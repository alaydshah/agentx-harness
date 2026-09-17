# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for targeted-drain snapshot root reservation."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import ConversationMetadata, TurnMetadata
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.structs import TurnToSend
from aiperf.plugin.enums import TimingMode
from aiperf.timing.branch_orchestrator import BranchOrchestrator
from aiperf.timing.concurrency import ConcurrencyManager
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.lifecycle import PhaseLifecycle
from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
from aiperf.timing.phase.runner import PhaseRunner
from aiperf.timing.phase.stop_conditions import StopConditionChecker
from aiperf.timing.request_cancellation import (
    RequestCancellationConfig,
    RequestCancellationSimulator,
)
from aiperf.timing.session_tree import SessionTreeRegistry
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import (
    CacheBustLedger,
    ConversationState,
    Trajectory,
    TrajectorySnapshot,
    TrajectorySource,
)


class _Transport:
    def __init__(self) -> None:
        self.credits = []
        self.fail = False

    async def send_credit(self, *, credit) -> None:
        if self.fail:
            raise RuntimeError("transport failure")
        self.credits.append(credit)


class _Clock:
    def __init__(self) -> None:
        self.pending = []

    def schedule_later(self, delay, coro, **kwargs) -> None:
        self.pending.append((delay, coro, kwargs))

    async def run(self) -> None:
        while self.pending:
            _, coroutine, _ = self.pending.pop(0)
            await coroutine

    def close(self) -> None:
        for _, coroutine, _ in self.pending:
            coroutine.close()
        self.pending = []


class _Fixture:
    def __init__(
        self,
        *,
        root_delay: int = 10,
        child_delay: int = 0,
        root_turn: int = 3,
        gated: bool = False,
        rootless: bool = False,
        target: int | None = 250,
        registry: bool = True,
        slots: int = 12,
        sessions: int = 12,
    ) -> None:
        self.config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.AGENTIC_REPLAY,
            concurrency=slots,
            prefill_concurrency=1,
            expected_num_sessions=sessions,
            agentic_drain_target_requests=target,
        )
        self.progress = PhaseProgressTracker(self.config)
        self.lifecycle = PhaseLifecycle(self.config)
        self.lifecycle.start()
        self.stop = StopConditionChecker(
            self.config, self.lifecycle, self.progress.counter
        )
        self.concurrency = ConcurrencyManager()
        self.concurrency.configure_for_phase(CreditPhase.PROFILING, slots, 1)
        self.registry = SessionTreeRegistry(self.concurrency) if registry else None
        self.router = _Transport()
        self.clock = _Clock()
        self.issuer = CreditIssuer(
            phase=CreditPhase.PROFILING,
            stop_checker=self.stop,
            progress=self.progress,
            concurrency_manager=self.concurrency,
            credit_router=self.router,
            cancellation_policy=RequestCancellationSimulator(
                RequestCancellationConfig()
            ),
            lifecycle=self.lifecycle,
            session_tree_registry=self.registry,
            defer_session_target_completion=True,
        )

        root = ConversationState(
            "root",
            "R",
            root_turn,
            next_dispatch_offset_ms=root_delay,
            root_correlation_id="R",
            waiting_on_children=gated,
            join_target_turn_index=root_turn if gated else None,
        )
        child = ConversationState(
            "root::sa:child",
            "C",
            1,
            next_dispatch_offset_ms=child_delay,
            agent_depth=1,
            parent_correlation_id="R",
            root_correlation_id="R",
            branch_mode=ConversationBranchMode.SPAWN,
        )
        states = (child,) if rootless else (child, root)
        self.trajectory = Trajectory(
            "root", 0, TrajectorySnapshot(t_star_ms=0, states=states)
        )

        # Use the real realized-source object while supplying compact synthetic
        # metadata instead of invoking dataset selection.
        self.source = TrajectorySource.__new__(TrajectorySource)
        self.source._metadata_lookup = {
            conversation_id: ConversationMetadata(
                conversation_id=conversation_id,
                turns=[TurnMetadata() for _ in range(8)],
            )
            for conversation_id in ("root", "root::sa:child")
        }
        self.source._drain_target_requests = target
        self.source._target_size = slots
        self.source._live_sessions_per_lane = 1
        self.source.trajectories = [self.trajectory]
        self.source._cache_bust_ledger = CacheBustLedger()
        self.orchestrator = BranchOrchestrator(
            self.source,
            self.issuer,
            session_tree_registry=self.registry,
            scheduler=self.clock,
        )
        self.strategy = AgenticReplayStrategy(
            config=self.config,
            conversation_source=self.source,
            scheduler=self.clock,
            stop_checker=self.stop,
            credit_issuer=self.issuer,
            lifecycle=self.lifecycle,
            branch_orchestrator=self.orchestrator,
            session_tree_registry=self.registry,
            progress=self.progress,
        )

    async def dispatch(self) -> None:
        await self.strategy._dispatch_snapshot_for_profiling(
            self.trajectory, lane=0, phase_t0_offset_ms=0
        )

    def cleanup(self) -> None:
        self.clock.close()
        self.issuer.stop_issuing()
        runner = PhaseRunner.__new__(PhaseRunner)
        AIPerfLoggerMixin.__init__(
            runner, logger_name="snapshot-reservation-test-cleanup"
        )
        runner._session_tree_registry = self.registry
        runner._config = self.config
        runner._branch_orchestrator = self.orchestrator
        runner._callback_handler = SimpleNamespace(
            set_branch_orchestrator=lambda _: None
        )
        PhaseRunner._detach_orchestrator_and_cleanup(runner)
        if self.registry:
            assert self.registry.open_count() == 0

    def release_prefill(self) -> None:
        self.concurrency.release_prefill_slot(CreditPhase.PROFILING)

    def slot_stats(self):
        return self.concurrency._session_limiter.global_stats


class TestSnapshotRootReservation(unittest.IsolatedAsyncioTestCase):
    async def test_child_before_root_uses_one_tree_slot(self):
        fixture = _Fixture()
        try:
            await fixture.dispatch()
            self.assertEqual(
                [credit.x_correlation_id for credit in fixture.router.credits],
                ["C"],
            )
            self.assertEqual(len(fixture.clock.pending), 1)
            self.assertGreater(fixture.clock.pending[0][0], 0)
            self.assertLessEqual(fixture.clock.pending[0][0], 0.01)
            self.assertEqual(fixture.progress.counter.sent_sessions, 0)

            fixture.release_prefill()
            await fixture.clock.run()

            self.assertEqual(
                [credit.x_correlation_id for credit in fixture.router.credits],
                ["C", "R"],
            )
            self.assertEqual(fixture.progress.counter.sent_sessions, 1)
            self.assertEqual(fixture.slot_stats().acquire_count, 1)
            self.assertEqual(fixture.registry.peak_open, 1)
            self.assertTrue(
                all(
                    credit.counts_toward_phase_target
                    for credit in fixture.router.credits
                )
            )
            self.assertTrue(
                all(
                    credit.max_tokens_override is None
                    for credit in fixture.router.credits
                )
            )
            self.assertEqual(
                [credit.turn_index for credit in fixture.router.credits], [1, 3]
            )
            fixture.registry.on_descendant_done("R")
            self.assertEqual(fixture.registry.open_count(), 1)
            fixture.registry.on_root_terminal("R")
            self.assertEqual(fixture.slot_stats().release_count, 1)
        finally:
            fixture.cleanup()
        self.assertEqual(fixture.slot_stats().release_count, 1)

    async def test_same_deadline_root_blocked_on_prefill(self):
        fixture = _Fixture(root_delay=0)
        task = asyncio.create_task(fixture.dispatch())
        try:
            await asyncio.sleep(0.01)
            self.assertFalse(task.done())
            self.assertEqual(
                [credit.x_correlation_id for credit in fixture.router.credits],
                ["C"],
            )
            self.assertEqual(fixture.registry.open_count(), 1)
            fixture.release_prefill()
            await asyncio.wait_for(task, 0.5)
            self.assertEqual(fixture.progress.counter.sent_sessions, 1)
            self.assertEqual(fixture.slot_stats().acquire_count, 1)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            fixture.cleanup()

    async def test_reserved_turn_zero_nonblocking_issue_counts_once(self):
        fixture = _Fixture(root_turn=0)
        try:
            self.assertTrue(await fixture.issuer.reserve_snapshot_root("R"))
            turn = TurnToSend(
                "root",
                "R",
                0,
                8,
                is_session_start=True,
                max_tokens_override=1024,
            )
            self.assertTrue(await fixture.issuer.try_issue_credit(turn))
            self.assertEqual(fixture.progress.counter.sent_sessions, 1)
            self.assertEqual(fixture.slot_stats().acquire_count, 1)
            self.assertEqual(fixture.router.credits[0].max_tokens_override, 1024)
        finally:
            fixture.cleanup()

    async def test_refused_root_admission_leaves_tree_owned_until_cleanup(self):
        fixture = _Fixture()
        try:
            self.assertTrue(await fixture.issuer.reserve_snapshot_root("R"))
            fixture.issuer.set_turn_admission(lambda _: False)
            self.assertFalse(
                await fixture.issuer.issue_credit(
                    TurnToSend("root", "R", 3, 8, is_session_start=True)
                )
            )
            self.assertEqual(fixture.slot_stats().release_count, 0)
            self.assertEqual(fixture.registry.open_count(), 1)
            self.assertIn("R", fixture.issuer._reserved_snapshot_roots)
            self.assertEqual(fixture.progress.counter.sent_sessions, 0)
        finally:
            fixture.cleanup()
        self.assertEqual(fixture.slot_stats().release_count, 1)
        self.assertFalse(fixture.issuer._reserved_snapshot_roots)

    async def test_cancelled_prefill_leaves_tree_owned_until_cleanup(self):
        fixture = _Fixture()
        self.assertTrue(await fixture.issuer.reserve_snapshot_root("R"))
        await fixture.concurrency.acquire_prefill_slot(
            CreditPhase.PROFILING, lambda: True
        )
        task = asyncio.create_task(
            fixture.issuer.issue_credit(
                TurnToSend("root", "R", 3, 8, is_session_start=True)
            )
        )
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(fixture.slot_stats().release_count, 0)
        self.assertEqual(fixture.progress.counter.sent_sessions, 0)
        fixture.cleanup()
        self.assertEqual(fixture.slot_stats().release_count, 1)

    async def test_duplicate_refuses_while_capacity_waits(self):
        fixture = _Fixture(slots=1)
        pending = None
        try:
            self.assertTrue(await fixture.issuer.reserve_snapshot_root("R"))
            self.assertFalse(await fixture.issuer.reserve_snapshot_root("R"))
            pending = asyncio.create_task(
                fixture.issuer.reserve_snapshot_root("another")
            )
            await asyncio.sleep(0.01)
            self.assertFalse(pending.done())
            self.assertEqual(fixture.slot_stats().acquire_count, 1)
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            fixture.cleanup()

    async def test_cancelled_capacity_wait_clears_pending_reservation(self):
        fixture = _Fixture(slots=1)
        pending = None
        try:
            self.assertTrue(await fixture.issuer.reserve_snapshot_root("R"))
            pending = asyncio.create_task(
                fixture.issuer.reserve_snapshot_root("another")
            )
            await asyncio.sleep(0.01)
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            self.assertNotIn("another", fixture.issuer._pending_snapshot_roots)
            self.assertNotIn("another", fixture.issuer._reserved_snapshot_roots)
            self.assertEqual(fixture.slot_stats().acquire_count, 1)
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            fixture.cleanup()

    async def test_registration_exception_releases_unowned_slot(self):
        fixture = _Fixture()
        original = fixture.registry.open_tree
        try:

            def fail_registration(*args, **kwargs):
                raise RuntimeError("registration")

            fixture.registry.open_tree = fail_registration
            with self.assertRaisesRegex(RuntimeError, "registration"):
                await fixture.issuer.reserve_snapshot_root("R")
            self.assertEqual(fixture.slot_stats().acquire_count, 1)
            self.assertEqual(fixture.slot_stats().release_count, 1)
        finally:
            fixture.registry.open_tree = original
            fixture.cleanup()

    async def test_router_failure_still_releases_once(self):
        fixture = _Fixture()
        fixture.router.fail = True
        try:
            self.assertTrue(await fixture.issuer.reserve_snapshot_root("R"))
            with self.assertRaisesRegex(RuntimeError, "transport"):
                await fixture.issuer.issue_credit(
                    TurnToSend("root", "R", 3, 8, is_session_start=True)
                )
            self.assertEqual(fixture.progress.counter.sent_sessions, 1)
            self.assertFalse(fixture.issuer._reserved_snapshot_roots)
        finally:
            fixture.cleanup()
        self.assertEqual(fixture.slot_stats().release_count, 1)

    async def test_gated_and_rootless_accounting_is_unchanged(self):
        for rootless in (False, True):
            with self.subTest(rootless=rootless):
                fixture = _Fixture(gated=not rootless, rootless=rootless)
                try:
                    await fixture.dispatch()
                    self.assertEqual(fixture.slot_stats().acquire_count, 1)
                    self.assertEqual(
                        fixture.progress.counter.sent_sessions,
                        0 if rootless else 1,
                    )
                    self.assertFalse(fixture.issuer._reserved_snapshot_roots)
                finally:
                    fixture.cleanup()

    async def test_target_mode_cannot_steal_quota_by_recycling(self):
        fixture = _Fixture(sessions=1)
        try:
            self.assertIsNone(fixture.source.next_recycle_conversation_id())
            await fixture.dispatch()
            fixture.release_prefill()
            await fixture.clock.run()
            self.assertEqual(fixture.progress.counter.sent_sessions, 1)
            self.assertIsNone(fixture.source.next_recycle_conversation_id())
            self.assertFalse(fixture.stop.can_start_new_session())
        finally:
            fixture.cleanup()

    async def test_legacy_root_without_registry_is_unchanged(self):
        fixture = _Fixture(registry=False, target=None)
        try:
            self.assertFalse(await fixture.issuer.reserve_snapshot_root("R"))
            self.assertTrue(
                await fixture.issuer.issue_credit(TurnToSend("root", "R", 0, 8))
            )
            self.assertEqual(fixture.progress.counter.sent_sessions, 1)
            self.assertEqual(fixture.slot_stats().acquire_count, 1)
        finally:
            fixture.cleanup()
            fixture.concurrency.release_session_slot(CreditPhase.PROFILING)

    async def test_stop_before_reservation_does_not_acquire(self):
        fixture = _Fixture()
        fixture.issuer.stop_issuing()
        try:
            self.assertFalse(await fixture.issuer.reserve_snapshot_root("R"))
            self.assertEqual(fixture.slot_stats().acquire_count, 0)
        finally:
            fixture.cleanup()

    async def test_two_initial_roots_reserve_last_quota_for_snapshot_root(self):
        fixture = _Fixture(sessions=2, slots=2)
        try:
            self.assertTrue(await fixture.issuer.reserve_snapshot_root("R"))
            self.assertTrue(
                await fixture.issuer.issue_credit(TurnToSend("other", "O", 0, 2))
            )
            fixture.release_prefill()
            self.assertTrue(
                await fixture.issuer.issue_credit(
                    TurnToSend("root", "R", 3, 8, is_session_start=True)
                )
            )
            self.assertEqual(fixture.progress.counter.sent_sessions, 2)
            self.assertEqual(fixture.slot_stats().acquire_count, 2)
            self.assertIsNone(fixture.source.next_recycle_conversation_id())
        finally:
            fixture.cleanup()
        self.assertEqual(fixture.slot_stats().release_count, 2)

    async def test_dispatch_waits_for_temporary_capacity(self):
        fixture = _Fixture(slots=1, root_delay=100)
        dispatch = None
        try:
            self.assertTrue(await fixture.issuer.reserve_snapshot_root("other"))
            dispatch = asyncio.create_task(fixture.dispatch())
            await asyncio.sleep(0.01)
            self.assertFalse(dispatch.done())
            self.assertEqual(fixture.clock.pending, [])
            self.assertEqual(fixture.router.credits, [])
            self.assertEqual(fixture.slot_stats().acquire_count, 1)
            fixture.registry.on_root_terminal("other")
            await asyncio.wait_for(dispatch, 0.5)
            self.assertEqual(
                [credit.x_correlation_id for credit in fixture.router.credits],
                ["C"],
            )
            self.assertEqual(fixture.slot_stats().acquire_count, 2)
            self.assertEqual(len(fixture.clock.pending), 1)
            self.assertGreater(fixture.clock.pending[0][0], 0)
            self.assertLess(fixture.clock.pending[0][0], 0.1)
        finally:
            if dispatch is not None and not dispatch.done():
                dispatch.cancel()
                await asyncio.gather(dispatch, return_exceptions=True)
            fixture.cleanup()

    async def test_capacity_wait_is_subtracted_from_gated_join_deadline(self):
        fixture = _Fixture(slots=1, root_delay=100)
        root = fixture.trajectory.snapshot.states[1]
        gated_child = ConversationState(
            "root::sa:child",
            "G",
            1,
            next_dispatch_offset_ms=100,
            agent_depth=1,
            parent_correlation_id="R",
            root_correlation_id="R",
            waiting_on_children=True,
            join_target_turn_index=1,
            branch_mode=ConversationBranchMode.SPAWN,
        )
        fixture.trajectory = Trajectory(
            "root",
            0,
            TrajectorySnapshot(t_star_ms=0, states=(gated_child, root)),
        )
        fixture.source.trajectories = [fixture.trajectory]
        captured = {}

        def seed_snapshot(states, **kwargs):
            captured.update(kwargs)

        fixture.strategy.branch_orchestrator = SimpleNamespace(
            seed_snapshot=seed_snapshot
        )
        dispatch = None
        try:
            self.assertTrue(await fixture.issuer.reserve_snapshot_root("other"))
            dispatch = asyncio.create_task(fixture.dispatch())
            await asyncio.sleep(0.02)
            fixture.registry.on_root_terminal("other")
            await asyncio.wait_for(dispatch, 0.5)

            gated_delay = captured["join_release_delays_ms"]["G"]
            self.assertGreater(gated_delay, 0)
            self.assertLess(gated_delay, 100)
        finally:
            if dispatch is not None and not dispatch.done():
                dispatch.cancel()
                await asyncio.gather(dispatch, return_exceptions=True)
            fixture.cleanup()

    async def test_non_targeted_replay_does_not_use_reservation(self):
        fixture = _Fixture(target=None)
        try:
            with self.assertRaisesRegex(
                RuntimeError, "snapshot descendant was refused"
            ):
                await fixture.dispatch()
            self.assertFalse(fixture.issuer._reserved_snapshot_roots)
        finally:
            fixture.cleanup()

    async def test_active_phase_target_must_match_source_selection(self):
        targeted = _Fixture(target=250)
        targeted.source._drain_target_requests = None
        try:
            with self.assertRaisesRegex(RuntimeError, "drain target differs"):
                await targeted.dispatch()
            self.assertEqual(targeted.router.credits, [])
        finally:
            targeted.cleanup()

        untargeted = _Fixture(target=None)
        untargeted.source._drain_target_requests = 250
        try:
            with self.assertRaisesRegex(RuntimeError, "drain target differs"):
                await untargeted.dispatch()
            self.assertEqual(untargeted.router.credits, [])
            self.assertFalse(untargeted.issuer._reserved_snapshot_roots)
        finally:
            untargeted.cleanup()

    async def test_active_phase_tree_shape_must_match_source_selection(self):
        fixture = _Fixture(target=250)
        fixture.source._target_size += 1
        try:
            with self.assertRaisesRegex(RuntimeError, "targeted tree shape differs"):
                await fixture.dispatch()
            self.assertEqual(fixture.router.credits, [])
            self.assertFalse(fixture.issuer._reserved_snapshot_roots)
        finally:
            fixture.cleanup()
