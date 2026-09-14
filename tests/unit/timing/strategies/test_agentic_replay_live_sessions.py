# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgenticReplayStrategy + LaneRotationGate under ``--agentic-live-sessions K``.

Uses a fake issuer that mirrors ``CreditIssuer.issue_credit``'s gate wiring
(``set_root_dispatch_gate`` / ``observe_issued``) so the strategy's lane
resolution, rotation, recycle-into-rotation and drain paths run end to end.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.credit.dispatch import ChildDispatchResult
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import Trajectory
from tests.unit.timing.strategies._shared_helpers import (
    _build_real_trajectory_source,
    _make_dataset,
)

TURNS = 4  # start_turn_index 0 -> profiling turns 1, 2, 3 per session


class _FakeIssuer:
    """issue_credit == replay gate (pass-through) -> lane gate -> wire."""

    def __init__(self) -> None:
        self.gate = None
        self.wire: list[Credit] = []
        self.accept = True
        self.submitted = 0
        self.replay_gate = MagicMock()
        self.replay_gate.completed_prefixes.return_value = ()
        self.replay_gate.pending_turns.return_value = ()
        self.replay_gate.pending_turns_by_root.return_value = {}

    def set_root_dispatch_gate(self, gate) -> None:
        self.gate = gate

    async def issue_credit(self, turn: TurnToSend) -> bool:
        self.submitted += 1
        if self.gate is None:
            return await self._ready(turn)
        return await self.gate.submit(turn, lambda: self._ready(turn))

    async def _ready(self, turn: TurnToSend) -> bool:
        if not self.accept:
            return False
        credit = Credit(
            id=len(self.wire),
            phase=CreditPhase.PROFILING,
            conversation_id=turn.conversation_id,
            x_correlation_id=turn.x_correlation_id,
            turn_index=turn.turn_index,
            num_turns=turn.num_turns,
            issued_at_ns=0,
            agent_depth=turn.agent_depth,
            parent_correlation_id=turn.parent_correlation_id,
            root_correlation_id=turn.root_correlation_id,
            branch_mode=turn.branch_mode,
        )
        self.wire.append(credit)
        if self.gate is not None:
            self.gate.observe_issued(credit)
        return True

    async def dispatch_child_turn(self, turn: TurnToSend) -> ChildDispatchResult:
        return (
            ChildDispatchResult.ISSUED
            if await self._ready(turn)
            else ChildDispatchResult.REJECTED
        )

    async def acquire_lane_credit(self, *args, **kwargs) -> bool:
        return True


def _strategy(k: int, trees: int):
    dataset = _make_dataset(num_traces=trees, turns_per_trace=TURNS)
    trajectories = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0)
        for i in range(trees)
    ]
    source = _build_real_trajectory_source(dataset=dataset, trajectories=trajectories)
    source._children_by_parent = {}
    cfg = MagicMock()
    cfg.phase = CreditPhase.PROFILING
    cfg.concurrency = trees
    cfg.agentic_live_sessions = k
    cfg.agentic_cache_warmup_duration_sec = None
    cfg.warmup_requests_per_lane = None
    issuer = _FakeIssuer()
    stop_checker = MagicMock()
    stop_checker.can_start_new_session.return_value = True
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=source,
        scheduler=LoopScheduler(),
        stop_checker=stop_checker,
        credit_issuer=issuer,
        lifecycle=MagicMock(),
    )
    return strategy, issuer, stop_checker


async def _settle() -> None:
    # schedule_later(0.0) fires on the next loop tick; give the task room.
    await asyncio.sleep(0.005)
    for _ in range(3):
        await asyncio.sleep(0)


async def _return(strategy: AgenticReplayStrategy, credit: Credit) -> None:
    """Mirror CreditCallbackHandler order: observe first, then dispatch."""
    strategy.observe_credit_return(credit)
    await strategy.handle_credit_return(credit)
    await _settle()


def _ids(credits: list[Credit]) -> list[str]:
    return [c.conversation_id for c in credits]


@pytest.mark.asyncio
async def test_k1_installs_no_gate_and_dispatches_every_lane() -> None:
    strategy, issuer, _ = _strategy(k=1, trees=6)
    await strategy.setup_phase()
    assert issuer.gate is None
    await strategy.execute_phase()
    assert _ids(issuer.wire) == [f"trace_{i}" for i in range(6)]
    assert [c.turn_index for c in issuer.wire] == [1] * 6
    strategy.scheduler.cancel_all()


@pytest.mark.asyncio
async def test_k3_rotates_two_lanes_and_recycles_into_the_rotation() -> None:
    strategy, issuer, _ = _strategy(k=3, trees=6)  # lanes: {0,1,2}, {3,4,5}
    await strategy.setup_phase()
    gate = issuer.gate
    assert gate is not None
    await strategy.execute_phase()
    # One main-agent request per dispatch lane; the other four trees wait.
    assert _ids(issuer.wire) == ["trace_0", "trace_3"]
    assert gate.queued_count == 4
    assert gate.in_flight_count == 2

    returned = 0
    while len(issuer.wire) < 20:
        credit = issuer.wire[returned]
        returned += 1
        await _return(strategy, credit)
        # Ledger: per lane issued - returned == in flight, never above one.
        for _lane, (issued, done, _queued) in gate.ledger().items():
            assert 0 <= issued - done <= 1
        assert gate.in_flight_count <= 2

    lane0 = [
        c for c in issuer.wire if c.conversation_id in {"trace_0", "trace_1", "trace_2"}
    ]
    lane1 = [
        c
        for c in issuer.wire
        if c.conversation_id not in {"trace_0", "trace_1", "trace_2"}
    ]
    # Wave order A,B,C,A,B,C,... through every profiling turn of each session.
    assert _ids(lane0[:9]) == ["trace_0", "trace_1", "trace_2"] * 3
    assert [c.turn_index for c in lane0[:9]] == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert _ids(lane1[:9]) == ["trace_3", "trace_4", "trace_5"] * 3
    # After a session's final turn returns, its tree recycles from the
    # sequential sampler into the SAME lane's rotation (turn 0, new identity),
    # queued behind the sessions already waiting.
    recycled = lane0[9]
    assert recycled.turn_index == 0
    assert recycled.x_correlation_id not in {t.x_correlation_id for t in lane0[:9]}
    # Every accepted submit is either on the wire or still queued.
    assert len(issuer.wire) + gate.queued_count == issuer.submitted
    assert gate.unresolved_count == 0
    strategy.scheduler.cancel_all()


@pytest.mark.asyncio
async def test_k3_drain_at_stop_empties_queues_without_wedging() -> None:
    strategy, issuer, stop_checker = _strategy(k=3, trees=6)
    await strategy.setup_phase()
    await strategy.execute_phase()
    gate = issuer.gate
    assert gate.queued_count == 4
    # Phase stop: nothing more may reach the wire.
    issuer.accept = False
    stop_checker.can_start_new_session.return_value = False
    for credit in list(issuer.wire):
        await _return(strategy, credit)
    assert len(issuer.wire) == 2  # nothing new reached the wire
    assert gate.queued_count == 0  # refused turns drained, none stuck
    assert gate.in_flight_count == 0
    assert all(issued == done for issued, done, _ in gate.ledger().values())
    strategy.scheduler.cancel_all()
