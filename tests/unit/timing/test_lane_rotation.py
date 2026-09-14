# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LaneRotationGate: one main-agent request per dispatch lane, FIFO rotation."""

from __future__ import annotations

import asyncio

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.timing.lane_rotation import LaneRotationGate

K = 4  # live sessions per lane in these tests


def _lane_of(root: str) -> int | None:
    # roots are "r<i>"; trajectory index i -> dispatch lane i // K
    if not root.startswith("r"):
        return None
    return int(root[1:]) // K


def _turn(root: str, turn_index: int = 0, *, depth: int = 0) -> TurnToSend:
    return TurnToSend(
        conversation_id=f"trace_{root}",
        x_correlation_id=root if depth == 0 else f"{root}-child",
        turn_index=turn_index,
        num_turns=10,
        agent_depth=depth,
        parent_correlation_id=root if depth else None,
        root_correlation_id=root if depth else None,
    )


class _Wire:
    """Fake issuer tail: records credits that 'reach the wire'."""

    def __init__(self, gate: LaneRotationGate) -> None:
        self.gate = gate
        self.credits: list[Credit] = []
        self.refused = 0

    def issue_fn(self, turn: TurnToSend, *, accept: bool = True):
        async def issue() -> bool:
            if not accept:
                self.refused += 1
                return False
            credit = Credit(
                id=len(self.credits),
                phase=CreditPhase.PROFILING,
                conversation_id=turn.conversation_id,
                x_correlation_id=turn.x_correlation_id,
                turn_index=turn.turn_index,
                num_turns=turn.num_turns,
                issued_at_ns=0,
                agent_depth=turn.agent_depth,
                parent_correlation_id=turn.parent_correlation_id,
                root_correlation_id=turn.root_correlation_id,
            )
            self.credits.append(credit)
            self.gate.observe_issued(credit)
            return True

        return issue

    def on_wire(self) -> list[str]:
        return [c.x_correlation_id for c in self.credits]


def _gate() -> tuple[LaneRotationGate, _Wire]:
    gate = LaneRotationGate(
        lane_of=_lane_of,
        run_soon=lambda coro: asyncio.get_running_loop().create_task(coro),
    )
    return gate, _Wire(gate)


async def _settle() -> None:
    for _ in range(4):
        await asyncio.sleep(0)


def _assert_ledger(gate: LaneRotationGate) -> None:
    """issued - returned == in_flight per lane, and never more than one."""
    for lane, (issued, returned, _queued) in gate.ledger().items():
        assert 0 <= issued - returned <= 1, (lane, issued, returned)
    assert gate.in_flight_count == sum(
        issued - returned for issued, returned, _ in gate.ledger().values()
    )


@pytest.mark.asyncio
async def test_children_bypass_the_gate() -> None:
    gate, wire = _gate()
    assert await gate.submit(_turn("r0"), wire.issue_fn(_turn("r0"))) is True
    child = _turn("r0", 3, depth=1)
    assert await gate.submit(child, wire.issue_fn(child)) is True
    # root on the wire AND the child issued immediately despite the busy lane
    assert wire.on_wire() == ["r0", "r0-child"]
    assert gate.queued_count == 0
    assert gate.in_flight_count == 1  # children are not lane traffic


@pytest.mark.asyncio
async def test_k4_wave_rotation_order_across_two_lanes() -> None:
    gate, wire = _gate()
    roots = [f"r{i}" for i in range(2 * K)]  # lane 0: r0..r3, lane 1: r4..r7
    for root in roots:
        turn = _turn(root)
        assert await gate.submit(turn, wire.issue_fn(turn)) is True
    # One main-agent request per lane on the wire, the rest queued in order.
    assert wire.on_wire() == ["r0", "r4"]
    assert gate.queued_count == 6
    _assert_ledger(gate)

    def ret(index: int) -> None:
        gate.observe_returned(wire.credits[index])

    ret(0)  # r0 returns -> r1 (next in lane 0), not r0 again
    await _settle()
    assert wire.on_wire() == ["r0", "r4", "r1"]
    # r0's next turn arrives (think gap elapsed) while lane 0 is busy: queued
    # behind r2, r3 -> full wave A,B,C,D,A.
    nxt = _turn("r0", 1)
    assert await gate.submit(nxt, wire.issue_fn(nxt)) is True
    assert wire.on_wire() == ["r0", "r4", "r1"]
    ret(1)  # r4 returns -> r5
    await _settle()
    ret(2)  # r1 -> r2
    await _settle()
    ret(3)  # r5 -> r6
    await _settle()
    ret(4)  # r2 -> r3
    await _settle()
    ret(6)  # r3 -> r0 (turn 1)
    await _settle()
    lane0 = [
        c.x_correlation_id for c in wire.credits if _lane_of(c.x_correlation_id) == 0
    ]
    lane1 = [
        c.x_correlation_id for c in wire.credits if _lane_of(c.x_correlation_id) == 1
    ]
    assert lane0 == ["r0", "r1", "r2", "r3", "r0"]
    assert lane1 == ["r4", "r5", "r6"]
    assert wire.credits[-1].turn_index == 1
    _assert_ledger(gate)
    assert gate.in_flight_count == 2


@pytest.mark.asyncio
async def test_refused_queued_turn_drains_to_the_next_one() -> None:
    gate, wire = _gate()
    head = _turn("r0")
    await gate.submit(head, wire.issue_fn(head))
    refused = _turn("r1")
    await gate.submit(refused, wire.issue_fn(refused, accept=False))
    ok = _turn("r2")
    await gate.submit(ok, wire.issue_fn(ok))
    assert gate.queued_count == 2
    gate.observe_returned(wire.credits[0])
    await _settle()
    # r1 was refused (never on the wire) so the lane moved straight on to r2.
    assert wire.on_wire() == ["r0", "r2"]
    assert wire.refused == 1
    assert gate.queued_count == 0
    assert gate.in_flight_count == 1
    _assert_ledger(gate)


@pytest.mark.asyncio
async def test_all_refused_at_stop_leaves_no_wedged_lane() -> None:
    gate, wire = _gate()
    head = _turn("r4")
    await gate.submit(head, wire.issue_fn(head))
    for root in ("r5", "r6", "r7"):
        turn = _turn(root)
        await gate.submit(turn, wire.issue_fn(turn, accept=False))
    gate.observe_returned(wire.credits[0])
    await _settle()
    assert wire.refused == 3
    assert gate.queued_count == 0
    assert gate.in_flight_count == 0
    assert gate.ledger() == {1: (1, 1, 0)}


@pytest.mark.asyncio
async def test_unresolved_lane_is_issued_ungated() -> None:
    gate, wire = _gate()
    busy = _turn("r0")
    await gate.submit(busy, wire.issue_fn(busy))
    unknown = TurnToSend(
        conversation_id="trace_x", x_correlation_id="x", turn_index=0, num_turns=2
    )
    assert await gate.submit(unknown, wire.issue_fn(unknown)) is True
    assert wire.on_wire() == ["r0", "x"]
    assert gate.unresolved_count == 1
    assert gate.queued_count == 0


@pytest.mark.asyncio
async def test_ledger_conservation_under_interleaving() -> None:
    """Every accepted depth-0 submit is exactly one of: queued, in flight, returned."""
    gate, wire = _gate()
    submitted = 0
    returned = 0
    order = [f"r{i}" for i in range(2 * K)] * 3
    for step, root in enumerate(order):
        turn = _turn(root, step)
        await gate.submit(turn, wire.issue_fn(turn))
        submitted += 1
        if step % 3 == 2 and returned < len(wire.credits):
            gate.observe_returned(wire.credits[returned])
            returned += 1
            await _settle()
        _assert_ledger(gate)
        assert gate.queued_count + len(wire.credits) == submitted
    while returned < len(wire.credits):
        gate.observe_returned(wire.credits[returned])
        returned += 1
        await _settle()
        _assert_ledger(gate)
        assert gate.queued_count + len(wire.credits) == submitted
    assert gate.queued_count == 0
    assert len(wire.credits) == submitted
