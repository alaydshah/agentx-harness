# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-lane round-robin admission for agentic replay main-agent turns.

``--agentic-live-sessions K`` decouples the number of LIVE session trees from
the number of main-agent requests on the wire. ``TrajectorySource`` builds
``concurrency x K`` trees and K consecutive trajectory indices
(``K*L .. K*L+K-1``) share dispatch lane ``L``. This gate admits at most ONE
depth-0 (main-agent) request per dispatch lane at a time and rotates the
lane's trees in arrival order. A tree's next main-agent turn reaches the gate
only after
its recorded think gap has elapsed (strategy timer) and its recorded
predecessors are complete (replay barrier), so when the lane already has a
request in flight the arrival is queued and released, FIFO, as that request
returns. Depth > 0 descendants (subagents, sidecars, flat agents) are never
gated: they run under their tree exactly as before.

The gate is installed on the PROFILING ``CreditIssuer`` only when K > 1, so
K == 1 (the default) leaves every existing dispatch path byte-identical.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aiperf.common.aiperf_logger import AIPerfLogger

if TYPE_CHECKING:
    from aiperf.credit.structs import Credit, TurnToSend

_logger = AIPerfLogger(__name__)

_Issue = Callable[[], Awaitable[bool]]


@dataclass(slots=True)
class _LaneState:
    """Rotation state of one dispatch lane."""

    queue: deque[tuple[TurnToSend, _Issue]] = field(default_factory=deque)
    """Main-agent turns waiting for the lane, in arrival (= rotation) order."""
    in_flight: int = 0
    """Depth-0 credits of this lane currently on the wire."""
    dispatching: bool = False
    """An admitted ``issue()`` has been started but has not resolved yet."""
    issued: int = 0
    """Depth-0 credits of this lane that reached the wire (ledger)."""
    returned: int = 0
    """Depth-0 credits of this lane that returned (ledger)."""


class LaneRotationGate:
    """One main-agent request per dispatch lane; queued arrivals rotate FIFO."""

    def __init__(
        self,
        *,
        lane_of: Callable[[str], int | None],
        run_soon: Callable[[Coroutine[Any, Any, Any]], object],
    ) -> None:
        """Args:
        lane_of: resolves a tree's ``root_correlation_id`` to its dispatch
            lane, or None when the tree is unknown (the turn is then issued
            immediately -- fail open, never wedge a request).
        run_soon: schedules a coroutine to run on the event loop (the
            strategy passes its ``LoopScheduler`` so phase teardown cancels
            queued releases with every other pending replay task).
        """
        self._lane_of = lane_of
        self._run_soon = run_soon
        self._lanes: dict[int, _LaneState] = {}
        self._unresolved = 0

    def _state(self, lane: int) -> _LaneState:
        state = self._lanes.get(lane)
        if state is None:
            state = _LaneState()
            self._lanes[lane] = state
        return state

    async def submit(self, turn: TurnToSend, issue: _Issue) -> bool:
        """Issue ``turn`` now if its lane is free, else queue it behind the lane.

        Returns ``issue()``'s result when issued inline and True
        (``issue_credit``'s "more may be sent") when queued, mirroring the
        replay barrier's retained-dispatch contract.
        """
        if turn.agent_depth > 0:
            return await issue()
        lane = self._lane_of(turn.effective_root_correlation_id)
        if lane is None:
            self._unresolved += 1
            _logger.warning(
                lambda: (
                    "LaneRotationGate: no dispatch lane for root "
                    f"{turn.effective_root_correlation_id!r} (conversation "
                    f"{turn.conversation_id!r} turn {turn.turn_index}); issuing "
                    "ungated"
                )
            )
            return await issue()
        state = self._state(lane)
        if state.in_flight == 0 and not state.dispatching and not state.queue:
            state.dispatching = True
            try:
                return await issue()
            finally:
                state.dispatching = False
                self._release(lane, state)
        state.queue.append((turn, issue))
        return True

    def _release(self, lane: int, state: _LaneState) -> None:
        """Admit the queue head once the lane has nothing on the wire.

        Claims ``dispatching`` synchronously so a main-agent turn arriving
        between this call and the scheduled ``issue()`` queues behind the head
        instead of racing it -- that ordering IS the rotation.
        """
        if state.in_flight > 0 or state.dispatching or not state.queue:
            return
        _turn, issue = state.queue.popleft()
        state.dispatching = True
        self._run_soon(self._issue_queued(lane, state, issue))

    async def _issue_queued(self, lane: int, state: _LaneState, issue: _Issue) -> None:
        try:
            await issue()
        finally:
            # A refused issue never reaches the wire (in_flight stays 0), so
            # the release below moves straight on to the next queued turn.
            state.dispatching = False
            self._release(lane, state)

    def observe_issued(self, credit: Credit) -> None:
        """Account one depth-0 credit reaching the wire (issuer hook)."""
        if credit.agent_depth > 0:
            return
        lane = self._lane_of(credit.effective_root_correlation_id)
        if lane is None:
            return
        state = self._state(lane)
        state.in_flight += 1
        state.issued += 1

    def observe_returned(self, credit: Credit) -> None:
        """Account one depth-0 credit returning and release the lane's head."""
        if credit.agent_depth > 0:
            return
        lane = self._lane_of(credit.effective_root_correlation_id)
        if lane is None:
            return
        state = self._lanes.get(lane)
        if state is None:
            return
        if state.in_flight > 0:
            state.in_flight -= 1
        state.returned += 1
        self._release(lane, state)

    @property
    def queued_count(self) -> int:
        """Main-agent turns currently waiting for a lane."""
        return sum(len(state.queue) for state in self._lanes.values())

    @property
    def in_flight_count(self) -> int:
        """Main-agent requests currently on the wire across all lanes."""
        return sum(state.in_flight for state in self._lanes.values())

    @property
    def unresolved_count(self) -> int:
        """Depth-0 submits that could not be mapped to a lane (issued ungated)."""
        return self._unresolved

    def ledger(self) -> dict[int, tuple[int, int, int]]:
        """Per-lane ``(issued, returned, queued)`` for logs and tests."""
        return {
            lane: (state.issued, state.returned, len(state.queue))
            for lane, state in sorted(self._lanes.items())
        }
