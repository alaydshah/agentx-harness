"""Stable-quiescence completion for fixed Agentic Replay sessions."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from aiperf.common.enums import CreditPhase
from aiperf.timing.phase.runner import PhaseRunner


def _quiescent_runner(*, drain_target_requests: int | None) -> PhaseRunner:
    runner = PhaseRunner.__new__(PhaseRunner)
    runner._config = SimpleNamespace(
        phase=CreditPhase.PROFILING,
        phase_index=0,
        agentic_drain_target_requests=drain_target_requests,
    )

    counter = SimpleNamespace(
        # One selected tree became rootless after warmup, so only 11 of the
        # configured 12 root sessions were ever counted.
        root_admission_closed=False,
        root_requests_sent=528,
        total_session_turns=528,
        in_flight=0,
    )
    runner._progress = SimpleNamespace(counter=counter)

    runner._session_tree_registry = MagicMock()
    runner._session_tree_registry.open_count.return_value = 0
    runner._session_tree_registry.pending_descendant_count = 0

    runner._branch_orchestrator = MagicMock()
    runner._branch_orchestrator.has_pending_branch_work.return_value = False

    runner._credit_issuer = MagicMock()
    runner._credit_issuer.replay_gate.has_pending_work.return_value = False

    runner._scheduler = SimpleNamespace(pending_count=0, running_count=0)
    runner._execution_task = None
    runner._scheduler_failure = None
    return runner


def test_drain_target_completes_after_rootless_selected_tree_drains() -> None:
    runner = _quiescent_runner(drain_target_requests=1000)

    assert runner._is_generation_complete() is True


def test_regular_session_target_still_requires_root_admission_to_close() -> None:
    runner = _quiescent_runner(drain_target_requests=None)

    assert runner._is_generation_complete() is False
