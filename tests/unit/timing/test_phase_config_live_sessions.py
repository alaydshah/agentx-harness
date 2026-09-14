# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``--agentic-live-sessions`` plumbing: phase -> CreditPhaseConfig / AIPerfConfig."""

from __future__ import annotations

import pydantic
import pytest

from aiperf.config.phases import PhaseConfig
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import (
    _build_agentic_warmup_config,
    _build_profiling_config,
)
from aiperf.timing.request_cancellation import RequestCancellationConfig
from tests.unit.config.test_validators import _agentic_phase, _make

_PHASE_ADAPTER = pydantic.TypeAdapter(PhaseConfig)


def _phase(**overrides) -> PhaseConfig:
    return _PHASE_ADAPTER.validate_python(
        {
            "name": "profiling",
            "type": "concurrency",
            "concurrency": 3,
            "duration": 1800,
            "timing_mode": TimingMode.AGENTIC_REPLAY,
            **overrides,
        }
    )


def _profiling(phase: PhaseConfig):
    return _build_profiling_config(
        phase,
        default_cancellation=RequestCancellationConfig(),
        phase_index=0,
        profiling_index=0,
    )


def test_default_is_one_tree_per_lane() -> None:
    phase = _phase()
    assert phase.agentic_live_sessions == 1
    profiling = _profiling(phase)
    assert profiling.concurrency == 3
    assert profiling.agentic_live_sessions == 1
    warmup = _build_agentic_warmup_config(phase)
    assert warmup is not None and warmup.concurrency == 3


def test_k4_sizes_session_slots_to_concurrency_x_k() -> None:
    phase = _phase(agentic_live_sessions=4)
    profiling = _profiling(phase)
    assert profiling.concurrency == 12  # every live tree holds a session slot
    assert profiling.agentic_live_sessions == 4
    assert profiling.expected_num_sessions is None
    warmup = _build_agentic_warmup_config(phase)
    assert warmup is not None
    assert warmup.concurrency == 12
    assert warmup.agentic_live_sessions == 4


def test_k4_warmup_request_budget_applies_per_tree() -> None:
    phase = _phase(agentic_live_sessions=4, warmup_requests_per_lane=10)
    warmup = _build_agentic_warmup_config(phase)
    assert warmup is not None
    assert warmup.total_expected_requests == 120


def test_non_agentic_phase_ignores_default_k() -> None:
    phase = _PHASE_ADAPTER.validate_python(
        {"name": "profiling", "type": "concurrency", "concurrency": 5, "requests": 10}
    )
    assert _profiling(phase).concurrency == 5


def test_aiperf_config_rejects_k_above_one_without_agentic_replay() -> None:
    with pytest.raises(ValueError, match="agentic-live-sessions > 1 requires"):
        _make(phases=_agentic_phase(agentic_live_sessions=4))


def test_aiperf_config_accepts_k_above_one_with_agentic_replay() -> None:
    cfg = _make(
        phases=_agentic_phase(agentic_live_sessions=4, timing_mode="agentic_replay")
    )
    assert cfg.benchmark.phases[0].agentic_live_sessions == 4


def test_aiperf_config_accepts_default_k_anywhere() -> None:
    cfg = _make(phases=_agentic_phase())
    assert cfg.benchmark.phases[0].agentic_live_sessions == 1
