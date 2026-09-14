# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TrajectorySource under ``--agentic-live-sessions K`` (concurrency x K trees)."""

from __future__ import annotations

import pytest

from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.trajectory_source import TrajectorySource

POOL = 20
CONCURRENCY = 3


def _dataset(*, timestamped: bool) -> DatasetMetadata:
    convs = []
    for i in range(POOL):
        turns = [
            TurnMetadata(
                timestamp_ms=(1_000.0 * t) if timestamped else None, delay_ms=None
            )
            for t in range(6)
        ]
        convs.append(ConversationMetadata(conversation_id=f"trace_{i}", turns=turns))
    return DatasetMetadata(
        conversations=convs, sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
    )


def _source(k: int | None, *, timestamped: bool = False) -> TrajectorySource:
    ds = _dataset(timestamped=timestamped)
    kwargs = {} if k is None else {"live_sessions_per_lane": k}
    return TrajectorySource(
        dataset_metadata=ds,
        dataset_sampler=SequentialSampler(
            [c.conversation_id for c in ds.conversations]
        ),
        concurrency=CONCURRENCY,
        random_seed=42,
        cache_bust_enabled=True,
        **kwargs,
    )


def _key(trajectory):
    return (
        trajectory.conversation_id,
        trajectory.start_turn_index,
        None if trajectory.snapshot is None else trajectory.snapshot.t_star_ms,
    )


@pytest.mark.parametrize("timestamped", [False, True])
def test_k1_is_the_baseline(timestamped: bool) -> None:
    baseline = _source(None, timestamped=timestamped)
    explicit = _source(1, timestamped=timestamped)
    assert len(baseline.trajectories) == CONCURRENCY
    assert [_key(t) for t in baseline.trajectories] == [
        _key(t) for t in explicit.trajectories
    ]
    assert baseline.live_sessions_per_lane == 1


@pytest.mark.parametrize("timestamped", [False, True])
def test_k4_builds_concurrency_x_k_trees_with_baseline_prefix(
    timestamped: bool,
) -> None:
    baseline = _source(None, timestamped=timestamped)
    waved = _source(4, timestamped=timestamped)
    assert waved.live_sessions_per_lane == 4
    assert len(waved.trajectories) == CONCURRENCY * 4
    # Same sampler draws and same per-index t* seeds: the first `concurrency`
    # trees are exactly the classic lanes; the rest continue the sequential
    # draw (trajectory i takes the i-th root).
    assert [_key(t) for t in waved.trajectories[:CONCURRENCY]] == [
        _key(t) for t in baseline.trajectories
    ]
    assert [t.conversation_id for t in waved.trajectories] == [
        f"trace_{i}" for i in range(CONCURRENCY * 4)
    ]
    # Distinct session identities per tree.
    assert len({t.x_correlation_id for t in waved.trajectories}) == CONCURRENCY * 4


def test_dispatch_lane_is_index_floor_div_k() -> None:
    waved = _source(4)
    lanes = [i // waved.live_sessions_per_lane for i in range(len(waved.trajectories))]
    assert lanes == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]


def test_warmup_primers_counted_per_tree() -> None:
    waved = _source(4, timestamped=True)
    assert len(waved.warmup_credit_counts_by_lane) == CONCURRENCY * 4


def test_live_sessions_below_one_rejected() -> None:
    with pytest.raises(ValueError, match="live_sessions_per_lane"):
        _source(0)
