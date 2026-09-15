# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic complete-tree selection near an agentic drain request target."""

from __future__ import annotations

import hashlib
import json

import pytest

from aiperf.common.enums import ConversationBranchMode
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.trajectory_source import TrajectorySource


def _dataset(*, reverse: bool = False) -> DatasetMetadata:
    conversations = [
        ConversationMetadata(
            conversation_id=f"trace_{turn_count}",
            turns=[
                TurnMetadata(timestamp_ms=float(index), delay_ms=None)
                for index in range(turn_count)
            ],
        )
        for turn_count in (11, 21, 31, 41, 51, 61)
    ]
    if reverse:
        conversations.reverse()
    return DatasetMetadata(
        conversations=conversations,
        sampling_strategy=DatasetSamplingStrategy.SHUFFLE,
    )


def _source(tmp_path, *, reverse: bool = False, target: int = 100, seed: int = 42):
    dataset = _dataset(reverse=reverse)
    return TrajectorySource(
        dataset_metadata=dataset,
        dataset_sampler=SequentialSampler(
            [conversation.conversation_id for conversation in dataset.conversations]
        ),
        concurrency=1,
        live_sessions_per_lane=3,
        expected_num_sessions=3,
        random_seed=seed,
        start_min_ratio=0,
        start_max_ratio=0,
        warmup_requests_per_lane=1,
        drain_target_requests=target,
        selection_artifact_dir=tmp_path,
        cache_bust_enabled=True,
    )


def test_target_selection_is_stable_across_dataset_order(tmp_path) -> None:
    first = _source(tmp_path / "first")
    second = _source(tmp_path / "second", reverse=True)

    assert first.drain_selection_manifest == second.drain_selection_manifest
    manifest = first.drain_selection_manifest
    assert manifest is not None
    assert manifest["target_requests"] == 100
    assert manifest["actual_requests"] == 90
    assert len(manifest["selected_trace_ids"]) == 3
    assert sum(manifest["selected_trace_profiling_requests"]) == 90
    assert first.next_recycle_conversation_id() is None


def test_seed_participates_in_the_frozen_selection(tmp_path) -> None:
    first = _source(tmp_path / "first", target=97, seed=42)
    second = _source(tmp_path / "second", target=97, seed=43)
    assert first.drain_selection_manifest is not None
    assert second.drain_selection_manifest is not None
    assert (
        first.drain_selection_manifest["selection_sha256"]
        != second.drain_selection_manifest["selection_sha256"]
    )


def test_selection_manifest_hashes_the_frozen_work_and_is_written(tmp_path) -> None:
    source = _source(tmp_path)
    assert source.drain_selection_manifest is not None
    manifest = dict(source.drain_selection_manifest)
    selection_hash = manifest.pop("selection_sha256")
    expected_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert selection_hash == expected_hash

    on_disk = json.loads((tmp_path / "agentic_drain_selection.json").read_text())
    assert on_disk == {**manifest, "selection_sha256": selection_hash}


def test_target_reports_the_progressive_greedy_complete_tree_total(tmp_path) -> None:
    source = _source(tmp_path, target=97)
    assert source.drain_selection_manifest is not None
    assert source.drain_selection_manifest["actual_requests"] == 90
    assert source.drain_selection_manifest["target_error_requests"] == 7


def test_target_mode_requires_session_budget_equal_to_live_tree_count(
    tmp_path,
) -> None:
    dataset = _dataset()
    with pytest.raises(ValueError, match="num-conversations.*concurrency"):
        TrajectorySource(
            dataset_metadata=dataset,
            dataset_sampler=SequentialSampler(
                [conversation.conversation_id for conversation in dataset.conversations]
            ),
            concurrency=1,
            live_sessions_per_lane=3,
            expected_num_sessions=4,
            random_seed=42,
            start_min_ratio=0,
            start_max_ratio=0,
            drain_target_requests=100,
            selection_artifact_dir=tmp_path,
            cache_bust_enabled=True,
        )


def test_target_mode_rejects_nonzero_trajectory_snapshots(tmp_path) -> None:
    dataset = _dataset()
    with pytest.raises(ValueError, match="requires turn-zero trajectories"):
        TrajectorySource(
            dataset_metadata=dataset,
            dataset_sampler=SequentialSampler(
                [conversation.conversation_id for conversation in dataset.conversations]
            ),
            concurrency=1,
            expected_num_sessions=1,
            random_seed=42,
            start_min_ratio=0.25,
            start_max_ratio=0.75,
            drain_target_requests=100,
            selection_artifact_dir=tmp_path,
            cache_bust_enabled=True,
        )


def test_planned_request_count_includes_descendant_turns_and_excludes_warmup(
    tmp_path,
) -> None:
    dataset = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="root",
                turns=[
                    TurnMetadata(timestamp_ms=0.0),
                    TurnMetadata(timestamp_ms=10.0, branch_ids=["branch"]),
                    TurnMetadata(timestamp_ms=30.0),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id="branch",
                        child_conversation_ids=["child"],
                        mode=ConversationBranchMode.SPAWN,
                        start_timestamp_ms=11.0,
                    )
                ],
            ),
            ConversationMetadata(
                conversation_id="child",
                turns=[
                    TurnMetadata(timestamp_ms=11.0),
                    TurnMetadata(timestamp_ms=20.0),
                    TurnMetadata(timestamp_ms=25.0),
                ],
                is_root=False,
                parent_conversation_id="root",
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    source = TrajectorySource(
        dataset_metadata=dataset,
        dataset_sampler=SequentialSampler(["root"]),
        concurrency=1,
        expected_num_sessions=1,
        random_seed=42,
        start_min_ratio=0,
        start_max_ratio=0,
        warmup_requests_per_lane=1,
        drain_target_requests=5,
        selection_artifact_dir=tmp_path,
        cache_bust_enabled=True,
    )
    assert source.drain_selection_manifest is not None
    assert source.drain_selection_manifest["actual_requests"] == 5
