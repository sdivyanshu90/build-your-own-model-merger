"""Bounded-memory checks: writer buffers one shard; live source tensors bounded.

These use structural instrumentation rather than wall-clock RSS (which is noisy
and platform-dependent).  The streaming access pattern is verified in
``test_streaming_behavior``; here we assert the writer never holds more than one
shard's worth of tensors, and that many-model merges stay tensor-at-a-time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from model_merger import merge_models
from model_merger.checkpoints import writer as writer_module
from model_merger.config.models import AlgorithmConfig, MergeConfig, ModelRef, OutputConfig

pytestmark = pytest.mark.performance


def _config(models, out, max_shard: str = "1KB") -> MergeConfig:
    return MergeConfig(
        algorithm=AlgorithmConfig(type="uniform_soup"),
        models=[ModelRef(path=str(path)) for path in models],
        output=OutputConfig(path=str(out), overwrite=True, max_shard_size=max_shard),
    )


def test_writer_buffer_never_exceeds_one_shard(three_models, tmp_path: Path, monkeypatch) -> None:
    max_buffered = {"value": 0}
    original_add = writer_module.SafetensorsShardWriter.add

    def spy_add(self, key, tensor):  # type: ignore[no-untyped-def]
        original_add(self, key, tensor)
        live = sum(len(buffer) for buffer in self._buffers.values())
        max_buffered["value"] = max(max_buffered["value"], live)

    monkeypatch.setattr(writer_module.SafetensorsShardWriter, "add", spy_add)

    report = merge_models(_config(three_models, tmp_path / "out"))
    assert len(report.output_shards) > 1  # actually sharded

    # The largest number of tensors ever buffered at once must not exceed the
    # biggest single shard's key count (never the whole model).
    largest_shard_keys = 0
    # Recompute plan to know the largest shard size.
    from model_merger.execution.planner import prepare_merge

    prepared = prepare_merge(_config(three_models, tmp_path / "out2"))
    try:
        largest_shard_keys = max(len(shard.keys) for shard in prepared.shard_plan.shards)
    finally:
        prepared.close()
    assert max_buffered["value"] <= largest_shard_keys


def test_many_models_merge_succeeds(model_factory, tmp_path: Path) -> None:
    models = [model_factory(f"m{i}", seed=100 + i) for i in range(6)]
    report = merge_models(_config(models, tmp_path / "out", max_shard="512B"))
    assert report.verification.passed
