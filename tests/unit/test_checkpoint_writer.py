"""Unit tests for shard planning and the streaming safetensors writer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from model_merger.checkpoints import SafetensorsCheckpoint
from model_merger.checkpoints.sharding import plan_shards
from model_merger.checkpoints.writer import SafetensorsShardWriter
from model_merger.exceptions import MergeExecutionError


def test_plan_single_shard() -> None:
    plan = plan_shards([("a", 10), ("b", 20)], max_shard_bytes=1000)
    assert not plan.is_sharded
    assert plan.shards[0].filename == "model.safetensors"
    assert plan.total_bytes == 30


def test_plan_multiple_shards() -> None:
    plan = plan_shards([("a", 60), ("b", 60), ("c", 60)], max_shard_bytes=100)
    assert plan.is_sharded
    assert len(plan.shards) == 3
    assert plan.shards[0].filename == "model-00001-of-00003.safetensors"
    weight_map = plan.weight_map()
    assert weight_map["a"] == "model-00001-of-00003.safetensors"


def test_plan_oversized_tensor_gets_own_shard() -> None:
    plan = plan_shards([("big", 500), ("small", 10)], max_shard_bytes=100)
    assert plan.shards[0].keys == ("big",)


def test_plan_rejects_empty() -> None:
    with pytest.raises(ValueError):
        plan_shards([], max_shard_bytes=100)


def test_writer_writes_and_indexes_shards(tmp_path: Path) -> None:
    tensors = {"a": torch.ones(4), "b": torch.zeros(4), "c": torch.arange(4.0)}
    sizes = [(k, v.numel() * 4) for k, v in tensors.items()]
    plan = plan_shards(sorted(sizes), max_shard_bytes=20, base_name="model")
    writer = SafetensorsShardWriter(tmp_path, plan)
    for key in sorted(tensors):
        writer.add(key, tensors[key])
    written = writer.finalize()
    assert any(path.name.endswith(".index.json") for path in written)
    # Re-open through the index and confirm contents survive.
    index = next(path for path in written if path.name.endswith(".index.json"))
    with SafetensorsCheckpoint(index) as ckpt:
        assert set(ckpt.keys()) == {"a", "b", "c"}
        assert torch.equal(ckpt.get_tensor("c"), torch.arange(4.0))


def test_writer_single_shard_no_index(tmp_path: Path) -> None:
    plan = plan_shards([("a", 16)], max_shard_bytes=1000)
    writer = SafetensorsShardWriter(tmp_path, plan)
    writer.add("a", torch.ones(4))
    written = writer.finalize()
    assert len(written) == 1
    assert written[0].name == "model.safetensors"


def test_writer_rejects_unknown_key(tmp_path: Path) -> None:
    plan = plan_shards([("a", 16)], max_shard_bytes=1000)
    writer = SafetensorsShardWriter(tmp_path, plan)
    with pytest.raises(MergeExecutionError):
        writer.add("unknown", torch.ones(4))
