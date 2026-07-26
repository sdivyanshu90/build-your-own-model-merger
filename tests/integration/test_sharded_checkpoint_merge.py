"""Sharded output and sharded input round-trips."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from model_merger import merge_models
from model_merger.checkpoints import SafetensorsCheckpoint, open_checkpoint
from model_merger.config.models import AlgorithmConfig, MergeConfig, ModelRef, OutputConfig

pytestmark = pytest.mark.integration


def _sharded_config(models, out, max_shard: str) -> MergeConfig:
    return MergeConfig(
        algorithm=AlgorithmConfig(type="uniform_soup"),
        models=[ModelRef(path=str(path)) for path in models],
        output=OutputConfig(path=str(out), overwrite=True, max_shard_size=max_shard),
    )


def test_small_shard_size_produces_index(three_models, tmp_path: Path) -> None:
    out = tmp_path / "merged"
    report = merge_models(_sharded_config(three_models, out, "1KB"))
    assert len(report.output_shards) > 1
    assert (out / "model.safetensors.index.json").is_file()


def test_sharded_output_reopens_and_matches(three_models, tmp_path: Path) -> None:
    out = tmp_path / "merged"
    merge_models(_sharded_config(three_models, out, "1KB"))
    index = out / "model.safetensors.index.json"
    sources = [open_checkpoint(path) for path in three_models]
    with SafetensorsCheckpoint(index) as merged:
        key = "lm_head.weight"
        expected = torch.stack([ckpt.get_tensor(key) for ckpt in sources]).mean(dim=0)
        assert torch.allclose(merged.get_tensor(key), expected, atol=1e-6)
    for ckpt in sources:
        ckpt.close()


def test_merge_from_sharded_input(three_models, tmp_path: Path) -> None:
    # First produce a sharded model, then feed it back in as a source.
    sharded = tmp_path / "sharded"
    merge_models(_sharded_config(three_models, sharded, "1KB"))

    out = tmp_path / "merged2"
    config = MergeConfig(
        algorithm=AlgorithmConfig(type="uniform_soup"),
        models=[ModelRef(path=str(sharded)), ModelRef(path=str(three_models[0]))],
        output=OutputConfig(path=str(out), overwrite=True),
    )
    report = merge_models(config)
    assert report.verification.passed
