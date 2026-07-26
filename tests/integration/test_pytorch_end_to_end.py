"""End-to-end merges of pickle-backed PyTorch checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from model_merger import merge_models
from model_merger.config.models import AlgorithmConfig, MergeConfig, ModelRef, OutputConfig

pytestmark = pytest.mark.integration


def _write_bin(path: Path, seed: int) -> Path:
    generator = torch.Generator().manual_seed(seed)
    state = {
        "layer.weight": torch.randn(4, 4, generator=generator),
        "layer.bias": torch.randn(4, generator=generator),
        "counter": torch.arange(3, dtype=torch.int64),
    }
    torch.save(state, path)
    return path


def test_merge_pytorch_files(tmp_path: Path) -> None:
    a = _write_bin(tmp_path / "a.bin", 1)
    b = _write_bin(tmp_path / "b.bin", 2)
    out = tmp_path / "merged"
    config = MergeConfig(
        algorithm=AlgorithmConfig(type="uniform_soup"),
        models=[ModelRef(path=str(a)), ModelRef(path=str(b))],
        output=OutputConfig(path=str(out), overwrite=True),
    )
    report = merge_models(config)
    assert report.verification.passed
    assert (out / "model.safetensors").is_file()


def test_pytorch_output_format(tmp_path: Path) -> None:
    a = _write_bin(tmp_path / "a.bin", 1)
    b = _write_bin(tmp_path / "b.bin", 2)
    out = tmp_path / "merged_pt"
    config = MergeConfig(
        algorithm=AlgorithmConfig(type="uniform_soup"),
        models=[ModelRef(path=str(a)), ModelRef(path=str(b))],
        output=OutputConfig(path=str(out), format="pytorch", overwrite=True),
    )
    report = merge_models(config)
    assert report.verification.passed
    assert (out / "pytorch_model.bin").is_file()
