"""End-to-end safetensors merges through the public API."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from model_merger import merge_models, verify_output
from model_merger.checkpoints import open_checkpoint

pytestmark = pytest.mark.integration


def test_uniform_merge_produces_verified_output(three_models, make_config, tmp_path: Path) -> None:
    out = tmp_path / "merged"
    report = merge_models(make_config(three_models, out))
    assert report.verification.passed
    assert (out / "model.safetensors").is_file()
    assert (out / "merge_report.json").is_file()
    assert verify_output(out).passed


def test_uniform_merge_values_are_mean(three_models, make_config, tmp_path: Path) -> None:
    out = tmp_path / "merged"
    merge_models(make_config(three_models, out))
    sources = [open_checkpoint(path) for path in three_models]
    merged = open_checkpoint(out)
    try:
        key = "model.layers.0.mlp.up_proj.weight"
        expected = torch.stack([ckpt.get_tensor(key) for ckpt in sources]).mean(dim=0)
        assert torch.allclose(merged.get_tensor(key), expected, atol=1e-6)
    finally:
        for ckpt in sources:
            ckpt.close()
        merged.close()


def test_output_dtype_preserved(three_models, make_config, tmp_path: Path) -> None:
    out = tmp_path / "merged"
    merge_models(make_config(three_models, out))
    with open_checkpoint(out) as ckpt:
        info = ckpt.tensor_info("model.norm.weight")
        assert info.dtype == torch.float32


def test_non_float_buffer_preserved(three_models, make_config, tmp_path: Path) -> None:
    out = tmp_path / "merged"
    merge_models(make_config(three_models, out))
    with open_checkpoint(out) as ckpt:
        buffer = ckpt.get_tensor("model.layers.0.self_attn.rotary_emb.inv_freq_ids")
        assert buffer.dtype == torch.int64
        assert torch.equal(buffer, torch.arange(4, dtype=torch.int64))


def test_slerp_merge(two_models, make_config, tmp_path: Path) -> None:
    out = tmp_path / "merged"
    report = merge_models(make_config(two_models, out, algorithm="slerp", t=0.5))
    assert report.verification.passed
    assert report.algorithm == "slerp"


def test_existing_output_not_overwritten(three_models, make_config, tmp_path: Path) -> None:
    out = tmp_path / "merged"
    merge_models(make_config(three_models, out, overwrite=True))
    from model_merger.exceptions import OutputExistsError

    with pytest.raises(OutputExistsError):
        merge_models(make_config(three_models, out, overwrite=False))
