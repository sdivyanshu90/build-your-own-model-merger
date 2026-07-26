"""Unit tests for checkpoint readers and the open_checkpoint factory."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from model_merger.checkpoints import (
    HuggingFaceCheckpoint,
    PyTorchCheckpoint,
    SafetensorsCheckpoint,
    open_checkpoint,
)
from model_merger.checkpoints.base import element_size
from model_merger.exceptions import CheckpointError


def test_safetensors_roundtrip(tmp_path: Path) -> None:
    state = {"w": torch.randn(3, 4), "b": torch.zeros(4)}
    path = tmp_path / "model.safetensors"
    save_file(state, str(path))
    with SafetensorsCheckpoint(path) as ckpt:
        assert set(ckpt.keys()) == {"w", "b"}
        info = ckpt.tensor_info("w")
        assert info.shape == (3, 4)
        assert info.dtype == torch.float32
        assert torch.equal(ckpt.get_tensor("w"), state["w"])


def test_safetensors_missing_key(tmp_path: Path) -> None:
    save_file({"w": torch.zeros(2)}, str(tmp_path / "model.safetensors"))
    with SafetensorsCheckpoint(tmp_path / "model.safetensors") as ckpt:
        with pytest.raises(CheckpointError, match="not present"):
            ckpt.get_tensor("missing")


def test_tensor_info_num_bytes(tmp_path: Path) -> None:
    save_file({"w": torch.zeros(3, 4, dtype=torch.float32)}, str(tmp_path / "m.safetensors"))
    with SafetensorsCheckpoint(tmp_path / "m.safetensors") as ckpt:
        assert ckpt.tensor_info("w").num_bytes == 3 * 4 * element_size(torch.float32)


def test_pytorch_weights_only_safe_load(tmp_path: Path) -> None:
    path = tmp_path / "pytorch_model.bin"
    torch.save({"w": torch.arange(4.0)}, path)
    with PyTorchCheckpoint(path) as ckpt:
        assert torch.equal(ckpt.get_tensor("w"), torch.arange(4.0))


def test_pytorch_unwraps_state_dict(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pt"
    torch.save({"state_dict": {"w": torch.ones(2)}, "epoch": 3}, path)
    with PyTorchCheckpoint(path) as ckpt:
        assert ckpt.keys() == ["w"]


def test_huggingface_directory(tmp_path: Path, model_factory) -> None:
    model_dir = model_factory("m", seed=5)
    with HuggingFaceCheckpoint(model_dir) as ckpt:
        assert ckpt.config_info is not None
        assert ckpt.config_info.model_type == "llama"
        assert "model.embed_tokens.weight" in ckpt.keys()
        assert ckpt.backend_format == "safetensors"


def test_huggingface_rejects_adapter(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text("{}")
    save_file({"w": torch.zeros(2)}, str(tmp_path / "adapter_model.safetensors"))
    with pytest.raises(CheckpointError, match="adapter"):
        HuggingFaceCheckpoint(tmp_path)


def test_open_checkpoint_dispatch(tmp_path: Path, model_factory) -> None:
    model_dir = model_factory("m", seed=1)
    assert open_checkpoint(model_dir).format == "huggingface"
    assert open_checkpoint(model_dir / "model.safetensors").format == "safetensors"


def test_open_checkpoint_missing(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="does not exist"):
        open_checkpoint(tmp_path / "nope")


def test_open_checkpoint_unsupported_type(tmp_path: Path) -> None:
    weird = tmp_path / "model.weird"
    weird.write_text("x")
    with pytest.raises(CheckpointError, match="unsupported"):
        open_checkpoint(weird)
