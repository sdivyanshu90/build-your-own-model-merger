"""Hugging Face model-directory reader.

Wraps a directory containing weights plus ``config.json`` and tokenizer files.
Weight access is delegated to :class:`SafetensorsCheckpoint` (preferred) or
:class:`PyTorchCheckpoint`, chosen by the files present.  The directory's config
and ancillary files are exposed for compatibility validation and for the
ancillary reconciliation strategy.

Adapter-only checkpoints (LoRA) are detected and rejected: merging adapters is a
different operation than merging full weights and is out of scope (documented in
``docs/limitations.md``).
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..exceptions import CheckpointError
from .base import Checkpoint, TensorInfo
from .metadata import (
    ModelConfigInfo,
    discover_ancillary_files,
    load_json,
    parse_config_info,
)
from .pytorch_checkpoint import PyTorchCheckpoint
from .safetensors_checkpoint import SafetensorsCheckpoint

__all__ = ["HuggingFaceCheckpoint"]


class HuggingFaceCheckpoint(Checkpoint):
    """Read a Hugging Face model directory (weights + config + tokenizer)."""

    def __init__(self, path: str | Path, *, allow_unsafe: bool = False) -> None:
        self.path = Path(path)
        self.format = "huggingface"
        if not self.path.is_dir():
            raise CheckpointError(f"hugging face checkpoint is not a directory: {self.path}")
        self._reject_adapter_only()
        self._backend = self._open_backend(allow_unsafe=allow_unsafe)
        self._config_info = self._load_config_info()
        self.ancillary_files = discover_ancillary_files(self.path)

    def _reject_adapter_only(self) -> None:
        adapter_config = self.path / "adapter_config.json"
        has_adapter_weights = (self.path / "adapter_model.safetensors").is_file() or (
            self.path / "adapter_model.bin"
        ).is_file()
        if adapter_config.is_file() and has_adapter_weights:
            raise CheckpointError(
                f"{self.path} looks like a LoRA/adapter checkpoint (adapter_config.json present); "
                f"merging adapters is not supported -- merge the base models instead"
            )

    def _open_backend(self, *, allow_unsafe: bool) -> Checkpoint:
        st_index = self.path / "model.safetensors.index.json"
        st_single = self.path / "model.safetensors"
        pt_index = self.path / "pytorch_model.bin.index.json"
        pt_single = self.path / "pytorch_model.bin"
        if st_index.is_file():
            return SafetensorsCheckpoint(st_index)
        if st_single.is_file():
            return SafetensorsCheckpoint(st_single)
        # Any other single safetensors file (e.g. custom-named export).
        other_safetensors = sorted(self.path.glob("*.safetensors"))
        if other_safetensors and not pt_index.is_file() and not pt_single.is_file():
            return SafetensorsCheckpoint(other_safetensors[0])
        if pt_index.is_file():
            raise CheckpointError(
                f"{self.path} uses a sharded pytorch_model.bin index; convert to safetensors "
                f"or provide a single-file checkpoint"
            )
        if pt_single.is_file():
            return PyTorchCheckpoint(pt_single, allow_unsafe=allow_unsafe)
        raise CheckpointError(f"no recognized weight files found in {self.path}")

    def _load_config_info(self) -> ModelConfigInfo | None:
        config_path = self.path / "config.json"
        if not config_path.is_file():
            return None
        return parse_config_info(load_json(config_path))

    @property
    def config_info(self) -> ModelConfigInfo | None:
        return self._config_info

    @property
    def backend_format(self) -> str:
        return self._backend.format

    # --- Checkpoint interface (delegates to the weight backend) ---

    def keys(self) -> list[str]:
        return self._backend.keys()

    def tensor_info(self, key: str) -> TensorInfo:
        return self._backend.tensor_info(key)

    def get_tensor(self, key: str) -> torch.Tensor:
        return self._backend.get_tensor(key)

    def raw_metadata(self) -> dict[str, str]:
        return self._backend.raw_metadata()

    def close(self) -> None:
        self._backend.close()
