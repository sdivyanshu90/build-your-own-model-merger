"""Checkpoint readers/writers and the :func:`open_checkpoint` dispatch factory."""

from __future__ import annotations

from pathlib import Path

from ..exceptions import CheckpointError
from .base import Checkpoint, TensorInfo, element_size
from .huggingface_checkpoint import HuggingFaceCheckpoint
from .metadata import ModelConfigInfo, discover_ancillary_files, parse_config_info
from .pytorch_checkpoint import PyTorchCheckpoint
from .safetensors_checkpoint import SafetensorsCheckpoint
from .sharding import ShardPlan, ShardSpec, plan_shards
from .writer import PyTorchStateDictWriter, SafetensorsShardWriter, copy_ancillary_file

__all__ = [
    "Checkpoint",
    "TensorInfo",
    "element_size",
    "SafetensorsCheckpoint",
    "PyTorchCheckpoint",
    "HuggingFaceCheckpoint",
    "ModelConfigInfo",
    "discover_ancillary_files",
    "parse_config_info",
    "ShardPlan",
    "ShardSpec",
    "plan_shards",
    "SafetensorsShardWriter",
    "PyTorchStateDictWriter",
    "copy_ancillary_file",
    "open_checkpoint",
]


def open_checkpoint(path: str | Path, *, allow_unsafe: bool = False) -> Checkpoint:
    """Open a checkpoint, selecting the reader by path shape.

    * A directory -> :class:`HuggingFaceCheckpoint`.
    * ``*.safetensors`` or a ``*.index.json`` -> :class:`SafetensorsCheckpoint`.
    * ``*.bin`` / ``*.pt`` / ``*.pth`` -> :class:`PyTorchCheckpoint`.

    Raises:
        CheckpointError: for a missing path or an unrecognized file type.
    """

    resolved = Path(path)
    if not resolved.exists():
        raise CheckpointError(f"checkpoint path does not exist: {resolved}")
    if resolved.is_dir():
        return HuggingFaceCheckpoint(resolved, allow_unsafe=allow_unsafe)
    name = resolved.name
    if resolved.suffix == ".safetensors" or name.endswith(".safetensors.index.json"):
        return SafetensorsCheckpoint(resolved)
    if name.endswith(".index.json"):
        return SafetensorsCheckpoint(resolved)
    if resolved.suffix in {".bin", ".pt", ".pth"}:
        return PyTorchCheckpoint(resolved, allow_unsafe=allow_unsafe)
    raise CheckpointError(
        f"unsupported checkpoint file type: {resolved} "
        f"(expected a directory, .safetensors, index .json, or .bin/.pt)"
    )
