"""Safetensors checkpoint reader (single-file and sharded).

Safetensors is the preferred format: it is not pickle-backed (no code execution
risk) and supports true lazy, per-tensor reads via ``safe_open``.  This reader
opens each shard once, caches the handle, and serves ``tensor_info`` from the
header (no data read) and ``get_tensor`` for a single tensor at a time.

A path may be a single ``.safetensors`` file or a ``*.index.json`` shard index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from ..exceptions import CheckpointError
from .base import Checkpoint, TensorInfo

__all__ = ["SafetensorsCheckpoint", "SAFETENSORS_DTYPES"]

#: Map safetensors dtype codes to torch dtypes.
SAFETENSORS_DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


class SafetensorsCheckpoint(Checkpoint):
    """Lazily read a single or sharded safetensors checkpoint."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.format = "safetensors"
        self._handles: dict[Path, Any] = {}
        self._index_metadata: dict[str, str] = {}
        if not self.path.exists():
            raise CheckpointError(f"safetensors checkpoint not found: {self.path}")

        if self.path.suffix == ".json" or self.path.name.endswith(".index.json"):
            self._weight_map = self._load_index(self.path)
        else:
            handle = self._handle(self.path)
            self._weight_map = dict.fromkeys(handle.keys(), self.path)

    def _load_index(self, index_path: Path) -> dict[str, Path]:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CheckpointError(f"failed to read shard index {index_path}: {error}") from error
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise CheckpointError(f"shard index {index_path} has no 'weight_map'")
        metadata = index.get("metadata")
        if isinstance(metadata, dict):
            self._index_metadata = {str(k): str(v) for k, v in metadata.items()}
        base = index_path.parent
        resolved: dict[str, Path] = {}
        for key, filename in weight_map.items():
            shard = base / filename
            if not shard.is_file():
                raise CheckpointError(f"shard file referenced by index is missing: {shard}")
            resolved[str(key)] = shard
        return resolved

    def _handle(self, shard: Path) -> Any:
        handle = self._handles.get(shard)
        if handle is None:
            try:
                handle = safe_open(str(shard), framework="pt", device="cpu")
            except Exception as error:  # safetensors raises assorted errors
                raise CheckpointError(
                    f"failed to open safetensors file {shard}: {error}"
                ) from error
            self._handles[shard] = handle
        return handle

    def keys(self) -> list[str]:
        return sorted(self._weight_map)

    def tensor_info(self, key: str) -> TensorInfo:
        shard = self._shard_for(key)
        try:
            sliced = self._handle(shard).get_slice(key)
            shape = tuple(int(dim) for dim in sliced.get_shape())
            dtype_code = sliced.get_dtype()
        except Exception as error:
            raise CheckpointError(
                f"failed to read info for {key!r} from {shard}: {error}"
            ) from error
        dtype = SAFETENSORS_DTYPES.get(dtype_code)
        if dtype is None:
            raise CheckpointError(f"unsupported safetensors dtype {dtype_code!r} for key {key!r}")
        return TensorInfo(key=key, shape=shape, dtype=dtype)

    def get_tensor(self, key: str) -> torch.Tensor:
        shard = self._shard_for(key)
        try:
            tensor: torch.Tensor = self._handle(shard).get_tensor(key)
        except Exception as error:
            raise CheckpointError(f"failed to load tensor {key!r} from {shard}: {error}") from error
        return tensor

    def _shard_for(self, key: str) -> Path:
        shard = self._weight_map.get(key)
        if shard is None:
            raise CheckpointError(f"tensor {key!r} not present in {self.path}")
        return shard

    def raw_metadata(self) -> dict[str, str]:
        metadata = dict(self._index_metadata)
        for shard in sorted(set(self._weight_map.values())):
            shard_meta = self._handle(shard).metadata()
            if shard_meta:
                metadata.update({str(k): str(v) for k, v in shard_meta.items()})
        return metadata

    def shard_files(self) -> list[Path]:
        """Return the distinct shard files backing this checkpoint."""

        return sorted(set(self._weight_map.values()))

    def close(self) -> None:
        # Dropping references lets safetensors release the mmap/file handle.
        self._handles.clear()
