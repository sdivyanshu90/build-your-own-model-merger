"""Streaming checkpoint writers.

The safetensors writer buffers **one shard at a time**: tensors are added in the
planned key order, and each shard is flushed to disk the moment its last key
arrives, then dropped from memory.  Peak writer memory is therefore ~one shard
(<= ``max_shard_size``), independent of model size.

Writers only ever write inside a caller-provided directory (the atomic staging
directory).  Shard/ancillary filenames are validated to be safe relative members
so nothing escapes it.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from ..exceptions import MergeExecutionError
from ..utilities.filesystem import is_safe_relative_member
from .sharding import ShardPlan

__all__ = ["SafetensorsShardWriter", "PyTorchStateDictWriter", "copy_ancillary_file"]


def copy_ancillary_file(source: Path, dest_dir: Path, *, name: str | None = None) -> Path:
    """Copy an ancillary file into ``dest_dir`` with a validated name.

    Raises:
        MergeExecutionError: if the destination name would escape ``dest_dir``.
    """

    filename = name or source.name
    if not is_safe_relative_member(filename):
        raise MergeExecutionError(f"unsafe ancillary filename: {filename!r}")
    dest = dest_dir / filename
    shutil.copyfile(source, dest)
    return dest


class SafetensorsShardWriter:
    """Write merged tensors to safetensors shards, one shard buffered at a time."""

    def __init__(
        self,
        out_dir: Path,
        plan: ShardPlan,
        *,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self._dir = out_dir
        self._plan = plan
        for shard in plan.shards:
            if not is_safe_relative_member(shard.filename):
                raise MergeExecutionError(f"unsafe shard filename: {shard.filename!r}")
        self._key_to_file = plan.weight_map()
        self._buffers: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        self._remaining: dict[str, set[str]] = {
            shard.filename: set(shard.keys) for shard in plan.shards
        }
        # safetensors metadata must be str->str; "format: pt" aids interop.
        self._metadata = {"format": "pt", **(metadata or {})}
        self.written_files: list[Path] = []

    def add(self, key: str, tensor: torch.Tensor) -> None:
        """Buffer a merged tensor; flush its shard when complete."""

        filename = self._key_to_file.get(key)
        if filename is None:
            raise MergeExecutionError(f"tensor {key!r} is not in the shard plan")
        self._buffers[filename][key] = tensor.contiguous()
        self._remaining[filename].discard(key)
        if not self._remaining[filename]:
            self._flush(filename)

    def _flush(self, filename: str) -> None:
        path = self._dir / filename
        try:
            save_file(self._buffers[filename], str(path), metadata=self._metadata)
        except Exception as error:
            raise MergeExecutionError(f"failed to write shard {path}: {error}") from error
        del self._buffers[filename]
        self.written_files.append(path)

    def finalize(self) -> list[Path]:
        """Assert all shards were written and emit the index if sharded."""

        missing = {name: keys for name, keys in self._remaining.items() if keys}
        if missing:
            raise MergeExecutionError(f"shards missing tensors at finalize: {missing}")
        if self._plan.is_sharded:
            index_path = self._dir / self._plan.index_filename
            index_path.write_text(
                json.dumps(self._plan.index_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            self.written_files.append(index_path)
        return list(self.written_files)


class PyTorchStateDictWriter:
    """Write all merged tensors to a single ``pytorch_model.bin``.

    Unlike the safetensors writer this holds the whole state dict in memory before
    saving (a pickle archive is written atomically as one object), so it is bounded
    by the full model size.  Prefer safetensors output for large models.
    """

    def __init__(self, out_dir: Path, *, filename: str = "pytorch_model.bin") -> None:
        if not is_safe_relative_member(filename):
            raise MergeExecutionError(f"unsafe output filename: {filename!r}")
        self._dir = out_dir
        self._filename = filename
        self._state: dict[str, torch.Tensor] = {}
        self.written_files: list[Path] = []

    def add(self, key: str, tensor: torch.Tensor) -> None:
        self._state[key] = tensor.contiguous()

    def finalize(self) -> list[Path]:
        path = self._dir / self._filename
        try:
            torch.save(self._state, path)
        except Exception as error:
            raise MergeExecutionError(f"failed to write {path}: {error}") from error
        self._state = {}
        self.written_files.append(path)
        return list(self.written_files)
