"""Abstract checkpoint interface and lightweight tensor descriptors.

A :class:`Checkpoint` exposes a model's tensors *lazily*: you can enumerate keys
and inspect each tensor's shape/dtype (:meth:`tensor_info`) without materializing
any data, and load exactly one tensor at a time (:meth:`get_tensor`).  This lazy
contract is what lets the executor merge tensor-at-a-time with bounded memory.

Concrete implementations live in sibling modules (safetensors, pytorch,
huggingface).  Use :func:`model_merger.checkpoints.open_checkpoint` to get the
right one for a path.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import torch

__all__ = ["TensorInfo", "Checkpoint", "element_size"]

# Cache element sizes so we never allocate a probe tensor per query.
_ELEMENT_SIZE: dict[torch.dtype, int] = {}


def element_size(dtype: torch.dtype) -> int:
    """Return the byte width of one element of ``dtype``."""

    cached = _ELEMENT_SIZE.get(dtype)
    if cached is None:
        cached = torch.empty((), dtype=dtype).element_size()
        _ELEMENT_SIZE[dtype] = cached
    return cached


@dataclass(frozen=True)
class TensorInfo:
    """Shape/dtype metadata for a single tensor, without its data."""

    key: str
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def num_elements(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def num_bytes(self) -> int:
        return self.num_elements * element_size(self.dtype)


class Checkpoint(ABC):
    """Read-only, lazily-loaded view of a model checkpoint."""

    #: Filesystem path this checkpoint was opened from.
    path: Path
    #: Short format identifier: ``"safetensors"``, ``"pytorch"``, ``"huggingface"``.
    format: str

    @abstractmethod
    def keys(self) -> list[str]:
        """Return all tensor keys in deterministic (sorted) order."""

    @abstractmethod
    def tensor_info(self, key: str) -> TensorInfo:
        """Return shape/dtype for ``key`` without loading the tensor."""

    @abstractmethod
    def get_tensor(self, key: str) -> torch.Tensor:
        """Load and return the tensor for ``key`` (CPU tensor)."""

    def raw_metadata(self) -> dict[str, str]:
        """Return container-level string metadata (empty if none)."""

        return {}

    def __contains__(self, key: str) -> bool:
        return key in set(self.keys())

    def infos(self) -> Iterator[TensorInfo]:
        """Yield :class:`TensorInfo` for every key, in key order."""

        for key in self.keys():
            yield self.tensor_info(key)

    def close(self) -> None:  # noqa: B027 - optional override; base default is a no-op
        """Release any open file handles.  Safe to call multiple times."""

    def __enter__(self) -> Checkpoint:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
