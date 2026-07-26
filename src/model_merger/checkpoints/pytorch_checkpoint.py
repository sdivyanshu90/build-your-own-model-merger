"""PyTorch (pickle-backed) checkpoint reader with a strict trust boundary.

``torch.load`` deserializes with pickle, which can execute arbitrary code from a
malicious file.  This reader defends the boundary:

* It loads with ``weights_only=True`` by default (torch >= 2.0), which restricts
  unpickling to tensors and primitive containers -- safe for untrusted files.
* If ``weights_only=True`` fails (a legacy checkpoint needs full unpickling), it
  raises :class:`UnsafeCheckpointError` *unless* the caller explicitly opts in via
  ``allow_unsafe=True``, in which case it retries with a loud warning.

Unlike safetensors, a pickle archive cannot be read one tensor at a time, so the
whole state dict is loaded once and served from memory.  This is bounded by the
size of a single model; for strictly bounded multi-model memory, use safetensors
inputs (documented in ``docs/memory-and-performance.md``).
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..exceptions import CheckpointError, UnsafeCheckpointError
from ..logging import get_logger
from .base import Checkpoint, TensorInfo

__all__ = ["PyTorchCheckpoint"]

_LOGGER = get_logger(__name__)


class PyTorchCheckpoint(Checkpoint):
    """Read a ``.bin`` / ``.pt`` state dict, defending the pickle boundary."""

    def __init__(self, path: str | Path, *, allow_unsafe: bool = False) -> None:
        self.path = Path(path)
        self.format = "pytorch"
        if not self.path.is_file():
            raise CheckpointError(f"pytorch checkpoint not found: {self.path}")
        self._state = self._load(allow_unsafe=allow_unsafe)

    def _load(self, *, allow_unsafe: bool) -> dict[str, torch.Tensor]:
        try:
            raw = torch.load(self.path, map_location="cpu", weights_only=True)
        except Exception as safe_error:
            if not allow_unsafe:
                raise UnsafeCheckpointError(
                    f"{self.path} could not be loaded safely (weights_only=True failed: "
                    f"{safe_error}). It may be a pickle-backed checkpoint that executes code. "
                    f"Re-run with allow_unsafe_pytorch=true / --allow-unsafe only if you trust it."
                ) from safe_error
            _LOGGER.warning(
                "loading %s with weights_only=False (UNSAFE pickle execution) at user request",
                self.path,
            )
            raw = torch.load(self.path, map_location="cpu", weights_only=False)

        state = self._extract_state_dict(raw)
        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                _LOGGER.debug("skipping non-tensor entry %r (%s)", key, type(value).__name__)
                continue
            cleaned[str(key)] = value
        if not cleaned:
            raise CheckpointError(f"no tensors found in pytorch checkpoint {self.path}")
        return cleaned

    @staticmethod
    def _extract_state_dict(raw: object) -> dict[str, object]:
        """Unwrap common checkpoint containers to a flat ``{key: tensor}`` map."""

        if isinstance(raw, dict):
            for wrapper_key in ("state_dict", "model_state_dict", "model"):
                inner = raw.get(wrapper_key)
                if isinstance(inner, dict):
                    return inner
            return raw
        type_name = getattr(raw, "__class__", type(raw)).__name__
        raise CheckpointError(f"unsupported pytorch checkpoint structure: {type_name}")

    def keys(self) -> list[str]:
        return sorted(self._state)

    def tensor_info(self, key: str) -> TensorInfo:
        tensor = self._get(key)
        return TensorInfo(key=key, shape=tuple(tensor.shape), dtype=tensor.dtype)

    def get_tensor(self, key: str) -> torch.Tensor:
        return self._get(key)

    def _get(self, key: str) -> torch.Tensor:
        tensor = self._state.get(key)
        if tensor is None:
            raise CheckpointError(f"tensor {key!r} not present in {self.path}")
        return tensor

    def close(self) -> None:
        self._state = {}
