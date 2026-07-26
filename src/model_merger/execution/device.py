"""Device resolution and safe tensor transfers."""

from __future__ import annotations

import torch

from ..exceptions import MergeExecutionError

__all__ = ["resolve_device", "to_device"]


def resolve_device(spec: str) -> torch.device:
    """Resolve a device spec to a concrete :class:`torch.device`.

    Args:
        spec: ``"cpu"``, ``"cuda"``, ``"cuda:N"``, or ``"auto"`` (CUDA if
            available, otherwise CPU).

    Raises:
        MergeExecutionError: if a CUDA device is requested but unavailable.
    """

    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cpu":
        return torch.device("cpu")
    if spec.startswith("cuda"):
        if not torch.cuda.is_available():
            raise MergeExecutionError(
                f"device {spec!r} requested but CUDA is not available; use 'cpu'"
            )
        return torch.device(spec)
    raise MergeExecutionError(f"unrecognized device spec: {spec!r}")


def to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a tensor to ``device`` (no-op if already there)."""

    if tensor.device == device:
        return tensor
    return tensor.to(device)
