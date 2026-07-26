"""SLERP and linear interpolation strategies (two-model merges).

SLERP operates **per tensor**: each tensor is flattened to a vector,
interpolated on the geodesic between the two models' versions, and reshaped.
Per-tensor (rather than global-vector) SLERP is a deliberate choice -- it keeps
memory bounded to a couple of tensors at a time and matches the de-facto
convention used by model-merging tools.  The rationale and alternatives are in
``docs/adr/0003-numerical-precision-policy.md`` and ``docs/slerp.md``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..exceptions import ConfigurationError
from .base import MergeStrategy
from .numerical import DEFAULT_SLERP_EPS, lerp, slerp_vectors

__all__ = ["Slerp", "LinearInterpolation", "validate_interpolation_t"]


def validate_interpolation_t(t: float, *, allow_extrapolation: bool) -> float:
    """Validate an interpolation coefficient.

    Args:
        t: The coefficient.
        allow_extrapolation: If false, ``t`` must lie in ``[0, 1]``.

    Raises:
        ConfigurationError: if ``t`` is non-finite, or outside ``[0, 1]`` while
            extrapolation is disabled.
    """

    value = float(t)
    if value != value or value in (float("inf"), float("-inf")):
        raise ConfigurationError(f"interpolation t must be finite, got {t!r}")
    if not allow_extrapolation and not (0.0 <= value <= 1.0):
        raise ConfigurationError(
            f"interpolation t={value} outside [0, 1]; set allow_extrapolation=true to permit it"
        )
    return value


class LinearInterpolation(MergeStrategy):
    """Plain LERP ``(1 - t) * A + t * B`` between two models."""

    name = "linear"
    required_models = 2

    def __init__(self, t: float, *, allow_extrapolation: bool = False) -> None:
        self.t = validate_interpolation_t(t, allow_extrapolation=allow_extrapolation)

    def merge(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        return lerp(tensors[0], tensors[1], self.t)


class Slerp(MergeStrategy):
    """Spherical linear interpolation between two models, per tensor.

    Args:
        t: Interpolation coefficient.
        dot_threshold: Absolute-cosine threshold above which the merge falls
            back to LERP (near-parallel / near-antiparallel guard).
        allow_extrapolation: Permit ``t`` outside ``[0, 1]``.
        eps: Zero-norm threshold below which a tensor triggers the LERP fallback.
    """

    name = "slerp"
    required_models = 2

    def __init__(
        self,
        t: float,
        *,
        dot_threshold: float = 0.9995,
        allow_extrapolation: bool = False,
        eps: float = DEFAULT_SLERP_EPS,
    ) -> None:
        self.t = validate_interpolation_t(t, allow_extrapolation=allow_extrapolation)
        if not (0.0 < dot_threshold <= 1.0):
            raise ConfigurationError(f"dot_threshold must be in (0, 1], got {dot_threshold!r}")
        self.dot_threshold = float(dot_threshold)
        self.eps = float(eps)

    def merge(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        a, b = tensors[0], tensors[1]
        original_shape = a.shape
        flat_a = a.reshape(-1)
        flat_b = b.reshape(-1)
        merged = slerp_vectors(
            flat_a,
            flat_b,
            self.t,
            dot_threshold=self.dot_threshold,
            eps=self.eps,
        )
        return merged.reshape(original_shape)
