"""Low-level numerical primitives shared by the merge algorithms.

This module is pure tensor math with no I/O.  It is the most heavily tested part
of the project because correctness and numerical stability are the top two
priorities.  Every function assumes its inputs have already been cast to a
suitable *compute* dtype (typically ``float32``) by the caller; the precision
policy (:mod:`model_merger.policies.precision`) owns that decision.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..exceptions import NumericalError

__all__ = [
    "weighted_sum",
    "slerp_vectors",
    "lerp",
    "assert_finite",
    "count_nonfinite",
    "DEFAULT_SLERP_EPS",
]

#: Norm below which a vector is treated as the zero vector (direction undefined).
DEFAULT_SLERP_EPS = 1e-8


def count_nonfinite(tensor: torch.Tensor) -> tuple[int, int]:
    """Return ``(nan_count, inf_count)`` for a tensor."""

    nan = int(torch.isnan(tensor).sum().item())
    inf = int(torch.isinf(tensor).sum().item())
    return nan, inf


def assert_finite(tensor: torch.Tensor, *, context: str) -> torch.Tensor:
    """Return ``tensor`` unchanged, or raise if it contains NaN/Inf.

    Args:
        tensor: The tensor to validate.
        context: Human-readable context (e.g. the tensor key) included in the
            error message.

    Raises:
        NumericalError: if any element is non-finite.
    """

    if not torch.isfinite(tensor).all():
        nan, inf = count_nonfinite(tensor)
        raise NumericalError(
            f"non-finite values in {context}: {nan} NaN, {inf} Inf "
            f"(out of {tensor.numel()} elements)"
        )
    return tensor


def weighted_sum(tensors: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    """Compute ``sum_i weights[i] * tensors[i]`` with in-place accumulation.

    Accumulation reuses a single buffer (``add_`` with an ``alpha`` scalar) so
    peak memory is one output-sized tensor rather than one per source.  Inputs
    must share shape and dtype and be non-empty.

    Raises:
        ValueError: on length mismatch or empty input.
    """

    if not tensors:
        raise ValueError("weighted_sum requires at least one tensor")
    if len(tensors) != len(weights):
        raise ValueError(f"tensor/weight length mismatch: {len(tensors)} vs {len(weights)}")
    acc = tensors[0].mul(float(weights[0]))
    for tensor, weight in zip(tensors[1:], weights[1:], strict=True):
        acc.add_(tensor, alpha=float(weight))
    return acc


def lerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """Linear interpolation ``(1 - t) * v0 + t * v1``.

    Implemented as ``v0 + t * (v1 - v0)`` (torch.lerp) which is exact at the
    endpoints regardless of ``t`` rounding.
    """

    return torch.lerp(v0, v1, float(t))


def slerp_vectors(
    v0: torch.Tensor,
    v1: torch.Tensor,
    t: float,
    *,
    dot_threshold: float = 0.9995,
    eps: float = DEFAULT_SLERP_EPS,
) -> torch.Tensor:
    """Spherical linear interpolation between two flat vectors.

    Implements::

        SLERP(v0, v1; t) = sin((1 - t) * Omega) / sin(Omega) * v0
                         + sin(t * Omega)       / sin(Omega) * v1

    with ``Omega = arccos(<v0/|v0|, v1/|v1|>)``.  The interpolation coefficients
    are applied to the *original* (unnormalized) vectors, so endpoint magnitudes
    are respected and the endpoints are reproduced exactly (``t=0`` -> ``v0``,
    ``t=1`` -> ``v1``).

    Numerical-stability handling:

    * **Zero-norm vector** (``|v| < eps``): direction is undefined, so we fall
      back to linear interpolation.
    * **Near-parallel or near-antiparallel** (``|cos| > dot_threshold``):
      ``sin(Omega) -> 0`` makes the division ill-conditioned, so we fall back to
      linear interpolation.  Using the *absolute* cosine catches both cases.
    * The cosine is clamped to ``[-1, 1]`` before ``arccos`` to stay inside its
      domain despite floating-point error.

    Args:
        v0: First endpoint, a 1-D tensor in the compute dtype.
        v1: Second endpoint, same shape/dtype as ``v0``.
        t: Interpolation coefficient (extrapolation, ``t`` outside ``[0, 1]``, is
            handled by the caller before this point).
        dot_threshold: Absolute-cosine threshold above which LERP is used.
        eps: Zero-norm threshold.

    Returns:
        The interpolated 1-D tensor in the same dtype as the inputs.
    """

    if v0.shape != v1.shape:
        raise ValueError(f"slerp shape mismatch: {tuple(v0.shape)} vs {tuple(v1.shape)}")

    norm0 = torch.linalg.vector_norm(v0)
    norm1 = torch.linalg.vector_norm(v1)
    if float(norm0) < eps or float(norm1) < eps:
        return lerp(v0, v1, t)

    unit0 = v0 / norm0
    unit1 = v1 / norm1
    dot = torch.clamp(torch.dot(unit0, unit1), -1.0, 1.0)
    if float(dot.abs()) > dot_threshold:
        return lerp(v0, v1, t)

    omega = torch.arccos(dot)
    sin_omega = torch.sin(omega)
    coeff0 = torch.sin((1.0 - t) * omega) / sin_omega
    coeff1 = torch.sin(t * omega) / sin_omega
    return coeff0 * v0 + coeff1 * v1
