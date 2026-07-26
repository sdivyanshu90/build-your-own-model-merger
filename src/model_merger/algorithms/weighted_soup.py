"""Weighted model soup: a convex (or, if enabled, affine) combination of models.

Given weights ``w_i`` the merged tensor is::

    theta_merged = sum_i w_i * theta_i        with   sum_i w_i = 1

Weight handling is decided once, at construction, by
:func:`normalize_weights`, so :meth:`merge` is a hot, allocation-light loop.

Interpolation vs. extrapolation vs. ensembling
----------------------------------------------
* **Interpolation**: all ``w_i >= 0`` and ``sum w_i = 1`` -- the result lies in
  the convex hull of the inputs.
* **Extrapolation**: some ``w_i < 0`` (only when ``allow_negative`` is set) --
  the result may lie outside the hull; use with care.
* **Ensembling** combines model *outputs* at inference time and is a different
  technique entirely (see ``docs/mathematical-foundations.md``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ..exceptions import ConfigurationError
from .base import MergeStrategy
from .numerical import weighted_sum

__all__ = ["WeightedSoup", "normalize_weights"]

#: Tolerance for the "weights already sum to 1" check in strict mode.
_SUM_TOLERANCE = 1e-6


def normalize_weights(
    weights: Sequence[float],
    *,
    normalize: bool,
    allow_negative: bool,
) -> list[float]:
    """Validate and (optionally) normalize merge weights.

    Args:
        weights: Raw per-model weights.
        normalize: If true, rescale so the weights sum to 1.  If false (strict
            mode) the weights must already sum to 1 within tolerance.
        allow_negative: If false, reject negative weights (extrapolation guard).

    Returns:
        The validated, possibly-normalized weight list.

    Raises:
        ConfigurationError: on empty input, non-finite weights, disallowed
            negatives, a non-positive normalization total, or (strict mode) a
            sum that is not 1.
    """

    if not weights:
        raise ConfigurationError("weighted soup requires at least one weight")
    values = [float(weight) for weight in weights]
    for index, weight in enumerate(values):
        if not math.isfinite(weight):
            raise ConfigurationError(f"weight {index} is not finite: {weight!r}")
        if weight < 0 and not allow_negative:
            raise ConfigurationError(
                f"negative weight {weight!r} at index {index}; "
                f"set allow_negative=true to enable extrapolation"
            )
    total = math.fsum(values)
    if normalize:
        if math.isclose(total, 0.0, abs_tol=1e-12):
            raise ConfigurationError("cannot normalize weights that sum to zero")
        return [weight / total for weight in values]
    if not math.isclose(total, 1.0, abs_tol=_SUM_TOLERANCE):
        raise ConfigurationError(
            f"weights must sum to 1 in strict mode (sum={total:.8f}); "
            f"enable normalize_weights to auto-scale"
        )
    return values


class WeightedSoup(MergeStrategy):
    """Merge inputs as ``sum_i w_i * theta_i`` with validated weights."""

    name = "weighted_soup"
    required_models = None

    def __init__(
        self,
        weights: Sequence[float],
        *,
        normalize: bool = True,
        allow_negative: bool = False,
    ) -> None:
        self.weights = normalize_weights(
            weights, normalize=normalize, allow_negative=allow_negative
        )

    def validate_inputs(self, tensors: Sequence[torch.Tensor]) -> None:
        super().validate_inputs(tensors)
        if len(tensors) != len(self.weights):
            from ..exceptions import TensorMismatchError

            raise TensorMismatchError(
                f"weighted_soup: {len(tensors)} tensors but {len(self.weights)} weights"
            )

    def merge(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        return weighted_sum(tensors, self.weights)
