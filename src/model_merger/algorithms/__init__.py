"""Merge algorithms: soups (uniform / weighted / greedy) and interpolation (SLERP / linear)."""

from __future__ import annotations

from .base import MergeStrategy
from .greedy_soup import (
    GreedySoupResult,
    GreedyStep,
    greedy_soup_selection,
    is_improvement,
)
from .numerical import assert_finite, lerp, slerp_vectors, weighted_sum
from .slerp import LinearInterpolation, Slerp, validate_interpolation_t
from .uniform_soup import UniformSoup
from .weighted_soup import WeightedSoup, normalize_weights

__all__ = [
    "MergeStrategy",
    "UniformSoup",
    "WeightedSoup",
    "normalize_weights",
    "Slerp",
    "LinearInterpolation",
    "validate_interpolation_t",
    "greedy_soup_selection",
    "GreedySoupResult",
    "GreedyStep",
    "is_improvement",
    "assert_finite",
    "weighted_sum",
    "slerp_vectors",
    "lerp",
]
