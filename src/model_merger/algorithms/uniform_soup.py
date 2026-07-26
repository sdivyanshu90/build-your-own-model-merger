"""Uniform model soup: the unweighted average of N compatible models.

Given parameter tensors ``theta_1 ... theta_n`` the merged tensor is::

    theta_soup = (1 / n) * sum_i theta_i

This is exactly a weighted soup with equal weights ``1/n``.  It is the simplest
and most common soup and assumes the sources share a training basin (see
``docs/model-soups.md``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .base import MergeStrategy
from .numerical import weighted_sum

__all__ = ["UniformSoup"]


class UniformSoup(MergeStrategy):
    """Average all inputs with equal weight."""

    name = "uniform_soup"
    required_models = None

    def merge(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        count = len(tensors)
        weight = 1.0 / count
        return weighted_sum(tensors, [weight] * count)
