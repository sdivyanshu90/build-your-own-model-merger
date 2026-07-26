"""Unit tests for weighted model soup and weight validation."""

from __future__ import annotations

import pytest
import torch

from model_merger.algorithms import WeightedSoup, normalize_weights
from model_merger.exceptions import ConfigurationError, TensorMismatchError


def test_weighted_matches_manual_combination() -> None:
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    result = WeightedSoup([0.25, 0.75])([a, b])
    assert torch.allclose(result, torch.tensor([0.25, 0.75]))


def test_equal_weights_match_uniform_mean() -> None:
    tensors = [torch.randn(5) for _ in range(3)]
    weighted = WeightedSoup([1 / 3, 1 / 3, 1 / 3])(tensors)
    mean = torch.stack(tensors).mean(dim=0)
    assert torch.allclose(weighted, mean, atol=1e-6)


def test_auto_normalization_scales_to_one() -> None:
    a = torch.tensor([2.0])
    b = torch.tensor([4.0])
    # Raw weights 1:1 normalize to 0.5:0.5 -> mean 3.0
    assert torch.allclose(WeightedSoup([10.0, 10.0])([a, b]), torch.tensor([3.0]))


def test_normalize_scaling_invariance() -> None:
    weights_small = normalize_weights([1.0, 2.0, 1.0], normalize=True, allow_negative=False)
    weights_big = normalize_weights([10.0, 20.0, 10.0], normalize=True, allow_negative=False)
    assert weights_small == pytest.approx(weights_big)


def test_strict_mode_requires_unit_sum() -> None:
    with pytest.raises(ConfigurationError):
        normalize_weights([0.3, 0.3], normalize=False, allow_negative=False)


def test_strict_mode_accepts_unit_sum() -> None:
    assert normalize_weights([0.3, 0.7], normalize=False, allow_negative=False) == [0.3, 0.7]


def test_rejects_nan_weight() -> None:
    with pytest.raises(ConfigurationError):
        normalize_weights([float("nan"), 1.0], normalize=True, allow_negative=False)


def test_rejects_negative_weight_by_default() -> None:
    with pytest.raises(ConfigurationError):
        normalize_weights([-0.5, 1.5], normalize=True, allow_negative=False)


def test_allows_negative_when_enabled() -> None:
    weights = normalize_weights([-0.5, 1.5], normalize=False, allow_negative=True)
    assert weights == [-0.5, 1.5]


def test_normalize_zero_sum_rejected() -> None:
    with pytest.raises(ConfigurationError):
        normalize_weights([1.0, -1.0], normalize=True, allow_negative=True)


def test_weight_count_must_match_tensors() -> None:
    with pytest.raises(TensorMismatchError):
        WeightedSoup([0.5, 0.5])([torch.zeros(2), torch.zeros(2), torch.zeros(2)])
