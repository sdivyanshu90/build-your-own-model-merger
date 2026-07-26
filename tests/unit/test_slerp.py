"""Unit tests for SLERP and linear interpolation strategies."""

from __future__ import annotations

import math

import pytest
import torch

from model_merger.algorithms import LinearInterpolation, Slerp
from model_merger.algorithms.numerical import slerp_vectors
from model_merger.exceptions import ConfigurationError


def test_slerp_endpoint_t0_returns_first() -> None:
    v0 = torch.tensor([1.0, 0.0, 0.0])
    v1 = torch.tensor([0.0, 2.0, 0.0])
    assert torch.allclose(Slerp(0.0)([v0, v1]), v0, atol=1e-6)


def test_slerp_endpoint_t1_returns_second() -> None:
    v0 = torch.tensor([1.0, 0.0, 0.0])
    v1 = torch.tensor([0.0, 2.0, 0.0])
    assert torch.allclose(Slerp(1.0)([v0, v1]), v1, atol=1e-6)


def test_slerp_midpoint_on_unit_circle_has_expected_angle() -> None:
    v0 = torch.tensor([1.0, 0.0])
    v1 = torch.tensor([0.0, 1.0])
    mid = Slerp(0.5)([v0, v1])
    # Halfway along a 90-degree arc between unit vectors -> 45 degrees, unit norm.
    assert mid.norm().item() == pytest.approx(1.0, abs=1e-5)
    assert mid[0].item() == pytest.approx(math.cos(math.pi / 4), abs=1e-5)
    assert mid[1].item() == pytest.approx(math.sin(math.pi / 4), abs=1e-5)


def test_slerp_parallel_falls_back_to_lerp_no_nan() -> None:
    v0 = torch.tensor([1.0, 2.0, 3.0])
    v1 = torch.tensor([2.0, 4.0, 6.0])  # exactly parallel
    result = Slerp(0.5)([v0, v1])
    assert torch.isfinite(result).all()
    assert torch.allclose(result, torch.lerp(v0, v1, 0.5))


def test_slerp_antiparallel_no_nan() -> None:
    v0 = torch.tensor([1.0, 0.0])
    v1 = torch.tensor([-1.0, 0.0])
    result = Slerp(0.5)([v0, v1])
    assert torch.isfinite(result).all()


def test_slerp_zero_vector_falls_back() -> None:
    v0 = torch.zeros(3)
    v1 = torch.tensor([1.0, 1.0, 1.0])
    result = Slerp(0.5)([v0, v1])
    assert torch.allclose(result, torch.lerp(v0, v1, 0.5))


def test_slerp_identical_vectors_return_same() -> None:
    v = torch.tensor([0.3, -0.4, 0.5])
    assert torch.allclose(Slerp(0.5)([v, v.clone()]), v, atol=1e-6)


def test_slerp_symmetry_swap_endpoints() -> None:
    v0 = torch.randn(16)
    v1 = torch.randn(16)
    forward = Slerp(0.3)([v0, v1])
    swapped = Slerp(0.7)([v1, v0])
    assert torch.allclose(forward, swapped, atol=1e-5)


def test_slerp_preserves_shape() -> None:
    v0 = torch.randn(4, 5)
    v1 = torch.randn(4, 5)
    assert Slerp(0.5)([v0, v1]).shape == (4, 5)


def test_slerp_rejects_extrapolation_by_default() -> None:
    with pytest.raises(ConfigurationError):
        Slerp(1.5)


def test_slerp_allows_extrapolation_when_enabled() -> None:
    strategy = Slerp(1.5, allow_extrapolation=True)
    assert strategy.t == 1.5


def test_slerp_rejects_bad_dot_threshold() -> None:
    with pytest.raises(ConfigurationError):
        Slerp(0.5, dot_threshold=1.5)


def test_linear_matches_torch_lerp() -> None:
    v0 = torch.randn(10)
    v1 = torch.randn(10)
    assert torch.allclose(LinearInterpolation(0.4)([v0, v1]), torch.lerp(v0, v1, 0.4))


def test_slerp_vectors_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        slerp_vectors(torch.zeros(3), torch.zeros(4), 0.5)
