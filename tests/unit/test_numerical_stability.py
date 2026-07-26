"""Unit tests for numerical primitives, finiteness, and precision benefit."""

from __future__ import annotations

import pytest
import torch

from model_merger.algorithms.numerical import (
    assert_finite,
    count_nonfinite,
    lerp,
    slerp_vectors,
    weighted_sum,
)
from model_merger.exceptions import NumericalError


def test_assert_finite_passes_for_finite() -> None:
    tensor = torch.randn(10)
    assert assert_finite(tensor, context="x") is tensor


def test_assert_finite_raises_on_nan() -> None:
    tensor = torch.tensor([1.0, float("nan")])
    with pytest.raises(NumericalError, match="NaN"):
        assert_finite(tensor, context="weight")


def test_assert_finite_raises_on_inf() -> None:
    tensor = torch.tensor([1.0, float("inf")])
    with pytest.raises(NumericalError, match="Inf"):
        assert_finite(tensor, context="weight")


def test_count_nonfinite() -> None:
    tensor = torch.tensor([1.0, float("nan"), float("inf"), float("-inf")])
    nan, inf = count_nonfinite(tensor)
    assert nan == 1
    assert inf == 2


def test_weighted_sum_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        weighted_sum([torch.zeros(2)], [0.5, 0.5])


def test_weighted_sum_empty() -> None:
    with pytest.raises(ValueError):
        weighted_sum([], [])


def test_lerp_endpoints_exact() -> None:
    a = torch.randn(8)
    b = torch.randn(8)
    assert torch.equal(lerp(a, b, 0.0), a)
    assert torch.equal(lerp(a, b, 1.0), b)


def test_float32_accumulation_beats_float16() -> None:
    """Averaging in float32 yields lower error than averaging in float16."""

    torch.manual_seed(0)
    reference = torch.randn(2000, dtype=torch.float64)
    tensors_f64 = [reference + 0.01 * torch.randn(2000, dtype=torch.float64) for _ in range(8)]
    true_mean = torch.stack(tensors_f64).mean(dim=0)

    tensors_f16 = [t.to(torch.float16) for t in tensors_f64]
    # Accumulate in float16 (lossy) vs float32 (policy default).
    acc16 = weighted_sum(tensors_f16, [1 / 8] * 8).to(torch.float64)
    acc32 = weighted_sum([t.to(torch.float32) for t in tensors_f16], [1 / 8] * 8).to(torch.float64)

    err16 = (acc16 - true_mean).abs().mean().item()
    err32 = (acc32 - true_mean).abs().mean().item()
    assert err32 < err16


def test_slerp_near_parallel_stability() -> None:
    base = torch.randn(64)
    perturbed = base + 1e-6 * torch.randn(64)
    result = slerp_vectors(base, perturbed, 0.5)
    assert torch.isfinite(result).all()
