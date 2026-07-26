"""Property-based tests for SLERP invariants."""

from __future__ import annotations

import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from model_merger.algorithms import Slerp
from model_merger.algorithms.numerical import slerp_vectors
from model_merger.exceptions import ConfigurationError

_sizes = st.integers(min_value=1, max_value=64)
_t = st.floats(min_value=0.0, max_value=1.0)


def _nondegenerate_pair(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(size * 7 + 1)
    v0 = torch.randn(size, generator=generator)
    v1 = torch.randn(size, generator=generator)
    return v0, v1


@settings(max_examples=100, deadline=None)
@given(size=_sizes)
def test_t0_returns_first(size: int) -> None:
    v0, v1 = _nondegenerate_pair(size)
    assert torch.allclose(slerp_vectors(v0, v1, 0.0), v0, atol=1e-5)


@settings(max_examples=100, deadline=None)
@given(size=_sizes)
def test_t1_returns_second(size: int) -> None:
    v0, v1 = _nondegenerate_pair(size)
    assert torch.allclose(slerp_vectors(v0, v1, 1.0), v1, atol=1e-5)


@settings(max_examples=100, deadline=None)
@given(size=_sizes, t=_t)
def test_swap_symmetry(size: int, t: float) -> None:
    v0, v1 = _nondegenerate_pair(size)
    forward = slerp_vectors(v0, v1, t)
    swapped = slerp_vectors(v1, v0, 1.0 - t)
    assert torch.allclose(forward, swapped, atol=1e-4)


@settings(max_examples=100, deadline=None)
@given(size=st.integers(min_value=1, max_value=16), t=_t)
def test_output_shape_preserved(size: int, t: float) -> None:
    v0 = torch.randn(size, 2)
    v1 = torch.randn(size, 2)
    assert Slerp(t)([v0, v1]).shape == (size, 2)


@settings(max_examples=100, deadline=None)
@given(size=_sizes, t=_t)
def test_near_parallel_no_nan(size: int, t: float) -> None:
    v0 = torch.randn(size)
    v1 = v0 + 1e-7 * torch.randn(size)
    assert torch.isfinite(slerp_vectors(v0, v1, t)).all()


@settings(max_examples=100, deadline=None)
@given(size=_sizes, t=_t)
def test_identical_vectors_return_same(size: int, t: float) -> None:
    v = torch.randn(size)
    assert torch.allclose(slerp_vectors(v, v.clone(), t), v, atol=1e-5)


@settings(max_examples=100, deadline=None)
@given(size=st.integers(min_value=2, max_value=32), t=_t)
def test_unit_sphere_result_stays_on_sphere(size: int, t: float) -> None:
    generator = torch.Generator().manual_seed(size + 3)
    v0 = torch.randn(size, generator=generator)
    v1 = torch.randn(size, generator=generator)
    u0, u1 = v0 / v0.norm(), v1 / v1.norm()
    dot = torch.dot(u0, u1).abs().item()
    assume(dot < 0.99)  # nondegenerate
    result = slerp_vectors(u0, u1, t)
    # Interpolating between unit vectors on the geodesic keeps ~unit norm.
    assert result.norm().item() == pytest.approx(1.0, abs=1e-3)


@settings(max_examples=50, deadline=None)
@given(t=st.floats(min_value=1.01, max_value=5.0))
def test_extrapolation_rejected_when_disabled(t: float) -> None:
    with pytest.raises(ConfigurationError):
        Slerp(t)
