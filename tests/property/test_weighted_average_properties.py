"""Property-based tests for weighted/uniform averaging invariants."""

from __future__ import annotations

import math

import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from model_merger.algorithms import UniformSoup, WeightedSoup, normalize_weights

_shapes = st.integers(min_value=1, max_value=32)
_finite = st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)


@settings(max_examples=100, deadline=None)
@given(size=_shapes)
def test_single_model_weight_one_is_identity(size: int) -> None:
    tensor = torch.randn(size)
    result = WeightedSoup([1.0], normalize=False)([tensor])
    assert torch.allclose(result, tensor, atol=1e-6)


@settings(max_examples=100, deadline=None)
@given(size=_shapes, n=st.integers(min_value=2, max_value=5))
def test_equal_weights_equal_mean(size: int, n: int) -> None:
    tensors = [torch.randn(size) for _ in range(n)]
    weighted = WeightedSoup([1.0 / n] * n)(tensors)
    mean = torch.stack(tensors).mean(dim=0)
    assert torch.allclose(weighted, mean, atol=1e-5)


@settings(max_examples=100, deadline=None)
@given(
    size=_shapes,
    weights=st.lists(st.floats(min_value=0.01, max_value=10), min_size=2, max_size=5),
)
def test_permutation_invariance(size: int, weights: list[float]) -> None:
    tensors = [torch.randn(size) for _ in weights]
    forward = WeightedSoup(list(weights))(tensors)
    order = list(range(len(weights)))[::-1]
    reversed_tensors = [tensors[i] for i in order]
    reversed_weights = [weights[i] for i in order]
    backward = WeightedSoup(reversed_weights)(reversed_tensors)
    assert torch.allclose(forward, backward, atol=1e-5)


@settings(max_examples=100, deadline=None)
@given(
    weights=st.lists(st.floats(min_value=0.01, max_value=10), min_size=2, max_size=5),
    scale=st.floats(min_value=0.1, max_value=100),
)
def test_positive_scaling_invariance_under_normalization(
    weights: list[float], scale: float
) -> None:
    tensors = [torch.randn(6) for _ in weights]
    base = WeightedSoup(list(weights), normalize=True)(tensors)
    scaled = WeightedSoup([w * scale for w in weights], normalize=True)(tensors)
    assert torch.allclose(base, scaled, atol=1e-5)


@settings(max_examples=50, deadline=None)
@given(size=_shapes, n=st.integers(min_value=1, max_value=5))
def test_output_shape_matches_input(size: int, n: int) -> None:
    tensors = [torch.randn(size, 3) for _ in range(n)]
    assert UniformSoup()(tensors).shape == (size, 3)


@settings(max_examples=100, deadline=None)
@given(values=st.lists(st.floats(min_value=0.0, max_value=100), min_size=1, max_size=8))
def test_normalized_weights_sum_to_one(values: list[float]) -> None:
    assume(math.fsum(values) > 1.0)  # avoid near-zero totals that amplify fp error
    normalized = normalize_weights(values, normalize=True, allow_negative=False)
    assert abs(math.fsum(normalized) - 1.0) < 1e-9


@settings(max_examples=50, deadline=None)
@given(size=_shapes, n=st.integers(min_value=2, max_value=4))
def test_finite_inputs_finite_outputs(size: int, n: int) -> None:
    tensors = [torch.randn(size) for _ in range(n)]
    assert torch.isfinite(UniformSoup()(tensors)).all()
