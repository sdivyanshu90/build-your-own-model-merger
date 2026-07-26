"""Unit tests for uniform model soup."""

from __future__ import annotations

import pytest
import torch

from model_merger.algorithms import UniformSoup
from model_merger.exceptions import TensorMismatchError


def test_uniform_is_arithmetic_mean() -> None:
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([3.0, 4.0, 5.0])
    result = UniformSoup()([a, b])
    assert torch.allclose(result, torch.tensor([2.0, 3.0, 4.0]))


def test_uniform_single_model_is_identity() -> None:
    a = torch.tensor([1.5, -2.0])
    assert torch.allclose(UniformSoup()([a]), a)


def test_uniform_three_models() -> None:
    tensors = [torch.full((4,), float(v)) for v in (3, 6, 9)]
    assert torch.allclose(UniformSoup()(tensors), torch.full((4,), 6.0))


def test_uniform_preserves_shape() -> None:
    tensors = [torch.randn(2, 3, 4) for _ in range(3)]
    assert UniformSoup()(tensors).shape == (2, 3, 4)


def test_uniform_rejects_shape_mismatch() -> None:
    with pytest.raises(TensorMismatchError):
        UniformSoup()([torch.zeros(3), torch.zeros(4)])


def test_uniform_rejects_empty() -> None:
    with pytest.raises(TensorMismatchError):
        UniformSoup()([])
