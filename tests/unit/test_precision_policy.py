"""Unit tests for the numerical precision policy and non-float policy."""

from __future__ import annotations

import pytest
import torch

from model_merger.exceptions import TensorMismatchError
from model_merger.policies.non_float_tensors import NonFloatTensorPolicy, tensors_equal
from model_merger.policies.precision import PrecisionPolicy
from model_merger.types import NonFloatPolicy, OutputDtypePolicy


def test_compute_dtype_promotes_half_to_float32() -> None:
    policy = PrecisionPolicy(compute_dtype=torch.float32)
    assert policy.compute_dtype_for([torch.float16, torch.bfloat16]) == torch.float32


def test_compute_dtype_respects_wider_input() -> None:
    policy = PrecisionPolicy(compute_dtype=torch.float32)
    assert policy.compute_dtype_for([torch.float64]) == torch.float64


def test_output_preserve_uses_first_dtype() -> None:
    policy = PrecisionPolicy(output_dtype_policy=OutputDtypePolicy.PRESERVE)
    assert policy.output_dtype_for([torch.float16, torch.float16]) == torch.float16


def test_output_highest() -> None:
    policy = PrecisionPolicy(output_dtype_policy=OutputDtypePolicy.HIGHEST)
    assert policy.output_dtype_for([torch.float16, torch.float32]) == torch.float32


def test_output_explicit() -> None:
    policy = PrecisionPolicy(output_dtype_policy=OutputDtypePolicy.BFLOAT16)
    assert policy.output_dtype_for([torch.float32]) == torch.bfloat16


def test_describe_is_json_friendly() -> None:
    described = PrecisionPolicy().describe()
    assert described["compute_dtype"] == "float32"
    assert described["output_dtype_policy"] == "preserve"


def test_unsafe_cast_warning(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    policy = PrecisionPolicy()
    huge = torch.tensor([1e6, -1e6])
    # Attach caplog's handler directly so the result does not depend on the
    # model_merger logger's propagation state (which other tests may change).
    mm_logger = logging.getLogger("model_merger")
    mm_logger.addHandler(caplog.handler)
    mm_logger.setLevel(logging.WARNING)
    try:
        policy.warn_on_unsafe_cast(huge, torch.float16, key="w")
    finally:
        mm_logger.removeHandler(caplog.handler)
    assert any("float16" in record.message for record in caplog.records)


def test_non_float_require_equal_passes() -> None:
    policy = NonFloatTensorPolicy(NonFloatPolicy.REQUIRE_EQUAL)
    a = torch.arange(4)
    result = policy.resolve("k", [a, a.clone()])
    assert torch.equal(result, a)


def test_non_float_require_equal_fails() -> None:
    policy = NonFloatTensorPolicy(NonFloatPolicy.REQUIRE_EQUAL)
    with pytest.raises(TensorMismatchError):
        policy.resolve("k", [torch.arange(4), torch.zeros(4, dtype=torch.int64)])


def test_non_float_take_first_and_last() -> None:
    a, b = torch.arange(3), torch.arange(3) + 1
    assert torch.equal(NonFloatTensorPolicy(NonFloatPolicy.TAKE_FIRST).resolve("k", [a, b]), a)
    assert torch.equal(NonFloatTensorPolicy(NonFloatPolicy.TAKE_LAST).resolve("k", [a, b]), b)


def test_non_float_error_policy() -> None:
    with pytest.raises(TensorMismatchError):
        NonFloatTensorPolicy(NonFloatPolicy.ERROR).resolve("k", [torch.arange(2)])


def test_tensors_equal() -> None:
    assert tensors_equal(torch.arange(3), torch.arange(3))
    assert not tensors_equal(torch.arange(3), torch.arange(3) + 1)
    assert not tensors_equal(torch.zeros(2), torch.zeros(3))
