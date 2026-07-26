"""The per-tensor merge engine (the bounded-memory hot path).

For each key the engine loads that one tensor from every source, casts to the
compute dtype, runs the strategy, validates finiteness, and casts to the output
dtype -- then the tensor is handed to the writer and released.  At no point are
all of a model's tensors resident.  Peak memory is roughly
``n_models * sizeof(largest tensor)`` in the compute dtype, plus one output shard
buffered by the writer.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..algorithms.base import MergeStrategy
from ..algorithms.numerical import assert_finite
from ..policies.non_float_tensors import NonFloatTensorPolicy
from ..policies.precision import PrecisionPolicy
from .device import to_device

__all__ = ["TensorMergeEngine"]


class TensorMergeEngine:
    """Merge individual tensors under a precision policy and device."""

    def __init__(
        self,
        precision: PrecisionPolicy,
        non_float_policy: NonFloatTensorPolicy,
        device: torch.device,
    ) -> None:
        self.precision = precision
        self.non_float_policy = non_float_policy
        self.device = device

    def merge_float(
        self,
        key: str,
        source_tensors: Sequence[torch.Tensor],
        strategy: MergeStrategy,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Merge floating tensors for ``key`` and return a CPU output tensor."""

        input_dtypes = [tensor.dtype for tensor in source_tensors]
        compute_dtype = self.precision.compute_dtype_for(input_dtypes)
        casts = [to_device(tensor, self.device).to(compute_dtype) for tensor in source_tensors]
        merged = strategy(casts)
        if self.precision.validate_finite:
            assert_finite(merged, context=key)
        self.precision.warn_on_unsafe_cast(merged, output_dtype, key=key)
        return merged.to(device="cpu", dtype=output_dtype)

    def resolve_non_float(self, key: str, source_tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        """Resolve non-floating tensors for ``key`` (CPU tensor, dtype preserved)."""

        chosen = self.non_float_policy.resolve(key, source_tensors)
        return chosen.to("cpu")
