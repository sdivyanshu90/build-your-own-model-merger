"""Numerical precision policy: compute dtype, output dtype, and finiteness.

Averaging or interpolating ``float16`` / ``bfloat16`` tensors directly loses
precision: each addition rounds to ~3-4 significant decimal digits, and the error
accumulates across models.  The safe default is therefore to **accumulate in
float32** even when inputs and outputs are half precision, then cast the result
back.  ``docs/numerical-precision-policy`` and ADR-0003 discuss the tradeoffs.

This policy is a plain value object (no torch state) so it is trivial to test.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..logging import get_logger
from ..types import OutputDtypePolicy, dtype_name, is_floating_dtype

__all__ = ["PrecisionPolicy", "FLOAT16_MAX"]

_LOGGER = get_logger(__name__)

#: Largest finite magnitude representable in IEEE float16.
FLOAT16_MAX = 65504.0

#: Precision rank used to select the "highest" dtype among inputs.
_DTYPE_RANK: dict[torch.dtype, int] = {
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float32: 3,
    torch.float64: 4,
}


@dataclass(frozen=True)
class PrecisionPolicy:
    """Decides compute and output dtypes for a merge.

    Attributes:
        compute_dtype: dtype used for the arithmetic.  Defaults to float32.  If a
            source tensor is *wider* than this (e.g. float64), the wider dtype is
            used so accumulation never loses input precision.
        output_dtype_policy: how the written dtype is chosen (see
            :class:`~model_merger.types.OutputDtypePolicy`).
        validate_finite: if true, merged tensors are checked for NaN/Inf.
    """

    compute_dtype: torch.dtype = torch.float32
    output_dtype_policy: OutputDtypePolicy = OutputDtypePolicy.PRESERVE
    validate_finite: bool = True

    def compute_dtype_for(self, input_dtypes: Sequence[torch.dtype]) -> torch.dtype:
        """Return the dtype the arithmetic should run in.

        Non-floating inputs are ignored here (they are handled by the non-float
        policy, not merged arithmetically).  The result is the wider of the
        configured compute dtype and the widest floating input dtype.
        """

        floating = [dtype for dtype in input_dtypes if is_floating_dtype(dtype)]
        best = self.compute_dtype
        for dtype in floating:
            if _DTYPE_RANK.get(dtype, 0) > _DTYPE_RANK.get(best, 0):
                best = dtype
        return best

    def output_dtype_for(self, input_dtypes: Sequence[torch.dtype]) -> torch.dtype:
        """Return the dtype merged tensors should be written in."""

        policy = self.output_dtype_policy
        if policy is OutputDtypePolicy.PRESERVE:
            return input_dtypes[0]
        if policy is OutputDtypePolicy.HIGHEST:
            return max(input_dtypes, key=lambda dtype: _DTYPE_RANK.get(dtype, 0))
        explicit = {
            OutputDtypePolicy.FLOAT16: torch.float16,
            OutputDtypePolicy.BFLOAT16: torch.bfloat16,
            OutputDtypePolicy.FLOAT32: torch.float32,
            OutputDtypePolicy.FLOAT64: torch.float64,
        }
        return explicit[policy]

    def warn_on_unsafe_cast(self, tensor: torch.Tensor, target: torch.dtype, *, key: str) -> None:
        """Log a warning if casting ``tensor`` to ``target`` would overflow.

        Only float16 has a small enough range to be a practical concern for
        merged weights; values beyond +/-65504 become Inf on cast.
        """

        if target is torch.float16 and is_floating_dtype(tensor.dtype):
            max_abs = float(tensor.abs().max().item()) if tensor.numel() else 0.0
            if max_abs > FLOAT16_MAX:
                _LOGGER.warning(
                    "casting %s to float16 will overflow: max |value| = %.4g > %.0f",
                    key,
                    max_abs,
                    FLOAT16_MAX,
                )

    def describe(self) -> dict[str, object]:
        """Return a JSON-serializable description for the merge report."""

        return {
            "compute_dtype": dtype_name(self.compute_dtype),
            "output_dtype_policy": self.output_dtype_policy.value,
            "validate_finite": self.validate_finite,
        }
