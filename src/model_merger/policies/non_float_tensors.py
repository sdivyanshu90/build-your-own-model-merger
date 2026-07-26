"""Policy for tensors that must not be arithmetically merged.

Integer buffers, boolean masks, position ids, batch-norm ``num_batches_tracked``
counters, and quantization metadata cannot be averaged meaningfully -- averaging
two ``int64`` position-id buffers, or two quantization scales, produces garbage.
These tensors are resolved by *selection*, not interpolation.

Policies:

* ``require_equal`` (default, safest): all sources must be bit-identical; error
  otherwise.
* ``take_first`` / ``take_last``: pick a designated source without comparison.
* ``error``: never allow a non-float tensor to be merged (fail immediately).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..exceptions import TensorMismatchError
from ..types import NonFloatPolicy

__all__ = ["NonFloatTensorPolicy", "tensors_equal"]


def tensors_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Return True if two tensors have identical shape, dtype, and contents."""

    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(torch.equal(a, b))


class NonFloatTensorPolicy:
    """Resolve a group of non-float source tensors into one output tensor."""

    def __init__(self, policy: NonFloatPolicy) -> None:
        self.policy = policy

    def resolve(self, key: str, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        """Return the chosen tensor for ``key`` according to the policy.

        Raises:
            TensorMismatchError: under ``require_equal`` when sources differ, or
                under ``error`` for any non-float tensor.
        """

        if not tensors:
            raise TensorMismatchError(f"no source tensors for non-float key {key!r}")

        if self.policy is NonFloatPolicy.ERROR:
            raise TensorMismatchError(
                f"non-float tensor {key!r} (dtype {tensors[0].dtype}) cannot be merged; "
                f"policy is 'error'"
            )
        if self.policy is NonFloatPolicy.TAKE_FIRST:
            return tensors[0]
        if self.policy is NonFloatPolicy.TAKE_LAST:
            return tensors[-1]

        # REQUIRE_EQUAL
        reference = tensors[0]
        for index, tensor in enumerate(tensors[1:], start=1):
            if not tensors_equal(reference, tensor):
                raise TensorMismatchError(
                    f"non-float tensor {key!r} differs across sources "
                    f"(model 0 vs model {index}); policy is 'require_equal'"
                )
        return reference
