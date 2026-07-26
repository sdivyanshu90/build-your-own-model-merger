"""Abstract base class for tensor-level merge strategies.

A :class:`MergeStrategy` merges the *same* tensor drawn from each source model
into one output tensor.  Strategies are pure functions of their inputs and their
configured parameters -- they perform no I/O, no device management, and no dtype
policy decisions.  The executor casts inputs to the compute dtype before calling
:meth:`merge` and casts the result to the output dtype afterwards.

Non-floating tensors are never passed to a strategy; the executor resolves those
via :class:`~model_merger.policies.non_float_tensors.NonFloatTensorPolicy`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from ..exceptions import TensorMismatchError

__all__ = ["MergeStrategy"]


class MergeStrategy(ABC):
    """Base class for algorithms that combine N same-shaped tensors into one."""

    #: Stable identifier used in reports and logs.
    name: str = "base"

    #: Exact number of models required, or ``None`` for "two or more".
    required_models: int | None = None

    def validate_inputs(self, tensors: Sequence[torch.Tensor]) -> None:
        """Validate arity and shape agreement before merging.

        Raises:
            TensorMismatchError: on arity or shape disagreement.
        """

        count = len(tensors)
        if count == 0:
            raise TensorMismatchError(f"{self.name}: no input tensors provided")
        if self.required_models is not None and count != self.required_models:
            raise TensorMismatchError(
                f"{self.name} requires exactly {self.required_models} models, got {count}"
            )
        if self.required_models is None and count < 1:
            raise TensorMismatchError(f"{self.name} requires at least one model")
        reference = tensors[0].shape
        for index, tensor in enumerate(tensors[1:], start=1):
            if tensor.shape != reference:
                raise TensorMismatchError(
                    f"{self.name}: shape mismatch at input {index}: "
                    f"{tuple(tensor.shape)} != {tuple(reference)}"
                )

    @abstractmethod
    def merge(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        """Merge ``tensors`` (already in compute dtype) into one tensor.

        Implementations may assume :meth:`validate_inputs` has passed.
        """

    def __call__(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        self.validate_inputs(tensors)
        return self.merge(tensors)
