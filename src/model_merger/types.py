"""Shared enums and type aliases used across :mod:`model_merger`.

These are deliberately dependency-light (only :mod:`torch` for dtype helpers) so
that configuration and reporting code can import them without pulling in the full
execution stack.
"""

from __future__ import annotations

from enum import Enum

import torch

__all__ = [
    "AlgorithmType",
    "OutputFormat",
    "OutputDtypePolicy",
    "NonFloatPolicy",
    "CompatibilityMode",
    "Severity",
    "MetricDirection",
    "MatchKind",
    "TensorKey",
    "DTYPE_BY_NAME",
    "resolve_dtype",
    "dtype_name",
    "is_floating_dtype",
]

#: A tensor key is the fully-qualified parameter name in a state dict.
TensorKey = str


class AlgorithmType(str, Enum):
    """Supported merge algorithms.

    ``linear`` is exposed for per-rule overrides (a plain LERP between two
    models) and is the numerically-safe fallback used by SLERP.
    """

    UNIFORM_SOUP = "uniform_soup"
    WEIGHTED_SOUP = "weighted_soup"
    GREEDY_SOUP = "greedy_soup"
    SLERP = "slerp"
    LINEAR = "linear"


class OutputFormat(str, Enum):
    """Checkpoint container written for the merged model."""

    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"


class OutputDtypePolicy(str, Enum):
    """How the dtype of written tensors is chosen.

    ``PRESERVE`` keeps each tensor at its source dtype; ``HIGHEST`` promotes to
    the widest dtype among the sources; the explicit dtype members force a cast.
    """

    PRESERVE = "preserve"
    HIGHEST = "highest"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class NonFloatPolicy(str, Enum):
    """Policy for tensors that cannot be meaningfully averaged.

    Integer/bool buffers, position ids, batch-norm counters, quantization
    metadata and similar tensors are not interpolated.  ``REQUIRE_EQUAL`` (the
    safe default) fails if the sources disagree.
    """

    TAKE_FIRST = "take_first"
    TAKE_LAST = "take_last"
    REQUIRE_EQUAL = "require_equal"
    ERROR = "error"


class CompatibilityMode(str, Enum):
    """Strictness of compatibility validation."""

    STRICT = "strict"
    PERMISSIVE = "permissive"


class Severity(str, Enum):
    """Severity classification for a compatibility finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"

    @property
    def rank(self) -> int:
        return {"info": 0, "warning": 1, "error": 2, "fatal": 3}[self.value]


class MetricDirection(str, Enum):
    """Whether a greedy-soup metric should be maximized or minimized."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class MatchKind(str, Enum):
    """The kind of tensor-selection predicate used by a layer rule."""

    EXACT = "exact"
    GLOB = "glob"
    REGEX = "regex"
    LAYER_RANGE = "layer_range"
    SUFFIX = "suffix"


#: Canonical mapping of dtype name -> torch dtype for configuration parsing.
DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "float": torch.float32,
    "half": torch.float16,
    "double": torch.float64,
}


def resolve_dtype(name: str) -> torch.dtype:
    """Return the :class:`torch.dtype` for a dtype name.

    Raises:
        KeyError: if the name is unknown (callers should convert to a
            :class:`~model_merger.exceptions.ConfigurationError`).
    """

    return DTYPE_BY_NAME[name.lower()]


def dtype_name(dtype: torch.dtype) -> str:
    """Return the short, round-trippable name for a torch dtype."""

    return str(dtype).removeprefix("torch.")


def is_floating_dtype(dtype: torch.dtype) -> bool:
    """True for real floating-point dtypes that may be interpolated."""

    return dtype.is_floating_point
