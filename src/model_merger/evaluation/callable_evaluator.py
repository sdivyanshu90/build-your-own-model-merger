"""Evaluator backed by a Python callable.

The callable receives the merged checkpoint path and returns a float.  It can be
supplied directly (programmatic API) or resolved from a ``module.path:function``
string (configuration).  Importing the module runs its top-level code -- that is
the user's own code and trust, and is documented as such.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path

from ..exceptions import ConfigurationError, EvaluationError
from .base import Evaluator

__all__ = ["CallableEvaluator"]

MetricCallable = Callable[[Path], float]


class CallableEvaluator(Evaluator):
    """Score a checkpoint by calling a user-provided function."""

    def __init__(self, function: MetricCallable) -> None:
        if not callable(function):
            raise ConfigurationError("callable evaluator requires a callable")
        self._function = function

    @classmethod
    def from_dotted_path(cls, spec: str) -> CallableEvaluator:
        """Build from a ``module.path:function`` specification.

        Raises:
            ConfigurationError: if the spec is malformed or cannot be imported.
        """

        if ":" not in spec:
            raise ConfigurationError(f"callable spec must be 'module:function', got {spec!r}")
        module_name, _, attribute = spec.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            raise ConfigurationError(
                f"cannot import evaluator module {module_name!r}: {error}"
            ) from error
        try:
            function = getattr(module, attribute)
        except AttributeError as error:
            raise ConfigurationError(
                f"module {module_name!r} has no attribute {attribute!r}"
            ) from error
        return cls(function)

    def evaluate(self, model_path: Path) -> float:
        try:
            result = self._function(Path(model_path))
        except Exception as error:
            raise EvaluationError(f"evaluator callable raised: {error}") from error
        try:
            return float(result)
        except (TypeError, ValueError) as error:
            raise EvaluationError(
                f"evaluator callable returned a non-numeric value: {result!r}"
            ) from error
