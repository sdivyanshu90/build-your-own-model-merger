"""Evaluators for greedy soup, plus a factory over evaluator configuration."""

from __future__ import annotations

from ..config.models import EvaluatorConfig
from ..exceptions import ConfigurationError
from .base import Evaluator
from .callable_evaluator import CallableEvaluator
from .command_evaluator import CommandEvaluator

__all__ = ["Evaluator", "CallableEvaluator", "CommandEvaluator", "build_evaluator"]


def build_evaluator(config: EvaluatorConfig) -> Evaluator:
    """Construct an :class:`Evaluator` from validated configuration."""

    if config.type == "command":
        if config.command is None:  # pragma: no cover - validated upstream
            raise ConfigurationError("command evaluator requires 'command'")
        return CommandEvaluator(
            config.command,
            placeholder=config.placeholder,
            metric_key=config.metric_key,
            timeout=config.timeout,
        )
    if config.callable is None:  # pragma: no cover - validated upstream
        raise ConfigurationError("callable evaluator requires 'callable'")
    return CallableEvaluator.from_dotted_path(config.callable)
