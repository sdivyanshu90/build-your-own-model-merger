"""Evaluator interface for greedy soup scoring.

An evaluator maps a *written* merged checkpoint (a directory or file path) to a
single scalar metric.  Greedy soup selection calls it repeatedly on candidate
soups; higher or lower is better per the configured
:class:`~model_merger.types.MetricDirection`.

Evaluators are a trust boundary: a callable evaluator imports user code and a
command evaluator spawns a user process.  Both are user-configured, so the trust
is the user's; the command evaluator additionally never uses a shell.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

__all__ = ["Evaluator"]


class Evaluator(ABC):
    """Base class mapping a checkpoint path to a scalar metric."""

    @abstractmethod
    def evaluate(self, model_path: Path) -> float:
        """Return the metric for the checkpoint at ``model_path``."""
