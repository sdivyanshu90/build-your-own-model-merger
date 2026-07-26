"""Evaluator backed by an external command.

The command is an **argument vector** (never a shell string), so there is no
shell interpretation and therefore no command-injection surface: the checkpoint
path is substituted into a placeholder argument and passed as a single argv
element even if it contains spaces or metacharacters.  The command must print
either a bare float or a JSON object containing the configured metric key.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from ..exceptions import EvaluationError
from ..logging import get_logger
from .base import Evaluator

__all__ = ["CommandEvaluator"]

_LOGGER = get_logger(__name__)


class CommandEvaluator(Evaluator):
    """Score a checkpoint by running an external program (no shell)."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        placeholder: str = "{model_path}",
        metric_key: str = "score",
        timeout: float | None = None,
    ) -> None:
        self._command = list(command)
        self._placeholder = placeholder
        self._metric_key = metric_key
        self._timeout = timeout

    def _build_argv(self, model_path: Path) -> list[str]:
        return [arg.replace(self._placeholder, str(model_path)) for arg in self._command]

    def evaluate(self, model_path: Path) -> float:
        argv = self._build_argv(Path(model_path))
        _LOGGER.debug("running evaluator command: %s", argv)
        try:
            completed = subprocess.run(  # noqa: S603 - argv list, shell=False by construction
                argv,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
                shell=False,
            )
        except FileNotFoundError as error:
            raise EvaluationError(f"evaluator command not found: {argv[0]!r}") from error
        except subprocess.TimeoutExpired as error:
            raise EvaluationError(
                f"evaluator command timed out after {self._timeout}s: {argv[0]!r}"
            ) from error

        if completed.returncode != 0:
            stderr = completed.stderr.strip()[:500]
            raise EvaluationError(f"evaluator command exited with {completed.returncode}: {stderr}")
        return self._parse_metric(completed.stdout)

    def _parse_metric(self, stdout: str) -> float:
        text = stdout.strip()
        if not text:
            raise EvaluationError("evaluator command produced no output")
        # Prefer a JSON object with the metric key; fall back to the last line as a float.
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            if self._metric_key not in parsed:
                raise EvaluationError(
                    f"evaluator JSON lacks metric key {self._metric_key!r}: keys={list(parsed)}"
                )
            value = parsed[self._metric_key]
        elif isinstance(parsed, int | float):
            value = parsed
        else:
            value = text.splitlines()[-1].strip()
        try:
            return float(value)
        except (TypeError, ValueError) as error:
            raise EvaluationError(f"cannot parse evaluator metric from {value!r}") from error
