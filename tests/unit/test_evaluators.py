"""Unit tests for evaluators and the evaluator factory."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from model_merger.config.models import EvaluatorConfig
from model_merger.evaluation import build_evaluator
from model_merger.evaluation.callable_evaluator import CallableEvaluator
from model_merger.evaluation.command_evaluator import CommandEvaluator
from model_merger.exceptions import ConfigurationError, EvaluationError


def test_callable_evaluator_direct(tmp_path: Path) -> None:
    evaluator = CallableEvaluator(lambda path: 0.42)
    assert evaluator.evaluate(tmp_path) == 0.42


def test_callable_evaluator_non_numeric() -> None:
    evaluator = CallableEvaluator(lambda path: "not a number")
    with pytest.raises(EvaluationError):
        evaluator.evaluate(Path())


def test_callable_evaluator_raises_wrapped() -> None:
    def boom(path: Path) -> float:
        raise RuntimeError("nope")

    with pytest.raises(EvaluationError):
        CallableEvaluator(boom).evaluate(Path())


def test_callable_from_dotted_path(tmp_path: Path) -> None:
    # os.path.getsize accepts a path and returns a number -> a valid evaluator.
    evaluator = CallableEvaluator.from_dotted_path("os.path:getsize")
    target = tmp_path / "f.bin"
    target.write_bytes(b"12345")
    assert evaluator.evaluate(target) == 5.0


def test_callable_from_dotted_bad_spec() -> None:
    with pytest.raises(ConfigurationError):
        CallableEvaluator.from_dotted_path("no_colon")


def test_callable_from_dotted_missing_attr() -> None:
    with pytest.raises(ConfigurationError):
        CallableEvaluator.from_dotted_path("math:not_a_function")


def test_command_evaluator_bare_float() -> None:
    evaluator = CommandEvaluator([sys.executable, "-c", "print(0.75)", "{model_path}"])
    assert evaluator.evaluate(Path()) == 0.75


def test_command_evaluator_json_missing_key() -> None:
    evaluator = CommandEvaluator(
        [sys.executable, "-c", "print('{\"other\": 1}')", "{model_path}"],
        metric_key="score",
    )
    with pytest.raises(EvaluationError, match="metric key"):
        evaluator.evaluate(Path())


def test_command_evaluator_empty_output() -> None:
    evaluator = CommandEvaluator([sys.executable, "-c", "pass", "{model_path}"])
    with pytest.raises(EvaluationError, match="no output"):
        evaluator.evaluate(Path())


def test_command_evaluator_not_found() -> None:
    evaluator = CommandEvaluator(["definitely-not-a-real-binary-xyz", "{model_path}"])
    with pytest.raises(EvaluationError, match="not found"):
        evaluator.evaluate(Path())


def test_command_evaluator_timeout() -> None:
    evaluator = CommandEvaluator(
        [sys.executable, "-c", "import time; time.sleep(5)", "{model_path}"], timeout=0.2
    )
    with pytest.raises(EvaluationError, match="timed out"):
        evaluator.evaluate(Path())


def test_build_evaluator_command() -> None:
    config = EvaluatorConfig(type="command", command=["echo", "{model_path}"])
    assert isinstance(build_evaluator(config), CommandEvaluator)


def test_build_evaluator_callable() -> None:
    config = EvaluatorConfig(type="callable", callable="math:sqrt")
    assert isinstance(build_evaluator(config), CallableEvaluator)
