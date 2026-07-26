"""Unit tests for the greedy soup selection policy (synthetic evaluator)."""

from __future__ import annotations

import pytest

from model_merger.algorithms.greedy_soup import greedy_soup_selection, is_improvement
from model_merger.exceptions import ConfigurationError
from model_merger.types import MetricDirection


def make_evaluator(scores: dict[tuple[str, ...], float]):
    def evaluate(subset: tuple[str, ...]) -> float:
        return scores[tuple(subset)]

    return evaluate


def test_seed_is_best_individual() -> None:
    scores = {("a",): 0.5, ("b",): 0.9, ("c",): 0.7, ("b", "c"): 0.4, ("b", "a"): 0.4}
    result = greedy_soup_selection(["a", "b", "c"], make_evaluator(scores))
    assert result.accepted[0] == "b"


def test_accepts_only_on_improvement() -> None:
    scores = {
        ("a",): 0.5,
        ("b",): 0.9,
        ("c",): 0.7,
        ("b", "c"): 0.95,  # improves -> accept c
        ("b", "c", "a"): 0.80,  # worse -> reject a
    }
    result = greedy_soup_selection(["a", "b", "c"], make_evaluator(scores))
    assert result.accepted == ("b", "c")
    assert result.best_score == pytest.approx(0.95)
    rejected = [step.candidate for step in result.history if not step.accepted]
    assert rejected == ["a"]


def test_minimize_direction() -> None:
    scores = {
        ("a",): 0.5,
        ("b",): 0.2,
        ("c",): 0.7,
        ("b", "a"): 0.1,  # lower is better -> accept a
        ("b", "a", "c"): 0.3,  # worse -> reject c
    }
    result = greedy_soup_selection(
        ["a", "b", "c"], make_evaluator(scores), direction=MetricDirection.MINIMIZE
    )
    assert result.accepted == ("b", "a")


def test_deterministic_tie_break_by_input_order() -> None:
    scores = {("a",): 0.5, ("b",): 0.5, ("a", "b"): 0.4}
    result = greedy_soup_selection(["a", "b"], make_evaluator(scores))
    # Equal individual scores -> seed is the first in input order.
    assert result.accepted[0] == "a"


def test_history_records_all_candidates() -> None:
    scores = {("a",): 0.5, ("b",): 0.9, ("b", "a"): 0.95}
    result = greedy_soup_selection(["a", "b"], make_evaluator(scores))
    assert len(result.history) == 2
    assert result.individual_scores == {"a": 0.5, "b": 0.9}


def test_rejects_empty_candidates() -> None:
    with pytest.raises(ConfigurationError):
        greedy_soup_selection([], make_evaluator({}))


def test_rejects_duplicate_candidates() -> None:
    with pytest.raises(ConfigurationError):
        greedy_soup_selection(["a", "a"], make_evaluator({}))


def test_is_improvement() -> None:
    assert is_improvement(0.6, 0.5, MetricDirection.MAXIMIZE)
    assert not is_improvement(0.5, 0.5, MetricDirection.MAXIMIZE)
    assert is_improvement(0.4, 0.5, MetricDirection.MINIMIZE)
