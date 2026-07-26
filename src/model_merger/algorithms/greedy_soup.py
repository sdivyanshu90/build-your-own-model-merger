"""Greedy model soup selection.

Greedy soups (Wortsman et al., 2022) build a soup incrementally:

1. Score every candidate individually.
2. Seed the soup with the single best candidate.
3. Consider the remaining candidates in descending score order; tentatively add
   each to the soup, re-score, and keep it only if the metric strictly improves.

This module implements the *selection policy* as a pure function parameterized by
an ``evaluate`` callback.  ``evaluate(subset)`` returns the metric for the uniform
soup of ``subset``.  Decoupling selection from checkpoint I/O makes the policy
exhaustively unit-testable with a synthetic evaluator, and lets the executor
supply the real "build soup, write it, run evaluator" implementation.

Determinism: candidates are ordered by ``(score, original_index)`` so ties break
by input order, and only *strict* improvements are accepted -- identical runs
produce identical soups.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..exceptions import ConfigurationError
from ..types import MetricDirection

__all__ = ["GreedyStep", "GreedySoupResult", "greedy_soup_selection", "is_improvement"]


@dataclass(frozen=True)
class GreedyStep:
    """One accept/reject decision in the greedy process."""

    candidate: str
    trial_set: tuple[str, ...]
    score: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class GreedySoupResult:
    """Outcome of greedy selection."""

    accepted: tuple[str, ...]
    best_score: float
    individual_scores: dict[str, float]
    history: tuple[GreedyStep, ...] = field(default_factory=tuple)


def is_improvement(candidate: float, incumbent: float, direction: MetricDirection) -> bool:
    """Return True if ``candidate`` strictly improves on ``incumbent``."""

    if direction is MetricDirection.MAXIMIZE:
        return candidate > incumbent
    return candidate < incumbent


def greedy_soup_selection(
    candidates: Sequence[str],
    evaluate: Callable[[tuple[str, ...]], float],
    *,
    direction: MetricDirection = MetricDirection.MAXIMIZE,
) -> GreedySoupResult:
    """Run greedy soup selection.

    Args:
        candidates: Unique candidate model names.
        evaluate: Callback returning the metric for the uniform soup of a subset.
        direction: Whether higher or lower metric is better.

    Returns:
        A :class:`GreedySoupResult` with the accepted subset (in acceptance
        order), the best score, per-candidate individual scores, and the full
        decision history.

    Raises:
        ConfigurationError: if ``candidates`` is empty or contains duplicates.
    """

    names = list(candidates)
    if not names:
        raise ConfigurationError("greedy soup requires at least one candidate")
    if len(set(names)) != len(names):
        raise ConfigurationError("greedy soup candidate names must be unique")

    individual: dict[str, float] = {name: float(evaluate((name,))) for name in names}
    original_index = {name: index for index, name in enumerate(names)}

    def sort_key(name: str) -> tuple[float, int]:
        score = individual[name]
        ordered = -score if direction is MetricDirection.MAXIMIZE else score
        return (ordered, original_index[name])

    ordered_names = sorted(names, key=sort_key)

    seed = ordered_names[0]
    accepted: list[str] = [seed]
    best_score = individual[seed]
    history: list[GreedyStep] = [
        GreedyStep(
            candidate=seed,
            trial_set=(seed,),
            score=best_score,
            accepted=True,
            reason="seed: best individual candidate",
        )
    ]

    for name in ordered_names[1:]:
        trial = (*accepted, name)
        score = float(evaluate(trial))
        improved = is_improvement(score, best_score, direction)
        if improved:
            accepted.append(name)
            reason = f"metric improved {best_score:.6g} -> {score:.6g}"
            best_score = score
        else:
            reason = f"metric did not improve (incumbent {best_score:.6g}, trial {score:.6g})"
        history.append(
            GreedyStep(
                candidate=name,
                trial_set=trial,
                score=score,
                accepted=improved,
                reason=reason,
            )
        )

    return GreedySoupResult(
        accepted=tuple(accepted),
        best_score=best_score,
        individual_scores=individual,
        history=tuple(history),
    )
