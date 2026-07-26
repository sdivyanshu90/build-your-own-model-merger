"""Helpers for assembling reports: run ids, timestamps, greedy conversion."""

from __future__ import annotations

import datetime
import uuid

from ..algorithms.greedy_soup import GreedySoupResult
from .models import GreedyStepRecord

__all__ = ["new_run_id", "utc_timestamp", "greedy_history_records"]


def new_run_id() -> str:
    """Return a short, unique run identifier."""

    return uuid.uuid4().hex[:16]


def utc_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string."""

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def greedy_history_records(result: GreedySoupResult) -> tuple[GreedyStepRecord, ...]:
    """Convert a :class:`GreedySoupResult` history to serializable records."""

    return tuple(
        GreedyStepRecord(
            candidate=step.candidate,
            trial_set=step.trial_set,
            score=step.score,
            accepted=step.accepted,
            reason=step.reason,
        )
        for step in result.history
    )
