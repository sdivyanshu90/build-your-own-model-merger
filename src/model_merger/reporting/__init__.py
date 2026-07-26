"""Merge plan/report data models and their serialization."""

from __future__ import annotations

from .generator import greedy_history_records, new_run_id, utc_timestamp
from .models import (
    AncillaryPlan,
    GreedyStepRecord,
    MergePlan,
    MergeReport,
    ModelSummary,
    TensorPlanEntry,
    VerificationResult,
)
from .serialization import (
    plan_to_json,
    plan_to_markdown,
    report_to_json,
    report_to_markdown,
    write_report_json,
    write_report_markdown,
)

__all__ = [
    "MergePlan",
    "MergeReport",
    "ModelSummary",
    "TensorPlanEntry",
    "AncillaryPlan",
    "GreedyStepRecord",
    "VerificationResult",
    "new_run_id",
    "utc_timestamp",
    "greedy_history_records",
    "report_to_json",
    "report_to_markdown",
    "write_report_json",
    "write_report_markdown",
    "plan_to_json",
    "plan_to_markdown",
]
