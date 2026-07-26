"""Unit tests for report/plan data models and serialization."""

from __future__ import annotations

import json

from model_merger.reporting.generator import new_run_id, utc_timestamp
from model_merger.reporting.models import (
    AncillaryPlan,
    MergeReport,
    ModelSummary,
    VerificationResult,
)
from model_merger.reporting.serialization import report_to_json, report_to_markdown


def _sample_report() -> MergeReport:
    return MergeReport(
        run_id="abc123",
        timestamp=utc_timestamp(),
        tool_version="0.1.0",
        algorithm="uniform_soup",
        algorithm_params={"type": "uniform_soup"},
        models=(ModelSummary(name="a", path="/tmp/a", format="huggingface", tensor_count=3),),
        output_path="/tmp/out",
        output_format="safetensors",
        output_hashes={"model.safetensors": "deadbeef"},
        output_shards=("model.safetensors",),
        tensor_count=3,
        per_rule_counts={"default": 3},
        non_float_count=1,
        skipped_keys=(),
        duration_seconds=1.25,
        environment={"tool_version": "0.1.0"},
        ancillary=AncillaryPlan(strategy="base", source_model="a", files=("config.json",)),
        verification=VerificationResult(passed=True, checks={"openable": True}),
    )


def test_run_id_unique() -> None:
    assert new_run_id() != new_run_id()


def test_report_json_roundtrips() -> None:
    report = _sample_report()
    data = json.loads(report_to_json(report))
    assert data["run_id"] == "abc123"
    assert data["output"]["hashes"]["model.safetensors"] == "deadbeef"
    assert data["tensors"]["count"] == 3


def test_report_markdown_contains_key_fields() -> None:
    text = report_to_markdown(_sample_report())
    assert "Merge report" in text
    assert "uniform_soup" in text
    assert "PASSED" in text


def test_verification_result_serializes() -> None:
    result = VerificationResult(passed=False, checks={"a": True, "b": False}, messages=("bad",))
    data = result.to_dict()
    assert data["passed"] is False
    assert data["checks"]["b"] is False
