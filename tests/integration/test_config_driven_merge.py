"""Merges driven by YAML/JSON configuration files."""

from __future__ import annotations

from pathlib import Path

import pytest

from model_merger import MergeConfig, merge_models, plan_merge

pytestmark = pytest.mark.integration


def _write_yaml(path: Path, models, out, body: str) -> Path:
    lines = ["models:"] + [f"  - {{path: {m}}}" for m in models]
    text = body + "\n" + "\n".join(lines) + f"\noutput: {{path: {out}, overwrite: true}}\n"
    path.write_text(text)
    return path


def test_yaml_uniform_merge(three_models, tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "cfg.yaml",
        three_models,
        tmp_path / "out",
        "algorithm: {type: uniform_soup}",
    )
    config = MergeConfig.from_file(config_path)
    report = merge_models(config)
    assert report.verification.passed


def test_yaml_layerwise_slerp(two_models, tmp_path: Path) -> None:
    body = (
        "algorithm: {type: slerp, t: 0.5}\n"
        "rules:\n"
        "  - name: embeddings\n"
        "    match: {regex: '.*embed.*'}\n"
        "    algorithm: {type: linear, t: 0.2}\n"
        "  - name: upper-layers\n"
        "    match: {layer_range: {start: 1, end: 1}}\n"
        "    algorithm: {type: slerp, t: 0.8}"
    )
    config_path = _write_yaml(tmp_path / "cfg.yaml", two_models, tmp_path / "out", body)
    config = MergeConfig.from_file(config_path)
    plan = plan_merge(config)
    assert "embeddings" in plan.per_rule_counts
    assert "upper-layers" in plan.per_rule_counts
    report = merge_models(config)
    assert report.verification.passed


def test_dry_run_plan_writes_nothing(three_models, tmp_path: Path) -> None:
    out = tmp_path / "out"
    config_path = _write_yaml(
        tmp_path / "cfg.yaml", three_models, out, "algorithm: {type: uniform_soup}"
    )
    config = MergeConfig.from_file(config_path)
    plan = plan_merge(config)
    assert plan.tensor_count > 0
    assert not out.exists()  # planning must not create output
