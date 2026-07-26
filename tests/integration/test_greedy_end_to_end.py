"""End-to-end greedy soup via a real command evaluator (subprocess, no shell)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from model_merger import merge_models
from model_merger.config.models import (
    AlgorithmConfig,
    EvaluatorConfig,
    GreedyConfig,
    MergeConfig,
    ModelRef,
    OutputConfig,
)

pytestmark = pytest.mark.integration

# A tiny evaluator: prints {"score": -mean|embed|} so smaller-magnitude soups win.
_EVAL_SCRIPT = """
import json, sys
from safetensors import safe_open
with safe_open(sys.argv[1] + "/model.safetensors", framework="pt") as f:
    t = f.get_tensor("model.embed_tokens.weight")
print(json.dumps({"score": float(-t.abs().mean())}))
"""


def _greedy_config(models, out, script: Path) -> MergeConfig:
    return MergeConfig(
        algorithm=AlgorithmConfig(type="greedy_soup"),
        models=[ModelRef(path=str(m), name=f"m{i}") for i, m in enumerate(models)],
        output=OutputConfig(path=str(out), overwrite=True),
        greedy=GreedyConfig(
            direction="maximize",
            evaluator=EvaluatorConfig(
                type="command",
                command=[sys.executable, str(script), "{model_path}"],
                metric_key="score",
            ),
        ),
    )


def test_greedy_soup_end_to_end(three_models, tmp_path: Path) -> None:
    script = tmp_path / "eval.py"
    script.write_text(_EVAL_SCRIPT)
    out = tmp_path / "greedy"
    report = merge_models(_greedy_config(three_models, out, script))
    assert report.verification.passed
    assert report.algorithm == "greedy_soup"
    assert report.algorithm_params["accepted"]
    assert len(report.greedy_history) == len(three_models)
    assert (out / "model.safetensors").is_file()


def test_greedy_history_recorded(three_models, tmp_path: Path) -> None:
    script = tmp_path / "eval.py"
    script.write_text(_EVAL_SCRIPT)
    report = merge_models(_greedy_config(three_models, tmp_path / "greedy", script))
    # First step is always the seed (accepted).
    assert report.greedy_history[0].accepted
    assert all(step.reason for step in report.greedy_history)
