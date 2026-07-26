"""Greedy soup with a Python callable evaluator.

Run: python examples/custom_evaluator.py

The evaluator receives the path to a merged checkpoint and returns a scalar.
Here we use a deterministic synthetic metric so the example is reproducible; a
real evaluator would load the model and score it on a held-out set (kept separate
from any final test set -- see docs/model-soups.md on evaluation leakage).
"""

from __future__ import annotations

from pathlib import Path

from _shared import make_tiny_models

from model_merger import MergeConfig, merge_models
from model_merger.config.models import (
    AlgorithmConfig,
    EvaluatorConfig,
    GreedyConfig,
    ModelRef,
    OutputConfig,
)


def deterministic_metric(model_path: Path) -> float:
    """Score a merged checkpoint: higher when the embedding magnitude is smaller."""

    from safetensors import safe_open

    with safe_open(str(Path(model_path) / "model.safetensors"), framework="pt") as handle:
        embed = handle.get_tensor("model.embed_tokens.weight")
    return float(-embed.abs().mean())


def main() -> None:
    base, models = make_tiny_models(3)
    config = MergeConfig(
        algorithm=AlgorithmConfig(type="greedy_soup"),
        models=[ModelRef(path=str(p), name=f"m{i}") for i, p in enumerate(models)],
        output=OutputConfig(path=str(base / "merged-greedy"), overwrite=True),
        greedy=GreedyConfig(
            direction="maximize",
            evaluator=EvaluatorConfig(
                type="callable",
                callable="custom_evaluator:deterministic_metric",
            ),
        ),
    )
    report = merge_models(config)
    print("accepted models:", report.algorithm_params["accepted"])
    print("decision history:")
    for step in report.greedy_history:
        verdict = "accept" if step.accepted else "reject"
        print(f"  {step.candidate}: {verdict} (score {step.score:.4f})")


if __name__ == "__main__":
    main()
