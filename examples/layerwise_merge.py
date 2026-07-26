"""Layer-wise merge with per-region rules. Run: python examples/layerwise_merge.py"""

from __future__ import annotations

from _shared import make_tiny_models

from model_merger import MergeConfig, merge_models, plan_merge
from model_merger.config.models import (
    AlgorithmConfig,
    LayerRangeConfig,
    MatchConfig,
    ModelRef,
    OutputConfig,
    RuleConfig,
)


def main() -> None:
    base, models = make_tiny_models(2)
    config = MergeConfig(
        algorithm=AlgorithmConfig(type="slerp", t=0.5),
        models=[ModelRef(path=str(models[0])), ModelRef(path=str(models[1]))],
        output=OutputConfig(path=str(base / "merged-layerwise"), overwrite=True),
        rules=[
            RuleConfig(
                name="embeddings",
                priority=10,
                match=MatchConfig(regex=".*embed.*weight"),
                algorithm=AlgorithmConfig(type="linear", t=0.2),
            ),
            RuleConfig(
                name="upper-layers",
                priority=5,
                match=MatchConfig(layer_range=LayerRangeConfig(start=1, end=1)),
                algorithm=AlgorithmConfig(type="slerp", t=0.8),
            ),
        ],
    )

    # Inspect the plan (a dry run) before writing anything.
    plan = plan_merge(config)
    print("per-rule tensor counts:", plan.per_rule_counts)

    report = merge_models(config)
    print(f"merged -> {report.output_path}  (verified: {report.verification.passed})")


if __name__ == "__main__":
    main()
