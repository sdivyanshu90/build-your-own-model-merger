"""SLERP via the Python API, with error handling. Run: python examples/python_slerp.py"""

from __future__ import annotations

from _shared import make_tiny_models

from model_merger import MergeConfig, ModelMergerError, merge_models
from model_merger.config.models import AlgorithmConfig, ModelRef, OutputConfig


def main() -> None:
    base, models = make_tiny_models(2)
    config = MergeConfig(
        algorithm=AlgorithmConfig(type="slerp", t=0.5, dot_threshold=0.9995),
        models=[ModelRef(path=str(models[0])), ModelRef(path=str(models[1]))],
        output=OutputConfig(path=str(base / "merged-slerp"), overwrite=True),
    )
    try:
        report = merge_models(config)
    except ModelMergerError as error:
        # Domain errors carry an actionable message and an exit code.
        print(f"merge failed [{type(error).__name__}]: {error.message}")
        raise SystemExit(error.exit_code) from error

    print(f"SLERP(t=0.5) -> {report.output_path}")
    print(f"algorithm params: {report.algorithm_params}")
    print(f"per-rule counts: {report.per_rule_counts}")


if __name__ == "__main__":
    main()
