"""Weighted soup via the Python API. Run: python examples/python_weighted_soup.py"""

from __future__ import annotations

from _shared import make_tiny_models

from model_merger import MergeConfig, merge_models
from model_merger.config.models import AlgorithmConfig, ModelRef, OutputConfig


def main() -> None:
    base, models = make_tiny_models(3)
    weights = [0.5, 0.3, 0.2]
    config = MergeConfig(
        algorithm=AlgorithmConfig(type="weighted_soup", normalize_weights=True),
        models=[ModelRef(path=str(p), weight=w) for p, w in zip(models, weights, strict=True)],
        output=OutputConfig(path=str(base / "merged-weighted"), overwrite=True),
    )
    report = merge_models(config)
    print(f"weights: {weights}")
    print(f"merged -> {report.output_path}  (verified: {report.verification.passed})")


if __name__ == "__main__":
    main()
