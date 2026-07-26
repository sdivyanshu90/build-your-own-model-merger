"""Uniform soup via the Python API. Run: python examples/python_uniform_soup.py"""

from __future__ import annotations

from _shared import make_tiny_models

from model_merger import MergeConfig, merge_models
from model_merger.config.models import AlgorithmConfig, ModelRef, OutputConfig


def main() -> None:
    base, models = make_tiny_models(3)
    config = MergeConfig(
        algorithm=AlgorithmConfig(type="uniform_soup"),
        models=[ModelRef(path=str(path)) for path in models],
        output=OutputConfig(path=str(base / "merged-uniform"), overwrite=True),
    )
    report = merge_models(config)
    print(f"merged {report.tensor_count} tensors -> {report.output_path}")
    print(f"verification passed: {report.verification.passed}")
    print(f"content hash: {report.output_hash}")


if __name__ == "__main__":
    main()
