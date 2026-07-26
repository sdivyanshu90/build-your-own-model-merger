"""Release sanity check: build tiny models, run every algorithm, verify outputs.

Usage:
    python scripts/verify_release.py

Exits 0 if all merges verify, non-zero otherwise.  Used by ``make release-check``
and CI to confirm the installed package works end to end without any downloads.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _extra in (_ROOT / "src", _ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import generate_tiny_test_models as gen  # noqa: E402

from model_merger import MergeConfig, __version__, merge_models, verify_output  # noqa: E402
from model_merger.config.models import (  # noqa: E402
    AlgorithmConfig,
    EvaluatorConfig,
    GreedyConfig,
    ModelRef,
    OutputConfig,
)


def _models(base: Path, count: int) -> list[Path]:
    return [gen.write_model(base / f"m{i}", seed=10 + i) for i in range(count)]


def main() -> int:
    print(f"model-merger {__version__}")
    base = Path(tempfile.mkdtemp(prefix="release-check-"))
    failures: list[str] = []

    def check(name: str, config: MergeConfig) -> None:
        try:
            report = merge_models(config)
            ok = report.verification.passed and verify_output(report.output_path).passed
        except Exception as error:  # noqa: BLE001 - report any failure
            ok = False
            print(f"  [FAIL] {name}: {error}")
        if ok:
            print(f"  [ OK ] {name}")
        else:
            failures.append(name)

    three = _models(base, 3)
    out = base / "out"

    check(
        "uniform_soup",
        MergeConfig(
            algorithm=AlgorithmConfig(type="uniform_soup"),
            models=[ModelRef(path=str(p)) for p in three],
            output=OutputConfig(path=str(out / "uniform"), overwrite=True),
        ),
    )
    check(
        "weighted_soup",
        MergeConfig(
            algorithm=AlgorithmConfig(type="weighted_soup"),
            models=[
                ModelRef(path=str(p), weight=w) for p, w in zip(three, [0.5, 0.3, 0.2], strict=True)
            ],
            output=OutputConfig(path=str(out / "weighted"), overwrite=True),
        ),
    )
    check(
        "slerp",
        MergeConfig(
            algorithm=AlgorithmConfig(type="slerp", t=0.5),
            models=[ModelRef(path=str(three[0])), ModelRef(path=str(three[1]))],
            output=OutputConfig(path=str(out / "slerp"), overwrite=True),
        ),
    )
    check(
        "greedy_soup",
        MergeConfig(
            algorithm=AlgorithmConfig(type="greedy_soup"),
            models=[ModelRef(path=str(p), name=f"m{i}") for i, p in enumerate(three)],
            output=OutputConfig(path=str(out / "greedy"), overwrite=True),
            greedy=GreedyConfig(
                direction="maximize",
                evaluator=EvaluatorConfig(
                    type="command",
                    command=[
                        sys.executable,
                        str(_ROOT / "examples" / "scoring_stub.py"),
                        "{model_path}",
                    ],
                ),
            ),
        ),
    )

    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("all algorithms verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
