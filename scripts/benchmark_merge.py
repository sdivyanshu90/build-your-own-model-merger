"""Benchmark a merge: wall-clock, peak memory, sizes, throughput.

Usage:
    python scripts/benchmark_merge.py CONFIG_PATH [--json]

Reports duration, peak resident memory (from the merge report), input/output
sizes, tensor count, throughput, device, and algorithm.  Intended for the
``performance`` characterization described in docs/memory-and-performance.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from model_merger import MergeConfig, merge_models  # noqa: E402


def _dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a merge.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = MergeConfig.from_file(args.config)
    input_bytes = sum(_dir_size(p) for p in config.resolved_model_paths())

    start = time.perf_counter()
    report = merge_models(config)
    wall = time.perf_counter() - start

    output_bytes = _dir_size(Path(report.output_path))
    throughput = output_bytes / wall if wall > 0 else 0.0
    result = {
        "algorithm": report.algorithm,
        "device": config.device,
        "wall_clock_seconds": round(wall, 4),
        "merge_seconds": round(report.duration_seconds, 4),
        "peak_memory_bytes": report.peak_memory_bytes,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "tensor_count": report.tensor_count,
        "throughput_bytes_per_s": round(throughput, 2),
        "verified": report.verification.passed,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print(f"{key:24} {value}")


if __name__ == "__main__":
    main()
