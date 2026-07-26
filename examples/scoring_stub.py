"""External evaluator stub used by configs/greedy_soup.example.yaml.

Usage (invoked by the merger, not directly):
    python examples/scoring_stub.py <merged_model_dir>

Prints a JSON object ``{"score": <float>}`` on stdout.  Replace the body with a
real evaluation (load the model, run it on a held-out set, print the metric).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def score(model_path: Path) -> float:
    from safetensors import safe_open

    with safe_open(str(model_path / "model.safetensors"), framework="pt") as handle:
        embed = handle.get_tensor("model.embed_tokens.weight")
    return float(-embed.abs().mean())


def main() -> None:
    if len(sys.argv) != 2:
        print(json.dumps({"error": "expected exactly one model path argument"}))
        raise SystemExit(2)
    print(json.dumps({"score": score(Path(sys.argv[1]))}))


if __name__ == "__main__":
    main()
