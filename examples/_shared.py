"""Shared setup for the runnable examples.

Makes the repo importable without installation and provides a helper to
materialize tiny models in a temporary directory so every example runs
end-to-end with no downloads.
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


def make_tiny_models(count: int = 3) -> tuple[Path, list[Path]]:
    """Create ``count`` tiny models in a temp dir; return (base_dir, model_paths)."""

    base = Path(tempfile.mkdtemp(prefix="model-merger-example-"))
    models = []
    for index in range(count):
        name = f"model-{chr(ord('a') + index)}"
        models.append(gen.write_model(base / name, seed=1000 + index))
    return base, models
