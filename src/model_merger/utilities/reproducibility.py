"""Reproducibility helpers: environment capture and deterministic seeding.

The merge itself is deterministic given identical inputs, algorithm, and
ordering -- it performs no sampling.  Seeding is provided for evaluators and for
any downstream code that does sample.  Environment capture records the exact
library and platform versions so a report can be reproduced (bit-for-bit within
the same hardware/library stack; see ``docs/limitations.md`` for cross-platform
caveats).
"""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass

import torch

__all__ = ["EnvironmentInfo", "capture_environment", "seed_everything"]


def _safe_version(module_name: str) -> str | None:
    try:
        import importlib.metadata as meta

        return meta.version(module_name)
    except Exception:
        return None


@dataclass(frozen=True)
class EnvironmentInfo:
    """Snapshot of the runtime relevant to reproducibility."""

    tool_version: str
    python_version: str
    platform: str
    torch_version: str
    numpy_version: str | None
    safetensors_version: str | None
    transformers_version: str | None
    cuda_available: bool
    cuda_version: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def capture_environment(tool_version: str) -> EnvironmentInfo:
    """Collect version and platform information for the merge report."""

    return EnvironmentInfo(
        tool_version=tool_version,
        python_version=platform.python_version(),
        platform=platform.platform(),
        torch_version=torch.__version__,
        numpy_version=_safe_version("numpy"),
        safetensors_version=_safe_version("safetensors"),
        transformers_version=_safe_version("transformers"),
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
    )


def seed_everything(seed: int) -> None:
    """Seed Python-visible RNGs for reproducible evaluation.

    Seeds :mod:`random`, NumPy, and torch (CPU and CUDA).  The merge core does
    not sample, but evaluators may; call this before running them.
    """

    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        torch.cuda.manual_seed_all(seed)
