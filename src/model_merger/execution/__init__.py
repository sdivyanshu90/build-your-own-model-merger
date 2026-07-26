"""Execution: device, progress, streaming engine, planner, executor, verification."""

from __future__ import annotations

from .device import resolve_device, to_device
from .executor import execute_merge
from .planner import PreparedMerge, build_strategy, prepare_merge
from .progress import ProgressReporter
from .streaming import TensorMergeEngine
from .verification import verify_output

__all__ = [
    "resolve_device",
    "to_device",
    "ProgressReporter",
    "TensorMergeEngine",
    "PreparedMerge",
    "prepare_merge",
    "build_strategy",
    "execute_merge",
    "verify_output",
]
