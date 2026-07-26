"""model_merger: merge compatible model checkpoints without retraining.

The public API surface is intentionally small and stable:

    from model_merger import MergeConfig, merge_models

    config = MergeConfig.from_file("configs/slerp.example.yaml")
    report = merge_models(config)
    print(report.output_path, report.output_hashes)

Anything not re-exported here is an implementation detail and may change between
minor releases.  See ``docs/python-api.md`` for the stability policy.
"""

from __future__ import annotations

from .api import inspect_model, merge_models, plan_merge, verify_output
from .config.models import MergeConfig
from .exceptions import (
    CheckpointError,
    CompatibilityError,
    ConfigurationError,
    EvaluationError,
    InsufficientDiskSpaceError,
    MergeExecutionError,
    ModelMergerError,
    NumericalError,
    OutputExistsError,
    TensorMismatchError,
    UnsafeCheckpointError,
    VerificationError,
)
from .reporting.models import MergePlan, MergeReport

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # High-level API
    "MergeConfig",
    "merge_models",
    "plan_merge",
    "inspect_model",
    "verify_output",
    "MergeReport",
    "MergePlan",
    # Exceptions
    "ModelMergerError",
    "ConfigurationError",
    "CheckpointError",
    "UnsafeCheckpointError",
    "CompatibilityError",
    "TensorMismatchError",
    "NumericalError",
    "EvaluationError",
    "OutputExistsError",
    "InsufficientDiskSpaceError",
    "MergeExecutionError",
    "VerificationError",
]
