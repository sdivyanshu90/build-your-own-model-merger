"""Structured plan and report data models.

These are plain frozen dataclasses (not Pydantic) so they are cheap to build in
the hot path and easy to serialize deterministically.  ``to_dict`` on the top-
level objects returns JSON-ready structures (paths as strings, enums as values,
keys sorted where order is not meaningful).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ModelSummary",
    "TensorPlanEntry",
    "AncillaryPlan",
    "MergePlan",
    "GreedyStepRecord",
    "VerificationResult",
    "MergeReport",
]


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ModelSummary:
    """Provenance for one source model."""

    name: str
    path: str
    format: str
    tensor_count: int
    weight: float | None = None
    content_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = _jsonify(self.__dict__)
        return result


@dataclass(frozen=True)
class TensorPlanEntry:
    """The resolved plan for a single tensor."""

    key: str
    rule_name: str
    algorithm: str
    compute_dtype: str
    output_dtype: str
    shape: tuple[int, ...]
    is_non_float: bool
    output_bytes: int

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = _jsonify(self.__dict__)
        return result


@dataclass(frozen=True)
class AncillaryPlan:
    """How non-tensor files will be reconciled."""

    strategy: str
    source_model: str | None
    files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = _jsonify(self.__dict__)
        return result


@dataclass(frozen=True)
class MergePlan:
    """The complete, side-effect-free plan for a merge (produced before writing)."""

    run_id: str
    tool_version: str
    algorithm: str
    algorithm_params: dict[str, Any]
    models: tuple[ModelSummary, ...]
    output_path: str
    output_format: str
    output_dtype_policy: str
    estimated_output_bytes: int
    shard_count: int
    shard_files: tuple[str, ...]
    tensor_count: int
    per_rule_counts: dict[str, int]
    non_float_keys: tuple[str, ...]
    ancillary: AncillaryPlan
    requires_unsafe_loading: bool
    warnings: tuple[str, ...] = ()
    compatibility_summary: dict[str, Any] = field(default_factory=dict)
    tensor_entries: tuple[TensorPlanEntry, ...] = ()

    def to_dict(self, *, include_tensor_entries: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "run_id": self.run_id,
            "tool_version": self.tool_version,
            "algorithm": self.algorithm,
            "algorithm_params": _jsonify(self.algorithm_params),
            "models": [model.to_dict() for model in self.models],
            "output": {
                "path": self.output_path,
                "format": self.output_format,
                "dtype_policy": self.output_dtype_policy,
                "estimated_bytes": self.estimated_output_bytes,
                "shard_count": self.shard_count,
                "shard_files": list(self.shard_files),
            },
            "tensor_count": self.tensor_count,
            "per_rule_counts": _jsonify(self.per_rule_counts),
            "non_float_key_count": len(self.non_float_keys),
            "non_float_keys": list(self.non_float_keys),
            "ancillary": self.ancillary.to_dict(),
            "requires_unsafe_loading": self.requires_unsafe_loading,
            "warnings": list(self.warnings),
            "compatibility": _jsonify(self.compatibility_summary),
        }
        if include_tensor_entries:
            data["tensor_entries"] = [entry.to_dict() for entry in self.tensor_entries]
        return data


@dataclass(frozen=True)
class GreedyStepRecord:
    """A serializable record of one greedy accept/reject decision."""

    candidate: str
    trial_set: tuple[str, ...]
    score: float
    accepted: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = _jsonify(self.__dict__)
        return result


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of post-write verification."""

    passed: bool
    checks: dict[str, bool]
    messages: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = _jsonify(self.__dict__)
        return result


@dataclass(frozen=True)
class MergeReport:
    """The full record of an executed merge."""

    run_id: str
    timestamp: str
    tool_version: str
    algorithm: str
    algorithm_params: dict[str, Any]
    models: tuple[ModelSummary, ...]
    output_path: str
    output_format: str
    output_hashes: dict[str, str]
    output_shards: tuple[str, ...]
    tensor_count: int
    per_rule_counts: dict[str, int]
    non_float_count: int
    skipped_keys: tuple[str, ...]
    duration_seconds: float
    environment: dict[str, Any]
    ancillary: AncillaryPlan
    verification: VerificationResult
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    peak_memory_bytes: int | None = None
    greedy_history: tuple[GreedyStepRecord, ...] = ()
    output_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "tool_version": self.tool_version,
            "algorithm": self.algorithm,
            "algorithm_params": _jsonify(self.algorithm_params),
            "models": [model.to_dict() for model in self.models],
            "output": {
                "path": self.output_path,
                "format": self.output_format,
                "hashes": _jsonify(self.output_hashes),
                "shards": list(self.output_shards),
                "combined_hash": self.output_hash,
            },
            "tensors": {
                "count": self.tensor_count,
                "per_rule_counts": _jsonify(self.per_rule_counts),
                "non_float_count": self.non_float_count,
                "skipped": list(self.skipped_keys),
            },
            "duration_seconds": self.duration_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "environment": _jsonify(self.environment),
            "ancillary": self.ancillary.to_dict(),
            "verification": self.verification.to_dict(),
            "greedy_history": [step.to_dict() for step in self.greedy_history],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }

    @property
    def output_hashes_list(self) -> list[str]:
        return [self.output_hashes[name] for name in sorted(self.output_hashes)]
