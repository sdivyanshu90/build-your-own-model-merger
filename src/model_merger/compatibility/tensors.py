"""Tensor-level compatibility analysis.

Compares the *shapes and dtypes* of every model's tensors using ``tensor_info``
only -- no tensor data is loaded, so this is cheap even for large checkpoints.

The analysis both produces compatibility findings and computes the merge plan's
key sets:

* ``merge_keys``  -- keys present in *all* models (the ones actually merged).
* ``passthrough`` -- keys present in only some models; allowed only when
  ``allow_missing``/``allow_extra`` is set, in which case each is copied verbatim
  from the first model that has it.
* shape mismatches on merge keys are fatal; dtype mismatches are errors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..checkpoints.base import Checkpoint, TensorInfo
from ..types import Severity, dtype_name
from .report import CompatibilityReport

__all__ = ["TensorCompatibility", "analyze_tensors"]


@dataclass
class TensorCompatibility:
    """Result of tensor compatibility analysis."""

    merge_keys: list[str]
    passthrough: dict[str, int]  # key -> owning model index
    info_by_key: dict[str, TensorInfo]  # canonical (model 0) info for merge keys
    report: CompatibilityReport = field(default_factory=CompatibilityReport)


def analyze_tensors(
    checkpoints: Sequence[Checkpoint],
    *,
    allow_missing: bool = False,
    allow_extra: bool = False,
) -> TensorCompatibility:
    """Analyze tensor compatibility across ``checkpoints``.

    Args:
        checkpoints: Source checkpoints (order defines model indices).
        allow_missing: Permit keys absent from some models (passthrough).
        allow_extra: Permit keys present in only some models (passthrough).

    Returns:
        A :class:`TensorCompatibility` with merge/passthrough key sets and findings.
    """

    report = CompatibilityReport()
    key_sets = [set(ckpt.keys()) for ckpt in checkpoints]
    union: set[str] = set().union(*key_sets) if key_sets else set()
    intersection: set[str] = set(key_sets[0]).intersection(*key_sets[1:]) if key_sets else set()

    partial = sorted(union - intersection)
    passthrough: dict[str, int] = {}
    if partial:
        severity = Severity.WARNING if (allow_missing or allow_extra) else Severity.ERROR
        report.add(
            severity,
            "tensor.key_set_mismatch",
            f"{len(partial)} tensor key(s) are not present in all models "
            f"(e.g. {partial[:3]}); "
            + (
                "they will be copied from the first model that has them"
                if severity is Severity.WARNING
                else "set allow_missing_keys/allow_extra_keys to permit partial keys"
            ),
        )
        if allow_missing or allow_extra:
            for key in partial:
                for index, keys in enumerate(key_sets):
                    if key in keys:
                        passthrough[key] = index
                        break

    merge_keys = sorted(intersection)
    info_by_key: dict[str, TensorInfo] = {}
    for key in merge_keys:
        infos = [ckpt.tensor_info(key) for ckpt in checkpoints]
        reference = infos[0]
        info_by_key[key] = reference
        for index, info in enumerate(infos[1:], start=1):
            if info.shape != reference.shape:
                report.add(
                    Severity.FATAL,
                    "tensor.shape_mismatch",
                    f"shape mismatch for {key!r}: model 0 {reference.shape} "
                    f"vs model {index} {info.shape}",
                )
            elif info.dtype != reference.dtype:
                report.add(
                    Severity.ERROR,
                    "tensor.dtype_mismatch",
                    f"dtype mismatch for {key!r}: model 0 {dtype_name(reference.dtype)} "
                    f"vs model {index} {dtype_name(info.dtype)}",
                )

    if not merge_keys and not passthrough:
        report.add(
            Severity.FATAL,
            "tensor.no_common_keys",
            "models share no common tensor keys; nothing to merge",
        )

    return TensorCompatibility(
        merge_keys=merge_keys,
        passthrough=passthrough,
        info_by_key=info_by_key,
        report=report,
    )
