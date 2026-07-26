"""Compatibility validation: tensors, architecture, tokenizer -> one report."""

from __future__ import annotations

from collections.abc import Sequence

from ..checkpoints.base import Checkpoint
from ..config.models import CompatibilityConfig
from .architecture import analyze_architecture
from .report import CompatibilityReport, Finding
from .tensors import TensorCompatibility, analyze_tensors
from .tokenizer import analyze_tokenizer

__all__ = [
    "CompatibilityReport",
    "Finding",
    "TensorCompatibility",
    "analyze_tensors",
    "analyze_architecture",
    "analyze_tokenizer",
    "validate_compatibility",
]


def validate_compatibility(
    checkpoints: Sequence[Checkpoint],
    config: CompatibilityConfig,
) -> tuple[CompatibilityReport, TensorCompatibility]:
    """Run all compatibility analyses and merge their findings.

    Returns:
        A tuple of the combined :class:`CompatibilityReport` and the
        :class:`TensorCompatibility` (whose key sets drive the merge plan).
    """

    tensor_result = analyze_tensors(
        checkpoints,
        allow_missing=config.allow_missing_keys,
        allow_extra=config.allow_extra_keys,
    )
    architecture_report = analyze_architecture(
        checkpoints,
        require_matching_config=config.require_matching_config,
        require_matching_vocab_size=config.require_matching_vocab_size,
    )
    tokenizer_report = analyze_tokenizer(
        checkpoints,
        require_matching_tokenizer=config.require_matching_tokenizer,
    )

    combined = CompatibilityReport()
    combined.extend(tensor_result.report)
    combined.extend(architecture_report)
    combined.extend(tokenizer_report)
    return combined, tensor_result
