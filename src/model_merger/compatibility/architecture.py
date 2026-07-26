"""Architecture-level compatibility (Hugging Face ``config.json``).

Compares model type, architecture class names, and vocabulary size, and rejects
quantized checkpoints (which need dequantization before any meaningful averaging).
Checkpoints without a config (plain safetensors / pytorch files) are noted as
"unverified" -- shape/dtype compatibility (in :mod:`.tensors`) still applies.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..checkpoints.huggingface_checkpoint import HuggingFaceCheckpoint
from ..checkpoints.metadata import ModelConfigInfo
from ..types import Severity
from .report import CompatibilityReport

__all__ = ["analyze_architecture"]


def _config_infos(checkpoints: Sequence[object]) -> list[ModelConfigInfo | None]:
    infos: list[ModelConfigInfo | None] = []
    for ckpt in checkpoints:
        if isinstance(ckpt, HuggingFaceCheckpoint):
            infos.append(ckpt.config_info)
        else:
            infos.append(None)
    return infos


def analyze_architecture(
    checkpoints: Sequence[object],
    *,
    require_matching_config: bool = True,
    require_matching_vocab_size: bool = True,
) -> CompatibilityReport:
    """Return architecture-compatibility findings across ``checkpoints``."""

    report = CompatibilityReport()
    infos = _config_infos(checkpoints)
    present = [info for info in infos if info is not None]

    if not present:
        report.add(
            Severity.INFO,
            "arch.no_config",
            "no config.json available; architecture compatibility not verified "
            "(tensor shapes/dtypes are still checked)",
        )
        return report

    for index, info in enumerate(infos):
        if info is not None and info.is_quantized:
            report.add(
                Severity.FATAL,
                "arch.quantized",
                f"model {index} is quantized (quantization_config present); "
                f"quantized merging is not supported",
            )

    reference = present[0]
    for index, info in enumerate(infos):
        if info is None:
            report.add(
                Severity.WARNING,
                "arch.missing_config",
                f"model {index} has no config.json; architecture cannot be compared",
            )
            continue
        if info.model_type != reference.model_type:
            report.add(
                Severity.FATAL,
                "arch.model_type_mismatch",
                f"model_type mismatch: {reference.model_type!r} vs {info.model_type!r}",
            )
        if require_matching_config and set(info.architectures) != set(reference.architectures):
            report.add(
                Severity.ERROR,
                "arch.architectures_mismatch",
                f"architectures differ: {reference.architectures} vs {info.architectures}",
            )
        if (
            require_matching_vocab_size
            and info.vocab_size is not None
            and reference.vocab_size is not None
            and info.vocab_size != reference.vocab_size
        ):
            report.add(
                Severity.ERROR,
                "arch.vocab_size_mismatch",
                f"vocab_size differs: {reference.vocab_size} vs {info.vocab_size}",
            )
        if info.transformers_version != reference.transformers_version:
            report.add(
                Severity.INFO,
                "arch.transformers_version",
                f"transformers_version differs: {reference.transformers_version} "
                f"vs {info.transformers_version}",
            )
    return report
