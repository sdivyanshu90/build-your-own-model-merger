"""Stable public Python API.

Four functions form the supported surface:

* :func:`merge_models` -- run a merge from a :class:`MergeConfig`, returning a
  :class:`~model_merger.reporting.models.MergeReport`.
* :func:`plan_merge` -- compute the plan without writing anything (dry run).
* :func:`inspect_model` -- summarize a checkpoint's tensors and metadata.
* :func:`verify_output` -- structurally verify a written checkpoint.

Everything else in the package is an implementation detail.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .checkpoints import open_checkpoint
from .checkpoints.base import element_size
from .checkpoints.huggingface_checkpoint import HuggingFaceCheckpoint
from .config.models import MergeConfig
from .execution.executor import execute_merge
from .execution.planner import prepare_merge
from .execution.verification import verify_output as _verify_output
from .reporting.models import MergePlan, MergeReport, VerificationResult
from .reporting.serialization import write_report_json, write_report_markdown
from .types import dtype_name

__all__ = ["merge_models", "plan_merge", "inspect_model", "verify_output"]


def merge_models(config: MergeConfig, *, progress: bool = False) -> MergeReport:
    """Execute a merge and return its report.

    If ``config.output.write_report`` is set, a ``merge_report.json`` (and, when
    ``report_markdown`` is set, ``merge_report.md``) is written into the output
    directory after the merge is verified.

    Args:
        config: A validated merge configuration.
        progress: Show a terminal progress bar (off by default for library use).

    Returns:
        The :class:`MergeReport` describing the completed, verified merge.
    """

    report = execute_merge(config, progress=progress)
    if config.output.write_report:
        output_dir = Path(report.output_path)
        if output_dir.is_dir():
            write_report_json(report, output_dir / "merge_report.json")
            if config.output.report_markdown:
                write_report_markdown(report, output_dir / "merge_report.md")
    return report


def plan_merge(config: MergeConfig) -> MergePlan:
    """Compute the merge plan without writing any output (a dry run)."""

    prepared = prepare_merge(config, hash_inputs=False)
    try:
        return prepared.plan
    finally:
        prepared.close()


def inspect_model(path: str | Path, *, allow_unsafe: bool = False) -> dict[str, Any]:
    """Return a JSON-friendly summary of a checkpoint.

    Includes tensor/parameter counts, byte size, a dtype histogram, a sample of
    keys, and (for Hugging Face directories) architecture/config information.
    """

    with open_checkpoint(path, allow_unsafe=allow_unsafe) as checkpoint:
        keys = checkpoint.keys()
        dtype_hist: Counter[str] = Counter()
        total_params = 0
        total_bytes = 0
        for key in keys:
            info = checkpoint.tensor_info(key)
            dtype_hist[dtype_name(info.dtype)] += 1
            total_params += info.num_elements
            total_bytes += info.num_elements * element_size(info.dtype)

        summary: dict[str, Any] = {
            "path": str(Path(path)),
            "format": checkpoint.format,
            "tensor_count": len(keys),
            "total_parameters": total_params,
            "total_bytes": total_bytes,
            "dtype_histogram": dict(sorted(dtype_hist.items())),
            "sample_keys": keys[:10],
            "metadata": checkpoint.raw_metadata(),
        }
        if isinstance(checkpoint, HuggingFaceCheckpoint) and checkpoint.config_info is not None:
            arch = checkpoint.config_info
            summary["architecture"] = {
                "model_type": arch.model_type,
                "architectures": list(arch.architectures),
                "vocab_size": arch.vocab_size,
                "hidden_size": arch.hidden_size,
                "tie_word_embeddings": arch.tie_word_embeddings,
                "is_quantized": arch.is_quantized,
            }
            summary["ancillary_files"] = [file.name for file in checkpoint.ancillary_files]
        return summary


def verify_output(
    path: str | Path, *, check_finite: bool = True, allow_unsafe: bool = False
) -> VerificationResult:
    """Structurally verify a written checkpoint (see execution.verification)."""

    return _verify_output(path, check_finite=check_finite, allow_unsafe=allow_unsafe)
