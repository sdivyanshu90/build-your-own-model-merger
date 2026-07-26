"""Execute a merge plan: write output atomically, verify it, build the report.

This module ties the pieces together for the four tensor-level algorithms and
orchestrates greedy soup (which is model selection on top of uniform soups).  It
enforces the safety contract: disk preflight before writing, atomic staging, and
post-write verification -- a merge is never reported successful until verification
passes.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import torch

from ..algorithms.greedy_soup import greedy_soup_selection
from ..checkpoints.huggingface_checkpoint import HuggingFaceCheckpoint
from ..checkpoints.writer import (
    PyTorchStateDictWriter,
    SafetensorsShardWriter,
    copy_ancillary_file,
)
from ..config.models import AlgorithmConfig, AncillaryConfig, MergeConfig, ModelRef, OutputConfig
from ..evaluation import build_evaluator
from ..exceptions import ConfigurationError, MergeExecutionError
from ..logging import get_logger
from ..reporting.generator import greedy_history_records, utc_timestamp
from ..reporting.models import MergeReport
from ..types import AlgorithmType, OutputFormat
from ..utilities.filesystem import AtomicDirectory, check_free_space
from ..utilities.hashing import hash_file
from ..utilities.reproducibility import capture_environment, seed_everything
from .planner import PreparedMerge, prepare_merge
from .progress import ProgressReporter
from .streaming import TensorMergeEngine
from .verification import content_hash, verify_output

__all__ = ["execute_merge"]

_LOGGER = get_logger(__name__)
_TOOL_VERSION = "0.1.0"


def execute_merge(
    config: MergeConfig, *, progress: bool = False, hash_inputs: bool = True
) -> MergeReport:
    """Execute a merge end to end and return its report.

    Dispatches greedy soup to its orchestrator; all other algorithms use the
    streaming tensor path.
    """

    seed_everything(config.seed)
    if config.algorithm.type is AlgorithmType.GREEDY_SOUP:
        return _execute_greedy(config, progress=progress)
    prepared = prepare_merge(config, hash_inputs=hash_inputs)
    try:
        return _execute_prepared(prepared, progress=progress)
    finally:
        prepared.close()


def _merge_key(prepared: PreparedMerge, engine: TensorMergeEngine, key: str) -> torch.Tensor:
    if key in prepared.passthrough:
        owner = prepared.passthrough[key]
        return prepared.checkpoints[owner].get_tensor(key).to("cpu")
    sources = [checkpoint.get_tensor(key) for checkpoint in prepared.checkpoints]
    if prepared.key_is_float[key]:
        strategy = prepared.strategies[prepared.key_rule[key]]
        return engine.merge_float(key, sources, strategy, prepared.key_output_dtype[key])
    return engine.resolve_non_float(key, sources)


def _execute_prepared(
    prepared: PreparedMerge,
    *,
    progress: bool = False,
    greedy_history: tuple = (),
    extra_warnings: tuple[str, ...] = (),
) -> MergeReport:
    config = prepared.config
    output_path = config.resolved_output_path()
    check_free_space(output_path.parent, prepared.shard_plan.total_bytes)

    engine = TensorMergeEngine(prepared.precision, prepared.non_float_policy, prepared.device)
    reporter = ProgressReporter(enabled=progress)
    start = time.perf_counter()

    with AtomicDirectory(output_path, overwrite=config.output.overwrite) as staging:
        writer = _build_writer(config, staging, prepared)
        with reporter.task("merging tensors", len(prepared.output_order)) as advance:
            for key in prepared.output_order:
                tensor = _merge_key(prepared, engine, key)
                writer.add(key, tensor)
                del tensor
                advance(1)
        writer.finalize()
        ancillary_warnings = _reconcile_ancillary(prepared, staging)

    duration = time.perf_counter() - start

    output_files = sorted(path for path in output_path.iterdir() if path.is_file())
    hashes = {path.name: hash_file(path) for path in output_files}
    expected_dtypes = {key: prepared.key_output_dtype[key] for key in prepared.output_order}
    verification = verify_output(
        output_path,
        expected_keys=set(prepared.output_order),
        expected_dtypes=expected_dtypes,
        check_finite=prepared.precision.validate_finite,
        allow_unsafe=config.allow_unsafe_pytorch,
    )
    if not verification.passed:
        raise MergeExecutionError(
            f"output verification failed: {'; '.join(verification.messages) or 'unknown reason'}"
        )

    plan = prepared.plan
    warnings = tuple(prepared.warnings) + tuple(extra_warnings) + tuple(ancillary_warnings)
    non_float_count = sum(
        1 for key in prepared.output_order if not prepared.key_is_float.get(key, False)
    )

    return MergeReport(
        run_id=plan.run_id,
        timestamp=utc_timestamp(),
        tool_version=_TOOL_VERSION,
        algorithm=plan.algorithm,
        algorithm_params=plan.algorithm_params,
        models=plan.models,
        output_path=str(output_path),
        output_format=config.output.format.value,
        output_hashes=hashes,
        output_shards=tuple(shard.filename for shard in prepared.shard_plan.shards),
        tensor_count=len(prepared.output_order),
        per_rule_counts=plan.per_rule_counts,
        non_float_count=non_float_count,
        skipped_keys=(),
        duration_seconds=duration,
        environment=capture_environment(_TOOL_VERSION).to_dict(),
        ancillary=plan.ancillary,
        verification=verification,
        warnings=warnings,
        errors=(),
        peak_memory_bytes=_peak_rss_bytes(),
        greedy_history=greedy_history,
        output_hash=content_hash(output_path, allow_unsafe=config.allow_unsafe_pytorch),
    )


def _build_writer(
    config: MergeConfig, staging: Path, prepared: PreparedMerge
) -> SafetensorsShardWriter | PyTorchStateDictWriter:
    if config.output.format is OutputFormat.SAFETENSORS:
        metadata = {
            "merged_by": f"model-merger/{_TOOL_VERSION}",
            "algorithm": config.algorithm.type.value,
        }
        return SafetensorsShardWriter(staging, prepared.shard_plan, metadata=metadata)
    return PyTorchStateDictWriter(staging)


def _reconcile_ancillary(prepared: PreparedMerge, staging: Path) -> list[str]:
    """Copy/reconcile ancillary files into the staging directory."""

    config = prepared.config
    strategy = config.ancillary.strategy
    warnings: list[str] = []

    hf_checkpoints = [
        (name, ckpt)
        for name, ckpt in zip(prepared.model_names, prepared.checkpoints, strict=True)
        if isinstance(ckpt, HuggingFaceCheckpoint)
    ]
    if not hf_checkpoints:
        return warnings

    if config.ancillary.base_model is not None:
        source_index = next(
            index
            for index, (name, _) in enumerate(hf_checkpoints)
            if name == config.ancillary.base_model
        )
    else:
        source_index = 0
    source_name, source_ckpt = hf_checkpoints[source_index]

    if strategy in {"require_equal", "fail_on_difference", "warn"}:
        differences = _ancillary_differences(hf_checkpoints)
        if differences:
            message = f"ancillary files differ across models: {sorted(differences)[:5]}"
            if strategy in {"require_equal", "fail_on_difference"}:
                raise MergeExecutionError(message)
            warnings.append(message)

    for source_file in source_ckpt.ancillary_files:
        copy_ancillary_file(source_file, staging)
    _LOGGER.debug(
        "copied %d ancillary files from %s", len(source_ckpt.ancillary_files), source_name
    )
    return warnings


def _ancillary_differences(hf_checkpoints: list[tuple[str, HuggingFaceCheckpoint]]) -> set[str]:
    reference_name, reference = hf_checkpoints[0]
    ref_hashes = {path.name: hash_file(path) for path in reference.ancillary_files}
    differing: set[str] = set()
    for _, other in hf_checkpoints[1:]:
        other_hashes = {path.name: hash_file(path) for path in other.ancillary_files}
        for name in set(ref_hashes) | set(other_hashes):
            if ref_hashes.get(name) != other_hashes.get(name):
                differing.add(name)
    return differing


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - non-unix
        return None
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes. Assume KiB (dominant CI platform).
    return int(max_rss) * 1024


# --- Greedy soup orchestration ---


def _execute_greedy(config: MergeConfig, *, progress: bool) -> MergeReport:
    greedy = config.greedy
    assert greedy is not None
    evaluator = build_evaluator(greedy.evaluator)
    direction = greedy.direction
    candidate_names = list(config.model_names)
    ref_by_name = {ref.name: ref for ref in config.models}

    temp_root = Path(greedy.temp_dir) if greedy.temp_dir else None
    temp_base = Path(tempfile.mkdtemp(prefix="greedy-soup-", dir=temp_root))
    cache: dict[tuple[str, ...], float] = {}
    counter = {"n": 0}

    def evaluate(subset: tuple[str, ...]) -> float:
        canonical = tuple(sorted(subset))
        if greedy.cache and canonical in cache:
            return cache[canonical]
        counter["n"] += 1
        soup_dir = temp_base / f"soup-{counter['n']:04d}"
        subset_refs = [ref_by_name[name] for name in subset]
        sub_config = _uniform_subconfig(config, subset_refs, soup_dir)
        prepared = prepare_merge(sub_config, hash_inputs=False)
        try:
            _execute_prepared(prepared, progress=False)
        finally:
            prepared.close()
        try:
            score = evaluator.evaluate(soup_dir)
        finally:
            if not greedy.keep_temp:
                shutil.rmtree(soup_dir, ignore_errors=True)
        if greedy.cache:
            cache[canonical] = score
        return score

    try:
        result = greedy_soup_selection(candidate_names, evaluate, direction=direction)
    finally:
        if not greedy.keep_temp:
            shutil.rmtree(temp_base, ignore_errors=True)

    _LOGGER.info(
        "greedy soup selected %d/%d models: %s (score %.6g)",
        len(result.accepted),
        len(candidate_names),
        result.accepted,
        result.best_score,
    )

    accepted_refs = [ref_by_name[name] for name in result.accepted]
    final_config = _uniform_subconfig(
        config, accepted_refs, config.resolved_output_path(), for_final=True
    )
    prepared = prepare_merge(final_config, hash_inputs=True)
    try:
        report = _execute_prepared(
            prepared,
            progress=progress,
            greedy_history=greedy_history_records(result),
            extra_warnings=(
                f"greedy soup selected {len(result.accepted)}/{len(candidate_names)} models",
            ),
        )
    finally:
        prepared.close()
    # Overwrite the algorithm label so the report reflects greedy provenance.
    return replace(
        report,
        algorithm="greedy_soup",
        algorithm_params={
            "type": "greedy_soup",
            "direction": direction.value,
            "accepted": list(result.accepted),
            "best_score": result.best_score,
            "individual_scores": result.individual_scores,
        },
    )


def _uniform_subconfig(
    config: MergeConfig,
    refs: list[ModelRef],
    output_dir: Path,
    *,
    for_final: bool = False,
) -> MergeConfig:
    """Build a uniform-soup config over ``refs`` writing to ``output_dir``."""

    if not refs:
        raise ConfigurationError("cannot build a soup with no models")
    model_copies = [ModelRef(path=ref.path, name=ref.name, weight=None) for ref in refs]
    output = OutputConfig(
        path=str(output_dir),
        format=config.output.format,
        max_shard_size=config.output.max_shard_size,
        overwrite=config.output.overwrite if for_final else True,
        atomic=True,
        write_report=False,
        report_markdown=False,
    )
    # Copy ancillary files from the first model in the subset (its config/tokenizer
    # are needed for the evaluator to load the soup, and base_model may not be in
    # the accepted subset).
    sub = MergeConfig(
        algorithm=AlgorithmConfig(type=AlgorithmType.UNIFORM_SOUP),
        models=model_copies,
        output=output,
        precision=config.precision,
        compatibility=config.compatibility,
        non_float_tensors=config.non_float_tensors,
        rules=[],
        greedy=None,
        ancillary=AncillaryConfig(strategy="first"),
        device=config.device,
        allow_unsafe_pytorch=config.allow_unsafe_pytorch,
        seed=config.seed,
    )
    sub.base_dir = config.base_dir
    return sub
