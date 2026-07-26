"""Merge planning: turn a validated config into a concrete, side-effect-free plan.

Planning opens the checkpoints (read-only), validates compatibility, resolves the
per-tensor algorithm/dtype, and computes the shard layout -- but writes nothing.
The result is a :class:`PreparedMerge` that the executor consumes.  Because
planning holds open file handles, callers must ``close()`` it (or use it via the
executor, which does).

Greedy soup is *not* planned here (it is model selection, not a per-tensor plan);
the executor handles it and reuses this planner for the resulting uniform soup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..algorithms.base import MergeStrategy
from ..algorithms.slerp import LinearInterpolation, Slerp
from ..algorithms.uniform_soup import UniformSoup
from ..algorithms.weighted_soup import WeightedSoup
from ..checkpoints import Checkpoint, open_checkpoint
from ..checkpoints.base import element_size
from ..checkpoints.huggingface_checkpoint import HuggingFaceCheckpoint
from ..checkpoints.sharding import ShardPlan, plan_shards
from ..compatibility import validate_compatibility
from ..compatibility.report import CompatibilityReport
from ..compatibility.tensors import TensorCompatibility
from ..config.models import AlgorithmConfig, MergeConfig
from ..exceptions import ConfigurationError
from ..policies.layer_rules import CompiledRule, LayerRuleResolver
from ..policies.matching import LayerRange, compile_predicate
from ..policies.non_float_tensors import NonFloatTensorPolicy
from ..policies.precision import PrecisionPolicy
from ..reporting.generator import new_run_id
from ..reporting.models import AncillaryPlan, MergePlan, ModelSummary, TensorPlanEntry
from ..types import AlgorithmType, OutputFormat, dtype_name
from ..utilities.hashing import hash_file
from .device import resolve_device

__all__ = ["PreparedMerge", "prepare_merge", "build_strategy"]

_TOOL_VERSION = "0.1.0"


def build_strategy(algorithm: AlgorithmConfig, weights: list[float | None]) -> MergeStrategy:
    """Construct a tensor-level :class:`MergeStrategy` from an algorithm config."""

    kind = algorithm.type
    if kind is AlgorithmType.UNIFORM_SOUP:
        return UniformSoup()
    if kind is AlgorithmType.WEIGHTED_SOUP:
        if any(weight is None for weight in weights):
            raise ConfigurationError("weighted_soup requires a weight for every model")
        return WeightedSoup(
            [float(weight) for weight in weights],  # type: ignore[arg-type]
            normalize=algorithm.normalize_weights,
            allow_negative=algorithm.allow_negative,
        )
    if kind is AlgorithmType.SLERP:
        assert algorithm.t is not None
        return Slerp(
            algorithm.t,
            dot_threshold=algorithm.dot_threshold,
            allow_extrapolation=algorithm.allow_extrapolation,
            eps=algorithm.eps,
        )
    if kind is AlgorithmType.LINEAR:
        assert algorithm.t is not None
        return LinearInterpolation(algorithm.t, allow_extrapolation=algorithm.allow_extrapolation)
    raise ConfigurationError(f"algorithm '{kind.value}' cannot be used as a per-tensor strategy")


def _build_resolver(config: MergeConfig) -> LayerRuleResolver:
    compiled: list[CompiledRule] = []
    for rule in config.rules:
        layer_range = None
        if rule.match.layer_range is not None:
            layer_range = LayerRange(rule.match.layer_range.start, rule.match.layer_range.end)
        predicate = compile_predicate(
            exact=rule.match.exact,
            glob=rule.match.glob,
            regex=rule.match.regex,
            suffix=rule.match.suffix,
            layer_range=layer_range,
            exclude=rule.match.exclude,
        )
        compiled.append(
            CompiledRule(
                name=rule.name,
                predicate=predicate,
                payload=rule.algorithm,
                priority=rule.priority,
            )
        )
    return LayerRuleResolver(compiled, config.algorithm, default_name="default")


def _is_pickle_backed(checkpoint: Checkpoint) -> bool:
    if checkpoint.format == "pytorch":
        return True
    if isinstance(checkpoint, HuggingFaceCheckpoint):
        return checkpoint.backend_format == "pytorch"
    return False


@dataclass
class PreparedMerge:
    """Everything needed to execute a (non-greedy) merge; holds open checkpoints."""

    config: MergeConfig
    checkpoints: list[Checkpoint]
    model_names: list[str]
    tensor_compat: TensorCompatibility
    resolver: LayerRuleResolver
    precision: PrecisionPolicy
    non_float_policy: NonFloatTensorPolicy
    device: torch.device
    output_order: list[str]
    passthrough: dict[str, int]
    shard_plan: ShardPlan
    strategies: dict[str, MergeStrategy]
    key_rule: dict[str, str]
    key_is_float: dict[str, bool]
    key_output_dtype: dict[str, torch.dtype]
    compat_report: CompatibilityReport
    plan: MergePlan
    warnings: list[str] = field(default_factory=list)

    def close(self) -> None:
        for checkpoint in self.checkpoints:
            checkpoint.close()


def prepare_merge(config: MergeConfig, *, hash_inputs: bool = False) -> PreparedMerge:
    """Open checkpoints, validate, and build the merge plan (writes nothing).

    Raises:
        CompatibilityError: if the models are not compatible under the config mode.
        ConfigurationError: for algorithm/weight problems discovered here.
    """

    paths = config.resolved_model_paths()
    checkpoints = [
        open_checkpoint(path, allow_unsafe=config.allow_unsafe_pytorch) for path in paths
    ]
    try:
        return _prepare_from_open(config, checkpoints, paths, hash_inputs=hash_inputs)
    except Exception:
        for checkpoint in checkpoints:
            checkpoint.close()
        raise


def _prepare_from_open(
    config: MergeConfig,
    checkpoints: list[Checkpoint],
    paths: list[Path],
    *,
    hash_inputs: bool,
) -> PreparedMerge:
    warnings: list[str] = []
    compat_report, tensor_compat = validate_compatibility(checkpoints, config.compatibility)
    compat_report.raise_if_incompatible(config.compatibility.mode)
    warnings.extend(
        finding.message
        for finding in compat_report.findings
        if finding.severity.value in {"warning", "error"}
        and not finding.is_blocking(config.compatibility.mode)
    )

    resolver = _build_resolver(config)
    precision = PrecisionPolicy(
        compute_dtype=torch.float32
        if config.precision.compute_dtype == "float32"
        else _dtype_from_name(config.precision.compute_dtype),
        output_dtype_policy=config.precision.output_dtype_policy,
        validate_finite=config.precision.validate_finite,
    )
    non_float_policy = NonFloatTensorPolicy(config.non_float_tensors.policy)
    device = resolve_device(config.device)
    weights = [ref.weight for ref in config.models]

    strategies: dict[str, MergeStrategy] = {}
    key_rule: dict[str, str] = {}
    key_is_float: dict[str, bool] = {}
    key_output_dtype: dict[str, torch.dtype] = {}
    per_rule_counts: dict[str, int] = {}
    non_float_keys: list[str] = []
    tensor_entries: list[TensorPlanEntry] = []
    output_entries: list[tuple[str, int]] = []

    for key in tensor_compat.merge_keys:
        input_dtypes = [ckpt.tensor_info(key).dtype for ckpt in checkpoints]
        reference = checkpoints[0].tensor_info(key)
        is_float = reference.dtype.is_floating_point
        key_is_float[key] = is_float
        resolved = resolver.resolve(key)
        rule_name = resolved.rule_name
        key_rule[key] = rule_name

        if is_float:
            if rule_name not in strategies:
                algorithm = resolved.payload
                assert isinstance(algorithm, AlgorithmConfig)
                strategies[rule_name] = build_strategy(algorithm, weights)
            output_dtype = precision.output_dtype_for(input_dtypes)
            compute_dtype = precision.compute_dtype_for(input_dtypes)
            algo_name = strategies[rule_name].name
            per_rule_counts[rule_name] = per_rule_counts.get(rule_name, 0) + 1
        else:
            output_dtype = reference.dtype
            compute_dtype = reference.dtype
            algo_name = f"non_float:{non_float_policy.policy.value}"
            non_float_keys.append(key)

        key_output_dtype[key] = output_dtype
        output_bytes = reference.num_elements * element_size(output_dtype)
        output_entries.append((key, output_bytes))
        tensor_entries.append(
            TensorPlanEntry(
                key=key,
                rule_name=rule_name if is_float else "non_float",
                algorithm=algo_name,
                compute_dtype=dtype_name(compute_dtype),
                output_dtype=dtype_name(output_dtype),
                shape=reference.shape,
                is_non_float=not is_float,
                output_bytes=output_bytes,
            )
        )

    for key, owner in sorted(tensor_compat.passthrough.items()):
        info = checkpoints[owner].tensor_info(key)
        key_is_float[key] = False
        key_rule[key] = "passthrough"
        key_output_dtype[key] = info.dtype
        output_bytes = info.num_elements * element_size(info.dtype)
        output_entries.append((key, output_bytes))
        per_rule_counts["passthrough"] = per_rule_counts.get("passthrough", 0) + 1
        tensor_entries.append(
            TensorPlanEntry(
                key=key,
                rule_name="passthrough",
                algorithm=f"copy_from_model_{owner}",
                compute_dtype=dtype_name(info.dtype),
                output_dtype=dtype_name(info.dtype),
                shape=info.shape,
                is_non_float=True,
                output_bytes=output_bytes,
            )
        )

    output_entries.sort(key=lambda item: item[0])
    output_order = [key for key, _ in output_entries]
    total_output_bytes = sum(size for _, size in output_entries)
    if config.output.format is OutputFormat.SAFETENSORS:
        base_name, extension = "model", ".safetensors"
        max_shard = config.output.max_shard_size_bytes
    else:
        # A single pickle archive is written as one file; force one shard.
        base_name, extension = "pytorch_model", ".bin"
        max_shard = max(total_output_bytes, 1)
    shard_plan = plan_shards(
        output_entries,
        max_shard_bytes=max_shard,
        base_name=base_name,
        extension=extension,
    )

    model_summaries = _model_summaries(config, checkpoints, paths, hash_inputs=hash_inputs)
    ancillary_plan = _ancillary_plan(config, checkpoints)
    requires_unsafe = any(_is_pickle_backed(ckpt) for ckpt in checkpoints)

    plan = MergePlan(
        run_id=new_run_id(),
        tool_version=_TOOL_VERSION,
        algorithm=config.algorithm.type.value,
        algorithm_params=_algorithm_params(config),
        models=tuple(model_summaries),
        output_path=str(config.resolved_output_path()),
        output_format=config.output.format.value,
        output_dtype_policy=config.precision.output_dtype,
        estimated_output_bytes=shard_plan.total_bytes,
        shard_count=len(shard_plan.shards),
        shard_files=tuple(shard.filename for shard in shard_plan.shards),
        tensor_count=len(output_order),
        per_rule_counts=per_rule_counts,
        non_float_keys=tuple(non_float_keys),
        ancillary=ancillary_plan,
        requires_unsafe_loading=requires_unsafe,
        warnings=tuple(warnings),
        compatibility_summary=compat_report.summary(config.compatibility.mode),
        tensor_entries=tuple(tensor_entries),
    )

    return PreparedMerge(
        config=config,
        checkpoints=checkpoints,
        model_names=list(config.model_names),
        tensor_compat=tensor_compat,
        resolver=resolver,
        precision=precision,
        non_float_policy=non_float_policy,
        device=device,
        output_order=output_order,
        passthrough=dict(tensor_compat.passthrough),
        shard_plan=shard_plan,
        strategies=strategies,
        key_rule=key_rule,
        key_is_float=key_is_float,
        key_output_dtype=key_output_dtype,
        compat_report=compat_report,
        plan=plan,
        warnings=warnings,
    )


def _dtype_from_name(name: str) -> torch.dtype:
    from ..types import resolve_dtype

    return resolve_dtype(name)


def _algorithm_params(config: MergeConfig) -> dict[str, object]:
    algorithm = config.algorithm
    params: dict[str, object] = {"type": algorithm.type.value}
    if algorithm.type in (AlgorithmType.SLERP, AlgorithmType.LINEAR):
        params["t"] = algorithm.t
    if algorithm.type is AlgorithmType.SLERP:
        params["dot_threshold"] = algorithm.dot_threshold
        params["allow_extrapolation"] = algorithm.allow_extrapolation
    if algorithm.type is AlgorithmType.WEIGHTED_SOUP:
        params["normalize_weights"] = algorithm.normalize_weights
        params["allow_negative"] = algorithm.allow_negative
        params["weights"] = [ref.weight for ref in config.models]
    return params


def _model_summaries(
    config: MergeConfig,
    checkpoints: list[Checkpoint],
    paths: list[Path],
    *,
    hash_inputs: bool,
) -> list[ModelSummary]:
    summaries: list[ModelSummary] = []
    for ref, checkpoint, path in zip(config.models, checkpoints, paths, strict=True):
        content_hash = None
        if hash_inputs and path.is_file():
            content_hash = hash_file(path)
        summaries.append(
            ModelSummary(
                name=ref.name or path.name,
                path=str(path),
                format=checkpoint.format,
                tensor_count=len(checkpoint.keys()),
                weight=ref.weight,
                content_hash=content_hash,
            )
        )
    return summaries


def _ancillary_plan(config: MergeConfig, checkpoints: list[Checkpoint]) -> AncillaryPlan:
    strategy = config.ancillary.strategy
    source_name = config.ancillary.base_model
    source_index = 0
    if source_name is not None:
        source_index = config.model_names.index(source_name)
    source_ckpt = checkpoints[source_index]
    files: tuple[str, ...] = ()
    if isinstance(source_ckpt, HuggingFaceCheckpoint):
        files = tuple(path.name for path in source_ckpt.ancillary_files)
    return AncillaryPlan(
        strategy=strategy,
        source_model=source_name or (config.model_names[0] if config.model_names else None),
        files=files,
    )
