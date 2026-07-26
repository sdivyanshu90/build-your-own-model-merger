"""Validated configuration models (Pydantic v2).

``MergeConfig`` is the single entry point users construct -- from a YAML/JSON file
(:meth:`MergeConfig.from_file`) or programmatically (:meth:`MergeConfig.from_dict`
or direct construction).  Unknown fields are rejected (``extra="forbid"``) so
typos surface as errors instead of being silently ignored.

Type validation is Pydantic's job; *semantic* validation (arity, cross-field
requirements, name uniqueness) lives in ``@model_validator`` methods here and in
:mod:`model_merger.config.validation`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..exceptions import ConfigurationError
from ..types import (
    AlgorithmType,
    CompatibilityMode,
    MetricDirection,
    NonFloatPolicy,
    OutputDtypePolicy,
    OutputFormat,
)
from .loaders import load_config_file
from .validation import (
    parse_size,
    resolve_path,
    validate_compute_dtype_name,
    validate_output_dtype_name,
)

__all__ = [
    "MergeConfig",
    "AlgorithmConfig",
    "ModelRef",
    "OutputConfig",
    "PrecisionConfig",
    "CompatibilityConfig",
    "NonFloatConfig",
    "MatchConfig",
    "LayerRangeConfig",
    "RuleConfig",
    "EvaluatorConfig",
    "GreedyConfig",
    "AncillaryConfig",
]


class StrictModel(BaseModel):
    """Base model that forbids unknown fields and strips surrounding whitespace."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AlgorithmConfig(StrictModel):
    """Algorithm selection and its parameters.

    Not every field applies to every algorithm; ``@model_validator`` enforces the
    per-type requirements (e.g. ``t`` is mandatory for SLERP/linear).
    """

    type: AlgorithmType
    t: float | None = None
    dot_threshold: float = 0.9995
    allow_extrapolation: bool = False
    normalize_weights: bool = True
    allow_negative: bool = False
    eps: float = 1e-8

    @model_validator(mode="after")
    def _check_parameters(self) -> AlgorithmConfig:
        if self.type in (AlgorithmType.SLERP, AlgorithmType.LINEAR) and self.t is None:
            raise ConfigurationError(f"'t' is required for algorithm '{self.type.value}'")
        if self.type is AlgorithmType.SLERP and not (0.0 < self.dot_threshold <= 1.0):
            raise ConfigurationError(f"dot_threshold must be in (0, 1], got {self.dot_threshold}")
        if self.eps <= 0:
            raise ConfigurationError(f"eps must be positive, got {self.eps}")
        return self


class ModelRef(StrictModel):
    """A source model: filesystem path, optional name, optional soup weight."""

    path: str
    name: str | None = None
    weight: float | None = None


class OutputConfig(StrictModel):
    """Where and how the merged model is written."""

    path: str
    format: OutputFormat = OutputFormat.SAFETENSORS
    max_shard_size: str | int = "5GB"
    overwrite: bool = False
    atomic: bool = True
    write_report: bool = True
    report_markdown: bool = False

    @property
    def max_shard_size_bytes(self) -> int:
        return parse_size(self.max_shard_size)


class PrecisionConfig(StrictModel):
    """Numerical precision policy."""

    compute_dtype: str = "float32"
    output_dtype: str = "preserve"
    validate_finite: bool = True

    @field_validator("compute_dtype")
    @classmethod
    def _check_compute(cls, value: str) -> str:
        return validate_compute_dtype_name(value)

    @field_validator("output_dtype")
    @classmethod
    def _check_output(cls, value: str) -> str:
        return validate_output_dtype_name(value)

    @property
    def output_dtype_policy(self) -> OutputDtypePolicy:
        return OutputDtypePolicy(self.output_dtype)


class CompatibilityConfig(StrictModel):
    """Compatibility validation strictness and toggles."""

    mode: CompatibilityMode = CompatibilityMode.STRICT
    require_matching_tokenizer: bool = True
    require_matching_config: bool = True
    require_matching_vocab_size: bool = True
    allow_extra_keys: bool = False
    allow_missing_keys: bool = False


class NonFloatConfig(StrictModel):
    """Policy for non-floating tensors."""

    policy: NonFloatPolicy = NonFloatPolicy.REQUIRE_EQUAL


class LayerRangeConfig(StrictModel):
    """Inclusive transformer layer-index range."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)


class MatchConfig(StrictModel):
    """Tensor-key match conditions for a rule (see policies.matching)."""

    exact: str | None = None
    glob: str | None = None
    regex: str | None = None
    suffix: str | None = None
    layer_range: LayerRangeConfig | None = None
    exclude: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_condition(self) -> MatchConfig:
        if not any(
            value is not None
            for value in (self.exact, self.glob, self.regex, self.suffix, self.layer_range)
        ):
            raise ConfigurationError(
                "match requires at least one of: exact, glob, regex, suffix, layer_range"
            )
        return self


class RuleConfig(StrictModel):
    """A named layer rule: match condition + algorithm override + priority."""

    name: str
    match: MatchConfig
    algorithm: AlgorithmConfig
    priority: int = 0


class EvaluatorConfig(StrictModel):
    """How a merged candidate is scored during greedy soup.

    ``command`` runs an external program with ``{model_path}`` substituted (no
    shell), expecting a float or ``{"score": <float>}`` on stdout.  ``callable``
    imports ``module:function`` and calls it with the checkpoint path.
    """

    type: Literal["command", "callable"]
    command: list[str] | None = None
    callable: str | None = None
    metric_key: str = "score"
    timeout: float | None = None
    placeholder: str = "{model_path}"

    @model_validator(mode="after")
    def _check(self) -> EvaluatorConfig:
        if self.type == "command":
            if not self.command:
                raise ConfigurationError("command evaluator requires a non-empty 'command'")
            if not any(self.placeholder in arg for arg in self.command):
                raise ConfigurationError(
                    f"command evaluator must reference {self.placeholder!r} in an argument"
                )
        else:
            if not self.callable or ":" not in self.callable:
                raise ConfigurationError(
                    "callable evaluator requires 'callable' as 'module.path:function'"
                )
        if self.timeout is not None and self.timeout <= 0:
            raise ConfigurationError("evaluator timeout must be positive")
        return self


class GreedyConfig(StrictModel):
    """Greedy soup selection settings."""

    direction: MetricDirection = MetricDirection.MAXIMIZE
    evaluator: EvaluatorConfig
    cache: bool = True
    keep_temp: bool = False
    temp_dir: str | None = None


class AncillaryConfig(StrictModel):
    """Strategy for non-tensor files (config.json, tokenizer, generation_config).

    * ``base`` / ``first`` -- copy from the named base model / the first model.
    * ``require_equal`` -- all sources must be byte-identical.
    * ``fail_on_difference`` -- like require_equal but a hard error.
    * ``warn`` -- copy from base/first but only warn on differences.
    """

    strategy: Literal["base", "first", "require_equal", "fail_on_difference", "warn"] = "base"
    base_model: str | None = None


class MergeConfig(StrictModel):
    """Root configuration for a merge run."""

    algorithm: AlgorithmConfig
    models: list[ModelRef] = Field(min_length=1)
    output: OutputConfig
    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)
    compatibility: CompatibilityConfig = Field(default_factory=CompatibilityConfig)
    non_float_tensors: NonFloatConfig = Field(default_factory=NonFloatConfig)
    rules: list[RuleConfig] = Field(default_factory=list)
    greedy: GreedyConfig | None = None
    ancillary: AncillaryConfig = Field(default_factory=AncillaryConfig)
    device: str = "cpu"
    allow_unsafe_pytorch: bool = False
    seed: int = 0

    # Directory used to resolve relative paths; set by from_file. Excluded from
    # serialization so a round-tripped config stays portable.
    base_dir: Path | None = Field(default=None, exclude=True, repr=False)

    @field_validator("device")
    @classmethod
    def _check_device(cls, value: str) -> str:
        allowed = value == "auto" or value == "cpu" or value.startswith("cuda")
        if not allowed:
            raise ConfigurationError(
                f"invalid device {value!r}; use 'cpu', 'cuda', 'cuda:N', or 'auto'"
            )
        return value

    @model_validator(mode="after")
    def _semantic_checks(self) -> MergeConfig:
        self._assign_names()
        algorithm = self.algorithm.type
        if algorithm in (AlgorithmType.SLERP, AlgorithmType.LINEAR) and len(self.models) != 2:
            raise ConfigurationError(
                f"algorithm '{algorithm.value}' requires exactly 2 models, got {len(self.models)}"
            )
        if algorithm is AlgorithmType.GREEDY_SOUP and self.greedy is None:
            raise ConfigurationError("greedy_soup algorithm requires a 'greedy' section")
        if algorithm is AlgorithmType.WEIGHTED_SOUP:
            missing = [ref.name for ref in self.models if ref.weight is None]
            if missing:
                raise ConfigurationError(
                    f"weighted_soup requires a 'weight' for every model; missing: {missing}"
                )
        self._check_referenced_names()
        return self

    def _assign_names(self) -> None:
        used: set[str] = set()
        for ref in self.models:
            if ref.name is None:
                ref.name = Path(ref.path).name or ref.path
            base = ref.name
            suffix = 1
            while ref.name in used:
                suffix += 1
                ref.name = f"{base}-{suffix}"
            used.add(ref.name)

    def _check_referenced_names(self) -> None:
        names = {ref.name for ref in self.models}
        if self.ancillary.base_model is not None and self.ancillary.base_model not in names:
            raise ConfigurationError(
                f"ancillary.base_model {self.ancillary.base_model!r} is not a configured model"
            )

    @property
    def model_names(self) -> list[str]:
        return [ref.name for ref in self.models if ref.name is not None]

    def resolved_model_paths(self) -> list[Path]:
        """Return each model path resolved against the config base directory."""

        return [resolve_path(self.base_dir, ref.path) for ref in self.models]

    def resolved_output_path(self) -> Path:
        """Return the output path resolved against the config base directory."""

        return resolve_path(self.base_dir, self.output.path)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_dir: Path | None = None) -> MergeConfig:
        """Build a config from a plain dict, attaching a base directory."""

        try:
            config = cls.model_validate(data)
        except ConfigurationError:
            raise
        except Exception as error:  # Pydantic ValidationError -> ConfigurationError
            raise ConfigurationError(str(error)) from error
        config.base_dir = base_dir
        return config

    @classmethod
    def from_file(cls, path: str | Path) -> MergeConfig:
        """Load, validate, and return a config from a YAML/JSON file.

        Relative model/output paths become relative to the file's directory.
        """

        config_path = Path(path).resolve()
        data = load_config_file(config_path)
        return cls.from_dict(data, base_dir=config_path.parent)

    def redacted_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict with obvious secrets masked, for logging.

        Environment-derived values are already expanded by the loader; this masks
        anything under keys that look sensitive so logs never echo tokens.
        """

        sensitive = ("token", "secret", "password", "api_key", "authorization")
        raw = self.model_dump(mode="json")

        def scrub(value: Any, key_hint: str = "") -> Any:
            if isinstance(value, dict):
                return {key: scrub(item, key) for key, item in value.items()}
            if isinstance(value, list):
                return [scrub(item, key_hint) for item in value]
            if isinstance(value, str) and any(token in key_hint.lower() for token in sensitive):
                return "***"
            return value

        scrubbed: dict[str, Any] = scrub(raw)
        return scrubbed
