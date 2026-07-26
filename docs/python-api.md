# Python API

The supported public surface is intentionally small and imported from the
top-level package:

```python
from model_merger import (
    MergeConfig, merge_models, plan_merge, inspect_model, verify_output,
    MergeReport, MergePlan,
    ModelMergerError,  # ... and the rest of the exception hierarchy
)
```

Everything else is an implementation detail and may change between minor releases.

## `merge_models(config, *, progress=False) -> MergeReport`

Executes a merge and returns its report. Writes `merge_report.json` into the
output directory when `config.output.write_report` is set. `progress` shows a
terminal bar (off by default for library use).

```python
config = MergeConfig.from_file("configs/uniform_soup.example.yaml")
report = merge_models(config)
assert report.verification.passed
print(report.output_path, report.output_hash)
```

## `plan_merge(config) -> MergePlan`

Computes the plan without writing anything (a dry run). Opens checkpoints,
validates, resolves per-tensor algorithms/dtypes, and computes the shard layout,
then closes the checkpoints.

```python
plan = plan_merge(config)
print(plan.tensor_count, plan.per_rule_counts, plan.shard_files)
```

## `inspect_model(path, *, allow_unsafe=False) -> dict`

Returns a JSON-friendly summary: tensor/parameter counts, byte size, dtype
histogram, sample keys, container metadata, and (for HF directories) architecture
info.

## `verify_output(path, *, check_finite=True, allow_unsafe=False) -> VerificationResult`

Structurally verifies a written checkpoint: opens it, loads every tensor, checks
floats are finite, and (for directories) parses `config.json`.

## Building a config programmatically

```python
from model_merger.config.models import (
    AlgorithmConfig, ModelRef, OutputConfig, MergeConfig,
)

config = MergeConfig(
    algorithm=AlgorithmConfig(type="slerp", t=0.35, dot_threshold=0.9995),
    models=[ModelRef(path="./model-a"), ModelRef(path="./model-b")],
    output=OutputConfig(path="./merged", format="safetensors", overwrite=False),
)
report = merge_models(config)
```

`MergeConfig.from_dict(data, base_dir=...)` builds from a plain dict and attaches a
base directory for relative-path resolution.

## Error handling

All expected failures derive from `ModelMergerError`, which carries a `.message`
and an `.exit_code`. Catch the base class (or a specific subclass) rather than
broad `Exception`:

```python
from model_merger import merge_models, CompatibilityError, ModelMergerError

try:
    report = merge_models(config)
except CompatibilityError as error:
    ...  # models are not compatible
except ModelMergerError as error:
    print(error.message)
    raise SystemExit(error.exit_code)
```

See the [exception hierarchy](cli-reference.md#exit-codes).

## Logging

The library never configures the root logger on import. To see library logs,
configure the namespaced logger explicitly:

```python
from model_merger.logging import configure_logging
configure_logging(level="INFO")        # or json_mode=True, quiet=True
```

Embedding applications keep full control of their own logging.

## The report object

`MergeReport` (and `MergePlan`) expose `.to_dict()` for JSON-safe serialization.
Notable fields: `run_id`, `timestamp`, `algorithm`, `algorithm_params`, `models`,
`output_path`, `output_hashes` (per-file), `output_hash` (content-based
reproducibility fingerprint), `tensor_count`, `per_rule_counts`,
`duration_seconds`, `peak_memory_bytes`, `environment`, `verification`, and
`greedy_history` for greedy runs.
