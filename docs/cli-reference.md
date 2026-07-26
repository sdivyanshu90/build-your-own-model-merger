# CLI reference

```
model-merger [GLOBAL OPTIONS] COMMAND [ARGS]
```

Also available as `python -m model_merger`.

## Global options

| Option | Effect |
| --- | --- |
| `-v`, `--verbose` | DEBUG-level logging |
| `-q`, `--quiet` | only log errors |
| `--debug` | show tracebacks for errors (otherwise a one-line message) |

Logs go to **stderr**; machine-readable output goes to **stdout**, so `--json`
output can be piped cleanly.

## Commands

### `inspect`
```
model-merger inspect MODEL_PATH [--json] [--allow-unsafe]
```
Summarize a checkpoint (tensors, parameters, size, dtypes, architecture).

### `validate`
```
model-merger validate CONFIG_PATH
```
Validate a config's syntax and semantics. Exit 0 if valid, 2 otherwise.

### `plan`
```
model-merger plan CONFIG_PATH [--json] [--full]
```
Compute and print the merge plan **without writing output**. `--full` includes
per-tensor entries.

### `merge`
```
model-merger merge CONFIG_PATH [--dry-run] [--overwrite] [--device D]
                               [--compute-dtype DT] [--json] [--no-progress]
```
Execute a merge. `--dry-run` plans only. `--overwrite`, `--device`, and
`--compute-dtype` override the config. `--json` prints the report as JSON.

### `verify`
```
model-merger verify OUTPUT_PATH [--json] [--allow-unsafe]
```
Verify a written checkpoint. Exit 0 if it passes, 12 otherwise.

### `schema`
```
model-merger schema
```
Print the JSON schema for the configuration.

### `version`
```
model-merger version
```

## Exit codes

Expected errors print a one-line message to stderr and exit with the code below
(unless `--debug`, which shows a traceback). These map one-to-one to the exception
hierarchy.

| Code | Exception | Meaning |
| --- | --- | --- |
| 0 | — | success |
| 1 | `ModelMergerError` | unclassified error |
| 2 | `ConfigurationError` | invalid configuration |
| 3 | `CheckpointError` | unreadable/malformed/unsupported checkpoint |
| 4 | `UnsafeCheckpointError` | pickle load blocked (needs `--allow-unsafe`) |
| 5 | `CompatibilityError` | models are not compatible |
| 6 | `TensorMismatchError` | tensor shape/dtype/presence disagreement |
| 7 | `NumericalError` | non-finite output or numerical invariant violation |
| 8 | `EvaluationError` | greedy evaluator failed |
| 9 | `OutputExistsError` | output exists and `--overwrite` not set |
| 10 | `InsufficientDiskSpaceError` | not enough free space |
| 11 | `MergeExecutionError` | failure during execution |
| 12 | `VerificationError` | output failed verification |

## Examples

```bash
# Inspect, then plan, then merge, then verify
model-merger inspect ./model-a
model-merger plan configs/slerp.example.yaml
model-merger merge configs/slerp.example.yaml --overwrite
model-merger verify ./merged-slerp

# Machine-readable
model-merger merge configs/uniform_soup.example.yaml --json --no-progress | jq .output

# Override device and dtype without editing the config
model-merger merge configs/uniform_soup.example.yaml --device cuda --compute-dtype float32
```
