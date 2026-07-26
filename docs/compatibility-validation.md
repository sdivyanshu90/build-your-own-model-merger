# Compatibility and validation

Before writing anything, Model Merger checks that the models can actually be
merged and produces a **compatibility report**. This is *validation* (on inputs),
distinct from *verification* (on outputs, after writing).

## What is checked

- **Tensors** (`compatibility/tensors.py`) — the key sets across models, and for
  every shared key, its shape and dtype. Uses lazy `tensor_info` only (no data
  loaded).
- **Architecture** (`compatibility/architecture.py`) — for Hugging Face models,
  `model_type`, architecture class names, and `vocab_size` from `config.json`;
  also rejects quantized models.
- **Tokenizer** (`compatibility/tokenizer.py`) — compares tokenizer-defining files
  by content hash across Hugging Face directories.

Checkpoints without a `config.json` (plain safetensors/pytorch) skip the
architecture/tokenizer checks with an informational note; tensor checks still
apply.

## Severity and modes

Each finding has a severity, and whether it **blocks** the merge depends on the
compatibility `mode`:

| Severity | strict mode | permissive mode |
| --- | --- | --- |
| `fatal` | blocks | blocks |
| `error` | blocks | warning (proceeds) |
| `warning` | proceeds | proceeds |
| `info` | proceeds | proceeds |

Examples:

- Shape mismatch on a shared tensor → **fatal** (always blocks).
- Different `model_type` → **fatal**.
- dtype mismatch, `architectures`/`vocab_size` mismatch, tokenizer file mismatch →
  **error** (blocks in strict, warns in permissive).
- Key-set mismatch → **error** in strict; **warning** with passthrough when
  `allow_missing_keys`/`allow_extra_keys` is set.

The merge is aborted **before any output is written** if a blocking finding
exists (`CompatibilityError`, exit code 5).

## Key-set handling

- **Merge keys** — keys present in *all* models are merged.
- **Passthrough keys** — keys present in only some models are, when allowed,
  copied verbatim from the first model that has them (and reported).

## Config toggles

```yaml
compatibility:
  mode: strict               # or permissive
  require_matching_tokenizer: true
  require_matching_config: true
  require_matching_vocab_size: true
  allow_extra_keys: false
  allow_missing_keys: false
```

## Inspecting findings

The full report (all findings, counts, max severity) is included in the plan:

```bash
model-merger plan config.yaml --json | jq .compatibility
```

## Verification vs validation

After a successful write, [verification](reproducibility.md#verification) re-opens
the output and confirms every tensor loads, floats are finite, dtypes/keys match
the plan, the shard index resolves, and any `config.json` parses. A merge is
never reported successful until verification passes.
