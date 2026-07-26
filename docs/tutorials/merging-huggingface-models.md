# Tutorial: merging Hugging Face models

**Goal.** Merge two Hugging Face model directories and keep a coherent set of
config/tokenizer files in the output.

**Prerequisites.** Two HF model directories that share an architecture and
tokenizer — typically two fine-tunes of the same base model. The tiny generated
models are HF-style directories and work here; substitute real ones the same way.

**Input assumptions.** Each directory contains weights (`model.safetensors` or a
sharded index) plus `config.json` and tokenizer files.

## Config

```yaml
algorithm:
  type: uniform_soup
models:
  - {path: ./finetune-a, name: a}
  - {path: ./finetune-b, name: b}
output:
  path: ./merged-hf
  format: safetensors
  overwrite: false
compatibility:
  mode: strict
  require_matching_tokenizer: true
  require_matching_config: true
ancillary:
  strategy: base
  base_model: a          # copy config/tokenizer from model "a"
```

## Ancillary files

The merged **weights** come from all models, but the **config and tokenizer** must
come from a single coherent source. The `ancillary` strategy controls this:

- `base` + `base_model` — copy from the named model (here, `a`).
- `first` — copy from the first model.
- `require_equal` / `fail_on_difference` — all sources must be byte-identical.
- `warn` — copy from base/first but warn on differences.

Compatibility validation compares tokenizer files by content hash; in `strict`
mode a tokenizer mismatch blocks the merge (a mismatched tokenizer would make the
merged embeddings incoherent).

## Run and verify

```bash
model-merger merge config.yaml --overwrite
model-merger verify ./merged-hf
ls ./merged-hf        # model.safetensors, config.json, tokenizer files, merge_report.json
```

## Loading the result

Because config and tokenizer are preserved, the output directory loads like any HF
model with your usual loader. Verification already confirmed `config.json` parses
and every tensor is finite and loadable.

## Common errors

- **`tokenizer file ... differs`** (error in strict) — the models use different
  tokenizers; align them, or use `mode: permissive` if you accept the risk.
- **`model_type mismatch`** (fatal) — different architectures cannot be merged.
- **adapter/LoRA directory rejected** — merge the base models, not adapters.
