# Tutorial: your first uniform soup

**Goal.** Average three tiny models into one and verify the result.

**Prerequisites.** The package installed (`pip install -e ".[dev]"`), a terminal.

**Input assumptions.** Three compatible models. We generate tiny ones locally so
there is nothing to download.

## 1. Create tiny models

```bash
python scripts/generate_tiny_test_models.py ./models
# writes ./models/model-a, model-b, model-c
```

Each is a small safetensors checkpoint with a minimal `config.json` and tokenizer
stub.

## 2. The config

`configs/uniform_soup.example.yaml` (already in the repo):

```yaml
algorithm:
  type: uniform_soup
models:
  - {path: ./models/model-a}
  - {path: ./models/model-b}
  - {path: ./models/model-c}
output:
  path: ./merged-uniform
  overwrite: false
```

## 3. Plan first (optional, writes nothing)

```bash
model-merger plan configs/uniform_soup.example.yaml
```

Expected: a plan showing the algorithm, models, estimated size, and per-rule
tensor counts (`default: N`).

## 4. Merge

```bash
model-merger merge configs/uniform_soup.example.yaml --overwrite
```

Expected output:

```
merged 11 tensors -> ./merged-uniform
run id: ...  duration: ...s
verification: PASSED
```

## 5. Verify

```bash
model-merger verify ./merged-uniform
```

Expected: `verification: PASSED` with each check `ok`.

## What happened internally

The planner opened the three models lazily, confirmed their tensors match in shape
and dtype, and resolved every floating tensor to the uniform-soup strategy (the
integer buffer used the `require_equal` non-float policy). The executor streamed
each tensor — loading it from all three models, averaging in float32, writing it —
then copied the ancillary files, wrote `merge_report.json`, and verified the
output.

## Common errors

- **`output path already exists` (exit 9)** — add `--overwrite` or change
  `output.path`.
- **`models are not compatible` (exit 5)** — your models differ in shape or
  architecture; run `model-merger plan ... --json | jq .compatibility`.

## Next

- [Weighted soup](weighted-soup.md) to bias toward some models.
- [First SLERP](first-slerp.md) to interpolate two models.
