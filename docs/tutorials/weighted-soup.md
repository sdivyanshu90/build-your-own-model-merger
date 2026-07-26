# Tutorial: a weighted soup

**Goal.** Average three models with unequal contributions.

**Prerequisites.** [First uniform soup](first-uniform-soup.md) completed; models in
`./models`.

## Config

`configs/weighted_soup.example.yaml`:

```yaml
algorithm:
  type: weighted_soup
  normalize_weights: true
models:
  - {path: ./models/model-a, weight: 0.50}
  - {path: ./models/model-b, weight: 0.30}
  - {path: ./models/model-c, weight: 0.20}
output:
  path: ./merged-weighted
  overwrite: false
```

The weights already sum to 1, but with `normalize_weights: true` any positive
weights work — `[5, 3, 2]` behaves identically.

## Run

```bash
model-merger merge configs/weighted_soup.example.yaml --overwrite
model-merger verify ./merged-weighted
```

## Verifying the math

Each merged tensor equals `0.5·A + 0.3·B + 0.2·C`. You can check one:

```python
from safetensors import safe_open
import torch

def load(p, k):
    with safe_open(f"{p}/model.safetensors", framework="pt") as f:
        return f.get_tensor(k)

key = "model.norm.weight"
a, b, c = (load(f"./models/model-{x}", key) for x in "abc")
expected = 0.5 * a + 0.3 * b + 0.2 * c
merged = load("./merged-weighted", key)
print(torch.allclose(merged, expected, atol=1e-6))  # True
```

## Strict weights

To require the weights to already sum to 1 (no auto-scaling), set
`normalize_weights: false`. A set that does not sum to 1 is then rejected with an
actionable error.

## Negative weights (extrapolation)

Negative weights are rejected unless `allow_negative: true`. They turn the merge
into extrapolation and can produce out-of-hull, even non-finite, results — use
sparingly and evaluate carefully.

## Common errors

- **`weighted_soup requires a weight for every model`** — add a `weight` to each
  model entry.
- **`weights must sum to 1 in strict mode`** — enable `normalize_weights` or fix
  the weights.
