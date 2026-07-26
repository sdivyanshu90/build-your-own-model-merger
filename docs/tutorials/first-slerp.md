# Tutorial: your first SLERP

**Goal.** Spherically interpolate two models at the midpoint.

**Prerequisites.** Models in `./models` (from
[First uniform soup](first-uniform-soup.md)).

**Input assumptions.** SLERP takes **exactly two** compatible models.

## Config

`configs/slerp.example.yaml`:

```yaml
algorithm:
  type: slerp
  t: 0.5                 # midpoint
  dot_threshold: 0.9995  # near-parallel -> LERP fallback
models:
  - {path: ./models/model-a}
  - {path: ./models/model-b}
output:
  path: ./merged-slerp
  overwrite: false
```

## Run

```bash
model-merger merge configs/slerp.example.yaml --overwrite
model-merger verify ./merged-slerp
```

Change `t` toward 0 to stay near `model-a`, toward 1 to lean to `model-b`.

## What happened internally

For each tensor, the merger flattened both models' versions to vectors, measured
the angle between their directions, and moved halfway along the arc — preserving
magnitude where LERP would shrink it. For tensors whose directions were nearly
identical (angle ≈ 0), it fell back to linear interpolation to avoid dividing by
`sin(0)`. All arithmetic ran in float32.

## Verifying an endpoint

At `t = 0` the output equals `model-a` exactly:

```bash
sed 's/t: 0.5/t: 0.0/' configs/slerp.example.yaml > /tmp/slerp0.yaml
model-merger merge /tmp/slerp0.yaml --overwrite
```

```python
from model_merger.execution.verification import content_hash
print(content_hash("./merged-slerp") == content_hash("./models/model-a"))  # True
```

## Layer-wise SLERP

To use different coefficients per region (e.g. gentle on embeddings, stronger in
upper layers), see `configs/layerwise_slerp.example.yaml` and
[SLERP: layer-wise](../slerp.md#layer-wise-slerp).

## Common errors

- **`algorithm 'slerp' requires exactly 2 models`** — SLERP is a two-model
  operation.
- **`interpolation t=... outside [0, 1]`** — use `t` in range, or set
  `allow_extrapolation: true`.
