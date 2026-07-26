# SLERP

Spherical linear interpolation blends **two** models along the shortest arc
between their weight directions, per tensor.

## The formula

For flattened tensors `v₀`, `v₁` and coefficient `t`:

```
Ω = arccos( (v̂₀ · v̂₁) )                  # angle between unit directions
SLERP(v₀, v₁; t) = sin((1−t)Ω)/sin(Ω) · v₀ + sin(tΩ)/sin(Ω) · v₁
```

The coefficients are applied to the **original** (unnormalized) vectors, so the
endpoints are reproduced exactly: `t = 0 → v₀`, `t = 1 → v₁`. See the
[worked example](mathematical-foundations.md#spherical-interpolation-slerp).

## Per-tensor, not global

Model Merger runs SLERP on **each tensor independently**: it flattens the tensor,
interpolates, and reshapes. The alternative — flattening the *entire model* into
one giant vector and interpolating once — would require holding a full model
vector in memory and breaks streaming. Per-tensor SLERP keeps memory bounded and
matches the de-facto convention of model-merging tools. See
[ADR-0003](adr/0003-numerical-precision-policy.md).

## Numerical stability

A naive implementation of the formula fails in several ways. Model Merger handles
each:

| Situation | Problem | Handling |
| --- | --- | --- |
| Zero-norm vector (`‖v‖ < eps`) | Direction undefined | Fall back to LERP |
| Nearly parallel (`\|cos\| > dot_threshold`) | `sin(Ω) → 0`, division blows up | Fall back to LERP |
| Nearly antiparallel (`\|cos\| > dot_threshold`) | `sin(Ω) → 0` | Fall back to LERP |
| `cos` slightly outside `[-1, 1]` | `arccos` returns NaN | Clamp before `arccos` |
| fp16/bf16 inputs | Precision loss in trig | Compute in float32 |

The threshold is configurable:

```yaml
algorithm:
  type: slerp
  t: 0.5
  dot_threshold: 0.9995   # above this |cos|, use LERP
  eps: 1e-8               # zero-norm cutoff
```

`dot_threshold` uses the **absolute** cosine, so it catches both the parallel and
antiparallel cases with one number.

## Interpolation coefficient and extrapolation

`t` normally lies in `[0, 1]`. Values outside that range **extrapolate** along the
geodesic. Extrapolation is disabled by default; enable it explicitly:

```yaml
algorithm:
  type: slerp
  t: 1.2
  allow_extrapolation: true
```

With extrapolation disabled, an out-of-range `t` is rejected at config time with
an actionable error.

## Layer-wise SLERP

Different parts of a model can use different coefficients (or a different
algorithm) via [rules](configuration-reference.md#rules). A common pattern:
gentle blending of embeddings, and progressively stronger interpolation toward
`model-b` in higher layers. See
[configs/layerwise_slerp.example.yaml](https://github.com/example/model-merger/blob/main/configs/layerwise_slerp.example.yaml).

## Output dtype and device

SLERP computes in the [precision policy's](numerical-precision-policy.md) compute
dtype (float32 by default) and writes in the output dtype (preserve by default).
Computation can run on CPU or CUDA (`device:`); the result is moved back to CPU
for writing.

## Choosing SLERP vs weighted soup of two models

A two-model weighted soup with weights `(1−t, t)` is exactly LERP. SLERP differs
only when the two tensors point in meaningfully different directions with similar
magnitude — then SLERP preserves magnitude along the arc while LERP shrinks it.
For near-parallel tensors the two coincide (SLERP falls back to LERP).
