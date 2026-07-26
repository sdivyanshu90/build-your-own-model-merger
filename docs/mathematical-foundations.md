# Mathematical foundations

This page builds up the math from first principles. It assumes basic Python and
high-school algebra, but not machine learning.

## Parameters, tensors, and state dicts

A neural network is, numerically, a large collection of **parameters** —
floating-point numbers. They are grouped into **tensors** (multi-dimensional
arrays): a weight matrix might be a `4096 × 4096` tensor, a bias a `4096` vector.
A checkpoint stores these tensors in a **state dict**: a mapping from a string
key (e.g. `model.layers.0.self_attn.q_proj.weight`) to a tensor.

Merging operates on state dicts: for each key present in all models, combine the
corresponding tensors into one output tensor.

## Flattening a tensor to a vector

Any tensor can be **flattened** into a 1-D vector by reading its elements in
order. A `2 × 3` matrix becomes a length-6 vector. Flattening loses no
information (we remember the original shape and restore it afterward) and lets us
use vector operations like norms and dot products.

## Norms

The **(Euclidean) norm** of a vector `v = [v₁, …, vₙ]` is its length:

```
‖v‖ = sqrt(v₁² + v₂² + … + vₙ²)
```

A **unit vector** has norm 1. Dividing a vector by its norm ("normalizing")
produces a unit vector pointing the same direction.

## Dot products and the angle between vectors

The **dot product** of two vectors is `v · w = v₁w₁ + v₂w₂ + … + vₙwₙ`. It relates
to the angle `Ω` between them:

```
cos(Ω) = (v · w) / (‖v‖ · ‖w‖)
```

So `cos(Ω)` for two **unit** vectors is just their dot product, and
`Ω = arccos(v̂ · ŵ)`. Because floating-point rounding can push the dot product
slightly outside `[-1, 1]`, we clamp before `arccos` — otherwise `arccos` returns
NaN.

## Linear interpolation (LERP)

To blend two vectors by a fraction `t ∈ [0, 1]`:

```
LERP(v, w; t) = (1 − t)·v + t·w
```

At `t = 0` you get `v`; at `t = 1` you get `w`; at `t = 0.5` the midpoint. LERP
moves along the **straight line** between the two vectors. This is exactly what a
two-model weighted average with weights `(1−t, t)` computes.

**Small numerical example.** With `v = [1, 0]`, `w = [0, 1]`, `t = 0.5`:
`LERP = [0.5, 0.5]`, whose norm is `0.707` — *shorter* than either input.

## Spherical interpolation (SLERP)

If the two vectors have similar magnitude and you care about **direction**, the
straight-line midpoint shrinks the magnitude (as above). SLERP instead moves
along the **arc** of the sphere, preserving the geodesic path between directions:

```
SLERP(v, w; t) = sin((1−t)·Ω)/sin(Ω) · v + sin(t·Ω)/sin(Ω) · w
```

where `Ω = arccos(v̂ · ŵ)`. At `t = 0` the coefficients are `(1, 0)` → `v`; at
`t = 1` they are `(0, 1)` → `w`. In between, the interpolant follows the shortest
arc.

**Same example** with unit vectors `v = [1, 0]`, `w = [0, 1]`, `Ω = 90°`,
`t = 0.5`: `SLERP = [0.707, 0.707]`, which has norm `1.0` and sits at exactly 45°.
The direction rotates uniformly; the magnitude is preserved.

**Degenerate cases.** When the two directions are nearly identical or nearly
opposite, `sin(Ω) → 0` and the formula divides by ~0. When a vector has zero norm,
its direction is undefined. In both cases Model Merger falls back to LERP, which is
well-defined everywhere. See [SLERP](slerp.md).

## Why weight-space averaging can work

Averaging the weights of two *arbitrary* independently trained networks usually
produces garbage: hidden units can be permuted, so "unit 5" in one model has no
relationship to "unit 5" in another. The average of two coherent-but-misaligned
solutions is not itself coherent.

Averaging works when the models occupy the **same loss basin** — typically because
they were fine-tuned from a **shared initialization**. Empirically, such models
are connected by paths of low loss (linear mode connectivity), so points *between*
them (including the average) can also have low loss. This is the premise behind
model soups (Wortsman et al., 2022).

Consequences that shape this tool's defaults:

- Merging is only attempted for tensors that **match in shape** across models; a
  shape mismatch is fatal.
- **Non-float** tensors (integer buffers, boolean masks, batch-norm counters) are
  not averaged — averaging them is meaningless. They are resolved by selection
  (`require_equal` by default).
- Compatibility of **architecture** and **tokenizer** is checked, because two
  models with different vocabularies do not share a weight space.

## Merging is not ensembling

**Ensembling** runs several models at inference time and combines their *outputs*
(e.g. averaging logits). It keeps every model in memory and multiplies inference
cost, but makes no assumptions about weight-space geometry.

**Merging** combines *weights* into a single model, so inference cost and memory
are those of one model — but it only works when the weights are compatible. The
two techniques are different tools; this project does merging.

## Further reading

- Wortsman et al., "Model soups", 2022.
- Frankle et al., "Linear mode connectivity and the lottery ticket hypothesis".
- Shoemake, "Animating rotation with quaternion curves" (the origin of SLERP).
