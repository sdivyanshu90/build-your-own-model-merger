# ADR 0003: Numerical precision and per-tensor SLERP

- Status: accepted
- Date: 2026-07-26

## Problem

Merged model quality is sensitive to floating-point precision. Half-precision
weights (`float16`/`bfloat16`) are common, and averaging them naively accumulates
rounding error. Separately, SLERP can be defined per-tensor or over a global
flattened model vector, and the naive SLERP formula is numerically fragile.

## Decisions

### 1. Accumulate in float32 by default

Merges compute in a **compute dtype** that is `float32` by default, promoted to a
wider dtype only if an input is wider (e.g. `float64`). Inputs in `float16`/
`bfloat16` are up-cast for the arithmetic and cast back to the output dtype
afterward. A demonstration test shows float32 accumulation yields lower error than
float16 accumulation for the same inputs.

**Tradeoff:** float32 uses more transient memory than half precision. Given the
bounded-memory streaming design (only a few tensors resident), this is
affordable, and correctness ranks above performance.

### 2. Output dtype defaults to `preserve`

The written dtype defaults to each tensor's source dtype (`preserve`), with
`highest` and explicit-dtype options. This keeps the merged model drop-in for the
same serving stack unless the user asks otherwise. Casting to `float16` warns on
overflow, and finiteness is validated.

### 3. SLERP is per-tensor

SLERP flattens and interpolates **each tensor independently**, then reshapes. The
global-vector alternative would require a full model vector in memory,
contradicting [ADR-0002](0002-streaming-merge-design.md). Per-tensor SLERP also
matches the de-facto convention of model-merging tools.

**Tradeoff:** per-tensor SLERP interpolates each tensor's direction separately
rather than the model's global direction. In practice this is the accepted and
useful behavior; a global variant is explicitly out of scope.

### 4. Numerically stable SLERP

The implementation clamps the cosine into `[-1, 1]` before `arccos`, and falls
back to LERP when either vector is near-zero or the vectors are near-parallel or
near-antiparallel (`|cos| > dot_threshold`, using the absolute cosine to catch
both). This avoids `NaN` and division by ~0. `dot_threshold` and `eps` are
configurable.

## Consequences

- The precision policy is a small, pure value object, unit-tested in isolation.
- SLERP degenerate cases are covered by both example and property-based tests
  (near-parallel produces no NaN; endpoints exact; swap symmetry).
- Users who need bit-exact behavior compare the content hash and pin their
  environment (see [Reproducibility](../reproducibility.md)).
