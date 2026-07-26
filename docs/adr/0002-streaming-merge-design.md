# ADR 0002: Streaming, tensor-at-a-time merge

- Status: accepted
- Date: 2026-07-26

## Problem

The tool must merge checkpoints larger than RAM with bounded, predictable memory,
while still supporting per-tensor algorithm choices, sharded output, atomic
writes, and post-write verification.

## Alternatives considered

1. **Whole-model in memory.** Load all state dicts, merge, save. Peak memory =
   sum of all models; violates the core requirement.
2. **Global-vector merge.** Flatten each model to one big vector and operate once
   (needed for a "global" SLERP). Requires a full model vector in memory and
   defeats streaming.
3. **Tensor-at-a-time streaming.** Iterate the sorted key union; for each key load
   that tensor from every model, merge, write, release.

## Decision

Adopt option 3. The **planner** computes the key order, per-key algorithm/dtype,
and a **shard plan** (contiguous runs of keys) up front. The **executor** iterates
keys in that order; the **writer** buffers one shard and flushes it the moment its
last key arrives.

SLERP is therefore **per-tensor**, not global (see
[ADR-0003](0003-numerical-precision-policy.md)) — the streaming design and the
global-vector variant are mutually exclusive, and streaming wins on memory.

## Consequences

- Peak memory ≈ `n_models × sizeof(largest tensor)` + one shard, independent of
  model size. Enforced by `tests/performance/` (grouped loads; writer buffer
  bounded by the largest shard).
- Because the shard plan groups keys into contiguous runs and the executor emits
  keys in the same sorted order, the writer only ever holds the current shard.
- Output is staged in a sibling directory and renamed atomically on success;
  failures roll back. Disk space is checked before writing.
- Verification re-streams the output (one tensor at a time) so it, too, stays
  within bounded memory.
- Greedy soup is layered on top: it is model selection that reuses the same
  streaming uniform-soup merge to materialize temporary candidates.
