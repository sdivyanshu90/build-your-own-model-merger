# Model Merger

Model Merger combines the weights of several compatible model checkpoints into a
single checkpoint, **without any retraining**. It implements two families of
techniques:

- **Model soups** — averaging weights across models (uniform, weighted, or a
  greedy subset selected by an evaluator).
- **SLERP** — spherical linear interpolation between two models along the shortest
  arc on a hypersphere, per tensor.

## Who this is for

You have two or more checkpoints that share an architecture — typically several
fine-tunes of the same base model — and you want to combine them into one model
that ideally inherits strengths from each. Merging is cheap (no gradients, no
data) but is **not guaranteed to help**; it must be evaluated. See
[Limitations](limitations.md).

## Design priorities

In order: **correctness, safety, reproducibility, numerical stability,
testability, bounded memory, clear failure behavior, maintainability,
documentation, performance.**

## Where to start

- New to merging? Read [Mathematical foundations](mathematical-foundations.md),
  then follow [First uniform soup](tutorials/first-uniform-soup.md).
- Want the mechanics? See [Architecture](architecture.md) and
  [Memory & performance](memory-and-performance.md).
- Ready to configure a real merge? See the
  [Configuration reference](configuration-reference.md).

## Key guarantees

- No source model is loaded in full alongside all the others — merges stream
  tensor-by-tensor with bounded memory.
- Untrusted pickle checkpoints are never executed without an explicit opt-in.
- Output is written atomically and verified before a merge is reported as
  successful; a partial write is never presented as success.
- Runs are deterministic in tensor content given identical inputs, algorithm, and
  ordering.
