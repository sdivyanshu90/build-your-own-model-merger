# Glossary

**Ancillary files** — non-weight files in a model directory: `config.json`,
`generation_config.json`, and tokenizer artifacts.

**Basin (loss basin)** — a region of weight space where loss is low. Models in the
same basin (e.g. fine-tunes of one base) tend to merge well.

**bfloat16 / float16 / float32 / float64** — floating-point formats of 16/16/32/64
bits, trading range and precision for size. See
[Numerical precision policy](numerical-precision-policy.md).

**Compute dtype** — the dtype the merge arithmetic runs in (float32 by default).

**Content hash (`output_hash`)** — a digest over tensor keys/dtypes/values; the
reproducibility fingerprint, independent of file byte order.

**Extrapolation** — using interpolation coefficients outside `[0, 1]` (or negative
weights), producing a result outside the convex hull of the inputs.

**Greedy soup** — a soup over a subset of models selected greedily by an evaluator.

**LERP** — linear interpolation, `(1−t)·a + t·b`.

**Merge key** — a tensor key present in *all* source models (the ones merged).

**Non-float tensor** — an integer/bool tensor (buffer, mask, counter, position
ids) that is selected rather than averaged.

**Passthrough key** — a tensor present in only some models, copied verbatim from
the first model that has it (when partial keys are allowed).

**Plan** — the side-effect-free description of a merge (models, per-tensor
algorithm/dtype, shard layout) produced before writing.

**SLERP** — spherical linear interpolation along the arc between two directions.

**Shard** — one output file when a checkpoint is split by size, indexed by a
`*.index.json`.

**Soup** — a model produced by averaging weights across models.

**State dict** — a mapping from parameter name to tensor.

**Streaming (tensor-at-a-time)** — processing one tensor across all models before
moving to the next, keeping memory bounded.

**Validation vs verification** — validation checks *inputs* before writing;
verification checks the *written output* afterward.

**weights_only** — the safe `torch.load` mode that refuses to unpickle arbitrary
code.
