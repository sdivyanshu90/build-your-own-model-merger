# Reproducibility

## What is guaranteed

Given **identical inputs, algorithm, parameters, and model ordering**, a merge
produces **bit-identical tensor content**. The merge performs no sampling; tensor
and shard ordering are deterministic (keys are processed in sorted order).

The merge report records everything needed to reproduce a run:

- tool version, Python, PyTorch, safetensors, transformers versions
- operating system / platform, device
- algorithm and its parameters
- input model paths (and optional content hashes)
- output file hashes and a **content hash** (`output_hash`)
- run id and timestamp

## The reproducibility fingerprint (`output_hash`)

`output_hash` is a **content hash**: a digest over tensor keys, dtypes, and
values, computed by streaming the output one tensor at a time. It depends only on
what the model *is*, not on container-level details.

This distinction matters because **raw safetensors file bytes are not
byte-stable**: the safetensors library serializes its `__metadata__` map via a
hash map with per-process random ordering, so two runs with identical tensors can
produce files that differ only in header byte order. The per-file hashes in
`output_hashes` therefore describe *this run's exact bytes* (useful for verifying
what is on disk), while `output_hash` is the value to compare across runs.

## What cannot be guaranteed bit-for-bit

- **Across different hardware** — CPU vs GPU, or different CPU/BLAS
  implementations, can produce different floating-point rounding.
- **Across library versions** — a different torch/BLAS version may reorder
  reductions.
- **Raw file bytes** — safetensors metadata ordering (above); use `output_hash`.
- **Peak memory** — `peak_memory_bytes` is whole-process and platform-dependent.

For strict reproducibility, pin the environment (the report captures the versions
you used) and compare `output_hash`.

## Verification

After writing, the output is re-opened and checked: every expected key is present,
every tensor loads, floating tensors are finite, dtypes match the plan, the shard
index resolves, `config.json` parses, and no staging/temporary files remain. A
merge is **not reported successful until verification passes**; a failure raises
`VerificationError` (exit 12).

Run verification standalone at any time:

```bash
model-merger verify ./merged --json
```

## Seeding

The merge core does not sample, but evaluators might. `seed` seeds Python,
NumPy, and torch RNGs before evaluators run, so a callable/command evaluator that
samples can be made deterministic.
