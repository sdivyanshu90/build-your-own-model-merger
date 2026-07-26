# Troubleshooting

Run any command with `--debug` to see a full traceback. Exit codes are listed in
the [CLI reference](cli-reference.md#exit-codes).

## "models are not compatible" (exit 5)

Compatibility validation found a blocking issue. Inspect the details:

```bash
model-merger plan config.yaml --json | jq .compatibility.findings
```

- **shape mismatch** (fatal) — the models are not the same architecture/size;
  merging is impossible.
- **model_type mismatch** (fatal) — different architectures.
- **dtype / vocab / tokenizer mismatch** (error) — blocks in `strict`. If you
  understand the risk, set `compatibility.mode: permissive`.
- **key-set mismatch** — set `allow_missing_keys`/`allow_extra_keys: true` to copy
  partial keys through, or align the models.

## "could not be loaded safely" (exit 4)

A PyTorch `.bin` needs full pickle unpickling. Only if you trust the file:

```yaml
allow_unsafe_pytorch: true
```
or `--allow-unsafe`. Better: convert it to safetensors.

## "output path already exists" (exit 9)

Pass `overwrite: true` / `--overwrite`, or choose a new `output.path`. Output is
never overwritten silently.

## "non-finite values in <key>" (exit 7)

The merge produced `NaN`/`Inf`. Common causes: extreme weights (extrapolation),
`float16` overflow (a warning precedes it), or already-corrupt inputs. Try
`compute_dtype: float32` (default), reduce extrapolation, or inspect the source
tensors. Set `validate_finite: false` only if you truly want to keep non-finite
output (not recommended).

## "insufficient disk space" (exit 10)

The estimated output plus margin exceeds free space. Free space, point
`output.path` at a larger volume, or reduce `output_dtype`.

## "evaluator command exited with N" / timed out (exit 8)

Your greedy evaluator failed. Run it by hand on a checkpoint directory:

```bash
python score.py /path/to/soup
```

It must print a bare float or `{"score": <float>}` and exit 0. Increase
`greedy.evaluator.timeout` if it is just slow.

## SLERP "interpolation t … outside [0, 1]" (exit 2)

`t` is out of range and extrapolation is disabled. Use `t` in `[0, 1]` or set
`allow_extrapolation: true`.

## Merge is slow or uses too much memory

- Prefer **safetensors** inputs/outputs (PyTorch inputs load fully; see
  [Memory & performance](memory-and-performance.md)).
- Lower `output.max_shard_size` to shrink the writer buffer.
- Use `--device cuda` if a GPU is available.

## Runtime issues the tool can't diagnose

Adapter/quantized checkpoints are rejected by design (see
[Checkpoint formats](checkpoint-formats.md)). For those, dequantize or merge the
base models first.
