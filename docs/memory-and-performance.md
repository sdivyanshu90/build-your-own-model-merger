# Memory and performance

Model Merger is designed to merge checkpoints **larger than system RAM**. This
page explains the memory model and how to tune it.

## The memory model

Merges stream **one tensor at a time**:

1. For a key, load that tensor from each of the `n` models (lazily).
2. Cast to the compute dtype (float32) and merge.
3. Cast to the output dtype and hand to the writer.
4. Release the tensor; move to the next key.

The writer buffers **one shard at a time**: it accumulates tensors for the current
shard and flushes them to disk the moment the shard is complete, then frees the
buffer.

**Peak memory ≈ `n_models × sizeof(largest tensor)` (in the compute dtype) + one
output shard (≤ `max_shard_size`).** No source model is ever fully resident, and
never are all models resident at once.

Two properties guarantee this and are enforced by tests
([`tests/performance/`](testing-strategy.md)):

- source loads are grouped per key (all models' copy of key *K* loaded together,
  released before *K+1*);
- the writer buffer never exceeds the largest single shard's tensor count.

### The exception: PyTorch inputs and outputs

A pickle archive cannot be read one tensor at a time. A `PyTorchCheckpoint` loads
its whole state dict once (bounded by *that* model's size). Merging `n` PyTorch
`.bin` models therefore holds up to `n` full models. For strictly bounded
multi-model memory, use **safetensors** inputs. Likewise, PyTorch **output** is a
single `torch.save`, buffering the full model. Use safetensors output for large
models.

## Tuning

| Knob | Effect |
| --- | --- |
| `output.max_shard_size` | Smaller shards → smaller writer buffer, more files |
| `precision.compute_dtype` | `float32` is the safe default; `bfloat16` halves compute memory at some accuracy cost |
| `precision.output_dtype` | `preserve` keeps source dtype; explicit dtype can shrink output |
| `device` | `cuda` moves compute to GPU (source/output still stream through CPU) |

## Disk preflight

Before writing, the executor estimates output size (from planned tensor
shapes/dtypes) and checks free space with a 5% margin, raising
`InsufficientDiskSpaceError` (exit 10) rather than filling the disk mid-write.

## Benchmarking

`scripts/benchmark_merge.py` reports wall-clock, peak resident memory (from the
report), input/output sizes, tensor count, throughput, device, and algorithm:

```bash
python scripts/benchmark_merge.py configs/uniform_soup.example.yaml --json
```

Peak memory in the report comes from `getrusage` (process peak RSS) on Unix — a
whole-process figure, useful as an upper bound but not a per-tensor measurement.

## Reproducibility note on peak memory

`peak_memory_bytes` is platform-dependent (KiB units on Linux, bytes on macOS) and
reflects the whole process, so treat it as indicative. The streaming *behavior*
(not the absolute RSS) is what the performance tests assert.
