# Checkpoint formats

Model Merger reads three input shapes and writes two. Use
[`model-merger inspect`](cli-reference.md#inspect) to see how a path is
interpreted.

## Reading

`open_checkpoint(path)` dispatches by path shape:

| Path | Reader |
| --- | --- |
| a directory | `HuggingFaceCheckpoint` |
| `*.safetensors` or `*.safetensors.index.json` | `SafetensorsCheckpoint` |
| `*.bin` / `*.pt` / `*.pth` | `PyTorchCheckpoint` |

All readers are **lazy**: they can report each tensor's shape and dtype without
loading data, and load exactly one tensor on demand. This is what enables the
streaming merge.

### Safetensors (preferred)

Safetensors is not pickle-backed (no code execution) and supports true per-tensor
reads. Both single-file and sharded (`model.safetensors.index.json` +
`model-00001-of-000NN.safetensors`) layouts are supported. Container metadata is
read from the file header.

### PyTorch state dicts

`torch.load` uses pickle, which can execute arbitrary code. The reader loads with
`weights_only=True` by default (safe for untrusted files). If that fails — a
legacy checkpoint needs full unpickling — it raises `UnsafeCheckpointError` unless
you pass `allow_unsafe_pytorch: true` / `--allow-unsafe`, in which case it retries
`weights_only=False` with a loud warning.

A pickle archive cannot be read one tensor at a time, so the whole state dict is
loaded once and served from memory. This is bounded by a single model's size; for
strictly bounded multi-model memory, use safetensors inputs. Common wrappers
(`{"state_dict": …}`, `{"model": …}`) are unwrapped automatically; non-tensor
entries are dropped.

### Hugging Face directories

A directory is read as a model plus its ancillary files. Weight backend selection
order: `model.safetensors.index.json` → `model.safetensors` → any other single
`*.safetensors` → `pytorch_model.bin`. The directory's `config.json` and tokenizer
files feed [compatibility validation](compatibility-validation.md) and the
[ancillary strategy](configuration-reference.md#ancillary).

**Adapters/LoRA** (`adapter_config.json` + `adapter_model.*`) are detected and
rejected — merging adapters is a different operation; merge the base models
instead. **Sharded `pytorch_model.bin` indexes** are not supported as input;
convert to safetensors.

## Writing

- **Safetensors** (default): tensors are written to shards sized by
  `output.max_shard_size`, with an index JSON when sharded. Shards are buffered
  one at a time (bounded memory). This is the verified, streaming path.
- **PyTorch**: a single `pytorch_model.bin` written via `torch.save`. This holds
  the whole state dict in memory before saving, so it is bounded by the full model
  size — prefer safetensors for large models.

Ancillary files (config, tokenizer, generation config) are copied into the output
directory per the ancillary strategy. A `merge_report.json` (and optionally
`merge_report.md`) is written after verification.

## Quantized and adapter checkpoints

Quantized checkpoints (`quantization_config` in `config.json`) are rejected —
averaging quantization scales/zero-points is meaningless, and there is no
dequantize-merge-requantize path here. Dequantize to a float checkpoint first if
you must merge.
