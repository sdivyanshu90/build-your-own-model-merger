# ADR 0001: A lazy checkpoint abstraction

- Status: accepted
- Date: 2026-07-26

## Problem

We must read three quite different on-disk shapes — safetensors (single and
sharded), pickle-backed PyTorch state dicts, and Hugging Face directories — and
feed them to one merge engine. We also need to inspect tensor shapes/dtypes
*without* loading data, so the planner can compute a shard layout and validate
compatibility cheaply, and so the executor can stream.

## Alternatives considered

1. **Load each model into a full state dict, then merge.** Simple, but holds every
   model in memory at once — impossible for checkpoints larger than RAM, and the
   opposite of the bounded-memory goal.
2. **Convert everything to safetensors up front.** Adds a full-size temp copy and
   an extra pass; wasteful and doubles disk use.
3. **A lazy `Checkpoint` interface** with `keys()`, `tensor_info(key)` (no data),
   `get_tensor(key)` (one tensor), and `raw_metadata()`, plus format-specific
   implementations behind an `open_checkpoint` factory.

## Decision

Adopt option 3. `Checkpoint` is an abstract base class; `SafetensorsCheckpoint`,
`PyTorchCheckpoint`, and `HuggingFaceCheckpoint` implement it.
`HuggingFaceCheckpoint` **delegates** weight access to a safetensors or pytorch
backend and adds config/tokenizer awareness. `open_checkpoint` dispatches by path
shape.

## Consequences

- The merge engine depends only on the abstract interface, so adding a format is
  localized.
- Safetensors gets true lazy per-tensor reads (bounded memory). PyTorch cannot —
  a pickle archive must be loaded whole — so its `get_tensor` serves from an
  in-memory dict, and we document that PyTorch inputs are bounded by a single
  model's size rather than by a tensor.
- `tensor_info` (shape/dtype only) makes planning and compatibility validation
  cheap even for huge models.
- Readers are context managers so file handles are released deterministically.
