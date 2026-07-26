# Limitations

Model merging is a cheap, weight-space operation with real constraints. Be honest
with yourself about what it can and cannot do.

## Merging does not guarantee

- **Improved quality.** The merged model may be worse than the best input on your
  task. Always evaluate.
- **Preserved safety alignment.** Averaging weights can weaken or alter alignment
  and guardrails in unpredictable ways.
- **Preserved calibration.** Confidence/probability calibration is not preserved.
- **Preserved tokenizer compatibility.** If sources use different tokenizers, the
  merged embeddings do not correspond to a single coherent vocabulary.
- **Successful merging of unrelated architectures.** Different `model_type`,
  shapes, or vocab sizes cannot be merged.
- **Successful merging of independently permuted parameter spaces.** Two models
  trained from different initializations may have permuted hidden units; their
  average is generally incoherent (no weight alignment is performed).
- **Removal of inherited biases or vulnerabilities.** The merged model can inherit
  biases and weaknesses from any source.
- **License compliance.** Merging does not change source licenses; you remain
  responsible for each source model's terms.

## Scope boundaries (by design)

- **Two models for SLERP/linear.** These interpolate between exactly two models.
- **No adapter/LoRA merging.** Adapter checkpoints are detected and rejected;
  merge the base models instead.
- **No quantized merging.** Quantized checkpoints are rejected; dequantize first.
- **No permutation alignment.** Techniques like weight matching / Git Re-Basin are
  out of scope.
- **No task-arithmetic vectors.** (TIES/DARE-style task vectors are not
  implemented here.)
- **Sharded `pytorch_model.bin` input** is unsupported; convert to safetensors.

## Practical caveats

- **PyTorch inputs load fully.** Bounded-memory streaming applies to safetensors;
  merging many `.bin` files holds multiple full models. See
  [Memory & performance](memory-and-performance.md).
- **Bit-for-bit reproducibility is not guaranteed across hardware/library
  versions.** Compare the content hash; see [Reproducibility](reproducibility.md).
- **Greedy soup can overfit its selection metric.** Keep the selection set
  separate from your final test set.

## The bottom line

Merging is a fast experiment, not a free lunch. Treat every merged model as an
unvalidated candidate and run a real evaluation — including safety and bias checks
— before relying on it. See
[Evaluating a merged model](tutorials/evaluating-a-merged-model.md).
