# Numerical precision policy

Merging half-precision weights naively loses accuracy: each addition in
`float16`/`bfloat16` rounds to ~3–4 significant decimal digits, and the error
compounds across models. Model Merger centralizes precision decisions in one
policy (`policies/precision.py`).

## Compute dtype

The arithmetic runs in the **compute dtype**, `float32` by default. If any source
tensor is *wider* than the configured compute dtype (e.g. `float64`), the wider
dtype is used so accumulation never loses input precision.

So for `float16`/`bfloat16` inputs, the merge accumulates in `float32` and only
then casts back — materially reducing error (see the demonstration test
`test_float32_accumulation_beats_float16`).

## Output dtype

Chosen by `precision.output_dtype`:

| Policy | Result |
| --- | --- |
| `preserve` (default) | each tensor keeps its source dtype |
| `highest` | the widest dtype among the sources |
| `float16` / `bfloat16` / `float32` / `float64` | forced cast |

## Finiteness

With `validate_finite: true` (default), every merged tensor is checked for
`NaN`/`Inf` and the merge aborts with a `NumericalError` if any appear — a partial
or corrupt result is never written.

## Unsafe-cast warning

Casting a merged tensor to `float16` overflows for magnitudes beyond ±65504.
When that would happen, a warning is logged (the value becomes `Inf`, which the
finiteness check then catches if enabled).

## dtype tradeoffs

| dtype | bits | approx. range | approx. precision | notes |
| --- | --- | --- | --- | --- |
| `float16` | 16 | ±65504 | ~3–4 digits | small range; overflow risk |
| `bfloat16` | 16 | ~±3.4e38 | ~2–3 digits | wide range, low precision; common for LLMs |
| `float32` | 32 | ~±3.4e38 | ~7 digits | the safe accumulation default |
| `float64` | 64 | ~±1.8e308 | ~15 digits | for numerical validation; 2× memory |

**Guidance:** keep `compute_dtype: float32` unless you have a specific reason.
Use `output_dtype: preserve` to match your serving stack, or force a dtype to
shrink the output. Use `float64` compute only to validate numerical behavior.

See [ADR-0003](adr/0003-numerical-precision-policy.md) for the rationale.
