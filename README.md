# Model Merger

Merge compatible machine-learning model checkpoints — **without retraining** —
using *model soups* (uniform / weighted / greedy) and *SLERP* (spherical linear
interpolation). Built for correctness, safety, reproducibility, and **bounded
memory** so it works on checkpoints larger than RAM.

- **Model soups**: uniform averaging, weighted averaging, and greedy soups with a
  pluggable evaluator.
- **SLERP**: per-tensor spherical interpolation with numerically stable fallbacks
  for degenerate vectors.
- **Formats**: safetensors (single + sharded), PyTorch state dicts (safe loading
  by default), and Hugging Face directories.
- **Interfaces**: a typed Python API, a polished CLI, and declarative YAML/JSON
  configs.
- **Safe by default**: no pickle execution, no overwrite, `require_equal` for
  non-float buffers, float32 accumulation, NaN/Inf rejection, atomic output, and
  post-write verification.

> Merging is not guaranteed to improve a model. It works best when the sources
> share a training basin (e.g. fine-tunes of a common base). Always evaluate the
> result. See [docs/limitations.md](docs/limitations.md).

---

## Five-minute quick start

```bash
# 1. Install (Python 3.10+)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Make three tiny models locally (no downloads)
python scripts/generate_tiny_test_models.py ./models

# 3. Merge them into a uniform soup
model-merger merge configs/uniform_soup.example.yaml --overwrite

# 4. Verify the result
model-merger verify ./merged-uniform
```

That writes a merged safetensors checkpoint plus a `merge_report.json` and prints
`verification: PASSED`.

### Python API

```python
from model_merger import MergeConfig, merge_models

config = MergeConfig.from_file("configs/slerp.example.yaml")
report = merge_models(config)

print(report.output_path)       # where the merged model was written
print(report.output_hash)       # reproducibility fingerprint (content hash)
print(report.verification.passed)
```

### CLI

```bash
model-merger inspect  MODEL_PATH        # summarize a checkpoint
model-merger validate CONFIG_PATH       # validate a config
model-merger plan     CONFIG_PATH       # dry-run: show the plan, write nothing
model-merger merge    CONFIG_PATH       # execute the merge
model-merger verify   OUTPUT_PATH       # verify a written checkpoint
model-merger schema                     # print the config JSON schema
model-merger version
```

Add `--json` for machine-readable output, `--dry-run` to plan without writing,
and `--debug` to see tracebacks. Exit codes are documented in
[docs/cli-reference.md](docs/cli-reference.md).

---

## How it works (in one paragraph)

The planner opens each source checkpoint lazily, validates compatibility (tensor
shapes/dtypes, architecture, tokenizer), and resolves — per tensor — which
algorithm and dtype apply. The executor then streams **one tensor at a time**:
it loads that tensor from every model, merges it in float32, casts to the output
dtype, and hands it to a writer that buffers **one shard at a time**. Peak memory
is roughly `n_models × sizeof(largest tensor)`, never all models at once. Output
is staged atomically and verified before the merge is reported as successful.

See [docs/architecture.md](docs/architecture.md) and
[docs/memory-and-performance.md](docs/memory-and-performance.md).

---

## Documentation

- [Mathematical foundations](docs/mathematical-foundations.md) — tensors, norms,
  LERP vs SLERP, why weight averaging can work.
- [Model soups](docs/model-soups.md) and [SLERP](docs/slerp.md).
- [Configuration reference](docs/configuration-reference.md),
  [Python API](docs/python-api.md), [CLI reference](docs/cli-reference.md).
- [Checkpoint formats](docs/checkpoint-formats.md),
  [Compatibility & validation](docs/compatibility-validation.md).
- [Security & trust](docs/security-and-trust.md),
  [Reproducibility](docs/reproducibility.md),
  [Limitations](docs/limitations.md).
- [Tutorials](docs/tutorials/) and [architecture decision records](docs/adr/).

Build the docs site with `make docs` (MkDocs Material).

## Development

```bash
make format      # ruff format + fix
make lint        # ruff check
make typecheck   # mypy (strict)
make test-all    # full suite with coverage (>= 90%)
make release-check
```

## License

[Apache 2.0](LICENSE). Merging does not change the license of the source models —
you remain responsible for complying with each source model's license.
