# Testing strategy

The suite is deterministic, isolated, network-free, and runs against tiny models
generated on the fly (`scripts/generate_tiny_test_models.py`). No test downloads a
large model.

Run it:

```bash
make test-all           # full suite with coverage
pytest -m "not slow and not performance"   # fast subset
pytest tests/unit -q    # one category
```

## Categories

### Unit (`tests/unit/`)
The numerical core and every policy in isolation: uniform/weighted/greedy soups,
SLERP endpoints/midpoint/fallbacks, weight validation, precision policy, non-float
policies, layer-rule matching and precedence, config parsing and semantic
validation, checkpoint readers/writer, compatibility classification, reporting
serialization, CLI (via Typer's in-process runner), evaluators, and verification.

### Property-based (`tests/property/`)
Hypothesis checks mathematical invariants across many inputs:

- **Weighted average**: single model with weight 1 is identity; equal weights =
  mean; permutation invariance; positive-scaling invariance under normalization;
  shape preservation; finite in → finite out; normalized weights sum to 1.
- **SLERP**: `t=0`→first, `t=1`→second; swap symmetry (`swap endpoints, t→1−t`);
  shape preservation; near-parallel produces no NaN; identical vectors return the
  same; unit inputs stay ~unit norm; extrapolation rejected when disabled.

Tolerances are dtype-aware (`atol` a few ULPs for trig-based SLERP).

### Integration (`tests/integration/`)
Full merges through the public API and CLI: safetensors, PyTorch files, Hugging
Face directories, sharded output and sharded input round-trips, config-driven
merges, greedy soup end-to-end (real command evaluator), dry-run planning,
existing-output protection, and ancillary reconciliation.

### Regression (`tests/regression/`)
Deterministic reference checks: merged tensors equal a recomputed reference within
an explicit tolerance (`1e-6` for averaging, `1e-5` for SLERP's trig), and two
runs produce identical tensor content and `output_hash`.

### Performance and memory (`tests/performance/`)
Structural (not wall-clock) assertions of streaming: source loads are grouped per
key (bounded live set), total loads equal `n_models × n_keys` (no eager
whole-model reads), and the writer buffer never exceeds the largest shard. Marked
`performance`.

### Security (`tests/security/`)
Pickle rejection by default (and opt-in load), malformed metadata rejection,
command-evaluator argv safety (no shell injection), path-containment checks, and
atomic-output rollback on failure.

## Coverage

The suite maintains **≥ 90%** line coverage (numerical algorithms and config
validation higher). Coverage is a floor, not the goal — tests assert *behavior and
invariants*, and core numerical behavior is not mocked.

## Determinism and isolation

Every test uses `tmp_path`, avoids execution-order dependence, and produces
meaningful failure messages. `conftest.py` adds `src`/`scripts` to the path so the
suite runs with or without an editable install.
