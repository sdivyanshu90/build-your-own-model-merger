# Configuration reference

Configuration is a validated Pydantic model. Unknown fields are **rejected**
(typos are errors, not silently ignored). Load from YAML/JSON with
`MergeConfig.from_file`, or build programmatically. Export the machine schema with
`model-merger schema`.

Relative model/output paths are resolved against the **config file's directory**.
`${VAR}` and `~` are expanded in string values.

## Top level

```yaml
algorithm: {...}        # required
models: [...]           # required, >= 1
output: {...}           # required
precision: {...}        # optional
compatibility: {...}    # optional
non_float_tensors: {...}# optional
rules: [...]            # optional
greedy: {...}           # required iff algorithm.type == greedy_soup
ancillary: {...}        # optional
device: cpu             # cpu | cuda | cuda:N | auto
allow_unsafe_pytorch: false
seed: 0
```

## `algorithm`

```yaml
algorithm:
  type: slerp           # uniform_soup | weighted_soup | greedy_soup | slerp | linear
  t: 0.5                # required for slerp/linear; 0..1 unless allow_extrapolation
  dot_threshold: 0.9995 # slerp only; |cos| above this -> LERP fallback
  allow_extrapolation: false
  normalize_weights: true   # weighted_soup
  allow_negative: false     # weighted_soup
  eps: 1e-8                 # slerp zero-norm cutoff
```

- `slerp` / `linear` require **exactly 2 models**.
- `weighted_soup` requires a `weight` on every model.
- `greedy_soup` requires a `greedy` section.

## `models`

```yaml
models:
  - path: ./model-a     # required
    name: model-a       # optional; defaults to the path basename (deduplicated)
    weight: 0.5         # required for weighted_soup, ignored otherwise
```

## `output`

```yaml
output:
  path: ./merged
  format: safetensors   # safetensors | pytorch
  max_shard_size: 5GB   # "5GB", "512MiB", or bytes
  overwrite: false
  atomic: true
  write_report: true
  report_markdown: false
```

Sizes: `KB/MB/GB/TB` are decimal (1000-based); `KiB/MiB/GiB/TiB` are binary
(1024-based); a bare number is bytes.

## `precision`

```yaml
precision:
  compute_dtype: float32    # arithmetic dtype (auto-promoted if an input is wider)
  output_dtype: preserve    # preserve | highest | float16 | bfloat16 | float32 | float64
  validate_finite: true     # reject NaN/Inf outputs
```

See [Numerical precision policy](numerical-precision-policy.md).

## `compatibility`

```yaml
compatibility:
  mode: strict              # strict | permissive
  require_matching_tokenizer: true
  require_matching_config: true
  require_matching_vocab_size: true
  allow_extra_keys: false
  allow_missing_keys: false
```

See [Compatibility & validation](compatibility-validation.md).

## `non_float_tensors`

```yaml
non_float_tensors:
  policy: require_equal     # require_equal | take_first | take_last | error
```

Applies to every non-floating tensor (integer/bool buffers, position ids,
batch-norm counters), regardless of any rule. `require_equal` (default) fails if
the sources disagree.

## `rules`

Ordered per-tensor algorithm overrides. For a key, the **first matching** rule
wins, ordered by `(-priority, declaration order)`; unmatched keys use the
top-level `algorithm`. Non-float tensors ignore rules.

```yaml
rules:
  - name: embeddings
    priority: 10
    match:
      exact: model.norm.weight   # any combination of the conditions below (AND)
      glob: "*.bias"
      regex: ".*embed.*"
      suffix: ".weight"
      layer_range: {start: 0, end: 11}
      exclude: ["*rotary*"]      # globs that disqualify a key
    algorithm:
      type: linear
      t: 0.2
```

At least one positive condition (`exact`/`glob`/`regex`/`suffix`/`layer_range`) is
required. `layer_range` matches the transformer block index parsed from the key
(`model.layers.N.`, `transformer.h.N.`, `encoder.block.N.`, `encoder.layer.N.`).

## `greedy`

```yaml
greedy:
  direction: maximize        # maximize | minimize
  cache: true
  keep_temp: false
  temp_dir: null             # where temp soups are written
  evaluator:
    type: command            # command | callable
    command: ["python", "score.py", "{model_path}"]  # command: argv, {model_path} substituted
    callable: "pkg.mod:fn"   # callable: dotted module:function
    metric_key: score        # command JSON metric key
    timeout: 300             # seconds
    placeholder: "{model_path}"
```

## `ancillary`

Strategy for non-tensor files (config, tokenizer, generation config):

```yaml
ancillary:
  strategy: base             # base | first | require_equal | fail_on_difference | warn
  base_model: model-a        # for strategy: base — must name a configured model
```

- `base` / `first` — copy from the base model / the first model.
- `require_equal` / `fail_on_difference` — all sources must be byte-identical, else
  a hard error.
- `warn` — copy from base/first, warn on differences.

## Security note

Configs are treated as trusted (they are yours). Values under keys that look
sensitive (`token`, `secret`, `password`, `api_key`, `authorization`) are redacted
from logs via `MergeConfig.redacted_dict()` and never written into reports.
