# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-26

### Added
- Uniform, weighted, and greedy model soups.
- SLERP and linear interpolation with numerically stable fallbacks.
- Per-tensor, bounded-memory streaming merge for safetensors checkpoints.
- Safetensors (single and sharded), PyTorch (`weights_only` by default), and
  Hugging Face directory readers; safetensors and PyTorch writers.
- Compatibility validation (tensors, architecture, tokenizer) with strict and
  permissive modes.
- Layer rules: match by exact name, glob, regex, suffix, or layer range.
- Non-float tensor policies: `require_equal` (default), `take_first`,
  `take_last`, `error`.
- Precision policy with float32 accumulation by default and configurable output
  dtype.
- Declarative YAML/JSON configuration (Pydantic), a typed Python API, and a
  Typer CLI (`inspect`, `validate`, `plan`, `merge`, `verify`, `schema`,
  `version`).
- Atomic output staging, disk-space preflight, and post-write verification.
- Structured JSON/Markdown merge reports with reproducibility metadata and a
  content-based reproducibility hash.
- Full unit, property-based, integration, security, performance, and regression
  test suites.

[Unreleased]: https://github.com/example/model-merger/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/example/model-merger/releases/tag/v0.1.0
