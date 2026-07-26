# Contributing

Thanks for your interest in improving Model Merger. This project prioritizes, in
order: **correctness, safety, reproducibility, numerical stability, testability,
bounded memory, clear failure behavior, maintainability, documentation, and
performance.** Please keep that ordering in mind when weighing tradeoffs.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
pre-commit install
```

## Workflow

```bash
make format      # auto-format and fix
make lint        # ruff check + format check
make typecheck   # mypy (strict)
make test-all    # full suite with coverage
make docs        # build docs with --strict
```

All of these must pass before a pull request is merged. CI runs them on every push.

## Ground rules

- **Every change ships with tests.** New numerical behavior needs both example-
  based and, where a property holds, property-based tests.
- **No placeholders.** No `TODO`, `FIXME`, bare `pass`, or `...` stubs in shipped
  code.
- **Keep I/O separate from math.** Algorithms in `algorithms/` are pure tensor
  functions; anything touching the filesystem lives in `checkpoints/` or
  `execution/`.
- **Safety defaults win.** Do not weaken a safe default (pickle rejection, no
  overwrite, `require_equal`, finite validation) without a documented rationale
  and an opt-in flag.
- **Type everything.** Public functions have annotations and docstrings.
- **Document the "why".** Comments and docs should explain rationale the code
  cannot, not restate the code.

## Commit and PR conventions

- Branch from `main`; use short, descriptive branch names (`fix/slerp-antiparallel`).
- Write imperative commit subjects ("Add greedy resume support").
- Reference issues where relevant and describe the tradeoffs in the PR body.
- Update `CHANGELOG.md` under `[Unreleased]`.

## Reporting security issues

Please do not open public issues for vulnerabilities. See [SECURITY.md](SECURITY.md).
