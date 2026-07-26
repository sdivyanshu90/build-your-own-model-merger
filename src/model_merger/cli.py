"""Command-line interface (Typer).

Commands:  ``inspect``, ``validate``, ``plan``, ``merge``, ``verify``, ``schema``,
``version``.  Expected failures (anything deriving from
:class:`~model_merger.exceptions.ModelMergerError`) print a one-line message to
stderr and exit with the error's documented code -- no traceback unless
``--debug`` is set.  ``--json`` makes a command emit machine-readable JSON on
stdout; logs always go to stderr so stdout stays clean.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer

from . import __version__
from .config.loaders import load_config_file
from .config.models import MergeConfig
from .exceptions import ModelMergerError
from .logging import configure_logging

app = typer.Typer(
    name="model-merger",
    help="Merge compatible model checkpoints (model soups + SLERP) with bounded memory.",
    no_args_is_help=True,
    add_completion=False,
)


def _echo_json(data: Any) -> None:
    typer.echo(json.dumps(data, indent=2, sort_keys=True))


def _fail(error: ModelMergerError, *, debug: bool) -> None:
    if debug:
        raise error
    typer.echo(f"error [{type(error).__name__}]: {error.message}", err=True)
    raise typer.Exit(code=error.exit_code)


@app.callback()
def _main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose (DEBUG) logging."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only log errors."),
    debug: bool = typer.Option(False, "--debug", help="Show tracebacks for errors."),
) -> None:
    """Configure logging and stash the debug flag for error handling."""

    import logging

    level = logging.DEBUG if verbose else logging.INFO
    configure_logging(level=level, quiet=quiet)
    ctx.obj = {"debug": debug}


def _load_with_overrides(
    config_path: Path,
    *,
    overwrite: bool | None = None,
    device: str | None = None,
    compute_dtype: str | None = None,
) -> MergeConfig:
    data = load_config_file(config_path)
    if overwrite:
        data.setdefault("output", {})["overwrite"] = True
    if device is not None:
        data["device"] = device
    if compute_dtype is not None:
        data.setdefault("precision", {})["compute_dtype"] = compute_dtype
    return MergeConfig.from_dict(data, base_dir=config_path.resolve().parent)


@app.command()
def version() -> None:
    """Print the tool version."""

    typer.echo(__version__)


@app.command()
def schema() -> None:
    """Print the JSON schema for the merge configuration."""

    _echo_json(MergeConfig.model_json_schema())


@app.command()
def inspect(
    ctx: typer.Context,
    model_path: Path = typer.Argument(..., help="Checkpoint file or Hugging Face directory."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
    allow_unsafe: bool = typer.Option(False, "--allow-unsafe", help="Permit pickle loading."),
) -> None:
    """Summarize a checkpoint's tensors and metadata."""

    from .api import inspect_model

    debug = ctx.obj["debug"]
    try:
        summary = inspect_model(model_path, allow_unsafe=allow_unsafe)
    except ModelMergerError as error:
        _fail(error, debug=debug)
        return
    if as_json:
        _echo_json(summary)
        return
    typer.echo(f"path:       {summary['path']}")
    typer.echo(f"format:     {summary['format']}")
    typer.echo(f"tensors:    {summary['tensor_count']}")
    typer.echo(f"parameters: {summary['total_parameters']:,}")
    typer.echo(f"size:       {summary['total_bytes']:,} bytes")
    typer.echo(f"dtypes:     {summary['dtype_histogram']}")
    if "architecture" in summary:
        typer.echo(f"arch:       {summary['architecture']}")


@app.command()
def validate(
    ctx: typer.Context,
    config_path: Path = typer.Argument(..., help="Merge configuration file."),
) -> None:
    """Validate a configuration file (syntax and semantics)."""

    debug = ctx.obj["debug"]
    try:
        config = MergeConfig.from_file(config_path)
    except ModelMergerError as error:
        _fail(error, debug=debug)
        return
    typer.echo(f"OK: {len(config.models)} model(s), algorithm '{config.algorithm.type.value}'")


@app.command()
def plan(
    ctx: typer.Context,
    config_path: Path = typer.Argument(..., help="Merge configuration file."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
    full: bool = typer.Option(False, "--full", help="Include per-tensor entries."),
) -> None:
    """Compute and print the merge plan without writing output."""

    from .api import plan_merge
    from .reporting.serialization import plan_to_markdown

    debug = ctx.obj["debug"]
    try:
        config = MergeConfig.from_file(config_path)
        merge_plan = plan_merge(config)
    except ModelMergerError as error:
        _fail(error, debug=debug)
        return
    if as_json:
        _echo_json(merge_plan.to_dict(include_tensor_entries=full))
    else:
        typer.echo(plan_to_markdown(merge_plan))


@app.command()
def merge(
    ctx: typer.Context,
    config_path: Path = typer.Argument(..., help="Merge configuration file."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan only; write nothing."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace existing output."),
    device: str | None = typer.Option(None, "--device", help="cpu | cuda | cuda:N | auto."),
    compute_dtype: str | None = typer.Option(None, "--compute-dtype", help="Compute dtype."),
    as_json: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable the progress bar."),
) -> None:
    """Execute a merge (or plan it with --dry-run)."""

    from .api import merge_models, plan_merge
    from .reporting.serialization import plan_to_markdown

    debug = ctx.obj["debug"]
    quiet = "-q" in sys.argv or "--quiet" in sys.argv
    try:
        config = _load_with_overrides(
            config_path, overwrite=overwrite, device=device, compute_dtype=compute_dtype
        )
        if dry_run:
            merge_plan = plan_merge(config)
            if as_json:
                _echo_json(merge_plan.to_dict())
            else:
                typer.echo(plan_to_markdown(merge_plan))
            return
        report = merge_models(config, progress=not no_progress and not as_json and not quiet)
    except ModelMergerError as error:
        _fail(error, debug=debug)
        return
    if as_json:
        _echo_json(report.to_dict())
    else:
        typer.echo(f"merged {report.tensor_count} tensors -> {report.output_path}")
        typer.echo(f"run id: {report.run_id}  duration: {report.duration_seconds:.3f}s")
        typer.echo(f"verification: {'PASSED' if report.verification.passed else 'FAILED'}")


@app.command()
def verify(
    ctx: typer.Context,
    output_path: Path = typer.Argument(..., help="Merged checkpoint to verify."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
    allow_unsafe: bool = typer.Option(False, "--allow-unsafe", help="Permit pickle loading."),
) -> None:
    """Verify a written checkpoint; exit non-zero if verification fails."""

    from .api import verify_output

    debug = ctx.obj["debug"]
    try:
        result = verify_output(output_path, allow_unsafe=allow_unsafe)
    except ModelMergerError as error:
        _fail(error, debug=debug)
        return
    if as_json:
        _echo_json(result.to_dict())
    else:
        typer.echo(f"verification: {'PASSED' if result.passed else 'FAILED'}")
        for name, ok in sorted(result.checks.items()):
            typer.echo(f"  {name}: {'ok' if ok else 'FAIL'}")
        for message in result.messages:
            typer.echo(f"  - {message}")
    if not result.passed:
        from .exceptions import VerificationError

        raise typer.Exit(code=VerificationError.exit_code)


if __name__ == "__main__":  # pragma: no cover
    app()
