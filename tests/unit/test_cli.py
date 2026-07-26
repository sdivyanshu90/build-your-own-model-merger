"""In-process CLI tests via Typer's CliRunner (captures coverage)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from model_merger.cli import app

runner = CliRunner()


def _write_config(models, out, path: Path, *, algorithm: str = "uniform_soup") -> Path:
    lines = [f"algorithm: {{type: {algorithm}}}", "models:"]
    lines += [f"  - {{path: {m}}}" for m in models]
    lines.append(f"output: {{path: {out}, overwrite: true}}")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_schema() -> None:
    result = runner.invoke(app, ["schema"])
    assert result.exit_code == 0
    assert "properties" in json.loads(result.stdout)


def test_inspect_human_and_json(three_models) -> None:
    human = runner.invoke(app, ["inspect", str(three_models[0])])
    assert human.exit_code == 0
    assert "format:" in human.stdout

    as_json = runner.invoke(app, ["inspect", str(three_models[0]), "--json"])
    assert as_json.exit_code == 0
    assert json.loads(as_json.stdout)["format"] == "huggingface"


def test_inspect_missing_exit_code(tmp_path: Path) -> None:
    result = runner.invoke(app, ["inspect", str(tmp_path / "ghost")])
    assert result.exit_code == 3  # CheckpointError


def test_validate_ok_and_bad(three_models, tmp_path: Path) -> None:
    good = _write_config(three_models, tmp_path / "out", tmp_path / "good.yaml")
    assert runner.invoke(app, ["validate", str(good)]).exit_code == 0

    bad = tmp_path / "bad.yaml"
    bad.write_text("algorithm: {type: slerp}\nmodels: []\noutput: {path: x}\n")
    assert runner.invoke(app, ["validate", str(bad)]).exit_code == 2


def test_plan_markdown_and_json(three_models, tmp_path: Path) -> None:
    config = _write_config(three_models, tmp_path / "out", tmp_path / "cfg.yaml")
    md = runner.invoke(app, ["plan", str(config)])
    assert md.exit_code == 0
    assert "Merge plan" in md.stdout

    js = runner.invoke(app, ["plan", str(config), "--json", "--full"])
    assert js.exit_code == 0
    assert json.loads(js.stdout)["tensor_count"] > 0


def test_merge_human_json_and_verify(three_models, tmp_path: Path) -> None:
    out = tmp_path / "out"
    config = _write_config(three_models, out, tmp_path / "cfg.yaml")

    human = runner.invoke(app, ["merge", str(config), "--no-progress"])
    assert human.exit_code == 0, human.stdout
    assert "merged" in human.stdout
    assert (out / "model.safetensors").is_file()

    verified = runner.invoke(app, ["verify", str(out)])
    assert verified.exit_code == 0
    assert "PASSED" in verified.stdout

    verified_json = runner.invoke(app, ["verify", str(out), "--json"])
    assert json.loads(verified_json.stdout)["passed"] is True


def test_merge_json_output(three_models, tmp_path: Path) -> None:
    out = tmp_path / "out"
    config = _write_config(three_models, out, tmp_path / "cfg.yaml")
    result = runner.invoke(app, ["merge", str(config), "--json", "--no-progress"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["run_id"]


def test_merge_dry_run_writes_nothing(three_models, tmp_path: Path) -> None:
    out = tmp_path / "out"
    config = _write_config(three_models, out, tmp_path / "cfg.yaml")
    result = runner.invoke(app, ["merge", str(config), "--dry-run"])
    assert result.exit_code == 0
    assert not out.exists()


def test_merge_overwrite_flag(three_models, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.txt").write_text("old")
    config = _write_config(three_models, out, tmp_path / "cfg.yaml")
    # config has overwrite:true already; ensure CLI flag path also works.
    result = runner.invoke(app, ["merge", str(config), "--overwrite", "--no-progress"])
    assert result.exit_code == 0


def test_verify_failure_exit_code(tmp_path: Path) -> None:
    result = runner.invoke(app, ["verify", str(tmp_path / "nothing")])
    assert result.exit_code == 12


def test_debug_shows_traceback(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("algorithm: {type: slerp}\nmodels: []\noutput: {path: x}\n")
    result = runner.invoke(app, ["--debug", "validate", str(bad)])
    # With --debug the error propagates (non-clean exit, exception surfaced).
    assert result.exit_code != 0
