"""CLI end-to-end tests via subprocess (real exit codes and stdout)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[2]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_ROOT / "src"), env.get("PYTHONPATH", "")])
    return subprocess.run(
        [sys.executable, "-m", "model_merger", *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _write_config(models, out, path: Path) -> Path:
    lines = ["algorithm: {type: uniform_soup}", "models:"]
    lines += [f"  - {{path: {m}}}" for m in models]
    lines.append(f"output: {{path: {out}, overwrite: true}}")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_version() -> None:
    result = run_cli("version")
    assert result.returncode == 0
    assert result.stdout.strip()


def test_schema_is_json() -> None:
    result = run_cli("schema")
    assert result.returncode == 0
    assert json.loads(result.stdout)["title"]


def test_inspect_json(three_models) -> None:
    result = run_cli("inspect", str(three_models[0]), "--json")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["format"] == "huggingface"


def test_validate_ok(three_models, tmp_path: Path) -> None:
    config = _write_config(three_models, tmp_path / "out", tmp_path / "cfg.yaml")
    result = run_cli("validate", str(config))
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_validate_bad_config_exit_code(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("algorithm: {type: nonsense}\nmodels: []\noutput: {path: x}\n")
    result = run_cli("validate", str(bad))
    assert result.returncode == 2  # ConfigurationError


def test_merge_and_verify(three_models, tmp_path: Path) -> None:
    out = tmp_path / "out"
    config = _write_config(three_models, out, tmp_path / "cfg.yaml")
    merged = run_cli("merge", str(config), "--no-progress")
    assert merged.returncode == 0, merged.stderr
    assert (out / "model.safetensors").is_file()

    verified = run_cli("verify", str(out))
    assert verified.returncode == 0
    assert "PASSED" in verified.stdout


def test_dry_run_writes_nothing(three_models, tmp_path: Path) -> None:
    out = tmp_path / "out"
    config = _write_config(three_models, out, tmp_path / "cfg.yaml")
    result = run_cli("merge", str(config), "--dry-run")
    assert result.returncode == 0
    assert not out.exists()


def test_verify_missing_output_exit_code(tmp_path: Path) -> None:
    result = run_cli("verify", str(tmp_path / "nope"))
    assert result.returncode == 12  # VerificationError
