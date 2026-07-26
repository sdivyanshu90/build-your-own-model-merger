"""Unit tests for post-write verification and content hashing."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from model_merger.execution.verification import content_hash, verify_output


def _write(path: Path, state: dict[str, torch.Tensor]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path / "model.safetensors"), metadata={"format": "pt"})


def test_verify_passes_for_good_output(tmp_path: Path) -> None:
    _write(tmp_path / "m", {"w": torch.randn(3), "b": torch.zeros(2)})
    result = verify_output(tmp_path / "m")
    assert result.passed
    assert result.checks["all_finite"]


def test_verify_detects_missing_path(tmp_path: Path) -> None:
    result = verify_output(tmp_path / "ghost")
    assert not result.passed


def test_verify_detects_unexpected_keys(tmp_path: Path) -> None:
    _write(tmp_path / "m", {"w": torch.randn(3), "extra": torch.zeros(1)})
    result = verify_output(tmp_path / "m", expected_keys={"w"})
    assert not result.passed
    assert not result.checks["keys_match"]


def test_verify_detects_dtype_mismatch(tmp_path: Path) -> None:
    _write(tmp_path / "m", {"w": torch.randn(3, dtype=torch.float32)})
    result = verify_output(tmp_path / "m", expected_dtypes={"w": torch.float16})
    assert not result.passed
    assert not result.checks["dtypes_match"]


def test_verify_detects_nonfinite(tmp_path: Path) -> None:
    _write(tmp_path / "m", {"w": torch.tensor([1.0, float("nan")])})
    result = verify_output(tmp_path / "m")
    assert not result.passed
    assert not result.checks["all_finite"]


def test_verify_hash_check(tmp_path: Path) -> None:
    _write(tmp_path / "m", {"w": torch.randn(3)})
    from model_merger.utilities.hashing import hash_file

    good = {"model.safetensors": hash_file(tmp_path / "m" / "model.safetensors")}
    assert verify_output(tmp_path / "m", expected_hashes=good).checks["hashes_match"]
    bad = {"model.safetensors": "0" * 64}
    assert not verify_output(tmp_path / "m", expected_hashes=bad).checks["hashes_match"]


def test_content_hash_is_stable_and_value_based(tmp_path: Path) -> None:
    state = {"w": torch.arange(6.0).reshape(2, 3), "b": torch.zeros(3)}
    _write(tmp_path / "a", state)
    # Re-save with different metadata ordering but same tensors.
    (tmp_path / "b").mkdir()
    save_file(state, str(tmp_path / "b" / "model.safetensors"), metadata={"x": "1", "y": "2"})
    assert content_hash(tmp_path / "a") == content_hash(tmp_path / "b")


def test_content_hash_changes_with_values(tmp_path: Path) -> None:
    _write(tmp_path / "a", {"w": torch.zeros(3)})
    _write(tmp_path / "b", {"w": torch.ones(3)})
    assert content_hash(tmp_path / "a") != content_hash(tmp_path / "b")
