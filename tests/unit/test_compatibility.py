"""Unit tests for compatibility analysis and finding classification."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from model_merger.checkpoints import SafetensorsCheckpoint, open_checkpoint
from model_merger.compatibility import analyze_tensors, validate_compatibility
from model_merger.compatibility.report import CompatibilityReport
from model_merger.config.models import CompatibilityConfig
from model_merger.exceptions import CompatibilityError
from model_merger.types import CompatibilityMode, Severity


def _make_safetensors(path: Path, state: dict[str, torch.Tensor]) -> SafetensorsCheckpoint:
    save_file(state, str(path))
    return SafetensorsCheckpoint(path)


def test_identical_keys_all_merge(tmp_path: Path) -> None:
    a = _make_safetensors(tmp_path / "a.safetensors", {"w": torch.zeros(3), "b": torch.zeros(2)})
    b = _make_safetensors(tmp_path / "b.safetensors", {"w": torch.ones(3), "b": torch.ones(2)})
    result = analyze_tensors([a, b])
    assert result.merge_keys == ["b", "w"]
    assert not result.passthrough


def test_shape_mismatch_is_fatal(tmp_path: Path) -> None:
    a = _make_safetensors(tmp_path / "a.safetensors", {"w": torch.zeros(3)})
    b = _make_safetensors(tmp_path / "b.safetensors", {"w": torch.zeros(4)})
    result = analyze_tensors([a, b])
    codes = {finding.code for finding in result.report.findings}
    assert "tensor.shape_mismatch" in codes
    assert result.report.max_severity is Severity.FATAL


def test_key_mismatch_error_in_strict(tmp_path: Path) -> None:
    a = _make_safetensors(
        tmp_path / "a.safetensors", {"w": torch.zeros(3), "extra": torch.zeros(1)}
    )
    b = _make_safetensors(tmp_path / "b.safetensors", {"w": torch.zeros(3)})
    result = analyze_tensors([a, b], allow_missing=False)
    assert any(f.code == "tensor.key_set_mismatch" for f in result.report.findings)


def test_key_mismatch_passthrough_when_allowed(tmp_path: Path) -> None:
    a = _make_safetensors(
        tmp_path / "a.safetensors", {"w": torch.zeros(3), "extra": torch.zeros(1)}
    )
    b = _make_safetensors(tmp_path / "b.safetensors", {"w": torch.zeros(3)})
    result = analyze_tensors([a, b], allow_missing=True)
    assert "extra" in result.passthrough
    assert result.passthrough["extra"] == 0


def test_report_blocking_semantics() -> None:
    report = CompatibilityReport()
    report.add(Severity.ERROR, "x", "an error")
    assert not report.is_compatible(CompatibilityMode.STRICT)
    assert report.is_compatible(CompatibilityMode.PERMISSIVE)


def test_report_fatal_always_blocks() -> None:
    report = CompatibilityReport()
    report.add(Severity.FATAL, "x", "fatal")
    assert not report.is_compatible(CompatibilityMode.PERMISSIVE)
    with pytest.raises(CompatibilityError):
        report.raise_if_incompatible(CompatibilityMode.PERMISSIVE)


def test_validate_compatibility_on_models(three_models) -> None:
    checkpoints = [open_checkpoint(path) for path in three_models]
    try:
        report, tensor_result = validate_compatibility(checkpoints, CompatibilityConfig())
        assert report.is_compatible(CompatibilityMode.STRICT)
        assert len(tensor_result.merge_keys) > 0
    finally:
        for ckpt in checkpoints:
            ckpt.close()


def test_architecture_mismatch_detected(tmp_path: Path, model_factory) -> None:
    a = model_factory("a", seed=1)
    b = model_factory("b", seed=2)
    # Corrupt model b's architecture.
    config_b = b / "config.json"
    import json

    data = json.loads(config_b.read_text())
    data["model_type"] = "gpt2"
    config_b.write_text(json.dumps(data))

    checkpoints = [open_checkpoint(a), open_checkpoint(b)]
    try:
        report, _ = validate_compatibility(checkpoints, CompatibilityConfig())
        with pytest.raises(CompatibilityError):
            report.raise_if_incompatible(CompatibilityMode.STRICT)
    finally:
        for ckpt in checkpoints:
            ckpt.close()
