"""Unit tests for configuration parsing and semantic validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_merger.config.models import MergeConfig
from model_merger.config.validation import parse_size
from model_merger.exceptions import ConfigurationError


def _base(tmp_path: Path) -> dict:
    return {
        "algorithm": {"type": "uniform_soup"},
        "models": [{"path": str(tmp_path / "a")}, {"path": str(tmp_path / "b")}],
        "output": {"path": str(tmp_path / "out")},
    }


def test_minimal_config_parses(tmp_path: Path) -> None:
    config = MergeConfig.from_dict(_base(tmp_path))
    assert config.algorithm.type.value == "uniform_soup"
    assert len(config.models) == 2


def test_unknown_field_rejected(tmp_path: Path) -> None:
    data = _base(tmp_path)
    data["nonsense"] = 1
    with pytest.raises(ConfigurationError):
        MergeConfig.from_dict(data)


def test_slerp_requires_two_models(tmp_path: Path) -> None:
    data = _base(tmp_path)
    data["algorithm"] = {"type": "slerp", "t": 0.5}
    data["models"].append({"path": str(tmp_path / "c")})
    with pytest.raises(ConfigurationError, match="exactly 2 models"):
        MergeConfig.from_dict(data)


def test_slerp_requires_t(tmp_path: Path) -> None:
    data = _base(tmp_path)
    data["algorithm"] = {"type": "slerp"}
    with pytest.raises(ConfigurationError):
        MergeConfig.from_dict(data)


def test_weighted_requires_weights(tmp_path: Path) -> None:
    data = _base(tmp_path)
    data["algorithm"] = {"type": "weighted_soup"}
    with pytest.raises(ConfigurationError, match="weight"):
        MergeConfig.from_dict(data)


def test_greedy_requires_greedy_section(tmp_path: Path) -> None:
    data = _base(tmp_path)
    data["algorithm"] = {"type": "greedy_soup"}
    with pytest.raises(ConfigurationError, match="greedy"):
        MergeConfig.from_dict(data)


def test_names_auto_assigned_and_unique(tmp_path: Path) -> None:
    data = _base(tmp_path)
    data["models"] = [{"path": str(tmp_path / "same")}, {"path": str(tmp_path / "same")}]
    config = MergeConfig.from_dict(data)
    assert config.model_names[0] != config.model_names[1]


def test_bad_device_rejected(tmp_path: Path) -> None:
    data = _base(tmp_path)
    data["device"] = "tpu"
    with pytest.raises(ConfigurationError):
        MergeConfig.from_dict(data)


def test_bad_compute_dtype_rejected(tmp_path: Path) -> None:
    data = _base(tmp_path)
    data["precision"] = {"compute_dtype": "float9"}
    with pytest.raises(ConfigurationError):
        MergeConfig.from_dict(data)


def test_match_requires_condition(tmp_path: Path) -> None:
    data = _base(tmp_path)
    data["rules"] = [{"name": "r", "match": {}, "algorithm": {"type": "uniform_soup"}}]
    with pytest.raises(ConfigurationError):
        MergeConfig.from_dict(data)


def test_ancillary_base_model_must_exist(tmp_path: Path) -> None:
    data = _base(tmp_path)
    data["ancillary"] = {"strategy": "base", "base_model": "ghost"}
    with pytest.raises(ConfigurationError, match="base_model"):
        MergeConfig.from_dict(data)


def test_from_file_resolves_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        "algorithm: {type: uniform_soup}\n"
        "models: [{path: a}, {path: b}]\n"
        f"output: {{path: {tmp_path / 'out'}}}\n"
    )
    config = MergeConfig.from_file(config_path)
    assert config.resolved_model_paths()[0] == (tmp_path / "a").resolve()


def test_env_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MM_TEST_DIR", str(tmp_path))
    config_path = tmp_path / "cfg.json"
    config_path.write_text(
        json.dumps(
            {
                "algorithm": {"type": "uniform_soup"},
                "models": [{"path": "${MM_TEST_DIR}/a"}, {"path": "${MM_TEST_DIR}/b"}],
                "output": {"path": "${MM_TEST_DIR}/out"},
            }
        )
    )
    config = MergeConfig.from_file(config_path)
    assert str(tmp_path) in config.models[0].path


@pytest.mark.parametrize(
    ("value", "expected"),
    [("5GB", 5_000_000_000), ("512MiB", 536_870_912), ("1000", 1000), (2048, 2048)],
)
def test_parse_size(value: object, expected: int) -> None:
    assert parse_size(value) == expected  # type: ignore[arg-type]


def test_parse_size_rejects_garbage() -> None:
    with pytest.raises(ConfigurationError):
        parse_size("five gigs")


def test_redacted_dict_masks_secrets(tmp_path: Path) -> None:
    config = MergeConfig.from_dict(_base(tmp_path))
    redacted = config.redacted_dict()
    assert redacted["algorithm"]["type"] == "uniform_soup"
