"""End-to-end merges of Hugging Face directories, including ancillary handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from model_merger import merge_models
from model_merger.config.models import (
    AlgorithmConfig,
    AncillaryConfig,
    CompatibilityConfig,
    MergeConfig,
    ModelRef,
    OutputConfig,
)
from model_merger.exceptions import MergeExecutionError

pytestmark = pytest.mark.integration


def _config(models, out, **kwargs) -> MergeConfig:
    return MergeConfig(
        algorithm=AlgorithmConfig(type="uniform_soup"),
        models=[ModelRef(path=str(path)) for path in models],
        output=OutputConfig(path=str(out), overwrite=True),
        **kwargs,
    )


def test_ancillary_files_copied(three_models, tmp_path: Path) -> None:
    out = tmp_path / "merged"
    merge_models(_config(three_models, out))
    assert (out / "config.json").is_file()
    assert (out / "tokenizer_config.json").is_file()
    assert (out / "generation_config.json").is_file()


def test_require_equal_ancillary_fails_on_difference(three_models, tmp_path: Path) -> None:
    # Corrupt one model's config so ancillary files differ.
    import json

    config_path = three_models[1] / "generation_config.json"
    data = json.loads(config_path.read_text())
    data["max_length"] = 999
    config_path.write_text(json.dumps(data))

    out = tmp_path / "merged"
    config = _config(
        three_models,
        out,
        ancillary=AncillaryConfig(strategy="require_equal"),
    )
    with pytest.raises(MergeExecutionError, match="ancillary"):
        merge_models(config)


def test_permissive_mode_allows_key_mismatch(two_models, tmp_path: Path) -> None:
    # Remove a tensor from one model by rewriting its safetensors without a key.
    from safetensors import safe_open
    from safetensors.torch import save_file

    path = two_models[0] / "model.safetensors"
    with safe_open(str(path), framework="pt") as handle:
        state = {key: handle.get_tensor(key) for key in handle.keys()}
    state.pop("model.norm.weight")
    save_file(state, str(path), metadata={"format": "pt"})

    out = tmp_path / "merged"
    config = _config(
        two_models,
        out,
        compatibility=CompatibilityConfig(mode="permissive", allow_missing_keys=True),
    )
    report = merge_models(config)
    assert report.verification.passed
