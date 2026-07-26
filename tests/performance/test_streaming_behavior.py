"""Verify the merge streams tensor-at-a-time rather than loading whole models.

We instrument ``SafetensorsCheckpoint.get_tensor`` to record the exact order of
tensor loads during a merge.  Two properties prove bounded-memory streaming:

* loads for a given key are *contiguous* across models (all models' copy of key
  ``K`` are loaded together, then released before key ``K+1``), so at most
  ``n_models`` source tensors are live at once; and
* the total number of loads equals ``n_models * n_keys`` -- no redundant or
  eager full-model reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from model_merger import merge_models
from model_merger.checkpoints import safetensors_checkpoint
from model_merger.config.models import AlgorithmConfig, MergeConfig, ModelRef, OutputConfig

pytestmark = pytest.mark.performance


def _config(models, out) -> MergeConfig:
    return MergeConfig(
        algorithm=AlgorithmConfig(type="uniform_soup"),
        models=[ModelRef(path=str(path)) for path in models],
        output=OutputConfig(path=str(out), overwrite=True),
    )


def test_loads_are_grouped_by_key(three_models, tmp_path: Path, monkeypatch) -> None:
    access_log: list[str] = []
    original = safetensors_checkpoint.SafetensorsCheckpoint.get_tensor
    source_dirs = {str(path) for path in three_models}

    def spy(self, key):  # type: ignore[no-untyped-def]
        # Only record reads from the *source* models, not the post-write
        # verification re-read of the output.
        if any(str(self.path).startswith(directory) for directory in source_dirs):
            access_log.append(key)
        return original(self, key)

    monkeypatch.setattr(safetensors_checkpoint.SafetensorsCheckpoint, "get_tensor", spy)

    merge_models(_config(three_models, tmp_path / "out"))

    # Every maximal run of identical keys should span at most n_models entries,
    # and each distinct key must appear as exactly one contiguous run.
    seen_runs: dict[str, int] = {}
    index = 0
    while index < len(access_log):
        key = access_log[index]
        run = 0
        while index < len(access_log) and access_log[index] == key:
            run += 1
            index += 1
        seen_runs[key] = seen_runs.get(key, 0) + 1
        assert run <= len(three_models), f"key {key} loaded {run} times in one run"

    # No key is revisited in a separate run (contiguity => bounded live set).
    assert all(count == 1 for count in seen_runs.values())


def test_total_loads_bounded(three_models, tmp_path: Path, monkeypatch) -> None:
    counter = {"n": 0}
    original = safetensors_checkpoint.SafetensorsCheckpoint.get_tensor
    source_dirs = {str(path) for path in three_models}

    def spy(self, key):  # type: ignore[no-untyped-def]
        if any(str(self.path).startswith(directory) for directory in source_dirs):
            counter["n"] += 1
        return original(self, key)

    monkeypatch.setattr(safetensors_checkpoint.SafetensorsCheckpoint, "get_tensor", spy)

    merge_models(_config(three_models, tmp_path / "out"))

    from model_merger.checkpoints import open_checkpoint

    with open_checkpoint(three_models[0]) as ckpt:
        n_keys = len(ckpt.keys())
    # Exactly one load per (model, key): no eager whole-model materialization.
    assert counter["n"] == len(three_models) * n_keys
