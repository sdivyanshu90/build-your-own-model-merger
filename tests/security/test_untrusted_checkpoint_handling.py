"""Security tests: pickle trust boundary, malformed metadata, evaluator argv."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from model_merger.checkpoints import PyTorchCheckpoint, SafetensorsCheckpoint
from model_merger.evaluation.command_evaluator import CommandEvaluator
from model_merger.exceptions import CheckpointError, UnsafeCheckpointError

pytestmark = pytest.mark.security


class _CustomPayload:
    """A non-tensor object that weights_only=True must refuse to unpickle."""

    def __init__(self) -> None:
        self.value = 1


def test_unsafe_pickle_rejected_by_default(tmp_path: Path) -> None:
    path = tmp_path / "evil.bin"
    torch.save({"w": torch.zeros(2), "payload": _CustomPayload()}, path)
    with pytest.raises(UnsafeCheckpointError):
        PyTorchCheckpoint(path)


def test_unsafe_pickle_loads_with_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "evil.bin"
    torch.save({"w": torch.ones(2), "payload": _CustomPayload()}, path)
    with PyTorchCheckpoint(path, allow_unsafe=True) as ckpt:
        # Non-tensor entries are dropped; only tensors are exposed.
        assert ckpt.keys() == ["w"]


def test_malformed_shard_index_rejected(tmp_path: Path) -> None:
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {}}))  # no weight_map
    with pytest.raises(CheckpointError, match="weight_map"):
        SafetensorsCheckpoint(index)


def test_shard_index_missing_file_rejected(tmp_path: Path) -> None:
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"w": "missing-shard.safetensors"}}))
    with pytest.raises(CheckpointError, match="missing"):
        SafetensorsCheckpoint(index)


def test_command_evaluator_no_shell_injection(tmp_path: Path) -> None:
    # A path containing shell metacharacters must be passed as one literal argv
    # element -- the program sees the exact string, nothing is executed.
    tricky = tmp_path / "model; echo pwned"
    tricky.mkdir()
    evaluator = CommandEvaluator(
        [
            sys.executable,
            "-c",
            "import sys, json; print(json.dumps({'score': len(sys.argv[1])}))",
            "{model_path}",
        ],
        metric_key="score",
    )
    score = evaluator.evaluate(tricky)
    assert score == len(str(tricky))


def test_command_evaluator_nonzero_exit(tmp_path: Path) -> None:
    from model_merger.exceptions import EvaluationError

    evaluator = CommandEvaluator([sys.executable, "-c", "import sys; sys.exit(3)", "{model_path}"])
    with pytest.raises(EvaluationError):
        evaluator.evaluate(tmp_path)
