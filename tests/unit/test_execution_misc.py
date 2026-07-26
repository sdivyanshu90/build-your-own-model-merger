"""Coverage for device resolution, progress, logging, hashing, and api helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

from model_merger import inspect_model, plan_merge
from model_merger.config.models import AlgorithmConfig, MergeConfig, ModelRef, OutputConfig
from model_merger.exceptions import MergeExecutionError
from model_merger.execution.device import resolve_device, to_device
from model_merger.execution.progress import ProgressReporter
from model_merger.logging import configure_logging, get_logger
from model_merger.utilities.hashing import hash_file, hash_state_dict, hash_tensor
from model_merger.utilities.reproducibility import capture_environment, seed_everything


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda"}


def test_resolve_device_bad() -> None:
    with pytest.raises(MergeExecutionError):
        resolve_device("tpu")


def test_resolve_cuda_without_cuda() -> None:
    if not torch.cuda.is_available():
        with pytest.raises(MergeExecutionError):
            resolve_device("cuda")


def test_to_device_noop() -> None:
    tensor = torch.zeros(2)
    assert to_device(tensor, torch.device("cpu")) is tensor


def test_progress_disabled_yields_noop() -> None:
    reporter = ProgressReporter(enabled=False)
    with reporter.task("x", 10) as advance:
        advance(1)
        advance()


def test_progress_enabled_runs() -> None:
    reporter = ProgressReporter(enabled=True)
    with reporter.task("merging", 3) as advance:
        for _ in range(3):
            advance(1)


def test_logging_configuration() -> None:
    configure_logging(level=logging.DEBUG, json_mode=True)
    logger = get_logger("model_merger.test")
    assert logger.name.startswith("model_merger")
    configure_logging(quiet=True)


def test_hashing_helpers(tmp_path: Path) -> None:
    file_path = tmp_path / "f.bin"
    file_path.write_bytes(b"hello")
    assert len(hash_file(file_path)) == 64
    assert hash_tensor(torch.zeros(3)) == hash_tensor(torch.zeros(3))
    a = hash_state_dict({"w": torch.zeros(2), "b": torch.ones(2)})
    b = hash_state_dict({"b": torch.ones(2), "w": torch.zeros(2)})
    assert a == b  # order-independent


def test_capture_environment_and_seed() -> None:
    info = capture_environment("9.9.9")
    assert info.tool_version == "9.9.9"
    assert "torch_version" in info.to_dict()
    seed_everything(123)  # must not raise


def test_inspect_and_plan(three_models, tmp_path: Path) -> None:
    summary = inspect_model(three_models[0])
    assert summary["tensor_count"] > 0
    assert "architecture" in summary

    config = MergeConfig(
        algorithm=AlgorithmConfig(type="uniform_soup"),
        models=[ModelRef(path=str(p)) for p in three_models],
        output=OutputConfig(path=str(tmp_path / "out")),
    )
    plan = plan_merge(config)
    assert plan.tensor_count > 0
    assert plan.algorithm == "uniform_soup"
