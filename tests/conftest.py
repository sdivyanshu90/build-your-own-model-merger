"""Shared pytest fixtures: tiny deterministic models and config helpers.

No test touches the network or a large model.  Every fixture builds small
safetensors checkpoints on the fly under ``tmp_path`` so tests are isolated,
deterministic, and fast.  ``src`` and ``scripts`` are added to ``sys.path`` so the
suite runs whether or not the package is installed.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _extra in (_ROOT / "src", _ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import generate_tiny_test_models as gen  # noqa: E402

from model_merger.config.models import (  # noqa: E402
    AlgorithmConfig,
    MergeConfig,
    ModelRef,
    OutputConfig,
)

ModelFactory = Callable[..., Path]


@pytest.fixture
def model_factory(tmp_path: Path) -> ModelFactory:
    """Return a factory that writes a tiny model directory."""

    counter = {"n": 0}

    def make(
        name: str | None = None,
        *,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
        write_pytorch: bool = False,
        with_ancillary: bool = True,
    ) -> Path:
        counter["n"] += 1
        model_name = name or f"model-{counter['n']}"
        directory = tmp_path / model_name
        return gen.write_model(
            directory,
            seed=seed if seed else 100 + counter["n"],
            dtype=dtype,
            write_pytorch=write_pytorch,
            with_ancillary=with_ancillary,
        )

    return make


@pytest.fixture
def two_models(model_factory: ModelFactory) -> tuple[Path, Path]:
    return model_factory("model-a", seed=11), model_factory("model-b", seed=22)


@pytest.fixture
def three_models(model_factory: ModelFactory) -> list[Path]:
    return [
        model_factory("model-a", seed=11),
        model_factory("model-b", seed=22),
        model_factory("model-c", seed=33),
    ]


@pytest.fixture
def make_config() -> Callable[..., MergeConfig]:
    """Return a helper that builds a MergeConfig for given model paths."""

    def build(
        models: Sequence[Path],
        output: Path,
        *,
        algorithm: str = "uniform_soup",
        weights: Sequence[float] | None = None,
        t: float | None = None,
        overwrite: bool = True,
        **algo_kwargs: object,
    ) -> MergeConfig:
        refs = []
        for index, path in enumerate(models):
            weight = weights[index] if weights is not None else None
            refs.append(ModelRef(path=str(path), weight=weight))
        algo = AlgorithmConfig(type=algorithm, t=t, **algo_kwargs)  # type: ignore[arg-type]
        return MergeConfig(
            algorithm=algo,
            models=refs,
            output=OutputConfig(path=str(output), overwrite=overwrite),
        )

    return build
