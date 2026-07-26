"""Regression tests: determinism and reference-value stability.

Rather than hard-coding opaque golden numbers, we recompute the expected result
from the source tensors with plain torch ops and assert equality within an
explicit tolerance.  Tolerances:

* uniform/weighted averaging is exact in float32 up to fp rounding, so ``atol``
  is ``1e-6``;
* SLERP involves trig (arccos/sin) whose float32 error is a few ULPs, so its
  ``atol`` is ``1e-5``.

Determinism is checked directly: two independent runs must produce byte-identical
outputs (equal combined hashes).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from model_merger import merge_models
from model_merger.algorithms.numerical import slerp_vectors
from model_merger.checkpoints import open_checkpoint
from model_merger.config.models import AlgorithmConfig, MergeConfig, ModelRef, OutputConfig

pytestmark = pytest.mark.integration

_KEY = "model.layers.0.self_attn.q_proj.weight"


def _config(models, out, **kw) -> MergeConfig:
    algo = AlgorithmConfig(type=kw.pop("algorithm", "uniform_soup"), **kw)
    return MergeConfig(
        algorithm=algo,
        models=[ModelRef(path=str(m)) for m in models],
        output=OutputConfig(path=str(out), overwrite=True),
    )


def test_uniform_output_matches_reference(three_models, tmp_path: Path) -> None:
    out = tmp_path / "out"
    merge_models(_config(three_models, out))
    sources = [open_checkpoint(p) for p in three_models]
    with open_checkpoint(out) as merged:
        reference = torch.stack([s.get_tensor(_KEY) for s in sources]).mean(dim=0)
        assert torch.allclose(merged.get_tensor(_KEY), reference, atol=1e-6)
    for s in sources:
        s.close()


def test_slerp_output_matches_reference(two_models, tmp_path: Path) -> None:
    out = tmp_path / "out"
    merge_models(_config(two_models, out, algorithm="slerp", t=0.5))
    a, b = (open_checkpoint(p) for p in two_models)
    with open_checkpoint(out) as merged:
        ta = a.get_tensor(_KEY).to(torch.float32)
        tb = b.get_tensor(_KEY).to(torch.float32)
        reference = slerp_vectors(ta.reshape(-1), tb.reshape(-1), 0.5).reshape(ta.shape)
        assert torch.allclose(merged.get_tensor(_KEY), reference, atol=1e-5)
    a.close()
    b.close()


def test_merge_is_deterministic(three_models, tmp_path: Path) -> None:
    # Reproducibility is defined at the level of tensor *content* (see
    # docs/reproducibility.md): safetensors' __metadata__ key ordering is not
    # byte-stable across processes, so the content hash -- not the raw file
    # hash -- is the reproducibility fingerprint.
    report1 = merge_models(_config(three_models, tmp_path / "out1"))
    report2 = merge_models(_config(three_models, tmp_path / "out2"))
    assert report1.output_hash == report2.output_hash

    # Tensor content is bit-identical between runs.
    merged1 = open_checkpoint(tmp_path / "out1")
    merged2 = open_checkpoint(tmp_path / "out2")
    try:
        assert merged1.keys() == merged2.keys()
        for key in merged1.keys():
            assert torch.equal(merged1.get_tensor(key), merged2.get_tensor(key))
    finally:
        merged1.close()
        merged2.close()
