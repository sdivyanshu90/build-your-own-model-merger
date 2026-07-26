"""Post-write verification.

A merge is only reported successful after the *written* output passes these
checks: every file re-opens, every tensor loads, floating tensors are finite,
shapes/dtypes match the plan, a shard index (if any) resolves, and any
``config.json`` parses.  This is the "verification" half of the validation-vs-
verification distinction: validation happens on inputs before writing;
verification happens on outputs after writing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from ..checkpoints import open_checkpoint
from ..checkpoints.metadata import load_json
from ..reporting.models import VerificationResult
from ..utilities.hashing import hash_file, hash_tensor

__all__ = ["verify_output", "content_hash"]


def content_hash(path: str | Path, *, allow_unsafe: bool = False) -> str:
    """Return a deterministic, order-independent hash of a checkpoint's tensors.

    Unlike a file hash, this depends only on tensor keys, dtypes, and values --
    not on container-level details such as safetensors' metadata key ordering
    (which is not byte-stable across processes).  It is the reproducibility
    fingerprint recorded in the merge report.  Tensors are hashed one at a time
    (bounded memory).
    """

    checkpoint = open_checkpoint(path, allow_unsafe=allow_unsafe)
    try:
        digest = hashlib.sha256()
        for key in checkpoint.keys():  # noqa: SIM118 - Checkpoint.keys() returns a sorted list
            digest.update(key.encode("utf-8"))
            digest.update(hash_tensor(checkpoint.get_tensor(key)).encode("utf-8"))
        return digest.hexdigest()
    finally:
        checkpoint.close()


def verify_output(
    output_path: str | Path,
    *,
    expected_keys: set[str] | None = None,
    expected_dtypes: dict[str, torch.dtype] | None = None,
    expected_hashes: dict[str, str] | None = None,
    check_finite: bool = True,
    allow_unsafe: bool = False,
) -> VerificationResult:
    """Verify a written checkpoint, returning a structured result.

    All arguments beyond ``output_path`` are optional; when omitted the
    corresponding check is skipped (used by the standalone ``verify`` command,
    which has no plan to compare against).
    """

    path = Path(output_path)
    checks: dict[str, bool] = {}
    messages: list[str] = []

    if not path.exists():
        return VerificationResult(False, {"exists": False}, (f"output does not exist: {path}",))
    checks["exists"] = True

    try:
        checkpoint = open_checkpoint(path, allow_unsafe=allow_unsafe)
    except Exception as error:
        return VerificationResult(False, {"openable": False}, (f"cannot open output: {error}",))
    checks["openable"] = True

    try:
        keys = set(checkpoint.keys())
        if expected_keys is not None:
            missing = expected_keys - keys
            extra = keys - expected_keys
            checks["keys_match"] = not missing and not extra
            if missing:
                messages.append(f"missing keys (e.g. {sorted(missing)[:5]})")
            if extra:
                messages.append(f"unexpected keys (e.g. {sorted(extra)[:5]})")

        all_loadable = True
        all_finite = True
        dtypes_match = True
        for key in keys:
            try:
                tensor = checkpoint.get_tensor(key)
            except Exception as error:
                all_loadable = False
                messages.append(f"cannot load {key!r}: {error}")
                continue
            if check_finite and tensor.dtype.is_floating_point and not torch.isfinite(tensor).all():
                all_finite = False
                messages.append(f"non-finite values in {key!r}")
            if (
                expected_dtypes is not None
                and key in expected_dtypes
                and tensor.dtype != expected_dtypes[key]
            ):
                dtypes_match = False
                messages.append(
                    f"dtype mismatch for {key!r}: {tensor.dtype} != {expected_dtypes[key]}"
                )
        checks["all_loadable"] = all_loadable
        if check_finite:
            checks["all_finite"] = all_finite
        if expected_dtypes is not None:
            checks["dtypes_match"] = dtypes_match
    finally:
        checkpoint.close()

    if path.is_dir():
        config_path = path / "config.json"
        if config_path.is_file():
            try:
                load_json(config_path)
                checks["config_parses"] = True
            except Exception as error:
                checks["config_parses"] = False
                messages.append(f"config.json failed to parse: {error}")

    if expected_hashes is not None:
        hashes_match = True
        for name, expected in expected_hashes.items():
            file_path = path / name if path.is_dir() else path
            if not file_path.is_file() or hash_file(file_path) != expected:
                hashes_match = False
                messages.append(f"hash mismatch or missing file: {name}")
        checks["hashes_match"] = hashes_match

    passed = all(checks.values())
    return VerificationResult(passed=passed, checks=checks, messages=tuple(messages))
