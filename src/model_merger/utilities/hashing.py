"""Deterministic content hashing for reproducibility and verification.

Hashes are used for two purposes:

* **Input provenance** -- record a stable fingerprint of each source model in the
  merge report so a run can be traced back to its exact inputs.
* **Output verification** -- confirm the bytes written to disk match the bytes the
  report claims were written.

File hashing is streamed in fixed-size chunks so hashing a multi-gigabyte shard
never loads it fully into memory.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

__all__ = ["hash_file", "hash_tensor", "hash_state_dict", "CHUNK_SIZE"]

#: Streaming read size for file hashing (1 MiB).
CHUNK_SIZE = 1 << 20


def hash_file(path: str | Path, *, algorithm: str = "sha256") -> str:
    """Return the hex digest of a file, read in bounded chunks.

    Args:
        path: File to hash.
        algorithm: Any name accepted by :func:`hashlib.new`.

    Returns:
        Lower-case hex digest string.
    """

    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tensor(tensor: torch.Tensor, *, algorithm: str = "sha256") -> str:
    """Return a stable hex digest of a tensor's contents.

    The tensor is moved to CPU and made contiguous first so the digest depends
    only on values and dtype, not on device or stride layout.
    """

    cpu = tensor.detach().to("cpu").contiguous()
    digest = hashlib.new(algorithm)
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(str(tuple(cpu.shape)).encode("utf-8"))
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def hash_state_dict(state_dict: dict[str, torch.Tensor], *, algorithm: str = "sha256") -> str:
    """Return an order-independent digest over a whole state dict.

    Keys are hashed in sorted order so the result does not depend on iteration
    order.  Each entry contributes its key and its per-tensor digest.
    """

    digest = hashlib.new(algorithm)
    for key in sorted(state_dict):
        digest.update(key.encode("utf-8"))
        digest.update(hash_tensor(state_dict[key], algorithm=algorithm).encode("utf-8"))
    return digest.hexdigest()
