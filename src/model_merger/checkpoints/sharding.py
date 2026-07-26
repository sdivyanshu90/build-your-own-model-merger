"""Deterministic shard planning for safetensors output.

Given the planned output tensors (key + byte size, in a fixed order) and a maximum
shard size, group keys into shards so no shard exceeds the limit -- except a single
tensor larger than the limit, which occupies its own (over-limit) shard because it
cannot be split.  Grouping walks keys in their given order, so shards are
contiguous runs of keys; this lets the writer buffer just one shard at a time.

Shard file names follow the Hugging Face convention
``model-00001-of-00003.safetensors`` when sharded, or ``model.safetensors`` for a
single shard.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = ["ShardSpec", "ShardPlan", "plan_shards"]


@dataclass(frozen=True)
class ShardSpec:
    """One output shard: its filename and the keys it contains."""

    filename: str
    keys: tuple[str, ...]
    num_bytes: int


@dataclass(frozen=True)
class ShardPlan:
    """The complete shard layout for an output checkpoint."""

    shards: tuple[ShardSpec, ...]
    total_bytes: int
    extension: str = ".safetensors"
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def is_sharded(self) -> bool:
        return len(self.shards) > 1

    @property
    def index_filename(self) -> str:
        return f"model{self.extension}.index.json"

    def weight_map(self) -> dict[str, str]:
        return {key: shard.filename for shard in self.shards for key in shard.keys}

    def index_dict(self) -> dict[str, object]:
        """Return the shard-index document (HF-compatible)."""

        return {
            "metadata": {"total_size": self.total_bytes},
            "weight_map": self.weight_map(),
        }


def plan_shards(
    entries: Sequence[tuple[str, int]],
    *,
    max_shard_bytes: int,
    base_name: str = "model",
    extension: str = ".safetensors",
) -> ShardPlan:
    """Group ``(key, num_bytes)`` entries into shards under ``max_shard_bytes``.

    Args:
        entries: Output tensors in write order.
        max_shard_bytes: Soft cap per shard (a single oversized tensor still gets
            its own shard).
        base_name: Filename stem.
        extension: Filename extension (``.safetensors``).

    Returns:
        A deterministic :class:`ShardPlan`.

    Raises:
        ValueError: on empty input or a non-positive shard cap.
    """

    if not entries:
        raise ValueError("cannot plan shards for an empty checkpoint")
    if max_shard_bytes <= 0:
        raise ValueError(f"max_shard_bytes must be positive, got {max_shard_bytes}")

    groups: list[list[str]] = []
    group_bytes: list[int] = []
    current: list[str] = []
    current_bytes = 0
    for key, size in entries:
        if current and current_bytes + size > max_shard_bytes:
            groups.append(current)
            group_bytes.append(current_bytes)
            current = []
            current_bytes = 0
        current.append(key)
        current_bytes += size
    groups.append(current)
    group_bytes.append(current_bytes)

    total_shards = len(groups)
    total_bytes = sum(group_bytes)
    shards: list[ShardSpec] = []
    for index, (keys, size) in enumerate(zip(groups, group_bytes, strict=True), start=1):
        if total_shards == 1:
            filename = f"{base_name}{extension}"
        else:
            filename = f"{base_name}-{index:05d}-of-{total_shards:05d}{extension}"
        shards.append(ShardSpec(filename=filename, keys=tuple(keys), num_bytes=size))

    return ShardPlan(shards=tuple(shards), total_bytes=total_bytes, extension=extension)
