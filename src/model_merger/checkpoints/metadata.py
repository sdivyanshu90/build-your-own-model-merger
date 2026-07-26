"""Hugging Face directory metadata: config parsing and ancillary-file discovery.

"Ancillary" files are everything that is not model weights: the model config,
generation config, and tokenizer artifacts.  They are copied or reconciled by the
writer according to the ancillary strategy; this module just finds and parses
them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..exceptions import CheckpointError

__all__ = [
    "ANCILLARY_FILENAMES",
    "WEIGHT_FILENAMES",
    "ModelConfigInfo",
    "load_json",
    "discover_ancillary_files",
    "parse_config_info",
]

#: Non-weight files that describe or accompany a model.
ANCILLARY_FILENAMES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "tokenizer.model",
    "added_tokens.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "chat_template.json",
)

#: Weight-file names/index patterns understood by the HF reader.
WEIGHT_FILENAMES: tuple[str, ...] = (
    "model.safetensors.index.json",
    "model.safetensors",
    "pytorch_model.bin.index.json",
    "pytorch_model.bin",
)


def load_json(path: Path) -> dict[str, Any]:
    """Parse a JSON file into a dict, raising CheckpointError on failure."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"failed to read JSON {path}: {error}") from error
    if not isinstance(data, dict):
        raise CheckpointError(f"expected a JSON object in {path}")
    return data


@dataclass(frozen=True)
class ModelConfigInfo:
    """Salient fields extracted from a Hugging Face ``config.json``."""

    model_type: str | None
    architectures: tuple[str, ...]
    vocab_size: int | None
    hidden_size: int | None
    tie_word_embeddings: bool | None
    is_quantized: bool
    transformers_version: str | None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def parse_config_info(config: dict[str, Any]) -> ModelConfigInfo:
    """Extract a :class:`ModelConfigInfo` from a parsed config dict."""

    architectures = config.get("architectures") or []
    return ModelConfigInfo(
        model_type=config.get("model_type"),
        architectures=tuple(str(name) for name in architectures),
        vocab_size=config.get("vocab_size"),
        hidden_size=config.get("hidden_size"),
        tie_word_embeddings=config.get("tie_word_embeddings"),
        is_quantized="quantization_config" in config,
        transformers_version=config.get("transformers_version"),
        raw=config,
    )


def discover_ancillary_files(directory: Path) -> list[Path]:
    """Return the ancillary files present in ``directory`` (sorted)."""

    present = [directory / name for name in ANCILLARY_FILENAMES]
    return sorted(path for path in present if path.is_file())
