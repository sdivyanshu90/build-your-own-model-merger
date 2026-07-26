"""Tensor-key pattern helpers: glob translation and layer-index extraction.

Layer rules (see :mod:`model_merger.policies.layer_rules`) match tensor keys by
exact string, glob, regex, suffix, or transformer layer-number range.  The
layer-number range needs to recover the integer layer index from a key such as
``model.layers.20.self_attn.q_proj.weight``.  We use a small set of well-known
patterns rather than guessing, and return ``None`` when no layer index is present
so the rule simply does not match (fail-safe).
"""

from __future__ import annotations

import fnmatch
import re

__all__ = ["glob_to_regex", "extract_layer_index", "LAYER_INDEX_PATTERNS"]

#: Regexes that capture a transformer block index from common key schemes.
LAYER_INDEX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:^|\.)layers\.(\d+)\."),  # LLaMA / Mistral / Qwen: model.layers.N.
    re.compile(r"(?:^|\.)h\.(\d+)\."),  # GPT-2 style: transformer.h.N.
    re.compile(r"(?:^|\.)block\.(\d+)\."),  # T5 style: encoder.block.N.
    re.compile(r"(?:^|\.)layer\.(\d+)\."),  # BERT style: encoder.layer.N.
)


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Compile a glob pattern into an anchored regex.

    Uses :func:`fnmatch.translate` so semantics match shell globbing (``*``,
    ``?``, ``[...]``).  The result is fully anchored.
    """

    return re.compile(fnmatch.translate(pattern))


def extract_layer_index(key: str) -> int | None:
    """Return the transformer layer index encoded in ``key``, or ``None``.

    Tries each known pattern in order and returns the first match.  Returns
    ``None`` for keys without a recognizable layer index (e.g. embeddings, final
    norm) so layer-range rules do not accidentally match them.
    """

    for pattern in LAYER_INDEX_PATTERNS:
        match = pattern.search(key)
        if match is not None:
            return int(match.group(1))
    return None
