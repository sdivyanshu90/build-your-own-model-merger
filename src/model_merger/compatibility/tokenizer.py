"""Tokenizer compatibility across Hugging Face directories.

Two models with mismatched tokenizers produce a merged model whose embeddings no
longer correspond to a single, coherent vocabulary -- a subtle failure that is
easy to miss.  We compare the tokenizer-defining files by content hash; a
mismatch is an error when ``require_matching_tokenizer`` is set.

Only Hugging Face directory checkpoints carry tokenizer files; other formats are
skipped with an informational note.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..checkpoints.huggingface_checkpoint import HuggingFaceCheckpoint
from ..types import Severity
from ..utilities.hashing import hash_file
from .report import CompatibilityReport

__all__ = ["analyze_tokenizer", "TOKENIZER_FILES"]

#: Files whose contents define tokenizer behavior.
TOKENIZER_FILES: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "tokenizer.model",
    "added_tokens.json",
)


def _tokenizer_fingerprint(directory: Path) -> dict[str, str]:
    fingerprint: dict[str, str] = {}
    for name in TOKENIZER_FILES:
        path = directory / name
        if path.is_file():
            fingerprint[name] = hash_file(path)
    return fingerprint


def analyze_tokenizer(
    checkpoints: Sequence[object],
    *,
    require_matching_tokenizer: bool = True,
) -> CompatibilityReport:
    """Return tokenizer-compatibility findings across ``checkpoints``."""

    report = CompatibilityReport()
    hf_dirs = [ckpt.path for ckpt in checkpoints if isinstance(ckpt, HuggingFaceCheckpoint)]
    if len(hf_dirs) < 2:
        report.add(
            Severity.INFO,
            "tokenizer.not_comparable",
            "fewer than two Hugging Face directories; tokenizer not compared",
        )
        return report

    fingerprints = [_tokenizer_fingerprint(directory) for directory in hf_dirs]
    reference = fingerprints[0]
    severity = Severity.ERROR if require_matching_tokenizer else Severity.WARNING

    for index, fingerprint in enumerate(fingerprints[1:], start=1):
        shared = set(reference) & set(fingerprint)
        for name in sorted(shared):
            if reference[name] != fingerprint[name]:
                report.add(
                    severity,
                    "tokenizer.file_mismatch",
                    f"tokenizer file {name!r} differs between model 0 and model {index}",
                )
        only_ref = set(reference) - set(fingerprint)
        only_other = set(fingerprint) - set(reference)
        for name in sorted(only_ref | only_other):
            report.add(
                Severity.WARNING,
                "tokenizer.file_presence",
                f"tokenizer file {name!r} is present in only one of model 0 / model {index}",
            )
    return report
