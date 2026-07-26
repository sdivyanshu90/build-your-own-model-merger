"""Security tests for path containment and atomic output."""

from __future__ import annotations

from pathlib import Path

import pytest

from model_merger.checkpoints.writer import copy_ancillary_file
from model_merger.exceptions import MergeExecutionError, OutputExistsError
from model_merger.utilities.filesystem import (
    AtomicDirectory,
    ensure_within,
    is_safe_relative_member,
)

pytestmark = pytest.mark.security


@pytest.mark.parametrize("name", ["../escape", "/abs/path", "..", "", "a/../../b"])
def test_unsafe_relative_names_rejected(name: str) -> None:
    assert not is_safe_relative_member(name)


@pytest.mark.parametrize("name", ["model.safetensors", "sub/file.json", "a.bin"])
def test_safe_relative_names_allowed(name: str) -> None:
    assert is_safe_relative_member(name)


def test_ensure_within_allows_child(tmp_path: Path) -> None:
    resolved = ensure_within(tmp_path, Path("child/file"))
    assert str(resolved).startswith(str(tmp_path.resolve()))


def test_ensure_within_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ensure_within(tmp_path, Path("../outside"))


def test_copy_ancillary_rejects_unsafe_name(tmp_path: Path) -> None:
    source = tmp_path / "src.txt"
    source.write_text("x")
    with pytest.raises(MergeExecutionError):
        copy_ancillary_file(source, tmp_path, name="../evil.txt")


def test_atomic_directory_commits_on_success(tmp_path: Path) -> None:
    target = tmp_path / "out"
    with AtomicDirectory(target) as staging:
        (staging / "file.txt").write_text("hello")
    assert (target / "file.txt").read_text() == "hello"


def test_atomic_directory_no_overwrite_by_default(tmp_path: Path) -> None:
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(OutputExistsError):
        with AtomicDirectory(target):
            pass


def test_atomic_directory_rollback_on_error(tmp_path: Path) -> None:
    target = tmp_path / "out"
    target.mkdir()
    (target / "original.txt").write_text("keep")
    with pytest.raises(RuntimeError):
        with AtomicDirectory(target, overwrite=True) as staging:
            (staging / "partial.txt").write_text("half")
            raise RuntimeError("boom")
    # Original content is untouched; no partial output leaked in.
    assert (target / "original.txt").read_text() == "keep"
    assert not (target / "partial.txt").exists()
    # No staging directories left behind.
    assert not any(p.name.startswith(".out.") for p in tmp_path.iterdir())
