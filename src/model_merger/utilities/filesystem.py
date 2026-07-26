"""Filesystem safety: atomic output staging, path containment, disk preflight.

Two threats are handled here:

* **Partial output presented as success.**  A crash mid-write must never leave a
  half-written directory that looks complete.  :class:`AtomicDirectory` writes to
  a sibling staging directory and renames it into place only after the caller
  signals success; any failure removes the staging directory.

* **Path traversal.**  Shard names and ancillary file names read from untrusted
  checkpoints must never escape the intended output directory.
  :func:`ensure_within` and :func:`is_safe_relative_member` enforce containment.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import TracebackType

from ..exceptions import InsufficientDiskSpaceError, OutputExistsError

__all__ = [
    "AtomicDirectory",
    "ensure_within",
    "is_safe_relative_member",
    "estimate_free_bytes",
    "check_free_space",
]


def is_safe_relative_member(name: str) -> bool:
    """Return True if ``name`` is a safe *relative* member of a directory.

    Rejects absolute paths, parent references (``..``), and empty names.  Used to
    validate shard filenames and ancillary file names that originate from
    checkpoint metadata (an untrusted source).
    """

    if not name or name in {".", ".."}:
        return False
    pure = Path(name)
    if pure.is_absolute():
        return False
    parts = pure.parts
    return not any(part == ".." for part in parts)


def ensure_within(base: Path, candidate: Path) -> Path:
    """Resolve ``candidate`` and assert it stays inside ``base``.

    Args:
        base: The directory that must contain ``candidate``.
        candidate: A path (possibly relative to ``base``).

    Returns:
        The resolved, contained path.

    Raises:
        ValueError: if the resolved candidate escapes ``base``.
    """

    base_resolved = base.resolve()
    target = candidate if candidate.is_absolute() else base_resolved / candidate
    resolved = target.resolve()
    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise ValueError(f"path {candidate!r} escapes base directory {base!r}")
    return resolved


def estimate_free_bytes(path: str | Path) -> int:
    """Return free bytes on the filesystem that would host ``path``.

    Walks up to the nearest existing ancestor so the check works before the
    output directory is created.
    """

    probe = Path(path).resolve()
    while not probe.exists():
        if probe.parent == probe:
            break
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return usage.free


def check_free_space(path: str | Path, required_bytes: int, *, margin: float = 1.05) -> None:
    """Raise if the estimated required space exceeds free space.

    Args:
        path: Destination whose filesystem is checked.
        required_bytes: Estimated bytes the output will consume.
        margin: Safety multiplier applied to ``required_bytes`` (default 5%).

    Raises:
        InsufficientDiskSpaceError: if free space is insufficient.
    """

    needed = int(required_bytes * margin)
    free = estimate_free_bytes(path)
    if free < needed:
        raise InsufficientDiskSpaceError(
            f"insufficient disk space at {path}: need ~{needed} bytes, only {free} bytes free"
        )


class AtomicDirectory:
    """Context manager for all-or-nothing directory output.

    Usage::

        with AtomicDirectory(final_path, overwrite=False) as staging:
            write_files_into(staging)
        # on clean exit the staging dir is renamed to ``final_path``

    On any exception the staging directory is removed and ``final_path`` is left
    untouched.  The rename is atomic on POSIX when staging and target share a
    filesystem; the staging directory is created as a sibling to guarantee this.
    """

    def __init__(self, final_path: str | Path, *, overwrite: bool = False) -> None:
        self.final_path = Path(final_path)
        self.overwrite = overwrite
        self._staging: Path | None = None
        self._committed = False

    @property
    def staging_path(self) -> Path:
        if self._staging is None:
            raise RuntimeError("AtomicDirectory used outside of its context")
        return self._staging

    def __enter__(self) -> Path:
        if self.final_path.exists():
            if not self.overwrite:
                raise OutputExistsError(
                    f"output path already exists: {self.final_path} "
                    f"(pass overwrite=True / --overwrite to replace it)"
                )
            if not self.final_path.is_dir():
                raise OutputExistsError(
                    f"output path exists and is not a directory: {self.final_path}"
                )
        parent = self.final_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        self._staging = Path(
            tempfile.mkdtemp(prefix=f".{self.final_path.name}.", suffix=".staging", dir=parent)
        )
        return self._staging

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        staging = self._staging
        if staging is None:
            return
        if exc_type is not None:
            shutil.rmtree(staging, ignore_errors=True)
            return
        # Success path: replace the target atomically where possible.
        if self.final_path.exists():
            backup = self.final_path.with_name(self.final_path.name + ".old")
            shutil.rmtree(backup, ignore_errors=True)
            self.final_path.replace(backup)
            try:
                staging.replace(self.final_path)
            except OSError:
                backup.replace(self.final_path)
                shutil.rmtree(staging, ignore_errors=True)
                raise
            shutil.rmtree(backup, ignore_errors=True)
        else:
            staging.replace(self.final_path)
        self._committed = True
