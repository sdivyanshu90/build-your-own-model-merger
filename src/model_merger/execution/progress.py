"""Minimal, disable-able progress reporting.

Wraps :mod:`rich` progress when enabled and attached to a terminal, and degrades
to a silent no-op otherwise (e.g. under ``--quiet``, ``--json``, or when output is
piped).  The merge core calls :meth:`ProgressReporter.task` and advances it once
per tensor; nothing else depends on rich.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

__all__ = ["ProgressReporter"]


class ProgressReporter:
    """Report progress over a known number of steps, or do nothing if disabled."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    @contextmanager
    def task(self, description: str, total: int) -> Iterator[Callable[[int], None]]:
        """Context manager yielding an ``advance(n=1)`` callable."""

        def _noop(count: int = 1) -> None:
            return None

        if not self.enabled or total <= 0:
            yield _noop
            return
        try:
            from rich.progress import (
                BarColumn,
                Progress,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
            )
        except Exception:  # pragma: no cover - rich always installed at runtime
            yield _noop
            return

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=True,
        )
        with progress:
            task_id = progress.add_task(description, total=total)

            def advance(count: int = 1) -> None:
                progress.advance(task_id, count)

            yield advance
