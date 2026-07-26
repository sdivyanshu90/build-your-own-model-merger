"""Logging configuration for :mod:`model_merger`.

The library never configures the root logger on import.  Applications embedding
the API keep full control of logging; only the CLI calls
:func:`configure_logging`.  All library modules obtain a logger via
:func:`get_logger`, which returns a child of the ``model_merger`` logger.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

__all__ = ["get_logger", "configure_logging", "LOGGER_NAME"]

LOGGER_NAME = "model_merger"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced logger.

    Args:
        name: Optional dotted suffix, typically ``__name__``.  The returned
            logger is always a descendant of the ``model_merger`` logger so that
            a single handler configuration governs the whole library.
    """

    if name is None or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    suffix = name.split(".")[-1]
    return logging.getLogger(LOGGER_NAME).getChild(suffix)


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per record for machine-readable logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(
    *,
    level: int | str = logging.INFO,
    json_mode: bool = False,
    quiet: bool = False,
) -> None:
    """Configure the ``model_merger`` logger (idempotent).

    Args:
        level: Logging level for library output.
        json_mode: If true, emit structured JSON lines instead of plain text.
        quiet: If true, raise the effective level to ``ERROR`` regardless of
            ``level`` so only failures are printed.

    Notes:
        Handlers are attached to the ``model_merger`` logger (not root) and
        ``propagate`` is disabled, so embedding applications are unaffected.
        Logs go to ``stderr`` to keep ``stdout`` clean for machine-readable
        command output.
    """

    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    effective = logging.ERROR if quiet else level
    logger.setLevel(effective)
    logger.propagate = False

    handler = logging.StreamHandler(stream=sys.stderr)
    if json_mode:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    handler.setLevel(effective)
    logger.addHandler(handler)
