"""Configuration validation helpers: size parsing, dtype names, path resolution.

These are separated from the Pydantic models so they can be unit-tested directly
and reused by loaders.  All raise :class:`ConfigurationError` with actionable
messages (never a bare ``ValueError`` that would surface as an internal error).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from ..exceptions import ConfigurationError
from ..types import DTYPE_BY_NAME, OutputDtypePolicy

__all__ = [
    "parse_size",
    "validate_compute_dtype_name",
    "validate_output_dtype_name",
    "resolve_path",
]

#: Multipliers for human-readable size strings.  Decimal (KB/MB/GB/TB) use 1000;
#: binary (KiB/MiB/GiB/TiB) use 1024.  A bare number is bytes.
_SIZE_UNITS: dict[str, int] = {
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}

_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*$")


def parse_size(value: str | int) -> int:
    """Convert a human-readable size to an integer number of bytes.

    Examples: ``"5GB" -> 5_000_000_000``, ``"512MiB" -> 536_870_912``,
    ``1048576 -> 1048576``.

    Raises:
        ConfigurationError: on an unparseable string, unknown unit, or a
            non-positive result.
    """

    if isinstance(value, bool):  # bool is an int subclass; reject explicitly.
        raise ConfigurationError(f"invalid size value: {value!r}")
    if isinstance(value, int):
        result = value
    else:
        match = _SIZE_RE.match(value)
        if match is None:
            raise ConfigurationError(
                f"cannot parse size {value!r} (e.g. '5GB', '512MiB', '1000000')"
            )
        number, unit = match.group(1), match.group(2).lower()
        unit = unit or "b"
        if unit not in _SIZE_UNITS:
            raise ConfigurationError(
                f"unknown size unit {unit!r} in {value!r}; "
                f"use one of: {', '.join(sorted(_SIZE_UNITS))}"
            )
        result = int(float(number) * _SIZE_UNITS[unit])
    if result <= 0:
        raise ConfigurationError(f"size must be positive, got {result} bytes")
    return result


def validate_compute_dtype_name(name: str) -> str:
    """Validate a compute dtype name, returning it unchanged."""

    if name.lower() not in DTYPE_BY_NAME:
        raise ConfigurationError(
            f"unknown compute dtype {name!r}; valid: {', '.join(sorted(DTYPE_BY_NAME))}"
        )
    return name.lower()


def validate_output_dtype_name(name: str) -> str:
    """Validate an output dtype policy name (``preserve``/``highest``/dtype)."""

    valid = {member.value for member in OutputDtypePolicy}
    if name.lower() not in valid:
        raise ConfigurationError(
            f"unknown output dtype policy {name!r}; valid: {', '.join(sorted(valid))}"
        )
    return name.lower()


def resolve_path(base_dir: Path | None, raw: str) -> Path:
    """Resolve a configured path: expand ``~`` and ``$VARS``, apply base dir.

    Relative paths are resolved against ``base_dir`` (the directory of the config
    file) so a config is portable regardless of the working directory.  Absolute
    paths are used as-is.
    """

    candidate = Path(os.path.expandvars(raw)).expanduser()
    if candidate.is_absolute() or base_dir is None:
        return candidate
    return (base_dir / candidate).resolve()
