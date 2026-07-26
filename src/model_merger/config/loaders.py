"""Load and preprocess configuration files (YAML or JSON).

Responsibilities:

* Read a ``.yaml``/``.yml``/``.json`` file into a plain dict.
* Recursively expand ``${VAR}`` / ``$VAR`` environment references in string
  values (never in keys), so secrets can live in the environment rather than the
  file.

Path resolution and schema validation happen later, in the Pydantic models, so
this module stays small and side-effect-free apart from reading the file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from ..exceptions import ConfigurationError

__all__ = ["load_config_file", "expand_env"]


def expand_env(value: Any) -> Any:
    """Recursively expand environment variables in string leaves of a structure."""

    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    return value


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Parse a YAML or JSON configuration file into a dict.

    Raises:
        ConfigurationError: if the file is missing, unreadable, not a mapping, or
            fails to parse.
    """

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"configuration file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    try:
        if suffix == ".json":
            data = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(text)
        else:
            # Fall back to YAML, which is a JSON superset.
            data = yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"failed to parse {config_path}: {error}") from error

    if data is None:
        raise ConfigurationError(f"configuration file is empty: {config_path}")
    if not isinstance(data, dict):
        raise ConfigurationError(f"configuration root must be a mapping, got {type(data).__name__}")
    expanded: dict[str, Any] = expand_env(data)
    return expanded
