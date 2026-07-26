"""Enable ``python -m model_merger`` as an alias for the CLI entry point."""

from __future__ import annotations

from .cli import app

if __name__ == "__main__":
    app()
