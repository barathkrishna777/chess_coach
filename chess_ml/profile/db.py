"""Shared local SQLite path helpers."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DB_PATH = Path("data/chess_ml.sqlite3")


def default_db_path() -> Path:
    """Return the configured local SQLite database path."""

    value = os.environ.get("CHESS_ML_DB_PATH")
    if value is None or not value.strip():
        return DEFAULT_DB_PATH
    return Path(value)
