"""SQLite-backed content-addressed cache for move explanations."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chess_ml.explanation.models import PROMPT_VERSION
from chess_ml.profile.db import default_db_path

KEY_VERSION = "explanation-cache-key.v1"


@dataclass(frozen=True)
class CachedExplanation:
    """A successful cached explanation."""

    text: str
    provider: str
    model: str
    response_json: dict[str, Any]


class ExplanationCache:
    """Small SQLite cache keyed by canonical prompt facts."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_db_path()
        self._ensure_schema()

    def get(self, cache_key: str) -> CachedExplanation | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT text, provider, model, response_json
                FROM explanation_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        response_json = json.loads(str(row["response_json"]))
        if not isinstance(response_json, dict):
            response_json = {}
        return CachedExplanation(
            text=str(row["text"]),
            provider=str(row["provider"]),
            model=str(row["model"]),
            response_json=response_json,
        )

    def set(
        self,
        *,
        cache_key: str,
        provider: str,
        model: str,
        text: str,
        request_json: dict[str, Any],
        response_json: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO explanation_cache (
                    cache_key,
                    prompt_version,
                    provider,
                    model,
                    text,
                    request_json,
                    response_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    PROMPT_VERSION,
                    provider,
                    model,
                    text,
                    _canonical_json(request_json),
                    _canonical_json(response_json),
                ),
            )
            connection.commit()

    def _ensure_schema(self) -> None:
        if self.path.parent != Path("."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS explanation_cache (
                    cache_key TEXT PRIMARY KEY,
                    prompt_version TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    text TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection


def cache_key_for_facts(facts: dict[str, Any]) -> str:
    payload = {
        "key_version": KEY_VERSION,
        "prompt_version": PROMPT_VERSION,
        "facts": facts,
    }
    return f"sha256:{hashlib.sha256(_canonical_json(payload).encode('utf-8')).hexdigest()}"


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
