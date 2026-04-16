"""SQLite profile store and dashboard aggregation."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeAlias, cast

from chess_ml.profile.db import default_db_path

GameResult: TypeAlias = Literal["1-0", "0-1", "1/2-1/2", "*"]
GameSource: TypeAlias = Literal["pgn_upload", "local_play"]
Side: TypeAlias = Literal["white", "black"]
MotifSeverity: TypeAlias = Literal["inaccuracy", "mistake", "blunder"]
GamePhase: TypeAlias = Literal["opening", "middlegame", "endgame"]

PHASES: tuple[GamePhase, ...] = ("opening", "middlegame", "endgame")


@dataclass(frozen=True)
class ProfilePlayer:
    """Player metadata saved with one reviewed game."""

    name: str | None
    elo: int | None


@dataclass(frozen=True)
class ProfilePlayers:
    """White and black player metadata."""

    white: ProfilePlayer
    black: ProfilePlayer


@dataclass(frozen=True)
class ProfileMotifOccurrence:
    """One detected motif attached to one reviewed move."""

    ply: int
    move_number: int
    side: Side
    san: str
    uci: str
    motif_id: str
    motif_label: str
    severity: MotifSeverity
    phase: GamePhase
    loss_cp: int | None
    score_cp: int | None


@dataclass(frozen=True)
class ProfileGameReview:
    """The profile-relevant facts from one completed review."""

    game_id: str
    players: ProfilePlayers
    result: GameResult
    source: GameSource
    ply_count: int
    motif_occurrences: tuple[ProfileMotifOccurrence, ...]


@dataclass(frozen=True)
class ProfileTotals:
    """Top-line local profile totals."""

    games_reviewed: int
    moves_reviewed: int
    flagged_moves: int
    motif_occurrences: int
    motif_rate_per_100_moves: float


@dataclass(frozen=True)
class MotifAggregate:
    """Aggregate count for one motif."""

    id: str
    label: str
    count: int
    rate_per_100_moves: float


@dataclass(frozen=True)
class PhaseAggregate:
    """Aggregate count for one game phase."""

    phase: GamePhase
    count: int
    rate_per_100_moves: float


@dataclass(frozen=True)
class RecentProfileGame:
    """One recently reviewed local game."""

    game_id: str
    players: ProfilePlayers
    result: GameResult
    source: GameSource
    created_at: str
    updated_at: str
    ply_count: int
    flagged_moves: int


@dataclass(frozen=True)
class ProfileDashboard:
    """Aggregated dashboard payload for the single local profile."""

    totals: ProfileTotals
    motifs: tuple[MotifAggregate, ...]
    phase_breakdown: tuple[PhaseAggregate, ...]
    recent_games: tuple[RecentProfileGame, ...]


class ProfileStore:
    """Small SQLite profile store for local reviewed games."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_db_path()
        self._ensure_schema()

    def save_review(
        self,
        review: ProfileGameReview,
        *,
        reviewed_at: datetime | None = None,
    ) -> None:
        """Upsert one analyzed game and replace its motif occurrence rows."""

        timestamp = _timestamp(reviewed_at)
        if self.path.parent != Path("."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("BEGIN")
            connection.execute(
                """
                INSERT INTO profile_games (
                    game_id,
                    white_name,
                    white_elo,
                    black_name,
                    black_elo,
                    result,
                    source,
                    ply_count,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    white_name = excluded.white_name,
                    white_elo = excluded.white_elo,
                    black_name = excluded.black_name,
                    black_elo = excluded.black_elo,
                    result = excluded.result,
                    source = excluded.source,
                    ply_count = excluded.ply_count,
                    updated_at = excluded.updated_at
                """,
                (
                    review.game_id,
                    review.players.white.name,
                    review.players.white.elo,
                    review.players.black.name,
                    review.players.black.elo,
                    review.result,
                    review.source,
                    review.ply_count,
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                "DELETE FROM profile_motif_occurrences WHERE game_id = ?",
                (review.game_id,),
            )
            connection.executemany(
                """
                INSERT INTO profile_motif_occurrences (
                    game_id,
                    ply,
                    move_number,
                    side,
                    san,
                    uci,
                    motif_id,
                    motif_label,
                    severity,
                    phase,
                    loss_cp,
                    score_cp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        review.game_id,
                        occurrence.ply,
                        occurrence.move_number,
                        occurrence.side,
                        occurrence.san,
                        occurrence.uci,
                        occurrence.motif_id,
                        occurrence.motif_label,
                        occurrence.severity,
                        occurrence.phase,
                        occurrence.loss_cp,
                        occurrence.score_cp,
                    )
                    for occurrence in review.motif_occurrences
                ],
            )
            connection.commit()

    def dashboard(self, *, recent_limit: int = 10) -> ProfileDashboard:
        """Return aggregate facts for the local profile dashboard."""

        with self._connect() as connection:
            totals_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS games_reviewed,
                    COALESCE(SUM(ply_count), 0) AS moves_reviewed
                FROM profile_games
                """
            ).fetchone()
            motif_count = _int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM profile_motif_occurrences"
                ).fetchone()["count"]
            )
            flagged_count = _int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM (
                        SELECT game_id, ply
                        FROM profile_motif_occurrences
                        GROUP BY game_id, ply
                    )
                    """
                ).fetchone()["count"]
            )

            games_reviewed = _int(totals_row["games_reviewed"])
            moves_reviewed = _int(totals_row["moves_reviewed"])
            motifs = tuple(
                MotifAggregate(
                    id=str(row["motif_id"]),
                    label=str(row["motif_label"]),
                    count=_int(row["count"]),
                    rate_per_100_moves=_rate(_int(row["count"]), moves_reviewed),
                )
                for row in connection.execute(
                    """
                    SELECT motif_id, motif_label, COUNT(*) AS count
                    FROM profile_motif_occurrences
                    GROUP BY motif_id, motif_label
                    ORDER BY count DESC, motif_label ASC
                    """
                )
            )

            phase_counts = {
                str(row["phase"]): _int(row["count"])
                for row in connection.execute(
                    """
                    SELECT phase, COUNT(*) AS count
                    FROM profile_motif_occurrences
                    GROUP BY phase
                    """
                )
            }
            phase_breakdown = tuple(
                PhaseAggregate(
                    phase=phase,
                    count=phase_counts.get(phase, 0),
                    rate_per_100_moves=_rate(phase_counts.get(phase, 0), moves_reviewed),
                )
                for phase in PHASES
            )

            recent_games = tuple(
                _recent_game(row)
                for row in connection.execute(
                    """
                    SELECT
                        games.game_id,
                        games.white_name,
                        games.white_elo,
                        games.black_name,
                        games.black_elo,
                        games.result,
                        games.source,
                        games.created_at,
                        games.updated_at,
                        games.ply_count,
                        COUNT(flagged.ply) AS flagged_moves
                    FROM profile_games AS games
                    LEFT JOIN (
                        SELECT game_id, ply
                        FROM profile_motif_occurrences
                        GROUP BY game_id, ply
                    ) AS flagged
                        ON games.game_id = flagged.game_id
                    GROUP BY games.game_id
                    ORDER BY games.updated_at DESC, games.created_at DESC, games.game_id ASC
                    LIMIT ?
                    """,
                    (recent_limit,),
                )
            )

        return ProfileDashboard(
            totals=ProfileTotals(
                games_reviewed=games_reviewed,
                moves_reviewed=moves_reviewed,
                flagged_moves=flagged_count,
                motif_occurrences=motif_count,
                motif_rate_per_100_moves=_rate(motif_count, moves_reviewed),
            ),
            motifs=motifs,
            phase_breakdown=phase_breakdown,
            recent_games=recent_games,
        )

    def _ensure_schema(self) -> None:
        if self.path.parent != Path("."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS profile_games (
                    game_id TEXT PRIMARY KEY,
                    white_name TEXT,
                    white_elo INTEGER,
                    black_name TEXT,
                    black_elo INTEGER,
                    result TEXT NOT NULL,
                    source TEXT NOT NULL,
                    ply_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS profile_motif_occurrences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    ply INTEGER NOT NULL,
                    move_number INTEGER NOT NULL,
                    side TEXT NOT NULL,
                    san TEXT NOT NULL,
                    uci TEXT NOT NULL,
                    motif_id TEXT NOT NULL,
                    motif_label TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    loss_cp INTEGER,
                    score_cp INTEGER,
                    FOREIGN KEY(game_id) REFERENCES profile_games(game_id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_profile_motifs_game
                ON profile_motif_occurrences(game_id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_profile_games_updated
                ON profile_games(updated_at)
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _recent_game(row: sqlite3.Row) -> RecentProfileGame:
    return RecentProfileGame(
        game_id=str(row["game_id"]),
        players=ProfilePlayers(
            white=ProfilePlayer(
                name=_optional_str(row["white_name"]),
                elo=_optional_int(row["white_elo"]),
            ),
            black=ProfilePlayer(
                name=_optional_str(row["black_name"]),
                elo=_optional_int(row["black_elo"]),
            ),
        ),
        result=_result(str(row["result"])),
        source=_source(str(row["source"])),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        ply_count=_int(row["ply_count"]),
        flagged_moves=_int(row["flagged_moves"]),
    )


def _timestamp(value: datetime | None) -> str:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).isoformat(timespec="seconds")


def _rate(count: int, moves_reviewed: int) -> float:
    if moves_reviewed <= 0:
        return 0.0
    return round((count / moves_reviewed) * 100, 2)


def _int(value: object) -> int:
    if isinstance(value, int):
        return value
    if value is None:
        return 0
    if isinstance(value, str | bytes | bytearray | float):
        return int(value)
    raise TypeError(f"Expected SQLite integer-compatible value, got {type(value).__name__}.")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _int(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _result(value: str) -> GameResult:
    if value in {"1-0", "0-1", "1/2-1/2", "*"}:
        return cast(GameResult, value)
    return "*"


def _source(value: str) -> GameSource:
    if value == "local_play":
        return "local_play"
    return "pgn_upload"
