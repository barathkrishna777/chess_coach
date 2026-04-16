"""SQLite profile store and dashboard aggregation."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
    fen_before: str | None = None
    best_move_uci: str | None = None
    best_move_san: str | None = None
    pv_json: str | None = None
    explanation_text: str | None = None
    explanation_status: str | None = None
    evidence_json: str | None = None


@dataclass(frozen=True)
class ProfileGameReview:
    """The profile-relevant facts from one completed review."""

    game_id: str
    players: ProfilePlayers
    result: GameResult
    source: GameSource
    ply_count: int
    motif_occurrences: tuple[ProfileMotifOccurrence, ...]
    eco_code: str | None = None
    opening_name: str | None = None


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
    opening: ProfileOpeningTag | None


@dataclass(frozen=True)
class ProfileOpeningTag:
    """Opening metadata attached to a reviewed game."""

    eco: str
    name: str


@dataclass(frozen=True)
class OpeningTopMotif:
    """Most frequent motif for one opening aggregate."""

    id: str
    label: str
    count: int


@dataclass(frozen=True)
class OpeningAggregate:
    """Aggregate profile facts for one detected opening."""

    eco: str
    name: str
    games: int
    avg_loss_cp: float
    top_motif: OpeningTopMotif | None


@dataclass(frozen=True)
class ProfileDashboard:
    """Aggregated dashboard payload for the single local profile."""

    totals: ProfileTotals
    motifs: tuple[MotifAggregate, ...]
    phase_breakdown: tuple[PhaseAggregate, ...]
    openings: tuple[OpeningAggregate, ...]
    recent_games: tuple[RecentProfileGame, ...]


@dataclass(frozen=True)
class DrillMove:
    """A move reference revealed after a drill attempt."""

    uci: str
    san: str


@dataclass(frozen=True)
class DrillContext:
    """Stored review context for one personal drill."""

    motif_label: str
    phase: GamePhase
    loss_cp: int | None
    score_cp: int | None
    played_move: DrillMove
    pv: tuple[DrillMove, ...]
    explanation_text: str | None
    explanation_status: str | None
    evidence: dict[str, object] | None


@dataclass(frozen=True)
class DrillPosition:
    """One trainable position from the user's reviewed games."""

    game_id: str
    ply: int
    move_number: int
    side: Side
    motif: str
    motif_label: str
    fen: str
    hint_text: str
    context: DrillContext


@dataclass(frozen=True)
class DrillResult:
    """A recorded drill attempt plus the gated answer."""

    correct: bool
    attempted_uci: str
    best_move: DrillMove
    next_due_at: str
    context: DrillContext


@dataclass(frozen=True)
class DrillTotals:
    """Top-line drill progress summary."""

    trainable_positions: int
    due_positions: int
    attempts: int
    correct_attempts: int


@dataclass(frozen=True)
class DrillMotifStats:
    """Drill progress summary for one motif."""

    motif: str
    motif_label: str
    trainable_positions: int
    due_positions: int
    attempts: int
    correct_attempts: int


@dataclass(frozen=True)
class DrillStats:
    """Aggregated drill progress payload."""

    totals: DrillTotals
    motifs: tuple[DrillMotifStats, ...]


class DrillNotFoundError(LookupError):
    """Raised when a submitted drill id no longer maps to a trainable row."""


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
                    eco_code,
                    opening_name,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    white_name = excluded.white_name,
                    white_elo = excluded.white_elo,
                    black_name = excluded.black_name,
                    black_elo = excluded.black_elo,
                    result = excluded.result,
                    source = excluded.source,
                    ply_count = excluded.ply_count,
                    eco_code = excluded.eco_code,
                    opening_name = excluded.opening_name,
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
                    review.eco_code,
                    review.opening_name,
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
                    score_cp,
                    fen_before,
                    best_move_uci,
                    best_move_san,
                    pv_json,
                    explanation_text,
                    explanation_status,
                    evidence_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        occurrence.fen_before,
                        occurrence.best_move_uci,
                        occurrence.best_move_san,
                        occurrence.pv_json,
                        occurrence.explanation_text,
                        occurrence.explanation_status,
                        occurrence.evidence_json,
                    )
                    for occurrence in review.motif_occurrences
                ],
            )
            connection.commit()

    def next_drill(
        self,
        motif: str | None = None,
        *,
        now: datetime | None = None,
    ) -> DrillPosition | None:
        """Return the next due personal drill position, if one exists."""

        timestamp = _timestamp(now)
        motif_filter = _motif_filter(motif)
        with self._connect() as connection:
            row = connection.execute(
                """
                WITH latest_attempt_ids AS (
                    SELECT MAX(id) AS id
                    FROM drill_attempts
                    GROUP BY game_id, ply, motif
                ),
                attempt_state AS (
                    SELECT attempts.game_id, attempts.ply, attempts.motif, attempts.next_due_at
                    FROM drill_attempts AS attempts
                    INNER JOIN latest_attempt_ids AS latest
                        ON attempts.id = latest.id
                )
                SELECT
                    occurrences.*,
                    games.created_at AS game_created_at,
                    games.updated_at AS game_updated_at,
                    attempt_state.next_due_at AS latest_due_at
                FROM profile_motif_occurrences AS occurrences
                INNER JOIN profile_games AS games
                    ON occurrences.game_id = games.game_id
                LEFT JOIN attempt_state
                    ON occurrences.game_id = attempt_state.game_id
                    AND occurrences.ply = attempt_state.ply
                    AND occurrences.motif_id = attempt_state.motif
                WHERE occurrences.fen_before IS NOT NULL
                    AND occurrences.best_move_uci IS NOT NULL
                    AND (? IS NULL OR occurrences.motif_id = ?)
                    AND (
                        attempt_state.next_due_at IS NULL
                        OR attempt_state.next_due_at <= ?
                    )
                ORDER BY
                    CASE WHEN attempt_state.next_due_at IS NULL THEN 0 ELSE 1 END,
                    COALESCE(attempt_state.next_due_at, ''),
                    games.updated_at ASC,
                    games.created_at ASC,
                    occurrences.game_id ASC,
                    occurrences.ply ASC,
                    occurrences.motif_id ASC
                LIMIT 1
                """,
                (motif_filter, motif_filter, timestamp),
            ).fetchone()

        if row is None:
            return None
        return _drill_position(row)

    def record_drill_attempt(
        self,
        *,
        game_id: str,
        ply: int,
        motif: str,
        attempted_uci: str,
        now: datetime | None = None,
    ) -> DrillResult:
        """Persist one drill attempt and return the gated answer/context."""

        timestamp = _timestamp(now)
        attempted_at = _datetime_from_timestamp(timestamp)
        with self._connect() as connection:
            connection.execute("BEGIN")
            row = connection.execute(
                """
                SELECT *
                FROM profile_motif_occurrences
                WHERE game_id = ?
                    AND ply = ?
                    AND motif_id = ?
                    AND fen_before IS NOT NULL
                    AND best_move_uci IS NOT NULL
                LIMIT 1
                """,
                (game_id, ply, motif),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise DrillNotFoundError("That drill is no longer available.")

            best_move_uci = str(row["best_move_uci"])
            correct = attempted_uci == best_move_uci
            next_due_at = _next_due_at(
                attempted_at,
                correct=correct,
                prior_correct_streak=_prior_correct_streak(connection, game_id, ply, motif),
            )
            connection.execute(
                """
                INSERT INTO drill_attempts (
                    game_id,
                    ply,
                    motif,
                    attempted_uci,
                    correct,
                    attempted_at,
                    next_due_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    ply,
                    motif,
                    attempted_uci,
                    1 if correct else 0,
                    timestamp,
                    next_due_at,
                ),
            )
            connection.commit()

        context = _drill_context(row)
        return DrillResult(
            correct=correct,
            attempted_uci=attempted_uci,
            best_move=DrillMove(
                uci=best_move_uci,
                san=_optional_str(row["best_move_san"]) or best_move_uci,
            ),
            next_due_at=next_due_at,
            context=context,
        )

    def drill_stats(self, *, now: datetime | None = None) -> DrillStats:
        """Return persisted drill progress summary."""

        timestamp = _timestamp(now)
        with self._connect() as connection:
            trainable_positions = _int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM profile_motif_occurrences
                    WHERE fen_before IS NOT NULL
                        AND best_move_uci IS NOT NULL
                    """
                ).fetchone()["count"]
            )
            due_positions = _int(
                connection.execute(
                    """
                    WITH latest_attempt_ids AS (
                        SELECT MAX(id) AS id
                        FROM drill_attempts
                        GROUP BY game_id, ply, motif
                    ),
                    attempt_state AS (
                        SELECT attempts.game_id,
                               attempts.ply,
                               attempts.motif,
                               attempts.next_due_at
                        FROM drill_attempts AS attempts
                        INNER JOIN latest_attempt_ids AS latest
                            ON attempts.id = latest.id
                    )
                    SELECT COUNT(*) AS count
                    FROM profile_motif_occurrences AS occurrences
                    LEFT JOIN attempt_state
                        ON occurrences.game_id = attempt_state.game_id
                        AND occurrences.ply = attempt_state.ply
                        AND occurrences.motif_id = attempt_state.motif
                    WHERE occurrences.fen_before IS NOT NULL
                        AND occurrences.best_move_uci IS NOT NULL
                        AND (
                            attempt_state.next_due_at IS NULL
                            OR attempt_state.next_due_at <= ?
                        )
                    """,
                    (timestamp,),
                ).fetchone()["count"]
            )
            attempts_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS attempts,
                    COALESCE(SUM(correct), 0) AS correct_attempts
                FROM drill_attempts
                """
            ).fetchone()
            motifs = tuple(
                DrillMotifStats(
                    motif=str(row["motif_id"]),
                    motif_label=str(row["motif_label"]),
                    trainable_positions=_int(row["trainable_positions"]),
                    due_positions=_int(row["due_positions"]),
                    attempts=_int(row["attempts"]),
                    correct_attempts=_int(row["correct_attempts"]),
                )
                for row in connection.execute(
                    """
                    WITH trainable AS (
                        SELECT *
                        FROM profile_motif_occurrences
                        WHERE fen_before IS NOT NULL
                            AND best_move_uci IS NOT NULL
                    ),
                    latest_attempt_ids AS (
                        SELECT MAX(id) AS id
                        FROM drill_attempts
                        GROUP BY game_id, ply, motif
                    ),
                    attempt_state AS (
                        SELECT attempts.game_id,
                               attempts.ply,
                               attempts.motif,
                               attempts.next_due_at
                        FROM drill_attempts AS attempts
                        INNER JOIN latest_attempt_ids AS latest
                            ON attempts.id = latest.id
                    ),
                    attempt_counts AS (
                        SELECT
                            motif,
                            COUNT(*) AS attempts,
                            COALESCE(SUM(correct), 0) AS correct_attempts
                        FROM drill_attempts
                        GROUP BY motif
                    )
                    SELECT
                        trainable.motif_id,
                        trainable.motif_label,
                        COUNT(*) AS trainable_positions,
                        COALESCE(SUM(
                            CASE
                                WHEN attempt_state.next_due_at IS NULL
                                    OR attempt_state.next_due_at <= ?
                                THEN 1
                                ELSE 0
                            END
                        ), 0) AS due_positions,
                        COALESCE(attempt_counts.attempts, 0) AS attempts,
                        COALESCE(attempt_counts.correct_attempts, 0) AS correct_attempts
                    FROM trainable
                    LEFT JOIN attempt_state
                        ON trainable.game_id = attempt_state.game_id
                        AND trainable.ply = attempt_state.ply
                        AND trainable.motif_id = attempt_state.motif
                    LEFT JOIN attempt_counts
                        ON trainable.motif_id = attempt_counts.motif
                    GROUP BY
                        trainable.motif_id,
                        trainable.motif_label,
                        attempt_counts.attempts,
                        attempt_counts.correct_attempts
                    ORDER BY trainable_positions DESC, trainable.motif_label ASC
                    """,
                    (timestamp,),
                )
            )

        return DrillStats(
            totals=DrillTotals(
                trainable_positions=trainable_positions,
                due_positions=due_positions,
                attempts=_int(attempts_row["attempts"]),
                correct_attempts=_int(attempts_row["correct_attempts"]),
            ),
            motifs=motifs,
        )

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

            top_motifs_by_opening = {
                (str(row["eco_code"]), str(row["opening_name"])): OpeningTopMotif(
                    id=str(row["motif_id"]),
                    label=str(row["motif_label"]),
                    count=_int(row["count"]),
                )
                for row in connection.execute(
                    """
                    SELECT eco_code, opening_name, motif_id, motif_label, count
                    FROM (
                        SELECT
                            games.eco_code,
                            games.opening_name,
                            occurrences.motif_id,
                            occurrences.motif_label,
                            COUNT(*) AS count,
                            ROW_NUMBER() OVER (
                                PARTITION BY games.eco_code, games.opening_name
                                ORDER BY COUNT(*) DESC,
                                         occurrences.motif_label ASC,
                                         occurrences.motif_id ASC
                            ) AS rank
                        FROM profile_games AS games
                        INNER JOIN profile_motif_occurrences AS occurrences
                            ON games.game_id = occurrences.game_id
                        WHERE games.eco_code IS NOT NULL
                            AND games.opening_name IS NOT NULL
                        GROUP BY
                            games.eco_code,
                            games.opening_name,
                            occurrences.motif_id,
                            occurrences.motif_label
                    )
                    WHERE rank = 1
                    """
                )
            }
            openings = tuple(
                OpeningAggregate(
                    eco=str(row["eco_code"]),
                    name=str(row["opening_name"]),
                    games=_int(row["games"]),
                    avg_loss_cp=round(float(row["avg_loss_cp"]), 2),
                    top_motif=top_motifs_by_opening.get(
                        (str(row["eco_code"]), str(row["opening_name"]))
                    ),
                )
                for row in connection.execute(
                    """
                    SELECT
                        games.eco_code,
                        games.opening_name,
                        COUNT(DISTINCT games.game_id) AS games,
                        COALESCE(AVG(occurrences.loss_cp), 0.0) AS avg_loss_cp
                    FROM profile_games AS games
                    LEFT JOIN profile_motif_occurrences AS occurrences
                        ON games.game_id = occurrences.game_id
                        AND occurrences.loss_cp IS NOT NULL
                    WHERE games.eco_code IS NOT NULL
                        AND games.opening_name IS NOT NULL
                    GROUP BY games.eco_code, games.opening_name
                    ORDER BY games DESC, avg_loss_cp DESC, games.opening_name ASC
                    """
                )
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
                        games.eco_code,
                        games.opening_name,
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
            openings=openings,
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
                    eco_code TEXT,
                    opening_name TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            _ensure_columns(
                connection,
                "profile_games",
                {
                    "eco_code": "TEXT",
                    "opening_name": "TEXT",
                },
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
                    fen_before TEXT,
                    best_move_uci TEXT,
                    best_move_san TEXT,
                    pv_json TEXT,
                    explanation_text TEXT,
                    explanation_status TEXT,
                    evidence_json TEXT,
                    FOREIGN KEY(game_id) REFERENCES profile_games(game_id) ON DELETE CASCADE
                )
                """
            )
            _ensure_columns(
                connection,
                "profile_motif_occurrences",
                {
                    "fen_before": "TEXT",
                    "best_move_uci": "TEXT",
                    "best_move_san": "TEXT",
                    "pv_json": "TEXT",
                    "explanation_text": "TEXT",
                    "explanation_status": "TEXT",
                    "evidence_json": "TEXT",
                },
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
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_profile_games_opening
                ON profile_games(eco_code, opening_name)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS drill_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    ply INTEGER NOT NULL,
                    motif TEXT NOT NULL,
                    attempted_uci TEXT NOT NULL,
                    correct INTEGER NOT NULL,
                    attempted_at TEXT NOT NULL,
                    next_due_at TEXT NOT NULL,
                    FOREIGN KEY(game_id) REFERENCES profile_games(game_id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_drill_attempts_motif_due
                ON drill_attempts(motif, next_due_at)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_drill_attempts_position
                ON drill_attempts(game_id, ply, motif)
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
        opening=_opening_tag(row["eco_code"], row["opening_name"]),
    )


def _opening_tag(eco_code: object, opening_name: object) -> ProfileOpeningTag | None:
    eco = _optional_str(eco_code)
    name = _optional_str(opening_name)
    if eco is None or name is None:
        return None
    return ProfileOpeningTag(eco=eco, name=name)


def _drill_position(row: sqlite3.Row) -> DrillPosition:
    motif_label = str(row["motif_label"])
    return DrillPosition(
        game_id=str(row["game_id"]),
        ply=_int(row["ply"]),
        move_number=_int(row["move_number"]),
        side=_side(str(row["side"])),
        motif=str(row["motif_id"]),
        motif_label=motif_label,
        fen=str(row["fen_before"]),
        hint_text=f"Find the engine best move for this {motif_label.lower()} position.",
        context=_drill_context(row),
    )


def _drill_context(row: sqlite3.Row) -> DrillContext:
    return DrillContext(
        motif_label=str(row["motif_label"]),
        phase=_phase(str(row["phase"])),
        loss_cp=_optional_int(row["loss_cp"]),
        score_cp=_optional_int(row["score_cp"]),
        played_move=DrillMove(uci=str(row["uci"]), san=str(row["san"])),
        pv=_parse_pv_json(_optional_str(row["pv_json"])),
        explanation_text=_optional_str(row["explanation_text"]),
        explanation_status=_optional_str(row["explanation_status"]),
        evidence=_parse_json_object(_optional_str(row["evidence_json"])),
    )


def _prior_correct_streak(
    connection: sqlite3.Connection,
    game_id: str,
    ply: int,
    motif: str,
) -> int:
    streak = 0
    for row in connection.execute(
        """
        SELECT correct
        FROM drill_attempts
        WHERE game_id = ?
            AND ply = ?
            AND motif = ?
        ORDER BY id DESC
        """,
        (game_id, ply, motif),
    ):
        if _int(row["correct"]) != 1:
            break
        streak += 1
    return streak


def _next_due_at(now: datetime, *, correct: bool, prior_correct_streak: int) -> str:
    if correct:
        correct_streak = prior_correct_streak + 1
        due_at = now + timedelta(days=2**correct_streak)
    else:
        due_at = now + timedelta(hours=1)
    return _timestamp(due_at)


def _datetime_from_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_pv_json(value: str | None) -> tuple[DrillMove, ...]:
    parsed = _parse_json_list(value)
    moves: list[DrillMove] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        uci = item.get("uci")
        san = item.get("san")
        if isinstance(uci, str) and isinstance(san, str):
            moves.append(DrillMove(uci=uci, san=san))
    return tuple(moves)


def _parse_json_list(value: str | None) -> list[object]:
    if value is None:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return cast(list[object], parsed)
    return []


def _parse_json_object(value: str | None) -> dict[str, object] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return cast(dict[str, object], parsed)
    return None


def _ensure_columns(
    connection: sqlite3.Connection,
    table: str,
    columns: dict[str, str],
) -> None:
    existing = {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}
    for column, definition in columns.items():
        if column not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _motif_filter(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped == "any":
        return None
    return stripped


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


def _side(value: str) -> Side:
    if value == "black":
        return "black"
    return "white"


def _phase(value: str) -> GamePhase:
    if value == "opening":
        return "opening"
    if value == "endgame":
        return "endgame"
    return "middlegame"
