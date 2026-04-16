"""Tests for the local profile store and dashboard API."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import chess
from fastapi import FastAPI
from fastapi.testclient import TestClient

from chess_ml.api.games import router as games_router
from chess_ml.api.profile import router as profile_router
from chess_ml.engine.stockfish import CentipawnScore, EngineEvaluation, EngineMove
from chess_ml.profile.store import (
    ProfileGameReview,
    ProfileMotifOccurrence,
    ProfilePlayer,
    ProfilePlayers,
    ProfileStore,
)


def test_profile_store_upserts_without_double_counting(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "profile.sqlite3")
    first_review = _review(
        "game-1",
        ply_count=10,
        motifs=(
            _occurrence(ply=3, motif_id="hanging_piece", label="Hanging piece"),
            _occurrence(ply=3, motif_id="missed_tactic", label="Missed tactic"),
        ),
    )
    replacement_review = _review(
        "game-1",
        ply_count=10,
        motifs=(_occurrence(ply=4, motif_id="allowed_tactic", label="Allowed tactic"),),
    )

    store.save_review(first_review, reviewed_at=datetime(2026, 4, 15, 12, tzinfo=UTC))
    store.save_review(replacement_review, reviewed_at=datetime(2026, 4, 15, 13, tzinfo=UTC))

    dashboard = store.dashboard()

    assert dashboard.totals.games_reviewed == 1
    assert dashboard.totals.moves_reviewed == 10
    assert dashboard.totals.flagged_moves == 1
    assert dashboard.totals.motif_occurrences == 1
    assert dashboard.totals.motif_rate_per_100_moves == 10.0
    assert [(motif.id, motif.count) for motif in dashboard.motifs] == [("allowed_tactic", 1)]
    assert dashboard.recent_games[0].flagged_moves == 1


def test_profile_store_aggregates_rates_and_phases(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "profile.sqlite3")
    store.save_review(
        _review(
            "game-1",
            ply_count=20,
            motifs=(
                _occurrence(ply=2, motif_id="hanging_piece", label="Hanging piece"),
                _occurrence(
                    ply=8,
                    motif_id="opening_inaccuracy",
                    label="Opening inaccuracy",
                    phase="opening",
                ),
                _occurrence(
                    ply=16,
                    motif_id="endgame_slip",
                    label="Endgame slip",
                    phase="endgame",
                ),
            ),
        )
    )

    dashboard = store.dashboard()

    assert dashboard.totals.games_reviewed == 1
    assert dashboard.totals.moves_reviewed == 20
    assert dashboard.totals.flagged_moves == 3
    assert dashboard.totals.motif_occurrences == 3
    assert dashboard.totals.motif_rate_per_100_moves == 15.0
    assert {motif.id: motif.rate_per_100_moves for motif in dashboard.motifs} == {
        "endgame_slip": 5.0,
        "hanging_piece": 5.0,
        "opening_inaccuracy": 5.0,
    }
    assert {phase.phase: phase.count for phase in dashboard.phase_breakdown} == {
        "opening": 2,
        "middlegame": 0,
        "endgame": 1,
    }


def test_profile_store_recent_games_sort_newest_first(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "profile.sqlite3")
    store.save_review(
        _review("older", ply_count=4),
        reviewed_at=datetime(2026, 4, 15, 12, tzinfo=UTC),
    )
    store.save_review(
        _review("newer", ply_count=4),
        reviewed_at=datetime(2026, 4, 15, 13, tzinfo=UTC),
    )

    dashboard = store.dashboard()

    assert [game.game_id for game in dashboard.recent_games] == ["newer", "older"]


def test_profile_store_persists_opening_metadata_idempotently(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "profile.sqlite3")
    store.save_review(
        _review(
            "game-1",
            ply_count=10,
            eco_code="B01",
            opening_name="Scandinavian Defense",
            motifs=(_occurrence(ply=3, motif_id="missed_tactic", label="Missed tactic"),),
        ),
        reviewed_at=datetime(2026, 4, 15, 12, tzinfo=UTC),
    )
    store.save_review(
        _review(
            "game-1",
            ply_count=10,
            eco_code="C20",
            opening_name="Wayward Queen Attack",
            motifs=(_occurrence(ply=4, motif_id="hanging_piece", label="Hanging piece"),),
        ),
        reviewed_at=datetime(2026, 4, 15, 13, tzinfo=UTC),
    )

    dashboard = store.dashboard()

    assert dashboard.totals.games_reviewed == 1
    assert [(opening.eco, opening.name, opening.games) for opening in dashboard.openings] == [
        ("C20", "Wayward Queen Attack", 1)
    ]
    assert dashboard.recent_games[0].opening is not None
    assert dashboard.recent_games[0].opening.eco == "C20"


def test_profile_store_aggregates_openings(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "profile.sqlite3")
    store.save_review(
        _review(
            "game-1",
            ply_count=10,
            eco_code="B01",
            opening_name="Scandinavian Defense",
            motifs=(
                _occurrence(
                    ply=3,
                    motif_id="missed_tactic",
                    label="Missed tactic",
                    loss_cp=300,
                ),
                _occurrence(
                    ply=5,
                    motif_id="missed_tactic",
                    label="Missed tactic",
                    loss_cp=100,
                ),
            ),
        ),
    )
    store.save_review(
        _review(
            "game-2",
            ply_count=8,
            eco_code="B01",
            opening_name="Scandinavian Defense",
            motifs=(_occurrence(ply=3, motif_id="hanging_piece", label="Hanging piece"),),
        ),
    )

    dashboard = store.dashboard()

    assert len(dashboard.openings) == 1
    opening = dashboard.openings[0]
    assert opening.eco == "B01"
    assert opening.name == "Scandinavian Defense"
    assert opening.games == 2
    assert opening.avg_loss_cp == 233.33
    assert opening.top_motif is not None
    assert opening.top_motif.id == "missed_tactic"
    assert opening.top_motif.count == 2


def test_profile_store_reads_existing_games_without_opening_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE profile_games (
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
            CREATE TABLE profile_motif_occurrences (
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
            """,
            (
                "legacy",
                "Ada",
                1500,
                "Turing",
                1600,
                "1-0",
                "pgn_upload",
                4,
                "2026-04-15T12:00:00+00:00",
                "2026-04-15T12:00:00+00:00",
            ),
        )

    dashboard = ProfileStore(db_path).dashboard()

    assert dashboard.totals.games_reviewed == 1
    assert dashboard.openings == ()
    assert dashboard.recent_games[0].opening is None


def test_profile_api_returns_empty_profile(tmp_path: Path) -> None:
    app = FastAPI()
    app.state.profile_store = ProfileStore(tmp_path / "profile.sqlite3")
    app.include_router(profile_router)
    client = TestClient(app)

    response = client.get("/api/profile/me")

    assert response.status_code == 200
    assert response.json() == {
        "schema_version": "profile-dashboard.v1",
        "totals": {
            "games_reviewed": 0,
            "moves_reviewed": 0,
            "flagged_moves": 0,
            "motif_occurrences": 0,
            "motif_rate_per_100_moves": 0.0,
        },
        "motifs": [],
        "phase_breakdown": [
            {"phase": "opening", "count": 0, "rate_per_100_moves": 0.0},
            {"phase": "middlegame", "count": 0, "rate_per_100_moves": 0.0},
            {"phase": "endgame", "count": 0, "rate_per_100_moves": 0.0},
        ],
        "openings": [],
        "recent_games": [],
    }


def test_post_games_writes_profile_rows(tmp_path: Path) -> None:
    app = FastAPI()
    app.state.review_lock = asyncio.Lock()
    app.state.stockfish_pool = _FakeStockfishPool()
    app.state.stockfish_error = ""
    app.state.profile_store = ProfileStore(tmp_path / "profile.sqlite3")
    app.include_router(games_router)
    app.include_router(profile_router)
    client = TestClient(app)

    response = client.post(
        "/api/games",
        json={
            "pgn": """
[Event "Fixture"]
[White "Ada"]
[Black "Turing"]
[WhiteElo "1500"]
[BlackElo "1600"]
[Result "*"]

1. e4 d5 *
""",
        },
    )

    assert response.status_code == 200
    assert response.json()["opening"] == {"eco": "B01", "name": "Scandinavian Defense"}
    profile_response = client.get("/api/profile/me")
    assert profile_response.status_code == 200
    profile = profile_response.json()
    assert profile["totals"]["games_reviewed"] == 1
    assert profile["totals"]["moves_reviewed"] == 2
    assert profile["recent_games"][0]["players"]["white"] == {"name": "Ada", "elo": 1500}
    assert profile["recent_games"][0]["players"]["black"] == {"name": "Turing", "elo": 1600}
    assert profile["recent_games"][0]["source"] == "pgn_upload"
    assert profile["recent_games"][0]["opening"] == {
        "eco": "B01",
        "name": "Scandinavian Defense",
    }
    assert profile["openings"] == [
        {
            "eco": "B01",
            "name": "Scandinavian Defense",
            "games": 1,
            "avg_loss_cp": 0.0,
            "top_motif": None,
        }
    ]


class _FakeStockfishPool:
    started = True
    depth = 1

    async def evaluate(self, fen: str, *, depth: int | None = None) -> EngineEvaluation:
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=False):
            return EngineEvaluation(
                status="terminal",
                depth=None,
                score=CentipawnScore(cp=0),
                best_move=None,
                pv=(),
                nodes=None,
                time_ms=0,
            )

        move = next(iter(board.legal_moves))
        engine_move = EngineMove(uci=move.uci(), san=board.san(move))
        return EngineEvaluation(
            status="ok",
            depth=depth or self.depth,
            score=CentipawnScore(cp=0),
            best_move=engine_move,
            pv=(engine_move,),
            nodes=1,
            time_ms=1,
        )


def _review(
    game_id: str,
    *,
    ply_count: int,
    motifs: tuple[ProfileMotifOccurrence, ...] = (),
    eco_code: str | None = None,
    opening_name: str | None = None,
) -> ProfileGameReview:
    return ProfileGameReview(
        game_id=game_id,
        players=ProfilePlayers(
            white=ProfilePlayer(name="Ada", elo=1500),
            black=ProfilePlayer(name="Turing", elo=1600),
        ),
        result="1-0",
        source="pgn_upload",
        ply_count=ply_count,
        eco_code=eco_code,
        opening_name=opening_name,
        motif_occurrences=motifs,
    )


def _occurrence(
    *,
    ply: int,
    motif_id: str,
    label: str,
    phase: Literal["opening", "middlegame", "endgame"] = "opening",
    loss_cp: int = 300,
) -> ProfileMotifOccurrence:
    return ProfileMotifOccurrence(
        ply=ply,
        move_number=(ply + 1) // 2,
        side="white" if ply % 2 == 1 else "black",
        san="e4",
        uci="e2e4",
        motif_id=motif_id,
        motif_label=label,
        severity="blunder",
        phase=phase,
        loss_cp=loss_cp,
        score_cp=loss_cp,
    )
