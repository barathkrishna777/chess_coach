"""Tests for Slice 12 personal drill mode."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from chess_ml.api.train import router as train_router
from chess_ml.profile.store import (
    ProfileGameReview,
    ProfileMotifOccurrence,
    ProfilePlayer,
    ProfilePlayers,
    ProfileStore,
)


def test_train_next_returns_due_drill_without_answer(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "profile.sqlite3")
    store.save_review(_review("game-1", motifs=(_drill_occurrence(),)))
    client = _client(store)

    response = client.get("/api/train/next?motif=missed_tactic")

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "training-drill.v1"
    assert body["motif"] == "missed_tactic"
    assert body["fen"] == _START_FEN
    assert "best_move" not in json.dumps(body)
    assert body["context"]["pv"] == []
    assert body["context"]["explanation_text"] is None


def test_train_result_records_attempt_and_reveals_answer(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "profile.sqlite3")
    store.save_review(_review("game-1", motifs=(_drill_occurrence(),)))
    client = _client(store)
    drill_id = client.get("/api/train/next").json()["drill_id"]

    response = client.post(
        "/api/train/result",
        json={"drill_id": drill_id, "attempted_uci": "e2e4"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "training-result.v1"
    assert body["correct"] is True
    assert body["best_move"] == {"uci": "e2e4", "san": "e4"}
    assert body["context"]["pv"] == [{"uci": "e2e4", "san": "e4"}]
    assert body["context"]["explanation_text"] == "Play e4 to take the center."

    stats_response = client.get("/api/train/stats")
    assert stats_response.status_code == 200
    stats = stats_response.json()
    assert stats["totals"]["trainable_positions"] == 1
    assert stats["totals"]["attempts"] == 1
    assert stats["totals"]["correct_attempts"] == 1


def test_drill_scheduling_uses_leitner_intervals(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "profile.sqlite3")
    store.save_review(_review("game-1", motifs=(_drill_occurrence(),)))
    now = datetime(2026, 4, 16, 12, tzinfo=UTC)

    incorrect = store.record_drill_attempt(
        game_id="game-1",
        ply=3,
        motif="missed_tactic",
        attempted_uci="g1f3",
        now=now,
    )

    assert incorrect.correct is False
    assert incorrect.next_due_at == (now + timedelta(hours=1)).isoformat(timespec="seconds")
    assert store.next_drill("missed_tactic", now=now + timedelta(minutes=30)) is None
    assert store.next_drill("missed_tactic", now=now + timedelta(hours=2)) is not None

    first_correct_at = now + timedelta(hours=2)
    first_correct = store.record_drill_attempt(
        game_id="game-1",
        ply=3,
        motif="missed_tactic",
        attempted_uci="e2e4",
        now=first_correct_at,
    )
    second_correct_at = first_correct_at + timedelta(days=2)
    second_correct = store.record_drill_attempt(
        game_id="game-1",
        ply=3,
        motif="missed_tactic",
        attempted_uci="e2e4",
        now=second_correct_at,
    )

    assert first_correct.correct is True
    assert first_correct.next_due_at == (first_correct_at + timedelta(days=2)).isoformat(
        timespec="seconds"
    )
    assert second_correct.correct is True
    assert second_correct.next_due_at == (second_correct_at + timedelta(days=4)).isoformat(
        timespec="seconds"
    )


def test_train_ignores_profile_rows_without_drill_detail(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "profile.sqlite3")
    store.save_review(
        _review(
            "game-1",
            motifs=(
                ProfileMotifOccurrence(
                    ply=3,
                    move_number=2,
                    side="white",
                    san="Nf3",
                    uci="g1f3",
                    motif_id="missed_tactic",
                    motif_label="Missed tactic",
                    severity="blunder",
                    phase="opening",
                    loss_cp=320,
                    score_cp=320,
                ),
            ),
        )
    )

    client = _client(store)
    response = client.get("/api/train/next?motif=missed_tactic")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_due_drill"
    assert store.drill_stats().totals.trainable_positions == 0


def _client(store: ProfileStore) -> TestClient:
    app = FastAPI()
    app.state.profile_store = store
    app.include_router(train_router)
    return TestClient(app)


_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _review(
    game_id: str,
    *,
    motifs: tuple[ProfileMotifOccurrence, ...],
) -> ProfileGameReview:
    return ProfileGameReview(
        game_id=game_id,
        players=ProfilePlayers(
            white=ProfilePlayer(name="Ada", elo=1500),
            black=ProfilePlayer(name="Turing", elo=1520),
        ),
        result="1-0",
        source="pgn_upload",
        ply_count=8,
        motif_occurrences=motifs,
    )


def _drill_occurrence() -> ProfileMotifOccurrence:
    return ProfileMotifOccurrence(
        ply=3,
        move_number=2,
        side="white",
        san="Nf3",
        uci="g1f3",
        motif_id="missed_tactic",
        motif_label="Missed tactic",
        severity="blunder",
        phase="opening",
        loss_cp=320,
        score_cp=320,
        fen_before=_START_FEN,
        best_move_uci="e2e4",
        best_move_san="e4",
        pv_json=json.dumps([{"uci": "e2e4", "san": "e4"}]),
        explanation_text="Play e4 to take the center.",
        explanation_status="ok",
        evidence_json=json.dumps(
            {
                "threshold_cp": 300,
                "score_kind": "cp",
                "phase": "opening",
                "piece": None,
                "attackers": [],
                "defenders": [],
                "best_move": {"uci": "e2e4", "san": "e4"},
                "opponent_reply": None,
                "related_ply": None,
            }
        ),
    )
