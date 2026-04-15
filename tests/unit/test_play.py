"""Tests for local play sessions and API routes."""

from __future__ import annotations

import asyncio

import chess
from fastapi import FastAPI
from fastapi.testclient import TestClient

from chess_ml.api.play import router as play_router
from chess_ml.engine.stockfish import EngineMove
from chess_ml.ingestion.pgn import parse_pgn
from chess_ml.play.session import InMemoryPlayStore, PlaySession


def test_new_game_returns_initial_state_and_legal_moves() -> None:
    client, _ = _client()

    response = client.post("/api/play/new")

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "play-state.v1"
    assert body["status"] == "active"
    assert body["result"] == "*"
    assert body["orientation"] == "white"
    assert body["fen"] == chess.STARTING_FEN
    assert _has_legal_destination(body["legal_moves"], "e2", "e4")


def test_legal_user_move_gets_bot_reply() -> None:
    client, opponent = _client()
    game_id = client.post("/api/play/new").json()["game_id"]

    response = client.post("/api/play/move", json={"game_id": game_id, "uci": "e2e4"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "active"
    assert body["moves"][0]["uci"] == "e2e4"
    assert body["moves"][0]["san"] == "e4"
    assert len(body["moves"]) == 2
    assert body["bot_move"] == {
        "uci": body["moves"][1]["uci"],
        "san": body["moves"][1]["san"],
    }
    assert opponent.calls == 1
    assert body["pgn"] is None


def test_illegal_move_is_rejected_without_mutating_session() -> None:
    client, opponent = _client()
    game_id = client.post("/api/play/new").json()["game_id"]

    illegal = client.post("/api/play/move", json={"game_id": game_id, "uci": "e2e5"})
    legal = client.post("/api/play/move", json={"game_id": game_id, "uci": "e2e4"})

    assert illegal.status_code == 400
    assert illegal.json()["error"]["code"] == "illegal_move"
    assert legal.status_code == 200
    assert legal.json()["moves"][0]["uci"] == "e2e4"
    assert opponent.calls == 1


def test_resign_returns_pgn_accepted_by_review_parser() -> None:
    client, _ = _client()
    game_id = client.post("/api/play/new").json()["game_id"]
    client.post("/api/play/move", json={"game_id": game_id, "uci": "e2e4"})

    response = client.post("/api/play/resign", json={"game_id": game_id})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "resigned"
    assert body["result"] == "0-1"
    assert body["pgn"] is not None
    parsed = parse_pgn(body["pgn"])
    assert parsed.result == "0-1"
    assert len(parsed.moves) == 2


def test_completed_game_does_not_request_bot_move() -> None:
    opponent = _FakeOpponent()
    session = PlaySession(
        game_id="mate-fixture",
        board=chess.Board("7k/6Q1/6K1/8/8/8/8/8 w - - 0 1"),
    )

    state = asyncio.run(session.apply_user_move("g7f8", opponent=opponent))

    assert state.status == "completed"
    assert state.result == "1-0"
    assert state.bot_move is None
    assert state.pgn is not None
    assert opponent.calls == 0


def test_missing_game_returns_not_found() -> None:
    client, _ = _client()

    response = client.post(
        "/api/play/move",
        json={"game_id": "missing", "uci": "e2e4"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "game_not_found"


def test_opponent_unavailable_returns_503() -> None:
    app = FastAPI()
    app.state.play_store = InMemoryPlayStore()
    app.state.play_opponent = None
    app.state.play_opponent_error = "No local opponent."
    app.include_router(play_router)
    client = TestClient(app)

    response = client.post("/api/play/new")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "opponent_unavailable"


class _FakeOpponent:
    def __init__(self) -> None:
        self.calls = 0

    async def choose_move(self, fen: str) -> EngineMove:
        self.calls += 1
        board = chess.Board(fen)
        move = next(iter(board.legal_moves))
        return EngineMove(uci=move.uci(), san=board.san(move))


def _client() -> tuple[TestClient, _FakeOpponent]:
    app = FastAPI()
    opponent = _FakeOpponent()
    app.state.play_store = InMemoryPlayStore()
    app.state.play_opponent = opponent
    app.state.play_opponent_error = ""
    app.include_router(play_router)
    return TestClient(app), opponent


def _has_legal_destination(
    legal_moves: list[dict[str, object]],
    from_square: str,
    to_square: str,
) -> bool:
    for group in legal_moves:
        if group["from_square"] != from_square:
            continue
        destinations = group["destinations"]
        assert isinstance(destinations, list)
        return any(
            isinstance(destination, dict) and destination.get("to_square") == to_square
            for destination in destinations
        )
    return False
