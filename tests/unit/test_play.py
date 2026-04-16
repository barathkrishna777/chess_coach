"""Tests for local play sessions and API routes."""

from __future__ import annotations

import asyncio

import chess
from fastapi import FastAPI
from fastapi.testclient import TestClient

from chess_ml.api.play import router as play_router
from chess_ml.engine.maia import MaiaSetupStatus
from chess_ml.engine.opponent import OpponentInfo, PlayOpponentStatus, SelectedOpponent
from chess_ml.engine.stockfish import CentipawnScore, EngineEvaluation, EngineMove
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
    assert body["user_color"] == "white"
    assert body["orientation"] == "white"
    assert body["hints_remaining"] == 3
    assert body["takebacks_remaining"] == 1
    assert body["opponent"]["kind"] == "stockfish"
    assert body["fen"] == chess.STARTING_FEN
    assert _has_legal_destination(body["legal_moves"], "e2", "e4")


def test_new_game_as_black_gets_bot_first_move_and_black_legal_moves() -> None:
    client, opponent = _client()

    response = client.post("/api/play/new", json={"user_color": "black"})

    assert response.status_code == 200
    body = response.json()
    assert body["user_color"] == "black"
    assert body["orientation"] == "black"
    assert len(body["moves"]) == 1
    assert body["moves"][0]["side"] == "white"
    assert body["bot_move"] == {
        "uci": body["moves"][0]["uci"],
        "san": body["moves"][0]["san"],
    }
    assert opponent.calls == 1
    assert _has_legal_destination(body["legal_moves"], "e7", "e5")


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


def test_resign_as_black_records_result_and_headers() -> None:
    client, _ = _client()
    game_id = client.post("/api/play/new", json={"user_color": "black"}).json()["game_id"]

    response = client.post("/api/play/resign", json={"game_id": game_id})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "resigned"
    assert body["result"] == "1-0"
    assert body["pgn"] is not None
    assert '[White "Stockfish fallback"]' in body["pgn"]
    assert '[Black "You"]' in body["pgn"]
    parsed = parse_pgn(body["pgn"])
    assert parsed.result == "1-0"


def test_completed_game_does_not_request_bot_move() -> None:
    opponent = _FakeOpponent()
    session = PlaySession(
        game_id="mate-fixture",
        opponent=_stockfish_info(),
        opponent_provider=opponent,
        board=chess.Board("7k/6Q1/6K1/8/8/8/8/8 w - - 0 1"),
    )

    state = asyncio.run(session.apply_user_move("g7f8"))

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


def test_takeback_restores_position_and_is_limited_to_one_use() -> None:
    client, _ = _client()
    initial = client.post("/api/play/new").json()
    game_id = initial["game_id"]
    moved = client.post("/api/play/move", json={"game_id": game_id, "uci": "e2e4"}).json()
    assert len(moved["moves"]) == 2

    takeback = client.post("/api/play/takeback", json={"game_id": game_id})
    second = client.post("/api/play/takeback", json={"game_id": game_id})

    assert takeback.status_code == 200
    body = takeback.json()
    assert body["fen"] == chess.STARTING_FEN
    assert body["moves"] == []
    assert body["takebacks_remaining"] == 0
    assert _has_legal_destination(body["legal_moves"], "e2", "e4")
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "takeback_unavailable"


def test_takeback_requires_a_user_move() -> None:
    client, _ = _client()
    white_game_id = client.post("/api/play/new").json()["game_id"]
    black_game_id = client.post("/api/play/new", json={"user_color": "black"}).json()["game_id"]

    white_response = client.post("/api/play/takeback", json={"game_id": white_game_id})
    black_response = client.post("/api/play/takeback", json={"game_id": black_game_id})

    assert white_response.status_code == 409
    assert white_response.json()["error"]["code"] == "takeback_unavailable"
    assert black_response.status_code == 409
    assert black_response.json()["error"]["code"] == "takeback_unavailable"


def test_hint_returns_best_move_and_is_limited_to_three_uses() -> None:
    stockfish_pool = _FakeStockfishPool()
    client, _ = _client(stockfish_pool=stockfish_pool)
    game_id = client.post("/api/play/new").json()["game_id"]

    first = client.get(f"/api/play/hint?session_id={game_id}")
    second = client.get(f"/api/play/hint?session_id={game_id}")
    third = client.get(f"/api/play/hint?session_id={game_id}")
    fourth = client.get(f"/api/play/hint?session_id={game_id}")

    assert first.status_code == 200
    first_body = first.json()
    best_move_uci = first_body["best_move"]["uci"]
    assert first_body["schema_version"] == "play-hint.v1"
    assert first_body["from_square"] == best_move_uci[:2]
    assert first_body["to_square"] == best_move_uci[2:4]
    assert first_body["hints_remaining"] == 2
    assert second.json()["hints_remaining"] == 1
    assert third.json()["hints_remaining"] == 0
    assert fourth.status_code == 409
    assert fourth.json()["error"]["code"] == "hint_unavailable"
    assert stockfish_pool.calls == 3


def test_promotion_uci_is_accepted_by_play_session() -> None:
    opponent = _FakeOpponent()
    session = PlaySession(
        game_id="promotion-fixture",
        opponent=_stockfish_info(),
        opponent_provider=opponent,
        board=chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1"),
    )

    state = asyncio.run(session.apply_user_move("a7a8q"))

    assert state.moves[0].uci == "a7a8q"
    assert state.moves[0].side == "white"


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


def test_new_game_selects_requested_maia_from_registry() -> None:
    app = FastAPI()
    registry = _FakeRegistry()
    app.state.play_store = InMemoryPlayStore()
    app.state.play_opponents = registry
    app.include_router(play_router)
    client = TestClient(app)

    response = client.post("/api/play/new", json={"opponent": "maia", "maia_rating": 1900})

    assert response.status_code == 200
    body = response.json()
    assert body["opponent"]["kind"] == "maia"
    assert body["opponent"]["maia_rating"] == 1900
    assert registry.requests == [("maia", 1900)]


def test_play_opponents_status_returns_setup_facts() -> None:
    app = FastAPI()
    app.state.play_store = InMemoryPlayStore()
    app.state.play_opponents = _FakeRegistry()
    app.include_router(play_router)
    client = TestClient(app)

    response = client.get("/api/play/opponents")

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "play-opponents.v1"
    assert body["default_requested"] == "auto"
    assert body["maia"]["available_ratings"] == [1500]
    assert body["maia"]["missing_weights"] == [1100, 1900]


def test_resign_pgn_records_selected_opponent_headers() -> None:
    client, _ = _client()
    game_id = client.post("/api/play/new").json()["game_id"]
    client.post("/api/play/move", json={"game_id": game_id, "uci": "e2e4"})

    body = client.post("/api/play/resign", json={"game_id": game_id}).json()

    assert '[Black "Stockfish fallback"]' in body["pgn"]


class _FakeOpponent:
    def __init__(self) -> None:
        self.calls = 0

    async def choose_move(self, fen: str) -> EngineMove:
        self.calls += 1
        board = chess.Board(fen)
        move = next(iter(board.legal_moves))
        return EngineMove(uci=move.uci(), san=board.san(move))


class _FakeStockfishPool:
    def __init__(self) -> None:
        self.calls = 0

    async def evaluate(self, fen: str, *, depth: int | None = None) -> EngineEvaluation:
        self.calls += 1
        board = chess.Board(fen)
        move = next(iter(board.legal_moves))
        engine_move = EngineMove(uci=move.uci(), san=board.san(move))
        return EngineEvaluation(
            status="ok",
            depth=depth,
            score=CentipawnScore(cp=0),
            best_move=engine_move,
            pv=(engine_move,),
            nodes=1,
            time_ms=1,
        )


class _FakeRegistry:
    def __init__(self) -> None:
        self.opponent = _FakeOpponent()
        self.requests: list[tuple[str, int]] = []

    async def select(self, *, requested: str, maia_rating: int) -> SelectedOpponent:
        self.requests.append((requested, maia_rating))
        return SelectedOpponent(
            provider=self.opponent,
            info=OpponentInfo(
                kind="maia" if requested == "maia" else "stockfish",
                requested=requested if requested in {"auto", "maia", "stockfish"} else "auto",
                label=f"Maia {maia_rating}" if requested == "maia" else "Stockfish fallback",
                engine="Lc0 Maia" if requested == "maia" else "Stockfish",
                maia_rating=maia_rating if requested == "maia" else None,
                fallback_reason=None,
            ),
        )

    def status(self) -> PlayOpponentStatus:
        return PlayOpponentStatus(
            default_requested="auto",
            default_maia_rating=1500,
            stockfish_path="/opt/homebrew/bin/stockfish",
            stockfish_available=True,
            stockfish_label="Stockfish fallback",
            maia=MaiaSetupStatus(
                lc0_path="/opt/homebrew/bin/lc0",
                lc0_available=True,
                weights_dir="checkpoints/maia",
                ratings=(1100, 1500, 1900),
                available_ratings=(1500,),
                missing_weights=(1100, 1900),
            ),
        )


def _client(stockfish_pool: _FakeStockfishPool | None = None) -> tuple[TestClient, _FakeOpponent]:
    app = FastAPI()
    opponent = _FakeOpponent()
    app.state.play_store = InMemoryPlayStore()
    app.state.play_opponent = opponent
    app.state.play_opponent_error = ""
    app.state.stockfish_pool = stockfish_pool
    app.state.stockfish_error = "" if stockfish_pool is not None else "No Stockfish pool."
    app.include_router(play_router)
    return TestClient(app), opponent


def _stockfish_info() -> OpponentInfo:
    return OpponentInfo(
        kind="stockfish",
        requested="stockfish",
        label="Stockfish fallback",
        engine="Stockfish",
        maia_rating=None,
        fallback_reason=None,
    )


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
