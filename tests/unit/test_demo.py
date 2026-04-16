"""Tests for Slice 9 demo profile seeding."""

from __future__ import annotations

import asyncio
from pathlib import Path

import chess

from chess_ml.engine.stockfish import CentipawnScore, EngineEvaluation, EngineMove, MateScore
from chess_ml.profile.demo import DEMO_PGN_PATHS, seed_demo_profile
from chess_ml.profile.store import ProfileStore


def test_demo_fixture_set_has_three_checked_in_games() -> None:
    assert [path.name for path in DEMO_PGN_PATHS] == [
        "hanging-piece.pgn",
        "missed-tactic.pgn",
        "mate-threat.pgn",
    ]
    assert all(path.exists() for path in DEMO_PGN_PATHS)


def test_demo_seed_upserts_three_games_without_duplicates(tmp_path: Path) -> None:
    db_path = tmp_path / "demo.sqlite3"
    evaluator = _FakeEvaluator()

    first = asyncio.run(seed_demo_profile(db_path=db_path, evaluator=evaluator, depth=1))
    second = asyncio.run(seed_demo_profile(db_path=db_path, evaluator=evaluator, depth=1))

    dashboard = ProfileStore(db_path).dashboard()
    assert first.games_seeded == 3
    assert second.games_seeded == 3
    assert dashboard.totals.games_reviewed == 3
    assert dashboard.totals.moves_reviewed == first.moves_reviewed == second.moves_reviewed
    assert dashboard.recent_games[0].players.white.name == "Priya"
    assert dashboard.recent_games[1].players.white.name == "Ada"
    assert dashboard.recent_games[2].players.white.name == "Nina"


class _FakeEvaluator:
    async def evaluate(self, fen: str, *, depth: int | None = None) -> EngineEvaluation:
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=False):
            outcome = board.outcome(claim_draw=False)
            score: CentipawnScore | MateScore
            if outcome is not None and outcome.winner is not None:
                score = MateScore(
                    mate_in=0,
                    winner="white" if outcome.winner == chess.WHITE else "black",
                )
            else:
                score = CentipawnScore(cp=0)
            return EngineEvaluation(
                status="terminal",
                depth=None,
                score=score,
                best_move=None,
                pv=(),
                nodes=None,
                time_ms=0,
            )

        move = next(iter(board.legal_moves))
        engine_move = EngineMove(uci=move.uci(), san=board.san(move))
        return EngineEvaluation(
            status="ok",
            depth=depth or 1,
            score=CentipawnScore(cp=0),
            best_move=engine_move,
            pv=(engine_move,),
            nodes=1,
            time_ms=1,
        )
