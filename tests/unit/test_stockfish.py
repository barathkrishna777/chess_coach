"""Tests for the Stockfish UCI wrapper."""

import asyncio
import os
from pathlib import Path

import pytest

from chess_ml.engine.stockfish import (
    DEFAULT_STOCKFISH_PATH,
    CentipawnScore,
    MateScore,
    StockfishPool,
)


def test_stockfish_evaluates_starting_position() -> None:
    stockfish_path = os.environ.get("CHESS_ML_STOCKFISH_PATH", DEFAULT_STOCKFISH_PATH)
    if not Path(stockfish_path).exists():
        pytest.skip(f"Stockfish binary not found at {stockfish_path}")

    async def run() -> None:
        pool = StockfishPool(path=stockfish_path, workers=1, depth=4, hash_mb=16)
        await pool.start()
        try:
            evaluation = await pool.evaluate(
                "rn1qkbnr/ppp2ppp/3b4/3pp3/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 2 5",
                depth=4,
            )
        finally:
            await pool.close()

        assert evaluation.status == "ok"
        assert evaluation.depth is not None
        assert isinstance(evaluation.score, CentipawnScore | MateScore)
        assert evaluation.best_move is not None
        assert evaluation.best_move.uci
        assert evaluation.pv

    asyncio.run(run())
