"""Tests for Maia/Lc0 play-opponent setup."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from chess_ml.engine.maia import MaiaConfig, MaiaPlayOpponent, maia_setup_status, parse_maia_rating
from chess_ml.engine.opponent import PlayOpponentRegistry, StockfishPlayConfig
from chess_ml.engine.stockfish import EngineMove, StockfishUnavailableError


def test_maia_config_resolves_weight_path(tmp_path: Path) -> None:
    config = MaiaConfig(lc0_path="/tmp/lc0", weights_dir=tmp_path, rating=1500)

    assert config.weight_path == tmp_path / "maia-1500.pb.gz"


def test_maia_setup_status_reports_missing_weights(tmp_path: Path) -> None:
    (tmp_path / "maia-1500.pb.gz").write_bytes(b"fixture")

    status = maia_setup_status(lc0_path="/missing/lc0", weights_dir=tmp_path)

    assert status.lc0_available is False
    assert status.available_ratings == (1500,)
    assert status.missing_weights == (1100, 1900)


def test_parse_maia_rating_rejects_unsupported_band() -> None:
    with pytest.raises(ValueError):
        parse_maia_rating(1300)


def test_maia_start_reports_missing_setup(tmp_path: Path) -> None:
    opponent = MaiaPlayOpponent(MaiaConfig(lc0_path=None, weights_dir=tmp_path, rating=1500))

    async def run() -> None:
        with pytest.raises(StockfishUnavailableError):
            await opponent.start()

    asyncio.run(run())


def test_registry_auto_falls_back_to_stockfish_when_maia_missing(tmp_path: Path) -> None:
    stockfish = _FakeStockfish()
    registry = PlayOpponentRegistry(
        stockfish=stockfish,
        maia_configs={
            1100: MaiaConfig(lc0_path=None, weights_dir=tmp_path, rating=1100),
            1500: MaiaConfig(lc0_path=None, weights_dir=tmp_path, rating=1500),
            1900: MaiaConfig(lc0_path=None, weights_dir=tmp_path, rating=1900),
        },
    )

    selected = asyncio.run(registry.select(requested="auto", maia_rating=1500))

    assert selected.info.kind == "stockfish"
    assert selected.info.requested == "auto"
    assert selected.info.fallback_reason is not None
    assert stockfish.started


def test_registry_explicit_maia_requires_setup(tmp_path: Path) -> None:
    registry = PlayOpponentRegistry(
        stockfish=_FakeStockfish(),
        maia_configs={
            1100: MaiaConfig(lc0_path=None, weights_dir=tmp_path, rating=1100),
            1500: MaiaConfig(lc0_path=None, weights_dir=tmp_path, rating=1500),
            1900: MaiaConfig(lc0_path=None, weights_dir=tmp_path, rating=1900),
        },
    )

    async def run() -> None:
        with pytest.raises(StockfishUnavailableError):
            await registry.select(requested="maia", maia_rating=1500)

    asyncio.run(run())


def test_real_maia_smoke_when_lc0_and_weights_exist() -> None:
    if os.environ.get("CHESS_ML_RUN_REAL_MAIA_SMOKE") != "1":
        pytest.skip("Set CHESS_ML_RUN_REAL_MAIA_SMOKE=1 to run the real Lc0/Maia smoke test.")

    config = MaiaConfig.from_env(rating=1500)
    if not config.available:
        pytest.skip("Lc0 and Maia 1500 weights are not both available.")

    async def run() -> None:
        opponent = MaiaPlayOpponent(config)
        await opponent.start()
        try:
            move = await opponent.choose_move(
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
            )
        finally:
            await opponent.close()

        assert move.uci
        assert move.san

    asyncio.run(run())


class _FakeStockfish:
    config = StockfishPlayConfig(path="/tmp/stockfish-fixture")

    def __init__(self) -> None:
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.started = False

    async def choose_move(self, fen: str) -> EngineMove:
        return EngineMove(uci="e7e5", san="e5")
