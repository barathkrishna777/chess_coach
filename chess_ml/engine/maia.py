"""Maia/Lc0 opponent wrapper for local play mode."""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import chess
import chess.engine

from chess_ml.engine.stockfish import EngineMove, StockfishProtocolError, StockfishUnavailableError

MaiaRating: TypeAlias = Literal[1100, 1500, 1900]

SUPPORTED_MAIA_RATINGS: tuple[MaiaRating, ...] = (1100, 1500, 1900)
DEFAULT_MAIA_RATING: MaiaRating = 1500
DEFAULT_MAIA_WEIGHTS_DIR = Path("checkpoints/maia")


@dataclass(frozen=True)
class MaiaConfig:
    """Configuration for one Maia rating band backed by Lc0."""

    lc0_path: str | None
    weights_dir: Path
    rating: MaiaRating

    @classmethod
    def from_env(cls, *, rating: MaiaRating = DEFAULT_MAIA_RATING) -> MaiaConfig:
        """Create Maia config from environment variables."""

        lc0_path = os.environ.get("CHESS_ML_LC0_PATH") or shutil.which("lc0")
        weights_dir = Path(os.environ.get("CHESS_ML_MAIA_WEIGHTS_DIR", DEFAULT_MAIA_WEIGHTS_DIR))
        return cls(lc0_path=lc0_path, weights_dir=weights_dir, rating=rating)

    @property
    def weight_path(self) -> Path:
        """Return the expected local weight file for this rating band."""

        return self.weights_dir / f"maia-{self.rating}.pb.gz"

    @property
    def available(self) -> bool:
        """Whether the configured Lc0 binary and Maia weight file exist."""

        return (
            self.lc0_path is not None and Path(self.lc0_path).exists() and self.weight_path.exists()
        )


@dataclass(frozen=True)
class MaiaSetupStatus:
    """Non-probing Maia setup facts for API/UI status."""

    lc0_path: str | None
    lc0_available: bool
    weights_dir: str
    ratings: tuple[MaiaRating, ...]
    available_ratings: tuple[MaiaRating, ...]
    missing_weights: tuple[MaiaRating, ...]


class MaiaPlayOpponent:
    """A Maia policy network served through an Lc0 UCI process."""

    def __init__(self, config: MaiaConfig) -> None:
        self.config = config
        self._transport: asyncio.SubprocessTransport | None = None
        self._protocol: chess.engine.UciProtocol | None = None
        self._lock = asyncio.Lock()

    @property
    def started(self) -> bool:
        """Whether the underlying Lc0 process is available."""

        return self._protocol is not None

    async def start(self) -> None:
        """Start Lc0 with the configured Maia weights."""

        if self.started:
            return
        if self.config.lc0_path is None or not Path(self.config.lc0_path).exists():
            raise StockfishUnavailableError(
                "Lc0 binary not found. Install it with `brew install lc0`."
            )
        if not self.config.weight_path.exists():
            raise StockfishUnavailableError(
                f"Maia {self.config.rating} weights not found at {self.config.weight_path}."
            )

        try:
            transport, protocol = await chess.engine.popen_uci(
                [self.config.lc0_path, f"--weights={self.config.weight_path}"]
            )
            self._transport = transport
            self._protocol = protocol
            await protocol.ping()
        except FileNotFoundError as exc:
            raise StockfishUnavailableError(
                f"Lc0 binary not found at {self.config.lc0_path}."
            ) from exc
        except PermissionError as exc:
            raise StockfishUnavailableError(
                f"Lc0 binary is not executable: {self.config.lc0_path}."
            ) from exc
        except chess.engine.EngineError as exc:
            await self.close()
            raise StockfishUnavailableError(f"Lc0 failed to start Maia: {exc}") from exc

    async def close(self) -> None:
        """Stop the Lc0 process."""

        protocol = self._protocol
        transport = self._transport
        self._protocol = None
        self._transport = None
        if protocol is not None:
            with suppress(chess.engine.EngineError, RuntimeError, BrokenPipeError):
                await protocol.quit()
        if transport is not None:
            transport.close()

    async def choose_move(self, fen: str) -> EngineMove:
        """Choose one legal Maia reply for a non-terminal position."""

        protocol = self._protocol
        if protocol is None:
            raise StockfishUnavailableError("Maia opponent has not started.")

        board = chess.Board(fen)
        if board.is_game_over(claim_draw=False):
            raise StockfishProtocolError("Cannot choose an opponent move in a terminal position.")

        async with self._lock:
            start = time.perf_counter()
            try:
                result = await protocol.play(board, chess.engine.Limit(nodes=1))
            except (chess.engine.EngineError, OSError) as exc:
                raise StockfishProtocolError(
                    f"Maia opponent failed during move selection: {exc}"
                ) from exc

        if result.move is None or result.move not in board.legal_moves:
            elapsed_ms = round((time.perf_counter() - start) * 1000)
            raise StockfishProtocolError(f"Maia returned no legal move after {elapsed_ms}ms.")

        return EngineMove(uci=result.move.uci(), san=board.san(result.move))


def maia_setup_status(
    *,
    lc0_path: str | None = None,
    weights_dir: Path | None = None,
) -> MaiaSetupStatus:
    """Return local Maia setup facts without starting Lc0."""

    resolved_lc0 = lc0_path or os.environ.get("CHESS_ML_LC0_PATH") or shutil.which("lc0")
    resolved_weights_dir = weights_dir or Path(
        os.environ.get("CHESS_ML_MAIA_WEIGHTS_DIR", DEFAULT_MAIA_WEIGHTS_DIR)
    )
    lc0_available = resolved_lc0 is not None and Path(resolved_lc0).exists()
    available = tuple(
        rating
        for rating in SUPPORTED_MAIA_RATINGS
        if (resolved_weights_dir / f"maia-{rating}.pb.gz").exists()
    )
    missing = tuple(rating for rating in SUPPORTED_MAIA_RATINGS if rating not in available)
    return MaiaSetupStatus(
        lc0_path=resolved_lc0,
        lc0_available=lc0_available,
        weights_dir=str(resolved_weights_dir),
        ratings=SUPPORTED_MAIA_RATINGS,
        available_ratings=available,
        missing_weights=missing,
    )


def parse_maia_rating(value: int) -> MaiaRating:
    """Validate and narrow a public Maia rating band."""

    if value in SUPPORTED_MAIA_RATINGS:
        return value
    raise ValueError(f"Unsupported Maia rating: {value}.")
