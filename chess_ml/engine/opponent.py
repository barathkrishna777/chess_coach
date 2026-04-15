"""Opponent move providers for local play mode."""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol

import chess
import chess.engine

from chess_ml.engine.stockfish import (
    DEFAULT_HASH_MB,
    DEFAULT_STOCKFISH_PATH,
    EngineMove,
    StockfishProtocolError,
    StockfishUnavailableError,
)

DEFAULT_PLAY_ELO = 1350
DEFAULT_PLAY_SKILL_LEVEL = 4
DEFAULT_PLAY_TIME_MS = 250


class OpponentMoveProvider(Protocol):
    """A swappable source of opponent moves."""

    async def choose_move(self, fen: str) -> EngineMove:
        """Return a legal move for the given position."""


@dataclass(frozen=True)
class StockfishPlayConfig:
    """Configuration for the low-strength Stockfish play opponent."""

    path: str = DEFAULT_STOCKFISH_PATH
    elo: int = DEFAULT_PLAY_ELO
    skill_level: int = DEFAULT_PLAY_SKILL_LEVEL
    move_time_ms: int = DEFAULT_PLAY_TIME_MS
    hash_mb: int = DEFAULT_HASH_MB

    @classmethod
    def from_env(cls) -> StockfishPlayConfig:
        """Create play-opponent config from environment variables."""

        path = (
            os.environ.get("CHESS_ML_PLAY_STOCKFISH_PATH")
            or os.environ.get("CHESS_ML_STOCKFISH_PATH")
            or DEFAULT_STOCKFISH_PATH
        )
        return cls(
            path=path,
            elo=_env_int("CHESS_ML_PLAY_STOCKFISH_ELO") or DEFAULT_PLAY_ELO,
            skill_level=_env_int("CHESS_ML_PLAY_STOCKFISH_SKILL_LEVEL") or DEFAULT_PLAY_SKILL_LEVEL,
            move_time_ms=_env_int("CHESS_ML_PLAY_STOCKFISH_TIME_MS") or DEFAULT_PLAY_TIME_MS,
            hash_mb=_env_int("CHESS_ML_PLAY_STOCKFISH_HASH_MB") or DEFAULT_HASH_MB,
        )


class StockfishPlayOpponent:
    """A low-strength Stockfish process used only for playing against the user."""

    def __init__(self, config: StockfishPlayConfig) -> None:
        self.config = config
        self._transport: asyncio.SubprocessTransport | None = None
        self._protocol: chess.engine.UciProtocol | None = None
        self._lock = asyncio.Lock()

    @classmethod
    def from_env(cls) -> StockfishPlayOpponent:
        """Build the default local play opponent."""

        return cls(StockfishPlayConfig.from_env())

    @property
    def started(self) -> bool:
        """Whether the underlying engine process is available."""

        return self._protocol is not None

    async def start(self) -> None:
        """Start and configure the play engine."""

        if self.started:
            return
        if not os.path.exists(self.config.path):
            raise StockfishUnavailableError(
                f"Stockfish play binary not found at {self.config.path}."
            )

        try:
            transport, protocol = await chess.engine.popen_uci(self.config.path)
            self._transport = transport
            self._protocol = protocol
            await protocol.configure(self._engine_options())
            await protocol.ping()
        except FileNotFoundError as exc:
            raise StockfishUnavailableError(
                f"Stockfish play binary not found at {self.config.path}."
            ) from exc
        except PermissionError as exc:
            raise StockfishUnavailableError(
                f"Stockfish play binary is not executable: {self.config.path}."
            ) from exc
        except chess.engine.EngineError as exc:
            await self.close()
            raise StockfishUnavailableError(
                f"Stockfish play engine failed to start: {exc}"
            ) from exc

    async def close(self) -> None:
        """Stop the play engine process."""

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
        """Choose one legal reply for a non-terminal position."""

        protocol = self._protocol
        if protocol is None:
            raise StockfishUnavailableError("Stockfish play engine has not started.")

        board = chess.Board(fen)
        if board.is_game_over(claim_draw=False):
            raise StockfishProtocolError("Cannot choose an opponent move in a terminal position.")

        async with self._lock:
            start = time.perf_counter()
            try:
                result = await protocol.play(
                    board,
                    chess.engine.Limit(time=self.config.move_time_ms / 1000),
                )
            except (chess.engine.EngineError, OSError) as exc:
                raise StockfishProtocolError(
                    f"Stockfish play engine failed during move selection: {exc}"
                ) from exc

        if result.move is None or result.move not in board.legal_moves:
            elapsed_ms = round((time.perf_counter() - start) * 1000)
            raise StockfishProtocolError(
                f"Stockfish play engine returned no legal move after {elapsed_ms}ms."
            )

        return EngineMove(uci=result.move.uci(), san=board.san(result.move))

    def _engine_options(self) -> dict[str, int | bool]:
        return {
            "Threads": 1,
            "Hash": self.config.hash_mb,
            "UCI_LimitStrength": True,
            "UCI_Elo": self.config.elo,
            "Skill Level": self.config.skill_level,
        }


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return int(value)
