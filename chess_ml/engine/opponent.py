"""Opponent move providers for local play mode."""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import chess
import chess.engine

from chess_ml.engine.maia import (
    DEFAULT_MAIA_RATING,
    SUPPORTED_MAIA_RATINGS,
    MaiaConfig,
    MaiaPlayOpponent,
    MaiaRating,
    MaiaSetupStatus,
    maia_setup_status,
)
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

RequestedOpponent: TypeAlias = Literal["auto", "maia", "stockfish"]
ActualOpponent: TypeAlias = Literal["maia", "stockfish"]


class OpponentMoveProvider(Protocol):
    """A swappable source of opponent moves."""

    async def choose_move(self, fen: str) -> EngineMove:
        """Return a legal move for the given position."""


@dataclass(frozen=True)
class OpponentInfo:
    """Public metadata for the play opponent selected for one session."""

    kind: ActualOpponent
    requested: RequestedOpponent
    label: str
    engine: str
    maia_rating: MaiaRating | None
    fallback_reason: str | None


@dataclass(frozen=True)
class SelectedOpponent:
    """A selected provider plus the metadata to stamp onto the session."""

    provider: OpponentMoveProvider
    info: OpponentInfo


@dataclass(frozen=True)
class PlayOpponentStatus:
    """Non-probing status for locally configured play opponents."""

    default_requested: RequestedOpponent
    default_maia_rating: MaiaRating
    stockfish_path: str
    stockfish_available: bool
    stockfish_label: str
    maia: MaiaSetupStatus


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


class PlayOpponentRegistry:
    """Selects and owns local play-opponent engine processes."""

    def __init__(
        self,
        *,
        stockfish: StockfishPlayOpponent,
        maia_configs: dict[MaiaRating, MaiaConfig],
        default_requested: RequestedOpponent = "auto",
        default_maia_rating: MaiaRating = DEFAULT_MAIA_RATING,
    ) -> None:
        self.stockfish = stockfish
        self.maia_configs = maia_configs
        self.default_requested = default_requested
        self.default_maia_rating = default_maia_rating
        self._maia_by_rating: dict[MaiaRating, MaiaPlayOpponent] = {}
        self._stockfish_error: str | None = None

    @classmethod
    def from_env(cls) -> PlayOpponentRegistry:
        """Create the local opponent registry from environment variables."""

        stockfish = StockfishPlayOpponent.from_env()
        maia_configs = {
            rating: MaiaConfig.from_env(rating=rating) for rating in SUPPORTED_MAIA_RATINGS
        }
        return cls(stockfish=stockfish, maia_configs=maia_configs)

    async def start_fallback(self) -> None:
        """Start Stockfish fallback if it is available."""

        try:
            await self.stockfish.start()
        except StockfishUnavailableError as exc:
            self._stockfish_error = str(exc)
        else:
            self._stockfish_error = None

    async def close(self) -> None:
        """Stop all engine processes owned by the registry."""

        await asyncio.gather(
            *(opponent.close() for opponent in self._maia_by_rating.values()),
            return_exceptions=True,
        )
        self._maia_by_rating.clear()
        if self.stockfish.started:
            await self.stockfish.close()

    async def select(
        self,
        *,
        requested: RequestedOpponent,
        maia_rating: MaiaRating,
    ) -> SelectedOpponent:
        """Return the provider to use for a new play session."""

        if requested == "stockfish":
            return await self._select_stockfish(requested="stockfish", fallback_reason=None)

        if requested == "maia":
            return await self._select_maia(requested="maia", maia_rating=maia_rating)

        maia_unavailable_reason = self._maia_unavailable_reason(maia_rating)
        if maia_unavailable_reason is None:
            try:
                return await self._select_maia(requested="auto", maia_rating=maia_rating)
            except StockfishUnavailableError as exc:
                maia_unavailable_reason = str(exc)

        return await self._select_stockfish(
            requested="auto",
            fallback_reason=maia_unavailable_reason,
        )

    def status(self) -> PlayOpponentStatus:
        """Return local setup status without starting Maia."""

        maia_status = maia_setup_status()
        return PlayOpponentStatus(
            default_requested=self.default_requested,
            default_maia_rating=self.default_maia_rating,
            stockfish_path=self.stockfish.config.path,
            stockfish_available=os.path.exists(self.stockfish.config.path),
            stockfish_label=_stockfish_label(self.stockfish.config),
            maia=maia_status,
        )

    async def _select_maia(
        self,
        *,
        requested: RequestedOpponent,
        maia_rating: MaiaRating,
    ) -> SelectedOpponent:
        config = self.maia_configs[maia_rating]
        opponent = self._maia_by_rating.get(maia_rating)
        if opponent is None:
            opponent = MaiaPlayOpponent(config)
            self._maia_by_rating[maia_rating] = opponent
        await opponent.start()
        return SelectedOpponent(
            provider=opponent,
            info=OpponentInfo(
                kind="maia",
                requested=requested,
                label=f"Maia {maia_rating}",
                engine="Lc0 Maia",
                maia_rating=maia_rating,
                fallback_reason=None,
            ),
        )

    async def _select_stockfish(
        self,
        *,
        requested: RequestedOpponent,
        fallback_reason: str | None,
    ) -> SelectedOpponent:
        if not self.stockfish.started:
            await self.start_fallback()
        if not self.stockfish.started:
            raise StockfishUnavailableError(
                self._stockfish_error or "Stockfish fallback opponent is unavailable."
            )

        return SelectedOpponent(
            provider=self.stockfish,
            info=OpponentInfo(
                kind="stockfish",
                requested=requested,
                label=_stockfish_label(self.stockfish.config),
                engine="Stockfish",
                maia_rating=None,
                fallback_reason=fallback_reason,
            ),
        )

    def _maia_unavailable_reason(self, rating: MaiaRating) -> str | None:
        config = self.maia_configs[rating]
        if config.lc0_path is None or not os.path.exists(config.lc0_path):
            return "Lc0 binary not found. Install it with `brew install lc0`."
        if not config.weight_path.exists():
            return f"Maia {rating} weights not found at {config.weight_path}."
        return None


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return int(value)


def _stockfish_label(config: StockfishPlayConfig) -> str:
    return f"Stockfish fallback ({config.elo} Elo)"
