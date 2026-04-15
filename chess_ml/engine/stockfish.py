"""Async Stockfish wrapper backed by a small UCI worker pool."""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal

import chess
import chess.engine

Side = Literal["white", "black"]
EngineStatus = Literal["ok", "terminal"]

DEFAULT_STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
DEFAULT_DEPTH = 16
DEFAULT_HASH_MB = 64


class StockfishUnavailableError(RuntimeError):
    """Raised when the Stockfish binary cannot be started."""


class StockfishProtocolError(RuntimeError):
    """Raised when Stockfish fails after startup."""


@dataclass(frozen=True)
class EngineMove:
    """A move in UCI plus SAN for the position where it is legal."""

    uci: str
    san: str


@dataclass(frozen=True)
class CentipawnScore:
    """A centipawn score from White's perspective."""

    cp: int


@dataclass(frozen=True)
class MateScore:
    """A forced mate score from White's perspective."""

    mate_in: int
    winner: Side


EngineScore = CentipawnScore | MateScore


@dataclass(frozen=True)
class EngineEvaluation:
    """Stockfish analysis for one FEN."""

    status: EngineStatus
    depth: int | None
    score: EngineScore
    best_move: EngineMove | None
    pv: tuple[EngineMove, ...]
    nodes: int | None
    time_ms: int


class _StockfishWorker:
    """One Stockfish subprocess with a python-chess UCI protocol."""

    def __init__(
        self,
        transport: asyncio.SubprocessTransport,
        protocol: chess.engine.UciProtocol,
    ) -> None:
        self._transport = transport
        self._protocol = protocol
        self._closed = False

    @classmethod
    async def start(cls, path: str, *, hash_mb: int) -> _StockfishWorker:
        try:
            transport, protocol = await chess.engine.popen_uci(path)
            await protocol.configure({"Threads": 1, "Hash": hash_mb})
            await protocol.ping()
        except FileNotFoundError as exc:
            raise StockfishUnavailableError(f"Stockfish binary not found at {path}.") from exc
        except PermissionError as exc:
            raise StockfishUnavailableError(f"Stockfish binary is not executable: {path}.") from exc
        except chess.engine.EngineError as exc:
            raise StockfishUnavailableError(f"Stockfish failed to initialize: {exc}") from exc

        return cls(transport, protocol)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with suppress(chess.engine.EngineError, RuntimeError, BrokenPipeError):
            await self._protocol.quit()
        self._transport.close()

    async def evaluate(self, fen: str, *, depth: int) -> EngineEvaluation:
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=False):
            return _terminal_evaluation(board)

        start = time.perf_counter()
        info = await self._protocol.analyse(board, chess.engine.Limit(depth=depth))
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        info_dict = info

        score_info = info_dict.get("score")
        if not isinstance(score_info, chess.engine.PovScore):
            raise StockfishProtocolError("Stockfish analysis did not include a score.")

        pv_moves = info_dict.get("pv", [])

        return EngineEvaluation(
            status="ok",
            depth=_optional_int(info_dict.get("depth")),
            score=_convert_score(score_info),
            best_move=_move_to_san(board, pv_moves[0]) if pv_moves else None,
            pv=_pv_to_san(board, pv_moves),
            nodes=_optional_int(info_dict.get("nodes")),
            time_ms=elapsed_ms,
        )


class StockfishPool:
    """A bounded pool of one-thread Stockfish workers."""

    def __init__(
        self,
        *,
        path: str = DEFAULT_STOCKFISH_PATH,
        workers: int | None = None,
        depth: int = DEFAULT_DEPTH,
        hash_mb: int = DEFAULT_HASH_MB,
    ) -> None:
        self.path = path
        self.workers = workers if workers is not None else _default_worker_count()
        self.depth = depth
        self.hash_mb = hash_mb
        self._queue: asyncio.Queue[_StockfishWorker] = asyncio.Queue()
        self._workers: list[_StockfishWorker] = []
        self._started = False

    @classmethod
    def from_env(cls) -> StockfishPool:
        """Create a pool using local environment overrides."""

        return cls(
            path=os.environ.get("CHESS_ML_STOCKFISH_PATH", DEFAULT_STOCKFISH_PATH),
            workers=_env_int("CHESS_ML_STOCKFISH_WORKERS"),
            depth=_env_int("CHESS_ML_STOCKFISH_DEPTH") or DEFAULT_DEPTH,
            hash_mb=_env_int("CHESS_ML_STOCKFISH_HASH_MB") or DEFAULT_HASH_MB,
        )

    @property
    def started(self) -> bool:
        return self._started

    async def start(self) -> None:
        if self._started:
            return

        if not os.path.exists(self.path):
            raise StockfishUnavailableError(f"Stockfish binary not found at {self.path}.")

        try:
            for _ in range(max(1, self.workers)):
                worker = await _StockfishWorker.start(self.path, hash_mb=self.hash_mb)
                self._workers.append(worker)
                self._queue.put_nowait(worker)
        except BaseException:
            await self.close()
            raise

        self._started = True

    async def close(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        workers = self._workers
        self._workers = []
        self._started = False
        await asyncio.gather(*(worker.close() for worker in workers), return_exceptions=True)

    async def evaluate(self, fen: str, *, depth: int | None = None) -> EngineEvaluation:
        if not self._started:
            raise StockfishUnavailableError("Stockfish pool has not started.")

        worker = await self._queue.get()
        try:
            return await worker.evaluate(fen, depth=depth or self.depth)
        except StockfishProtocolError:
            raise
        except (chess.engine.EngineError, OSError) as exc:
            raise StockfishProtocolError(f"Stockfish failed during analysis: {exc}") from exc
        finally:
            self._queue.put_nowait(worker)


def _default_worker_count() -> int:
    cpu_count = os.cpu_count() or 2
    return min(4, max(1, cpu_count // 2))


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return int(value)


def _terminal_evaluation(board: chess.Board) -> EngineEvaluation:
    outcome = board.outcome(claim_draw=False)
    if outcome is not None and outcome.winner is not None:
        winner: Side = "white" if outcome.winner == chess.WHITE else "black"
        score: EngineScore = MateScore(mate_in=0, winner=winner)
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


def _convert_score(score: chess.engine.PovScore) -> EngineScore:
    white_score = score.white()
    mate = white_score.mate()
    if mate is not None:
        winner: Side = "white" if mate >= 0 else "black"
        return MateScore(mate_in=abs(mate), winner=winner)

    cp = white_score.score()
    if cp is None:
        raise StockfishProtocolError("Stockfish score was neither centipawn nor mate.")
    return CentipawnScore(cp=cp)


def _pv_to_san(board: chess.Board, moves: list[chess.Move]) -> tuple[EngineMove, ...]:
    line_board = board.copy(stack=False)
    san_moves: list[EngineMove] = []
    for move in moves:
        san_moves.append(_move_to_san(line_board, move))
        line_board.push(move)
    return tuple(san_moves)


def _move_to_san(board: chess.Board, move: chess.Move) -> EngineMove:
    return EngineMove(uci=move.uci(), san=board.san(move))


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str | float):
        return int(value)
    return None
