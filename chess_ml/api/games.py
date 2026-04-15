"""Game review API routes."""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from collections.abc import Sequence
from typing import Annotated, Literal, cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from chess_ml.classifier.classify import classify_moves
from chess_ml.classifier.motifs import AnalyzedMove
from chess_ml.classifier.motifs import Motif as ClassifiedMotif
from chess_ml.engine.stockfish import (
    CentipawnScore,
    EngineEvaluation,
    EngineMove,
    EngineScore,
    MateScore,
    StockfishPool,
    StockfishProtocolError,
    StockfishUnavailableError,
)
from chess_ml.ingestion.pgn import ParsedPgnGame, ParsedPgnMove, PgnParseError, parse_pgn

router = APIRouter(prefix="/api/games", tags=["games"])

MAX_PGN_CHARS = 200_000
MAX_PLIES = 160
DEFAULT_ANALYSIS_TIMEOUT_SECONDS = 14.5


class CreateGameRequest(BaseModel):
    """Payload for submitting one PGN for review."""

    model_config = ConfigDict(extra="forbid")

    pgn: str = Field(min_length=1)


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, str] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class PlayerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None
    elo: int | None


class PlayersModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    white: PlayerModel
    black: PlayerModel


class MoveRefModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uci: str
    san: str


class CentipawnScoreModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["cp"]
    cp: int


class MateScoreModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["mate"]
    mate_in: int
    winner: Literal["white", "black"]


ScoreModel = Annotated[CentipawnScoreModel | MateScoreModel, Field(discriminator="type")]


class EngineAnalysisModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "terminal"]
    depth: int | None
    score: ScoreModel
    best_move: MoveRefModel | None
    pv: list[MoveRefModel]
    nodes: int | None
    time_ms: int


class MotifPieceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    color: Literal["white", "black"]
    role: Literal["pawn", "knight", "bishop", "rook", "queen"]
    square: str


class MotifEvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threshold_cp: int
    score_kind: Literal["cp", "mate"]
    phase: Literal["opening", "middlegame", "endgame"]
    piece: MotifPieceModel | None
    attackers: list[str]
    defenders: list[str]
    best_move: MoveRefModel | None
    opponent_reply: MoveRefModel | None
    related_ply: int | None


class MotifModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Literal[
        "hanging_piece",
        "missed_tactic",
        "allowed_tactic",
        "endgame_slip",
        "opening_inaccuracy",
    ]
    label: str
    severity: Literal["inaccuracy", "mistake", "blunder"]
    source: Literal["heuristic"]
    score_cp: int | None
    evidence: MotifEvidenceModel


class AnnotatedMoveModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ply: int
    move_number: int
    side: Literal["white", "black"]
    san: str
    uci: str
    from_square: str
    to_square: str
    promotion: Literal["q", "r", "b", "n"] | None
    fen_before: str
    fen_after: str
    analysis_before: EngineAnalysisModel
    analysis_after: EngineAnalysisModel
    eval_delta_cp_white: int | None
    loss_cp: int | None
    is_engine_best: bool
    motifs: list[MotifModel]


class AnalysisSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: str
    depth: int
    positions_evaluated: int
    elapsed_ms: int


class AnnotatedGameModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["annotated-game.v1"]
    game_id: str
    headers: dict[str, str]
    players: PlayersModel
    result: Literal["1-0", "0-1", "1/2-1/2", "*"]
    initial_fen: str
    final_fen: str
    analysis: AnalysisSummaryModel
    moves: list[AnnotatedMoveModel]


@router.post("", response_model=AnnotatedGameModel)
async def create_annotated_game(
    payload: CreateGameRequest,
    request: Request,
) -> AnnotatedGameModel | JSONResponse:
    """Parse and analyze a PGN, returning a complete raw review."""

    if len(payload.pgn) > MAX_PGN_CHARS:
        return _error_response(
            413,
            "pgn_too_large",
            f"PGN must be {MAX_PGN_CHARS} characters or fewer.",
            {"max_characters": str(MAX_PGN_CHARS)},
        )

    try:
        parsed_game = parse_pgn(payload.pgn, max_plies=MAX_PLIES)
    except PgnParseError as exc:
        return _pgn_error_response(exc)

    pool = cast(StockfishPool | None, getattr(request.app.state, "stockfish_pool", None))
    if pool is None or not pool.started:
        message = str(getattr(request.app.state, "stockfish_error", "Stockfish is unavailable."))
        return _error_response(503, "stockfish_unavailable", message)

    review_lock = cast(asyncio.Lock, request.app.state.review_lock)
    if review_lock.locked():
        return _error_response(
            429,
            "analysis_busy",
            "Another game review is already running. Try again in a moment.",
        )

    await review_lock.acquire()
    try:
        return await asyncio.wait_for(
            _annotate_game(parsed_game, pool=pool, depth=pool.depth),
            timeout=_analysis_timeout_seconds(),
        )
    except TimeoutError:
        return _error_response(
            504,
            "analysis_timeout",
            "Stockfish analysis exceeded the Slice 1 wall-clock budget.",
        )
    except StockfishUnavailableError as exc:
        return _error_response(503, "stockfish_unavailable", str(exc))
    except StockfishProtocolError as exc:
        return _error_response(500, "engine_protocol_error", str(exc))
    finally:
        review_lock.release()


async def _annotate_game(
    parsed_game: ParsedPgnGame,
    *,
    pool: StockfishPool,
    depth: int,
) -> AnnotatedGameModel:
    started_at = time.perf_counter()
    unique_fens = _unique_positions(parsed_game)
    evaluations = await _evaluate_positions(unique_fens, pool=pool, depth=depth)

    analyzed_moves = [
        _analyzed_move(move, evaluations[move.fen_before], evaluations[move.fen_after])
        for move in parsed_game.moves
    ]
    motif_lists = classify_moves(analyzed_moves, initial_fen=parsed_game.initial_fen)
    annotated_moves = [
        _annotate_move(
            move,
            evaluations[move.fen_before],
            evaluations[move.fen_after],
            motifs,
        )
        for move, motifs in zip(parsed_game.moves, motif_lists, strict=True)
    ]
    elapsed_ms = round((time.perf_counter() - started_at) * 1000)

    return AnnotatedGameModel(
        schema_version="annotated-game.v1",
        game_id=_game_id(parsed_game.normalized_pgn),
        headers=parsed_game.headers,
        players=_players(parsed_game.headers),
        result=_result(parsed_game.result),
        initial_fen=parsed_game.initial_fen,
        final_fen=parsed_game.final_fen,
        analysis=AnalysisSummaryModel(
            engine="Stockfish",
            depth=depth,
            positions_evaluated=len(unique_fens),
            elapsed_ms=elapsed_ms,
        ),
        moves=annotated_moves,
    )


async def _evaluate_positions(
    fens: list[str],
    *,
    pool: StockfishPool,
    depth: int,
) -> dict[str, EngineEvaluation]:
    async def evaluate_one(fen: str) -> tuple[str, EngineEvaluation]:
        return fen, await pool.evaluate(fen, depth=depth)

    tasks = [asyncio.create_task(evaluate_one(fen)) for fen in fens]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    return dict(results)


def _unique_positions(parsed_game: ParsedPgnGame) -> list[str]:
    seen: set[str] = set()
    fens: list[str] = []
    for fen in [parsed_game.initial_fen, *(move.fen_after for move in parsed_game.moves)]:
        if fen not in seen:
            seen.add(fen)
            fens.append(fen)
    return fens


def _annotate_move(
    move: ParsedPgnMove,
    analysis_before: EngineEvaluation,
    analysis_after: EngineEvaluation,
    motifs: Sequence[ClassifiedMotif],
) -> AnnotatedMoveModel:
    return AnnotatedMoveModel(
        ply=move.ply,
        move_number=move.move_number,
        side=move.side,
        san=move.san,
        uci=move.uci,
        from_square=move.from_square,
        to_square=move.to_square,
        promotion=move.promotion,
        fen_before=move.fen_before,
        fen_after=move.fen_after,
        analysis_before=_engine_analysis_model(analysis_before),
        analysis_after=_engine_analysis_model(analysis_after),
        eval_delta_cp_white=_eval_delta_cp_white(analysis_before.score, analysis_after.score),
        loss_cp=_loss_cp(move.side, analysis_before.score, analysis_after.score),
        is_engine_best=(
            analysis_before.best_move is not None and analysis_before.best_move.uci == move.uci
        ),
        motifs=[_motif_model(motif) for motif in motifs],
    )


def _analyzed_move(
    move: ParsedPgnMove,
    analysis_before: EngineEvaluation,
    analysis_after: EngineEvaluation,
) -> AnalyzedMove:
    return AnalyzedMove(
        ply=move.ply,
        move_number=move.move_number,
        side=move.side,
        san=move.san,
        uci=move.uci,
        fen_before=move.fen_before,
        fen_after=move.fen_after,
        analysis_before=analysis_before,
        analysis_after=analysis_after,
    )


def _engine_analysis_model(evaluation: EngineEvaluation) -> EngineAnalysisModel:
    return EngineAnalysisModel(
        status=evaluation.status,
        depth=evaluation.depth,
        score=_score_model(evaluation.score),
        best_move=_move_ref_model(evaluation.best_move),
        pv=[_required_move_ref_model(move) for move in evaluation.pv],
        nodes=evaluation.nodes,
        time_ms=evaluation.time_ms,
    )


def _score_model(score: EngineScore) -> ScoreModel:
    if isinstance(score, CentipawnScore):
        return CentipawnScoreModel(type="cp", cp=score.cp)
    if isinstance(score, MateScore):
        return MateScoreModel(type="mate", mate_in=score.mate_in, winner=score.winner)
    raise TypeError(f"Unsupported score type: {type(score).__name__}")


def _move_ref_model(move: EngineMove | None) -> MoveRefModel | None:
    if move is None:
        return None
    return _required_move_ref_model(move)


def _required_move_ref_model(move: EngineMove) -> MoveRefModel:
    return MoveRefModel(uci=move.uci, san=move.san)


def _motif_model(motif: ClassifiedMotif) -> MotifModel:
    evidence = motif.evidence
    piece = evidence.piece
    return MotifModel(
        id=motif.id,
        label=motif.label,
        severity=motif.severity,
        source=motif.source,
        score_cp=motif.score_cp,
        evidence=MotifEvidenceModel(
            threshold_cp=evidence.threshold_cp,
            score_kind=evidence.score_kind,
            phase=evidence.phase,
            piece=(
                MotifPieceModel(color=piece.color, role=piece.role, square=piece.square)
                if piece is not None
                else None
            ),
            attackers=list(evidence.attackers),
            defenders=list(evidence.defenders),
            best_move=(
                MoveRefModel(uci=evidence.best_move.uci, san=evidence.best_move.san)
                if evidence.best_move is not None
                else None
            ),
            opponent_reply=(
                MoveRefModel(uci=evidence.opponent_reply.uci, san=evidence.opponent_reply.san)
                if evidence.opponent_reply is not None
                else None
            ),
            related_ply=evidence.related_ply,
        ),
    )


def _eval_delta_cp_white(before: EngineScore, after: EngineScore) -> int | None:
    if isinstance(before, CentipawnScore) and isinstance(after, CentipawnScore):
        return after.cp - before.cp
    return None


def _loss_cp(
    side: Literal["white", "black"], before: EngineScore, after: EngineScore
) -> int | None:
    delta = _eval_delta_cp_white(before, after)
    if delta is None:
        return None
    if side == "white":
        return max(0, -delta)
    return max(0, delta)


def _players(headers: dict[str, str]) -> PlayersModel:
    return PlayersModel(
        white=PlayerModel(name=headers.get("White"), elo=_optional_elo(headers.get("WhiteElo"))),
        black=PlayerModel(name=headers.get("Black"), elo=_optional_elo(headers.get("BlackElo"))),
    )


def _optional_elo(value: str | None) -> int | None:
    if value is None or value in {"", "?"}:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _result(result: str) -> Literal["1-0", "0-1", "1/2-1/2", "*"]:
    if result in {"1-0", "0-1", "1/2-1/2", "*"}:
        return cast(Literal["1-0", "0-1", "1/2-1/2", "*"], result)
    return "*"


def _game_id(normalized_pgn: str) -> str:
    return f"sha256:{hashlib.sha256(normalized_pgn.encode('utf-8')).hexdigest()}"


def _analysis_timeout_seconds() -> float:
    value = os.environ.get("CHESS_ML_ANALYSIS_TIMEOUT_SECONDS")
    if value is None or not value.strip():
        return DEFAULT_ANALYSIS_TIMEOUT_SECONDS
    return float(value)


def _pgn_error_response(exc: PgnParseError) -> JSONResponse:
    status_code = 413 if exc.code in {"pgn_too_large", "too_many_plies"} else 400
    return _error_response(status_code, exc.code, exc.message, exc.details)


def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, str] | None = None,
) -> JSONResponse:
    envelope = ErrorEnvelope(error=ErrorBody(code=code, message=message, details=details or {}))
    return JSONResponse(status_code=status_code, content=envelope.model_dump())
