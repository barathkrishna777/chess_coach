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
from chess_ml.classifier.motifs import AnalyzedMove, MotifEvidence, MoveRef, PieceRef
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
from chess_ml.explanation.models import ExplanationRequest, LineMove, MoveExplanation
from chess_ml.explanation.service import ExplanationService, ExplanationServiceStatus
from chess_ml.ingestion.pgn import ParsedPgnGame, ParsedPgnMove, PgnParseError, parse_pgn
from chess_ml.profile.store import (
    ProfileGameReview,
    ProfileMotifOccurrence,
    ProfilePlayer,
    ProfilePlayers,
    ProfileStore,
)

router = APIRouter(prefix="/api/games", tags=["games"])

MAX_PGN_CHARS = 200_000
MAX_PLIES = 160
DEFAULT_ANALYSIS_TIMEOUT_SECONDS = 14.5


class AnalysisTimeoutError(TimeoutError):
    """Raised when Stockfish analysis exceeds the review budget."""


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
    source: Literal["heuristic", "learned", "ensemble"]
    score_cp: int | None
    evidence: MotifEvidenceModel


class MoveExplanationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["move-explanation.v1"]
    status: Literal["ok", "unavailable", "error"]
    text: str | None
    source: Literal["cache", "llm", "fallback"] | None
    provider: Literal["anthropic", "codex", "ollama"] | None
    model: str | None
    prompt_version: Literal["grounded-coach.v1"]
    reason: (
        Literal[
            "api_key_missing",
            "provider_error",
            "invalid_response",
            "local_model_unavailable",
            "timeout",
        ]
        | None
    )
    timeout_seconds: float | None = None
    retryable: bool = False


class ExplanationStatusModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["explanation-status.v1"]
    enabled: bool
    configured: bool
    provider: Literal["anthropic", "codex", "ollama"] | None
    model: str | None
    timeout_seconds: float
    availability: Literal["not_checked"]
    reason: Literal["disabled", "api_key_missing", "unknown_provider"] | None


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
    explanation: MoveExplanationModel | None


class ExplainMoveRequest(BaseModel):
    """Payload for lazily explaining one already analyzed move."""

    model_config = ConfigDict(extra="forbid")

    move: AnnotatedMoveModel
    actual_line: list[MoveRefModel] = Field(default_factory=list, max_length=6)


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
        annotated_game = await _annotate_game(
            parsed_game,
            pool=pool,
            depth=pool.depth,
        )
        profile_store = cast(ProfileStore | None, getattr(request.app.state, "profile_store", None))
        if profile_store is not None:
            profile_store.save_review(_profile_review(annotated_game))
        return annotated_game
    except AnalysisTimeoutError:
        return _error_response(
            504,
            "analysis_timeout",
            "Stockfish analysis exceeded the wall-clock budget.",
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
    try:
        evaluations = await asyncio.wait_for(
            _evaluate_positions(unique_fens, pool=pool, depth=depth),
            timeout=_analysis_timeout_seconds(),
        )
    except TimeoutError as exc:
        raise AnalysisTimeoutError from exc

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
            None,
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


@router.post("/explain", response_model=MoveExplanationModel)
async def explain_move(
    payload: ExplainMoveRequest,
    request: Request,
) -> MoveExplanationModel:
    """Generate a coaching explanation for one selected flagged move."""

    explanation_service = cast(ExplanationService, request.app.state.explanation_service)
    explanation = await explanation_service.explain(
        _explanation_request_from_model(payload.move, payload.actual_line)
    )
    if explanation is None:
        explanation = MoveExplanation(
            status="unavailable",
            text=None,
            source=None,
            provider=None,
            model=None,
            reason="api_key_missing",
        )
    model = _explanation_model(explanation)
    if model is None:
        raise RuntimeError("Explanation model cannot be None for explain endpoint.")
    return model


@router.get("/explain/status", response_model=ExplanationStatusModel)
async def explanation_status(request: Request) -> ExplanationStatusModel:
    """Return local coach configuration without probing the provider."""

    explanation_service = cast(ExplanationService, request.app.state.explanation_service)
    return _explanation_status_model(explanation_service.status())


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
    explanation: MoveExplanation | None,
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
        explanation=_explanation_model(explanation),
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


def _explanation_request(
    move: ParsedPgnMove,
    analysis_before: EngineEvaluation,
    analysis_after: EngineEvaluation,
    motifs: Sequence[ClassifiedMotif],
) -> ExplanationRequest:
    return ExplanationRequest(
        ply=move.ply,
        move_number=move.move_number,
        side=move.side,
        san=move.san,
        uci=move.uci,
        fen_before=move.fen_before,
        fen_after=move.fen_after,
        analysis_before=analysis_before,
        analysis_after=analysis_after,
        loss_cp=_loss_cp(move.side, analysis_before.score, analysis_after.score),
        actual_line=(),
        motifs=tuple(motifs),
    )


def _explanation_request_from_model(
    move: AnnotatedMoveModel,
    actual_line: Sequence[MoveRefModel],
) -> ExplanationRequest:
    return ExplanationRequest(
        ply=move.ply,
        move_number=move.move_number,
        side=move.side,
        san=move.san,
        uci=move.uci,
        fen_before=move.fen_before,
        fen_after=move.fen_after,
        analysis_before=_engine_evaluation(move.analysis_before),
        analysis_after=_engine_evaluation(move.analysis_after),
        loss_cp=move.loss_cp,
        actual_line=tuple(
            _line_move(move, index, line_move) for index, line_move in enumerate(actual_line)
        ),
        motifs=tuple(_classified_motif(motif) for motif in move.motifs),
    )


def _line_move(selected: AnnotatedMoveModel, index: int, move: MoveRefModel) -> LineMove:
    ply = selected.ply + index
    return LineMove(
        ply=ply,
        side="white" if ply % 2 == 1 else "black",
        san=move.san,
        uci=move.uci,
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


def _engine_evaluation(model: EngineAnalysisModel) -> EngineEvaluation:
    return EngineEvaluation(
        status=model.status,
        depth=model.depth,
        score=_engine_score(model.score),
        best_move=_engine_move(model.best_move),
        pv=tuple(_required_engine_move(move) for move in model.pv),
        nodes=model.nodes,
        time_ms=model.time_ms,
    )


def _score_model(score: EngineScore) -> ScoreModel:
    if isinstance(score, CentipawnScore):
        return CentipawnScoreModel(type="cp", cp=score.cp)
    if isinstance(score, MateScore):
        return MateScoreModel(type="mate", mate_in=score.mate_in, winner=score.winner)
    raise TypeError(f"Unsupported score type: {type(score).__name__}")


def _engine_score(score: ScoreModel) -> EngineScore:
    if score.type == "cp":
        return CentipawnScore(cp=score.cp)
    return MateScore(mate_in=score.mate_in, winner=score.winner)


def _move_ref_model(move: EngineMove | None) -> MoveRefModel | None:
    if move is None:
        return None
    return _required_move_ref_model(move)


def _required_move_ref_model(move: EngineMove) -> MoveRefModel:
    return MoveRefModel(uci=move.uci, san=move.san)


def _engine_move(move: MoveRefModel | None) -> EngineMove | None:
    if move is None:
        return None
    return _required_engine_move(move)


def _required_engine_move(move: MoveRefModel) -> EngineMove:
    return EngineMove(uci=move.uci, san=move.san)


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


def _classified_motif(motif: MotifModel) -> ClassifiedMotif:
    evidence = motif.evidence
    piece = evidence.piece
    return ClassifiedMotif(
        id=motif.id,
        label=motif.label,
        severity=motif.severity,
        source=motif.source,
        score_cp=motif.score_cp,
        evidence=MotifEvidence(
            threshold_cp=evidence.threshold_cp,
            score_kind=evidence.score_kind,
            phase=evidence.phase,
            piece=(
                PieceRef(color=piece.color, role=piece.role, square=piece.square)
                if piece is not None
                else None
            ),
            attackers=tuple(evidence.attackers),
            defenders=tuple(evidence.defenders),
            best_move=(
                MoveRef(uci=evidence.best_move.uci, san=evidence.best_move.san)
                if evidence.best_move is not None
                else None
            ),
            opponent_reply=(
                MoveRef(uci=evidence.opponent_reply.uci, san=evidence.opponent_reply.san)
                if evidence.opponent_reply is not None
                else None
            ),
            related_ply=evidence.related_ply,
        ),
    )


def _explanation_model(explanation: MoveExplanation | None) -> MoveExplanationModel | None:
    if explanation is None:
        return None
    return MoveExplanationModel(
        schema_version="move-explanation.v1",
        status=explanation.status,
        text=explanation.text,
        source=explanation.source,
        provider=explanation.provider,
        model=explanation.model,
        prompt_version="grounded-coach.v1",
        reason=explanation.reason,
        timeout_seconds=explanation.timeout_seconds,
        retryable=explanation.retryable,
    )


def _explanation_status_model(status: ExplanationServiceStatus) -> ExplanationStatusModel:
    return ExplanationStatusModel(
        schema_version="explanation-status.v1",
        enabled=status.enabled,
        configured=status.configured,
        provider=status.provider,
        model=status.model,
        timeout_seconds=status.timeout_seconds,
        availability=status.availability,
        reason=status.reason,
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


def _profile_review(game: AnnotatedGameModel) -> ProfileGameReview:
    return ProfileGameReview(
        game_id=game.game_id,
        players=ProfilePlayers(
            white=ProfilePlayer(name=game.players.white.name, elo=game.players.white.elo),
            black=ProfilePlayer(name=game.players.black.name, elo=game.players.black.elo),
        ),
        result=game.result,
        source=_profile_source(game.headers),
        ply_count=len(game.moves),
        motif_occurrences=tuple(
            ProfileMotifOccurrence(
                ply=move.ply,
                move_number=move.move_number,
                side=move.side,
                san=move.san,
                uci=move.uci,
                motif_id=motif.id,
                motif_label=motif.label,
                severity=motif.severity,
                phase=motif.evidence.phase,
                loss_cp=move.loss_cp,
                score_cp=motif.score_cp,
            )
            for move in game.moves
            for motif in move.motifs
        ),
    )


def _profile_source(headers: dict[str, str]) -> Literal["pgn_upload", "local_play"]:
    event = headers.get("Event", "").strip().lower()
    if event == "chess_ml local play":
        return "local_play"
    return "pgn_upload"


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
