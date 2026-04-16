"""Personal drill mode API routes backed by reviewed profile positions."""

from __future__ import annotations

import base64
import json
from typing import Literal, cast

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from chess_ml.profile.store import (
    DrillContext,
    DrillMove,
    DrillNotFoundError,
    DrillPosition,
    DrillResult,
    DrillStats,
    ProfileStore,
)

router = APIRouter(prefix="/api/train", tags=["train"])


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, str] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class MoveRefModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uci: str
    san: str


class TrainingContextModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    motif_label: str
    phase: Literal["opening", "middlegame", "endgame"]
    loss_cp: int | None
    score_cp: int | None
    played_move: MoveRefModel
    pv: list[MoveRefModel]
    explanation_text: str | None
    explanation_status: str | None
    evidence: dict[str, object] | None


class TrainingDrillModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["training-drill.v1"]
    drill_id: str
    game_id: str
    ply: int
    move_number: int
    side: Literal["white", "black"]
    motif: str
    motif_label: str
    fen: str
    hint_text: str
    context: TrainingContextModel


class SubmitTrainingResultRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drill_id: str = Field(min_length=1)
    attempted_uci: str = Field(min_length=4, max_length=5)


class TrainingResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["training-result.v1"]
    correct: bool
    attempted_uci: str
    best_move: MoveRefModel
    next_due_at: str
    context: TrainingContextModel


class TrainingStatsTotalsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trainable_positions: int
    due_positions: int
    attempts: int
    correct_attempts: int


class TrainingMotifStatsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    motif: str
    motif_label: str
    trainable_positions: int
    due_positions: int
    attempts: int
    correct_attempts: int


class TrainingStatsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["training-stats.v1"]
    totals: TrainingStatsTotalsModel
    motifs: list[TrainingMotifStatsModel]


@router.get("/next", response_model=TrainingDrillModel)
async def next_training_drill(
    request: Request,
    motif: str | None = Query(default=None),
) -> TrainingDrillModel | JSONResponse:
    """Return the next due drill without revealing the answer."""

    drill = _store(request).next_drill(motif)
    if drill is None:
        return _error_response(
            404,
            "no_due_drill",
            "No due drills are available for that motif yet.",
        )
    return _training_drill_model(drill)


@router.post("/result", response_model=TrainingResultModel)
async def submit_training_result(
    payload: SubmitTrainingResultRequest,
    request: Request,
) -> TrainingResultModel | JSONResponse:
    """Record a drill attempt and reveal the gated answer."""

    try:
        drill_key = _decode_drill_id(payload.drill_id)
    except ValueError as exc:
        return _error_response(400, "invalid_drill_id", str(exc))

    try:
        result = _store(request).record_drill_attempt(
            game_id=drill_key.game_id,
            ply=drill_key.ply,
            motif=drill_key.motif,
            attempted_uci=payload.attempted_uci,
        )
    except DrillNotFoundError as exc:
        return _error_response(404, "drill_not_found", str(exc))

    return _training_result_model(result)


@router.get("/stats", response_model=TrainingStatsModel)
async def training_stats(request: Request) -> TrainingStatsModel:
    """Return local personal-drill progress."""

    return _training_stats_model(_store(request).drill_stats())


def _training_drill_model(drill: DrillPosition) -> TrainingDrillModel:
    return TrainingDrillModel(
        schema_version="training-drill.v1",
        drill_id=_encode_drill_id(drill.game_id, drill.ply, drill.motif),
        game_id=drill.game_id,
        ply=drill.ply,
        move_number=drill.move_number,
        side=drill.side,
        motif=drill.motif,
        motif_label=drill.motif_label,
        fen=drill.fen,
        hint_text=drill.hint_text,
        context=_context_model(drill.context, reveal_answer=False),
    )


def _training_result_model(result: DrillResult) -> TrainingResultModel:
    return TrainingResultModel(
        schema_version="training-result.v1",
        correct=result.correct,
        attempted_uci=result.attempted_uci,
        best_move=_move_model(result.best_move),
        next_due_at=result.next_due_at,
        context=_context_model(result.context, reveal_answer=True),
    )


def _training_stats_model(stats: DrillStats) -> TrainingStatsModel:
    return TrainingStatsModel(
        schema_version="training-stats.v1",
        totals=TrainingStatsTotalsModel(
            trainable_positions=stats.totals.trainable_positions,
            due_positions=stats.totals.due_positions,
            attempts=stats.totals.attempts,
            correct_attempts=stats.totals.correct_attempts,
        ),
        motifs=[
            TrainingMotifStatsModel(
                motif=motif.motif,
                motif_label=motif.motif_label,
                trainable_positions=motif.trainable_positions,
                due_positions=motif.due_positions,
                attempts=motif.attempts,
                correct_attempts=motif.correct_attempts,
            )
            for motif in stats.motifs
        ],
    )


def _context_model(context: DrillContext, *, reveal_answer: bool) -> TrainingContextModel:
    return TrainingContextModel(
        motif_label=context.motif_label,
        phase=context.phase,
        loss_cp=context.loss_cp,
        score_cp=context.score_cp,
        played_move=_move_model(context.played_move),
        pv=[_move_model(move) for move in context.pv] if reveal_answer else [],
        explanation_text=context.explanation_text if reveal_answer else None,
        explanation_status=context.explanation_status if reveal_answer else None,
        evidence=_evidence_payload(context.evidence, reveal_answer=reveal_answer),
    )


def _move_model(move: DrillMove) -> MoveRefModel:
    return MoveRefModel(uci=move.uci, san=move.san)


def _evidence_payload(
    evidence: dict[str, object] | None,
    *,
    reveal_answer: bool,
) -> dict[str, object] | None:
    if evidence is None:
        return None
    if reveal_answer:
        return evidence
    return {key: value for key, value in evidence.items() if key != "best_move"}


class _DrillKey(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game_id: str
    ply: int
    motif: str


def _encode_drill_id(game_id: str, ply: int, motif: str) -> str:
    payload = json.dumps(
        {"game_id": game_id, "ply": ply, "motif": motif},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_drill_id(value: str) -> _DrillKey:
    padding = "=" * (-len(value) % 4)
    try:
        raw = base64.urlsafe_b64decode(f"{value}{padding}".encode("ascii"))
        parsed = json.loads(raw.decode("utf-8"))
        return _DrillKey.model_validate(parsed)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError("The drill id is not valid.") from exc


def _store(request: Request) -> ProfileStore:
    return cast(ProfileStore, request.app.state.profile_store)


def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, str] | None = None,
) -> JSONResponse:
    envelope = ErrorEnvelope(error=ErrorBody(code=code, message=message, details=details or {}))
    return JSONResponse(status_code=status_code, content=envelope.model_dump())
