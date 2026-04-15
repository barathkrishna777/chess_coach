"""Local play API routes."""

from __future__ import annotations

from typing import Literal, cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from chess_ml.engine.opponent import OpponentMoveProvider
from chess_ml.engine.stockfish import StockfishProtocolError, StockfishUnavailableError
from chess_ml.play.session import (
    IllegalMoveError,
    InMemoryPlayStore,
    LegalMoveGroup,
    PlayGameFinishedError,
    PlayGameNotFoundError,
    PlayMove,
    PlayState,
)

router = APIRouter(prefix="/api/play", tags=["play"])


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


class PlayMoveModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ply: int
    side: Literal["white", "black"]
    san: str
    uci: str


class LegalMoveDestinationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    to_square: str
    promotions: list[Literal["q", "r", "b", "n"]]


class LegalMoveGroupModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_square: str
    destinations: list[LegalMoveDestinationModel]


class PlayStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["play-state.v1"]
    game_id: str
    status: Literal["active", "completed", "resigned"]
    result: Literal["1-0", "0-1", "1/2-1/2", "*"]
    fen: str
    orientation: Literal["white"]
    legal_moves: list[LegalMoveGroupModel]
    moves: list[PlayMoveModel]
    bot_move: MoveRefModel | None
    pgn: str | None


class SubmitMoveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game_id: str
    uci: str


class ResignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game_id: str


@router.post("/new", response_model=PlayStateModel)
async def new_game(request: Request) -> PlayStateModel | JSONResponse:
    """Start a new white-only play session."""

    if _opponent(request) is None:
        return _opponent_unavailable_response(request)

    store = _store(request)
    session = store.create()
    return _state_model(session.state(bot_move=None))


@router.post("/move", response_model=PlayStateModel)
async def submit_move(
    payload: SubmitMoveRequest,
    request: Request,
) -> PlayStateModel | JSONResponse:
    """Apply a user move and return the bot reply if the game continues."""

    opponent = _opponent(request)
    if opponent is None:
        return _opponent_unavailable_response(request)

    try:
        session = _store(request).get(payload.game_id)
        state = await session.apply_user_move(payload.uci, opponent=opponent)
    except PlayGameNotFoundError:
        return _error_response(
            404,
            "game_not_found",
            "That play session no longer exists. Start a new game.",
        )
    except PlayGameFinishedError:
        return _error_response(409, "game_already_finished", "This game has already finished.")
    except IllegalMoveError as exc:
        return _error_response(400, "illegal_move", str(exc))
    except (StockfishUnavailableError, StockfishProtocolError) as exc:
        return _error_response(503, "opponent_unavailable", str(exc))

    return _state_model(state)


@router.post("/resign", response_model=PlayStateModel)
async def resign(
    payload: ResignRequest,
    request: Request,
) -> PlayStateModel | JSONResponse:
    """Resign a local play session."""

    try:
        session = _store(request).get(payload.game_id)
        state = session.resign()
    except PlayGameNotFoundError:
        return _error_response(
            404,
            "game_not_found",
            "That play session no longer exists. Start a new game.",
        )
    except PlayGameFinishedError:
        return _error_response(409, "game_already_finished", "This game has already finished.")

    return _state_model(state)


def _state_model(state: PlayState) -> PlayStateModel:
    return PlayStateModel(
        schema_version="play-state.v1",
        game_id=state.game_id,
        status=state.status,
        result=state.result,
        fen=state.fen,
        orientation=state.orientation,
        legal_moves=[_legal_move_group_model(group) for group in state.legal_moves],
        moves=[_play_move_model(move) for move in state.moves],
        bot_move=_move_ref_model(state.bot_move),
        pgn=state.pgn,
    )


def _legal_move_group_model(group: LegalMoveGroup) -> LegalMoveGroupModel:
    return LegalMoveGroupModel(
        from_square=group.from_square,
        destinations=[
            LegalMoveDestinationModel(
                to_square=destination.to_square,
                promotions=list(destination.promotions),
            )
            for destination in group.destinations
        ],
    )


def _play_move_model(move: PlayMove) -> PlayMoveModel:
    return PlayMoveModel(ply=move.ply, side=move.side, san=move.san, uci=move.uci)


def _move_ref_model(move: PlayMove | None) -> MoveRefModel | None:
    if move is None:
        return None
    return MoveRefModel(uci=move.uci, san=move.san)


def _store(request: Request) -> InMemoryPlayStore:
    return cast(InMemoryPlayStore, request.app.state.play_store)


def _opponent(request: Request) -> OpponentMoveProvider | None:
    return cast(OpponentMoveProvider | None, getattr(request.app.state, "play_opponent", None))


def _opponent_unavailable_response(request: Request) -> JSONResponse:
    message = str(getattr(request.app.state, "play_opponent_error", "Play opponent unavailable."))
    return _error_response(503, "opponent_unavailable", message)


def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, str] | None = None,
) -> JSONResponse:
    envelope = ErrorEnvelope(error=ErrorBody(code=code, message=message, details=details or {}))
    return JSONResponse(status_code=status_code, content=envelope.model_dump())
