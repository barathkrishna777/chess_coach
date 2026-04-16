"""Local play API routes."""

from __future__ import annotations

from typing import Literal, cast

from fastapi import APIRouter, Body, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from chess_ml.engine.maia import DEFAULT_MAIA_RATING, MaiaRating, parse_maia_rating
from chess_ml.engine.opponent import (
    OpponentInfo,
    OpponentMoveProvider,
    PlayOpponentRegistry,
    PlayOpponentStatus,
    RequestedOpponent,
    SelectedOpponent,
)
from chess_ml.engine.stockfish import (
    EngineMove,
    StockfishPool,
    StockfishProtocolError,
    StockfishUnavailableError,
)
from chess_ml.play.session import (
    HintUnavailableError,
    IllegalMoveError,
    InMemoryPlayStore,
    LegalMoveGroup,
    PlayGameFinishedError,
    PlayGameNotFoundError,
    PlayMove,
    PlayState,
    Side,
    TakebackUnavailableError,
)

router = APIRouter(prefix="/api/play", tags=["play"])
NEW_GAME_BODY = Body(default=None)


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


class PlayOpponentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["maia", "stockfish"]
    requested: Literal["auto", "maia", "stockfish"]
    label: str
    engine: str
    maia_rating: Literal[1100, 1500, 1900] | None
    fallback_reason: str | None


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
    opponent: PlayOpponentModel
    user_color: Literal["white", "black"]
    status: Literal["active", "completed", "resigned"]
    result: Literal["1-0", "0-1", "1/2-1/2", "*"]
    fen: str
    orientation: Literal["white", "black"]
    legal_moves: list[LegalMoveGroupModel]
    moves: list[PlayMoveModel]
    bot_move: MoveRefModel | None
    hints_remaining: int
    takebacks_remaining: int
    pgn: str | None


class PlayHintModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["play-hint.v1"]
    game_id: str
    best_move: MoveRefModel
    from_square: str
    to_square: str
    promotion: Literal["q", "r", "b", "n"] | None
    hints_remaining: int


class NewGameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    opponent: Literal["auto", "maia", "stockfish"] = "auto"
    maia_rating: Literal[1100, 1500, 1900] = DEFAULT_MAIA_RATING
    user_color: Literal["white", "black"] = "white"


class SubmitMoveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game_id: str
    uci: str


class ResignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game_id: str


class TakebackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game_id: str


class MaiaOpponentStatusModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lc0_path: str | None
    lc0_available: bool
    weights_dir: str
    ratings: list[Literal[1100, 1500, 1900]]
    available_ratings: list[Literal[1100, 1500, 1900]]
    missing_weights: list[Literal[1100, 1500, 1900]]


class PlayOpponentStatusModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["play-opponents.v1"]
    default_requested: Literal["auto", "maia", "stockfish"]
    default_maia_rating: Literal[1100, 1500, 1900]
    stockfish_path: str
    stockfish_available: bool
    stockfish_label: str
    maia: MaiaOpponentStatusModel


@router.post("/new", response_model=PlayStateModel)
async def new_game(
    request: Request,
    payload: NewGameRequest | None = NEW_GAME_BODY,
) -> PlayStateModel | JSONResponse:
    """Start a new local play session."""

    requested = payload.opponent if payload is not None else "auto"
    rating = _maia_rating(payload.maia_rating if payload is not None else DEFAULT_MAIA_RATING)
    user_color: Side = payload.user_color if payload is not None else "white"

    try:
        selected = await _select_opponent(request, requested=requested, maia_rating=rating)
    except StockfishUnavailableError as exc:
        return _error_response(503, "opponent_unavailable", str(exc))

    store = _store(request)
    session = store.create(
        opponent=selected.info,
        opponent_provider=selected.provider,
        user_color=user_color,
    )
    try:
        return _state_model(await session.start())
    except (StockfishUnavailableError, StockfishProtocolError) as exc:
        return _error_response(503, "opponent_unavailable", str(exc))


@router.get("/opponents", response_model=PlayOpponentStatusModel)
async def opponents(request: Request) -> PlayOpponentStatusModel:
    """Return local opponent setup status without probing Maia."""

    registry = _registry(request)
    if registry is not None:
        return _opponent_status_model(registry.status())

    opponent = _opponent(request)
    return PlayOpponentStatusModel(
        schema_version="play-opponents.v1",
        default_requested="stockfish",
        default_maia_rating=DEFAULT_MAIA_RATING,
        stockfish_path="",
        stockfish_available=opponent is not None,
        stockfish_label="Stockfish fallback",
        maia=MaiaOpponentStatusModel(
            lc0_path=None,
            lc0_available=False,
            weights_dir="checkpoints/maia",
            ratings=[1100, 1500, 1900],
            available_ratings=[],
            missing_weights=[1100, 1500, 1900],
        ),
    )


@router.post("/move", response_model=PlayStateModel)
async def submit_move(
    payload: SubmitMoveRequest,
    request: Request,
) -> PlayStateModel | JSONResponse:
    """Apply a user move and return the bot reply if the game continues."""

    try:
        session = _store(request).get(payload.game_id)
        state = await session.apply_user_move(payload.uci)
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


@router.post("/takeback", response_model=PlayStateModel)
async def takeback(
    payload: TakebackRequest,
    request: Request,
) -> PlayStateModel | JSONResponse:
    """Undo the latest user move and bot reply if a takeback is available."""

    try:
        session = _store(request).get(payload.game_id)
        state = await session.takeback()
    except PlayGameNotFoundError:
        return _error_response(
            404,
            "game_not_found",
            "That play session no longer exists. Start a new game.",
        )
    except PlayGameFinishedError:
        return _error_response(
            409,
            "takeback_unavailable",
            "Takeback is only available while the game is in progress.",
        )
    except TakebackUnavailableError as exc:
        return _error_response(409, "takeback_unavailable", str(exc))
    except IllegalMoveError as exc:
        return _error_response(409, "takeback_unavailable", str(exc))

    return _state_model(state)


@router.get("/hint", response_model=PlayHintModel)
async def hint(
    session_id: str,
    request: Request,
) -> PlayHintModel | JSONResponse:
    """Return one Stockfish best-move hint for the current user turn."""

    try:
        session = _store(request).get(session_id)
    except PlayGameNotFoundError:
        return _error_response(
            404,
            "game_not_found",
            "That play session no longer exists. Start a new game.",
        )

    pool = _stockfish_pool(request)
    if pool is None:
        message = str(getattr(request.app.state, "stockfish_error", "Stockfish is unavailable."))
        return _error_response(503, "stockfish_unavailable", message)

    try:
        fen = await session.hint_fen()
        evaluation = await pool.evaluate(fen, depth=14)
        if evaluation.best_move is None:
            return _error_response(
                409,
                "hint_unavailable",
                "No legal hint is available in the current position.",
            )
        hints_remaining = await session.record_hint_used(fen)
    except PlayGameFinishedError:
        return _error_response(
            409,
            "hint_unavailable",
            "Hints are only available while the game is in progress.",
        )
    except HintUnavailableError as exc:
        return _error_response(409, "hint_unavailable", str(exc))
    except (StockfishUnavailableError, StockfishProtocolError) as exc:
        return _error_response(503, "stockfish_unavailable", str(exc))

    return _hint_model(
        game_id=session.game_id,
        move=evaluation.best_move,
        hints_remaining=hints_remaining,
    )


def _state_model(state: PlayState) -> PlayStateModel:
    return PlayStateModel(
        schema_version="play-state.v1",
        game_id=state.game_id,
        opponent=_opponent_model(state.opponent),
        user_color=state.user_color,
        status=state.status,
        result=state.result,
        fen=state.fen,
        orientation=state.orientation,
        legal_moves=[_legal_move_group_model(group) for group in state.legal_moves],
        moves=[_play_move_model(move) for move in state.moves],
        bot_move=_move_ref_model(state.bot_move),
        hints_remaining=state.hints_remaining,
        takebacks_remaining=state.takebacks_remaining,
        pgn=state.pgn,
    )


def _opponent_model(opponent: OpponentInfo) -> PlayOpponentModel:
    return PlayOpponentModel(
        kind=opponent.kind,
        requested=opponent.requested,
        label=opponent.label,
        engine=opponent.engine,
        maia_rating=opponent.maia_rating,
        fallback_reason=opponent.fallback_reason,
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


def _hint_model(*, game_id: str, move: EngineMove, hints_remaining: int) -> PlayHintModel:
    uci = move.uci
    san = move.san
    promotion = uci[4] if len(uci) == 5 else None
    if promotion not in {"q", "r", "b", "n"}:
        promotion = None
    return PlayHintModel(
        schema_version="play-hint.v1",
        game_id=game_id,
        best_move=MoveRefModel(uci=uci, san=san),
        from_square=uci[:2],
        to_square=uci[2:4],
        promotion=cast(Literal["q", "r", "b", "n"] | None, promotion),
        hints_remaining=hints_remaining,
    )


def _store(request: Request) -> InMemoryPlayStore:
    return cast(InMemoryPlayStore, request.app.state.play_store)


def _stockfish_pool(request: Request) -> StockfishPool | None:
    return cast(StockfishPool | None, getattr(request.app.state, "stockfish_pool", None))


async def _select_opponent(
    request: Request,
    *,
    requested: RequestedOpponent,
    maia_rating: MaiaRating,
) -> SelectedOpponent:
    registry = _registry(request)
    if registry is not None:
        return await registry.select(requested=requested, maia_rating=maia_rating)

    opponent = _opponent(request)
    if opponent is None:
        raise StockfishUnavailableError(
            str(getattr(request.app.state, "play_opponent_error", "Play opponent unavailable."))
        )
    return SelectedOpponent(
        provider=opponent,
        info=OpponentInfo(
            kind="stockfish",
            requested="stockfish" if requested == "stockfish" else requested,
            label="Stockfish fallback",
            engine="Stockfish",
            maia_rating=None,
            fallback_reason=None,
        ),
    )


def _registry(request: Request) -> PlayOpponentRegistry | None:
    return cast(PlayOpponentRegistry | None, getattr(request.app.state, "play_opponents", None))


def _opponent(request: Request) -> OpponentMoveProvider | None:
    return cast(OpponentMoveProvider | None, getattr(request.app.state, "play_opponent", None))


def _opponent_unavailable_response(request: Request) -> JSONResponse:
    message = str(getattr(request.app.state, "play_opponent_error", "Play opponent unavailable."))
    return _error_response(503, "opponent_unavailable", message)


def _opponent_status_model(status: PlayOpponentStatus) -> PlayOpponentStatusModel:
    return PlayOpponentStatusModel(
        schema_version="play-opponents.v1",
        default_requested=status.default_requested,
        default_maia_rating=status.default_maia_rating,
        stockfish_path=status.stockfish_path,
        stockfish_available=status.stockfish_available,
        stockfish_label=status.stockfish_label,
        maia=MaiaOpponentStatusModel(
            lc0_path=status.maia.lc0_path,
            lc0_available=status.maia.lc0_available,
            weights_dir=status.maia.weights_dir,
            ratings=list(status.maia.ratings),
            available_ratings=list(status.maia.available_ratings),
            missing_weights=list(status.maia.missing_weights),
        ),
    )


def _maia_rating(value: int) -> MaiaRating:
    return parse_maia_rating(value)


def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, str] | None = None,
) -> JSONResponse:
    envelope = ErrorEnvelope(error=ErrorBody(code=code, message=message, details=details or {}))
    return JSONResponse(status_code=status_code, content=envelope.model_dump())
