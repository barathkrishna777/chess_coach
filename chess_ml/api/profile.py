"""Local profile dashboard API routes."""

from __future__ import annotations

from typing import Literal, cast

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict

from chess_ml.profile.store import (
    MotifAggregate,
    OpeningAggregate,
    OpeningTopMotif,
    PhaseAggregate,
    ProfileDashboard,
    ProfileOpeningTag,
    ProfilePlayer,
    ProfilePlayers,
    ProfileStore,
    ProfileTotals,
    RecentProfileGame,
)

router = APIRouter(prefix="/api/profile", tags=["profile"])


class ProfilePlayerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None
    elo: int | None


class ProfilePlayersModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    white: ProfilePlayerModel
    black: ProfilePlayerModel


class OpeningTagModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eco: str
    name: str


class ProfileTotalsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    games_reviewed: int
    moves_reviewed: int
    flagged_moves: int
    motif_occurrences: int
    motif_rate_per_100_moves: float


class MotifAggregateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    count: int
    rate_per_100_moves: float


class PhaseAggregateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: Literal["opening", "middlegame", "endgame"]
    count: int
    rate_per_100_moves: float


class OpeningTopMotifModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    count: int


class OpeningAggregateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eco: str
    name: str
    games: int
    avg_loss_cp: float
    top_motif: OpeningTopMotifModel | None


class RecentProfileGameModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game_id: str
    players: ProfilePlayersModel
    result: Literal["1-0", "0-1", "1/2-1/2", "*"]
    source: Literal["pgn_upload", "local_play"]
    created_at: str
    updated_at: str
    ply_count: int
    flagged_moves: int
    opening: OpeningTagModel | None


class ProfileDashboardModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["profile-dashboard.v1"]
    totals: ProfileTotalsModel
    motifs: list[MotifAggregateModel]
    phase_breakdown: list[PhaseAggregateModel]
    openings: list[OpeningAggregateModel]
    recent_games: list[RecentProfileGameModel]


@router.get("/me", response_model=ProfileDashboardModel)
async def profile_me(request: Request) -> ProfileDashboardModel:
    """Return the single local profile dashboard."""

    store = cast(ProfileStore, request.app.state.profile_store)
    return _dashboard_model(store.dashboard())


def _dashboard_model(dashboard: ProfileDashboard) -> ProfileDashboardModel:
    return ProfileDashboardModel(
        schema_version="profile-dashboard.v1",
        totals=_totals_model(dashboard.totals),
        motifs=[_motif_model(motif) for motif in dashboard.motifs],
        phase_breakdown=[_phase_model(phase) for phase in dashboard.phase_breakdown],
        openings=[_opening_aggregate_model(opening) for opening in dashboard.openings],
        recent_games=[_recent_game_model(game) for game in dashboard.recent_games],
    )


def _totals_model(totals: ProfileTotals) -> ProfileTotalsModel:
    return ProfileTotalsModel(
        games_reviewed=totals.games_reviewed,
        moves_reviewed=totals.moves_reviewed,
        flagged_moves=totals.flagged_moves,
        motif_occurrences=totals.motif_occurrences,
        motif_rate_per_100_moves=totals.motif_rate_per_100_moves,
    )


def _motif_model(motif: MotifAggregate) -> MotifAggregateModel:
    return MotifAggregateModel(
        id=motif.id,
        label=motif.label,
        count=motif.count,
        rate_per_100_moves=motif.rate_per_100_moves,
    )


def _phase_model(phase: PhaseAggregate) -> PhaseAggregateModel:
    return PhaseAggregateModel(
        phase=phase.phase,
        count=phase.count,
        rate_per_100_moves=phase.rate_per_100_moves,
    )


def _opening_aggregate_model(opening: OpeningAggregate) -> OpeningAggregateModel:
    return OpeningAggregateModel(
        eco=opening.eco,
        name=opening.name,
        games=opening.games,
        avg_loss_cp=opening.avg_loss_cp,
        top_motif=_top_motif_model(opening.top_motif),
    )


def _top_motif_model(motif: OpeningTopMotif | None) -> OpeningTopMotifModel | None:
    if motif is None:
        return None
    return OpeningTopMotifModel(id=motif.id, label=motif.label, count=motif.count)


def _recent_game_model(game: RecentProfileGame) -> RecentProfileGameModel:
    return RecentProfileGameModel(
        game_id=game.game_id,
        players=_players_model(game.players),
        result=game.result,
        source=game.source,
        created_at=game.created_at,
        updated_at=game.updated_at,
        ply_count=game.ply_count,
        flagged_moves=game.flagged_moves,
        opening=_opening_tag_model(game.opening),
    )


def _opening_tag_model(opening: ProfileOpeningTag | None) -> OpeningTagModel | None:
    if opening is None:
        return None
    return OpeningTagModel(eco=opening.eco, name=opening.name)


def _players_model(players: ProfilePlayers) -> ProfilePlayersModel:
    return ProfilePlayersModel(
        white=_player_model(players.white),
        black=_player_model(players.black),
    )


def _player_model(player: ProfilePlayer) -> ProfilePlayerModel:
    return ProfilePlayerModel(name=player.name, elo=player.elo)
