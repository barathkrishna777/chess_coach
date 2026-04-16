"""Compact ECO opening detection for reviewed games."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import chess

MAX_OPENING_PLIES = 20


@dataclass(frozen=True)
class OpeningTag:
    """Detected ECO code and human-readable opening name."""

    eco: str
    name: str


@dataclass(frozen=True)
class _OpeningLine:
    moves: tuple[str, ...]
    eco: str
    name: str


_ECO_LINES: tuple[_OpeningLine, ...] = (
    _OpeningLine(
        moves=("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"),
        eco="B90",
        name="Sicilian Defense: Najdorf Variation",
    ),
    _OpeningLine(
        moves=("e2e4", "e7e5", "d1h5", "g8f6", "f1c4"),
        eco="C20",
        name="Wayward Queen Attack",
    ),
    _OpeningLine(
        moves=("e2e4", "d7d5", "e4d5", "d8d5", "b1c3"),
        eco="B01",
        name="Scandinavian Defense",
    ),
    _OpeningLine(
        moves=("e2e4", "e7e5", "g1f3", "b8c6", "f1b5"),
        eco="C60",
        name="Ruy Lopez",
    ),
    _OpeningLine(
        moves=("d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"),
        eco="E60",
        name="King's Indian Defense",
    ),
    _OpeningLine(
        moves=("e2e4", "c7c5", "g1f3", "d7d6"),
        eco="B50",
        name="Sicilian Defense",
    ),
    _OpeningLine(
        moves=("e2e4", "e7e6"),
        eco="C00",
        name="French Defense",
    ),
    _OpeningLine(
        moves=("e2e4", "c7c6"),
        eco="B10",
        name="Caro-Kann Defense",
    ),
    _OpeningLine(
        moves=("d2d4", "d7d5", "c2c4"),
        eco="D06",
        name="Queen's Gambit",
    ),
    _OpeningLine(
        moves=("e2e4", "d7d5"),
        eco="B01",
        name="Scandinavian Defense",
    ),
    _OpeningLine(
        moves=("e2e4", "e7e5", "d1h5"),
        eco="C20",
        name="Wayward Queen Attack",
    ),
    _OpeningLine(
        moves=("e2e4", "e7e5"),
        eco="C20",
        name="King's Pawn Game",
    ),
    _OpeningLine(
        moves=("f2f3",),
        eco="A00",
        name="Barnes Opening",
    ),
)

_SORTED_ECO_LINES = tuple(sorted(_ECO_LINES, key=lambda line: len(line.moves), reverse=True))


def detect_opening(
    moves_uci: Sequence[str],
    *,
    initial_fen: str = chess.STARTING_FEN,
) -> OpeningTag | None:
    """Return the longest matching ECO line for a standard-start game."""

    if initial_fen != chess.STARTING_FEN:
        return None

    early_moves = tuple(moves_uci[:MAX_OPENING_PLIES])
    for line in _SORTED_ECO_LINES:
        if len(line.moves) <= len(early_moves) and early_moves[: len(line.moves)] == line.moves:
            return OpeningTag(eco=line.eco, name=line.name)
    return None
