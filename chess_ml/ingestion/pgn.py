"""PGN parsing for game review ingestion."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal, cast

import chess
import chess.pgn

Side = Literal["white", "black"]
Promotion = Literal["q", "r", "b", "n"]


class PgnParseError(ValueError):
    """Raised when a submitted PGN cannot be accepted for review."""

    def __init__(self, code: str, message: str, details: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class ParsedPgnMove:
    """One legal mainline PGN move with positions before and after it."""

    ply: int
    move_number: int
    side: Side
    san: str
    uci: str
    from_square: str
    to_square: str
    promotion: Promotion | None
    fen_before: str
    fen_after: str


@dataclass(frozen=True)
class ParsedPgnGame:
    """A single standard chess game parsed from PGN."""

    normalized_pgn: str
    headers: dict[str, str]
    result: str
    initial_fen: str
    final_fen: str
    moves: tuple[ParsedPgnMove, ...]


def parse_pgn(pgn_text: str, *, max_plies: int = 160) -> ParsedPgnGame:
    """Parse exactly one standard PGN game into a mainline move list."""

    normalized_pgn = _normalize_pgn(pgn_text)
    if not normalized_pgn:
        raise PgnParseError("invalid_pgn", "PGN is empty.")

    stream = io.StringIO(normalized_pgn)
    game = chess.pgn.read_game(stream)
    if game is None:
        raise PgnParseError("invalid_pgn", "Could not parse PGN.")

    second_game = chess.pgn.read_game(stream)
    if second_game is not None:
        raise PgnParseError(
            "multiple_games_not_supported",
            "Upload exactly one game at a time.",
        )

    if game.errors:
        first_error = game.errors[0]
        raise PgnParseError(
            "invalid_pgn",
            "PGN contains illegal or malformed moves.",
            {"parser_error": str(first_error)},
        )

    headers = {str(key): str(value) for key, value in game.headers.items()}
    _validate_standard_variant(headers)

    board = game.board()
    if board.chess960:
        raise PgnParseError("unsupported_variant", "Chess960 PGNs are not supported in Slice 1.")

    initial_fen = board.fen()
    parsed_moves: list[ParsedPgnMove] = []

    for move in game.mainline_moves():
        if len(parsed_moves) >= max_plies:
            raise PgnParseError(
                "too_many_plies",
                f"PGN contains more than the Slice 1 limit of {max_plies} plies.",
                {"max_plies": str(max_plies)},
            )

        if move not in board.legal_moves:
            raise PgnParseError(
                "invalid_pgn",
                "PGN contains an illegal mainline move.",
                {"uci": move.uci(), "fen": board.fen()},
            )

        side: Side = "white" if board.turn == chess.WHITE else "black"
        fen_before = board.fen()
        san = board.san(move)
        move_number = board.fullmove_number

        board.push(move)
        parsed_moves.append(
            ParsedPgnMove(
                ply=len(parsed_moves) + 1,
                move_number=move_number,
                side=side,
                san=san,
                uci=move.uci(),
                from_square=chess.square_name(move.from_square),
                to_square=chess.square_name(move.to_square),
                promotion=_promotion_symbol(move),
                fen_before=fen_before,
                fen_after=board.fen(),
            )
        )

    if not parsed_moves:
        raise PgnParseError("invalid_pgn", "PGN must contain at least one move.")

    return ParsedPgnGame(
        normalized_pgn=normalized_pgn,
        headers=headers,
        result=headers.get("Result", "*"),
        initial_fen=initial_fen,
        final_fen=board.fen(),
        moves=tuple(parsed_moves),
    )


def _normalize_pgn(pgn_text: str) -> str:
    return "\n".join(line.rstrip() for line in pgn_text.strip().splitlines())


def _validate_standard_variant(headers: dict[str, str]) -> None:
    variant = headers.get("Variant", "").strip().lower()
    if variant and variant not in {"standard", "chess"}:
        raise PgnParseError(
            "unsupported_variant",
            f"Unsupported PGN variant: {headers['Variant']}.",
            {"variant": headers["Variant"]},
        )


def _promotion_symbol(move: chess.Move) -> Promotion | None:
    if move.promotion is None:
        return None
    return cast(Promotion, chess.piece_symbol(move.promotion))
