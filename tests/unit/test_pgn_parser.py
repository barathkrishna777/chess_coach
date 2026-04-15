"""Tests for PGN ingestion."""

import pytest

from chess_ml.ingestion.pgn import PgnParseError, parse_pgn


def test_parse_pgn_returns_mainline_positions() -> None:
    pgn = """
[Event "Casual"]
[White "Ada"]
[Black "Turing"]
[WhiteElo "1500"]
[BlackElo "1600"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
"""

    parsed = parse_pgn(pgn)

    assert parsed.headers["White"] == "Ada"
    assert parsed.headers["Black"] == "Turing"
    assert parsed.result == "1-0"
    assert len(parsed.moves) == 6
    assert parsed.moves[0].move_number == 1
    assert parsed.moves[0].side == "white"
    assert parsed.moves[0].san == "e4"
    assert parsed.moves[0].uci == "e2e4"
    assert parsed.moves[0].from_square == "e2"
    assert parsed.moves[0].to_square == "e4"
    assert parsed.moves[0].fen_before == parsed.initial_fen
    assert parsed.moves[-1].san == "a6"
    assert parsed.moves[-1].fen_after == parsed.final_fen


def test_parse_pgn_rejects_multiple_games() -> None:
    pgn = """
[Result "*"]

1. e4 *

[Result "*"]

1. d4 *
"""

    with pytest.raises(PgnParseError) as exc_info:
        parse_pgn(pgn)

    assert exc_info.value.code == "multiple_games_not_supported"


def test_parse_pgn_rejects_unsupported_variant() -> None:
    pgn = """
[Variant "Three-check"]
[Result "*"]

1. e4 *
"""

    with pytest.raises(PgnParseError) as exc_info:
        parse_pgn(pgn)

    assert exc_info.value.code == "unsupported_variant"
