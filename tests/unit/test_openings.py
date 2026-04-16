"""Tests for compact ECO opening detection."""

from __future__ import annotations

import chess

from chess_ml.classifier.openings import detect_opening


def test_detect_opening_chooses_longest_matching_line() -> None:
    opening = detect_opening(
        [
            "e2e4",
            "c7c5",
            "g1f3",
            "d7d6",
            "d2d4",
            "c5d4",
            "f3d4",
            "g8f6",
            "b1c3",
            "a7a6",
            "f1e2",
        ]
    )

    assert opening is not None
    assert opening.eco == "B90"
    assert opening.name == "Sicilian Defense: Najdorf Variation"


def test_detect_opening_skips_non_standard_initial_fen() -> None:
    opening = detect_opening(["e2e4", "d7d5"], initial_fen=chess.Board.empty().fen())

    assert opening is None
