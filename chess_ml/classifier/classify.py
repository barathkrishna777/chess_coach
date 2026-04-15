"""Classifier entrypoints for annotated game moves."""

from __future__ import annotations

from collections.abc import Sequence

from chess_ml.classifier.motifs import AnalyzedMove, Motif, detect_motifs


def classify_moves(
    moves: Sequence[AnalyzedMove],
    *,
    initial_fen: str,
) -> list[tuple[Motif, ...]]:
    """Classify each analyzed move with zero or more heuristic motifs."""

    return detect_motifs(moves, initial_fen=initial_fen)
