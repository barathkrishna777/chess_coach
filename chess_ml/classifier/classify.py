"""Classifier entrypoints for annotated game moves."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import chess

from chess_ml.classifier.learned import LearnedMotifClassifier, learned_classifier_from_env
from chess_ml.classifier.motifs import (
    HANGING_PIECE_THRESHOLD_CP,
    MATE_EFFECTIVE_CP,
    OPENING_INACCURACY_EXCLUSIVE_MAX_CP,
    OPENING_INACCURACY_MIN_CP,
    TABLEBASE_RELEVANT_PIECES,
    TACTIC_THRESHOLD_CP,
    AnalyzedMove,
    GamePhase,
    Motif,
    MotifEvidence,
    MotifId,
    MotifSeverity,
    MoveRef,
    PieceRef,
    PieceRole,
    ScoreKind,
    detect_motifs,
)
from chess_ml.engine.stockfish import CentipawnScore, EngineMove, MateScore

_LABELS: dict[MotifId, str] = {
    "hanging_piece": "Hanging piece",
    "missed_tactic": "Missed tactic",
    "allowed_tactic": "Allowed tactic",
    "endgame_slip": "Endgame slip",
    "opening_inaccuracy": "Opening inaccuracy",
}


def classify_moves(
    moves: Sequence[AnalyzedMove],
    *,
    initial_fen: str,
    learned_classifier: LearnedMotifClassifier | None = None,
) -> list[tuple[Motif, ...]]:
    """Classify each analyzed move with heuristic motifs plus optional learned signals."""

    heuristic_results = detect_motifs(moves, initial_fen=initial_fen)
    classifier = (
        learned_classifier if learned_classifier is not None else learned_classifier_from_env()
    )
    if classifier is None:
        return heuristic_results

    learned_predictions = classifier.predict(list(moves))
    starts_from_standard_position = initial_fen == chess.STARTING_FEN
    combined: list[tuple[Motif, ...]] = []
    for move, heuristic_motifs, predictions in zip(
        moves,
        heuristic_results,
        learned_predictions,
        strict=True,
    ):
        by_id = {motif.id: motif for motif in heuristic_motifs}
        merged: list[Motif] = list(heuristic_motifs)
        for prediction in predictions:
            existing = by_id.get(prediction.label)
            if existing is not None:
                index = merged.index(existing)
                merged[index] = replace(existing, source="ensemble")
                continue

            learned_motif = _learned_motif(
                move,
                prediction.label,
                starts_from_standard_position=starts_from_standard_position,
            )
            if learned_motif is not None:
                merged.append(learned_motif)
                by_id[prediction.label] = learned_motif
        combined.append(tuple(merged))
    return combined


def _learned_motif(
    move: AnalyzedMove,
    motif_id: MotifId,
    *,
    starts_from_standard_position: bool,
) -> Motif | None:
    before_board = chess.Board(move.fen_before)
    after_board = chess.Board(move.fen_after)
    loss = _loss_for_move(move)
    phase = _phase(before_board, move.move_number)

    if motif_id == "opening_inaccuracy":
        if (
            not starts_from_standard_position
            or move.move_number > 15
            or loss.score_cp is None
            or not (
                OPENING_INACCURACY_MIN_CP <= loss.score_cp < OPENING_INACCURACY_EXCLUSIVE_MAX_CP
            )
        ):
            return None
        return _motif(move, motif_id, loss, phase, threshold=OPENING_INACCURACY_MIN_CP)

    if motif_id == "hanging_piece":
        if loss.effective_cp < HANGING_PIECE_THRESHOLD_CP:
            return None
        piece, attackers, defenders = _hanging_piece_evidence(
            after_board,
            _color_from_side(move.side),
        )
        if piece is None:
            return None
        return _motif(
            move,
            motif_id,
            loss,
            phase,
            threshold=HANGING_PIECE_THRESHOLD_CP,
            piece=piece,
            attackers=attackers,
            defenders=defenders,
        )

    if motif_id == "endgame_slip" and _piece_count(before_board) > TABLEBASE_RELEVANT_PIECES:
        return None
    if motif_id in {"missed_tactic", "endgame_slip"}:
        if loss.effective_cp < TACTIC_THRESHOLD_CP:
            return None
        best_move = _move_ref(move.analysis_before.best_move)
        if best_move is None or best_move.uci == move.uci:
            return None
        return _motif(move, motif_id, loss, phase, threshold=TACTIC_THRESHOLD_CP)

    if motif_id == "allowed_tactic":
        if loss.effective_cp < TACTIC_THRESHOLD_CP or move.analysis_after.best_move is None:
            return None
        return _motif(move, motif_id, loss, phase, threshold=TACTIC_THRESHOLD_CP)

    return None


def _motif(
    move: AnalyzedMove,
    motif_id: MotifId,
    loss: _Loss,
    phase: GamePhase,
    *,
    threshold: int,
    piece: PieceRef | None = None,
    attackers: tuple[str, ...] = (),
    defenders: tuple[str, ...] = (),
) -> Motif:
    return Motif(
        id=motif_id,
        label=_LABELS[motif_id],
        severity=_severity(loss),
        source="learned",
        score_cp=loss.score_cp,
        evidence=MotifEvidence(
            threshold_cp=threshold,
            score_kind=loss.score_kind,
            phase=phase,
            piece=piece,
            attackers=attackers,
            defenders=defenders,
            best_move=_move_ref(
                move.analysis_after.best_move
                if motif_id == "allowed_tactic"
                else move.analysis_before.best_move
            ),
            opponent_reply=None,
            related_ply=None,
        ),
    )


class _Loss:
    def __init__(
        self,
        *,
        effective_cp: int,
        score_cp: int | None,
        score_kind: ScoreKind,
    ) -> None:
        self.effective_cp = effective_cp
        self.score_cp = score_cp
        self.score_kind = score_kind


def _loss_for_move(move: AnalyzedMove) -> _Loss:
    before = _score_for_side(move.analysis_before.score, move.side)
    after = _score_for_side(move.analysis_after.score, move.side)
    score_cp = _bounded_loss_cp(move)
    return _Loss(
        effective_cp=max(0, before - after),
        score_cp=score_cp,
        score_kind="mate" if score_cp is None else "cp",
    )


def _score_for_side(score: CentipawnScore | MateScore, side: str) -> int:
    if isinstance(score, CentipawnScore):
        return score.cp if side == "white" else -score.cp
    if score.winner == side:
        return MATE_EFFECTIVE_CP - score.mate_in
    return -MATE_EFFECTIVE_CP + score.mate_in


def _bounded_loss_cp(move: AnalyzedMove) -> int | None:
    before = move.analysis_before.score
    after = move.analysis_after.score
    if not isinstance(before, CentipawnScore) or not isinstance(after, CentipawnScore):
        return None
    delta = after.cp - before.cp
    if move.side == "white":
        return max(0, -delta)
    return max(0, delta)


def _severity(loss: _Loss) -> MotifSeverity:
    if loss.score_kind == "mate" or loss.effective_cp >= TACTIC_THRESHOLD_CP:
        return "blunder"
    if loss.effective_cp >= 150:
        return "mistake"
    return "inaccuracy"


def _phase(board: chess.Board, move_number: int) -> GamePhase:
    if _piece_count(board) <= TABLEBASE_RELEVANT_PIECES:
        return "endgame"
    if move_number <= 15:
        return "opening"
    return "middlegame"


def _piece_count(board: chess.Board) -> int:
    return len(board.piece_map())


def _hanging_piece_evidence(
    board: chess.Board,
    color: chess.Color,
) -> tuple[PieceRef | None, tuple[str, ...], tuple[str, ...]]:
    candidates: list[tuple[int, PieceRef, tuple[str, ...], tuple[str, ...]]] = []
    for square, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type in {chess.PAWN, chess.KING}:
            continue
        attackers = board.attackers(not color, square)
        defenders = board.attackers(color, square)
        if len(attackers) <= len(defenders):
            continue
        candidates.append(
            (
                _piece_value(piece.piece_type),
                PieceRef(
                    color="white" if color == chess.WHITE else "black",
                    role=_piece_role(piece.piece_type),
                    square=chess.square_name(square),
                ),
                _piece_square_refs(board, attackers),
                _piece_square_refs(board, defenders),
            )
        )
    if not candidates:
        return None, (), ()
    _value, piece_ref, attacker_refs, defender_refs = max(candidates, key=lambda item: item[0])
    return piece_ref, attacker_refs, defender_refs


def _piece_square_refs(board: chess.Board, squares: chess.SquareSet) -> tuple[str, ...]:
    refs: list[str] = []
    for square in sorted(squares):
        piece = board.piece_at(square)
        refs.append(
            chess.square_name(square)
            if piece is None
            else f"{chess.square_name(square)} {piece.symbol().upper()}"
        )
    return tuple(refs)


def _piece_role(piece_type: chess.PieceType) -> PieceRole:
    if piece_type == chess.KNIGHT:
        return "knight"
    if piece_type == chess.BISHOP:
        return "bishop"
    if piece_type == chess.ROOK:
        return "rook"
    if piece_type == chess.QUEEN:
        return "queen"
    return "pawn"


def _piece_value(piece_type: chess.PieceType) -> int:
    if piece_type == chess.QUEEN:
        return 900
    if piece_type == chess.ROOK:
        return 500
    if piece_type in {chess.BISHOP, chess.KNIGHT}:
        return 300
    return 100


def _move_ref(move: EngineMove | None) -> MoveRef | None:
    if move is None:
        return None
    return MoveRef(uci=move.uci, san=move.san)


def _color_from_side(side: str) -> chess.Color:
    return chess.WHITE if side == "white" else chess.BLACK
