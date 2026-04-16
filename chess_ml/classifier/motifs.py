"""Heuristic chess weakness motif detectors for Slice 2."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import chess

from chess_ml.engine.stockfish import CentipawnScore, EngineEvaluation, EngineMove, MateScore
from chess_ml.ingestion.pgn import Side

MotifId: TypeAlias = Literal[
    "hanging_piece",
    "missed_tactic",
    "allowed_tactic",
    "endgame_slip",
    "opening_inaccuracy",
]
MotifSeverity: TypeAlias = Literal["inaccuracy", "mistake", "blunder"]
MotifSource: TypeAlias = Literal["heuristic", "learned", "ensemble"]
GamePhase: TypeAlias = Literal["opening", "middlegame", "endgame"]
PieceRole: TypeAlias = Literal["pawn", "knight", "bishop", "rook", "queen"]
ScoreKind: TypeAlias = Literal["cp", "mate"]

HANGING_PIECE_THRESHOLD_CP = 200
TACTIC_THRESHOLD_CP = 300
OPENING_INACCURACY_MIN_CP = 50
OPENING_INACCURACY_EXCLUSIVE_MAX_CP = 150
TABLEBASE_RELEVANT_PIECES = 7
MATE_EFFECTIVE_CP = 10_000


@dataclass(frozen=True)
class MoveRef:
    """A move reference in the position where it is legal."""

    uci: str
    san: str


@dataclass(frozen=True)
class PieceRef:
    """A piece and the square it occupies."""

    color: Side
    role: PieceRole
    square: str


@dataclass(frozen=True)
class MotifEvidence:
    """Grounding data for one detected motif."""

    threshold_cp: int
    score_kind: ScoreKind
    phase: GamePhase
    piece: PieceRef | None
    attackers: tuple[str, ...]
    defenders: tuple[str, ...]
    best_move: MoveRef | None
    opponent_reply: MoveRef | None
    related_ply: int | None


@dataclass(frozen=True)
class Motif:
    """One heuristic motif label on a move."""

    id: MotifId
    label: str
    severity: MotifSeverity
    source: MotifSource
    score_cp: int | None
    evidence: MotifEvidence


@dataclass(frozen=True)
class AnalyzedMove:
    """A parsed move plus the engine evaluations needed for classification."""

    ply: int
    move_number: int
    side: Side
    san: str
    uci: str
    fen_before: str
    fen_after: str
    analysis_before: EngineEvaluation
    analysis_after: EngineEvaluation


@dataclass(frozen=True)
class _Loss:
    effective_cp: int
    score_cp: int | None
    score_kind: ScoreKind


@dataclass(frozen=True)
class _HangingPieceCandidate:
    piece: PieceRef
    attackers: tuple[str, ...]
    defenders: tuple[str, ...]
    value: int


def detect_motifs(
    moves: Sequence[AnalyzedMove],
    *,
    initial_fen: str,
) -> list[tuple[Motif, ...]]:
    """Return one motif tuple per analyzed move."""

    starts_from_standard_position = initial_fen == chess.STARTING_FEN
    results: list[tuple[Motif, ...]] = []

    for index, move in enumerate(moves):
        before_board = chess.Board(move.fen_before)
        after_board = chess.Board(move.fen_after)
        loss = _loss_for_move(move)
        phase = _phase(before_board, move.move_number)
        detected: list[Motif] = []

        hanging_piece = _detect_hanging_piece(move, after_board, loss, phase)
        if hanging_piece is not None:
            detected.append(hanging_piece)

        tactic_motif = _detect_missed_or_endgame_tactic(move, before_board, loss, phase)
        if tactic_motif is not None:
            detected.append(tactic_motif)

        allowed_tactic = _detect_allowed_tactic(
            move,
            after_board,
            moves[index + 1] if index + 1 < len(moves) else None,
            loss,
            phase,
        )
        if allowed_tactic is not None:
            detected.append(allowed_tactic)

        opening_inaccuracy = _detect_opening_inaccuracy(
            move,
            loss,
            phase,
            starts_from_standard_position=starts_from_standard_position,
            higher_severity_exists=any(motif.severity != "inaccuracy" for motif in detected),
        )
        if opening_inaccuracy is not None:
            detected.append(opening_inaccuracy)

        results.append(tuple(detected))

    return results


def _detect_hanging_piece(
    move: AnalyzedMove,
    after_board: chess.Board,
    loss: _Loss,
    phase: GamePhase,
) -> Motif | None:
    if loss.effective_cp < HANGING_PIECE_THRESHOLD_CP:
        return None

    candidate = _best_hanging_piece(after_board, _color_from_side(move.side))
    if candidate is None:
        return None

    return Motif(
        id="hanging_piece",
        label="Hanging piece",
        severity=_severity(loss),
        source="heuristic",
        score_cp=loss.score_cp,
        evidence=MotifEvidence(
            threshold_cp=HANGING_PIECE_THRESHOLD_CP,
            score_kind=loss.score_kind,
            phase=phase,
            piece=candidate.piece,
            attackers=candidate.attackers,
            defenders=candidate.defenders,
            best_move=_move_ref(move.analysis_before.best_move),
            opponent_reply=None,
            related_ply=None,
        ),
    )


def _detect_missed_or_endgame_tactic(
    move: AnalyzedMove,
    before_board: chess.Board,
    loss: _Loss,
    phase: GamePhase,
) -> Motif | None:
    if loss.effective_cp < TACTIC_THRESHOLD_CP:
        return None

    best_move = _legal_move(before_board, move.analysis_before.best_move)
    if best_move is None or best_move.uci() == move.uci:
        return None
    if not _is_tactical(before_board, best_move, move.analysis_before):
        return None

    is_endgame_slip = _piece_count(before_board) <= TABLEBASE_RELEVANT_PIECES
    motif_id: MotifId = "endgame_slip" if is_endgame_slip else "missed_tactic"
    label = "Endgame slip" if is_endgame_slip else "Missed tactic"

    return Motif(
        id=motif_id,
        label=label,
        severity=_severity(loss),
        source="heuristic",
        score_cp=loss.score_cp,
        evidence=MotifEvidence(
            threshold_cp=TACTIC_THRESHOLD_CP,
            score_kind=loss.score_kind,
            phase=phase,
            piece=None,
            attackers=(),
            defenders=(),
            best_move=_move_ref(move.analysis_before.best_move),
            opponent_reply=None,
            related_ply=None,
        ),
    )


def _detect_allowed_tactic(
    move: AnalyzedMove,
    after_board: chess.Board,
    next_move: AnalyzedMove | None,
    loss: _Loss,
    phase: GamePhase,
) -> Motif | None:
    if loss.effective_cp < TACTIC_THRESHOLD_CP:
        return None

    opponent_best = _legal_move(after_board, move.analysis_after.best_move)
    if opponent_best is None or not _is_tactical(after_board, opponent_best, move.analysis_after):
        return None

    opponent_reply: MoveRef | None = None
    related_ply: int | None = None
    if next_move is not None and next_move.uci == opponent_best.uci():
        opponent_reply = MoveRef(uci=next_move.uci, san=next_move.san)
        related_ply = next_move.ply

    return Motif(
        id="allowed_tactic",
        label="Allowed tactic",
        severity=_severity(loss),
        source="heuristic",
        score_cp=loss.score_cp,
        evidence=MotifEvidence(
            threshold_cp=TACTIC_THRESHOLD_CP,
            score_kind=loss.score_kind,
            phase=phase,
            piece=None,
            attackers=(),
            defenders=(),
            best_move=_move_ref(move.analysis_after.best_move),
            opponent_reply=opponent_reply,
            related_ply=related_ply,
        ),
    )


def _detect_opening_inaccuracy(
    move: AnalyzedMove,
    loss: _Loss,
    phase: GamePhase,
    *,
    starts_from_standard_position: bool,
    higher_severity_exists: bool,
) -> Motif | None:
    if not starts_from_standard_position or higher_severity_exists:
        return None
    if move.move_number > 15 or loss.score_cp is None:
        return None
    if not (OPENING_INACCURACY_MIN_CP <= loss.score_cp < OPENING_INACCURACY_EXCLUSIVE_MAX_CP):
        return None

    return Motif(
        id="opening_inaccuracy",
        label="Opening inaccuracy",
        severity="inaccuracy",
        source="heuristic",
        score_cp=loss.score_cp,
        evidence=MotifEvidence(
            threshold_cp=OPENING_INACCURACY_MIN_CP,
            score_kind="cp",
            phase=phase,
            piece=None,
            attackers=(),
            defenders=(),
            best_move=_move_ref(move.analysis_before.best_move),
            opponent_reply=None,
            related_ply=None,
        ),
    )


def _best_hanging_piece(board: chess.Board, color: chess.Color) -> _HangingPieceCandidate | None:
    candidates: list[_HangingPieceCandidate] = []
    for square, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type in {chess.PAWN, chess.KING}:
            continue

        attackers = board.attackers(not color, square)
        defenders = board.attackers(color, square)
        if len(attackers) <= len(defenders):
            continue

        role = _piece_role(piece.piece_type)
        if role is None:
            continue

        candidates.append(
            _HangingPieceCandidate(
                piece=PieceRef(
                    color=_side_from_color(color),
                    role=role,
                    square=chess.square_name(square),
                ),
                attackers=_piece_square_refs(board, attackers),
                defenders=_piece_square_refs(board, defenders),
                value=_piece_value(piece.piece_type),
            )
        )

    return max(
        candidates,
        key=lambda candidate: (
            candidate.value,
            len(candidate.attackers) - len(candidate.defenders),
            candidate.piece.square,
        ),
        default=None,
    )


def _loss_for_move(move: AnalyzedMove) -> _Loss:
    before = _score_for_side(move.analysis_before.score, move.side)
    after = _score_for_side(move.analysis_after.score, move.side)
    effective_cp = max(0, before - after)
    score_cp = _bounded_loss_cp(move)
    score_kind: ScoreKind = "mate" if score_cp is None else "cp"
    return _Loss(effective_cp=effective_cp, score_cp=score_cp, score_kind=score_kind)


def _score_for_side(score: CentipawnScore | MateScore, side: Side) -> int:
    if isinstance(score, CentipawnScore):
        return score.cp if side == "white" else -score.cp

    winner_side = score.winner
    if winner_side == side:
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


def _is_tactical(
    board: chess.Board,
    move: chess.Move,
    evaluation: EngineEvaluation,
) -> bool:
    if move not in board.legal_moves:
        return False
    if board.is_capture(move) or move.promotion is not None or board.gives_check(move):
        return True

    next_board = board.copy(stack=False)
    next_board.push(move)
    if next_board.is_checkmate():
        return True

    return (
        isinstance(evaluation.score, MateScore)
        and evaluation.best_move is not None
        and evaluation.best_move.uci == move.uci()
    )


def _phase(board: chess.Board, move_number: int) -> GamePhase:
    if _piece_count(board) <= TABLEBASE_RELEVANT_PIECES:
        return "endgame"
    if move_number <= 15:
        return "opening"
    return "middlegame"


def _piece_count(board: chess.Board) -> int:
    return len(board.piece_map())


def _legal_move(board: chess.Board, move: EngineMove | None) -> chess.Move | None:
    if move is None:
        return None
    try:
        parsed = chess.Move.from_uci(move.uci)
    except ValueError:
        return None
    if parsed not in board.legal_moves:
        return None
    return parsed


def _move_ref(move: EngineMove | None) -> MoveRef | None:
    if move is None:
        return None
    return MoveRef(uci=move.uci, san=move.san)


def _piece_square_refs(board: chess.Board, squares: chess.SquareSet) -> tuple[str, ...]:
    refs: list[str] = []
    for square in sorted(squares):
        piece = board.piece_at(square)
        if piece is None:
            refs.append(chess.square_name(square))
            continue
        refs.append(f"{chess.square_name(square)} {piece.symbol().upper()}")
    return tuple(refs)


def _piece_role(piece_type: chess.PieceType) -> PieceRole | None:
    if piece_type == chess.PAWN:
        return "pawn"
    if piece_type == chess.KNIGHT:
        return "knight"
    if piece_type == chess.BISHOP:
        return "bishop"
    if piece_type == chess.ROOK:
        return "rook"
    if piece_type == chess.QUEEN:
        return "queen"
    return None


def _piece_value(piece_type: chess.PieceType) -> int:
    if piece_type == chess.QUEEN:
        return 900
    if piece_type == chess.ROOK:
        return 500
    if piece_type in {chess.BISHOP, chess.KNIGHT}:
        return 300
    if piece_type == chess.PAWN:
        return 100
    return 0


def _color_from_side(side: Side) -> chess.Color:
    return chess.WHITE if side == "white" else chess.BLACK


def _side_from_color(color: chess.Color) -> Side:
    return "white" if color == chess.WHITE else "black"
