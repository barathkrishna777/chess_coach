"""Heuristic chess weakness motif detectors for Slice 2."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import chess

from chess_ml.engine.stockfish import CentipawnScore, EngineEvaluation, EngineMove, MateScore
from chess_ml.ingestion.pgn import Side

MotifId: TypeAlias = Literal[
    "hanging_piece",
    "missed_tactic",
    "allowed_tactic",
    "endgame_slip",
    "opening_inaccuracy",
    "pin",
    "fork",
    "overloaded_defender",
    "discovered_attack",
]
MotifSeverity: TypeAlias = Literal["inaccuracy", "mistake", "blunder"]
MotifSource: TypeAlias = Literal["heuristic", "learned", "ensemble"]
GamePhase: TypeAlias = Literal["opening", "middlegame", "endgame"]
PieceRole: TypeAlias = Literal["pawn", "knight", "bishop", "rook", "queen"]
ScoreKind: TypeAlias = Literal["cp", "mate"]

HANGING_PIECE_THRESHOLD_CP = 200
TACTIC_THRESHOLD_CP = 300
TACTICAL_PATTERN_THRESHOLD_CP = 200
OPENING_INACCURACY_MIN_CP = 50
OPENING_INACCURACY_EXCLUSIVE_MAX_CP = 150
TABLEBASE_RELEVANT_PIECES = 7
MATE_EFFECTIVE_CP = 10_000
KING_VALUE = 20_000
VALUABLE_TARGET_MIN_CP = 300

_MOTIF_LABELS: dict[MotifId, str] = {
    "hanging_piece": "Hanging piece",
    "missed_tactic": "Missed tactic",
    "allowed_tactic": "Allowed tactic",
    "endgame_slip": "Endgame slip",
    "opening_inaccuracy": "Opening inaccuracy",
    "pin": "Pin",
    "fork": "Fork",
    "overloaded_defender": "Overloaded defender",
    "discovered_attack": "Discovered attack",
}


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


@dataclass(frozen=True)
class _PatternCandidate:
    piece: PieceRef
    attackers: tuple[str, ...]
    defenders: tuple[str, ...]
    value: int


@dataclass(frozen=True)
class _PinContext:
    attackers: tuple[chess.Square, ...]
    anchor_square: chess.Square
    anchor_value: int


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

        detected.extend(
            _detect_tactical_patterns(
                move,
                before_board,
                after_board,
                moves[index + 1] if index + 1 < len(moves) else None,
                loss,
                phase,
            )
        )

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
        label=_MOTIF_LABELS["hanging_piece"],
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

    return Motif(
        id=motif_id,
        label=_MOTIF_LABELS[motif_id],
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
        label=_MOTIF_LABELS["allowed_tactic"],
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
        label=_MOTIF_LABELS["opening_inaccuracy"],
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


def _detect_tactical_patterns(
    move: AnalyzedMove,
    before_board: chess.Board,
    after_board: chess.Board,
    next_move: AnalyzedMove | None,
    loss: _Loss,
    phase: GamePhase,
) -> tuple[Motif, ...]:
    if loss.effective_cp < TACTICAL_PATTERN_THRESHOLD_CP:
        return ()

    user_color = _color_from_side(move.side)
    motifs: list[Motif] = []
    seen: set[MotifId] = set()

    best_before = _legal_move(before_board, move.analysis_before.best_move)
    if best_before is not None and best_before.uci() != move.uci:
        best_before_board = _board_after(before_board, best_before)
        missed_patterns = (
            (
                "pin",
                _best_pin_candidate(
                    best_before_board,
                    not user_color,
                    previous_board=before_board,
                ),
            ),
            (
                "fork",
                _best_fork_candidate(
                    best_before_board,
                    user_color,
                    previous_board=before_board,
                ),
            ),
            (
                "overloaded_defender",
                _best_overloaded_defender(before_board, best_before, not user_color),
            ),
            (
                "discovered_attack",
                _best_discovered_attack(before_board, best_before, user_color),
            ),
        )
        for motif_id, candidate in missed_patterns:
            typed_motif_id = cast(MotifId, motif_id)
            if candidate is not None and typed_motif_id not in seen:
                motifs.append(
                    _pattern_motif(
                        typed_motif_id,
                        candidate,
                        move,
                        loss,
                        phase,
                        best_move=_move_ref(move.analysis_before.best_move),
                        opponent_reply=None,
                        related_ply=None,
                    )
                )
                seen.add(typed_motif_id)

    best_after = _legal_move(after_board, move.analysis_after.best_move)
    if best_after is not None:
        best_after_board = _board_after(after_board, best_after)
        opponent_color = not user_color
        opponent_reply, related_ply = _actual_opponent_reply(next_move, best_after)
        allowed_patterns = (
            (
                "pin",
                _best_pin_candidate(
                    best_after_board,
                    user_color,
                    previous_board=after_board,
                ),
            ),
            (
                "fork",
                _best_fork_candidate(
                    best_after_board,
                    opponent_color,
                    previous_board=after_board,
                ),
            ),
            (
                "overloaded_defender",
                _best_overloaded_defender(after_board, best_after, user_color),
            ),
            (
                "discovered_attack",
                _best_discovered_attack(after_board, best_after, opponent_color),
            ),
        )
        for motif_id, candidate in allowed_patterns:
            typed_motif_id = cast(MotifId, motif_id)
            if candidate is not None and typed_motif_id not in seen:
                motifs.append(
                    _pattern_motif(
                        typed_motif_id,
                        candidate,
                        move,
                        loss,
                        phase,
                        best_move=_move_ref(move.analysis_after.best_move),
                        opponent_reply=opponent_reply,
                        related_ply=related_ply,
                    )
                )
                seen.add(typed_motif_id)

    return tuple(motifs)


def _pattern_motif(
    motif_id: MotifId,
    candidate: _PatternCandidate,
    move: AnalyzedMove,
    loss: _Loss,
    phase: GamePhase,
    *,
    best_move: MoveRef | None,
    opponent_reply: MoveRef | None,
    related_ply: int | None,
) -> Motif:
    return Motif(
        id=motif_id,
        label=_MOTIF_LABELS[motif_id],
        severity=_severity(loss),
        source="heuristic",
        score_cp=loss.score_cp,
        evidence=MotifEvidence(
            threshold_cp=TACTICAL_PATTERN_THRESHOLD_CP,
            score_kind=loss.score_kind,
            phase=phase,
            piece=candidate.piece,
            attackers=candidate.attackers,
            defenders=candidate.defenders,
            best_move=best_move,
            opponent_reply=opponent_reply,
            related_ply=related_ply,
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


def _best_pin_candidate(
    board: chess.Board,
    victim_color: chess.Color,
    *,
    previous_board: chess.Board,
) -> _PatternCandidate | None:
    candidates: list[_PatternCandidate] = []
    for square, piece in board.piece_map().items():
        if piece.color != victim_color or piece.piece_type == chess.KING:
            continue
        if _pin_context(previous_board, victim_color, square) is not None:
            continue

        context = _pin_context(board, victim_color, square)
        role = _piece_role(piece.piece_type)
        if context is None or role is None:
            continue
        candidates.append(
            _PatternCandidate(
                piece=PieceRef(
                    color=_side_from_color(victim_color),
                    role=role,
                    square=chess.square_name(square),
                ),
                attackers=_piece_square_refs_from_squares(board, context.attackers),
                defenders=_piece_square_refs_from_squares(board, (context.anchor_square,)),
                value=context.anchor_value + _piece_value(piece.piece_type),
            )
        )

    return _best_pattern_candidate(candidates)


def _pin_context(
    board: chess.Board,
    victim_color: chess.Color,
    square: chess.Square,
) -> _PinContext | None:
    victim = board.piece_at(square)
    if victim is None or victim.color != victim_color or victim.piece_type == chess.KING:
        return None

    attackers: list[chess.Square] = []
    anchors: list[tuple[int, chess.Square]] = []
    for attacker_square in board.attackers(not victim_color, square):
        attacker = board.piece_at(attacker_square)
        if attacker is None or not _is_sliding_piece(attacker.piece_type):
            continue
        anchor = _pin_anchor_behind(board, attacker_square, square, victim_color)
        if anchor is None:
            continue
        anchor_square, anchor_piece = anchor
        anchor_value = _piece_value_or_king(anchor_piece)
        if anchor_piece.piece_type != chess.KING and anchor_value <= _piece_value(
            victim.piece_type
        ):
            continue
        attackers.append(attacker_square)
        anchors.append((anchor_value, anchor_square))

    if not attackers or not anchors:
        return None
    anchor_value, anchor_square = max(anchors, key=lambda item: (item[0], item[1]))
    return _PinContext(
        attackers=tuple(sorted(attackers)),
        anchor_square=anchor_square,
        anchor_value=anchor_value,
    )


def _pin_anchor_behind(
    board: chess.Board,
    attacker_square: chess.Square,
    victim_square: chess.Square,
    victim_color: chess.Color,
) -> tuple[chess.Square, chess.Piece] | None:
    direction = _direction_between(attacker_square, victim_square)
    if direction is None:
        return None
    for square in _ray_from(victim_square, direction):
        piece = board.piece_at(square)
        if piece is None:
            continue
        if piece.color == victim_color:
            return square, piece
        return None
    return None


def _best_fork_candidate(
    board: chess.Board,
    attacker_color: chess.Color,
    *,
    previous_board: chess.Board,
) -> _PatternCandidate | None:
    candidates: list[_PatternCandidate] = []
    for square, piece in board.piece_map().items():
        if piece.color != attacker_color or piece.piece_type == chess.KING:
            continue
        targets = _fork_targets(board, square, attacker_color)
        if len(targets) < 2 or len(_fork_targets(previous_board, square, attacker_color)) >= 2:
            continue
        role = _piece_role(piece.piece_type)
        if role is None:
            continue
        candidates.append(
            _PatternCandidate(
                piece=PieceRef(
                    color=_side_from_color(attacker_color),
                    role=role,
                    square=chess.square_name(square),
                ),
                attackers=_piece_square_refs_from_squares(board, targets),
                defenders=(),
                value=sum(_piece_value_or_king(board.piece_at(target)) for target in targets),
            )
        )

    return _best_pattern_candidate(candidates)


def _fork_targets(
    board: chess.Board,
    attacker_square: chess.Square,
    attacker_color: chess.Color,
) -> tuple[chess.Square, ...]:
    attacker = board.piece_at(attacker_square)
    if attacker is None or attacker.color != attacker_color:
        return ()
    attacker_value = _piece_value(attacker.piece_type)
    targets: list[chess.Square] = []
    for target_square in board.attacks(attacker_square):
        target = board.piece_at(target_square)
        if (
            target is not None
            and target.color != attacker_color
            and target.piece_type != chess.KING
            and _piece_value(target.piece_type) >= attacker_value
        ):
            targets.append(target_square)
    return tuple(sorted(targets))


def _best_overloaded_defender(
    board: chess.Board,
    move: chess.Move,
    defender_color: chess.Color,
) -> _PatternCandidate | None:
    if move not in board.legal_moves or not board.is_capture(move) or board.is_en_passant(move):
        return None

    captured_square = move.to_square
    captured_piece = board.piece_at(captured_square)
    if captured_piece is None or captured_piece.color != defender_color:
        return None

    candidates: list[_PatternCandidate] = []
    for defender_square in board.attackers(defender_color, captured_square):
        defender = board.piece_at(defender_square)
        if defender is None or defender.piece_type == chess.KING:
            continue
        defended_targets = _defended_targets(board, defender_square, defender_color)
        other_targets = tuple(square for square in defended_targets if square != captured_square)
        if len(defended_targets) < 2:
            continue
        role = _piece_role(defender.piece_type)
        if role is None:
            continue
        candidates.append(
            _PatternCandidate(
                piece=PieceRef(
                    color=_side_from_color(defender_color),
                    role=role,
                    square=chess.square_name(defender_square),
                ),
                attackers=_piece_square_refs_from_squares(board, (captured_square,)),
                defenders=_piece_square_refs_from_squares(board, other_targets),
                value=sum(
                    _piece_value_or_king(board.piece_at(target)) for target in defended_targets
                ),
            )
        )

    return _best_pattern_candidate(candidates)


def _defended_targets(
    board: chess.Board,
    defender_square: chess.Square,
    defender_color: chess.Color,
) -> tuple[chess.Square, ...]:
    targets: list[chess.Square] = []
    for target_square in board.attacks(defender_square):
        target = board.piece_at(target_square)
        if (
            target is not None
            and target.color == defender_color
            and target.piece_type != chess.KING
        ):
            targets.append(target_square)
    return tuple(sorted(targets))


def _best_discovered_attack(
    board: chess.Board,
    move: chess.Move,
    attacker_color: chess.Color,
) -> _PatternCandidate | None:
    if move not in board.legal_moves:
        return None
    moved_piece = board.piece_at(move.from_square)
    if moved_piece is None or moved_piece.color != attacker_color:
        return None

    after_board = _board_after(board, move)
    candidates: list[_PatternCandidate] = []
    for attacker_square, attacker in after_board.piece_map().items():
        if (
            attacker.color != attacker_color
            or not _is_sliding_piece(attacker.piece_type)
            or attacker_square == move.to_square
        ):
            continue
        role = _piece_role(attacker.piece_type)
        if role is None:
            continue
        for target_square in after_board.attacks(attacker_square):
            target = after_board.piece_at(target_square)
            if (
                target is None
                or target.color == attacker_color
                or target.piece_type == chess.KING
                or _piece_value(target.piece_type) < VALUABLE_TARGET_MIN_CP
                or target_square in board.attacks(attacker_square)
                or not _line_contains(attacker_square, target_square, move.from_square)
            ):
                continue
            candidates.append(
                _PatternCandidate(
                    piece=PieceRef(
                        color=_side_from_color(attacker_color),
                        role=role,
                        square=chess.square_name(attacker_square),
                    ),
                    attackers=_piece_square_refs_from_squares(after_board, (target_square,)),
                    defenders=_piece_square_refs_from_squares(board, (move.from_square,)),
                    value=_piece_value(target.piece_type) + _piece_value(attacker.piece_type),
                )
            )

    return _best_pattern_candidate(candidates)


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


def _piece_square_refs_from_squares(
    board: chess.Board,
    squares: Iterable[chess.Square],
) -> tuple[str, ...]:
    refs: list[str] = []
    for square in sorted(squares):
        piece = board.piece_at(square)
        if piece is None:
            refs.append(chess.square_name(square))
            continue
        refs.append(f"{chess.square_name(square)} {piece.symbol().upper()}")
    return tuple(refs)


def _best_pattern_candidate(candidates: Sequence[_PatternCandidate]) -> _PatternCandidate | None:
    return max(
        candidates,
        key=lambda candidate: (
            candidate.value,
            len(candidate.attackers),
            len(candidate.defenders),
            candidate.piece.square,
        ),
        default=None,
    )


def _actual_opponent_reply(
    next_move: AnalyzedMove | None,
    expected: chess.Move,
) -> tuple[MoveRef | None, int | None]:
    if next_move is None or next_move.uci != expected.uci():
        return None, None
    return MoveRef(uci=next_move.uci, san=next_move.san), next_move.ply


def _board_after(board: chess.Board, move: chess.Move) -> chess.Board:
    next_board = board.copy(stack=False)
    next_board.push(move)
    return next_board


def _is_sliding_piece(piece_type: chess.PieceType) -> bool:
    return piece_type in {chess.BISHOP, chess.ROOK, chess.QUEEN}


def _piece_value_or_king(piece: chess.Piece | None) -> int:
    if piece is None:
        return 0
    if piece.piece_type == chess.KING:
        return KING_VALUE
    return _piece_value(piece.piece_type)


def _direction_between(
    start: chess.Square,
    end: chess.Square,
) -> tuple[int, int] | None:
    start_file = chess.square_file(start)
    start_rank = chess.square_rank(start)
    end_file = chess.square_file(end)
    end_rank = chess.square_rank(end)
    file_delta = end_file - start_file
    rank_delta = end_rank - start_rank

    file_step = _sign(file_delta)
    rank_step = _sign(rank_delta)
    if file_delta == 0 and rank_delta != 0:
        return 0, rank_step
    if rank_delta == 0 and file_delta != 0:
        return file_step, 0
    if abs(file_delta) == abs(rank_delta) and file_delta != 0:
        return file_step, rank_step
    return None


def _ray_from(square: chess.Square, direction: tuple[int, int]) -> tuple[chess.Square, ...]:
    file_step, rank_step = direction
    file_index = chess.square_file(square) + file_step
    rank_index = chess.square_rank(square) + rank_step
    squares: list[chess.Square] = []
    while 0 <= file_index <= 7 and 0 <= rank_index <= 7:
        squares.append(chess.square(file_index, rank_index))
        file_index += file_step
        rank_index += rank_step
    return tuple(squares)


def _line_contains(
    start: chess.Square,
    end: chess.Square,
    middle: chess.Square,
) -> bool:
    direction = _direction_between(start, end)
    if direction is None:
        return False
    for square in _ray_from(start, direction):
        if square == middle:
            return True
        if square == end:
            return False
    return False


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


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
