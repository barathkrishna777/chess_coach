"""Tests for Slice 2 heuristic motif classification."""

from pathlib import Path
from typing import Literal, cast

import chess

from chess_ml.api.games import _motif_model
from chess_ml.classifier.classify import classify_moves
from chess_ml.classifier.motifs import (
    AnalyzedMove,
    Motif,
    MotifEvidence,
    MotifId,
    MoveRef,
    PieceRef,
)
from chess_ml.engine.stockfish import CentipawnScore, EngineEvaluation, EngineMove, MateScore
from chess_ml.ingestion.pgn import ParsedPgnGame, parse_pgn


def test_hanging_piece_tags_loose_queen() -> None:
    parsed = parse_pgn(
        """
[Event "Fixture"]
[Result "*"]

1. e4 e5 2. Qh5 Nf6 3. Bc4 *
"""
    )

    motifs = _classify(
        parsed,
        {
            5: _spec(before_cp=0, after_cp=-650, best_before="g1f3"),
        },
    )

    hanging = _required_motif(motifs[4], "hanging_piece")
    assert hanging.severity == "blunder"
    assert hanging.score_cp == 650
    assert hanging.evidence.piece is not None
    assert hanging.evidence.piece.role == "queen"
    assert hanging.evidence.piece.square == "h5"
    assert "f6 N" in hanging.evidence.attackers


def test_missed_tactic_tags_unplayed_best_capture() -> None:
    parsed = parse_pgn(
        """
[Event "Fixture"]
[Result "*"]

1. e4 d5 2. exd5 Qxd5 3. Nc3 a6 4. Nf3 *
"""
    )

    motifs = _classify(
        parsed,
        {
            7: _spec(before_cp=740, after_cp=85, best_before="c3d5"),
        },
    )

    missed = _required_motif(motifs[6], "missed_tactic")
    assert missed.severity == "blunder"
    assert missed.score_cp == 655
    assert missed.evidence.best_move is not None
    assert missed.evidence.best_move.uci == "c3d5"
    assert missed.evidence.best_move.san == "Nxd5"
    assert _motif_ids(motifs[6]) == ["missed_tactic"]


def test_allowed_tactic_tags_immediate_opponent_best_move() -> None:
    parsed = parse_pgn(
        """
[Event "Fixture"]
[Result "*"]

1. e4 d5 2. exd5 Qxd5 3. Nc3 a6 4. Nxd5 *
"""
    )

    motifs = _classify(
        parsed,
        {
            6: _spec(
                before_cp=80,
                after_cp=702,
                best_before="d5d8",
                best_after="c3d5",
            ),
        },
    )

    assert _motif_ids(motifs[5]) == ["hanging_piece", "allowed_tactic"]
    allowed = _required_motif(motifs[5], "allowed_tactic")
    assert allowed.evidence.best_move is not None
    assert allowed.evidence.best_move.uci == "c3d5"
    assert allowed.evidence.opponent_reply is not None
    assert allowed.evidence.opponent_reply.uci == "c3d5"
    assert allowed.evidence.related_ply == 7


def test_endgame_slip_replaces_missed_tactic_in_low_piece_position() -> None:
    parsed = parse_pgn(
        """
[Event "Fixture"]
[SetUp "1"]
[FEN "6k1/8/8/8/3q4/8/8/3Q2K1 w - - 0 1"]
[Result "*"]

1. Kf1 *
"""
    )

    motifs = _classify(
        parsed,
        {
            1: _spec(before_cp=538, after_cp=-530, best_before="d1d4"),
        },
    )

    assert "endgame_slip" in _motif_ids(motifs[0])
    assert "missed_tactic" not in _motif_ids(motifs[0])
    endgame_slip = _required_motif(motifs[0], "endgame_slip")
    assert endgame_slip.evidence.best_move is not None
    assert endgame_slip.evidence.best_move.san == "Qxd4"


def test_opening_inaccuracy_tags_small_early_loss() -> None:
    parsed = parse_pgn(
        """
[Event "Fixture"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 h6 *
"""
    )

    motifs = _classify(
        parsed,
        {
            6: _spec(before_cp=-17, after_cp=70, best_before="g8f6"),
        },
    )

    assert _motif_ids(motifs[5]) == ["opening_inaccuracy"]
    opening = motifs[5][0]
    assert opening.severity == "inaccuracy"
    assert opening.score_cp == 87
    assert opening.evidence.phase == "opening"


def test_opening_inaccuracy_skips_custom_starting_fen() -> None:
    parsed = parse_pgn(
        """
[Event "Fixture"]
[SetUp "1"]
[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"]
[Result "*"]

1... h6 *
"""
    )

    motifs = _classify(
        parsed,
        {
            1: _spec(before_cp=-17, after_cp=70, best_before="g8f6"),
        },
    )

    assert motifs[0] == ()


def test_mate_swing_uses_effective_loss_without_public_score_cp() -> None:
    parsed = parse_pgn(
        """
[Event "Fixture"]
[Result "*"]

1. f3 e5 2. g4 *
"""
    )

    motifs = _classify(
        parsed,
        {
            3: _spec(
                before_cp=-70,
                after_mate=("black", 1),
                best_before="e2e4",
                best_after="d8h4",
            ),
        },
    )

    allowed = _required_motif(motifs[2], "allowed_tactic")
    assert allowed.severity == "blunder"
    assert allowed.score_cp is None
    assert allowed.evidence.score_kind == "mate"
    assert allowed.evidence.best_move is not None
    assert allowed.evidence.best_move.san == "Qh4#"


def test_pin_tags_missed_pin_best_line() -> None:
    parsed = parse_pgn(
        """
[Event "Fixture"]
[SetUp "1"]
[FEN "4k3/3n4/8/8/2B5/8/8/4K3 w - - 0 1"]
[Result "*"]

1. Ke2 *
"""
    )

    motifs = _classify(
        parsed,
        {
            1: _spec(before_cp=300, after_cp=0, best_before="c4b5"),
        },
    )

    pin = _required_motif(motifs[0], "pin")
    assert pin.severity == "blunder"
    assert pin.evidence.piece is not None
    assert pin.evidence.piece.role == "knight"
    assert pin.evidence.piece.square == "d7"
    assert "b5 B" in pin.evidence.attackers
    assert "e8 K" in pin.evidence.defenders


def test_pin_tags_allowed_pin_best_reply() -> None:
    parsed = parse_pgn(
        """
[Event "Fixture"]
[SetUp "1"]
[FEN "4k3/8/8/2b5/8/2N5/P7/4K3 w - - 0 1"]
[Result "*"]

1. a3 *
"""
    )

    motifs = _classify(
        parsed,
        {
            1: _spec(before_cp=250, after_cp=0, best_after="c5b4"),
        },
    )

    pin = _required_motif(motifs[0], "pin")
    assert pin.severity == "mistake"
    assert pin.evidence.best_move is not None
    assert pin.evidence.best_move.uci == "c5b4"
    assert pin.evidence.piece is not None
    assert pin.evidence.piece.square == "c3"


def test_fork_tags_missed_fork_best_line() -> None:
    parsed = parse_pgn(
        """
[Event "Fixture"]
[SetUp "1"]
[FEN "k2q3r/8/8/6N1/8/8/8/4K3 w - - 0 1"]
[Result "*"]

1. Ke2 *
"""
    )

    motifs = _classify(
        parsed,
        {
            1: _spec(before_cp=280, after_cp=0, best_before="g5f7"),
        },
    )

    fork = _required_motif(motifs[0], "fork")
    assert fork.evidence.piece is not None
    assert fork.evidence.piece.role == "knight"
    assert fork.evidence.piece.square == "f7"
    assert set(fork.evidence.attackers) == {"d8 Q", "h8 R"}


def test_fork_tags_allowed_fork_best_reply() -> None:
    parsed = parse_pgn(
        """
[Event "Fixture"]
[SetUp "1"]
[FEN "k7/8/8/8/6n1/8/P7/3QK2R w - - 0 1"]
[Result "*"]

1. a3 *
"""
    )

    motifs = _classify(
        parsed,
        {
            1: _spec(before_cp=260, after_cp=0, best_after="g4f2"),
        },
    )

    fork = _required_motif(motifs[0], "fork")
    assert fork.evidence.best_move is not None
    assert fork.evidence.best_move.uci == "g4f2"
    assert fork.evidence.piece is not None
    assert fork.evidence.piece.square == "f2"
    assert set(fork.evidence.attackers) == {"d1 Q", "h1 R"}


def test_overloaded_defender_tags_missed_capture() -> None:
    parsed = parse_pgn(
        """
[Event "Fixture"]
[SetUp "1"]
[FEN "k7/8/3q4/8/3r4/4B3/7b/4K3 w - - 0 1"]
[Result "*"]

1. Kf1 *
"""
    )

    motifs = _classify(
        parsed,
        {
            1: _spec(before_cp=260, after_cp=0, best_before="e3d4"),
        },
    )

    overloaded = _required_motif(motifs[0], "overloaded_defender")
    assert overloaded.evidence.piece is not None
    assert overloaded.evidence.piece.role == "queen"
    assert overloaded.evidence.piece.square == "d6"
    assert overloaded.evidence.attackers == ("d4 R",)
    assert "h2 B" in overloaded.evidence.defenders


def test_overloaded_defender_tags_allowed_capture() -> None:
    parsed = parse_pgn(
        """
[Event "Fixture"]
[SetUp "1"]
[FEN "k7/7B/8/8/8/3Q4/P3b3/3RK3 w - - 0 1"]
[Result "*"]

1. a3 *
"""
    )

    motifs = _classify(
        parsed,
        {
            1: _spec(before_cp=250, after_cp=0, best_after="e2d1"),
        },
    )

    overloaded = _required_motif(motifs[0], "overloaded_defender")
    assert overloaded.evidence.best_move is not None
    assert overloaded.evidence.best_move.uci == "e2d1"
    assert overloaded.evidence.piece is not None
    assert overloaded.evidence.piece.square == "d3"
    assert overloaded.evidence.attackers == ("d1 R",)
    assert "h7 B" in overloaded.evidence.defenders


def test_discovered_attack_tags_missed_discovery() -> None:
    parsed = parse_pgn(
        """
[Event "Fixture"]
[SetUp "1"]
[FEN "k3q3/8/8/8/4N3/8/8/K3R3 w - - 0 1"]
[Result "*"]

1. Kb1 *
"""
    )

    motifs = _classify(
        parsed,
        {
            1: _spec(before_cp=260, after_cp=0, best_before="e4f6"),
        },
    )

    discovered = _required_motif(motifs[0], "discovered_attack")
    assert discovered.evidence.piece is not None
    assert discovered.evidence.piece.role == "rook"
    assert discovered.evidence.piece.square == "e1"
    assert discovered.evidence.attackers == ("e8 Q",)
    assert discovered.evidence.defenders == ("e4 N",)


def test_discovered_attack_tags_allowed_discovery() -> None:
    parsed = parse_pgn(
        """
[Event "Fixture"]
[SetUp "1"]
[FEN "k3r3/8/8/4n3/8/8/P7/K3Q3 w - - 0 1"]
[Result "*"]

1. a3 *
"""
    )

    motifs = _classify(
        parsed,
        {
            1: _spec(before_cp=250, after_cp=0, best_after="e5f3"),
        },
    )

    discovered = _required_motif(motifs[0], "discovered_attack")
    assert discovered.evidence.best_move is not None
    assert discovered.evidence.best_move.uci == "e5f3"
    assert discovered.evidence.piece is not None
    assert discovered.evidence.piece.square == "e8"
    assert discovered.evidence.attackers == ("e1 Q",)
    assert discovered.evidence.defenders == ("e5 N",)


def test_demo_missed_tactic_fixture_keeps_missed_tactic_label() -> None:
    parsed = parse_pgn(Path("tests/fixtures/demo/missed-tactic.pgn").read_text())

    motifs = _classify(
        parsed,
        {
            7: _spec(before_cp=740, after_cp=85, best_before="c3d5"),
        },
    )

    assert "missed_tactic" in _motif_ids(motifs[6])


def test_api_motif_model_accepts_richer_motif_ids() -> None:
    for motif_id, label in (
        ("pin", "Pin"),
        ("fork", "Fork"),
        ("overloaded_defender", "Overloaded defender"),
        ("discovered_attack", "Discovered attack"),
    ):
        model = _motif_model(_api_motif(cast(MotifId, motif_id), label))
        assert model.id == motif_id
        assert model.label == label


def test_normal_developing_moves_have_no_motifs() -> None:
    parsed = parse_pgn(
        """
[Event "Fixture"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 *
"""
    )

    assert all(motifs == () for motifs in _classify(parsed, {}))


ScoreSpec = int | tuple[Literal["white", "black"], int]


def _classify(
    parsed: ParsedPgnGame,
    specs: dict[int, "_MoveSpec"],
) -> list[tuple[Motif, ...]]:
    moves: list[AnalyzedMove] = []
    for parsed_move in parsed.moves:
        spec = specs.get(parsed_move.ply, _spec(before_cp=0, after_cp=0))
        moves.append(
            AnalyzedMove(
                ply=parsed_move.ply,
                move_number=parsed_move.move_number,
                side=parsed_move.side,
                san=parsed_move.san,
                uci=parsed_move.uci,
                fen_before=parsed_move.fen_before,
                fen_after=parsed_move.fen_after,
                analysis_before=_evaluation(
                    parsed_move.fen_before,
                    spec.before_score,
                    best_uci=spec.best_before,
                ),
                analysis_after=_evaluation(
                    parsed_move.fen_after,
                    spec.after_score,
                    best_uci=spec.best_after,
                ),
            )
        )
    return classify_moves(moves, initial_fen=parsed.initial_fen)


class _MoveSpec:
    def __init__(
        self,
        *,
        before_score: ScoreSpec,
        after_score: ScoreSpec,
        best_before: str | None = None,
        best_after: str | None = None,
    ) -> None:
        self.before_score = before_score
        self.after_score = after_score
        self.best_before = best_before
        self.best_after = best_after


def _spec(
    *,
    before_cp: int,
    after_cp: int | None = None,
    after_mate: tuple[Literal["white", "black"], int] | None = None,
    best_before: str | None = None,
    best_after: str | None = None,
) -> _MoveSpec:
    assert after_cp is not None or after_mate is not None
    return _MoveSpec(
        before_score=before_cp,
        after_score=after_mate if after_mate is not None else after_cp,
        best_before=best_before,
        best_after=best_after,
    )


def _evaluation(
    fen: str,
    score: ScoreSpec,
    *,
    best_uci: str | None,
) -> EngineEvaluation:
    board = chess.Board(fen)
    best_move = _engine_move(board, best_uci) if best_uci is not None else None
    return EngineEvaluation(
        status="ok",
        depth=1,
        score=_score(score),
        best_move=best_move,
        pv=(best_move,) if best_move is not None else (),
        nodes=1,
        time_ms=1,
    )


def _score(score: ScoreSpec) -> CentipawnScore | MateScore:
    if isinstance(score, int):
        return CentipawnScore(cp=score)
    return MateScore(winner=score[0], mate_in=score[1])


def _engine_move(board: chess.Board, uci: str) -> EngineMove:
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves
    return EngineMove(uci=uci, san=board.san(move))


def _required_motif(motifs: tuple[Motif, ...], motif_id: str) -> Motif:
    matches = [motif for motif in motifs if motif.id == motif_id]
    assert len(matches) == 1
    return matches[0]


def _motif_ids(motifs: tuple[Motif, ...]) -> list[str]:
    return [motif.id for motif in motifs]


def _api_motif(motif_id: MotifId, label: str) -> Motif:
    return Motif(
        id=motif_id,
        label=label,
        severity="mistake",
        source="heuristic",
        score_cp=220,
        evidence=MotifEvidence(
            threshold_cp=200,
            score_kind="cp",
            phase="middlegame",
            piece=PieceRef(color="white", role="knight", square="f3"),
            attackers=("d4 Q",),
            defenders=("e2 K",),
            best_move=MoveRef(uci="g1f3", san="Nf3"),
            opponent_reply=None,
            related_ply=None,
        ),
    )
