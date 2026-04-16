"""Tests for Slice 8 learned classifier utilities."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import chess
import torch

from chess_ml.classifier.classify import classify_moves
from chess_ml.classifier.config import LABEL_ORDER, ClassifierConfig
from chess_ml.classifier.learned import LearnedMotifClassifier, LearnedPrediction
from chess_ml.classifier.model import BOARD_PLANES, METADATA_FEATURES, encode_analyzed_move
from chess_ml.classifier.motifs import AnalyzedMove, MotifId
from chess_ml.classifier.train import train_rows
from chess_ml.engine.stockfish import CentipawnScore, EngineEvaluation, EngineMove
from chess_ml.ingestion.lichess import (
    LabeledPositionExample,
    examples_for_game,
    read_examples_parquet,
    read_standard_games,
    write_examples_parquet,
)
from chess_ml.ingestion.pgn import parse_pgn


def test_fixture_pgn_reader_accepts_multiple_games() -> None:
    games = list(
        read_standard_games(
            "tests/fixtures/classifier/slice8-mini.pgn",
            max_games=8,
            max_plies=80,
        )
    )

    assert len(games) == 3
    assert games[0].headers["White"] == "Ada"


def test_examples_for_game_uses_heuristics_as_weak_labels() -> None:
    parsed = parse_pgn(
        """
[Event "Fixture"]
[Result "*"]

1. e4 *
"""
    )
    evaluations = {
        parsed.initial_fen: _evaluation(parsed.initial_fen, cp=0, best_uci="d2d4"),
        parsed.final_fen: _evaluation(parsed.final_fen, cp=-80, best_uci="e7e5"),
    }

    examples = examples_for_game(parsed, evaluations)

    assert len(examples) == 1
    assert examples[0].loss_cp == 80
    assert examples[0].labels == ("opening_inaccuracy",)


def test_parquet_roundtrip_preserves_label_columns(tmp_path: Path) -> None:
    path = tmp_path / "examples.parquet"
    write_examples_parquet(
        [
            LabeledPositionExample(
                game_id="game-1",
                ply=1,
                move_number=1,
                side="white",
                san="e4",
                uci="e2e4",
                from_square="e2",
                to_square="e4",
                fen_before=chess.STARTING_FEN,
                fen_after=chess.STARTING_FEN,
                eval_before_cp=0,
                eval_after_cp=-80,
                loss_cp=80,
                is_engine_best=False,
                labels=("opening_inaccuracy",),
            )
        ],
        path,
    )

    rows = read_examples_parquet(path)

    assert rows[0]["uci"] == "e2e4"
    assert rows[0]["label_opening_inaccuracy"] is True
    assert rows[0]["label_hanging_piece"] is False


def test_encode_analyzed_move_shapes_are_stable() -> None:
    parsed = parse_pgn(
        """
[Result "*"]

1. e4 *
"""
    )
    move = parsed.moves[0]
    analyzed = AnalyzedMove(
        ply=move.ply,
        move_number=move.move_number,
        side=move.side,
        san=move.san,
        uci=move.uci,
        fen_before=move.fen_before,
        fen_after=move.fen_after,
        analysis_before=_evaluation(move.fen_before, cp=0, best_uci="d2d4"),
        analysis_after=_evaluation(move.fen_after, cp=-80, best_uci="e7e5"),
    )

    encoded = encode_analyzed_move(analyzed)

    assert tuple(encoded.board.shape) == (BOARD_PLANES, 8, 8)
    assert tuple(encoded.metadata.shape) == (METADATA_FEATURES,)


def test_training_is_reproducible_on_tiny_rows(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = train_rows(_training_rows(), config)
    second = train_rows(_training_rows(), config)

    assert first.report["training"] == second.report["training"]
    assert first.report["metrics"] == second.report["metrics"]


def test_checkpoint_loader_validates_trained_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = train_rows(_training_rows(), config)
    checkpoint_path = tmp_path / "classifier.pt"
    torch.save(result.checkpoint, checkpoint_path)

    classifier = LearnedMotifClassifier.from_checkpoint(checkpoint_path)

    assert classifier.thresholds["opening_inaccuracy"] == config.thresholds["opening_inaccuracy"]


def test_classify_moves_can_add_learned_only_motif() -> None:
    parsed = parse_pgn(
        """
[Result "*"]

1. e4 *
"""
    )
    move = parsed.moves[0]
    analyzed = AnalyzedMove(
        ply=move.ply,
        move_number=move.move_number,
        side=move.side,
        san=move.san,
        uci=move.uci,
        fen_before=move.fen_before,
        fen_after=move.fen_after,
        analysis_before=_evaluation(move.fen_before, cp=200, best_uci="d2d4"),
        analysis_after=_evaluation(move.fen_after, cp=-200, best_uci="e7e5"),
    )

    motifs = classify_moves(
        [analyzed],
        initial_fen=parsed.initial_fen,
        learned_classifier=cast(LearnedMotifClassifier, _FakeLearned("missed_tactic")),
    )

    assert [motif.id for motif in motifs[0]] == ["missed_tactic"]
    assert motifs[0][0].source == "learned"


def test_classify_moves_marks_heuristic_and_learned_agreement_as_ensemble() -> None:
    parsed = parse_pgn(
        """
[Result "*"]

1. e4 *
"""
    )
    move = parsed.moves[0]
    analyzed = AnalyzedMove(
        ply=move.ply,
        move_number=move.move_number,
        side=move.side,
        san=move.san,
        uci=move.uci,
        fen_before=move.fen_before,
        fen_after=move.fen_after,
        analysis_before=_evaluation(move.fen_before, cp=0, best_uci="d2d4"),
        analysis_after=_evaluation(move.fen_after, cp=-80, best_uci="e7e5"),
    )

    motifs = classify_moves(
        [analyzed],
        initial_fen=parsed.initial_fen,
        learned_classifier=cast(LearnedMotifClassifier, _FakeLearned("opening_inaccuracy")),
    )

    assert [motif.id for motif in motifs[0]] == ["opening_inaccuracy"]
    assert motifs[0][0].source == "ensemble"


class _FakeLearned:
    def __init__(self, label: str) -> None:
        self.label = cast(MotifId, label)

    def predict(self, moves: list[AnalyzedMove]) -> list[list[LearnedPrediction]]:
        return [[LearnedPrediction(label=self.label, probability=0.99)] for _move in moves]


def _training_rows() -> list[dict[str, object]]:
    return [
        _row("e2e4", label_opening_inaccuracy=True, loss_cp=80),
        _row("d2d4", loss_cp=0),
        _row("g1f3", label_missed_tactic=True, loss_cp=360),
        _row("c2c4", loss_cp=20),
    ]


def _row(
    uci: str,
    *,
    loss_cp: int,
    label_opening_inaccuracy: bool = False,
    label_missed_tactic: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "fen_before": chess.STARTING_FEN,
        "uci": uci,
        "side": "white",
        "eval_before_cp": 0,
        "loss_cp": loss_cp,
    }
    for label in LABEL_ORDER:
        row[f"label_{label}"] = False
    row["label_opening_inaccuracy"] = label_opening_inaccuracy
    row["label_missed_tactic"] = label_missed_tactic
    return row


def _config(tmp_path: Path) -> ClassifierConfig:
    return ClassifierConfig(
        seed=20260416,
        source_pgn=Path("tests/fixtures/classifier/slice8-mini.pgn"),
        dataset_path=tmp_path / "slice8.parquet",
        max_games=2,
        max_plies_per_game=40,
        analysis_depth=1,
        checkpoint_path=tmp_path / "classifier.pt",
        hidden_channels=4,
        dropout=0.0,
        epochs=2,
        batch_size=2,
        learning_rate=0.001,
        train_fraction=0.75,
        eval_report_path=tmp_path / "report.json",
        thresholds=dict.fromkeys(LABEL_ORDER, 0.5),
    )


def _evaluation(fen: str, *, cp: int, best_uci: str) -> EngineEvaluation:
    board = chess.Board(fen)
    move = chess.Move.from_uci(best_uci)
    assert move in board.legal_moves
    engine_move = EngineMove(uci=best_uci, san=board.san(move))
    return EngineEvaluation(
        status="ok",
        depth=1,
        score=CentipawnScore(cp=cp),
        best_move=engine_move,
        pv=(engine_move,),
        nodes=1,
        time_ms=1,
    )
