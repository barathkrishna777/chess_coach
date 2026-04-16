"""Tests for Slice 8 learned classifier utilities."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import chess
import torch

from chess_ml.classifier.classify import classify_moves
from chess_ml.classifier.config import (
    DEFAULT_LICHESS_SOURCE_URL,
    LABEL_ORDER,
    ClassifierConfig,
    load_classifier_config,
)
from chess_ml.classifier.learned import LearnedMotifClassifier, LearnedPrediction
from chess_ml.classifier.model import (
    BOARD_PLANES,
    METADATA_FEATURES,
    SmallMotifNet,
    encode_analyzed_move,
)
from chess_ml.classifier.motifs import AnalyzedMove, MotifId
from chess_ml.classifier.train import (
    CHECKPOINT_SCHEMA_VERSION,
    calibrate_thresholds,
    split_rows_by_game,
    train_rows,
)
from chess_ml.engine.stockfish import CentipawnScore, EngineEvaluation, EngineMove
from chess_ml.ingestion.lichess import (
    LabeledPositionExample,
    build_dataset,
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


def test_slice16_config_defaults_to_real_lichess_run() -> None:
    config = load_classifier_config("configs/classifier/slice16-lichess-v1.toml")

    assert config.slice_name == "slice16-lichess-v1"
    assert config.source_url == DEFAULT_LICHESS_SOURCE_URL
    assert config.raw_path == Path("data/raw/lichess/lichess_db_standard_rated_2013-01.pgn.zst")
    assert config.dataset_path == Path("data/processed/slice16-lichess-v1.parquet")
    assert config.target_examples == 10000
    assert config.min_elo == 1200
    assert config.max_elo == 2000
    assert config.rated_only is True
    assert config.validation_fraction == 0.15


def test_pgn_reader_filters_rated_games_by_elo(tmp_path: Path) -> None:
    pgn_path = tmp_path / "lichess.pgn"
    pgn_path.write_text(
        """
[Event "Rated Classical game"]
[White "Ada"]
[Black "Turing"]
[WhiteElo "1500"]
[BlackElo "1600"]
[Result "*"]

1. e4 *

[Event "Too high"]
[White "Grace"]
[Black "Noether"]
[WhiteElo "2100"]
[BlackElo "1600"]
[Rated "True"]
[Result "*"]

1. d4 *

[Event "Casual"]
[White "Hypatia"]
[Black "Lovelace"]
[WhiteElo "1500"]
[BlackElo "1600"]
[Rated "False"]
[Result "*"]

1. c4 *
""",
        encoding="utf-8",
    )

    games = list(
        read_standard_games(
            pgn_path,
            max_games=10,
            max_plies=20,
            min_elo=1200,
            max_elo=2000,
            rated_only=True,
        )
    )

    assert len(games) == 1
    assert games[0].headers["Event"] == "Rated Classical game"


def test_build_dataset_stops_after_target_examples(tmp_path: Path) -> None:
    pgn_path = tmp_path / "source.pgn"
    pgn_path.write_text(
        """
[Event "Game 1"]
[White "A"]
[Black "B"]
[WhiteElo "1500"]
[BlackElo "1600"]
[Rated "True"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 *

[Event "Game 2"]
[White "C"]
[Black "D"]
[WhiteElo "1500"]
[BlackElo "1600"]
[Rated "True"]
[Result "*"]

1. d4 d5 2. c4 e6 *

[Event "Game 3"]
[White "E"]
[Black "F"]
[WhiteElo "1500"]
[BlackElo "1600"]
[Rated "True"]
[Result "*"]

1. c4 e5 2. Nc3 Nf6 *
""",
        encoding="utf-8",
    )
    config = _config(tmp_path, source_pgn=pgn_path, target_examples=5, max_games=10)

    summary = asyncio.run(build_dataset(config, evaluator=_FakeEvaluator()))

    assert summary.games_read == 2
    assert summary.examples_written == 8
    assert summary.dataset_path.exists()


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
    assert all(f"label_{label}" in rows[0] for label in LABEL_ORDER)


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


def test_training_report_records_slice16_eval_sections(tmp_path: Path) -> None:
    config = _config(tmp_path)

    result = train_rows(_training_rows(), config)
    source = cast(dict[str, object], result.report["source"])
    distribution = cast(dict[str, int], result.report["label_distribution"])
    splits = cast(dict[str, int], result.report["splits"])
    thresholds = cast(dict[str, dict[str, float]], result.report["thresholds"])
    metrics = cast(dict[str, object], result.report["metrics"])

    assert result.report["schema_version"] == "classifier-eval-report.v1"
    assert result.report["slice"] == "slice16-lichess-v1"
    assert source["path"] == "tests/fixtures/classifier/slice8-mini.pgn"
    assert set(distribution) == set(LABEL_ORDER)
    assert splits["train_examples"] > 0
    assert splits["test_examples"] > 0
    assert set(thresholds["calibrated"]) == set(LABEL_ORDER)
    assert "validation" in metrics


def test_checkpoint_loader_validates_trained_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = train_rows(_training_rows(), config)
    checkpoint_path = tmp_path / "classifier.pt"
    torch.save(result.checkpoint, checkpoint_path)

    classifier = LearnedMotifClassifier.from_checkpoint(checkpoint_path)

    assert set(classifier.thresholds) == set(LABEL_ORDER)
    assert classifier.label_order == LABEL_ORDER


def test_game_level_splits_are_deterministic_and_isolated() -> None:
    rows = [
        _row("e2e4", game_id=f"game-{game}", loss_cp=0)
        for game in range(8)
        for _position in range(2)
    ]

    first = split_rows_by_game(
        rows,
        train_fraction=0.6,
        validation_fraction=0.2,
        seed=20260416,
    )
    second = split_rows_by_game(
        rows,
        train_fraction=0.6,
        validation_fraction=0.2,
        seed=20260416,
    )

    assert first.train.tolist() == second.train.tolist()
    assert first.validation.tolist() == second.validation.tolist()
    assert first.test.tolist() == second.test.tolist()

    train_games = _games_for_indices(rows, first.train)
    validation_games = _games_for_indices(rows, first.validation)
    test_games = _games_for_indices(rows, first.test)
    assert train_games.isdisjoint(validation_games)
    assert train_games.isdisjoint(test_games)
    assert validation_games.isdisjoint(test_games)


def test_threshold_calibration_uses_validation_signal() -> None:
    probabilities = torch.full((3, len(LABEL_ORDER)), 0.1)
    probabilities[:, 0] = torch.tensor([0.1, 0.8, 0.7])
    targets = torch.zeros((3, len(LABEL_ORDER)))
    targets[:, 0] = torch.tensor([0.0, 1.0, 1.0])

    thresholds = calibrate_thresholds(
        probabilities,
        targets,
        defaults=dict.fromkeys(LABEL_ORDER, 0.9),
    )

    assert thresholds["hanging_piece"] == 0.7
    assert thresholds["missed_tactic"] == 0.9


def test_checkpoint_loader_accepts_old_label_order_prefix(tmp_path: Path) -> None:
    old_label_order = LABEL_ORDER[:5]
    model = SmallMotifNet(hidden_channels=4, dropout=0.0, label_count=len(old_label_order))
    checkpoint_path = tmp_path / "old-classifier.pt"
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "label_order": list(old_label_order),
            "thresholds": dict.fromkeys(old_label_order, 0.5),
            "hidden_channels": 4,
            "dropout": 0.0,
            "model_state": model.state_dict(),
        },
        checkpoint_path,
    )

    classifier = LearnedMotifClassifier.from_checkpoint(checkpoint_path)

    assert classifier.label_order == old_label_order
    assert set(classifier.thresholds) == set(old_label_order)


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
        _row("e2e4", game_id="game-1", label_opening_inaccuracy=True, loss_cp=80),
        _row("d2d4", game_id="game-2", loss_cp=0),
        _row("g1f3", game_id="game-3", label_missed_tactic=True, loss_cp=360),
        _row("c2c4", game_id="game-4", loss_cp=20),
    ]


def _row(
    uci: str,
    *,
    loss_cp: int,
    game_id: str = "game-1",
    label_opening_inaccuracy: bool = False,
    label_missed_tactic: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "game_id": game_id,
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


def _config(
    tmp_path: Path,
    *,
    source_pgn: Path = Path("tests/fixtures/classifier/slice8-mini.pgn"),
    target_examples: int | None = None,
    max_games: int = 2,
) -> ClassifierConfig:
    return ClassifierConfig(
        slice_name="slice16-lichess-v1",
        seed=20260416,
        source_pgn=source_pgn,
        dataset_path=tmp_path / "slice8.parquet",
        max_games=max_games,
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
        target_examples=target_examples,
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


class _FakeEvaluator:
    async def evaluate(self, fen: str, *, depth: int | None = None) -> EngineEvaluation:
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=False):
            return EngineEvaluation(
                status="terminal",
                depth=None,
                score=CentipawnScore(cp=0),
                best_move=None,
                pv=(),
                nodes=None,
                time_ms=1,
            )
        move = next(iter(board.legal_moves))
        engine_move = EngineMove(uci=move.uci(), san=board.san(move))
        return EngineEvaluation(
            status="ok",
            depth=depth,
            score=CentipawnScore(cp=0),
            best_move=engine_move,
            pv=(engine_move,),
            nodes=1,
            time_ms=1,
        )


def _games_for_indices(rows: list[dict[str, object]], indices: torch.Tensor) -> set[str]:
    return {str(rows[index]["game_id"]) for index in indices.tolist()}
