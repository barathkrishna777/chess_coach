"""Deterministic Slice 8 training and eval command."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from chess_ml.classifier.config import LABEL_ORDER, ClassifierConfig, load_classifier_config
from chess_ml.classifier.model import (
    SmallMotifNet,
    encode_row,
    label_vector_from_row,
    labels_from_row,
    stack_encoded,
)
from chess_ml.classifier.motifs import MotifId
from chess_ml.ingestion.lichess import read_examples_parquet, sha256_file

CHECKPOINT_SCHEMA_VERSION = "classifier-checkpoint.v1"
EVAL_REPORT_SCHEMA_VERSION = "classifier-eval-report.v1"


@dataclass(frozen=True)
class TrainResult:
    """The model, checkpoint payload, and report produced by training."""

    model: SmallMotifNet
    checkpoint: dict[str, object]
    report: dict[str, object]


@dataclass(frozen=True)
class TensorDataset:
    """In-memory tensors for a small local classifier dataset."""

    boards: torch.Tensor
    metadata: torch.Tensor
    labels: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.labels.shape[0])


@dataclass(frozen=True)
class SplitIndices:
    """Deterministic row indices for train/validation/test splits."""

    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor
    train_games: int
    validation_games: int
    test_games: int


def train_from_config(config_path: str | Path | None = None) -> TrainResult:
    """Train from the configured parquet dataset and write checkpoint/report."""

    config = (
        load_classifier_config(config_path) if config_path is not None else load_classifier_config()
    )
    rows = read_examples_parquet(config.dataset_path)
    result = train_rows(rows, config)
    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.checkpoint, config.checkpoint_path)
    config.eval_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.eval_report_path.write_text(
        json.dumps(result.report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def train_rows(rows: list[dict[str, object]], config: ClassifierConfig) -> TrainResult:
    """Train a deterministic tiny classifier on already loaded rows."""

    if not rows:
        raise ValueError("Cannot train classifier without any rows.")
    _set_seed(config.seed)
    dataset = _tensor_dataset(rows)
    splits = split_rows_by_game(
        rows,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    model = SmallMotifNet(
        hidden_channels=config.hidden_channels,
        dropout=config.dropout,
        label_count=len(LABEL_ORDER),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    loss_history: list[float] = []
    generator = torch.Generator().manual_seed(config.seed)

    model.train()
    for _epoch in range(config.epochs):
        permutation = splits.train[torch.randperm(len(splits.train), generator=generator)]
        for batch_indices in _batches(permutation, config.batch_size):
            optimizer.zero_grad()
            logits = model(dataset.boards[batch_indices], dataset.metadata[batch_indices])
            loss = loss_fn(logits, dataset.labels[batch_indices])
            loss.backward()
            optimizer.step()
            loss_history.append(round(float(loss.detach().item()), 6))

    calibrated_thresholds = calibrate_model_thresholds(
        model,
        dataset,
        splits.validation,
        defaults=config.thresholds,
    )
    validation_metrics = (
        evaluate_model(model, dataset, splits.validation, thresholds=calibrated_thresholds)
        if len(splits.validation) > 0
        else _empty_metrics()
    )
    model_metrics = evaluate_model(model, dataset, splits.test, thresholds=calibrated_thresholds)
    baseline_metrics = evaluate_predictions(
        dataset.labels[splits.test],
        dataset.labels[splits.test],
    )
    report = _report(
        config,
        rows=rows,
        splits=splits,
        examples=dataset.size,
        baseline_metrics=baseline_metrics,
        validation_metrics=validation_metrics,
        model_metrics=model_metrics,
        loss_history=loss_history,
        calibrated_thresholds=calibrated_thresholds,
    )
    checkpoint: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "label_order": list(LABEL_ORDER),
        "seed": config.seed,
        "hidden_channels": config.hidden_channels,
        "dropout": config.dropout,
        "thresholds": {label: calibrated_thresholds[label] for label in LABEL_ORDER},
        "model_state": model.state_dict(),
        "metrics": model_metrics,
    }
    return TrainResult(model=model, checkpoint=checkpoint, report=report)


def evaluate_model(
    model: SmallMotifNet,
    dataset: TensorDataset,
    indices: torch.Tensor,
    *,
    thresholds: Mapping[MotifId, float],
) -> dict[str, object]:
    """Evaluate a model on the selected rows."""

    if len(indices) == 0:
        return _empty_metrics()
    model.eval()
    probabilities = predict_probabilities(model, dataset, indices)
    threshold_tensor = torch.tensor(
        [float(thresholds[label]) for label in LABEL_ORDER],
        dtype=torch.float32,
    )
    predictions = probabilities >= threshold_tensor
    return evaluate_predictions(dataset.labels[indices], predictions.to(dtype=torch.float32))


def predict_probabilities(
    model: SmallMotifNet,
    dataset: TensorDataset,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Return sigmoid probabilities for selected dataset rows."""

    model.eval()
    with torch.no_grad():
        logits = model(dataset.boards[indices], dataset.metadata[indices])
        return torch.sigmoid(logits)


def calibrate_model_thresholds(
    model: SmallMotifNet,
    dataset: TensorDataset,
    indices: torch.Tensor,
    *,
    defaults: Mapping[MotifId, float],
) -> dict[MotifId, float]:
    """Calibrate per-label thresholds on validation rows."""

    if len(indices) == 0:
        return {label: float(defaults[label]) for label in LABEL_ORDER}
    probabilities = predict_probabilities(model, dataset, indices)
    targets = dataset.labels[indices]
    return calibrate_thresholds(probabilities, targets, defaults=defaults)


def calibrate_thresholds(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    defaults: Mapping[MotifId, float],
) -> dict[MotifId, float]:
    """Choose deterministic per-label thresholds that maximize validation F1."""

    calibrated: dict[MotifId, float] = {}
    for index, label in enumerate(LABEL_ORDER):
        default = float(defaults[label])
        target = targets[:, index] > 0.5
        if int(target.sum().item()) == 0:
            calibrated[label] = round(default, 4)
            continue

        values = probabilities[:, index]
        candidates = sorted(
            {
                default,
                0.05,
                0.10,
                0.20,
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.80,
                0.90,
                0.95,
                *(float(value.item()) for value in values),
            }
        )
        best_threshold = default
        best_key = (-1.0, -1.0, -1.0, -1.0)
        for candidate in candidates:
            predicted = values >= candidate
            row = _metric_row_from_bool(target, predicted)
            key = (
                float(row["f1"]),
                float(row["precision"]),
                float(row["recall"]),
                -abs(candidate - default),
            )
            if key > best_key:
                best_key = key
                best_threshold = candidate
        calibrated[label] = round(float(best_threshold), 4)
    return calibrated


def evaluate_predictions(targets: torch.Tensor, predictions: torch.Tensor) -> dict[str, object]:
    """Return per-label and micro precision/recall/f1 metrics."""

    per_label: dict[str, dict[str, float | int]] = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for index, label in enumerate(LABEL_ORDER):
        target = targets[:, index] > 0.5
        predicted = predictions[:, index] > 0.5
        tp = int(torch.logical_and(target, predicted).sum().item())
        fp = int(torch.logical_and(~target, predicted).sum().item())
        fn = int(torch.logical_and(target, ~predicted).sum().item())
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_label[label] = _metric_row(tp, fp, fn)

    return {
        "per_label": per_label,
        "micro": _metric_row(total_tp, total_fp, total_fn),
    }


def _metric_row_from_bool(target: torch.Tensor, predicted: torch.Tensor) -> dict[str, float | int]:
    tp = int(torch.logical_and(target, predicted).sum().item())
    fp = int(torch.logical_and(~target, predicted).sum().item())
    fn = int(torch.logical_and(target, ~predicted).sum().item())
    return _metric_row(tp, fp, fn)


def _empty_metrics() -> dict[str, object]:
    return {
        "per_label": {label: _metric_row(0, 0, 0) for label in LABEL_ORDER},
        "micro": _metric_row(0, 0, 0),
    }


def _tensor_dataset(rows: list[dict[str, object]]) -> TensorDataset:
    encoded = [encode_row(row) for row in rows]
    boards, metadata = stack_encoded(encoded)
    labels = torch.stack([label_vector_from_row(row) for row in rows])
    return TensorDataset(boards=boards, metadata=metadata, labels=labels)


def split_rows_by_game(
    rows: list[dict[str, object]],
    *,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> SplitIndices:
    """Split rows deterministically by game id, not by individual position."""

    if not rows:
        raise ValueError("Cannot split an empty dataset.")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1.")
    if not (0.0 <= validation_fraction < 1.0):
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must leave a test split.")

    row_game_ids = [_row_game_id(row, index) for index, row in enumerate(rows)]
    ordered_games = sorted(
        set(row_game_ids), key=lambda game_id: (_hash_key(seed, game_id), game_id)
    )
    train_games, validation_games, test_games = _split_game_ids(
        ordered_games,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )

    train_set = set(train_games)
    validation_set = set(validation_games)
    test_set = set(test_games)
    return SplitIndices(
        train=_indices_for_games(row_game_ids, train_set),
        validation=_indices_for_games(row_game_ids, validation_set),
        test=_indices_for_games(row_game_ids, test_set),
        train_games=len(train_set),
        validation_games=len(validation_set),
        test_games=len(test_set),
    )


def _split_game_ids(
    ordered_games: list[str],
    *,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    game_count = len(ordered_games)
    if game_count == 1:
        only = tuple(ordered_games)
        return only, only, only
    if game_count == 2:
        return (ordered_games[0],), (), (ordered_games[1],)

    train_count = max(1, min(game_count - 2, round(game_count * train_fraction)))
    validation_count = max(
        1, min(game_count - train_count - 1, round(game_count * validation_fraction))
    )
    test_count = game_count - train_count - validation_count
    if test_count < 1:
        validation_count -= 1
        test_count = 1
    train_end = train_count
    validation_end = train_count + validation_count
    return (
        tuple(ordered_games[:train_end]),
        tuple(ordered_games[train_end:validation_end]),
        tuple(ordered_games[validation_end : validation_end + test_count]),
    )


def _indices_for_games(row_game_ids: list[str], games: set[str]) -> torch.Tensor:
    return torch.tensor(
        [index for index, game_id in enumerate(row_game_ids) if game_id in games],
        dtype=torch.long,
    )


def _row_game_id(row: Mapping[str, object], index: int) -> str:
    value = row.get("game_id")
    if isinstance(value, str) and value:
        return value
    return f"row:{index}"


def _hash_key(seed: int, game_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{game_id}".encode()).hexdigest()
    return int(digest, 16)


def _batches(indices: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def _metric_row(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _report(
    config: ClassifierConfig,
    *,
    rows: list[dict[str, object]],
    splits: SplitIndices,
    examples: int,
    baseline_metrics: dict[str, object],
    validation_metrics: dict[str, object],
    model_metrics: dict[str, object],
    loss_history: list[float],
    calibrated_thresholds: Mapping[MotifId, float],
) -> dict[str, object]:
    source_path = config.raw_path or config.source_pgn
    return {
        "schema_version": EVAL_REPORT_SCHEMA_VERSION,
        "slice": config.slice_name,
        "status": "trained",
        "seed": config.seed,
        "source": {
            "url": config.source_url,
            "path": str(source_path),
            "sha256": sha256_file(source_path) if source_path.exists() else None,
            "min_elo": config.min_elo,
            "max_elo": config.max_elo,
            "rated_only": config.rated_only,
            "target_examples": config.target_examples,
        },
        "dataset_path": str(config.dataset_path),
        "checkpoint_path": str(config.checkpoint_path),
        "eval_report_path": str(config.eval_report_path),
        "label_order": list(LABEL_ORDER),
        "examples": examples,
        "label_distribution": _label_distribution(rows),
        "splits": {
            "train_examples": len(splits.train),
            "validation_examples": len(splits.validation),
            "test_examples": len(splits.test),
            "train_games": splits.train_games,
            "validation_games": splits.validation_games,
            "test_games": splits.test_games,
        },
        "train_examples": len(splits.train),
        "eval_examples": len(splits.test),
        "training": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "train_fraction": config.train_fraction,
            "validation_fraction": config.validation_fraction,
            "loss_steps": len(loss_history),
            "loss_first": loss_history[:5],
            "loss_last": loss_history[-5:],
        },
        "thresholds": {
            "configured": {label: config.thresholds[label] for label in LABEL_ORDER},
            "calibrated": {label: calibrated_thresholds[label] for label in LABEL_ORDER},
        },
        "metrics": {
            "baseline": baseline_metrics,
            "validation": validation_metrics,
            "model": model_metrics,
        },
        "notes": (
            "Slice 16 trains on weak labels generated from Stockfish-gated heuristics. "
            "Baseline metrics compare heuristic weak labels against themselves; learned "
            "metrics are held out by game id and use validation-calibrated thresholds."
        ),
    }


def _label_distribution(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = dict.fromkeys(LABEL_ORDER, 0)
    for row in rows:
        for label in labels_from_row(row):
            counts[label] += 1
    return counts


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def main() -> None:
    """CLI entrypoint for `python -m chess_ml.classifier.train`."""

    result = train_from_config()
    report = result.report
    print(
        f"Trained {report['slice']} classifier on "
        f"{report['examples']} examples; wrote {report['checkpoint_path']} "
        f"and {report['dataset_path']}; refreshed {report['eval_report_path']}"
    )


if __name__ == "__main__":
    main()
