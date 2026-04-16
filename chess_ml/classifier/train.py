"""Deterministic Slice 8 training and eval command."""

from __future__ import annotations

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
    stack_encoded,
)
from chess_ml.classifier.motifs import MotifId
from chess_ml.ingestion.lichess import read_examples_parquet

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
    train_indices, eval_indices = _split_indices(
        dataset.labels,
        train_fraction=config.train_fraction,
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
        permutation = train_indices[torch.randperm(len(train_indices), generator=generator)]
        for batch_indices in _batches(permutation, config.batch_size):
            optimizer.zero_grad()
            logits = model(dataset.boards[batch_indices], dataset.metadata[batch_indices])
            loss = loss_fn(logits, dataset.labels[batch_indices])
            loss.backward()
            optimizer.step()
            loss_history.append(round(float(loss.detach().item()), 6))

    model_metrics = evaluate_model(model, dataset, eval_indices, thresholds=config.thresholds)
    baseline_metrics = evaluate_predictions(
        dataset.labels[eval_indices],
        dataset.labels[eval_indices],
    )
    report = _report(
        config,
        examples=dataset.size,
        train_examples=len(train_indices),
        eval_examples=len(eval_indices),
        baseline_metrics=baseline_metrics,
        model_metrics=model_metrics,
        loss_history=loss_history,
    )
    checkpoint: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "label_order": list(LABEL_ORDER),
        "seed": config.seed,
        "hidden_channels": config.hidden_channels,
        "dropout": config.dropout,
        "thresholds": {label: config.thresholds[label] for label in LABEL_ORDER},
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

    model.eval()
    with torch.no_grad():
        logits = model(dataset.boards[indices], dataset.metadata[indices])
        probabilities = torch.sigmoid(logits)
    threshold_tensor = torch.tensor(
        [float(thresholds[label]) for label in LABEL_ORDER],
        dtype=torch.float32,
    )
    predictions = probabilities >= threshold_tensor
    return evaluate_predictions(dataset.labels[indices], predictions.to(dtype=torch.float32))


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


def _tensor_dataset(rows: list[dict[str, object]]) -> TensorDataset:
    encoded = [encode_row(row) for row in rows]
    boards, metadata = stack_encoded(encoded)
    labels = torch.stack([label_vector_from_row(row) for row in rows])
    return TensorDataset(boards=boards, metadata=metadata, labels=labels)


def _split_indices(
    labels: torch.Tensor,
    *,
    train_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    size = int(labels.shape[0])
    if size <= 0:
        raise ValueError("Cannot split an empty dataset.")
    if size == 1:
        single = torch.tensor([0], dtype=torch.long)
        return single, single

    positive_mask = labels.sum(dim=1) > 0
    all_indices = torch.arange(size, dtype=torch.long)
    positive_indices = all_indices[positive_mask]
    negative_indices = all_indices[~positive_mask]

    train_parts: list[torch.Tensor] = []
    eval_parts: list[torch.Tensor] = []
    generator = torch.Generator().manual_seed(seed)
    for group in (positive_indices, negative_indices):
        if len(group) == 0:
            continue
        group_train, group_eval = _split_group(
            group, train_fraction=train_fraction, generator=generator
        )
        train_parts.append(group_train)
        eval_parts.append(group_eval)

    train_indices = torch.cat(train_parts) if train_parts else all_indices[:1]
    eval_indices = torch.cat(eval_parts) if eval_parts else all_indices[-1:]
    return train_indices.sort().values, eval_indices.sort().values


def _split_group(
    indices: torch.Tensor,
    *,
    train_fraction: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(indices) == 1:
        return indices, indices
    shuffled = indices[torch.randperm(len(indices), generator=generator)]
    train_size = max(1, min(len(indices) - 1, round(len(indices) * train_fraction)))
    return shuffled[:train_size], shuffled[train_size:]


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
    examples: int,
    train_examples: int,
    eval_examples: int,
    baseline_metrics: dict[str, object],
    model_metrics: dict[str, object],
    loss_history: list[float],
) -> dict[str, object]:
    return {
        "schema_version": EVAL_REPORT_SCHEMA_VERSION,
        "slice": "slice8-v1",
        "status": "trained",
        "seed": config.seed,
        "dataset_path": str(config.dataset_path),
        "checkpoint_path": str(config.checkpoint_path),
        "eval_report_path": str(config.eval_report_path),
        "label_order": list(LABEL_ORDER),
        "examples": examples,
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "training": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "train_fraction": config.train_fraction,
            "loss_history": loss_history,
        },
        "metrics": {
            "baseline": baseline_metrics,
            "model": model_metrics,
        },
        "notes": (
            "Fixture-sized Slice 8 smoke eval. Baseline metrics compare heuristic weak labels "
            "against themselves; model metrics confirm the checkpoint/eval path is reproducible, "
            "not production-quality motif recall."
        ),
    }


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
        "Trained Slice 8 classifier on "
        f"{report['examples']} examples; wrote {report['checkpoint_path']} "
        f"and {report['dataset_path']}; refreshed {report['eval_report_path']}"
    )


if __name__ == "__main__":
    main()
