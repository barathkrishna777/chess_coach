"""Optional runtime inference for the Slice 8 learned classifier."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast

import torch

from chess_ml.classifier.config import (
    DEFAULT_CLASSIFIER_CHECKPOINT_PATH,
    LABEL_ORDER,
    load_classifier_config,
)
from chess_ml.classifier.model import SmallMotifNet, encode_analyzed_move, stack_encoded
from chess_ml.classifier.motifs import AnalyzedMove, MotifId
from chess_ml.classifier.train import CHECKPOINT_SCHEMA_VERSION


@dataclass(frozen=True)
class LearnedPrediction:
    """One learned motif probability."""

    label: MotifId
    probability: float


class LearnedMotifClassifier:
    """A schema-checked local learned classifier checkpoint."""

    def __init__(
        self,
        *,
        model: SmallMotifNet,
        thresholds: Mapping[MotifId, float],
    ) -> None:
        self.model = model
        self.thresholds = dict(thresholds)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> LearnedMotifClassifier:
        """Load and validate a local checkpoint."""

        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Classifier checkpoint must be a dictionary.")
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Unsupported classifier checkpoint schema.")
        if tuple(checkpoint.get("label_order", ())) != LABEL_ORDER:
            raise ValueError("Classifier checkpoint label order does not match this build.")

        hidden_channels = _int(checkpoint, "hidden_channels")
        dropout = _float(checkpoint, "dropout")
        thresholds = _thresholds(checkpoint.get("thresholds"))
        model = SmallMotifNet(
            hidden_channels=hidden_channels,
            dropout=dropout,
            label_count=len(LABEL_ORDER),
        )
        state = checkpoint.get("model_state")
        if not isinstance(state, dict):
            raise ValueError("Classifier checkpoint is missing model_state.")
        model.load_state_dict(state)
        return cls(model=model, thresholds=thresholds)

    def predict(self, moves: list[AnalyzedMove]) -> list[list[LearnedPrediction]]:
        """Return above-threshold learned motif predictions for each move."""

        if not moves:
            return []
        encoded = [encode_analyzed_move(move) for move in moves]
        boards, metadata = stack_encoded(encoded)
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(boards, metadata))
        results: list[list[LearnedPrediction]] = []
        for row in probabilities:
            move_predictions: list[LearnedPrediction] = []
            for index, label in enumerate(LABEL_ORDER):
                probability = float(row[index].item())
                if probability >= self.thresholds[label]:
                    move_predictions.append(LearnedPrediction(label=label, probability=probability))
            results.append(move_predictions)
        return results


@lru_cache(maxsize=1)
def learned_classifier_from_env() -> LearnedMotifClassifier | None:
    """Load the optional runtime classifier, or None for heuristic-only mode."""

    configured_path = os.environ.get("CHESS_ML_CLASSIFIER_CHECKPOINT")
    if configured_path is not None and not configured_path.strip():
        return None

    checkpoint_path = (
        Path(configured_path) if configured_path is not None else _default_checkpoint_path()
    )
    if not checkpoint_path.exists():
        return None

    try:
        return LearnedMotifClassifier.from_checkpoint(checkpoint_path)
    except (OSError, RuntimeError, ValueError):
        return None


def _default_checkpoint_path() -> Path:
    configured_config_path = os.environ.get("CHESS_ML_CLASSIFIER_CONFIG")
    if configured_config_path is None or not configured_config_path.strip():
        return DEFAULT_CLASSIFIER_CHECKPOINT_PATH
    try:
        return load_classifier_config(configured_config_path).checkpoint_path
    except (OSError, ValueError):
        return DEFAULT_CLASSIFIER_CHECKPOINT_PATH


def _thresholds(value: object) -> dict[MotifId, float]:
    if not isinstance(value, dict):
        raise ValueError("Classifier checkpoint thresholds must be a dictionary.")
    raw = cast(dict[object, object], value)
    thresholds: dict[MotifId, float] = {}
    for label in LABEL_ORDER:
        item = raw.get(label)
        if not isinstance(item, int | float):
            raise ValueError(f"Classifier checkpoint threshold missing for {label}.")
        thresholds[label] = float(item)
    return thresholds


def _int(mapping: Mapping[object, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, int):
        return value
    raise ValueError(f"Classifier checkpoint value {key} must be an integer.")


def _float(mapping: Mapping[object, object], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"Classifier checkpoint value {key} must be numeric.")
