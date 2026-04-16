"""Configuration for the Slice 8 learned classifier."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from chess_ml.classifier.motifs import MotifId

DEFAULT_CLASSIFIER_CONFIG_PATH = Path("configs/classifier/slice8-v1.toml")
DEFAULT_CLASSIFIER_CHECKPOINT_PATH = Path("checkpoints/classifier/slice8-v1.pt")
DEFAULT_CLASSIFIER_DATASET_PATH = Path("data/processed/slice8-v1.parquet")
DEFAULT_CLASSIFIER_EVAL_REPORT_PATH = Path("docs/evals/009-slice8-classifier-v1.json")

LABEL_ORDER: tuple[MotifId, ...] = (
    "hanging_piece",
    "missed_tactic",
    "allowed_tactic",
    "endgame_slip",
    "opening_inaccuracy",
    "pin",
    "fork",
    "overloaded_defender",
    "discovered_attack",
)


@dataclass(frozen=True)
class ClassifierConfig:
    """Parsed configuration for ingestion, training, eval, and inference."""

    seed: int
    source_pgn: Path
    dataset_path: Path
    max_games: int
    max_plies_per_game: int
    analysis_depth: int
    checkpoint_path: Path
    hidden_channels: int
    dropout: float
    epochs: int
    batch_size: int
    learning_rate: float
    train_fraction: float
    eval_report_path: Path
    thresholds: dict[MotifId, float]


def load_classifier_config(path: str | Path = DEFAULT_CLASSIFIER_CONFIG_PATH) -> ClassifierConfig:
    """Load the checked-in classifier config."""

    config_path = Path(path)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    run = _section(raw, "run")
    data = _section(raw, "data")
    model = _section(raw, "model")
    training = _section(raw, "training")
    eval_section = _section(raw, "eval")
    thresholds = _section(raw, "thresholds")

    return ClassifierConfig(
        seed=_int(run, "seed", 20260416),
        source_pgn=_path(data, "source_pgn", Path("tests/fixtures/classifier/slice8-mini.pgn")),
        dataset_path=_path(data, "dataset_path", DEFAULT_CLASSIFIER_DATASET_PATH),
        max_games=_int(data, "max_games", 8),
        max_plies_per_game=_int(data, "max_plies_per_game", 80),
        analysis_depth=_int(data, "analysis_depth", 4),
        checkpoint_path=_path(model, "checkpoint_path", DEFAULT_CLASSIFIER_CHECKPOINT_PATH),
        hidden_channels=_int(model, "hidden_channels", 32),
        dropout=_float(model, "dropout", 0.1),
        epochs=_int(training, "epochs", 8),
        batch_size=_int(training, "batch_size", 16),
        learning_rate=_float(training, "learning_rate", 0.003),
        train_fraction=_float(training, "train_fraction", 0.75),
        eval_report_path=_path(
            eval_section,
            "report_path",
            DEFAULT_CLASSIFIER_EVAL_REPORT_PATH,
        ),
        thresholds={label: _float(thresholds, label, 0.62) for label in LABEL_ORDER},
    )


def _section(raw: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = raw.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Classifier config section [{name}] must be a table.")
    return cast(Mapping[str, object], value)


def _int(section: Mapping[str, object], key: str, default: int) -> int:
    value = section.get(key, default)
    if isinstance(value, int):
        return value
    raise ValueError(f"Classifier config value {key} must be an integer.")


def _float(section: Mapping[str, object], key: str, default: float) -> float:
    value = section.get(key, default)
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"Classifier config value {key} must be a number.")


def _path(section: Mapping[str, object], key: str, default: Path) -> Path:
    value = section.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        return Path(value)
    raise ValueError(f"Classifier config value {key} must be a path string.")
