"""Shared explanation dataclasses and literal types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from chess_ml.classifier.motifs import Motif
from chess_ml.engine.stockfish import EngineEvaluation
from chess_ml.ingestion.pgn import Side

PROMPT_VERSION = "grounded-coach.v1"
EXPLANATION_SCHEMA_VERSION = "move-explanation.v1"

ExplanationProvider: TypeAlias = Literal["anthropic", "codex", "ollama"]
ExplanationSource: TypeAlias = Literal["cache", "llm"]
ExplanationStatus: TypeAlias = Literal["ok", "unavailable", "error"]
ExplanationReason: TypeAlias = Literal[
    "api_key_missing",
    "provider_error",
    "invalid_response",
    "local_model_unavailable",
    "timeout",
]


@dataclass(frozen=True)
class LineMove:
    """One move in a short SAN/UCI continuation."""

    ply: int
    side: Side
    san: str
    uci: str


@dataclass(frozen=True)
class ExplanationRequest:
    """The engine-grounded facts needed to explain one flagged move."""

    ply: int
    move_number: int
    side: Side
    san: str
    uci: str
    fen_before: str
    fen_after: str
    analysis_before: EngineEvaluation
    analysis_after: EngineEvaluation
    loss_cp: int | None
    actual_line: tuple[LineMove, ...]
    motifs: tuple[Motif, ...]


@dataclass(frozen=True)
class MoveExplanation:
    """Public move-level explanation status returned by the API."""

    status: ExplanationStatus
    text: str | None
    source: ExplanationSource | None
    provider: ExplanationProvider | None
    model: str | None
    reason: ExplanationReason | None
    timeout_seconds: float | None = None
    retryable: bool = False
    schema_version: str = EXPLANATION_SCHEMA_VERSION
    prompt_version: str = PROMPT_VERSION
