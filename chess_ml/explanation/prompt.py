"""Prompt assembly and response validation for grounded coaching explanations."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from chess_ml.classifier.motifs import Motif, MoveRef
from chess_ml.engine.stockfish import CentipawnScore, EngineEvaluation, EngineMove, MateScore
from chess_ml.explanation.models import PROMPT_VERSION, ExplanationRequest

SYSTEM_PROMPT = """You are a chess coach explaining one mistake to a 1200-2000 rated club player.
Stockfish is ground truth: never contradict the provided engine best move or PV.
Use only the supplied FEN, moves, engine line, eval swing, and motif evidence; do not invent tactics.
Reference concrete pieces, squares, or moves from the supplied facts.
Teach exactly one practical lesson.
Return strict JSON only: {"text":"...","referenced_move_uci":"..."}. The text must be at most 3 sentences and about 70 words or fewer."""

SEVERITY_RANK: Mapping[str, int] = {"blunder": 3, "mistake": 2, "inaccuracy": 1}
MOTIF_RANK: Mapping[str, int] = {
    "allowed_tactic": 5,
    "missed_tactic": 4,
    "endgame_slip": 3,
    "hanging_piece": 2,
    "opening_inaccuracy": 1,
}
TACTIC_AFTER_MOTIFS = {"allowed_tactic"}


class InvalidExplanationResponseError(ValueError):
    """Raised when the provider output cannot be trusted."""


@dataclass(frozen=True)
class BuiltPrompt:
    """A provider-ready prompt plus structured facts for cache keys and tests."""

    system_prompt: str
    user_prompt: str
    facts: dict[str, Any]
    expected_move_uci: str | None
    expected_move_san: str | None
    primary_motif_id: str
    prompt_version: str = PROMPT_VERSION


@dataclass(frozen=True)
class ValidatedExplanation:
    """A parsed and validated provider response."""

    text: str
    response_json: dict[str, Any]


def build_prompt(request: ExplanationRequest) -> BuiltPrompt:
    """Build the grounded prompt for one flagged move."""

    primary = primary_motif(request.motifs)
    line_source: Literal["before", "after"] = (
        "after" if primary.id in TACTIC_AFTER_MOTIFS else "before"
    )
    line_analysis = request.analysis_after if line_source == "after" else request.analysis_before
    expected_move = line_analysis.best_move
    facts = {
        "prompt_version": PROMPT_VERSION,
        "task": "Explain exactly one chess mistake.",
        "move": {
            "ply": request.ply,
            "move_number": request.move_number,
            "side": request.side,
            "san": request.san,
            "uci": request.uci,
        },
        "position": {
            "fen_before": request.fen_before,
            "fen_after": request.fen_after,
        },
        "engine": {
            "line_source": line_source,
            "ground_truth_best_move": _move_payload(expected_move),
            "ground_truth_pv": [_move_payload(move) for move in line_analysis.pv],
            "before": _analysis_payload(request.analysis_before),
            "after": _analysis_payload(request.analysis_after),
            "loss_cp": request.loss_cp,
        },
        "motifs": [_motif_payload(motif) for motif in request.motifs],
        "primary_motif_id": primary.id,
        "game_phase": primary.evidence.phase,
        "output_contract": {
            "format": "strict JSON",
            "schema": {"text": "string", "referenced_move_uci": "string|null"},
            "max_sentences": 3,
            "max_words": 70,
        },
    }
    user_prompt = (
        "Use these engine-grounded facts to write one short coaching explanation.\n"
        "Do not mention any move as best unless it is the ground_truth_best_move or in "
        "ground_truth_pv.\n\n"
        f"{json.dumps(facts, sort_keys=True, separators=(',', ':'))}"
    )
    return BuiltPrompt(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        facts=facts,
        expected_move_uci=expected_move.uci if expected_move is not None else None,
        expected_move_san=expected_move.san if expected_move is not None else None,
        primary_motif_id=primary.id,
    )


def primary_motif(motifs: tuple[Motif, ...]) -> Motif:
    """Select one deterministic teaching target from one or more motifs."""

    if not motifs:
        raise ValueError("Cannot build an explanation prompt without motifs.")
    return max(
        motifs,
        key=lambda motif: (
            SEVERITY_RANK[motif.severity],
            MOTIF_RANK[motif.id],
            motif.id,
        ),
    )


def validate_provider_response(raw_text: str, prompt: BuiltPrompt) -> ValidatedExplanation:
    """Parse and validate strict JSON from the provider."""

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        parsed = _extract_json_object(raw_text)
        if parsed is None:
            raise InvalidExplanationResponseError("Explanation response was not JSON.") from exc

    if not isinstance(parsed, dict):
        raise InvalidExplanationResponseError("Explanation response must be a JSON object.")

    text = parsed.get("text")
    referenced_move = parsed.get("referenced_move_uci")
    if not isinstance(text, str) or not text.strip():
        raise InvalidExplanationResponseError("Explanation text must be a non-empty string.")
    if referenced_move is not None and not isinstance(referenced_move, str):
        raise InvalidExplanationResponseError("referenced_move_uci must be a string or null.")
    if not _references_engine_move(text, referenced_move, prompt):
        raise InvalidExplanationResponseError("Explanation referenced a non-engine move.")
    text = _trim_to_sentence_limit(text, max_sentences=3)
    text = _trim_to_word_limit(text, max_words=80)

    return ValidatedExplanation(text=text.strip(), response_json=parsed)


def _analysis_payload(analysis: EngineEvaluation) -> dict[str, Any]:
    return {
        "status": analysis.status,
        "depth": analysis.depth,
        "score": _score_payload(analysis.score),
        "best_move": _move_payload(analysis.best_move),
        "pv": [_move_payload(move) for move in analysis.pv],
        "nodes": analysis.nodes,
    }


def _score_payload(score: CentipawnScore | MateScore) -> dict[str, Any]:
    if isinstance(score, CentipawnScore):
        return {"type": "cp", "cp": score.cp}
    return {"type": "mate", "mate_in": score.mate_in, "winner": score.winner}


def _move_payload(move: EngineMove | None) -> dict[str, str] | None:
    if move is None:
        return None
    return {"uci": move.uci, "san": move.san}


def _motif_payload(motif: Motif) -> dict[str, Any]:
    evidence = motif.evidence
    piece = evidence.piece
    return {
        "id": motif.id,
        "label": motif.label,
        "severity": motif.severity,
        "source": motif.source,
        "score_cp": motif.score_cp,
        "evidence": {
            "threshold_cp": evidence.threshold_cp,
            "score_kind": evidence.score_kind,
            "phase": evidence.phase,
            "piece": (
                {"color": piece.color, "role": piece.role, "square": piece.square}
                if piece is not None
                else None
            ),
            "attackers": list(evidence.attackers),
            "defenders": list(evidence.defenders),
            "best_move": _move_ref_payload(evidence.best_move),
            "opponent_reply": _move_ref_payload(evidence.opponent_reply),
            "related_ply": evidence.related_ply,
        },
    }


def _move_ref_payload(move: MoveRef | None) -> dict[str, str] | None:
    if move is None:
        return None
    return {"uci": move.uci, "san": move.san}


def _sentence_count(text: str) -> int:
    fragments = [fragment for fragment in re.split(r"[.!?]+", text) if fragment.strip()]
    return len(fragments)


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw_text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _references_engine_move(
    text: str,
    referenced_move: str | None,
    prompt: BuiltPrompt,
) -> bool:
    expected_refs = {
        ref
        for ref in (prompt.expected_move_uci, prompt.expected_move_san)
        if ref is not None and ref.strip()
    }
    if not expected_refs:
        return referenced_move is None
    if referenced_move is None:
        return any(ref in text for ref in expected_refs)
    if referenced_move in expected_refs:
        return True
    return any(ref in text for ref in expected_refs)


def _trim_to_sentence_limit(text: str, *, max_sentences: int) -> str:
    matches = list(re.finditer(r"[^.!?]+[.!?]*", text))
    sentences = [match.group(0).strip() for match in matches if match.group(0).strip()]
    if len(sentences) <= max_sentences:
        return text.strip()
    return " ".join(sentences[:max_sentences]).strip()


def _trim_to_word_limit(text: str, *, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).rstrip(".,;:") + "."
