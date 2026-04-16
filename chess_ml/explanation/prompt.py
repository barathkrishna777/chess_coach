"""Prompt assembly and response validation for grounded coaching explanations."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import chess

from chess_ml.classifier.motifs import Motif, MoveRef
from chess_ml.engine.stockfish import CentipawnScore, EngineEvaluation, EngineMove, MateScore
from chess_ml.explanation.models import PROMPT_VERSION, ExplanationRequest

SYSTEM_PROMPT = """You are a chess coach explaining one mistake to a 1200-2000 rated club player.
Stockfish is ground truth: never contradict the provided engine best move or main line.
Use only the supplied FEN, played move, actual game line, Stockfish line, eval swing, and motif evidence; do not invent tactics.
Compare what the player did with what Stockfish recommended, then explain what changed.
Reference concrete pieces, squares, or moves from those two supplied lines.
Teach exactly one practical lesson; only discuss deeper positional ideas when the concrete lines do not already explain the mistake.
Return strict JSON only: {"text":"...","referenced_move_uci":"..."}. The text must be at most 3 sentences and about 70 words or fewer."""

SEVERITY_RANK: Mapping[str, int] = {"blunder": 3, "mistake": 2, "inaccuracy": 1}
MOTIF_RANK: Mapping[str, int] = {
    "allowed_tactic": 9,
    "missed_tactic": 8,
    "endgame_slip": 7,
    "fork": 6,
    "pin": 5,
    "discovered_attack": 4,
    "overloaded_defender": 3,
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
    line_source = _line_source(primary, request)
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
        "actual_line": [
            {
                "ply": move.ply,
                "side": move.side,
                "san": move.san,
                "uci": move.uci,
            }
            for move in request.actual_line
        ],
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
            "required_content": (
                "Compare the played move or actual line with Stockfish's best move or main line."
            ),
        },
    }
    user_prompt = (
        "Use these engine-grounded facts to write one short coaching explanation.\n"
        "Your job is comparison, not discovery: contrast move.san and actual_line with "
        "engine.ground_truth_best_move and engine.ground_truth_pv.\n"
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


def _line_source(
    primary: Motif,
    request: ExplanationRequest,
) -> Literal["before", "after"]:
    if primary.id in TACTIC_AFTER_MOTIFS:
        return "after"
    evidence_move = primary.evidence.best_move
    if (
        evidence_move is not None
        and request.analysis_after.best_move is not None
        and evidence_move.uci == request.analysis_after.best_move.uci
    ):
        return "after"
    return "before"


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
    if not _references_engine_line(text, referenced_move, prompt):
        raise InvalidExplanationResponseError("Explanation referenced a non-engine move.")
    if _claims_different_engine_move(text, prompt):
        raise InvalidExplanationResponseError("Explanation claimed a different engine move.")
    text = _trim_to_sentence_limit(text, max_sentences=3)
    text = _trim_to_word_limit(text, max_words=80)

    return ValidatedExplanation(text=text.strip(), response_json=parsed)


def build_fallback_explanation(request: ExplanationRequest, prompt: BuiltPrompt) -> str:
    """Build a deterministic explanation from supplied Stockfish and motif facts only."""

    primary = primary_motif(request.motifs)
    best_move = _ground_truth_best_move(prompt)
    pv = _ground_truth_pv(prompt)
    line_source = prompt.facts["engine"]["line_source"]
    best_label = _move_label(best_move) if best_move is not None else None
    pv_label = _line_label(pv)
    sentences = [_fallback_intro(request, primary, best_label, line_source)]

    proof = _fallback_proof_sentence(pv_label, request.loss_cp)
    if proof:
        sentences.append(proof)
    sentences.append(_fallback_lesson(primary))

    return " ".join(sentences)


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


def _references_engine_line(
    text: str,
    referenced_move: str | None,
    prompt: BuiltPrompt,
) -> bool:
    engine_refs = _engine_line_refs(prompt)
    if not engine_refs:
        return referenced_move is None
    if referenced_move is None:
        return _text_contains_any_ref(text, engine_refs)
    if referenced_move != prompt.expected_move_uci:
        return False
    return _text_contains_any_ref(text, engine_refs)


def _claims_different_engine_move(text: str, prompt: BuiltPrompt) -> bool:
    allowed_refs = _engine_line_refs(prompt)
    if not allowed_refs:
        return False

    legal_refs = _legal_move_refs(prompt)
    disallowed_refs = legal_refs - allowed_refs
    if not disallowed_refs:
        return False

    claim_patterns = (
        r"\b(?:stockfish|engine|best|recommended|recommendation|wanted|better)\b",
        r"\bshould\s+have\s+played\b",
        r"\bwas\s+best\b",
    )
    sentences = [fragment for fragment in re.split(r"(?<=[.!?])\s+", text) if fragment.strip()]
    for sentence in sentences:
        lowered = sentence.lower()
        if not any(re.search(pattern, lowered) for pattern in claim_patterns):
            continue
        if _text_contains_any_ref(sentence, disallowed_refs):
            return True
    return False


def _engine_line_refs(prompt: BuiltPrompt) -> set[str]:
    refs: set[str] = set()
    best_move = _ground_truth_best_move(prompt)
    if best_move is not None:
        refs.update(_refs_for_move(best_move))
    for move in _ground_truth_pv(prompt):
        refs.update(_refs_for_move(move))
    return refs


def _legal_move_refs(prompt: BuiltPrompt) -> set[str]:
    facts = prompt.facts
    position = facts["position"]
    engine = facts["engine"]
    fen = position["fen_after"] if engine["line_source"] == "after" else position["fen_before"]
    if not isinstance(fen, str):
        return set()

    try:
        board = chess.Board(fen)
    except ValueError:
        return set()

    refs: set[str] = set()
    for move in board.legal_moves:
        refs.add(move.uci())
        refs.add(board.san(move))
    return refs


def _ground_truth_best_move(prompt: BuiltPrompt) -> dict[str, str] | None:
    move = prompt.facts["engine"]["ground_truth_best_move"]
    if not _is_move_payload(move):
        return None
    return cast(dict[str, str], move)


def _ground_truth_pv(prompt: BuiltPrompt) -> list[dict[str, str]]:
    pv = prompt.facts["engine"]["ground_truth_pv"]
    if not isinstance(pv, list):
        return []
    return [move for move in pv if _is_move_payload(move)]


def _is_move_payload(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return isinstance(value.get("uci"), str) and isinstance(value.get("san"), str)


def _refs_for_move(move: dict[str, str]) -> set[str]:
    return {move["uci"], move["san"]}


def _text_contains_any_ref(text: str, refs: set[str]) -> bool:
    return any(_text_contains_ref(text, ref) for ref in refs if ref.strip())


def _text_contains_ref(text: str, ref: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(ref)}(?![A-Za-z0-9])", text) is not None


def _move_label(move: dict[str, str]) -> str:
    return f"{move['san']} ({move['uci']})"


def _line_label(pv: list[dict[str, str]]) -> str:
    return " ".join(_move_label(move) for move in pv[:4])


def _fallback_intro(
    request: ExplanationRequest,
    motif: Motif,
    best_label: str | None,
    line_source: object,
) -> str:
    move_label = f"{request.san} ({request.uci})"
    motif_label = motif.label.lower()
    phase = _phase_label(motif.evidence.phase)
    article = _article_for(phase)
    if best_label is None:
        return f"{move_label} is tagged as {article} {phase} {motif_label}."
    if line_source == "after":
        return (
            f"After {move_label}, Stockfish says the key reply is {best_label}, "
            f"so this is {article} {phase} {motif_label}."
        )
    return (
        f"Instead of {move_label}, Stockfish wanted {best_label}, "
        f"which is why this is tagged as {article} {phase} {motif_label}."
    )


def _fallback_proof_sentence(pv_label: str, loss_cp: int | None) -> str | None:
    loss = _loss_label(loss_cp)
    if pv_label and loss:
        return f"The supplied main line starts {pv_label}, and the recorded loss is {loss}."
    if pv_label:
        return f"The supplied main line starts {pv_label}."
    if loss:
        return f"The recorded loss is {loss}."
    return None


def _fallback_lesson(motif: Motif) -> str:
    evidence = motif.evidence
    if motif.id == "allowed_tactic":
        reply = (
            f" like {evidence.opponent_reply.san} ({evidence.opponent_reply.uci})"
            if evidence.opponent_reply is not None
            else ""
        )
        return f"Lesson: before making the move, check the opponent's best reply{reply}."
    if motif.id in {"missed_tactic", "endgame_slip"}:
        if evidence.best_move is not None:
            best = f"{evidence.best_move.san} ({evidence.best_move.uci})"
            return f"Lesson: compare your candidate with Stockfish's concrete move {best}."
        return "Lesson: compare your candidate with the concrete Stockfish line."
    if motif.id == "hanging_piece" and evidence.piece is not None:
        piece = evidence.piece
        return f"Lesson: count attacks and defenders on the {piece.role} on {piece.square}."
    if motif.id == "pin" and evidence.piece is not None:
        piece = evidence.piece
        return f"Lesson: before moving, check whether the {piece.role} on {piece.square} is pinned."
    if motif.id == "fork" and evidence.piece is not None:
        piece = evidence.piece
        return f"Lesson: look for one {piece.role} attacking two valuable targets."
    if motif.id == "overloaded_defender" and evidence.piece is not None:
        piece = evidence.piece
        return (
            f"Lesson: notice when the {piece.role} on {piece.square} has too many defensive jobs."
        )
    if motif.id == "discovered_attack" and evidence.piece is not None:
        piece = evidence.piece
        return f"Lesson: check whether moving a blocker opens the {piece.role} on {piece.square}."
    if motif.id == "opening_inaccuracy":
        return "Lesson: save this engine line as the concrete opening improvement to review."
    return "Lesson: use the Stockfish line as the concrete correction for this position."


def _phase_label(phase: str) -> str:
    if phase == "opening":
        return "opening"
    if phase == "middlegame":
        return "middlegame"
    return "endgame"


def _article_for(text: str) -> str:
    return "an" if text[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def _loss_label(loss_cp: int | None) -> str | None:
    if loss_cp is None:
        return None
    if loss_cp == 0:
        return "no recorded centipawn loss"
    return f"{loss_cp / 100:.2f} pawns"


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
