"""Tests for Slice 3 grounded explanation layer."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Literal

import chess
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from chess_ml.api.games import _annotate_move
from chess_ml.api.games import router as games_router
from chess_ml.classifier.classify import classify_moves
from chess_ml.classifier.config import LABEL_ORDER
from chess_ml.classifier.motifs import (
    AnalyzedMove,
    Motif,
    MotifEvidence,
    MotifId,
    MoveRef,
    PieceRef,
)
from chess_ml.engine.stockfish import CentipawnScore, EngineEvaluation, EngineMove, MateScore
from chess_ml.explanation.cache import ExplanationCache, cache_key_for_facts
from chess_ml.explanation.client import (
    DEFAULT_TIMEOUT_SECONDS,
    ClientResponse,
    LocalProviderUnavailableError,
    client_from_env,
    select_client_from_env,
)
from chess_ml.explanation.models import (
    ExplanationProvider,
    ExplanationRequest,
    LineMove,
    MoveExplanation,
)
from chess_ml.explanation.prompt import (
    SYSTEM_PROMPT,
    InvalidExplanationResponseError,
    build_fallback_explanation,
    build_prompt,
    validate_provider_response,
)
from chess_ml.explanation.service import ExplanationService
from chess_ml.ingestion.pgn import ParsedPgnGame, ParsedPgnMove, parse_pgn


def test_prompt_uses_before_line_for_missed_tactic() -> None:
    request = _request_for_ply(
        """
[Event "Fixture"]
[Result "*"]

1. e4 d5 2. exd5 Qxd5 3. Nc3 a6 4. Nf3 *
""",
        7,
        {
            7: _spec(before_cp=740, after_cp=85, best_before="c3d5"),
        },
    )

    prompt = build_prompt(request)

    assert prompt.primary_motif_id == "missed_tactic"
    assert prompt.expected_move_uci == "c3d5"
    assert prompt.facts["engine"]["line_source"] == "before"
    assert prompt.facts["position"]["fen_before"] == request.fen_before
    assert prompt.facts["actual_line"][0] == {
        "ply": 7,
        "side": "white",
        "san": "Nf3",
        "uci": "g1f3",
    }
    assert "c3d5" in prompt.user_prompt
    assert "Nxd5" in prompt.user_prompt


def test_prompt_uses_after_line_for_allowed_tactic() -> None:
    request = _request_for_ply(
        """
[Event "Fixture"]
[Result "*"]

1. e4 d5 2. exd5 Qxd5 3. Nc3 a6 4. Nxd5 *
""",
        6,
        {
            6: _spec(
                before_cp=80,
                after_cp=702,
                best_before="d5d8",
                best_after="c3d5",
            ),
        },
    )

    prompt = build_prompt(request)

    assert prompt.primary_motif_id == "allowed_tactic"
    assert prompt.expected_move_uci == "c3d5"
    assert prompt.facts["engine"]["line_source"] == "after"
    assert prompt.facts["engine"]["best_move_side"] == "white"
    assert prompt.facts["engine"]["best_move_role"] == "opponent_reply_after_player_move"
    assert prompt.facts["engine"]["ground_truth_best_move"] == {"uci": "c3d5", "san": "Nxd5"}
    assert prompt.facts["engine"]["ground_truth_main_line"][0] == {
        "move_number": 4,
        "side": "white",
        "san": "Nxd5",
        "uci": "c3d5",
    }
    assert "opponent's reply after the player's move" in prompt.user_prompt


@pytest.mark.parametrize("motif_id", LABEL_ORDER)
def test_prompt_includes_motif_specific_teaching_guidance(motif_id: MotifId) -> None:
    prompt = build_prompt(_request_with_motif(motif_id))
    guidance = prompt.facts["teaching_guidance"]
    output_contract = prompt.facts["output_contract"]

    assert guidance["motif_id"] == motif_id
    assert isinstance(guidance["pattern"], str)
    assert guidance["pattern"]
    assert isinstance(guidance["focus"], str)
    assert guidance["focus"]
    assert isinstance(guidance["evidence_fields"], list)
    assert guidance["evidence_fields"]
    assert output_contract["max_words"] == 95
    assert output_contract["user_facing_line_name"] == "main line"
    assert "PV" in output_contract["disallowed_user_facing_terms"]


def test_system_prompt_contains_grounding_rules_without_image_language() -> None:
    prompt = build_prompt(_missed_tactic_request())

    assert "Stockfish is ground truth" in SYSTEM_PROMPT
    assert "never contradict" in SYSTEM_PROMPT
    assert "Compare what the player did with what Stockfish recommended" in SYSTEM_PROMPT
    assert "strict JSON" in SYSTEM_PROMPT
    assert "at most 3 sentences" in SYSTEM_PROMPT
    assert "about 95 words" in SYSTEM_PROMPT
    lowered = f"{SYSTEM_PROMPT}\n{prompt.user_prompt}".lower()
    assert "image" not in lowered
    assert "ocr" not in lowered
    assert "screenshot" not in lowered
    assert "computer vision" not in lowered


def test_cache_key_is_deterministic_and_tracks_engine_line() -> None:
    request = _missed_tactic_request()
    facts = build_prompt(request).facts
    copied_facts = json.loads(json.dumps(facts))

    assert cache_key_for_facts(facts) == cache_key_for_facts(copied_facts)

    changed_facts = copy.deepcopy(facts)
    changed_facts["engine"]["ground_truth_pv"] = [{"uci": "g1f3", "san": "Nf3"}]

    assert cache_key_for_facts(facts) != cache_key_for_facts(changed_facts)

    changed_guidance = copy.deepcopy(facts)
    changed_guidance["teaching_guidance"]["focus"] = "Different teaching focus."

    assert cache_key_for_facts(facts) != cache_key_for_facts(changed_guidance)


def test_validation_accepts_wrapped_json_and_san_reference() -> None:
    prompt = build_prompt(_missed_tactic_request())

    validated = validate_provider_response(
        '```json\n{"text":"Stockfish wanted Nxd5 from c3, improving the supplied engine line. Check forcing captures before developing.","referenced_move_uci":"c3d5"}\n```',
        prompt,
    )

    assert validated.text.startswith("Stockfish wanted Nxd5")


def test_validation_trims_long_local_model_answer() -> None:
    prompt = build_prompt(_missed_tactic_request())

    validated = validate_provider_response(
        '{"text":"Stockfish wanted Nxd5 from c3, winning the queen on d5. This is the forcing capture. Before a quiet move, check captures. Extra sentence should not survive.","referenced_move_uci":"c3d5"}',
        prompt,
    )

    assert validated.text.count(".") == 3
    assert "Extra sentence" not in validated.text


def test_validation_rejects_wrong_referenced_move_uci() -> None:
    prompt = build_prompt(_missed_tactic_request())

    with pytest.raises(InvalidExplanationResponseError):
        validate_provider_response(
            '{"text":"Stockfish wanted Nxd5 from c3, improving the supplied engine line.","referenced_move_uci":"a2a3"}',
            prompt,
        )


def test_validation_rejects_invented_best_move_in_text() -> None:
    prompt = build_prompt(_missed_tactic_request())

    with pytest.raises(InvalidExplanationResponseError):
        validate_provider_response(
            '{"text":"Stockfish recommended a3 as the best move, though Nxd5 was also in the line.","referenced_move_uci":"c3d5"}',
            prompt,
        )


def test_validation_rejects_generic_advice_without_engine_line() -> None:
    prompt = build_prompt(_missed_tactic_request())

    with pytest.raises(InvalidExplanationResponseError):
        validate_provider_response(
            '{"text":"You should look for forcing moves before playing quiet development.","referenced_move_uci":"c3d5"}',
            prompt,
        )


def test_validation_rejects_user_facing_pv_wording() -> None:
    prompt = build_prompt(_missed_tactic_request())

    with pytest.raises(InvalidExplanationResponseError):
        validate_provider_response(
            '{"text":"Stockfish wanted Nxd5 from c3 in the PV. Check forcing captures first.","referenced_move_uci":"c3d5"}',
            prompt,
        )


def test_validation_rejects_opponent_reply_described_as_users_move() -> None:
    prompt = build_prompt(_allowed_tactic_request())

    with pytest.raises(InvalidExplanationResponseError):
        validate_provider_response(
            '{"text":"Your best move was Nxd5, so you should have played it instead. Check forcing moves first.","referenced_move_uci":"c3d5"}',
            prompt,
        )


def test_validation_accepts_opponent_reply_described_as_opponents_move() -> None:
    prompt = build_prompt(_allowed_tactic_request())

    validated = validate_provider_response(
        '{"text":"After a6, White had the reply Nxd5 in the supplied main line. Before playing a quiet move, check the opponent\\u0027s forcing reply.","referenced_move_uci":"c3d5"}',
        prompt,
    )

    assert "White had the reply Nxd5" in validated.text


def test_missing_api_key_returns_unavailable(tmp_path: Path) -> None:
    service = ExplanationService(cache=ExplanationCache(tmp_path / "cache.sqlite3"), client=None)

    explanation = asyncio.run(service.explain(_missed_tactic_request()))

    assert explanation is not None
    assert explanation.status == "unavailable"
    assert explanation.reason == "api_key_missing"
    assert explanation.text is None


def test_auto_provider_defaults_to_ollama(monkeypatch) -> None:
    monkeypatch.delenv("CHESS_ML_EXPLANATION_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = client_from_env()

    assert client is not None
    assert client.provider == "ollama"


def test_local_provider_unavailable_is_non_fatal(tmp_path: Path) -> None:
    service = ExplanationService(
        cache=ExplanationCache(tmp_path / "cache.sqlite3"),
        client=_UnavailableLocalClient(),
    )

    explanation = asyncio.run(service.explain(_missed_tactic_request()))

    assert explanation is not None
    assert explanation.status == "unavailable"
    assert explanation.provider == "ollama"
    assert explanation.reason == "local_model_unavailable"
    assert explanation.retryable is True


def test_service_level_timeout_is_non_fatal_and_not_cached(tmp_path: Path) -> None:
    request = _missed_tactic_request()
    cache = ExplanationCache(tmp_path / "cache.sqlite3")
    service = ExplanationService(
        cache=cache,
        client=_SlowClient(),
        timeout_seconds=0.01,
    )

    explanation = asyncio.run(service.explain(request))

    assert explanation is not None
    assert explanation.status == "error"
    assert explanation.reason == "timeout"
    assert explanation.retryable is True
    assert explanation.timeout_seconds == 0.01
    assert cache.get(cache_key_for_facts(build_prompt(request).facts)) is None


def test_cache_hit_skips_client_call(tmp_path: Path) -> None:
    request = _missed_tactic_request()
    client = _FakeClient(
        '{"text":"Stockfish wanted Nxd5 from c3, winning the queen on d5. Before a quiet developing move, check forcing captures first.","referenced_move_uci":"c3d5"}'
    )
    service = ExplanationService(
        cache=ExplanationCache(tmp_path / "cache.sqlite3"),
        client=client,
    )

    first = asyncio.run(service.explain(request))
    second = asyncio.run(service.explain(request))

    assert first is not None
    assert first.status == "ok"
    assert first.source == "llm"
    assert second is not None
    assert second.status == "ok"
    assert second.source == "cache"
    assert second.retryable is False
    assert client.calls == 1


def test_invalid_provider_response_returns_uncached_fallback(tmp_path: Path) -> None:
    request = _missed_tactic_request()
    client = _FakeClient(
        '{"text":"Play a3 instead. That is the lesson.","referenced_move_uci":"a2a3"}'
    )
    cache = ExplanationCache(tmp_path / "cache.sqlite3")
    service = ExplanationService(cache=cache, client=client)

    explanation = asyncio.run(service.explain(request))

    assert explanation is not None
    assert explanation.status == "ok"
    assert explanation.source == "fallback"
    assert explanation.text is not None
    assert "Play a3 instead" not in explanation.text
    assert "Nxd5" in explanation.text
    assert "c3d5" in explanation.text
    assert explanation.reason == "invalid_response"
    assert explanation.retryable is False
    assert cache.get(cache_key_for_facts(build_prompt(request).facts)) is None


def test_fallback_explanation_uses_only_supplied_stockfish_and_motif_facts() -> None:
    request = _missed_tactic_request()
    prompt = build_prompt(request)

    text = build_fallback_explanation(request, prompt)

    assert request.san in text
    assert request.uci in text
    assert "Nxd5" in text
    assert "c3d5" in text
    assert "missed tactic" in text.lower()
    assert "recorded loss is 6.55 pawns" in text
    assert "Lesson:" in text
    assert "Grounding:" not in text
    assert "queen" not in text.lower()
    assert "fork" not in text.lower()
    assert "a3" not in text


def test_allowed_tactic_fallback_explains_the_actionable_check() -> None:
    request = _request_for_ply(
        """
[Event "Fixture"]
[Result "*"]

1. e4 d5 2. exd5 Qxd5 3. Nc3 a6 4. Nxd5 *
""",
        6,
        {
            6: _spec(
                before_cp=80,
                after_cp=702,
                best_before="d5d8",
                best_after="c3d5",
            ),
        },
    )
    prompt = build_prompt(request)

    text = build_fallback_explanation(request, prompt)

    assert text.startswith("After a6 (a7a6), Stockfish says the key reply is Nxd5 (c3d5)")
    assert "an opening allowed tactic" in text
    assert "recorded loss is 6.22 pawns" in text
    assert "opponent's best reply Nxd5 (c3d5)" in text
    assert "ply 7" in text
    assert "Grounding:" not in text


@pytest.mark.parametrize(
    ("motif_id", "expected_fragments"),
    [
        ("allowed_tactic", ("allowed tactic", "e5 (e7e5)", "ply 2")),
        ("endgame_slip", ("endgame slip", "in the endgame", "e4 (e2e4)")),
        ("pin", ("pin", "knight on f3", "attackers black bishop on g4")),
        ("fork", ("fork", "knight on f3", "multiple targets")),
        ("overloaded_defender", ("overloaded defender", "knight on f3", "defensive jobs")),
        ("discovered_attack", ("discovered attack", "knight on f3", "opened lines")),
        ("hanging_piece", ("hanging piece", "attackers black bishop on g4", "defenders")),
        ("missed_tactic", ("missed tactic", "in the middlegame", "e4 (e2e4)")),
        ("opening_inaccuracy", ("opening inaccuracy", "opening improvement", "e4 (e2e4)")),
    ],
)
def test_fallback_explanation_teaches_each_motif(
    motif_id: MotifId,
    expected_fragments: tuple[str, ...],
) -> None:
    request = _request_with_motif(motif_id)
    prompt = build_prompt(request)

    text = build_fallback_explanation(request, prompt)

    assert "Lesson:" in text
    assert "main line" in text
    assert "PV" not in text
    assert "principal variation" not in text.lower()
    for fragment in expected_fragments:
        assert fragment in text


def test_explanation_status_endpoint_does_not_contact_provider(tmp_path: Path) -> None:
    client = _FakeClient('{"text":"Stockfish wanted Nxd5 from c3.","referenced_move_uci":"c3d5"}')
    service = ExplanationService(
        cache=ExplanationCache(tmp_path / "cache.sqlite3"),
        client=client,
        timeout_seconds=7.5,
    )
    app = FastAPI()
    app.state.explanation_service = service
    app.include_router(games_router)
    api = TestClient(app)

    response = api.get("/api/games/explain/status")

    assert response.status_code == 200
    assert response.json() == {
        "schema_version": "explanation-status.v1",
        "enabled": True,
        "configured": True,
        "provider": "anthropic",
        "model": "fake-explainer",
        "timeout_seconds": 7.5,
        "availability": "not_checked",
        "reason": None,
    }
    assert client.calls == 0


def test_default_explanation_timeout_allows_local_ollama_cold_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHESS_ML_EXPLANATION_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("CHESS_ML_EXPLANATION_PROVIDER", "ollama")

    selection = select_client_from_env()

    assert DEFAULT_TIMEOUT_SECONDS == 45.0
    assert selection.timeout_seconds == 45.0


def test_api_move_model_includes_nullable_explanation() -> None:
    parsed = parse_pgn(
        """
[Event "Fixture"]
[Result "*"]

1. e4 *
"""
    )
    move = parsed.moves[0]
    before = _evaluation(move.fen_before, 0, best_uci="e2e4")
    after = _evaluation(move.fen_after, 0, best_uci="e7e5")

    unflagged = _annotate_move(move, before, after, (), None)
    flagged = _annotate_move(
        move,
        before,
        after,
        (),
        MoveExplanation(
            status="unavailable",
            text=None,
            source=None,
            provider=None,
            model=None,
            reason="api_key_missing",
        ),
    )

    assert unflagged.model_dump()["explanation"] is None
    assert flagged.model_dump()["explanation"]["schema_version"] == "move-explanation.v1"
    assert flagged.model_dump()["explanation"]["status"] == "unavailable"


def _missed_tactic_request() -> ExplanationRequest:
    return _request_for_ply(
        """
[Event "Fixture"]
[Result "*"]

1. e4 d5 2. exd5 Qxd5 3. Nc3 a6 4. Nf3 *
""",
        7,
        {
            7: _spec(before_cp=740, after_cp=85, best_before="c3d5"),
        },
    )


def _allowed_tactic_request() -> ExplanationRequest:
    return _request_for_ply(
        """
[Event "Fixture"]
[Result "*"]

1. e4 d5 2. exd5 Qxd5 3. Nc3 a6 4. Nxd5 *
""",
        6,
        {
            6: _spec(
                before_cp=80,
                after_cp=702,
                best_before="d5d8",
                best_after="c3d5",
            ),
        },
    )


def _request_with_motif(motif_id: MotifId) -> ExplanationRequest:
    parsed = parse_pgn(
        """
[Event "Fixture"]
[Result "*"]

1. d4 *
"""
    )
    move = parsed.moves[0]
    before = _evaluation(move.fen_before, 0, best_uci="e2e4")
    after = _evaluation(move.fen_after, -320, best_uci="e7e5")
    return ExplanationRequest(
        ply=move.ply,
        move_number=move.move_number,
        side=move.side,
        san=move.san,
        uci=move.uci,
        fen_before=move.fen_before,
        fen_after=move.fen_after,
        analysis_before=before,
        analysis_after=after,
        loss_cp=320,
        actual_line=_actual_line(parsed, move.ply),
        motifs=(_motif(motif_id),),
    )


def _motif(motif_id: MotifId) -> Motif:
    phase = "endgame" if motif_id == "endgame_slip" else "opening"
    if motif_id in {"missed_tactic", "pin", "fork", "overloaded_defender", "discovered_attack"}:
        phase = "middlegame"
    best_move = (
        MoveRef(uci="e7e5", san="e5")
        if motif_id == "allowed_tactic"
        else MoveRef(
            uci="e2e4",
            san="e4",
        )
    )
    return Motif(
        id=motif_id,
        label=motif_id.replace("_", " ").title(),
        severity="inaccuracy" if motif_id == "opening_inaccuracy" else "mistake",
        source="heuristic",
        score_cp=320,
        evidence=MotifEvidence(
            threshold_cp=200,
            score_kind="cp",
            phase=phase,
            piece=PieceRef(color="white", role="knight", square="f3"),
            attackers=("black bishop on g4", "black queen on d6"),
            defenders=("white queen on d1",),
            best_move=best_move,
            opponent_reply=MoveRef(uci="e7e5", san="e5") if motif_id == "allowed_tactic" else None,
            related_ply=2 if motif_id == "allowed_tactic" else None,
        ),
    )


def _request_for_ply(
    pgn: str,
    ply: int,
    specs: dict[int, _MoveSpec],
) -> ExplanationRequest:
    parsed = parse_pgn(pgn)
    analyzed_moves = _analyzed_moves(parsed, specs)
    motif_lists = classify_moves(analyzed_moves, initial_fen=parsed.initial_fen)
    move = parsed.moves[ply - 1]
    analyzed = analyzed_moves[ply - 1]
    motifs = motif_lists[ply - 1]
    assert motifs
    return ExplanationRequest(
        ply=move.ply,
        move_number=move.move_number,
        side=move.side,
        san=move.san,
        uci=move.uci,
        fen_before=move.fen_before,
        fen_after=move.fen_after,
        analysis_before=analyzed.analysis_before,
        analysis_after=analyzed.analysis_after,
        loss_cp=_loss_cp(move, analyzed.analysis_before, analyzed.analysis_after),
        actual_line=_actual_line(parsed, ply),
        motifs=motifs,
    )


def _actual_line(parsed: ParsedPgnGame, ply: int) -> tuple[LineMove, ...]:
    return tuple(
        LineMove(
            ply=move.ply,
            side=move.side,
            san=move.san,
            uci=move.uci,
        )
        for move in parsed.moves[ply - 1 : ply + 5]
    )


def _analyzed_moves(
    parsed: ParsedPgnGame,
    specs: dict[int, _MoveSpec],
) -> list[AnalyzedMove]:
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
    return moves


ScoreSpec = int | tuple[Literal["white", "black"], int]


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


def _loss_cp(
    move: ParsedPgnMove,
    before: EngineEvaluation,
    after: EngineEvaluation,
) -> int | None:
    if not isinstance(before.score, CentipawnScore) or not isinstance(after.score, CentipawnScore):
        return None
    delta = after.score.cp - before.score.cp
    if move.side == "white":
        return max(0, -delta)
    return max(0, delta)


class _FakeClient:
    provider: ExplanationProvider = "anthropic"
    model = "fake-explainer"

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    async def complete(self, prompt: object) -> ClientResponse:
        self.calls += 1
        return ClientResponse(
            content=self.content,
            response_json={"raw": self.content},
            provider=self.provider,
            model=self.model,
        )


class _UnavailableLocalClient:
    provider: ExplanationProvider = "ollama"
    model = "fake-local"

    async def complete(self, prompt: object) -> ClientResponse:
        raise LocalProviderUnavailableError("Ollama is not reachable.")


class _SlowClient:
    provider: ExplanationProvider = "ollama"
    model = "slow-local"

    async def complete(self, prompt: object) -> ClientResponse:
        await asyncio.sleep(1)
        return ClientResponse(
            content='{"text":"Stockfish wanted Nxd5 from c3.","referenced_move_uci":"c3d5"}',
            response_json={},
            provider=self.provider,
            model=self.model,
        )
