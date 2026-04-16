# Plan 004 — Slice 3: Grounded Explanation Layer

Status: approved for implementation
Owner: barathkrishna
Last updated: 2026-04-15

## Summary

Slice 3 adds one short, engine-grounded coaching explanation for each flagged move
returned by `POST /api/games`, without changing the core review flow or adding image
inputs. The implementation uses a provider-adapter design: prompt assembly, cache keys,
validation, API schema, and tests are provider-independent; the concrete LLM client is
selected by environment and can be replaced without changing the review contract.

Product decision after initial implementation: explanations are local/open-source first
for now, using Ollama by default. Hosted API providers remain opt-in adapters for a later
quality/cost pass.

## API And Schema

Keep `schema_version: "annotated-game.v1"` and add an always-present nullable
`explanation` field to each move. The upload path returns `null` for every move so PGN
review is not blocked by local LLM latency; selected flagged moves are explained through
an on-demand endpoint.

```ts
type MoveExplanation = {
  schema_version: "move-explanation.v1";
  status: "ok" | "unavailable" | "error";
  text: string | null;
  source: "cache" | "llm" | null;
  provider: "anthropic" | "codex" | "ollama" | null;
  model: string | null;
  prompt_version: "grounded-coach.v1";
  reason:
    | "api_key_missing"
    | "local_model_unavailable"
    | "provider_error"
    | "invalid_response"
    | "timeout"
    | null;
};
```

Move shape becomes:

```ts
type AnnotatedMove = {
  // existing Slice 1/2 fields
  motifs: Motif[];
  explanation: MoveExplanation | null;
};
```

Rules:

- `status: "ok"` requires non-empty `text`, at most 3 sentences.
- `status: "unavailable"` is used for missing local key/config and never fails review.
- `status: "error"` is used for provider timeout/error or failed response validation and
  does not fail the review flow.
- No explanation is generated automatically during `POST /api/games`.
- No explanation is requested for moves with `motifs.length === 0`.

## Prompt And Grounding

Add `chess_ml/explanation/prompt.py` with structured prompt assembly around
`ExplanationRequest`.

Prompt input includes:

- `fen_before`, `fen_after`, `user_move` SAN/UCI, side to move, move number.
- `actual_line`: the selected move plus up to five following moves from the game, in SAN
  and UCI.
- `analysis_before.best_move`, `analysis_before.pv`, `analysis_after.best_move`,
  `analysis_after.pv`.
- `loss_cp`, mate/CP score kind, selected primary motif, all motif IDs, motif evidence
  piece/square/attackers/defenders.
- `game_phase` from motif evidence.

Primary motif selection is deterministic: highest severity first, then `allowed_tactic`,
`missed_tactic`, `endgame_slip`, `hanging_piece`, `opening_inaccuracy`. The prompt may
receive all motifs but must teach one lesson.

System prompt rules:

- Stockfish is ground truth; never recommend a move that is not the provided engine best
  move/PV.
- The LLM compares the player's move/actual game line with Stockfish's best move/PV; it
  does not calculate chess.
- Use only provided facts; do not infer hidden tactics beyond the PV/evidence.
- Reference concrete moves, pieces, or squares.
- Teach exactly one practical lesson for a 1200-2000 club player.
- Output strict JSON: `{ "text": "...", "referenced_move_uci": "..." | null }`.
- The `text` must be <=3 sentences and no more than about 70 words.

Grounding semantics:

- For missed tactic, endgame slip, opening inaccuracy, and most hanging-piece
  explanations, the recommended line is `analysis_before.pv`.
- For allowed tactic, the ground-truth line is `analysis_after.pv`, because the tactic is
  available to the opponent after the user move.
- Response validation rejects text that names a different `referenced_move_uci`, exceeds
  sentence limit, or returns malformed JSON. Rejected responses are not cached.

## Cache And Storage

Add `chess_ml/explanation/cache.py` using standard SQLite, with default DB path
`data/chess_ml.sqlite3`, overridable by `CHESS_ML_DB_PATH`. Ensure the directory/table is
created on startup; `*.sqlite3` is already gitignored.

Use table `explanation_cache`:

```sql
cache_key TEXT PRIMARY KEY,
prompt_version TEXT NOT NULL,
provider TEXT NOT NULL,
model TEXT NOT NULL,
text TEXT NOT NULL,
request_json TEXT NOT NULL,
response_json TEXT NOT NULL,
created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
```

Cache key is content-addressed:

```text
sha256(canonical_json({
  key_version: "explanation-cache-key.v1",
  prompt_version: "grounded-coach.v1",
  fen_before,
  fen_after,
  user_move_uci,
  primary_motif_id,
  motifs: stable motif IDs + severity + score/evidence fields,
  engine_before: score + best_move_uci + pv_uci[],
  engine_after: score + best_move_uci + pv_uci[]
}))
```

This intentionally includes PV/eval facts so an explanation cannot be reused after
engine depth or prompt grounding changes make the old text stale. Cache successful
validated explanations only; do not cache provider errors, invalid responses, or
missing-key placeholders.

## Local Config And Missing Keys

Add an `ExplanationClient` protocol and an `ExplanationService` that accepts any client.
Concrete provider selection comes from env:

- `CHESS_ML_EXPLANATION_PROVIDER=auto|ollama|anthropic|codex|disabled`.
- Default `auto` uses local Ollama, not a hosted API key.
- Ollama defaults: `CHESS_ML_OLLAMA_BASE_URL=http://localhost:11434` and
  `CHESS_ML_OLLAMA_MODEL=qwen3:8b`.
- Hosted adapters remain opt-in: `ANTHROPIC_API_KEY` for Anthropic and
  `CODEX_API_KEY`/`OPENAI_API_KEY` for the Codex/OpenAI-compatible path.
- Load `.env` locally so the documented gitignored key workflow actually works.

`POST /api/games` returns the full engine review and motifs without calling Ollama. When
the user asks for a selected move's explanation and no local model is running, the
on-demand endpoint returns:

```json
{
  "schema_version": "move-explanation.v1",
  "status": "unavailable",
  "text": null,
  "source": null,
  "provider": "ollama",
  "model": "qwen3:8b",
  "prompt_version": "grounded-coach.v1",
  "reason": "local_model_unavailable"
}
```

Frontend behavior: selected flagged moves show a `Generate coach note` button while
`explanation` is `null`, explanation text when `ok`, a short local Ollama setup note when
`unavailable`, and a non-blocking retry-style message when `error`.

## Test Plan

Python unit tests:

- Prompt assembly includes FEN, user move, engine PV, motif evidence, and the correct
  before/after PV for `missed_tactic` vs `allowed_tactic`.
- System prompt contains the grounding rules, sentence limit, JSON output requirement,
  and no image/OCR/screenshot language.
- Cache key is deterministic, changes when PV/prompt version changes, and ignores JSON
  ordering noise.
- Cache hit returns stored text and does not call the client.
- Missing explicit hosted API key returns `unavailable` and does not call the client.
- Default provider selection returns an Ollama client, and local Ollama connection/model
  failures return `unavailable` without failing review.
- Fake client success is validated, stored, and returned as `ok`.
- Fake client responses that mention the wrong move, exceed 3 sentences, or return
  malformed JSON become `error` and are not cached.

API/frontend checks:

- `POST /api/games` remains successful and does not call Ollama when explanations are
  configured.
- Moves initially include `explanation: null`; the on-demand endpoint returns a
  `move-explanation.v1` status object for a selected flagged move.
- Frontend TypeScript types and current-move panel handle `ok`, `unavailable`, `error`,
  and `null`.
- Run `make check` and `cd web && npm run typecheck`.

## Assumptions To Validate

- Additive `explanation` under `annotated-game.v1` is acceptable; no schema-version bump.
- One explanation per flagged move, not one per motif, is the right UX for Slice 3.
- Synchronous explanations inside `POST /api/games` are acceptable for this slice;
  streaming can remain deferred.
- Default cache DB path `data/chess_ml.sqlite3` is acceptable and can later share the
  Slice 6 profile DB.
- Provider adapter is the right path; local Ollama is the default for now, while hosted
  API providers remain opt-in for later quality testing.
- Missing local model or hosted keys should return `unavailable` instead of deterministic
  fallback coaching text, so the app stays honest and review remains usable.
