# Plan 010 - Coach Grounding Repair

Status: approved for implementation
Owner: barathkrishna
Last updated: 2026-04-16

## Summary

Stop untrusted LLM chess advice from appearing in review. The backend will return
a deterministic Stockfish-grounded fallback when an LLM response fails validation,
with explicit metadata showing it is fallback text. Bad model output is never
displayed or cached.

This slice prioritizes a boring, factual coach note over fluent but unreliable
analysis. Stockfish remains the only source of chess truth; motifs provide
coaching labels and evidence; the LLM only paraphrases validated facts.

## Key Changes

- Add deterministic fallback generation in the explanation layer:
  - Use only the selected move, Stockfish best move/PV, eval loss, primary motif,
    motif evidence, phase, and actual game line already present in
    `ExplanationRequest`.
  - Keep wording factual: name the played move, Stockfish move/line, loss if
    available, motif, phase, and concrete motif evidence when present.
  - Do not infer tactics such as "wins the queen", "fork", or "mate threat" unless
    that exact claim is already represented by supplied facts.
- Tighten provider validation:
  - Require `referenced_move_uci` to exactly match the expected Stockfish best-move
    UCI when a best move exists.
  - Require displayed text to reference the expected Stockfish best move or PV when
    the prompt has a concrete engine line.
  - Reject generic text that omits the engine move/line.
  - Reject text that claims a different best, recommended, or Stockfish move from
    the supplied best move/PV.
- Update the public explanation contract:
  - Extend `MoveExplanation.source` from `"cache" | "llm"` to
    `"cache" | "llm" | "fallback"`.
  - Invalid LLM output returns `status: "ok"`, `source: "fallback"`,
    `reason: "invalid_response"`, fallback text, and attempted provider/model
    metadata.
  - Provider missing, timeout, local model unavailable, and provider errors keep
    their current unavailable/error behavior.
  - Do not cache invalid LLM responses or deterministic fallback responses; only
    validated LLM text remains cacheable.
- Update the review UI so `source: "fallback"` is shown as Stockfish-grounded
  fallback, not model advice.
- Preserve lazy coach-note behavior: `POST /api/games` still does not generate
  explanations.

## Tests

- Add or update explanation tests for:
  - wrong `referenced_move_uci` is rejected.
  - invented best/recommended move in text is rejected even if the expected move is
    also mentioned.
  - generic advice with no engine move reference is rejected.
  - valid grounded explanation still passes.
  - invalid provider output returns `source: "fallback"`, `status: "ok"`,
    `reason: "invalid_response"`, and is not cached.
  - fallback text mentions only supplied Stockfish/motif facts and does not include
    the rejected provider text or invented tactical claims.
  - cache hits for validated LLM output still skip provider calls.
- Required verification:
  - `make check`
  - `cd web && npm run typecheck`
  - `cd web && npm run build` if frontend changes become layout-significant.

## Assumptions

- The simplest compatible product behavior is to show deterministic fallback when
  model validation fails.
- Fallback is trusted because it is generated from existing Stockfish and motif
  facts, but it is not cached in SQLite for Slice 10 to avoid a cache schema
  migration.
- The LLM remains optional and local-first through Ollama; this repair does not add
  hosted-provider work, cloud deployment, image inputs, auth, payments, or sharing.
