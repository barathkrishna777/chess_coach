# Plan 006 — Slice 5: Coach Note Latency And UX Hardening

Status: approved for implementation
Owner: barathkrishna
Last updated: 2026-04-15

## Summary

Make local coach notes feel deliberate and trustworthy even when Ollama is slow or
missing. Keep explanations lazy and Stockfish-grounded, preserve both PGN upload review
and `/play` post-game review, and avoid any fallback chess advice that bypasses the
existing grounding pipeline.

## Key Changes

- Keep coach notes on demand. `POST /api/games` continues to return review data with
  `explanation: null`, and selected flagged moves use `POST /api/games/explain`.
- Tighten timeout handling with `CHESS_ML_EXPLANATION_TIMEOUT_SECONDS`, defaulting to 15
  seconds, enforced at the service layer with `asyncio.wait_for` so every provider path
  obeys the same budget.
- Preserve strict grounding. Cache only validated successful explanations; keep timeout,
  provider failure, unavailable model, and invalid/untrusted responses uncached; never
  return generic fallback chess advice.
- Add a lightweight `GET /api/games/explain/status` endpoint returning provider, model,
  enabled/configured state, and timeout. This endpoint must not contact Ollama or any
  hosted provider.
- Add additive explanation metadata if useful for the UI, such as timeout seconds and
  retryability, while keeping `status: "ok" | "unavailable" | "error"`.
- Upgrade the review coach panel to explicitly communicate: not requested, generating,
  cached, unavailable because local Ollama/model is missing, timed out, invalid/untrusted
  response, and retry available.
- Document explanation configuration in README: `CHESS_ML_EXPLANATION_PROVIDER`,
  `CHESS_ML_EXPLANATION_TIMEOUT_SECONDS`, `CHESS_ML_OLLAMA_BASE_URL`,
  `CHESS_ML_OLLAMA_MODEL`, and `CHESS_ML_DB_PATH`.

## Test Plan

- Python tests cover service-level timeout, local provider unavailable, invalid response
  not cached, cache hit skipping the client, and the status endpoint returning config
  without contacting Ollama.
- Frontend type checking covers the explanation status/reason/source combinations used
  by the coach panel.
- Required verification before declaring done:
  - `make check`
  - `cd web && npm run typecheck`
  - `cd web && npm run build` if the UI changes are broad enough to warrant it.

## Assumptions

- Default explanation timeout is 15 seconds: short enough to avoid frozen-feeling local
  UX, long enough for a small local model on reasonable hardware.
- “Generating” and “not requested” remain frontend states; backend stores only completed
  outcomes.
- “Cached” is represented by `status: "ok"` plus `source: "cache"`, not a new backend
  status.
- The status/config indicator is informational, not a settings UI and not a startup
  health probe.
