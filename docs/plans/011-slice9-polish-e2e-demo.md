# Plan 011 - Slice 9: Polish + End-to-End Demo

Status: approved for implementation
Owner: barathkrishna
Last updated: 2026-04-16

## Summary

Finish the MVP hardening slice by making a fresh clone immediately demoable,
adding deterministic end-to-end coverage, and tightening frontend failure copy.
The slice stays local-first: Stockfish remains required for review/demo analysis,
while Maia and Ollama stay optional enhancements.

Existing Slice 8 eval report edits in `docs/evals/009-slice8-classifier-v1.json`
are not part of this slice and must not be overwritten, staged, or committed.

## Key Changes

- Add three checked-in demo PGNs under `tests/fixtures/demo/` that produce visible
  deterministic motif coverage for dashboard and review.
- Replace the failing `make demo` stub with `python -m chess_ml.profile.demo`,
  which analyzes the demo PGNs through the existing parse/evaluate/classify/profile
  path and upserts by deterministic `game_id`.
- Refactor the internal game annotation helper enough for the API, demo seeder,
  and tests to share the same Stockfish-like evaluator interface without changing
  public API schemas.
- Add Playwright e2e support with `npm run e2e` and `make e2e`, using a temporary
  seeded SQLite DB, disabled coach providers, disabled learned checkpoint loading,
  and forced Maia-unavailable auto-mode fallback.
- Harden frontend copy for bad PGN, API/network failure, Stockfish unavailable,
  and analysis timeout, while keeping local coach notes lazy and non-blocking.

## Public Interfaces

- `make demo` seeds three sample reviewed games into `CHESS_ML_DB_PATH` or
  `data/chess_ml.sqlite3`; rerunning it refreshes the same rows.
- `CHESS_ML_DEMO_STOCKFISH_DEPTH` controls demo analysis depth and defaults to `6`.
- `make e2e` runs the Playwright MVP suite.
- Frontend API helpers throw typed local errors with `status` and backend `code`
  when available.

## Test Plan

- Add unit/integration tests for demo seeding with a fake Stockfish-like evaluator,
  including repeated seeding without duplicate profile rows.
- Add Playwright tests for PGN upload to review, seeded dashboard, local play with
  Stockfish fallback, post-game review, bad PGN copy, API-down copy, and
  Stockfish-timeout/unavailable copy.
- Required verification:
  - `make demo`
  - `make check`
  - `cd web && npm run typecheck`
  - `cd web && npm run build`
  - `make e2e`
  - Start `make serve` if practical, or state clearly if not run.

## Assumptions

- Stockfish is mandatory for demo seeding because it is the product's source of
  chess truth.
- Demo seeding writes profile/dashboard rows only; it does not create cached LLM
  advice and does not require Ollama.
- The e2e suite uses the real local API and web app for happy paths, with targeted
  request interception only for deterministic frontend failure-state assertions.
- No image upload, screenshot, OCR, computer vision, cloud deployment, auth,
  payments, or sharing are added.
