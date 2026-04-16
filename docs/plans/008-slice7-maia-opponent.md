# Plan 008 - Slice 7: Maia/Lc0 Human-Like Opponent

Status: approved for implementation
Owner: barathkrishna
Last updated: 2026-04-15

## Summary

Add Maia as the default human-like play opponent while preserving the existing
low-strength Stockfish opponent as a local fallback. This slice keeps the loop thin:
choose a Maia rating band, play as White, resign or finish, then review the PGN through
the existing Stockfish-grounded review pipeline.

Maia is only a play opponent. Stockfish remains the sole authority for review analysis,
PVs, evals, motifs, and explanation grounding.

## Key Changes

- Add `chess_ml/engine/maia.py`, wrapping an Lc0 UCI process loaded with Maia v1
  weights. Use `go nodes 1` through python-chess so Maia behaves as a policy imitation
  model rather than a search-heavy analysis engine.
- Support rating bands `1100`, `1500`, and `1900` for this slice. Default to Maia
  `1500`, matching the target club-player range without creating a large selector.
- Add a play-opponent selector:
  - `auto` chooses Maia when `lc0` and the requested weight exist, otherwise uses the
    existing Stockfish fallback.
  - `maia` requires Maia setup and returns `503 opponent_unavailable` if it is missing.
  - `stockfish` always uses the existing low-strength Stockfish opponent.
- Extend `POST /api/play/new` with an optional body:
  - `opponent: "auto" | "maia" | "stockfish"`; default `auto`.
  - `maia_rating: 1100 | 1500 | 1900`; default `1500`.
  Existing no-body calls continue to work.
- Add opponent metadata to `play-state.v1`: actual kind, label, engine name, requested
  kind, rating band, and fallback reason.
- Add `GET /api/play/opponents` for non-probing setup status so the frontend can explain
  whether Maia, Stockfish fallback, or both are locally available.
- Add `scripts/fetch_maia_weights.sh` and `make setup-maia`. `make setup` runs the fetch
  script after package installs. Downloaded weights live in gitignored
  `checkpoints/maia/`.
- Update `/play` with a small opponent/rating selector, clear actual-opponent labeling,
  and fallback messaging. Keep the current board, move submission, resign, and
  post-game review flow intact.
- Update README setup notes for `brew install lc0`, Maia weights, env vars, and the
  analysis boundary.

## API And Storage

`play-state.v1` gains an additive `opponent` field. No database migration is needed.
The PGN exporter keeps `Event "chess_ml local play"` so dashboard source detection keeps
working; `Black` becomes the actual opponent label and `BlackElo` is set for Maia
rating bands.

Environment variables:

- `CHESS_ML_LC0_PATH`: Lc0 executable override; default uses `lc0` on `PATH`.
- `CHESS_ML_MAIA_WEIGHTS_DIR`: Maia weights directory; default `checkpoints/maia`.
- Existing Stockfish play env vars remain unchanged for fallback mode.

## Test Plan

- Python tests cover Maia config resolution, missing setup status, invalid rating
  rejection, no-body `POST /api/play/new` compatibility, explicit Maia failure,
  `auto` fallback, Stockfish selection, and PGN opponent headers.
- Add a skipped real-Lc0 smoke test that runs only when the configured Lc0 binary and a
  Maia weight file exist locally.
- Frontend typecheck covers the new play request, opponent status, and play-state
  metadata.
- Manual verification: start Maia 1500 when installed, make several moves, resign, and
  confirm review loads through `POST /api/games`.

Required verification before declaring done:

- `make check`
- `cd web && npm run typecheck`
- `cd web && npm run build`

## Assumptions

- Maia v1 via Lc0 is the MVP integration path. Maia-2 is deferred because it uses a
  Python/PyTorch inference path and would make this slice broader.
- Default startup is `auto` + Maia `1500`, with fallback to Stockfish to preserve
  local-first usability on fresh clones.
- No clocks, opening books, bot personalities, cloud setup, auth, sharing, payments,
  image inputs, OCR, or computer vision are part of this slice.
