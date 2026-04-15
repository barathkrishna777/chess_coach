# Plan 007 - Slice 6: Profile Dashboard

Status: approved for implementation
Owner: barathkrishna
Last updated: 2026-04-15

## Summary

Add the local profile loop: every successful game review is saved into SQLite, and a
new `/dashboard` page shows the user's reviewed-games weakness profile. This slice
keeps the product single-user and local-first. It does not add accounts, cloud sync,
sharing, payments, deployment, image inputs, OCR, screenshots, advanced analytics, or
chart-library integration.

For this slice, the dashboard aggregates all flagged moves from reviewed games. The
store still records move side and player metadata so a later local-identity slice can
filter to "my side" without changing the persistence shape.

## Key Changes

- Add a small explicit SQLite profile store under `chess_ml/profile/`, using the same
  `CHESS_ML_DB_PATH` default as the explanation cache: `data/chess_ml.sqlite3`.
- Persist successful `POST /api/games` reviews after analysis and motif classification,
  whether the PGN came from upload or the `/play` post-game review path.
- Upsert reviews by deterministic `game_id`; re-analyzing the same PGN replaces that
  game's motif rows before inserting the new set so aggregates never double-count.
- Add `GET /api/profile/me`, returning `profile-dashboard.v1` from stored rows only.
  This endpoint must not run Stockfish, invoke the classifier, call Ollama, or depend on
  coach notes.
- Add `/dashboard` with local summary totals, a ranked motif list, phase breakdown,
  recent games, and a clear empty state.
- Add navigation links between Review PGN, Play, and Dashboard.

## API And Storage

`GET /api/profile/me` returns:

- `totals`: games reviewed, moves reviewed, flagged moves, motif occurrences, and motif
  occurrences per 100 reviewed moves.
- `motifs`: ranked rows with motif id, label, count, and rate per 100 reviewed moves.
- `phase_breakdown`: opening, middlegame, and endgame rows with counts and rates.
- `recent_games`: latest saved reviews with game id, players, result, source,
  timestamps, ply count, and flagged move count.

SQLite tables:

- `profile_games`: one row per `game_id`, including player names/elos, result, source,
  ply count, and created/updated timestamps.
- `profile_motif_occurrences`: one row per game/move/motif, including ply, SAN/UCI,
  side, motif id/label/severity, phase, and loss/score fields.

Use `CREATE TABLE IF NOT EXISTS` for this MVP. Keep the existing `explanation_cache`
table intact.

## Test Plan

- Profile store unit tests cover insert/upsert, same-game reinsert without
  double-counting, motif counts/rates, phase breakdown, and newest-first recent games.
- API tests cover an empty profile on a fresh DB and `POST /api/games` writing profile
  rows through a fake Stockfish pool.
- Required verification:
  - `make check`
  - `cd web && npm run typecheck`
- Run `cd web && npm run build` if the dashboard UI changes are broad enough to warrant
  a production build check.

## Assumptions

- "Profile" means the local reviewed-games corpus for this slice, not a named account.
- Rates use total reviewed plies as the denominator.
- `flagged_moves` counts distinct game/ply pairs with at least one motif, while
  `motif_occurrences` counts every motif chip.
- Recent games sort by `updated_at` so re-analyzing a PGN refreshes it without
  duplicating it.
