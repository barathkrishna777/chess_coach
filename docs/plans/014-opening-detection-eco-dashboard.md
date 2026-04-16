# Plan 014 — Opening Detection + ECO Dashboard Drilldown

Status: **approved for implementation**
Owner: barathkrishna
Last updated: 2026-04-16

## Summary

Implement Slice 14 only: tag analyzed standard-start games with the longest matching ECO line, persist that tag with reviewed games, expose opening aggregates in the profile API, show the opening in review, and add an Openings dashboard section that filters recent games.

No Slice 15 motif work, opening preparation, transposition detection, network lookup, auth/cloud/sharing, image input, or ML training is included.

## API And Data Contracts

- Add a compact checked-in ECO table as Python data alongside a new detector in `chess_ml/classifier/openings.py`; no new package dependency and no runtime network calls.
- Detector API:
  - `OpeningTag(eco: str, name: str)`
  - `detect_opening(moves_uci: Sequence[str], *, initial_fen: str = chess.STARTING_FEN) -> OpeningTag | None`
  - Return `None` for non-standard initial FENs.
  - Compare only the first 20 plies.
  - Choose the longest ECO entry whose UCI sequence is a prefix of the game sequence.
- Extend `annotated-game.v1` with `opening: { eco: string, name: string } | null`.
- Extend `profile-dashboard.v1` with:
  - `openings: [{ eco: string, name: string, games: number, avg_loss_cp: number, top_motif: { id: string, label: string, count: number } | null }]`
  - `recent_games[].opening: { eco: string, name: string } | null`
- Keep internal engine/API field names such as `pv` unchanged, but all new user-facing copy must say “main line,” never “PV.”

## Implementation Changes

- Review pipeline:
  - In the PGN review path, derive `moves_uci` from parsed moves, run the detector during `annotate_game`, and attach the opening tag to the returned game.
  - Include the opening when converting an annotated game into `ProfileGameReview`.
  - The ECO table must cover the three demo fixtures at minimum and include common prefix/longer-line cases needed to prove longest-match behavior, including `B01 — Scandinavian Defense` for the existing missed-tactic demo.
- Profile storage:
  - Add nullable `eco_code` and `opening_name` fields to the game review dataclass and `profile_games`.
  - Use the existing `_ensure_columns` migration helper so old SQLite DBs are upgraded automatically.
  - Preserve idempotent `ON CONFLICT(game_id)` upsert semantics and update opening fields on replacement review.
  - Include opening metadata in recent games while keeping old rows with null opening metadata readable.
- Profile aggregation:
  - Build openings from `profile_games` left-joined to motif occurrences.
  - `games` is distinct reviewed games per ECO/name.
  - `avg_loss_cp` is the average non-null motif loss for that opening, rounded to two decimals, defaulting to `0.0` when no motif loss exists.
  - `top_motif` is the highest-count motif for that opening, tie-broken by label; `null` when no motifs exist.
- Frontend:
  - Add `OpeningTag`/opening aggregate types in the web type contract and no `any`.
  - Show `Opening: B01 — Scandinavian Defense` or equivalent concise text in the review surface when `game.opening` exists.
  - Add an Openings dashboard section below the summary/motif/phase area and above Recent games.
  - Opening rows show ECO/name, games, average loss, and top motif.
  - Clicking an opening row sets local dashboard state and filters Recent games to matching `recent_games[].opening.eco`; include a clear/reset control and a small label indicating the active filter.
  - Do not touch `web/components/Board.tsx` movement behavior unless a type update is unavoidable; preserve existing click and drag play tests.

## Tests And Verification

- Backend/unit tests:
  - Add detector tests proving longest matching line wins and non-standard initial FEN returns no tag.
  - Extend PGN/parser or games API tests so analyzed game JSON includes opening metadata.
  - Extend profile store tests for idempotent opening persistence, replacement update by `game_id`, profile API `openings`, and old DB migration/readability without manual deletion.
  - Extend demo tests so seeded demo profile has at least two distinct openings.
- Playwright:
  - Uploaded PGN review shows opening metadata.
  - Dashboard displays Openings aggregate rows after demo seed.
  - Selecting `B01` or another seeded opening filters Recent games and reset restores all games.
  - Keep existing upload review, dashboard, training drill, play drag/click, promotion, black play, takeback, hint, error, and network/engine failure paths green.
- Required commands before declaring done:
  - `make check`
  - `cd web && npm run typecheck`
  - `cd web && npm run build`
  - `make e2e`
  - If uv/npm/Playwright cache or local server startup is blocked by sandboxing, request escalation and rerun.

## Assumptions

- The compact ECO table is intentionally small for Slice 14 and not a full opening book; it must be easy to expand later.
- No transposition detection: only exact UCI-prefix matching from the game’s early main line.
- Existing reviewed games without opening metadata remain valid and simply show no opening.
- The current worktree is clean at planning time; if implementation starts after new user edits appear, inspect and preserve them before touching overlapping files.
