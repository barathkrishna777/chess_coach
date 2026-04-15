# Plan 005 — Slice 4: In-App Play Loop With Stockfish Fallback

Status: approved for implementation
Owner: barathkrishna
Last updated: 2026-04-15

## Summary

Build `/play` next, before dashboard. This is the better product move because it creates
the core loop: play a game locally, finish or resign, immediately review it with the
existing Stockfish/motif/on-demand coach-note flow. Dashboard can follow once the app is
producing first-party games worth aggregating.

Use low-strength Stockfish for the first opponent slice, not Maia/Lc0 yet. Keep it
behind a small opponent interface so Maia can replace the fallback later without
rewriting `/play`, the session model, or the frontend.

## Key Changes

- Add a new backend play API:
  - `POST /api/play/new` starts a white-only local game and returns `play-state.v1`.
  - `POST /api/play/move` accepts one user UCI move, validates it with `python-chess`,
    applies it, gets the bot reply if the game is still active, and returns updated
    state.
  - `POST /api/play/resign` ends the game as a white resignation and returns a valid PGN.
  - Responses include FEN, move history, legal move destinations, game status, result,
    PGN when finished, and the latest bot move.
- Add a play-domain layer:
  - In-memory local sessions keyed by UUID for this slice.
  - `OpponentMoveProvider` protocol with a `choose_move(fen) -> EngineMove` shape.
  - `StockfishPlayOpponent` as the v1 provider, using a separate Stockfish process/pool
    from review analysis so low-skill play settings never affect ground-truth review
    evaluations.
  - Default opponent config: local Stockfish, low-strength mode using available UCI
    limiting options, short per-move time budget, env-configurable.
- Add `/play` frontend:
  - Interactive chessground board, white-only.
  - Legal moves come from the backend; the frontend does not do chess analysis.
  - User moves by dragging/clicking; bot replies are applied after the API returns.
  - Resign button and automatic post-game path.
  - When the game ends or the user resigns, convert the returned PGN through existing
    `POST /api/games`, then show the same review UI with motifs and lazy coach notes.
  - Promotion support should be minimal but legal: if a from/to pair has multiple
    promotion moves, ask for promotion piece before submitting UCI.
- Refactor frontend review code just enough to reuse it:
  - Extract shared annotated-game types and review UI from `web/app/page.tsx` into
    reusable modules/components.
  - Keep the existing upload page behavior unchanged.
  - Use the extracted review component from both `/` and `/play` after analysis
    completes.

## API And Interfaces

`play-state.v1` should include:

- `game_id`
- `status: "active" | "completed" | "resigned"`
- `result: "1-0" | "0-1" | "1/2-1/2" | "*"`
- `fen`
- `orientation: "white"`
- `legal_moves`, grouped by source square with destination and promotion options
- `moves`, with SAN/UCI/side/ply
- `bot_move: MoveRef | null`
- `pgn: string | null`

Error responses should follow the existing envelope style:

- `400 illegal_move`
- `404 game_not_found`
- `409 game_already_finished`
- `503 opponent_unavailable`

## Test Plan

Python unit/API tests:

- Starting a game returns initial FEN and legal white moves.
- A legal user move produces a valid bot reply and updated FEN.
- Illegal moves are rejected without mutating the session.
- Resignation returns `0-1` and a PGN accepted by the existing PGN parser.
- Completed games return PGN and do not request another bot move.
- Opponent provider can be faked in tests so core API tests do not require Stockfish.

Frontend checks:

- `/play` typechecks with strict TypeScript.
- Existing upload/review page still typechecks after extracting shared components.
- Manual local flow: `make serve`, open `/play`, play a few moves, resign, verify review
  loads, select a flagged move, generate a coach note.

Required verification before declaring done:

- `make check`
- `cd web && npm run typecheck`

## Assumptions

- First play slice is white-only.
- Low-strength Stockfish is acceptable for v1 as long as opponent selection is swappable
  later.
- No clocks, rating selector, dashboard/profile writes, Maia/Lc0 setup, auth, sharing,
  cloud deployment, image inputs, OCR, or computer vision in this slice.
- Review remains the source of truth for analysis; the play opponent is only an opponent,
  never the explanation or evaluation authority.
