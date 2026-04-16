# Plan 013 — Play-Mode Ergonomics

Status: **approved, implementation in progress**
Owner: barathkrishna
Last updated: 2026-04-16

## Summary

Slice 13 removes friction from the local play loop without expanding product scope. It adds an in-board promotion chooser, lets the user play either color, adds one takeback and three Stockfish-grounded hints per game, and gives the post-game review screen a `Play again` path that preserves the selected opponent, rating, and color.

This slice does not add auth, multi-user state, cloud behavior, image inputs, classifier changes, or explanation pipeline changes.

## API And Session Changes

- Extend `POST /api/play/new` with `user_color: "white" | "black"`, defaulting to `"white"`.
- Return `user_color`, `orientation`, `hints_remaining`, and `takebacks_remaining` in every `play-state.v1` response.
- Store `user_color` on `PlaySession`; only expose legal moves when it is the user's turn.
- If the user plays Black, the selected local opponent makes White's first move before the new-game response returns.
- Add `POST /api/play/takeback` with `{ game_id }`; undo the latest user move plus any following bot reply, limit to one success per active game, and return 409 when unavailable.
- Add `GET /api/play/hint?session_id={id}`; use the existing Stockfish pool at short depth and return the best move as structured data, limited to three successful hints per active game.

## Frontend Changes

- Replace `window.prompt()` promotion handling in `/play` with React state and an in-board promotion dialog.
- Add a White/Black selector to the play setup and pass the selected color to the API.
- Flip board orientation for Black and keep `turnColor`/`movableColor` aligned with the user color.
- Add Takeback and Hint buttons; render hints as chessground arrows and clear arrows after moves, takebacks, resigns, or new games.
- Minimally extend `Board` with programmatic auto-shapes while preserving existing click and drag behavior.
- Show `Play again` above post-game review and start a fresh game with the prior opponent/rating/color settings.

## Tests

- Backend tests cover Black start, Black resignation headers/result, takeback success and 409 limits, hint success and limit, and promotion UCI acceptance.
- Playwright tests cover existing play drag behavior, click movement, promotion dialog flow, Black orientation/bot first move, takeback, hint arrow, and Play again.

## Verification

Run before declaring the slice complete:

- `make check`
- `cd web && npm run typecheck`
- `cd web && npm run build`
- `make e2e`
