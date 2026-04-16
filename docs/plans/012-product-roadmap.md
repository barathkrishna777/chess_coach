# Plan 012 — Product Roadmap: From Analyzer to Coach

Status: **approved, not yet implemented**
Owner: barathkrishna
Last updated: 2026-04-16

This document covers the product/engineering roadmap for chess_ml after Slice 10. It is intended as a Codex implementation guide. Read [docs/plans/001-mvp.md](001-mvp.md) first for the canonical architecture and hard constraints.

---

## Executive Summary

The technical foundation is solid: Stockfish analysis is correct and grounded, the heuristic classifier reliably tags blunders, coach notes are validated and never contradict the engine, the play loop works, and the dashboard is a real aggregated weakness profile. `make check && make demo && make serve` works on a fresh clone.

The gap is product surface, not architecture. A 1200–2000 rated club player who uploads a game sees move-by-move eval changes and a few sentences of text. That is a better Lichess analysis board, not a chess coach. The next set of slices should convert the analysis output into interactive lessons — letting the user touch the board, drill their own mistakes, and navigate naturally.

No new ML/grounding work is needed to unlock the highest-value improvements. The `MotifEvidence` dataclass already has attackers, defenders, piece refs, best_move, opponent_reply, and related_ply. The learned classifier just needs a real Lichess training run (~10k+ examples). The explanation system is already well-designed.

**Recommendation: start with Slice 11.** It is pure UI on existing data, requires no new API endpoints, and turns the analyzer into a coach in a single focused sprint.

---

## Diagnosis

### Strengths

1. **Grounding pipeline is correct.** Stockfish PV + eval delta is the single source of truth. The LLM only paraphrases; it never invents moves. Validated via `referenced_move_uci` match + engine-line reference check + disallowed-claims detection. Invalid LLM responses are silently dropped and replaced with deterministic Stockfish-grounded fallback text. This is the right architecture.

2. **Heuristic classifier is reliable.** Five motifs (hanging_piece, missed_tactic, allowed_tactic, endgame_slip, opening_inaccuracy) with tuned CP thresholds. Rich `MotifEvidence` dataclass with full tactical context. False positive rate is low because all detectors require an eval-delta gate.

3. **Profile aggregation is real.** `profile_motif_occurrences` table, upsert by SHA-256 game_id, phase breakdown, motif ranking. Dashboard surfaces an actual weakness pattern across games, not just a list of games.

4. **Play loop works.** Maia 1100/1500/1900 with Stockfish fallback. Post-game review through the existing `POST /api/games` pipeline. Play + review is a coherent user journey.

5. **Demo story is solid.** `make demo` seeds 3 fixture games idempotently. Fresh clone → `make demo && make serve` → working product in under 2 minutes.

6. **E2E coverage.** 5 Playwright tests covering the critical path. Engine failure and network failure copy is tested.

### Weaknesses by Area

#### UX — Review flow (highest priority)

- **No keyboard navigation.** The user must click each move in the move list. Arrow-key nav (← → to step, ↑↓ to jump to prev/next flagged move) is the standard chess UI pattern and is expected.
- **No PV replay on the board.** The `Best` and `Line` fields are shown as text but there is no way to step through the engine's continuation on the board. This is the biggest missing feature for a 1200-rated player — they need to see *what would have happened* on the board, not just read a move sequence.
- **No "try the engine's move" interaction.** After seeing a mistake, the most valuable coaching moment is: can the user find the best continuation themselves? There is no way to do this today.
- **No top-of-review summary card.** On load, the user sees nothing until they click a move. A summary card (2 biggest mistakes, most common motif, accuracy score) would orient them immediately.
- **No jump-to-next-blunder shortcut.** Stepping through 40 moves to find the 3 blunders is tedious.

#### UX — Play mode

- **`window.prompt()` for promotion.** This is a known wart (play/page.tsx:367). It breaks the UI illusion and will fail in automated tests.
- **White-only.** The user always plays white. Playing black is a common training request.
- **No takeback.** Takeback is expected in a coaching context — the point is to learn, not to suffer.
- **No "play again" button.** After post-game review, the user must navigate back to `/play` manually.
- **No hint system.** During play, there is no way to ask the engine for a hint. Even one hint per game that counts against a score would be valuable.

#### Coaching value — Personalization

- **Dashboard motif counts are not actionable.** "You have 14 missed tactics" tells the user they have a problem but doesn't give them a path to fix it. The highest-leverage next step is a drill mode that generates a puzzle from each of their own flagged positions.
- **No opening detection.** The opening_inaccuracy motif fires on moves in the first 15 plies with a 50–150cp drop, but the dashboard does not surface which openings the user plays and which ones they blunder in. A 1400-rated player who blunders in the Sicilian consistently would benefit enormously from knowing that.
- **Explanation quality is limited by the 70-word constraint.** For some motifs (especially endgame_slip and allowed_tactic), the explanation does not teach the underlying pattern — it just names the best move. The prompt could be improved.

#### Coaching value — Motif coverage

- **Pins, forks, overloaded defenders, and discovered attacks are not detected.** The heuristic suite covers blunder magnitude (eval delta) and some tactical context (hanging piece, missed tactic, allowed tactic) but not the *pattern* of the tactic. A user who keeps walking into pins will not see that pattern surfaced — they will just see repeated "missed tactic" tags.
- **Learned classifier F1=0.** The Slice 8 model is a reproducibility smoke test on 28 fixture examples. It needs a real Lichess ingestion run (~10k+ examples) before it contributes value. Until then, it is silent and heuristics are the entire classifier.

#### Engineering — Minor

- **Single `review_lock` serializes game analysis.** If two tabs are open (or a test is slow), the second request blocks. For MVP this is fine, but it will become an issue as soon as there are real concurrent users.
- **Coach note latency for cold Ollama (~10–30s).** The status endpoint and UI states are well-designed (not_requested / generating / cached / unavailable / timeout / invalid). The latency itself is a local hardware constraint, but the UX could do more to set expectations.
- **`MotifEvidence` fields not surfaced in UI.** `attackers`, `defenders`, `opponent_reply`, `related_ply` are computed and stored but not shown. These would be valuable in the review panel.

---

## Prioritized Roadmap

### Tier 1 — Do these next (pure value, no new ML)

| Slice | What | Why |
|-------|------|-----|
| 11 | Interactive review (PV board replay, keyboard nav, try-engine-move, summary card) | Turns analyzer into coach. Pure UI on existing data. |
| 12 | Puzzle-from-your-blunders drill mode | Makes the dashboard actionable. Closes the practice loop. |
| 13 | Play-mode ergonomics (promotion dialog, play black, takeback, hint, play again) | Removes friction from the coaching-by-playing path. |

### Tier 2 — High value, more scope

| Slice | What | Why |
|-------|------|-----|
| 14 | Opening detection + ECO dashboard drilldown | First personalization signal that generalizes across games. |
| 15 | Richer motif heuristics (pin, fork, overloaded defender, discovered attack) | More precise coaching vocabulary; better drill categorization. |

### Tier 3 — After Tier 2 (ML/infra work)

| Slice | What | Why |
|-------|------|-----|
| 16 | Real Lichess training run for learned classifier | ~10k examples needed for F1 > 0 on held-out data. |
| 17 | Explanation quality upgrade (longer motif-specific templates, multi-turn) | Current 70-word single-response is good but limited for complex positions. |

### What NOT to build (deferred past Tier 3)

- **Multi-user auth.** Single local user, hardcoded ID. Do not add auth until the product is proven.
- **Cloud deployment.** `make serve` is the target. No Docker, no Heroku, no managed databases.
- **Puzzle generation from external sources.** Only puzzles from the user's own games. External puzzle DBs are a different product.
- **Style similarity / opponent prep.** Interesting but far outside MVP scope.
- **Large from-scratch classifier.** The SmallMotifNet (~1–5M params) is the right size. Do not train a transformer on chess positions for this use case.

---

## Slice 11 — Interactive Review

**Status:** Not started.

### Goal

Convert the review screen from a passive annotator into an active teaching tool. After this slice, the user can:

1. Navigate moves with ← → arrow keys; ↑ ↓ jumps to previous/next flagged move.
2. Click `Try engine's move` on any flagged move to enter a free-play mode on the board where they can explore the engine's best continuation themselves.
3. Step through the engine's PV line on the board (click `Show best line`, then ← → to advance through the continuation).
4. See a summary card at the top of the review with the accuracy score, top 2 blunders, and most common motif.
5. See `MotifEvidence` details in the move panel (attacker/defender piece lists, opponent reply square).

### What is NOT in scope for Slice 11

- New API endpoints (all data is already in the review response).
- New motifs or classifier changes.
- Keyboard nav on the play screen.
- Any ML work.

### Technical approach

**API:** No changes needed. The `annotated-game.v1` response already includes `pv` (engine continuation), `best_move`, `eval_before_cp`, `eval_after_cp`, `motifs` (with `MotifEvidence` fields), and `move_san` for every annotated move.

The `MotifEvidence` fields are serialized in `chess_ml/api/games.py`. Check the response schema to confirm `attackers`, `defenders`, `opponent_reply`, and `related_ply` are included in the JSON response. If they are not currently serialized, add them to the `AnnotatedMoveModel` pydantic schema and ensure they are populated from `MotifEvidence`.

**Frontend — Keyboard navigation:**

In `web/components/GameReview.tsx`, add a `useEffect` that registers `keydown` handlers:
- `ArrowRight` → advance to next move (wrap at end)
- `ArrowLeft` → go to previous move (floor at 0)
- `ArrowDown` → find next flagged move (motifs.length > 0) from current index
- `ArrowUp` → find previous flagged move from current index

The selected move index is already state in `GameReview`. This is a pure state update.

**Frontend — PV board replay:**

Add a `pvMode` state to `GameReview`. When the user clicks `Show best line` on a flagged move, enter PV mode:
- Parse the PV from the current move's `pv` field (space-separated UCI move list, e.g., `"e2e4 e7e5 g1f3"`).
- Reconstruct board positions by replaying PV moves from the `fen_after` of the current move using `chess.js` (already in the web project as a chessground dependency, or add `chess.js` if not present).
- Show a secondary board or toggle the main board into PV mode with its own step index.
- ← → in PV mode steps through PV positions (not game moves).
- Press `Esc` or click `Back to game` to exit PV mode.

The simplest implementation: a single `pvStep` integer and a `pvPositions: string[]` array derived from the selected move's PV. Render the main board from `pvPositions[pvStep]` when in PV mode.

**Frontend — Try engine's move:**

Add a `tryMode` state. When the user clicks `Try engine's move`:
- Enter free-play mode on the board from `fen_after` of the flagged move.
- Enable `legalDests` by calling a new local chess logic helper (not API) to compute legal moves from the FEN.
- When the user makes a move, show a comparison: did they find the engine's best move? Show green/yellow/red feedback chip.
- This is local-only chess logic — no API call needed. Use `chess.js` to compute legal moves.

**Frontend — Summary card:**

At the top of `GameReview`, before the move list, render a `ReviewSummary` component that receives the full annotated move list and computes:
- **Accuracy**: `(1 - total_cp_loss / max_possible_loss) * 100`, clamped to 0–100. Or simpler: percent of moves with no motif.
- **Top 2 blunders**: moves sorted by `loss_cp` descending, show move number + motif name.
- **Most common motif**: group motif_occurrences across all moves, take the max.

This is a pure client-side computation from the existing response data. No API changes.

**Frontend — MotifEvidence in panel:**

In `CurrentMovePanel`, expand the motif chip section to show:
- Attacker/defender piece lists (e.g., "Rook on d1 attacks undefended Queen on d7").
- `opponent_reply` square if present ("Black played …Rxd7").
- These come from the serialized `MotifEvidence` fields in the annotated move.

### Files to modify

- `web/components/GameReview.tsx` — keyboard nav, PV mode state, try-engine-move state, summary card, evidence panel
- `web/app/page.tsx` — pass `autoFocus` to GameReview or handle focus for keyboard events
- `chess_ml/api/games.py` — verify `MotifEvidence` fields are serialized in `AnnotatedMoveModel`
- `chess_ml/classifier/motifs.py` — verify `MotifEvidence` field names match what's serialized
- Possibly add `chess.js` to `web/package.json` if not already present

### Done when

1. Arrow-key navigation works on the review screen (no mouse required after initial load).
2. Clicking `Show best line` on a flagged move steps the board through the engine's PV.
3. `Try engine's move` prompts the user to find the best move and gives feedback.
4. Summary card shows accuracy %, top 2 blunders, most common motif.
5. MotifEvidence details (attacker/defender) visible in the move panel for tagged moves.
6. `make check` passes. `make e2e` passes (extend one Playwright test to assert keyboard nav works).

### Risks

- `chess.js` legal move computation may conflict with chessground's internal state. Use chess.js only for computing `legalDests` (a Map), then pass it to the Board component — don't let chess.js and chessground both own board state.
- PV move parsing from UCI strings (`e2e4`) requires handling promotions (e.g., `e7e8q`). Handle the 5-character case.
- The `pvPositions` array needs to be recomputed whenever the selected move changes. Use `useMemo` with the selected move index as the dependency.

---

## Slice 12 — Puzzle-from-Your-Blunders Drill Mode

**Status:** Not started. Depends on Slice 11 being complete (or can be built in parallel if team size allows).

### Goal

Make the dashboard actionable. After this slice, the user can:

1. Click `Drill this motif` on any motif in the dashboard to enter a drill screen (`/train`).
2. The drill screen shows a position from one of their own flagged games (the FEN just before the blunder).
3. The user makes a move on the board. They see immediate feedback: correct (found engine's best) / close (within 50cp of best) / incorrect.
4. After feedback, they see the full coach explanation for that position.
5. Positions are served in a simple Leitner-style spaced repetition order: incorrect positions come back sooner.
6. Progress is persisted in SQLite.

### What is NOT in scope for Slice 12

- External puzzle databases.
- Adaptive difficulty beyond basic Leitner-box scheduling.
- Timed drills or ratings.
- Any new Stockfish analysis (all positions are already analyzed in the profile store).

### Technical approach

**API — New endpoints:**

```
GET  /api/train/next?motif={motif_name}   → next drill position for this motif
POST /api/train/result                    → record attempt result
GET  /api/train/stats                     → drill progress summary
```

`GET /api/train/next` queries `profile_motif_occurrences` for positions with the given motif, joins to the game's annotated data for FEN + best_move + motif_evidence, and applies Leitner scheduling (due_at timestamp or attempt count). Returns: `{drill_id, fen, motif, hint_text, best_move_uci}`. Do NOT return `best_move_uci` to the frontend until after the user submits — gate it on the POST.

**Database — New table:**

```sql
CREATE TABLE drill_attempts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id     TEXT NOT NULL,
    ply         INTEGER NOT NULL,
    motif       TEXT NOT NULL,
    correct     INTEGER NOT NULL,   -- 1 = correct, 0 = incorrect
    cp_loss     INTEGER,            -- how far from best move
    attempted_at TEXT NOT NULL,
    next_due_at  TEXT              -- ISO8601, null = due now
);
```

Leitner logic: if correct, `next_due_at = now + 2^attempt_count days`. If incorrect, `next_due_at = now + 1 hour`. Simple and correct.

**Frontend — `/train` screen:**

New `web/app/train/page.tsx`. Layout: board on left, drill info on right.

- On load, fetch the first drill position (motif from query param or "any").
- Show the position on the board. Player is always the side to move (board orientation = side_to_move from FEN).
- On user move: POST result, reveal feedback (correct/close/incorrect with explanation text).
- `Next puzzle` button fetches the next drill position.

The board in drill mode uses the same `Board` component with `legalDests` computed locally (chess.js).

**Dashboard integration:**

On `web/app/dashboard/page.tsx`, each motif row in the motif bar chart gets a `Drill →` link that navigates to `/train?motif={motif_name}`.

### Done when

1. `GET /api/train/next` returns a position from the user's blunders.
2. Drill screen shows a board and accepts a move.
3. Feedback is immediate and correct (engine's best move is the gold standard).
4. Leitner scheduling persists attempt history in SQLite.
5. Dashboard motif rows link to the drill screen.
6. `make check` passes.

---

## Slice 13 — Play-Mode Ergonomics

**Status:** Not started. Can be built independently of Slices 11/12.

### Goal

Remove the remaining friction from the play loop. After this slice:

1. Pawn promotion shows an in-board overlay dialog (not `window.prompt()`).
2. The user can play as black (board flips, Maia plays white).
3. Takeback button (one ply, both user and Maia move undone, costs one takeback per game).
4. Hint button (shows the engine's best move as an arrow on the board, costs one hint per game).
5. `Play again` button in post-game review navigates back to `/play` with the same settings.

### What is NOT in scope for Slice 13

- Takeback count persistence across sessions.
- Multiple takebacks per move.
- Any classifier or explanation changes.

### Technical approach

**Promotion dialog:**

In `web/app/play/page.tsx`, replace `window.prompt()` with a React overlay component. When a pawn reaches the back rank, render a `PromotionDialog` with four piece buttons (Queen, Rook, Bishop, Knight). The overlay appears over the board, positioned at the promotion square. Selecting a piece resolves a pending `promotionResolve` ref and submits the move.

**Play as black:**

In `POST /api/play/new`, add an optional `user_color: "white" | "black"` field (default `"white"`). If `"black"`, the session stores `user_color = "black"` and Maia/Stockfish makes the first move immediately after `new`. The API's `POST /api/play/move` already handles the white/black distinction at the protocol level.

Frontend: add a color selector (White / Black / Random) to the start screen. Pass `user_color` in the new-game request. Set board orientation accordingly.

**Takeback:**

New API endpoint: `POST /api/play/takeback`. Removes the last two half-moves (user + opponent) from session history and returns the updated position. `PlaySession` in `chess_ml/play/session.py` needs a `takeback()` method. Limit: `max_takebacks_per_game = 1` (configurable, stored in session). Return 409 if no moves to take back or limit exceeded.

Frontend: `Takeback` button, disabled after 1 use or when no user move exists. Shows remaining takebacks.

**Hint:**

New API endpoint: `GET /api/play/hint?session_id={id}`. Calls Stockfish on the current FEN (depth 14, short timeout) and returns `{best_move_uci: "e2e4"}`. Limit: `max_hints_per_game = 3` (stored in session). Frontend renders the hint as a chessground arrow (`shapes` prop). Arrow disappears after the user makes their move.

**Play again:**

In the post-game review screen (`web/app/play/page.tsx`), add a `Play again` button next to `Review game` that resets state to the start-game form with the same opponent settings.

### Files to modify

- `web/app/play/page.tsx` — promotion dialog, color selector, takeback/hint buttons, play again
- `chess_ml/api/play.py` — `user_color` param in new-game, takeback endpoint, hint endpoint
- `chess_ml/play/session.py` — `user_color`, `takeback()`, `hints_used`, `takebacks_used`
- `chess_ml/engine/stockfish.py` — hint uses the existing pool, just a short-depth eval call

### Done when

1. Promotion uses an in-board overlay, not `window.prompt()`. Playwright test for promotion works.
2. User can start a game as black; Maia plays the first move.
3. Takeback undoes the last user move and Maia response; button disabled after 1 use.
4. Hint shows an arrow for the engine's best move; limited to 3 per game.
5. `Play again` button returns to start screen with same settings.
6. `make check && make e2e` passes.

---

## Slice 14 — Opening Detection + ECO Dashboard Drilldown

**Status:** Not started. Depends on Slice 11 being complete (dashboard link target).

### Goal

Surface which openings the user plays and where they go wrong. After this slice:

1. Each analyzed game is tagged with ECO code + opening name (e.g., `B20 — Sicilian Defence`).
2. The dashboard has an `Openings` tab showing: opening name, games played, average accuracy, most common motif per opening.
3. Clicking an opening filters the recent games list to show only games with that opening.
4. The review screen shows the opening name in the header once it diverges from theory (move where eval first drops below −0.3 or first inaccuracy).

### What is NOT in scope for Slice 14

- Opening preparation / repertoire building.
- Transposition detection.
- Any external opening books beyond the bundled ECO table.

### Technical approach

**ECO tagging:**

Bundle a compact ECO lookup table as a JSON file in the repo (`data/eco.json`, ~600KB). Format: `{moves_uci: string[], eco: string, name: string}[]` sorted by move sequence length descending. To tag a game, replay the first 20 plies and at each ply find the longest matching sequence in the table.

`python-chess` can replay moves from PGN to produce UCI sequences. Add `chess_ml/classifier/openings.py` with `detect_opening(moves_uci: list[str]) → OpeningTag | None`.

Wire into `POST /api/games`: run opening detection on the parsed move sequence, store `eco_code` and `opening_name` in `profile_games`. Add these columns to the existing migration.

**Dashboard openings tab:**

`GET /api/profile/me` returns `profile-dashboard.v1`. Extend it to include an `openings` array: `{eco: string, name: string, games: int, avg_loss_cp: float, top_motif: string | null}[]`. Compute from `profile_games JOIN profile_motif_occurrences`.

Frontend: add an `Openings` tab to the dashboard. Show a table with sortable columns (games, avg loss, top motif).

**Review header:**

In `GameReview.tsx`, if the annotated game response includes `opening_name`, show it in the review header beneath the player names.

### Done when

1. Uploaded games show ECO code and opening name in the review header.
2. Dashboard `Openings` tab shows games-by-opening with accuracy and top motif.
3. Clicking an opening filters the recent games list.
4. `make check` passes. Demo seeds include at least 2 different openings.

---

## Slice 15 — Richer Motif Heuristics

**Status:** Not started. Can be built independently (classifier layer only, no frontend changes required beyond new chip labels).

### Goal

Add four new motif detectors to the heuristic classifier, giving the coaching vocabulary more precision:

1. **Pin** — a piece is pinned to the king or a higher-value piece (cannot move without exposing attack).
2. **Fork** — a single piece attacks two or more opponent pieces simultaneously (eval delta confirms it was a mistake to allow/miss it).
3. **Overloaded defender** — a piece is defending two or more pieces simultaneously; capturing one removes the defense of the other.
4. **Discovered attack** — moving one piece reveals an attack from a piece behind it.

### What is NOT in scope for Slice 15

- Training the learned classifier on these new labels (that is Slice 16).
- Any frontend changes beyond adding the new motif chip colors/labels.
- Skewer detection (similar to pin, defer).

### Technical approach

All detectors live in `chess_ml/classifier/motifs.py`. They follow the same pattern as existing detectors: they receive `(board_before, board_after, move, engine_eval_before, engine_eval_after)` and return `MotifEvidence | None`.

**Pin detector:**
Use `python-chess`'s `board.is_pinned(color, square)` method. A move "missed" a pin tactic if: after the opponent's best reply (from `opponent_reply` in MotifEvidence), a piece of the user's color becomes pinned to their king and the eval dropped ≥ 200cp. A move "allowed" a pin if the opponent's move creates a pin on a user piece.

**Fork detector:**
After `board_after`, check if any opponent piece attacks 2+ user pieces of higher or equal value. If the fork resulted from the user's move (they moved a piece that is now forked, or they failed to capture a forking piece), and eval delta ≥ 200cp.

**Overloaded defender:**
For each user piece after `board_after`, count how many user pieces it defends. If it defends ≥ 2 and one of those pieces is captured on the next move (opponent's best reply), tag as overloaded_defender.

**Discovered attack:**
Compare the attack map before and after the move. If moving a piece reveals that a previously blocked piece now attacks a valuable target (and that attack was not defended), tag as discovered_attack. Use `python-chess` attack maps.

**Label order and config:**

Add the 4 new motifs to `chess_ml/classifier/config.py`'s `LABEL_ORDER`. They go after the existing 5 so the existing checkpoint stays compatible. New labels are heuristic-only until Slice 16.

**Test fixtures:**

Add hand-curated FEN positions for each new motif to `tests/fixtures/`. Each should have a clear human-verifiable correct answer. Minimum 2 positions per motif.

### Done when

1. All 4 new detectors pass their fixture tests.
2. Uploading the `demo/missed-tactic.pgn` fixture still produces the same existing motif tags (regression test).
3. `LABEL_ORDER` updated in config.
4. Frontend motif chips show the new labels with appropriate colors.
5. `make check` passes.

---

## Implementation Order

```
Slice 11 (Interactive review) ← START HERE
    ↓ can parallelize ↓
Slice 12 (Drill mode)          Slice 13 (Play ergonomics)    Slice 15 (Richer motifs)
    ↓
Slice 14 (Openings)
    ↓
Slice 16 (Lichess training run — real F1 > 0 on learned classifier)
    ↓
Slice 17 (Explanation quality upgrade)
```

Slices 12, 13, and 15 have no dependencies on each other and can be implemented in any order after Slice 11 is done. Slice 14 depends on Slice 11's dashboard extension.

---

## Key Files Reference

Codex should read these files before implementing any slice in this roadmap:

| File | Why |
|------|-----|
| `CLAUDE.md` | Hard constraints (no image inputs, grounding rules, locked stack) |
| `docs/plans/001-mvp.md` | Full architecture, slice history, data flow |
| `chess_ml/api/games.py` | Review pipeline, `AnnotatedGameModel` schema |
| `chess_ml/classifier/motifs.py` | `MotifEvidence` dataclass, all heuristic detectors |
| `chess_ml/classifier/config.py` | `LABEL_ORDER`, thresholds |
| `chess_ml/explanation/prompt.py` | Grounding validation, fallback text |
| `web/components/GameReview.tsx` | Review UI — the main target for Slice 11 |
| `web/app/play/page.tsx` | Play loop — the main target for Slice 13 |
| `web/e2e/mvp.spec.ts` | Existing Playwright tests — do not regress |
| `chess_ml/play/session.py` | In-memory play session — extend for Slice 13 |
| `chess_ml/profile/store.py` | SQLite profile tables — extend for Slice 12/14 |

---

## Hard Constraints Reminder

These apply to every slice in this roadmap:

1. **No image inputs.** PGN, FEN, or board interaction only. Never add screenshot upload, OCR, or image analysis.
2. **Explanations never contradict the engine.** Every coach note must pass `validate_provider_response()`. Fallback text is always Stockfish-grounded, never invented.
3. **`make check` must pass** before any slice is declared done. Ruff + mypy strict + pytest all green.
4. **`make e2e` must pass.** The 5 existing Playwright tests must not regress. Add new Playwright tests for each slice's golden path.
5. **Reproducible ML.** Any new training or eval in Slice 16 must have a fixed seed, checked-in config, and recorded eval report in `docs/evals/`.
