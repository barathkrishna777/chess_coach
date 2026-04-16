# Plan 015 - Richer Motif Heuristics

Status: **approved for implementation**
Owner: barathkrishna
Last updated: 2026-04-16

## Summary

Add four heuristic-only motif labels: `pin`, `fork`, `overloaded_defender`, and
`discovered_attack`. These extend the existing Stockfish-gated motif classifier
and surface through the existing review, dashboard, drill, and explanation paths
without new API endpoints, ML training, auth, cloud, sharing, or image input.

## Key Changes

- Extend motif contracts in `chess_ml/classifier/motifs.py`,
  `chess_ml/api/games.py`, and `web/lib/types.ts` with the four new motif IDs
  and labels.
- Add `TACTICAL_PATTERN_THRESHOLD_CP = 200`; all new detectors require
  engine-confirmed loss at or above that threshold.
- Implement detectors in `chess_ml/classifier/motifs.py` using `python-chess`
  board state only:
  - `pin`: new absolute or relative pin after Stockfish's best continuation.
  - `fork`: one attacker newly attacks two or more equal-or-higher-value targets.
  - `overloaded_defender`: one defender protects multiple targets and
    Stockfish's line exploits that workload.
  - `discovered_attack`: a candidate move opens a line attack from a sliding
    piece onto a valuable target.
- Keep existing broad labels (`missed_tactic`, `allowed_tactic`, `endgame_slip`)
  intact; new motifs are additional vocabulary, not replacements.
- Reuse current `MotifEvidence` fields: `piece` is the key
  attacking/pinned/defending piece; `attackers` and `defenders` carry compact
  square-piece refs for targets or tactical context.
- Append new labels after the existing five in `LABEL_ORDER` and config
  thresholds.
- Preserve old Slice 8 checkpoint compatibility by letting learned checkpoints
  predict only their saved label prefix; new labels remain heuristic-only until
  Slice 16.

## UI And Explanation

- Frontend motif chips can keep severity-based colors; only the TypeScript union
  needs the new IDs.
- Dashboard and drill mode should work automatically because motif IDs/labels
  are stored as strings.
- Add prompt rank/fallback lesson entries for the new motifs, keeping broad
  tactic motifs ranked higher so Stockfish line selection remains grounded.

## Test Plan

- Add at least two unit fixtures per new motif in `tests/unit/test_motifs.py` or
  `tests/fixtures/motifs/`.
- Add regression coverage proving existing demo motif tags still appear for
  `tests/fixtures/demo/missed-tactic.pgn`.
- Add classifier config/learned-loader tests for appended-label compatibility
  with old 5-label checkpoints.
- Extend API/type tests enough to prove the new motif IDs serialize without
  schema failures.
- Required verification: `make check`, `cd web && npm run typecheck`,
  `cd web && npm run build`, and `make e2e`.

## Assumptions

- No learned model retraining happens in Slice 15.
- No schema migration is needed because motif IDs are already stored as text.
- New tactical motifs may co-exist on a move with existing motifs; that is
  expected and useful for coaching/drills.
