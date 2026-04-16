# Plan 016 - Real Lichess Training Run For Learned Classifier

Status: **implemented**
Owner: barathkrishna
Last updated: 2026-04-16

## Summary

Upgrade the learned motif classifier from a Slice 8 fixture smoke test to a
reproducible real Lichess run with roughly 10,000 weak-labeled examples,
game-level train/validation/test splits, validation-calibrated thresholds, an
ignored local checkpoint, and a checked-in eval report.

No API, frontend, explanation, auth, cloud, sharing, image input, or large
from-scratch model work is included.

Implemented eval: `docs/evals/016-lichess-classifier-v1.json` records 10,027
examples from 203 Lichess games, deterministic game-level splits, and held-out
model micro F1 of 0.3664 against the current Stockfish-gated weak labels.

## Key Changes

- Add `configs/classifier/slice16-lichess-v1.toml` as the default
  `make ingest` / `make train` config.
- Download the configured Lichess standard rated PGN archive only when the raw
  ignored file is missing; `make serve` never performs runtime data downloads.
- Stream/decompress local PGN archives, filter both players to 1200-2000 Elo,
  require rated games, cap at about 10,000 examples, and weak-label with the
  current nine-label heuristic suite.
- Split by deterministic `game_id` hash so one game's positions never cross
  train/validation/test boundaries.
- Calibrate per-label learned thresholds on validation examples and report final
  test metrics with those thresholds.
- Keep broad heuristic/Stockfish gates as runtime safety. New tactical labels
  remain heuristic-grounded until learned evidence can be trusted.

## Test Plan

- Unit-test Slice 16 config parsing, Lichess PGN filtering, target-example stop
  behavior, all nine parquet label columns, game-level split determinism, split
  isolation, threshold calibration, and report schema.
- Keep old Slice 8 five-label checkpoint prefix loading green.
- Required verification before declaring done:
  `make check`, `cd web && npm run typecheck`, `cd web && npm run build`,
  and `make e2e`.

## Assumptions

- The January 2013 Lichess standard archive is small enough for a first local
  real run while still being real Lichess data.
- The eval report records source URL, raw file hash when available, dataset row
  count, label distribution, split sizes, calibrated thresholds, and held-out
  metrics.
- Raw archives, parquet datasets, and checkpoints remain ignored artifacts.
