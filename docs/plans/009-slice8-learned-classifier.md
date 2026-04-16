# Plan 009 - Slice 8: Learned Classifier v1

Status: approved for implementation
Owner: barathkrishna
Last updated: 2026-04-16

## Summary

Add a small, reproducible PyTorch motif classifier as an optional contributor to
the existing heuristic classifier. The learned model is local-first and safe by
default: `make serve` continues to use heuristic v0 when no compatible checkpoint
exists, and heuristics remain the baseline and safety net even when a checkpoint is
available.

Stockfish remains the only source of review truth. The learned classifier predicts
motif likelihoods from Stockfish-grounded position facts; it never replaces engine
evals, PVs, or explanation grounding.

## Key Changes

- Add deterministic Slice 8 ML assets:
  - checked-in config at `configs/classifier/slice8-v1.toml`
  - checked-in eval report at `docs/evals/009-slice8-classifier-v1.json`
  - small fixture PGNs under `tests/fixtures/classifier/`
- Add ingestion utilities in `chess_ml/ingestion/lichess.py` that read local PGN
  files (`.pgn`, `.pgn.bz2`, `.pgn.zst`), parse standard games, analyze positions
  with Stockfish, weak-label motifs with heuristic v0, and write parquet examples
  to `data/processed/`.
- Add a compact PyTorch multi-label classifier using 12 piece planes, move
  from/to planes, side/castling/en-passant metadata, eval-before, and loss
  features. Training uses fixed seed, deterministic split, stable label order,
  BCE-with-logits loss, config thresholds, and gitignored checkpoints under
  `checkpoints/classifier/`.
- Wire optional ensemble inference through `chess_ml/classifier/classify.py`.
  Heuristics always run. A learned checkpoint is loaded only if present and
  metadata-compatible. Learned-only motifs may be added only with evidence that can
  be grounded from Stockfish facts; heuristic motifs are never removed.
- Update `make ingest` and `make train` to run the configured local sample path,
  and correct Slice 8 help text.

## Public Interfaces

- `Motif.source` expands from `heuristic` to `heuristic | learned | ensemble`.
- `make ingest` builds the default small local dataset unless overridden by config
  or environment.
- `make train` trains, evaluates, writes `checkpoints/classifier/slice8-v1.pt`,
  and refreshes the checked-in eval report.
- Runtime defaults:
  - checkpoint path: `checkpoints/classifier/slice8-v1.pt`
  - config path: `configs/classifier/slice8-v1.toml`
  - disable learned inference with `CHESS_ML_CLASSIFIER_CHECKPOINT=""`

## Test Plan

- Preserve existing heuristic motif tests to prove v0 behavior is unchanged.
- Add ingestion tests with fixture PGNs and deterministic fake engine evaluations.
- Add encoder/model tests for tensor shape, label order, and checkpoint metadata
  compatibility.
- Add training reproducibility smoke test with a tiny fixture dataset and fixed
  seed.
- Add eval report tests for stable baseline/model metric shape.
- Required verification:
  - `make check`
  - `cd web && npm run typecheck`
  - `cd web && npm run build` if frontend changes are significant

## Assumptions

- No model checkpoint is committed; checkpoints remain gitignored and reproducible
  via `make train`.
- The default dataset path is fixture-sized so fresh clones can exercise the slice
  without downloading a large Lichess dump.
- Real Lichess data is supported by pointing ingestion at a larger local PGN/zst
  file, not by downloading large data during normal setup.
- Learned predictions are coaching signals, not chess truth. Explanations continue
  to use only Stockfish PV/eval plus motif evidence.
