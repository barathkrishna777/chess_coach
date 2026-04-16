# chess_ml

Personalized chess coach: weakness classifier + LLM explanation layer + profile store, centered on an interactive chessboard. MVP runs locally.

See [CLAUDE.md](CLAUDE.md) for project conventions and [docs/plans/001-mvp.md](docs/plans/001-mvp.md) for the full plan.

## Prerequisites

- macOS or Linux
- [uv](https://github.com/astral-sh/uv) for Python
- Node.js 20+
- Stockfish: `brew install stockfish`
- Optional Maia play opponent: `brew install lc0`
- Optional local explanations: [Ollama](https://ollama.com), then
  `ollama pull qwen3:8b`

## Setup

```bash
make setup
```

This installs Python dependencies via uv (using Python 3.11), installs npm
dependencies for the web app, and downloads Maia 1100/1500/1900 weights into
`checkpoints/maia/`. The weights are gitignored.

For the core PGN review, seeded dashboard, and Stockfish fallback play demo, only
Stockfish is required after dependencies are installed. Maia and Ollama improve
the local experience, but they are not required for `make demo`.

If you only need to refresh Maia weights later:

```bash
make setup-maia
```

## Fresh-Clone Demo

Seed a local profile database with three checked-in Stockfish-analyzed games:

```bash
make demo
```

`make demo` writes to `data/chess_ml.sqlite3` by default, or to
`CHESS_ML_DB_PATH` when that environment variable is set. It is idempotent:
rerunning it refreshes the same three demo games instead of duplicating profile
rows. The seeder requires Stockfish because the profile data must be grounded in
engine analysis; it does not require Maia weights or Ollama.

Then run the app:

```bash
make serve
```

Boots the FastAPI backend on http://localhost:8000 and the Next.js frontend on http://localhost:3000. Open the frontend URL to see the chessboard.

Slice 3 explanations are generated on demand for a selected flagged move and default to
a local open-source model through Ollama at http://localhost:11434. If Ollama or the
configured model is not running, review still works and the coach panel shows a setup
note instead of blocking PGN analysis.

## Local play opponents

`/play` defaults to `auto` mode with Maia 1500. If `lc0` or the selected Maia
weight is missing, the app falls back to the existing low-strength Stockfish
opponent and labels that fallback in the UI.

Configuration:

- `CHESS_ML_LC0_PATH`: Lc0 executable path. Default finds `lc0` on `PATH`.
- `CHESS_ML_MAIA_WEIGHTS_DIR`: Maia weights directory. Default `checkpoints/maia`.
- `CHESS_ML_PLAY_STOCKFISH_PATH`: optional Stockfish path for fallback play.
- `CHESS_ML_PLAY_STOCKFISH_ELO`: fallback Stockfish Elo. Default `1350`.

Maia is never used for review analysis or coach-note grounding. It only chooses
the black moves during local play; finished games still go through Stockfish for
evaluation, motifs, and explanations.

## Local coach notes

Coach notes are intentionally lazy: the app asks for one only when you click
`Generate coach note` on a flagged move. Stockfish and the motif classifier provide the
grounding facts; the LLM only turns those facts into short teaching text. Timed-out
or failed provider calls are not cached. Untrusted model responses are rejected and
not cached; the UI may show a deterministic Stockfish-grounded fallback instead.

Configuration:

- `CHESS_ML_EXPLANATION_PROVIDER`: `auto`, `ollama`, `anthropic`, `codex`, or
  `disabled`. Default `auto` uses local Ollama.
- `CHESS_ML_EXPLANATION_TIMEOUT_SECONDS`: provider timeout budget. Default `45`, which
  gives local Ollama enough room for a cold model load plus one short JSON response.
- `CHESS_ML_OLLAMA_BASE_URL`: local Ollama base URL. Default `http://localhost:11434`.
- `CHESS_ML_OLLAMA_MODEL`: local model name. Default `qwen3:8b`.
- `CHESS_ML_DB_PATH`: SQLite path for the explanation cache and local profile store. Default
  `data/chess_ml.sqlite3`.

For local explanations:

```bash
ollama pull qwen3:8b
ollama serve
```

## Learned classifier

Slice 16 adds an optional local PyTorch classifier on top of the heuristic motif
baseline using a reproducible real Lichess training run. Heuristics always run.
If no compatible checkpoint exists at `checkpoints/classifier/slice16-lichess-v1.pt`,
review stays on heuristic v0 and `make serve` continues to work on a fresh clone.

The default config is `configs/classifier/slice16-lichess-v1.toml`. It downloads
the configured Lichess archive only when the ignored raw file is missing, builds
the ignored parquet dataset, and trains the ignored local checkpoint:

```bash
make ingest
make train
```

`make ingest` writes weak-labeled parquet examples to
`data/processed/slice16-lichess-v1.parquet`. `make train` refreshes the
gitignored checkpoint at `checkpoints/classifier/slice16-lichess-v1.pt` and writes
the reproducible eval report at `docs/evals/016-lichess-classifier-v1.json`.

The original Slice 8 fixture smoke config remains available at
`configs/classifier/slice8-v1.toml` for quick local classifier-path checks.

Configuration:

- `CHESS_ML_CLASSIFIER_CONFIG`: optional config path. Default
  `configs/classifier/slice16-lichess-v1.toml`.
- `CHESS_ML_CLASSIFIER_CHECKPOINT`: optional checkpoint path. Default comes from the
  config. Set it to an empty string to force heuristic-only classification.

The learned classifier never performs chess analysis. Stockfish still supplies evals,
PVs, and best moves; the model only contributes motif probabilities that must be
grounded in those facts before appearing in review.

## Check

```bash
make check
```

Runs ruff, mypy, and pytest. Must pass before any commit.

Frontend verification:

```bash
cd web && npm run typecheck
cd web && npm run build
```

End-to-end verification:

```bash
make e2e
```

The Playwright suite starts FastAPI and Next.js locally, seeds a temporary demo
database, disables coach providers and learned checkpoint loading, and forces the
Maia-unavailable path so auto play uses the Stockfish fallback. If Playwright has
not installed a browser yet, run `cd web && npx playwright install chromium`.

## Status

Slices 0-9 are implemented through the learned classifier v1 path, with the Slice
10 coach-grounding repair also in place. Current app supports PGN review,
Stockfish analysis, heuristic and optional learned motifs, lazy grounded coach
notes, local play against Maia with Stockfish fallback, post-game review,
`/dashboard` profile aggregation, `make demo`, and Playwright MVP coverage. See
[docs/plans/001-mvp.md](docs/plans/001-mvp.md) for the updated slice map.
