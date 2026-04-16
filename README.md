# chess_ml

Personalized chess coach: weakness classifier + LLM explanation layer + profile store, centered on an interactive chessboard. MVP runs locally.

See [CLAUDE.md](CLAUDE.md) for project conventions and [docs/plans/001-mvp.md](docs/plans/001-mvp.md) for the full plan.

## Prerequisites

- macOS or Linux
- [uv](https://github.com/astral-sh/uv) for Python
- Node.js 20+
- Stockfish: `brew install stockfish`
- Maia play opponent: `brew install lc0`
- Optional local explanations: [Ollama](https://ollama.com), then
  `ollama pull qwen3:8b`

## Setup

```bash
make setup
```

This installs Python dependencies via uv (using Python 3.11), installs npm
dependencies for the web app, and downloads Maia 1100/1500/1900 weights into
`checkpoints/maia/`. The weights are gitignored.

If you only need to refresh Maia weights later:

```bash
make setup-maia
```

## Run

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
grounding facts; the LLM only turns those facts into short teaching text. Failed,
timed-out, or untrusted model responses are not cached and never become fallback chess
advice.

Configuration:

- `CHESS_ML_EXPLANATION_PROVIDER`: `auto`, `ollama`, `anthropic`, `codex`, or
  `disabled`. Default `auto` uses local Ollama.
- `CHESS_ML_EXPLANATION_TIMEOUT_SECONDS`: provider timeout budget. Default `15`.
- `CHESS_ML_OLLAMA_BASE_URL`: local Ollama base URL. Default `http://localhost:11434`.
- `CHESS_ML_OLLAMA_MODEL`: local model name. Default `qwen3:8b`.
- `CHESS_ML_DB_PATH`: SQLite path for the explanation cache and local profile store. Default
  `data/chess_ml.sqlite3`.

For local explanations:

```bash
ollama pull qwen3:8b
ollama serve
```

## Check

```bash
make check
```

Runs ruff, mypy, and pytest. Must pass before any commit.

## Status

Slices 0-6 are implemented through the local profile dashboard. Current app supports
PGN review, Stockfish analysis, heuristic motifs, lazy grounded coach notes, local play
against Maia with Stockfish fallback, post-game review, and `/dashboard` profile aggregation.
See [docs/plans/001-mvp.md](docs/plans/001-mvp.md) for the updated slice map.
