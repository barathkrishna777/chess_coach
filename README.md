# chess_ml

Personalized chess coach: weakness classifier + LLM explanation layer + profile store, centered on an interactive chessboard. MVP runs locally.

See [CLAUDE.md](CLAUDE.md) for project conventions and [docs/plans/001-mvp.md](docs/plans/001-mvp.md) for the full plan.

## Prerequisites

- macOS or Linux
- [uv](https://github.com/astral-sh/uv) for Python
- Node.js 20+
- Stockfish: `brew install stockfish`
- Optional local explanations: [Ollama](https://ollama.com), then
  `ollama pull qwen3:8b`

## Setup

```bash
make setup
```

This installs Python dependencies via uv (using Python 3.11), installs npm dependencies for the web app, and leaves you ready to run.

## Run

```bash
make serve
```

Boots the FastAPI backend on http://localhost:8000 and the Next.js frontend on http://localhost:3000. Open the frontend URL to see the chessboard.

Slice 3 explanations default to a local open-source model through Ollama at
http://localhost:11434. If Ollama or the configured model is not running, review still
works and flagged moves show a setup note instead of a coaching explanation.

## Check

```bash
make check
```

Runs ruff, mypy, and pytest. Must pass before any commit.

## Status

Slice 0 (scaffold) — in progress. See [docs/plans/001-mvp.md](docs/plans/001-mvp.md).
