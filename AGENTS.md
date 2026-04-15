# Chess ML Coaching Product

A locally-runnable chess coaching product: personalized weakness classifier + LLM explanation layer + profile store, centered on an interactive chessboard. Target user: 1200–2000 rated club players.

The canonical long-form plan lives at [docs/plans/001-mvp.md](docs/plans/001-mvp.md). Read it before starting non-trivial work.

## Hard constraints (non-negotiable)

1. **No image inputs anywhere.** All inputs are plain-text PGN, FEN strings, or direct interaction with the live chessground board. Never add screenshot upload, OCR, computer vision, or any image analysis to this product.
2. **Explanations never contradict the engine.** The classifier detects motifs; Stockfish produces ground-truth PV and eval; Codex only paraphrases and teaches. Any explanation that contradicts the engine's best line is a bug — fix it, don't ship it.
3. **Reproducible ML.** Every training run has a fixed seed, a checked-in config file, and a recorded eval result. No ad-hoc "I'll re-run it later" experiments.
4. **Locally runnable end-to-end.** `make serve` must boot the full product. The only external dependency is the Codex API for the explanation layer.
5. **Thin vertical slices, not horizontal layers.** Every slice ends with a demoable `make serve` state. No "I'll come back to this later" stubs that break the user-facing flow.

## Stack (locked decisions — do not re-litigate)

- **Python 3.11+** with **uv** for package management.
- **PyTorch** for ML training and inference.
- **FastAPI** for the HTTP API, **SQLite** for the profile store (MVP).
- **python-chess** for PGN parsing and board state.
- **Stockfish** binary for ground-truth evaluation (installed via `brew install stockfish`).
- **Maia** (via Lc0) for human-like play — see Slice 5 in the plan.
- **Next.js (App Router)** + **TypeScript** + **Tailwind** + **chessground** for the frontend.
- **Codex API** (`Codex-opus-4-6` primary, `Codex-sonnet-4-6` fallback for cheap moves) for explanations, with prompt caching on the system prompt.

## Local run

- `make setup` — one-time: install Python deps via uv, install npm deps, fetch any required weights.
- `make check` — ruff + mypy + pytest. Must pass before declaring any task done.
- `make serve` — boots FastAPI (port 8000) and Next.js dev (port 3000). Open http://localhost:3000.
- `make ingest` — downloads a Lichess slice and builds the processed dataset (Slice 6).
- `make train` — trains the weakness classifier (Slice 6).
- `make demo` — seeds the DB with sample games so the product is immediately usable on a fresh clone (Slice 7).

## Directory layout

```
chess_ml/
├── AGENTS.md                    # this file
├── Makefile                     # ingest/train/serve/check/demo
├── pyproject.toml               # uv-managed Python project
├── docs/plans/                  # numbered long-form plans (checked in)
├── data/                        # raw/ processed/ puzzles/  (gitignored contents)
├── chess_ml/                    # the Python package
│   ├── api/                     # FastAPI routes
│   ├── engine/                  # Stockfish + Maia wrappers
│   ├── ingestion/               # PGN → labeled positions
│   ├── classifier/              # weakness classifier
│   ├── explanation/             # LLM prompting + grounding
│   └── profile/                 # SQLite profile store
├── web/                         # Next.js frontend
├── tests/                       # pytest unit + e2e (playwright)
├── notebooks/                   # exploratory only, never source of truth
└── checkpoints/                 # model weights (gitignored)
```

## Conventions

- **Python:** ruff for lint + format (line length 100), mypy in strict mode for the `chess_ml` package. Type-annotate everything in the package; tests can be looser.
- **Frontend:** prettier defaults, TypeScript strict mode. No `any` in checked-in code.
- **Tests:** pytest for Python, playwright for the full end-to-end path (once we have it). Unit tests live next to the module they test (`tests/unit/<module>/`).
- **Data files:** raw Lichess dumps and trained checkpoints are gitignored. Datasets are stored as parquet.
- **Secrets:** `ANTHROPIC_API_KEY` via `.env` (gitignored). Never commit keys.
- **Commits:** create new commits rather than amending. Small, focused commits preferred over big batches.

## Model and reasoning guidance for sessions

- Use **Opus 4.6** for planning, architecture design, prompt engineering (Slice 3), classifier design (Slice 6), and hard debugging.
- Use **Sonnet 4.6** for bulk implementation of well-defined slices where the plan is already clear.
- When in plan mode, always commit the resulting plan to `docs/plans/NNN-xxx.md` before implementing against it.
- Run `make check` before marking any task complete. "It compiles" is not "it works."

## Key external resources

- Lichess open database: https://database.lichess.org
- Lichess puzzle database: https://database.lichess.org/#puzzles
- Maia Chess weights + code: https://github.com/CSSLab/maia-chess
- Stockfish: https://stockfishchess.org/download/
- chessground (board library): https://github.com/lichess-org/chessground
- python-chess docs: https://python-chess.readthedocs.io

## What NOT to do

- Do not add image upload, OCR, or computer vision anywhere. (See hard constraint 1.)
- Do not train a large from-scratch model as part of MVP. The Slice 6 classifier is deliberately small (~1–5M params).
- Do not skip `make check` to "save time." A broken main branch stops everything.
- Do not generate explanations without passing through the grounding pipeline (FEN + engine PV + motif).
- Do not add cloud deployment, auth, payments, or sharing until MVP is demoable locally.
