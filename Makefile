.PHONY: help setup check lint typecheck test serve serve-api serve-web ingest train demo clean

PYTHON := uv run python
UVICORN := uv run uvicorn

help:
	@echo "Targets:"
	@echo "  setup    - install Python + npm dependencies"
	@echo "  check    - ruff + mypy + pytest (must pass before commit)"
	@echo "  serve    - boot API (8000) and web (3000) concurrently"
	@echo "  ingest   - download Lichess slice and build dataset (Slice 6)"
	@echo "  train    - train the weakness classifier (Slice 6)"
	@echo "  demo     - seed sample games for demo (Slice 7)"
	@echo "  clean    - remove caches and build artifacts"

setup:
	uv sync
	cd web && npm install

check: lint typecheck test

lint:
	uv run ruff check chess_ml tests
	uv run ruff format --check chess_ml tests

typecheck:
	uv run mypy chess_ml

test:
	uv run pytest

serve:
	@echo "Starting API on :8000 and web on :3000 (Ctrl-C to stop both)"
	@trap 'kill 0' INT TERM EXIT; \
	$(MAKE) serve-api & \
	$(MAKE) serve-web & \
	wait

serve-api:
	$(UVICORN) chess_ml.api.main:app --reload --port 8000

serve-web:
	cd web && npm run dev

ingest:
	@echo "Slice 6 — not yet implemented"
	@exit 1

train:
	@echo "Slice 6 — not yet implemented"
	@exit 1

demo:
	@echo "Slice 7 — not yet implemented"
	@exit 1

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf web/.next
