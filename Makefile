.PHONY: help setup setup-maia check lint typecheck test serve serve-api serve-web ingest train demo e2e clean

PYTHON := uv run python
UVICORN := uv run uvicorn

help:
	@echo "Targets:"
	@echo "  setup    - install Python + npm dependencies"
	@echo "  setup-maia - download Maia weights for local play"
	@echo "  check    - ruff + mypy + pytest (must pass before commit)"
	@echo "  serve    - boot API (8000) and web (3000) concurrently"
	@echo "  ingest   - build the Slice 16 weak-labeled Lichess classifier dataset"
	@echo "  train    - train the Slice 16 learned weakness classifier"
	@echo "  demo     - seed three Slice 9 sample games into the local profile DB"
	@echo "  e2e      - run the Playwright MVP end-to-end suite"
	@echo "  clean    - remove caches and build artifacts"

setup:
	uv sync
	cd web && npm install
	./scripts/fetch_maia_weights.sh

setup-maia:
	./scripts/fetch_maia_weights.sh

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
	$(PYTHON) -m chess_ml.ingestion.lichess

train: ingest
	$(PYTHON) -m chess_ml.classifier.train

demo:
	$(PYTHON) -m chess_ml.profile.demo

e2e:
	cd web && npm run e2e

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf web/.next
