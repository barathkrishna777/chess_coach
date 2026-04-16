"""FastAPI application entrypoint.

Slice 0: only exposes /health. Subsequent slices add /api/games, /api/play,
/api/profile under this app.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chess_ml import __version__
from chess_ml.api.games import router as games_router
from chess_ml.api.play import router as play_router
from chess_ml.api.profile import router as profile_router
from chess_ml.api.train import router as train_router
from chess_ml.engine.opponent import PlayOpponentRegistry
from chess_ml.engine.stockfish import StockfishPool, StockfishUnavailableError
from chess_ml.explanation.service import service_from_env
from chess_ml.play.session import InMemoryPlayStore
from chess_ml.profile.store import ProfileStore


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
    """Own local engine resources for the API process."""

    fastapi_app.state.review_lock = asyncio.Lock()
    fastapi_app.state.explanation_service = service_from_env()
    fastapi_app.state.play_store = InMemoryPlayStore()
    fastapi_app.state.profile_store = ProfileStore()
    pool = StockfishPool.from_env()
    play_opponents = PlayOpponentRegistry.from_env()
    try:
        await pool.start()
    except StockfishUnavailableError as exc:
        fastapi_app.state.stockfish_pool = None
        fastapi_app.state.stockfish_error = str(exc)
    else:
        fastapi_app.state.stockfish_pool = pool
        fastapi_app.state.stockfish_error = ""

    await play_opponents.start_fallback()
    fastapi_app.state.play_opponents = play_opponents

    try:
        yield
    finally:
        await play_opponents.close()
        if pool.started:
            await pool.close()


app = FastAPI(
    title="chess_ml",
    version=__version__,
    description="Personalized chess coaching API",
    lifespan=lifespan,
)

# The Next.js dev server runs on :3000 and calls this API on :8000.
# In production we'd serve them from the same origin; for local dev we allow CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    """Response payload for /health."""

    status: str
    version: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe. Returns 200 with version info if the app is up."""
    return HealthResponse(status="ok", version=__version__)


app.include_router(games_router)
app.include_router(play_router)
app.include_router(profile_router)
app.include_router(train_router)
