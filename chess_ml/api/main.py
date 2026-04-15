"""FastAPI application entrypoint.

Slice 0: only exposes /health. Subsequent slices add /api/games, /api/play,
/api/profile under this app.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chess_ml import __version__

app = FastAPI(
    title="chess_ml",
    version=__version__,
    description="Personalized chess coaching API",
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
