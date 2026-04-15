"""Smoke test: /health endpoint responds 200 with status=ok."""

from fastapi.testclient import TestClient

from chess_ml import __version__
from chess_ml.api.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
