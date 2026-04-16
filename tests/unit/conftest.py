"""Shared unit-test fixtures."""

from collections.abc import Iterator

import pytest

from chess_ml.classifier.learned import learned_classifier_from_env


@pytest.fixture(autouse=True)
def disable_runtime_learned_classifier(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep unit tests on heuristic v0 unless they pass an explicit fake/model."""

    monkeypatch.setenv("CHESS_ML_CLASSIFIER_CHECKPOINT", "")
    learned_classifier_from_env.cache_clear()
    yield
    learned_classifier_from_env.cache_clear()
