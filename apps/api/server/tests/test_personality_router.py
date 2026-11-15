"""Tests for the descriptor-projected personality history crossing.

The route lives in ``routers/personality.py`` (descriptor + projection, see
docs/TRANSPORT_PROJECTIONS.md) — its tests live here so the module name, the
test file name and the docs parity check agree.
"""

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_personality_history_empty():
    """The descriptor-projected crossing serves the same envelope as before."""
    from infrastructure.auth import require_auth_if_enabled
    from infrastructure.exception_handlers import register_app_error_handler
    from routers.personality import SPEC as PERSONALITY_HISTORY_SPEC
    from routers.personality import router as personality_router

    assert PERSONALITY_HISTORY_SPEC.name == "consciousness.personality_history"

    mock_eng = MagicMock()
    mock_eng.self_model.episodes = []

    app = FastAPI()
    register_app_error_handler(app)
    app.include_router(personality_router)
    app.dependency_overrides[require_auth_if_enabled] = lambda: None

    with patch("domain.cognition.get_consciousness", return_value=mock_eng):
        res = TestClient(app).get("/consciousness/personality/history")

    assert res.status_code == 200
    data = res.json()
    assert "history" in data["data"]
    assert "labels" in data["data"]
    assert isinstance(data["data"]["history"], list)


def test_personality_history_with_episodes():
    import time

    from infrastructure.auth import require_auth_if_enabled
    from infrastructure.exception_handlers import register_app_error_handler
    from routers.personality import router as personality_router

    from domain.cognition._internal.consciousness.self_model import SelfEpisode

    mock_eng = MagicMock()
    mock_eng.config.is_enabled.return_value = True
    mock_eng.config.level = 1

    episodes = [
        SelfEpisode(
            timestamp=time.time() - 100 + i,
            input_text=f"test {i}",
            response=f"response {i}",
            qualia={"novelty": 0.8 if i % 2 == 0 else 0.3, "coherence": 0.5, "valence": 0.6},
            self_insight="test insight",
            growth_delta=0.06 if i % 3 == 0 else 0.01,
        )
        for i in range(10)
    ]
    mock_eng.self_model.episodes = episodes

    app = FastAPI()
    register_app_error_handler(app)
    app.include_router(personality_router)
    app.dependency_overrides[require_auth_if_enabled] = lambda: None

    with patch("domain.cognition.get_consciousness", return_value=mock_eng):
        res = TestClient(app).get("/consciousness/personality/history")

    assert res.status_code == 200
    data = res.json()
    history = data["data"]["history"]
    assert len(history) > 0
    assert "voice" in history[0]
    assert "traits" in history[0]
