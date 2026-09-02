"""
Tests for ReadinessGateMiddleware — blocks inference until model loaded.

Key behavior:
- OPTIONS (CORS preflight) requests must ALWAYS pass through, even when
  the model is not ready. Blocking them breaks browser CORS and causes
  the frontend retry loop observed in production logs.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.middleware import ReadinessGateMiddleware


@pytest.fixture
def app():
    _app = FastAPI()
    _app.add_middleware(ReadinessGateMiddleware)

    @_app.get("/chat")
    async def chat():
        return {"ok": True}

    @_app.post("/chat")
    async def chat_post():
        return {"ok": True}

    @_app.post("/inference/generate")
    async def generate():
        return {"ok": True}

    @_app.get("/health")
    async def health():
        return {"ok": True}

    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def model_not_ready():
    """Force model-not-ready state."""
    import state as server_state
    from startup_progress import STARTUP_PHASE

    saved_model = server_state.model
    saved_provider = server_state.provider
    saved_phase = STARTUP_PHASE["phase"]

    server_state.model = None
    server_state.provider = None
    STARTUP_PHASE["phase"] = "ready"
    try:
        yield
    finally:
        server_state.model = saved_model
        server_state.provider = saved_provider
        STARTUP_PHASE["phase"] = saved_phase


class TestReadinessGateOPTIONS:
    """OPTIONS (CORS preflight) must always pass through."""

    def test_options_chat_passes_when_model_not_ready(self, client, model_not_ready):
        resp = client.options("/chat")
        assert resp.status_code != 503

    def test_options_inference_generate_passes_when_model_not_ready(self, client, model_not_ready):
        resp = client.options("/inference/generate")
        assert resp.status_code != 503

    def test_post_chat_blocked_when_model_not_ready(self, client, model_not_ready):
        resp = client.post("/chat")
        assert resp.status_code == 503

    def test_get_chat_blocked_when_model_not_ready(self, client, model_not_ready):
        resp = client.get("/chat")
        assert resp.status_code == 503

    def test_health_always_passes(self, client, model_not_ready):
        resp = client.get("/health")
        assert resp.status_code == 200
