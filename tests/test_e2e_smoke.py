"""
E2E smoke test — validates 12 key endpoints through the full FastAPI stack.

Starts the server in-process via ``TestClient(app)`` with lifespan triggered
by the ``with`` context manager.  No model is loaded (autoload disabled
via env vars), so generation endpoints return 503 — this proves the full
routing, middleware, serialization, and error-handling pipeline works.

Run: ``pytest tests/test_e2e_smoke.py -v``
"""

import os

# Force-disable autoload + background services so lifespan completes quickly.
# These MUST be set before importing the app (config reads env vars at import time).
os.environ["MAN_AUTOLOAD_MODEL"] = ""
os.environ["MAN_AUTO_WORKFLOW"] = "false"
os.environ["MAN_HEALTH_MONITOR"] = "false"
os.environ["MAN_WATCHDOG"] = "false"
os.environ["MAN_WEB"] = "false"

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client():
    """Start the FastAPI app with full lifespan via ``with TestClient(app)``."""
    from apps.api.server.main import app

    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_health_has_required_fields(self, client):
        resp = client.get("/health")
        data = resp.json()
        for field in ("model_loaded", "model_type", "inference_count"):
            assert field in data, f"Missing field: {field}"

    def test_health_live_returns_200(self, client):
        assert client.get("/health/live").status_code == 200


class TestSouls:
    def test_list_souls(self, client):
        resp = client.get("/souls")
        assert resp.status_code == 200
        data = resp.json()
        assert "souls" in data
        assert isinstance(data["souls"], list)

    def test_current_soul(self, client):
        resp = client.get("/souls/current")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data or "soul" in data
        assert isinstance(data, dict)


class TestModels:
    def test_list_models(self, client):
        resp = client.get("/models")
        assert resp.status_code == 200
        models = resp.json()
        assert isinstance(models, list)

    def test_models_hf(self, client):
        resp = client.get("/models/hf")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)


class TestChat:
    def test_sessions_list(self, client):
        resp = client.get("/chat/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    def test_create_session(self, client):
        resp = client.post("/chat/sessions", json={"session_id": "e2e-test-session"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "created"

    def test_chat_returns_503_when_no_model(self, client):
        """Without a loaded model, chat should return 503."""
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 503
        data = resp.json()
        assert "detail" in data or "error" in data or "message" in data

    def test_chat_stream_returns_503_when_no_model(self, client):
        resp = client.post("/chat/stream", json={
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 503


class TestInferenceGenerate:
    def test_generate_returns_503_when_no_model(self, client):
        resp = client.post("/inference/generate", json={"prompt": "Hello"})
        assert resp.status_code == 503
        data = resp.json()
        assert "detail" in data or "error" in data or "message" in data

    def test_generate_stream_returns_503_when_no_model(self, client):
        resp = client.post("/inference/generate/stream", json={"prompt": "Hi"})
        assert resp.status_code == 503


class TestDatasets:
    def test_list_datasets(self, client):
        resp = client.get("/datasets")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, (list, dict))
        if isinstance(data, dict):
            assert "datasets" in data


class TestTokenizer:
    def test_tokenizer_stats(self, client):
        """May return 503 if no model is loaded (tokenizer depends on model)."""
        resp = client.get("/tokenizer/stats")
        assert resp.status_code in {200, 503}
        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data, dict)


class TestAutoTrain:
    def test_list_checkpoints(self, client):
        resp = client.get("/auto-train/checkpoints")
        assert resp.status_code == 200
        data = resp.json()
        assert "checkpoints" in data
        assert isinstance(data["checkpoints"], list)


class TestWorkflow:
    def test_workflow_status(self, client):
        resp = client.get("/workflow/status")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)


class Test404:
    def test_nonexistent_route_returns_404(self, client):
        resp = client.get("/nonexistent-route")
        assert resp.status_code == 404


class TestSystem:
    def test_system_metrics(self, client):
        resp = client.get("/system/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_system_info(self, client):
        resp = client.get("/system/info")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
