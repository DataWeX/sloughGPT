"""
E2E smoke test — validates 14 key endpoints through the full FastAPI stack.

Starts the server in-process via ``TestClient(app)`` with lifespan triggered
by the ``with`` context manager.  Model loading is skipped by default (set
``MAN_AUTOLOAD_MODEL=gpt2`` in the environment to enable real generation).

Tests the full routing, middleware, serialization, and error-handling pipeline.
Generation-dependent endpoints are skipped when no model is loaded.

Run: ``pytest tests/test_e2e_smoke.py -v``
"""

import os

# Disable background services so lifespan completes quickly.
# These MUST be set before importing the app (config reads env vars at import time).
# Don't touch MAN_AUTOLOAD_MODEL — let the user's shell env (or lack thereof)
# determine whether a model is loaded. Set it to "gpt2" for generation tests:
#   MAN_AUTOLOAD_MODEL=gpt2 pytest tests/test_e2e_smoke.py
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


@pytest.fixture(scope="module")
def model_loaded(client):
    """Check whether a model is loaded after startup.

    Polls /health up to 30s for the model to finish loading (autoload runs
    in a background task and may not be complete when the lifespan yields).
    """
    import time
    deadline = time.time() + 20
    while time.time() < deadline:
        resp = client.get("/health")
        data = resp.json()
        if data.get("model_loaded"):
            return True
        time.sleep(1)
    return False


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_health_fields(self, client):
        resp = client.get("/health")
        data = resp.json()
        for field in ("model_loaded", "model_type", "inference_count"):
            assert field in data, f"Missing field: {field}"

    def test_health_live(self, client):
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
        assert isinstance(data, dict)
        assert "name" in data or "soul" in data


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


class TestChatSessions:
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


class TestChatGenerate:
    """Generation endpoints — only tested when a model is loaded."""

    def test_chat_non_streaming(self, client, model_loaded):
        if not model_loaded:
            pytest.skip("No model loaded")
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 20,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data

    def test_chat_stream(self, client, model_loaded):
        if not model_loaded:
            pytest.skip("No model loaded")
        resp = client.post("/chat/stream", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 20,
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_generate(self, client, model_loaded):
        if not model_loaded:
            pytest.skip("No model loaded")
        resp = client.post("/inference/generate", json={
            "prompt": "Hello", "max_new_tokens": 20,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data

    def test_generate_stream(self, client, model_loaded):
        if not model_loaded:
            pytest.skip("No model loaded")
        resp = client.post("/inference/generate/stream", json={
            "prompt": "Hi", "max_new_tokens": 20,
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")


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
        resp = client.get("/tokenizer/stats")
        assert resp.status_code in {200, 503}


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

    def test_system_disk(self, client):
        resp = client.get("/system/disk")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)


class TestRouting:
    def test_nonexistent_route_returns_404(self, client):
        resp = client.get("/nonexistent-route")
        assert resp.status_code == 404

    def test_docs_redirect(self, client):
        resp = client.get("/docs", follow_redirects=False)
        assert resp.status_code in {200, 302, 307}


class TestHealthDetailed:
    def test_detailed_health(self, client):
        resp = client.get("/health/detailed")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
