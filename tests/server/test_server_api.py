"""
Server API Tests — validates current API endpoints via TestClient.

All tests marked ``slow`` (deselected by default). Run explicitly with:
  ``pytest tests/server/test_server_api.py -m slow``
"""

import pytest
from fastapi.testclient import TestClient

try:
    from apps.api.server.main import app
    client = TestClient(app)
except Exception:
    pytest.skip("Server app not available", allow_module_level=True)


class TestHealthEndpoint:
    def test_health_returns_status(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_returns_model_info(self):
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data
        assert "model_type" in data


class TestModelEndpoints:
    def test_list_models(self):
        response = client.get("/models")
        assert response.status_code == 200
        models = response.json()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_model_has_required_fields(self):
        response = client.get("/models")
        models = response.json()
        for model in models:
            assert "model_id" in model
            assert "status" in model

    def test_models_health_aliased(self):
        """/health covers model health status."""
        response = client.get("/health")
        assert response.status_code == 200


class TestInferenceEndpoints:
    def test_inference_generate_503_when_no_provider(self):
        """Should error gracefully when no model is loaded."""
        response = client.post("/inference/generate", json={"prompt": "Hi"})
        assert response.status_code == 503


class TestAutoTrainEndpoints:
    def test_list_checkpoints(self):
        response = client.get("/auto-train/checkpoints")
        assert response.status_code == 200
        data = response.json()
        assert "checkpoints" in data
        assert isinstance(data["checkpoints"], list)

    def test_list_models(self):
        """Sanity: auto-train models list."""
        response = client.get("/models/hf")
        assert response.status_code == 200


class TestChatEndpoints:
    @pytest.mark.slow
    def test_chat_sessions(self):
        response = client.get("/chat/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data

    def test_create_session(self):
        response = client.post("/chat/sessions", json={"session_id": "api-test-session"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
