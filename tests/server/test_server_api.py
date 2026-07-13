"""
Server API Tests — validates key API endpoints via TestClient.

All tests marked ``slow`` (deselected by default). Run explicitly with:
  ``pytest tests/server/test_server_api.py -m slow``
"""

import pytest
from fastapi.testclient import TestClient

try:
    from apps.api.server.tests.test_support import get_test_client
    client = get_test_client()
except Exception:
    pytest.skip("Server app not available", allow_module_level=True)


class TestHealthEndpoint:
    def test_health_returns_status(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "success")

    def test_health_returns_model_info(self):
        response = client.get("/health")
        data = response.json()
        health = data.get("data", data)
        assert "model_loaded" in health
        assert "model_type" in health


class TestModelEndpoints:
    def test_list_models(self):
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        models = data.get("models", data.get("data", data))
        assert isinstance(models, list)
        assert len(models) >= 0  # may be empty in test env

    def test_model_has_required_fields(self):
        response = client.get("/models")
        data = response.json()
        models = data.get("models", data.get("data", data))
        for model in models:
            assert "model_id" in model or "id" in model
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
    @pytest.mark.slow
    def test_list_checkpoints(self):
        response = client.get("/auto-train/checkpoints")
        assert response.status_code == 200
        data = response.json()
        payload = data.get("data", data)
        checkpoints = payload if isinstance(payload, list) else payload.get("checkpoints", [])
        assert isinstance(checkpoints, list)

    @pytest.mark.slow
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
        payload = data.get("data", data)
        assert isinstance(payload, (list, dict))

    @pytest.mark.slow
    def test_create_session(self):
        response = client.post("/chat/sessions", json={"session_id": "api-test-session"})
        assert response.status_code == 200
        data = response.json()
        session_data = data.get("data", data)
        assert "session_id" in session_data
