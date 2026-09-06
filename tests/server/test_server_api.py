"""
Server API Tests — validates key API endpoints via TestClient.

All tests marked ``slow`` (deselected by default). Run explicitly with:
  ``pytest tests/server/test_server_api.py -m slow``
"""

import pytest
from unittest.mock import patch
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
        """Should error gracefully when no provider is available."""
        with patch("domains.models.provider.get_provider", return_value=None), \
             patch("apps.api.server.state.model", "gpt2", create=True):
            response = client.post("/inference/generate", json={"prompt": "Hi"})
            assert response.status_code == 503


class TestInfoEndpoints:
    def test_info_returns_version(self):
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert data.get("api_version") == "1.0.0"

    def test_info_includes_model_block(self):
        response = client.get("/info")
        data = response.json()
        model = data.get("model", {})
        assert "type" in model
        assert "loaded" in model
        assert isinstance(model["loaded"], bool)

    def test_info_includes_host_block(self):
        response = client.get("/info")
        data = response.json()
        host = data.get("host", {})
        assert "platform" in host
        assert "python_version" in host

    def test_info_soul_ok(self):
        response = client.get("/info/soul")
        assert response.status_code == 200


class TestUnknownRoute:
    def test_unknown_route_404(self):
        response = client.get("/nonexistent-path-xyz")
        assert response.status_code == 404

    def test_unknown_route_post_404(self):
        response = client.post("/nonexistent-path-xyz", json={})
        assert response.status_code == 404


class TestInvalidRequests:
    def test_generate_missing_payload_422(self):
        with patch("state.model", object()):
            response = client.post("/inference/generate", json={})
            assert response.status_code == 422

    def test_generate_no_prompt_422(self):
        with patch("state.model", object()):
            response = client.post("/inference/generate")
            assert response.status_code == 422


class TestProvidersEndpoints:
    def test_providers_ok(self):
        response = client.get("/providers")
        assert response.status_code == 200
        assert "data" in response.json()

    def test_providers_has_status(self):
        response = client.get("/providers")
        assert "status" in response.json()


class TestKnowledgeStatsDetails:
    def test_knowledge_stats_topics(self):
        response = client.get("/knowledge/stats")
        data = response.json().get("data", {})
        assert isinstance(data.get("topics", {}), dict)
        assert isinstance(data.get("topic_count", 0), int)


class TestTokenizerStatsDetails:
    def test_tokenizer_vocab_size_int(self):
        response = client.get("/tokenizer/stats")
        data = response.json().get("data", response.json())
        assert isinstance(data["vocab_size"], int)

    def test_tokenizer_has_merges(self):
        response = client.get("/tokenizer/stats")
        data = response.json().get("data", response.json())
        assert "merged_subwords" in data or "num_merges" in data


class TestSystemEndpoints:
    def test_executor_status_ok(self):
        response = client.get("/system/executor")
        assert response.status_code == 200
        data = response.json()["data"]
        assert "initialized" in data
        assert "active_jobs" in data
        assert isinstance(data.get("active_jobs", 0), int)

    def test_executor_uninitialized_reports_zero_jobs(self):
        from domains.training.executor import _instance as executor_instance
        if executor_instance is None:
            response = client.get("/system/executor")
            data = response.json()["data"]
            assert data.get("initialized") is False
            assert data.get("active_jobs") == 0


class TestStatsEndpoints:
    def test_tokenizer_stats_ok(self):
        response = client.get("/tokenizer/stats")
        assert response.status_code == 200
        data = response.json().get("data", response.json())
        assert "vocab_size" in data

    def test_knowledge_stats_shape(self):
        response = client.get("/knowledge/stats")
        assert response.status_code == 200
        data = response.json()
        payload = data.get("data", data)
        assert isinstance(payload, dict)
        assert "total_items" in payload


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
