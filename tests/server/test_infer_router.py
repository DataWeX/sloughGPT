"""
Tests for the unified inference router — /infer prefix endpoints.
"""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.infer import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestInfer:
    @patch("state.model")
    @patch("domains.models.provider.get_provider")
    def test_generates_text(self, mock_get_prov, mock_model, client):
        provider = AsyncMock()
        provider.chat.return_value = "Hello world"
        mock_get_prov.return_value = provider
        mock_model.model_id = "test-model"
        resp = client.post("/infer", json={"prompt": "Hi"})
        assert resp.status_code == 200
        assert resp.json()["text"] == "Hello world"

    @patch("state.model", None)
    def test_returns_503_when_no_model(self, client):
        resp = client.post("/infer", json={"prompt": "Hi"})
        assert resp.status_code == 503

    @patch("state.model")
    @patch("domains.models.provider.get_provider", return_value=None)
    def test_returns_503_when_no_provider(self, mock_prov, mock_model, client):
        resp = client.post("/infer", json={"prompt": "Hi"})
        assert resp.status_code == 503

    @patch("state.model")
    @patch("domains.models.provider.get_provider")
    def test_passes_temperature(self, mock_get_prov, mock_model, client):
        provider = AsyncMock()
        provider.chat.return_value = "ok"
        mock_get_prov.return_value = provider
        mock_model.model_id = "m"
        client.post("/infer", json={"prompt": "Hi", "temperature": 0.1})
        _, kwargs = provider.chat.call_args
        assert kwargs.get("temperature") == 0.1

    @patch("state.model")
    @patch("domains.models.provider.get_provider")
    def test_passes_top_p(self, mock_get_prov, mock_model, client):
        provider = AsyncMock()
        provider.chat.return_value = "ok"
        mock_get_prov.return_value = provider
        mock_model.model_id = "m"
        client.post("/infer", json={"prompt": "Hi", "top_p": 0.5})
        _, kwargs = provider.chat.call_args
        assert kwargs.get("top_p") == 0.5

    @patch("state.model")
    @patch("domains.models.provider.get_provider")
    def test_passes_max_new_tokens(self, mock_get_prov, mock_model, client):
        provider = AsyncMock()
        provider.chat.return_value = "ok"
        mock_get_prov.return_value = provider
        mock_model.model_id = "m"
        client.post("/infer", json={"prompt": "Hi", "max_new_tokens": 100})
        _, kwargs = provider.chat.call_args
        assert kwargs.get("max_tokens") == 100

    @patch("state.model")
    @patch("domains.models.provider.get_provider")
    def test_provider_exception_returns_500(self, mock_get_prov, mock_model, client):
        provider = AsyncMock()
        provider.chat.side_effect = RuntimeError("OOM")
        mock_get_prov.return_value = provider
        mock_model.model_id = "m"
        resp = client.post("/infer", json={"prompt": "Hi"})
        assert resp.status_code == 500

    @patch("state.model")
    @patch("domains.models.provider.get_provider")
    def test_response_has_model_field(self, mock_get_prov, mock_model, client):
        provider = AsyncMock()
        provider.chat.return_value = "test"
        mock_get_prov.return_value = provider
        mock_model.model_id = "gpt2"
        resp = client.post("/infer", json={"prompt": "Hi"})
        assert "model" in resp.json()

    def test_missing_prompt_rejected(self, client):
        resp = client.post("/infer", json={})
        assert resp.status_code == 422

    def test_empty_prompt_accepted(self, client):
        """Empty prompt is valid per schema (no min_length constraint)."""
        with patch("state.model") as mock_model, \
             patch("domains.models.provider.get_provider") as mock_prov:
            provider = AsyncMock()
            provider.chat.return_value = ""
            mock_prov.return_value = provider
            mock_model.model_id = "m"
            resp = client.post("/infer", json={"prompt": ""})
            assert resp.status_code == 200


class TestInferTokenize:
    def test_tokenizes_text(self, client):
        resp = client.post("/infer/tokenize", json={"text": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] > 0
        assert len(data["tokens"]) == data["count"]

    def test_tokenizes_empty_string(self, client):
        resp = client.post("/infer/tokenize", json={"text": ""})
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_token_ids_are_integers(self, client):
        resp = client.post("/infer/tokenize", json={"text": "abc"})
        ids = resp.json()["ids"]
        assert all(isinstance(i, int) for i in ids)

    def test_unicode_text(self, client):
        resp = client.post("/infer/tokenize", json={"text": "héllo wörld"})
        assert resp.status_code == 200
        assert resp.json()["count"] > 0

    def test_missing_text_rejected(self, client):
        resp = client.post("/infer/tokenize", json={})
        assert resp.status_code == 422


class TestInferDetokenize:
    def test_detokenizes_ids(self, client):
        resp = client.post("/infer/detokenize", json={"ids": [104, 101, 108, 108, 111]})
        assert resp.status_code == 200
        assert resp.json()["text"] == "hello"

    def test_empty_ids(self, client):
        resp = client.post("/infer/detokenize", json={"ids": []})
        assert resp.status_code == 200
        assert resp.json()["text"] == ""
        assert resp.json()["count"] == 0

    def test_single_id(self, client):
        resp = client.post("/infer/detokenize", json={"ids": [65]})
        assert resp.status_code == 200
        assert resp.json()["text"] == "A"

    def test_invalid_utf8_handled(self, client):
        resp = client.post("/infer/detokenize", json={"ids": [255, 254]})
        assert resp.status_code == 200
        assert resp.json()["count"] == 2

    def test_missing_ids_rejected(self, client):
        resp = client.post("/infer/detokenize", json={})
        assert resp.status_code == 422


class TestInferHealth:
    def test_returns_no_model_status(self, client):
        resp = client.get("/infer/health")
        assert resp.status_code == 200
        assert resp.json()["model_loaded"] is False

    def test_no_model_has_streaming_false(self, client):
        resp = client.get("/infer/health")
        assert resp.json()["has_streaming"] is False

    def test_no_model_status_value(self, client):
        resp = client.get("/infer/health")
        assert resp.json()["status"] == "no_model"

    def test_model_loaded_status(self, client):
        mock_info = type("Info", (), {
            "model_id": "gpt2",
            "model_type": "hf",
            "has_streaming": True,
            "has_embedding": False,
        })()
        mock_model = type("MockModel", (), {
            "info": lambda self: mock_info,
            "embed": None,
        })()
        with patch("apps.api.server.routers.infer.InferRouter._get_model_interface", return_value=mock_model):
            resp = client.get("/infer/health")
            assert resp.json()["model_loaded"] is True

    def test_model_loaded_has_streaming(self, client):
        mock_info = type("Info", (), {
            "model_id": "gpt2",
            "model_type": "hf",
            "has_streaming": True,
            "has_embedding": False,
        })()
        mock_model = type("MockModel", (), {
            "info": lambda self: mock_info,
            "embed": None,
        })()
        with patch("apps.api.server.routers.infer.InferRouter._get_model_interface", return_value=mock_model):
            resp = client.get("/infer/health")
            assert resp.json()["has_streaming"] is True


class TestInferInfo:
    def test_returns_503_when_no_model(self, client):
        resp = client.get("/infer/info")
        assert resp.status_code == 503

    def test_returns_model_info(self, client):
        mock_info = type("Info", (), {
            "model_id": "gpt2",
            "model_type": "hf",
            "num_parameters": 124000000,
            "vocab_size": 50257,
            "max_context": 1024,
            "num_layers": 12,
            "has_tokenizer": True,
            "has_streaming": True,
            "has_embedding": False,
            "extra": {},
        })()
        mock_model = type("MockModel", (), {
            "info": lambda self: mock_info,
        })()
        with patch("apps.api.server.routers.infer.InferRouter._get_model_interface", return_value=mock_model):
            resp = client.get("/infer/info")
            assert resp.status_code == 200
            assert resp.json()["model_id"] == "gpt2"
            assert resp.json()["num_parameters"] == 124000000


class TestInferStream:
    @patch("state.model", None)
    def test_stream_returns_error_when_no_model(self, client):
        resp = client.post("/infer/stream", json={"prompt": "Hi"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_stream_missing_prompt_rejected(self, client):
        resp = client.post("/infer/stream", json={})
        assert resp.status_code == 422
