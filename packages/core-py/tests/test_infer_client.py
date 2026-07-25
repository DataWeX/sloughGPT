"""
Tests for InferClient — Python SDK for the /infer API.

Uses mocked HTTP responses to test the client without a live server.
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from domains.infrastructure.infer_client import (
    InferClient,
    InferResult,
    EmbedResult,
    TokenizeResult,
    DetokenizeResult,
    HealthResult,
    InfoResult,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def client():
    return InferClient(base_url="http://localhost:8000", timeout=10)


@pytest.fixture
def mock_get(monkeypatch):
    """Mock _get method on the client."""
    mock = MagicMock()
    monkeypatch.setattr(client := InferClient(), "_get", mock)
    return mock, client


def _mock_response(data, status=200):
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data
    resp.raise_for_status.return_value = None
    resp.headers = {"Content-Type": "application/json"}
    return resp


# ── generate() tests ────────────────────────────────────────────────


class TestInferClientGenerate:
    def test_generate_returns_infer_result(self, client):
        """generate() returns InferResult with text."""
        with patch.object(client, "_post", return_value={
            "text": "Hello world",
            "model": "gpt2",
            "tokens_generated": 2,
            "elapsed_ms": 100.5,
        }):
            result = client.generate("Hello")
        assert isinstance(result, InferResult)
        assert result.text == "Hello world"
        assert result.model == "gpt2"
        assert result.tokens_generated == 2
        assert result.elapsed_ms == 100.5

    def test_generate_passes_params(self, client):
        """generate() passes all parameters to the API."""
        with patch.object(client, "_post", return_value={"text": "", "model": "", "tokens_generated": 0}) as mock:
            client.generate("test", max_new_tokens=50, temperature=0.5, top_p=0.8, top_k=30)
        call_args = mock.call_args
        assert call_args[0][0] == ""  # path
        payload = call_args[0][1]
        assert payload["prompt"] == "test"
        assert payload["max_new_tokens"] == 50
        assert payload["temperature"] == 0.5
        assert payload["top_p"] == 0.8
        assert payload["top_k"] == 30

    def test_generate_includes_model_when_set(self, client):
        """generate() includes model in payload when provided."""
        with patch.object(client, "_post", return_value={"text": "", "model": "", "tokens_generated": 0}) as mock:
            client.generate("test", model="custom-model")
        payload = mock.call_args[0][1]
        assert payload["model"] == "custom-model"

    def test_generate_excludes_model_when_none(self, client):
        """generate() excludes model from payload when None."""
        with patch.object(client, "_post", return_value={"text": "", "model": "", "tokens_generated": 0}) as mock:
            client.generate("test")
        payload = mock.call_args[0][1]
        assert "model" not in payload


# ── generate_stream() tests ─────────────────────────────────────────


class TestInferClientGenerateStream:
    def test_generate_stream_yields_tokens(self, client):
        """generate_stream() yields tokens from SSE events."""
        events = [
            'data: {"stream":"infer","phase":"STREAMING","status":"working","data":{"token":"Hello"}}',
            'data: {"stream":"infer","phase":"STREAMING","status":"working","data":{"token":" world"}}',
            'data: {"stream":"infer","phase":"STREAMING","status":"complete","data":{},"meta":{"tokens":2}}',
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.iter_lines.return_value = iter(events)

        with patch.object(client, "_post", side_effect=Exception("should not be called")):
            pass

        # Mock the requests.post call directly
        with patch("domains.infrastructure.infer_client._requests") as mock_req:
            mock_req.post.return_value = mock_resp
            tokens = list(client.generate_stream("Hello"))
        assert tokens == ["Hello", " world"]

    def test_generate_stream_handles_error(self, client):
        """generate_stream() raises on error events."""
        events = [
            'data: {"stream":"infer","phase":"STREAMING","status":"error","data":{"error":"boom"}}',
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.iter_lines.return_value = iter(events)

        with patch("domains.infrastructure.infer_client._requests") as mock_req:
            mock_req.post.return_value = mock_resp
            with pytest.raises(RuntimeError, match="boom"):
                list(client.generate_stream("Hello"))


# ── embed() tests ───────────────────────────────────────────────────


class TestInferClientEmbed:
    def test_embed_returns_list(self, client):
        """embed() returns list of floats."""
        with patch.object(client, "_post", return_value={
            "embedding": [0.1, 0.2, 0.3],
            "dimensions": 3,
            "model": "test",
        }):
            result = client.embed("hello")
        assert isinstance(result, list)
        assert result == [0.1, 0.2, 0.3]

    def test_embed_result_returns_full_metadata(self, client):
        """embed_result() returns EmbedResult with dimensions and model."""
        with patch.object(client, "_post", return_value={
            "embedding": [0.1, 0.2],
            "dimensions": 2,
            "model": "ngram",
        }):
            result = client.embed_result("hello")
        assert isinstance(result, EmbedResult)
        assert result.dimensions == 2
        assert result.model == "ngram"


# ── tokenize() tests ────────────────────────────────────────────────


class TestInferClientTokenize:
    def test_tokenize_returns_result(self, client):
        """tokenize() returns TokenizeResult."""
        with patch.object(client, "_post", return_value={
            "tokens": ["hello", "world"],
            "ids": [1, 2],
            "count": 2,
        }):
            result = client.tokenize("hello world")
        assert isinstance(result, TokenizeResult)
        assert result.tokens == ["hello", "world"]
        assert result.ids == [1, 2]
        assert result.count == 2


# ── detokenize() tests ──────────────────────────────────────────────


class TestInferClientDetokenize:
    def test_detokenize_returns_text(self, client):
        """detokenize() returns DecdetokenizeResult."""
        with patch.object(client, "_post", return_value={
            "text": "hello world",
            "count": 2,
        }):
            result = client.detokenize([1, 2])
        assert isinstance(result, DetokenizeResult)
        assert result.text == "hello world"
        assert result.count == 2


# ── health() tests ──────────────────────────────────────────────────


class TestInferClientHealth:
    def test_health_returns_result(self, client):
        """health() returns HealthResult."""
        with patch.object(client, "_get", return_value={
            "status": "ready",
            "model_loaded": True,
            "model_id": "gpt2",
            "engine_type": "NumpyEngine",
            "has_streaming": True,
            "has_embedding": False,
        }):
            result = client.health()
        assert isinstance(result, HealthResult)
        assert result.status == "ready"
        assert result.model_loaded is True
        assert result.model_id == "gpt2"


# ── info() tests ────────────────────────────────────────────────────


class TestInferClientInfo:
    def test_info_returns_result(self, client):
        """info() returns InfoResult."""
        with patch.object(client, "_get", return_value={
            "model_id": "gpt2",
            "model_type": "NumpyEngine",
            "num_parameters": 124000000,
            "vocab_size": 50257,
            "max_context": 1024,
            "num_layers": 12,
            "has_tokenizer": True,
            "has_streaming": True,
            "has_embedding": False,
            "extra": {},
        }):
            result = client.info()
        assert isinstance(result, InfoResult)
        assert result.model_id == "gpt2"
        assert result.num_parameters == 124000000


# ── is_ready() tests ────────────────────────────────────────────────


class TestInferClientReady:
    def test_is_ready_true(self, client):
        """is_ready() returns True when model loaded and status ready."""
        with patch.object(client, "health", return_value=HealthResult(
            status="ready", model_loaded=True,
        )):
            assert client.is_ready() is True

    def test_is_ready_false_no_model(self, client):
        """is_ready() returns False when no model loaded."""
        with patch.object(client, "health", return_value=HealthResult(
            status="no_model", model_loaded=False,
        )):
            assert client.is_ready() is False

    def test_is_ready_false_on_error(self, client):
        """is_ready() returns False when health() raises."""
        with patch.object(client, "health", side_effect=Exception("conn refused")):
            assert client.is_ready() is False


# ── URL construction tests ──────────────────────────────────────────


class TestInferClientURL:
    def test_url_construction(self, client):
        """_url() constructs correct URL."""
        assert client._url("/health") == "http://localhost:8000/infer/health"
        assert client._url("") == "http://localhost:8000/infer"

    def test_url_strips_trailing_slash(self):
        """Base URL trailing slash is stripped."""
        c = InferClient(base_url="http://localhost:8000/")
        assert c._url("/health") == "http://localhost:8000/infer/health"

    def test_repr(self, client):
        """repr() shows base_url."""
        assert "localhost:8000" in repr(client)
