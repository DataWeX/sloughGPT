"""
Tests for /inference/generate and /inference/generate/stream endpoints.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from apps.api.server.main import app
    return TestClient(app)


class AsyncIteratorMock:
    """Async iterator that yields tokens."""
    def __init__(self, tokens):
        self._tokens = tokens

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._tokens:
            raise StopAsyncIteration
        return self._tokens.pop(0)


@pytest.fixture
def mock_provider():
    """Mock the provider pipeline to avoid model loading."""
    provider = MagicMock()
    provider.chat = AsyncMock(return_value="Hello! How are you today?")
    provider.chat_stream = MagicMock(return_value=AsyncIteratorMock(["Hello!", " How", " are", " you?"]))
    provider.model_id = "test-model"
    with patch("domains.models.provider.get_provider", return_value=provider):
        yield provider


class TestGenerateEndpoint:
    """Tests for POST /inference/generate."""

    def test_generate_returns_text(self, client, mock_provider):
        """Should return generated text with model info."""
        response = client.post(
            "/inference/generate",
            json={"prompt": "Hello", "max_new_tokens": 10}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Hello! How are you today?"
        assert data["model"] == "gpt2"
        assert data["tokens_generated"] > 0

    def test_generate_passes_params(self, client, mock_provider):
        """Should pass generation params to provider."""
        client.post(
            "/inference/generate",
            json={
                "prompt": "Hi",
                "max_new_tokens": 50,
                "temperature": 0.5,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.1,
            }
        )
        mock_provider.chat.assert_called_once()
        kwargs = mock_provider.chat.call_args[1]
        assert kwargs["max_tokens"] == 50
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_p"] == 0.8
        assert kwargs["top_k"] == 20
        assert kwargs["repetition_penalty"] == 1.1

    def test_generate_no_provider_returns_503(self, client):
        """Should return 503 when no provider is available."""
        with patch("domains.models.provider.get_provider", return_value=None):
            response = client.post(
                "/inference/generate",
                json={"prompt": "Hello"}
            )
        assert response.status_code == 503


class TestGenerateStreamEndpoint:
    """Tests for POST /inference/generate/stream."""

    def test_generate_stream_returns_sse(self, client, mock_provider):
        """Should return SSE tokens from provider."""
        response = client.post(
            "/inference/generate/stream",
            json={"prompt": "Hello", "max_new_tokens": 10}
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = []
        for line in response.text.split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        assert len(events) >= 2
        working = [e for e in events if e["status"] == "working"]
        complete = [e for e in events if e["status"] == "complete"]
        assert len(working) >= 1
        assert len(complete) == 1
        assert working[0]["stream"] == "generate"
        assert "token" in working[0]["data"]

    def test_generate_stream_tokens_ordered(self, client, mock_provider):
        """Should yield tokens in correct order."""
        response = client.post(
            "/inference/generate/stream",
            json={"prompt": "Hi", "max_new_tokens": 5}
        )
        tokens = []
        for line in response.text.split("\n"):
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event["status"] == "working" and event["data"].get("token"):
                    tokens.append(event["data"]["token"])

        assert len(tokens) == 4
        assert "".join(tokens) == "Hello! How are you?"

    def test_generate_stream_complete_meta(self, client, mock_provider):
        """Should include token count and elapsed time in final event."""
        response = client.post(
            "/inference/generate/stream",
            json={"prompt": "Hi", "max_new_tokens": 10}
        )
        for line in response.text.split("\n"):
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event["status"] == "complete":
                    assert "tokens" in event["meta"]
                    assert event["meta"]["tokens"] == 4
                    assert "elapsed_ms" in event["meta"]
