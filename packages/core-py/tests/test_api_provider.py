"""Tests for external API provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from domain.inference._internal.api_provider import ApiProvider


class TestApiProviderInit:
    def test_valid_init(self):
        p = ApiProvider(api_key="sk-test", api_url="https://api.openai.com/v1", model="gpt-4o")
        assert p.model_id == "gpt-4o"
        assert p.api_key == "sk-test"
        assert p.api_url == "https://api.openai.com/v1"

    def test_strips_trailing_slash(self):
        p = ApiProvider(api_key="sk-test", api_url="https://api.openai.com/v1/")
        assert p.api_url == "https://api.openai.com/v1"

    def test_empty_api_key_raises(self):
        with pytest.raises(ValueError, match="API key required"):
            ApiProvider(api_key="", api_url="https://api.openai.com/v1")

    def test_empty_api_url_raises(self):
        with pytest.raises(ValueError, match="API URL required"):
            ApiProvider(api_key="sk-test", api_url="")

    def test_capabilities(self):
        p = ApiProvider(api_key="sk-test", api_url="https://api.openai.com/v1")
        caps = p.capabilities
        assert caps.chat is True
        assert caps.streaming is True
        assert caps.embedding is False

    def test_metadata(self):
        p = ApiProvider(api_key="sk-test", api_url="https://api.openai.com/v1", model="gpt-4")
        meta = p.metadata
        assert meta["provider"] == "api"
        assert meta["model_id"] == "gpt-4"

    def test_embed_returns_empty(self):
        p = ApiProvider(api_key="sk-test", api_url="https://api.openai.com/v1")
        assert p.embed("hello") == []


class TestApiProviderChat:
    @pytest.mark.asyncio
    async def test_chat_success(self):
        p = ApiProvider(api_key="sk-test", api_url="https://api.openai.com/v1", model="gpt-4o")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hello!"}}]}
        mock_response.raise_for_status = MagicMock()

        with patch("domain.inference._internal.api_provider.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await p.chat([{"role": "user", "content": "Hi"}])
            assert result == "Hello!"

    @pytest.mark.asyncio
    async def test_chat_stream_yields_tokens(self):
        p = ApiProvider(api_key="sk-test", api_url="https://api.openai.com/v1", model="gpt-4o")

        async def mock_aiter_lines():
            for line in [
                'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                'data: {"choices":[{"delta":{"content":" world"}}]}',
                "data: [DONE]",
            ]:
                yield line

        mock_response = AsyncMock()
        mock_response.aiter_lines = mock_aiter_lines
        mock_response.raise_for_status = MagicMock()

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("domain.inference._internal.api_provider.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.stream = MagicMock(return_value=mock_stream_ctx)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            tokens = []
            async for token in p.chat_stream([{"role": "user", "content": "Hi"}]):
                tokens.append(token)
            assert tokens == ["Hello", " world"]


class TestApiProviderConnection:
    def test_test_connection_success(self):
        p = ApiProvider(api_key="sk-test", api_url="https://api.openai.com/v1", model="gpt-4o")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "gpt-4o"}, {"id": "gpt-3.5-turbo"}]}
        mock_response.raise_for_status = MagicMock()

        with patch("domain.inference._internal.api_provider.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = p.test_connection()
            assert result["status"] == "connected"
            assert result["model_available"] is True

    def test_test_connection_failure(self):
        p = ApiProvider(api_key="sk-test", api_url="https://api.openai.com/v1")

        with patch("domain.inference._internal.api_provider.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = p.test_connection()
            assert result["status"] == "error"


class TestApiProviderTokenUsage:
    def _make_p(self):
        return ApiProvider(api_key="sk-test", api_url="https://api.openai.com/v1", model="gpt-4o")

    def test_usage_less_until_chat(self):
        p = self._make_p()
        assert p.last_usage is None
        assert p.usage_tokens() == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_usage_tokens_fallback_estimate(self):
        p = self._make_p()
        usage = p.usage_tokens(
            text="Hello beautiful world", messages=[{"role": "user", "content": "Hi there"}]
        )
        assert usage["completion_tokens"] > 0
        assert usage["prompt_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    @pytest.mark.asyncio
    async def test_chat_captures_usage_from_response(self):
        p = self._make_p()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        }
        mock_response.raise_for_status = MagicMock()

        with patch("domain.inference._internal.api_provider.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await p.chat([{"role": "user", "content": "Hi"}])
            assert result == "Hello!"
            assert p.last_usage == {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}

    @pytest.mark.asyncio
    async def test_chat_empty_usage_keeps_none(self):
        p = self._make_p()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "No usage here"}}]}
        mock_response.raise_for_status = MagicMock()

        with patch("domain.inference._internal.api_provider.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            await p.chat([{"role": "user", "content": "Hi"}])
            assert p.last_usage is None

    @pytest.mark.asyncio
    async def test_chat_stream_captures_trailing_usage_chunk(self):
        p = self._make_p()

        async def mock_aiter_lines():
            yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}'
            yield 'data: {"choices":[{"delta":{"content":" world"}}]}'
            yield 'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}'
            yield "data: [DONE]"

        mock_response = AsyncMock()
        mock_response.aiter_lines = mock_aiter_lines
        mock_response.raise_for_status = MagicMock()

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("domain.inference._internal.api_provider.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.stream = MagicMock(return_value=mock_stream_ctx)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            tokens = []
            async for token in p.chat_stream([{"role": "user", "content": "Hi"}]):
                tokens.append(token)
            assert tokens == ["Hello", " world"]
            assert p.last_usage == {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}

    @pytest.mark.asyncio
    async def test_chat_stream_payload_requests_usage(self):
        p = self._make_p()
        captured = {}

        async def mock_aiter_lines():
            yield "data: [DONE]"

        mock_response = AsyncMock()
        mock_response.aiter_lines = mock_aiter_lines
        mock_response.raise_for_status = MagicMock()

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        def fake_stream(method, url, headers, json=None):
            captured["json"] = json
            return mock_stream_ctx

        with patch("domain.inference._internal.api_provider.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.stream = MagicMock(side_effect=fake_stream)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            async for _ in p.chat_stream([{"role": "user", "content": "Hi"}]):
                pass

            assert captured["json"].get("stream_options", {}).get("include_usage") is True

    def test_metadata_includes_last_usage(self):
        p = self._make_p()
        p.last_usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        assert p.metadata["last_usage"] == {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        }
