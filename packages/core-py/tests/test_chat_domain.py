"""Tests for chat.domain — ChatDomain, ChatRequest, ChatResponse."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from domains.chat.domain import (
    ChatDomain,
    ChatRequest,
    ChatResponse,
    get_chat_domain,
)


# ── ChatRequest ───────────────────────────────────────────────────────────


class TestChatRequest:

    def test_defaults(self):
        req = ChatRequest(messages=[{"role": "user", "content": "hi"}])
        assert req.model == "gpt2"
        assert req.temperature == 0.8
        assert req.max_tokens == 256

    def test_custom(self):
        req = ChatRequest(
            messages=[],
            model="llama",
            temperature=0.5,
            max_tokens=100,
            session_id="s1",
        )
        assert req.model == "llama"
        assert req.session_id == "s1"


# ── ChatResponse ──────────────────────────────────────────────────────────


class TestChatResponse:

    def test_defaults(self):
        resp = ChatResponse(text="hello", session_id="s1")
        assert resp.done is True
        assert resp.tokens_generated == 0
        assert resp.duration_ms == 0

    def test_custom(self):
        resp = ChatResponse(
            text="hi",
            session_id="s1",
            tokens_generated=5,
            duration_ms=100,
        )
        assert resp.tokens_generated == 5
        assert resp.duration_ms == 100


# ── ChatDomain ────────────────────────────────────────────────────────────


class TestChatDomain:

    def test_init(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        assert domain.log_dir.exists()

    def test_set_engine(self):
        domain = ChatDomain()
        engine = MagicMock()
        domain.set_engine(engine)
        assert domain._engine is engine

    def test_build_prompt_simple(self):
        prompt = ChatDomain._build_prompt(
            "You are helpful",
            [{"role": "user", "content": "hi"}],
            "hi",
        )
        assert "You are helpful" in prompt
        assert "User: hi" in prompt
        assert "Assistant:" in prompt

    def test_build_prompt_with_history(self):
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        prompt = ChatDomain._build_prompt("sys", messages, "q2")
        assert "System: sys" in prompt
        assert "User: q1" in prompt
        assert "Assistant: a1" in prompt

    def test_build_prompt_no_system(self):
        prompt = ChatDomain._build_prompt(
            "",
            [{"role": "user", "content": "hi"}],
            "hi",
        )
        assert "System:" not in prompt

    @pytest.mark.asyncio
    async def test_respond_calls_generate(self):
        domain = ChatDomain()
        mock_provider = AsyncMock()
        mock_provider.chat.return_value = "Hello there!"

        with patch("domains.models.provider.get_provider", return_value=mock_provider):
            resp = await domain.respond(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt2",
            )
            assert resp.text == "Hello there!"
            assert resp.session_id == "default"
            assert resp.tokens_generated > 0

    @pytest.mark.asyncio
    async def test_respond_no_provider(self):
        domain = ChatDomain()
        with patch("domains.models.provider.get_provider", return_value=None):
            resp = await domain.respond(
                messages=[{"role": "user", "content": "hi"}],
            )
            assert "Error" in resp.text

    @pytest.mark.asyncio
    async def test_respond_timeout(self):
        domain = ChatDomain()
        mock_provider = AsyncMock()
        mock_provider.chat.side_effect = asyncio.TimeoutError()

        with patch("domains.models.provider.get_provider", return_value=mock_provider):
            resp = await domain.respond(
                messages=[{"role": "user", "content": "hi"}],
            )
            assert "timed out" in resp.text

    def test_get_stats_empty(self):
        domain = ChatDomain()
        with patch.object(domain, "get_recent_responses", return_value=[]):
            stats = domain.get_stats()
            assert stats["total"] == 0

    def test_get_stats_with_responses(self):
        domain = ChatDomain()
        responses = [
            {"tokens_generated": 10, "duration_ms": 100, "model": "gpt2"},
            {"tokens_generated": 20, "duration_ms": 200, "model": "gpt2"},
        ]
        with patch.object(domain, "get_recent_responses", return_value=responses):
            stats = domain.get_stats()
            assert stats["total"] == 2
            assert stats["avg_tokens"] == 15.0
            assert stats["avg_duration_ms"] == 150.0


# ── Singleton ─────────────────────────────────────────────────────────────


class TestSingleton:

    def test_get_returns_same(self):
        with patch("domains.chat.domain.ChatDomain"):
            import domains.chat.domain as mod
            mod._chat_domain = None
            d1 = get_chat_domain()
            d2 = get_chat_domain()
            assert d1 is d2
            mod._chat_domain = None
