"""Tests for ChatDomain — respond, build_prompt, log, stats."""

import json
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from domains.chat.domain import ChatDomain, ChatRequest, ChatResponse


# ── _build_prompt ────────────────────────────────────────────────────────

class TestBuildPrompt:

    def test_system_prompt_included(self):
        result = ChatDomain._build_prompt(
            "You are helpful.", [], "Hello"
        )
        assert "System: You are helpful." in result
        assert "User: Hello" in result
        assert "Assistant:" in result

    def test_all_roles_included(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Tell me a joke"},
        ]
        result = ChatDomain._build_prompt("Be concise.", msgs, "Tell me a joke")
        assert "System: Be concise." in result
        assert "User: Hi" in result
        assert "Assistant: Hello!" in result
        assert "Tell me a joke" in result

    def test_omits_last_user_message_duplication(self):
        msgs = [
            {"role": "user", "content": "Tell me a joke"},
        ]
        result = ChatDomain._build_prompt("", msgs, "Tell me a joke")
        # The last message content should appear only once
        assert result.count("Tell me a joke") == 1

    def test_no_system_prompt_skips_system_line(self):
        result = ChatDomain._build_prompt("", [], "Hi")
        assert "System:" not in result
        assert "User: Hi" in result
        assert "Assistant:" in result


# ── _log & get_recent_responses ─────────────────────────────────────────

class TestLogging:

    @pytest.fixture
    def domain(self):
        tmp = tempfile.mkdtemp()
        d = ChatDomain(log_dir=tmp)
        yield d

    def test_log_creates_file(self, domain):
        domain._log(
            user_message="hello",
            assistant_response="world",
            model="gpt2",
            temperature=0.8,
            max_tokens=256,
            session_id="s1",
            user_id="u1",
            tokens_generated=2,
            duration_ms=100,
        )
        files = list(Path(domain.log_dir).iterdir())
        assert len(files) == 1
        content = files[0].read_text()
        assert "hello" in content
        assert "world" in content
        assert "gpt2" in content

    def test_get_recent_responses_returns_entries(self, domain):
        domain._log(
            user_message="msg1", assistant_response="resp1",
            model="gpt2", temperature=0.8, max_tokens=256,
            session_id="s1", user_id="u1", tokens_generated=1, duration_ms=50,
        )
        domain._log(
            user_message="msg2", assistant_response="resp2",
            model="gpt2", temperature=0.8, max_tokens=256,
            session_id="s1", user_id="u1", tokens_generated=2, duration_ms=60,
        )
        recent = domain.get_recent_responses(limit=10)
        assert len(recent) == 2
        assert recent[0]["user_message"] == "msg1"
        assert recent[1]["assistant_response"] == "resp2"

    def test_get_recent_responses_empty_when_no_file(self, domain):
        assert domain.get_recent_responses() == []

    def test_get_recent_responses_limit(self, domain):
        for i in range(5):
            domain._log(
                user_message=f"msg{i}", assistant_response=f"resp{i}",
                model="gpt2", temperature=0.8, max_tokens=256,
                session_id="s1", user_id="u1", tokens_generated=1, duration_ms=50,
            )
        recent = domain.get_recent_responses(limit=2)
        assert len(recent) == 2
        assert recent[0]["user_message"] == "msg3"
        assert recent[1]["user_message"] == "msg4"


# ── get_stats ────────────────────────────────────────────────────────────

class TestGetStats:

    @pytest.fixture
    def domain(self):
        tmp = tempfile.mkdtemp()
        d = ChatDomain(log_dir=tmp)
        yield d

    def test_stats_empty_when_no_responses(self, domain):
        stats = domain.get_stats()
        assert stats == {"total": 0}

    def test_stats_aggregates_correctly(self, domain):
        for i in range(3):
            domain._log(
                user_message=f"msg{i}", assistant_response="x",
                model="gpt2", temperature=0.8, max_tokens=256,
                session_id="s1", user_id="u1",
                tokens_generated=(i + 1) * 10, duration_ms=(i + 1) * 100,
            )
        stats = domain.get_stats()
        assert stats["total"] == 3
        assert stats["avg_tokens"] == 20.0  # (10+20+30)/3
        assert stats["avg_duration_ms"] == 200.0  # (100+200+300)/3
        assert stats["unique_models"] == ["gpt2"]


# ── respond ──────────────────────────────────────────────────────────────

class TestRespond:

    @pytest.fixture
    def domain(self):
        tmp = tempfile.mkdtemp()
        d = ChatDomain(log_dir=tmp)
        yield d

    @pytest.mark.asyncio
    async def test_respond_returns_chat_response(self, domain):
        with patch.object(domain, "_generate", new=AsyncMock(return_value="Hello back")):
            resp = await domain.respond(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt2",
            )
            assert isinstance(resp, ChatResponse)
            assert resp.text == "Hello back"
            assert resp.session_id == "default"
            assert resp.tokens_generated == 2

    @pytest.mark.asyncio
    async def test_respond_logs_response(self, domain):
        with patch.object(domain, "_generate", new=AsyncMock(return_value="Hello back")):
            await domain.respond(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt2",
            )
            recent = domain.get_recent_responses(limit=1)
            assert len(recent) == 1
            assert recent[0]["user_message"] == "Hi"

    @pytest.mark.asyncio
    async def test_respond_sets_session_id(self, domain):
        with patch.object(domain, "_generate", new=AsyncMock(return_value="Hi")):
            resp = await domain.respond(
                messages=[{"role": "user", "content": "Hey"}],
                session_id="custom-session",
            )
            assert resp.session_id == "custom-session"

    @pytest.mark.asyncio
    async def test_respond_empty_generate_returns_no_response(self, domain):
        with patch.object(domain, "_generate", new=AsyncMock(return_value="")):
            resp = await domain.respond(
                messages=[{"role": "user", "content": "Hi"}],
            )
            assert resp.text == "[no response]"
            assert resp.tokens_generated == 0


# ── Session-id threading ────────────────────────────────────────────────

class TestSessionIdThreading:

    @pytest.fixture
    def domain(self):
        tmp = tempfile.mkdtemp()
        d = ChatDomain(log_dir=tmp)
        yield d

    @pytest.mark.asyncio
    async def test_respond_passes_session_id_to_generate(self, domain):
        """respond() must forward session_id into _generate() for KV reuse."""
        captured = {}
        async def _fake_generate(**kwargs):
            captured.update(kwargs)
            return "Hello"
        with patch.object(domain, "_generate", new=_fake_generate):
            await domain.respond(
                messages=[{"role": "user", "content": "Hi"}],
                session_id="sess-kv",
            )
        assert captured["session_id"] == "sess-kv"

    @pytest.mark.asyncio
    async def test_generate_passes_session_id_to_provider_chat(self, domain):
        """_generate() must thread session_id into provider.chat()."""
        provider = MagicMock()
        provider.chat = AsyncMock(return_value="Hello back")
        with patch("domains.models.provider.get_provider", return_value=provider):
            result = await domain._generate(
                user_msg="Hi",
                system_prompt="",
                model="gpt2",
                temperature=0.8,
                max_tokens=16,
                session_id="sess-kv",
            )
        assert result == "Hello back"
        _, kwargs = provider.chat.call_args
        assert kwargs["session_id"] == "sess-kv"

    @pytest.mark.asyncio
    async def test_generate_defaults_session_id(self, domain):
        """_generate() must default session_id so direct callers still work."""
        provider = MagicMock()
        provider.chat = AsyncMock(return_value="ok")
        with patch("domains.models.provider.get_provider", return_value=provider):
            await domain._generate(
                user_msg="Hi", system_prompt="", model="gpt2",
                temperature=0.8, max_tokens=16,
            )
        _, kwargs = provider.chat.call_args
        assert kwargs["session_id"] == "default"


# ── Factories / dataclasses ─────────────────────────────────────────────

class TestChatRequest:

    def test_defaults(self):
        r = ChatRequest(messages=[{"role": "user", "content": "Hi"}])
        assert r.model == "gpt2"
        assert r.temperature == 0.8
        assert r.max_tokens == 256

    def test_custom_values(self):
        r = ChatRequest(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt2-medium",
            temperature=0.5,
            session_id="abc",
        )
        assert r.model == "gpt2-medium"
        assert r.temperature == 0.5
        assert r.session_id == "abc"
        assert r.system_prompt == ""


class TestChatResponse:

    def test_defaults(self):
        r = ChatResponse(text="Hi", session_id="s1")
        assert r.done is True
        assert r.tokens_generated == 0
        assert r.duration_ms == 0

    def test_custom(self):
        r = ChatResponse(text="Hi", session_id="s1", tokens_generated=5, duration_ms=200)
        assert r.tokens_generated == 5
        assert r.duration_ms == 200
