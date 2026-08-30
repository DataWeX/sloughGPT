"""Meaningful tests for ChatDomain — prompt building, logging, stats, request/response construction."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from domains.chat.domain import ChatDomain, ChatRequest, ChatResponse, get_chat_domain


class TestChatRequest:
    def test_defaults(self):
        req = ChatRequest(messages=[{"role": "user", "content": "hi"}])
        assert req.model == "gpt2"
        assert req.temperature == 0.8
        assert req.max_tokens == 256
        assert req.session_id is None

    def test_custom_values(self):
        req = ChatRequest(
            messages=[],
            model="custom",
            temperature=0.1,
            max_tokens=100,
            session_id="s1",
            user_id="u1",
        )
        assert req.model == "custom"
        assert req.session_id == "s1"

    def test_system_prompt_default(self):
        req = ChatRequest(messages=[])
        assert req.system_prompt == ""

    def test_system_prompt_custom(self):
        req = ChatRequest(messages=[], system_prompt="Be helpful")
        assert req.system_prompt == "Be helpful"

    def test_messages_stored(self):
        msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        req = ChatRequest(messages=msgs)
        assert len(req.messages) == 2

    def test_user_id_default(self):
        req = ChatRequest(messages=[])
        assert req.user_id is None


class TestChatResponse:
    def test_defaults(self):
        resp = ChatResponse(text="hello", session_id="s1")
        assert resp.done is True
        assert resp.tokens_generated == 0
        assert resp.duration_ms == 0

    def test_full_construction(self):
        resp = ChatResponse(text="hi", session_id="s1", tokens_generated=5, duration_ms=100)
        assert resp.tokens_generated == 5
        assert resp.duration_ms == 100

    def test_done_can_be_false(self):
        resp = ChatResponse(text="", session_id="s", done=False)
        assert resp.done is False

    def test_text_stored(self):
        resp = ChatResponse(text="response text", session_id="s")
        assert resp.text == "response text"


class TestBuildPrompt:
    def test_system_only(self):
        result = ChatDomain._build_prompt("Be helpful", [], "Hello")
        assert result == "System: Be helpful\nUser: Hello\nAssistant:"

    def test_multi_turn(self):
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        result = ChatDomain._build_prompt("", messages, "Q2")
        assert "User: Q1" in result
        assert "Assistant: A1" in result
        assert result.endswith("User: Q2\nAssistant:")

    def test_system_in_context(self):
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        result = ChatDomain._build_prompt("", messages, "Q3")
        assert "System: Be helpful" in result
        assert "User: Q1" in result
        assert "Assistant: A1" in result
        assert result.endswith("User: Q3\nAssistant:")

    def test_no_messages(self):
        result = ChatDomain._build_prompt("", [], "Q")
        assert result == "User: Q\nAssistant:"

    def test_assistant_role_in_context(self):
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        result = ChatDomain._build_prompt("sys", messages, "Q3")
        assert "Assistant: A1" in result
        assert "User: Q1" in result
        assert result.endswith("User: Q3\nAssistant:")

    def test_unknown_role_defaults_to_user(self):
        """Unknown role is treated as 'user' when processed."""
        # Only the prior message (before last) is processed in the loop.
        # With 2 messages, cutoff=1, so only i=0 is processed.
        messages = [
            {"role": "user", "content": "first"},
            {"role": "custom", "content": "second"},
        ]
        result = ChatDomain._build_prompt("", messages, "Q")
        assert "User: first" in result

    def test_unknown_role_in_prior_messages(self):
        """Unknown role in an earlier message defaults to 'User:'."""
        messages = [
            {"role": "custom", "content": "custom content"},
            {"role": "user", "content": "follow up"},
        ]
        result = ChatDomain._build_prompt("", messages, "Q")
        assert "User: custom content" in result

    def test_missing_content_defaults_empty(self):
        messages = [{"role": "user"}]
        result = ChatDomain._build_prompt("", messages, "Q")
        assert "User: " in result

    def test_system_and_multi_turn_combined(self):
        """System prompt + prior messages (before last) are included."""
        messages = [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Follow up"},
        ]
        result = ChatDomain._build_prompt("Be verbose", messages, "Q")
        # system_prompt adds "System: Be verbose" at the top
        assert result.startswith("System: Be verbose")
        assert "System: Be concise" in result
        assert "User: Hi" in result
        assert "Assistant: Hello" in result

    def test_only_last_user_message_in_prompt(self):
        """Last message (user_msg param) is added at end, not from messages."""
        messages = [{"role": "user", "content": "old question"}]
        result = ChatDomain._build_prompt("", messages, "new question")
        assert result.endswith("User: new question\nAssistant:")

    def test_empty_system_prompt_not_added(self):
        result = ChatDomain._build_prompt("", [], "Q")
        assert not result.startswith("System:")


class TestChatDomainLogging:
    def test_log_writes_jsonl(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._log("user msg", "assistant resp", "gpt2", 0.8, 256, "s1", "u1", 10, 50)

        log_files = list((tmp_path / "logs").glob("responses_*.jsonl"))
        assert len(log_files) == 1
        line = log_files[0].read_text().strip()
        entry = json.loads(line)
        assert entry["user_message"] == "user msg"
        assert entry["assistant_response"] == "assistant resp"
        assert entry["model"] == "gpt2"
        assert entry["tokens_generated"] == 10
        assert entry["duration_ms"] == 50

    def test_log_truncates_long_messages(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        long_msg = "x" * 1000
        domain._log(long_msg, long_msg, "m", 0.5, 100, "s", "u", 0, 0)

        log_files = list((tmp_path / "logs").glob("responses_*.jsonl"))
        entry = json.loads(log_files[0].read_text().strip())
        assert len(entry["user_message"]) == 500
        assert len(entry["assistant_response"]) == 1000

    def test_get_recent_responses_empty(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        assert domain.get_recent_responses() == []

    def test_get_recent_responses_limit(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        for i in range(5):
            domain._log(f"msg{i}", f"resp{i}", "m", 0.5, 100, "s", "u", 0, 0)
        recent = domain.get_recent_responses(limit=3)
        assert len(recent) == 3
        assert recent[0]["user_message"] == "msg2"

    def test_get_recent_responses_corrupted_line(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        log_file = tmp_path / "logs" / f"responses_20260101.jsonl"
        log_file.write_text("not json\nvalid json\n")
        # This will skip the bad line
        responses = domain.get_recent_responses()
        assert isinstance(responses, list)

    def test_get_stats_empty(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        assert domain.get_stats() == {"total": 0}

    def test_get_stats_averages(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._log("m1", "r1", "gpt2", 0.8, 256, "s", "u", 10, 100)
        domain._log("m2", "r2", "gpt2", 0.5, 128, "s", "u", 20, 200)
        stats = domain.get_stats()
        assert stats["total"] == 2
        assert stats["avg_tokens"] == 15.0
        assert stats["avg_duration_ms"] == 150.0
        assert "gpt2" in stats["unique_models"]

    def test_log_creates_parent_dir(self, tmp_path):
        """Log dir is created automatically."""
        domain = ChatDomain(log_dir=str(tmp_path / "a" / "b" / "logs"))
        domain._log("msg", "resp", "m", 0.5, 100, "s", "u", 0, 0)
        log_files = list((tmp_path / "a" / "b" / "logs").glob("responses_*.jsonl"))
        assert len(log_files) == 1

    def test_log_has_timestamp(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._log("msg", "resp", "m", 0.5, 100, "s", "u", 0, 0)
        log_files = list((tmp_path / "logs").glob("responses_*.jsonl"))
        entry = json.loads(log_files[0].read_text().strip())
        assert "timestamp" in entry

    def test_log_all_fields_present(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._log("u", "a", "gpt2", 0.8, 256, "sid", "uid", 5, 100)
        log_files = list((tmp_path / "logs").glob("responses_*.jsonl"))
        entry = json.loads(log_files[0].read_text().strip())
        expected_keys = {
            "timestamp", "user_message", "assistant_response", "model",
            "temperature", "max_tokens", "session_id", "user_id",
            "tokens_generated", "duration_ms",
        }
        assert expected_keys.issubset(entry.keys())

    def test_get_recent_responses_reads_correct_day(self, tmp_path):
        """Responses are read from today's log file."""
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._log("today", "resp", "m", 0.5, 100, "s", "u", 0, 0)
        responses = domain.get_recent_responses()
        assert len(responses) == 1
        assert responses[0]["user_message"] == "today"

    def test_get_stats_multiple_models(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._log("m", "r", "model_a", 0.5, 100, "s", "u", 10, 50)
        domain._log("m", "r", "model_b", 0.5, 100, "s", "u", 10, 50)
        stats = domain.get_stats()
        assert len(stats["unique_models"]) == 2

    def test_get_recent_responses_limit_one(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        for i in range(3):
            domain._log(f"m{i}", f"r{i}", "m", 0.5, 100, "s", "u", 0, 0)
        recent = domain.get_recent_responses(limit=1)
        assert len(recent) == 1


class TestSetEngine:
    def test_set_engine(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain.set_engine("mock_engine")
        assert domain._engine == "mock_engine"

    def test_set_engine_none(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain.set_engine(None)
        assert domain._engine is None

    def test_init_default_engine(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        assert domain._engine is None

    def test_init_custom_log_dir(self, tmp_path):
        custom_dir = tmp_path / "custom_logs"
        domain = ChatDomain(log_dir=str(custom_dir))
        assert domain.log_dir == custom_dir


class TestRespond:
    @pytest.mark.asyncio
    async def test_respond_calls_generate(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._generate = AsyncMock(return_value="Hello there!")

        resp = await domain.respond(
            messages=[{"role": "user", "content": "Hi"}],
            model="test",
            session_id="s1",
        )
        assert resp.text == "Hello there!"
        assert resp.session_id == "s1"
        assert resp.tokens_generated == 2

    @pytest.mark.asyncio
    async def test_respond_empty_response(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._generate = AsyncMock(return_value="")

        resp = await domain.respond(messages=[{"role": "user", "content": "Q"}])
        assert resp.text == "[no response]"
        assert resp.tokens_generated == 0

    @pytest.mark.asyncio
    async def test_respond_logs_to_disk(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._generate = AsyncMock(return_value="response text")

        await domain.respond(
            messages=[{"role": "user", "content": "Q"}],
            user_id="u1",
        )
        log_files = list((tmp_path / "logs").glob("responses_*.jsonl"))
        assert len(log_files) == 1
        entry = json.loads(log_files[0].read_text().strip())
        assert entry["user_id"] == "u1"

    @pytest.mark.asyncio
    async def test_respond_extracts_last_user_message(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._generate = AsyncMock(return_value="ok")

        await domain.respond(
            messages=[
                {"role": "assistant", "content": "previous"},
                {"role": "user", "content": "the actual question"},
            ],
        )
        log_files = list((tmp_path / "logs").glob("responses_*.jsonl"))
        entry = json.loads(log_files[0].read_text().strip())
        assert entry["user_message"] == "the actual question"

    @pytest.mark.asyncio
    async def test_respond_default_session(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._generate = AsyncMock(return_value="ok")

        resp = await domain.respond(messages=[{"role": "user", "content": "Q"}])
        assert resp.session_id == "default"

    @pytest.mark.asyncio
    async def test_respond_default_user(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._generate = AsyncMock(return_value="ok")

        await domain.respond(messages=[{"role": "user", "content": "Q"}])
        log_files = list((tmp_path / "logs").glob("responses_*.jsonl"))
        entry = json.loads(log_files[0].read_text().strip())
        assert entry["user_id"] == "default"


class TestGetChatDomain:
    def test_singleton(self, tmp_path):
        import domains.chat.domain as mod
        old = mod._chat_domain
        try:
            mod._chat_domain = None
            a = get_chat_domain()
            b = get_chat_domain()
            assert a is b
        finally:
            mod._chat_domain = old

    def test_singleton_returns_chat_domain(self, tmp_path):
        import domains.chat.domain as mod
        old = mod._chat_domain
        try:
            mod._chat_domain = None
            d = get_chat_domain()
            assert isinstance(d, ChatDomain)
        finally:
            mod._chat_domain = old


class TestRespondWithSystemPrompt:
    @pytest.mark.asyncio
    async def test_respond_passes_system_prompt(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        captured = {}

        async def fake_generate(**kwargs):
            captured["system_prompt"] = kwargs.get("system_prompt", "")
            return "ok"

        domain._generate = fake_generate
        await domain.respond(
            messages=[{"role": "user", "content": "Q"}],
            system_prompt="Be concise",
        )
        assert captured["system_prompt"] == "Be concise"

    @pytest.mark.asyncio
    async def test_respond_passes_temperature(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        captured = {}

        async def fake_generate(**kwargs):
            captured["temperature"] = kwargs.get("temperature")
            return "ok"

        domain._generate = fake_generate
        await domain.respond(
            messages=[{"role": "user", "content": "Q"}],
            temperature=0.3,
        )
        assert captured["temperature"] == 0.3
