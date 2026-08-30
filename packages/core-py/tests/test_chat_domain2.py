"""Comprehensive tests for domains.chat.domain — dataclasses, prompt building, logging, stats, singleton, respond logic."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock
from domains.chat.domain import (
    ChatDomain,
    ChatRequest,
    ChatResponse,
    get_chat_domain,
    __all__ as domain_all,
)


# ---------------------------------------------------------------------------
# ChatRequest
# ---------------------------------------------------------------------------

class TestChatRequestConstruction:
    def test_minimal(self):
        req = ChatRequest(messages=[])
        assert req.messages == []
        assert req.model == "gpt2"
        assert req.system_prompt == ""
        assert req.temperature == 0.8
        assert req.max_tokens == 256
        assert req.session_id is None
        assert req.user_id is None

    def test_all_fields(self):
        req = ChatRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="llama",
            system_prompt="You are helpful",
            temperature=0.2,
            max_tokens=512,
            session_id="sess-99",
            user_id="user-42",
        )
        assert req.model == "llama"
        assert req.system_prompt == "You are helpful"
        assert req.temperature == 0.2
        assert req.max_tokens == 512
        assert req.session_id == "sess-99"
        assert req.user_id == "user-42"

    def test_messages_are_mutable(self):
        msgs = [{"role": "user", "content": "a"}]
        req = ChatRequest(messages=msgs)
        req.messages.append({"role": "assistant", "content": "b"})
        assert len(req.messages) == 2

    def test_extreme_temperature(self):
        assert ChatRequest(messages=[], temperature=0.0).temperature == 0.0
        assert ChatRequest(messages=[], temperature=2.0).temperature == 2.0

    def test_zero_max_tokens(self):
        assert ChatRequest(messages=[], max_tokens=0).max_tokens == 0

    def test_negative_max_tokens(self):
        assert ChatRequest(messages=[], max_tokens=-1).max_tokens == -1

    def test_equality(self):
        a = ChatRequest(messages=[{"role": "user", "content": "hi"}], model="gpt2")
        b = ChatRequest(messages=[{"role": "user", "content": "hi"}], model="gpt2")
        assert a == b

    def test_inequality(self):
        a = ChatRequest(messages=[], model="gpt2")
        b = ChatRequest(messages=[], model="llama")
        assert a != b

    def test_hashable_messages_list_not_shared(self):
        a = ChatRequest(messages=[{"role": "user", "content": "x"}])
        b = ChatRequest(messages=[{"role": "user", "content": "x"}])
        assert a.messages is not b.messages

    def test_system_prompt_default_empty(self):
        req = ChatRequest(messages=[])
        assert req.system_prompt == ""


# ---------------------------------------------------------------------------
# ChatResponse
# ---------------------------------------------------------------------------

class TestChatResponseConstruction:
    def test_minimal(self):
        resp = ChatResponse(text="hi", session_id="s1")
        assert resp.text == "hi"
        assert resp.session_id == "s1"
        assert resp.done is True
        assert resp.tokens_generated == 0
        assert resp.duration_ms == 0

    def test_all_fields(self):
        resp = ChatResponse(
            text="hello",
            session_id="s2",
            done=False,
            tokens_generated=42,
            duration_ms=1500,
        )
        assert resp.done is False
        assert resp.tokens_generated == 42
        assert resp.duration_ms == 1500

    def test_empty_text(self):
        assert ChatResponse(text="", session_id="s1").text == ""

    def test_equality(self):
        a = ChatResponse(text="x", session_id="s1")
        b = ChatResponse(text="x", session_id="s1")
        assert a == b

    def test_inequality_by_text(self):
        a = ChatResponse(text="x", session_id="s1")
        b = ChatResponse(text="y", session_id="s1")
        assert a != b

    def test_inequality_by_session(self):
        a = ChatResponse(text="x", session_id="s1")
        b = ChatResponse(text="x", session_id="s2")
        assert a != b

    def test_large_tokens(self):
        resp = ChatResponse(text="t", session_id="s", tokens_generated=999999)
        assert resp.tokens_generated == 999999

    def test_done_can_be_false(self):
        resp = ChatResponse(text="streaming...", session_id="s", done=False)
        assert resp.done is False


# ---------------------------------------------------------------------------
# _build_prompt  (static method)
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_system_only_no_messages(self):
        result = ChatDomain._build_prompt("Be helpful", [], "Hello")
        assert result == "System: Be helpful\nUser: Hello\nAssistant:"

    def test_no_messages_no_system(self):
        result = ChatDomain._build_prompt("", [], "Q")
        assert result == "User: Q\nAssistant:"

    def test_multi_turn_context(self):
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        result = ChatDomain._build_prompt("", messages, "Q2")
        # cutoff = len(messages)-1 = 2, so only indices 0,1 are included
        assert "User: Q1" in result
        assert "Assistant: A1" in result
        assert result.endswith("User: Q2\nAssistant:")

    def test_system_in_context_messages(self):
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

    def test_system_prompt_overrides_context_system(self):
        messages = [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Q1"},
        ]
        result = ChatDomain._build_prompt("Override system", messages, "Q2")
        assert result.startswith("System: Override system\n")
        assert "System: Be concise" in result

    def test_cutoff_excludes_last_message(self):
        """_build_prompt always excludes the last element of messages
        because user_msg is passed separately."""
        messages = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "last-in-list"},
        ]
        result = ChatDomain._build_prompt("", messages, "final")
        assert "User: first" in result
        assert "User: last-in-list" not in result
        assert result.endswith("User: final\nAssistant:")

    def test_single_message_excluded(self):
        messages = [{"role": "user", "content": "only"}]
        result = ChatDomain._build_prompt("", messages, "Q")
        # single message is the last → excluded
        assert "only" not in result
        assert result == "User: Q\nAssistant:"

    def test_empty_messages_list(self):
        result = ChatDomain._build_prompt("sys", [], "Q")
        assert result == "System: sys\nUser: Q\nAssistant:"

    def test_unknown_role_not_in_if_chain(self):
        """Roles not in (system, user, assistant) are silently dropped
        by the if/elif chain."""
        messages = [{"role": "moderator", "content": "hello"}]
        result = ChatDomain._build_prompt("", messages, "Q")
        assert "moderator" not in result
        assert result == "User: Q\nAssistant:"

    def test_missing_role_defaults_to_user(self):
        """When role key is absent, m.get('role', 'user') returns 'user'.
        Use 2 messages so the first (missing-role) is in prior context."""
        messages = [{"content": "no role"}, {"role": "user", "content": "Q2"}]
        result = ChatDomain._build_prompt("", messages, "Q3")
        assert "User: no role" in result
        assert result.endswith("User: Q3\nAssistant:")

    def test_missing_content_defaults_to_empty(self):
        """Missing content defaults to '' via m.get('content', '')."""
        messages = [{"role": "user"}, {"role": "user", "content": "Q2"}]
        result = ChatDomain._build_prompt("", messages, "Q3")
        assert "User: \n" in result

    def test_both_keys_missing(self):
        """Both role and content missing — role defaults to 'user', content to ''."""
        messages = [{"other": "data"}, {"role": "user", "content": "Q2"}]
        result = ChatDomain._build_prompt("", messages, "Q3")
        assert "User: \n" in result

    def test_many_turns(self):
        messages = [
            {"role": "user", "content": f"Q{i}"}
            if i % 2 == 0
            else {"role": "assistant", "content": f"A{i}"}
            for i in range(20)
        ]
        result = ChatDomain._build_prompt("", messages, "Qfinal")
        # First message is included
        assert "User: Q0" in result
        # Last message (index 19) is excluded — it's the cutoff
        assert "Assistant: A19" not in result
        assert result.endswith("User: Qfinal\nAssistant:")

    def test_none_system_prompt_treated_as_falsy(self):
        result = ChatDomain._build_prompt(None, [], "Q")
        assert result == "User: Q\nAssistant:"

    def test_whitespace_content_preserved(self):
        messages = [{"role": "user", "content": "   "}]
        result = ChatDomain._build_prompt("", messages, "Q")
        # The only message is the last → excluded
        assert "   " not in result

    def test_empty_string_content_in_prior_turn(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": ""},
        ]
        result = ChatDomain._build_prompt("", messages, "Q")
        assert "User: first" in result
        # empty content message is excluded (last)
        assert result.endswith("User: Q\nAssistant:")

    def test_system_message_in_prior_turn(self):
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ]
        result = ChatDomain._build_prompt("", messages, "Q2")
        # cutoff = 2, so indices 0,1 are included; index 2 (A1) is excluded
        assert "System: rules" in result
        assert "User: Q1" in result
        assert "Assistant: A1" not in result
        assert result.endswith("User: Q2\nAssistant:")


# ---------------------------------------------------------------------------
# Logging (_log, get_recent_responses)
# ---------------------------------------------------------------------------

class TestLoggingRoundTrip:
    def test_log_writes_jsonl(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._log("user msg", "assistant resp", "gpt2", 0.8, 256, "s1", "u1", 10, 50)

        log_files = list((tmp_path / "logs").glob("responses_*.jsonl"))
        assert len(log_files) == 1
        entry = json.loads(log_files[0].read_text().strip())
        assert entry["user_message"] == "user msg"
        assert entry["assistant_response"] == "assistant resp"
        assert entry["model"] == "gpt2"
        assert entry["tokens_generated"] == 10
        assert entry["duration_ms"] == 50
        assert entry["session_id"] == "s1"
        assert entry["user_id"] == "u1"

    def test_log_truncates_user_message_500(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._log("A" * 600, "short", "m", 0.5, 100, "s", "u", 0, 0)
        responses = domain.get_recent_responses()
        assert len(responses[0]["user_message"]) == 500

    def test_log_truncates_assistant_response_1000(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._log("short", "B" * 1200, "m", 0.5, 100, "s", "u", 0, 0)
        responses = domain.get_recent_responses()
        assert len(responses[0]["assistant_response"]) == 1000

    def test_log_fields_present(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._log("u", "a", "gpt2", 0.8, 256, "s1", "u1", 5, 100)
        entry = domain.get_recent_responses()[0]
        required_keys = {
            "timestamp", "user_message", "assistant_response", "model",
            "temperature", "max_tokens", "session_id", "user_id",
            "tokens_generated", "duration_ms",
        }
        assert required_keys.issubset(set(entry.keys()))

    def test_timestamp_is_iso_format(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._log("u", "a", "m", 0.5, 100, "s", "u", 0, 0)
        ts = domain.get_recent_responses()[0]["timestamp"]
        assert "T" in ts

    def test_multiple_logs_same_day(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        for i in range(10):
            domain._log(f"msg{i}", f"resp{i}", "m", 0.5, 100, "s", "u", i, i * 10)
        responses = domain.get_recent_responses(limit=100)
        assert len(responses) == 10
        assert responses[0]["user_message"] == "msg0"
        assert responses[9]["user_message"] == "msg9"

    def test_get_recent_responses_empty(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        assert domain.get_recent_responses() == []

    def test_get_recent_responses_limit(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        for i in range(5):
            domain._log(f"m{i}", f"r{i}", "m", 0.5, 100, "s", "u", 0, 0)
        recent = domain.get_recent_responses(limit=3)
        assert len(recent) == 3
        assert recent[0]["user_message"] == "m2"

    def test_get_recent_responses_limit_1(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        for i in range(5):
            domain._log(f"m{i}", f"r{i}", "m", 0.5, 100, "s", "u", 0, 0)
        recent = domain.get_recent_responses(limit=1)
        assert len(recent) == 1
        assert recent[0]["user_message"] == "m4"

    def test_get_recent_responses_limit_exceeds_count(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._log("m", "r", "m", 0.5, 100, "s", "u", 0, 0)
        recent = domain.get_recent_responses(limit=1000)
        assert len(recent) == 1

    def test_corrupted_jsonl_skips_bad_lines(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        # get_recent_responses reads today's file — write to that
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        log_file = domain.log_dir / f"responses_{today}.jsonl"
        log_file.write_text('{"valid": true}\nnot json\n')
        responses = domain.get_recent_responses()
        assert len(responses) == 1
        assert responses[0]["valid"] is True

    def test_empty_log_file(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        log_file = domain.log_dir / "responses_20260101.jsonl"
        log_file.write_text("")
        assert domain.get_recent_responses() == []

    def test_only_whitespace_log_file(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        log_file = domain.log_dir / "responses_20260101.jsonl"
        log_file.write_text("   \n  \n")
        assert domain.get_recent_responses() == []


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestStatsComputation:
    def test_empty(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        assert domain.get_stats() == {"total": 0}

    def test_single_entry(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._log("m", "r", "gpt2", 0.8, 256, "s", "u", 10, 200)
        stats = domain.get_stats()
        assert stats["total"] == 1
        assert stats["avg_tokens"] == 10.0
        assert stats["avg_duration_ms"] == 200.0
        assert stats["unique_models"] == ["gpt2"]

    def test_multiple_models(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._log("m", "r", "gpt2", 0.8, 256, "s", "u", 10, 100)
        domain._log("m", "r", "llama", 0.5, 128, "s", "u", 20, 300)
        domain._log("m", "r", "gpt2", 0.8, 256, "s", "u", 30, 500)
        stats = domain.get_stats()
        assert stats["total"] == 3
        assert stats["avg_tokens"] == 20.0
        assert stats["avg_duration_ms"] == 300.0
        assert set(stats["unique_models"]) == {"gpt2", "llama"}

    def test_zero_tokens_and_duration(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._log("m", "r", "m", 0.5, 100, "s", "u", 0, 0)
        domain._log("m", "r", "m", 0.5, 100, "s", "u", 0, 0)
        stats = domain.get_stats()
        assert stats["avg_tokens"] == 0.0
        assert stats["avg_duration_ms"] == 0.0

    def test_stats_uses_limit_100(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        for i in range(150):
            domain._log("m", "r", "m", 0.5, 100, "s", "u", i, i)
        stats = domain.get_stats()
        assert stats["total"] == 100
        # last 100 entries: i=50..149, avg tokens = (50+149)/2 = 99.5
        assert stats["avg_tokens"] == 99.5
        assert stats["avg_duration_ms"] == 99.5

    def test_unique_models_dedup(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._log("m", "r", "gpt2", 0.8, 256, "s", "u", 0, 0)
        domain._log("m", "r", "gpt2", 0.8, 256, "s", "u", 0, 0)
        stats = domain.get_stats()
        assert len(stats["unique_models"]) == 1


# ---------------------------------------------------------------------------
# ChatDomain __init__
# ---------------------------------------------------------------------------

class TestChatDomainInit:
    def test_creates_log_dir(self, tmp_path):
        log_dir = tmp_path / "custom_logs"
        domain = ChatDomain(log_dir=str(log_dir))
        assert log_dir.exists()
        assert log_dir.is_dir()

    def test_default_engine_is_none(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        assert domain._engine is None

    def test_log_dir_anchored_to_repo_root(self):
        domain = ChatDomain(log_dir="test_data/logs")
        assert domain.log_dir.exists()
        assert domain.log_dir.name == "logs"

    def test_engine_can_be_set_via_init(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"), engine="my_engine")
        assert domain._engine == "my_engine"


# ---------------------------------------------------------------------------
# set_engine
# ---------------------------------------------------------------------------

class TestSetEngine:
    def test_set_engine(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain.set_engine("some_engine")
        assert domain._engine == "some_engine"

    def test_set_engine_overwrites(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain.set_engine("engine1")
        domain.set_engine("engine2")
        assert domain._engine == "engine2"

    def test_set_engine_to_none(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain.set_engine("engine")
        domain.set_engine(None)
        assert domain._engine is None


# ---------------------------------------------------------------------------
# get_chat_domain singleton
# ---------------------------------------------------------------------------

class TestGetChatDomainSingleton:
    def test_returns_same_instance(self):
        import domains.chat.domain as mod
        original = mod._chat_domain
        mod._chat_domain = None
        try:
            d1 = get_chat_domain()
            d2 = get_chat_domain()
            assert d1 is d2
        finally:
            mod._chat_domain = original

    def test_returns_chat_domain_type(self):
        import domains.chat.domain as mod
        original = mod._chat_domain
        mod._chat_domain = None
        try:
            result = get_chat_domain()
            assert isinstance(result, ChatDomain)
        finally:
            mod._chat_domain = original


# ---------------------------------------------------------------------------
# respond (async, mocked _generate)
# ---------------------------------------------------------------------------

class TestRespondLogic:
    @pytest.mark.asyncio
    async def test_finds_last_user_message(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._generate = AsyncMock(return_value="ok")
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        resp = await domain.respond(messages=messages, session_id="s1")
        assert resp.text == "ok"
        call_kwargs = domain._generate.call_args
        assert call_kwargs.kwargs["user_msg"] == "second"

    @pytest.mark.asyncio
    async def test_no_user_message_returns_empty_string(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._generate = AsyncMock(return_value="ok")
        messages = [{"role": "assistant", "content": "hello"}]
        resp = await domain.respond(messages=messages, session_id="s1")
        assert resp.text == "ok"
        call_kwargs = domain._generate.call_args
        assert call_kwargs.kwargs["user_msg"] == ""

    @pytest.mark.asyncio
    async def test_empty_messages(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._generate = AsyncMock(return_value="ok")
        resp = await domain.respond(messages=[], session_id="s1")
        assert resp.text == "ok"
        call_kwargs = domain._generate.call_args
        assert call_kwargs.kwargs["user_msg"] == ""

    @pytest.mark.asyncio
    async def test_token_counting(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._generate = AsyncMock(return_value="one two three four five")
        resp = await domain.respond(
            messages=[{"role": "user", "content": "Q"}],
            session_id="s1",
        )
        assert resp.tokens_generated == 5

    @pytest.mark.asyncio
    async def test_token_count_empty_response(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._generate = AsyncMock(return_value="")
        resp = await domain.respond(
            messages=[{"role": "user", "content": "Q"}],
            session_id="s1",
        )
        assert resp.tokens_generated == 0
        assert resp.text == "[no response]"

    @pytest.mark.asyncio
    async def test_none_response_defaults_to_no_response(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._generate = AsyncMock(return_value=None)
        resp = await domain.respond(
            messages=[{"role": "user", "content": "Q"}],
            session_id="s1",
        )
        assert resp.text == "[no response]"
        assert resp.tokens_generated == 0

    @pytest.mark.asyncio
    async def test_duration_ms_non_negative(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._generate = AsyncMock(return_value="ok")
        resp = await domain.respond(
            messages=[{"role": "user", "content": "Q"}],
            session_id="s1",
        )
        assert resp.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_session_id_passed_through(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._generate = AsyncMock(return_value="ok")
        resp = await domain.respond(
            messages=[{"role": "user", "content": "Q"}],
            session_id="custom-session",
        )
        assert resp.session_id == "custom-session"

    @pytest.mark.asyncio
    async def test_logs_to_disk(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._generate = AsyncMock(return_value="response")
        await domain.respond(
            messages=[{"role": "user", "content": "Q"}],
            model="test-model",
            session_id="s1",
            user_id="u1",
        )
        responses = domain.get_recent_responses()
        assert len(responses) == 1
        assert responses[0]["model"] == "test-model"
        assert responses[0]["user_id"] == "u1"

    @pytest.mark.asyncio
    async def test_response_dataclass_fields(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._generate = AsyncMock(return_value="hello world")
        resp = await domain.respond(
            messages=[{"role": "user", "content": "Q"}],
            session_id="s1",
        )
        assert isinstance(resp, ChatResponse)
        assert resp.done is True
        assert resp.text == "hello world"
        assert resp.session_id == "s1"

    @pytest.mark.asyncio
    async def test_generate_receives_system_prompt(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._generate = AsyncMock(return_value="ok")
        await domain.respond(
            messages=[{"role": "user", "content": "Q"}],
            system_prompt="Be concise",
            session_id="s1",
        )
        call_kwargs = domain._generate.call_args
        assert call_kwargs.kwargs["system_prompt"] == "Be concise"

    @pytest.mark.asyncio
    async def test_generate_receives_temperature_and_max_tokens(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._generate = AsyncMock(return_value="ok")
        await domain.respond(
            messages=[{"role": "user", "content": "Q"}],
            temperature=0.1,
            max_tokens=64,
            session_id="s1",
        )
        call_kwargs = domain._generate.call_args
        assert call_kwargs.kwargs["temperature"] == 0.1
        assert call_kwargs.kwargs["max_tokens"] == 64

    @pytest.mark.asyncio
    async def test_multiple_user_messages_picks_last(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain._generate = AsyncMock(return_value="ok")
        messages = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        await domain.respond(messages=messages, session_id="s1")
        call_kwargs = domain._generate.call_args
        assert call_kwargs.kwargs["user_msg"] == "third"


# ---------------------------------------------------------------------------
# __all__ exports
# ---------------------------------------------------------------------------

class TestAllExports:
    def test_all_exports(self):
        assert "ChatRequest" in domain_all
        assert "ChatResponse" in domain_all
        assert "ChatDomain" in domain_all
        assert "get_chat_domain" in domain_all
        assert len(domain_all) == 4
