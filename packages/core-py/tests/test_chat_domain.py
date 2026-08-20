"""Meaningful tests for ChatDomain — prompt building, logging, stats, request/response construction."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from domains.chat.domain import ChatDomain, ChatRequest, ChatResponse


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


class TestSetEngine:
    def test_set_engine(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        domain.set_engine("mock_engine")
        assert domain._engine == "mock_engine"


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
