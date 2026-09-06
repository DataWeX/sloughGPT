"""Tests for domains.chat.domain — ChatDomain, ChatRequest, ChatResponse."""

from __future__ import annotations

import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch

from domains.chat.domain import (
    ChatRequest,
    ChatResponse,
    ChatDomain,
)


# ── Dataclasses ───────────────────────────────────────────────────────────────

class TestDataclasses:
    def test_chat_request_defaults(self):
        req = ChatRequest(messages=[{"role": "user", "content": "hi"}])
        assert req.model == "gpt2"
        assert req.temperature == 0.8
        assert req.max_tokens == 256

    def test_chat_response_defaults(self):
        resp = ChatResponse(text="hello", session_id="s1")
        assert resp.done is True
        assert resp.tokens_generated == 0
        assert resp.duration_ms == 0


# ── ChatDomain._build_prompt ─────────────────────────────────────────────────

class TestBuildPrompt:
    def test_simple(self):
        prompt = ChatDomain._build_prompt(
            system_prompt="",
            messages=[{"role": "user", "content": "Hello"}],
            user_msg="Hello",
        )
        assert "User: Hello" in prompt
        assert "Assistant:" in prompt

    def test_with_system(self):
        prompt = ChatDomain._build_prompt(
            system_prompt="You are helpful.",
            messages=[{"role": "user", "content": "Hi"}],
            user_msg="Hi",
        )
        assert "System: You are helpful." in prompt

    def test_conversation_history(self):
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        prompt = ChatDomain._build_prompt(
            system_prompt="",
            messages=messages,
            user_msg="Q2",
        )
        assert "User: Q1" in prompt
        assert "Assistant: A1" in prompt
        assert "User: Q2" in prompt

    def test_system_message_in_history(self):
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ]
        prompt = ChatDomain._build_prompt(
            system_prompt="",
            messages=messages,
            user_msg="Hi",
        )
        assert "System: Be concise." in prompt


# ── ChatDomain ────────────────────────────────────────────────────────────────

class TestChatDomain:
    def test_log_and_recent(self, tmp_path):
        from domains.feedback.response_tracker import ResponseTracker
        tracker = ResponseTracker(log_dir=str(tmp_path / "logs"))
        with patch("domains.feedback.response_tracker.get_response_tracker", return_value=tracker):
            domain = ChatDomain(log_dir=str(tmp_path / "logs"))
            domain._log(
                user_message="hello",
                assistant_response="world",
                model="gpt2",
                temperature=0.8,
                max_tokens=256,
                session_id="s1",
                user_id="u1",
                tokens_generated=5,
                duration_ms=100,
            )
            recent = domain.get_recent_responses()
        assert len(recent) == 1
        assert recent[0]["user_message"] == "hello"
        assert recent[0]["model"] == "gpt2"

    def test_get_recent_limit(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        for i in range(5):
            domain._log(
                user_message=f"msg {i}",
                assistant_response=f"resp {i}",
                model="gpt2",
                temperature=0.8,
                max_tokens=256,
                session_id="s1",
                user_id="u1",
                tokens_generated=1,
                duration_ms=10,
            )
        recent = domain.get_recent_responses(limit=2)
        assert len(recent) == 2
        assert recent[0]["user_message"] == "msg 3"

    def test_get_stats_empty(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        stats = domain.get_stats()
        assert stats == {"total": 0}

    def test_get_stats(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        for i in range(3):
            domain._log(
                user_message="hi",
                assistant_response="hello",
                model="gpt2",
                temperature=0.8,
                max_tokens=256,
                session_id="s1",
                user_id="u1",
                tokens_generated=10 + i,
                duration_ms=100 + i,
            )
        stats = domain.get_stats()
        assert stats["total"] == 3
        assert stats["avg_tokens"] == 11.0
        assert "gpt2" in stats["unique_models"]

    def test_set_engine(self, tmp_path):
        domain = ChatDomain(log_dir=str(tmp_path / "logs"))
        assert domain._engine is None
        domain.set_engine("mock_engine")
        assert domain._engine == "mock_engine"
