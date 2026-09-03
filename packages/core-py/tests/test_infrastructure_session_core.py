"""Tests for SessionCore — session context storage and retrieval."""
from __future__ import annotations

from domains.infrastructure.session_core import SessionCore


class TestSessionCore:
    def test_store_and_get(self):
        result = SessionCore.store_context("s1", [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])
        assert result["status"] == "stored"
        assert result["message_count"] == 2

        msgs = SessionCore.get_messages("s1")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"
        assert msgs[1]["content"] == "hi"

    def test_get_empty_session(self):
        msgs = SessionCore.get_messages("nonexistent")
        assert msgs == []

    def test_overwrite_session(self):
        SessionCore.store_context("s2", [{"role": "user", "content": "first"}])
        SessionCore.store_context("s2", [{"role": "user", "content": "second"}])
        msgs = SessionCore.get_messages("s2")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "second"

    def test_list_sessions(self):
        SessionCore.store_context("s3", [{"role": "user", "content": "x"}])
        sessions = SessionCore.list_sessions()
        assert isinstance(sessions, list)
