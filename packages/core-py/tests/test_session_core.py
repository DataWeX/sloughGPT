"""Tests for session core — store, retrieve, list sessions."""

import pytest
from unittest.mock import patch, MagicMock
from domains.infrastructure.session_core import SessionCore


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_feedback():
    with patch('domains.infrastructure.session_core.get_message_feedback') as mock:
        mock_fb = MagicMock()
        mock.return_value = mock_fb
        yield mock_fb


# ── store_context ─────────────────────────────────────────────────────────

class TestStoreContext:

    def test_returns_stored_status(self, mock_feedback):
        result = SessionCore.store_context("s1", [{"role": "user", "content": "hi"}])
        assert result["status"] == "stored"
        assert result["session_id"] == "s1"
        assert result["message_count"] == 1

    def test_message_count_matches(self, mock_feedback):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        result = SessionCore.store_context("s2", msgs)
        assert result["message_count"] == 3

    def test_empty_messages(self, mock_feedback):
        result = SessionCore.store_context("s3", [])
        assert result["status"] == "stored"
        assert result["message_count"] == 0

    def test_calls_store_session_context(self, mock_feedback):
        msgs = [{"role": "user", "content": "hello"}]
        SessionCore.store_context("s4", msgs)
        mock_feedback.store_session_context.assert_called_once()

    def test_converts_message_data(self, mock_feedback):
        msgs = [{"role": "assistant", "content": "response"}]
        SessionCore.store_context("s5", msgs)
        call_args = mock_feedback.store_session_context.call_args
        stored_msgs = call_args[0][1]
        assert stored_msgs[0].role == "assistant"
        assert stored_msgs[0].content == "response"

    def test_message_defaults_role_to_user(self, mock_feedback):
        msgs = [{"content": "no role"}]
        SessionCore.store_context("s6", msgs)
        call_args = mock_feedback.store_session_context.call_args
        stored_msgs = call_args[0][1]
        assert stored_msgs[0].role == "user"

    def test_message_defaults_content_to_empty(self, mock_feedback):
        msgs = [{"role": "user"}]
        SessionCore.store_context("s7", msgs)
        call_args = mock_feedback.store_session_context.call_args
        stored_msgs = call_args[0][1]
        assert stored_msgs[0].content == ""


# ── get_messages ──────────────────────────────────────────────────────────

class TestGetMessages:

    def test_returns_stored_messages(self, mock_feedback):
        msg1 = MagicMock(role="user", content="hi")
        msg2 = MagicMock(role="assistant", content="hello")
        mock_feedback.get_session_context.return_value = [msg1, msg2]
        result = SessionCore.get_messages("s1")
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "hi"}
        assert result[1] == {"role": "assistant", "content": "hello"}

    def test_returns_empty_list_when_none(self, mock_feedback):
        mock_feedback.get_session_context.return_value = None
        result = SessionCore.get_messages("no-session")
        assert result == []

    def test_returns_empty_list_when_empty(self, mock_feedback):
        mock_feedback.get_session_context.return_value = []
        result = SessionCore.get_messages("empty")
        assert result == []

    def test_preserves_order(self, mock_feedback):
        msgs = [
            MagicMock(role="user", content="first"),
            MagicMock(role="assistant", content="second"),
            MagicMock(role="user", content="third"),
        ]
        mock_feedback.get_session_context.return_value = msgs
        result = SessionCore.get_messages("s2")
        assert [m["content"] for m in result] == ["first", "second", "third"]

    def test_calls_get_session_context(self, mock_feedback):
        mock_feedback.get_session_context.return_value = []
        SessionCore.get_messages("s3")
        mock_feedback.get_session_context.assert_called_once_with("s3")


# ── list_sessions ─────────────────────────────────────────────────────────

class TestListSessions:

    def test_returns_list(self, mock_feedback):
        mock_feedback.list_conversations.return_value = [
            {"session_id": "s1", "message_count": 5},
            {"session_id": "s2", "message_count": 3},
        ]
        result = SessionCore.list_sessions()
        assert len(result) == 2
        assert result[0]["session_id"] == "s1"

    def test_calls_list_conversations(self, mock_feedback):
        mock_feedback.list_conversations.return_value = []
        SessionCore.list_sessions()
        mock_feedback.list_conversations.assert_called_once()

    def test_empty_sessions(self, mock_feedback):
        mock_feedback.list_conversations.return_value = []
        result = SessionCore.list_sessions()
        assert result == []
