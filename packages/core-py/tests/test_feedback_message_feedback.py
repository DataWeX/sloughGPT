"""Tests for MessageFeedback — in-memory feedback and session context."""
from __future__ import annotations

from domains.feedback.message_feedback import MessageData, MessageFeedback, get_message_feedback


class TestMessageData:
    def test_creation(self):
        m = MessageData(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"


class TestMessageFeedback:
    def test_record_feedback(self):
        fb = MessageFeedback()
        entry = fb.record_feedback("msg1", "thumbs_up", session_id="s1")
        assert entry["message_id"] == "msg1"
        assert entry["rating"] == "thumbs_up"

    def test_get_feedback(self):
        fb = MessageFeedback()
        fb.record_feedback("msg1", "thumbs_up")
        result = fb.get_feedback("msg1")
        assert result is not None
        assert result["rating"] == "thumbs_up"

    def test_get_feedback_missing(self):
        fb = MessageFeedback()
        assert fb.get_feedback("nonexistent") is None

    def test_store_and_get_session_context(self):
        fb = MessageFeedback()
        msgs = [MessageData(role="user", content="hi")]
        fb.store_session_context("s1", msgs)
        result = fb.get_session_context("s1")
        assert result is not None
        assert len(result) == 1
        assert result[0].content == "hi"

    def test_clear_session_context(self):
        fb = MessageFeedback()
        fb.store_session_context("s1", [MessageData(role="user", content="hi")])
        fb.clear_session_context("s1")
        assert fb.get_session_context("s1") is None

    def test_record_regeneration(self):
        fb = MessageFeedback()
        entry = fb.record_regeneration("orig1", "new1", session_id="s1")
        assert entry["original_message_id"] == "orig1"
        assert entry["new_message_id"] == "new1"

    def test_list_conversations(self):
        fb = MessageFeedback()
        fb.store_session_context("s1", [MessageData(role="user", content="a")])
        fb.store_session_context("s2", [MessageData(role="user", content="b"), MessageData(role="assistant", content="c")])
        convs = fb.list_conversations()
        assert len(convs) == 2
        assert convs[0]["message_count"] == 1

    def test_get_stats(self):
        fb = MessageFeedback()
        fb.record_feedback("m1", "thumbs_up")
        fb.record_feedback("m2", "thumbs_down")
        fb.store_session_context("s1", [MessageData(role="user", content="x")])
        stats = fb.get_stats()
        assert stats["thumbs_up"] == 1
        assert stats["thumbs_down"] == 1
        assert stats["active_sessions"] == 1

    def test_context_truncation(self):
        fb = MessageFeedback()
        long_context = "x" * 2000
        entry = fb.record_feedback("m1", "thumbs_up", context=long_context)
        assert len(entry["context"]) == 1000


class TestSingleton:
    def test_get_message_feedback(self):
        a = get_message_feedback()
        b = get_message_feedback()
        assert a is b
