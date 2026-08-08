"""Tests for domains.feedback.message_feedback — in-memory feedback store."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import pytest

from domains.feedback.message_feedback import (
    MessageData,
    MessageFeedback,
    get_message_feedback,
)


class TestMessageData:
    def test_creation(self):
        msg = MessageData(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_equality(self):
        a = MessageData(role="assistant", content="Hi")
        b = MessageData(role="assistant", content="Hi")
        assert a == b

    def test_inequality(self):
        a = MessageData(role="user", content="Hello")
        b = MessageData(role="assistant", content="Hello")
        assert a != b


class TestRecordFeedback:
    def test_basic(self):
        mf = MessageFeedback()
        entry = mf.record_feedback("msg1", "thumbs_up", session_id="s1")
        assert entry["message_id"] == "msg1"
        assert entry["rating"] == "thumbs_up"
        assert entry["session_id"] == "s1"
        assert "timestamp" in entry

    def test_overwrites_same_message(self):
        mf = MessageFeedback()
        mf.record_feedback("msg1", "thumbs_up")
        mf.record_feedback("msg1", "thumbs_down")
        fb = mf.get_feedback("msg1")
        assert fb["rating"] == "thumbs_down"

    def test_context_truncation(self):
        mf = MessageFeedback()
        long_ctx = "x" * 2000
        entry = mf.record_feedback("msg1", "thumbs_up", context=long_ctx)
        assert len(entry["context"]) == 1000

    def test_context_not_truncated(self):
        mf = MessageFeedback()
        short_ctx = "good"
        entry = mf.record_feedback("msg1", "thumbs_up", context=short_ctx)
        assert entry["context"] == "good"

    def test_no_context(self):
        mf = MessageFeedback()
        entry = mf.record_feedback("msg1", "thumbs_up")
        assert "context" not in entry


class TestGetFeedback:
    def test_existing(self):
        mf = MessageFeedback()
        mf.record_feedback("msg1", "thumbs_up")
        fb = mf.get_feedback("msg1")
        assert fb is not None
        assert fb["rating"] == "thumbs_up"

    def test_nonexistent(self):
        mf = MessageFeedback()
        assert mf.get_feedback("missing") is None


class TestSessionContext:
    def test_store_and_get(self):
        mf = MessageFeedback()
        msgs = [MessageData(role="user", content="Hi"), MessageData(role="assistant", content="Hello")]
        mf.store_session_context("sess1", msgs)
        result = mf.get_session_context("sess1")
        assert result is not None
        assert len(result) == 2
        assert result[0].role == "user"

    def test_returns_reference(self):
        mf = MessageFeedback()
        msgs = [MessageData(role="user", content="Hi")]
        mf.store_session_context("sess1", msgs)
        result = mf.get_session_context("sess1")
        result.append(MessageData(role="assistant", content="Added"))
        original = mf.get_session_context("sess1")
        assert len(original) == 2

    def test_nonexistent(self):
        mf = MessageFeedback()
        assert mf.get_session_context("missing") is None

    def test_clear(self):
        mf = MessageFeedback()
        mf.store_session_context("sess1", [MessageData(role="user", content="Hi")])
        mf.clear_session_context("sess1")
        assert mf.get_session_context("sess1") is None

    def test_clear_nonexistent(self):
        mf = MessageFeedback()
        mf.clear_session_context("missing")


class TestRecordRegeneration:
    def test_basic(self):
        mf = MessageFeedback()
        entry = mf.record_regeneration("orig1", "new1", session_id="s1")
        assert entry["original_message_id"] == "orig1"
        assert entry["new_message_id"] == "new1"
        assert entry["session_id"] == "s1"
        assert "timestamp" in entry

    def test_overwrites_same_original(self):
        mf = MessageFeedback()
        mf.record_regeneration("orig1", "new1")
        mf.record_regeneration("orig1", "new2")
        stats = mf.get_stats()
        assert stats["total_regenerations"] == 1


class TestListConversations:
    def test_empty(self):
        mf = MessageFeedback()
        assert mf.list_conversations() == []

    def test_with_sessions(self):
        mf = MessageFeedback()
        mf.store_session_context("s1", [MessageData(role="user", content="Hi"), MessageData(role="assistant", content="Hello")])
        mf.store_session_context("s2", [MessageData(role="user", content="Bye")])
        convos = mf.list_conversations()
        assert len(convos) == 2
        ids = {c["session_id"] for c in convos}
        assert ids == {"s1", "s2"}
        for c in convos:
            assert "message_count" in c


class TestGetStats:
    def test_empty(self):
        mf = MessageFeedback()
        stats = mf.get_stats()
        assert stats["total_feedback"] == 0
        assert stats["thumbs_up"] == 0
        assert stats["thumbs_down"] == 0
        assert stats["total_regenerations"] == 0
        assert stats["active_sessions"] == 0

    def test_mixed(self):
        mf = MessageFeedback()
        mf.record_feedback("m1", "thumbs_up")
        mf.record_feedback("m2", "thumbs_down")
        mf.record_feedback("m3", "thumbs_up")
        mf.record_regeneration("o1", "n1")
        mf.store_session_context("s1", [MessageData(role="user", content="Hi")])
        stats = mf.get_stats()
        assert stats["total_feedback"] == 3
        assert stats["thumbs_up"] == 2
        assert stats["thumbs_down"] == 1
        assert stats["total_regenerations"] == 1
        assert stats["active_sessions"] == 1


class TestConcurrency:
    def test_concurrent_feedback(self):
        mf = MessageFeedback()
        errors = []

        def writer(n):
            try:
                for i in range(50):
                    mf.record_feedback(f"msg_{n}_{i}", "thumbs_up")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert mf.get_stats()["total_feedback"] == 200

    def test_concurrent_sessions(self):
        mf = MessageFeedback()
        errors = []

        def writer(n):
            try:
                for i in range(20):
                    mf.store_session_context(f"s_{n}_{i}", [MessageData(role="user", content=f"msg {i}")])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(mf.list_conversations()) == 80


class TestSingleton:
    def test_same_instance(self):
        a = get_message_feedback()
        b = get_message_feedback()
        assert a is b

    def test_singleton_is_message_feedback(self):
        mf = get_message_feedback()
        assert isinstance(mf, MessageFeedback)
