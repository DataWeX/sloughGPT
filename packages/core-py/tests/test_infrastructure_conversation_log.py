"""Tests for ConversationLogger — API conversation logging."""
from __future__ import annotations

import json
import os

from domains.infrastructure.conversation_log import (
    ConversationLogger,
    capture,
    get_conversation_logger,
    reset_conversation_logger,
)


class TestConversationLogger:
    def test_record(self, tmp_path):
        logger = ConversationLogger(data_dir=tmp_path)
        result = logger.record("hello", "hi there", model="gpt2")
        assert result is not None

    def test_creates_files(self, tmp_path):
        logger = ConversationLogger(data_dir=tmp_path)
        logger.record("hello", "hi", model="test")
        assert (tmp_path / "corpus.jsonl").exists()
        assert (tmp_path / "input.txt").exists()

    def test_corpus_format(self, tmp_path):
        logger = ConversationLogger(data_dir=tmp_path)
        logger.record("q", "a", model="m")
        with open(tmp_path / "corpus.jsonl") as f:
            row = json.loads(f.readline())
        assert "messages" in row
        assert row["messages"][0]["role"] == "user"
        assert row["messages"][1]["content"] == "a"

    def test_text_format(self, tmp_path):
        logger = ConversationLogger(data_dir=tmp_path)
        logger.record("question", "answer")
        text = (tmp_path / "input.txt").read_text()
        assert "User: question" in text
        assert "Assistant: answer" in text

    def test_disabled(self, tmp_path):
        os.environ["MAN_CAPTURE_CONVERSATIONS"] = "0"
        try:
            logger = ConversationLogger(data_dir=tmp_path)
            result = logger.record("hello", "hi")
            assert result is None
        finally:
            del os.environ["MAN_CAPTURE_CONVERSATIONS"]

    def test_empty_prompt_returns_none(self, tmp_path):
        logger = ConversationLogger(data_dir=tmp_path)
        assert logger.record("", "response") is None

    def test_empty_response_returns_none(self, tmp_path):
        logger = ConversationLogger(data_dir=tmp_path)
        assert logger.record("prompt", "") is None


class TestCapture:
    def test_capture(self, tmp_path):
        reset_conversation_logger()
        # Redirect to tmp_path by creating logger manually
        logger = ConversationLogger(data_dir=tmp_path)
        from domains.infrastructure.conversation_log import _logger_lock, _logger
        import domains.infrastructure.conversation_log as mod
        mod._logger = logger
        try:
            result = capture("hello", "world", model="test")
            assert result is True
        finally:
            mod._logger = None


class TestSingleton:
    def test_get_conversation_logger(self):
        reset_conversation_logger()
        a = get_conversation_logger()
        b = get_conversation_logger()
        assert a is b
