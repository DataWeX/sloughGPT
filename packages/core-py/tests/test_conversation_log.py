"""Tests for ConversationLogger — recording exchanges, file output, enable/disable.

Covers:
  - record() writes to corpus.jsonl and input.txt
  - record() returns bytes written (1) or None when disabled
  - Enabled/disabled via env var
  - Empty prompt/response returns None
  - Meta dict is merged into corpus row
  - capture() wrapper never raises
  - get_conversation_logger() returns singleton
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from domains.infrastructure.conversation_log import (
    ConversationLogger,
    get_conversation_logger,
    capture,
)


@pytest.fixture
def tmp_log_dir(tmp_path):
    return tmp_path / "conv_log"


class TestConversationLogger:
    def test_init_creates_dir(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        assert tmp_log_dir.exists()

    def test_record_writes_corpus(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        result = logger.record("Hello", "Hi there", model="gpt2")
        assert result == 1
        lines = logger.corpus_path.read_text().strip().split("\n")
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["messages"][0]["content"] == "Hello"
        assert row["messages"][1]["content"] == "Hi there"
        assert row["meta"]["model"] == "gpt2"

    def test_record_writes_text(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Hello", "Hi there")
        text = logger.text_path.read_text()
        assert "User: Hello" in text
        assert "Assistant: Hi there" in text

    def test_record_returns_one(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        assert logger.record("Q", "A") == 1

    def test_record_empty_prompt_returns_none(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        assert logger.record("", "response") is None

    def test_record_empty_response_returns_none(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        assert logger.record("prompt", "") is None

    def test_record_whitespace_only_returns_none(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        assert logger.record("  ", "  ") is None

    def test_record_disabled(self, tmp_log_dir, monkeypatch):
        monkeypatch.setenv("MAN_CAPTURE_CONVERSATIONS", "0")
        logger = ConversationLogger(tmp_log_dir)
        assert logger.record("Q", "A") is None

    def test_enabled_default(self, tmp_log_dir, monkeypatch):
        monkeypatch.delenv("MAN_CAPTURE_CONVERSATIONS", raising=False)
        logger = ConversationLogger(tmp_log_dir)
        assert logger.enabled is True

    def test_enabled_disabled(self, tmp_log_dir, monkeypatch):
        monkeypatch.setenv("MAN_CAPTURE_CONVERSATIONS", "0")
        logger = ConversationLogger(tmp_log_dir)
        assert logger.enabled is False

    def test_record_with_meta(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Q", "A", meta={"custom": "value"})
        row = json.loads(logger.corpus_path.read_text().strip())
        assert row["meta"]["custom"] == "value"

    def test_record_with_temperature(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Q", "A", temperature=0.8)
        row = json.loads(logger.corpus_path.read_text().strip())
        assert row["meta"]["temperature"] == 0.8

    def test_multiple_records(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Q1", "A1")
        logger.record("Q2", "A2")
        lines = logger.corpus_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_meta_merged(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Q", "A", meta={"extra": 1}, model="test")
        row = json.loads(logger.corpus_path.read_text().strip())
        assert row["meta"]["model"] == "test"
        assert row["meta"]["extra"] == 1


class TestCapture:
    def test_capture_returns_true(self, tmp_log_dir):
        from domains.infrastructure import conversation_log
        old = conversation_log._logger
        try:
            conversation_log._logger = ConversationLogger(tmp_log_dir)
            assert capture("Q", "A") is True
        finally:
            conversation_log._logger = old

    def test_capture_returns_false_when_disabled(self, tmp_log_dir, monkeypatch):
        monkeypatch.setenv("MAN_CAPTURE_CONVERSATIONS", "0")
        from domains.infrastructure import conversation_log
        old = conversation_log._logger
        try:
            conversation_log._logger = ConversationLogger(tmp_log_dir)
            assert capture("Q", "A") is False
        finally:
            conversation_log._logger = old


class TestSingleton:
    def test_get_returns_same_instance(self):
        a = get_conversation_logger()
        b = get_conversation_logger()
        assert a is b
