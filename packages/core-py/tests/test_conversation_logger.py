"""Tests for ConversationLogger — API response capture for training data."""

import json
import os
from pathlib import Path

import pytest

from domains.infrastructure.conversation_log import ConversationLogger, capture


@pytest.fixture
def logger(tmp_path: Path):
    return ConversationLogger(data_dir=tmp_path / "api_conversations")


class TestConversationLogger:
    def test_record_writes_both_formats(self, logger, tmp_path):
        logger.record("Hello", "Hi there!", model="qwen", tokens_generated=3, elapsed_ms=1500.0, temperature=0.8)
        corpus = (tmp_path / "api_conversations" / "corpus.jsonl").read_text()
        text = (tmp_path / "api_conversations" / "input.txt").read_text()
        row = json.loads(corpus.splitlines()[0])
        assert row["messages"] == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        assert row["meta"]["model"] == "qwen"
        assert row["meta"]["tokens_generated"] == 3
        assert row["meta"]["elapsed_ms"] == 1500.0
        assert row["meta"]["temperature"] == 0.8
        assert "captured_at" in row["meta"]
        assert text == "User: Hello\nAssistant: Hi there!\n\n"

    def test_record_appends_multiple_rows(self, logger, tmp_path):
        logger.record("Q1", "A1")
        logger.record("Q2", "A2")
        lines = (tmp_path / "api_conversations" / "corpus.jsonl").read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["messages"][0]["content"] == "Q2"

    def test_record_skips_empty(self, logger, tmp_path):
        assert logger.record("", "Answer") is None
        assert logger.record("Question", "") is None
        assert not (tmp_path / "api_conversations" / "corpus.jsonl").exists()

    def test_disabled_via_env(self, logger, tmp_path, monkeypatch):
        monkeypatch.setenv("MAN_CAPTURE_CONVERSATIONS", "0")
        assert logger.record("Q", "A") is None
        assert not (tmp_path / "api_conversations" / "corpus.jsonl").exists()

    def test_enabled_by_default(self, logger):
        assert logger.enabled is True

    def test_meta_extra_merged(self, logger, tmp_path):
        logger.record("Q", "A", meta={"session_id": "s1"})
        row = json.loads((tmp_path / "api_conversations" / "corpus.jsonl").read_text())
        assert row["meta"]["session_id"] == "s1"

    def test_creates_data_dir(self, tmp_path):
        target = tmp_path / "nested" / "dataset"
        ConversationLogger(data_dir=target).record("Q", "A")
        assert target.is_dir()
        assert (target / "corpus.jsonl").exists()


class TestCaptureHelper:
    def test_capture_returns_true_and_writes(self, monkeypatch, tmp_path):
        logger = ConversationLogger(data_dir=tmp_path / "api_conversations")
        monkeypatch.setattr(
            "domains.infrastructure.conversation_log._logger", logger
        )
        assert capture("Q", "A", model="m", tokens_generated=1, elapsed_ms=2.0, temperature=0.9) is True
        row = json.loads((tmp_path / "api_conversations" / "corpus.jsonl").read_text())
        assert row["meta"]["model"] == "m"
        assert row["meta"]["elapsed_ms"] == 2.0

    def test_capture_never_raises_on_error(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("disk full")
        monkeypatch.setattr(
            "domains.infrastructure.conversation_log.get_conversation_logger", boom
        )
        assert capture("Q", "A") is False


class TestGetConversationLogger:
    def test_get_conversation_logger_singleton(self, monkeypatch, tmp_path):
        """get_conversation_logger lazily creates and returns a singleton."""
        import domains.infrastructure.conversation_log as cl
        monkeypatch.setattr(cl, "_logger", None)
        logger = cl.get_conversation_logger()
        assert isinstance(logger, ConversationLogger)
        assert cl.get_conversation_logger() is logger

    def test_get_conversation_logger_returns_existing(self, monkeypatch, tmp_path):
        """get_conversation_logger returns the existing singleton."""
        import domains.infrastructure.conversation_log as cl
        logger = ConversationLogger(data_dir=tmp_path / "api_conversations")
        monkeypatch.setattr(cl, "_logger", logger)
        assert cl.get_conversation_logger() is logger
