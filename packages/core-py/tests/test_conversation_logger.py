"""Tests for ConversationLogger — API response capture for training data."""

import json
import os
from pathlib import Path
from datetime import datetime, timezone

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

    def test_record_returns_byte_count(self, logger, tmp_path):
        result = logger.record("Q", "A")
        assert result is not None
        assert result >= 1

    def test_record_returns_none_when_disabled(self, logger, tmp_path, monkeypatch):
        monkeypatch.setenv("MAN_CAPTURE_CONVERSATIONS", "0")
        assert logger.record("Q", "A") is None

    def test_captured_at_is_valid_iso(self, logger, tmp_path):
        logger.record("Q", "A")
        row = json.loads((tmp_path / "api_conversations" / "corpus.jsonl").read_text())
        ts = row["meta"]["captured_at"]
        # Should parse without error
        datetime.fromisoformat(ts)

    def test_default_model_unknown(self, logger, tmp_path):
        logger.record("Q", "A")
        row = json.loads((tmp_path / "api_conversations" / "corpus.jsonl").read_text())
        assert row["meta"]["model"] == "unknown"

    def test_default_tokens_zero(self, logger, tmp_path):
        logger.record("Q", "A")
        row = json.loads((tmp_path / "api_conversations" / "corpus.jsonl").read_text())
        assert row["meta"]["tokens_generated"] == 0

    def test_default_elapsed_zero(self, logger, tmp_path):
        logger.record("Q", "A")
        row = json.loads((tmp_path / "api_conversations" / "corpus.jsonl").read_text())
        assert row["meta"]["elapsed_ms"] == 0.0

    def test_temperature_none_by_default(self, logger, tmp_path):
        logger.record("Q", "A")
        row = json.loads((tmp_path / "api_conversations" / "corpus.jsonl").read_text())
        assert row["meta"]["temperature"] is None

    def test_text_format_exact(self, logger, tmp_path):
        logger.record("What is 2+2?", "4")
        text = (tmp_path / "api_conversations" / "input.txt").read_text()
        assert text == "User: What is 2+2?\nAssistant: 4\n\n"

    def test_whitespace_only_prompt_skipped(self, logger, tmp_path):
        assert logger.record("   ", "Answer") is None

    def test_whitespace_only_response_skipped(self, logger, tmp_path):
        assert logger.record("Question", "   ") is None

    def test_strips_whitespace(self, logger, tmp_path):
        logger.record("  Hello  ", "  World  ")
        row = json.loads((tmp_path / "api_conversations" / "corpus.jsonl").read_text())
        assert row["messages"][0]["content"] == "Hello"
        assert row["messages"][1]["content"] == "World"

    def test_jsonl_is_valid_json_per_line(self, logger, tmp_path):
        for i in range(5):
            logger.record(f"Q{i}", f"A{i}")
        lines = (tmp_path / "api_conversations" / "corpus.jsonl").read_text().splitlines()
        for line in lines:
            row = json.loads(line)
            assert "messages" in row
            assert "meta" in row

    def test_multiple_meta_keys(self, logger, tmp_path):
        logger.record("Q", "A", meta={"k1": "v1", "k2": 42})
        row = json.loads((tmp_path / "api_conversations" / "corpus.jsonl").read_text())
        assert row["meta"]["k1"] == "v1"
        assert row["meta"]["k2"] == 42

    def test_special_characters_in_prompt(self, logger, tmp_path):
        logger.record("Hello\nWorld\t!", "Response\nwith\nnewlines")
        row = json.loads((tmp_path / "api_conversations" / "corpus.jsonl").read_text())
        assert "\n" in row["messages"][0]["content"]
        assert "\t" in row["messages"][0]["content"]

    def test_unicode_content(self, logger, tmp_path):
        logger.record("日本語テスト", "応答")
        row = json.loads((tmp_path / "api_conversations" / "corpus.jsonl").read_text())
        assert row["messages"][0]["content"] == "日本語テスト"
        assert row["messages"][1]["content"] == "応答"

    def test_corpus_path_location(self, logger, tmp_path):
        assert logger.corpus_path == tmp_path / "api_conversations" / "corpus.jsonl"

    def test_text_path_location(self, logger, tmp_path):
        assert logger.text_path == tmp_path / "api_conversations" / "input.txt"

    def test_data_dir_set(self, logger, tmp_path):
        assert logger.data_dir == tmp_path / "api_conversations"

    def test_lock_exists(self, logger):
        import threading
        assert isinstance(logger._lock, type(threading.Lock()))

    def test_many_records(self, logger, tmp_path):
        for i in range(100):
            logger.record(f"Q{i}", f"A{i}")
        lines = (tmp_path / "api_conversations" / "corpus.jsonl").read_text().splitlines()
        assert len(lines) == 100

    def test_text_appends(self, logger, tmp_path):
        logger.record("Q1", "A1")
        logger.record("Q2", "A2")
        text = (tmp_path / "api_conversations" / "input.txt").read_text()
        assert "Q1" in text
        assert "Q2" in text


# ── TestCaptureHelper ────────────────────────────────────────────────────────


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

    def test_capture_passes_all_params(self, monkeypatch, tmp_path):
        logger = ConversationLogger(data_dir=tmp_path / "api_conversations")
        monkeypatch.setattr("domains.infrastructure.conversation_log._logger", logger)
        capture("Q", "A", model="test", tokens_generated=10, elapsed_ms=50.0, temperature=0.7, meta={"key": "val"})
        row = json.loads((tmp_path / "api_conversations" / "corpus.jsonl").read_text())
        assert row["meta"]["model"] == "test"
        assert row["meta"]["tokens_generated"] == 10
        assert row["meta"]["key"] == "val"

    def test_capture_returns_false_when_disabled(self, monkeypatch, tmp_path):
        logger = ConversationLogger(data_dir=tmp_path / "api_conversations")
        monkeypatch.setattr("domains.infrastructure.conversation_log._logger", logger)
        monkeypatch.setenv("MAN_CAPTURE_CONVERSATIONS", "0")
        assert capture("Q", "A") is False

    def test_capture_returns_false_on_empty(self, monkeypatch, tmp_path):
        logger = ConversationLogger(data_dir=tmp_path / "api_conversations")
        monkeypatch.setattr("domains.infrastructure.conversation_log._logger", logger)
        assert capture("", "A") is False
        assert capture("Q", "") is False

    def test_capture_with_none_response(self, monkeypatch, tmp_path):
        logger = ConversationLogger(data_dir=tmp_path / "api_conversations")
        monkeypatch.setattr("domains.infrastructure.conversation_log._logger", logger)
        # Should not raise
        result = capture("Q", None)
        assert result is False


# ── TestGetConversationLogger ────────────────────────────────────────────────


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

    def test_reset_clears_singleton(self, monkeypatch, tmp_path):
        import domains.infrastructure.conversation_log as cl
        logger = ConversationLogger(data_dir=tmp_path / "api_conversations")
        monkeypatch.setattr(cl, "_logger", logger)
        cl.reset_conversation_logger()
        assert cl._logger is None

    def test_reset_then_get_creates_new(self, monkeypatch, tmp_path):
        import domains.infrastructure.conversation_log as cl
        logger = ConversationLogger(data_dir=tmp_path / "api_conversations")
        monkeypatch.setattr(cl, "_logger", logger)
        cl.reset_conversation_logger()
        new_logger = cl.get_conversation_logger()
        assert new_logger is not logger

    def test_reset_is_idempotent(self, monkeypatch):
        import domains.infrastructure.conversation_log as cl
        monkeypatch.setattr(cl, "_logger", None)
        cl.reset_conversation_logger()
        cl.reset_conversation_logger()
        assert cl._logger is None


# ── Thread safety ────────────────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_records(self, tmp_path):
        import threading
        logger = ConversationLogger(data_dir=tmp_path / "api_conversations")
        errors = []

        def record(i):
            try:
                logger.record(f"Q{i}", f"A{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        lines = (tmp_path / "api_conversations" / "corpus.jsonl").read_text().splitlines()
        assert len(lines) == 20
