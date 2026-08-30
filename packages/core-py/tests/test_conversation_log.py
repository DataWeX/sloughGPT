"""Tests for ConversationLogger — recording exchanges, file output, enable/disable.

Covers:
  - record() writes to corpus.jsonl and input.txt
  - record() returns bytes written (1) or None when disabled
  - Enabled/disabled via env var
  - Empty prompt/response returns None
  - Meta dict is merged into corpus row
  - capture() wrapper never raises
  - get_conversation_logger() returns singleton
  - reset_conversation_logger() clears singleton
  - Thread safety, unicode, edge cases
"""

import json
import os
import tempfile
import threading
import pytest
from pathlib import Path
from domains.infrastructure.conversation_log import (
    ConversationLogger,
    get_conversation_logger,
    reset_conversation_logger,
    capture,
)


@pytest.fixture
def tmp_log_dir(tmp_path):
    return tmp_path / "conv_log"


# ── ConversationLogger.__init__ ──────────────────────────────────────────────

class TestInit:
    def test_creates_directory(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        assert tmp_log_dir.exists()

    def test_creates_nested_directory(self, tmp_log_dir):
        nested = tmp_log_dir / "a" / "b" / "c"
        logger = ConversationLogger(nested)
        assert nested.exists()

    def test_existing_directory_not_overwritten(self, tmp_log_dir):
        tmp_log_dir.mkdir(parents=True, exist_ok=True)
        (tmp_log_dir / "existing.txt").write_text("keep")
        logger = ConversationLogger(tmp_log_dir)
        assert (tmp_log_dir / "existing.txt").read_text() == "keep"

    def test_corpus_path(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        assert logger.corpus_path == tmp_log_dir / "corpus.jsonl"

    def test_text_path(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        assert logger.text_path == tmp_log_dir / "input.txt"


# ── ConversationLogger.enabled ───────────────────────────────────────────────

class TestEnabled:
    def test_enabled_default(self, tmp_log_dir, monkeypatch):
        monkeypatch.delenv("MAN_CAPTURE_CONVERSATIONS", raising=False)
        logger = ConversationLogger(tmp_log_dir)
        assert logger.enabled is True

    def test_enabled_disabled_value_0(self, tmp_log_dir, monkeypatch):
        monkeypatch.setenv("MAN_CAPTURE_CONVERSATIONS", "0")
        logger = ConversationLogger(tmp_log_dir)
        assert logger.enabled is False

    def test_enabled_enabled_value_1(self, tmp_log_dir, monkeypatch):
        monkeypatch.setenv("MAN_CAPTURE_CONVERSATIONS", "1")
        logger = ConversationLogger(tmp_log_dir)
        assert logger.enabled is True

    def test_enabled_any_nonzero_string(self, tmp_log_dir, monkeypatch):
        monkeypatch.setenv("MAN_CAPTURE_CONVERSATIONS", "yes")
        logger = ConversationLogger(tmp_log_dir)
        assert logger.enabled is True

    def test_enabled_empty_string_is_enabled(self, tmp_log_dir, monkeypatch):
        monkeypatch.setenv("MAN_CAPTURE_CONVERSATIONS", "")
        logger = ConversationLogger(tmp_log_dir)
        # Empty string != "0" so enabled is True
        assert logger.enabled is True


# ── ConversationLogger.record() ──────────────────────────────────────────────

class TestRecord:
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

    def test_record_none_prompt_returns_none(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        assert logger.record(None, "response") is None

    def test_record_none_response_returns_none(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        assert logger.record("prompt", None) is None

    def test_record_whitespace_only_returns_none(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        assert logger.record("  ", "  ") is None

    def test_record_tabs_only_returns_none(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        assert logger.record("\t\t", "\t\t") is None

    def test_record_newlines_only_returns_none(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        assert logger.record("\n\n", "\n\n") is None

    def test_record_disabled(self, tmp_log_dir, monkeypatch):
        monkeypatch.setenv("MAN_CAPTURE_CONVERSATIONS", "0")
        logger = ConversationLogger(tmp_log_dir)
        assert logger.record("Q", "A") is None

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

    def test_record_with_tokens_generated(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Q", "A", tokens_generated=42)
        row = json.loads(logger.corpus_path.read_text().strip())
        assert row["meta"]["tokens_generated"] == 42

    def test_record_with_elapsed_ms(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Q", "A", elapsed_ms=123.456)
        row = json.loads(logger.corpus_path.read_text().strip())
        assert row["meta"]["elapsed_ms"] == 123.5  # rounded to 1 decimal

    def test_record_default_model(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Q", "A")
        row = json.loads(logger.corpus_path.read_text().strip())
        assert row["meta"]["model"] == "unknown"

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

    def test_record_none_meta(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Q", "A", meta=None)
        row = json.loads(logger.corpus_path.read_text().strip())
        assert "meta" in row

    def test_record_empty_meta_dict(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Q", "A", meta={})
        row = json.loads(logger.corpus_path.read_text().strip())
        assert "meta" in row

    def test_record_unicode_content(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("你好世界", "こんにちは")
        row = json.loads(logger.corpus_path.read_text().strip())
        assert row["messages"][0]["content"] == "你好世界"
        assert row["messages"][1]["content"] == "こんにちは"

    def test_record_emoji_content(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("test 🎉", "response 🚀")
        row = json.loads(logger.corpus_path.read_text().strip())
        assert "🎉" in row["messages"][0]["content"]

    def test_record_long_content(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        long_q = "Q" * 10000
        long_q = long_q + "A" * 10000
        result = logger.record(long_q, "A" * 10000)
        assert result == 1

    def test_record_special_characters(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("line1\nline2\ttab", "backslash \\ quote \"")
        row = json.loads(logger.corpus_path.read_text().strip())
        assert "line1\nline2\ttab" in row["messages"][0]["content"]

    def test_record_captured_at_timestamp(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Q", "A")
        row = json.loads(logger.corpus_path.read_text().strip())
        assert "captured_at" in row["meta"]
        assert "T" in row["meta"]["captured_at"]  # ISO format

    def test_record_text_format(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Question", "Answer")
        text = logger.text_path.read_text()
        assert text.startswith("User: Question\n")
        assert "Assistant: Answer\n\n" in text

    def test_record_multiple_text_appended(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Q1", "A1")
        logger.record("Q2", "A2")
        text = logger.text_path.read_text()
        assert "User: Q1" in text
        assert "User: Q2" in text

    def test_record_corpus_valid_jsonl(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        for i in range(5):
            logger.record(f"Q{i}", f"A{i}")
        lines = logger.corpus_path.read_text().strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            row = json.loads(line)
            assert "messages" in row
            assert "meta" in row

    def test_record_temperature_none(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Q", "A")
        row = json.loads(logger.corpus_path.read_text().strip())
        assert row["meta"]["temperature"] is None


# ── Thread safety ────────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_records(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        errors = []

        def writer(i):
            try:
                logger.record(f"Q{i}", f"A{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        lines = logger.corpus_path.read_text().strip().split("\n")
        assert len(lines) == 20


# ── capture() ────────────────────────────────────────────────────────────────

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

    def test_capture_with_all_params(self, tmp_log_dir):
        from domains.infrastructure import conversation_log
        old = conversation_log._logger
        try:
            conversation_log._logger = ConversationLogger(tmp_log_dir)
            result = capture(
                "Q", "A",
                model="test-model",
                tokens_generated=10,
                elapsed_ms=50.0,
                temperature=0.5,
                meta={"key": "val"},
            )
            assert result is True
            row = json.loads((tmp_log_dir / "corpus.jsonl").read_text().strip())
            assert row["meta"]["model"] == "test-model"
            assert row["meta"]["tokens_generated"] == 10
            assert row["meta"]["temperature"] == 0.5
            assert row["meta"]["key"] == "val"
        finally:
            conversation_log._logger = old

    def test_capture_returns_false_when_empty(self, tmp_log_dir):
        from domains.infrastructure import conversation_log
        old = conversation_log._logger
        try:
            conversation_log._logger = ConversationLogger(tmp_log_dir)
            assert capture("", "A") is False
            assert capture("Q", "") is False
        finally:
            conversation_log._logger = old


# ── Singleton / reset ────────────────────────────────────────────────────────

class TestSingleton:
    def test_get_returns_same_instance(self):
        a = get_conversation_logger()
        b = get_conversation_logger()
        assert a is b

    def test_reset_clears_singleton(self):
        reset_conversation_logger()
        a = get_conversation_logger()
        reset_conversation_logger()
        b = get_conversation_logger()
        assert a is not b

    def test_reset_then_get_creates_fresh(self):
        reset_conversation_logger()
        logger = get_conversation_logger()
        assert logger is not None
        reset_conversation_logger()


# ── Independent directories ──────────────────────────────────────────────────

class TestIndependentDirs:
    def test_separate_loggers_different_dirs(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        logger_a = ConversationLogger(dir_a)
        logger_b = ConversationLogger(dir_b)
        logger_a.record("QA", "AA")
        logger_b.record("QB", "AB")
        assert (dir_a / "corpus.jsonl").exists()
        assert (dir_b / "corpus.jsonl").exists()
        row_a = json.loads((dir_a / "corpus.jsonl").read_text().strip())
        row_b = json.loads((dir_b / "corpus.jsonl").read_text().strip())
        assert row_a["messages"][0]["content"] == "QA"
        assert row_b["messages"][0]["content"] == "QB"

    def test_same_logger_same_dir(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Q1", "A1")
        logger.record("Q2", "A2")
        lines = logger.corpus_path.read_text().strip().split("\n")
        assert len(lines) == 2


# ── Record edge cases ────────────────────────────────────────────────────────

class TestRecordEdgeCases:
    def test_record_disabled_no_files_created(self, tmp_log_dir, monkeypatch):
        monkeypatch.setenv("MAN_CAPTURE_CONVERSATIONS", "0")
        logger = ConversationLogger(tmp_log_dir)
        assert logger.record("Q", "A") is None
        assert not logger.corpus_path.exists()
        assert not logger.text_path.exists()

    def test_record_single_char_prompt(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        assert logger.record("x", "y") == 1
        row = json.loads(logger.corpus_path.read_text().strip())
        assert row["messages"][0]["content"] == "x"

    def test_record_single_char_response(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        assert logger.record("Q", "A") == 1
        row = json.loads(logger.corpus_path.read_text().strip())
        assert row["messages"][1]["content"] == "A"

    def test_record_mixed_whitespace_and_content(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        assert logger.record("  Q  ", "  A  ") == 1

    def test_record_meta_overrides_default_keys(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Q", "A", meta={"model": "override"})
        row = json.loads(logger.corpus_path.read_text().strip())
        assert row["meta"]["model"] == "override"

    def test_record_many_meta_keys(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        meta = {f"key{i}": i for i in range(50)}
        logger.record("Q", "A", meta=meta)
        row = json.loads(logger.corpus_path.read_text().strip())
        for i in range(50):
            assert row["meta"][f"key{i}"] == i

    def test_record_elapsed_ms_zero(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Q", "A", elapsed_ms=0.0)
        row = json.loads(logger.corpus_path.read_text().strip())
        assert row["meta"]["elapsed_ms"] == 0.0

    def test_record_tokens_generated_zero(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Q", "A", tokens_generated=0)
        row = json.loads(logger.corpus_path.read_text().strip())
        assert row["meta"]["tokens_generated"] == 0

    def test_record_model_custom_string(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Q", "A", model="custom-model-123")
        row = json.loads(logger.corpus_path.read_text().strip())
        assert row["meta"]["model"] == "custom-model-123"

    def test_record_messages_have_roles(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Q", "A")
        row = json.loads(logger.corpus_path.read_text().strip())
        assert row["messages"][0]["role"] == "user"
        assert row["messages"][1]["role"] == "assistant"

    def test_record_text_has_double_newline_separator(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Q", "A")
        text = logger.text_path.read_text()
        assert "\n\n" in text


# ── capture() edge cases ─────────────────────────────────────────────────────

class TestCaptureEdgeCases:
    def test_capture_returns_false_on_exception(self, tmp_log_dir, monkeypatch):
        monkeypatch.setenv("MAN_CAPTURE_CONVERSATIONS", "1")
        from domains.infrastructure import conversation_log
        old = conversation_log._logger
        try:
            broken_logger = ConversationLogger(tmp_log_dir)
            broken_logger.corpus_path = None
            conversation_log._logger = broken_logger
            assert capture("Q", "A") is False
        finally:
            conversation_log._logger = old

    def test_capture_returns_false_when_none_prompt(self, tmp_log_dir):
        from domains.infrastructure import conversation_log
        old = conversation_log._logger
        try:
            conversation_log._logger = ConversationLogger(tmp_log_dir)
            assert capture(None, "A") is False
        finally:
            conversation_log._logger = old

    def test_capture_returns_false_when_none_response(self, tmp_log_dir):
        from domains.infrastructure import conversation_log
        old = conversation_log._logger
        try:
            conversation_log._logger = ConversationLogger(tmp_log_dir)
            assert capture("Q", None) is False
        finally:
            conversation_log._logger = old

    def test_capture_default_params(self, tmp_log_dir):
        from domains.infrastructure import conversation_log
        old = conversation_log._logger
        try:
            conversation_log._logger = ConversationLogger(tmp_log_dir)
            assert capture("Q", "A") is True
            row = json.loads((tmp_log_dir / "corpus.jsonl").read_text().strip())
            assert row["meta"]["model"] == "unknown"
            assert row["meta"]["temperature"] is None
        finally:
            conversation_log._logger = old


# ── Singleton thread safety ──────────────────────────────────────────────────

class TestSingletonThreadSafety:
    def test_concurrent_get_singleton(self):
        reset_conversation_logger()
        results = []

        def getter():
            results.append(get_conversation_logger())

        threads = [threading.Thread(target=getter) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 10
        assert all(r is results[0] for r in results)
        reset_conversation_logger()

    def test_reset_and_get_interleaved(self):
        reset_conversation_logger()
        instances = []
        for _ in range(5):
            reset_conversation_logger()
            instances.append(get_conversation_logger())
        assert len(set(id(i) for i in instances)) == 5
        reset_conversation_logger()


# ── Content edge cases ──────────────────────────────────────────────────────

class TestContentEdgeCases:
    def test_record_single_char_only(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        assert logger.record("a", "b") == 1
        row = json.loads(logger.corpus_path.read_text().strip())
        assert row["messages"][0]["content"] == "a"
        assert row["messages"][1]["content"] == "b"

    def test_record_multiline_prompt(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        prompt = "line1\nline2\nline3"
        logger.record(prompt, "response")
        row = json.loads(logger.corpus_path.read_text().strip())
        assert "\n" in row["messages"][0]["content"]

    def test_record_tabs_in_content(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        assert logger.record("q\tq", "a\ta") == 1

    def test_record_json_in_content(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        json_str = '{"key": "value"}'
        logger.record(json_str, "response")
        row = json.loads(logger.corpus_path.read_text().strip())
        assert json_str in row["messages"][0]["content"]

    def test_record_double_quotes_in_content(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record('say "hello"', "ok")
        row = json.loads(logger.corpus_path.read_text().strip())
        assert 'say "hello"' in row["messages"][0]["content"]

    def test_record_backslash_in_content(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("path\\to\\file", "ok")
        row = json.loads(logger.corpus_path.read_text().strip())
        assert "path\\to\\file" in row["messages"][0]["content"]

    def test_record_control_chars(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("q\ra", "ok")
        row = json.loads(logger.corpus_path.read_text().strip())
        assert "\r" in row["messages"][0]["content"]

    def test_record_with_many_meta_keys(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        meta = {f"k{i}": i for i in range(100)}
        logger.record("Q", "A", meta=meta)
        row = json.loads(logger.corpus_path.read_text().strip())
        for i in range(100):
            assert row["meta"][f"k{i}"] == i


# ── Thread safety extended ──────────────────────────────────────────────────

class TestThreadSafetyExtended:
    def test_concurrent_records_same_content(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        errors = []

        def writer():
            try:
                for i in range(5):
                    logger.record("Q", "A")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        lines = logger.corpus_path.read_text().strip().split("\n")
        assert len(lines) == 20

    def test_concurrent_record_and_read(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("init", "init")
        errors = []

        def writer():
            try:
                for i in range(5):
                    logger.record(f"Q{i}", f"A{i}")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(5):
                    if logger.corpus_path.exists():
                        logger.corpus_path.read_text()
            except Exception as e:
                errors.append(e)

        threads = ([threading.Thread(target=writer) for _ in range(2)] +
                   [threading.Thread(target=reader) for _ in range(2)])
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ── Logger module constants ─────────────────────────────────────────────────

class TestModuleConstants:
    def test_default_dir_exists(self):
        from domains.infrastructure import conversation_log
        # _DEFAULT_DIR is set at import time
        assert isinstance(conversation_log._DEFAULT_DIR, Path)

    def test_logger_lock_exists(self):
        from domains.infrastructure import conversation_log
        assert isinstance(conversation_log._logger_lock, type(threading.Lock()))

    def test_singleton_lock_exists(self):
        from domains.infrastructure import conversation_log
        assert hasattr(conversation_log, '_logger_lock')

    def test_capture_with_extra_meta(self, tmp_log_dir):
        from domains.infrastructure import conversation_log
        old = conversation_log._logger
        try:
            conversation_log._logger = ConversationLogger(tmp_log_dir)
            capture("Q", "A", meta={"nested": {"deep": True}})
            row = json.loads((tmp_log_dir / "corpus.jsonl").read_text().strip())
            assert row["meta"]["nested"]["deep"] is True
        finally:
            conversation_log._logger = old

    def test_capture_with_zero_tokens(self, tmp_log_dir):
        from domains.infrastructure import conversation_log
        old = conversation_log._logger
        try:
            conversation_log._logger = ConversationLogger(tmp_log_dir)
            result = capture("Q", "A", tokens_generated=0)
            assert result is True
            row = json.loads((tmp_log_dir / "corpus.jsonl").read_text().strip())
            assert row["meta"]["tokens_generated"] == 0
        finally:
            conversation_log._logger = old

    def test_capture_with_negative_elapsed(self, tmp_log_dir):
        from domains.infrastructure import conversation_log
        old = conversation_log._logger
        try:
            conversation_log._logger = ConversationLogger(tmp_log_dir)
            result = capture("Q", "A", elapsed_ms=-1.0)
            assert result is True
            row = json.loads((tmp_log_dir / "corpus.jsonl").read_text().strip())
            assert row["meta"]["elapsed_ms"] == -1.0
        finally:
            conversation_log._logger = old


# ── Persistence and file format ─────────────────────────────────────────────

class TestFileFormat:
    def test_corpus_is_valid_jsonl(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        for i in range(10):
            logger.record(f"Q{i}", f"A{i}")
        text = logger.corpus_path.read_text()
        lines = text.strip().split("\n")
        for line in lines:
            row = json.loads(line)
            assert isinstance(row, dict)
            assert "messages" in row
            assert len(row["messages"]) == 2

    def test_text_starts_with_user(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("First", "Response")
        text = logger.text_path.read_text()
        assert text.startswith("User: First")

    def test_text_file_utf8(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("日本語", "中国語")
        text = logger.text_path.read_text(encoding="utf-8")
        assert "日本語" in text
        assert "中国語" in text

    def test_corpus_has_timestamps(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        logger.record("Q", "A")
        row = json.loads(logger.corpus_path.read_text().strip())
        assert "captured_at" in row["meta"]

    def test_record_many_entries_performance(self, tmp_log_dir):
        logger = ConversationLogger(tmp_log_dir)
        for i in range(100):
            logger.record(f"Q{i}", f"A{i}")
        lines = logger.corpus_path.read_text().strip().split("\n")
        assert len(lines) == 100
