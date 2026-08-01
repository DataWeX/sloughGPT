"""Tests for domains/shell/log_buffer.py — LogBuffer + LogBufferHandler."""

import logging
import time
from domains.shell.log_buffer import (
    LogBuffer,
    LogBufferHandler,
    LogEntry,
    get_log_buffer,
)


class TestLogEntry:
    def test_create_entry(self):
        e = LogEntry(timestamp=1.0, level="INFO", source="test", message="hello")
        assert e.timestamp == 1.0
        assert e.level == "INFO"
        assert e.source == "test"
        assert e.message == "hello"
        assert e.context == {}


class TestLogBuffer:
    def test_append_and_get(self):
        b = LogBuffer(max_size=10)
        b.append(LogEntry(1.0, "INFO", "src", "msg"))
        assert len(b) == 1
        entries = b.get()
        assert len(entries) == 1
        assert entries[0].message == "msg"

    def test_filter_by_level(self):
        b = LogBuffer(max_size=10)
        b.append(LogEntry(1.0, "INFO", "src", "info"))
        b.append(LogEntry(2.0, "ERROR", "src", "err"))
        entries = b.get(level="ERROR")
        assert len(entries) == 1
        assert entries[0].message == "err"

    def test_filter_by_level_case_insensitive(self):
        b = LogBuffer(max_size=10)
        b.append(LogEntry(1.0, "info", "src", "m"))
        assert len(b.get(level="INFO")) == 1

    def test_filter_by_source_substring(self):
        b = LogBuffer(max_size=10)
        b.append(LogEntry(1.0, "INFO", "slo.kernel", "k"))
        b.append(LogEntry(2.0, "INFO", "slo.shell.repl", "r"))
        entries = b.get(source="kernel")
        assert len(entries) == 1
        assert entries[0].message == "k"

    def test_limit(self):
        b = LogBuffer(max_size=10)
        for i in range(5):
            b.append(LogEntry(float(i), "INFO", "src", f"msg{i}"))
        entries = b.get(limit=2)
        assert len(entries) == 2
        assert entries[-1].message == "msg4"

    def test_offset(self):
        b = LogBuffer(max_size=10)
        for i in range(5):
            b.append(LogEntry(float(i), "INFO", "src", f"msg{i}"))
        entries = b.get(offset=3)
        assert len(entries) == 2
        assert entries[0].message == "msg3"

    def test_clear(self):
        b = LogBuffer(max_size=10)
        b.append(LogEntry(1.0, "INFO", "src", "m"))
        b.clear()
        assert len(b) == 0

    def test_ring_buffer_eviction(self):
        b = LogBuffer(max_size=3)
        for i in range(5):
            b.append(LogEntry(float(i), "INFO", "src", f"msg{i}"))
        assert len(b) == 3
        entries = b.get()
        assert entries[0].message == "msg2"
        assert entries[-1].message == "msg4"

    def test_empty_get_returns_empty(self):
        b = LogBuffer(max_size=10)
        assert b.get() == []

    def test_get_does_not_mutate(self):
        b = LogBuffer(max_size=10)
        b.append(LogEntry(1.0, "INFO", "src", "m"))
        b.get()
        assert len(b) == 1


class TestLogBufferSingleton:
    def test_singleton_returns_same_instance(self):
        a = get_log_buffer()
        b = get_log_buffer()
        assert a is b

    def test_singleton_is_logbuffer(self):
        assert isinstance(get_log_buffer(), LogBuffer)


class TestLogBufferHandler:
    def test_handler_feeds_buffer(self):
        b = LogBuffer(max_size=10)
        handler = LogBufferHandler(b)
        logger = logging.getLogger("test_handler")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info("test message")
        entries = b.get()
        assert len(entries) >= 1
        assert entries[-1].message == "test message"
        logger.removeHandler(handler)

    def test_handler_captures_level_and_source(self):
        b = LogBuffer(max_size=10)
        handler = LogBufferHandler(b)
        logger = logging.getLogger("test.level.source")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.warning("warn msg")
        entries = b.get()
        entry = entries[-1]
        assert entry.level == "WARNING"
        assert entry.source == "test.level.source"
        logger.removeHandler(handler)

    def test_handler_error_safe(self):
        handler = LogBufferHandler()
        # Trigger handleError path by passing None as record
        handler.emit(None)  # should not raise

    def test_handler_default_buffer(self):
        handler = LogBufferHandler()
        assert handler._buffer is get_log_buffer()
