"""Tests for shell/log_buffer.py — LogEntry, LogBuffer, LogBufferHandler, singleton."""

import logging
import time
from unittest.mock import MagicMock
import pytest
from domains.shell.log_buffer import (
    LogEntry, LogBuffer, LogBufferHandler, get_log_buffer,
)


class TestLogEntry:
    def test_creation(self):
        e = LogEntry(timestamp=1.0, level="INFO", source="slo.test", message="hello")
        assert e.timestamp == 1.0
        assert e.level == "INFO"
        assert e.source == "slo.test"
        assert e.message == "hello"
        assert e.context == {}

    def test_with_context(self):
        e = LogEntry(timestamp=1.0, level="INFO", source="slo", message="ok",
                      context={"req": "123"})
        assert e.context["req"] == "123"


class TestLogBuffer:
    def test_append_and_len(self):
        buf = LogBuffer(max_size=100)
        assert len(buf) == 0
        buf.append(LogEntry(1.0, "INFO", "slo", "a"))
        buf.append(LogEntry(2.0, "ERROR", "slo", "b"))
        assert len(buf) == 2

    def test_ring_buffer_overflow(self):
        buf = LogBuffer(max_size=3)
        for i in range(5):
            buf.append(LogEntry(float(i), "INFO", "slo", f"msg{i}"))
        assert len(buf) == 3
        entries = buf.get()
        assert entries[0].message == "msg2"
        assert entries[2].message == "msg4"

    def test_get_all(self):
        buf = LogBuffer()
        buf.append(LogEntry(1.0, "INFO", "slo.a", "one"))
        buf.append(LogEntry(2.0, "ERROR", "slo.b", "two"))
        result = buf.get()
        assert len(result) == 2

    def test_get_filter_level(self):
        buf = LogBuffer()
        buf.append(LogEntry(1.0, "INFO", "slo", "info"))
        buf.append(LogEntry(2.0, "ERROR", "slo", "err"))
        buf.append(LogEntry(3.0, "info", "slo", "lower"))  # case-insensitive
        result = buf.get(level="INFO")
        assert len(result) == 2

    def test_get_filter_source(self):
        buf = LogBuffer()
        buf.append(LogEntry(1.0, "INFO", "slo.kernel", "k"))
        buf.append(LogEntry(2.0, "INFO", "slo.api", "a"))
        result = buf.get(source="kernel")
        assert len(result) == 1
        assert result[0].source == "slo.kernel"

    def test_get_limit_returns_last_n(self):
        buf = LogBuffer()
        for i in range(10):
            buf.append(LogEntry(float(i), "INFO", "slo", f"msg{i}"))
        result = buf.get(limit=3)
        assert len(result) == 3
        assert result[0].message == "msg7"

    def test_get_offset(self):
        buf = LogBuffer()
        for i in range(5):
            buf.append(LogEntry(float(i), "INFO", "slo", f"msg{i}"))
        result = buf.get(offset=2)
        assert len(result) == 3
        assert result[0].message == "msg2"

    def test_get_offset_then_limit(self):
        buf = LogBuffer()
        for i in range(10):
            buf.append(LogEntry(float(i), "INFO", "slo", f"msg{i}"))
        # offset slices from front, limit takes last N
        result = buf.get(offset=3, limit=2)
        assert len(result) == 2
        assert result[0].message == "msg8"

    def test_clear(self):
        buf = LogBuffer()
        buf.append(LogEntry(1.0, "INFO", "slo", "x"))
        buf.clear()
        assert len(buf) == 0
        assert buf.get() == []


class TestLogBufferHandler:
    def test_emit_feeds_buffer(self):
        buf = LogBuffer()
        handler = LogBufferHandler(buffer=buf)
        record = logging.LogRecord(
            name="slo.test", level=logging.INFO, pathname="", lineno=0,
            msg="hello %s", args=("world",), exc_info=None,
        )
        handler.emit(record)
        entries = buf.get()
        assert len(entries) == 1
        assert entries[0].message == "hello world"
        assert entries[0].level == "INFO"
        assert entries[0].source == "slo.test"

    def test_emit_with_exception(self):
        buf = LogBuffer()
        handler = LogBufferHandler(buffer=buf)
        record = logging.LogRecord(
            name="slo.err", level=logging.ERROR, pathname="", lineno=0,
            msg="fail", args=(), exc_info=(ValueError, ValueError("bad"), None),
        )
        handler.emit(record)
        entries = buf.get()
        assert len(entries) == 1

    def test_emit_error_does_not_propagate(self):
        buf = MagicMock()
        buf.append.side_effect = RuntimeError("disk full")
        handler = LogBufferHandler(buffer=buf)
        record = logging.LogRecord(
            name="slo", level=logging.INFO, pathname="", lineno=0,
            msg="x", args=(), exc_info=None,
        )
        handler.emit(record)  # should not raise


class TestSingleton:
    def test_get_log_buffer_returns_same_instance(self):
        a = get_log_buffer()
        b = get_log_buffer()
        assert a is b

    def test_singleton_is_log_buffer(self):
        assert isinstance(get_log_buffer(), LogBuffer)
