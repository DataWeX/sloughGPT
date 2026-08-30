"""Tests for shell/log_buffer.py — LogEntry, LogBuffer, LogBufferHandler, singleton."""

import logging
import threading
import time
from unittest.mock import MagicMock
import pytest
from domains.shell.log_buffer import (
    LogEntry, LogBuffer, LogBufferHandler, get_log_buffer,
)


# ── LogEntry ─────────────────────────────────────────────────────────

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

    def test_timestamp_is_float(self):
        e = LogEntry(timestamp=1234567890.123, level="INFO", source="slo", message="x")
        assert isinstance(e.timestamp, float)
        assert e.timestamp == 1234567890.123

    def test_level_preserves_case(self):
        e = LogEntry(timestamp=0.0, level="warning", source="slo", message="x")
        assert e.level == "warning"

    def test_empty_source(self):
        e = LogEntry(timestamp=0.0, level="INFO", source="", message="x")
        assert e.source == ""

    def test_empty_message(self):
        e = LogEntry(timestamp=0.0, level="INFO", source="slo", message="")
        assert e.message == ""

    def test_context_mutation(self):
        e = LogEntry(timestamp=0.0, level="INFO", source="slo", message="x")
        e.context["key"] = "value"
        assert e.context["key"] == "value"

    def test_context_independent_per_entry(self):
        e1 = LogEntry(timestamp=0.0, level="INFO", source="slo", message="a")
        e2 = LogEntry(timestamp=0.0, level="INFO", source="slo", message="b")
        e1.context["x"] = 1
        assert "x" not in e2.context

    def test_dataclass_fields(self):
        e = LogEntry(timestamp=5.0, level="ERROR", source="slo.err", message="fail")
        assert e.timestamp == 5.0
        assert e.level == "ERROR"
        assert e.source == "slo.err"
        assert e.message == "fail"

    def test_long_message(self):
        long_msg = "x" * 10000
        e = LogEntry(timestamp=0.0, level="INFO", source="slo", message=long_msg)
        assert len(e.message) == 10000

    def test_context_multiple_keys(self):
        e = LogEntry(timestamp=0.0, level="INFO", source="slo", message="x",
                      context={"a": 1, "b": 2, "c": 3})
        assert len(e.context) == 3

    def test_zero_timestamp(self):
        e = LogEntry(timestamp=0.0, level="INFO", source="slo", message="x")
        assert e.timestamp == 0.0

    def test_negative_timestamp(self):
        e = LogEntry(timestamp=-1.0, level="INFO", source="slo", message="x")
        assert e.timestamp == -1.0

    def test_context_empty_dict_default(self):
        e1 = LogEntry(timestamp=0.0, level="INFO", source="slo", message="a")
        e2 = LogEntry(timestamp=0.0, level="INFO", source="slo", message="b")
        assert e1.context is not e2.context


# ── LogBuffer ────────────────────────────────────────────────────────

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

    def test_default_max_size(self):
        buf = LogBuffer()
        for i in range(2500):
            buf.append(LogEntry(float(i), "INFO", "slo", f"msg{i}"))
        assert len(buf) == 2000

    def test_max_size_one(self):
        buf = LogBuffer(max_size=1)
        buf.append(LogEntry(1.0, "INFO", "slo", "first"))
        buf.append(LogEntry(2.0, "INFO", "slo", "second"))
        assert len(buf) == 1
        assert buf.get()[0].message == "second"

    def test_max_size_zero(self):
        buf = LogBuffer(max_size=0)
        buf.append(LogEntry(1.0, "INFO", "slo", "x"))
        assert len(buf) == 0

    def test_get_no_filters(self):
        buf = LogBuffer()
        buf.append(LogEntry(1.0, "INFO", "slo", "a"))
        buf.append(LogEntry(2.0, "ERROR", "slo", "b"))
        result = buf.get()
        assert len(result) == 2

    def test_get_level_case_insensitive_lower(self):
        buf = LogBuffer()
        buf.append(LogEntry(1.0, "info", "slo", "a"))
        buf.append(LogEntry(2.0, "ERROR", "slo", "b"))
        result = buf.get(level="INFO")
        assert len(result) == 1
        assert result[0].level == "info"

    def test_get_level_case_insensitive_upper(self):
        buf = LogBuffer()
        buf.append(LogEntry(1.0, "INFO", "slo", "a"))
        result = buf.get(level="error")
        assert len(result) == 0

    def test_get_source_substring_match(self):
        buf = LogBuffer()
        buf.append(LogEntry(1.0, "INFO", "slo.kernel.init", "k"))
        buf.append(LogEntry(2.0, "INFO", "slo.api.server", "a"))
        result = buf.get(source="kernel")
        assert len(result) == 1

    def test_get_source_case_insensitive(self):
        buf = LogBuffer()
        buf.append(LogEntry(1.0, "INFO", "slo.Kernel", "k"))
        result = buf.get(source="kernel")
        assert len(result) == 1

    def test_get_limit_larger_than_entries(self):
        buf = LogBuffer()
        buf.append(LogEntry(1.0, "INFO", "slo", "a"))
        result = buf.get(limit=100)
        assert len(result) == 1

    def test_get_offset_beyond_entries(self):
        buf = LogBuffer()
        buf.append(LogEntry(1.0, "INFO", "slo", "a"))
        result = buf.get(offset=100)
        assert len(result) == 0

    def test_get_offset_zero(self):
        buf = LogBuffer()
        buf.append(LogEntry(1.0, "INFO", "slo", "a"))
        result = buf.get(offset=0)
        assert len(result) == 1

    def test_get_limit_zero_returns_all(self):
        buf = LogBuffer()
        buf.append(LogEntry(1.0, "INFO", "slo", "a"))
        buf.append(LogEntry(2.0, "INFO", "slo", "b"))
        result = buf.get(limit=0)
        assert len(result) == 2

    def test_combined_level_source_filter(self):
        buf = LogBuffer()
        buf.append(LogEntry(1.0, "INFO", "slo.kernel", "k"))
        buf.append(LogEntry(2.0, "ERROR", "slo.kernel", "k"))
        buf.append(LogEntry(3.0, "INFO", "slo.api", "a"))
        result = buf.get(level="INFO", source="kernel")
        assert len(result) == 1

    def test_empty_buffer_get(self):
        buf = LogBuffer()
        assert buf.get() == []
        assert buf.get(level="INFO") == []
        assert buf.get(source="x") == []
        assert buf.get(limit=5) == []

    def test_thread_safety_concurrent_appends(self):
        buf = LogBuffer(max_size=10000)
        errors = []

        def appender(thread_id):
            try:
                for i in range(100):
                    buf.append(LogEntry(float(i), "INFO", f"slo.t{thread_id}", f"msg{i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=appender, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(buf) == 1000

    def test_thread_safety_concurrent_reads(self):
        buf = LogBuffer(max_size=10000)
        for i in range(100):
            buf.append(LogEntry(float(i), "INFO", "slo", f"msg{i}"))
        errors = []

        def reader():
            try:
                for _ in range(50):
                    result = buf.get()
                    _ = len(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_clear_during_concurrent_reads(self):
        buf = LogBuffer(max_size=10000)
        for i in range(100):
            buf.append(LogEntry(float(i), "INFO", "slo", f"msg{i}"))
        errors = []

        def reader():
            try:
                for _ in range(20):
                    buf.get()
            except Exception as e:
                errors.append(e)

        def clearer():
            for _ in range(10):
                time.sleep(0.001)
                buf.clear()

        threads = [threading.Thread(target=reader) for _ in range(5)]
        threads.append(threading.Thread(target=clearer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_append_returns_none(self):
        buf = LogBuffer()
        result = buf.append(LogEntry(0.0, "INFO", "slo", "x"))
        assert result is None

    def test_clear_returns_none(self):
        buf = LogBuffer()
        buf.append(LogEntry(0.0, "INFO", "slo", "x"))
        result = buf.clear()
        assert result is None

    def test_get_preserves_order(self):
        buf = LogBuffer()
        for i in range(20):
            buf.append(LogEntry(float(i), "INFO", "slo", f"msg{i}"))
        result = buf.get()
        for i in range(19):
            assert result[i].timestamp <= result[i + 1].timestamp

    def test_get_level_exact_match_only(self):
        buf = LogBuffer()
        buf.append(LogEntry(1.0, "INFO", "slo", "a"))
        buf.append(LogEntry(2.0, "INFORMATION", "slo", "b"))
        result = buf.get(level="INFO")
        assert len(result) == 1

    def test_get_source_full_path_match(self):
        buf = LogBuffer()
        buf.append(LogEntry(1.0, "INFO", "slo.kernel.driver", "k"))
        result = buf.get(source="slo.kernel.driver")
        assert len(result) == 1

    def test_max_size_large(self):
        buf = LogBuffer(max_size=5000)
        for i in range(6000):
            buf.append(LogEntry(float(i), "INFO", "slo", f"msg{i}"))
        assert len(buf) == 5000

    def test_len_thread_safety(self):
        buf = LogBuffer(max_size=10000)

        def appender():
            for i in range(100):
                buf.append(LogEntry(float(i), "INFO", "slo", f"m{i}"))

        threads = [threading.Thread(target=appender) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # After all threads complete, buffer should have exactly 500 entries
        assert len(buf) == 500

    def test_get_limit_and_offset_combined_large(self):
        buf = LogBuffer()
        for i in range(100):
            buf.append(LogEntry(float(i), "INFO", "slo", f"msg{i}"))
        result = buf.get(offset=10, limit=5)
        assert len(result) == 5
        # offset=10 removes the 10 oldest → 90 remain; limit=5 takes last 5 → msg95..msg99
        assert result[0].message == "msg95"

    def test_multiple_level_filters(self):
        buf = LogBuffer()
        buf.append(LogEntry(1.0, "INFO", "slo", "a"))
        buf.append(LogEntry(2.0, "ERROR", "slo", "b"))
        buf.append(LogEntry(3.0, "WARNING", "slo", "c"))
        assert len(buf.get(level="INFO")) == 1
        assert len(buf.get(level="ERROR")) == 1
        assert len(buf.get(level="WARNING")) == 1
        assert len(buf.get(level="DEBUG")) == 0

    def test_multiple_source_filters(self):
        buf = LogBuffer()
        buf.append(LogEntry(1.0, "INFO", "slo.kernel", "k"))
        buf.append(LogEntry(2.0, "INFO", "slo.api", "a"))
        buf.append(LogEntry(3.0, "INFO", "slo.db", "d"))
        assert len(buf.get(source="kernel")) == 1
        assert len(buf.get(source="api")) == 1
        assert len(buf.get(source="slo")) == 3


# ── LogBufferHandler ─────────────────────────────────────────────────

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

    def test_emit_preserves_timestamp(self):
        buf = LogBuffer()
        handler = LogBufferHandler(buffer=buf)
        ts = 1234567890.5
        record = logging.LogRecord(
            name="slo", level=logging.INFO, pathname="", lineno=0,
            msg="x", args=(), exc_info=None,
        )
        record.created = ts
        handler.emit(record)
        entries = buf.get()
        assert entries[0].timestamp == ts

    def test_emit_level_debug(self):
        buf = LogBuffer()
        handler = LogBufferHandler(buffer=buf)
        record = logging.LogRecord(
            name="slo.debug", level=logging.DEBUG, pathname="", lineno=0,
            msg="debug msg", args=(), exc_info=None,
        )
        handler.emit(record)
        entries = buf.get()
        assert entries[0].level == "DEBUG"

    def test_emit_level_warning(self):
        buf = LogBuffer()
        handler = LogBufferHandler(buffer=buf)
        record = logging.LogRecord(
            name="slo.warn", level=logging.WARNING, pathname="", lineno=0,
            msg="warn msg", args=(), exc_info=None,
        )
        handler.emit(record)
        entries = buf.get()
        assert entries[0].level == "WARNING"

    def test_emit_level_critical(self):
        buf = LogBuffer()
        handler = LogBufferHandler(buffer=buf)
        record = logging.LogRecord(
            name="slo.crit", level=logging.CRITICAL, pathname="", lineno=0,
            msg="crit msg", args=(), exc_info=None,
        )
        handler.emit(record)
        entries = buf.get()
        assert entries[0].level == "CRITICAL"

    def test_emit_preserves_source(self):
        buf = LogBuffer()
        handler = LogBufferHandler(buffer=buf)
        record = logging.LogRecord(
            name="slo.infrastructure.model_server", level=logging.INFO, pathname="", lineno=0,
            msg="x", args=(), exc_info=None,
        )
        handler.emit(record)
        entries = buf.get()
        assert entries[0].source == "slo.infrastructure.model_server"

    def test_emit_message_with_args(self):
        buf = LogBuffer()
        handler = LogBufferHandler(buffer=buf)
        record = logging.LogRecord(
            name="slo", level=logging.INFO, pathname="", lineno=0,
            msg="loaded %d items", args=(1000,), exc_info=None,
        )
        handler.emit(record)
        entries = buf.get()
        assert entries[0].message == "loaded 1000 items"

    def test_emit_message_no_args(self):
        buf = LogBuffer()
        handler = LogBufferHandler(buffer=buf)
        record = logging.LogRecord(
            name="slo", level=logging.INFO, pathname="", lineno=0,
            msg="simple message", args=(), exc_info=None,
        )
        handler.emit(record)
        entries = buf.get()
        assert entries[0].message == "simple message"

    def test_default_buffer_uses_singleton(self):
        handler = LogBufferHandler()
        record = logging.LogRecord(
            name="slo.test", level=logging.INFO, pathname="", lineno=0,
            msg="x", args=(), exc_info=None,
        )
        handler.emit(record)
        singleton = get_log_buffer()
        entries = singleton.get()
        assert len(entries) >= 1

    def test_multiple_emits(self):
        buf = LogBuffer()
        handler = LogBufferHandler(buffer=buf)
        for i in range(10):
            record = logging.LogRecord(
                name="slo", level=logging.INFO, pathname="", lineno=0,
                msg=f"msg{i}", args=(), exc_info=None,
            )
            handler.emit(record)
        assert len(buf) == 10

    def test_emit_message_with_multiple_args(self):
        buf = LogBuffer()
        handler = LogBufferHandler(buffer=buf)
        record = logging.LogRecord(
            name="slo", level=logging.INFO, pathname="", lineno=0,
            msg="values: %s, %d, %.2f", args=("a", 42, 3.14), exc_info=None,
        )
        handler.emit(record)
        entries = buf.get()
        assert "a" in entries[0].message
        assert "42" in entries[0].message
        assert "3.14" in entries[0].message

    def test_emit_returns_none(self):
        buf = LogBuffer()
        handler = LogBufferHandler(buffer=buf)
        record = logging.LogRecord(
            name="slo", level=logging.INFO, pathname="", lineno=0,
            msg="x", args=(), exc_info=None,
        )
        result = handler.emit(record)
        assert result is None

    def test_emit_preserves_level_case(self):
        buf = LogBuffer()
        handler = LogBufferHandler(buffer=buf)
        for level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
            record = logging.LogRecord(
                name="slo", level=level, pathname="", lineno=0,
                msg="x", args=(), exc_info=None,
            )
            handler.emit(record)
        entries = buf.get()
        levels = [e.level for e in entries]
        assert "DEBUG" in levels
        assert "INFO" in levels
        assert "WARNING" in levels
        assert "ERROR" in levels
        assert "CRITICAL" in levels

    def test_emit_record_with_zero_args(self):
        buf = LogBuffer()
        handler = LogBufferHandler(buffer=buf)
        record = logging.LogRecord(
            name="slo", level=logging.INFO, pathname="", lineno=0,
            msg="no args here", args=(), exc_info=None,
        )
        handler.emit(record)
        entries = buf.get()
        assert entries[0].message == "no args here"

    def test_emit_record_with_empty_msg(self):
        buf = LogBuffer()
        handler = LogBufferHandler(buffer=buf)
        record = logging.LogRecord(
            name="slo", level=logging.INFO, pathname="", lineno=0,
            msg="", args=(), exc_info=None,
        )
        handler.emit(record)
        entries = buf.get()
        assert entries[0].message == ""


# ── Singleton ────────────────────────────────────────────────────────

class TestSingleton:
    def test_get_log_buffer_returns_same_instance(self):
        a = get_log_buffer()
        b = get_log_buffer()
        assert a is b

    def test_singleton_is_log_buffer(self):
        assert isinstance(get_log_buffer(), LogBuffer)

    def test_singleton_has_entries_from_handler(self):
        buf = get_log_buffer()
        initial_count = len(buf)
        handler = LogBufferHandler(buffer=buf)
        record = logging.LogRecord(
            name="slo.singleton", level=logging.INFO, pathname="", lineno=0,
            msg="singleton test", args=(), exc_info=None,
        )
        handler.emit(record)
        assert len(buf) == initial_count + 1

    def test_singleton_thread_safety(self):
        results = []

        def get_singleton():
            results.append(get_log_buffer())

        threads = [threading.Thread(target=get_singleton) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r is results[0] for r in results)

    def test_singleton_persists_data(self):
        buf = get_log_buffer()
        initial = len(buf)
        handler = LogBufferHandler(buffer=buf)
        record = logging.LogRecord(
            name="slo.persist", level=logging.INFO, pathname="", lineno=0,
            msg="persist", args=(), exc_info=None,
        )
        handler.emit(record)
        buf2 = get_log_buffer()
        assert len(buf2) == initial + 1

    def test_singleton_class_type(self):
        buf = get_log_buffer()
        assert type(buf).__name__ == "LogBuffer"
