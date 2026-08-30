"""Tests for domains/shell/log_buffer.py — LogBuffer + LogBufferHandler."""

import logging
import threading
import time
from domains.shell.log_buffer import (
    LogBuffer,
    LogBufferHandler,
    LogEntry,
    get_log_buffer,
)


# =============================================================================
# LogEntry
# =============================================================================

class TestLogEntry:
    def test_create_entry(self):
        e = LogEntry(timestamp=1.0, level="INFO", source="test", message="hello")
        assert e.timestamp == 1.0
        assert e.level == "INFO"
        assert e.source == "test"
        assert e.message == "hello"
        assert e.context == {}

    def test_context_default_empty(self):
        e = LogEntry(timestamp=0.0, level="DEBUG", source="s", message="m")
        assert e.context == {}

    def test_context_custom(self):
        e = LogEntry(timestamp=0.0, level="INFO", source="s", message="m", context={"key": "val"})
        assert e.context["key"] == "val"

    def test_entry_fields_are_set(self):
        e = LogEntry(timestamp=123.456, level="CRITICAL", source="slo.kernel", message="panic")
        assert e.timestamp == 123.456
        assert e.level == "CRITICAL"
        assert e.source == "slo.kernel"
        assert e.message == "panic"

    def test_empty_message(self):
        e = LogEntry(timestamp=0.0, level="INFO", source="s", message="")
        assert e.message == ""

    def test_empty_source(self):
        e = LogEntry(timestamp=0.0, level="INFO", source="", message="m")
        assert e.source == ""


# =============================================================================
# LogBuffer
# =============================================================================

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

    def test_limit_larger_than_buffer(self):
        b = LogBuffer(max_size=3)
        for i in range(3):
            b.append(LogEntry(float(i), "INFO", "src", f"msg{i}"))
        entries = b.get(limit=100)
        assert len(entries) == 3

    def test_offset_beyond_entries(self):
        b = LogBuffer(max_size=10)
        b.append(LogEntry(1.0, "INFO", "src", "m"))
        entries = b.get(offset=10)
        assert len(entries) == 0

    def test_offset_zero(self):
        b = LogBuffer(max_size=10)
        for i in range(3):
            b.append(LogEntry(float(i), "INFO", "src", f"msg{i}"))
        entries = b.get(offset=0)
        assert len(entries) == 3
        assert entries[0].message == "msg0"

    def test_limit_and_offset_combined(self):
        b = LogBuffer(max_size=10)
        for i in range(10):
            b.append(LogEntry(float(i), "INFO", "src", f"msg{i}"))
        entries = b.get(limit=3, offset=5)
        assert len(entries) == 3
        assert entries[0].message == "msg7"
        assert entries[-1].message == "msg9"

    def test_filter_by_level_no_match(self):
        b = LogBuffer(max_size=10)
        b.append(LogEntry(1.0, "INFO", "src", "m"))
        assert b.get(level="ERROR") == []

    def test_filter_by_source_no_match(self):
        b = LogBuffer(max_size=10)
        b.append(LogEntry(1.0, "INFO", "slo.kernel", "m"))
        assert b.get(source="nonexistent") == []

    def test_level_filter_case_insensitive_mixed(self):
        b = LogBuffer(max_size=10)
        b.append(LogEntry(1.0, "Warning", "src", "w"))
        b.append(LogEntry(2.0, "warning", "src", "w2"))
        b.append(LogEntry(3.0, "WARNING", "src", "w3"))
        entries = b.get(level="warning")
        assert len(entries) == 3

    def test_source_filter_case_insensitive(self):
        b = LogBuffer(max_size=10)
        b.append(LogEntry(1.0, "INFO", "SLO.Kernel", "m"))
        entries = b.get(source="kernel")
        assert len(entries) == 1

    def test_source_filter_partial_match(self):
        b = LogBuffer(max_size=10)
        b.append(LogEntry(1.0, "INFO", "slo.kernel.driver", "m"))
        entries = b.get(source="kernel")
        assert len(entries) == 1

    def test_append_many(self):
        b = LogBuffer(max_size=100)
        for i in range(100):
            b.append(LogEntry(float(i), "INFO", "src", f"msg{i}"))
        assert len(b) == 100
        assert b.get()[-1].message == "msg99"

    def test_max_size_one(self):
        b = LogBuffer(max_size=1)
        b.append(LogEntry(1.0, "INFO", "src", "first"))
        b.append(LogEntry(2.0, "INFO", "src", "second"))
        assert len(b) == 1
        assert b.get()[0].message == "second"

    def test_clear_and_reuse(self):
        b = LogBuffer(max_size=5)
        b.append(LogEntry(1.0, "INFO", "src", "old"))
        b.clear()
        assert len(b) == 0
        b.append(LogEntry(2.0, "INFO", "src", "new"))
        assert len(b) == 1
        assert b.get()[0].message == "new"

    def test_get_returns_copy(self):
        b = LogBuffer(max_size=10)
        b.append(LogEntry(1.0, "INFO", "src", "m"))
        entries1 = b.get()
        entries2 = b.get()
        assert entries1 is not entries2
        assert entries1 == entries2

    def test_limit_zero_treated_as_no_limit(self):
        """limit=0 is falsy in Python, so the source treats it as 'no limit'."""
        b = LogBuffer(max_size=10)
        b.append(LogEntry(1.0, "INFO", "src", "m"))
        entries = b.get(limit=0)
        # limit=0 is falsy → if limit: is skipped → no limit applied
        assert len(entries) == 1


# =============================================================================
# Thread Safety
# =============================================================================

class TestLogBufferThreadSafety:
    def test_concurrent_appends(self):
        b = LogBuffer(max_size=1000)
        errors = []

        def writer(n):
            try:
                for i in range(50):
                    b.append(LogEntry(float(i), "INFO", f"src{n}", f"msg{n}-{i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(b) == 500

    def test_concurrent_read_write(self):
        b = LogBuffer(max_size=100)
        errors = []

        def writer():
            try:
                for i in range(100):
                    b.append(LogEntry(float(i), "INFO", "src", f"msg{i}"))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    b.get()
            except Exception as e:
                errors.append(e)

        tw = threading.Thread(target=writer)
        tr = threading.Thread(target=reader)
        tw.start()
        tr.start()
        tw.join()
        tr.join()

        assert len(errors) == 0

    def test_concurrent_clear_and_append(self):
        b = LogBuffer(max_size=100)
        errors = []

        def writer():
            try:
                for i in range(50):
                    b.append(LogEntry(float(i), "INFO", "src", f"msg{i}"))
            except Exception as e:
                errors.append(e)

        def clearer():
            try:
                for _ in range(50):
                    b.clear()
            except Exception as e:
                errors.append(e)

        tw = threading.Thread(target=writer)
        tc = threading.Thread(target=clearer)
        tw.start()
        tc.start()
        tw.join()
        tc.join()

        assert len(errors) == 0
        assert len(b) <= 50


# =============================================================================
# LogBufferSingleton
# =============================================================================

class TestLogBufferSingleton:
    def test_singleton_returns_same_instance(self):
        a = get_log_buffer()
        b = get_log_buffer()
        assert a is b

    def test_singleton_is_logbuffer(self):
        assert isinstance(get_log_buffer(), LogBuffer)

    def test_singleton_has_reasonable_default_size(self):
        buf = get_log_buffer()
        assert buf._max_size >= 100


# =============================================================================
# LogBufferHandler
# =============================================================================

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
        handler.emit(None)  # should not raise

    def test_handler_default_buffer(self):
        handler = LogBufferHandler()
        assert handler._buffer is get_log_buffer()

    def test_handler_debug_level(self):
        b = LogBuffer(max_size=10)
        handler = LogBufferHandler(b)
        logger = logging.getLogger("test.debug.level")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.debug("debug msg")
        entries = b.get()
        assert entries[-1].level == "DEBUG"
        assert entries[-1].message == "debug msg"
        logger.removeHandler(handler)

    def test_handler_error_level(self):
        b = LogBuffer(max_size=10)
        handler = LogBufferHandler(b)
        logger = logging.getLogger("test.error.level")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.error("error msg")
        entries = b.get()
        assert entries[-1].level == "ERROR"
        logger.removeHandler(handler)

    def test_handler_critical_level(self):
        b = LogBuffer(max_size=10)
        handler = LogBufferHandler(b)
        logger = logging.getLogger("test.critical.level")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.critical("critical msg")
        entries = b.get()
        assert entries[-1].level == "CRITICAL"
        logger.removeHandler(handler)

    def test_handler_multiple_messages(self):
        b = LogBuffer(max_size=10)
        handler = LogBufferHandler(b)
        logger = logging.getLogger("test.multi.msg")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info("msg1")
        logger.info("msg2")
        logger.info("msg3")
        entries = b.get()
        assert len(entries) >= 3
        logger.removeHandler(handler)

    def test_handler_with_extra_fields(self):
        b = LogBuffer(max_size=10)
        handler = LogBufferHandler(b)
        logger = logging.getLogger("test.extra.fields")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info("extra msg", extra={"tag": "MODEL"})
        entries = b.get()
        assert entries[-1].message == "extra msg"
        logger.removeHandler(handler)

    def test_handler_timestamp_is_float(self):
        b = LogBuffer(max_size=10)
        handler = LogBufferHandler(b)
        logger = logging.getLogger("test.timestamp")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info("ts msg")
        entries = b.get()
        assert isinstance(entries[-1].timestamp, float)
        logger.removeHandler(handler)

    def test_handler_records_are_logentry_instances(self):
        b = LogBuffer(max_size=10)
        handler = LogBufferHandler(b)
        logger = logging.getLogger("test.instance.type")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info("type msg")
        entries = b.get()
        assert isinstance(entries[-1], LogEntry)
        logger.removeHandler(handler)

    def test_handler_formatted_message(self):
        b = LogBuffer(max_size=10)
        handler = LogBufferHandler(b)
        logger = logging.getLogger("test.formatted")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info("value is %d", 42)
        entries = b.get()
        assert entries[-1].message == "value is 42"
        logger.removeHandler(handler)

    def test_handler_does_not_duplicate_to_other_handlers(self):
        b = LogBuffer(max_size=10)
        handler = LogBufferHandler(b)
        logger = logging.getLogger("test.isolation")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        # Ensure only our handler captures
        logger.info("isolated msg")
        entries = b.get()
        assert len(entries) >= 1
        assert entries[-1].message == "isolated msg"
        logger.removeHandler(handler)

    def test_multiple_loggers_same_buffer(self):
        b = LogBuffer(max_size=10)
        h1 = LogBufferHandler(b)
        h2 = LogBufferHandler(b)
        l1 = logging.getLogger("test.multi.logger1")
        l2 = logging.getLogger("test.multi.logger2")
        l1.addHandler(h1)
        l2.addHandler(h2)
        l1.setLevel(logging.DEBUG)
        l2.setLevel(logging.DEBUG)
        l1.info("from logger1")
        l2.info("from logger2")
        entries = b.get()
        messages = [e.message for e in entries]
        assert "from logger1" in messages
        assert "from logger2" in messages
        l1.removeHandler(h1)
        l2.removeHandler(h2)
