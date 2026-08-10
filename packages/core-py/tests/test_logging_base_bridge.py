"""Tests for logging/base.py and logging/bridge.py — LogLevel, LogRecord, Logger, BridgeHandler.

Covers:
  - LogLevel ordering (comparison operators)
  - ErrorCode and LogTag enums
  - LogRecord creation
  - Logger ABC — emit, debug/info/warning/error/critical, level filtering
  - TaggedLogger — tag attached to records
  - ChildLogger — delegates to parent
  - BridgeHandler — converts logging.LogRecord to LogRecord
"""

import logging
import pytest
from unittest.mock import MagicMock
from domains.logging.base import (
    Logger, LogLevel, LogRecord, ErrorCode, LogTag, TaggedLogger, ChildLogger,
)
from domains.logging.bridge import BridgeHandler, _LEVEL_MAP


class TestLogLevel:
    def test_ordering(self):
        assert LogLevel.DEBUG < LogLevel.INFO < LogLevel.WARNING < LogLevel.ERROR < LogLevel.CRITICAL

    def test_ge(self):
        assert LogLevel.ERROR >= LogLevel.WARNING
        assert LogLevel.ERROR >= LogLevel.ERROR
        assert not (LogLevel.INFO >= LogLevel.WARNING)

    def test_le(self):
        assert LogLevel.DEBUG <= LogLevel.INFO
        assert LogLevel.WARNING <= LogLevel.WARNING
        assert not (LogLevel.ERROR <= LogLevel.INFO)

    def test_eq(self):
        assert LogLevel.INFO == LogLevel.INFO

    def test_comparison_with_non_loglevel(self):
        assert LogLevel.INFO.__ge__("info") is NotImplemented
        assert LogLevel.INFO.__gt__(42) is NotImplemented
        assert LogLevel.INFO.__le__("info") is NotImplemented
        assert LogLevel.INFO.__lt__(42) is NotImplemented


class TestErrorCode:
    def test_all_values_are_strings(self):
        for code in ErrorCode:
            assert isinstance(code.value, str)
            assert code.value.startswith("E_")

    def test_member_count(self):
        assert len(ErrorCode) >= 15


class TestLogTag:
    def test_all_values_are_strings(self):
        for tag in LogTag:
            assert isinstance(tag.value, str)


class TestLogRecord:
    def test_creation(self):
        r = LogRecord(level=LogLevel.INFO, message="hello")
        assert r.level == LogLevel.INFO
        assert r.message == "hello"
        assert r.logger == "slo"
        assert r.context == {}
        assert r.exception is None
        assert r.error_code is None
        assert r.tag is None

    def test_with_context(self):
        r = LogRecord(level=LogLevel.ERROR, message="err", context={"req": "123"})
        assert r.context["req"] == "123"

    def test_with_error_code(self):
        r = LogRecord(level=LogLevel.ERROR, message="oom", error_code="E_MODEL_OOM")
        assert r.error_code == "E_MODEL_OOM"

    def test_with_tag(self):
        r = LogRecord(level=LogLevel.INFO, message="ok", tag="REQ")
        assert r.tag == "REQ"


class _CollectingLogger(Logger):
    """Test logger that collects emitted records."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.records = []

    def emit(self, record: LogRecord):
        self.records.append(record)


class TestLogger:
    def test_debug_emits(self):
        log = _CollectingLogger(level=LogLevel.DEBUG)
        log.debug("test msg")
        assert len(log.records) == 1
        assert log.records[0].level == LogLevel.DEBUG

    def test_info_emits(self):
        log = _CollectingLogger(level=LogLevel.DEBUG)
        log.info("test")
        assert len(log.records) == 1
        assert log.records[0].level == LogLevel.INFO

    def test_warning_emits(self):
        log = _CollectingLogger(level=LogLevel.DEBUG)
        log.warning("test")
        assert log.records[0].level == LogLevel.WARNING

    def test_error_emits(self):
        log = _CollectingLogger(level=LogLevel.DEBUG)
        log.error("test", exception="ValueError: bad")
        assert log.records[0].level == LogLevel.ERROR
        assert log.records[0].exception == "ValueError: bad"

    def test_critical_emits(self):
        log = _CollectingLogger(level=LogLevel.DEBUG)
        log.critical("test")
        assert log.records[0].level == LogLevel.CRITICAL

    def test_level_filtering(self):
        log = _CollectingLogger(level=LogLevel.WARNING)
        log.debug("nope")
        log.info("nope")
        log.warning("yes")
        log.error("yes")
        assert len(log.records) == 2

    def test_exception_method(self):
        log = _CollectingLogger(level=LogLevel.DEBUG)
        try:
            raise ValueError("oops")
        except ValueError as e:
            log.exception("caught", exc=e)
        assert "ValueError" in log.records[0].exception

    def test_set_context(self):
        log = _CollectingLogger(level=LogLevel.DEBUG)
        log.set_context(req_id="abc")
        log.info("test")
        assert log.records[0].context["req_id"] == "abc"

    def test_clear_context(self):
        log = _CollectingLogger(level=LogLevel.DEBUG, context={"k": "v"})
        log.clear_context()
        log.info("test")
        assert log.records[0].context == {}

    def test_name_property(self):
        log = _CollectingLogger(name="slo.test")
        assert log.name == "slo.test"

    def test_level_property_setter(self):
        log = _CollectingLogger(level=LogLevel.DEBUG)
        log.level = LogLevel.ERROR
        assert log.level == LogLevel.ERROR

    def test_repr(self):
        log = _CollectingLogger(name="slo.x", level=LogLevel.INFO)
        assert "slo.x" in repr(log)


class TestTaggedLogger:
    def test_tag_attached(self):
        log = _CollectingLogger(level=LogLevel.DEBUG)
        tagged = log.tag("MODEL")
        tagged.info("loaded")
        assert log.records[0].tag == "MODEL"

    def test_tag_always_uses_parent_tag(self):
        log = _CollectingLogger(level=LogLevel.DEBUG)
        tagged = log.tag("DEFAULT")
        tagged.info("test", tag="ignored")
        assert log.records[0].tag == "DEFAULT"

    def test_level_delegates_to_parent(self):
        log = _CollectingLogger(level=LogLevel.WARNING)
        tagged = log.tag("REQ")
        tagged.info("nope")
        assert len(log.records) == 0
        tagged.warning("yes")
        assert len(log.records) == 1


class TestChildLogger:
    def test_emits_through_parent(self):
        log = _CollectingLogger(level=LogLevel.DEBUG)
        child = log.child("sub")
        child.info("from child")
        assert len(log.records) == 1
        assert log.records[0].logger == "slo.sub"

    def test_level_delegates_to_parent(self):
        log = _CollectingLogger(level=LogLevel.ERROR)
        child = log.child("sub")
        child.warning("nope")
        assert len(log.records) == 0

    def test_context_merged(self):
        log = _CollectingLogger(level=LogLevel.DEBUG, context={"a": 1})
        child = log.child("sub", b=2)
        child.info("test")
        assert log.records[0].context["a"] == 1
        assert log.records[0].context["b"] == 2


class TestBridgeHandler:
    def _make_record(self, level, msg, **extra):
        record = logging.LogRecord(
            name="slo.test", level=level, pathname="", lineno=0,
            msg=msg, args=(), exc_info=None,
        )
        for k, v in extra.items():
            setattr(record, k, v)
        return record

    def test_info_level_mapping(self):
        mock_logger = MagicMock()
        handler = BridgeHandler(mock_logger)
        record = self._make_record(logging.INFO, "hello")
        handler.emit(record)
        mock_logger.emit.assert_called_once()
        emitted = mock_logger.emit.call_args[0][0]
        assert emitted.level == LogLevel.INFO
        assert emitted.message == "hello"

    def test_error_level_mapping(self):
        mock_logger = MagicMock()
        handler = BridgeHandler(mock_logger)
        record = self._make_record(logging.ERROR, "fail")
        handler.emit(record)
        emitted = mock_logger.emit.call_args[0][0]
        assert emitted.level == LogLevel.ERROR

    def test_extra_context_merged(self):
        mock_logger = MagicMock()
        handler = BridgeHandler(mock_logger)
        record = self._make_record(logging.INFO, "test", context={"req": "123"})
        handler.emit(record)
        emitted = mock_logger.emit.call_args[0][0]
        assert emitted.context["req"] == "123"

    def test_error_code_extracted(self):
        mock_logger = MagicMock()
        handler = BridgeHandler(mock_logger)
        record = self._make_record(logging.ERROR, "oom", error_code="E_MODEL_OOM")
        handler.emit(record)
        emitted = mock_logger.emit.call_args[0][0]
        assert emitted.error_code == "E_MODEL_OOM"

    def test_tag_extracted(self):
        mock_logger = MagicMock()
        handler = BridgeHandler(mock_logger)
        record = self._make_record(logging.INFO, "req", tag="REQ")
        handler.emit(record)
        emitted = mock_logger.emit.call_args[0][0]
        assert emitted.tag == "REQ"

    def test_exception_captured(self):
        mock_logger = MagicMock()
        handler = BridgeHandler(mock_logger)
        record = logging.LogRecord(
            name="slo.test", level=logging.ERROR, pathname="", lineno=0,
            msg="err", args=(), exc_info=(ValueError, ValueError("bad"), None),
        )
        handler.emit(record)
        emitted = mock_logger.emit.call_args[0][0]
        assert "ValueError" in emitted.exception

    def test_debug_includes_path_and_line(self):
        mock_logger = MagicMock()
        handler = BridgeHandler(mock_logger)
        record = self._make_record(logging.DEBUG, "trace", pathname="/foo.py", lineno=42)
        handler.emit(record)
        emitted = mock_logger.emit.call_args[0][0]
        assert emitted.context["path"] == "/foo.py"
        assert emitted.context["line"] == 42

    def test_info_excludes_path_and_line(self):
        mock_logger = MagicMock()
        handler = BridgeHandler(mock_logger)
        record = self._make_record(logging.INFO, "info", pathname="/foo.py", lineno=42)
        handler.emit(record)
        emitted = mock_logger.emit.call_args[0][0]
        assert "path" not in emitted.context

    def test_level_map_completeness(self):
        assert logging.DEBUG in _LEVEL_MAP
        assert logging.INFO in _LEVEL_MAP
        assert logging.WARNING in _LEVEL_MAP
        assert logging.ERROR in _LEVEL_MAP
        assert logging.CRITICAL in _LEVEL_MAP
