"""Tests for domains.logging.bridge — BridgeHandler and record_extra_context."""

import logging

import pytest
from domains.logging.bridge import BridgeHandler, record_extra_context, _LEVEL_MAP
from domains.logging.base import Logger, LogLevel, LogRecord
from domains.logging.console_logger import ConsoleLogger


class TestRecordExtraContext:
    def test_empty_record(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="test", args=(), exc_info=None,
        )
        assert record_extra_context(record) == {}

    def test_explicit_context(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="test", args=(), exc_info=None,
        )
        record.context = {"key": "value"}
        ctx = record_extra_context(record)
        assert ctx["key"] == "value"

    def test_auto_capture_non_standard_extra(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="test", args=(), exc_info=None,
        )
        record.mode = "guard"
        record.elapsed_ms = 100
        ctx = record_extra_context(record)
        assert ctx["mode"] == "guard"
        assert ctx["elapsed_ms"] == 100

    def test_standard_attrs_excluded(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="/foo", lineno=42,
            msg="test", args=(), exc_info=None,
        )
        ctx = record_extra_context(record)
        assert "name" not in ctx
        assert "pathname" not in ctx
        assert "lineno" not in ctx

    def test_handled_attrs_excluded(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="test", args=(), exc_info=None,
        )
        record.error_code = "E_001"
        record.tag = "INFRA"
        ctx = record_extra_context(record)
        assert "error_code" not in ctx
        assert "tag" not in ctx

    def test_context_wins_over_auto(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="test", args=(), exc_info=None,
        )
        record.context = {"mode": "explicit"}
        record.mode = "auto"
        ctx = record_extra_context(record)
        assert ctx["mode"] == "explicit"


class TestBridgeHandler:
    def _make_handler(self):
        logger = ConsoleLogger("slo.test", level=LogLevel.DEBUG)
        return BridgeHandler(logger)

    def test_emit_debug(self):
        handler = self._make_handler()
        record = logging.LogRecord(
            name="slo.test", level=logging.DEBUG, pathname="/foo", lineno=10,
            msg="debug msg", args=(), exc_info=None,
        )
        handler.emit(record)  # should not raise

    def test_emit_info(self):
        handler = self._make_handler()
        record = logging.LogRecord(
            name="slo.test", level=logging.INFO, pathname="", lineno=0,
            msg="info msg", args=(), exc_info=None,
        )
        handler.emit(record)

    def test_emit_error(self):
        handler = self._make_handler()
        record = logging.LogRecord(
            name="slo.test", level=logging.ERROR, pathname="", lineno=0,
            msg="error msg", args=(), exc_info=None,
        )
        handler.emit(record)

    def test_emit_with_exc_info(self):
        handler = self._make_handler()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="slo.test", level=logging.ERROR, pathname="", lineno=0,
            msg="failed", args=(), exc_info=exc_info,
        )
        handler.emit(record)

    def test_emit_with_extra_fields(self):
        handler = self._make_handler()
        record = logging.LogRecord(
            name="slo.test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=(), exc_info=None,
        )
        record.error_code = "E_MODEL_OOM"
        record.tag = "MODEL"
        record.custom_field = 42
        handler.emit(record)

    def test_emit_with_context_extra(self):
        handler = self._make_handler()
        record = logging.LogRecord(
            name="slo.test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=(), exc_info=None,
        )
        record.context = {"model": "gpt2", "tokens": 512}
        handler.emit(record)

    def test_debug_includes_path_and_line(self):
        handler = self._make_handler()
        record = logging.LogRecord(
            name="slo.test", level=logging.DEBUG, pathname="/foo/bar.py", lineno=42,
            msg="trace", args=(), exc_info=None,
        )
        handler.emit(record)

    def test_unknown_level_maps_to_info(self):
        handler = self._make_handler()
        record = logging.LogRecord(
            name="slo.test", level=999, pathname="", lineno=0,
            msg="unknown", args=(), exc_info=None,
        )
        handler.emit(record)  # should not raise, defaults to INFO


class TestLevelMap:
    def test_all_standard_levels_mapped(self):
        assert _LEVEL_MAP[logging.DEBUG] == LogLevel.DEBUG
        assert _LEVEL_MAP[logging.INFO] == LogLevel.INFO
        assert _LEVEL_MAP[logging.WARNING] == LogLevel.WARNING
        assert _LEVEL_MAP[logging.ERROR] == LogLevel.ERROR
        assert _LEVEL_MAP[logging.CRITICAL] == LogLevel.CRITICAL
