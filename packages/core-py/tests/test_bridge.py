"""Tests for domains.logging.bridge — BridgeHandler and record_extra_context."""

from __future__ import annotations

import logging
import pytest

from domains.logging.bridge import BridgeHandler, record_extra_context
from domains.logging.base import LogLevel
from domains.logging.console_logger import ConsoleLogger


class TestRecordExtraContext:
    def test_empty_record(self):
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        ctx = record_extra_context(record)
        assert isinstance(ctx, dict)

    def test_with_context_extra(self):
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.context = {"key": "value"}
        ctx = record_extra_context(record)
        assert ctx.get("key") == "value"

    def test_with_non_standard_extra(self):
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.custom_field = "custom_value"
        ctx = record_extra_context(record)
        assert ctx.get("custom_field") == "custom_value"

    def test_standard_attrs_excluded(self):
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        ctx = record_extra_context(record)
        assert "name" not in ctx
        assert "msg" not in ctx
        assert "levelname" not in ctx


class TestBridgeHandler:
    def test_emit(self):
        logger = ConsoleLogger("test", level=LogLevel.DEBUG)
        handler = BridgeHandler(logger)
        record = logging.LogRecord("test", logging.INFO, "", 0, "test message", (), None)
        handler.emit(record)

    def test_emit_with_context(self):
        logger = ConsoleLogger("test", level=LogLevel.DEBUG)
        handler = BridgeHandler(logger)
        record = logging.LogRecord("test", logging.INFO, "", 0, "test message", (), None)
        record.context = {"key": "value"}
        handler.emit(record)

    def test_emit_with_error_code(self):
        logger = ConsoleLogger("test", level=LogLevel.DEBUG)
        handler = BridgeHandler(logger)
        record = logging.LogRecord("test", logging.ERROR, "", 0, "error msg", (), None)
        record.error_code = "E_TEST"
        handler.emit(record)
