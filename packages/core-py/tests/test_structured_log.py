"""Tests for structured logging."""

from __future__ import annotations

import json
import logging
import time
from io import StringIO

import pytest

from domains.infrastructure.structured_log import (
    JSONFormatter,
    LogContext,
    StructuredLogger,
    get_request_id,
    get_log_context,
    setup_structured_logging,
)


class TestJSONFormatter:
    def test_format_basic(self):
        fmt = JSONFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello", (), None)
        out = json.loads(fmt.format(record))
        assert out["msg"] == "hello"
        assert out["level"] == "INFO"
        assert out["logger"] == "test"
        assert "ts" in out

    def test_format_with_extra(self):
        fmt = JSONFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello", (), None)
        # Simulate Logger._log() which does record.__dict__.update(extra)
        record.__dict__.update({"tag": "TEST", "tokens": 42})
        out = json.loads(fmt.format(record))
        assert out["tag"] == "TEST"
        assert out["tokens"] == 42

    def test_format_with_context(self):
        fmt = JSONFormatter()
        with LogContext(request_id="abc", model_id="gpt2"):
            record = logging.LogRecord("test", logging.INFO, "", 0, "ctx work", (), None)
            out = json.loads(fmt.format(record))
            assert out["request_id"] == "abc"
            assert out["model_id"] == "gpt2"
            assert out["msg"] == "ctx work"


class TestLogContext:
    def test_sets_request_id(self):
        with LogContext(request_id="test-123"):
            assert get_request_id() == "test-123"

    def test_restores_after_exit(self):
        with LogContext(request_id="outer"):
            with LogContext(request_id="inner"):
                assert get_request_id() == "inner"
            assert get_request_id() == "outer"

    def test_auto_generates_request_id(self):
        with LogContext(model_id="test"):
            rid = get_request_id()
            assert rid is not None
            assert len(rid) == 8

    def test_context_merge(self):
        with LogContext(a=1, b=2):
            with LogContext(b=3, c=4):
                ctx = get_log_context()
                assert ctx["a"] == 1
                assert ctx["b"] == 3
                assert ctx["c"] == 4

    def test_context_is_thread_local(self):
        from threading import Thread
        results = []
        with LogContext(request_id="main"):
            def worker():
                results.append(get_request_id())
            t = Thread(target=worker)
            t.start()
            t.join()
        # Worker thread should not inherit main thread's context
        assert results[0] is None or results[0] != "main"


class TestStructuredLogger:
    def test_info_with_extras(self):
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(JSONFormatter())
        log = StructuredLogger("test.slogger", level=logging.DEBUG)
        log._logger.addHandler(handler)
        log._logger.setLevel(logging.DEBUG)

        log.info("hello", tag="TEST", count=10)
        handler.flush()
        out = json.loads(buf.getvalue())
        assert out["msg"] == "hello"
        assert out["tag"] == "TEST"
        assert out["count"] == 10
        assert out["level"] == "INFO"

    def test_context_inherits_to_wrapper(self):
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(JSONFormatter())
        log = StructuredLogger("test.ctx", level=logging.DEBUG)
        log._logger.addHandler(handler)
        log._logger.setLevel(logging.DEBUG)

        with LogContext(request_id="ctx-99"):
            log.info("inside context")
        handler.flush()
        out = json.loads(buf.getvalue())
        assert out["request_id"] == "ctx-99"

    def test_structured_logger_proxies_standard_attrs(self):
        log = StructuredLogger("test.proxy")
        assert log._logger.name == "test.proxy"
        assert hasattr(log, "info")
        assert hasattr(log, "debug")
        assert hasattr(log, "warning")
        assert hasattr(log, "error")


class TestSetupStructuredLogging:
    def test_root_handler_installed(self):
        setup_structured_logging(logging.DEBUG)
        root = logging.getLogger()
        assert any(isinstance(h.formatter, JSONFormatter) for h in root.handlers)
        # Should have removed old handlers
        assert len(root.handlers) >= 1
