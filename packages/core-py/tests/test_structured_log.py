"""Tests for structured logging."""

from __future__ import annotations

import json
import logging
import sys
import time
from io import StringIO

import pytest

from domains.infrastructure.structured_log import (
    JSONFormatter,
    LogContext,
    StructuredLogger,
    get_request_id,
    get_log_context,
    request_log_middleware,
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

    def test_format_with_exception(self):
        fmt = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            record = logging.LogRecord(
                "test", logging.ERROR, "", 0, "failed", (), sys.exc_info()
            )
        out = json.loads(fmt.format(record))
        assert "exception" in out
        assert "ValueError" in out["exception"]


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

    def test_structured_logger_proxies_undeclared_attrs(self):
        log = StructuredLogger("test.proxy2")
        assert log.name == "test.proxy2"
        assert log.getChild("sub").name == "test.proxy2.sub"

    def test_info_with_positional_args(self):
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(JSONFormatter())
        log = StructuredLogger("test.args", level=logging.DEBUG)
        log._logger.addHandler(handler)
        log._logger.setLevel(logging.DEBUG)

        log.info("value %s", 42)
        handler.flush()
        out = json.loads(buf.getvalue())
        assert out["msg"] == "value 42"

    def test_info_with_non_dict_extra(self):
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(JSONFormatter())
        log = StructuredLogger("test.extra", level=logging.DEBUG)
        log._logger.addHandler(handler)
        log._logger.setLevel(logging.DEBUG)

        log.info("hello", tag="x", extra="not-a-dict")
        handler.flush()
        out = json.loads(buf.getvalue())
        assert out["tag"] == "x"

    @pytest.mark.parametrize("method,level,msg", [
        ("debug", "DEBUG", "dbg"),
        ("warning", "WARNING", "warn"),
        ("error", "ERROR", "err"),
        ("critical", "CRITICAL", "crit"),
    ])
    def test_level_methods(self, method, level, msg):
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(JSONFormatter())
        log = StructuredLogger("test.levels", level=logging.DEBUG)
        log._logger.addHandler(handler)
        log._logger.setLevel(logging.DEBUG)

        getattr(log, method)(msg)
        handler.flush()
        out = json.loads(buf.getvalue())
        assert out["msg"] == msg
        assert out["level"] == level


class TestSetupStructuredLogging:
    def test_root_handler_installed(self):
        setup_structured_logging(logging.DEBUG)
        root = logging.getLogger()
        assert any(isinstance(h.formatter, JSONFormatter) for h in root.handlers)
        # Should have removed old handlers
        assert len(root.handlers) >= 1


class _Headers:
    def __init__(self, mapping):
        self._m = mapping

    def get(self, key, default=None):
        return self._m.get(key, default)


class _URL:
    path = "/api/models"


class _Request:
    method = "POST"

    def __init__(self, request_id=None):
        self.headers = _Headers({"X-Request-Id": request_id} if request_id else {})
        self.url = _URL()


class _Response:
    status_code = 200

    def __init__(self):
        self.headers = {}


class TestRequestLogMiddleware:
    async def test_generates_request_id_and_sets_header(self):
        req = _Request()
        resp = _Response()

        async def call_next(r):
            assert r.url.path == "/api/models"
            assert get_request_id() is not None
            return resp

        out = await request_log_middleware(req, call_next)
        assert out is resp
        assert len(resp.headers["X-Request-Id"]) == 8

    async def test_uses_incoming_request_id(self):
        req = _Request(request_id="abc-1234")
        resp = _Response()
        seen = {}

        async def call_next(r):
            seen["rid"] = get_request_id()
            return resp

        await request_log_middleware(req, call_next)
        assert seen["rid"] == "abc-1234"
        assert resp.headers["X-Request-Id"] == "abc-1234"
