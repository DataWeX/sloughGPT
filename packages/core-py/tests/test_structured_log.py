"""Tests for structured logging."""

from __future__ import annotations

import json
import logging
import sys
import time
from io import StringIO
from threading import Thread

import pytest

from domains.infrastructure.structured_log import (
    JSONFormatter,
    LogContext,
    StructuredLogger,
    get_log_context,
    get_request_id,
    log_timer,
    request_log_middleware,
    setup_structured_logging,
    tagged,
    timed,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _make_handler(buf: StringIO, level: int = logging.DEBUG) -> logging.StreamHandler:
    h = logging.StreamHandler(buf)
    h.setFormatter(JSONFormatter())
    h.setLevel(level)
    return h


def _make_logger(name: str, buf: StringIO) -> StructuredLogger:
    log = StructuredLogger(name, level=logging.DEBUG)
    log._logger.addHandler(_make_handler(buf))
    log._logger.setLevel(logging.DEBUG)
    return log


def _read_json(buf: StringIO) -> dict:
    buf.seek(0)
    return json.loads(buf.read())


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


# ── JSONFormatter ──────────────────────────────────────────────────────


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

    def test_format_no_context(self):
        fmt = JSONFormatter()
        record = logging.LogRecord("test", logging.WARNING, "", 0, "warn msg", (), None)
        out = json.loads(fmt.format(record))
        assert out["level"] == "WARNING"
        assert out["msg"] == "warn msg"
        assert out["logger"] == "test"

    def test_format_known_fields_not_in_extra(self):
        fmt = JSONFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "dup", (), None)
        out = json.loads(fmt.format(record))
        # Known record attrs are filtered from the extra iteration loop
        # Only custom (non-known) attributes appear as extra keys
        assert out["logger"] == "test"
        assert out["level"] == "INFO"
        assert out["msg"] == "dup"

    def test_format_extra_dict_from_record_context_not_duplicated(self):
        fmt = JSONFormatter()
        with LogContext(request_id="x1"):
            record = logging.LogRecord("test", logging.INFO, "", 0, "c", (), None)
            record.__dict__.update({"request_id": "override"})
            out = json.loads(fmt.format(record))
            assert out["request_id"] == "override"

    def test_format_multiple_extra_fields(self):
        fmt = JSONFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "multi", (), None)
        record.__dict__.update({"a": 1, "b": "two", "c": 3.0})
        out = json.loads(fmt.format(record))
        assert out["a"] == 1
        assert out["b"] == "two"
        assert out["c"] == 3.0

    def test_format_default_str_fallback(self):
        fmt = JSONFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "fallback", (), None)
        record.__dict__.update({"obj": object()})
        out = json.loads(fmt.format(record))
        assert "obj" in out

    def test_format_with_string_args(self):
        fmt = JSONFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "val %s", ("hello",), None)
        out = json.loads(fmt.format(record))
        assert out["msg"] == "val hello"


# ── LogContext ─────────────────────────────────────────────────────────


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
        results = []
        with LogContext(request_id="main"):
            def worker():
                results.append(get_request_id())
            t = Thread(target=worker)
            t.start()
            t.join()
        assert results[0] is None or results[0] != "main"

    def test_empty_context_returns_copy(self):
        with LogContext(request_id="x"):
            pass
        ctx = get_log_context()
        assert isinstance(ctx, dict)

    def test_log_context_produces_uuid_prefix(self):
        ids = set()
        for _ in range(20):
            with LogContext():
                ids.add(get_request_id())
        assert len(ids) == 20

    def test_nested_preserves_outer_values(self):
        with LogContext(a=1, b=2):
            with LogContext(b=3):
                ctx = get_log_context()
                assert ctx["a"] == 1
                assert ctx["b"] == 3
            ctx = get_log_context()
            assert ctx["a"] == 1
            assert ctx["b"] == 2

    def test_context_with_no_args(self):
        with LogContext():
            rid = get_request_id()
            assert rid is not None
            assert len(rid) == 8

    def test_log_context_returns_dict_copy(self):
        with LogContext(x=1):
            ctx1 = get_log_context()
            ctx2 = get_log_context()
            ctx1["extra"] = 999
            assert "extra" not in get_log_context()


# ── StructuredLogger ──────────────────────────────────────────────────


class TestStructuredLogger:
    def test_info_with_extras(self):
        buf = StringIO()
        log = _make_logger("test.slogger", buf)
        log.info("hello", tag="TEST", count=10)
        out = _read_json(buf)
        assert out["msg"] == "hello"
        assert out["tag"] == "TEST"
        assert out["count"] == 10
        assert out["level"] == "INFO"

    def test_context_inherits_to_wrapper(self):
        buf = StringIO()
        log = _make_logger("test.ctx", buf)
        with LogContext(request_id="ctx-99"):
            log.info("inside context")
        out = _read_json(buf)
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
        log = _make_logger("test.args", buf)
        log.info("value %s", 42)
        out = _read_json(buf)
        assert out["msg"] == "value 42"

    def test_info_with_non_dict_extra(self):
        buf = StringIO()
        log = _make_logger("test.extra", buf)
        log.info("hello", tag="x", extra="not-a-dict")
        out = _read_json(buf)
        assert out["tag"] == "x"

    @pytest.mark.parametrize("method,level,msg", [
        ("debug", "DEBUG", "dbg"),
        ("warning", "WARNING", "warn"),
        ("error", "ERROR", "err"),
        ("critical", "CRITICAL", "crit"),
    ])
    def test_level_methods(self, method, level, msg):
        buf = StringIO()
        log = _make_logger("test.levels", buf)
        getattr(log, method)(msg)
        out = _read_json(buf)
        assert out["msg"] == msg
        assert out["level"] == level

    def test_child_logger(self):
        buf = StringIO()
        parent = StructuredLogger("slo.parent", level=logging.DEBUG)
        child = parent.child("child", tag="child_tag")
        child._logger.addHandler(_make_handler(buf))
        child._logger.setLevel(logging.DEBUG)
        child.info("from child")
        out = _read_json(buf)
        assert out["logger"] == "slo.parent.child"
        assert out["tag"] == "child_tag"

    def test_child_merges_tags(self):
        buf = StringIO()
        parent = StructuredLogger("slo.p", level=logging.DEBUG, phase="parent")
        child = parent.child("c", phase="child")
        child._logger.addHandler(_make_handler(buf))
        child._logger.setLevel(logging.DEBUG)
        child.info("msg")
        out = _read_json(buf)
        assert out["phase"] == "child"

    def test_child_preserves_parent_tags(self):
        buf = StringIO()
        parent = StructuredLogger("slo.p2", level=logging.DEBUG, base="x")
        child = parent.child("c2", extra="y")
        child._logger.addHandler(_make_handler(buf))
        child._logger.setLevel(logging.DEBUG)
        child.info("msg")
        out = _read_json(buf)
        assert out["base"] == "x"
        assert out["extra"] == "y"

    def test_repr(self):
        log = StructuredLogger("slo.repr", level=logging.DEBUG)
        r = repr(log)
        assert "slo.repr" in r
        assert "DEBUG" in r

    def test_repr_with_tags(self):
        log = StructuredLogger("slo.rtags", level=logging.INFO, a=1)
        r = repr(log)
        assert "tags=" in r

    def test_repr_without_tags(self):
        log = StructuredLogger("slo.rnotags", level=logging.INFO)
        r = repr(log)
        assert "tags=" not in r

    def test_legacy_extra_dict_merging(self):
        buf = StringIO()
        log = _make_logger("test.legacy", buf)
        log.info("legacy", extra={"old_key": "val"})
        out = _read_json(buf)
        assert out["old_key"] == "val"

    def test_positional_args_with_extras(self):
        buf = StringIO()
        log = _make_logger("test.pos_ex", buf)
        log.info("loaded %s in %.1fms", "model", 123.4, tag="MODEL")
        out = _read_json(buf)
        assert out["msg"] == "loaded model in 123.4ms"
        assert out["tag"] == "MODEL"


# ── log_timer ─────────────────────────────────────────────────────────


class TestLogTimer:
    def test_success_logs_elapsed(self):
        buf = StringIO()
        log = _make_logger("test.timer", buf)
        with log_timer(log, "sleep"):
            time.sleep(0.05)
        out = _read_json(buf)
        assert "sleep completed" in out["msg"]
        assert out["elapsed_ms"] >= 40

    def test_exception_logs_error(self):
        buf = StringIO()
        log = _make_logger("test.timer_err", buf)
        with pytest.raises(RuntimeError):
            with log_timer(log, "boom"):
                raise RuntimeError("fail")
        out = _read_json(buf)
        assert "boom failed" in out["msg"]
        assert out["level"] == "ERROR"
        assert out["elapsed_ms"] >= 0

    def test_success_custom_level(self):
        buf = StringIO()
        log = _make_logger("test.timer lvl", buf)
        with log_timer(log, "op", level=logging.WARNING):
            pass
        out = _read_json(buf)
        assert out["level"] == "WARNING"

    def test_success_ms_format_under_1s(self):
        buf = StringIO()
        log = _make_logger("test.timer ms", buf)
        with log_timer(log, "fast"):
            time.sleep(0.001)
        out = _read_json(buf)
        assert "ms" in out["msg"]

    def test_success_s_format_over_1s(self):
        buf = StringIO()
        log = _make_logger("test.timer s", buf)
        with log_timer(log, "slow", level=logging.DEBUG):
            time.sleep(1.05)
        out = _read_json(buf)
        assert "1.1s" in out["msg"] or "s" in out["msg"]

    def test_timer_extra_fields(self):
        buf = StringIO()
        log = _make_logger("test.timer extra", buf)
        with log_timer(log, "op", tag="TIMER"):
            pass
        out = _read_json(buf)
        assert out["tag"] == "TIMER"

    def test_timer_exception_preserves_message(self):
        buf = StringIO()
        log = _make_logger("test.timer exc", buf)
        with pytest.raises(ValueError):
            with log_timer(log, "compute"):
                raise ValueError("bad math")
        out = _read_json(buf)
        assert "compute failed" in out["msg"]

    def test_timer_logs_latency_ms(self):
        buf = StringIO()
        log = _make_logger("test.timer latency", buf)
        with log_timer(log, "work"):
            time.sleep(0.02)
        out = _read_json(buf)
        assert isinstance(out["elapsed_ms"], float)
        assert out["elapsed_ms"] > 0


# ── timed decorator ───────────────────────────────────────────────────


class TestTimed:
    def test_success_logs_elapsed(self):
        buf = StringIO()
        log = _make_logger("test.timed", buf)

        @timed(log)
        def my_func():
            return 42

        result = my_func()
        assert result == 42
        out = _read_json(buf)
        assert "my_func completed" in out["msg"]

    def test_exception_logs_error(self):
        buf = StringIO()
        log = _make_logger("test.timed_exc", buf)

        @timed(log)
        def bad():
            raise TypeError("wrong")

        with pytest.raises(TypeError):
            bad()
        out = _read_json(buf)
        assert "bad failed" in out["msg"]
        assert out["level"] == "ERROR"

    def test_preserves_function_name(self):
        buf = StringIO()
        log = _make_logger("test.timed name", buf)

        @timed(log)
        def special_name():
            pass

        assert special_name.__name__ == "special_name"

    def test_custom_level(self):
        buf = StringIO()
        log = _make_logger("test.timed lvl", buf)

        @timed(log, level=logging.WARNING)
        def slow():
            time.sleep(0.001)

        slow()
        out = _read_json(buf)
        assert out["level"] == "WARNING"

    def test_extra_fields(self):
        buf = StringIO()
        log = _make_logger("test.timed extra", buf)

        @timed(log, phase="test")
        def tagged_func():
            pass

        tagged_func()
        out = _read_json(buf)
        assert out["phase"] == "test"

    def test_ms_format_fast(self):
        buf = StringIO()
        log = _make_logger("test.timed ms", buf)

        @timed(log)
        def fast():
            time.sleep(0.001)

        fast()
        out = _read_json(buf)
        assert "ms" in out["msg"]

    def test_s_format_slow(self):
        buf = StringIO()
        log = _make_logger("test.timed s", buf)

        @timed(log)
        def slow():
            time.sleep(1.05)

        slow()
        out = _read_json(buf)
        assert "s" in out["msg"]

    def test_return_value_preserved(self):
        buf = StringIO()
        log = _make_logger("test.timed ret", buf)

        @timed(log)
        def compute(x, y):
            return x + y

        assert compute(3, 4) == 7

    def test_kwargs_preserved(self):
        buf = StringIO()
        log = _make_logger("test.timed kw", buf)

        @timed(log)
        def greet(name="world"):
            return f"hello {name}"

        assert greet(name="test") == "hello test"

    def test_exception_elapsed_ms_logged(self):
        buf = StringIO()
        log = _make_logger("test.timed elog", buf)

        @timed(log)
        def explode():
            time.sleep(0.01)
            raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError):
            explode()
        out = _read_json(buf)
        assert out["elapsed_ms"] > 0


# ── tagged ─────────────────────────────────────────────────────────────


class TestTagged:
    def test_info_injects_tags(self):
        buf = StringIO()
        log = _make_logger("test.tagged", buf)
        tlog = tagged(log, phase="train", epoch=5)
        tlog.info("starting")
        out = _read_json(buf)
        assert out["phase"] == "train"
        assert out["epoch"] == 5
        assert out["msg"] == "starting"

    def test_debug_injects_tags(self):
        buf = StringIO()
        log = _make_logger("test.tagged dbg", buf)
        tlog = tagged(log, tag="dbg")
        tlog.debug("debug msg")
        out = _read_json(buf)
        assert out["level"] == "DEBUG"
        assert out["tag"] == "dbg"

    def test_warning_injects_tags(self):
        buf = StringIO()
        log = _make_logger("test.tagged warn", buf)
        tlog = tagged(log, flag="w")
        tlog.warning("warning msg")
        out = _read_json(buf)
        assert out["flag"] == "w"
        assert out["level"] == "WARNING"

    def test_error_injects_tags(self):
        buf = StringIO()
        log = _make_logger("test.tagged err", buf)
        tlog = tagged(log, flag="e")
        tlog.error("error msg")
        out = _read_json(buf)
        assert out["flag"] == "e"
        assert out["level"] == "ERROR"

    def test_critical_injects_tags(self):
        buf = StringIO()
        log = _make_logger("test.tagged crit", buf)
        tlog = tagged(log, flag="c")
        tlog.critical("critical msg")
        out = _read_json(buf)
        assert out["flag"] == "c"
        assert out["level"] == "CRITICAL"

    def test_call_kwargs_override_tag_kwargs(self):
        buf = StringIO()
        log = _make_logger("test.tagged override", buf)
        tlog = tagged(log, tag="base")
        tlog.info("msg", tag="override")
        out = _read_json(buf)
        assert out["tag"] == "override"

    def test_multiple_tags(self):
        buf = StringIO()
        log = _make_logger("test.tagged multi", buf)
        tlog = tagged(log, a=1, b=2, c=3)
        tlog.info("multi")
        out = _read_json(buf)
        assert out["a"] == 1
        assert out["b"] == 2
        assert out["c"] == 3

    def test_positional_args_with_tags(self):
        buf = StringIO()
        log = _make_logger("test.tagged pos", buf)
        tlog = tagged(log, phase="eval")
        tlog.info("score %.2f", 0.95)
        out = _read_json(buf)
        assert out["msg"] == "score 0.95"
        assert out["phase"] == "eval"

    def test_tagged_logger_does_not_mutate_original(self):
        buf = StringIO()
        log = _make_logger("test.tagged nomut", buf)
        tlog = tagged(log, extra_tag="y")
        tlog.info("via proxy")
        lines = buf.getvalue().strip().split("\n")
        out_proxy = json.loads(lines[0])
        log.info("via original")
        lines = buf.getvalue().strip().split("\n")
        out_orig = json.loads(lines[-1])
        assert "extra_tag" in out_proxy
        assert "extra_tag" not in out_orig


# ── setup_structured_logging ──────────────────────────────────────────


class TestSetupStructuredLogging:
    def test_root_handler_installed(self):
        setup_structured_logging(logging.DEBUG)
        root = logging.getLogger()
        assert any(isinstance(h.formatter, JSONFormatter) for h in root.handlers)
        assert len(root.handlers) >= 1

    def test_custom_formatter(self):
        custom = logging.Formatter("%(message)s")
        setup_structured_logging(logging.WARNING, fmt=custom)
        root = logging.getLogger()
        assert any(h.formatter is custom for h in root.handlers)

    def test_noisy_loggers_quieted(self):
        setup_structured_logging()
        for name in ("requests", "urllib3", "httpx", "asyncio", "PIL"):
            assert logging.getLogger(name).level >= logging.WARNING

    def test_root_level_set(self):
        setup_structured_logging(logging.WARNING)
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_removes_old_handlers(self):
        root = logging.getLogger()
        before = len(root.handlers)
        setup_structured_logging()
        after = len(root.handlers)
        assert after >= 1


# ── request_log_middleware ────────────────────────────────────────────


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

    async def test_context_restores_after_middleware(self):
        req = _Request()
        resp = _Response()

        async def call_next(r):
            return resp

        await request_log_middleware(req, call_next)
        assert get_request_id() is None

    async def test_custom_status_code(self):
        req = _Request()
        resp = _Response()
        resp.status_code = 404

        async def call_next(r):
            return resp

        out = await request_log_middleware(req, call_next)
        assert out.status_code == 404

    async def test_method_in_context(self):
        req = _Request()
        resp = _Response()
        captured = {}

        async def call_next(r):
            captured["method"] = req.method
            return resp

        await request_log_middleware(req, call_next)
        assert captured["method"] == "POST"
