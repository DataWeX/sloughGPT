"""Tests for the OOP logging hierarchy (domains.logging)."""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from unittest.mock import MagicMock

import pytest

import sys as _sys
_sys.path.insert(0, "/Users/mac/sloughGPT/packages/core-py")

from domains.logging import (
    LogLevel,
    LogRecord,
    Logger,
    ChildLogger,
    ConsoleLogger,
    CLILogger,
    ShellLogger,
    WebLogger,
    BridgeHandler,
    get_logger,
    set_global,
    get_global,
)


# ══════════════════════════════════════════════════════════════════════
#  LogLevel
# ══════════════════════════════════════════════════════════════════════

class TestLogLevel:
    def test_ordering(self):
        assert LogLevel.DEBUG < LogLevel.INFO < LogLevel.WARNING < LogLevel.ERROR < LogLevel.CRITICAL

    def test_ge(self):
        assert LogLevel.INFO >= LogLevel.DEBUG
        assert LogLevel.WARNING >= LogLevel.WARNING

    def test_le(self):
        assert LogLevel.DEBUG <= LogLevel.INFO
        assert LogLevel.ERROR <= LogLevel.ERROR

    def test_comparison_with_non_loglevel_returns_not_implemented(self):
        assert LogLevel.INFO.__ge__("info") is NotImplemented
        assert LogLevel.INFO.__gt__("info") is NotImplemented
        assert LogLevel.INFO.__le__("info") is NotImplemented
        assert LogLevel.INFO.__lt__("info") is NotImplemented


# ══════════════════════════════════════════════════════════════════════
#  LogRecord
# ══════════════════════════════════════════════════════════════════════

class TestLogRecord:
    def test_defaults(self):
        r = LogRecord(level=LogLevel.INFO, message="hello")
        assert r.level == LogLevel.INFO
        assert r.message == "hello"
        assert r.logger == "man"
        assert r.context == {}
        assert r.exception is None
        assert r.timestamp > 0

    def test_frozen(self):
        r = LogRecord(level=LogLevel.WARNING, message="test")
        with pytest.raises(AttributeError):
            r.message = "changed"  # type: ignore

    def test_context(self):
        r = LogRecord(level=LogLevel.INFO, message="ok", context={"port": 8000})
        assert r.context["port"] == 8000


# ══════════════════════════════════════════════════════════════════════
#  Logger ABC
# ══════════════════════════════════════════════════════════════════════

class _MemoryLogger(Logger):
    """Test logger that stores emitted records in a list."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.records = []  # type: list[LogRecord]

    def emit(self, record: LogRecord) -> None:
        self.records.append(record)


class TestLoggerABC:
    def test_debug_below_level_not_emitted(self):
        log = _MemoryLogger(level=LogLevel.INFO)
        log.debug("should not appear")
        assert len(log.records) == 0

    def test_info_emitted(self):
        log = _MemoryLogger(level=LogLevel.INFO)
        log.info("hello")
        assert len(log.records) == 1
        assert log.records[0].message == "hello"
        assert log.records[0].level == LogLevel.INFO

    def test_context_merged(self):
        log = _MemoryLogger(context={"a": 1})
        log.info("test", b=2)
        assert log.records[0].context == {"a": 1, "b": 2}

    def test_context_override(self):
        log = _MemoryLogger(context={"a": 1})
        log.info("test", a=99)
        assert log.records[0].context == {"a": 99}

    def test_exception_method(self):
        log = _MemoryLogger()
        try:
            raise ValueError("bad")
        except ValueError as e:
            log.exception("failed", exc=e)
        assert len(log.records) == 1
        assert "ValueError: bad" in log.records[0].exception

    def test_child_logger(self):
        parent = _MemoryLogger(name="man.api")
        child = parent.child("inference")
        child.info("generating")
        assert len(parent.records) == 1
        assert parent.records[0].logger == "man.api.inference"

    def test_child_inherits_level(self):
        parent = _MemoryLogger(name="man", level=LogLevel.WARNING)
        child = parent.child("sub")
        child.info("should not emit")
        assert len(parent.records) == 0
        child.warning("should emit")
        assert len(parent.records) == 1

    def test_set_level(self):
        log = _MemoryLogger(level=LogLevel.WARNING)
        log.info("suppressed")
        assert len(log.records) == 0
        log.level = LogLevel.DEBUG
        log.info("now emits")
        assert len(log.records) == 1

    def test_repr(self):
        log = _MemoryLogger(name="man.test", level=LogLevel.ERROR)
        assert "man.test" in repr(log)
        assert "error" in repr(log)


# ══════════════════════════════════════════════════════════════════════
#  ConsoleLogger
# ══════════════════════════════════════════════════════════════════════

class TestConsoleLogger:
    def test_emit_writes_to_stream(self):
        stream = io.StringIO()
        log = ConsoleLogger("man.test", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.info("hello world")
        output = stream.getvalue()
        assert "hello world" in output
        assert "man.test" in output
        assert "INFO" in output

    def test_colors_disabled_no_ansi(self):
        stream = io.StringIO()
        log = ConsoleLogger("man.test", stream=stream, colors=False)
        log.warning("careful")
        output = stream.getvalue()
        # No ANSI escape codes when colors=False
        assert "\033[" not in output

    def test_exception_in_output(self):
        stream = io.StringIO()
        log = ConsoleLogger("man.test", stream=stream, colors=False)
        log.error("boom", exception="RuntimeError: OOM")
        output = stream.getvalue()
        assert "boom" in output
        assert "RuntimeError: OOM" in output

    def test_context_in_output(self):
        stream = io.StringIO()
        log = ConsoleLogger("man.test", stream=stream, colors=False)
        log.info("started", port=8000)
        output = stream.getvalue()
        assert "port=8000" in output


# ══════════════════════════════════════════════════════════════════════
#  ShellLogger
# ══════════════════════════════════════════════════════════════════════

class TestShellLogger:
    def test_emit_writes_to_stream(self):
        stream = io.StringIO()
        log = ShellLogger("man.shell", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.info("test message")
        output = stream.getvalue()
        assert "test message" in output
        assert "man.shell" in output


# ══════════════════════════════════════════════════════════════════════
#  WebLogger
# ══════════════════════════════════════════════════════════════════════

class TestWebLogger:
    def test_record_to_dict(self):
        log = WebLogger("man.web")
        r = LogRecord(level=LogLevel.INFO, message="test", context={"k": "v"})
        d = log._record_to_dict(r)
        assert d["level"] == "info"
        assert d["message"] == "test"
        assert d["context"]["k"] == "v"

    def test_to_json_and_back(self):
        log = WebLogger("man.web")
        r = LogRecord(level=LogLevel.WARNING, message="hello", context={"a": 1})
        raw = log.to_json(r)
        r2 = log.from_json(raw)
        assert r2.level == LogLevel.WARNING
        assert r2.message == "hello"
        assert r2.context["a"] == 1

    def test_emit_with_console(self):
        mock_console = MagicMock()
        log = WebLogger("man.web", console=mock_console)
        log.info("browser test")
        mock_console.log.assert_called_once()
        args = mock_console.log.call_args[0]
        assert "browser test" in args[0]

    def test_emit_with_writable(self):
        stream = io.StringIO()
        log = WebLogger("man.web", writable=stream)
        log.error("ssr error")
        output = stream.getvalue()
        data = json.loads(output)
        assert data["level"] == "error"
        assert data["message"] == "ssr error"


# ══════════════════════════════════════════════════════════════════════
#  BridgeHandler
# ══════════════════════════════════════════════════════════════════════

class TestBridgeHandler:
    def test_routes_standard_logging_to_our_logger(self):
        mem = _MemoryLogger(name="man")
        handler = BridgeHandler(mem)

        std_logger = logging.getLogger("man.test.bridge")
        std_logger.addHandler(handler)
        std_logger.setLevel(logging.DEBUG)

        std_logger.info("from standard logging")
        assert len(mem.records) == 1
        assert mem.records[0].message == "from standard logging"
        assert mem.records[0].logger == "man.test.bridge"

        std_logger.removeHandler(handler)

    def test_exception_captured(self):
        mem = _MemoryLogger(name="man")
        handler = BridgeHandler(mem)

        std_logger = logging.getLogger("man.test.exc")
        std_logger.addHandler(handler)
        std_logger.setLevel(logging.DEBUG)

        try:
            raise TypeError("oops")
        except TypeError:
            std_logger.exception("failed")

        assert len(mem.records) == 1
        assert "TypeError" in mem.records[0].exception

        std_logger.removeHandler(handler)


# ══════════════════════════════════════════════════════════════════════
#  Factory & Globals
# ══════════════════════════════════════════════════════════════════════

class TestFactory:
    def test_get_logger_api(self):
        log = get_logger("api", name="man.test")
        assert isinstance(log, ConsoleLogger)

    def test_get_logger_cli(self):
        log = get_logger("cli", name="man.test")
        assert isinstance(log, CLILogger)

    def test_get_logger_shell(self):
        log = get_logger("shell", name="man.test")
        assert isinstance(log, ShellLogger)

    def test_get_logger_web(self):
        log = get_logger("web", name="man.test")
        assert isinstance(log, WebLogger)

    def test_get_logger_aliases(self):
        assert isinstance(get_logger("server"), ConsoleLogger)
        assert isinstance(get_logger("console"), ConsoleLogger)
        assert isinstance(get_logger("repl"), ShellLogger)
        assert isinstance(get_logger("browser"), WebLogger)

    def test_get_logger_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown logger interface"):
            get_logger("nonexistent")

    def test_set_and_get_global(self):
        mem = _MemoryLogger(name="global.test")
        set_global(mem)
        assert get_global() is mem

    def test_global_default_is_console(self):
        set_global(None)  # type: ignore
        g = get_global()
        assert isinstance(g, ConsoleLogger)
