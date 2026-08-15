"""Tests for ConsoleLogger — human/json formatting, exception parsing, levels."""

import io
import json

import pytest

from domains.logging.base import LogLevel, LogRecord
from domains.logging.console_logger import ConsoleLogger


def _record(
    message="hello",
    level=LogLevel.INFO,
    logger="slo.api.inference",
    context=None,
    exception=None,
    error_code=None,
    tag=None,
    timestamp=1_700_000_000.0,
):
    return LogRecord(
        level=level,
        message=message,
        logger=logger,
        timestamp=timestamp,
        context=context or {},
        exception=exception,
        error_code=error_code,
        tag=tag,
    )


_ANSI_CODES = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
    "RED": "\033[31m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "WHITE": "\033[37m",
    "GREY": "\033[90m",
    "BG_RED": "\033[41m",
}


@pytest.fixture
def ansi_on(monkeypatch):
    import domains.logging.console_logger as cl

    for name, code in _ANSI_CODES.items():
        monkeypatch.setattr(cl._Ansi, name, code)


# ── Construction ───────────────────────────────────────────────────────────

class TestConstruction:

    def test_default_stream_is_stderr_capture(self):
        log = ConsoleLogger("slo.api")
        assert log.name == "slo.api"
        assert log.level == LogLevel.INFO

    def test_override_stream_and_format(self):
        buf = io.StringIO()
        log = ConsoleLogger("slo.api", format="json", stream=buf)
        assert log._format == "json"
        assert log._stream is buf

    def test_repr(self):
        log = ConsoleLogger("slo.api")
        assert "ConsoleLogger" in repr(log)


# ── Human format ───────────────────────────────────────────────────────────

class TestHumanFormat:

    def test_info_line(self):
        log = ConsoleLogger("slo.api", colors=False)
        line = log._format_record(_record(message="server started", context={"port": 8000}))
        assert "server started" in line
        assert "INF" in line
        assert "port=8000" in line
        assert "inference" in line  # last segment of logger name

    def test_error_line_includes_exception_and_code(self):
        log = ConsoleLogger("slo.api", colors=False)
        line = log._format_record(
            _record(
                message="load failed",
                level=LogLevel.ERROR,
                exception="RuntimeError: OOM",
                error_code="E_MODEL_OOM",
            )
        )
        assert "ERR" in line
        assert "OOM" in line
        assert "E_MODEL_OOM" in line

    def test_tag_appears_in_brackets(self):
        log = ConsoleLogger("slo.api", colors=False)
        line = log._format_record(_record(message="loaded", tag="MODEL"))
        assert "[MODEL]" in line

    def test_critical_badge(self):
        log = ConsoleLogger("slo.api", colors=False)
        line = log._format_record(_record(message="boom", level=LogLevel.CRITICAL))
        assert "CRI" in line

    def test_colors_emit_ansi_when_enabled(self, ansi_on):
        log = ConsoleLogger("slo.api", colors=True)
        line = log._format_record(_record(message="colored", tag="MODEL"))
        assert "\033[" in line

    def test_colors_true_emits_ansi_even_when_stderr_not_tty(self, monkeypatch):
        # Regression: ANSI codes must not be frozen to "" by import-time tty
        # detection — an explicit colors=True must force colors on any stream.
        monkeypatch.setattr("sys.stderr", io.StringIO())
        log = ConsoleLogger("slo.api", colors=True)
        line = log._format_record(_record(message="colored", tag="MODEL"))
        assert "\033[" in line

    def test_no_colors_no_ansi(self):
        log = ConsoleLogger("slo.api", colors=False)
        line = log._format_record(_record(message="plain", tag="MODEL"))
        assert "\033[" not in line

    def test_unknown_level_fallback(self):
        log = ConsoleLogger("slo.api", colors=False)
        line = log._format_record(_record(message="x", level="weird"))
        assert "???" in line


# ── JSON format ────────────────────────────────────────────────────────────

class TestJsonFormat:

    def test_json_line_is_valid(self):
        log = ConsoleLogger("slo.api", format="json")
        line = log._format_record(_record(message="hello", context={"a": 1}, tag="REQ"))
        data = json.loads(line)
        assert data["level"] == "INFO"
        assert data["logger"] == "slo.api.inference"
        assert data["msg"] == "hello"
        assert data["tag"] == "REQ"
        assert data["ctx"] == {"a": 1}

    def test_json_includes_code_and_exception(self):
        log = ConsoleLogger("slo.api", format="json")
        data = json.loads(
            log._format_record(
                _record(
                    message="failed",
                    level=LogLevel.ERROR,
                    error_code="E_MODEL_CRASH",
                    exception="OSError: nope",
                )
            )
        )
        assert data["code"] == "E_MODEL_CRASH"
        assert data["err"] == "OSError: nope"
        assert data["level"] == "ERROR"

    def test_json_level_mapping(self):
        log = ConsoleLogger("slo.api", format="json")
        assert json.loads(log._format_record(_record(level=LogLevel.WARNING)))["level"] == "WARN"
        assert json.loads(log._format_record(_record(level=LogLevel.DEBUG)))["level"] == "DEBUG"
        assert json.loads(log._format_record(_record(level=LogLevel.CRITICAL)))["level"] == "CRIT"

    def test_json_timestamp_iso(self):
        log = ConsoleLogger("slo.api", format="json")
        data = json.loads(log._format_record(_record(timestamp=1_700_000_000.0)))
        assert data["ts"].startswith("2023-11-14T22:13:20.000+00:00")


# ── Exception parsing ──────────────────────────────────────────────────────

class TestExceptionParsing:

    def test_parse_simple_type_and_message(self):
        log = ConsoleLogger("slo.api")
        exc_type, exc_msg, file_info = log._parse_exception("ValueError: bad input")
        assert exc_type == "ValueError"
        assert exc_msg == "bad input"
        assert file_info is None

    def test_parse_plain_message(self):
        log = ConsoleLogger("slo.api")
        exc_type, exc_msg, file_info = log._parse_exception("just a message")
        assert exc_type is None
        assert exc_msg == "just a message"

    def test_parse_traceback_extracts_file_info(self):
        log = ConsoleLogger("slo.api")
        tb = (
            "Traceback (most recent call last):\n"
            "  File 'domains/foo.py', line 42, in run\n"
            "    do_thing()\n"
            "RuntimeError: exploded"
        )
        exc_type, exc_msg, file_info = log._parse_exception(tb)
        assert exc_type == "RuntimeError"
        assert exc_msg == "exploded"
        assert file_info == "domains/foo.py:42 in run()"

    def test_get_exception_color_programming(self, ansi_on):
        log = ConsoleLogger("slo.api")
        assert log._get_exception_color("ValueError") == "\033[31m" + "\033[1m"

    def test_get_exception_color_system(self, ansi_on):
        log = ConsoleLogger("slo.api")
        assert log._get_exception_color("RuntimeError") == "\033[35m" + "\033[1m"

    def test_get_exception_color_transient(self, ansi_on):
        log = ConsoleLogger("slo.api")
        assert log._get_exception_color("TimeoutError") == "\033[33m" + "\033[1m"

    def test_get_exception_color_dependency(self, ansi_on):
        log = ConsoleLogger("slo.api")
        assert log._get_exception_color("ImportError") == "\033[36m" + "\033[1m"

    def test_get_exception_color_unknown(self, ansi_on):
        log = ConsoleLogger("slo.api")
        assert log._get_exception_color("WeirdError") == "\033[31m"

    def test_format_exception(self):
        log = ConsoleLogger("slo.api", colors=False)
        parts = log._format_exception("TypeError: nope")
        assert "[TypeError]" in parts[0]
        assert "nope" in parts[1]


# ── emit ───────────────────────────────────────────────────────────────────

class TestEmit:

    def test_emit_writes_line(self):
        buf = io.StringIO()
        log = ConsoleLogger("slo.api", stream=buf, colors=False)
        log.info("booted", port=8000)
        out = buf.getvalue()
        assert "booted" in out
        assert out.endswith("\n")

    def test_emit_respects_level(self):
        buf = io.StringIO()
        log = ConsoleLogger("slo.api", stream=buf, level=LogLevel.WARNING, colors=False)
        log.info("ignored")
        log.warning("kept")
        out = buf.getvalue()
        assert "ignored" not in out
        assert "kept" in out

    def test_emit_swallows_stream_error(self):
        class Broken:
            def write(self, _):
                raise OSError("closed")

            def flush(self):
                raise OSError("closed")

        log = ConsoleLogger("slo.api", stream=Broken(), colors=False)
        log.info("won't raise")
