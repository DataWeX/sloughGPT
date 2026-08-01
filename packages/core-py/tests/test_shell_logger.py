"""Tests for ShellLogger — ANSI formatting, levels, stream handling."""

import io
from datetime import datetime

import pytest

from domains.logging.base import LogLevel, LogRecord
from domains.logging.shell_logger import ShellLogger


def _record(
    message="hello",
    level=LogLevel.INFO,
    logger="slo.shell",
    context=None,
    exception=None,
    timestamp=1_700_000_000.0,
):
    return LogRecord(
        level=level,
        message=message,
        logger=logger,
        timestamp=timestamp,
        context=context or {},
        exception=exception,
    )


@pytest.fixture
def ansi_on(monkeypatch):
    import domains.logging.shell_logger as sh

    codes = {
        "RESET": "\033[0m",
        "BOLD": "\033[1m",
        "DIM": "\033[2m",
        "RED": "\033[31m",
        "GREEN": "\033[32m",
        "YELLOW": "\033[33m",
        "CYAN": "\033[36m",
        "GREY": "\033[90m",
    }
    for name, code in codes.items():
        monkeypatch.setattr(sh._Ansi, name, code)


class TestConstruction:

    def test_default_stream(self):
        log = ShellLogger("slo.shell")
        assert log.name == "slo.shell"
        assert log.level == LogLevel.INFO

    def test_override_stream(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf)
        assert log._stream is buf

    def test_colors_flag(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, colors=False)
        assert log._colors is False
        log2 = ShellLogger("slo.shell", stream=buf, colors=True)
        assert log2._colors is True


class TestFormatting:

    def test_plain_line(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(message="model loaded", context={"model": "gpt2"}))
        assert "model loaded" in line
        assert "INFO" in line
        assert "model=gpt2" in line
        assert "slo.shell" in line

    def test_icon_per_level(self):
        log = ShellLogger("slo.shell", colors=False)
        assert "ℹ" in log._format_record(_record(level=LogLevel.INFO))
        assert "✗" in log._format_record(_record(level=LogLevel.ERROR))
        assert "✗" in log._format_record(_record(level=LogLevel.CRITICAL))
        assert "!" in log._format_record(_record(level=LogLevel.WARNING))
        assert "·" in log._format_record(_record(level=LogLevel.DEBUG))

    def test_exception_included(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(message="failed", exception="FileNotFoundError"))
        assert "FileNotFoundError" in line

    def test_colors_emit_ansi(self, ansi_on):
        log = ShellLogger("slo.shell", colors=True)
        line = log._format_record(_record(message="x"))
        assert "\033[" in line

    def test_no_colors_no_ansi(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(message="x"))
        assert "\033[" not in line

    def test_time_formatted(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(timestamp=1_700_000_000.0))
        expected = datetime.fromtimestamp(1_700_000_000.0).strftime("%H:%M:%S")
        assert expected in line


class TestEmit:

    def test_emit_writes_line(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, colors=False)
        log.info("ready")
        assert "ready" in buf.getvalue()

    def test_emit_respects_level(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, level=LogLevel.WARNING, colors=False)
        log.debug("hidden")
        log.error("shown")
        out = buf.getvalue()
        assert "hidden" not in out
        assert "shown" in out

    def test_emit_swallows_stream_error(self):
        class Broken:
            def write(self, _):
                raise OSError("closed")

            def flush(self):
                raise OSError("closed")

        log = ShellLogger("slo.shell", stream=Broken(), colors=False)
        log.info("won't raise")
