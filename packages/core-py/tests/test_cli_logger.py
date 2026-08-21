"""Tests for CLILogger — native ANSI CLI logger."""

import io

import pytest

from domains.logging.base import LogLevel, LogRecord
from domains.logging.cli_logger import (
    CLILogger,
    set_cli_terminal,
    _TERMINAL_ENABLED,
    _color_enabled,
    _term_width,
)


def _record(
    message="hello",
    level=LogLevel.INFO,
    logger="slo.cli",
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


@pytest.fixture
def buf():
    return io.StringIO()


@pytest.fixture
def logger(buf):
    return CLILogger("test.slo", level=LogLevel.DEBUG, stream=buf, colors=False)


@pytest.fixture(autouse=True)
def reset_terminal():
    """Reset global terminal state between tests."""
    import domains.logging.cli_logger as mod
    mod._TERMINAL_ENABLED = True
    yield
    mod._TERMINAL_ENABLED = True


# ── Construction ────────────────────────────────────────────────────────

class TestConstruction:
    def test_creates_with_name(self, logger):
        assert logger.name == "test.slo"

    def test_creates_with_default_name(self):
        log = CLILogger()
        assert log.name == "slo.cli"

    def test_creates_with_level(self):
        log = CLILogger("x", level=LogLevel.WARNING)
        assert log.level == LogLevel.WARNING

    def test_creates_with_context(self):
        log = CLILogger("x", context={"env": "test"})
        assert log.context == {"env": "test"}

    def test_creates_with_stream(self):
        s = io.StringIO()
        log = CLILogger("x", stream=s)
        assert log._stream is s

    def test_creates_with_colors_explicit(self):
        log = CLILogger("x", colors=True)
        assert log._colors is True

    def test_creates_with_colors_default(self):
        log = CLILogger("x")
        assert isinstance(log._colors, bool)

    def test_repr(self):
        log = CLILogger("x")
        assert "CLILogger" in repr(log)


# ── Color detection ─────────────────────────────────────────────────────

class TestColorDetection:
    def test_no_color_env_disables(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert _color_enabled() is False

    def test_force_color_env_enables(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("FORCE_COLOR", "1")
        assert _color_enabled() is True

    def test_slo_log_color_true(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        monkeypatch.setenv("SLO_LOG_COLOR", "true")
        assert _color_enabled() is True

    def test_slo_log_color_false(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        monkeypatch.setenv("SLO_LOG_COLOR", "0")
        assert _color_enabled() is False


# ── Emit ────────────────────────────────────────────────────────────────

class TestEmit:
    def test_emit_writes_line(self, logger, buf):
        logger.emit(_record(message="booted"))
        out = buf.getvalue()
        assert "booted" in out
        assert out.endswith("\n")

    def test_emit_respects_level(self, buf):
        log = CLILogger("x", stream=buf, level=LogLevel.WARNING, colors=False)
        log.info("ignored")
        log.warning("kept")
        out = buf.getvalue()
        assert "ignored" not in out
        assert "kept" in out

    def test_emit_includes_exception(self, logger, buf):
        logger.emit(_record(message="fail", exception="RuntimeError: boom"))
        out = buf.getvalue()
        assert "fail" in out
        assert "RuntimeError: boom" in out

    def test_emit_includes_context(self, logger, buf):
        logger.emit(_record(message="msg", context={"k": "v"}))
        out = buf.getvalue()
        assert "k=v" in out

    def test_emit_includes_level_label(self, logger, buf):
        logger.emit(_record(message="test", level=LogLevel.ERROR))
        out = buf.getvalue()
        assert "[error]" in out

    def test_emit_includes_logger_name(self, logger, buf):
        logger.emit(_record(message="test", logger="slo.api"))
        out = buf.getvalue()
        assert "slo.api" in out

    def test_emit_no_colors_no_ansi(self, buf):
        log = CLILogger("x", stream=buf, colors=False)
        log.emit(_record(message="plain"))
        out = buf.getvalue()
        assert "\033[" not in out

    def test_emit_respects_terminal_disabled(self, buf):
        log = CLILogger("x", stream=buf, colors=False)
        set_cli_terminal(False)
        log.emit(_record(message="suppressed"))
        assert buf.getvalue() == ""

    def test_emit_swallows_stream_error(self):
        class Broken:
            def write(self, _):
                raise OSError("closed")
            def flush(self):
                raise OSError("closed")

        log = CLILogger("x", stream=Broken(), colors=False)
        log.emit(_record(message="won't raise"))


# ── Success ─────────────────────────────────────────────────────────────

class TestSuccess:
    def test_success_prints(self, logger, buf):
        logger.success("done", step="1")
        out = buf.getvalue()
        assert "✓" in out
        assert "done" in out
        assert "step=1" in out

    def test_success_without_context(self, logger, buf):
        logger.success("done")
        out = buf.getvalue()
        assert "✓" in out
        assert "done" in out

    def test_success_respects_terminal_disabled(self, buf):
        log = CLILogger("x", stream=buf, colors=False)
        set_cli_terminal(False)
        log.success("hidden")
        assert buf.getvalue() == ""


# ── Step ────────────────────────────────────────────────────────────────

class TestStep:
    def test_step_prints(self, logger, buf):
        logger.step("processing", file="data.csv")
        out = buf.getvalue()
        assert "→" in out
        assert "processing" in out
        assert "file=data.csv" in out


# ── Header ──────────────────────────────────────────────────────────────

class TestHeader:
    def test_header_prints_two_lines(self, logger, buf):
        logger.header("Title")
        lines = buf.getvalue().splitlines()
        assert len(lines) == 2
        assert "Title" in lines[0]

    def test_header_custom_char(self, logger, buf):
        logger.header("Title", char="-")
        lines = buf.getvalue().splitlines()
        assert "-" in lines[1]


# ── Section ─────────────────────────────────────────────────────────────

class TestSection:
    def test_section_prints_three_lines(self, logger, buf):
        logger.section("Section")
        lines = buf.getvalue().splitlines()
        assert len(lines) == 3
        assert "Section" in lines[1]


# ── Table ───────────────────────────────────────────────────────────────

class TestTable:
    def test_table_with_rows(self, logger, buf):
        logger.table(["Name", "Val"], [["a", "1"], ["b", "2"]])
        out = buf.getvalue()
        assert "Name" in out
        assert "Val" in out
        assert "a" in out
        assert "2" in out

    def test_table_empty_returns(self, logger, buf):
        logger.table(["Name"], [])
        assert buf.getvalue() == ""

    def test_table_with_align(self, logger, buf):
        logger.table(["A"], [["x"]], align=["r"])
        out = buf.getvalue()
        assert "x" in out

    def test_table_header_bold_in_ansi(self, buf):
        log = CLILogger("x", stream=buf, colors=True)
        log.table(["Col"], [["val"]])
        out = buf.getvalue()
        assert "\033[1m" in out  # BOLD for header


# ── JSON ────────────────────────────────────────────────────────────────

class TestJson:
    def test_json_prints(self, logger, buf):
        logger.json({"key": "value"})
        out = buf.getvalue()
        assert '"key"' in out
        assert '"value"' in out

    def test_json_indented(self, logger, buf):
        logger.json({"a": 1}, indent=2)
        out = buf.getvalue()
        assert "  " in out  # indented


# ── Status ──────────────────────────────────────────────────────────────

class TestStatus:
    def test_status_ok(self, logger, buf):
        logger.status("Health", "OK", status="ok")
        out = buf.getvalue()
        assert "✓" in out
        assert "Health: OK" in out

    def test_status_warn(self, logger, buf):
        logger.status("Load", "High", status="warn")
        out = buf.getvalue()
        assert "!" in out

    def test_status_error(self, logger, buf):
        logger.status("DB", "Down", status="error")
        out = buf.getvalue()
        assert "✗" in out

    def test_status_info(self, logger, buf):
        logger.status("Version", "1.0", status="info")
        out = buf.getvalue()
        assert "ℹ" in out


# ── Divider ─────────────────────────────────────────────────────────────

class TestDivider:
    def test_divider_prints(self, logger, buf):
        logger.divider()
        out = buf.getvalue()
        assert "-" in out
        assert out.endswith("\n")

    def test_divider_custom_char(self, logger, buf):
        logger.divider("=")
        out = buf.getvalue()
        assert "=" in out


# ── KeyValue ────────────────────────────────────────────────────────────

class TestKeyValue:
    def test_key_value_prints(self, logger, buf):
        logger.key_value("API", "http://localhost:8000")
        out = buf.getvalue()
        assert "API:" in out
        assert "http://localhost:8000" in out

    def test_key_value_empty_key_prints_value_only(self, logger, buf):
        logger.key_value("", "Press Ctrl+C to stop")
        out = buf.getvalue()
        assert "Press Ctrl+C to stop" in out
        assert ": " not in out

    def test_key_value_custom_indent(self, logger, buf):
        logger.key_value("K", "V", indent=4)
        out = buf.getvalue()
        assert "    K:" in out


# ── Blank ───────────────────────────────────────────────────────────────

class TestBlank:
    def test_blank_prints_empty_line(self, logger, buf):
        logger.blank()
        assert buf.getvalue() == "\n"

    def test_blank_multiple(self, logger, buf):
        logger.blank(count=3)
        assert buf.getvalue() == "\n\n\n"


# ── Command ─────────────────────────────────────────────────────────────

class TestCommand:
    def test_command_prints(self, logger, buf):
        logger.command("sloughgpt dev", "Start dev servers")
        out = buf.getvalue()
        assert "sloughgpt dev" in out
        assert "Start dev servers" in out

    def test_command_without_description(self, logger, buf):
        logger.command("sloughgpt shell")
        out = buf.getvalue()
        assert "sloughgpt shell" in out


# ── Timer ───────────────────────────────────────────────────────────────

class TestTimer:
    def test_timer_logs_elapsed(self, logger, buf):
        import time
        with logger.timer("test op"):
            time.sleep(0.01)
        out = buf.getvalue()
        assert "test op" in out
        assert "ms" in out

    def test_timer_logs_on_exception(self, logger, buf):
        import time
        try:
            with logger.timer("fail op"):
                time.sleep(0.01)
                raise ValueError("boom")
        except ValueError:
            pass
        out = buf.getvalue()
        assert "fail op" in out
        assert "ms" in out


# ── SetCliTerminal ─────────────────────────────────────────────────────

class TestSetCliTerminal:
    def test_disable_and_reenable(self):
        import domains.logging.cli_logger as mod
        set_cli_terminal(False)
        assert mod._TERMINAL_ENABLED is False
        set_cli_terminal(True)
        assert mod._TERMINAL_ENABLED is True
