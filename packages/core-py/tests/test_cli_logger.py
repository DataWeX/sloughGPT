"""Tests for CLILogger — Rich-based CLI logger."""

import pytest
from unittest.mock import MagicMock, patch
from domains.logging.base import LogLevel, LogRecord
from domains.logging.cli_logger import (
    CLILogger,
    set_cli_terminal,
    _TERMINAL_ENABLED,
    _ensure_rich,
)


@pytest.fixture
def logger():
    return CLILogger("test.slo", level=LogLevel.DEBUG)


@pytest.fixture(autouse=True)
def reset_rich():
    """Reset global Rich state between tests."""
    import domains.logging.cli_logger as mod
    mod._console = None
    mod._Table = None
    mod._Panel = None
    mod._Syntax = None
    mod._box = None
    mod._TERMINAL_ENABLED = True
    yield
    mod._TERMINAL_ENABLED = True


class TestCLILoggerInit:
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

    def test_ensure_rich_called(self, logger):
        import domains.logging.cli_logger as mod
        assert mod._console is not None

    def test_ensure_rich_idempotent(self):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        first = mod._console
        _ensure_rich()
        assert mod._console is first


class TestCLILoggerEmit:
    def test_emit_calls_console_print(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        record = LogRecord(level=LogLevel.INFO, message="hello")
        logger.emit(record)
        mod._console.print.assert_called_once()

    def test_emit_respects_terminal_disabled(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        set_cli_terminal(False)
        record = LogRecord(level=LogLevel.INFO, message="hello")
        logger.emit(record)
        mod._console.print.assert_not_called()

    def test_emit_includes_exception(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        record = LogRecord(level=LogLevel.ERROR, message="fail", exception="RuntimeError: boom")
        logger.emit(record)
        mod._console.print.assert_called_once()

    def test_emit_includes_context(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        record = LogRecord(level=LogLevel.INFO, message="msg", context={"k": "v"})
        logger.emit(record)
        mod._console.print.assert_called_once()


class TestCLILoggerSuccess:
    def test_success_prints(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        logger.success("done", step="1")
        mod._console.print.assert_called_once()

    def test_success_without_context(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        logger.success("done")
        mod._console.print.assert_called_once()


class TestCLILoggerStep:
    def test_step_prints(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        logger.step("processing", file="data.csv")
        mod._console.print.assert_called_once()


class TestCLILoggerHeader:
    def test_header_prints_two_lines(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        mod._console.width = 40
        logger.header("Title")
        assert mod._console.print.call_count == 2


class TestCLILoggerSection:
    def test_section_prints_three_lines(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        mod._console.width = 40
        logger.section("Section")
        assert mod._console.print.call_count == 3


class TestCLILoggerTable:
    def test_table_with_rows(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        logger.table(["Name", "Val"], [["a", "1"], ["b", "2"]])
        mod._console.print.assert_called_once()

    def test_table_empty_returns(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        logger.table(["Name"], [])
        mod._console.print.assert_not_called()

    def test_table_with_align(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        logger.table(["A"], [["x"]], align=["r"])
        mod._console.print.assert_called_once()


class TestCLILoggerJson:
    def test_json_prints(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        logger.json({"key": "value"})
        mod._console.print.assert_called_once()


class TestCLILoggerStatus:
    def test_status_ok(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        logger.status("Health", "OK", status="ok")
        mod._console.print.assert_called_once()

    def test_status_warn(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        logger.status("Load", "High", status="warn")
        mod._console.print.assert_called_once()

    def test_status_error(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        logger.status("DB", "Down", status="error")
        mod._console.print.assert_called_once()

    def test_status_info(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        logger.status("Version", "1.0", status="info")
        mod._console.print.assert_called_once()


class TestCLILoggerDivider:
    def test_divider_prints(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        mod._console.width = 40
        logger.divider()
        mod._console.print.assert_called_once()

    def test_divider_custom_char(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        mod._console.width = 20
        logger.divider("=")
        mod._console.print.assert_called_once()


class TestCLILoggerKeyValue:
    def test_key_value_prints(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        logger.key_value("API", "http://localhost:8000")
        mod._console.print.assert_called_once()

    def test_key_value_empty_key_prints_value_only(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        logger.key_value("", "Press Ctrl+C to stop")
        mod._console.print.assert_called_once()
        args = mod._console.print.call_args[0][0]
        assert "Press Ctrl+C to stop" in args
        assert ": " not in args

    def test_key_value_custom_indent(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        logger.key_value("K", "V", indent=4)
        mod._console.print.assert_called_once()


class TestCLILoggerBlank:
    def test_blank_prints_empty_line(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        logger.blank()
        mod._console.print.assert_called_once_with("")

    def test_blank_multiple(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        logger.blank(count=3)
        assert mod._console.print.call_count == 3


class TestCLILoggerCommand:
    def test_command_prints(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        logger.command("sloughgpt dev", "Start dev servers")
        mod._console.print.assert_called_once()

    def test_command_without_description(self, logger):
        import domains.logging.cli_logger as mod
        mod._console = MagicMock()
        logger.command("sloughgpt shell")
        mod._console.print.assert_called_once()


class TestSetCliTerminal:
    def test_disable_and_reenable(self):
        import domains.logging.cli_logger as mod
        set_cli_terminal(False)
        assert mod._TERMINAL_ENABLED is False
        set_cli_terminal(True)
        assert mod._TERMINAL_ENABLED is True
