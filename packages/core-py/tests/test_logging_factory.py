"""Tests for domains.logging.__init__ — factory functions and global logger."""

import pytest
from domains.logging import (
    get_logger, set_global, get_global,
    ConsoleLogger, CLILogger, ShellLogger, WebLogger,
    LogLevel,
)


class TestGetLogger:
    def test_api_returns_console(self):
        log = get_logger("api")
        assert isinstance(log, ConsoleLogger)

    def test_server_returns_console(self):
        log = get_logger("server")
        assert isinstance(log, ConsoleLogger)

    def test_console_returns_console(self):
        log = get_logger("console")
        assert isinstance(log, ConsoleLogger)

    def test_cli_returns_cli(self):
        log = get_logger("cli")
        assert isinstance(log, CLILogger)

    def test_shell_returns_shell(self):
        log = get_logger("shell")
        assert isinstance(log, ShellLogger)

    def test_repl_returns_shell(self):
        log = get_logger("repl")
        assert isinstance(log, ShellLogger)

    def test_web_returns_web(self):
        log = get_logger("web")
        assert isinstance(log, WebLogger)

    def test_browser_returns_web(self):
        log = get_logger("browser")
        assert isinstance(log, WebLogger)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown logger interface"):
            get_logger("nonexistent")

    def test_custom_name(self):
        log = get_logger("api", name="slo.custom")
        assert log.name == "slo.custom"

    def test_custom_level(self):
        log = get_logger("api", level=LogLevel.DEBUG)
        assert log.level == LogLevel.DEBUG


class TestGlobalLogger:
    def test_set_and_get(self):
        log = ConsoleLogger("slo.test_global")
        set_global(log)
        assert get_global() is log

    def test_get_creates_default(self):
        set_global(None)
        # Reset to force creation
        import domains.logging as pkg
        pkg._global_logger = None
        log = get_global()
        assert isinstance(log, ConsoleLogger)
        # Cleanup
        pkg._global_logger = None
