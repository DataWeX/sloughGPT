"""Tests for shell.repl — ShellREPL execute() and helper methods."""

from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch

from domains.shell.repl import ShellREPL, _color, _COLOR_ENABLED


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_repl():
    """Create a ShellREPL with mocked dependencies."""
    mock_os = MagicMock()
    mock_cmds = MagicMock()
    repl = ShellREPL.__new__(ShellREPL)
    repl.os = mock_os
    repl.cmds = mock_cmds
    repl.state = MagicMock()
    repl.state.history = []
    repl.state.aliases = {}
    repl.state.env = {}
    repl._history = []
    repl._running = False
    repl._use_tui = False
    repl._bg_threads = {}
    repl._next_bg_id = 1
    repl._piped_input = ""
    repl._aborted = False
    repl._env = {"PS1": "λ", "SHELL": "sloughgpt", "HOME": "/tmp", "TERM": "xterm-256color"}
    repl._last_exit_code = 0
    repl._cmd_count = 0
    repl._dir_stack = []
    repl._chat_session_id = None
    repl._chat_history = []
    repl._completion_cache_obj = None
    repl._completion_cache = {}
    repl._aliases = {"q": "exit", "quit": "exit", "h": "help", "?": "help"}
    repl._ext_cmds = {}

    from domains.shell.io import MemoryIO
    repl.io = MemoryIO()

    from domains.shell.console import Console
    repl.console = Console(repl.io, has_readline=False)

    from domains.logging import ShellLogger, LogLevel
    repl.log = ShellLogger("slo.shell.test", level=LogLevel.DEBUG)

    from domains.shell.log_buffer import get_log_buffer
    repl._log_buffer = get_log_buffer()

    from domains.shell.log_display import LineModeLogDisplay
    repl._log_display = LineModeLogDisplay(repl._log_buffer)

    from domains.shell.audit import get_shell_audit_logger
    repl._audit = get_shell_audit_logger()

    from domains.shell.permissions import ShellPermissions
    repl._perms = ShellPermissions()

    repl.COMMANDS = {}
    repl._print = lambda *a, **kw: None
    return repl


# ── execute() ───────────────────────────────────────────────────────────────


class TestExecute:

    def test_empty_line(self):
        repl = _make_repl()
        output, code = repl.execute("")
        assert output == ""
        assert code == 0

    def test_whitespace_line(self):
        repl = _make_repl()
        output, code = repl.execute("   ")
        assert output == ""
        assert code == 0

    def test_unknown_command(self):
        repl = _make_repl()
        output, code = repl.execute("notacommand")
        assert code == 127
        assert "Unknown command" in output

    def test_execute_returns_tuple(self):
        repl = _make_repl()
        result = repl.execute("echo hello")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_cmd_count_increments(self):
        repl = _make_repl()
        initial = repl._cmd_count
        repl.execute("notacommand")
        assert repl._cmd_count == initial + 1

    def test_history_updated(self):
        repl = _make_repl()
        repl.execute("notacommand")
        assert "notacommand" in repl._history


# ── _expand_alias ───────────────────────────────────────────────────────────


class TestExpandAlias:

    def test_expand_alias(self):
        repl = _make_repl()
        repl._aliases = {"ll": "ls -la"}
        result = repl._expand_alias("ll foo")
        assert result == "ls -la foo"

    def test_no_alias(self):
        repl = _make_repl()
        repl._aliases = {}
        result = repl._expand_alias("ls foo")
        assert result == "ls foo"


# ── _suggest_command ────────────────────────────────────────────────────────


class TestSuggestCommand:

    def test_suggest_similar(self):
        repl = _make_repl()
        repl.COMMANDS = {"help": None, "exit": None, "load": None}
        suggestion = repl._suggest_command("hep")
        assert suggestion == "help"

    def test_no_suggestion(self):
        repl = _make_repl()
        repl.COMMANDS = {"help": None}
        suggestion = repl._suggest_command("xyz")
        assert suggestion is None


# ── _render_prompt ──────────────────────────────────────────────────────────


class TestRenderPrompt:

    def test_default_prompt(self):
        repl = _make_repl()
        prompt = repl._render_prompt()
        assert "λ" in prompt

    def test_error_code_prefix(self):
        repl = _make_repl()
        repl._last_exit_code = 1
        prompt = repl._render_prompt()
        assert "[1]" in prompt


# ── _format_error ───────────────────────────────────────────────────────────


class TestFormatError:

    def test_format_error(self):
        repl = _make_repl()
        err = ValueError("test error")
        msg = repl._format_error(err, "testcmd")
        assert "test error" in msg or "ValueError" in msg


# ── _update_color_state ─────────────────────────────────────────────────────


class TestUpdateColorState:

    def test_no_color_true(self):
        repl = _make_repl()
        repl._env["NO_COLOR"] = "1"
        repl._update_color_state()
        import domains.shell.repl as mod
        # After update, colors should be empty
        assert mod._C_CYAN == "" or mod._COLOR_ENABLED is False

    def test_no_color_false(self):
        repl = _make_repl()
        repl._env.pop("NO_COLOR", None)
        repl._update_color_state()


# ── _complete ───────────────────────────────────────────────────────────────


class TestComplete:

    def test_complete_first_word(self):
        repl = _make_repl()
        repl.COMMANDS = {"help": None, "exit": None}
        result = repl._complete("he", 0)
        assert result == "help"

    def test_complete_no_match(self):
        repl = _make_repl()
        repl.COMMANDS = {"help": None}
        result = repl._complete("xyz", 0)
        assert result is None


# ── _complete_args_for ──────────────────────────────────────────────────────


class TestCompleteArgsFor:

    def test_train_subcommands(self):
        repl = _make_repl()
        result = repl._complete_args_for_uncached("train")
        assert "status" in result
        assert "stop" in result

    def test_finetuned_subcommands(self):
        repl = _make_repl()
        result = repl._complete_args_for_uncached("finetuned")
        assert "load" in result
        assert "rm" in result
