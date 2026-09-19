"""Tests for shell.repl — ShellREPL execute() and helper methods."""

from __future__ import annotations

from unittest.mock import MagicMock

from domain.shell._internal.repl import ShellREPL

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

    from domain.shell._internal.io import MemoryIO

    repl.io = MemoryIO()

    from domain.shell._internal.console import Console

    repl.console = Console(repl.io, has_readline=False)

    from domain.logging import LogLevel, ShellLogger

    repl.log = ShellLogger("slo.shell.test", level=LogLevel.DEBUG)

    from domain.shell._internal.log_buffer import get_log_buffer

    repl._log_buffer = get_log_buffer()

    from domain.shell._internal.log_display import LineModeLogDisplay

    repl._log_display = LineModeLogDisplay(repl._log_buffer)

    from domain.shell._internal.audit import get_shell_audit_logger

    repl._audit = get_shell_audit_logger()

    from domain.shell._internal.permissions import ShellPermissions

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
        import domain.shell._internal.repl as mod

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


# ── traceback command ───────────────────────────────────────────────────────


class TestTracebackCommand:
    def _boom_repl(self):
        repl = _make_repl()

        def _boom(self, args=""):
            raise RuntimeError("kaboom-test")

        repl.COMMANDS["boom"] = _boom
        repl.COMMANDS["traceback"] = ShellREPL._cmd_traceback
        printed: list[str] = []
        repl._print = lambda *a, **kw: printed.append(" ".join(str(x) for x in a))  # noqa: E731
        return repl, printed

    def test_error_stashes_traceback(self):
        repl, _ = self._boom_repl()
        repl.execute("boom")
        assert repl._last_exit_code == 1
        assert repl._last_traceback is not None
        assert "RuntimeError" in repl._last_traceback
        assert "kaboom-test" in repl._last_traceback

    def test_traceback_command_prints_last_error(self):
        repl, printed = self._boom_repl()
        repl.execute("boom")
        out, code = repl.execute("traceback")
        assert code == 0
        assert "RuntimeError" in out

    def test_traceback_without_error(self):
        repl, printed = self._boom_repl()
        out, code = repl.execute("traceback")
        assert code == 0
        assert "No recent traceback" in out


# ── man ─────────────────────────────────────────────────────────────────


class TestManCommand:
    def _man_repl(self):
        repl = _make_repl()
        repl.COMMANDS["man"] = ShellREPL._cmd_man
        repl.COMMANDS["echo"] = ShellREPL._cmd_echo
        repl.COMMANDS["exit"] = ShellREPL._cmd_exit
        return repl

    def test_man_no_args_shows_usage(self):
        repl = self._man_repl()
        out, code = repl.execute("man")
        assert code == 0
        assert "Usage: man <command>" in out

    def test_man_renders_sections(self):
        repl = self._man_repl()
        out, code = repl.execute("man echo")
        assert code == 0
        assert "ECHO(1)" in out
        assert "NAME" in out
        assert "SYNOPSIS" in out
        assert "DESCRIPTION" in out
        assert "EXIT STATUS" in out

    def test_man_resolves_alias(self):
        repl = self._man_repl()
        out, code = repl.execute("man q")
        assert code == 0
        assert "EXIT(1)" in out

    def test_man_unknown_suggests_and_fails(self):
        repl = self._man_repl()
        out, code = repl.execute("man ech")
        assert code == 1
        assert "No manual entry for ech" in out


# ── _dump_json ──────────────────────────────────────────────────────────


class TestDumpJson:
    def test_plain_json_without_color(self, monkeypatch):
        repl = _make_repl()
        monkeypatch.setenv("NO_COLOR", "1")
        import json as _json

        obj = {"name": "x", "n": 2}
        assert repl._dump_json(obj) == _json.dumps(obj, indent=2, default=str)

    def test_valid_json_always(self, monkeypatch):
        # Highlighting must never corrupt the payload shape for parsers:
        # strip ANSI and the result must still parse.
        import json as _json
        import re as _re

        repl = _make_repl()
        monkeypatch.delenv("NO_COLOR", raising=False)
        obj = {"name": "x", "items": [1, 2], "ok": True}
        out = repl._dump_json(obj)
        stripped = _re.sub(r"\x1b\[[0-9;]*m", "", out)
        assert _json.loads(stripped) == obj
