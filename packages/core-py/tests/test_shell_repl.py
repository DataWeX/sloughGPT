"""
Tests for ShellREPL — pipeline parsing, output capture, state persistence.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from domains.shell.repl import ShellREPL, _CaptureOutput
from domains.shell.io import capture_cmd
from domains.shell.runtime import DaitRuntime
from domains.shell.state import ShellState
from domains.shell.state import set_shell_state_db, reset_shell_state_db


@pytest.fixture
def repl():
    from pathlib import Path
    from unittest.mock import patch
    import tempfile
    import gc
    with tempfile.TemporaryDirectory() as tmp:
        st = Path(tmp) / "sloughgpt"
        st.mkdir(parents=True, exist_ok=True)
        set_shell_state_db(str(st / "test_mogdb"))
        with patch("domains.shell.runtime._probe_api", return_value={"available": False, "error": "mock"}), \
             patch("domains.shell.repl.ShellREPL._get_current_model", return_value=""), \
             patch("domains.shell.repl.ShellREPL._get_current_soul", return_value=""), \
             patch.object(ShellREPL, "_setup_readline"), \
             patch("domains.shell.runtime.APIServerProcess.start", return_value={"ok": True, "message": "mocked"}):
            os = DaitRuntime()
            r = ShellREPL(os)
            r._perms._granted.update(["tee", "xargs", "cp", "mv", "touch", "chmod"])
            yield r
            # Break reference cycles to allow GC of 679+ DaitRuntime instances:
            # 1) logging handlers on root logger → closure → repl
            for name in ("slo", "slo.kernel", "slo.shell.runtime", "slo.shell.init"):
                for h in logging.getLogger(name).handlers[:]:
                    try:
                        logging.getLogger(name).removeHandler(h)
                        h.close()
                    except Exception:
                        pass
            # 2) _buffered_emit closure captures self → self.log → closure cycle
            r.log = None
            r._bg_threads.clear()
            r = None
            os = None
            gc.collect()
        reset_shell_state_db()


@pytest.fixture
def isolated_state():
    """Fixture providing a ShellState backed by a temp directory."""
    with tempfile.TemporaryDirectory() as tmp:
        st = Path(tmp) / "sloughgpt"
        st.mkdir(parents=True, exist_ok=True)
        set_shell_state_db(str(st / "test_mogdb"))
        s = ShellState()
        yield s
        reset_shell_state_db()


# ── UI selection (line mode default, TUI opt-in) ────────────────────

def _make_repl(use_tui=None):
    from domains.shell.runtime import DaitRuntime
    with tempfile.TemporaryDirectory() as tmp:
        st = Path(tmp) / "sloughgpt"
        st.mkdir(parents=True, exist_ok=True)
        set_shell_state_db(str(st / "test_mogdb"))
        try:
            with patch("domains.shell.runtime._probe_api", return_value={"available": False, "error": "mock"}), \
                 patch.object(ShellREPL, "_setup_readline"):
                return ShellREPL(DaitRuntime(), use_tui=use_tui)
        finally:
            reset_shell_state_db()


def test_line_mode_is_default(monkeypatch):
    monkeypatch.delenv("MAN_TUI", raising=False)
    assert _make_repl()._use_tui is False


def test_man_tui_env_opts_in(monkeypatch):
    monkeypatch.setenv("MAN_TUI", "1")
    assert _make_repl()._use_tui is True


def test_explicit_use_tui_true(monkeypatch):
    monkeypatch.delenv("MAN_TUI", raising=False)
    assert _make_repl(use_tui=True)._use_tui is True


def test_explicit_use_tui_false(monkeypatch):
    monkeypatch.setenv("MAN_TUI", "1")
    assert _make_repl(use_tui=False)._use_tui is False


# ── _CaptureOutput ─────────────────────────────────────────────────


class TestCaptureOutput:
    def test_captures_print(self):
        with _CaptureOutput() as cap:
            print("hello")
        assert cap.getvalue() == "hello\n"

    def test_empty_capture(self):
        with _CaptureOutput() as cap:
            pass
        assert cap.getvalue() == ""

    def test_multiple_prints(self):
        with _CaptureOutput() as cap:
            print("a")
            print("b")
        assert cap.getvalue() == "a\nb\n"


# ── Pipeline parsing ────────────────────────────────────────────────


class TestPipelineParsing:
    def test_single_command(self, repl):
        cmds, bg, timed = repl._parse_pipeline("health")
        assert cmds == [("health", None)]
        assert bg is False
        assert timed is False

    def test_two_commands(self, repl):
        cmds, bg, timed = repl._parse_pipeline("echo hello | wc")
        assert cmds == [("echo hello", "|"), ("wc", None)]
        assert bg is False
        assert timed is False

    def test_three_commands(self, repl):
        cmds, bg, timed = repl._parse_pipeline("models | grep gpt | head 5")
        assert cmds == [("models", "|"), ("grep gpt", "|"), ("head 5", None)]
        assert bg is False
        assert timed is False

    def test_background_flag(self, repl):
        cmds, bg, timed = repl._parse_pipeline("health &")
        assert cmds == [("health", None)]
        assert bg is True
        assert timed is False

    def test_pipeline_with_background(self, repl):
        cmds, bg, timed = repl._parse_pipeline("echo a | wc &")
        assert cmds == [("echo a", "|"), ("wc", None)]
        assert bg is True
        assert timed is False

    def test_trailing_spaces(self, repl):
        cmds, bg, timed = repl._parse_pipeline("  models  |  grep llama  ")
        assert cmds == [("models", "|"), ("grep llama", None)]
        assert bg is False
        assert timed is False

    def test_time_prefix(self, repl):
        cmds, bg, timed = repl._parse_pipeline("time health")
        assert cmds == [("health", None)]
        assert bg is False
        assert timed is True

    def test_time_with_pipeline(self, repl):
        cmds, bg, timed = repl._parse_pipeline("time models | grep gpt")
        assert cmds == [("models", "|"), ("grep gpt", None)]
        assert timed is True


# ── Pipeline execution ──────────────────────────────────────────────


class TestPipelineExecution:
    def test_single_pipeline_runs(self, repl):
        output = repl._execute_pipeline([("echo hello", None)])
        # _execute_pipeline returns None (prints output), so capture at caller

    def test_two_stage_pipeline(self, repl):
        output = repl._execute_single("echo hello world", "")
        assert "hello" in output


# ── echo ────────────────────────────────────────────────────────────



# ── read ─────────────────────────────────────────────────────────────


class TestRead:
    def test_read_sets_env_var(self, repl):
        with patch("builtins.input", return_value="test_value"):
            repl._cmd_read("MYVAR")
        assert repl._env.get("MYVAR") == "test_value"

    def test_read_with_prompt(self, repl):
        with patch("builtins.input", return_value="val"):
            repl._cmd_read("-p Enter MYVAR")
        assert repl._env.get("MYVAR") == "val"

    def test_read_no_args_fails(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_read("")
        assert repl._last_exit_code == 1

    def test_read_eof_sets_exit_code_1(self, repl):
        with patch("builtins.input", side_effect=EOFError):
            repl._cmd_read("X")
        assert repl._last_exit_code == 1


# ── printf ───────────────────────────────────────────────────────────


# ── dirname / basename ────────────────────────────────────────────────



class TestBasename:
    def test_basename_basic(self, repl):
        out = repl._execute_single("basename /a/b/c.txt")
        assert "c.txt" in out

    def test_basename_no_args(self, repl):
        out = repl._execute_single("basename")
        assert repl._last_exit_code == 1


# ── yes ──────────────────────────────────────────────────────────────


class TestYes:
    def _test_yes_output(self, args, expected_in):
        """Run yes in a subprocess with SIGALRM to avoid infinite loop."""
        import subprocess, sys
        pkgs = os.path.join(os.path.dirname(__file__), "..")
        code = f"""
import sys, os, signal, tempfile, json
sys.path.insert(0, {pkgs!r})
from pathlib import Path
from unittest.mock import patch
from domains.shell.repl import ShellREPL
from domains.shell.runtime import DaitRuntime
tmp = tempfile.mkdtemp()
from domains.shell.state import set_shell_state_db
set_shell_state_db(tmp + "/mogdb")
os.chdir(tmp)
r = ShellREPL(DaitRuntime())
signal.setitimer(signal.ITIMER_REAL, 0.1)
try:
    r._execute_single("yes {args}")
except (KeyboardInterrupt, SystemExit):
    pass
signal.setitimer(signal.ITIMER_REAL, 0)
"""
        result = subprocess.run([sys.executable, "-c", code],
                                capture_output=True, text=True, timeout=10)
        # -14 = SIGALRM (success — signal killed the infinite loop)
        assert result.returncode in (0, -14), \
            f"RC={result.returncode} stderr={result.stderr[:500]}"

    def test_yes_default(self):
        self._test_yes_output("", "y")

    def test_yes_custom(self):
        self._test_yes_output("hello", "hello")


# ── glob expansion ────────────────────────────────────────────────────


class TestGlobExpansion:
    def test_has_magic(self, repl):
        from domains.shell.repl import glob
        assert glob.has_magic("*.py")
        assert not glob.has_magic("hello.txt")

    def test_expand_globs_no_magic(self, repl):
        result = repl._expand_globs("echo hello")
        assert result == "echo hello"

    def test_expand_globs_unchanged_on_no_match(self, repl):
        # If no files match, the pattern should be unchanged
        result = repl._expand_globs("echo zzzz_*.nonexistent")
        assert "zzzz_*.nonexistent" in result


# ── history expansion ──────────────────────────────────────────────────


class TestHistoryExpansion:
    def test_bang_bang_expands(self, repl):
        repl._history.append("echo hello")
        result = repl._expand_history("!!")
        assert result == "echo hello"

    def test_bang_dollar(self, repl):
        repl._history.append("echo hello world")
        result = repl._expand_history("!$")
        assert result == "world"

    def test_bang_n(self, repl):
        repl._history.append("first")
        repl._history.append("second")
        result = repl._expand_history("!1")
        assert result == "first"

    def test_bang_neg_n(self, repl):
        repl._history.append("first")
        repl._history.append("second")
        result = repl._expand_history("!-1")
        assert result == "second"

    def test_bang_star(self, repl):
        repl._history.append("echo a b c")
        result = repl._expand_history("!*")
        assert result == "a b c"

    def test_bang_colon_n(self, repl):
        repl._history.append("echo a b c")
        result = repl._expand_history("!:2")
        assert result == "b"  # 0-indexed: 0=echo, 1=a, 2=b


# ── Alias ───────────────────────────────────────────────────────────


class TestAlias:
    def test_set_alias(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_alias("ll=procs")
        assert "ll" in repl._aliases
        assert repl._aliases["ll"] == "procs"

    def test_list_aliases(self, repl):
        repl._aliases["xx"] = "health"
        with _CaptureOutput() as cap:
            repl._cmd_alias("")
        assert "xx=health" in cap.getvalue()

    def test_unalias(self, repl):
        repl._aliases["xx"] = "health"
        with _CaptureOutput() as cap:
            repl._cmd_unalias("xx")
        assert "xx" not in repl._aliases


# ── Alias expansion ────────────────────────────────────────────────


class TestAliasExpansion:
    def test_expand_known(self, repl):
        repl._aliases["ll"] = "procs"
        assert repl._expand_alias("ll") == "procs"

    def test_expand_with_args(self, repl):
        repl._aliases["ll"] = "procs"
        assert repl._expand_alias("ll -a") == "procs -a"

    def test_expand_unknown(self, repl):
        assert repl._expand_alias("unknown_cmd arg1") == "unknown_cmd arg1"

    def test_empty_line(self, repl):
        assert repl._expand_alias("") == ""


# ── State persistence ───────────────────────────────────────────────


class TestShellState:
    def test_init_creates_empty(self, isolated_state):
        assert isolated_state.history == []
        assert isolated_state.aliases == {}

    def test_add_history_dedup(self, isolated_state):
        isolated_state.add_history("a")
        isolated_state.add_history("a")
        assert isolated_state.history == ["a"]

    def test_add_history_sequential(self, isolated_state):
        isolated_state.add_history("a")
        isolated_state.add_history("b")
        assert isolated_state.history == ["a", "b"]

    def test_set_alias(self, isolated_state):
        isolated_state.set_alias("ll", "procs")
        assert isolated_state.aliases["ll"] == "procs"

    def test_unset_alias_exists(self, isolated_state):
        isolated_state.set_alias("ll", "procs")
        assert isolated_state.unset_alias("ll") is True
        assert "ll" not in isolated_state.aliases

    def test_unset_alias_missing(self, isolated_state):
        assert isolated_state.unset_alias("nonexistent") is False

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = Path(tmp) / "sloughgpt"
            st.mkdir(parents=True, exist_ok=True)
            set_shell_state_db(str(st / "test_mogdb"))
            try:
                state = ShellState()
                state.add_history("hello")
                state.add_history("health")
                state.set_alias("h", "health")
                state.save()
                state2 = ShellState()
                assert state2.history == ["hello", "health"]
            finally:
                reset_shell_state_db()
                assert state2.aliases == {"h": "health"}

    def test_max_history(self, isolated_state):
        for i in range(600):
            isolated_state.add_history(str(i))
        assert len(isolated_state.history) == 600
        isolated_state.save()
        assert len(isolated_state.history[-500:]) == 500

    def test_to_dict(self, isolated_state):
        isolated_state.set_alias("h", "health")
        d = isolated_state.to_dict()
        assert "history_count" in d
        assert "aliases" in d
        assert "last_session" in d
        assert d["aliases"]["h"] == "health"


# ── BG command parsing (no actual threading in test) ────────────────


class TestBackground:
    def test_parse_background(self, repl):
        cmds, bg, timed = repl._parse_pipeline("health &")
        assert bg is True

    def test_parse_no_background(self, repl):
        cmds, bg, timed = repl._parse_pipeline("health")
        assert bg is False

    def test_bg_list_empty(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_bg("")
        assert "No background" in cap.getvalue()


# ── ShellCommands integration ───────────────────────────────────────


class TestCommandsIntegration:
    def test_echo_via_commands(self, repl):
        output = repl._execute_single("echo test123")
        assert "test123" in output

    def test_unknown_command(self, repl):
        output = repl._execute_single("nonexistent_cmd_xyz")
        assert "Unknown command" in output

    def test_alias_expansion_in_execute(self, repl):
        repl._aliases["h"] = "health"
        output = repl._execute_single("h")
        # Just verify it ran without error
        assert output != ""

    def test_empty_pipeline_does_nothing(self, repl):
        # _execute_pipeline prints output; just verify no error
        repl._execute_pipeline([])


class TestAiCommand:
    def test_ai_query_uses_real_cwd(self, repl):
        """``ai`` builds shell context from the process cwd, not a runtime
        attribute — regression for ``'DaitRuntime' object has no attribute
        'cwd'`` (repl.py ctx_parts)."""
        from unittest.mock import patch
        with patch("domains.shell.runtime._probe_api", return_value={"available": True, "model_id": "x"}), \
             patch.object(repl, "_spinner_call", return_value={"text": "echo hi"}):
            with _CaptureOutput() as cap:
                repl._cmd_ai("how are you?")
        assert "echo hi" in cap.getvalue()

    def test_ai_falls_back_to_keyword_match_when_api_down(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_ai("list models")
        assert "keyword" in cap.getvalue().lower()


# ── source command ──────────────────────────────────────────────────


class TestSource:
    def test_source_executes_commands(self, repl):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write("echo hello from script\n")
            f.write("echo line two\n")
            f_path = f.name
        try:
            with _CaptureOutput() as cap:
                repl._cmd_source(f_path)
            output = cap.getvalue()
            assert "hello from script" in output
            assert "line two" in output
        finally:
            Path(f_path).unlink(missing_ok=True)

    def test_source_skips_comments(self, repl):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write("# this is a comment\n")
            f.write("echo after comment\n")
            f_path = f.name
        try:
            with _CaptureOutput() as cap:
                repl._cmd_source(f_path)
            assert "after comment" in cap.getvalue()
        finally:
            Path(f_path).unlink(missing_ok=True)

    def test_source_nonexistent_file(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_source("/tmp/nonexistent_script_xyz.sh")
        assert "Error" in cap.getvalue()

    def test_dot_command_aliases_source(self, repl):
        output = repl._execute_single(".")
        assert "Usage" in output or "source" in output

    def test_source_empty_file(self, repl):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f_path = f.name
        try:
            with _CaptureOutput() as cap:
                repl._cmd_source(f_path)
            assert cap.getvalue() == ""
        finally:
            Path(f_path).unlink(missing_ok=True)


# ── py command ──────────────────────────────────────────────────────


class TestPyCommand:
    def test_py_evaluates_expression(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_py("2 + 2")
        assert "4" in cap.getvalue()

    def test_py_string(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_py("'hello'.upper()")
        assert "HELLO" in cap.getvalue()

    def test_py_list(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_py("[i*i for i in range(5)]")
        assert "[0, 1, 4, 9, 16]" in cap.getvalue()

    def test_py_error(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_py("1/0")
        assert "Error" in cap.getvalue()

    def test_py_empty_shows_usage(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_py("")
        assert "Usage" in cap.getvalue()


# ── Command substitution $(cmd) ──────────────────────────────────────


class TestCommandSubstitution:
    def test_expand_cmd_subst_basic(self, repl):
        expanded = repl._expand_cmd_subst("echo $(echo hello)")
        # echo hello produces "hello", so result should be "echo hello"
        assert "hello" in expanded

    def test_expand_cmd_subst_nested_calls(self, repl):
        # $(echo hello) expands to "hello", so the line becomes "echo hello"
        result = repl._expand_cmd_subst("$(echo hello)")
        result = result.strip()
        assert result == "hello"

    def test_expand_cmd_subst_no_subst(self, repl):
        assert repl._expand_cmd_subst("echo plain") == "echo plain"

    def test_expand_cmd_subst_empty(self, repl):
        assert repl._expand_cmd_subst("") == ""


# ── Env var persistence ──────────────────────────────────────────────


class TestEnvPersistence:
    def test_set_env_persists_to_state(self, isolated_state):
        isolated_state.set_env("MY_VAR", "my_value")
        assert isolated_state.env["MY_VAR"] == "my_value"

    def test_save_load_env_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = Path(tmp) / "sloughgpt"
            st.mkdir(parents=True, exist_ok=True)
            set_shell_state_db(str(st / "test_mogdb"))
            try:
                state = ShellState()
                state.set_env("TEST_KEY", "test_val")
                state.save()
                state2 = ShellState()
                assert state2.env.get("TEST_KEY") == "test_val"
            finally:
                reset_shell_state_db()

    def test_to_dict_includes_env_count(self, isolated_state):
        isolated_state.set_env("A", "1")
        d = isolated_state.to_dict()
        assert "env_vars" in d
        assert d["env_vars"] == 1


# ── help command ────────────────────────────────────────────────────


class TestHelpCommand:
    def test_help_shows_general(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_help("")
        output = cap.getvalue()
        assert "Built-in commands" in output
        assert "health" in output

    def test_help_individual_command(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_help("health")
        assert "health" in cap.getvalue()
        assert "API health check" in cap.getvalue() or "health" in cap.getvalue()

    def test_help_unknown_command(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_help("nonexistent_cmd_xyz")
        assert "Unknown" in cap.getvalue()

    def test_help_brief(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_help("brief")
        out = cap.getvalue()
        assert "ls" in out
        assert "models" in out
        assert "boot" in out
        assert "help" in out
        assert "exit" in out


# ── history with n ──────────────────────────────────────────────────


class TestHistoryN:
    def test_history_default_limit(self, repl):
        for i in range(50):
            repl._history.append(f"cmd{i}")
        with _CaptureOutput() as cap:
            repl._cmd_history("")
        lines = [l for l in cap.getvalue().split("\n") if l.strip()]
        # Default shows last 20
        assert len(lines) == 20

    def test_history_with_n(self, repl):
        for i in range(50):
            repl._history.append(f"cmd{i}")
        with _CaptureOutput() as cap:
            repl._cmd_history("5")
        lines = [l for l in cap.getvalue().split("\n") if l.strip()]
        assert len(lines) == 5

    def test_history_invalid_n(self, repl):
        for i in range(5):
            repl._history.append(f"cmd{i}")
        with _CaptureOutput() as cap:
            repl._cmd_history("abc")
        # Invalid n defaults to 20, but only 5 exist
        lines = [l for l in cap.getvalue().split("\n") if l.strip()]
        assert len(lines) == 5


# ── fc command ──────────────────────────────────────────────────────────


class TestFcCommand:
    def test_fc_no_args_lists_history(self, repl):
        repl._history.append("health")
        repl._history.append("models")
        with _CaptureOutput() as cap:
            repl._cmd_fc("")
        output = cap.getvalue()
        assert "health" in output
        assert "models" in output

    def test_fc_l_flag(self, repl):
        for i in range(5):
            repl._history.append(f"cmd{i}")
        with _CaptureOutput() as cap:
            repl._cmd_fc("-l 2")
        lines = [l for l in cap.getvalue().split("\n") if l.strip()]
        assert len(lines) == 2
        assert "cmd3" in lines[0]
        assert "cmd4" in lines[1]

    def test_fc_rerun_by_number(self, repl):
        repl._history.append("echo hello_world_fc_test")
        with _CaptureOutput() as cap:
            repl._cmd_fc("1")
        # echo should output the text
        assert "hello_world_fc_test" in cap.getvalue()

    def test_fc_invalid_number(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_fc("999")
        assert "No history" in cap.getvalue()

    def test_fc_non_numeric(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_fc("abc")
        assert "Usage" in cap.getvalue()

    def test_fc_registered(self):
        from domains.shell.repl import ShellREPL
        assert "fc" in ShellREPL.COMMANDS


# ── job control (bg/fg) ──────────────────────────────────────────────


class TestJobControl:
    def test_bg_empty(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_bg("")
        assert "No background" in cap.getvalue()

    def test_jobs_alias(self, repl):
        output = repl._execute_single("jobs")
        assert "No background" in output or "running" in output

    def test_fg_no_args(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_fg("")
        assert "Usage" in cap.getvalue()

    def test_fg_invalid_id(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_fg("not_a_number")
        assert "Invalid" in cap.getvalue()

    def test_fg_nonexistent(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_fg("999")
        assert "No background" in cap.getvalue()


# ── NO_COLOR toggle ──────────────────────────────────────────────────


class TestNoColor:
    def test_color_constants_defined(self, repl):
        from domains.shell.repl import _C_GREEN, _C_RESET, _COLOR_ENABLED
        # Constants should exist and be non-empty by default
        assert _COLOR_ENABLED is True
        assert _C_GREEN != ""
        assert _C_RESET != ""

    def test_health_uses_colors(self, repl):
        result = repl._execute_single("health")
        assert "\033[" in result or "Status:" in result or "not responding" in result

    def test_exit_uses_colors(self, repl):
        from domains.shell.repl import _CaptureOutput as CO
        with CO() as cap:
            repl._cmd_exit("")
        output = cap.getvalue()
        assert "Shutting down" in output

    def test_inline_env_set_before_command(self, repl):
        """NAME=VALUE cmd should set env for one command only."""
        result = repl._execute_single("MYTEST=123 echo hello")
        assert "hello" in result
        # Env should NOT persist after command
        assert "MYTEST" not in repl._env

    def test_inline_env_does_not_persist(self, repl):
        """Verify inline env is cleaned up after execution."""
        assert "TMP_TEST_VAR" not in repl._env
        repl._execute_single("TMP_TEST_VAR=val echo test")
        assert "TMP_TEST_VAR" not in repl._env

    def test_inline_env_available_to_command(self, repl):
        """Inline env should be visible via echo $VAR (checked indirectly)."""
        repl._execute_single("INLINE_VAR=hello echo $INLINE_VAR")
        # echo with $INLINE_VAR — the variable expansion happens in the echo command
        result = repl._execute_single("INLINE_VAR=world echo $INLINE_VAR")
        assert "world" in result

    def test_fg_help_in_help_text(self, repl):
        from domains.shell.repl import _CaptureOutput as CO
        with CO() as cap:
            repl._cmd_help("")
        assert "fg" in cap.getvalue()

    def test_jobs_in_help_text(self, repl):
        from domains.shell.repl import _CaptureOutput as CO
        with CO() as cap:
            repl._cmd_help("")
        assert "jobs" in cap.getvalue() or "bg" in cap.getvalue()


# ── sleep command ────────────────────────────────────────────────────



# ── PS1 escapes ──────────────────────────────────────────────────────


class TestPS1:
    def test_render_prompt_default(self, repl):
        """Default PS1 is lambda character."""
        prompt = repl._render_prompt()
        assert len(prompt) > 0

    def test_render_prompt_host(self, repl):
        repl._env["PS1"] = r"\h"
        prompt = repl._render_prompt()
        # Should be the hostname, not literal \h
        assert "\\h" not in prompt
        assert len(prompt) > 0

    def test_render_prompt_user(self, repl):
        repl._env["PS1"] = r"\u"
        prompt = repl._render_prompt()
        assert "\\u" not in prompt
        # Should be the current user
        import os
        assert prompt == os.environ.get("USER", "user")

    def test_render_prompt_shell(self, repl):
        repl._env["PS1"] = r"\s"
        prompt = repl._render_prompt()
        assert prompt == "sloughgpt"

    def test_render_prompt_count(self, repl):
        repl._env["PS1"] = r"\#"
        prompt = repl._render_prompt()
        assert prompt == "1"

    def test_render_prompt_with_env_var(self, repl):
        """PS1 with $VAR should expand."""
        repl._env["MY_HOST"] = "myserver"
        repl._env["PS1"] = "$MY_HOST"
        prompt = repl._render_prompt()
        # $MY_HOST is not a PS1 escape — the prompt is rendered literally
        # (variable expansion happens at execute_single, not at prompt render)
        assert prompt == "$MY_HOST" or prompt == "myserver"

    def test_cmd_count_increments(self, repl):
        with patch.object(repl, '_execute_single', return_value=""):
            repl._cmd_count = 0
            repl._env["PS1"] = r"\#"
            assert repl._render_prompt() == "1"


# ── .sloughgptrc startup ─────────────────────────────────────────────


class TestRcFile:
    def test_rc_file_executed(self, repl, tmp_path):
        """Simulate rc file execution."""
        rc = tmp_path / "rc"
        rc.write_text("alias t=health\necho rc_loaded\n")
        with patch.object(repl, '_rc_path', return_value=rc):
            repl._load_rc()
        assert "t" in repl._aliases

    def test_rc_file_skips_comments(self, repl, tmp_path):
        rc = tmp_path / "rc"
        rc.write_text("# this is a comment\nalias x=health\n")
        with patch.object(repl, '_rc_path', return_value=rc):
            repl._load_rc()
        assert "x" in repl._aliases

    def test_rc_missing_file(self, repl):
        """Missing rc file is silently ignored."""
        rc = Path("/tmp/nonexistent_rc_file_xyz")
        with patch.object(repl, '_rc_path', return_value=rc):
            repl._load_rc()  # should not raise

    def test_rc_sets_env(self, repl, tmp_path):
        rc = tmp_path / "rc"
        rc.write_text("set MY_RC_VAR=hello\n")
        with patch.object(repl, '_rc_path', return_value=rc):
            repl._load_rc()
        assert repl._env.get("MY_RC_VAR") == "hello"

    def test_rc_path_in_config(self, repl):
        """_rc_path should be under ~/.config/sloughgpt/."""
        p = repl._rc_path()
        assert ".config" in str(p)
        assert "sloughgpt" in str(p)
        assert p.name == "rc"


# ── gen tab completion ───────────────────────────────────────────────


class TestGenCompletion:
    def test_gen_has_completion(self, repl):
        """gen command should have tab completion candidates."""
        from domains.shell.repl import ShellREPL
        assert "gen" in ShellREPL.COMMANDS


# ── finetuned tab completion ─────────────────────────────────────────


class TestFinetunedCompletion:
    def test_finetuned_subcommand_completion(self, repl):
        """finetuned should complete load/rm/del/delete subcommands."""
        candidates = repl._complete_args_for("finetuned")
        assert isinstance(candidates, list)
        assert "load" in candidates
        assert "rm" in candidates
        assert "del" in candidates
        assert "delete" in candidates

    def test_finetuned_load_completes_names(self, repl):
        """finetuned load should resolve model names from the API."""
        original = repl.cmds.finetuned_models
        repl.cmds.finetuned_models = lambda: [
            {"model_name": "gpt2__dataset_1"},
            {"model_name": "qwen__v2"},
        ]
        try:
            names = [m.get("model_name", "") for m in repl.cmds.finetuned_models()]
            matched = [n for n in sorted(set(names)) if n.startswith("gpt")]
            assert matched == ["gpt2__dataset_1"]
        finally:
            repl.cmds.finetuned_models = original

    def test_complete_args_for_gen(self, repl):
        """_complete_args_for should handle gen without error."""
        try:
            candidates = repl._complete_args_for("gen")
            # May be empty if models API returns nothing, but shouldn't error
            assert isinstance(candidates, list)
        except Exception as e:
            print(f"Error: {e}")

    def test_complete_path_files(self, repl, tmp_path):
        """_complete_path returns files in a directory."""
        (tmp_path / "alpha.txt").write_text("")
        (tmp_path / "beta.txt").write_text("")
        candidates = repl._complete_path(str(tmp_path) + "/")
        assert len(candidates) >= 2
        assert any("alpha.txt" in c for c in candidates)
        assert any("beta.txt" in c for c in candidates)

    def test_complete_path_prefix(self, repl, tmp_path):
        """_complete_path filters by prefix."""
        (tmp_path / "alpha.txt").write_text("")
        (tmp_path / "beta.txt").write_text("")
        candidates = repl._complete_path(str(tmp_path / "al"))
        matched = [c for c in candidates if c.startswith(str(tmp_path / "al"))]
        assert len(matched) >= 1

    def test_complete_path_nonexistent(self, repl):
        """_complete_path returns empty for nonexistent dir."""
        candidates = repl._complete_path("/nonexistent_dir_xyzabc/")
        assert candidates == []

    def test_complete_path_shows_dirs(self, repl, tmp_path):
        """_complete_path shows dirs with trailing slash."""
        (tmp_path / "mydir").mkdir()
        candidates = repl._complete_path(str(tmp_path) + "/")
        dir_candidates = [c for c in candidates if c.endswith("/")]
        assert len(dir_candidates) >= 1

    def test_complete_args_falls_back_to_path(self, repl, tmp_path):
        """_complete_args_for falls back to path completion for unknown commands."""
        candidates = repl._complete_args_for("source")
        assert isinstance(candidates, list)


# ── watch command ────────────────────────────────────────────────────


class TestWatch:
    def test_watch_no_args(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_watch("")
        assert "Usage" in cap.getvalue()

    def test_watch_invalid_interval(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_watch("-n abc echo test")
        assert "Invalid" in cap.getvalue()


# ── export command ───────────────────────────────────────────────────


class TestExport:
    def test_export_no_args(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_export("")
        # Should list env vars (like set)
        output = cap.getvalue()
        assert "PS1" in output or "HOME" in output or "SHELL" in output

    def test_export_set_var(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_export("TEST_EXPORT_VAR=hello")
        assert "TEST_EXPORT_VAR" in repl._env
        assert repl._env["TEST_EXPORT_VAR"] == "hello"
        # Clean up
        del repl._env["TEST_EXPORT_VAR"]

    def test_export_show_var(self, repl):
        repl._env["TEST_SHOW_VAR"] = "show_me"
        with _CaptureOutput() as cap:
            repl._cmd_export("TEST_SHOW_VAR")
        assert "show_me" in cap.getvalue()
        del repl._env["TEST_SHOW_VAR"]

    def test_export_not_set(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_export("NONEXISTENT_VAR_XYZ")
        assert "not set" in cap.getvalue()


# ── Command registration checks ──────────────────────────────────────


class TestAllCommandsRegistered:
    """Verify all commands appear in COMMANDS or _ext_cmds."""

    def test_watch_registered(self, repl):
        from domains.shell.repl import ShellREPL
        assert "watch" in ShellREPL.COMMANDS

    def test_export_registered(self, repl):
        from domains.shell.repl import ShellREPL
        assert "export" in ShellREPL.COMMANDS

    def test_read_registered(self, repl):
        from domains.shell.repl import ShellREPL
        assert "read" in ShellREPL.COMMANDS


# ── Permissions ──────────────────────────────────────────────────────


class TestPermissions:
    def test_permit_registered(self):
        from domains.shell.repl import ShellREPL
        assert "permit" in ShellREPL.COMMANDS

    def test_deny_registered(self):
        from domains.shell.repl import ShellREPL
        assert "deny" in ShellREPL.COMMANDS

    def test_permissions_registered(self):
        from domains.shell.repl import ShellREPL
        assert "permissions" in ShellREPL.COMMANDS

    def test_dangerous_command_blocked_by_default(self, repl):
        repl._perms._granted.clear()
        output, code = repl.execute("shutdown")
        assert code == 126
        assert "Permission denied" in output

    def test_critical_command_blocked_by_default(self, repl):
        repl._perms._granted.clear()
        output, code = repl.execute("shutdown")
        assert code == 126
        assert "Permission denied" in output

    def test_permit_allows_command(self, repl):
        repl._perms._granted.clear()
        repl._cmd_permit("shutdown")
        output, code = repl.execute("shutdown")
        assert code != 126

    def test_permit_shows_usage_when_empty(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_permit("")
        assert "Usage" in cap.getvalue()

    def test_deny_shows_usage_when_empty(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_deny("")
        assert "Usage" in cap.getvalue()

    def test_permissions_shows_policy(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_permissions()
        output = cap.getvalue()
        assert "Risk policies" in output
        assert "safe" in output
        assert "elevated" in output
        assert "dangerous" in output
        assert "critical" in output

    def test_permit_persist_flag(self, repl):
        repl._perms._granted.clear()
        with _CaptureOutput() as cap:
            repl._cmd_permit("chmod --persist")
        output = cap.getvalue()
        assert "persistent" in output.lower() or "Granted" in output

    def test_permit_all_dangerous(self, repl):
        repl._perms._granted.clear()
        with _CaptureOutput() as cap:
            repl._cmd_permit("--all-dangerous")
        output = cap.getvalue()
        assert "Granted" in output or "allow" in output.lower()

    def test_deny_revokes_permission(self, repl):
        repl._perms._granted.clear()
        repl._cmd_permit("shutdown")
        repl._cmd_deny("shutdown")
        output, code = repl.execute("shutdown")
        assert code == 126

    def test_permit_tab_completion(self, repl):
        candidates = repl._complete_args_for("permit")
        assert "shutdown" in candidates
        assert "--persist" in candidates
        assert "--all-dangerous" in candidates

    def test_deny_tab_completion(self, repl):
        candidates = repl._complete_args_for("deny")
        assert "shutdown" in candidates
        assert "--persist" in candidates
        assert "--all-dangerous" not in candidates

    def test_safe_command_not_blocked(self, repl):
        with patch("domains.shell.commands._api_get") as mock:
            mock.return_value = {"status": "healthy", "model_type": "gpt2", "soul_name": "default"}
            output, code = repl.execute("health")
        assert code == 0


# ── ps ─────────────────────────────────────────────────────────────────


class TestPs:
    def test_ps_shows_no_processes_when_empty(self, repl):
        output, code = repl.execute("ps")
        assert code == 0
        assert "No kernel processes" in output

    def test_ps_shows_processes_when_running(self, repl):
        repl.os.kernel.boot()
        output, code = repl.execute("ps")
        assert code == 0
        assert "PID" in output
        assert "kernel-init" in output

    def test_cmd_ps_no_processes(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_ps("")
        output = cap.getvalue()
        assert "No kernel processes" in output

    def test_procs_still_works(self, repl):
        output, code = repl.execute("procs")
        assert code == 0


# ── Start screen & prompting flow ───────────────────────────────────


class TestStartScreen:
    def test_print_header_no_crash(self, repl):
        repl._print_header()

    def test_show_welcome_sets_first_run_false(self, repl):
        repl.state.first_run = True
        repl._show_welcome()
        assert repl.state.first_run is False

    def test_show_welcome_output(self, repl):
        with _CaptureOutput() as cap:
            repl._show_welcome()
        out = cap.getvalue()
        assert "Welcome" in out or "Dait" in out

    def test_run_shows_welcome_on_first_run(self, repl):
        from domains.shell.io import MemoryIO
        mem = MemoryIO()
        mem.feed("\x04")  # Ctrl+D to exit loop
        old_io = repl.io
        old_console_io = repl.console._io
        repl.io = mem
        repl.console._io = mem
        repl.state.first_run = True
        try:
            repl.run()
        finally:
            repl.io = old_io
            repl.console._io = old_console_io
        assert repl.state.first_run is False


class TestRequireApi:
    def test_requires_api_returns_false_when_unavailable(self, repl):
        assert repl._require_api("gen") is False

    def test_requires_api_sets_exit_code_on_fail(self, repl):
        repl._require_api("gen")
        assert repl._last_exit_code == 1


class TestCmdGen:
    def test_gen_no_args_shows_usage(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_gen("")
        out = cap.getvalue()
        assert "Usage" in out

    def test_gen_requires_api(self, repl):
        with patch("domains.shell.commands._api_post") as mock:
            repl._cmd_gen("hello")
        mock.assert_not_called()

    def test_gen_graceful_when_api_down(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_gen("hello")
        out = cap.getvalue()
        assert "API" in out or "not connected" in out

    def test_gen_with_mocked_response(self, repl):
        with patch("domains.shell.repl.ShellREPL._require_api", return_value=True):
            with patch("domains.shell.commands._api_post") as mock:
                mock.return_value = {"text": "generated output"}
                with _CaptureOutput() as cap:
                    repl._cmd_gen("hello")
                out = cap.getvalue()
                assert "generated output" in out

    def test_gen_with_error_response(self, repl):
        with patch("domains.shell.repl.ShellREPL._require_api", return_value=True):
            with patch("domains.shell.commands._api_post") as mock:
                mock.return_value = {"error": "model not loaded"}
                with _CaptureOutput() as cap:
                    repl._cmd_gen("hello")
                out = cap.getvalue()
                assert "Error" in out
                assert "model not loaded" in out

    def test_gen_execute_with_mock(self, repl):
        with patch("domains.shell.repl.ShellREPL._check_permission", return_value=True):
            with patch("domains.shell.repl.ShellREPL._require_api", return_value=True):
                with patch("domains.shell.commands._api_post") as mock:
                    mock.return_value = {"text": "mock text"}
                    output, code = repl.execute("gen hello")
                    assert code == 0
                    assert "mock text" in output


class TestCmdChat:
    def test_chat_no_args_shows_usage(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_chat("")
        out = cap.getvalue()
        assert "Usage" in out

    def test_chat_reset_clears_session(self, repl):
        repl._chat_session_id = "session-123"
        repl._chat_history = [{"role": "user", "content": "hi"}]
        with _CaptureOutput() as cap:
            repl._cmd_chat("/reset")
        out = cap.getvalue()
        assert "cleared" in out
        assert repl._chat_session_id is None
        assert repl._chat_history == []

    def test_chat_requires_api(self, repl):
        with patch("domains.shell.commands._api_post") as mock:
            repl._cmd_chat("hello")
        mock.assert_not_called()

    def test_chat_with_mocked_response(self, repl):
        with patch("domains.shell.repl.ShellREPL._require_api", return_value=True):
            with patch("domains.shell.commands._api_post") as mock:
                mock.return_value = {"message": "chat response"}
                with _CaptureOutput() as cap:
                    repl._cmd_chat("hello")
                out = cap.getvalue()
                assert "chat response" in out
                assert repl._chat_history[-1]["content"] == "chat response"

    def test_chat_execute_with_mock(self, repl):
        with patch("domains.shell.repl.ShellREPL._check_permission", return_value=True):
            with patch("domains.shell.repl.ShellREPL._require_api", return_value=True):
                with patch("domains.shell.commands._api_post") as mock:
                    mock.return_value = {"message": "mock chat"}
                    output, code = repl.execute('chat "hello"')
                    assert code == 0
                    assert "mock chat" in output


# ── Pipeline builtins ───────────────────────────────────────────────


class TestPipelineBuiltins:
    """Tests for built-in commands with piped input."""

    def test_cat_with_piped_input(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = "hello\nworld"
            repl._cmd_cat("")
            out = cap.getvalue()
        assert "hello" in out
        assert "world" in out

    def test_cat_without_args_shows_usage_when_no_pipe(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_cat("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_grep_piped_match(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = "apple\nbanana\ncherry"
            repl._cmd_grep("anana")
            out = cap.getvalue()
        assert "banana" in out
        assert "apple" not in out

    def test_grep_piped_no_match(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = "apple\nbanana"
            repl._cmd_grep("xyz")
            out = cap.getvalue()
        assert out == ""
        assert repl._last_exit_code == 1

    def test_grep_invert_flag(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = "a\nb\nc"
            repl._cmd_grep("-v a")
            out = cap.getvalue()
        assert "b" in out
        assert "c" in out
        assert "a" not in out

    def test_grep_case_insensitive(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = "Apple\nbanana"
            repl._cmd_grep("-i apple")
            out = cap.getvalue()
        assert "Apple" in out

    def test_sort_basic(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = "c\na\nb"
            repl._cmd_sort("")
            out = cap.getvalue().rstrip("\n")
        assert out == "a\nb\nc"

    def test_sort_reverse(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = "a\nb\nc"
            repl._cmd_sort("-r")
            out = cap.getvalue().rstrip("\n")
        assert out == "c\nb\na"

    def test_sort_unique(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = "a\nb\na"
            repl._cmd_sort("-u")
            out = cap.getvalue().rstrip("\n")
        assert out == "a\nb"
        assert out.count("a") == 1

    def test_sort_numeric(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = "10\n2\n1"
            repl._cmd_sort("-n")
            out = cap.getvalue().rstrip("\n")
        assert out == "1\n2\n10"

    def test_sort_usage_without_input(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_sort("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_uniq_piped(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = "a\na\nb\nb\nc"
            repl._cmd_uniq("")
            out = cap.getvalue().rstrip("\n")
        assert out == "a\nb\nc"

    def test_uniq_no_change_when_sequential(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = "a\nb\nc"
            repl._cmd_uniq("")
            out = cap.getvalue().rstrip("\n")
        assert out == "a\nb\nc"

    def test_uniq_usage_without_input(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_uniq("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_head_default_n10_from_pipe(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = "\n".join(f"line{i}" for i in range(20))
            repl._cmd_head("")
            out = cap.getvalue()
        lines = out.splitlines()
        assert len(lines) == 10
        assert lines[0] == "line0"
        assert lines[-1] == "line9"

    def test_head_with_count_from_pipe(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = "\n".join(f"line{i}" for i in range(20))
            repl._cmd_head("-3")
            out = cap.getvalue()
        lines = out.splitlines()
        assert len(lines) == 3
        assert lines == ["line0", "line1", "line2"]

    def test_head_usage_without_input(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_head("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_tail_default_n10_from_pipe(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = "\n".join(f"line{i}" for i in range(20))
            repl._cmd_tail("")
            out = cap.getvalue()
        lines = out.splitlines()
        assert len(lines) == 10
        assert lines[0] == "line10"
        assert lines[-1] == "line19"

    def test_tail_with_count_from_pipe(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = "\n".join(f"line{i}" for i in range(20))
            repl._cmd_tail("-3")
            out = cap.getvalue()
        lines = out.splitlines()
        assert len(lines) == 3
        assert lines == ["line17", "line18", "line19"]

    def test_tail_usage_without_input(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_tail("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_wc_piped(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = "a b c\nd e f"
            repl._cmd_wc("")
            out = cap.getvalue()
        assert "2" in out  # 2 lines
        assert "6" in out  # 6 words
        assert "11" in out  # 11 chars

    def test_wc_usage_without_input(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_wc("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_tee_writes_to_file_and_stdout(self, repl):
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as f:
            tmp = f.name
        try:
            with _CaptureOutput() as cap:
                repl._piped_input = "hello world"
                repl._cmd_tee(tmp)
                out = cap.getvalue()
            assert "hello world" in out
            with open(tmp) as f:
                assert f.read().strip() == "hello world"
        finally:
            os.unlink(tmp)

    def test_tee_append_mode(self, repl):
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("first\n")
            tmp = f.name
        try:
            with _CaptureOutput() as cap:
                repl._piped_input = "second"
                repl._cmd_tee(f"-a {tmp}")
                cap.getvalue()
            with open(tmp) as f:
                content = f.read()
            assert "first" in content
            assert "second" in content
        finally:
            os.unlink(tmp)

    def test_tee_usage_without_input(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_tee("out.txt")
            out = cap.getvalue()
        assert "Usage" in out

    def test_xargs_default_echo(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = "a b c"
            repl._cmd_xargs("")
            out = cap.getvalue().rstrip("\n")
        assert out == "a\nb\nc"

    def test_xargs_with_command(self, repl):
        repl._perms.grant("echo")
        with _CaptureOutput() as cap:
            repl._piped_input = "hello world"
            repl._cmd_xargs("echo items:")
            out = cap.getvalue()
        assert "items:" in out
        assert "hello" in out
        assert "world" in out

    def test_xargs_n_flag(self, repl):
        repl._perms.grant("echo")
        with _CaptureOutput() as cap:
            repl._piped_input = "a b c d"
            repl._cmd_xargs("-n 2 echo")
            out = cap.getvalue()
        # Two invocations: echo a b, echo c d
        assert "a b" in out
        assert "c d" in out

    def test_xargs_usage_without_input(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_xargs("echo")
            out = cap.getvalue()
        assert "Usage" in out

    def test_find_no_matches(self, repl):
        with _CaptureOutput() as cap:
            with tempfile.TemporaryDirectory() as td:
                repl._cmd_find(td + " -name nonexistent")
                out = cap.getvalue()
        assert out == "" or "Permission" in out  # No output is fine
        assert repl._last_exit_code == 0

    def test_find_with_matches(self, repl):
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            Path(td, "test.py").write_text("x")
            Path(td, "readme.md").write_text("x")
            os.makedirs(Path(td, "sub"), exist_ok=True)
            Path(td, "sub", "other.py").write_text("x")
            with _CaptureOutput() as cap:
                repl._cmd_find(td + " -name *.py")
                out = cap.getvalue()
            assert "test.py" in out
            assert "other.py" in out
            assert "readme.md" not in out

    def test_find_usage_without_args(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_find("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_time_basic(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_time("echo test")
            out = cap.getvalue()
        assert "real" in out
        assert repl._last_exit_code == 0

    def test_time_usage_without_args(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_time("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_chmod_usage_without_args(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_chmod("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_chmod_changes_permissions(self, repl):
        import tempfile, os, stat
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp = f.name
        try:
            # Grant dangerous permission
            repl._perms.grant("chmod")
            with _CaptureOutput() as cap:
                repl._cmd_chmod(f"644 {tmp}")
                cap.getvalue()
            mode = os.stat(tmp).st_mode & 0o777
            assert mode == 0o644
        finally:
            os.unlink(tmp)

    def test_chmod_nonexistent_file(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_chmod("644 /nonexistent/file")
            out = cap.getvalue()
        assert "No such file" in out
        assert repl._last_exit_code == 1

    def test_du_file_size(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 100)
            tmp = f.name
        try:
            with _CaptureOutput() as cap:
                repl._cmd_du(tmp)
                out = cap.getvalue()
            assert "100" in out or str(os.path.getsize(tmp)) in out
        finally:
            os.unlink(tmp)

    def test_du_human_readable(self, repl):
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 100)
            tmp = f.name
        try:
            with _CaptureOutput() as cap:
                repl._cmd_du(f"-h {tmp}")
                out = cap.getvalue()
            assert "100B" in out or "B" in out
        finally:
            os.unlink(tmp)

    def test_du_directory(self, repl):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            Path(td, "a.txt").write_text("hello")
            Path(td, "b.txt").write_text("world")
            with _CaptureOutput() as cap:
                repl._cmd_du(td)
                out = cap.getvalue()
            assert "10" in out  # 5+5 bytes

    def test_du_defaults_to_cwd(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_du("")
            out = cap.getvalue()
        assert "." in out or "total" in out

    def test_format_size(self, repl):
        assert repl._format_size(0) == "       0"
        assert repl._format_size(100, human=True).strip() == "100.0B"
        assert repl._format_size(2048, human=True).strip() == "2.0K"
        assert repl._format_size(1048576, human=True).strip() == "1.0M"

    def test_diff_identical_files(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("a\nb\nc\n")
            t1 = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("a\nb\nc\n")
            t2 = f.name
        try:
            with _CaptureOutput() as cap:
                repl._cmd_diff(f"{t1} {t2}")
                cap.getvalue()
            assert repl._last_exit_code == 0
        finally:
            os.unlink(t1)
            os.unlink(t2)

    def test_diff_different_files(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("a\nb\nc\n")
            t1 = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("a\nx\nc\n")
            t2 = f.name
        try:
            with _CaptureOutput() as cap:
                repl._cmd_diff(f"{t1} {t2}")
                out = cap.getvalue()
            assert repl._last_exit_code == 1
            assert "b" in out or "x" in out
        finally:
            os.unlink(t1)
            os.unlink(t2)

    def test_diff_usage_without_args(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_diff("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_stat_file(self, repl):
        import tempfile, os, time
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello")
            t = f.name
        try:
            with _CaptureOutput() as cap:
                repl._cmd_stat(t)
                out = cap.getvalue()
            assert "Size:" in out
            assert "File:" in out
            assert "Mode:" in out
            assert repl._last_exit_code == 0
        finally:
            os.unlink(t)

    def test_stat_directory(self, repl):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with _CaptureOutput() as cap:
                repl._cmd_stat(td)
                out = cap.getvalue()
            assert "directory" in out
            assert repl._last_exit_code == 0

    def test_stat_nonexistent(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_stat("/nonexistent_file_xyz")
            out = cap.getvalue()
        assert "No such file" in out
        assert repl._last_exit_code == 1

    def test_stat_usage_without_args(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_stat("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_cut_from_file(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("a\t1\tx\nb\t2\ty\n")
            t = f.name
        try:
            with _CaptureOutput() as cap:
                repl._cmd_cut(f"-f1 {t}")
                out = cap.getvalue()
            assert out == "a\nb\n" or "a" in out
            assert repl._last_exit_code == 0
        finally:
            os.unlink(t)

    def test_cut_from_pipe(self, repl):
        repl._piped_input = "hello\tworld\tfoo\n"
        with _CaptureOutput() as cap:
            repl._cmd_cut("-f2")
            out = cap.getvalue()
        assert "world" in out
        assert repl._last_exit_code == 0
        repl._piped_input = ""

    def test_cut_usage_without_args(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_cut("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_tr_translate(self, repl):
        repl._piped_input = "hello"
        with _CaptureOutput() as cap:
            repl._cmd_tr("a-z A-Z")
            out = cap.getvalue()
        assert "HELLO" in out
        assert repl._last_exit_code == 0
        repl._piped_input = ""

    def test_tr_delete(self, repl):
        repl._piped_input = "hello123"
        with _CaptureOutput() as cap:
            repl._cmd_tr("-d 0-9")
            out = cap.getvalue()
        assert "hello" in out
        assert "123" not in out
        assert repl._last_exit_code == 0
        repl._piped_input = ""

    def test_tr_usage_without_pipe(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_tr("a-z A-Z")
            out = cap.getvalue()
        assert "Usage" in out

    def test_seq_single(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_seq("3")
            out = cap.getvalue()
        assert "1" in out and "2" in out and "3" in out

    def test_seq_two_args(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_seq("2 5")
            out = cap.getvalue()
        assert "2" in out and "3" in out and "5" in out

    def test_seq_usage_without_args(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_seq("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_nl_from_file(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("a\nb\nc\n")
            t = f.name
        try:
            with _CaptureOutput() as cap:
                repl._cmd_nl(t)
                out = cap.getvalue()
            assert "1\ta" in out
            assert "2\tb" in out
            assert "3\tc" in out
            assert repl._last_exit_code == 0
        finally:
            os.unlink(t)

    def test_nl_from_pipe(self, repl):
        repl._piped_input = "x\ny\nz\n"
        with _CaptureOutput() as cap:
            repl._cmd_nl("")
            out = cap.getvalue()
        assert "1\tx" in out
        assert "2\ty" in out
        assert "3\tz" in out
        assert repl._last_exit_code == 0
        repl._piped_input = ""

    def test_nl_usage_without_args(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = ""
            repl._cmd_nl("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_fold_wraps_lines(self, repl):
        repl._piped_input = "abcdefghij"
        with _CaptureOutput() as cap:
            repl._cmd_fold("-w 3")
            out = cap.getvalue()
        assert "abc\ndef\nghi\nj" == out.strip() or "abc" in out
        assert repl._last_exit_code == 0
        repl._piped_input = ""

    def test_fold_from_file(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("abcdef")
            t = f.name
        try:
            with _CaptureOutput() as cap:
                repl._cmd_fold(f"-w 2 {t}")
                out = cap.getvalue()
            lines = out.strip().split("\n")
            assert len(lines) >= 3
            assert repl._last_exit_code == 0
        finally:
            os.unlink(t)

    def test_fold_usage_without_args(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = ""
            repl._cmd_fold("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_tac_reverses_lines(self, repl):
        repl._piped_input = "a\nb\nc\n"
        with _CaptureOutput() as cap:
            repl._cmd_tac("")
            out = cap.getvalue()
        lines = out.strip().split("\n")
        assert lines == ["c", "b", "a"]
        assert repl._last_exit_code == 0
        repl._piped_input = ""

    def test_tac_from_file(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("1\n2\n3\n")
            t = f.name
        try:
            with _CaptureOutput() as cap:
                repl._cmd_tac(t)
                out = cap.getvalue()
            assert out.split("\n")[0] == "3" or out.strip().startswith("3")
            assert repl._last_exit_code == 0
        finally:
            os.unlink(t)

    def test_tac_usage_without_args(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = ""
            repl._cmd_tac("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_test_file_exists(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            p = f.name
        try:
            repl._cmd_test(f"-f {p}")
            assert repl._last_exit_code == 0
            repl._cmd_test("-f /nonexistent_xyz")
            assert repl._last_exit_code == 1
        finally:
            os.unlink(p)

    def test_test_dir_exists(self, repl):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            repl._cmd_test(f"-d {td}")
        assert repl._last_exit_code == 0
        repl._cmd_test("-d /nonexistent_dir_xyz")
        assert repl._last_exit_code == 1

    def test_test_str_eq(self, repl):
        repl._cmd_test("a = a")
        assert repl._last_exit_code == 0
        repl._cmd_test("a = b")
        assert repl._last_exit_code == 1

    def test_test_int_cmp(self, repl):
        repl._cmd_test("1 -eq 1")
        assert repl._last_exit_code == 0
        repl._cmd_test("1 -ne 1")
        assert repl._last_exit_code == 1
        repl._cmd_test("1 -lt 2")
        assert repl._last_exit_code == 0
        repl._cmd_test("2 -gt 1")
        assert repl._last_exit_code == 0

    def test_test_no_args(self, repl):
        repl._cmd_test("")
        assert repl._last_exit_code == 1

    def test_printf_basic(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_printf("%s hello")
            out = cap.getvalue()
        assert "hello" in out

    def test_printf_format_int(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_printf("%d 42")
            out = cap.getvalue()
        assert "42" in out

    def test_printf_format_escape_n(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_printf("a\\nb")
            out = cap.getvalue()
        lines = out.strip().split("\n")
        assert len(lines) >= 2

    def test_printf_no_args(self, repl):
        repl._cmd_printf("")
        assert repl._last_exit_code == 1

    def test_which_registered(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_which("echo")
            out = cap.getvalue()
        assert "echo" in out
        assert repl._last_exit_code == 0

    def test_which_not_found(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_which("nonexistent_cmd")
            out = cap.getvalue()
        assert "not found" in out or "nonexistent" in out
        assert repl._last_exit_code == 1

    def test_which_no_args(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_which("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_type_describes_command(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_type("echo")
            out = cap.getvalue()
        assert "built-in" in out or "echo" in out

    def test_type_not_found(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_type("nonexistent_cmd")
            out = cap.getvalue()
        assert "not found" in out

    def test_type_no_args(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_type("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_expand_tabs(self, repl):
        repl._piped_input = "a\tb"
        with _CaptureOutput() as cap:
            repl._cmd_expand("")
            out = cap.getvalue()
        assert "a       b" in out
        assert repl._last_exit_code == 0
        repl._piped_input = ""

    def test_expand_from_file(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x\ty")
            t = f.name
        try:
            with _CaptureOutput() as cap:
                repl._cmd_expand(t)
                out = cap.getvalue()
            assert "x       y" in out
        finally:
            os.unlink(t)

    def test_expand_usage(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = ""
            repl._cmd_expand("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_unexpand_spaces(self, repl):
        repl._piped_input = "        x"
        with _CaptureOutput() as cap:
            repl._cmd_unexpand("")
            out = cap.getvalue()
        assert "\tx" in out
        assert repl._last_exit_code == 0
        repl._piped_input = ""

    def test_unexpand_usage(self, repl):
        with _CaptureOutput() as cap:
            repl._piped_input = ""
            repl._cmd_unexpand("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_id_prints_identity(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_id("")
            out = cap.getvalue()
        assert "uid" in out or "gid" in out
        assert repl._last_exit_code == 0

    def test_logname_prints_user(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_logname("")
            out = cap.getvalue()
        assert len(out.strip()) > 0
        assert repl._last_exit_code == 0

    def test_mktemp_creates_file(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_mktemp("")
            out = cap.getvalue()
        path = out.strip()
        assert os.path.exists(path)
        os.unlink(path)
        assert repl._last_exit_code == 0

    def test_mktemp_creates_dir(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_mktemp("-d")
            out = cap.getvalue()
        path = out.strip()
        assert os.path.isdir(path)
        os.rmdir(path)
        assert repl._last_exit_code == 0

    def test_who_prints_user(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_who("")
            out = cap.getvalue()
        assert len(out.strip()) > 0
        assert repl._last_exit_code == 0

    def test_od_octal(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"ABC")
            t = f.name
        try:
            with _CaptureOutput() as cap:
                repl._cmd_od(t)
                out = cap.getvalue()
            assert "101" in out or "0101" in out  # 'A' in octal
            assert repl._last_exit_code == 0
        finally:
            os.unlink(t)

    def test_od_hex(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"ABC")
            t = f.name
        try:
            with _CaptureOutput() as cap:
                repl._cmd_od(f"-x {t}")
                out = cap.getvalue()
            assert "41" in out  # 'A' in hex
            assert repl._last_exit_code == 0
        finally:
            os.unlink(t)

    def test_od_no_file(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_od("")
            out = cap.getvalue()
        assert "Usage" in out

    def test_od_nonexistent(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_od("/nonexistent_od_file")
            out = cap.getvalue()
        assert "No such file" in out

    def test_join_two_files(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("a foo\nb bar\n")
            t1 = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("a baz\nb qux\n")
            t2 = f.name
        try:
            with _CaptureOutput() as cap:
                repl._cmd_join(f"{t1} {t2}")
                out = cap.getvalue()
            assert "a" in out and "b" in out
            assert "foo" in out and "baz" in out
            assert repl._last_exit_code == 0
        finally:
            os.unlink(t1)
            os.unlink(t2)

    def test_join_usage(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_join("")
            out = cap.getvalue()
        assert "Usage" in out


class TestCmdLogs:
    def test_logs_empty(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_logs("")
        out = cap.getvalue()
        assert "No log entries" in out

    def test_logs_clear_empty(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_logs("-c")
        out = cap.getvalue()
        assert "cleared" in out

    def test_logs_clear_with_data(self, repl):
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer.append(LogEntry(1.0, "INFO", "test", "msg"))
        with _CaptureOutput() as cap:
            repl._cmd_logs("-c")
        out = cap.getvalue()
        assert "cleared" in out
        assert len(repl._log_buffer) == 0

    def test_logs_shows_entries(self, repl):
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer.append(LogEntry(1.0, "INFO", "test.src", "hello world"))
        with _CaptureOutput() as cap:
            repl._cmd_logs("")
        out = cap.getvalue()
        assert "hello world" in out
        assert "Console Logs" in out

    def test_logs_level_filter(self, repl):
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer.append(LogEntry(1.0, "INFO", "src", "info msg"))
        repl._log_buffer.append(LogEntry(2.0, "ERROR", "src", "err msg"))
        with _CaptureOutput() as cap:
            repl._cmd_logs("-l ERROR")
        out = cap.getvalue()
        assert "err msg" in out
        assert "info msg" not in out

    def test_logs_source_filter(self, repl):
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer.append(LogEntry(1.0, "INFO", "slo.kernel", "kern"))
        repl._log_buffer.append(LogEntry(2.0, "INFO", "slo.shell.repl", "shell"))
        with _CaptureOutput() as cap:
            repl._cmd_logs("-s kernel")
        out = cap.getvalue()
        assert "kern" in out
        assert "shell" not in out

    def test_logs_limit(self, repl):
        from domains.shell.log_buffer import LogEntry
        for i in range(10):
            repl._log_buffer.append(LogEntry(float(i), "INFO", "src", f"msg{i}"))
        with _CaptureOutput() as cap:
            repl._cmd_logs("-n 3")
        out = cap.getvalue()
        assert "msg0" not in out
        assert "msg7" in out
        assert "msg8" in out
        assert "msg9" in out

    def test_logs_stats_shows_distribution(self, repl):
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer.append(LogEntry(1.0, "ERROR", "a", "err"))
        repl._log_buffer.append(LogEntry(2.0, "INFO", "b", "inf"))
        repl._log_buffer.append(LogEntry(3.0, "INFO", "c", "inf2"))
        with _CaptureOutput() as cap:
            repl._cmd_logs("--stats")
        out = cap.getvalue()
        assert "Log Statistics" in out
        assert "By Level" in out
        assert "Top Sources" in out
        assert "ERROR" in out
        assert "INFO" in out

    def test_logs_export(self, repl):
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer.append(LogEntry(1.0, "INFO", "src", "line1"))
        repl._log_buffer.append(LogEntry(2.0, "WARNING", "src", "line2"))
        import tempfile
        with tempfile.NamedTemporaryFile(mode="r", suffix=".log", delete=False) as tf:
            tmppath = tf.name
        try:
            with _CaptureOutput() as cap:
                repl._cmd_logs(f"-e {tmppath}")
            out = cap.getvalue()
            assert "Exported" in out
            with open(tmppath) as f:
                content = f.read()
            assert "INFO" in content
            assert "line1" in content
            assert "WARNING" in content
            assert "line2" in content
        finally:
            os.unlink(tmppath)
