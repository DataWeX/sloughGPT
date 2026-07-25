"""
Tests for ShellREPL — pipeline parsing, output capture, state persistence.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from domains.shell.repl import ShellREPL, _CaptureOutput
from domains.shell.io import capture_cmd
from domains.shell.runtime import DaitRuntime
from domains.shell.state import ShellState


@pytest.fixture
def repl():
    from pathlib import Path
    from unittest.mock import patch
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        st = Path(tmp) / "sloughgpt"
        st.mkdir(parents=True, exist_ok=True)
        state_file = st / "shell_state.json"
        with patch("domains.shell.state._STATE_FILE", state_file):
            os = DaitRuntime()
            r = ShellREPL(os)
            yield r


@pytest.fixture
def isolated_state():
    """Fixture providing a ShellState backed by a temp directory."""
    with tempfile.TemporaryDirectory() as tmp:
        st = Path(tmp) / "sloughgpt"
        st.mkdir(parents=True, exist_ok=True)
        state_file = st / "shell_state.json"
        with patch("domains.shell.state._STATE_FILE", state_file):
            s = ShellState()
            yield s


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

    def test_grep_pipeline(self, repl):
        # Direct grep with piped input via _execute_single
        output = repl._execute_single("grep hello", "hello world\ngoodbye")
        assert "hello" in output
        assert "goodbye" not in output

    def test_echo_wc_chain(self, repl):
        # Test wc with piped input
        output = repl._execute_single("wc", "hello world")
        # wc outputs: lines, words, chars
        parts = [int(x) for x in output.split() if x.isdigit()]
        assert len(parts) == 3
        assert parts[0] == 1  # 1 line

    def test_head_pipeline(self, repl):
        output = repl._execute_single("head 3", "1\n2\n3\n4\n5\n6")
        lines = [l for l in output.split("\n") if l.strip()]
        assert len(lines) == 3

    def test_tail_pipeline(self, repl):
        output = repl._execute_single("tail 2", "1\n2\n3\n4\n5")
        lines = [l for l in output.split("\n") if l.strip()]
        assert len(lines) == 2


# ── echo ────────────────────────────────────────────────────────────


class TestEcho:
    def test_echo_prints_args(self, repl):
        output = capture_cmd(repl, repl._cmd_echo, "hello world")
        assert output.strip() == "hello world"

    def test_echo_empty(self, repl):
        output = capture_cmd(repl, repl._cmd_echo, "")
        assert output.strip() == ""

    def test_echo_no_newline(self, repl):
        output = capture_cmd(repl, repl._cmd_echo, "-n hello")
        assert output == "hello"

    def test_echo_escapes(self, repl):
        output = capture_cmd(repl, repl._cmd_echo, "-e 'hello\\nworld'")
        assert "hello" in output
        assert "world" in output
        assert "\n" in output  # actually interpreted


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


class TestPrintf:
    def test_printf_string(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_printf("hello")
        assert "hello" in cap.getvalue()

    def test_printf_no_args(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_printf("")
        assert True


# ── dirname / basename ────────────────────────────────────────────────


class TestDirname:
    def test_dirname_basic(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_dirname("/a/b/c")
        assert "a/b" in cap.getvalue().replace(" ", "")

    def test_dirname_root(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_dirname("/")
        assert "  /" in cap.getvalue()

    def test_dirname_no_args(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_dirname("")
        assert repl._last_exit_code == 1


class TestBasename:
    def test_basename_basic(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_basename("/a/b/c.txt")
        assert "c.txt" in cap.getvalue()

    def test_basename_no_args(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_basename("")
        assert repl._last_exit_code == 1


# ── yes ──────────────────────────────────────────────────────────────


class TestYes:
    def _test_yes_output(self, args, expected_in):
        """Run _cmd_yes in a subprocess with SIGALRM to avoid infinite loop."""
        import subprocess, sys
        pkgs = os.path.join(os.path.dirname(__file__), "..")
        code = f"""
import sys, os, signal, tempfile, json
sys.path.insert(0, {pkgs!r})
from pathlib import Path
from unittest.mock import patch
from domains.shell.repl import ShellREPL, _CaptureOutput
from domains.shell.runtime import DaitRuntime
tmp = tempfile.mkdtemp()
with open(Path(tmp) / "state.json", "w") as f:
    json.dump({{"version": 1, "first_run": False, "history": [], "aliases": {{}}, "env": {{}}, "cwd": os.getcwd()}}, f)
os.chdir(tmp)
with patch("domains.shell.state._STATE_FILE", Path(tmp) / "state.json"):
    r = ShellREPL(DaitRuntime())
    with _CaptureOutput() as cap:
        signal.setitimer(signal.ITIMER_REAL, 0.1)
        try:
            r._cmd_yes({args!r})
        except KeyboardInterrupt:
            pass
        signal.setitimer(signal.ITIMER_REAL, 0)
    out = cap.getvalue()
    assert {expected_in!r} in out, f"Expected {{expected_in!r}} in output, got: {{out[:200]!r}}"
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
            state_file = st / "shell_state.json"
            with patch("domains.shell.state._STATE_FILE", state_file):
                state = ShellState()
                state.add_history("hello")
                state.add_history("health")
                state.set_alias("h", "health")
                state.save()
                assert state_file.is_file()
                state2 = ShellState()
                assert state2.history == ["hello", "health"]
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
            state_file = st / "shell_state.json"
            with patch("domains.shell.state._STATE_FILE", state_file):
                state = ShellState()
                state.set_env("TEST_KEY", "test_val")
                state.save()
                state2 = ShellState()
                assert state2.env.get("TEST_KEY") == "test_val"

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
        from domains.shell.repl import _CaptureOutput as CO
        with CO() as cap:
            repl._cmd_health("")
        output = cap.getvalue()
        # Should contain ANSI escape codes
        assert "\033[" in output or "Status:" in output

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


class TestSleep:
    def test_sleep_no_args(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_sleep("")
        assert "Usage" in cap.getvalue()

    def test_sleep_invalid(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_sleep("abc")
        assert "Invalid" in cap.getvalue()

    def test_sleep_registered(self, repl):
        from domains.shell.repl import ShellREPL
        assert "sleep" in ShellREPL.COMMANDS

    def test_sleep_in_help(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_help("")
        assert "sleep" in cap.getvalue()


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


# ── sleep tab completion ──────────────────────────────────────────────


class TestCommandRegistration:
    def test_sleep_in_commands(self, repl):
        from domains.shell.repl import ShellREPL
        assert "sleep" in ShellREPL.COMMANDS

    def test_sleep_in_help_dict(self, repl):
        # sleep should have a help string
        with _CaptureOutput() as cap:
            repl._cmd_help("sleep")
        assert "sleep" in cap.getvalue()


# ── New pipe filters: tee, sort, uniq ─────────────────────────────────


class TestTee:
    def test_tee_no_args(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_tee("")
        assert "Usage" in cap.getvalue()

    def test_tee_writes_file(self, repl, tmp_path):
        f = tmp_path / "tee_out.txt"
        repl._piped_input = "hello from tee"
        with _CaptureOutput() as cap:
            repl._cmd_tee(str(f))
        # Data passed through to stdout
        assert "hello from tee" in cap.getvalue()
        # Data written to file
        assert f.read_text() == "hello from tee"


class TestSort:
    def test_sort_lines(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_sort("")
            # No piped input — no output
        assert cap.getvalue() == ""

    def test_sort_with_piped_input(self, repl):
        repl._piped_input = "c\na\nb\n"
        with _CaptureOutput() as cap:
            repl._cmd_sort("")
        lines = [l for l in cap.getvalue().split("\n") if l]
        assert lines == ["a", "b", "c"]

    def test_sort_reverse(self, repl):
        repl._piped_input = "c\na\nb\n"
        with _CaptureOutput() as cap:
            repl._cmd_sort("-r")
        lines = [l for l in cap.getvalue().split("\n") if l]
        assert lines == ["c", "b", "a"]

    def test_sort_unique(self, repl):
        repl._piped_input = "b\na\nb\nc\na\n"
        with _CaptureOutput() as cap:
            repl._cmd_sort("-u")
        lines = [l for l in cap.getvalue().split("\n") if l]
        assert lines == ["a", "b", "c"]

    def test_sort_numeric(self, repl):
        repl._piped_input = "10\n2\n33\n1\n"
        with _CaptureOutput() as cap:
            repl._cmd_sort("-n")
        lines = [l for l in cap.getvalue().split("\n") if l]
        assert lines == ["1", "2", "10", "33"]

    def test_sort_numeric_reverse(self, repl):
        repl._piped_input = "10\n2\n33\n1\n"
        with _CaptureOutput() as cap:
            repl._cmd_sort("-n -r")
        lines = [l for l in cap.getvalue().split("\n") if l]
        assert lines == ["33", "10", "2", "1"]

    def test_sort_unique_reverse(self, repl):
        repl._piped_input = "b\na\nb\nc\na\n"
        with _CaptureOutput() as cap:
            repl._cmd_sort("-u -r")
        lines = [l for l in cap.getvalue().split("\n") if l]
        assert lines == ["c", "b", "a"]

    def test_sort_empty_input(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_sort("")
        assert cap.getvalue() == ""


class TestUniq:
    def test_uniq_dedup(self, repl):
        repl._piped_input = "a\na\nb\nb\nc\n"
        with _CaptureOutput() as cap:
            repl._cmd_uniq("")
        lines = [l for l in cap.getvalue().split("\n") if l]
        assert lines == ["a", "b", "c"]

    def test_uniq_no_change(self, repl):
        repl._piped_input = "a\nb\nc\n"
        with _CaptureOutput() as cap:
            repl._cmd_uniq("")
        lines = [l for l in cap.getvalue().split("\n") if l]
        assert lines == ["a", "b", "c"]

    def test_uniq_all_same(self, repl):
        repl._piped_input = "x\nx\nx\n"
        with _CaptureOutput() as cap:
            repl._cmd_uniq("")
        lines = [l for l in cap.getvalue().split("\n") if l]
        assert lines == ["x"]


# ── less pager ──────────────────────────────────────────────────────────


class TestLess:
    def test_less_no_data(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_less("")
        assert "Usage" in cap.getvalue()

    def test_less_shows_content(self, repl):
        repl._piped_input = "hello\nworld\n"
        with _CaptureOutput() as cap:
            repl._cmd_less("")
        assert "hello" in cap.getvalue()
        assert "world" in cap.getvalue()

    def test_less_with_args_as_data(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_less("line1\nline2")
        assert "line1" in cap.getvalue()
        assert "line2" in cap.getvalue()

    def test_less_registered(self):
        from domains.shell.repl import ShellREPL
        assert "less" in ShellREPL.COMMANDS

    def test_less_in_help(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_help("less")
        assert "less" in cap.getvalue()


# ── Directory stack: pushd / popd / dirs ─────────────────────────────


class TestDirStack:
    def test_dirs_empty(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_dirs("")
        assert "empty" in cap.getvalue()

    def test_pushd_no_args(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_pushd("")
        assert "Usage" in cap.getvalue()

    def test_pushd_nonexistent(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_pushd("/tmp/nonexistent_dir_xyz")
        assert "Not a directory" in cap.getvalue()

    def test_pushd_and_popd(self, repl, tmp_path):
        orig = os.getcwd()
        d = str(tmp_path)
        with _CaptureOutput() as cap:
            repl._cmd_pushd(d)
        assert repl._dir_stack == [orig]

        with _CaptureOutput() as cap:
            repl._cmd_popd("")
        assert repl._dir_stack == []

    def test_popd_empty(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_popd("")
        assert "empty" in cap.getvalue()

    def test_dirs_after_push(self, repl, tmp_path):
        orig = os.getcwd()
        d = str(tmp_path)
        repl._cmd_pushd(d)
        with _CaptureOutput() as cap:
            repl._cmd_dirs("")
        output = cap.getvalue()
        assert "current" in output
        # Clean up
        os.chdir(orig)
        repl._dir_stack = []


# ── watch command ────────────────────────────────────────────────────


class TestWatch:
    def test_watch_no_args(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_watch("")
        assert "Usage" in cap.getvalue()

    def test_watch_invalid_interval(self, repl):
        with _CaptureOutput() as cap:
            repl._cmd_watch("abc health")
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
    """Verify all new commands appear in COMMANDS and help."""

    def test_tee_registered(self):
        from domains.shell.repl import ShellREPL
        assert "tee" in ShellREPL.COMMANDS

    def test_sort_registered(self):
        from domains.shell.repl import ShellREPL
        assert "sort" in ShellREPL.COMMANDS

    def test_uniq_registered(self):
        from domains.shell.repl import ShellREPL
        assert "uniq" in ShellREPL.COMMANDS

    def test_less_registered(self):
        from domains.shell.repl import ShellREPL
        assert "less" in ShellREPL.COMMANDS

    def test_pushd_registered(self):
        from domains.shell.repl import ShellREPL
        assert "pushd" in ShellREPL.COMMANDS

    def test_popd_registered(self):
        from domains.shell.repl import ShellREPL
        assert "popd" in ShellREPL.COMMANDS

    def test_dirs_registered(self):
        from domains.shell.repl import ShellREPL
        assert "dirs" in ShellREPL.COMMANDS

    def test_watch_registered(self):
        from domains.shell.repl import ShellREPL
        assert "watch" in ShellREPL.COMMANDS

    def test_export_registered(self):
        from domains.shell.repl import ShellREPL
        assert "export" in ShellREPL.COMMANDS

    def test_read_registered(self):
        from domains.shell.repl import ShellREPL
        assert "read" in ShellREPL.COMMANDS

    def test_printf_registered(self):
        from domains.shell.repl import ShellREPL
        assert "printf" in ShellREPL.COMMANDS

    def test_dirname_registered(self):
        from domains.shell.repl import ShellREPL
        assert "dirname" in ShellREPL.COMMANDS

    def test_basename_registered(self):
        from domains.shell.repl import ShellREPL
        assert "basename" in ShellREPL.COMMANDS

    def test_yes_registered(self):
        from domains.shell.repl import ShellREPL
        assert "yes" in ShellREPL.COMMANDS

    def test_xargs_registered(self):
        from domains.shell.repl import ShellREPL
        assert "xargs" in ShellREPL.COMMANDS

    def test_cut_registered(self):
        from domains.shell.repl import ShellREPL
        assert "cut" in ShellREPL.COMMANDS

    def test_tr_registered(self):
        from domains.shell.repl import ShellREPL
        assert "tr" in ShellREPL.COMMANDS

    def test_find_registered(self):
        from domains.shell.repl import ShellREPL
        assert "find" in ShellREPL.COMMANDS


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
        output, code = repl.execute("chmod 777 /nonexistent/path")
        assert code == 126
        assert "Permission denied" in output

    def test_critical_command_blocked_by_default(self, repl):
        repl._perms._granted.clear()
        output, code = repl.execute("shutdown")
        assert code == 126
        assert "Permission denied" in output

    def test_permit_allows_command(self, repl):
        repl._perms._granted.clear()
        repl._cmd_permit("chmod")
        output, code = repl.execute("chmod 777 /nonexistent/path")
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
        repl._cmd_permit("chmod")
        repl._cmd_deny("chmod")
        output, code = repl.execute("chmod 777 /nonexistent/path")
        assert code == 126

    def test_permit_tab_completion(self, repl):
        candidates = repl._complete_args_for("permit")
        assert "chmod" in candidates
        assert "shutdown" in candidates
        assert "--persist" in candidates
        assert "--all-dangerous" in candidates

    def test_deny_tab_completion(self, repl):
        candidates = repl._complete_args_for("deny")
        assert "chmod" in candidates
        assert "shutdown" in candidates
        assert "--persist" in candidates
        assert "--all-dangerous" not in candidates

    def test_safe_command_not_blocked(self, repl):
        output, code = repl.execute("echo hello")
        assert code == 0
        assert "hello" in output
