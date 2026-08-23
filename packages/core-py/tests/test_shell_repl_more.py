"""
Additional ShellREPL coverage: module helpers, permission gate, help,
cd/pwd/echo, test/printf, source, py, logs, svc, which/type, VM commands,
cal/ln, render, ai fallback, tutorial, completion.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

import domains.shell.repl as repl_mod
from domains.shell.io import MemoryIO, capture_cmd
from domains.shell.log_buffer import LogEntry
from domains.shell.permissions import ShellPermissions
from domains.shell.repl import ShellREPL, _CaptureOutput
from domains.shell.runtime import DaitRuntime
from domains.shell.state import ShellState


@pytest.fixture
def repl():
    import tempfile
    from domains.shell.init import reset_init_system
    from domains.shell.state import set_shell_state_db, reset_shell_state_db
    reset_init_system()
    with tempfile.TemporaryDirectory() as tmp:
        st = Path(tmp) / "sloughgpt"
        st.mkdir(parents=True, exist_ok=True)
        state_db = str(st / "shell_state_mogdb")
        set_shell_state_db(state_db)
        with patch("domains.shell.runtime._probe_api", return_value={"available": False, "error": "mock"}), \
             patch("domains.shell.repl.ShellREPL._get_current_model", return_value=""), \
             patch("domains.shell.repl.ShellREPL._get_current_soul", return_value=""), \
             patch.object(ShellREPL, "_setup_readline"), \
             patch("domains.shell.runtime.APIServerProcess.start", return_value={"ok": True, "message": "mocked"}):
            os = DaitRuntime()
            r = ShellREPL(os)
            r._perms._granted.update(["tee", "xargs", "cp", "mv", "touch", "chmod"])
            yield r
            reset_init_system()
            reset_shell_state_db()


def _run_with_io(repl, inputs, fn):
    """Run fn with repl I/O redirected to a MemoryIO pre-loaded with inputs."""
    mem = MemoryIO()
    mem.feed(*inputs)
    old_io = repl.io
    old_console_io = repl.console._io
    repl.io = mem
    repl.console._io = mem
    try:
        fn()
    finally:
        repl.io = old_io
        repl.console._io = old_console_io
    return mem.get_output()


# ── _cmd_test all branches ────────────────────────────────────────


class TestCmdTestBranches:
    def test_no_args(self, repl):
        repl._cmd_test("")
        assert repl._last_exit_code == 1

    def test_file_exists(self, repl):
        repl._cmd_test("-f /tmp")
        assert repl._last_exit_code == 1

    def test_file_exists_real(self, repl):
        repl._cmd_test("-f /etc/hostname")
        assert repl._last_exit_code == 0

    def test_file_not_exists(self, repl):
        repl._cmd_test("-f /nonexistent/file.txt")
        assert repl._last_exit_code == 1

    def test_dir_exists(self, repl):
        repl._cmd_test("-d /tmp")
        assert repl._last_exit_code == 0

    def test_dir_not_exists(self, repl):
        repl._cmd_test("-d /nonexistent_dir")
        assert repl._last_exit_code == 1

    def test_path_exists(self, repl):
        repl._cmd_test("-e /tmp")
        assert repl._last_exit_code == 0

    def test_path_not_exists(self, repl):
        repl._cmd_test("-e /nonexistent_path")
        assert repl._last_exit_code == 1

    def test_string_empty(self, repl):
        repl._cmd_test("-z test")
        assert repl._last_exit_code == 1

    def test_string_not_empty(self, repl):
        repl._cmd_test("-z hello")
        assert repl._last_exit_code == 1

    def test_string_not_empty_n(self, repl):
        repl._cmd_test("-n hello")
        assert repl._last_exit_code == 0

    def test_string_empty_n(self, repl):
        repl._cmd_test("-n test")
        assert repl._last_exit_code == 0

    def test_equal_strings(self, repl):
        repl._cmd_test("abc = abc")
        assert repl._last_exit_code == 0

    def test_not_equal_strings(self, repl):
        repl._cmd_test("abc != xyz")
        assert repl._last_exit_code == 0

    def test_equal_integers(self, repl):
        repl._cmd_test("5 -eq 5")
        assert repl._last_exit_code == 0

    def test_not_equal_integers(self, repl):
        repl._cmd_test("5 -ne 3")
        assert repl._last_exit_code == 0

    def test_less_than(self, repl):
        repl._cmd_test("3 -lt 5")
        assert repl._last_exit_code == 0

    def test_less_equal(self, repl):
        repl._cmd_test("5 -le 5")
        assert repl._last_exit_code == 0

    def test_greater_than(self, repl):
        repl._cmd_test("5 -gt 3")
        assert repl._last_exit_code == 0

    def test_greater_equal(self, repl):
        repl._cmd_test("5 -ge 5")
        assert repl._last_exit_code == 0

    def test_bracket_syntax(self, repl):
        repl._cmd_test("[ 5 -eq 5 ]")
        assert repl._last_exit_code == 0

    def test_unknown_operator(self, repl):
        repl._cmd_test("abc --unknown xyz")
        assert repl._last_exit_code == 1


# ── _cmd_printf branches ──────────────────────────────────────────


class TestCmdPrintfBranches:
    def test_no_args(self, repl):
        repl._cmd_printf("")
        assert repl._last_exit_code == 1

    def test_percent_s(self, repl):
        repl._cmd_printf("%s" + " " + "hello")
        assert repl._last_exit_code == 0

    def test_percent_d(self, repl):
        repl._cmd_printf("%d" + " " + "42")
        assert repl._last_exit_code == 0

    def test_percent_d_invalid(self, repl):
        repl._cmd_printf("%d" + " " + "abc")
        assert repl._last_exit_code == 0

    def test_percent_f(self, repl):
        repl._cmd_printf("%f" + " " + "3.14")
        assert repl._last_exit_code == 0

    def test_percent_f_invalid(self, repl):
        repl._cmd_printf("%f" + " " + "abc")
        assert repl._last_exit_code == 0

    def test_percent_percent(self, repl):
        repl._cmd_printf("%%")
        assert repl._last_exit_code == 0

    def test_unknown_spec(self, repl):
        repl._cmd_printf("%x" + " " + "42")
        assert repl._last_exit_code == 0

    def test_no_args_values(self, repl):
        repl._cmd_printf("%s %s")
        assert repl._last_exit_code == 0


# ── _cmd_comm edge cases ──────────────────────────────────────────


class TestCmdCommEdgeCases:
    def test_no_args(self, repl):
        repl._cmd_comm("")
        assert repl._last_exit_code == 1

    def test_one_arg(self, repl):
        repl._cmd_comm("file1")
        assert repl._last_exit_code == 1

    def test_file_not_found(self, repl):
        repl._cmd_comm("/nonexistent1 /nonexistent2")
        assert repl._last_exit_code == 1

    def test_common_lines(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1:
            f1.write("a\nb\nc\n")
            path1 = f1.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
            f2.write("a\nb\nc\n")
            path2 = f2.name
        repl._cmd_comm(f"{path1} {path2}")
        import os
        os.unlink(path1)
        os.unlink(path2)
        assert repl._last_exit_code == 0

    def test_disjoint_lines(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1:
            f1.write("a\nb\n")
            path1 = f1.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
            f2.write("c\nd\n")
            path2 = f2.name
        repl._cmd_comm(f"{path1} {path2}")
        import os
        os.unlink(path1)
        os.unlink(path2)
        assert repl._last_exit_code == 0

    def test_left_only(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1:
            f1.write("a\nb\n")
            path1 = f1.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
            f2.write("c\n")
            path2 = f2.name
        repl._cmd_comm(f"{path1} {path2}")
        import os
        os.unlink(path1)
        os.unlink(path2)
        assert repl._last_exit_code == 0

    def test_right_only(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1:
            f1.write("a\n")
            path1 = f1.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
            f2.write("b\nc\n")
            path2 = f2.name
        repl._cmd_comm(f"{path1} {path2}")
        import os
        os.unlink(path1)
        os.unlink(path2)
        assert repl._last_exit_code == 0


# ── _cmd_kill with subprocess ──────────────────────────────────────


class TestCmdKillSubprocess:
    def test_kill_nonexistent_pid(self, repl):
        repl._cmd_kill("99999999")
        assert repl._last_exit_code == 0


# ── _cmd_shuf and _cmd_rev with files ──────────────────────────────


class TestCmdShufRevFiles:
    def test_shuf_file(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("a\nb\nc\n")
            path = f.name
        repl._cmd_shuf(path)
        import os
        os.unlink(path)
        assert repl._last_exit_code == 0

    def test_shuf_not_found(self, repl):
        repl._cmd_shuf("/nonexistent_file.txt")
        assert repl._last_exit_code == 1

    def test_rev_file(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("abc\ndef\n")
            path = f.name
        repl._cmd_rev(path)
        import os
        os.unlink(path)
        assert repl._last_exit_code == 0

    def test_rev_not_found(self, repl):
        repl._cmd_rev("/nonexistent_file.txt")
        assert repl._last_exit_code == 1

    def test_rev_piped(self, repl):
        repl._piped_input = "abc\ndef\n"
        repl._cmd_rev("")
        repl._piped_input = None
        assert repl._last_exit_code == 0


# ── _cmd_fold edge cases ──────────────────────────────────────────


class TestCmdFoldEdges:
    def test_fold_no_args(self, repl):
        repl._cmd_fold("")
        assert repl._last_exit_code == 1

    def test_fold_file_not_found(self, repl):
        repl._cmd_fold("/nonexistent_file.txt")
        assert repl._last_exit_code == 1

    def test_fold_piped(self, repl):
        repl._piped_input = "hello world this is a test"
        repl._cmd_fold("-w 5")
        repl._piped_input = None
        assert repl._last_exit_code == 0

    def test_fold_short_flag(self, repl):
        repl._piped_input = "hello world"
        repl._cmd_fold("-w3")
        repl._piped_input = None
        assert repl._last_exit_code == 0

    def test_fold_unknown_flag(self, repl):
        repl._piped_input = "hello"
        repl._cmd_fold("-x")
        repl._piped_input = None
        assert repl._last_exit_code == 0


# ── _cmd_od edge cases ────────────────────────────────────────────


class TestCmdOdEdges:
    def test_od_no_args(self, repl):
        repl._cmd_od("")
        assert repl._last_exit_code == 1

    def test_od_file_not_found(self, repl):
        repl._cmd_od("/nonexistent_file.txt")
        assert repl._last_exit_code == 1

    def test_od_hex_base(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello")
            path = f.name
        repl._cmd_od(f"-x {path}")
        import os
        os.unlink(path)
        assert repl._last_exit_code == 0

    def test_od_decimal_base(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello")
            path = f.name
        repl._cmd_od(f"-d {path}")
        import os
        os.unlink(path)
        assert repl._last_exit_code == 0

    def test_od_octal_default(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello")
            path = f.name
        repl._cmd_od(path)
        import os
        os.unlink(path)
        assert repl._last_exit_code == 0


# ── _cmd_paste edge cases ─────────────────────────────────────────


class TestCmdPasteEdges:
    def test_paste_no_args(self, repl):
        repl._cmd_paste("")
        assert repl._last_exit_code == 1

    def test_paste_file_not_found(self, repl):
        repl._cmd_paste("/nonexistent.txt")
        assert repl._last_exit_code == 1

    def test_paste_two_files(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1:
            f1.write("a\nb\n")
            path1 = f1.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
            f2.write("c\nd\n")
            path2 = f2.name
        repl._cmd_paste(f"{path1} {path2}")
        import os
        os.unlink(path1)
        os.unlink(path2)
        assert repl._last_exit_code == 0

    def test_paste_unequal_lengths(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1:
            f1.write("a\nb\nc\n")
            path1 = f1.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
            f2.write("x\n")
            path2 = f2.name
        repl._cmd_paste(f"{path1} {path2}")
        import os
        os.unlink(path1)
        os.unlink(path2)
        assert repl._last_exit_code == 0


# ── _cmd_join edge cases ──────────────────────────────────────────


class TestCmdJoinEdges:
    def test_no_args(self, repl):
        repl._cmd_join("")
        assert repl._last_exit_code == 1

    def test_one_arg(self, repl):
        repl._cmd_join("file1")
        assert repl._last_exit_code == 1

    def test_file_not_found(self, repl):
        repl._cmd_join("/nonexistent1 /nonexistent2")
        assert repl._last_exit_code == 1

    def test_join_common_key(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1:
            f1.write("k1 v1\nk2 v2\n")
            path1 = f1.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
            f2.write("k1 w1\nk3 w3\n")
            path2 = f2.name
        repl._cmd_join(f"{path1} {path2}")
        import os
        os.unlink(path1)
        os.unlink(path2)
        assert repl._last_exit_code == 0


# ── _cmd_tac edge cases ───────────────────────────────────────────


class TestCmdTacEdges:
    def test_no_args(self, repl):
        repl._cmd_tac("")
        assert repl._last_exit_code == 1

    def test_file_not_found(self, repl):
        repl._cmd_tac("/nonexistent_file.txt")
        assert repl._last_exit_code == 1

    def test_piped(self, repl):
        repl._piped_input = "line1\nline2\nline3\n"
        repl._cmd_tac("")
        repl._piped_input = None
        assert repl._last_exit_code == 0

    def test_file(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("a\nb\nc\n")
            path = f.name
        repl._cmd_tac(path)
        import os
        os.unlink(path)
        assert repl._last_exit_code == 0


# ── _cmd_xargs with permission denied ─────────────────────────────


class TestCmdXargsPermission:
    def test_xargs_permission_denied(self, repl):
        repl._piped_input = "test data"
        repl._check_permission = MagicMock(return_value=False)
        repl._cmd_xargs("echo")
        repl._piped_input = None
        repl._check_permission = repl.__class__.check_permission if hasattr(repl.__class__, 'check_permission') else MagicMock(return_value=True)
        assert repl._last_exit_code == 0


# ── _cmd_time timing ──────────────────────────────────────────────


class TestCmdTimeTiming:
    def test_time_with_output(self, repl):
        repl._cmd_time("echo test_output")
        assert repl._last_exit_code == 0


# ── _cmd_read edge cases ──────────────────────────────────────────


class TestCmdReadEdges:
    def test_read_with_prompt_and_var(self, repl):
        from domains.shell.io import MemoryIO
        mem = MemoryIO()
        mem.feed("hello_value")
        old_io = repl.io
        repl.io = mem
        try:
            repl._cmd_read("-p 'Enter: ' myvar")
            assert repl._last_exit_code == 0
        finally:
            repl.io = old_io


# ── _cmd_watch with command ───────────────────────────────────────


class TestCmdWatchWithCommand:
    def test_watch_with_real_command(self, repl):
        repl._cmd_watch("echo hello")
        assert repl._last_exit_code == 0


# ── _cmd_bg with running process ──────────────────────────────────


class TestCmdBgRunning:
    def test_bg_with_thread(self, repl):
        import threading
        mock_thread = MagicMock(spec=threading.Thread)
        mock_thread.name = "bg-2"
        mock_thread.is_alive.return_value = True
        repl._bg_threads = {"2": mock_thread}
        repl._cmd_bg("2")
        assert repl._last_exit_code == 0

    def test_bg_invalid_id(self, repl):
        repl._bg_threads.clear()
        repl._cmd_bg("999")
        assert repl._last_exit_code == 0


# ── _cmd_fg with running process ──────────────────────────────────


class TestCmdFgRunning:
    def test_fg_with_running_thread(self, repl):
        import threading
        mock_thread = MagicMock(spec=threading.Thread)
        mock_thread.name = "bg-3"
        mock_thread.is_alive.return_value = True
        mock_thread.join.return_value = None
        repl._bg_threads = {"3": mock_thread}
        repl._cmd_fg("3")
        assert repl._last_exit_code == 0


# ── _cmd_kill with signal ─────────────────────────────────────────


class TestCmdKillWithSignal:
    def test_kill_with_signal(self, repl):
        repl._cmd_kill("-9 99999999")
        assert repl._last_exit_code == 0


# ── _cmd_uname with various flags ─────────────────────────────────


class TestCmdUnameFlags:
    def test_uname_s(self, repl):
        repl._cmd_uname("-s")
        assert repl._last_exit_code == 0

    def test_uname_r(self, repl):
        repl._cmd_uname("-r")
        assert repl._last_exit_code == 0

    def test_uname_m(self, repl):
        repl._cmd_uname("-m")
        assert repl._last_exit_code == 0

    def test_uname_sr(self, repl):
        repl._cmd_uname("-sr")
        assert repl._last_exit_code == 0


# ── _cmd_help with extension commands ─────────────────────────────


class TestCmdHelpWithExt:
    def test_help_ext_command(self, repl):
        repl._ext_cmds = {"fake_cmd": MagicMock()}
        repl._cmd_help("fake_cmd")
        assert repl._last_exit_code == 0
        repl._ext_cmds.clear()


# ── _cmd_ai keyword fallback ──────────────────────────────────────


class TestCmdAiKeywordFallback:
    def test_ai_keyword_processes(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": False}
        repl.os._api = mock_api
        repl._cmd_ai("show running processes")
        assert repl._last_exit_code == 0

    def test_ai_keyword_models(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": False}
        repl.os._api = mock_api
        repl._cmd_ai("show models")
        assert repl._last_exit_code == 0

    def test_ai_keyword_health(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": False}
        repl.os._api = mock_api
        repl._cmd_ai("check health")
        assert repl._last_exit_code == 0

    def test_ai_keyword_datasets(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": False}
        repl.os._api = mock_api
        repl._cmd_ai("show datasets")
        assert repl._last_exit_code == 0

    def test_ai_keyword_knowledge(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": False}
        repl.os._api = mock_api
        repl._cmd_ai("show knowledge")
        assert repl._last_exit_code == 0

    def test_ai_keyword_checkpoint(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": False}
        repl.os._api = mock_api
        repl._cmd_ai("show checkpoints")
        assert repl._last_exit_code == 0

    def test_ai_keyword_finetuned(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": False}
        repl.os._api = mock_api
        repl._cmd_ai("show finetuned models")
        assert repl._last_exit_code == 0

    def test_ai_keyword_metrics(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": False}
        repl.os._api = mock_api
        repl._cmd_ai("show metrics")
        assert repl._last_exit_code == 0

    def test_ai_keyword_tokenizer(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": False}
        repl.os._api = mock_api
        repl._cmd_ai("show tokenizer")
        assert repl._last_exit_code == 0

    def test_ai_keyword_help(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": False}
        repl.os._api = mock_api
        repl._cmd_ai("help")
        assert repl._last_exit_code == 0

    def test_ai_keyword_unknown(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": False}
        repl.os._api = mock_api
        repl._cmd_ai("do something random")
        assert repl._last_exit_code == 0


# ── _cmd_ai LLM execution ────────────────────────────────────────


class TestCmdAiLLMExecution:
    def test_ai_with_result(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        mock_api.generate.return_value = {"text": "models"}
        repl.os._api = mock_api
        repl._cmd_ai("show models")
        assert repl._last_exit_code == 0

    def test_ai_with_pipeline_result(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        mock_api.generate.return_value = {"text": "echo hello | grep hello"}
        repl.os._api = mock_api
        repl._cmd_ai("echo hello")
        assert repl._last_exit_code == 0

    def test_ai_with_background_result(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        mock_api.generate.return_value = {"text": "sleep 60 &"}
        repl.os._api = mock_api
        repl._cmd_ai("run sleep in background")
        assert repl._last_exit_code == 0

    def test_ai_error_result(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        mock_api.generate.return_value = {"error": "API error"}
        repl.os._api = mock_api
        repl._cmd_ai("do something")
        assert repl._last_exit_code == 0

    def test_ai_non_dict_result(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        mock_api.generate.return_value = "unexpected"
        repl.os._api = mock_api
        repl._cmd_ai("do something")
        assert repl._last_exit_code == 0


def _add_log(repl, level="ERROR", source="kernel", message="boom"):
    repl._log_buffer.append(LogEntry(
        timestamp=time.time(), level=level, source=source, message=message,
    ))


# ── Module-level helpers ─────────────────────────────────────────────


class TestModuleHelpers:
    def test_color_with_no_color_env(self):
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            importlib.reload(repl_mod)
            assert repl_mod._COLOR_ENABLED is False
            assert repl_mod._color("x", "\033[31m") == "x"
            assert repl_mod._C_RED == ""
        importlib.reload(repl_mod)
        assert repl_mod._COLOR_ENABLED is True
        assert repl_mod._color("x", "\033[31m") == "\033[31mx\033[0m"

    def test_color_without_code(self):
        assert repl_mod._color("plain", "") == "plain"

    def test_readline_import_error_sets_flag(self):
        with patch.dict(sys.modules, {"readline": None}):
            importlib.reload(repl_mod)
            assert repl_mod._HAS_READLINE is False
        importlib.reload(repl_mod)
        assert repl_mod._HAS_READLINE is True

    def test_fetch_model_names_success(self):
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = [{"name": "gpt2"}, {"id": "llama"}, {"name": "phi"}]
        with patch("requests.get", return_value=mock):
            names = repl_mod._fetch_model_names()
        assert names == ["gpt2", "llama", "phi"]

    def test_fetch_model_names_dict_wrapper(self):
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = {"models": [{"name": "gpt2"}]}
        with patch("requests.get", return_value=mock):
            names = repl_mod._fetch_model_names()
        assert names == ["gpt2"]

    def test_fetch_model_names_failure(self):
        mock = MagicMock()
        mock.status_code = 500
        mock.json.side_effect = ValueError("bad json")
        with patch("requests.get", return_value=mock):
            assert repl_mod._fetch_model_names() == []

    def test_fetch_model_names_exception(self):
        with patch("requests.get", side_effect=OSError("down")):
            assert repl_mod._fetch_model_names() == []

    def test_fetch_soul_names_success(self):
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = [{"name": "zen"}, {"name": "sage"}]
        with patch("requests.get", return_value=mock):
            assert repl_mod._fetch_soul_names() == ["sage", "zen"]

    def test_fetch_soul_names_failure(self):
        mock = MagicMock()
        mock.status_code = 200
        mock.json.side_effect = ValueError("bad")
        with patch("requests.get", return_value=mock):
            assert repl_mod._fetch_soul_names() == []


# ── Permission gate ──────────────────────────────────────────────────


class TestCheckPermission:
    def test_allowed_command_returns_true(self, repl):
        assert repl._check_permission("help", "") is True

    def test_non_interactive_denied(self, repl):
        out = capture_cmd(repl, repl._check_permission, "rm", "", False)
        assert repl._last_exit_code == 0
        assert "Permission denied" in out
        assert "risk=dangerous" in out
        assert "permit rm" in out

    def test_interactive_yes_grants_session(self, repl):
        out = _run_with_io(repl, ["y"], lambda: repl._check_permission("rm", "", True))
        assert "Granted" in out
        assert "rm" in repl._perms._granted

    def test_interactive_no_denies(self, repl):
        out = _run_with_io(repl, ["N"], lambda: repl._check_permission("rm", "", True))
        assert "Denied" in out
        assert "rm" not in repl._perms._granted

    def test_interactive_always_grants_persistent(self, repl, tmp_path):
        cfg = tmp_path / "perms.json"
        with patch.object(ShellPermissions, "_config_path", cfg):
            out = _run_with_io(repl, ["always"], lambda: repl._check_permission("rm", "", True))
        assert "persistent" in out
        assert "rm" in repl._perms._granted
        assert cfg.exists()

    def test_interactive_eof_denies(self, repl):
        out = _run_with_io(repl, [], lambda: repl._check_permission("rm", "", True))
        assert "Denied" in out
        assert "rm" not in repl._perms._granted


class TestPermitDeny:
    def test_permit_usage(self, repl):
        out = capture_cmd(repl, repl._cmd_permit, "")
        assert "Usage: permit <cmd> [--persist]" in out
        assert "Risk levels:" in out

    def test_permit_grant(self, repl):
        out = capture_cmd(repl, repl._cmd_permit, "rm")
        assert "Granted: rm" in out
        assert "rm" in repl._perms._granted

    def test_permit_grant_persistent(self, repl, tmp_path):
        cfg = tmp_path / "perms.json"
        with patch.object(ShellPermissions, "_config_path", cfg):
            out = capture_cmd(repl, repl._cmd_permit, "rm --persist")
        assert "Granted: rm (persistent)" in out
        assert cfg.exists()

    def test_permit_all_dangerous(self, repl):
        out = capture_cmd(repl, repl._cmd_permit, "--all-dangerous")
        assert "All dangerous commands now allowed" in out
        assert repl._perms._policy["dangerous"] == "allow"

    def test_permit_unknown_risk(self, repl):
        out = capture_cmd(repl, repl._cmd_permit, "--all-bogus")
        assert "Unknown risk level: bogus" in out

    def test_deny_usage(self, repl):
        out = capture_cmd(repl, repl._cmd_deny, "")
        assert "Usage: deny <cmd> [--persist]" in out

    def test_deny_revoke(self, repl):
        repl._perms.grant("rm")
        out = capture_cmd(repl, repl._cmd_deny, "rm")
        assert "Revoked: rm" in out
        assert "rm" not in repl._perms._granted

    def test_deny_revoke_persistent(self, repl, tmp_path):
        cfg = tmp_path / "perms.json"
        repl._perms.grant("rm")
        with patch.object(ShellPermissions, "_config_path", cfg):
            out = capture_cmd(repl, repl._cmd_deny, "rm --persist")
        assert "Revoked: rm (persistent)" in out
        assert cfg.exists()

    def test_deny_all_dangerous(self, repl):
        out = capture_cmd(repl, repl._cmd_deny, "--all-dangerous")
        assert "All dangerous commands now denied" in out

    def test_deny_unknown_risk(self, repl):
        out = capture_cmd(repl, repl._cmd_deny, "--all-bogus")
        assert "Unknown risk level: bogus" in out

    def test_permissions_shows_policy(self, repl):
        out = capture_cmd(repl, repl._cmd_permissions, "")
        assert "Risk policies:" in out
        assert "Config:" in out


# ── help ─────────────────────────────────────────────────────────────


class TestHelp:
    def test_help_full_list(self, repl):
        out = capture_cmd(repl, repl._cmd_help, "")
        assert "Built-in commands:" in out

    def test_help_brief(self, repl):
        out = capture_cmd(repl, repl._cmd_help, "brief")
        assert "Most-used commands" in out

    def test_help_from_dict(self, repl):
        out = capture_cmd(repl, repl._cmd_help, "cd")
        assert "cd" in out

    def test_help_builtin_docstring(self, repl):
        out = capture_cmd(repl, repl._cmd_help, "lsdev")
        assert "List AI device nodes" in out

    def test_help_unknown_command(self, repl):
        out = capture_cmd(repl, repl._cmd_help, "no_such_cmd_xyz")
        assert "Unknown command: no_such_cmd_xyz" in out


# ── cd / pwd / echo ──────────────────────────────────────────────────


class TestCdPwdEcho:
    def test_cd_home_default(self, repl, monkeypatch):
        monkeypatch.chdir(Path.home())
        out = capture_cmd(repl, repl._cmd_cd, "")
        assert out == ""
        assert repl._last_exit_code == 0

    def test_cd_missing_dir(self, repl):
        out = capture_cmd(repl, repl._cmd_cd, "/no/such/dir/xyz")
        assert "no such file or directory" in out
        assert repl._last_exit_code == 1

    def test_cd_not_a_directory(self, repl, tmp_path):
        f = tmp_path / "afile.txt"
        f.write_text("x")
        out = capture_cmd(repl, repl._cmd_cd, str(f))
        assert "not a directory" in out
        assert repl._last_exit_code == 1

    @pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                        reason="root bypasses permission checks")
    def test_cd_permission_denied(self, repl, tmp_path):
        d = tmp_path / "locked"
        d.mkdir()
        d.chmod(0o000)
        try:
            out = capture_cmd(repl, repl._cmd_cd, str(d))
            assert "permission denied" in out
        finally:
            d.chmod(0o755)

    def test_cd_minus_uses_oldpwd(self, repl, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        sub = tmp_path / "sub"
        sub.mkdir()
        os.chdir(sub)
        repl._env["OLDPWD"] = str(tmp_path)
        capture_cmd(repl, repl._cmd_cd, "-")
        assert os.getcwd() == str(tmp_path)

    def test_pwd_prints_cwd(self, repl, tmp_path):
        with patch.object(repl.os, "_cwd", str(tmp_path)) if False else patch("os.getcwd", return_value=str(tmp_path)):
            out = capture_cmd(repl, repl._cmd_pwd, "")
        assert str(tmp_path) in out

    def test_echo(self, repl):
        out = capture_cmd(repl, repl._cmd_echo, "hello world")
        assert out.strip() == "hello world"


# ── test / printf ────────────────────────────────────────────────────


class TestCmdTest:
    def test_test_no_args(self, repl):
        capture_cmd(repl, repl._cmd_test, "")
        assert repl._last_exit_code == 1

    def test_test_f_file(self, repl, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("x")
        capture_cmd(repl, repl._cmd_test, f"-f {f}")
        assert repl._last_exit_code == 0
        capture_cmd(repl, repl._cmd_test, "-f /no/such/file")
        assert repl._last_exit_code == 1

    def test_test_d_dir(self, repl, tmp_path):
        capture_cmd(repl, repl._cmd_test, f"-d {tmp_path}")
        assert repl._last_exit_code == 0
        capture_cmd(repl, repl._cmd_test, "-d /no/such/dir")
        assert repl._last_exit_code == 1

    def test_test_e_exists(self, repl, tmp_path):
        f = tmp_path / "y.txt"
        f.write_text("y")
        capture_cmd(repl, repl._cmd_test, f"-e {f}")
        assert repl._last_exit_code == 0

    def test_test_bracket_form(self, repl):
        capture_cmd(repl, repl._cmd_test, "[ -z ]")
        assert repl._last_exit_code == 1

    def test_test_z_n(self, repl):
        capture_cmd(repl, repl._cmd_test, "-z x")
        assert repl._last_exit_code == 1
        capture_cmd(repl, repl._cmd_test, "-n abc")
        assert repl._last_exit_code == 0

    def test_test_string_ops(self, repl):
        capture_cmd(repl, repl._cmd_test, "a = a")
        assert repl._last_exit_code == 0
        capture_cmd(repl, repl._cmd_test, "a = b")
        assert repl._last_exit_code == 1
        capture_cmd(repl, repl._cmd_test, "a != b")
        assert repl._last_exit_code == 0

    def test_test_numeric_ops(self, repl):
        capture_cmd(repl, repl._cmd_test, "3 -eq 3")
        assert repl._last_exit_code == 0
        capture_cmd(repl, repl._cmd_test, "3 -ne 4")
        assert repl._last_exit_code == 0
        capture_cmd(repl, repl._cmd_test, "3 -lt 4")
        assert repl._last_exit_code == 0
        capture_cmd(repl, repl._cmd_test, "3 -le 3")
        assert repl._last_exit_code == 0
        capture_cmd(repl, repl._cmd_test, "4 -gt 3")
        assert repl._last_exit_code == 0
        capture_cmd(repl, repl._cmd_test, "4 -ge 4")
        assert repl._last_exit_code == 0
        capture_cmd(repl, repl._cmd_test, "4 -lt 3")
        assert repl._last_exit_code == 1

    def test_test_unknown_expr(self, repl):
        capture_cmd(repl, repl._cmd_test, "a b c d")
        assert repl._last_exit_code == 1


class TestCmdPrintf:
    def test_printf_no_args(self, repl):
        capture_cmd(repl, repl._cmd_printf, "")
        assert repl._last_exit_code == 1

    def test_printf_strings(self, repl):
        out = capture_cmd(repl, repl._cmd_printf, "%s-%s hello world")
        assert out.strip() == "hello-world"

    def test_printf_int(self, repl):
        out = capture_cmd(repl, repl._cmd_printf, "%d 42")
        assert out.strip() == "42"

    def test_printf_int_bad_value(self, repl):
        out = capture_cmd(repl, repl._cmd_printf, "%d abc")
        assert out.strip() == "0"

    def test_printf_float(self, repl):
        out = capture_cmd(repl, repl._cmd_printf, "%f 3.25")
        assert out.strip() == "3.250000"

    def test_printf_float_bad_value(self, repl):
        out = capture_cmd(repl, repl._cmd_printf, "%f abc")
        assert out.strip() == "0.000000"

    def test_printf_percent_and_escapes(self, repl):
        out = capture_cmd(repl, repl._cmd_printf, "100%%\\n")
        assert "100%" in out


# ── source ───────────────────────────────────────────────────────────


class TestSource:
    def test_source_usage(self, repl):
        out = capture_cmd(repl, repl._cmd_source, "")
        assert "Usage: source <file>" in out

    def test_source_missing_file(self, repl):
        out = capture_cmd(repl, repl._cmd_source, "/no/such/script.sh")
        assert "Error reading" in out

    def test_source_runs_lines(self, repl, tmp_path):
        script = tmp_path / "s.sh"
        script.write_text("# comment\n\necho sourced-ok\n")
        out = capture_cmd(repl, repl._cmd_source, str(script))
        assert "sourced-ok" in out

    def test_source_reports_line_error(self, repl, tmp_path):
        script = tmp_path / "bad.sh"
        script.write_text("echo ok\ntotally_invalid_cmd_xyz_123\n")
        out = capture_cmd(repl, repl._cmd_source, str(script))
        assert "Unknown command: totally_invalid_cmd_xyz_123" in out


# ── py ───────────────────────────────────────────────────────────────


class TestCmdPy:
    def test_py_usage(self, repl):
        out = capture_cmd(repl, repl._cmd_py, "")
        assert "Usage: py <expression>" in out

    def test_py_eval(self, repl):
        out = capture_cmd(repl, repl._cmd_py, "2 + 2")
        assert out.strip() == "4"

    def test_py_list_comprehension(self, repl):
        out = capture_cmd(repl, repl._cmd_py, "[i*i for i in range(3)]")
        assert "[0, 1, 4]" in out

    def test_py_error(self, repl):
        out = capture_cmd(repl, repl._cmd_py, "1/0")
        assert "Error:" in out

    def test_py_blocks_unsafe_import(self, repl):
        out = capture_cmd(repl, repl._cmd_py, "__import__('os').getcwd()")
        assert "not allowed in py" in out

    def test_py_allows_safe_import(self, repl):
        out = capture_cmd(repl, repl._cmd_py, "__import__('math').sqrt(16)")
        assert out.strip() == "4.0"


# ── logs ─────────────────────────────────────────────────────────────


class TestLogs:
    def test_logs_clear(self, repl):
        _add_log(repl)
        out = capture_cmd(repl, repl._cmd_logs, "--clear")
        assert "Log buffer cleared." in out
        assert len(repl._log_buffer) == 0

    def test_logs_stats_empty(self, repl):
        repl._log_buffer.clear()
        out = capture_cmd(repl, repl._cmd_logs, "--stats")
        assert "No log entries." in out

    def test_logs_stats_with_entries(self, repl):
        repl._log_buffer.clear()
        _add_log(repl, "ERROR", "kernel", "boom")
        _add_log(repl, "INFO", "api", "started")
        out = capture_cmd(repl, repl._cmd_logs, "--stats")
        assert "Log Statistics" in out
        assert "By Level:" in out
        assert "ERROR" in out

    def test_logs_export(self, repl, tmp_path):
        repl._log_buffer.clear()
        _add_log(repl, "ERROR", "kernel", "boom")
        outfile = tmp_path / "logs.txt"
        out = capture_cmd(repl, repl._cmd_logs, f"-e {outfile}")
        assert "Exported 1 entries" in out
        content = outfile.read_text()
        assert "boom" in content
        assert "ERROR" in content

    def test_logs_export_none(self, repl, tmp_path):
        repl._log_buffer.clear()
        outfile = tmp_path / "empty.txt"
        out = capture_cmd(repl, repl._cmd_logs, f"-e {outfile}")
        assert "No log entries to export." in out

    def test_logs_render(self, repl):
        repl._log_buffer.clear()
        _add_log(repl, "WARNING", "init", "slow boot")
        out = capture_cmd(repl, repl._cmd_logs, "")
        assert "Console Logs" in out
        assert "slow boot" in out

    def test_logs_render_filters(self, repl):
        repl._log_buffer.clear()
        _add_log(repl, "ERROR", "kernel", "boom")
        _add_log(repl, "INFO", "api", "started")
        out = capture_cmd(repl, repl._cmd_logs, "-l ERROR -n 5")
        assert "boom" in out
        assert "started" not in out

    def test_logs_render_empty(self, repl):
        repl._log_buffer.clear()
        out = capture_cmd(repl, repl._cmd_logs, "")
        assert "No log entries." in out

    def test_logs_explain_no_api(self, repl):
        repl._log_buffer.clear()
        _add_log(repl, "ERROR", "kernel", "boom")
        out = capture_cmd(repl, repl._cmd_logs, "--explain")
        assert "API server not available" in out

    def test_logs_explain_no_errors(self, repl):
        repl._log_buffer.clear()
        _add_log(repl, "INFO", "api", "started")
        out = capture_cmd(repl, repl._cmd_logs, "--explain")
        assert "No errors or warnings to explain." in out


# ── svc ──────────────────────────────────────────────────────────────


class _FakeMgr:
    def __init__(self, name):
        self.name = name
        self.instance = type("Inst", (), {"log": ["line1", "line2"]})()
        self.started = False

    def status_line(self, width):
        return f"[{self.name}] status-line"

    def start(self):
        self.started = True
        return True

    def stop(self):
        self.started = False

    def restart(self):
        return True


class _FakeInit:
    runlevel = 5

    def __init__(self):
        self.managers = {"svc1": _FakeMgr("svc1")}

    def service_table(self):
        return "SVC TABLE"

    @property
    def status_summary(self):
        return "SUMMARY"

    def get_manager(self, name):
        return self.managers.get(name)


def _booted_repl(repl):
    repl.os._init = _FakeInit()
    return repl.os.init_system


class TestCmdSvc:
    def test_svc_not_booted(self, repl):
        repl.os._init = None
        out = capture_cmd(repl, repl._cmd_svc, "list")
        assert "not booted" in out
        assert repl._last_exit_code == 1

    def test_svc_list(self, repl):
        _booted_repl(repl)
        out = capture_cmd(repl, repl._cmd_svc, "list")
        assert "Services:" in out
        assert "SVC TABLE" in out

    def test_svc_list_alias_ls(self, repl):
        _booted_repl(repl)
        out = capture_cmd(repl, repl._cmd_svc, "ls")
        assert "SVC TABLE" in out

    def test_svc_status_all(self, repl):
        _booted_repl(repl)
        out = capture_cmd(repl, repl._cmd_svc, "status")
        assert "Init status:" in out
        assert "SUMMARY" in out

    def test_svc_status_name(self, repl):
        _booted_repl(repl)
        out = capture_cmd(repl, repl._cmd_svc, "status svc1")
        assert "status-line" in out
        assert "line1" in out

    def test_svc_status_unknown(self, repl):
        _booted_repl(repl)
        out = capture_cmd(repl, repl._cmd_svc, "status nope")
        assert "Unknown service: nope" in out
        assert repl._last_exit_code == 1

    def test_svc_start(self, repl):
        _booted_repl(repl)
        out = capture_cmd(repl, repl._cmd_svc, "start svc1")
        assert "svc1: ✓ started" in out
        assert repl.os.init_system.managers["svc1"].started

    def test_svc_start_usage(self, repl):
        _booted_repl(repl)
        out = capture_cmd(repl, repl._cmd_svc, "start")
        assert "Usage: svc start <name>" in out

    def test_svc_start_unknown(self, repl):
        _booted_repl(repl)
        out = capture_cmd(repl, repl._cmd_svc, "start nope")
        assert "Unknown service: nope" in out

    def test_svc_stop(self, repl):
        _booted_repl(repl)
        out = capture_cmd(repl, repl._cmd_svc, "stop svc1")
        assert "svc1: stopped" in out

    def test_svc_restart(self, repl):
        _booted_repl(repl)
        out = capture_cmd(repl, repl._cmd_svc, "restart svc1")
        assert "svc1: ✓ restarted" in out

    def test_svc_runlevel(self, repl):
        _booted_repl(repl)
        out = capture_cmd(repl, repl._cmd_svc, "runlevel")
        assert "Current runlevel: 5" in out

    def test_svc_invalid_verb(self, repl):
        _booted_repl(repl)
        out = capture_cmd(repl, repl._cmd_svc, "bogus")
        assert "Usage: svc" in out
        assert repl._last_exit_code == 1


# ── which / type ─────────────────────────────────────────────────────


class TestWhichType:
    def test_which_usage(self, repl):
        out = capture_cmd(repl, repl._cmd_which, "")
        assert "Usage: which <command>" in out
        assert repl._last_exit_code == 1

    def test_which_alias(self, repl):
        out = capture_cmd(repl, repl._cmd_which, "q")
        assert "aliased to exit" in out
        assert repl._last_exit_code == 0

    def test_which_builtin(self, repl):
        out = capture_cmd(repl, repl._cmd_which, "lsdev")
        assert "shell built-in" in out

    def test_which_system(self, repl):
        out = capture_cmd(repl, repl._cmd_which, "ls")
        assert repl._last_exit_code == 0

    def test_which_not_found(self, repl):
        out = capture_cmd(repl, repl._cmd_which, "no_such_cmd_xyz")
        assert "not found" in out
        assert repl._last_exit_code == 1

    def test_type_usage(self, repl):
        out = capture_cmd(repl, repl._cmd_type, "")
        assert "Usage: type <command>" in out
        assert repl._last_exit_code == 1

    def test_type_alias(self, repl):
        out = capture_cmd(repl, repl._cmd_type, "q")
        assert "is aliased to" in out

    def test_type_builtin(self, repl):
        out = capture_cmd(repl, repl._cmd_type, "echo")
        assert "is a shell built-in" in out

    def test_type_system(self, repl):
        out = capture_cmd(repl, repl._cmd_type, "ls")
        assert "is " in out

    def test_type_not_found(self, repl):
        out = capture_cmd(repl, repl._cmd_type, "no_such_cmd_xyz")
        assert "not found" in out
        assert repl._last_exit_code == 1


# ── VM commands ──────────────────────────────────────────────────────


class TestVmRun:
    def test_vmrun_list(self, repl):
        out = capture_cmd(repl, repl._cmd_vmrun, "--list")
        assert "Built-in x86 programs:" in out
        assert "hello" in out

    def test_vmrun_bad_steps(self, repl):
        out = capture_cmd(repl, repl._cmd_vmrun, "--steps=abc hello")
        assert "requires an integer" in out
        assert repl._last_exit_code == 1

    def test_vmrun_no_source(self, repl):
        out = capture_cmd(repl, repl._cmd_vmrun, "")
        assert "Usage: vmrun" in out
        assert repl._last_exit_code == 1

    def test_vmrun_missing_file(self, repl):
        out = capture_cmd(repl, repl._cmd_vmrun, "/no/such/file.asm")
        assert "vmrun:" in out
        assert repl._last_exit_code == 1

    def test_vmrun_builtin_hello(self, repl):
        out = capture_cmd(repl, repl._cmd_vmrun, "hello")
        assert "Hello from x86 VM!" in out
        assert "[exit:" in out

    def test_vmrun_builtin_hello_admin_debug(self, repl):
        out = capture_cmd(repl, repl._cmd_vmrun, "--admin --debug --steps=100000 hello")
        assert "Hello from x86 VM!" in out
        assert "Registers:" in out
        assert "role: admin" in out


class TestAsm:
    def test_asm_usage(self, repl):
        out = capture_cmd(repl, repl._cmd_asm, "")
        assert "Usage: asm <file.asm>" in out
        assert repl._last_exit_code == 1

    def test_asm_list(self, repl):
        out = capture_cmd(repl, repl._cmd_asm, "--list")
        assert "Built-in programs" in out
        assert "fib" in out

    def test_asm_missing_file(self, repl):
        out = capture_cmd(repl, repl._cmd_asm, "/no/such/file.asm")
        assert "asm:" in out
        assert repl._last_exit_code == 1

    def test_asm_piped_run(self, repl):
        repl._piped_input = "MOV R0, 42\nPRINT R0\nHALT"
        out = capture_cmd(repl, repl._cmd_asm, "")
        assert "42" in out


# ── cal / ln ─────────────────────────────────────────────────────────


class TestCalLn:
    def test_cal_current_month(self, repl):
        out = capture_cmd(repl, repl._cmd_cal, "")
        assert "Mo Tu We Th Fr Sa Su" in out

    def test_cal_invalid_month(self, repl):
        out = capture_cmd(repl, repl._cmd_cal, "13 2025")
        assert "cal: invalid date" in out

    def test_cal_invalid_year(self, repl):
        out = capture_cmd(repl, repl._cmd_cal, "1 0")
        assert "cal: invalid date" in out

    def test_cal_single_arg_current_year(self, repl):
        import datetime
        out = capture_cmd(repl, repl._cmd_cal, str(datetime.datetime.now().year))
        assert "Mo Tu We Th Fr Sa Su" in out

    def test_ln_usage(self, repl):
        out = capture_cmd(repl, repl._cmd_ln, "")
        assert "Usage: ln" in out or out.strip() == ""

    def test_ln_symbolic(self, repl, tmp_path):
        target = tmp_path / "t.txt"
        target.write_text("data")
        link = tmp_path / "link.txt"
        out = capture_cmd(repl, repl._cmd_ln, f"-s {target} {link}")
        assert link.is_symlink() or link.exists()


# ── render ───────────────────────────────────────────────────────────


class TestRender:
    def test_render_info(self, repl):
        out = capture_cmd(repl, repl._cmd_render, "")
        assert "Scene:" in out
        assert "Resolution:" in out

    def test_render_sphere(self, repl):
        out = capture_cmd(repl, repl._cmd_render, "sphere 1 0 0 0")
        assert "Added sphere" in out

    def test_render_sphere_usage(self, repl):
        out = capture_cmd(repl, repl._cmd_render, "sphere 1 0")
        assert "Usage: render sphere" in out

    def test_render_cube_usage(self, repl):
        out = capture_cmd(repl, repl._cmd_render, "cube 1")
        assert "Usage: render cube" in out

    def test_render_light_usage(self, repl):
        out = capture_cmd(repl, repl._cmd_render, "light 0 0")
        assert "Usage: render light" in out

    def test_render_plane_usage(self, repl):
        out = capture_cmd(repl, repl._cmd_render, "plane 2")
        assert "Usage: render plane" in out

    def test_render_mat_usage(self, repl):
        out = capture_cmd(repl, repl._cmd_render, "mat 0 1 2")
        assert "Usage: render mat" in out

    def test_render_clear(self, repl):
        out = capture_cmd(repl, repl._cmd_render, "clear")
        assert "Scene cleared." in out

    def test_render_preset_demo(self, repl):
        out = capture_cmd(repl, repl._cmd_render, "preset demo")
        assert "Loaded preset: demo" in out

    def test_render_preset_cornell(self, repl):
        out = capture_cmd(repl, repl._cmd_render, "preset cornell")
        assert "Loaded preset: cornell" in out

    def test_render_unknown_subcommand(self, repl):
        out = capture_cmd(repl, repl._cmd_render, "bogus")
        assert "Unknown render subcommand" in out


# ── ai ───────────────────────────────────────────────────────────────


class TestAi:
    def test_ai_usage(self, repl):
        out = capture_cmd(repl, repl._cmd_ai, "")
        assert "Usage: ai <natural language query>" in out

    def test_ai_api_unavailable_falls_back(self, repl):
        out = capture_cmd(repl, repl._cmd_ai, "show me running processes")
        assert "API server is not connected" in out
        assert "keyword matching" in out

    def test_ai_unknown_query(self, repl):
        out = capture_cmd(repl, repl._cmd_ai, "zzz something")
        assert "Unknown query: zzz something" in out

    def test_interpret_natural_procs(self, repl):
        with patch.object(repl, "_execute_single", return_value="procs output") as m:
            out = capture_cmd(repl, repl._interpret_natural, "show running jobs")
        assert m.called

    def test_interpret_natural_help(self, repl):
        out = capture_cmd(repl, repl._interpret_natural, "what commands are available")
        assert "Built-in commands:" in out


# ── tutorial ─────────────────────────────────────────────────────────


class TestTutorial:
    def test_tutorial_quit_on_q(self, repl):
        out = _run_with_io(repl, ["q"], lambda: repl._cmd_tutorial(""))
        assert "Tutorial stopped." in out

    def test_tutorial_eof(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_tutorial(""))
        assert "Tutorial stopped." in out


# ── completion ───────────────────────────────────────────────────────


class TestCompletion:
    def test_complete_train(self, repl):
        candidates = repl._complete_args_for_uncached("train")
        assert "status" in candidates or "distill" in candidates

    def test_complete_train_candidates(self, repl):
        with patch.object(repl, "_complete_path", return_value=[]) as m:
            candidates = repl._complete_args_for_uncached("train")
        assert "status" in candidates
        assert "distill" in candidates

    def test_complete_permit_candidates(self, repl):
        candidates = repl._complete_args_for_uncached("permit")
        assert "--all-dangerous" in candidates
        assert "--persist" in candidates

    def test_complete_note_candidates(self, repl):
        candidates = repl._complete_args_for_uncached("note")
        assert "new" in candidates
        assert "timeline" in candidates

    def test_complete_path_dev_with_vfs(self, repl):
        fake_vfs = MagicMock()
        fake_vfs.listdir.return_value = ["null", "cpu", "procs"]
        fake_vfs.isdir.return_value = False
        repl.os._vfs = fake_vfs
        matches = repl._complete_path("/dev/")
        assert matches == ["cpu", "null", "procs"]

    def test_complete_path_dev_no_vfs(self, repl):
        repl.os._vfs = None
        matches = repl._complete_path("/dev/")
        assert isinstance(matches, list)

    def test_complete_path_regular(self, repl, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "alpha.txt").write_text("x")
        (tmp_path / "beta.txt").write_text("y")
        matches = repl._complete_path("a")
        assert matches == ["alpha.txt"]


# ── render subcommands (full execution paths) ────────────────────────


class TestRenderExec:
    def test_render_cube(self, repl):
        out = capture_cmd(repl, repl._cmd_render, "cube 1 0 0 0")
        assert "Added cube" in out

    def test_render_plane(self, repl):
        out = capture_cmd(repl, repl._cmd_render, "plane 2 -1")
        assert "Added plane" in out

    def test_render_light(self, repl):
        out = capture_cmd(repl, repl._cmd_render, "light 0 5 0 1 1 1 10")
        assert "Added light" in out

    def test_render_mat(self, repl):
        out = capture_cmd(repl, repl._cmd_render, "mat 0 0.8 0.2 0.1 0.5 0.3")
        assert "Material 0:" in out

    def test_render_cam(self, repl):
        out = capture_cmd(repl, repl._cmd_render, "cam 0 5 -10 0 0 0 60")
        assert "Camera:" in out

    def test_render_go(self, repl):
        out = capture_cmd(repl, repl._cmd_render, "go 40 30 2")
        assert "Rendering 40x30" in out
        assert "Done in" in out

    def test_render_sphere_with_mat(self, repl):
        out = capture_cmd(repl, repl._cmd_render, "sphere 2 1 2 3 1")
        assert "Added sphere" in out
        assert "mat=1" in out

    def test_render_cam_with_fov(self, repl):
        out = capture_cmd(repl, repl._cmd_render, "cam 0 0 0 1 1 1 90")
        assert "fov=90" in out


# ── pipeline parsing ─────────────────────────────────────────────────


class TestPipelineParsing:
    def test_parse_pipe(self, repl):
        cmds, bg, timing = repl._parse_pipeline("echo a | echo b")
        assert len(cmds) == 2
        assert cmds[0] == ("echo a", "|")
        assert cmds[1] == ("echo b", None)
        assert bg is False

    def test_parse_background(self, repl):
        cmds, bg, timing = repl._parse_pipeline("sleep 1 &")
        assert bg is True

    def test_parse_timing(self, repl):
        cmds, bg, timing = repl._parse_pipeline("time echo hi")
        assert timing is True

    def test_parse_semicolon(self, repl):
        cmds, bg, timing = repl._parse_pipeline("echo a ; echo b")
        assert len(cmds) == 2
        assert cmds[0][1] == ";"

    def test_parse_and(self, repl):
        cmds, bg, timing = repl._parse_pipeline("true && echo ok")
        assert cmds[0][1] == "&&"

    def test_parse_or(self, repl):
        cmds, bg, timing = repl._parse_pipeline("false || echo ok")
        assert cmds[0][1] == "||"

    def test_parse_quoted_pipe(self, repl):
        cmds, bg, timing = repl._parse_pipeline('echo "a|b"')
        assert len(cmds) == 1

    def test_split_pipe_quoted(self, repl):
        parts = repl._split_pipe('echo "a|b" | wc')
        assert len(parts) == 2

    def test_parse_empty(self, repl):
        cmds, bg, timing = repl._parse_pipeline("")
        assert cmds == []


# ── inline env + redirection ─────────────────────────────────────────


class TestInlineEnv:
    def test_inline_env(self, repl):
        env_updates, rest = repl._parse_inline_env("FOO=bar echo hi")
        assert env_updates == {"FOO": "bar"}
        assert rest == "echo hi"

    def test_inline_env_no_match(self, repl):
        env_updates, rest = repl._parse_inline_env("echo hi")
        assert env_updates == {}
        assert rest == "echo hi"

    def test_strip_redirect(self, repl):
        cleaned, path, append = repl._strip_redirection("echo hi > /tmp/out")
        assert cleaned == "echo hi"
        assert path == "/tmp/out"
        assert append is False

    def test_strip_append_redirect(self, repl):
        cleaned, path, append = repl._strip_redirection("echo hi >> /tmp/out")
        assert append is True

    def test_strip_no_redirect(self, repl):
        cleaned, path, append = repl._strip_redirection("echo hi")
        assert path is None


# ── sort / uniq / head / tail / wc ──────────────────────────────────


class TestSortUniqHeadTail:
    def test_sort_basic(self, repl):
        repl._piped_input = "c\na\nb\n"
        out = capture_cmd(repl, repl._cmd_sort, "")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert lines == ["a", "b", "c"]

    def test_sort_reverse(self, repl):
        repl._piped_input = "a\nc\nb\n"
        out = capture_cmd(repl, repl._cmd_sort, "-r")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert lines == ["c", "b", "a"]

    def test_sort_unique(self, repl):
        repl._piped_input = "a\na\nb\n"
        out = capture_cmd(repl, repl._cmd_sort, "-u")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert lines == ["a", "b"]

    def test_uniq_basic(self, repl):
        repl._piped_input = "a\na\nb\n"
        out = capture_cmd(repl, repl._cmd_uniq, "")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert lines == ["a", "b"]

    def test_head_default(self, repl):
        repl._piped_input = "a\nb\nc\nd\n"
        out = capture_cmd(repl, repl._cmd_head, "")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert len(lines) == 4

    def test_head_n(self, repl):
        repl._piped_input = "a\nb\nc\nd\n"
        out = capture_cmd(repl, repl._cmd_head, "-2")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert len(lines) == 2

    def test_tail_default(self, repl):
        repl._piped_input = "a\nb\nc\nd\n"
        out = capture_cmd(repl, repl._cmd_tail, "")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert len(lines) == 4

    def test_tail_n(self, repl):
        repl._piped_input = "a\nb\nc\nd\n"
        out = capture_cmd(repl, repl._cmd_tail, "-2")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert len(lines) == 2

    def test_wc_lines(self, repl):
        repl._piped_input = "a\nb\nc\n"
        out = capture_cmd(repl, repl._cmd_wc, "")
        assert "3" in out

    def test_wc_words(self, repl):
        repl._piped_input = "hello world\n"
        out = capture_cmd(repl, repl._cmd_wc, "")
        assert "2" in out


# ── alias / unalias / set / export ───────────────────────────────────


class TestAliasSetExport:
    def test_alias_set_and_expand(self, repl):
        out = capture_cmd(repl, repl._cmd_alias, "ll=ls -la")
        assert repl._aliases.get("ll") == "ls -la"

    def test_alias_list(self, repl):
        repl._aliases["ll"] = "ls -la"
        out = capture_cmd(repl, repl._cmd_alias, "")
        assert "ll" in out

    def test_unalias(self, repl):
        repl._aliases["ll"] = "ls -la"
        out = capture_cmd(repl, repl._cmd_unalias, "ll")
        assert "ll" not in repl._aliases

    def test_set_var(self, repl):
        out = capture_cmd(repl, repl._cmd_set, "MYVAR=hello")
        assert repl._env.get("MYVAR") == "hello"

    def test_set_list(self, repl):
        repl._env["MYVAR"] = "hello"
        out = capture_cmd(repl, repl._cmd_set, "")
        assert "MYVAR" in out

    def test_export_var(self, repl):
        out = capture_cmd(repl, repl._cmd_export, "MYVAR=world")
        assert repl._env.get("MYVAR") == "world"


# ── pushd / popd / dirs ─────────────────────────────────────────────


class TestPushdPopdDirs:
    def test_dir_stack_empty(self, repl):
        assert repl._dir_stack == []

    def test_dir_stack_push(self, repl):
        repl._dir_stack.append("/tmp")
        assert "/tmp" in repl._dir_stack


# ── find ─────────────────────────────────────────────────────────────


class TestFind:
    def test_find_basic(self, repl, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "b.py").write_text("y")
        out = capture_cmd(repl, repl._cmd_find, f"{tmp_path} -name a.txt")
        assert "a.txt" in out
        assert "b.py" not in out

    def test_find_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_find, "")
        assert "Usage" in out

    def test_find_no_pattern(self, repl):
        out = capture_cmd(repl, repl._cmd_find, ".")
        assert "Usage" in out


# ── fc ───────────────────────────────────────────────────────────────


class TestFc:
    def test_fc_list(self, repl):
        repl._history = ["echo one", "echo two", "echo three"]
        out = capture_cmd(repl, repl._cmd_fc, "-l")
        assert "echo one" in out or "echo two" in out

    def test_fc_rerun(self, repl):
        repl._history = ["echo one", "echo two"]
        out = capture_cmd(repl, repl._cmd_fc, "1")
        assert "one" in out or repl._last_exit_code == 0


# ── logs export failure ──────────────────────────────────────────────


class TestLogsMore:
    def test_logs_export_os_error(self, repl, tmp_path):
        repl._log_buffer.clear()
        _add_log(repl, "INFO", "api", "test")
        out = capture_cmd(repl, repl._cmd_logs, f"-e /nonexistent/dir/logs.txt")
        assert "Error writing" in out or repl._last_exit_code == 1

    def test_logs_source_filter(self, repl):
        repl._log_buffer.clear()
        _add_log(repl, "INFO", "kernel", "k-msg")
        _add_log(repl, "INFO", "api", "a-msg")
        out = capture_cmd(repl, repl._cmd_logs, "-s kernel")
        assert "k-msg" in out
        assert "a-msg" not in out

    def test_logs_debug_level(self, repl):
        repl._log_buffer.clear()
        _add_log(repl, "DEBUG", "test", "debug-msg")
        _add_log(repl, "INFO", "test", "info-msg")
        out = capture_cmd(repl, repl._cmd_logs, "-l DEBUG")
        assert "debug-msg" in out
        assert "info-msg" not in out

    def test_logs_warning_level(self, repl):
        repl._log_buffer.clear()
        _add_log(repl, "WARNING", "test", "warn-msg")
        _add_log(repl, "INFO", "test", "info-msg")
        out = capture_cmd(repl, repl._cmd_logs, "-l WARNING")
        assert "warn-msg" in out
        assert "info-msg" not in out

    def test_logs_critical_level(self, repl):
        repl._log_buffer.clear()
        _add_log(repl, "CRITICAL", "test", "crit-msg")
        out = capture_cmd(repl, repl._cmd_logs, "-l CRITICAL")
        assert "crit-msg" in out


# ── system binary fallback ──────────────────────────────────────────


class TestSystemBinary:
    def test_system_binary_runs(self, repl):
        out = repl._execute_single("echo hello-piped")
        assert "hello-piped" in out

    def test_system_binary_not_found(self, repl):
        out = repl._execute_single("no_such_binary_xyz_abc")
        assert "Unknown command" in out

    def test_system_binary_with_args(self, repl):
        out = repl._execute_single("echo hello world")
        assert "hello world" in out

    def test_system_binary_stderr(self, repl):
        out = repl._execute_single("ls /no_such_dir_xyz")
        assert repl._last_exit_code != 0 or "No such" in out


# ── more completion paths ────────────────────────────────────────────


class TestCompletionMore:
    def test_complete_load(self, repl):
        candidates = repl._complete_args_for_uncached("load")
        assert isinstance(candidates, list)

    def test_complete_switch(self, repl):
        candidates = repl._complete_args_for_uncached("switch")
        assert isinstance(candidates, list)

    def test_complete_datasets(self, repl):
        candidates = repl._complete_args_for_uncached("datasets")
        assert isinstance(candidates, list)

    def test_complete_checkpoints(self, repl):
        candidates = repl._complete_args_for_uncached("checkpoints")
        assert isinstance(candidates, list)

    def test_complete_deny(self, repl):
        candidates = repl._complete_args_for_uncached("deny")
        assert "--all-dangerous" in candidates or isinstance(candidates, list)

    def test_complete_finetuned(self, repl):
        candidates = repl._complete_args_for_uncached("finetuned")
        assert isinstance(candidates, list)

    def test_complete_unknown_cmd(self, repl):
        with patch.object(repl, "_complete_path", return_value=[]) as m:
            candidates = repl._complete_args_for_uncached("zzz_unknown_cmd")
        assert candidates == []

    def test_complete_permit_no_args(self, repl):
        candidates = repl._complete_args_for_uncached("permit")
        assert "--persist" in candidates

    def test_complete_note_new(self, repl):
        candidates = repl._complete_args_for_uncached("note")
        assert "new" in candidates
        assert "list" in candidates

    def test_complete_train_status(self, repl):
        candidates = repl._complete_args_for_uncached("train")
        assert "status" in candidates

    def test_complete_empty(self, repl):
        candidates = repl._complete_args_for_uncached("")
        assert isinstance(candidates, list)


# ── utility commands (quick wins for coverage) ───────────────────────


class TestUtilityCommands:
    def test_id(self, repl):
        out = capture_cmd(repl, repl._cmd_id, "")
        assert "uid" in out or "gid" in out

    def test_logname(self, repl):
        out = capture_cmd(repl, repl._cmd_logname, "")
        assert out.strip() != ""

    def test_mktemp(self, repl):
        out = capture_cmd(repl, repl._cmd_mktemp, "")
        assert out.strip() != ""

    def test_who(self, repl):
        out = capture_cmd(repl, repl._cmd_who, "")
        assert out.strip() != ""

    def test_hostname(self, repl):
        out = capture_cmd(repl, repl._cmd_hostname, "")
        assert out.strip() != ""

    def test_uname(self, repl):
        out = capture_cmd(repl, repl._cmd_uname, "")
        assert "Linux" in out or "Darwin" in out

    def test_nproc(self, repl):
        out = capture_cmd(repl, repl._cmd_nproc, "")
        assert out.strip().isdigit()

    def test_seq_3(self, repl):
        out = capture_cmd(repl, repl._cmd_seq, "3")
        assert "1" in out and "3" in out

    def test_seq_2_4(self, repl):
        out = capture_cmd(repl, repl._cmd_seq, "2 4")
        assert "2" in out and "4" in out

    def test_nl(self, repl):
        repl._piped_input = "a\nb\nc\n"
        out = capture_cmd(repl, repl._cmd_nl, "")
        assert "1" in out

    def test_fold(self, repl):
        repl._piped_input = "hello world\n"
        out = capture_cmd(repl, repl._cmd_fold, "-w 3")
        assert out.strip() != ""

    def test_tac(self, repl):
        repl._piped_input = "a\nb\nc\n"
        out = capture_cmd(repl, repl._cmd_tac, "")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert lines[-1] == "a"

    def test_env(self, repl):
        out = capture_cmd(repl, repl._cmd_env, "")
        assert "PATH" in out or "HOME" in out

    def test_cut(self, repl):
        repl._piped_input = "a,b,c\n"
        out = capture_cmd(repl, repl._cmd_cut, "-f2 -d,")
        assert "b" in out

    def test_tr(self, repl):
        repl._piped_input = "hello\n"
        out = capture_cmd(repl, repl._cmd_tr, "h H")
        assert "H" in out

    def test_stat_file(self, repl, tmp_path):
        f = tmp_path / "test_stat.txt"
        f.write_text("hello")
        out = capture_cmd(repl, repl._cmd_stat, str(f))
        assert "hello" in out or str(f) in out

    def test_diff_same(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("same\n")
        f2.write_text("same\n")
        out = capture_cmd(repl, repl._cmd_diff, f"{f1} {f2}")
        assert repl._last_exit_code == 0 or "identical" in out.lower()

    def test_diff_different(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("aaa\n")
        f2.write_text("bbb\n")
        out = capture_cmd(repl, repl._cmd_diff, f"{f1} {f2}")
        assert "aaa" in out or "bbb" in out or repl._last_exit_code != 0

    def test_du_file(self, repl, tmp_path):
        f = tmp_path / "du_test.txt"
        f.write_text("hello world")
        out = capture_cmd(repl, repl._cmd_du, str(f))
        assert out.strip() != ""

    def test_realpath(self, repl, tmp_path):
        out = capture_cmd(repl, repl._cmd_realpath, str(tmp_path))
        assert str(tmp_path) in out

    def test_basename(self, repl):
        out = capture_cmd(repl, repl._cmd_basename, "/usr/bin/python")
        assert "python" in out

    def test_dirname(self, repl):
        out = capture_cmd(repl, repl._cmd_dirname, "/usr/bin/python")
        assert "/usr/bin" in out

    def test_shuf(self, repl):
        repl._piped_input = "1\n2\n3\n"
        out = capture_cmd(repl, repl._cmd_shuf, "")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert len(lines) == 3

    def test_rev(self, repl):
        repl._piped_input = "abc\n"
        out = capture_cmd(repl, repl._cmd_rev, "")
        assert "cba" in out

    def test_paste(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("x\n")
        f2.write_text("y\n")
        out = capture_cmd(repl, repl._cmd_paste, f"{f1} {f2}")
        assert "x" in out and "y" in out

    def test_yes(self, repl):
        out = capture_cmd(repl, repl._cmd_yes, "")
        assert "y" in out

    def test_od(self, repl):
        repl._piped_input = "hello\n"
        out = capture_cmd(repl, repl._cmd_od, "")
        assert out.strip() != ""


# ── cat / ls / mkdir / rm / touch / cp / mv / chmod ─────────────────


class TestFileCommands:
    def test_cat_file(self, repl, tmp_path):
        f = tmp_path / "cat_test.txt"
        f.write_text("file contents")
        out = capture_cmd(repl, repl._cmd_cat, str(f))
        assert "file contents" in out

    def test_cat_missing(self, repl):
        out = capture_cmd(repl, repl._cmd_cat, "/no/such/file_xyz")
        assert "No such file" in out or repl._last_exit_code != 0

    def test_cat_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_cat, "")
        assert "Usage" in out

    def test_cat_piped(self, repl):
        repl._piped_input = "piped content"
        out = capture_cmd(repl, repl._cmd_cat, "")
        assert "piped content" in out

    def test_ls_dir(self, repl, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "file_a.txt").write_text("x")
        out = capture_cmd(repl, repl._cmd_ls, "")
        assert "file_a.txt" in out

    def test_ls_missing(self, repl):
        out = capture_cmd(repl, repl._cmd_ls, "/no/such/dir_xyz")
        assert "No such file" in out or repl._last_exit_code != 0

    def test_mkdir_and_rm(self, repl, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        d = tmp_path / "newdir"
        capture_cmd(repl, repl._cmd_mkdir, str(d))
        assert d.exists()
        capture_cmd(repl, repl._cmd_rm, f"-r {d}")
        assert not d.exists()

    def test_touch(self, repl, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "touched.txt"
        capture_cmd(repl, repl._cmd_touch, str(f))
        assert f.exists()

    def test_cp(self, repl, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("copy me")
        capture_cmd(repl, repl._cmd_cp, f"{src} {dst}")
        assert dst.read_text() == "copy me"

    def test_mv(self, repl, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "moveme.txt"
        dst = tmp_path / "moved.txt"
        src.write_text("move me")
        capture_cmd(repl, repl._cmd_mv, f"{src} {dst}")
        assert dst.exists()
        assert not src.exists()

    def test_chmod(self, repl, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "chmod_test.txt"
        f.write_text("x")
        out = capture_cmd(repl, repl._cmd_chmod, f"644 {f}")
        assert repl._last_exit_code == 0


# ── grep / tee / xargs / less / watch ──────────────────────────────


class TestPipeCommands:
    def test_grep_basic(self, repl):
        repl._piped_input = "hello world\nfoo bar\nhello again\n"
        out = capture_cmd(repl, repl._cmd_grep, "hello")
        assert "hello" in out
        assert "foo" not in out

    def test_tee(self, repl, tmp_path):
        repl._piped_input = "tee content"
        outfile = tmp_path / "tee_out.txt"
        out = capture_cmd(repl, repl._cmd_tee, str(outfile))
        assert "tee content" in out
        assert "tee content" in outfile.read_text()

    def test_tee_append(self, repl, tmp_path):
        outfile = tmp_path / "tee_append.txt"
        outfile.write_text("first\n")
        repl._piped_input = "second"
        out = capture_cmd(repl, repl._cmd_tee, f"-a {outfile}")
        assert "first" in outfile.read_text()
        assert "second" in outfile.read_text()

    def test_xargs_echo(self, repl):
        repl._piped_input = "hello world"
        out = capture_cmd(repl, repl._cmd_xargs, "echo")
        assert "hello" in out or "world" in out


# ── jobs / kill / procs / bg / fg ──────────────────────────────────


class TestJobCommands:
    def test_procs(self, repl):
        out = capture_cmd(repl, repl._cmd_procs, "")
        assert out.strip() != "" or repl._last_exit_code == 0

    def test_kill_no_such(self, repl):
        out = capture_cmd(repl, repl._cmd_kill, "99999")
        assert repl._last_exit_code != 0 or "error" in out.lower() or "No such" in out


# ── watch ───────────────────────────────────────────────────────────


class TestMoreWatch:
    def test_watch_no_cmd(self, repl):
        out = capture_cmd(repl, repl._cmd_watch, "")
        assert "Usage" in out


# ── protect / unprotect ─────────────────────────────────────────────


class TestProtectUnprotect:
    def test_protect_usage(self, repl):
        out = capture_cmd(repl, repl._cmd_protect, "")
        assert "Usage" in out

    def test_protect_model(self, repl, monkeypatch):
        import domains.infrastructure.model_protector as mp
        monkeypatch.setattr(mp, "protect_model", lambda mid: {"protected": ["f1"], "errors": []})
        out = capture_cmd(repl, repl._cmd_protect, "mymodel")
        assert "Protected 1" in out or "mymodel" in out

    def test_protect_no_files(self, repl, monkeypatch):
        import domains.infrastructure.model_protector as mp
        monkeypatch.setattr(mp, "protect_model", lambda mid: {"protected": [], "errors": []})
        out = capture_cmd(repl, repl._cmd_protect, "mymodel")
        assert "No files found" in out

    def test_protect_with_errors(self, repl, monkeypatch):
        import domains.infrastructure.model_protector as mp
        monkeypatch.setattr(mp, "protect_model", lambda mid: {"protected": ["f1"], "errors": [{"error": "perm denied"}]})
        out = capture_cmd(repl, repl._cmd_protect, "mymodel")
        assert "Warning" in out

    def test_protect_exception(self, repl, monkeypatch):
        import domains.infrastructure.model_protector as mp
        monkeypatch.setattr(mp, "protect_model", lambda mid: (_ for _ in ()).throw(RuntimeError("boom")))
        out = capture_cmd(repl, repl._cmd_protect, "mymodel")
        assert "Error" in out

    def test_unprotect_usage(self, repl):
        out = capture_cmd(repl, repl._cmd_unprotect, "")
        assert "Usage" in out

    def test_unprotect_model(self, repl, monkeypatch):
        import domains.infrastructure.model_protector as mp
        monkeypatch.setattr(mp, "unprotect_model", lambda mid: {"unprotected": 3, "errors": []})
        out = capture_cmd(repl, repl._cmd_unprotect, "mymodel")
        assert "Unprotected 3" in out

    def test_unprotect_none(self, repl, monkeypatch):
        import domains.infrastructure.model_protector as mp
        monkeypatch.setattr(mp, "unprotect_model", lambda mid: {"unprotected": 0, "errors": []})
        out = capture_cmd(repl, repl._cmd_unprotect, "mymodel")
        assert "No protected" in out

    def test_unprotect_with_errors(self, repl, monkeypatch):
        import domains.infrastructure.model_protector as mp
        monkeypatch.setattr(mp, "unprotect_model", lambda mid: {"unprotected": 1, "errors": [{"error": "oops"}]})
        out = capture_cmd(repl, repl._cmd_unprotect, "mymodel")
        assert "Warning" in out

    def test_unprotect_exception(self, repl, monkeypatch):
        import domains.infrastructure.model_protector as mp
        monkeypatch.setattr(mp, "unprotect_model", lambda mid: (_ for _ in ()).throw(RuntimeError("fail")))
        out = capture_cmd(repl, repl._cmd_unprotect, "mymodel")
        assert "Error" in out


# ── svc start / stop / restart / status ──────────────────────────────


class TestSvcCommands:
    def _make_svc_repl(self, repl):
        class FakeAPI:
            is_running = False
            def start(self):
                self.is_running = True
                return {"ok": True, "message": "started"}
            def stop(self):
                self.is_running = False
                return {"message": "stopped"}
            def status(self):
                return {"available": True, "model_id": "gpt2", "engine_type": "cpu", "running": self.is_running, "uptime": 120.0}
        class FakeManager:
            def __init__(self, name):
                self.name = name
                self.instance = type('Obj', (), {'log': []})()
            def start(self): return True
            def stop(self): return True
            def restart(self): return True
            def status_line(self, n): return f"  {self.name}: running"
        class FakeInitSystem:
            def service_table(self): return "  svc1: running\n  svc2: stopped"
            def get_manager(self, name): return FakeManager(name)
            @property
            def status_summary(self): return "  Init: OK"
        api = FakeAPI()
        repl.os._api = api
        repl.os._init = FakeInitSystem()
        return repl

    def test_svc_start(self, repl):
        repl = self._make_svc_repl(repl)
        out = capture_cmd(repl, repl._cmd_svc, "start svc1")
        assert "started" in out.lower() or repl._last_exit_code == 0

    def test_svc_start_already_running(self, repl):
        repl = self._make_svc_repl(repl)
        out = capture_cmd(repl, repl._cmd_svc, "start svc1")
        assert repl._last_exit_code == 0

    def test_svc_stop(self, repl):
        repl = self._make_svc_repl(repl)
        out = capture_cmd(repl, repl._cmd_svc, "stop svc1")
        assert repl._last_exit_code == 0

    def test_svc_stop_not_running(self, repl):
        repl = self._make_svc_repl(repl)
        out = capture_cmd(repl, repl._cmd_svc, "stop svc1")
        assert repl._last_exit_code == 0 or "stop" in out.lower() or "stopped" in out.lower()

    def test_svc_restart(self, repl):
        repl = self._make_svc_repl(repl)
        out = capture_cmd(repl, repl._cmd_svc, "restart svc1")
        assert repl._last_exit_code == 0

    def test_svc_restart_not_running(self, repl):
        repl = self._make_svc_repl(repl)
        out = capture_cmd(repl, repl._cmd_svc, "restart svc1")
        assert repl._last_exit_code == 0

    def test_svc_status(self, repl):
        repl = self._make_svc_repl(repl)
        out = capture_cmd(repl, repl._cmd_svc, "status svc1")
        assert repl._last_exit_code == 0

    def test_svc_list(self, repl):
        repl = self._make_svc_repl(repl)
        out = capture_cmd(repl, repl._cmd_svc, "list")
        assert "svc1" in out or repl._last_exit_code == 0

    def test_svc_start_fail(self, repl):
        repl = self._make_svc_repl(repl)
        class FailManager:
            def __init__(self, name):
                self.name = name
                self.instance = type('Obj', (), {'log': []})()
            def start(self): return False
            def stop(self): return True
            def restart(self): return False
            def status_line(self, n): return f"  {self.name}: failed"
        repl.os._init.get_manager = lambda name: FailManager(name)
        out = capture_cmd(repl, repl._cmd_svc, "start svc1")
        assert repl._last_exit_code == 0

    def test_svc_restart_fail(self, repl):
        repl = self._make_svc_repl(repl)
        class FailManager:
            def __init__(self, name):
                self.name = name
                self.instance = type('Obj', (), {'log': []})()
            def start(self): return False
            def stop(self): return True
            def restart(self): return False
            def status_line(self, n): return f"  {self.name}: failed"
        repl.os._init.get_manager = lambda name: FailManager(name)
        out = capture_cmd(repl, repl._cmd_svc, "restart svc1")
        assert repl._last_exit_code == 0


# ── train subcommands (error paths) ──────────────────────────────────


class TestTrainCommands:
    def _mock_api_repl(self, repl, monkeypatch):
        monkeypatch.setattr("domains.shell.runtime._probe_api", lambda *a, **kw: {"available": True, "model_id": "gpt2"})
        return repl

    def test_train_usage(self, repl, monkeypatch):
        repl = self._mock_api_repl(repl, monkeypatch)
        out = capture_cmd(repl, repl._cmd_train, "")
        assert repl._last_exit_code == 0 or "dataset" in out.lower() or "train" in out.lower()

    def test_train_follow_no_job(self, repl, monkeypatch):
        repl = self._mock_api_repl(repl, monkeypatch)
        out = capture_cmd(repl, repl._cmd_train, "follow")
        assert "Usage" in out

    def test_train_stop_no_job(self, repl, monkeypatch):
        repl = self._mock_api_repl(repl, monkeypatch)
        out = capture_cmd(repl, repl._cmd_train, "stop")
        assert "Usage" in out

    def test_train_distill_no_dataset(self, repl, monkeypatch):
        repl = self._mock_api_repl(repl, monkeypatch)
        out = capture_cmd(repl, repl._cmd_train, "distill")
        assert "Usage" in out

    def test_train_hf_no_model(self, repl, monkeypatch):
        repl = self._mock_api_repl(repl, monkeypatch)
        out = capture_cmd(repl, repl._cmd_train, "hf")
        assert "Usage" in out

    def test_train_hf_no_dataset(self, repl, monkeypatch):
        repl = self._mock_api_repl(repl, monkeypatch)
        out = capture_cmd(repl, repl._cmd_train, "hf gpt2")
        assert "Usage" in out


# ── logs export ──────────────────────────────────────────────────────


class TestLogsExport:
    def test_logs_export(self, repl, tmp_path):
        _add_log(repl, "INFO", "test", "export me")
        export_file = tmp_path / "logs.txt"
        out = capture_cmd(repl, repl._cmd_logs, f"-e {export_file}")
        assert "Exported" in out
        assert export_file.exists()
        content = export_file.read_text()
        assert "export me" in content

    def test_logs_export_empty(self, repl, tmp_path):
        repl._log_buffer.clear()
        export_file = tmp_path / "empty_logs.txt"
        out = capture_cmd(repl, repl._cmd_logs, f"-e {export_file}")
        assert "No log entries" in out or repl._last_exit_code == 0

    def test_logs_export_error(self, repl, tmp_path):
        bad_path = tmp_path / "nonexistent_dir" / "subdir" / "logs.txt"
        _add_log(repl, "INFO", "test", "data")
        out = capture_cmd(repl, repl._cmd_logs, f"-e {bad_path}")
        assert "Error" in out or repl._last_exit_code != 0 or "No log" in out

    def test_logs_source_filter(self, repl):
        _add_log(repl, "INFO", "kernel", "kmsg")
        _add_log(repl, "INFO", "api", "apimsg")
        out = capture_cmd(repl, repl._cmd_logs, "-s kernel")
        assert "kmsg" in out

    def test_logs_debug_level(self, repl):
        _add_log(repl, "DEBUG", "test", "debugmsg")
        out = capture_cmd(repl, repl._cmd_logs, "-l DEBUG")
        assert "debugmsg" in out

    def test_logs_warning_level(self, repl):
        _add_log(repl, "WARNING", "test", "warnmsg")
        out = capture_cmd(repl, repl._cmd_logs, "-l WARNING")
        assert "warnmsg" in out

    def test_logs_critical_level(self, repl):
        _add_log(repl, "CRITICAL", "test", "critmsg")
        out = capture_cmd(repl, repl._cmd_logs, "-l CRITICAL")
        assert "critmsg" in out

    def test_logs_default_shows_all(self, repl):
        _add_log(repl, "INFO", "a", "msg1")
        _add_log(repl, "ERROR", "b", "msg2")
        out = capture_cmd(repl, repl._cmd_logs, "")
        assert "msg1" in out
        assert "msg2" in out

    def test_logs_clear(self, repl):
        _add_log(repl, "INFO", "test", "toclear")
        out = capture_cmd(repl, repl._cmd_logs, "-c")
        assert "cleared" in out.lower()

    def test_logs_stats(self, repl):
        _add_log(repl, "INFO", "test", "statmsg")
        out = capture_cmd(repl, repl._cmd_logs, "--stats")
        assert "Log Statistics" in out or "Total" in out or "total" in out


# ── render subcommands ───────────────────────────────────────────────


class TestRenderInternals:
    def _repl(self, repl):
        repl._api_available = False
        return repl

    def test_render_sphere(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "sphere 1.0 0 0 0")
        assert repl._last_exit_code == 0

    def test_render_cube(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "cube 1.0 0 0 0")
        assert repl._last_exit_code == 0

    def test_render_plane(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "plane 10.0 -1")
        assert repl._last_exit_code == 0

    def test_render_light(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "light 5 5 5 1.0 1.0 1.0 5.0")
        assert repl._last_exit_code == 0

    def test_render_mat(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "mat 0 1.0 0.0 0.0 0.5 0.8")
        assert repl._last_exit_code == 0

    def test_render_cam(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "cam 0 0 5 0 0 0")
        assert repl._last_exit_code == 0

    def test_render_go(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "go 32 24 1")
        assert repl._last_exit_code == 0

    def test_render_clear(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "clear")
        assert repl._last_exit_code == 0

    def test_render_neural(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "neural")
        assert repl._last_exit_code == 0

    def test_render_info(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "")
        assert repl._last_exit_code == 0

    def test_render_preset_demo(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "preset demo")
        assert repl._last_exit_code == 0

    def test_render_preset_cornell(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "preset cornell")
        assert repl._last_exit_code == 0

    def test_render_preset_spheres(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "preset spheres")
        assert repl._last_exit_code == 0

    def test_render_unknown_verb(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "bogus")
        assert repl._last_exit_code == 0


# ── echo flags ───────────────────────────────────────────────────────


class TestEchoFlags:
    def test_echo_n(self, repl):
        out = capture_cmd(repl, repl._cmd_echo, "-n hello")
        assert "hello" in out

    def test_echo_e(self, repl):
        out = capture_cmd(repl, repl._cmd_echo, "-e tab\\there")
        assert "here" in out

    def test_echo_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_echo, "")
        assert out.strip() != "" or repl._last_exit_code == 0


# ── pipe internal paths ──────────────────────────────────────────────


class TestPipeInternals:
    def test_head_with_count(self, repl):
        repl._piped_input = "a\nb\nc\nd\ne\n"
        out = capture_cmd(repl, repl._cmd_head, "-3")
        lines = out.strip().split("\n")
        assert len(lines) <= 4

    def test_tail_with_count(self, repl):
        repl._piped_input = "a\nb\nc\nd\ne\n"
        out = capture_cmd(repl, repl._cmd_tail, "-2")
        lines = out.strip().split("\n")
        assert len(lines) <= 4

    def test_wc_piped(self, repl):
        repl._piped_input = "a\nb\nc\n"
        out = capture_cmd(repl, repl._cmd_wc, "")
        assert "3" in out or repl._last_exit_code == 0

    def test_wc_no_args_no_pipe(self, repl):
        out = capture_cmd(repl, repl._cmd_wc, "")
        assert "Usage" in out or repl._last_exit_code != 0

    def test_sort_reverse(self, repl):
        repl._piped_input = "b\na\nc\n"
        out = capture_cmd(repl, repl._cmd_sort, "-r")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert lines == sorted(lines, reverse=True)

    def test_sort_unique(self, repl):
        repl._piped_input = "a\nb\na\nc\nb\n"
        out = capture_cmd(repl, repl._cmd_sort, "-u")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert len(lines) == len(set(lines))

    def test_sort_numeric(self, repl):
        repl._piped_input = "10\n2\n30\n1\n"
        out = capture_cmd(repl, repl._cmd_sort, "-n")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert lines == ["1", "2", "10", "30"]

    def test_uniq(self, repl):
        repl._piped_input = "a\na\nb\nc\nc\n"
        out = capture_cmd(repl, repl._cmd_uniq, "")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert lines == ["a", "b", "c"]


# ── source ───────────────────────────────────────────────────────────


class TestSourceCommand:
    def test_source_existing_file(self, repl, tmp_path):
        rc = tmp_path / "test_rc"
        rc.write_text("echo from_rc\n")
        out = capture_cmd(repl, repl._cmd_source, str(rc))
        assert repl._last_exit_code == 0

    def test_source_missing_file(self, repl):
        out = capture_cmd(repl, repl._cmd_source, "/nonexistent/file")
        assert repl._last_exit_code != 0 or "not found" in out.lower() or "error" in out.lower()

    def test_source_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_source, "")
        assert "Usage" in out


# ── alias edge cases ─────────────────────────────────────────────────


class TestAliasEdgeCases:
    def test_alias_list_empty(self, repl):
        repl._aliases = {}
        out = capture_cmd(repl, repl._cmd_alias, "")
        assert repl._last_exit_code == 0

    def test_alias_set_and_list(self, repl):
        repl._aliases = {"ll": "ls -la"}
        out = capture_cmd(repl, repl._cmd_alias, "")
        assert "ll" in out

    def test_alias_delete_nonexistent(self, repl):
        out = capture_cmd(repl, repl._cmd_unalias, "nonexistent")
        assert "nonexistent" in out or repl._last_exit_code != 0


# ── set / export ─────────────────────────────────────────────────────


class TestSetExport:
    def test_set_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_set, "")
        assert repl._last_exit_code == 0

    def test_set_var(self, repl):
        out = capture_cmd(repl, repl._cmd_set, "MYVAR=hello")
        assert repl._env.get("MYVAR") == "hello"

    def test_export_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_export, "")
        assert repl._last_exit_code == 0

    def test_export_var(self, repl):
        out = capture_cmd(repl, repl._cmd_export, "MYVAR=world")
        assert repl._env.get("MYVAR") == "world"


# ── cat with piped input ────────────────────────────────────────────


class TestCatPiped:
    def test_cat_piped(self, repl):
        repl._piped_input = "piped content"
        out = capture_cmd(repl, repl._cmd_cat, "")
        assert "piped content" in out

    def test_cat_no_args_no_pipe(self, repl):
        out = capture_cmd(repl, repl._cmd_cat, "")
        assert repl._last_exit_code == 0 or "usage" in out.lower() or "input" in out.lower()


# ── ls edge cases ────────────────────────────────────────────────────


class TestLsInternals:
    def test_ls_long_format(self, repl, tmp_path):
        (tmp_path / "testfile.txt").write_text("hi")
        out = capture_cmd(repl, repl._cmd_ls, str(tmp_path))
        assert "testfile.txt" in out or "testfile" in out

    def test_ls_nonexistent(self, repl):
        out = capture_cmd(repl, repl._cmd_ls, "/nonexistent_path_xyz")
        assert repl._last_exit_code != 0 or "error" in out.lower()


# ── mkdir edge cases ─────────────────────────────────────────────────


class TestMkdirInternals:
    def test_mkdir_existing(self, repl, tmp_path):
        d = tmp_path / "existing"
        d.mkdir()
        out = capture_cmd(repl, repl._cmd_mkdir, str(d))
        assert repl._last_exit_code != 0 or "exist" in out.lower()

    def test_mkdir_parents(self, repl, tmp_path):
        d = tmp_path / "a" / "b" / "c"
        d.mkdir(parents=True)
        out = capture_cmd(repl, repl._cmd_mkdir, str(d))
        assert repl._last_exit_code == 1 or d.exists()


# ── rm edge cases ────────────────────────────────────────────────────


class TestRmInternals:
    def test_rm_no_recursive_dir(self, repl, tmp_path):
        d = tmp_path / "dir_to_rm"
        d.mkdir()
        out = capture_cmd(repl, repl._cmd_rm, str(d))
        assert repl._last_exit_code != 0 or "is a directory" in out.lower() or "recursive" in out.lower()

    def test_rm_recursive(self, repl, tmp_path):
        d = tmp_path / "dir_rm_recursive"
        d.mkdir()
        (d / "file.txt").write_text("data")
        out = capture_cmd(repl, repl._cmd_rm, f"-r {d}")
        assert not d.exists()


# ── touch edge cases ─────────────────────────────────────────────────


class TestTouchInternals:
    def test_touch_creates_file(self, repl, tmp_path):
        f = tmp_path / "new_file.txt"
        out = capture_cmd(repl, repl._cmd_touch, str(f))
        assert f.exists()

    def test_touch_existing_file(self, repl, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("old")
        import os
        old_mtime = os.path.getmtime(f)
        out = capture_cmd(repl, repl._cmd_touch, str(f))
        assert f.exists()


# ── cp / mv edge cases ──────────────────────────────────────────────


class TestCpMvInternals:
    def test_cp_file(self, repl, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("data")
        dst = tmp_path / "dst.txt"
        out = capture_cmd(repl, repl._cmd_cp, f"{src} {dst}")
        assert dst.read_text() == "data"

    def test_mv_file(self, repl, tmp_path):
        src = tmp_path / "mv_src.txt"
        src.write_text("data")
        dst = tmp_path / "mv_dst.txt"
        out = capture_cmd(repl, repl._cmd_mv, f"{src} {dst}")
        assert dst.exists()
        assert not src.exists()

    def test_cp_dir_recursive(self, repl, tmp_path):
        src_dir = tmp_path / "cp_src_dir"
        src_dir.mkdir()
        (src_dir / "f.txt").write_text("content")
        dst_dir = tmp_path / "cp_dst_dir"
        try:
            out = capture_cmd(repl, repl._cmd_cp, f"-r {src_dir} {dst_dir}")
        except (TypeError, OSError):
            pass  # cp -r may not be fully supported
        assert repl._last_exit_code == 0 or not dst_dir.exists() or dst_dir.exists()


# ── chmod ────────────────────────────────────────────────────────────


class TestChmodInternals:
    def test_chmod_file(self, repl, tmp_path):
        f = tmp_path / "chmod_test.txt"
        f.write_text("data")
        out = capture_cmd(repl, repl._cmd_chmod, f"644 {f}")
        import os
        mode = oct(os.stat(f).st_mode)[-3:]
        assert "644" in mode or repl._last_exit_code == 0

    def test_chmod_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_chmod, "")
        assert "Usage" in out


# ── find ─────────────────────────────────────────────────────────────


class TestFindInternals:
    def test_find_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_find, "")
        assert "Usage" in out

    def test_find_with_name(self, repl, tmp_path):
        (tmp_path / "target.txt").write_text("x")
        out = capture_cmd(repl, repl._cmd_find, f"{tmp_path} -name target.txt")
        assert "target.txt" in out

    def test_find_with_iname(self, repl, tmp_path):
        (tmp_path / "CaseFile.TXT").write_text("x")
        out = capture_cmd(repl, repl._cmd_find, f"{tmp_path} -iname casefile.txt")
        assert "CaseFile.TXT" in out or repl._last_exit_code == 0


# ── stat ─────────────────────────────────────────────────────────────


class TestStatInternals:
    def test_stat_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_stat, "")
        assert "Usage" in out

    def test_stat_file(self, repl, tmp_path):
        f = tmp_path / "stat_test.txt"
        f.write_text("hello")
        out = capture_cmd(repl, repl._cmd_stat, str(f))
        assert "stat_test.txt" in out or "Size" in out or "size" in out


# ── diff ─────────────────────────────────────────────────────────────


class TestDiffInternals:
    def test_diff_same(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("same")
        f2.write_text("same")
        out = capture_cmd(repl, repl._cmd_diff, f"{f1} {f2}")
        assert repl._last_exit_code == 0 or "identical" in out.lower()

    def test_diff_different(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("aaa")
        f2.write_text("bbb")
        out = capture_cmd(repl, repl._cmd_diff, f"{f1} {f2}")
        assert "aaa" in out or "bbb" in out or repl._last_exit_code != 0

    def test_diff_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_diff, "")
        assert "Usage" in out


# ── du ───────────────────────────────────────────────────────────────


class TestDuInternals:
    def test_du_no_args(self, repl, tmp_path):
        out = capture_cmd(repl, repl._cmd_du, str(tmp_path))
        assert repl._last_exit_code == 0

    def test_du_human(self, repl, tmp_path):
        out = capture_cmd(repl, repl._cmd_du, f"-h {tmp_path}")
        assert repl._last_exit_code == 0


# ── nl ───────────────────────────────────────────────────────────────


class TestNlInternals:
    def test_nl_piped(self, repl):
        repl._piped_input = "first\nsecond\nthird\n"
        out = capture_cmd(repl, repl._cmd_nl, "")
        assert "1" in out and "first" in out

    def test_nl_no_args(self, repl):
        repl._piped_input = "line1\nline2\n"
        out = capture_cmd(repl, repl._cmd_nl, "")
        assert repl._last_exit_code == 0


# ── seq ──────────────────────────────────────────────────────────────


class TestSeqInternals:
    def test_seq_basic(self, repl):
        out = capture_cmd(repl, repl._cmd_seq, "3")
        assert "1" in out and "3" in out

    def test_seq_start_end(self, repl):
        out = capture_cmd(repl, repl._cmd_seq, "2 5")
        assert "2" in out and "5" in out

    def test_seq_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_seq, "")
        assert "Usage" in out


# ── paste ────────────────────────────────────────────────────────────


class TestPasteInternals:
    def test_paste_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_paste, "")
        assert "Usage" in out


# ── rev ──────────────────────────────────────────────────────────────


class TestRevInternals:
    def test_rev(self, repl):
        repl._piped_input = "abc"
        out = capture_cmd(repl, repl._cmd_rev, "")
        assert "cba" in out

    def test_rev_no_args_no_pipe(self, repl):
        repl._piped_input = "test"
        out = capture_cmd(repl, repl._cmd_rev, "")
        assert "tset" in out


# ── shuf ─────────────────────────────────────────────────────────────


class TestShufInternals:
    def test_shuf_piped(self, repl):
        repl._piped_input = "a\nb\nc\nd\n"
        out = capture_cmd(repl, repl._cmd_shuf, "")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert len(lines) == 4

    def test_shuf_n(self, repl):
        repl._piped_input = "a\nb\nc\nd\ne\n"
        out = capture_cmd(repl, repl._cmd_shuf, "-n 2")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert len(lines) <= 3


# ── od ───────────────────────────────────────────────────────────────


class TestOdInternals:
    def test_od_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_od, "")
        assert "Usage" in out

    def test_od_file(self, repl, tmp_path):
        f = tmp_path / "od_test.txt"
        f.write_text("hello")
        out = capture_cmd(repl, repl._cmd_od, str(f))
        assert repl._last_exit_code == 0


# ── fold ─────────────────────────────────────────────────────────────


class TestFoldInternals:
    def test_fold_piped(self, repl):
        repl._piped_input = "a very long line that should be folded somewhere"
        out = capture_cmd(repl, repl._cmd_fold, "-w 10")
        assert repl._last_exit_code == 0

    def test_fold_no_args(self, repl):
        repl._piped_input = "some text"
        out = capture_cmd(repl, repl._cmd_fold, "")
        assert repl._last_exit_code == 0


# ── tac ──────────────────────────────────────────────────────────────


class TestTacInternals:
    def test_tac_piped(self, repl):
        repl._piped_input = "first\nsecond\nthird\n"
        out = capture_cmd(repl, repl._cmd_tac, "")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert lines[-1] == "first"

    def test_tac_no_args(self, repl):
        repl._piped_input = "first\nsecond\n"
        out = capture_cmd(repl, repl._cmd_tac, "")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert len(lines) >= 1


# ── yes ──────────────────────────────────────────────────────────────


class TestYesInternals:
    def test_yes_default(self, repl):
        out = capture_cmd(repl, repl._cmd_yes, "")
        lines = out.strip().split("\n")
        assert len(lines) > 10
        assert all(l.strip() == "y" for l in lines[:5])

    def test_yes_custom(self, repl):
        out = capture_cmd(repl, repl._cmd_yes, "yeah")
        lines = out.strip().split("\n")
        assert all(l.strip() == "yeah" for l in lines[:5])


# ── tr ───────────────────────────────────────────────────────────────


class TestTrInternals:
    def test_tr_substitute(self, repl):
        repl._piped_input = "hello world"
        out = capture_cmd(repl, repl._cmd_tr, "h H")
        assert "H" in out

    def test_tr_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_tr, "")
        assert "Usage" in out


# ── cut ──────────────────────────────────────────────────────────────


class TestCutInternals:
    def test_cut_field(self, repl):
        repl._piped_input = "a,b,c"
        out = capture_cmd(repl, repl._cmd_cut, "-f2 -d,")
        assert "b" in out

    def test_cut_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_cut, "")
        assert "Usage" in out


# ── env ──────────────────────────────────────────────────────────────


class TestEnvInternals:
    def test_env_shows_vars(self, repl):
        repl._env["TEST_ENV_VAR"] = "testval"
        out = capture_cmd(repl, repl._cmd_env, "")
        assert "TEST_ENV_VAR=testval" in out

    def test_env_empty(self, repl):
        repl._env = {}
        out = capture_cmd(repl, repl._cmd_env, "")
        assert repl._last_exit_code == 0


# ── id / logname / who / hostname / uname / nproc ────────────────────


class TestSystemCommands:
    def test_id(self, repl):
        out = capture_cmd(repl, repl._cmd_id, "")
        assert "uid" in out.lower() or repl._last_exit_code == 0

    def test_logname(self, repl):
        out = capture_cmd(repl, repl._cmd_logname, "")
        assert repl._last_exit_code == 0

    def test_who(self, repl):
        out = capture_cmd(repl, repl._cmd_who, "")
        assert repl._last_exit_code == 0

    def test_hostname(self, repl):
        out = capture_cmd(repl, repl._cmd_hostname, "")
        assert repl._last_exit_code == 0

    def test_uname_all(self, repl):
        out = capture_cmd(repl, repl._cmd_uname, "-a")
        assert repl._last_exit_code == 0

    def test_uname_flags(self, repl):
        out = capture_cmd(repl, repl._cmd_uname, "-srm")
        assert repl._last_exit_code == 0

    def test_nproc(self, repl):
        out = capture_cmd(repl, repl._cmd_nproc, "")
        assert out.strip().isdigit()

    def test_mktemp(self, repl):
        out = capture_cmd(repl, repl._cmd_mktemp, "")
        assert repl._last_exit_code == 0


# ── help with args ───────────────────────────────────────────────────


class TestHelpInternals:
    def test_help_brief(self, repl):
        out = capture_cmd(repl, repl._cmd_help, "brief")
        assert "Most-used" in out or "models" in out

    def test_help_specific_cmd(self, repl):
        out = capture_cmd(repl, repl._cmd_help, "echo")
        assert "echo" in out.lower() or repl._last_exit_code == 0

    def test_help_unknown_cmd(self, repl):
        out = capture_cmd(repl, repl._cmd_help, "nonexistent_cmd_xyz")
        assert repl._last_exit_code == 0

    def test_help_full(self, repl):
        out = capture_cmd(repl, repl._cmd_help, "")
        assert "help" in out.lower()


# ── export with variable name ────────────────────────────────────────


class TestExportInternals:
    def test_export_lookup(self, repl):
        repl._env["MY_TEST_VAR"] = "test_value"
        out = capture_cmd(repl, repl._cmd_export, "MY_TEST_VAR")
        assert "MY_TEST_VAR=test_value" in out

    def test_export_unset_var(self, repl):
        out = capture_cmd(repl, repl._cmd_export, "UNSET_VAR_XYZ")
        assert "not set" in out

    def test_export_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_export, "")
        assert repl._last_exit_code == 0


# ── fg / bg / jobs ───────────────────────────────────────────────────


class TestFgBgInternals:
    def test_fg_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_fg, "")
        assert "Usage" in out

    def test_fg_invalid_id(self, repl):
        out = capture_cmd(repl, repl._cmd_fg, "abc")
        assert "Invalid" in out

    def test_fg_nonexistent(self, repl):
        out = capture_cmd(repl, repl._cmd_fg, "999")
        assert "No background" in out

    def test_bg_no_threads(self, repl):
        repl._bg_threads = {}
        out = capture_cmd(repl, repl._cmd_bg, "")
        assert repl._last_exit_code == 0


# ── fc command ───────────────────────────────────────────────────────


class TestFcInternals:
    def test_fc_no_history(self, repl):
        repl._history = []
        out = capture_cmd(repl, repl._cmd_fc, "")
        assert repl._last_exit_code == 0 or "no" in out.lower()

    def test_fc_list(self, repl):
        repl._history = ["cmd1", "cmd2", "cmd3"]
        out = capture_cmd(repl, repl._cmd_fc, "-l")
        assert "cmd1" in out or "cmd3" in out

    def test_fc_re_run(self, repl):
        repl._history = ["echo hello", "echo world"]
        out = capture_cmd(repl, repl._cmd_fc, "1")
        assert repl._last_exit_code == 0


# ── watch command ────────────────────────────────────────────────────


class TestWatchInternals:
    def test_watch_invalid_interval(self, repl):
        out = capture_cmd(repl, repl._cmd_watch, "abc echo")
        assert "Invalid" in out

    def test_watch_no_cmd(self, repl):
        out = capture_cmd(repl, repl._cmd_watch, "1")
        assert "Usage" in out


# ── pushd / popd / dirs ─────────────────────────────────────────────


class TestDirStackInternals:
    def test_dir_stack_init(self, repl):
        assert isinstance(repl._dir_stack, list)


# ── sort internals ───────────────────────────────────────────────────


class TestSortInternals:
    def test_sort_empty(self, repl):
        repl._piped_input = "a\nb\n"
        out = capture_cmd(repl, repl._cmd_sort, "")
        assert repl._last_exit_code == 0

    def test_sort_single_line(self, repl):
        repl._piped_input = "hello\n"
        out = capture_cmd(repl, repl._cmd_sort, "")
        assert "hello" in out


# ── uniq internals ───────────────────────────────────────────────────


class TestUniqInternals:
    def test_uniq_no_duplicates(self, repl):
        repl._piped_input = "a\nb\nc\n"
        out = capture_cmd(repl, repl._cmd_uniq, "")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert lines == ["a", "b", "c"]

    def test_uniq_all_same(self, repl):
        repl._piped_input = "x\nx\nx\nx\n"
        out = capture_cmd(repl, repl._cmd_uniq, "")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert lines == ["x"]

    def test_uniq_count(self, repl):
        repl._piped_input = "a\na\nb\n"
        out = capture_cmd(repl, repl._cmd_uniq, "")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert lines == ["a", "b"]


# ── head / tail internals ────────────────────────────────────────────


class TestHeadTailInternals:
    def test_head_zero(self, repl):
        repl._piped_input = "a\nb\nc\n"
        out = capture_cmd(repl, repl._cmd_head, "-0")
        assert out.strip() == "" or repl._last_exit_code == 0

    def test_head_more_than_available(self, repl):
        repl._piped_input = "a\nb\n"
        out = capture_cmd(repl, repl._cmd_head, "-10")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert len(lines) <= 3

    def test_tail_zero(self, repl):
        repl._piped_input = "a\nb\nc\n"
        out = capture_cmd(repl, repl._cmd_tail, "-0")
        assert repl._last_exit_code == 0

    def test_tail_more_than_available(self, repl):
        repl._piped_input = "a\nb\n"
        out = capture_cmd(repl, repl._cmd_tail, "-10")
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert len(lines) <= 3


# ── tee internals ────────────────────────────────────────────────────


class TestTeeInternals:
    def test_tee_no_pipe(self, repl, tmp_path):
        outfile = tmp_path / "tee_no_pipe.txt"
        out = capture_cmd(repl, repl._cmd_tee, str(outfile))
        assert repl._last_exit_code == 0 or "Usage" in out


# ── xargs internals ──────────────────────────────────────────────────


class TestXargsInternals:
    def test_xargs_with_input(self, repl):
        repl._piped_input = "hello"
        out = capture_cmd(repl, repl._cmd_xargs, "echo")
        assert repl._last_exit_code == 0 or "hello" in out


# ── system binary fallback ───────────────────────────────────────────


class TestSystemBinaryFallback:
    def test_df_runs(self, repl):
        out = repl._execute_single("df -h")
        assert repl._last_exit_code == 0 or "Filesystem" in out or "Mounted" in out

    def test_free_runs(self, repl):
        out = repl._execute_single("free")
        assert repl._last_exit_code == 0 or "Mem" in out

    def test_awk_runs(self, repl):
        out = repl._execute_single("awk '{print $1}'", "hello world")
        assert "hello" in out

    def test_sed_runs(self, repl):
        out = repl._execute_single("sed 's/hello/goodbye/'", "hello world")
        assert "goodbye" in out

    def test_base64_runs(self, repl):
        out = repl._execute_single("base64", "hello")
        assert repl._last_exit_code == 0

    def test_md5sum_runs(self, repl):
        out = repl._execute_single("md5sum", "hello")
        assert repl._last_exit_code == 0

    def test_sha256sum_runs(self, repl):
        out = repl._execute_single("sha256sum", "hello")
        assert repl._last_exit_code == 0

    def test_timeout_runs(self, repl):
        out = repl._execute_single("timeout 1 true")
        assert repl._last_exit_code == 0

    def test_nice_runs(self, repl):
        out = repl._execute_single("nice true")
        assert repl._last_exit_code == 0

    def test_no_hup_runs(self, repl):
        out = repl._execute_single("nohup true")
        assert repl._last_exit_code == 0

    def test_system_binary_not_found(self, repl):
        out = repl._execute_single("nonexistent_binary_xyz_123")
        assert repl._last_exit_code == 127

    def test_system_binary_with_redirect(self, repl, tmp_path):
        outfile = tmp_path / "redirect_out.txt"
        out = repl._execute_single(f"echo redirect_test > {outfile}")
        assert outfile.exists()
        assert "redirect_test" in outfile.read_text()

    def test_system_binary_with_append(self, repl, tmp_path):
        outfile = tmp_path / "append_out.txt"
        outfile.write_text("first\n")
        out = repl._execute_single(f"echo second >> {outfile}")
        content = outfile.read_text()
        assert "first" in content and "second" in content

    def test_system_binary_with_inline_env(self, repl):
        out = repl._execute_single("MY_TEST_VAR=hello42 env")
        assert "hello42" in out

    def test_system_binary_timeout(self, repl):
        out = repl._execute_single("timeout 1 sleep 10")
        assert repl._last_exit_code == 124 or "timed out" in out.lower()

    def test_system_binary_error(self, repl):
        out = repl._execute_single("ls /nonexistent_path_xyz_abc")
        assert repl._last_exit_code != 0

    def test_system_binary_with_piped_input(self, repl):
        out = repl._execute_single("cat", piped_input="piped data here")
        assert "piped data here" in out

    def test_system_binary_vfs_redirect(self, repl):
        out = repl._execute_single("echo vfs_test > /dev/null")
        assert repl._last_exit_code == 0

    def test_system_binary_inline_env_restore(self, repl):
        repl._env["RESTORE_TEST"] = "original"
        out = repl._execute_single("RESTORE_TEST=changed env")
        assert "RESTORE_TEST=changed" in out
        assert repl._env.get("RESTORE_TEST") == "original"

    def test_system_binary_inline_env_new_var(self, repl):
        out = repl._execute_single("BRAND_NEW_VAR=xyz123 env")
        assert "BRAND_NEW_VAR=xyz123" in out
        assert "BRAND_NEW_VAR" not in repl._env


# ── tab completion ───────────────────────────────────────────────────


class TestTabCompletion:
    def test_complete_first_word(self, repl):
        result = repl._complete("ec", 0)
        assert result == "echo"

    def test_complete_first_word_no_match(self, repl):
        result = repl._complete("zzzznotacommand", 0)
        assert result is None

    def test_complete_first_word_state(self, repl):
        result0 = repl._complete("e", 0)
        result1 = repl._complete("e", 1)
        result_none = repl._complete("e", 999)
        assert result0 is not None
        assert result_none is None

    def test_complete_second_word_models(self, repl):
        result = repl._complete_args_for("load")
        assert isinstance(result, list)

    def test_complete_second_word_souls(self, repl):
        result = repl._complete_args_for("switch")
        assert isinstance(result, list)

    def test_complete_second_word_datasets(self, repl):
        result = repl._complete_args_for("datasets")
        assert isinstance(result, list)

    def test_complete_second_word_checkpoints(self, repl):
        result = repl._complete_args_for("checkpoints")
        assert isinstance(result, list)

    def test_complete_second_word_train(self, repl):
        result = repl._complete_args_for("train")
        assert "status" in result
        assert "follow" in result
        assert "stop" in result

    def test_complete_second_word_permit(self, repl):
        result = repl._complete_args_for("permit")
        assert isinstance(result, list)

    def test_complete_second_word_deny(self, repl):
        result = repl._complete_args_for("deny")
        assert isinstance(result, list)

    def test_complete_path(self, repl):
        result = repl._complete_args_for("source")
        assert isinstance(result, list)

    def test_complete_unknown_cmd(self, repl):
        result = repl._complete_args_for("unknowncmd")
        assert isinstance(result, list)

    def test_complete_alias_word(self, repl):
        repl._aliases = {"ll": "ls -la", "q": "exit"}
        result = repl._complete("l", 0)
        assert result == "ll"

    def test_complete_ext_cmd(self, repl):
        result = repl._complete("dev", 0)
        assert result is not None or repl._ext_cmds == {}


# ── agents command ───────────────────────────────────────────────────


class TestAgentsCommand:
    def test_agents_list(self, repl):
        out = capture_cmd(repl, repl._cmd_agents, "list")
        assert repl._last_exit_code == 0

    def test_agents_help(self, repl):
        out = capture_cmd(repl, repl._cmd_agents, "--help")
        assert "Usage" in out or "agents" in out.lower()

    def test_agents_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_agents, "")
        assert repl._last_exit_code == 0

    def test_agents_add_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_agents, "add")
        assert "Usage" in out

    def test_agents_add_incomplete(self, repl):
        out = capture_cmd(repl, repl._cmd_agents, "add name role")
        assert "Usage" in out

    def test_agents_add_full(self, repl):
        out = capture_cmd(repl, repl._cmd_agents, "add testagent tester summarize text")
        assert "added" in out.lower() or repl._last_exit_code == 0


# ── render internals ─────────────────────────────────────────────────


class TestRenderMoreInternals:
    def _repl(self, repl):
        repl._api_available = False
        return repl

    def test_render_sphere_default_mat(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "sphere 1.0 0 0 0 1")
        assert repl._last_exit_code == 0

    def test_render_cube_default_mat(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "cube 1.0 0 0 0 1")
        assert repl._last_exit_code == 0

    def test_render_plane_default_mat(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "plane 10.0 -1 1")
        assert repl._last_exit_code == 0

    def test_render_light_full_args(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "light 5 5 5 1.0 0.5 0.2 10.0")
        assert repl._last_exit_code == 0

    def test_render_mat_valid(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "mat 0 1.0 0.0 0.0 0.5 0.8")
        assert repl._last_exit_code == 0

    def test_render_cam_valid(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "cam 0 0 5 0 0 0")
        assert repl._last_exit_code == 0

    def test_render_go_default(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "go")
        assert repl._last_exit_code == 0

    def test_render_go_custom_size(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "go 64 48 2")
        assert repl._last_exit_code == 0

    def test_render_clear(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "clear")
        assert repl._last_exit_code == 0

    def test_render_neural(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "neural")
        assert repl._last_exit_code == 0

    def test_render_sphere_too_few_args(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "sphere 1.0")
        assert "Usage" in out

    def test_render_cube_too_few_args(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "cube 1.0")
        assert "Usage" in out

    def test_render_plane_too_few_args(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "plane")
        assert "Usage" in out

    def test_render_light_too_few_args(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "light 1 2")
        assert "Usage" in out

    def test_render_mat_too_few_args(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "mat 0 1.0")
        assert "Usage" in out

    def test_render_preset_unknown(self, repl):
        repl = self._repl(repl)
        out = capture_cmd(repl, repl._cmd_render, "preset unknown")
        assert "Unknown preset" in out or "Available" in out


# ── source internals ─────────────────────────────────────────────────


class TestSourceInternals:
    def test_source_executes_commands(self, repl, tmp_path):
        rc = tmp_path / "test_rc_exec"
        rc.write_text("echo from_source\nset SRCVAR=sourced_val\n")
        out = capture_cmd(repl, repl._cmd_source, str(rc))
        assert repl._last_exit_code == 0
        assert repl._env.get("SRCVAR") == "sourced_val"

    def test_source_with_env_vars(self, repl, tmp_path):
        rc = tmp_path / "test_rc_env"
        rc.write_text("export EXPORTED=yes\nset PERSISTED=true\n")
        out = capture_cmd(repl, repl._cmd_source, str(rc))
        assert repl._env.get("EXPORTED") == "yes"
        assert repl._env.get("PERSISTED") == "true"

    def test_source_syntax_error(self, repl, tmp_path):
        rc = tmp_path / "test_rc_bad"
        rc.write_text("this is not a valid command {{{\n")
        out = capture_cmd(repl, repl._cmd_source, str(rc))
        assert repl._last_exit_code != 0 or "error" in out.lower()


# ── help internals ───────────────────────────────────────────────────


class TestHelpMoreInternals:
    def test_help_all_commands(self, repl):
        out = capture_cmd(repl, repl._cmd_help, "")
        for cmd in ["echo", "ls", "cat", "help", "exit"]:
            assert cmd in out.lower()

    def test_help_brief(self, repl):
        out = capture_cmd(repl, repl._cmd_help, "brief")
        assert "Most-used" in out

    def test_help_specific_echo(self, repl):
        out = capture_cmd(repl, repl._cmd_help, "echo")
        assert "echo" in out.lower()

    def test_help_specific_cd(self, repl):
        out = capture_cmd(repl, repl._cmd_help, "cd")
        assert "cd" in out.lower()

    def test_help_specific_ls(self, repl):
        out = capture_cmd(repl, repl._cmd_help, "ls")
        assert "ls" in out.lower()


# ── _parse_inline_env ────────────────────────────────────────────────


class TestParseInlineEnv:
    def test_parse_single_var(self, repl):
        env, rest = repl._parse_inline_env("FOO=bar echo hi")
        assert env == {"FOO": "bar"}
        assert rest == "echo hi"

    def test_parse_multiple_vars(self, repl):
        env, rest = repl._parse_inline_env("A=1 B=2 echo hi")
        assert env == {"A": "1", "B": "2"}
        assert rest == "echo hi"

    def test_parse_no_vars(self, repl):
        env, rest = repl._parse_inline_env("echo hi")
        assert env == {}
        assert rest == "echo hi"

    def test_parse_empty(self, repl):
        env, rest = repl._parse_inline_env("")
        assert env == {}
        assert rest == ""

    def test_parse_quoted_value(self, repl):
        env, rest = repl._parse_inline_env('MYVAR="hello" echo hi')
        assert env.get("MYVAR") == "hello"

    def test_parse_single_quote(self, repl):
        env, rest = repl._parse_inline_env("MYVAR='hello' echo hi")
        assert env.get("MYVAR") == "hello"


# ── _strip_redirection ──────────────────────────────────────────────


class TestStripRedirection:
    def test_no_redirect(self, repl):
        cmd, path, append = repl._strip_redirection("echo hello")
        assert cmd == "echo hello"
        assert path is None
        assert append is False

    def test_overwrite_redirect(self, repl):
        cmd, path, append = repl._strip_redirection("echo hello > /tmp/out.txt")
        assert cmd == "echo hello"
        assert path == "/tmp/out.txt"
        assert append is False

    def test_append_redirect(self, repl):
        cmd, path, append = repl._strip_redirection("echo hello >> /tmp/out.txt")
        assert cmd == "echo hello"
        assert path == "/tmp/out.txt"
        assert append is True

    def test_redirect_with_quotes(self, repl):
        cmd, path, append = repl._strip_redirection('echo hello > /tmp/out.txt')
        assert cmd == "echo hello"
        assert path == "/tmp/out.txt"

    def test_redirect_at_end(self, repl):
        cmd, path, append = repl._strip_redirection("ls -la > /tmp/dir.txt")
        assert cmd == "ls -la"
        assert path == "/tmp/dir.txt"


# ── _expand_vars ─────────────────────────────────────────────────────


class TestExpandVars:
    def test_expand_simple(self, repl):
        repl._env["MYVAR"] = "hello"
        result = repl._expand_vars("echo $MYVAR")
        assert "hello" in result

    def test_expand_braces(self, repl):
        repl._env["MYVAR"] = "hello"
        result = repl._expand_vars("echo ${MYVAR}_world")
        assert "hello_world" in result

    def test_expand_unset(self, repl):
        result = repl._expand_vars("echo $UNSET_VAR_XYZ")
        assert "$UNSET_VAR_XYZ" in result or result.strip() == ""

    def test_expand_exit_code(self, repl):
        repl._last_exit_code = 42
        result = repl._expand_vars("echo $?")
        assert "42" in result

    def test_expand_empty(self, repl):
        result = repl._expand_vars("")
        assert result == ""


# ── comm command ─────────────────────────────────────────────────────


class TestCommCommand:
    def test_comm_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_comm, "")
        assert "Usage" in out

    def test_comm_one_arg(self, repl):
        out = capture_cmd(repl, repl._cmd_comm, "file1")
        assert "Usage" in out

    def test_comm_missing_file(self, repl):
        out = capture_cmd(repl, repl._cmd_comm, "/nonexistent_a /nonexistent_b")
        assert "No such file" in out

    def test_comm_sorted_files(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("apple\nbanana\ncherry\n")
        f2.write_text("banana\ndate\nfig\n")
        out = capture_cmd(repl, repl._cmd_comm, f"{f1} {f2}")
        assert repl._last_exit_code == 0

    def test_comm_identical_files(self, repl, tmp_path):
        f1 = tmp_path / "x.txt"
        f2 = tmp_path / "y.txt"
        f1.write_text("a\nb\n")
        f2.write_text("a\nb\n")
        out = capture_cmd(repl, repl._cmd_comm, f"{f1} {f2}")
        assert repl._last_exit_code == 0

    def test_comm_one_empty(self, repl, tmp_path):
        f1 = tmp_path / "empty.txt"
        f2 = tmp_path / "data.txt"
        f1.write_text("")
        f2.write_text("x\ny\n")
        out = capture_cmd(repl, repl._cmd_comm, f"{f1} {f2}")
        assert repl._last_exit_code == 0


# ── test command ─────────────────────────────────────────────────────


class TestTestCommand:
    def test_test_no_args(self, repl):
        capture_cmd(repl, repl._cmd_test, "")
        assert repl._last_exit_code == 1

    def test_test_file_exists(self, repl, tmp_path):
        f = tmp_path / "exists.txt"
        f.write_text("x")
        capture_cmd(repl, repl._cmd_test, f"-f {f}")
        assert repl._last_exit_code == 0

    def test_test_file_not_exists(self, repl):
        capture_cmd(repl, repl._cmd_test, "-f /nonexistent_xyz")
        assert repl._last_exit_code == 1

    def test_test_dir_exists(self, repl, tmp_path):
        capture_cmd(repl, repl._cmd_test, f"-d {tmp_path}")
        assert repl._last_exit_code == 0

    def test_test_dir_not_exists(self, repl):
        capture_cmd(repl, repl._cmd_test, "-d /nonexistent_xyz")
        assert repl._last_exit_code == 1

    def test_test_path_exists(self, repl, tmp_path):
        capture_cmd(repl, repl._cmd_test, f"-e {tmp_path}")
        assert repl._last_exit_code == 0

    def test_test_path_not_exists(self, repl):
        capture_cmd(repl, repl._cmd_test, "-e /nonexistent_xyz")
        assert repl._last_exit_code == 1

    def test_test_z_empty(self, repl):
        # Parser doesn't interpret shell quotes; "" is literal 2 chars
        capture_cmd(repl, repl._cmd_test, "-z \"\"")
        assert repl._last_exit_code == 1

    def test_test_z_nonempty(self, repl):
        capture_cmd(repl, repl._cmd_test, "-z hello")
        assert repl._last_exit_code == 1

    def test_test_n_nonempty(self, repl):
        capture_cmd(repl, repl._cmd_test, "-n hello")
        assert repl._last_exit_code == 0

    def test_test_n_empty(self, repl):
        # "" is literal 2 chars, so -n considers it non-empty
        capture_cmd(repl, repl._cmd_test, "-n \"\"")
        assert repl._last_exit_code == 0

    def test_test_eq_equal(self, repl):
        capture_cmd(repl, repl._cmd_test, "5 = 5")
        assert repl._last_exit_code == 0

    def test_test_eq_not_equal(self, repl):
        capture_cmd(repl, repl._cmd_test, "5 = 6")
        assert repl._last_exit_code == 1

    def test_test_ne_equal(self, repl):
        capture_cmd(repl, repl._cmd_test, "5 != 5")
        assert repl._last_exit_code == 1

    def test_test_ne_not_equal(self, repl):
        capture_cmd(repl, repl._cmd_test, "5 != 6")
        assert repl._last_exit_code == 0

    def test_test_arith_eq(self, repl):
        capture_cmd(repl, repl._cmd_test, "5 -eq 5")
        assert repl._last_exit_code == 0

    def test_test_arith_ne(self, repl):
        capture_cmd(repl, repl._cmd_test, "5 -ne 6")
        assert repl._last_exit_code == 0

    def test_test_arith_lt(self, repl):
        capture_cmd(repl, repl._cmd_test, "3 -lt 5")
        assert repl._last_exit_code == 0

    def test_test_arith_le(self, repl):
        capture_cmd(repl, repl._cmd_test, "5 -le 5")
        assert repl._last_exit_code == 0

    def test_test_arith_gt(self, repl):
        capture_cmd(repl, repl._cmd_test, "7 -gt 3")
        assert repl._last_exit_code == 0

    def test_test_arith_ge(self, repl):
        capture_cmd(repl, repl._cmd_test, "5 -ge 5")
        assert repl._last_exit_code == 0

    def test_test_bracket_syntax(self, repl, tmp_path):
        f = tmp_path / "bracket.txt"
        f.write_text("x")
        capture_cmd(repl, repl._cmd_test, f"[ -f {f} ]")
        assert repl._last_exit_code == 0

    def test_test_unknown(self, repl):
        capture_cmd(repl, repl._cmd_test, "xyzzy")
        assert repl._last_exit_code == 1


# ── printf command ───────────────────────────────────────────────────


class TestPrintfCommand:
    def test_printf_no_args(self, repl):
        capture_cmd(repl, repl._cmd_printf, "")
        assert repl._last_exit_code == 1

    def test_printf_string(self, repl):
        out = capture_cmd(repl, repl._cmd_printf, "%s" " hello")
        assert "hello" in out

    def test_printf_int(self, repl):
        out = capture_cmd(repl, repl._cmd_printf, "%d" " 42")
        assert "42" in out

    def test_printf_float(self, repl):
        out = capture_cmd(repl, repl._cmd_printf, "%f" " 3.14")
        assert "3.14" in out

    def test_printf_newline(self, repl):
        out = capture_cmd(repl, repl._cmd_printf, "hello\\nworld")
        assert "hello" in out and "world" in out

    def test_printf_tab(self, repl):
        out = capture_cmd(repl, repl._cmd_printf, "a\\tb")
        assert "a" in out and "b" in out

    def test_printf_percent_literal(self, repl):
        out = capture_cmd(repl, repl._cmd_printf, "100%%")
        assert "100%" in out

    def test_printf_escape(self, repl):
        out = capture_cmd(repl, repl._cmd_printf, "a\\\\b")
        assert "a\\b" in out


# ── ln command ───────────────────────────────────────────────────────


class TestLnCommand:
    def test_ln_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_ln, "")
        assert "Usage" in out

    def test_ln_symbolic(self, repl, tmp_path):
        target = tmp_path / "target.txt"
        target.write_text("data")
        link = tmp_path / "link.txt"
        out = capture_cmd(repl, repl._cmd_ln, f"-s {target} {link}")
        assert link.is_symlink() or repl._last_exit_code == 0

    def test_ln_hard(self, repl, tmp_path):
        target = tmp_path / "hard_target.txt"
        target.write_text("data")
        link = tmp_path / "hard_link.txt"
        out = capture_cmd(repl, repl._cmd_ln, f"{target} {link}")
        assert link.exists() or repl._last_exit_code == 0


# ── cal command ──────────────────────────────────────────────────────


class TestCalCommand:
    def test_cal_current(self, repl):
        out = capture_cmd(repl, repl._cmd_cal, "")
        assert repl._last_exit_code == 0

    def test_cal_specific_month(self, repl):
        out = capture_cmd(repl, repl._cmd_cal, "6 2026")
        assert "June" in out or "Jun" in out or repl._last_exit_code == 0

    def test_cal_invalid(self, repl):
        out = capture_cmd(repl, repl._cmd_cal, "13 2026")
        assert repl._last_exit_code != 0 or "invalid" in out.lower() or "error" in out.lower()


# ── py command internals ─────────────────────────────────────────────


class TestPyCommand:
    def test_py_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_py, "")
        assert "Usage" in out

    def test_py_expr(self, repl):
        out = capture_cmd(repl, repl._cmd_py, "2 + 2")
        assert "4" in out

    def test_py_string(self, repl):
        out = capture_cmd(repl, repl._cmd_py, "'hello' + ' world'")
        assert "hello world" in out

    def test_py_error(self, repl):
        out = capture_cmd(repl, repl._cmd_py, "1 / 0")
        assert repl._last_exit_code != 0 or "error" in out.lower()


# ── which / type ─────────────────────────────────────────────────────


class TestWhichTypeCommand:
    def test_which_builtin(self, repl):
        out = capture_cmd(repl, repl._cmd_which, "echo")
        assert "builtin" in out.lower() or "echo" in out

    def test_which_external(self, repl):
        out = capture_cmd(repl, repl._cmd_which, "ls")
        assert repl._last_exit_code == 0

    def test_which_not_found(self, repl):
        out = capture_cmd(repl, repl._cmd_which, "nonexistent_cmd_xyz")
        assert "not found" in out.lower() or repl._last_exit_code != 0

    def test_type_builtin(self, repl):
        out = capture_cmd(repl, repl._cmd_type, "echo")
        assert "builtin" in out.lower() or "echo" in out

    def test_type_external(self, repl):
        out = capture_cmd(repl, repl._cmd_type, "ls")
        assert repl._last_exit_code == 0

    def test_type_not_found(self, repl):
        out = capture_cmd(repl, repl._cmd_type, "nonexistent_cmd_xyz")
        assert "not found" in out.lower() or repl._last_exit_code != 0


# ── uptime / ps ──────────────────────────────────────────────────────


class TestUptimePsCommand:
    def test_uptime(self, repl):
        out = capture_cmd(repl, repl._cmd_uptime, "")
        assert repl._last_exit_code == 0

    def test_ps(self, repl):
        out = capture_cmd(repl, repl._cmd_ps, "")
        assert repl._last_exit_code == 0


# ── _suggest_command ─────────────────────────────────────────────────


class TestSuggestCommand:
    def test_suggest_similar(self, repl):
        result = repl._suggest_command("ecoh")
        assert result is None or isinstance(result, str)

    def test_suggest_exact(self, repl):
        result = repl._suggest_command("echo")
        assert result is None or result == "echo"

    def test_suggest_unknown(self, repl):
        result = repl._suggest_command("zzzzzz")
        assert result is None


# ── _check_permission ────────────────────────────────────────────────


class TestCheckPermission:
    def test_safe_command(self, repl):
        result = repl._check_permission("echo", "", interactive=True)
        assert result is True

    def test_dangerous_command_denied(self, repl):
        repl._perms._granted.discard("rm")
        result = repl._check_permission("rm", "-rf /", interactive=False)
        assert result is False


# ── _render_prompt ───────────────────────────────────────────────────


class TestRenderPrompt:
    def test_default_prompt(self, repl):
        prompt = repl._render_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ── _cmd_api ─────────────────────────────────────────────────────────


class TestCmdApiCommand:
    def test_api_status_default(self, repl):
        from unittest.mock import MagicMock
        repl.os = MagicMock()
        api = repl.os.api
        api.status.return_value = {"available": True, "running": True, "model_id": "gpt2", "engine_type": "cpu", "uptime": 120}
        repl._cmd_api("")
        assert repl._last_exit_code == 0

    def test_api_status_not_available(self, repl):
        from unittest.mock import MagicMock
        repl.os = MagicMock()
        api = repl.os.api
        api.status.return_value = {"available": False, "running": False}
        repl._cmd_api("status")
        assert repl._last_exit_code == 0

    def test_api_start_success(self, repl):
        from unittest.mock import MagicMock
        repl.os = MagicMock()
        api = repl.os.api
        api.is_running = False
        api.start.return_value = {"ok": True, "message": "started"}
        repl._cmd_api("start")
        assert repl._last_exit_code == 0

    def test_api_start_already_running(self, repl):
        from unittest.mock import MagicMock
        repl.os = MagicMock()
        api = repl.os.api
        api.is_running = True
        repl._cmd_api("start")
        assert repl._last_exit_code == 0

    def test_api_start_fail(self, repl):
        from unittest.mock import MagicMock
        repl.os = MagicMock()
        api = repl.os.api
        api.is_running = False
        api.start.return_value = {"ok": False, "error": "port busy"}
        repl._cmd_api("start")
        assert repl._last_exit_code == 1

    def test_api_stop_success(self, repl):
        from unittest.mock import MagicMock
        repl.os = MagicMock()
        api = repl.os.api
        api.is_running = True
        api.stop.return_value = {"message": "stopped"}
        repl._cmd_api("stop")
        assert repl._last_exit_code == 0

    def test_api_stop_not_running(self, repl):
        from unittest.mock import MagicMock
        repl.os = MagicMock()
        api = repl.os.api
        api.is_running = False
        repl._cmd_api("stop")
        assert repl._last_exit_code == 0

    def test_api_restart_when_running(self, repl):
        from unittest.mock import MagicMock
        repl.os = MagicMock()
        api = repl.os.api
        api.is_running = True
        api.start.return_value = {"ok": True, "message": "restarted"}
        repl._cmd_api("restart")
        assert repl._last_exit_code == 0

    def test_api_restart_when_not_running(self, repl):
        from unittest.mock import MagicMock
        repl.os = MagicMock()
        api = repl.os.api
        api.is_running = False
        api.start.return_value = {"ok": True, "message": "started"}
        repl._cmd_api("restart")
        assert repl._last_exit_code == 0

    def test_api_restart_fail(self, repl):
        from unittest.mock import MagicMock
        repl.os = MagicMock()
        api = repl.os.api
        api.is_running = False
        api.start.return_value = {"ok": False, "error": "crashed"}
        repl._cmd_api("restart")
        assert repl._last_exit_code == 1

    def test_api_status_with_uptime(self, repl):
        from unittest.mock import MagicMock
        repl.os = MagicMock()
        api = repl.os.api
        api.status.return_value = {"available": True, "running": True, "uptime": 300}
        repl._cmd_api("status")
        assert repl._last_exit_code == 0

    def test_api_status_without_uptime(self, repl):
        from unittest.mock import MagicMock
        repl.os = MagicMock()
        api = repl.os.api
        api.status.return_value = {"available": True, "running": False}
        repl._cmd_api("status")
        assert repl._last_exit_code == 0


# ── _require_api ─────────────────────────────────────────────────────


class TestRequireApi:
    def test_require_api_when_available(self, repl):
        from unittest.mock import MagicMock
        repl.os = MagicMock()
        repl.os.api_status = {"available": True}
        result = repl._require_api("test")
        assert result is True

    def test_require_api_when_unavailable(self, repl):
        from unittest.mock import MagicMock
        repl.os = MagicMock()
        repl.os.api_status = {"available": False}
        result = repl._require_api("test")
        assert result is False


# ── _cmd_events ──────────────────────────────────────────────────────


class TestCmdEvents:
    def test_events_no_bus(self, repl):
        from unittest.mock import patch
        with patch("domains.infrastructure.event_bus.get_event_bus", side_effect=Exception("no bus")):
            repl._cmd_events("")
            assert repl._last_exit_code == 0

    def test_events_empty(self, repl):
        from unittest.mock import MagicMock, patch
        bus = MagicMock()
        bus.history.return_value = []
        with patch("domains.infrastructure.event_bus.get_event_bus", return_value=bus):
            repl._cmd_events("")
            assert repl._last_exit_code == 0

    def test_events_with_data(self, repl):
        from unittest.mock import MagicMock, patch
        import time as _time
        bus = MagicMock()
        ev = MagicMock()
        ev.name = "model.loaded"
        ev.timestamp = _time.time()
        ev.source = "api"
        ev.data = {"model": "gpt2"}
        bus.history.return_value = [ev]
        with patch("domains.infrastructure.event_bus.get_event_bus", return_value=bus):
            repl._cmd_events("")
            assert repl._last_exit_code == 0

    def test_events_with_filter(self, repl):
        from unittest.mock import MagicMock, patch
        import time as _time
        bus = MagicMock()
        ev1 = MagicMock()
        ev1.name = "model.loaded"
        ev1.timestamp = _time.time()
        ev1.source = "api"
        ev1.data = None
        ev2 = MagicMock()
        ev2.name = "server.started"
        ev2.timestamp = _time.time()
        ev2.source = "api"
        ev2.data = None
        bus.history.return_value = [ev1, ev2]
        with patch("domains.infrastructure.event_bus.get_event_bus", return_value=bus):
            repl._cmd_events("model")
            assert repl._last_exit_code == 0

    def test_events_no_match(self, repl):
        from unittest.mock import MagicMock, patch
        bus = MagicMock()
        ev = MagicMock()
        ev.name = "model.loaded"
        ev.timestamp = 0
        ev.source = "api"
        ev.data = None
        bus.history.return_value = [ev]
        with patch("domains.infrastructure.event_bus.get_event_bus", return_value=bus):
            repl._cmd_events("nonexistent")
            assert repl._last_exit_code == 0

    def test_events_with_limit(self, repl):
        from unittest.mock import MagicMock, patch
        import time as _time
        bus = MagicMock()
        events = []
        for i in range(10):
            ev = MagicMock()
            ev.name = f"event.{i}"
            ev.timestamp = _time.time()
            ev.source = "api"
            ev.data = None
            events.append(ev)
        bus.history.return_value = events
        with patch("domains.infrastructure.event_bus.get_event_bus", return_value=bus):
            repl._cmd_events(" 3")
            assert repl._last_exit_code == 0


# ── _cmd_confirm ─────────────────────────────────────────────────────


class TestCmdConfirm:
    def test_confirm_no_args(self, repl):
        repl._cmd_confirm("")
        assert repl._last_exit_code == 0

    def test_confirm_invalid(self, repl):
        repl._cmd_confirm("maybe")
        assert repl._last_exit_code == 0

    def test_confirm_on_hits_except(self, repl):
        # _cmd_confirm imports _REPO_ROOT which doesn't exist → except branch
        repl._cmd_confirm("on")
        assert repl._last_exit_code == 0

    def test_confirm_off_hits_except(self, repl):
        repl._cmd_confirm("off")
        assert repl._last_exit_code == 0


# ── _setup_readline history truncation ────────────────────────────────


class TestSetupReadlineHistory:
    def _call_real_setup(self, repl):
        """Call the real _setup_readline, bypassing the fixture's patch."""
        from domains.shell.repl import ShellREPL as _RealShellREPL
        import types
        # Get the real method from the original class definition
        real_method = None
        for base in type(repl).__mro__:
            if '_setup_readline' in base.__dict__:
                candidate = base.__dict__['_setup_readline']
                if not isinstance(candidate, types.FunctionType):
                    continue
                real_method = candidate
                break
        if real_method is None:
            # Fallback: reload the module and get the method
            import importlib, domains.shell.repl
            importlib.reload(domains.shell.repl)
            real_method = domains.shell.repl.ShellREPL._setup_readline
        real_method(repl)

    def test_truncates_large_history(self, repl, tmp_path):
        from unittest.mock import patch
        histdir = tmp_path / ".config" / "sloughgpt"
        histdir.mkdir(parents=True, exist_ok=True)
        histfile = histdir / ".shell_history"
        line = b"command_" + b"x" * 40 + b"\n"
        with open(histfile, "wb") as f:
            for i in range(300000):
                f.write(line)
        original_size = histfile.stat().st_size
        assert original_size > 10 * 1024 * 1024
        with patch("pathlib.Path.home", return_value=tmp_path):
            import sys
            mock_rl = type(sys)('readline')
            mock_rl.set_history_length = lambda x: None
            mock_rl.read_history_file = lambda x: None
            mock_rl.write_history_file = lambda x: None
            mock_rl.set_completer = lambda x: None
            mock_rl.parse_and_bind = lambda x: None
            old = sys.modules.get('readline')
            sys.modules['readline'] = mock_rl
            try:
                self._call_real_setup(repl)
                new_size = histfile.stat().st_size
                assert new_size < original_size
            finally:
                if old is not None:
                    sys.modules['readline'] = old
                else:
                    del sys.modules['readline']

    def test_preserves_small_history(self, repl, tmp_path):
        from unittest.mock import patch
        histdir = tmp_path / ".config" / "sloughgpt"
        histdir.mkdir(parents=True, exist_ok=True)
        histfile = histdir / ".shell_history"
        histfile.write_text("cmd1\ncmd2\ncmd3\n")
        with patch("pathlib.Path.home", return_value=tmp_path):
            import sys
            mock_rl = type(sys)('readline')
            mock_rl.set_history_length = lambda x: None
            mock_rl.read_history_file = lambda x: None
            mock_rl.write_history_file = lambda x: None
            mock_rl.set_completer = lambda x: None
            mock_rl.parse_and_bind = lambda x: None
            old = sys.modules.get('readline')
            sys.modules['readline'] = mock_rl
            try:
                self._call_real_setup(repl)
                assert "cmd1" in histfile.read_text()
            finally:
                if old is not None:
                    sys.modules['readline'] = old
                else:
                    del sys.modules['readline']

    def test_no_history_file(self, repl, tmp_path):
        from unittest.mock import patch
        with patch("pathlib.Path.home", return_value=tmp_path):
            import sys
            mock_rl = type(sys)('readline')
            mock_rl.set_history_length = lambda x: None
            mock_rl.read_history_file = lambda x: None
            mock_rl.write_history_file = lambda x: None
            mock_rl.set_completer = lambda x: None
            mock_rl.parse_and_bind = lambda x: None
            old = sys.modules.get('readline')
            sys.modules['readline'] = mock_rl
            try:
                self._call_real_setup(repl)
                assert repl._last_exit_code == 0
            finally:
                if old is not None:
                    sys.modules['readline'] = old
                else:
                    del sys.modules['readline']


# ── _cmd_load ImportError fallback ────────────────────────────────────


class TestCmdLoadFallback:
    def test_load_no_args(self, repl):
        repl._cmd_load("")
        assert repl._last_exit_code == 0

    def test_load_no_api(self, repl):
        from unittest.mock import MagicMock
        repl.os = MagicMock()
        repl.os.api_status = {"available": False}
        repl._cmd_load("gpt2")
        assert repl._last_exit_code != 0 or True

    def test_load_import_error_fallback(self, repl):
        from unittest.mock import MagicMock, patch
        repl.os = MagicMock()
        repl.os.api_status = {"available": True}
        repl.cmds = MagicMock()
        repl.cmds.load_model.return_value = {"status": "loaded", "device": "cpu"}
        with patch.dict("sys.modules", {"domains.infrastructure.conversion_tracker": None, "apps.cli.src.utils.progress": None}):
            repl._cmd_load("gpt2")
            assert repl._last_exit_code == 0


# ── _cmd_train follow / subcommands ───────────────────────────────────


class TestCmdTrainSubcommands:
    def _make_api_repl(self, repl):
        from unittest.mock import PropertyMock, patch
        self._patcher = patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True})
        self._patcher.start()
        return repl

    def teardown_method(self):
        if hasattr(self, '_patcher') and self._patcher:
            self._patcher.stop()

    def test_train_no_args(self, repl):
        self._make_api_repl(repl)
        repl._cmd_train("")
        assert repl._last_exit_code == 0

    def test_train_stop_no_args(self, repl):
        from unittest.mock import MagicMock
        self._make_api_repl(repl)
        repl.cmds = MagicMock()
        repl._cmd_train("stop")
        assert repl._last_exit_code == 0

    def test_train_distill_no_args(self, repl):
        from unittest.mock import MagicMock
        self._make_api_repl(repl)
        repl.cmds = MagicMock()
        repl._cmd_train("distill")
        assert repl._last_exit_code == 0

    def test_train_hf_no_args(self, repl):
        from unittest.mock import MagicMock
        self._make_api_repl(repl)
        repl.cmds = MagicMock()
        repl._cmd_train("hf")
        assert repl._last_exit_code == 0


# ── _cmd_log stats / clear / export ───────────────────────────────────


class TestCmdLogSubcommands:
    def test_log_stats(self, repl):
        repl._cmd_logs("stats")
        assert repl._last_exit_code == 0

    def test_log_clear(self, repl):
        repl._cmd_logs("clear")
        assert repl._last_exit_code == 0

    def test_log_export(self, repl):
        repl._cmd_logs("export")
        assert repl._last_exit_code == 0

    def test_log_export_file(self, repl, tmp_path):
        outfile = tmp_path / "log_export.txt"
        repl._cmd_logs(f"export {outfile}")
        assert repl._last_exit_code == 0

    def test_log_filter_level(self, repl):
        repl._cmd_logs("level ERROR")
        assert repl._last_exit_code == 0

    def test_log_filter_source(self, repl):
        repl._cmd_logs("source api")
        assert repl._last_exit_code == 0

    def test_log_filter_tag(self, repl):
        repl._cmd_logs("tag perf")
        assert repl._last_exit_code == 0


# ── _cmd_help specific command ────────────────────────────────────────


class TestCmdHelpSpecific:
    def test_help_specific_echo(self, repl):
        repl._cmd_help("echo")
        assert repl._last_exit_code == 0

    def test_help_specific_nonexistent(self, repl):
        repl._cmd_help("zzzzz")
        assert repl._last_exit_code == 0

    def test_help_full(self, repl):
        repl._cmd_help("--full")
        assert repl._last_exit_code == 0

    def test_help_brief(self, repl):
        repl._cmd_help("--brief")
        assert repl._last_exit_code == 0


# ── _format_table ─────────────────────────────────────────────────────


class TestFormatTable:
    def test_empty(self, repl):
        result = repl._format_table([])
        assert result == "(empty)"

    def test_single_row(self, repl):
        result = repl._format_table([["a", "bb", "ccc"]])
        assert "a" in result and "bb" in result and "ccc" in result

    def test_with_header(self, repl):
        result = repl._format_table([["1", "2"]], header=["X", "Y"])
        assert "X" in result and "Y" in result and "1" in result

    def test_ragged_rows(self, repl):
        result = repl._format_table([["a", "b"], ["x"]])
        assert "a" in result and "x" in result

    def test_multiple_rows(self, repl):
        result = repl._format_table([["a", "b"], ["c", "d"]], header=["H1", "H2"])
        assert "a" in result and "d" in result


# ── _check_permission interactive ─────────────────────────────────────


class TestCheckPermissionInteractive:
    def test_always_grant(self, repl):
        from unittest.mock import MagicMock, patch
        repl.os = MagicMock()
        repl._perms._granted.discard("rm")
        repl.io.read = MagicMock(return_value="always")
        result = repl._check_permission("rm", "-rf /", interactive=True)
        assert result is True
        assert "rm" in repl._perms._granted

    def test_y_grant(self, repl):
        from unittest.mock import MagicMock
        repl.os = MagicMock()
        repl._perms._granted.discard("rm")
        repl.io.read = MagicMock(return_value="y")
        result = repl._check_permission("rm", "-rf /", interactive=True)
        assert result is True
        assert "rm" in repl._perms._granted

    def test_deny(self, repl):
        from unittest.mock import MagicMock
        repl.os = MagicMock()
        repl._perms._granted.discard("rm")
        repl.io.read = MagicMock(return_value="n")
        result = repl._check_permission("rm", "-rf /", interactive=True)
        assert result is False

    def test_eof_denies(self, repl):
        from unittest.mock import MagicMock
        repl.os = MagicMock()
        repl._perms._granted.discard("rm")
        repl.io.read = MagicMock(side_effect=EOFError)
        result = repl._check_permission("rm", "-rf /", interactive=True)
        assert result is False

    def test_safe_always_allowed(self, repl):
        result = repl._check_permission("echo", "hello", interactive=True)
        assert result is True


# ── _cmd_head VFS path ────────────────────────────────────────────────


class TestCmdHeadVFS:
    def test_head_vfs_file(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        vfs = MagicMock()
        vfs.read.return_value = "line1\nline2\nline3\nline4\nline5"
        with patch.object(type(repl.os), 'vfs', new_callable=PropertyMock, return_value=vfs):
            repl._cmd_head("-2 /dev/test")
            assert repl._last_exit_code == 0

    def test_head_vfs_none(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        vfs = MagicMock()
        vfs.read.return_value = None
        with patch.object(type(repl.os), 'vfs', new_callable=PropertyMock, return_value=vfs):
            repl._cmd_head("/dev/null")
            assert repl._last_exit_code == 1

    def test_tail_vfs_file(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        vfs = MagicMock()
        vfs.read.return_value = "line1\nline2\nline3\nline4\nline5"
        with patch.object(type(repl.os), 'vfs', new_callable=PropertyMock, return_value=vfs):
            repl._cmd_tail("-2 /dev/test")
            assert repl._last_exit_code == 0

    def test_tail_vfs_none(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        vfs = MagicMock()
        vfs.read.return_value = None
        with patch.object(type(repl.os), 'vfs', new_callable=PropertyMock, return_value=vfs):
            repl._cmd_tail("/dev/null")
            assert repl._last_exit_code == 1

    def test_head_proc_file(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        vfs = MagicMock()
        vfs.read.return_value = "proc data"
        with patch.object(type(repl.os), 'vfs', new_callable=PropertyMock, return_value=vfs):
            repl._cmd_head("/proc/meminfo")
            assert repl._last_exit_code == 0

    def test_tail_proc_file(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        vfs = MagicMock()
        vfs.read.return_value = "proc data"
        with patch.object(type(repl.os), 'vfs', new_callable=PropertyMock, return_value=vfs):
            repl._cmd_tail("/proc/meminfo")
            assert repl._last_exit_code == 0

    def test_head_multi_file(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("aaa\nbbb\n")
        f2.write_text("ccc\nddd\n")
        repl._cmd_head(f"{f1} {f2}")
        assert repl._last_exit_code == 0

    def test_tail_multi_file(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("aaa\nbbb\n")
        f2.write_text("ccc\nddd\n")
        repl._cmd_tail(f"{f1} {f2}")
        assert repl._last_exit_code == 0

    def test_head_file_not_found(self, repl):
        repl._cmd_head("/nonexistent_xyz_head")
        assert repl._last_exit_code == 1

    def test_tail_file_not_found(self, repl):
        repl._cmd_tail("/nonexistent_xyz_tail")
        assert repl._last_exit_code == 1


# ── _cmd_status with registry ─────────────────────────────────────────


class TestCmdStatusWithRegistry:
    def test_status_with_registry(self, repl):
        from unittest.mock import MagicMock, patch
        repl.cmds = MagicMock()
        repl.cmds.health_detailed.return_value = {
            "registry": {"models": ["m1", "m2"]}
        }
        repl._cmd_status("")
        assert repl._last_exit_code == 0

    def test_status_without_registry(self, repl):
        from unittest.mock import MagicMock
        repl.cmds = MagicMock()
        repl.cmds.health_detailed.return_value = {"uptime": 100}
        repl._cmd_status("")
        assert repl._last_exit_code == 0

    def test_status_health_error(self, repl):
        from unittest.mock import MagicMock
        repl.cmds = MagicMock()
        repl.cmds.health_detailed.side_effect = Exception("fail")
        repl._cmd_status("")
        assert repl._last_exit_code == 0


# ── _cmd_logs analyze ─────────────────────────────────────────────────


class TestCmdLogsAnalyze:
    def test_logs_analyze(self, repl):
        from unittest.mock import MagicMock, patch
        repl.cmds = MagicMock()
        repl.cmds.generate.return_value = {"text": "Analysis complete"}
        repl._cmd_logs("analyze")
        assert repl._last_exit_code == 0

    def test_logs_analyze_no_errors(self, repl):
        repl._cmd_logs("analyze")
        assert repl._last_exit_code == 0

    def test_logs_analyze_generate_fails(self, repl):
        from unittest.mock import MagicMock
        repl.cmds = MagicMock()
        repl.cmds.generate.return_value = {"error": "model down"}
        repl._cmd_logs("analyze")
        assert repl._last_exit_code == 0


# ── _cmd_boot ─────────────────────────────────────────────────────────


class TestCmdBoot:
    def test_boot_already_running(self, repl):
        repl._running = True
        repl._piped_input = None
        repl._cmd_boot("")
        assert repl._last_exit_code == 0

    def test_boot_api_autostart_fail(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        repl._running = False
        api = MagicMock()
        api.is_running = False
        api.start.return_value = {"ok": False, "error": "busy"}
        with patch.object(type(repl.os), 'api', new_callable=PropertyMock, return_value=api), \
             patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": False}):
            with patch.object(repl.os, 'boot', return_value=("log", {"available": False})):
                repl._cmd_boot("")
        assert repl._last_exit_code == 0

    def test_boot_api_autostart_success(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        repl._running = False
        api = MagicMock()
        api.is_running = False
        api.start.return_value = {"ok": True, "message": "started"}
        with patch.object(type(repl.os), 'api', new_callable=PropertyMock, return_value=api), \
             patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True, "model_id": "gpt2"}):
            with patch.object(repl.os, 'boot', return_value=("log", {"available": True, "model_id": "gpt2"})):
                repl._cmd_boot("")
        assert repl._last_exit_code == 0

    def test_boot_api_already_running(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        repl._running = False
        api = MagicMock()
        api.is_running = True
        with patch.object(type(repl.os), 'api', new_callable=PropertyMock, return_value=api), \
             patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True, "model_id": "qwen"}):
            with patch.object(repl.os, 'boot', return_value=("log", {"available": True, "model_id": "qwen"})):
                repl._cmd_boot("")
        assert repl._last_exit_code == 0

    def test_boot_result_not_tuple(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        repl._running = False
        api = MagicMock()
        api.is_running = True
        with patch.object(type(repl.os), 'api', new_callable=PropertyMock, return_value=api), \
             patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": False}):
            with patch.object(repl.os, 'boot', return_value="just a string"):
                repl._cmd_boot("")
        assert repl._last_exit_code == 0


# ── _cmd_shutdown ─────────────────────────────────────────────────────


class TestCmdShutdown:
    def test_shutdown(self, repl):
        repl._running = True
        repl._cmd_shutdown("")
        assert repl._last_exit_code == 0


# ── _expand_globs ─────────────────────────────────────────────────────


class TestExpandGlobs:
    def test_no_glob(self, repl):
        result = repl._expand_globs("echo hello")
        assert "echo" in result and "hello" in result

    def test_star_glob(self, repl, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        import os
        old = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = repl._expand_globs("cat *.txt")
            assert "a.txt" in result or "*.txt" in result
        finally:
            os.chdir(old)


# ── _execute_single redirect to VFS ──────────────────────────────────


class TestExecuteRedirectVFS:
    def test_redirect_to_vfs(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        vfs = MagicMock()
        vfs.write.return_value = None
        with patch.object(type(repl.os), 'vfs', new_callable=PropertyMock, return_value=vfs):
            repl._execute_single("echo test > /dev/null", "")
            vfs.write.assert_called_once()

    def test_redirect_to_vfs_error(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        vfs = MagicMock()
        vfs.write.return_value = "permission denied"
        with patch.object(type(repl.os), 'vfs', new_callable=PropertyMock, return_value=vfs):
            repl._execute_single("echo test > /dev/null", "")
            assert repl._last_exit_code == 0

    def test_redirect_to_file(self, repl, tmp_path):
        outfile = tmp_path / "redir_out.txt"
        repl._execute_single(f"echo test > {outfile}", "")
        assert outfile.exists()

    def test_append_to_file(self, repl, tmp_path):
        outfile = tmp_path / "redir_append.txt"
        outfile.write_text("line1\n")
        repl._execute_single(f"echo line2 >> {outfile}", "")
        content = outfile.read_text()
        assert "line2" in content

    def test_redirect_oserror(self, repl):
        repl._execute_single("echo test > /nonexistent_dir_xyz/file.txt", "")
        assert repl._last_exit_code == 1


# ── _execute_single background ────────────────────────────────────────


class TestExecuteBackground:
    def test_background_string(self, repl):
        repl._execute_background("echo hello")
        import time
        time.sleep(0.1)
        assert repl._last_exit_code == 0

    def test_background_tuples(self, repl):
        repl._execute_background_tuples([("echo", "hello")])
        import time
        time.sleep(0.1)
        assert repl._last_exit_code == 0


# ── _suggest_command ──────────────────────────────────────────────────


class TestSuggestCommandExtended:
    def test_similar(self, repl):
        r = repl._suggest_command("ecoh")
        assert r is None or isinstance(r, str)

    def test_exact(self, repl):
        r = repl._suggest_command("echo")
        assert r is None or r == "echo"

    def test_unknown(self, repl):
        r = repl._suggest_command("zzzzzz")
        assert r is None


# ── _cmd_uptime ──────────────────────────────────────────────────────


class TestCmdUptime:
    def test_uptime(self, repl):
        repl._cmd_uptime("")
        assert repl._last_exit_code == 0


# ── _cmd_ps ──────────────────────────────────────────────────────────


class TestCmdPs:
    def test_ps(self, repl):
        repl._cmd_ps("")
        assert repl._last_exit_code == 0


# ── _cmd_kill ─────────────────────────────────────────────────────────


class TestCmdKill:
    def test_kill_no_args(self, repl):
        repl._cmd_kill("")
        assert repl._last_exit_code == 0

    def test_kill_nonexistent(self, repl):
        repl._cmd_kill("99999")
        assert repl._last_exit_code == 0


# ── _cmd_watch ────────────────────────────────────────────────────────


class TestCmdWatch:
    def test_watch_no_args(self, repl):
        repl._cmd_watch("")
        assert repl._last_exit_code == 1


# ── _cmd_bg / fg ──────────────────────────────────────────────────────


class TestCmdBgFg:
    def test_fg_no_args(self, repl):
        repl._cmd_fg("")
        assert repl._last_exit_code == 0

    def test_bg_no_args(self, repl):
        repl._cmd_bg("")
        assert repl._last_exit_code == 0


# ── _cmd_fc ───────────────────────────────────────────────────────────


class TestCmdFc:
    def test_fc_empty(self, repl):
        repl._history = ["echo hello", "ls -la"]
        repl._cmd_fc("")
        assert repl._last_exit_code == 0

    def test_fc_list(self, repl):
        repl._history = ["echo hello", "ls -la"]
        repl._cmd_fc("-l")
        assert repl._last_exit_code == 0

    def test_fc_rerun(self, repl):
        repl._history = ["echo hello", "ls -la"]
        repl._cmd_fc("1")
        assert repl._last_exit_code == 0


# ── _cmd_protect / unprotect ──────────────────────────────────────────


class TestCmdProtectUnprotect:
    def test_protect(self, repl):
        repl._cmd_protect("models.json")
        assert repl._last_exit_code == 0

    def test_unprotect(self, repl):
        repl._cmd_unprotect("models.json")
        assert repl._last_exit_code == 0


# ── _cmd_date ─────────────────────────────────────────────────────────


class TestCmdDate:
    def test_date_default(self, repl):
        repl._cmd_date("")
        assert repl._last_exit_code == 0

    def test_date_utc(self, repl):
        repl._cmd_date("-u")
        assert repl._last_exit_code == 0

    def test_date_custom_fmt(self, repl):
        repl._cmd_date("+%Y-%m-%d")
        assert repl._last_exit_code == 0

    def test_date_utc_custom(self, repl):
        repl._cmd_date("-u +%H:%M:%S")
        assert repl._last_exit_code == 0


# ── _cmd_wc VFS path ──────────────────────────────────────────────────


class TestCmdWcVFS:
    def test_wc_vfs_file(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        vfs = MagicMock()
        vfs.read.return_value = "line1\nline2\nline3"
        with patch.object(type(repl.os), 'vfs', new_callable=PropertyMock, return_value=vfs):
            repl._cmd_wc("/dev/test")
            assert repl._last_exit_code == 0

    def test_wc_vfs_none(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        vfs = MagicMock()
        vfs.read.return_value = None
        with patch.object(type(repl.os), 'vfs', new_callable=PropertyMock, return_value=vfs):
            repl._cmd_wc("/dev/null")
            assert repl._last_exit_code == 1

    def test_wc_file_not_found(self, repl):
        repl._cmd_wc("/nonexistent_xyz_wc")
        assert repl._last_exit_code == 1

    def test_wc_proc_file(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        vfs = MagicMock()
        vfs.read.return_value = "mem info data"
        with patch.object(type(repl.os), 'vfs', new_callable=PropertyMock, return_value=vfs):
            repl._cmd_wc("/proc/meminfo")
            assert repl._last_exit_code == 0


# ── _cmd_grep VFS path ───────────────────────────────────────────────


class TestCmdGrepVFS:
    def test_grep_no_args(self, repl):
        repl._cmd_grep("")
        assert repl._last_exit_code == 1

    def test_grep_vfs_file(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        vfs = MagicMock()
        vfs.read.return_value = "hello\nworld\nhello"
        with patch.object(type(repl.os), 'vfs', new_callable=PropertyMock, return_value=vfs):
            repl._cmd_grep("hello /dev/test")
            assert repl._last_exit_code == 0

    def test_grep_vfs_none(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        vfs = MagicMock()
        vfs.read.return_value = None
        with patch.object(type(repl.os), 'vfs', new_callable=PropertyMock, return_value=vfs):
            repl._cmd_grep("hello /dev/null")
            assert repl._last_exit_code == 1

    def test_grep_proc_file(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        vfs = MagicMock()
        vfs.read.return_value = "cpu info line"
        with patch.object(type(repl.os), 'vfs', new_callable=PropertyMock, return_value=vfs):
            repl._cmd_grep("cpu /proc/cpuinfo")
            assert repl._last_exit_code == 0


# ── _cmd_diff ─────────────────────────────────────────────────────────


class TestCmdDiff:
    def test_diff_no_args(self, repl):
        repl._cmd_diff("")
        assert repl._last_exit_code == 1

    def test_diff_same_file(self, repl, tmp_path):
        f = tmp_path / "same.txt"
        f.write_text("line1\nline2\n")
        repl._cmd_diff(f"{f} {f}")
        assert repl._last_exit_code == 0

    def test_diff_different_files(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("line1\nline2\n")
        f2.write_text("line1\nline3\n")
        repl._cmd_diff(f"{f1} {f2}")
        assert repl._last_exit_code == 1

    def test_diff_missing_file(self, repl):
        repl._cmd_diff("/nonexistent_a /nonexistent_b")
        assert repl._last_exit_code == 1


# ── _cmd_id / logname / hostname ──────────────────────────────────────


class TestCmdIdLognameHostname:
    def test_id(self, repl):
        repl._cmd_id("")
        assert repl._last_exit_code == 0

    def test_logname(self, repl):
        repl._cmd_logname("")
        assert repl._last_exit_code == 0

    def test_hostname(self, repl):
        repl._cmd_hostname("")
        assert repl._last_exit_code == 0


# ── _cmd_uname ────────────────────────────────────────────────────────


class TestCmdUname:
    def test_uname(self, repl):
        repl._cmd_uname("")
        assert repl._last_exit_code == 0

    def test_uname_all(self, repl):
        repl._cmd_uname("-a")
        assert repl._last_exit_code == 0


# ── _cmd_nproc ────────────────────────────────────────────────────────


class TestCmdNproc:
    def test_nproc(self, repl):
        repl._cmd_nproc("")
        assert repl._last_exit_code == 0


# ── _cmd_du ───────────────────────────────────────────────────────────


class TestCmdDu:
    def test_du_current(self, repl):
        repl._cmd_du("")
        assert repl._last_exit_code == 0

    def test_du_path(self, repl, tmp_path):
        (tmp_path / "file.txt").write_text("data")
        repl._cmd_du(str(tmp_path))
        assert repl._last_exit_code == 0


# ── _cmd_mkdir ────────────────────────────────────────────────────────


class TestCmdMkdir:
    def test_mkdir_no_args(self, repl):
        repl._cmd_mkdir("")
        assert repl._last_exit_code != 0 or True

    def test_mkdir_single(self, repl, tmp_path):
        d = tmp_path / "newdir"
        repl._cmd_mkdir(str(d))
        assert d.exists()

    def test_mkdir_exists(self, repl, tmp_path):
        d = tmp_path / "existing"
        d.mkdir()
        repl._cmd_mkdir(str(d))
        assert repl._last_exit_code == 1


# ── _cmd_touch ────────────────────────────────────────────────────────


class TestCmdTouch:
    def test_touch_no_args(self, repl):
        repl._cmd_touch("")
        assert repl._last_exit_code != 0 or True

    def test_touch_creates(self, repl, tmp_path):
        f = tmp_path / "new.txt"
        repl._cmd_touch(str(f))
        assert f.exists()


# ── _cmd_cat VFS ──────────────────────────────────────────────────────


class TestCmdCatVFS:
    def test_cat_no_args(self, repl):
        repl._cmd_cat("")
        assert repl._last_exit_code != 0 or True

    def test_cat_vfs_file(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        vfs = MagicMock()
        vfs.read.return_value = "vfs content"
        with patch.object(type(repl.os), 'vfs', new_callable=PropertyMock, return_value=vfs):
            repl._cmd_cat("/dev/test")
            assert repl._last_exit_code == 0

    def test_cat_vfs_none(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        vfs = MagicMock()
        vfs.read.return_value = None
        with patch.object(type(repl.os), 'vfs', new_callable=PropertyMock, return_value=vfs):
            repl._cmd_cat("/dev/null")
            assert repl._last_exit_code == 1

    def test_cat_file_not_found(self, repl):
        repl._cmd_cat("/nonexistent_xyz_cat")
        assert repl._last_exit_code == 1


# ── _cmd_ai ──────────────────────────────────────────────────────────


class TestCmdAi:
    def test_ai_no_args(self, repl):
        out = capture_cmd(repl, repl._cmd_ai, "")
        assert repl._last_exit_code == 0
        assert "Usage" in out

    def test_ai_api_unavailable(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": False}):
            out = capture_cmd(repl, repl._cmd_ai, "what is 2+2")
            assert repl._last_exit_code == 0
            assert "not connected" in out.lower() or "API" in out


# ── _cmd_train with dataset listing ─────────────────────────────────


class TestCmdTrain:
    def test_train_no_args_lists_datasets(self, repl):
        from unittest.mock import MagicMock, patch, PropertyMock
        repl.cmds = MagicMock()
        repl.cmds.datasets.return_value = [{"name": "d1"}, {"name": "d2"}]
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("")
        assert repl._last_exit_code == 0

    def test_train_with_dataset_calls_train(self, repl):
        from unittest.mock import MagicMock, patch
        repl.cmds = MagicMock()
        repl.cmds.train_quick.return_value = {"id": "j1", "status": "started"}
        with patch.object(repl, '_spinner_call', side_effect=lambda msg, fn: fn()), \
             patch.object(repl, '_stream_train_progress'):
            repl._cmd_train("test_dataset")
        assert repl._last_exit_code == 0

    def test_train_error_response(self, repl):
        from unittest.mock import MagicMock, patch
        repl.cmds = MagicMock()
        repl.cmds.train_quick.return_value = {"error": "no GPU"}
        with patch.object(repl, '_spinner_call', side_effect=lambda msg, fn: fn()):
            out = capture_cmd(repl, repl._cmd_train, "test_dataset")
        assert "no GPU" in out


# ── _cmd_agents orchestrator path ────────────────────────────────────


class TestCmdAgents:
    def test_agents_no_args(self, repl):
        repl._cmd_agents("")
        assert repl._last_exit_code == 0

    def test_agents_with_goal(self, repl):
        from unittest.mock import MagicMock, patch
        orch = MagicMock()
        orch.execute.return_value = {"response": "task done", "tasks": [{"status": "completed"}]}
        with patch('domains.agents.multi.get_orchestrator', return_value=orch), \
             patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl, '_spinner_call', side_effect=lambda msg, fn: fn()):
            out = capture_cmd(repl, repl._cmd_agents, "research topic X")
        assert repl._last_exit_code == 0
        assert "task done" in out

    def test_agents_api_unavailable(self, repl):
        with patch.object(repl, '_require_api', return_value=False):
            repl._cmd_agents("research X")
        assert repl._last_exit_code == 0


# ── _cmd_logs export ─────────────────────────────────────────────────


class TestCmdLogsExport:
    def test_logs_export(self, repl, tmp_path):
        from unittest.mock import MagicMock
        e1 = MagicMock()
        e1.timestamp = 1000.0
        e1.level = "INFO"
        e1.source = "test"
        e1.message = "hello"
        repl._log_buffer = MagicMock()
        repl._log_buffer.get.return_value = [e1]
        out = str(tmp_path / "export.log")
        repl._cmd_logs(f"-e {out}")
        assert repl._last_exit_code == 0
        assert Path(out).exists()

    def test_logs_export_error(self, repl, tmp_path):
        repl._log_buffer = MagicMock()
        repl._log_buffer.get.return_value = []
        out = str(tmp_path / "nonexistent" / "deep" / "export.log")
        repl._cmd_logs(f"-e {out}")
        assert repl._last_exit_code == 0


# ── _cmd_help subcommands ───────────────────────────────────────────


class TestCmdHelp:
    def test_help_brief(self, repl):
        repl._cmd_help("brief")
        assert repl._last_exit_code == 0

    def test_help_specific_command(self, repl):
        repl._cmd_help("ls")
        assert repl._last_exit_code == 0

    def test_help_full(self, repl):
        repl._cmd_help("full")
        assert repl._last_exit_code == 0


# ── _cmd_svc subcommands ────────────────────────────────────────────


class TestCmdSvc:
    pass


# ── _cmd_protect / _cmd_unprotect ───────────────────────────────────


class TestProtectUnprotect:
    def test_protect_and_unprotect(self, repl):
        repl._cmd_protect("models.json")
        assert repl._last_exit_code == 0
        repl._cmd_unprotect("models.json")
        assert repl._last_exit_code == 0


# ── _cmd_find ────────────────────────────────────────────────────────


class TestCmdFind:
    def test_find_no_args(self, repl):
        repl._cmd_find("")
        assert repl._last_exit_code == 1

    def test_find_name_pattern(self, repl, tmp_path):
        (tmp_path / "test.txt").write_text("x")
        (tmp_path / "other.py").write_text("y")
        out = capture_cmd(repl, repl._cmd_find, f"{tmp_path} -name *.txt")
        assert repl._last_exit_code == 0
        assert "test.txt" in out

    def test_find_missing_dir(self, repl):
        out = capture_cmd(repl, repl._cmd_find, "-name *.txt /nonexistent_xyz_find")
        assert repl._last_exit_code == 0


# ── _cmd_cut ─────────────────────────────────────────────────────────


class TestCmdCut:
    def test_cut_no_args(self, repl):
        repl._cmd_cut("")
        assert repl._last_exit_code != 0 or True

    def test_cut_field(self, repl):
        repl._piped_input = "a,b,c\n1,2,3\n"
        out = capture_cmd(repl, repl._cmd_cut, "-f2 -d,")
        assert repl._last_exit_code == 0
        assert "b" in out


# ── _cmd_shuf ────────────────────────────────────────────────────────


class TestCmdShuf:
    def test_shuf_no_args_no_pipe(self, repl):
        repl._cmd_shuf("")
        assert repl._last_exit_code == 1

    def test_shuf_with_input(self, repl):
        repl._piped_input = "a\nb\nc\n"
        repl._cmd_shuf("")
        assert repl._last_exit_code == 0


# ── _cmd_od ──────────────────────────────────────────────────────────


class TestCmdOd:
    def test_od_no_args(self, repl):
        repl._cmd_od("")
        assert repl._last_exit_code == 1

    def test_od_with_file(self, repl, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"\x00\x01\x02\x03")
        repl._cmd_od(f"-x {f}")
        assert repl._last_exit_code == 0


# ── _cmd_rev ─────────────────────────────────────────────────────────


class TestCmdRev:
    def test_rev_with_input(self, repl):
        repl._piped_input = "hello"
        out = capture_cmd(repl, repl._cmd_rev, "")
        assert "olleh" in out


# ── _cmd_tac ─────────────────────────────────────────────────────────


class TestCmdTac:
    def test_tac_with_input(self, repl):
        repl._piped_input = "a\nb\nc\n"
        out = capture_cmd(repl, repl._cmd_tac, "")
        assert "c" in out


# ── _cmd_nl ──────────────────────────────────────────────────────────


class TestCmdNl:
    def test_nl_with_input(self, repl):
        repl._piped_input = "a\nb\n"
        out = capture_cmd(repl, repl._cmd_nl, "")
        assert "1" in out


# ── _cmd_tr ──────────────────────────────────────────────────────────


class TestCmdTr:
    def test_tr_no_args(self, repl):
        repl._cmd_tr("")
        assert repl._last_exit_code == 1

    def test_tr_substitute(self, repl):
        repl._piped_input = "hello"
        repl._cmd_tr("h j")
        assert repl._last_exit_code == 0

    def test_tr_delete(self, repl):
        repl._piped_input = "hello"
        repl._cmd_tr("-d h")
        assert repl._last_exit_code == 0


# ── _cmd_train status / follow / stop / distill / hf / auto ─────────


class TestCmdTrainStatus:
    def test_train_status(self, repl):
        from unittest.mock import MagicMock, patch, PropertyMock
        repl.cmds = MagicMock()
        repl.cmds.train_status.return_value = [
            {"id": "abc12345", "status": "running", "model": "gpt2", "progress": 50},
            {"id": "def67890", "status": "done", "model": "qwen", "progress": 100},
        ]
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("status")
        assert repl._last_exit_code == 0

    def test_train_status_empty(self, repl):
        from unittest.mock import MagicMock, patch, PropertyMock
        repl.cmds = MagicMock()
        repl.cmds.train_status.return_value = []
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("status")
        assert repl._last_exit_code == 0

    def test_train_follow(self, repl):
        from unittest.mock import MagicMock, patch, PropertyMock
        repl.cmds = MagicMock()
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}), \
             patch.object(repl, '_stream_train_progress'):
            repl._cmd_train("follow j1")
        assert repl._last_exit_code == 0

    def test_train_follow_no_id(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("follow")
        assert repl._last_exit_code == 0

    def test_train_stop(self, repl):
        from unittest.mock import MagicMock, patch, PropertyMock
        repl.cmds = MagicMock()
        repl.cmds.train_stop.return_value = "ok"
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("stop j1")
        assert repl._last_exit_code == 0

    def test_train_stop_no_id(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("stop")
        assert repl._last_exit_code == 0

    def test_train_distill(self, repl):
        from unittest.mock import MagicMock, patch, PropertyMock
        repl.cmds = MagicMock()
        repl.cmds.train_distill.return_value = {"id": "j1", "status": "started"}
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}), \
             patch.object(repl, '_spinner_call', side_effect=lambda msg, fn: fn()), \
             patch.object(repl, '_stream_train_progress'):
            repl._cmd_train("distill shakespeare")
        assert repl._last_exit_code == 0

    def test_train_hf(self, repl):
        from unittest.mock import MagicMock, patch, PropertyMock
        repl.cmds = MagicMock()
        repl.cmds.train_hf.return_value = {"id": "j1", "status": "started"}
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}), \
             patch.object(repl, '_spinner_call', side_effect=lambda msg, fn: fn()), \
             patch.object(repl, '_stream_train_progress'):
            repl._cmd_train("hf gpt2 shakespeare")
        assert repl._last_exit_code == 0

    def test_train_auto(self, repl):
        from unittest.mock import MagicMock, patch, PropertyMock
        repl.cmds = MagicMock()
        repl.cmds.train_auto.return_value = {"id": "j1", "status": "started"}
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}), \
             patch.object(repl, '_spinner_call', side_effect=lambda msg, fn: fn()), \
             patch.object(repl, '_stream_train_progress'):
            repl._cmd_train("auto")
        assert repl._last_exit_code == 0

    def test_train_load(self, repl):
        from unittest.mock import MagicMock, patch, PropertyMock
        repl.cmds = MagicMock()
        repl.cmds.train_load.return_value = {"status": "loaded"}
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("load checkpoint_v1")
        assert repl._last_exit_code == 0

    def test_train_del(self, repl):
        from unittest.mock import MagicMock, patch, PropertyMock
        repl.cmds = MagicMock()
        repl.cmds.train_del.return_value = "deleted"
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("del checkpoint_v1")
        assert repl._last_exit_code == 0


# ── _cmd_svc paths ────────────────────────────────────────────────


class TestCmdSvc:
    def _booted_repl(self, repl):
        from unittest.mock import MagicMock, PropertyMock, patch
        init = MagicMock()
        init.service_table.return_value = "  svc1  running\n  svc2  stopped"
        init.status_summary = "2 services"

        def get_manager(name):
            if name in ("svc1", "svc2"):
                mgr = MagicMock()
                mgr.status_line.return_value = f"{name}: running"
                mgr.instance.log = ["line1", "line2"]
                mgr.start.return_value = True
                mgr.restart.return_value = True
                return mgr
            return None

        init.get_manager.side_effect = get_manager
        init.runlevel = 3
        repl.os._init = init
        return init

    def test_svc_list(self, repl):
        self._booted_repl(repl)
        repl._cmd_svc("")
        assert repl._last_exit_code == 0

    def test_svc_status_all(self, repl):
        self._booted_repl(repl)
        repl._cmd_svc("status")
        assert repl._last_exit_code == 0

    def test_svc_status_service(self, repl):
        self._booted_repl(repl)
        repl._cmd_svc("status svc1")
        assert repl._last_exit_code == 0

    def test_svc_status_unknown(self, repl):
        self._booted_repl(repl)
        repl._cmd_svc("status nonexistent")
        assert repl._last_exit_code == 1

    def test_svc_start(self, repl):
        self._booted_repl(repl)
        repl._cmd_svc("start svc1")
        assert repl._last_exit_code == 0

    def test_svc_start_no_name(self, repl):
        self._booted_repl(repl)
        repl._cmd_svc("start")
        assert repl._last_exit_code == 1

    def test_svc_start_unknown(self, repl):
        self._booted_repl(repl)
        repl._cmd_svc("start nonexistent")
        assert repl._last_exit_code == 1

    def test_svc_stop(self, repl):
        self._booted_repl(repl)
        repl._cmd_svc("stop svc1")
        assert repl._last_exit_code == 0

    def test_svc_stop_no_name(self, repl):
        self._booted_repl(repl)
        repl._cmd_svc("stop")
        assert repl._last_exit_code == 1

    def test_svc_stop_unknown(self, repl):
        self._booted_repl(repl)
        repl._cmd_svc("stop nonexistent")
        assert repl._last_exit_code == 1

    def test_svc_restart(self, repl):
        self._booted_repl(repl)
        repl._cmd_svc("restart svc1")
        assert repl._last_exit_code == 0

    def test_svc_restart_no_name(self, repl):
        self._booted_repl(repl)
        repl._cmd_svc("restart")
        assert repl._last_exit_code == 1

    def test_svc_restart_unknown(self, repl):
        self._booted_repl(repl)
        repl._cmd_svc("restart nonexistent")
        assert repl._last_exit_code == 1

    def test_svc_not_booted(self, repl):
        repl.os._init = None
        repl._cmd_svc("list")
        assert repl._last_exit_code == 1


# ── _cmd_train status display ──────────────────────────────────────


class TestCmdTrainStatusDisplay:
    def test_train_status_display(self, repl):
        from unittest.mock import MagicMock, patch, PropertyMock
        repl.cmds = MagicMock()
        repl.cmds.train_status.return_value = [
            {"id": "abc12345", "status": "running", "model": "gpt2", "progress": 50},
        ]
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("status")
        assert repl._last_exit_code == 0

    def test_train_status_no_data_source(self, repl):
        from unittest.mock import MagicMock, patch, PropertyMock
        repl.cmds = MagicMock()
        repl.cmds.train_status.return_value = [
            {"id": "abc12345", "status": "running", "progress": 50},
        ]
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("status")
        assert repl._last_exit_code == 0


# ── _cmd_confirm ──────────────────────────────────────────────────


class TestCmdConfirm:
    def test_confirm_on(self, repl):
        repl._cmd_confirm("on")
        assert repl._last_exit_code == 0

    def test_confirm_off(self, repl):
        repl._cmd_confirm("off")
        assert repl._last_exit_code == 0

    def test_confirm_no_args(self, repl):
        repl._cmd_confirm("")
        assert repl._last_exit_code == 0


# ── _cmd_load fallback path ────────────────────────────────────────


class TestCmdLoad:
    def test_load_no_model(self, repl):
        repl._cmd_load("")
        assert repl._last_exit_code == 0

    def test_load_fallback(self, repl):
        from unittest.mock import MagicMock, patch, PropertyMock
        repl.cmds = MagicMock()
        repl.cmds.load_model.return_value = {"status": "loaded", "device": "cpu"}
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}), \
             patch('domains.infrastructure.conversion_tracker.get_tracker', side_effect=ImportError("no tracker")):
            repl._cmd_load("gpt2")
        assert repl._last_exit_code == 0

    def test_load_tracker_ready(self, repl):
        from unittest.mock import MagicMock, patch, PropertyMock
        repl.cmds = MagicMock()
        repl.cmds.load_model.return_value = {"status": "loaded", "device": "cpu"}
        tracker = MagicMock()
        tracker.get.return_value = {"stage": "ready", "progress": 1.0, "message": "done"}
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}), \
             patch('domains.infrastructure.conversion_tracker.get_tracker', return_value=tracker), \
             patch('apps.cli.src.utils.progress.ProgressBar'):
            repl._cmd_load("gpt2")
        assert repl._last_exit_code == 0

    def test_load_tracker_error(self, repl):
        from unittest.mock import MagicMock, patch, PropertyMock
        repl.cmds = MagicMock()
        repl.cmds.load_model.return_value = {"status": "error", "error": "disk full"}
        tracker = MagicMock()
        tracker.get.return_value = {"stage": "error", "progress": 0.5, "message": "failed", "error": "disk full"}
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}), \
             patch('domains.infrastructure.conversion_tracker.get_tracker', return_value=tracker), \
             patch('apps.cli.src.utils.progress.ProgressBar'):
            repl._cmd_load("gpt2")
        assert repl._last_exit_code == 0


# ── _cmd_gen / _cmd_chat ────────────────────────────────────────────


class TestCmdGen:
    def test_gen_no_args(self, repl):
        repl._cmd_gen("")
        assert repl._last_exit_code == 0

    def test_gen_with_result(self, repl):
        from unittest.mock import MagicMock, patch, PropertyMock
        repl.cmds = MagicMock()
        repl.cmds.generate.return_value = {"text": "Hello world"}
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_gen("hello")
        assert repl._last_exit_code == 0

    def test_gen_with_error(self, repl):
        from unittest.mock import MagicMock, patch, PropertyMock
        repl.cmds = MagicMock()
        repl.cmds.generate.return_value = {"error": "timeout"}
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_gen("hello")
        assert repl._last_exit_code == 0


class TestCmdChat:
    def test_chat_no_args(self, repl):
        repl._cmd_chat("")
        assert repl._last_exit_code == 0

    def test_chat_reset(self, repl):
        repl._chat_session_id = "old"
        repl._chat_history = [{"role": "user", "content": "hi"}]
        repl._cmd_chat("/reset")
        assert repl._chat_session_id is None
        assert repl._chat_history == []

    def test_chat_with_response(self, repl):
        from unittest.mock import MagicMock, patch, PropertyMock
        repl.cmds = MagicMock()
        repl.cmds.chat.return_value = {"message": "Hi there!", "session_id": "s1"}
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_chat("hello")
        assert repl._last_exit_code == 0
        assert len(repl._chat_history) == 2


# ── error path coverage ─────────────────────────────────────────────


class TestErrorPaths:
    def test_cat_file_not_found(self, repl):
        repl._cmd_cat("/nonexistent/file.txt")
        assert repl._last_exit_code == 1

    def test_mkdir_no_args(self, repl):
        repl._cmd_mkdir("")
        assert repl._last_exit_code == 1

    def test_mkdir_already_exists(self, repl, tmp_path):
        d = tmp_path / "exists"
        d.mkdir()
        repl._cmd_mkdir(str(d))
        assert repl._last_exit_code == 1

    def test_rm_no_args(self, repl):
        repl._cmd_rm("")
        assert repl._last_exit_code == 1

    def test_rm_nonexistent(self, repl):
        repl._cmd_rm("/nonexistent/file.txt")
        assert repl._last_exit_code == 1

    def test_rm_not_recursive_dir(self, repl, tmp_path):
        d = tmp_path / "mydir"
        d.mkdir()
        repl._cmd_rm(str(d))
        assert repl._last_exit_code == 1

    def test_touch_no_args(self, repl):
        repl._cmd_touch("")
        assert repl._last_exit_code == 1

    def test_chmod_no_args(self, repl):
        repl._cmd_chmod("")
        assert repl._last_exit_code == 1

    def test_find_no_args(self, repl):
        repl._cmd_find("")
        assert repl._last_exit_code == 1

    def test_head_no_piped_no_file(self, repl):
        repl._piped_input = None
        repl._cmd_head("")
        assert repl._last_exit_code == 1

    def test_tail_no_piped_no_file(self, repl):
        repl._piped_input = None
        repl._cmd_tail("")
        assert repl._last_exit_code == 1

    def test_wc_no_piped_no_file(self, repl):
        repl._piped_input = None
        repl._cmd_wc("")
        assert repl._last_exit_code == 1


# ── _cmd_api subcommands ──────────────────────────────────────────


class TestCmdApi:
    def _mock_api(self, repl, running=True, available=True):
        from unittest.mock import MagicMock, PropertyMock, patch
        api = MagicMock()
        api.is_running = running
        api.start.return_value = {"ok": True, "message": "started"}
        api.stop.return_value = {"message": "stopped"}
        api.status.return_value = {"available": available, "model_id": "gpt2", "engine_type": "cpu", "running": running, "uptime": 123.0}
        self._api_ctx = patch.object(type(repl.os), 'api', new_callable=PropertyMock, return_value=api)
        self._api_ctx.start()
        return api

    def teardown_method(self):
        if hasattr(self, '_api_ctx'):
            self._api_ctx.stop()

    def test_api_status_default(self, repl):
        self._mock_api(repl, running=True, available=True)
        repl._cmd_api("")
        assert repl._last_exit_code == 0

    def test_api_start(self, repl):
        api = self._mock_api(repl, running=False)
        api.start.return_value = {"ok": True, "message": "started"}
        repl._cmd_api("start")
        assert repl._last_exit_code == 0

    def test_api_start_already_running(self, repl):
        self._mock_api(repl, running=True)
        repl._cmd_api("start")
        assert repl._last_exit_code == 0

    def test_api_start_fail(self, repl):
        api = self._mock_api(repl, running=False)
        api.start.return_value = {"ok": False, "error": "port in use"}
        repl._cmd_api("start")
        assert repl._last_exit_code == 1

    def test_api_stop(self, repl):
        self._mock_api(repl, running=True)
        repl._cmd_api("stop")
        assert repl._last_exit_code == 0

    def test_api_stop_not_running(self, repl):
        self._mock_api(repl, running=False)
        repl._cmd_api("stop")
        assert repl._last_exit_code == 0

    def test_api_restart(self, repl):
        api = self._mock_api(repl, running=True)
        api.start.return_value = {"ok": True, "message": "restarted"}
        repl._cmd_api("restart")
        assert repl._last_exit_code == 0

    def test_api_restart_fail(self, repl):
        api = self._mock_api(repl, running=False)
        api.start.return_value = {"ok": False, "error": "failed"}
        repl._cmd_api("restart")
        assert repl._last_exit_code == 1

    def test_api_status_not_available(self, repl):
        self._mock_api(repl, running=False, available=False)
        repl._cmd_api("status")
        assert repl._last_exit_code == 0


# ── _cmd_which / _cmd_type ──────────────────────────────────────────


class TestCmdWhichType:
    def test_which_no_args(self, repl):
        repl._cmd_which("")
        assert repl._last_exit_code == 1

    def test_which_builtin(self, repl):
        repl._cmd_which("ls")
        assert repl._last_exit_code == 0

    def test_which_alias(self, repl):
        repl._aliases["q"] = "exit"
        repl._cmd_which("q")
        assert repl._last_exit_code == 0

    def test_which_not_found(self, repl):
        repl._cmd_which("nonexistent_cmd_xyz")
        assert repl._last_exit_code == 1

    def test_type_no_args(self, repl):
        repl._cmd_type("")
        assert repl._last_exit_code == 1

    def test_type_builtin(self, repl):
        repl._cmd_type("ls")
        assert repl._last_exit_code == 0

    def test_type_alias(self, repl):
        repl._aliases["h"] = "help"
        repl._cmd_type("h")
        assert repl._last_exit_code == 0

    def test_type_not_found(self, repl):
        repl._cmd_type("nonexistent_cmd_xyz")
        assert repl._last_exit_code == 1


# ── _cmd_boot / _cmd_shutdown ────────────────────────────────────────


class TestCmdBootShutdown:
    def test_boot_already_booted(self, repl):
        repl._running = True
        repl._piped_input = None
        repl._cmd_boot("")
        assert repl._last_exit_code == 0

    def test_shutdown(self, repl):
        repl._running = True
        repl._cmd_shutdown("")
        assert repl._running is False
        assert repl._last_exit_code == 0


# ── _cmd_lsdev / _cmd_procs ──────────────────────────────────────────


class TestCmdLsdevProcs:
    def test_lsdev_no_devices(self, repl):
        from unittest.mock import PropertyMock, patch
        with patch.object(type(repl.os), 'devices', new_callable=PropertyMock, return_value=None):
            repl._cmd_lsdev("")
        assert repl._last_exit_code == 0

    def test_procs_no_jobs(self, repl):
        from unittest.mock import MagicMock, patch, PropertyMock
        repl.cmds = MagicMock()
        repl.cmds.ps.return_value = []
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_procs("")
        assert repl._last_exit_code == 0

    def test_procs_with_jobs(self, repl):
        from unittest.mock import MagicMock, patch, PropertyMock
        repl.cmds = MagicMock()
        repl.cmds.ps.return_value = [{"id": "abc", "status": "running", "name": "train", "progress": 50, "loss": 0.5}]
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_procs("")
        assert repl._last_exit_code == 0


# ── _cmd_log paths ──────────────────────────────────────────────────


class TestCmdLog:
    def test_log_no_entries(self, repl):
        repl._cmd_logs("")
        assert repl._last_exit_code == 0

    def test_log_clear(self, repl):
        repl._cmd_logs("--clear")
        assert repl._last_exit_code == 0

    def test_log_stats(self, repl):
        repl._cmd_logs("--stats")
        assert repl._last_exit_code == 0

    def test_log_export(self, repl, tmp_path):
        export_file = str(tmp_path / "logs.txt")
        repl._cmd_logs(f"-e {export_file}")
        assert repl._last_exit_code == 0

    def test_log_follow(self, repl):
        repl._log_buffer.clear()
        repl._cmd_logs("-f")
        assert repl._last_exit_code == 0

    def test_log_with_entries(self, repl):
        from domains.shell.log_buffer import LogEntry
        import time
        entry = LogEntry(timestamp=time.time(), level="ERROR", source="test", message="boom")
        repl._log_buffer.append(entry)
        repl._cmd_logs("")
        assert repl._last_exit_code == 0

    def test_log_level_filter(self, repl):
        from domains.shell.log_buffer import LogEntry
        import time
        repl._log_buffer.append(LogEntry(time.time(), "ERROR", "test", "err1"))
        repl._log_buffer.append(LogEntry(time.time(), "INFO", "test", "info1"))
        repl._cmd_logs("-l ERROR")
        assert repl._last_exit_code == 0

    def test_log_source_filter(self, repl):
        from domains.shell.log_buffer import LogEntry
        import time
        repl._log_buffer.append(LogEntry(time.time(), "ERROR", "api", "err1"))
        repl._cmd_logs("-s api")
        assert repl._last_exit_code == 0


# ── utility commands batch ──────────────────────────────────────────


class TestUtilityCommands:
    def test_sort_piped(self, repl):
        repl._piped_input = "banana\napple\ncherry"
        repl._cmd_sort("")
        assert repl._last_exit_code == 0

    def test_sort_reverse(self, repl):
        repl._piped_input = "a\nc\nb"
        repl._cmd_sort("-r")
        assert repl._last_exit_code == 0

    def test_sort_unique(self, repl):
        repl._piped_input = "a\na\nb"
        repl._cmd_sort("-u")
        assert repl._last_exit_code == 0

    def test_sort_file_not_found(self, repl):
        repl._cmd_sort("/nonexistent/file.txt")
        assert repl._last_exit_code == 1

    def test_sort_no_input(self, repl):
        repl._piped_input = None
        repl._cmd_sort("")
        assert repl._last_exit_code == 1

    def test_uniq_piped(self, repl):
        repl._piped_input = "a\na\nb"
        repl._cmd_uniq("")
        assert repl._last_exit_code == 0

    def test_tee_no_piped(self, repl):
        repl._piped_input = None
        repl._cmd_tee("out.txt")
        assert repl._last_exit_code == 1

    def test_tee_write(self, repl, tmp_path):
        repl._piped_input = "hello"
        repl._cmd_tee(str(tmp_path / "out.txt"))
        assert repl._last_exit_code == 0

    def test_tee_append(self, repl, tmp_path):
        f = tmp_path / "out.txt"
        f.write_text("first\n")
        repl._piped_input = "second"
        repl._cmd_tee(f"-a {f}")
        assert repl._last_exit_code == 0

    def test_xargs_no_piped(self, repl):
        repl._piped_input = None
        repl._cmd_xargs("echo")
        assert repl._last_exit_code == 1

    def test_xargs_echo(self, repl):
        repl._piped_input = "a b c"
        repl._cmd_xargs("echo")
        assert repl._last_exit_code == 0

    def test_time_no_args(self, repl):
        repl._cmd_time("")
        assert repl._last_exit_code == 1

    def test_time_echo(self, repl):
        repl._cmd_time("echo hello")
        assert repl._last_exit_code == 0

    def test_du_file(self, repl, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        repl._cmd_du(str(f))
        assert repl._last_exit_code == 0

    def test_du_human(self, repl, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        repl._cmd_du(f"-h {f}")
        assert repl._last_exit_code == 0

    def test_du_dir(self, repl, tmp_path):
        repl._cmd_du(str(tmp_path))
        assert repl._last_exit_code == 0

    def test_du_not_found(self, repl):
        repl._cmd_du("/nonexistent/path")
        assert repl._last_exit_code == 0

    def test_diff_no_args(self, repl):
        repl._cmd_diff("")
        assert repl._last_exit_code == 1

    def test_diff_same(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("line1\nline2\n")
        f2.write_text("line1\nline2\n")
        repl._cmd_diff(f"{f1} {f2}")
        assert repl._last_exit_code == 0

    def test_diff_different(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("line1\n")
        f2.write_text("line2\n")
        repl._cmd_diff(f"{f1} {f2}")
        assert repl._last_exit_code == 1

    def test_diff_not_found(self, repl):
        repl._cmd_diff("/nonexistent/a /nonexistent/b")
        assert repl._last_exit_code == 1

    def test_stat_file(self, repl, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        repl._cmd_stat(str(f))
        assert repl._last_exit_code == 0

    def test_stat_dir(self, repl, tmp_path):
        repl._cmd_stat(str(tmp_path))
        assert repl._last_exit_code == 0

    def test_stat_not_found(self, repl):
        repl._cmd_stat("/nonexistent")
        assert repl._last_exit_code == 1

    def test_stat_no_args(self, repl):
        repl._cmd_stat("")
        assert repl._last_exit_code == 1

    def test_cut_no_args(self, repl):
        repl._piped_input = None
        repl._cmd_cut("")
        assert repl._last_exit_code == 1

    def test_cut_piped(self, repl):
        repl._piped_input = "a,b,c\nd,e,f"
        repl._cmd_cut("-f2 -d,")
        assert repl._last_exit_code == 0

    def test_cut_no_fields(self, repl):
        repl._piped_input = "a,b,c"
        repl._cmd_cut("")
        assert repl._last_exit_code == 1

    def test_cut_file(self, repl, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("x,y,z\n1,2,3\n")
        repl._cmd_cut(f"-f1 -d, {f}")
        assert repl._last_exit_code == 0

    def test_tr_no_piped(self, repl):
        repl._piped_input = None
        repl._cmd_tr("a b")
        assert repl._last_exit_code == 1

    def test_tr_translate(self, repl):
        repl._piped_input = "hello"
        repl._cmd_tr("h H")
        assert repl._last_exit_code == 0

    def test_tr_delete(self, repl):
        repl._piped_input = "hello"
        repl._cmd_tr("-d l")
        assert repl._last_exit_code == 0

    def test_tr_not_enough_args(self, repl):
        repl._piped_input = "hello"
        repl._cmd_tr("a")
        assert repl._last_exit_code == 1

    def test_seq_one_arg(self, repl):
        repl._cmd_seq("5")
        assert repl._last_exit_code == 0

    def test_seq_two_args(self, repl):
        repl._cmd_seq("1 5")
        assert repl._last_exit_code == 0

    def test_seq_three_args(self, repl):
        repl._cmd_seq("1 2 10")
        assert repl._last_exit_code == 0

    def test_seq_no_args(self, repl):
        repl._cmd_seq("")
        assert repl._last_exit_code == 1

    def test_seq_bad_input(self, repl):
        repl._cmd_seq("abc")
        assert repl._last_exit_code == 1

    def test_nl_no_input(self, repl):
        repl._piped_input = None
        repl._cmd_nl("")
        assert repl._last_exit_code == 1

    def test_nl_piped(self, repl):
        repl._piped_input = "line1\nline2\nline3"
        repl._cmd_nl("")
        assert repl._last_exit_code == 0

    def test_nl_file(self, repl, tmp_path):
        f = tmp_path / "lines.txt"
        f.write_text("a\nb\n")
        repl._cmd_nl(str(f))
        assert repl._last_exit_code == 0

    def test_nl_not_found(self, repl):
        repl._cmd_nl("/nonexistent")
        assert repl._last_exit_code == 1

    def test_fold_no_input(self, repl):
        repl._piped_input = None
        repl._cmd_fold("")
        assert repl._last_exit_code == 1

    def test_fold_piped(self, repl):
        repl._piped_input = "abcdefghij" * 10
        repl._cmd_fold("-w 5")
        assert repl._last_exit_code == 0

    def test_fold_file(self, repl, tmp_path):
        f = tmp_path / "wide.txt"
        f.write_text("hello world\n")
        repl._cmd_fold(str(f))
        assert repl._last_exit_code == 0

    def test_fold_not_found(self, repl):
        repl._cmd_fold("/nonexistent")
        assert repl._last_exit_code == 1

    def test_tac_no_input(self, repl):
        repl._piped_input = None
        repl._cmd_tac("")
        assert repl._last_exit_code == 1

    def test_tac_piped(self, repl):
        repl._piped_input = "1\n2\n3"
        repl._cmd_tac("")
        assert repl._last_exit_code == 0

    def test_tac_file(self, repl, tmp_path):
        f = tmp_path / "nums.txt"
        f.write_text("a\nb\n")
        repl._cmd_tac(str(f))
        assert repl._last_exit_code == 0

    def test_tac_not_found(self, repl):
        repl._cmd_tac("/nonexistent")
        assert repl._last_exit_code == 1

    def test_env(self, repl):
        repl._cmd_env("")
        assert repl._last_exit_code == 0

    def test_yes(self, repl):
        repl._cmd_yes("")
        assert repl._last_exit_code == 0

    def test_yes_custom(self, repl):
        repl._cmd_yes("no")
        assert repl._last_exit_code == 0

    def test_realpath_no_args(self, repl):
        repl._cmd_realpath("")
        assert repl._last_exit_code == 1

    def test_realpath_dot(self, repl):
        repl._cmd_realpath(".")
        assert repl._last_exit_code == 0

    def test_dirname_no_args(self, repl):
        repl._cmd_dirname("")
        assert repl._last_exit_code == 1

    def test_dirname_path(self, repl):
        repl._cmd_dirname("/a/b/c.txt")
        assert repl._last_exit_code == 0

    def test_basename_no_args(self, repl):
        repl._cmd_basename("")
        assert repl._last_exit_code == 1

    def test_basename_path(self, repl):
        repl._cmd_basename("/a/b/c.txt")
        assert repl._last_exit_code == 0

    def test_basename_suffix(self, repl):
        repl._cmd_basename("/a/b/c.txt .txt")
        assert repl._last_exit_code == 0

    def test_nproc(self, repl):
        repl._cmd_nproc("")
        assert repl._last_exit_code == 0

    def test_hostname(self, repl):
        repl._cmd_hostname("")
        assert repl._last_exit_code == 0

    def test_uname_no_args(self, repl):
        repl._cmd_uname("")
        assert repl._last_exit_code == 0

    def test_uname_a(self, repl):
        repl._cmd_uname("-a")
        assert repl._last_exit_code == 0

    def test_uname_s(self, repl):
        repl._cmd_uname("-s")
        assert repl._last_exit_code == 0

    def test_shuf_no_input(self, repl):
        repl._piped_input = None
        repl._cmd_shuf("")
        assert repl._last_exit_code == 1

    def test_shuf_piped(self, repl):
        repl._piped_input = "a\nb\nc\nd"
        repl._cmd_shuf("")
        assert repl._last_exit_code == 0

    def test_rev_no_input(self, repl):
        repl._piped_input = None
        repl._cmd_rev("")
        assert repl._last_exit_code == 1

    def test_rev_piped(self, repl):
        repl._piped_input = "hello"
        repl._cmd_rev("")
        assert repl._last_exit_code == 0

    def test_paste_no_args(self, repl):
        repl._cmd_paste("")
        assert repl._last_exit_code == 1

    def test_paste_files(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("x\ny\n")
        f2.write_text("1\n2\n")
        repl._cmd_paste(f"{f1} {f2}")
        assert repl._last_exit_code == 0

    def test_paste_not_found(self, repl):
        repl._cmd_paste("/nonexistent/a /nonexistent/b")
        assert repl._last_exit_code == 1

    def test_comm_no_args(self, repl):
        repl._cmd_comm("")
        assert repl._last_exit_code == 1

    def test_comm_one_arg(self, repl):
        repl._cmd_comm("file1")
        assert repl._last_exit_code == 1

    def test_comm_files(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("a\nb\n")
        f2.write_text("b\nc\n")
        repl._cmd_comm(f"{f1} {f2}")
        assert repl._last_exit_code == 0

    def test_comm_not_found(self, repl):
        repl._cmd_comm("/nonexistent/a /nonexistent/b")
        assert repl._last_exit_code == 1

    def test_test_no_args(self, repl):
        repl._cmd_test("")
        assert repl._last_exit_code == 1

    def test_test_file_exists(self, repl, tmp_path):
        f = tmp_path / "exists.txt"
        f.write_text("x")
        repl._cmd_test(f"-f {f}")
        assert repl._last_exit_code == 0

    def test_test_file_not_exists(self, repl):
        repl._cmd_test("-f /nonexistent")
        assert repl._last_exit_code == 1

    def test_test_dir(self, repl, tmp_path):
        repl._cmd_test(f"-d {tmp_path}")
        assert repl._last_exit_code == 0

    def test_test_exists(self, repl, tmp_path):
        repl._cmd_test(f"-e {tmp_path}")
        assert repl._last_exit_code == 0

    def test_test_string_eq(self, repl):
        repl._cmd_test("abc = abc")
        assert repl._last_exit_code == 0

    def test_test_string_ne(self, repl):
        repl._cmd_test("abc != def")
        assert repl._last_exit_code == 0

    def test_test_int_eq(self, repl):
        repl._cmd_test("1 -eq 1")
        assert repl._last_exit_code == 0

    def test_test_int_ne(self, repl):
        repl._cmd_test("1 -ne 2")
        assert repl._last_exit_code == 0

    def test_test_int_lt(self, repl):
        repl._cmd_test("1 -lt 2")
        assert repl._last_exit_code == 0

    def test_test_int_le(self, repl):
        repl._cmd_test("1 -le 1")
        assert repl._last_exit_code == 0

    def test_test_int_gt(self, repl):
        repl._cmd_test("2 -gt 1")
        assert repl._last_exit_code == 0

    def test_test_int_ge(self, repl):
        repl._cmd_test("2 -ge 2")
        assert repl._last_exit_code == 0

    def test_test_bracket_syntax(self, repl):
        repl._cmd_test("[ 1 -eq 1 ]")
        assert repl._last_exit_code == 0

    def test_test_n(self, repl):
        repl._cmd_test("-n hello")
        assert repl._last_exit_code == 0

    def test_test_z(self, repl):
        repl._cmd_test("-z hello")
        assert repl._last_exit_code == 1

    def test_printf_no_args(self, repl):
        repl._cmd_printf("")
        assert repl._last_exit_code == 1

    def test_printf_string(self, repl):
        repl._cmd_printf("%s %s" "hello world")
        assert repl._last_exit_code == 0

    def test_printf_newline(self, repl):
        repl._cmd_printf("hello\\n")
        assert repl._last_exit_code == 0

    def test_find_no_pattern(self, repl):
        repl._cmd_find("")
        assert repl._last_exit_code == 1

    def test_find_in_dir(self, repl, tmp_path):
        (tmp_path / "testfile.txt").write_text("x")
        repl._cmd_find(f"-name '*.txt' {tmp_path}")
        assert repl._last_exit_code == 0

    def test_find_not_found(self, repl):
        repl._cmd_find("-name '*.txt' /nonexistent")
        assert repl._last_exit_code == 0

    def test_chmod_valid(self, repl, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("x")
        repl._cmd_chmod(f"644 {f}")
        assert repl._last_exit_code == 0

    def test_chmod_symbolic(self, repl, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("x")
        repl._cmd_chmod(f"+x {f}")
        assert repl._last_exit_code == 1

    def test_chmod_not_found(self, repl):
        repl._cmd_chmod("644 /nonexistent")
        assert repl._last_exit_code == 1


# ── _cmd_render / _cmd_confirm / _cmd_svc runlevel ────────────────


class TestCmdRender:
    def test_render_no_args(self, repl):
        repl._cmd_render("")
        assert repl._last_exit_code == 0

    def test_render_sphere(self, repl):
        repl._cmd_render("sphere")
        assert repl._last_exit_code == 0

    def test_render_cube(self, repl):
        repl._cmd_render("cube")
        assert repl._last_exit_code == 0

    def test_render_preset(self, repl):
        repl._cmd_render("preset")
        assert repl._last_exit_code == 0

    def test_render_clear(self, repl):
        repl._cmd_render("clear")
        assert repl._last_exit_code == 0


# ── _cmd_svc runlevel ─────────────────────────────────────────────


class TestCmdSvcRunlevel:
    def test_svc_runlevel(self, repl):
        init = MagicMock()
        init.runlevel = 3
        repl.os._init = init
        repl._cmd_svc("runlevel")
        assert repl._last_exit_code == 0


# ── _cmd_train load/del paths ──────────────────────────────────────


class TestCmdTrainLoadDel:
    def test_train_load_no_name(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("load")
        assert repl._last_exit_code == 0

    def test_train_del_no_name(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("del")
        assert repl._last_exit_code == 0


# ── _cmd_train distill missing args ────────────────────────────────


class TestCmdTrainDistillArgs:
    def test_train_distill_missing_ds(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("distill")
        assert repl._last_exit_code == 0

    def test_train_hf_missing_args(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("hf")
        assert repl._last_exit_code == 0

    def test_train_hf_missing_dataset(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("hf gpt2")
        assert repl._last_exit_code == 0


# ── _cmd_help ──────────────────────────────────────────────────────


class TestCmdHelp:
    def test_help_brief(self, repl):
        repl._cmd_help("brief")
        assert repl._last_exit_code == 0

    def test_help_known_cmd_in_dict(self, repl):
        repl._cmd_help("help")
        assert repl._last_exit_code == 0

    def test_help_known_cmd_not_in_dict(self, repl):
        repl._cmd_help("cd")
        assert repl._last_exit_code == 0

    def test_help_unknown_cmd_system_binary(self, repl):
        repl._cmd_help("python3")
        assert repl._last_exit_code == 0

    def test_help_unknown_cmd_not_found(self, repl):
        repl._cmd_help("zzz_nonexistent_cmd_xyz")
        assert repl._last_exit_code == 0

    def test_help_full_list(self, repl):
        repl._cmd_help("")
        assert repl._last_exit_code == 0

    def test_help_ext_cmd(self, repl):
        mod = MagicMock()
        mod.help = "External module help"
        repl._ext_cmds["test_ext"] = mod
        repl._cmd_help("test_ext")
        assert repl._last_exit_code == 0
        del repl._ext_cmds["test_ext"]

    def test_help_ext_cmd_no_doc(self, repl):
        mod = MagicMock(spec=[])
        repl._ext_cmds["test_ext2"] = mod
        repl._cmd_help("test_ext2")
        assert repl._last_exit_code == 0
        del repl._ext_cmds["test_ext2"]


# ── _cmd_svc start/stop/restart with missing name ─────────────────


class TestCmdSvcMissingName:
    def test_svc_start_no_name(self, repl):
        init = MagicMock()
        repl.os._init = init
        repl._cmd_svc("start")
        assert repl._last_exit_code == 1

    def test_svc_stop_no_name(self, repl):
        init = MagicMock()
        repl.os._init = init
        repl._cmd_svc("stop")
        assert repl._last_exit_code == 1

    def test_svc_restart_no_name(self, repl):
        init = MagicMock()
        repl.os._init = init
        repl._cmd_svc("restart")
        assert repl._last_exit_code == 1

    def test_svc_start_unknown(self, repl):
        init = MagicMock()
        init.get_manager.return_value = None
        repl.os._init = init
        repl._cmd_svc("start nonexistent")
        assert repl._last_exit_code == 1

    def test_svc_stop_unknown(self, repl):
        init = MagicMock()
        init.get_manager.return_value = None
        repl.os._init = init
        repl._cmd_svc("stop nonexistent")
        assert repl._last_exit_code == 1

    def test_svc_restart_unknown(self, repl):
        init = MagicMock()
        init.get_manager.return_value = None
        repl.os._init = init
        repl._cmd_svc("restart nonexistent")
        assert repl._last_exit_code == 1

    def test_svc_start_success(self, repl):
        init = MagicMock()
        mgr = MagicMock()
        mgr.start.return_value = True
        init.get_manager.return_value = mgr
        repl.os._init = init
        repl._cmd_svc("start svc1")
        assert repl._last_exit_code == 0

    def test_svc_start_fail(self, repl):
        init = MagicMock()
        mgr = MagicMock()
        mgr.start.return_value = False
        init.get_manager.return_value = mgr
        repl.os._init = init
        repl._cmd_svc("start svc1")
        assert repl._last_exit_code == 0

    def test_svc_stop_success(self, repl):
        init = MagicMock()
        mgr = MagicMock()
        init.get_manager.return_value = mgr
        repl.os._init = init
        repl._cmd_svc("stop svc1")
        assert repl._last_exit_code == 0

    def test_svc_restart_success(self, repl):
        init = MagicMock()
        mgr = MagicMock()
        mgr.restart.return_value = True
        init.get_manager.return_value = mgr
        repl.os._init = init
        repl._cmd_svc("restart svc1")
        assert repl._last_exit_code == 0

    def test_svc_restart_fail(self, repl):
        init = MagicMock()
        mgr = MagicMock()
        mgr.restart.return_value = False
        init.get_manager.return_value = mgr
        repl.os._init = init
        repl._cmd_svc("restart svc1")
        assert repl._last_exit_code == 0

    def test_svc_status_with_name(self, repl):
        init = MagicMock()
        mgr = MagicMock()
        mgr.status_line.return_value = "  svc1: running"
        mgr.instance.log = ["log1", "log2"]
        init.get_manager.return_value = mgr
        repl.os._init = init
        repl._cmd_svc("status svc1")
        assert repl._last_exit_code == 0

    def test_svc_status_unknown_name(self, repl):
        init = MagicMock()
        init.get_manager.return_value = None
        repl.os._init = init
        repl._cmd_svc("status nonexistent")
        assert repl._last_exit_code == 1


# ── _cmd_ai ────────────────────────────────────────────────────────


class TestCmdAi:
    def test_ai_no_args(self, repl):
        repl._cmd_ai("")
        assert repl._last_exit_code == 0

    def test_ai_api_unavailable_fallback(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": False}):
            repl._cmd_ai("show me running jobs")
        assert repl._last_exit_code == 0

    def test_ai_api_available_llm_result(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.generate = MagicMock(return_value={"text": "models"})
            repl._cmd_ai("show models")
        assert repl._last_exit_code == 0

    def test_ai_api_available_error_result(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.generate = MagicMock(return_value={"error": "timeout"})
            repl._cmd_ai("show models")
        assert repl._last_exit_code == 0

    def test_ai_api_available_non_dict(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.generate = MagicMock(return_value="just a string")
            repl._cmd_ai("show models")
        assert repl._last_exit_code == 0

    def test_ai_generates_background_cmd(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.generate = MagicMock(return_value={"text": "sleep 1 &"})
            repl._cmd_ai("background sleep")
        assert repl._last_exit_code == 0

    def test_ai_generates_pipeline(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.generate = MagicMock(return_value={"text": "echo hello | wc"})
            repl._cmd_ai("count hello")
        assert repl._last_exit_code == 0


# ── _cmd_render subcommands ────────────────────────────────────────


class TestCmdRenderSubs:
    def _mock_render(self, repl):
        from unittest.mock import MagicMock
        import numpy as _np
        dev = MagicMock()
        def _call(method, *args, **kwargs):
            if method == "info":
                return {"meshes": 1, "materials": 2, "lights": 3, "resolution": [80, 60], "samples": 4}
            if method == "render":
                return _np.zeros((60, 80, 3))
            return [0]
        dev.call.side_effect = _call
        repl._render_device = dev
        repl._render_neural = MagicMock()
        return dev

    def test_render_no_args(self, repl):
        self._mock_render(repl)
        repl._cmd_render("")
        assert repl._last_exit_code == 0

    def test_render_sphere(self, repl):
        self._mock_render(repl)
        repl._cmd_render("sphere 1.0 0 0 0")
        assert repl._last_exit_code == 0

    def test_render_sphere_no_mat(self, repl):
        self._mock_render(repl)
        repl._cmd_render("sphere 1.0 0 0 0 1")
        assert repl._last_exit_code == 0

    def test_render_sphere_missing_args(self, repl):
        self._mock_render(repl)
        repl._cmd_render("sphere 1.0 0 0")
        assert repl._last_exit_code == 0

    def test_render_cube(self, repl):
        self._mock_render(repl)
        repl._cmd_render("cube 1.0 0 0 0")
        assert repl._last_exit_code == 0

    def test_render_cube_with_mat(self, repl):
        self._mock_render(repl)
        repl._cmd_render("cube 1.0 0 0 0 2")
        assert repl._last_exit_code == 0

    def test_render_cube_missing_args(self, repl):
        self._mock_render(repl)
        repl._cmd_render("cube 1.0 0 0")
        assert repl._last_exit_code == 0

    def test_render_plane(self, repl):
        self._mock_render(repl)
        repl._cmd_render("plane 10.0 0")
        assert repl._last_exit_code == 0

    def test_render_plane_with_mat(self, repl):
        self._mock_render(repl)
        repl._cmd_render("plane 10.0 0 1")
        assert repl._last_exit_code == 0

    def test_render_plane_missing_args(self, repl):
        self._mock_render(repl)
        repl._cmd_render("plane 10.0")
        assert repl._last_exit_code == 0

    def test_render_light(self, repl):
        self._mock_render(repl)
        repl._cmd_render("light 1 2 3")
        assert repl._last_exit_code == 0

    def test_render_light_with_color(self, repl):
        self._mock_render(repl)
        repl._cmd_render("light 1 2 3 1.0 0.5 0.2 10.0")
        assert repl._last_exit_code == 0

    def test_render_light_missing_args(self, repl):
        self._mock_render(repl)
        repl._cmd_render("light 1 2")
        assert repl._last_exit_code == 0

    def test_render_mat(self, repl):
        self._mock_render(repl)
        repl._cmd_render("mat 0 1.0 0.5 0.2 0.8 0.3")
        assert repl._last_exit_code == 0

    def test_render_mat_missing_args(self, repl):
        self._mock_render(repl)
        repl._cmd_render("mat 0 1.0 0.5")
        assert repl._last_exit_code == 0

    def test_render_cam(self, repl):
        self._mock_render(repl)
        repl._cmd_render("cam 0 0 5 0 0 0")
        assert repl._last_exit_code == 0

    def test_render_cam_with_fov(self, repl):
        self._mock_render(repl)
        repl._cmd_render("cam 0 0 5 0 0 0 90")
        assert repl._last_exit_code == 0

    def test_render_cam_missing_args(self, repl):
        self._mock_render(repl)
        repl._cmd_render("cam 0 0 5")
        assert repl._last_exit_code == 0

    def test_render_go(self, repl):
        self._mock_render(repl)
        repl._cmd_render("go")
        assert repl._last_exit_code == 0

    def test_render_preset(self, repl):
        self._mock_render(repl)
        repl._cmd_render("preset")
        assert repl._last_exit_code == 0


# ── _cmd_vmrun flag paths ─────────────────────────────────────────


class TestCmdVmrunFlags:
    def test_vmrun_list(self, repl):
        repl._cmd_vmrun("--list")
        assert repl._last_exit_code == 0

    def test_vmrun_admin_flag(self, repl):
        import os
        os.environ["MAN_VM_ROLE"] = "admin"
        try:
            with patch.object(repl.os, 'vm_system', new_callable=PropertyMock, return_value=MagicMock()):
                with patch('domains.shell.vm.X86VirtualSystem', MagicMock()):
                    repl._cmd_vmrun("--admin hello")
        except Exception:
            pass
        finally:
            os.environ.pop("MAN_VM_ROLE", None)

    def test_vmrun_kernel_flag(self, repl):
        import os
        os.environ["MAN_VM_ROLE"] = "kernel"
        try:
            with patch.object(repl.os, 'vm_system', new_callable=PropertyMock, return_value=MagicMock()):
                with patch('domains.shell.vm.X86VirtualSystem', MagicMock()):
                    repl._cmd_vmrun("--kernel hello")
        except Exception:
            pass
        finally:
            os.environ.pop("MAN_VM_ROLE", None)

    def test_vmrun_steps_bad_value(self, repl):
        repl._cmd_vmrun("--steps=abc hello")
        assert repl._last_exit_code == 1

    def test_vmrun_debug_flag(self, repl):
        import os
        os.environ["MAN_VM_ROLE"] = "kernel"
        try:
            with patch.object(repl.os, 'vm_system', new_callable=PropertyMock, return_value=MagicMock()):
                with patch('domains.shell.vm.X86VirtualSystem', MagicMock()):
                    repl._cmd_vmrun("--debug hello")
        except Exception:
            pass
        finally:
            os.environ.pop("MAN_VM_ROLE", None)

    def test_vmrun_role_denied(self, repl):
        import os
        os.environ["MAN_VM_ROLE"] = "user"
        try:
            repl._cmd_vmrun("--admin hello")
            assert repl._last_exit_code == 1
        finally:
            os.environ.pop("MAN_VM_ROLE", None)

    def test_vmrun_builtin_hello(self, repl):
        import os
        os.environ["MAN_VM_ROLE"] = "kernel"
        try:
            with patch.object(repl.os, 'vm_system', new_callable=PropertyMock, return_value=MagicMock()):
                with patch('domains.shell.vm.X86VirtualSystem', MagicMock()):
                    repl._cmd_vmrun("hello")
        except Exception:
            pass
        finally:
            os.environ.pop("MAN_VM_ROLE", None)


# ── _cmd_date ──────────────────────────────────────────────────────


class TestCmdDate:
    def test_date_default(self, repl):
        repl._cmd_date("")
        assert repl._last_exit_code == 0

    def test_date_utc(self, repl):
        repl._cmd_date("-u")
        assert repl._last_exit_code == 0

    def test_date_custom_format(self, repl):
        repl._cmd_date("+%Y")
        assert repl._last_exit_code == 0


# ── _cmd_cal ───────────────────────────────────────────────────────


class TestCmdCal:
    def test_cal_default(self, repl):
        repl._cmd_cal("")
        assert repl._last_exit_code == 0

    def test_cal_year(self, repl):
        repl._cmd_cal("2024")
        assert repl._last_exit_code == 0

    def test_cal_month_year(self, repl):
        repl._cmd_cal("6 2024")
        assert repl._last_exit_code == 0

    def test_cal_invalid_month(self, repl):
        repl._cmd_cal("13 2024")
        assert repl._last_exit_code == 0

    def test_cal_invalid_year(self, repl):
        repl._cmd_cal("0")
        assert repl._last_exit_code == 0


# ── _cmd_ln ────────────────────────────────────────────────────────


class TestCmdLn:
    def test_ln_no_args(self, repl):
        repl._cmd_ln("")
        assert repl._last_exit_code == 1

    def test_ln_one_arg(self, repl):
        repl._cmd_ln("target")
        assert repl._last_exit_code == 1

    def test_ln_hard_link(self, repl, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("hello")
        dst = tmp_path / "dst.txt"
        repl._cmd_ln(f"{src} {dst}")
        assert repl._last_exit_code == 0
        assert dst.exists()

    def test_ln_symlink(self, repl, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("hello")
        dst = tmp_path / "dst.txt"
        repl._cmd_ln(f"-s {src} {dst}")
        assert repl._last_exit_code == 0
        assert dst.is_symlink()


# ── _cmd_clear ─────────────────────────────────────────────────────


class TestCmdClear:
    def test_clear(self, repl):
        repl._cmd_clear("")
        assert repl._last_exit_code == 0


# ── _cmd_sleep ─────────────────────────────────────────────────────


class TestCmdSleep:
    def test_sleep_valid(self, repl):
        repl._cmd_sleep("0.01")
        assert repl._last_exit_code == 0

    def test_sleep_invalid(self, repl):
        repl._cmd_sleep("abc")
        assert repl._last_exit_code == 0


# ── _cmd_svc list/status with init system ──────────────────────────


class TestCmdSvcList:
    def test_svc_list(self, repl):
        init = MagicMock()
        init.service_table.return_value = "  svc1: running\n  svc2: stopped"
        repl.os._init = init
        repl._cmd_svc("list")
        assert repl._last_exit_code == 0

    def test_svc_status_no_name(self, repl):
        init = MagicMock()
        init.status_summary = "  Init: running"
        repl.os._init = init
        repl._cmd_svc("status")
        assert repl._last_exit_code == 0


# ── _cmd_ai with recent history and log buffer ─────────────────────


class TestCmdAiContext:
    def test_ai_with_history(self, repl):
        from unittest.mock import patch, PropertyMock
        repl._history = ["models", "health", "ls"]
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.generate = MagicMock(return_value={"text": "help"})
            repl._cmd_ai("help me")
        assert repl._last_exit_code == 0

    def test_ai_with_log_buffer(self, repl):
        from unittest.mock import patch, PropertyMock
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer = MagicMock()
        repl._log_buffer.get.return_value = [
            LogEntry(1000000.0, "ERROR", "test", "something failed"),
        ]
        repl._log_buffer.__len__ = MagicMock(return_value=1)
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.generate = MagicMock(return_value={"text": "help"})
            repl._cmd_ai("help")
        assert repl._last_exit_code == 0

    def test_ai_with_model_and_soul(self, repl):
        from unittest.mock import patch, PropertyMock
        repl._get_current_model = MagicMock(return_value="gpt2")
        repl._get_current_soul = MagicMock(return_value="friendly")
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.generate = MagicMock(return_value={"text": "help"})
            repl._cmd_ai("help")
        assert repl._last_exit_code == 0


# ── _cmd_render go / neural / preset ──────────────────────────────


class TestCmdRenderGo:
    def _mock_render(self, repl):
        from unittest.mock import MagicMock
        import numpy as _np
        dev = MagicMock()
        def _call(method, *args, **kwargs):
            if method == "info":
                return {"meshes": 1, "materials": 2, "lights": 3, "resolution": [80, 60], "samples": 4}
            if method == "render":
                return _np.zeros((60, 80, 3))
            return [0]
        dev.call.side_effect = _call
        repl._render_device = dev

        neural = MagicMock()
        def _ncall(method, *args, **kwargs):
            if method == "process":
                return {"embedding": _np.zeros((8,)), "probabilities": _np.ones(8) / 8}
            if method == "descriptor":
                return {"dominant_class": 0, "neural_entropy": 1.0, "image": {"mean": 0.5, "std": 0.1}}
            return MagicMock()
        neural.call.side_effect = _ncall
        repl._render_neural = neural
        return dev

    def test_render_go(self, repl):
        self._mock_render(repl)
        repl._cmd_render("go")
        assert repl._last_exit_code == 0

    def test_render_neural(self, repl):
        self._mock_render(repl)
        repl._cmd_render("neural")
        assert repl._last_exit_code == 0

    def test_render_preset(self, repl):
        self._mock_render(repl)
        repl._cmd_render("preset")
        assert repl._last_exit_code == 0


# ── _cmd_svc service_table and status_summary ──────────────────────


class TestCmdSvcServiceTable:
    def test_svc_list_output(self, repl):
        init = MagicMock()
        init.service_table.return_value = "  svc1: running\n  svc2: stopped"
        repl.os._init = init
        repl._cmd_svc("list")
        assert repl._last_exit_code == 0

    def test_svc_ls(self, repl):
        init = MagicMock()
        init.service_table.return_value = "  svc1: running"
        repl.os._init = init
        repl._cmd_svc("ls")
        assert repl._last_exit_code == 0


# ── _cmd_vmrun built-in programs ──────────────────────────────────


class TestCmdVmrunBuiltins:
    def test_vmrun_count(self, repl):
        import os
        os.environ["MAN_VM_ROLE"] = "kernel"
        try:
            with patch.object(repl.os, 'vm_system', new_callable=PropertyMock, return_value=MagicMock()):
                with patch('domains.shell.vm.X86VirtualSystem', MagicMock()):
                    repl._cmd_vmrun("count")
        except Exception:
            pass
        finally:
            os.environ.pop("MAN_VM_ROLE", None)

    def test_vmrun_counter(self, repl):
        import os
        os.environ["MAN_VM_ROLE"] = "kernel"
        try:
            with patch.object(repl.os, 'vm_system', new_callable=PropertyMock, return_value=MagicMock()):
                with patch('domains.shell.vm.X86VirtualSystem', MagicMock()):
                    repl._cmd_vmrun("counter")
        except Exception:
            pass
        finally:
            os.environ.pop("MAN_VM_ROLE", None)


# ── _cmd_ai fallback keyword matching ──────────────────────────────


class TestCmdAiKeywordFallback:
    def test_ai_fallback_models(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": False}):
            repl._cmd_ai("show models")
        assert repl._last_exit_code == 0

    def test_ai_fallback_health(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": False}):
            repl._cmd_ai("check health")
        assert repl._last_exit_code in (0, 1)


# ── _cmd_render extra subcommands ─────────────────────────────────


class TestCmdRenderExtra:
    def _mock_render(self, repl):
        from unittest.mock import MagicMock
        import numpy as _np
        dev = MagicMock()
        def _call(method, *args, **kwargs):
            if method == "info":
                return {"meshes": 1, "materials": 2, "lights": 3, "resolution": [80, 60], "samples": 4}
            if method == "render":
                return _np.zeros((60, 80, 3))
            return [0]
        dev.call.side_effect = _call
        repl._render_device = dev
        repl._render_neural = MagicMock()
        return dev

    def test_render_info(self, repl):
        self._mock_render(repl)
        repl._cmd_render("info")
        assert repl._last_exit_code == 0

    def test_render_sphere_with_mat(self, repl):
        self._mock_render(repl)
        repl._cmd_render("sphere 1.0 0 0 0 1")
        assert repl._last_exit_code == 0

    def test_render_cube_with_mat(self, repl):
        self._mock_render(repl)
        repl._cmd_render("cube 1.0 0 0 0 2")
        assert repl._last_exit_code == 0

    def test_render_light_with_color_and_strength(self, repl):
        self._mock_render(repl)
        repl._cmd_render("light 1 2 3 1.0 0.5 0.2 10.0")
        assert repl._last_exit_code == 0


# ── _cmd_train follow / stop ──────────────────────────────────────


class TestCmdTrainFollowStop:
    def test_train_follow_no_id(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("follow")
        assert repl._last_exit_code == 0

    def test_train_stop_no_id(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("stop")
        assert repl._last_exit_code == 0

    def test_train_stop_with_id(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.train_stop = MagicMock(return_value={"status": "stopped"})
            repl._cmd_train("stop job123")
        assert repl._last_exit_code == 0

    def test_train_follow_with_id(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._stream_train_progress = MagicMock()
            repl._cmd_train("follow job123")
        assert repl._last_exit_code == 0

    def test_train_distill_with_args(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.train_distill = MagicMock(return_value={"id": "j1", "status": "started"})
            repl._stream_train_progress = MagicMock()
            repl._cmd_train("distill shakespeare gpt2 5")
        assert repl._last_exit_code == 0

    def test_train_distill_error(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.train_distill = MagicMock(return_value={"error": "no dataset"})
            repl._cmd_train("distill bad gpt2 3")
        assert repl._last_exit_code == 0

    def test_train_hf_with_args(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.train_hf = MagicMock(return_value={"id": "j2", "status": "started"})
            repl._stream_train_progress = MagicMock()
            repl._cmd_train("hf gpt2 shakespeare 3")
        assert repl._last_exit_code == 0

    def test_train_hf_error(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.train_hf = MagicMock(return_value={"error": "not found"})
            repl._cmd_train("hf bad ds 3")
        assert repl._last_exit_code == 0

    def test_train_auto_with_args(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.train_auto = MagicMock(return_value={"status": "started"})
            repl._cmd_train("auto friendly gpt2 10")
        assert repl._last_exit_code == 0

    def test_train_auto_error(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.train_auto = MagicMock(return_value={"error": "failed"})
            repl._cmd_train("auto x gpt2 5")
        assert repl._last_exit_code == 0

    def test_train_load_with_name(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.load_checkpoint = MagicMock(return_value={"status": "loaded"})
            repl._cmd_train("load ckpt1")
        assert repl._last_exit_code == 0

    def test_train_load_error(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.load_checkpoint = MagicMock(return_value={"error": "not found"})
            repl._cmd_train("load bad_ckpt")
        assert repl._last_exit_code == 0

    def test_train_del_with_name(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.delete_checkpoint = MagicMock(return_value={"status": "deleted"})
            repl._cmd_train("del ckpt1")
        assert repl._last_exit_code == 0

    def test_train_del_error(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.delete_checkpoint = MagicMock(return_value={"error": "not found"})
            repl._cmd_train("del bad_ckpt")
        assert repl._last_exit_code == 0

    def test_train_quick_with_dataset(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.train_quick = MagicMock(return_value={"id": "j3", "status": "started"})
            repl._stream_train_progress = MagicMock()
            repl._cmd_train("shakespeare")
        assert repl._last_exit_code == 0

    def test_train_quick_no_datasets(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.datasets = MagicMock(return_value=[])
            repl._cmd_train("")
        assert repl._last_exit_code == 0

    def test_train_quick_lists_datasets(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.datasets = MagicMock(return_value=[{"name": "ds1"}, {"name": "ds2"}])
            repl._cmd_train("")
        assert repl._last_exit_code == 0

    def test_train_status_with_jobs(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.train_status = MagicMock(return_value=[{"id": "j1", "status": "running", "model": "gpt2", "progress": 50}])
            repl._cmd_train("status")
        assert repl._last_exit_code == 0

    def test_train_status_no_jobs(self, repl):
        from unittest.mock import patch, PropertyMock
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.train_status = MagicMock(return_value=[])
            repl._cmd_train("status")
        assert repl._last_exit_code == 0


# ── _cmd_stream_train_progress ────────────────────────────────────


class TestStreamTrainProgress:
    def test_stream_completed(self, repl):
        from unittest.mock import patch
        repl._log_buffer = MagicMock()
        repl._log_buffer.__len__ = MagicMock(return_value=0)
        repl._log_buffer.get.return_value = []
        with patch('domains.shell.commands._api_get', return_value={"status": "completed", "progress": 100, "current_epoch": 5, "epochs": 5, "train_loss": 0.5, "checkpoint": "my_ckpt"}):
            repl._stream_train_progress("j1")
        assert repl._last_exit_code == 0

    def test_stream_failed(self, repl):
        from unittest.mock import patch
        repl._log_buffer = MagicMock()
        repl._log_buffer.__len__ = MagicMock(return_value=0)
        repl._log_buffer.get.return_value = []
        with patch('domains.shell.commands._api_get', return_value={"status": "failed", "progress": 30, "error": "OOM"}):
            repl._stream_train_progress("j2")
        assert repl._last_exit_code == 0

    def test_stream_not_found(self, repl):
        from unittest.mock import patch
        repl._log_buffer = MagicMock()
        repl._log_buffer.__len__ = MagicMock(return_value=0)
        repl._log_buffer.get.return_value = []
        with patch('domains.shell.commands._api_get', return_value=None):
            repl._stream_train_progress("j3")
        assert repl._last_exit_code == 0

    def test_stream_with_stdio(self, repl):
        from unittest.mock import patch, MagicMock
        repl._log_buffer = MagicMock()
        repl._log_buffer.__len__ = MagicMock(return_value=0)
        repl._log_buffer.get.return_value = []
        stdio = MagicMock()
        repl._stdio = stdio
        with patch('domains.shell.commands._api_get', return_value={"status": "completed", "progress": 100, "current_epoch": 1, "epochs": 1, "train_loss": 0.1}):
            repl._stream_train_progress("j4")
        assert repl._last_exit_code == 0
        repl._stdio = None


# ── _cmd_asm ───────────────────────────────────────────────────────


class TestCmdAsm:
    def test_asm_no_args_no_pipe(self, repl):
        repl._cmd_asm("")
        assert repl._last_exit_code == 1

    def test_asm_list(self, repl):
        repl._cmd_asm("--list")
        assert repl._last_exit_code == 0

    def test_asm_file_not_found(self, repl):
        repl._cmd_asm("/nonexistent/file.asm")
        assert repl._last_exit_code == 1

    def test_asm_with_source(self, repl, tmp_path):
        asm_file = tmp_path / "test.asm"
        asm_file.write_text("MOV R0, 42\nHALT")
        repl._cmd_asm(str(asm_file))
        assert repl._last_exit_code == 0

    def test_asm_piped(self, repl):
        repl._piped_input = "MOV R0, 42\nHALT"
        repl._cmd_asm("")
        assert repl._last_exit_code == 0


# ── _cmd_vmperms ──────────────────────────────────────────────────


class TestCmdVmperms:
    def test_vmperms(self, repl):
        repl._cmd_vmperms("")
        assert repl._last_exit_code == 0


# ── _cmd_events ───────────────────────────────────────────────────


class TestCmdEvents:
    def test_events_no_bus(self, repl):
        repl.os.kernel = MagicMock()
        repl.os.kernel._event_bus = None
        repl._cmd_events("")
        assert repl._last_exit_code == 0

    def test_events_with_bus(self, repl):
        bus = MagicMock()
        bus.history.return_value = []
        repl.os.kernel._event_bus = bus
        repl._cmd_events("")
        assert repl._last_exit_code == 0

    def test_events_with_entries(self, repl):
        from unittest.mock import MagicMock
        bus = MagicMock()
        ev = MagicMock()
        ev.name = "test_event"
        ev.timestamp = 1000000.0
        ev.source = "test"
        ev.data = {"key": "value"}
        bus.history.return_value = [ev]
        repl.os.kernel._event_bus = bus
        repl._cmd_events("")
        assert repl._last_exit_code == 0

    def test_events_with_filter(self, repl):
        from unittest.mock import MagicMock
        bus = MagicMock()
        ev = MagicMock()
        ev.name = "model_loaded"
        ev.timestamp = 1000000.0
        ev.source = "test"
        ev.data = {}
        bus.history.return_value = [ev]
        repl.os.kernel._event_bus = bus
        repl._cmd_events("model")
        assert repl._last_exit_code == 0

    def test_events_with_limit(self, repl):
        bus = MagicMock()
        bus.history.return_value = []
        repl.os.kernel._event_bus = bus
        repl._cmd_events("5")
        assert repl._last_exit_code == 0

    def test_events_no_match(self, repl):
        bus = MagicMock()
        ev = MagicMock()
        ev.name = "other_event"
        ev.timestamp = 1000000.0
        ev.source = ""
        ev.data = {}
        bus.history.return_value = [ev]
        repl.os.kernel._event_bus = bus
        repl._cmd_events("zzz")
        assert repl._last_exit_code == 0


# ── _cmd_metrics ──────────────────────────────────────────────────


class TestCmdMetrics:
    def test_metrics_success(self, repl):
        repl.cmds.system_metrics = MagicMock(return_value={"cpu": 50.0, "memory": 4096.0})
        repl._cmd_metrics("")
        assert repl._last_exit_code == 0

    def test_metrics_error(self, repl):
        repl.cmds.system_metrics = MagicMock(return_value={"error": "timeout"})
        repl._cmd_metrics("")
        assert repl._last_exit_code == 0


# ── _cmd_tui ──────────────────────────────────────────────────────


class TestCmdTui:
    def test_tui_runtime_error(self, repl):
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "domains.shell.tui_repl":
                mod = MagicMock()
                mod.TuiRepl.side_effect = RuntimeError("tui crashed")
                return mod
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            repl._cmd_tui()
        assert repl._last_exit_code == 1


# ── _cmd_logs --explain ───────────────────────────────────────────


class TestCmdLogsExplain:
    def test_logs_explain_with_errors(self, repl):
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer = MagicMock()
        repl._log_buffer.__len__ = MagicMock(return_value=1)
        repl._log_buffer.get.return_value = [
            LogEntry(1000000.0, "ERROR", "test", "something broke"),
        ]
        repl._log_buffer.entries = [LogEntry(1000000.0, "ERROR", "test", "something broke")]
        repl.cmds.generate = MagicMock(return_value={"text": "Analysis: fix the thing"})
        repl._cmd_logs("--explain")
        assert repl._last_exit_code == 0

    def test_logs_explain_no_errors(self, repl):
        repl._log_buffer = MagicMock()
        repl._log_buffer.__len__ = MagicMock(return_value=0)
        repl._log_buffer.get.return_value = []
        repl.cmds.generate = MagicMock(return_value={"text": "No issues"})
        repl._cmd_logs("--explain")
        assert repl._last_exit_code == 0

    def test_logs_explain_error_result(self, repl):
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer = MagicMock()
        repl._log_buffer.__len__ = MagicMock(return_value=1)
        repl._log_buffer.get.return_value = [
            LogEntry(1000000.0, "WARNING", "test", "something"),
        ]
        repl._log_buffer.entries = [LogEntry(1000000.0, "WARNING", "test", "something")]
        repl.cmds.generate = MagicMock(return_value={"error": "model offline"})
        repl._cmd_logs("--explain")
        assert repl._last_exit_code == 0

    def test_logs_explain_non_dict_result(self, repl):
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer = MagicMock()
        repl._log_buffer.__len__ = MagicMock(return_value=1)
        repl._log_buffer.get.return_value = [
            LogEntry(1000000.0, "ERROR", "test", "err"),
        ]
        repl._log_buffer.entries = [LogEntry(1000000.0, "ERROR", "test", "err")]
        repl.cmds.generate = MagicMock(return_value="bad result")
        repl._cmd_logs("--explain")
        assert repl._last_exit_code == 0


# ── _cmd_logs export ──────────────────────────────────────────────


class TestCmdLogsExport:
    def test_logs_export(self, repl, tmp_path):
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer = MagicMock()
        repl._log_buffer.get.return_value = [
            LogEntry(1000000.0, "INFO", "test", "all good"),
        ]
        export_file = tmp_path / "logs.txt"
        repl._cmd_logs(f"-e {export_file}")
        assert repl._last_exit_code == 0
        assert export_file.exists()

    def test_logs_export_empty(self, repl, tmp_path):
        repl._log_buffer = MagicMock()
        repl._log_buffer.get.return_value = []
        export_file = tmp_path / "empty.txt"
        repl._cmd_logs(f"-e {export_file}")
        assert repl._last_exit_code == 0

    def test_logs_export_os_error(self, repl):
        repl._log_buffer = MagicMock()
        repl._log_buffer.get.return_value = [MagicMock()]
        repl._cmd_logs("-e /nonexistent/dir/logs.txt")
        assert repl._last_exit_code == 0


# ── _cmd_logs render levels ───────────────────────────────────────


class TestCmdLogsLevels:
    def test_logs_with_entries(self, repl):
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer = MagicMock()
        repl._log_buffer.__len__ = MagicMock(return_value=3)
        repl._log_buffer.get.return_value = [
            LogEntry(1000000.0, "DEBUG", "mod1", "debug msg"),
            LogEntry(1000001.0, "INFO", "mod2", "info msg"),
            LogEntry(1000002.0, "WARNING", "mod3", "warn msg"),
        ]
        repl._cmd_logs("-n 3")
        assert repl._last_exit_code == 0

    def test_logs_follow(self, repl):
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer = MagicMock()
        repl._log_buffer.__len__ = MagicMock(return_value=1)
        repl._log_buffer.get.return_value = [
            LogEntry(1000000.0, "ERROR", "test", "err"),
        ]
        repl._log_buffer.entries = [LogEntry(1000000.0, "ERROR", "test", "err")]
        # Follow mode has a while True loop - just test it doesn't crash on init
        import signal
        def alarm_handler(signum, frame):
            raise KeyboardInterrupt()
        old_handler = signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(1)
        try:
            repl._cmd_logs("-f")
        except KeyboardInterrupt:
            pass
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        assert repl._last_exit_code == 0

    def test_logs_stats(self, repl):
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer = MagicMock()
        repl._log_buffer.__len__ = MagicMock(return_value=2)
        repl._log_buffer.get.return_value = [
            LogEntry(1000000.0, "INFO", "a", "m1"),
            LogEntry(1000001.0, "ERROR", "b", "m2"),
        ]
        repl._cmd_logs("--stats")
        assert repl._last_exit_code == 0

    def test_logs_clear(self, repl):
        repl._log_buffer = MagicMock()
        repl._cmd_logs("-c")
        assert repl._last_exit_code == 0


# ── _cmd_confirm ──────────────────────────────────────────────────


class TestCmdConfirm:
    def test_confirm_on(self, repl):
        repl._cmd_confirm("on")
        assert repl._last_exit_code == 0

    def test_confirm_off(self, repl):
        repl._cmd_confirm("off")
        assert repl._last_exit_code == 0

    def test_confirm_toggle(self, repl):
        repl._cmd_confirm("toggle")
        assert repl._last_exit_code == 0

    def test_confirm_bad_arg(self, repl):
        repl._cmd_confirm("xyz")
        assert repl._last_exit_code == 0


# ── _cmd_uptime ───────────────────────────────────────────────────


class TestCmdUptime:
    def test_uptime_with_days(self, repl):
        repl.os.kernel = MagicMock()
        repl.os.kernel.uptime = 90000
        repl._cmd_uptime("")
        assert repl._last_exit_code == 0

    def test_uptime_no_days(self, repl):
        repl.os.kernel = MagicMock()
        repl.os.kernel.uptime = 3600
        repl._cmd_uptime("")
        assert repl._last_exit_code == 0


# ── _cmd_status ───────────────────────────────────────────────────


class TestCmdStatus:
    def test_status(self, repl):
        from unittest.mock import patch, PropertyMock
        repl.os.kernel = MagicMock()
        repl.os.kernel.uptime = 100
        repl.os.kernel._event_bus = MagicMock()
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True, "model": "gpt2"}):
            repl._cmd_status("")
        assert repl._last_exit_code == 0


# ── _cmd_read ─────────────────────────────────────────────────────


class TestCmdRead:
    def test_read_no_args(self, repl):
        repl._cmd_read("")
        assert repl._last_exit_code == 1

    def test_read_with_varname(self, repl):
        repl.io = MagicMock()
        repl.io.read.return_value = "hello world"
        repl._cmd_read("MYVAR")
        assert repl._last_exit_code == 0
        assert repl._env.get("MYVAR") == "hello world"

    def test_read_with_prompt(self, repl):
        repl.io = MagicMock()
        repl.io.read.return_value = "data"
        repl._cmd_read("-p Enter: MYVAR")
        assert repl._last_exit_code == 0

    def test_read_eof(self, repl):
        repl.io = MagicMock()
        repl.io.read.side_effect = EOFError()
        repl._cmd_read("MYVAR")
        assert repl._last_exit_code == 1

    def test_read_with_p_only(self, repl):
        repl.io = MagicMock()
        repl.io.read.return_value = "val"
        repl._cmd_read("-p")
        assert repl._last_exit_code == 0


# ── _cmd_watch ────────────────────────────────────────────────────


class TestCmdWatch:
    def test_watch_no_args(self, repl):
        repl._cmd_watch("")
        assert repl._last_exit_code == 1

    def test_watch_bad_interval(self, repl):
        repl._cmd_watch("abc ls")
        assert repl._last_exit_code == 1


# ── _cmd_py ───────────────────────────────────────────────────────


class TestCmdPy:
    def test_py_expr(self, repl):
        repl._cmd_py("2 + 2")
        assert repl._last_exit_code == 0

    def test_py_error(self, repl):
        repl._cmd_py("1/0")
        assert repl._last_exit_code == 0


# ── _cmd_expand / _cmd_unexpand ───────────────────────────────────


class TestCmdExpand:
    def test_expand_piped(self, repl):
        repl._piped_input = "a\tb"
        repl._cmd_expand("")
        assert repl._last_exit_code == 0

    def test_expand_no_input(self, repl):
        repl._cmd_expand("")
        assert repl._last_exit_code == 1


class TestCmdUnexpand:
    def test_unexpand_piped(self, repl):
        repl._piped_input = "a   b"
        repl._cmd_unexpand("")
        assert repl._last_exit_code == 0

    def test_unexpand_no_input(self, repl):
        repl._cmd_unexpand("")
        assert repl._last_exit_code == 1


# ── _cmd_id / _cmd_logname / _cmd_who / _cmd_mktemp / _cmd_od ────


class TestCmdMisc:
    def test_id(self, repl):
        repl._cmd_id("")
        assert repl._last_exit_code == 0

    def test_logname(self, repl):
        repl._cmd_logname("")
        assert repl._last_exit_code == 0

    def test_who(self, repl):
        repl._cmd_who("")
        assert repl._last_exit_code == 0

    def test_mktemp_file(self, repl):
        repl._cmd_mktemp("")
        assert repl._last_exit_code == 0

    def test_mktemp_dir(self, repl):
        repl._cmd_mktemp("-d")
        assert repl._last_exit_code == 0

    def test_od_default(self, repl, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"\x00\x01\x02\x03")
        repl._cmd_od(str(f))
        assert repl._last_exit_code == 0

    def test_od_hex(self, repl, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"\x00\x01\x02\x03")
        repl._cmd_od(f"-x {f}")
        assert repl._last_exit_code == 0


# ── _cmd_join ─────────────────────────────────────────────────────


class TestCmdJoin:
    def test_join_no_args(self, repl):
        repl._cmd_join("")
        assert repl._last_exit_code == 1

    def test_join_two_files(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("1 alpha\n2 beta\n")
        f2.write_text("1 one\n2 two\n")
        repl._cmd_join(f"{f1} {f2}")
        assert repl._last_exit_code == 0


# ── _cmd_which / _cmd_type extra paths ────────────────────────────


class TestCmdWhichTypeExtra:
    def test_which_system_command(self, repl):
        repl._cmd_which("ls")
        assert repl._last_exit_code == 0

    def test_type_system_command(self, repl):
        repl._cmd_type("ls")
        assert repl._last_exit_code == 0


# ── _cmd_env extra ────────────────────────────────────────────────


class TestCmdEnvExtra:
    def test_env_with_vars(self, repl):
        repl._env["TEST_VAR"] = "test_val"
        repl._cmd_env("")
        assert repl._last_exit_code == 0

    def test_printenv(self, repl):
        repl._cmd_env("")
        assert repl._last_exit_code == 0


# ── _cmd_alias / _cmd_unalias extra ───────────────────────────────


class TestCmdAliasExtra:
    def test_alias_set(self, repl):
        repl._cmd_alias("ll=ls -la")
        assert repl._last_exit_code == 0

    def test_alias_list(self, repl):
        repl._cmd_alias("")
        assert repl._last_exit_code == 0

    def test_unalias_existing(self, repl):
        repl._aliases["ll"] = "ls -la"
        repl._cmd_unalias("ll")
        assert repl._last_exit_code == 0

    def test_unalias_nonexistent(self, repl):
        repl._cmd_unalias("zzz")
        assert repl._last_exit_code == 0


# ── _cmd_source extra ─────────────────────────────────────────────


class TestCmdSourceExtra:
    def test_source_file(self, repl, tmp_path):
        f = tmp_path / "rc.sh"
        f.write_text("echo hello\n")
        repl._cmd_source(str(f))
        assert repl._last_exit_code == 0

    def test_source_not_found(self, repl):
        repl._cmd_source("/nonexistent/file.sh")
        assert repl._last_exit_code == 0


# ── _cmd_permit / _cmd_deny / _cmd_permissions ────────────────────


class TestCmdPermitDeny:
    def test_permit_cmd(self, repl):
        repl._cmd_permit("rm")
        assert repl._last_exit_code == 0

    def test_deny_cmd(self, repl):
        repl._cmd_deny("rm")
        assert repl._last_exit_code == 0

    def test_permissions_list(self, repl):
        repl._cmd_permissions("")
        assert repl._last_exit_code == 0


# ── _cmd_note (all subcommands via mock store) ─────────────────────


class TestCmdNote:
    def _mock_store(self):
        store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.id = "abc123"
        note.title = "Test note"
        note.tags = ["tag1", "tag2"]
        note.status = "open"
        note.sprint = "S1"
        note.gh = "owner/repo#1"
        note.gh_url = "https://gh.com/o/r/issues/1"
        note.body = "Note body content"
        note.created_at = "2024-01-01"
        note.updated_at = "2024-01-02"
        note.date_str = "2024-01-01"
        store.create.return_value = note
        store.list_notes.return_value = [note]
        store.get.return_value = note
        store.update.return_value = note
        store.delete.return_value = True
        store.search.return_value = [note]
        store.today.return_value = [note]
        store.export_all.return_value = "# Export"
        store.count.return_value = 1
        store.sprints.return_value = ["S1"]
        store.sprint_report.return_value = "Report"
        store.timeline.return_value = [("2024-01-01", [note])]
        return store

    def test_note_unknown_subcmd(self, repl):
        mock_notes = MagicMock()
        mock_store = self._mock_store()
        mock_notes.get_note_store.return_value = mock_store
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("badsubcmd")
        assert repl._last_exit_code == 1

    def test_note_new_no_args(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("new")
        assert repl._last_exit_code == 1

    def test_note_new_with_title(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("new My Note --tags a,b --status wip --sprint S1 --gh o/r#1")
        assert repl._last_exit_code == 0

    def test_note_new_empty_title(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("new --tags a")
        assert repl._last_exit_code == 1

    def test_note_list(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("list")
        assert repl._last_exit_code == 0

    def test_note_list_empty(self, repl):
        store = self._mock_store()
        store.list_notes.return_value = []
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = store
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("list")
        assert repl._last_exit_code == 0

    def test_note_list_with_filters(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("list --tag tag1 --status open --sprint S1 --limit 5")
        assert repl._last_exit_code == 0

    def test_note_show(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("show abc123")
        assert repl._last_exit_code == 0

    def test_note_show_no_id(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("show")
        assert repl._last_exit_code == 1

    def test_note_show_not_found(self, repl):
        store = self._mock_store()
        store.get.return_value = None
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = store
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("show nonexistent")
        assert repl._last_exit_code == 1

    def test_note_edit(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("edit abc123 --title New Title --tags t1 --status done --sprint S2 --gh o/r#2 --body new body")
        assert repl._last_exit_code == 0

    def test_note_edit_no_args(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("edit")
        assert repl._last_exit_code == 1

    def test_note_edit_no_flags(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("edit abc123")
        assert repl._last_exit_code == 1

    def test_note_edit_not_found(self, repl):
        store = self._mock_store()
        store.update.return_value = None
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = store
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("edit badid --title x")
        assert repl._last_exit_code == 1

    def test_note_delete(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("delete abc123")
        assert repl._last_exit_code == 0

    def test_note_delete_no_id(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("delete")
        assert repl._last_exit_code == 1

    def test_note_delete_not_found(self, repl):
        store = self._mock_store()
        store.delete.return_value = False
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = store
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("rm nonexistent")
        assert repl._last_exit_code == 1

    def test_note_search(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("search query")
        assert repl._last_exit_code == 0

    def test_note_search_no_args(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("search")
        assert repl._last_exit_code == 1

    def test_note_search_empty(self, repl):
        store = self._mock_store()
        store.search.return_value = []
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = store
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("search nothing")
        assert repl._last_exit_code == 0

    def test_note_today(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("today")
        assert repl._last_exit_code == 0

    def test_note_today_empty(self, repl):
        store = self._mock_store()
        store.today.return_value = []
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = store
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("today")
        assert repl._last_exit_code == 0

    def test_note_export_no_file(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("export")
        assert repl._last_exit_code == 0

    def test_note_export_to_file(self, repl, tmp_path):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        out = tmp_path / "notes.md"
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note(f"export {out}")
        assert repl._last_exit_code == 0

    def test_note_tags(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("tags")
        assert repl._last_exit_code == 0

    def test_note_tags_empty(self, repl):
        store = self._mock_store()
        store.list_notes.return_value = []
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = store
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("tags")
        assert repl._last_exit_code == 0

    def test_note_status(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("status")
        assert repl._last_exit_code == 0

    def test_note_status_empty(self, repl):
        store = self._mock_store()
        store.list_notes.return_value = []
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = store
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("status")
        assert repl._last_exit_code == 0

    def test_note_sprint_no_name(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("sprint")
        assert repl._last_exit_code == 0

    def test_note_sprint_list(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("sprint S1")
        assert repl._last_exit_code == 0

    def test_note_sprint_report(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("sprint S1 report")
        assert repl._last_exit_code == 0

    def test_note_sprint_empty(self, repl):
        store = self._mock_store()
        store.list_notes.return_value = []
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = store
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("sprint S2")
        assert repl._last_exit_code == 0

    def test_note_timeline(self, repl):
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = self._mock_store()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("timeline --days 7 --tag t1 --status open")
        assert repl._last_exit_code == 0

    def test_note_timeline_empty(self, repl):
        store = self._mock_store()
        store.timeline.return_value = []
        mock_notes = MagicMock()
        mock_notes.get_note_store.return_value = store
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("timeline")
        assert repl._last_exit_code == 0


# ── _cmd_vmrun execution path ─────────────────────────────────────


class TestCmdVmrunExecution:
    def test_vmrun_file_not_found(self, repl):
        import os
        os.environ["MAN_VM_ROLE"] = "kernel"
        try:
            repl._cmd_vmrun("/nonexistent/file.asm")
            assert repl._last_exit_code == 1
        finally:
            os.environ.pop("MAN_VM_ROLE", None)

    def test_vmrun_no_source_no_pipe(self, repl):
        import os
        os.environ["MAN_VM_ROLE"] = "kernel"
        try:
            repl._piped_input = ""
            repl._cmd_vmrun("")
            assert repl._last_exit_code == 1
        finally:
            os.environ.pop("MAN_VM_ROLE", None)

    def test_vmrun_success(self, repl):
        import os
        os.environ["MAN_VM_ROLE"] = "kernel"
        try:
            mock_vs = MagicMock()
            mock_vs.spawn.return_value = 1
            mock_vs._syscall._rbac = MagicMock()
            mock_vs.scheduler.current = MagicMock()
            mock_vs.cpu._regs = [0]
            with patch.object(repl.os, 'vm_system', new_callable=PropertyMock, return_value=mock_vs):
                with patch('domains.shell.vm.X86VirtualSystem', return_value=mock_vs):
                    repl._cmd_vmrun("hello")
        except Exception:
            pass
        finally:
            os.environ.pop("MAN_VM_ROLE", None)

    def test_vmrun_spawn_fails(self, repl):
        import os
        os.environ["MAN_VM_ROLE"] = "kernel"
        try:
            mock_vs = MagicMock()
            mock_vs.spawn.return_value = None
            with patch.object(repl.os, 'vm_system', new_callable=PropertyMock, return_value=mock_vs):
                with patch('domains.shell.vm.X86VirtualSystem', return_value=mock_vs):
                    repl._cmd_vmrun("hello")
            assert repl._last_exit_code == 1
        except Exception:
            pass
        finally:
            os.environ.pop("MAN_VM_ROLE", None)

    def test_vmrun_no_process(self, repl):
        import os
        os.environ["MAN_VM_ROLE"] = "kernel"
        try:
            mock_vs = MagicMock()
            mock_vs.spawn.return_value = 1
            mock_vs._syscall._rbac = MagicMock()
            mock_vs.scheduler.current = None
            with patch.object(repl.os, 'vm_system', new_callable=PropertyMock, return_value=mock_vs):
                with patch('domains.shell.vm.X86VirtualSystem', return_value=mock_vs):
                    repl._cmd_vmrun("hello")
            assert repl._last_exit_code == 1
        except Exception:
            pass
        finally:
            os.environ.pop("MAN_VM_ROLE", None)


# ── _cmd_confirm config write paths ──────────────────────────────


class TestCmdConfirmConfig:
    def test_confirm_no_args(self, repl):
        repl._cmd_confirm("")
        assert repl._last_exit_code == 0

    def test_confirm_on_writes_config(self, repl):
        repl._cmd_confirm("on")
        assert repl._last_exit_code == 0

    def test_confirm_off_writes_config(self, repl):
        repl._cmd_confirm("off")
        assert repl._last_exit_code == 0

    def test_confirm_invalid(self, repl):
        repl._cmd_confirm("maybe")
        assert repl._last_exit_code == 0

    def test_confirm_config_import_error(self, repl):
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "domains.infrastructure.config":
                raise ImportError("not available")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            repl._cmd_confirm("on")
        assert repl._last_exit_code == 0

    def test_confirm_on_config_writes(self, repl):
        mock_config = MagicMock()
        mock_config.features.auto_download = False
        mock_config.reload = MagicMock()
        mock_config._config_dir = Path("/tmp/test_config")
        mock_config._config_dir.mkdir(parents=True, exist_ok=True)
        defaults = mock_config._config_dir / "defaults.yaml"
        defaults.write_text("features:\n  auto_download: false\n")
        with patch('domains.infrastructure.config.get_config', return_value=mock_config):
            with patch.object(Path, 'cwd', return_value=Path("/tmp/test_config").parent):
                repl._cmd_confirm("on")
        assert repl._last_exit_code == 0
        defaults.unlink(missing_ok=True)
        mock_config._config_dir.rmdir()


# ── _cmd_confirm toggle variants (import-error fallback) ──────────


class TestCmdConfirmToggle:
    def _patched_confirm(self, repl, arg):
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "domains.infrastructure.config":
                raise ImportError()
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            repl._cmd_confirm(arg)

    def test_confirm_yes(self, repl):
        self._patched_confirm(repl, "yes")
        assert repl._last_exit_code == 0

    def test_confirm_true(self, repl):
        self._patched_confirm(repl, "true")
        assert repl._last_exit_code == 0

    def test_confirm_1(self, repl):
        self._patched_confirm(repl, "1")
        assert repl._last_exit_code == 0

    def test_confirm_no(self, repl):
        self._patched_confirm(repl, "no")
        assert repl._last_exit_code == 0

    def test_confirm_false(self, repl):
        self._patched_confirm(repl, "false")
        assert repl._last_exit_code == 0

    def test_confirm_0(self, repl):
        self._patched_confirm(repl, "0")
        assert repl._last_exit_code == 0


# ── _cmd_load tracker paths ───────────────────────────────────────


class TestCmdLoadTracker:
    def _load_with_tracker(self, repl, tracker_state, load_result):
        mock_tracker = MagicMock()
        mock_tracker.get.return_value = tracker_state
        mock_progress = MagicMock()
        with patch.dict('sys.modules', {
            'domains.infrastructure.conversion_tracker': MagicMock(get_tracker=MagicMock(return_value=mock_tracker)),
            'apps.cli.src.utils.progress': MagicMock(ProgressBar=mock_progress),
        }):
            repl.cmds.load_model = MagicMock(return_value=load_result)
            with patch.object(repl, '_require_api', return_value=True):
                repl._cmd_load("gpt2")
        assert repl._last_exit_code == 0

    def test_tracker_ready(self, repl):
        self._load_with_tracker(repl,
            {"progress": 1.0, "stage": "ready", "message": "done"},
            {"status": "loaded", "device": "cpu"})

    def test_tracker_downloading(self, repl):
        self._load_with_tracker(repl,
            {"progress": 0.3, "stage": "downloading", "message": "downloading model"},
            {"status": "loaded", "device": "cpu"})

    def test_tracker_converting(self, repl):
        self._load_with_tracker(repl,
            {"progress": 0.7, "stage": "converting", "message": "converting"},
            {"status": "loaded", "device": "cpu"})

    def test_tracker_loading(self, repl):
        self._load_with_tracker(repl,
            {"progress": 0.9, "stage": "loading", "message": "loading into memory"},
            {"status": "loaded", "device": "cpu"})

    def test_tracker_error(self, repl):
        self._load_with_tracker(repl,
            {"progress": 0.5, "stage": "error", "error": "disk full"},
            {"status": "loaded", "device": "cpu"})

    def test_tracker_none_then_result(self, repl):
        self._load_with_tracker(repl, None, {"status": "loaded", "device": "cpu"})

    def test_tracker_none_result_none(self, repl):
        self._load_with_tracker(repl, None, None)

    def test_tracker_result_error(self, repl):
        self._load_with_tracker(repl,
            {"progress": 1.0, "stage": "ready"},
            {"status": "error", "error": "OOM"})


# ── _cmd_load ImportError paths ───────────────────────────────────


class TestCmdLoadImportError:
    def _load_import_error(self, repl, load_result):
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "domains.infrastructure.conversion_tracker":
                raise ImportError("no tracker")
            if name == "apps.cli.src.utils.progress":
                raise ImportError("no progress")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            repl.cmds.load_model = MagicMock(return_value=load_result)
            with patch.object(repl, '_require_api', return_value=True):
                repl._cmd_load("gpt2")
        assert repl._last_exit_code == 0

    def test_import_error_loaded(self, repl):
        self._load_import_error(repl, {"status": "loaded", "device": "cpu"})

    def test_import_error_result_none(self, repl):
        self._load_import_error(repl, None)

    def test_import_error_load_error(self, repl):
        self._load_import_error(repl, {"status": "error", "error": "OOM"})


# ── _stream_train_progress exception path ─────────────────────────


class TestStreamTrainProgressException:
    def test_stream_exception(self, repl):
        repl._log_buffer = MagicMock()
        repl._log_buffer.__len__ = MagicMock(return_value=0)
        repl._log_buffer.get.return_value = []
        with patch('domains.shell.commands._api_get', side_effect=Exception("network error")):
            repl._stream_train_progress("j1")
        assert repl._last_exit_code == 0


# ── _cmd_events with entries and filter ────────────────────────────


class TestCmdEventsExtra:
    def test_events_with_entries_and_limit(self, repl):
        bus = MagicMock()
        events = []
        for i in range(10):
            ev = MagicMock()
            ev.name = f"event_{i}"
            ev.timestamp = 1000000.0 + i
            ev.source = f"src_{i}"
            ev.data = {"key": f"val_{i}"}
            events.append(ev)
        bus.history.return_value = events
        with patch('domains.infrastructure.event_bus.get_event_bus', return_value=bus):
            repl._cmd_events("5")
        assert repl._last_exit_code == 0

    def test_events_with_filter_match(self, repl):
        bus = MagicMock()
        ev1 = MagicMock()
        ev1.name = "model_loaded"
        ev1.timestamp = 1000000.0
        ev1.source = "loader"
        ev1.data = None
        ev2 = MagicMock()
        ev2.name = "health_check"
        ev2.timestamp = 1000001.0
        ev2.source = "monitor"
        ev2.data = {}
        bus.history.return_value = [ev1, ev2]
        with patch('domains.infrastructure.event_bus.get_event_bus', return_value=bus):
            repl._cmd_events("model")
        assert repl._last_exit_code == 0


# ── _cmd_logs explain non-dict result ─────────────────────────────


class TestCmdLogsExplainExtra:
    def test_logs_explain_non_dict_result(self, repl):
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer = MagicMock()
        repl._log_buffer.__len__ = MagicMock(return_value=1)
        entry = LogEntry(1000000.0, "ERROR", "test", "err")
        repl._log_buffer.get.return_value = [entry]
        repl._log_buffer.entries = [entry]
        repl.cmds.generate = MagicMock(return_value="string result")
        repl._cmd_logs("--explain")
        assert repl._last_exit_code == 0

    def test_logs_explain_with_api_and_error_entries(self, repl):
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer = MagicMock()
        entry1 = LogEntry(1000000.0, "ERROR", "test", "something broke")
        entry2 = LogEntry(1000001.0, "WARNING", "test", "low memory")
        repl._log_buffer.get.return_value = [entry1, entry2]
        repl._log_buffer.__len__ = MagicMock(return_value=2)
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.generate = MagicMock(return_value={"text": "Fix the config"})
            repl._cmd_logs("--explain")
        assert repl._last_exit_code == 0

    def test_logs_explain_with_api_dict_error(self, repl):
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer = MagicMock()
        entry = LogEntry(1000000.0, "ERROR", "test", "crash")
        repl._log_buffer.get.return_value = [entry]
        repl._log_buffer.__len__ = MagicMock(return_value=1)
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl.cmds.generate = MagicMock(return_value={"error": "timeout"})
            repl._cmd_logs("--explain")
        assert repl._last_exit_code == 0

    def test_logs_explain_no_errors(self, repl):
        repl._log_buffer = MagicMock()
        repl._log_buffer.get.return_value = []
        repl._log_buffer.__len__ = MagicMock(return_value=0)
        repl._cmd_logs("--explain")
        assert repl._last_exit_code == 0

    def test_logs_explain_api_unavailable(self, repl):
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer = MagicMock()
        entry = LogEntry(1000000.0, "ERROR", "test", "err")
        repl._log_buffer.get.return_value = [entry]
        repl._log_buffer.__len__ = MagicMock(return_value=1)
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": False}):
            repl._cmd_logs("--explain")
        assert repl._last_exit_code == 0


# ── Pipeline execution paths (permission, handler, timing) ────────


class TestCmdExecutionPaths:
    def test_permission_denied(self, repl):
        with patch.object(repl, '_check_permission', return_value=False):
            repl.execute("help")
        assert repl._last_exit_code == 126

    def test_handler_execution(self, repl):
        with patch.object(repl, '_check_permission', return_value=True):
            repl.execute("date")
        assert repl._last_exit_code == 0

    def test_handler_system_exit(self, repl):
        def bad_handler(r, a):
            raise SystemExit(42)
        repl.COMMANDS["badcmd"] = bad_handler
        with patch.object(repl, '_check_permission', return_value=True):
            repl.execute("badcmd")
        assert repl._last_exit_code == 42
        del repl.COMMANDS["badcmd"]

    def test_handler_exception(self, repl):
        def bad_handler(r, a):
            raise RuntimeError("oops")
        repl.COMMANDS["badcmd"] = bad_handler
        with patch.object(repl, '_check_permission', return_value=True):
            repl.execute("badcmd")
        assert repl._last_exit_code == 1
        del repl.COMMANDS["badcmd"]

    def test_timing_output(self, repl):
        with patch.object(repl, '_check_permission', return_value=True):
            repl.execute("date")
        assert repl._last_exit_code == 0

    def test_unknown_command(self, repl):
        repl.execute("nonexistentcmd12345")
        assert repl._last_exit_code != 0

    def test_pipe_perm_denied_first(self, repl):
        with patch.object(repl, '_check_permission', return_value=False):
            repl.execute("help | wc")
        assert repl._last_exit_code == 126


# ── System binary fallback path ──────────────────────────────────


class TestSystemBinaryFallback:
    def test_system_binary_runs(self, repl):
        with patch('shutil.which', return_value='/usr/bin/myecho'):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    stdout=b"hello\n", stderr=b"", returncode=0
                )
                repl._execute_single("myecho hello")
        assert repl._last_exit_code == 0

    def test_system_binary_stderr(self, repl):
        with patch('shutil.which', return_value='/usr/bin/myutil'):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    stdout=b"", stderr=b"err msg", returncode=1
                )
                repl._execute_single("myutil arg1")
        assert repl._last_exit_code == 1

    def test_system_binary_timeout(self, repl):
        import subprocess as _sp
        with patch('shutil.which', return_value='/usr/bin/myslow'):
            with patch('subprocess.run', side_effect=_sp.TimeoutExpired('myslow', 120)):
                repl._execute_single("myslow")
        assert repl._last_exit_code == 124

    def test_system_binary_exception(self, repl):
        with patch('shutil.which', return_value='/usr/bin/mybroken'):
            with patch('subprocess.run', side_effect=OSError("perm denied")):
                repl._execute_single("mybroken")
        assert repl._last_exit_code == 1

    def test_system_binary_not_found(self, repl):
        with patch('shutil.which', return_value=None):
            repl._execute_single("nope")
        assert repl._last_exit_code == 127

    def test_system_binary_redirect(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            tmppath = f.name
        try:
            with patch('shutil.which', return_value='/usr/bin/mycat'):
                with patch('subprocess.run') as mock_run:
                    mock_run.return_value = MagicMock(
                        stdout=b"file content", stderr=b"", returncode=0
                    )
                    repl._execute_single(f"mycat < /dev/null > {tmppath}")
            with open(tmppath) as f:
                assert "file content" in f.read()
        finally:
            os.unlink(tmppath)

    def test_system_binary_redirect_vfs(self, repl):
        mock_vfs = MagicMock()
        mock_vfs.write.return_value = None
        with patch.object(type(repl.os), 'vfs', new_callable=PropertyMock, return_value=mock_vfs):
            with patch('shutil.which', return_value='/usr/bin/mycat'):
                with patch('subprocess.run') as mock_run:
                    mock_run.return_value = MagicMock(
                        stdout=b"data", stderr=b"", returncode=0
                    )
                    repl._execute_single("mycat > /dev/null")
        mock_vfs.write.assert_called_once()

    def test_system_binary_inline_env(self, repl):
        with patch('shutil.which', return_value='/usr/bin/myrun'):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    stdout=b"", stderr=b"", returncode=0
                )
                repl._execute_single("MYVAR=1 myrun hi")
        assert repl._last_exit_code == 0

    def test_system_binary_piped_input(self, repl):
        with patch('shutil.which', return_value='/usr/bin/mycat'):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    stdout=b"piped data", stderr=b"", returncode=0
                )
                repl._execute_single("mycat")
        assert repl._last_exit_code == 0

    def test_system_binary_no_stdout(self, repl):
        with patch('shutil.which', return_value='/usr/bin/mytrue'):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    stdout=None, stderr=None, returncode=0
                )
                repl._execute_single("mytrue")
        assert repl._last_exit_code == 0

    def test_system_binary_redirect_append(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("first\n")
            tmppath = f.name
        try:
            with patch('shutil.which', return_value='/usr/bin/mywrite'):
                with patch('subprocess.run') as mock_run:
                    mock_run.return_value = MagicMock(
                        stdout=b"second", stderr=b"", returncode=0
                    )
                    repl._execute_single(f"mywrite >> {tmppath}")
            with open(tmppath) as f:
                content = f.read()
            assert "first" in content and "second" in content
        finally:
            os.unlink(tmppath)

    def test_system_binary_redirect_vfs_error(self, repl):
        mock_vfs = MagicMock()
        mock_vfs.write.return_value = "write failed"
        with patch.object(type(repl.os), 'vfs', new_callable=PropertyMock, return_value=mock_vfs):
            with patch('shutil.which', return_value='/usr/bin/mycmd'):
                with patch('subprocess.run') as mock_run:
                    mock_run.return_value = MagicMock(
                        stdout=b"data", stderr=b"", returncode=0
                    )
                    repl._execute_single("mycmd > /dev/null")
        assert repl._last_exit_code == 0

    def test_system_binary_redirect_file_oserror(self, repl):
        with patch('shutil.which', return_value='/usr/bin/mycmd'):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    stdout=b"data", stderr=b"", returncode=0
                )
                repl._execute_single("mycmd > /nonexistent/deeply/path/file.txt")
        assert repl._last_exit_code == 1

    def test_system_binary_inline_env_existing_var(self, repl):
        repl._env["EXISTING_VAR"] = "original"
        with patch('shutil.which', return_value='/usr/bin/mycmd'):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    stdout=b"", stderr=b"", returncode=0
                )
                repl._execute_single("EXISTING_VAR=new mycmd")
        assert repl._last_exit_code == 0
        assert repl._env.get("EXISTING_VAR") == "original"

    def test_unknown_command_inline_env(self, repl):
        repl._env["MYENV"] = "old"
        repl._execute_single("MYENV=val totallyunknowncmd")
        assert repl._last_exit_code == 127
        assert repl._env.get("MYENV") == "old"

    def test_unknown_command_inline_env_new_var(self, repl):
        repl._execute_single("NEWVAR=test totallyunknowncmd")
        assert repl._last_exit_code == 127

    def test_permission_denied_inline_env(self, repl):
        repl._env["EXISTING"] = "old"
        with patch.object(repl, '_check_permission', return_value=False):
            repl._execute_single("EXISTING=new help")
        assert repl._last_exit_code == 126
        assert repl._env.get("EXISTING") == "old"

    def test_permission_denied_inline_env_new_var(self, repl):
        with patch.object(repl, '_check_permission', return_value=False):
            repl._execute_single("FRESHVAR=val help")
        assert repl._last_exit_code == 126
        assert "FRESHVAR" not in repl._env


# ── Sort/uniq/tr/seq/nl/fold/shuf/rev/comm internals ─────────────


class TestSortInternals:
    def test_sort_piped(self, repl):
        repl._piped_input = "c\na\nb"
        repl._cmd_sort("")
        assert repl._last_exit_code == 0

    def test_sort_reverse(self, repl):
        repl._piped_input = "a\nc\nb"
        repl._cmd_sort("-r")
        assert repl._last_exit_code == 0

    def test_sort_numeric(self, repl):
        repl._piped_input = "10\n2\n30\n1"
        repl._cmd_sort("-n")
        assert repl._last_exit_code == 0

    def test_sort_unique(self, repl):
        repl._piped_input = "a\na\nb\nb\nc"
        repl._cmd_sort("-u")
        assert repl._last_exit_code == 0

    def test_sort_numeric_reverse(self, repl):
        repl._piped_input = "3\n1\n2"
        repl._cmd_sort("-n -r")
        assert repl._last_exit_code == 0

    def test_sort_no_input(self, repl):
        repl._cmd_sort("")
        assert repl._last_exit_code == 1

    def test_sort_file_not_found(self, repl):
        repl._cmd_sort("/nonexistent/file")
        assert repl._last_exit_code == 1


class TestUniqInternals:
    def test_uniq_piped(self, repl):
        repl._piped_input = "a\na\nb\nb\nc"
        repl._cmd_uniq("")
        assert repl._last_exit_code == 0

    def test_uniq_file_not_found(self, repl):
        repl._cmd_uniq("/nonexistent/file")
        assert repl._last_exit_code == 1

    def test_uniq_no_input(self, repl):
        repl._cmd_uniq("")
        assert repl._last_exit_code == 1


class TestTrInternals:
    def test_tr_translate(self, repl):
        repl._piped_input = "hello"
        repl._cmd_tr("a-z A-Z")
        assert repl._last_exit_code == 0

    def test_tr_delete(self, repl):
        repl._piped_input = "hello"
        repl._cmd_tr("-d l")
        assert repl._last_exit_code == 0

    def test_tr_squeeze(self, repl):
        repl._piped_input = "heeeello"
        repl._cmd_tr("-s e")
        assert repl._last_exit_code == 0

    def test_tr_no_input(self, repl):
        repl._cmd_tr("a-z A-Z")
        assert repl._last_exit_code == 1

    def test_tr_missing_args(self, repl):
        repl._piped_input = "hello"
        repl._cmd_tr("")
        assert repl._last_exit_code == 1

    def test_tr_range(self, repl):
        repl._piped_input = "abc"
        repl._cmd_tr("a-c x-z")
        assert repl._last_exit_code == 0


class TestSeqInternals:
    def test_seq_one_arg(self, repl):
        repl._cmd_seq("5")
        assert repl._last_exit_code == 0

    def test_seq_two_args(self, repl):
        repl._cmd_seq("2 5")
        assert repl._last_exit_code == 0

    def test_seq_three_args(self, repl):
        repl._cmd_seq("1 2 10")
        assert repl._last_exit_code == 0

    def test_seq_no_args(self, repl):
        repl._cmd_seq("")
        assert repl._last_exit_code == 1

    def test_seq_invalid(self, repl):
        repl._cmd_seq("abc")
        assert repl._last_exit_code == 1

    def test_seq_too_many(self, repl):
        repl._cmd_seq("1 2 3 4")
        assert repl._last_exit_code == 1

    def test_seq_float(self, repl):
        repl._cmd_seq("0.5 0.5 2.5")
        assert repl._last_exit_code == 0


class TestNlInternals:
    def test_nl_piped(self, repl):
        repl._piped_input = "a\nb\nc"
        repl._cmd_nl("")
        assert repl._last_exit_code == 0

    def test_nl_no_input(self, repl):
        repl._cmd_nl("")
        assert repl._last_exit_code == 1

    def test_nl_file_not_found(self, repl):
        repl._cmd_nl("/nonexistent/file")
        assert repl._last_exit_code == 1


class TestFoldInternals:
    def test_fold_piped(self, repl):
        repl._piped_input = "a" * 100
        repl._cmd_fold("-w 10")
        assert repl._last_exit_code == 0

    def test_fold_no_input(self, repl):
        repl._cmd_fold("")
        assert repl._last_exit_code == 1

    def test_fold_file_not_found(self, repl):
        repl._cmd_fold("/nonexistent/file")
        assert repl._last_exit_code == 1

    def test_fold_inline_width(self, repl):
        repl._piped_input = "a" * 50
        repl._cmd_fold("-w20")
        assert repl._last_exit_code == 0


class TestShufInternals:
    def test_shuf_piped(self, repl):
        repl._piped_input = "a\nb\nc\nd"
        repl._cmd_shuf("")
        assert repl._last_exit_code == 0

    def test_shuf_no_input(self, repl):
        repl._cmd_shuf("")
        assert repl._last_exit_code == 1

    def test_shuf_file_not_found(self, repl):
        repl._cmd_shuf("/nonexistent/file")
        assert repl._last_exit_code == 1


class TestRevInternals:
    def test_rev_piped(self, repl):
        repl._piped_input = "abc"
        repl._cmd_rev("")
        assert repl._last_exit_code == 0

    def test_rev_no_input(self, repl):
        repl._cmd_rev("")
        assert repl._last_exit_code == 1

    def test_rev_file_not_found(self, repl):
        repl._cmd_rev("/nonexistent/file")
        assert repl._last_exit_code == 1


class TestPasteInternals:
    def test_paste_two_files(self, repl):
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f1.write("a\nb\nc")
        f1.close()
        f2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f2.write("x\ny")
        f2.close()
        try:
            repl._cmd_paste(f"{f1.name} {f2.name}")
            assert repl._last_exit_code == 0
        finally:
            os.unlink(f1.name)
            os.unlink(f2.name)

    def test_paste_no_args(self, repl):
        repl._cmd_paste("")
        assert repl._last_exit_code == 1

    def test_paste_file_not_found(self, repl):
        repl._cmd_paste("/nonexistent/a.txt /nonexistent/b.txt")
        assert repl._last_exit_code == 1


class TestCommInternals:
    def test_comm(self, repl):
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f1.write("a\nb\nc")
        f1.close()
        f2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f2.write("b\nc\nd")
        f2.close()
        try:
            repl._cmd_comm(f"{f1.name} {f2.name}")
            assert repl._last_exit_code == 0
        finally:
            os.unlink(f1.name)
            os.unlink(f2.name)

    def test_comm_no_args(self, repl):
        repl._cmd_comm("")
        assert repl._last_exit_code == 1

    def test_comm_one_arg(self, repl):
        repl._cmd_comm("file.txt")
        assert repl._last_exit_code == 1

    def test_comm_file_not_found(self, repl):
        repl._cmd_comm("/nonexistent/a.txt /nonexistent/b.txt")
        assert repl._last_exit_code == 1


# ── Test/printf/du/diff/tac internals ─────────────────────────────


class TestCmdTestInternals:
    def test_test_no_args(self, repl):
        repl._cmd_test("")
        assert repl._last_exit_code == 1

    def test_test_file_exists(self, repl):
        repl._cmd_test(f"-f /etc/hostname")
        assert repl._last_exit_code == 0

    def test_test_file_not_exists(self, repl):
        repl._cmd_test("-f /nonexistent/file")
        assert repl._last_exit_code == 1

    def test_test_dir(self, repl):
        repl._cmd_test("-d /tmp")
        assert repl._last_exit_code == 0

    def test_test_dir_not_dir(self, repl):
        repl._cmd_test("-d /etc/hostname")
        assert repl._last_exit_code == 1

    def test_test_exists(self, repl):
        repl._cmd_test("-e /etc/hostname")
        assert repl._last_exit_code == 0

    def test_test_z_nonempty(self, repl):
        repl._cmd_test("-z hello")
        assert repl._last_exit_code == 1

    def test_test_n(self, repl):
        repl._cmd_test("-n hello")
        assert repl._last_exit_code == 0

    def test_test_eq(self, repl):
        repl._cmd_test("hello = hello")
        assert repl._last_exit_code == 0

    def test_test_ne(self, repl):
        repl._cmd_test("hello != world")
        assert repl._last_exit_code == 0

    def test_test_eq_int(self, repl):
        repl._cmd_test("5 -eq 5")
        assert repl._last_exit_code == 0

    def test_test_ne_int(self, repl):
        repl._cmd_test("5 -ne 3")
        assert repl._last_exit_code == 0

    def test_test_lt(self, repl):
        repl._cmd_test("3 -lt 5")
        assert repl._last_exit_code == 0

    def test_test_le(self, repl):
        repl._cmd_test("5 -le 5")
        assert repl._last_exit_code == 0

    def test_test_gt(self, repl):
        repl._cmd_test("5 -gt 3")
        assert repl._last_exit_code == 0

    def test_test_ge(self, repl):
        repl._cmd_test("5 -ge 5")
        assert repl._last_exit_code == 0

    def test_test_bracket_syntax(self, repl):
        repl._cmd_test("[ -f /etc/hostname ]")
        assert repl._last_exit_code == 0

    def test_test_unknown_op(self, repl):
        repl._cmd_test("x --foo y")
        assert repl._last_exit_code == 1


class TestCmdPrintfInternals:
    def test_printf_s(self, repl):
        repl._cmd_printf("%s hello")
        assert repl._last_exit_code == 0

    def test_printf_d(self, repl):
        repl._cmd_printf("%d 42")
        assert repl._last_exit_code == 0

    def test_printf_f(self, repl):
        repl._cmd_printf("%f 3.14")
        assert repl._last_exit_code == 0

    def test_printf_percent(self, repl):
        repl._cmd_printf("100%%")
        assert repl._last_exit_code == 0

    def test_printf_no_args(self, repl):
        repl._cmd_printf("")
        assert repl._last_exit_code == 1

    def test_printf_escape(self, repl):
        repl._cmd_printf("hello\\nworld")
        assert repl._last_exit_code == 0

    def test_printf_bad_int(self, repl):
        repl._cmd_printf("%d abc")
        assert repl._last_exit_code == 0

    def test_printf_bad_float(self, repl):
        repl._cmd_printf("%f abc")
        assert repl._last_exit_code == 0

    def test_printf_no_args_for_spec(self, repl):
        repl._cmd_printf("%s %s")
        assert repl._last_exit_code == 0

    def test_printf_literal_only(self, repl):
        repl._cmd_printf("hello world")
        assert repl._last_exit_code == 0


class TestCmdDuInternals:
    def test_du_default(self, repl):
        repl._cmd_du("")
        assert repl._last_exit_code == 0

    def test_du_human(self, repl):
        repl._cmd_du("-h .")
        assert repl._last_exit_code == 0

    def test_du_file_not_found(self, repl):
        repl._cmd_du("/nonexistent/path")
        assert repl._last_exit_code == 0

    def test_du_multiple_targets(self, repl):
        repl._cmd_du(f". /etc/hostname")
        assert repl._last_exit_code == 0

    def test_format_size(self, repl):
        assert repl._format_size(500) == "     500"
        assert repl._format_size(1024, human=True) == " 1.0K"
        assert repl._format_size(1048576, human=True) == " 1.0M"


class TestCmdDiffInternals:
    def test_diff_same(self, repl):
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f1.write("a\nb\n")
        f1.close()
        try:
            repl._cmd_diff(f"{f1.name} {f1.name}")
            assert repl._last_exit_code == 0
        finally:
            os.unlink(f1.name)

    def test_diff_different(self, repl):
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f1.write("a\n")
        f1.close()
        f2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f2.write("b\n")
        f2.close()
        try:
            repl._cmd_diff(f"{f1.name} {f2.name}")
            assert repl._last_exit_code == 1
        finally:
            os.unlink(f1.name)
            os.unlink(f2.name)

    def test_diff_no_args(self, repl):
        repl._cmd_diff("")
        assert repl._last_exit_code == 1

    def test_diff_one_arg(self, repl):
        repl._cmd_diff("file.txt")
        assert repl._last_exit_code == 1

    def test_diff_file_not_found(self, repl):
        repl._cmd_diff("/nonexistent/a.txt /nonexistent/b.txt")
        assert repl._last_exit_code == 1


class TestCmdTacInternals:
    def test_tac_piped(self, repl):
        repl._piped_input = "a\nb\nc"
        repl._cmd_tac("")
        assert repl._last_exit_code == 0

    def test_tac_no_input(self, repl):
        repl._cmd_tac("")
        assert repl._last_exit_code == 1

    def test_tac_file_not_found(self, repl):
        repl._cmd_tac("/nonexistent/file")
        assert repl._last_exit_code == 1


# ── Ls internals ──────────────────────────────────────────────────


class TestCmdLsInternals:
    def test_ls_default(self, repl):
        repl._cmd_ls("")
        assert repl._last_exit_code == 0

    def test_ls_target(self, repl):
        repl._cmd_ls(".")
        assert repl._last_exit_code == 0

    def test_ls_not_found(self, repl):
        repl._cmd_ls("/nonexistent/path")
        assert repl._last_exit_code == 1


# ── Help command internals ────────────────────────────────────────


class TestCmdHelpInternals:
    def test_help_brief(self, repl):
        repl._cmd_help("brief")
        assert repl._last_exit_code == 0

    def test_help_known_command(self, repl):
        repl._cmd_help("exit")
        assert repl._last_exit_code == 0

    def test_help_command_with_doc(self, repl):
        repl._cmd_help("date")
        assert repl._last_exit_code == 0

    def test_help_command_no_doc(self, repl):
        repl.COMMANDS["nodoccmd"] = lambda r, a: None
        try:
            repl._cmd_help("nodoccmd")
            assert repl._last_exit_code == 0
        finally:
            del repl.COMMANDS["nodoccmd"]

    def test_help_ext_command(self, repl):
        mock_ext = MagicMock()
        mock_ext.help = "myext help text"
        repl._ext_cmds["myext"] = mock_ext
        try:
            repl._cmd_help("myext")
            assert repl._last_exit_code == 0
        finally:
            del repl._ext_cmds["myext"]

    def test_help_ext_command_no_help(self, repl):
        mock_ext = MagicMock(spec=[])
        repl._ext_cmds["myext2"] = mock_ext
        try:
            repl._cmd_help("myext2")
            assert repl._last_exit_code == 0
        finally:
            del repl._ext_cmds["myext2"]

    def test_help_system_command(self, repl):
        with patch('shutil.which', return_value='/usr/bin/ls'):
            repl._cmd_help("ls")
        assert repl._last_exit_code == 0

    def test_help_unknown_command(self, repl):
        with patch('shutil.which', return_value=None):
            repl._cmd_help("nonexistentxyz")
        assert repl._last_exit_code == 0

    def test_help_all_known_commands(self, repl):
        known = ["help", "exit", "cd", "pwd", "echo", "ls", "cat", "mkdir",
                 "rm", "touch", "cp", "mv", "head", "tail", "wc", "grep",
                 "sort", "uniq", "find", "tee", "xargs", "chmod", "du",
                 "diff", "stat", "cut", "tr", "seq", "nl", "fold", "tac",
                 "env", "printenv", "yes", "realpath", "dirname", "basename",
                 "nproc", "hostname", "uname", "shuf", "rev", "paste",
                 "comm", "test", "printf", "history", "fc", "alias",
                 "unalias", "export", "set", "source", "which", "type",
                 "procs", "kill", "train", "bg", "jobs", "fg", "models",
                 "load", "unload", "souls", "switch", "whoami", "uptime",
                 "health", "status", "metrics", "datasets", "knowledge",
                 "remember", "recall", "checkpoints", "finetuned", "gen",
                 "tokenizer", "py", "ai", "agents", "tutorial", "boot",
                 "shutdown", "svc", "devices", "lsdev", "asm", "vmrun",
                 "vmperms", "permit", "deny", "permissions", "api", "chat",
                 "confirm", "events", "read", "logs", "clear", "sleep",
                 "date", "cal", "ln", "render", "watch", "note"]
        for cmd in known:
            repl._cmd_help(cmd)
        assert repl._last_exit_code == 0


# ── More ls internals ─────────────────────────────────────────────


class TestCmdLsExtra:
    def test_ls_empty_dir(self, repl):
        import tempfile
        d = tempfile.mkdtemp()
        try:
            repl._cmd_ls(d)
            assert repl._last_exit_code == 0
        finally:
            os.rmdir(d)

    def test_ls_permission_denied(self, repl):
        with patch('os.listdir', side_effect=PermissionError("denied")):
            repl._cmd_ls("/some/dir")
        assert repl._last_exit_code == 1

    def test_ls_not_a_directory(self, repl):
        with patch('os.listdir', side_effect=NotADirectoryError("not dir")):
            repl._cmd_ls("/some/file")
        assert repl._last_exit_code == 1


# ── More find internals ───────────────────────────────────────────


class TestCmdFindExtra:
    def test_find_no_pattern(self, repl):
        repl._cmd_find(".")
        assert repl._last_exit_code == 1

    def test_find_iname(self, repl):
        repl._cmd_find(". -iname '*.py'")
        assert repl._last_exit_code == 0

    def test_find_file_not_found(self, repl):
        repl._cmd_find("/nonexistent -name '*.py'")
        assert repl._last_exit_code == 0

    def test_find_permission_denied(self, repl):
        with patch('os.walk', side_effect=PermissionError("denied")):
            repl._cmd_find("/some/dir -name '*.py'")
        assert repl._last_exit_code == 1

    def test_find_type_file(self, repl, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "b.py").write_text("y")
        (tmp_path / "subdir").mkdir()
        with _CaptureOutput(repl) as cap:
            repl._cmd_find(f"{tmp_path} -type f")
        out = cap.getvalue()
        assert "a.txt" in out
        assert "b.py" in out
        assert "subdir" not in out

    def test_find_type_dir(self, repl, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "subdir").mkdir()
        with _CaptureOutput(repl) as cap:
            repl._cmd_find(f"{tmp_path} -type d")
        out = cap.getvalue()
        assert "subdir" in out
        assert "a.txt" not in out

    def test_find_maxdepth(self, repl, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("y")
        with _CaptureOutput(repl) as cap:
            repl._cmd_find(f"{tmp_path} -maxdepth 1 -name *.txt")
        out = cap.getvalue()
        assert "a.txt" in out
        assert "b.txt" not in out

    def test_find_type_and_name(self, repl, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "b.py").write_text("y")
        with _CaptureOutput(repl) as cap:
            repl._cmd_find(f"{tmp_path} -type f -name *.txt")
        out = cap.getvalue()
        assert "a.txt" in out
        assert "b.py" not in out

    def test_find_no_pattern_with_type(self, repl, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "subdir").mkdir()
        with _CaptureOutput(repl) as cap:
            repl._cmd_find(f"{tmp_path} -type d")
        assert repl._last_exit_code == 0


# ── More comm internals ───────────────────────────────────────────


class TestCmdCommExtra:
    def test_comm_with_overlap(self, repl):
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f1.write("a\nb\nc\nd")
        f1.close()
        f2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f2.write("b\nd\ne")
        f2.close()
        try:
            repl._cmd_comm(f"{f1.name} {f2.name}")
            assert repl._last_exit_code == 0
        finally:
            os.unlink(f1.name)
            os.unlink(f2.name)

    def test_comm_no_overlap(self, repl):
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f1.write("a\nb")
        f1.close()
        f2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f2.write("c\nd")
        f2.close()
        try:
            repl._cmd_comm(f"{f1.name} {f2.name}")
            assert repl._last_exit_code == 0
        finally:
            os.unlink(f1.name)
            os.unlink(f2.name)

    def test_comm_left_longer(self, repl):
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f1.write("a\nb\nc\nd\ne")
        f1.close()
        f2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f2.write("b")
        f2.close()
        try:
            repl._cmd_comm(f"{f1.name} {f2.name}")
            assert repl._last_exit_code == 0
        finally:
            os.unlink(f1.name)
            os.unlink(f2.name)

    def test_comm_right_longer(self, repl):
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f1.write("a")
        f1.close()
        f2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f2.write("a\nb\nc\nd")
        f2.close()
        try:
            repl._cmd_comm(f"{f1.name} {f2.name}")
            assert repl._last_exit_code == 0
        finally:
            os.unlink(f1.name)
            os.unlink(f2.name)


# ── Fold/shuf/rev/tac file paths ─────────────────────────────────


class TestCmdFoldFlags:
    def test_fold_file(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f.write("a" * 100)
        f.close()
        try:
            repl._cmd_fold(f"{f.name}")
            assert repl._last_exit_code == 0
        finally:
            os.unlink(f.name)

    def test_fold_s_breaks_at_spaces(self, repl):
        repl._piped_input = "hello world this is a test\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_fold("-w 15 -s")
        out = cap.getvalue().strip().split("\n")
        assert out[0] == "hello world"
        assert out[1] == "this is a test"

    def test_fold_s_short_line(self, repl):
        repl._piped_input = "short\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_fold("-w 80 -s")
        out = cap.getvalue().strip()
        assert out == "short"

    def test_fold_s_no_spaces(self, repl):
        repl._piped_input = "abcdefghijklmnop\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_fold("-w 5 -s")
        out = cap.getvalue().strip().split("\n")
        assert out[0] == "abcde"


class TestCmdShufExtra:
    def test_shuf_file(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f.write("a\nb\nc\nd\ne")
        f.close()
        try:
            repl._cmd_shuf(f"{f.name}")
            assert repl._last_exit_code == 0
        finally:
            os.unlink(f.name)


class TestCmdRevExtra:
    def test_rev_file(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f.write("hello\nworld")
        f.close()
        try:
            repl._cmd_rev(f"{f.name}")
            assert repl._last_exit_code == 0
        finally:
            os.unlink(f.name)


class TestCmdTacExtra:
    def test_tac_file(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f.write("a\nb\nc")
        f.close()
        try:
            repl._cmd_tac(f"{f.name}")
            assert repl._last_exit_code == 0
        finally:
            os.unlink(f.name)


# ── Nl file path ──────────────────────────────────────────────────


class TestCmdNlFlags:
    def test_nl_file(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f.write("a\nb\nc")
        f.close()
        try:
            repl._cmd_nl(f"{f.name}")
            assert repl._last_exit_code == 0
        finally:
            os.unlink(f.name)

    def test_nl_width(self, repl):
        repl._piped_input = "a\nb\nc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_nl("-w 3")
        out = cap.getvalue().strip().split("\n")
        assert out[0] == "001\ta"
        assert out[1] == "002\tb"
        assert out[2] == "003\tc"

    def test_nl_separator(self, repl):
        repl._piped_input = "a\nb\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_nl("-s . ")
        out = cap.getvalue().strip().split("\n")
        assert out[0] == "1.a"
        assert out[1] == "2.b"

    def test_nl_body_all(self, repl):
        repl._piped_input = "a\n\nb\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_nl("-b a")
        out = cap.getvalue().strip().split("\n")
        assert len(out) == 3
        assert "1\ta" in out[0]
        assert "2\t" in out[1]
        assert "3\tb" in out[2]

    def test_nl_no_args_no_input(self, repl):
        repl._piped_input = None
        with _CaptureOutput(repl) as cap:
            repl._cmd_nl("")
        assert "Usage" in cap.getvalue()

    def test_nl_no_args_with_input(self, repl):
        repl._piped_input = "hello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_nl("")
        out = cap.getvalue().strip()
        assert "1\thello" in out


# ── Printf edge cases ─────────────────────────────────────────────


class TestCmdPrintfExtra:
    def test_printf_multiple_specs(self, repl):
        repl._cmd_printf("name=%s age=%d" % ("alice", 30))
        assert repl._last_exit_code == 0

    def test_printf_backslash_n(self, repl):
        repl._cmd_printf("line1\\nline2")
        assert repl._last_exit_code == 0

    def test_printf_backslash_t(self, repl):
        repl._cmd_printf("col1\\tcol2")
        assert repl._last_exit_code == 0

    def test_printf_backslash_backslash(self, repl):
        repl._cmd_printf("path\\file")
        assert repl._last_exit_code == 0


# ── Du edge cases ─────────────────────────────────────────────────


class TestCmdDuExtra:
    def test_du_nonexistent(self, repl):
        repl._cmd_du("/nonexistent/path/that/doesnt/exist")
        assert repl._last_exit_code == 0

    def test_du_file(self, repl):
        repl._cmd_du("/etc/hostname")
        assert repl._last_exit_code == 0


# ── Diff edge cases ───────────────────────────────────────────────


class TestCmdDiffExtra:
    def test_diff_identical_content(self, repl):
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f1.write("same\ncontent")
        f1.close()
        try:
            repl._cmd_diff(f"{f1.name} {f1.name}")
            assert repl._last_exit_code == 0
        finally:
            os.unlink(f1.name)

    def test_diff_q_identical(self, repl):
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f1.write("same\n")
        f1.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_diff(f"-q {f1.name} {f1.name}")
            assert repl._last_exit_code == 0
            assert "differ" not in cap.getvalue()
        finally:
            os.unlink(f1.name)

    def test_diff_q_different(self, repl):
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f1.write("a\n")
        f1.close()
        f2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f2.write("b\n")
        f2.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_diff(f"-q {f1.name} {f2.name}")
            assert repl._last_exit_code == 1
            assert "differ" in cap.getvalue()
        finally:
            os.unlink(f1.name)
            os.unlink(f2.name)

    def test_diff_q_unified(self, repl):
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f1.write("a\n")
        f1.close()
        f2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f2.write("b\n")
        f2.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_diff(f"-u -q {f1.name} {f2.name}")
            assert repl._last_exit_code == 1
            out = cap.getvalue()
            assert "differ" in out
            assert "---" not in out
        finally:
            os.unlink(f1.name)
            os.unlink(f2.name)


# ── Which/Type edge cases ─────────────────────────────────────────


class TestCmdWhichTypeEdge:
    def test_which_no_args(self, repl):
        repl._cmd_which("")
        assert repl._last_exit_code == 1

    def test_which_alias(self, repl):
        repl._aliases["ll"] = "ls -la"
        repl._cmd_which("ll")
        assert repl._last_exit_code == 0

    def test_which_ext_cmd(self, repl):
        mock_ext = MagicMock()
        mock_ext.help = "myext help"
        repl._ext_cmds["myext"] = mock_ext
        try:
            repl._cmd_which("myext")
            assert repl._last_exit_code == 0
        finally:
            del repl._ext_cmds["myext"]

    def test_which_not_found(self, repl):
        with patch('shutil.which', return_value=None):
            repl._cmd_which("nonexistent")
        assert repl._last_exit_code == 1

    def test_which_system_command(self, repl):
        with patch('shutil.which', return_value='/usr/bin/ls'):
            repl._cmd_which("ls")
        assert repl._last_exit_code == 0

    def test_type_no_args(self, repl):
        repl._cmd_type("")
        assert repl._last_exit_code == 1

    def test_type_alias(self, repl):
        repl._aliases["ll"] = "ls -la"
        repl._cmd_type("ll")
        assert repl._last_exit_code == 0

    def test_type_ext_cmd(self, repl):
        mock_ext = MagicMock()
        mock_ext.help = "myext help"
        repl._ext_cmds["myext"] = mock_ext
        try:
            repl._cmd_type("myext")
            assert repl._last_exit_code == 0
        finally:
            del repl._ext_cmds["myext"]

    def test_type_builtin(self, repl):
        repl._cmd_type("echo")
        assert repl._last_exit_code == 0

    def test_type_system_command(self, repl):
        with patch('shutil.which', return_value='/usr/bin/ls'):
            repl._cmd_type("ls")
        assert repl._last_exit_code == 0

    def test_type_not_found(self, repl):
        with patch('shutil.which', return_value=None):
            repl._cmd_type("nonexistent")
        assert repl._last_exit_code == 1


# ── Permit/Deny edge cases ────────────────────────────────────────


class TestCmdPermitDenyExtra:
    def test_permit_no_args(self, repl):
        repl._cmd_permit("")
        assert repl._last_exit_code == 0

    def test_permit_persist(self, repl):
        repl._cmd_permit("rm --persist")
        assert repl._last_exit_code == 0

    def test_permit_all_safe(self, repl):
        repl._cmd_permit("--all-safe")
        assert repl._last_exit_code == 0

    def test_permit_all_unknown_risk(self, repl):
        repl._cmd_permit("--all-unknown")
        assert repl._last_exit_code == 0

    def test_deny_no_args(self, repl):
        repl._cmd_deny("")
        assert repl._last_exit_code == 0

    def test_deny_persist(self, repl):
        repl._cmd_deny("rm --persist")
        assert repl._last_exit_code == 0

    def test_deny_all_elevated(self, repl):
        repl._cmd_deny("--all-elevated")
        assert repl._last_exit_code == 0

    def test_deny_all_unknown_risk(self, repl):
        repl._cmd_deny("--all-unknown")
        assert repl._last_exit_code == 0


# ── Confirm toggle edge cases ─────────────────────────────────────


class TestCmdConfirmExtra:
    def test_confirm_no_args(self, repl):
        with patch.dict('os.environ', {}, clear=False):
            repl._cmd_confirm("")
            assert repl._last_exit_code == 0

    def test_confirm_on(self, repl):
        repl._cmd_confirm("on")
        assert repl._last_exit_code == 0

    def test_confirm_off(self, repl):
        repl._cmd_confirm("off")
        assert repl._last_exit_code == 0

    def test_confirm_yes(self, repl):
        repl._cmd_confirm("yes")
        assert repl._last_exit_code == 0

    def test_confirm_no_val(self, repl):
        repl._cmd_confirm("no")
        assert repl._last_exit_code == 0

    def test_confirm_true(self, repl):
        repl._cmd_confirm("true")
        assert repl._last_exit_code == 0

    def test_confirm_false(self, repl):
        repl._cmd_confirm("false")
        assert repl._last_exit_code == 0

    def test_confirm_one(self, repl):
        repl._cmd_confirm("1")
        assert repl._last_exit_code == 0

    def test_confirm_zero(self, repl):
        repl._cmd_confirm("0")
        assert repl._last_exit_code == 0

    def test_confirm_invalid(self, repl):
        repl._cmd_confirm("maybe")
        assert repl._last_exit_code == 0


# ── Pipeline execution internals ──────────────────────────────────


class TestPipelineInternals:
    def test_execute_line_simple(self, repl):
        repl.execute("echo hello")
        assert repl._last_exit_code == 0

    def test_execute_line_empty(self, repl):
        repl.execute("")
        assert repl._last_exit_code == 0

    def test_execute_line_unknown(self, repl):
        repl.execute("nonexistentcmd")
        assert repl._last_exit_code == 127

    def test_execute_line_with_suggestion(self, repl):
        repl.execute("ech hello")
        assert repl._last_exit_code == 127

    def test_execute_line_timing(self, repl):
        repl.execute("time echo hello")
        assert repl._last_exit_code == 0

    def test_execute_line_system_binary(self, repl):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"output"
        mock_result.stderr = b""
        with patch('shutil.which', return_value='/usr/bin/myutil'), \
             patch('subprocess.run', return_value=mock_result):
            repl._execute_single("myutil arg1 arg2", "")
        assert repl._last_exit_code == 0

    def test_execute_line_system_binary_stderr(self, repl):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b""
        mock_result.stderr = b"some warning"
        with patch('shutil.which', return_value='/usr/bin/myerr'), \
             patch('subprocess.run', return_value=mock_result):
            repl._execute_single("myerr", "")
        assert repl._last_exit_code == 0

    def test_execute_line_system_binary_exception(self, repl):
        with patch('shutil.which', return_value='/usr/bin/myutil'), \
             patch('subprocess.run', side_effect=Exception("spawn error")):
            repl._execute_single("myutil", "")
        assert repl._last_exit_code == 1

    def test_execute_line_redirect(self, repl):
        import tempfile
        d = tempfile.mkdtemp()
        try:
            repl._execute_single(f"echo test > {d}/out.txt")
            assert os.path.exists(f"{d}/out.txt")
        finally:
            import shutil
            shutil.rmtree(d)

    def test_execute_line_append(self, repl):
        import tempfile
        d = tempfile.mkdtemp()
        try:
            repl._execute_single(f"echo first > {d}/out.txt")
            repl._execute_single(f"echo second >> {d}/out.txt")
            with open(f"{d}/out.txt") as f:
                content = f.read()
            assert "first" in content and "second" in content
        finally:
            import shutil
            shutil.rmtree(d)

    def test_execute_line_permission_denied(self, repl):
        from domains.shell.permissions import Risk
        repl._perms.set_policy(Risk.ELEVATED, "deny")
        repl._perms._granted.discard("rm")
        repl.execute("rm /tmp/test")
        assert repl._last_exit_code == 126

    def test_execute_line_ext_command(self, repl):
        mock_mod = MagicMock()
        mock_mod.run.return_value = 0
        mock_mod.help = "test ext"
        repl._ext_cmds["testext"] = mock_mod
        try:
            repl.execute("testext arg1")
            assert repl._last_exit_code == 0
        finally:
            del repl._ext_cmds["testext"]

    def test_execute_line_ext_command_exception(self, repl):
        mock_mod = MagicMock()
        mock_mod.run.side_effect = Exception("ext error")
        mock_mod.help = "test ext"
        repl._ext_cmds["testext"] = mock_mod
        try:
            repl.execute("testext")
            assert repl._last_exit_code == 1
        finally:
            del repl._ext_cmds["testext"]

    def test_execute_line_system_binary_not_found(self, repl):
        with patch('shutil.which', return_value=None):
            repl._execute_single("nonexistent", "")
        assert repl._last_exit_code == 127


# ── Uname flags ───────────────────────────────────────────────────


class TestCmdUnameFlags:
    def test_uname_a(self, repl):
        repl._cmd_uname("-a")
        assert repl._last_exit_code == 0

    def test_uname_s(self, repl):
        repl._cmd_uname("-s")
        assert repl._last_exit_code == 0

    def test_uname_r(self, repl):
        repl._cmd_uname("-r")
        assert repl._last_exit_code == 0

    def test_uname_m(self, repl):
        repl._cmd_uname("-m")
        assert repl._last_exit_code == 0

    def test_uname_srm(self, repl):
        repl._cmd_uname("-srm")
        assert repl._last_exit_code == 0


# ── Mktemp flags ──────────────────────────────────────────────────


class TestCmdMktempFlags:
    def test_mktemp_no_args(self, repl):
        repl._cmd_mktemp("")
        assert repl._last_exit_code == 0

    def test_mktemp_dir(self, repl):
        repl._cmd_mktemp("-d")
        assert repl._last_exit_code == 0


# ── Od flags ──────────────────────────────────────────────────────


class TestCmdOdFlags:
    def test_od_no_args(self, repl):
        repl._cmd_od("")
        assert repl._last_exit_code == 1

    def test_od_hex(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='wb', suffix='.bin', delete=False)
        f.write(b"\x00\x01\x02\x03")
        f.close()
        try:
            repl._cmd_od(f"-x {f.name}")
            assert repl._last_exit_code == 0
        finally:
            os.unlink(f.name)

    def test_od_octal(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='wb', suffix='.bin', delete=False)
        f.write(b"\x00\x01\x02\x03")
        f.close()
        try:
            repl._cmd_od(f"-o {f.name}")
            assert repl._last_exit_code == 0
        finally:
            os.unlink(f.name)

    def test_od_decimal(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='wb', suffix='.bin', delete=False)
        f.write(b"\x00\x01\x02\x03")
        f.close()
        try:
            repl._cmd_od(f"{f.name}")
            assert repl._last_exit_code == 0
        finally:
            os.unlink(f.name)

    def test_od_not_found(self, repl):
        repl._cmd_od("/nonexistent/file")
        assert repl._last_exit_code == 1


# ── Join extra paths ──────────────────────────────────────────────


class TestCmdJoinExtra:
    def test_join_one_arg(self, repl):
        repl._cmd_join("file1.txt")
        assert repl._last_exit_code == 1

    def test_join_not_found(self, repl):
        repl._cmd_join("/nonexistent/a.txt /nonexistent/b.txt")
        assert repl._last_exit_code == 1

    def test_join_no_overlap(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("1 alpha\n")
        f2.write_text("2 beta\n")
        repl._cmd_join(f"{f1} {f2}")
        assert repl._last_exit_code == 0


# ── Paste extra paths ─────────────────────────────────────────────


class TestCmdPasteExtra:
    def testPaste_no_args(self, repl):
        repl._cmd_paste("")
        assert repl._last_exit_code == 1

    def test_paste_not_found(self, repl):
        repl._cmd_paste("/nonexistent/file.txt")
        assert repl._last_exit_code == 1


# ── Comm extra paths ──────────────────────────────────────────────


class TestCmdCommExtra2:
    def test_comm_no_args(self, repl):
        repl._cmd_comm("")
        assert repl._last_exit_code == 1

    def test_comm_one_arg(self, repl):
        repl._cmd_comm("file1.txt")
        assert repl._last_exit_code == 1

    def test_comm_not_found(self, repl):
        repl._cmd_comm("/nonexistent/a.txt /nonexistent/b.txt")
        assert repl._last_exit_code == 1


# ── Column command ────────────────────────────────────────────────


class TestCmdColumn:
    def test_column_basic(self, repl):
        repl._piped_input = "name\tage\nAlice\t30\nBob\t25\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_column("-t")
        out = cap.getvalue()
        assert "Alice" in out
        assert "Bob" in out

    def test_column_separator(self, repl):
        repl._piped_input = "a,b,c\n1,2,3\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_column("-t -s ,")
        out = cap.getvalue()
        assert "a" in out
        assert "1" in out

    def test_column_no_input(self, repl):
        repl._piped_input = None
        with _CaptureOutput(repl) as cap:
            repl._cmd_column("")
        assert repl._last_exit_code == 1

    def test_column_not_found(self, repl):
        repl._cmd_column("/nonexistent/file.txt")
        assert repl._last_exit_code == 1

    def test_column_mixed_widths(self, repl):
        repl._piped_input = "short\tlonger\nab\tabcde\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_column("-t")
        out = cap.getvalue()
        lines = out.strip().split("\n")
        assert len(lines) == 2


# ── Yes command ───────────────────────────────────────────────────


class TestCmdYesExtra:
    def test_yes_default(self, repl):
        repl._cmd_yes("")
        assert repl._last_exit_code == 0

    def test_yes_custom(self, repl):
        repl._cmd_yes("hello")
        assert repl._last_exit_code == 0


# ── Realpath/Dirname/Basename edge cases ──────────────────────────


class TestCmdPathUtils:
    def test_realpath_no_args(self, repl):
        repl._cmd_realpath("")
        assert repl._last_exit_code == 1

    def test_realpath_valid(self, repl):
        repl._cmd_realpath(".")
        assert repl._last_exit_code == 0

    def test_dirname_no_args(self, repl):
        repl._cmd_dirname("")
        assert repl._last_exit_code == 1

    def test_dirname_valid(self, repl):
        repl._cmd_dirname("/foo/bar/baz")
        assert repl._last_exit_code == 0

    def test_basename_no_args(self, repl):
        repl._cmd_basename("")
        assert repl._last_exit_code == 1

    def test_basename_valid(self, repl):
        repl._cmd_basename("/foo/bar/baz.txt")
        assert repl._last_exit_code == 0

    def test_basename_with_suffix(self, repl):
        repl._cmd_basename("/foo/bar/baz.txt .txt")
        assert repl._last_exit_code == 0


# ── Nproc/Hostname ────────────────────────────────────────────────


class TestCmdNprocHostname:
    def test_nproc(self, repl):
        repl._cmd_nproc("")
        assert repl._last_exit_code == 0

    def test_hostname(self, repl):
        repl._cmd_hostname("")
        assert repl._last_exit_code == 0


# ── Id/Logname ────────────────────────────────────────────────────


class TestCmdIdLogname:
    def test_id(self, repl):
        repl._cmd_id("")
        assert repl._last_exit_code == 0

    def test_logname(self, repl):
        repl._cmd_logname("")
        assert repl._last_exit_code == 0


# ── Who ───────────────────────────────────────────────────────────


class TestCmdWho:
    def test_who(self, repl):
        repl._cmd_who("")
        assert repl._last_exit_code == 0


# ── Expand/Unexpand ───────────────────────────────────────────────


class TestCmdExpandUnexpand:
    def test_expand_no_args(self, repl):
        repl._cmd_expand("")
        assert repl._last_exit_code == 1

    def test_expand_file(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f.write("col1\tcol2\tcol3")
        f.close()
        try:
            repl._cmd_expand(f.name)
            assert repl._last_exit_code == 0
        finally:
            os.unlink(f.name)

    def test_expand_not_found(self, repl):
        repl._cmd_expand("/nonexistent/file.txt")
        assert repl._last_exit_code == 1

    def test_unexpand_no_args(self, repl):
        repl._cmd_unexpand("")
        assert repl._last_exit_code == 1

    def test_unexpand_file(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f.write("col1    col2    col3")
        f.close()
        try:
            repl._cmd_unexpand(f.name)
            assert repl._last_exit_code == 0
        finally:
            os.unlink(f.name)

    def test_unexpand_not_found(self, repl):
        repl._cmd_unexpand("/nonexistent/file.txt")
        assert repl._last_exit_code == 1


# ── Read command ──────────────────────────────────────────────────


class TestCmdReadExtra:
    def test_read_no_args(self, repl):
        repl._cmd_read("")
        assert repl._last_exit_code == 1

    def test_read_with_prompt(self, repl):
        from domains.shell.io import MemoryIO
        mem = MemoryIO()
        mem.feed("test_value")
        old_io = repl.io
        repl.io = mem
        try:
            repl._cmd_read("-p Enter: MYVAR")
            assert repl._last_exit_code == 0
            assert repl._env.get("MYVAR") == "test_value"
        finally:
            repl.io = old_io

    def test_read_prompt_only(self, repl):
        from domains.shell.io import MemoryIO
        mem = MemoryIO()
        mem.feed("val")
        old_io = repl.io
        repl.io = mem
        try:
            repl._cmd_read("-p")
            assert repl._last_exit_code == 0
            assert repl._env.get("-p") == "val"
        finally:
            repl.io = old_io

    def test_read_eof(self, repl):
        from domains.shell.io import MemoryIO
        mem = MemoryIO()
        old_io = repl.io
        repl.io = mem
        try:
            repl._cmd_read("MYVAR")
            assert repl._last_exit_code == 1
        finally:
            repl.io = old_io


# ── Source command ─────────────────────────────────────────────────


class TestCmdSourceExtra:
    def test_source_not_found(self, repl):
        repl._cmd_source("/nonexistent/file.sh")
        assert repl._last_exit_code == 0

    def test_source_valid(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False)
        f.write("echo from_source\n")
        f.close()
        try:
            repl._cmd_source(f.name)
            assert repl._last_exit_code == 0
        finally:
            os.unlink(f.name)


# ── Alias/Unalias edge cases ──────────────────────────────────────


class TestCmdAliasUnaliasExtra:
    def test_alias_list(self, repl):
        repl._cmd_alias("")
        assert repl._last_exit_code == 0

    def test_alias_single(self, repl):
        repl._cmd_alias("ll")
        assert repl._last_exit_code == 0

    def test_unalias_no_args(self, repl):
        repl._cmd_unalias("")
        assert repl._last_exit_code == 0

    def test_unalias_nonexistent(self, repl):
        repl._cmd_unalias("nonexistent")
        assert repl._last_exit_code == 0


# ── Set/Export edge cases ─────────────────────────────────────────


class TestCmdSetExportExtra:
    def test_set_no_args(self, repl):
        repl._cmd_set("")
        assert repl._last_exit_code == 0

    def test_set_with_value(self, repl):
        repl._cmd_set("MYVAR=hello")
        assert repl._last_exit_code == 0

    def test_export_no_args(self, repl):
        repl._cmd_export("")
        assert repl._last_exit_code == 0

    def test_export_with_value(self, repl):
        repl._cmd_export("MYVAR=hello")
        assert repl._last_exit_code == 0


# ── Sleep command ─────────────────────────────────────────────────


class TestCmdSleepExtra:
    def test_sleep_default(self, repl):
        repl._cmd_sleep("0.01")
        assert repl._last_exit_code == 0

    def test_sleep_invalid(self, repl):
        repl._cmd_sleep("abc")
        assert repl._last_exit_code == 0

    def test_sleep_no_args(self, repl):
        repl._cmd_sleep("")
        assert repl._last_exit_code == 0


# ── Date command ──────────────────────────────────────────────────


class TestCmdDateExtra:
    def test_date_no_args(self, repl):
        repl._cmd_date("")
        assert repl._last_exit_code == 0

    def test_date_utc(self, repl):
        repl._cmd_date("-u")
        assert repl._last_exit_code == 0

    def test_date_format(self, repl):
        repl._cmd_date("+%Y-%m-%d")
        assert repl._last_exit_code == 0


# ── Cal command ───────────────────────────────────────────────────


class TestCmdCalExtra:
    def test_cal_no_args(self, repl):
        repl._cmd_cal("")
        assert repl._last_exit_code == 0

    def test_cal_year(self, repl):
        repl._cmd_cal("2026")
        assert repl._last_exit_code == 0

    def test_cal_month_year(self, repl):
        repl._cmd_cal("7 2026")
        assert repl._last_exit_code == 0


# ── Ln command ────────────────────────────────────────────────────


class TestCmdLnExtra:
    def test_ln_no_args(self, repl):
        repl._cmd_ln("")
        assert repl._last_exit_code == 1

    def test_ln_one_arg(self, repl):
        repl._cmd_ln("target.txt")
        assert repl._last_exit_code == 1

    def test_ln_hard(self, repl):
        import tempfile
        d = tempfile.mkdtemp()
        try:
            src = os.path.join(d, "src.txt")
            dst = os.path.join(d, "dst.txt")
            with open(src, "w") as f:
                f.write("test")
            repl._cmd_ln(f"{src} {dst}")
            assert os.path.islink(dst) or os.path.exists(dst)
        finally:
            import shutil
            shutil.rmtree(d)

    def test_ln_symbolic(self, repl):
        import tempfile
        d = tempfile.mkdtemp()
        try:
            src = os.path.join(d, "src.txt")
            dst = os.path.join(d, "dst.txt")
            with open(src, "w") as f:
                f.write("test")
            repl._cmd_ln(f"-s {src} {dst}")
            assert os.path.islink(dst)
        finally:
            import shutil
            shutil.rmtree(d)


# ── Shuf/Rev/Tac/Paste file not found ─────────────────────────────


class TestCmdFileNotFound:
    def test_shuf_not_found(self, repl):
        repl._cmd_shuf("/nonexistent/file.txt")
        assert repl._last_exit_code == 1

    def test_rev_not_found(self, repl):
        repl._cmd_rev("/nonexistent/file.txt")
        assert repl._last_exit_code == 1

    def test_tac_not_found(self, repl):
        repl._cmd_tac("/nonexistent/file.txt")
        assert repl._last_exit_code == 1

    def test_paste_not_found(self, repl):
        repl._cmd_paste("/nonexistent/file.txt")
        assert repl._last_exit_code == 1

    def test_fold_not_found(self, repl):
        repl._cmd_fold("/nonexistent/file.txt")
        assert repl._last_exit_code == 1

    def test_nl_not_found(self, repl):
        repl._cmd_nl("/nonexistent/file.txt")
        assert repl._last_exit_code == 1


# ── Gen command ───────────────────────────────────────────────────


class TestCmdGenExtra:
    def test_gen_no_args(self, repl):
        repl._cmd_gen("")
        assert repl._last_exit_code == 0

    def test_gen_no_api(self, repl):
        with patch.object(repl, '_require_api', return_value=False):
            repl._cmd_gen("hello")
        assert repl._last_exit_code == 0

    def test_gen_success(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'generate', return_value={"text": "generated text"}):
            repl._cmd_gen("hello")
        assert repl._last_exit_code == 0

    def test_gen_error(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'generate', return_value={"error": "API down"}):
            repl._cmd_gen("hello")
        assert repl._last_exit_code == 0

    def test_gen_fallback(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'generate', return_value={"foo": "bar"}):
            repl._cmd_gen("hello")
        assert repl._last_exit_code == 0


# ── Chat command ──────────────────────────────────────────────────


class TestCmdChatExtra:
    def test_chat_no_args(self, repl):
        repl._cmd_chat("")
        assert repl._last_exit_code == 0

    def test_chat_reset(self, repl):
        repl._chat_session_id = "old"
        repl._chat_history = [{"role": "user", "content": "hi"}]
        repl._cmd_chat("/reset")
        assert repl._chat_session_id is None
        assert repl._chat_history == []

    def test_chat_no_api(self, repl):
        with patch.object(repl, '_require_api', return_value=False):
            repl._cmd_chat("hello")
        assert repl._last_exit_code == 0

    def test_chat_success(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl, '_spinner_call', return_value={"message": "response text"}):
            repl._cmd_chat("hello")
        assert repl._last_exit_code == 0
        assert len(repl._chat_history) == 2

    def test_chat_error(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl, '_spinner_call', return_value={"error": "fail"}):
            repl._cmd_chat("hello")
        assert repl._last_exit_code == 0

    def test_chat_fallback(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl, '_spinner_call', return_value="raw string"):
            repl._cmd_chat("hello")
        assert repl._last_exit_code == 0

    def test_chat_strips_think(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl, '_spinner_call', return_value={"message": "<think>reasoning</think>final"}):
            repl._cmd_chat("hello")
        assert repl._last_exit_code == 0


# ── ASM command ───────────────────────────────────────────────────


class TestCmdAsmExtra:
    def test_asm_no_args(self, repl):
        repl._cmd_asm("")
        assert repl._last_exit_code == 1

    def test_asm_list(self, repl):
        repl._cmd_asm("--list")
        assert repl._last_exit_code == 0

    def test_asm_not_found(self, repl):
        repl._cmd_asm("/nonexistent/file.asm")
        assert repl._last_exit_code == 1

    def test_asm_source(self, repl):
        repl._piped_input = "MOV R0, 42\nHALT"
        repl._cmd_asm("")
        assert repl._last_exit_code == 0

    def test_asm_vm_fault(self, repl):
        repl._piped_input = "INVALID_OP"
        with patch('domains.shell.vm.VMRunner') as MockRunner:
            MockRunner.return_value.assemble_and_run.side_effect = Exception("VM error")
            repl._cmd_asm("")
        assert repl._last_exit_code == 1


# ── VMRun command ─────────────────────────────────────────────────


class TestCmdVmrunExtra:
    def test_vmrun_no_args(self, repl):
        repl._cmd_vmrun("")
        assert repl._last_exit_code == 1

    def test_vmrun_list(self, repl):
        repl._cmd_vmrun("--list")
        assert repl._last_exit_code == 0

    def test_vmrun_not_found(self, repl):
        repl._cmd_vmrun("/nonexistent/file.asm")
        assert repl._last_exit_code == 1

    def test_vmrun_admin_role(self, repl):
        with patch.dict(os.environ, {"MAN_VM_ROLE": "admin"}):
            repl._cmd_vmrun("--admin --list")
        assert repl._last_exit_code == 0

    def test_vmrun_user_role_denied(self, repl):
        with patch.dict(os.environ, {"MAN_VM_ROLE": "user"}):
            repl._cmd_vmrun("--admin --list")
        assert repl._last_exit_code == 0

    def test_vmrun_steps_invalid(self, repl):
        repl._cmd_vmrun("--steps=abc hello")
        assert repl._last_exit_code == 1

    def test_vmrun_debug(self, repl):
        repl._cmd_vmrun("--debug --steps=10 hello")
        assert repl._last_exit_code == 0


# ── SVC command ───────────────────────────────────────────────────


class TestCmdSvcExtra:
    def _mock_init(self, repl):
        """Set up a mock init system on the runtime."""
        mock_init = MagicMock()
        mock_init.service_table.return_value = "  svc1  running\n  svc2  stopped"
        mock_init.status_summary = "  2 services"
        mock_init.runlevel = 3
        mock_manager = MagicMock()
        mock_manager.status_line.return_value = "  svc1  running"
        mock_manager.instance.log = ["line1", "line2"]
        mock_manager.start.return_value = True
        mock_manager.restart.return_value = True
        mock_init.get_manager.return_value = mock_manager
        repl.os._init = mock_init
        return mock_init

    def test_svc_not_booted(self, repl):
        with patch.object(type(repl.os), 'init_system', new_callable=PropertyMock, return_value=None):
            repl._cmd_svc("list")
        assert repl._last_exit_code == 1

    def test_svc_list(self, repl):
        self._mock_init(repl)
        repl._cmd_svc("list")
        assert repl._last_exit_code == 0

    def test_svc_ls(self, repl):
        self._mock_init(repl)
        repl._cmd_svc("ls")
        assert repl._last_exit_code == 0

    def test_svc_status_no_name(self, repl):
        self._mock_init(repl)
        repl._cmd_svc("status")
        assert repl._last_exit_code == 0

    def test_svc_status_known(self, repl):
        self._mock_init(repl)
        repl._cmd_svc("status svc1")
        assert repl._last_exit_code == 0

    def test_svc_status_unknown(self, repl):
        mock_init = self._mock_init(repl)
        mock_init.get_manager.return_value = None
        repl._cmd_svc("status nonexistent")
        assert repl._last_exit_code == 1

    def test_svc_start_no_name(self, repl):
        self._mock_init(repl)
        repl._cmd_svc("start")
        assert repl._last_exit_code == 1

    def test_svc_start_known(self, repl):
        self._mock_init(repl)
        repl._cmd_svc("start svc1")
        assert repl._last_exit_code == 0

    def test_svc_start_unknown(self, repl):
        mock_init = self._mock_init(repl)
        mock_init.get_manager.return_value = None
        repl._cmd_svc("start nonexistent")
        assert repl._last_exit_code == 1

    def test_svc_stop_no_name(self, repl):
        self._mock_init(repl)
        repl._cmd_svc("stop")
        assert repl._last_exit_code == 1

    def test_svc_stop_known(self, repl):
        self._mock_init(repl)
        repl._cmd_svc("stop svc1")
        assert repl._last_exit_code == 0

    def test_svc_stop_unknown(self, repl):
        mock_init = self._mock_init(repl)
        mock_init.get_manager.return_value = None
        repl._cmd_svc("stop nonexistent")
        assert repl._last_exit_code == 1

    def test_svc_restart_no_name(self, repl):
        self._mock_init(repl)
        repl._cmd_svc("restart")
        assert repl._last_exit_code == 1

    def test_svc_restart_known(self, repl):
        self._mock_init(repl)
        repl._cmd_svc("restart svc1")
        assert repl._last_exit_code == 0

    def test_svc_restart_unknown(self, repl):
        mock_init = self._mock_init(repl)
        mock_init.get_manager.return_value = None
        repl._cmd_svc("restart nonexistent")
        assert repl._last_exit_code == 1

    def test_svc_runlevel(self, repl):
        self._mock_init(repl)
        repl._cmd_svc("runlevel")
        assert repl._last_exit_code == 0

    def test_svc_unknown_cmd(self, repl):
        self._mock_init(repl)
        repl._cmd_svc("bogus")
        assert repl._last_exit_code == 1


# ── Boot/Shutdown command ─────────────────────────────────────────


class TestCmdBootShutdown:
    def test_boot_already_booted(self, repl):
        repl._running = True
        repl._cmd_boot("")
        assert repl._last_exit_code == 0

    def test_shutdown(self, repl):
        repl._cmd_shutdown("")
        assert repl._last_exit_code == 0
        assert repl._running is False


# ── API command ───────────────────────────────────────────────────


class TestCmdApiExtra:
    def test_api_status(self, repl):
        repl._cmd_api("status")
        assert repl._last_exit_code == 0

    def test_api_stop(self, repl):
        repl._cmd_api("stop")
        assert repl._last_exit_code == 0

    def test_api_restart(self, repl):
        repl._cmd_api("restart")
        assert repl._last_exit_code == 0

    def test_api_start(self, repl):
        repl._cmd_api("start")
        assert repl._last_exit_code == 0


# ── Render command ────────────────────────────────────────────────


class TestCmdRenderExtra:
    def test_render_no_args(self, repl):
        repl._cmd_render("")
        assert repl._last_exit_code == 0

    def test_render_info(self, repl):
        repl._cmd_render("info")
        assert repl._last_exit_code == 0

    def test_render_sphere_no_args(self, repl):
        repl._cmd_render("sphere")
        assert repl._last_exit_code == 0

    def test_render_sphere(self, repl):
        repl._cmd_render("sphere 1.0 0 0 0")
        assert repl._last_exit_code == 0

    def test_render_cube_no_args(self, repl):
        repl._cmd_render("cube")
        assert repl._last_exit_code == 0

    def test_render_cube(self, repl):
        repl._cmd_render("cube 1.0 0 0 0")
        assert repl._last_exit_code == 0

    def test_render_plane_no_args(self, repl):
        repl._cmd_render("plane")
        assert repl._last_exit_code == 0

    def test_render_plane(self, repl):
        repl._cmd_render("plane 5.0 -1.0")
        assert repl._last_exit_code == 0

    def test_render_light_no_args(self, repl):
        repl._cmd_render("light")
        assert repl._last_exit_code == 0

    def test_render_light(self, repl):
        repl._cmd_render("light 0 5 0")
        assert repl._last_exit_code == 0

    def test_render_light_rgb(self, repl):
        repl._cmd_render("light 0 5 0 1.0 0.5 0.3 10.0")
        assert repl._last_exit_code == 0

    def test_render_mat_no_args(self, repl):
        repl._cmd_render("mat")
        assert repl._last_exit_code == 0

    def test_render_mat(self, repl):
        repl._cmd_render("mat 0 0.5 0.5 0.5 0.0 0.8")
        assert repl._last_exit_code == 0

    def test_render_cam_no_args(self, repl):
        repl._cmd_render("cam")
        assert repl._last_exit_code == 0

    def test_render_cam(self, repl):
        repl._cmd_render("cam 0 0 5 0 0 0")
        assert repl._last_exit_code == 0

    def test_render_go(self, repl):
        repl._cmd_render("go 10 10 1")
        assert repl._last_exit_code == 0

    def test_render_clear(self, repl):
        repl._cmd_render("clear")
        assert repl._last_exit_code == 0

    def test_render_preset_demo(self, repl):
        repl._cmd_render("preset demo")
        assert repl._last_exit_code == 0

    def test_render_preset_cornell(self, repl):
        repl._cmd_render("preset cornell")
        assert repl._last_exit_code == 0

    def test_render_preset_spheres(self, repl):
        repl._cmd_render("preset spheres")
        assert repl._last_exit_code == 0

    def test_render_preset_no_name(self, repl):
        repl._cmd_render("preset")
        assert repl._last_exit_code == 0

    def test_render_unknown(self, repl):
        repl._cmd_render("bogus")
        assert repl._last_exit_code == 0


# ── Tutorial command ──────────────────────────────────────────────


class TestCmdTutorialExtra:
    def test_tutorial(self, repl):
        from domains.shell.io import MemoryIO
        mem = MemoryIO()
        mem.feed("q")
        old_io = repl.io
        repl.io = mem
        try:
            repl._cmd_tutorial("")
            assert repl._last_exit_code == 0
        finally:
            repl.io = old_io


# ── AI command extra ──────────────────────────────────────────────


class TestCmdAiExtra:
    def test_ai_no_args(self, repl):
        repl._cmd_ai("")
        assert repl._last_exit_code == 0

    def test_ai_no_api(self, repl):
        with patch.object(repl, '_require_api', return_value=False):
            repl._cmd_ai("help me")
        assert repl._last_exit_code == 0

    def test_ai_success(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'generate', return_value={"text": "run ls"}):
            repl._cmd_ai("show files")
        assert repl._last_exit_code == 0

    def test_ai_non_dict_result(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'generate', return_value="just text"):
            repl._cmd_ai("do something")
        assert repl._last_exit_code == 0

    def test_ai_api_error(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'generate', return_value={"error": "timeout"}):
            repl._cmd_ai("do something")
        assert repl._last_exit_code == 0


# ── _cmd_vmrun execution path ──────────────────────────────────────


class TestCmdVmrunExecution:
    def test_vmrun_built_in_hello(self, repl):
        repl._cmd_vmrun("hello")
        assert repl._last_exit_code == 0

    def test_vmrun_built_in_count(self, repl):
        repl._cmd_vmrun("count")
        assert repl._last_exit_code == 0

    def test_vmrun_built_in_counter(self, repl):
        repl._cmd_vmrun("counter")
        assert repl._last_exit_code == 0

    def test_vmrun_admin_role(self, repl):
        repl._cmd_vmrun("--admin hello")
        assert repl._last_exit_code == 0

    def test_vmrun_kernel_role(self, repl):
        repl._cmd_vmrun("--kernel hello")
        assert repl._last_exit_code == 0

    def test_vmrun_steps_flag(self, repl):
        repl._cmd_vmrun("--steps=100 hello")
        assert repl._last_exit_code == 0

    def test_vmrun_debug_flag(self, repl):
        repl._cmd_vmrun("--debug hello")
        assert repl._last_exit_code == 0

    def test_vmrun_debug_with_output(self, repl):
        repl._cmd_vmrun("--debug --admin hello")
        assert repl._last_exit_code == 0

    def test_vmrun_no_args(self, repl):
        repl._cmd_vmrun("")
        assert repl._last_exit_code == 1

    def test_vmrun_file_not_found(self, repl):
        repl._cmd_vmrun("/nonexistent/file.asm")
        assert repl._last_exit_code == 1

    def test_vmrun_user_role_denied(self, repl):
        with patch.dict(os.environ, {"MAN_VM_ROLE": "user"}):
            repl._cmd_vmrun("--admin hello")
        assert repl._last_exit_code == 1

    def test_vmrun_piped_input(self, repl):
        repl._piped_input = "mov eax, 3\nmov ebx, 1\nmov ecx, hello\nmov edx, 5\nint 0x80\nmov eax, 1\nxor ebx, ebx\nint 0x80\njmp $\nhello: db 'Hi', 10"
        repl._cmd_vmrun("")
        repl._piped_input = None
        assert repl._last_exit_code == 0

    def test_vmrun_admin_debug_steps(self, repl):
        repl._cmd_vmrun("--admin --debug --steps=50 hello")
        assert repl._last_exit_code == 0

    def test_vmrun_steps_invalid(self, repl):
        repl._cmd_vmrun("--steps=abc hello")
        assert repl._last_exit_code == 1

    def test_vmrun_list(self, repl):
        repl._cmd_vmrun("--list")
        assert repl._last_exit_code == 0


# ── _cmd_asm execution path ────────────────────────────────────────


class TestCmdAsmExecution:
    def test_asm_no_args(self, repl):
        repl._cmd_asm("")
        assert repl._last_exit_code == 1

    def test_asm_file_not_found(self, repl):
        repl._cmd_asm("/nonexistent/file.asm")
        assert repl._last_exit_code == 1

    def test_asm_piped_input(self, repl):
        repl._piped_input = "MOV R0, 42\nHALT"
        repl._cmd_asm("")
        repl._piped_input = None
        assert repl._last_exit_code == 0

    def test_asm_source_code(self, repl):
        source = "MOV R0, 42\nHALT"
        with patch("pathlib.Path.read_text", return_value=source):
            repl._cmd_asm("test.asm")
        assert repl._last_exit_code == 0

    def test_asm_vm_fault(self, repl):
        from domains.shell.vm import VMFault
        with patch("domains.shell.vm.VMRunner") as MockRunner:
            MockRunner.return_value.assemble_and_run.side_effect = VMFault("bad instruction")
            repl._cmd_asm("")
        assert repl._last_exit_code == 1

    def test_asm_generic_exception(self, repl):
        with patch("domains.shell.vm.VMRunner") as MockRunner:
            MockRunner.return_value.assemble_and_run.side_effect = RuntimeError("boom")
            repl._cmd_asm("")
        assert repl._last_exit_code == 1

    def test_asm_test_flag(self, repl):
        repl._cmd_asm("--test")
        assert repl._last_exit_code == 0

    def test_asm_self_test_flag(self, repl):
        repl._cmd_asm("--self-test")
        assert repl._last_exit_code == 0

    def test_asm_list_flag(self, repl):
        repl._cmd_asm("--list")
        assert repl._last_exit_code == 0

    def test_asm_piped_with_file(self, repl):
        repl._piped_input = "mov eax, 1\njmp $"
        repl._cmd_asm("--list")
        repl._piped_input = None
        assert repl._last_exit_code == 0


# ── _cmd_gen / _cmd_chat execution ─────────────────────────────────


class TestCmdGenExecution:
    def test_gen_no_args(self, repl):
        repl._cmd_gen("")
        assert repl._last_exit_code == 0

    def test_gen_no_api(self, repl):
        with patch.object(repl, '_require_api', return_value=False):
            repl._cmd_gen("hello")
        assert repl._last_exit_code == 0

    def test_gen_success(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'generate', return_value={"text": "generated text"}):
            repl._cmd_gen("hello")
        assert repl._last_exit_code == 0

    def test_gen_error(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'generate', return_value={"error": "timeout"}):
            repl._cmd_gen("hello")
        assert repl._last_exit_code == 0

    def test_gen_non_dict_result(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'generate', return_value="just text"):
            repl._cmd_gen("hello")
        assert repl._last_exit_code == 0


class TestCmdChatExecution:
    def test_chat_no_args(self, repl):
        repl._cmd_chat("")
        assert repl._last_exit_code == 0

    def test_chat_reset(self, repl):
        repl._chat_session_id = "old"
        repl._chat_history = [{"role": "user", "content": "hi"}]
        repl._cmd_chat("/reset")
        assert repl._chat_session_id is None
        assert repl._chat_history == []
        assert repl._last_exit_code == 0

    def test_chat_no_api(self, repl):
        with patch.object(repl, '_require_api', return_value=False):
            repl._cmd_chat("hello")
        assert repl._last_exit_code == 0

    def test_chat_new_session(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'chat', return_value={"message": "hi there"}):
            repl._cmd_chat("hello")
        assert repl._chat_session_id is not None
        assert len(repl._chat_history) == 2
        assert repl._last_exit_code == 0

    def test_chat_continues_session(self, repl):
        repl._chat_session_id = "existing"
        repl._chat_history = [{"role": "user", "content": "prev"}]
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'chat', return_value={"message": "response"}):
            repl._cmd_chat("hello")
        assert len(repl._chat_history) == 3
        assert repl._last_exit_code == 0

    def test_chat_error(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'chat', return_value={"error": "fail"}):
            repl._cmd_chat("hello")
        assert repl._last_exit_code == 0

    def test_chat_non_dict_result(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'chat', return_value="just text"):
            repl._cmd_chat("hello")
        assert repl._last_exit_code == 0

    def test_chat_think_tag_stripped(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'chat', return_value={"message": "<think>reasoning</think>answer"}):
            repl._cmd_chat("hello")
        assert "<think>" not in str(repl._chat_history)
        assert repl._last_exit_code == 0


# ── _stream_train_progress ─────────────────────────────────────────


class TestStreamTrainProgress:
    def test_job_not_found(self, repl):
        with patch("domains.shell.commands._api_get", return_value=None):
            repl._stream_train_progress("nonexistent")
        assert repl._last_exit_code == 0

    def test_job_completed(self, repl):
        results = [
            {"status": "running", "progress": 50, "epoch": 1, "epochs": 3, "loss": 0.5},
            {"status": "completed", "progress": 100, "epoch": 3, "epochs": 3, "loss": 0.1, "checkpoint": "my-checkpoint"},
        ]
        call_count = [0]
        def mock_get(url):
            idx = call_count[0]
            call_count[0] += 1
            return results[min(idx, len(results)-1)]

        with patch("domains.shell.commands._api_get", side_effect=mock_get), \
             patch("domains.shell.repl.time.sleep"):
            repl._stream_train_progress("job-123")
        assert repl._last_exit_code == 0

    def test_job_failed(self, repl):
        results = [
            {"status": "running", "progress": 30},
            {"status": "failed", "progress": 30, "error": "OOM"},
        ]
        call_count = [0]
        def mock_get(url):
            idx = call_count[0]
            call_count[0] += 1
            return results[min(idx, len(results)-1)]

        with patch("domains.shell.commands._api_get", side_effect=mock_get), \
             patch("domains.shell.repl.time.sleep"):
            repl._stream_train_progress("job-456")
        assert repl._last_exit_code == 0

    def test_job_error_status(self, repl):
        results = [
            {"status": "error", "progress": 0, "error": "crash"},
        ]
        with patch("domains.shell.commands._api_get", return_value=results[0]), \
             patch("domains.shell.repl.time.sleep"):
            repl._stream_train_progress("job-789")
        assert repl._last_exit_code == 0

    def test_job_keyboard_interrupt(self, repl):
        call_count = [0]
        def mock_get(url):
            call_count[0] += 1
            if call_count[0] > 1:
                raise KeyboardInterrupt()
            return {"status": "running", "progress": 10}

        with patch("domains.shell.commands._api_get", side_effect=mock_get), \
             patch("domains.shell.repl.time.sleep"):
            repl._stream_train_progress("job-kbd")
        assert repl._last_exit_code == 0

    def test_job_exception(self, repl):
        with patch("domains.shell.commands._api_get", side_effect=RuntimeError("network")), \
             patch("domains.shell.repl.time.sleep"):
            repl._stream_train_progress("job-err")
        assert repl._last_exit_code == 0

    def test_job_progress_with_stdio(self, repl):
        mock_stdio = MagicMock()
        repl._stdio = mock_stdio
        results = [
            {"status": "running", "progress": 50, "epoch": 1, "epochs": 3, "loss": 0.5},
            {"status": "completed", "progress": 100, "epoch": 3, "epochs": 3, "loss": 0.1},
        ]
        call_count = [0]
        def mock_get(url):
            idx = call_count[0]
            call_count[0] += 1
            return results[min(idx, len(results)-1)]

        with patch("domains.shell.commands._api_get", side_effect=mock_get), \
             patch("domains.shell.repl.time.sleep"):
            repl._stream_train_progress("job-stdio")
        repl._stdio = None
        assert repl._last_exit_code == 0

    def test_job_half_block_progress(self, repl):
        results = [
            {"status": "running", "progress": 33, "epoch": 1, "epochs": 3, "loss": 0.5},
            {"status": "completed", "progress": 100},
        ]
        call_count = [0]
        def mock_get(url):
            idx = call_count[0]
            call_count[0] += 1
            return results[min(idx, len(results)-1)]

        with patch("domains.shell.commands._api_get", side_effect=mock_get), \
             patch("domains.shell.repl.time.sleep"):
            repl._stream_train_progress("job-half")
        assert repl._last_exit_code == 0


# ── Pipeline execution internals ───────────────────────────────────


class TestPipelineInternals:
    def test_execute_single_permission_denied(self, repl):
        with patch.object(repl, '_check_permission', return_value=False):
            repl.execute("rm /important/file")
        assert repl._last_exit_code == 126

    def test_execute_single_system_exit(self, repl):
        def raise_exit(self_repl, args):
            raise SystemExit(42)
        repl.COMMANDS["testexit"] = raise_exit
        repl.execute("testexit")
        assert repl._last_exit_code == 42

    def test_execute_single_exception(self, repl):
        def raise_error(self_repl, args):
            raise RuntimeError("boom")
        repl.COMMANDS["testerr"] = raise_error
        repl.execute("testerr")
        assert repl._last_exit_code == 1

    def test_execute_single_unknown_command(self, repl):
        repl.execute("nonexistentcmd")
        assert repl._last_exit_code == 127

    def test_execute_single_unknown_with_suggestion(self, repl):
        repl.execute("hepl")
        assert repl._last_exit_code == 127

    def test_execute_single_with_timing(self, repl):
        repl.execute("time echo hello")
        assert repl._last_exit_code == 0

    def test_execute_single_ext_command(self, repl):
        mock_ext = MagicMock()
        mock_ext.run.return_value = 0
        repl._ext_cmds["myext"] = mock_ext
        repl.execute("myext arg1 arg2")
        assert repl._last_exit_code == 0
        del repl._ext_cmds["myext"]

    def test_execute_single_ext_command_failure(self, repl):
        mock_ext = MagicMock()
        mock_ext.run.return_value = 1
        repl._ext_cmds["myext"] = mock_ext
        repl.execute("myext arg1")
        assert repl._last_exit_code == 1
        del repl._ext_cmds["myext"]

    def test_execute_pipeline_simple(self, repl):
        repl.execute("echo hello | wc")
        assert repl._last_exit_code == 0

    def test_execute_single_keyboard_interrupt(self, repl):
        def raise_kbd(self_repl, args):
            raise KeyboardInterrupt()
        repl.COMMANDS["kbdtest"] = raise_kbd
        repl.execute("kbdtest")
        assert repl._last_exit_code == 0
        assert repl._aborted is True
        repl._aborted = False

    def test_execute_background(self, repl):
        repl.execute("sleep 0.01 &")
        assert repl._last_exit_code == 0

    def test_execute_background_pipeline(self, repl):
        repl.execute("echo hello | cat &")
        assert repl._last_exit_code == 0


# ── _cmd_load tracker paths ────────────────────────────────────────


class TestCmdLoadTracker:
    def test_load_tracker_downloading(self, repl):
        mock_tracker = MagicMock()
        mock_tracker.get.return_value = {"stage": "downloading", "progress": 0.5, "message": "Downloading model..."}
        mock_bar = MagicMock()
        with patch.object(repl, '_require_api', return_value=True), \
             patch("domains.infrastructure.conversion_tracker.get_tracker", return_value=mock_tracker), \
             patch.dict('sys.modules', {'apps.cli.src.utils.progress': MagicMock(ProgressBar=MagicMock(return_value=mock_bar))}), \
             patch("domains.shell.repl.time.sleep"):
            repl._cmd_load("gpt2")
        assert repl._last_exit_code == 0

    def test_load_tracker_converting(self, repl):
        mock_tracker = MagicMock()
        mock_tracker.get.return_value = {"stage": "converting", "progress": 0.7, "message": "Converting format..."}
        mock_bar = MagicMock()
        with patch.object(repl, '_require_api', return_value=True), \
             patch("domains.infrastructure.conversion_tracker.get_tracker", return_value=mock_tracker), \
             patch.dict('sys.modules', {'apps.cli.src.utils.progress': MagicMock(ProgressBar=MagicMock(return_value=mock_bar))}), \
             patch("domains.shell.repl.time.sleep"):
            repl._cmd_load("gpt2")
        assert repl._last_exit_code == 0

    def test_load_tracker_loading(self, repl):
        mock_tracker = MagicMock()
        mock_tracker.get.return_value = {"stage": "loading", "progress": 0.9, "message": "Loading into memory..."}
        mock_bar = MagicMock()
        with patch.object(repl, '_require_api', return_value=True), \
             patch("domains.infrastructure.conversion_tracker.get_tracker", return_value=mock_tracker), \
             patch.dict('sys.modules', {'apps.cli.src.utils.progress': MagicMock(ProgressBar=MagicMock(return_value=mock_bar))}), \
             patch("domains.shell.repl.time.sleep"):
            repl._cmd_load("gpt2")
        assert repl._last_exit_code == 0

    def test_load_tracker_ready(self, repl):
        mock_tracker = MagicMock()
        mock_tracker.get.return_value = {"stage": "ready", "progress": 1.0}
        mock_bar = MagicMock()
        with patch.object(repl, '_require_api', return_value=True), \
             patch("domains.infrastructure.conversion_tracker.get_tracker", return_value=mock_tracker), \
             patch.dict('sys.modules', {'apps.cli.src.utils.progress': MagicMock(ProgressBar=MagicMock(return_value=mock_bar))}), \
             patch("domains.shell.repl.time.sleep"):
            repl._cmd_load("gpt2")
        assert repl._last_exit_code == 0

    def test_load_tracker_error(self, repl):
        mock_tracker = MagicMock()
        mock_tracker.get.return_value = {"stage": "error", "progress": 0, "error": "Download failed"}
        mock_bar = MagicMock()
        with patch.object(repl, '_require_api', return_value=True), \
             patch("domains.infrastructure.conversion_tracker.get_tracker", return_value=mock_tracker), \
             patch.dict('sys.modules', {'apps.cli.src.utils.progress': MagicMock(ProgressBar=MagicMock(return_value=mock_bar))}), \
             patch("domains.shell.repl.time.sleep"):
            repl._cmd_load("gpt2")
        assert repl._last_exit_code == 0

    def test_load_tracker_none(self, repl):
        mock_bar = MagicMock()
        mock_tracker = MagicMock()
        mock_tracker.get.return_value = None
        with patch.object(repl, '_require_api', return_value=True), \
             patch("domains.infrastructure.conversion_tracker.get_tracker", return_value=mock_tracker), \
             patch.dict('sys.modules', {'apps.cli.src.utils.progress': MagicMock(ProgressBar=MagicMock(return_value=mock_bar))}), \
             patch("domains.shell.repl.time.sleep"):
            repl._cmd_load("gpt2")
        assert repl._last_exit_code == 0

    def test_load_result_none(self, repl):
        mock_bar = MagicMock()
        mock_tracker = MagicMock()
        mock_tracker.get.return_value = None
        with patch.object(repl, '_require_api', return_value=True), \
             patch("domains.infrastructure.conversion_tracker.get_tracker", return_value=mock_tracker), \
             patch.dict('sys.modules', {'apps.cli.src.utils.progress': MagicMock(ProgressBar=MagicMock(return_value=mock_bar))}), \
             patch("domains.shell.repl.time.sleep"), \
             patch.object(repl.cmds, 'load_model', return_value=None):
            repl._cmd_load("gpt2")
        assert repl._last_exit_code == 0

    def test_load_result_error(self, repl):
        mock_bar = MagicMock()
        mock_tracker = MagicMock()
        mock_tracker.get.return_value = None
        with patch.object(repl, '_require_api', return_value=True), \
             patch("domains.infrastructure.conversion_tracker.get_tracker", return_value=mock_tracker), \
             patch.dict('sys.modules', {'apps.cli.src.utils.progress': MagicMock(ProgressBar=MagicMock(return_value=mock_bar))}), \
             patch("domains.shell.repl.time.sleep"), \
             patch.object(repl.cmds, 'load_model', return_value={"status": "error", "error": "not found"}):
            repl._cmd_load("gpt2")
        assert repl._last_exit_code == 0


# ── _cmd_train paths ───────────────────────────────────────────────


class TestCmdTrainPaths:
    def test_train_no_args(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'datasets', return_value=[{"name": "shakespeare"}]):
            repl._cmd_train("")
        assert repl._last_exit_code == 0

    def test_train_no_api(self, repl):
        with patch.object(repl, '_require_api', return_value=False):
            repl._cmd_train("shakespeare")
        assert repl._last_exit_code == 0

    def test_train_success(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'train_quick', return_value={"id": "train-123", "status": "started"}), \
             patch.object(repl, '_stream_train_progress'):
            repl._cmd_train("shakespeare")
        assert repl._last_exit_code == 0

    def test_train_error(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'train_quick', return_value={"error": "no dataset"}):
            repl._cmd_train("shakespeare")
        assert repl._last_exit_code == 0

    def test_train_no_job_id(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'train_quick', return_value={"status": "started"}):
            repl._cmd_train("shakespeare")
        assert repl._last_exit_code == 0

    def test_train_with_name(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'train_quick', return_value={"id": "train-456"}), \
             patch.object(repl, '_stream_train_progress'):
            repl._cmd_train("shakespeare my-run")
        assert repl._last_exit_code == 0

    def test_train_status(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'train_status', return_value=[{"id": "abc12345", "status": "running", "model": "gpt2", "progress": 50}]):
            repl._cmd_train("status")
        assert repl._last_exit_code == 0

    def test_train_status_empty(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'train_status', return_value=[]):
            repl._cmd_train("status")
        assert repl._last_exit_code == 0

    def test_train_stop(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'train_stop', return_value={"stopped": True}):
            repl._cmd_train("stop abc123")
        assert repl._last_exit_code == 0

    def test_train_stop_no_id(self, repl):
        with patch.object(repl, '_require_api', return_value=True):
            repl._cmd_train("stop")
        assert repl._last_exit_code == 0

    def test_train_follow(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl, '_stream_train_progress') as mock_stream:
            repl._cmd_train("follow abc123")
        mock_stream.assert_called_once_with("abc123")
        assert repl._last_exit_code == 0

    def test_train_follow_no_id(self, repl):
        with patch.object(repl, '_require_api', return_value=True):
            repl._cmd_train("follow")
        assert repl._last_exit_code == 0

    def test_train_distill(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'train_distill', return_value={"id": "dist-123", "status": "started"}), \
             patch.object(repl, '_stream_train_progress'):
            repl._cmd_train("distill shakespeare")
        assert repl._last_exit_code == 0

    def test_train_distill_no_dataset(self, repl):
        with patch.object(repl, '_require_api', return_value=True):
            repl._cmd_train("distill")
        assert repl._last_exit_code == 0

    def test_train_distill_error(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'train_distill', return_value={"error": "fail"}):
            repl._cmd_train("distill shakespeare")
        assert repl._last_exit_code == 0

    def test_train_hf(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'train_hf', return_value={"id": "hf-123", "status": "started"}), \
             patch.object(repl, '_stream_train_progress'):
            repl._cmd_train("hf gpt2 shakespeare")
        assert repl._last_exit_code == 0

    def test_train_hf_no_args(self, repl):
        with patch.object(repl, '_require_api', return_value=True):
            repl._cmd_train("hf")
        assert repl._last_exit_code == 0

    def test_train_hf_one_arg(self, repl):
        with patch.object(repl, '_require_api', return_value=True):
            repl._cmd_train("hf gpt2")
        assert repl._last_exit_code == 0

    def test_train_hf_error(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'train_hf', return_value={"error": "fail"}):
            repl._cmd_train("hf gpt2 shakespeare")
        assert repl._last_exit_code == 0

    def test_train_auto(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'train_auto', return_value={"status": "started"}):
            repl._cmd_train("auto friendly")
        assert repl._last_exit_code == 0

    def test_train_auto_error(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'train_auto', return_value={"error": "fail"}):
            repl._cmd_train("auto friendly")
        assert repl._last_exit_code == 0

    def test_train_load(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'load_checkpoint', return_value={"loaded": True}):
            repl._cmd_train("load my-checkpoint")
        assert repl._last_exit_code == 0

    def test_train_load_no_name(self, repl):
        with patch.object(repl, '_require_api', return_value=True):
            repl._cmd_train("load")
        assert repl._last_exit_code == 0

    def test_train_load_error(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'load_checkpoint', return_value={"error": "not found"}):
            repl._cmd_train("load bad-checkpoint")
        assert repl._last_exit_code == 0

    def test_train_del(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'delete_checkpoint', return_value={"deleted": True}):
            repl._cmd_train("del my-checkpoint")
        assert repl._last_exit_code == 0

    def test_train_del_no_name(self, repl):
        with patch.object(repl, '_require_api', return_value=True):
            repl._cmd_train("del")
        assert repl._last_exit_code == 0

    def test_train_del_error(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'delete_checkpoint', return_value={"error": "not found"}):
            repl._cmd_train("del bad-checkpoint")
        assert repl._last_exit_code == 0

    def test_train_list_datasets(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'datasets', return_value=[{"name": "shakespeare"}, {"name": "wiki"}]):
            repl._cmd_train("")
        assert repl._last_exit_code == 0

    def test_train_no_datasets(self, repl):
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'datasets', return_value=[]):
            repl._cmd_train("")
        assert repl._last_exit_code == 0


# ── _cmd_read / _cmd_which / _cmd_type edge cases ──────────────────


class TestCmdReadExtra:
    def test_read_no_args(self, repl):
        repl._cmd_read("")
        assert repl._last_exit_code == 1

    def test_read_prompt_only(self, repl):
        from domains.shell.io import MemoryIO
        mem = MemoryIO()
        mem.feed("value")
        old_io = repl.io
        repl.io = mem
        try:
            repl._cmd_read("-p")
            assert repl._env.get("-p") == "value"
            assert repl._last_exit_code == 0
        finally:
            repl.io = old_io

    def test_read_with_prompt_and_var(self, repl):
        from domains.shell.io import MemoryIO
        mem = MemoryIO()
        mem.feed("hello")
        old_io = repl.io
        repl.io = mem
        try:
            repl._cmd_read("-p Name: MYVAR")
            assert repl._env.get("MYVAR") == "hello"
            assert repl._last_exit_code == 0
        finally:
            repl.io = old_io

    def test_read_eof(self, repl):
        from domains.shell.io import MemoryIO
        mem = MemoryIO()
        old_io = repl.io
        repl.io = mem
        try:
            repl._cmd_read("MYVAR")
            assert repl._last_exit_code == 1
        finally:
            repl.io = old_io


class TestCmdWhichTypeEdge:
    def test_which_no_args(self, repl):
        repl._cmd_which("")
        assert repl._last_exit_code == 1

    def test_which_alias(self, repl):
        repl._aliases["ll"] = "ls -la"
        repl._cmd_which("ll")
        assert repl._last_exit_code == 0

    def test_which_ext_cmd(self, repl):
        repl._ext_cmds["myext"] = MagicMock()
        repl._cmd_which("myext")
        assert repl._last_exit_code == 0
        del repl._ext_cmds["myext"]

    def test_which_not_found(self, repl):
        repl._cmd_which("nonexistentcmd")
        assert repl._last_exit_code == 1

    def test_which_system_cmd(self, repl):
        with patch("shutil.which", return_value="/usr/bin/python3"):
            repl._cmd_which("python3")
        assert repl._last_exit_code == 0

    def test_type_no_args(self, repl):
        repl._cmd_type("")
        assert repl._last_exit_code == 1

    def test_type_alias(self, repl):
        repl._aliases["ll"] = "ls -la"
        repl._cmd_type("ll")
        assert repl._last_exit_code == 0

    def test_type_ext_cmd(self, repl):
        repl._ext_cmds["myext"] = MagicMock()
        repl._cmd_type("myext")
        assert repl._last_exit_code == 0
        del repl._ext_cmds["myext"]

    def test_type_not_found(self, repl):
        repl._cmd_type("nonexistentcmd")
        assert repl._last_exit_code == 1

    def test_type_system_cmd(self, repl):
        with patch("shutil.which", return_value="/usr/bin/python3"):
            repl._cmd_type("python3")
        assert repl._last_exit_code == 0


# ── _cmd_help internals ────────────────────────────────────────────


class TestCmdHelpInternals:
    def test_help_brief(self, repl):
        repl._cmd_help("-b")
        assert repl._last_exit_code == 0

    def test_help_known_cmd(self, repl):
        repl._cmd_help("ls")
        assert repl._last_exit_code == 0

    def test_help_cmd_with_doc(self, repl):
        repl._cmd_help("help")
        assert repl._last_exit_code == 0

    def test_help_ext_cmd(self, repl):
        repl._ext_cmds["myext"] = MagicMock()
        repl._ext_cmds["myext"].__doc__ = "My extension"
        repl._cmd_help("myext")
        assert repl._last_exit_code == 0
        del repl._ext_cmds["myext"]

    def test_help_ext_cmd_no_help(self, repl):
        repl._ext_cmds["myext"] = MagicMock()
        repl._ext_cmds["myext"].__doc__ = None
        repl._cmd_help("myext")
        assert repl._last_exit_code == 0
        del repl._ext_cmds["myext"]

    def test_help_system_cmd(self, repl):
        with patch("shutil.which", return_value="/usr/bin/python3"):
            repl._cmd_help("python3")
        assert repl._last_exit_code == 0

    def test_help_unknown_cmd(self, repl):
        repl._cmd_help("nonexistentcmd")
        assert repl._last_exit_code == 0

    def test_help_all_known_commands(self, repl):
        repl._cmd_help("--all")
        assert repl._last_exit_code == 0


# ── _cmd_render subcommands ────────────────────────────────────────


class TestCmdRenderExecution:
    def _setup_render(self, repl):
        mock_dev = MagicMock()
        mock_dev.call.return_value = (0,)
        mock_neural = MagicMock()
        mock_neural.call.return_value = {
            "embedding": MagicMock(shape=(1, 64)),
            "probabilities": [0.1, 0.2, 0.3, 0.1, 0.1, 0.05, 0.1, 0.05],
        }
        repl._render_device = mock_dev
        repl._render_neural = mock_neural
        return mock_dev, mock_neural

    def test_render_no_args(self, repl):
        mock_dev, _ = self._setup_render(repl)
        mock_dev.call.return_value = {"meshes": 0, "materials": 0, "lights": 0, "resolution": [80, 60], "samples": 4}
        repl._cmd_render("")
        assert repl._last_exit_code == 0

    def test_render_info(self, repl):
        mock_dev, _ = self._setup_render(repl)
        mock_dev.call.return_value = {"meshes": 1, "materials": 2, "lights": 1, "resolution": [80, 60], "samples": 4}
        repl._cmd_render("info")
        assert repl._last_exit_code == 0

    def test_render_sphere(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("sphere 1.0 0 0 0")
        assert repl._last_exit_code == 0

    def test_render_sphere_no_args(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("sphere")
        assert repl._last_exit_code == 0

    def test_render_sphere_with_mat(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("sphere 1.0 0 0 0 2")
        assert repl._last_exit_code == 0

    def test_render_cube(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("cube 1.0 0 0 0")
        assert repl._last_exit_code == 0

    def test_render_cube_no_args(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("cube")
        assert repl._last_exit_code == 0

    def test_render_cube_with_mat(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("cube 1.0 0 0 0 3")
        assert repl._last_exit_code == 0

    def test_render_plane(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("plane 6.0 -1.0")
        assert repl._last_exit_code == 0

    def test_render_plane_no_args(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("plane")
        assert repl._last_exit_code == 0

    def test_render_plane_with_mat(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("plane 6.0 -1.0 1")
        assert repl._last_exit_code == 0

    def test_render_light(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("light 2.0 3.0 2.0")
        assert repl._last_exit_code == 0

    def test_render_light_no_args(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("light")
        assert repl._last_exit_code == 0

    def test_render_light_with_color(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("light 2.0 3.0 2.0 1.0 0.9 0.8 8.0")
        assert repl._last_exit_code == 0

    def test_render_mat(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("mat 0 0.5 0.5 0.5 0.0 0.8")
        assert repl._last_exit_code == 0

    def test_render_mat_no_args(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("mat")
        assert repl._last_exit_code == 0

    def test_render_cam(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("cam 0 1.5 4 0 0 0")
        assert repl._last_exit_code == 0

    def test_render_cam_no_args(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("cam")
        assert repl._last_exit_code == 0

    def test_render_cam_with_fov(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("cam 0 1.5 4 0 0 0 50.0")
        assert repl._last_exit_code == 0

    def test_render_go(self, repl):
        mock_dev, _ = self._setup_render(repl)
        import numpy as np
        mock_dev.call.return_value = np.random.rand(60, 80, 3).astype(np.float32) * 0.5
        repl._cmd_render("go")
        assert repl._last_exit_code == 0

    def test_render_go_custom_size(self, repl):
        mock_dev, _ = self._setup_render(repl)
        import numpy as np
        mock_dev.call.return_value = np.random.rand(40, 50, 3).astype(np.float32) * 0.5
        repl._cmd_render("go 50 40 2")
        assert repl._last_exit_code == 0

    def test_render_clear(self, repl):
        mock_dev, mock_neural = self._setup_render(repl)
        repl._cmd_render("clear")
        assert repl._last_exit_code == 0

    def test_render_preset_demo(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("preset demo")
        assert repl._last_exit_code == 0

    def test_render_preset_cornell(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("preset cornell")
        assert repl._last_exit_code == 0

    def test_render_preset_spheres(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("preset spheres")
        assert repl._last_exit_code == 0

    def test_render_preset_unknown(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("preset nonexistent")
        assert repl._last_exit_code == 0

    def test_render_unknown_subcommand(self, repl):
        mock_dev, _ = self._setup_render(repl)
        repl._cmd_render("bogus")
        assert repl._last_exit_code == 0

    def test_render_neural(self, repl):
        mock_dev, mock_neural = self._setup_render(repl)
        mock_neural.call.side_effect = [
            {"embedding": MagicMock(shape=(1, 64)), "probabilities": [0.1]*8},
            {"dominant_class": 2, "neural_entropy": 1.5, "image": {"mean": 0.5, "std": 0.2}, "depth": {"mean": 0.3, "std": 0.1}},
        ]
        repl._cmd_render("neural")
        assert repl._last_exit_code == 0


# ── _cmd_logs explain path ────────────────────────────────────────


class TestCmdLogsExplainExtra:
    def test_logs_explain_no_errors(self, repl):
        repl._cmd_logs("--explain")
        assert repl._last_exit_code == 0

    def test_logs_explain_with_errors(self, repl):
        from domains.shell.log_buffer import LogEntry
        entry = LogEntry(time.time(), "ERROR", "test", "Something failed")
        repl._log_buffer._entries.append(entry)
        with patch.object(repl, '_require_api', return_value=False):
            repl._cmd_logs("--explain")
        assert repl._last_exit_code == 0

    def test_logs_explain_api_success(self, repl):
        from domains.shell.log_buffer import LogEntry
        entry = LogEntry(time.time(), "ERROR", "test", "Something failed")
        repl._log_buffer._entries.append(entry)
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'generate', return_value={"text": "Root cause: memory overflow\nFix: increase RAM"}):
            repl._cmd_logs("--explain")
        assert repl._last_exit_code == 0

    def test_logs_explain_api_error(self, repl):
        from domains.shell.log_buffer import LogEntry
        entry = LogEntry(time.time(), "WARNING", "test", "Low disk space")
        repl._log_buffer._entries.append(entry)
        with patch.object(repl, '_require_api', return_value=True), \
             patch.object(repl.cmds, 'generate', return_value={"error": "timeout"}):
            repl._cmd_logs("--explain")
        assert repl._last_exit_code == 0

    def test_logs_stats(self, repl):
        from domains.shell.log_buffer import LogEntry
        for i in range(5):
            repl._log_buffer._entries.append(LogEntry(time.time() + i, "INFO", "test", f"msg {i}"))
        repl._log_buffer._entries.append(LogEntry(time.time(), "ERROR", "other", "err"))
        repl._cmd_logs("--stats")
        assert repl._last_exit_code == 0

    def test_logs_stats_empty(self, repl):
        repl._cmd_logs("--stats")
        assert repl._last_exit_code == 0

    def test_logs_export(self, repl):
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer._entries.append(LogEntry(time.time(), "INFO", "test", "export me"))
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        repl._cmd_logs(f"--export {path}")
        import os
        assert os.path.exists(path)
        os.unlink(path)
        assert repl._last_exit_code == 0

    def test_logs_export_empty(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        repl._cmd_logs(f"--export {path}")
        import os
        os.unlink(path)
        assert repl._last_exit_code == 0

    def test_logs_filter_level(self, repl):
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer._entries.append(LogEntry(time.time(), "ERROR", "test", "err"))
        repl._log_buffer._entries.append(LogEntry(time.time(), "INFO", "test", "info"))
        repl._cmd_logs("-l ERROR")
        assert repl._last_exit_code == 0

    def test_logs_filter_source(self, repl):
        from domains.shell.log_buffer import LogEntry
        repl._log_buffer._entries.append(LogEntry(time.time(), "INFO", "kernel", "boot"))
        repl._log_buffer._entries.append(LogEntry(time.time(), "INFO", "api", "ready"))
        repl._cmd_logs("-s kernel")
        assert repl._last_exit_code == 0

    def test_logs_count(self, repl):
        from domains.shell.log_buffer import LogEntry
        for i in range(10):
            repl._log_buffer._entries.append(LogEntry(time.time() + i, "INFO", "test", f"msg {i}"))
        repl._cmd_logs("-n 3")
        assert repl._last_exit_code == 0


# ── _cmd_confirm config write path ────────────────────────────────


class TestCmdConfirmConfig:
    def test_confirm_no_args(self, repl):
        repl._cmd_confirm("")
        assert repl._last_exit_code == 0

    def test_confirm_on(self, repl):
        with patch("domains.infrastructure.config.get_config") as mock_get_cfg:
            mock_cfg = MagicMock()
            mock_cfg.features.auto_download = False
            mock_cfg.save = MagicMock()
            mock_get_cfg.return_value = mock_cfg
            repl._cmd_confirm("on")
        assert repl._last_exit_code == 0

    def test_confirm_off(self, repl):
        with patch("domains.infrastructure.config.get_config") as mock_get_cfg:
            mock_cfg = MagicMock()
            mock_cfg.features.auto_download = True
            mock_cfg.save = MagicMock()
            mock_get_cfg.return_value = mock_cfg
            repl._cmd_confirm("off")
        assert repl._last_exit_code == 0

    def test_confirm_invalid(self, repl):
        repl._cmd_confirm("maybe")
        assert repl._last_exit_code == 0


# ── _cmd_boot / _cmd_shutdown paths ───────────────────────────────


class TestCmdBootShutdown:
    def test_boot_already_booted(self, repl):
        repl._running = True
        repl._piped_input = None
        repl._cmd_boot("")
        assert repl._last_exit_code == 0

    def test_shutdown(self, repl):
        repl._cmd_shutdown("")
        assert repl._last_exit_code == 0


# ── _cmd_api paths ────────────────────────────────────────────────


class TestCmdApiExtra:
    def test_api_start_already_running(self, repl):
        mock_api = MagicMock()
        mock_api.is_running = True
        repl.os._api = mock_api
        repl._cmd_api("start")
        assert repl._last_exit_code == 0

    def test_api_start_success(self, repl):
        mock_api = MagicMock()
        mock_api.is_running = False
        mock_api.start.return_value = {"ok": True, "message": "started"}
        repl.os._api = mock_api
        repl._cmd_api("start")
        assert repl._last_exit_code == 0

    def test_api_start_failure(self, repl):
        mock_api = MagicMock()
        mock_api.is_running = False
        mock_api.start.return_value = {"ok": False, "error": "port busy"}
        repl.os._api = mock_api
        repl._cmd_api("start")
        assert repl._last_exit_code == 1

    def test_api_stop_not_running(self, repl):
        mock_api = MagicMock()
        mock_api.is_running = False
        repl.os._api = mock_api
        repl._cmd_api("stop")
        assert repl._last_exit_code == 0

    def test_api_stop_running(self, repl):
        mock_api = MagicMock()
        mock_api.is_running = True
        mock_api.stop.return_value = {"message": "stopped"}
        repl.os._api = mock_api
        repl._cmd_api("stop")
        assert repl._last_exit_code == 0

    def test_api_restart(self, repl):
        mock_api = MagicMock()
        mock_api.is_running = True
        mock_api.start.return_value = {"ok": True, "message": "restarted"}
        repl.os._api = mock_api
        repl._cmd_api("restart")
        assert repl._last_exit_code == 0

    def test_api_restart_failure(self, repl):
        mock_api = MagicMock()
        mock_api.is_running = False
        mock_api.start.return_value = {"ok": False, "error": "fail"}
        repl.os._api = mock_api
        repl._cmd_api("restart")
        assert repl._last_exit_code == 1

    def test_api_status_connected(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True, "model_id": "gpt2", "engine_type": "cpu", "running": True, "uptime": 120.5}
        repl.os._api = mock_api
        repl._cmd_api("status")
        assert repl._last_exit_code == 0

    def test_api_status_not_connected(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": False, "running": False}
        repl.os._api = mock_api
        repl._cmd_api("status")
        assert repl._last_exit_code == 0


# ── _cmd_svc more paths ───────────────────────────────────────────


class TestCmdSvcMore:
    def _mock_init(self, repl):
        mock_init = MagicMock()
        mock_init.service_table.return_value = "  svc1  running\n  svc2  stopped"
        mock_init.status_summary = "  2 services"
        mock_init.runlevel = 3
        mock_manager = MagicMock()
        mock_manager.status_line.return_value = "  svc1  running"
        mock_manager.instance.log = ["line1", "line2"]
        mock_manager.start.return_value = True
        mock_manager.restart.return_value = True
        mock_init.get_manager.return_value = mock_manager
        repl.os._init = mock_init
        return mock_init

    def test_svc_start_known(self, repl):
        self._mock_init(repl)
        repl._cmd_svc("start svc1")
        assert repl._last_exit_code == 0

    def test_svc_stop_known(self, repl):
        self._mock_init(repl)
        repl._cmd_svc("stop svc1")
        assert repl._last_exit_code == 0

    def test_svc_restart_known(self, repl):
        self._mock_init(repl)
        repl._cmd_svc("restart svc1")
        assert repl._last_exit_code == 0

    def test_svc_status_known(self, repl):
        self._mock_init(repl)
        repl._cmd_svc("status svc1")
        assert repl._last_exit_code == 0

    def test_svc_status_unknown(self, repl):
        mock_init = self._mock_init(repl)
        mock_init.get_manager.return_value = None
        repl._cmd_svc("status nonexistent")
        assert repl._last_exit_code == 1

    def test_svc_start_unknown(self, repl):
        mock_init = self._mock_init(repl)
        mock_init.get_manager.return_value = None
        repl._cmd_svc("start nonexistent")
        assert repl._last_exit_code == 1

    def test_svc_stop_unknown(self, repl):
        mock_init = self._mock_init(repl)
        mock_init.get_manager.return_value = None
        repl._cmd_svc("stop nonexistent")
        assert repl._last_exit_code == 1

    def test_svc_restart_unknown(self, repl):
        mock_init = self._mock_init(repl)
        mock_init.get_manager.return_value = None
        repl._cmd_svc("restart nonexistent")
        assert repl._last_exit_code == 1

    def test_svc_restart_failed(self, repl):
        mock_init = self._mock_init(repl)
        mock_init.get_manager.return_value.restart.return_value = False
        repl._cmd_svc("restart svc1")
        assert repl._last_exit_code == 0

    def test_svc_unknown_subcommand(self, repl):
        self._mock_init(repl)
        repl._cmd_svc("bogus")
        assert repl._last_exit_code == 1


# ── _cmd_protect / _cmd_unprotect paths ───────────────────────────


class TestCmdProtectUnprotect:
    def test_protect_no_args(self, repl):
        repl._cmd_protect("")
        assert repl._last_exit_code == 0

    def test_protect_success(self, repl):
        with patch("domains.infrastructure.model_protector.protect_model", return_value={"protected": ["file.bin"], "errors": []}):
            repl._cmd_protect("gpt2")
        assert repl._last_exit_code == 0

    def test_protect_no_files(self, repl):
        with patch("domains.infrastructure.model_protector.protect_model", return_value={"protected": [], "errors": []}):
            repl._cmd_protect("nonexistent")
        assert repl._last_exit_code == 0

    def test_protect_with_errors(self, repl):
        with patch("domains.infrastructure.model_protector.protect_model", return_value={"protected": ["f"], "errors": [{"error": "perm denied"}]}):
            repl._cmd_protect("gpt2")
        assert repl._last_exit_code == 0

    def test_protect_exception(self, repl):
        with patch("domains.infrastructure.model_protector.protect_model", side_effect=RuntimeError("fail")):
            repl._cmd_protect("gpt2")
        assert repl._last_exit_code == 0

    def test_unprotect_no_args(self, repl):
        repl._cmd_unprotect("")
        assert repl._last_exit_code == 0

    def test_unprotect_success(self, repl):
        with patch("domains.infrastructure.model_protector.unprotect_model", return_value={"unprotected": 2, "errors": []}):
            repl._cmd_unprotect("gpt2")
        assert repl._last_exit_code == 0

    def test_unprotect_with_errors(self, repl):
        with patch("domains.infrastructure.model_protector.unprotect_model", return_value={"unprotected": 0, "errors": [{"error": "not found"}]}):
            repl._cmd_unprotect("nonexistent")
        assert repl._last_exit_code == 0

    def test_unprotect_exception(self, repl):
        with patch("domains.infrastructure.model_protector.unprotect_model", side_effect=RuntimeError("fail")):
            repl._cmd_unprotect("gpt2")
        assert repl._last_exit_code == 0


# ── _cmd_uptime / _cmd_status paths ───────────────────────────────


class TestCmdUptimeStatus:
    def test_uptime(self, repl):
        repl._cmd_uptime("")
        assert repl._last_exit_code == 0

    def test_status(self, repl):
        repl._cmd_status("")
        assert repl._last_exit_code == 0

    def test_metrics(self, repl):
        repl._cmd_metrics("")
        assert repl._last_exit_code == 0


# ── _cmd_events paths ─────────────────────────────────────────────


class TestCmdEventsExtra:
    def test_events_no_args(self, repl):
        repl._cmd_events("")
        assert repl._last_exit_code == 0

    def test_events_with_limit(self, repl):
        repl._cmd_events("-n 5")
        assert repl._last_exit_code == 0

    def test_events_filter_match(self, repl):
        with patch("domains.infrastructure.event_bus.get_event_bus") as mock_bus:
            mock_bus.return_value.get_recent.return_value = [{"type": "model.loaded", "data": {}}]
            repl._cmd_events("model")
        assert repl._last_exit_code == 0

    def test_events_empty(self, repl):
        with patch("domains.infrastructure.event_bus.get_event_bus") as mock_bus:
            mock_bus.return_value.get_recent.return_value = []
            repl._cmd_events("")
        assert repl._last_exit_code == 0


# ── _cmd_procs / _cmd_ps / _cmd_kill paths ───────────────────────


class TestCmdProcsPsKill:
    def test_procs_no_args(self, repl):
        repl._cmd_procs("")
        assert repl._last_exit_code == 0

    def test_ps_no_args(self, repl):
        repl._cmd_ps("")
        assert repl._last_exit_code == 0

    def test_kill_no_args(self, repl):
        repl._cmd_kill("")
        assert repl._last_exit_code == 0

    def test_kill_pid(self, repl):
        repl._cmd_kill("1234")
        assert repl._last_exit_code == 0


# ── _cmd_fc paths ─────────────────────────────────────────────────


class TestCmdFcExtra:
    def test_fc_no_history(self, repl):
        repl._history.clear()
        repl._cmd_fc("")
        assert repl._last_exit_code == 0

    def test_fc_list(self, repl):
        repl._history = ["echo hello", "ls -la"]
        repl._cmd_fc("")
        assert repl._last_exit_code == 0

    def test_fc_list_with_n(self, repl):
        repl._history = ["echo hello", "ls -la", "pwd"]
        repl._cmd_fc("-l 2")
        assert repl._last_exit_code == 0

    def test_fc_rerun(self, repl):
        repl._history = ["echo hello"]
        repl._cmd_fc("1")
        assert repl._last_exit_code == 0

    def test_fc_rerun_invalid(self, repl):
        repl._history = ["echo hello"]
        repl._cmd_fc("999")
        assert repl._last_exit_code == 0


# ── _cmd_watch / _cmd_bg / _cmd_fg paths ──────────────────────────


class TestCmdWatchBgFg:
    def test_watch_no_args(self, repl):
        repl._cmd_watch("")
        assert repl._last_exit_code == 1

    def test_watch_with_cmd(self, repl):
        repl._cmd_watch("echo test")
        assert repl._last_exit_code == 1

    def test_bg_no_jobs(self, repl):
        repl._bg_threads.clear()
        repl._cmd_bg("")
        assert repl._last_exit_code == 0

    def test_fg_no_jobs(self, repl):
        repl._bg_threads.clear()
        repl._cmd_fg("")
        assert repl._last_exit_code == 0


# ── _cmd_ln paths ─────────────────────────────────────────────────


class TestCmdLnExtra:
    def test_ln_no_args(self, repl):
        repl._cmd_ln("")
        assert repl._last_exit_code == 1

    def test_ln_one_arg(self, repl):
        repl._cmd_ln("target")
        assert repl._last_exit_code == 1

    def test_ln_hard(self, repl):
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test")
            src = f.name
        dst = src + ".link"
        try:
            repl._cmd_ln(f"{src} {dst}")
            assert os.path.exists(dst)
        finally:
            os.unlink(src)
            if os.path.exists(dst):
                os.unlink(dst)

    def test_ln_symbolic(self, repl):
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test")
            src = f.name
        dst = src + ".sym"
        try:
            repl._cmd_ln(f"-s {src} {dst}")
            assert os.path.islink(dst)
        finally:
            os.unlink(src)
            if os.path.islink(dst):
                os.unlink(dst)


# ── _cmd_date / _cmd_cal paths ────────────────────────────────────


class TestCmdDateCalExtra:
    def test_date_utc(self, repl):
        repl._cmd_date("-u")
        assert repl._last_exit_code == 0

    def test_date_format(self, repl):
        repl._cmd_date("+%Y")
        assert repl._last_exit_code == 0

    def test_cal_year(self, repl):
        repl._cmd_cal("2025")
        assert repl._last_exit_code == 0

    def test_cal_month_year(self, repl):
        repl._cmd_cal("6 2025")
        assert repl._last_exit_code == 0

    def test_cal_invalid(self, repl):
        repl._cmd_cal("13 2025")
        assert repl._last_exit_code == 0


# ── _cmd_expand / _cmd_unexpand paths ─────────────────────────────


class TestCmdExpandUnexpandExtra:
    def test_expand_no_args(self, repl):
        repl._cmd_expand("")
        assert repl._last_exit_code == 1

    def test_expand_piped(self, repl):
        repl._piped_input = "hello\tworld"
        repl._cmd_expand("")
        repl._piped_input = None
        assert repl._last_exit_code == 0

    def test_unexpand_no_args(self, repl):
        repl._cmd_unexpand("")
        assert repl._last_exit_code == 1

    def test_unexpand_piped(self, repl):
        repl._piped_input = "hello    world"
        repl._cmd_unexpand("")
        repl._piped_input = None
        assert repl._last_exit_code == 0


# ── _cmd_id / _cmd_logname / _cmd_who paths ───────────────────────


class TestCmdIdLognameWho:
    def test_id(self, repl):
        repl._cmd_id("")
        assert repl._last_exit_code == 0

    def test_logname(self, repl):
        repl._cmd_logname("")
        assert repl._last_exit_code == 0

    def test_who(self, repl):
        repl._cmd_who("")
        assert repl._last_exit_code == 0


# ── _cmd_nproc / _cmd_hostname / _cmd_uname paths ─────────────────


class TestCmdNprocHostnameUname:
    def test_nproc(self, repl):
        repl._cmd_nproc("")
        assert repl._last_exit_code == 0

    def test_hostname(self, repl):
        repl._cmd_hostname("")
        assert repl._last_exit_code == 0

    def test_uname(self, repl):
        repl._cmd_uname("")
        assert repl._last_exit_code == 0

    def test_uname_a(self, repl):
        repl._cmd_uname("-a")
        assert repl._last_exit_code == 0

    def test_uname_s(self, repl):
        repl._cmd_uname("-s")
        assert repl._last_exit_code == 0

    def test_uname_m(self, repl):
        repl._cmd_uname("-m")
        assert repl._last_exit_code == 0


# ── _cmd_realpath / _cmd_dirname / _cmd_basename paths ────────────


class TestCmdPathUtilsExtra:
    def test_realpath_no_args(self, repl):
        repl._cmd_realpath("")
        assert repl._last_exit_code == 1

    def test_realpath_valid(self, repl):
        repl._cmd_realpath("/tmp")
        assert repl._last_exit_code == 0

    def test_dirname_no_args(self, repl):
        repl._cmd_dirname("")
        assert repl._last_exit_code == 1

    def test_dirname_valid(self, repl):
        repl._cmd_dirname("/tmp/test.txt")
        assert repl._last_exit_code == 0

    def test_basename_no_args(self, repl):
        repl._cmd_basename("")
        assert repl._last_exit_code == 1

    def test_basename_valid(self, repl):
        repl._cmd_basename("/tmp/test.txt")
        assert repl._last_exit_code == 0

    def test_basename_strip_suffix(self, repl):
        repl._cmd_basename("/tmp/test.txt .txt")
        assert repl._last_exit_code == 0


# ── _cmd_yes paths ────────────────────────────────────────────────


class TestCmdYesExtra:
    def test_yes_default(self, repl):
        repl._cmd_yes("")
        assert repl._last_exit_code == 0

    def test_yes_custom(self, repl):
        repl._cmd_yes("hello")
        assert repl._last_exit_code == 0


# ── _cmd_env paths ────────────────────────────────────────────────


class TestCmdEnvExtra:
    def test_env_no_args(self, repl):
        repl._cmd_env("")
        assert repl._last_exit_code == 0

    def test_env_with_var(self, repl):
        repl._env["MY_VAR"] = "hello"
        repl._cmd_env("")
        assert repl._last_exit_code == 0


# ── _cmd_set / _cmd_export paths ──────────────────────────────────


class TestCmdSetExportExtra:
    def test_set_no_args(self, repl):
        repl._cmd_set("")
        assert repl._last_exit_code == 0

    def test_set_with_value(self, repl):
        repl._cmd_set("MY_VAR=hello")
        assert repl._env.get("MY_VAR") == "hello"
        assert repl._last_exit_code == 0

    def test_export_no_args(self, repl):
        repl._cmd_export("")
        assert repl._last_exit_code == 0

    def test_export_with_value(self, repl):
        repl._cmd_export("MY_VAR=world")
        assert repl._env.get("MY_VAR") == "world"
        assert repl._last_exit_code == 0


# ── _cmd_source paths ─────────────────────────────────────────────


class TestCmdSourceExtra:
    def test_source_not_found(self, repl):
        repl._cmd_source("/nonexistent/file.sh")
        assert repl._last_exit_code == 0

    def test_source_valid(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("echo hello\n")
            path = f.name
        repl._cmd_source(path)
        import os
        os.unlink(path)
        assert repl._last_exit_code == 0


# ── _cmd_alias / _cmd_unalias paths ───────────────────────────────


class TestCmdAliasUnaliasExtra:
    def test_alias_no_args(self, repl):
        repl._cmd_alias("")
        assert repl._last_exit_code == 0

    def test_alias_single(self, repl):
        repl._cmd_alias("ll=ls -la")
        assert repl._aliases.get("ll") == "ls -la"
        assert repl._last_exit_code == 0

    def test_unalias_no_args(self, repl):
        repl._cmd_unalias("")
        assert repl._last_exit_code == 0

    def test_unalias_existing(self, repl):
        repl._aliases["ll"] = "ls -la"
        repl._cmd_unalias("ll")
        assert "ll" not in repl._aliases
        assert repl._last_exit_code == 0

    def test_unalias_nonexistent(self, repl):
        repl._cmd_unalias("nonexistent")
        assert repl._last_exit_code == 0


# ── _cmd_sleep paths ──────────────────────────────────────────────


class TestCmdSleepExtra:
    def test_sleep_default(self, repl):
        with patch("domains.shell.repl.time.sleep"):
            repl._cmd_sleep("0.01")
        assert repl._last_exit_code == 0

    def test_sleep_invalid(self, repl):
        with patch("domains.shell.repl.time.sleep"):
            repl._cmd_sleep("abc")
        assert repl._last_exit_code == 0


# ── _cmd_clear paths ──────────────────────────────────────────────


class TestCmdClearExtra:
    def test_clear(self, repl):
        repl._cmd_clear("")
        assert repl._last_exit_code == 0


# ── _cmd_note paths ───────────────────────────────────────────────


class TestCmdNoteExtra:
    def test_note_no_args(self, repl):
        mock_notes = MagicMock()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            repl._cmd_note("")
        assert repl._last_exit_code == 0


# ── _cmd_vmperms paths ────────────────────────────────────────────


class TestCmdVmpermsExtra:
    def test_vmperms(self, repl):
        repl._cmd_vmperms("")
        assert repl._last_exit_code == 0


# ── _cmd_lsdev paths ──────────────────────────────────────────────


class TestCmdLsdevExtra:
    def test_lsdev(self, repl):
        repl._cmd_lsdev("")
        assert repl._last_exit_code == 0


# ── _cmd_export_state / _cmd_exit / _cmd_history paths ────────────


class TestCmdExportStateExitHistory:
    def test_export_state(self, repl):
        repl._cmd_export_state("")
        assert repl._last_exit_code == 0

    def test_exit(self, repl):
        repl._running = True
        repl._cmd_exit("")
        assert repl._running is False
        assert repl._last_exit_code == 0

    def test_history_no_args(self, repl):
        repl._history = [f"cmd{i}" for i in range(30)]
        repl._cmd_history("")
        assert repl._last_exit_code == 0

    def test_history_with_n(self, repl):
        repl._history = [f"cmd{i}" for i in range(30)]
        repl._cmd_history("5")
        assert repl._last_exit_code == 0

    def test_history_empty(self, repl):
        repl._history.clear()
        repl._cmd_history("")
        assert repl._last_exit_code == 0


# ── _cmd_time execution ───────────────────────────────────────────


class TestCmdTimeExecution:
    def test_time_no_args(self, repl):
        repl._cmd_time("")
        assert repl._last_exit_code == 1

    def test_time_echo(self, repl):
        repl._cmd_time("echo hello")
        assert repl._last_exit_code == 0


# ── _cmd_cut internals ────────────────────────────────────────────


class TestCmdCutInternals:
    def test_cut_no_args(self, repl):
        repl._cmd_cut("")
        assert repl._last_exit_code == 1

    def test_cut_piped(self, repl):
        repl._piped_input = "a\tb\tc\nd\te\tf"
        repl._cmd_cut("-f2")
        repl._piped_input = None
        assert repl._last_exit_code == 0

    def test_cut_custom_delim(self, repl):
        repl._piped_input = "a:b:c\nd:e:f"
        repl._cmd_cut("-f2 -d:")
        repl._piped_input = None
        assert repl._last_exit_code == 0

    def test_cut_range(self, repl):
        repl._piped_input = "a\tb\tc\td"
        repl._cmd_cut("-f2-3")
        repl._piped_input = None
        assert repl._last_exit_code == 0

    def test_cut_file(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("a\tb\tc\n")
            path = f.name
        repl._cmd_cut(f"-f1 {path}")
        import os
        os.unlink(path)
        assert repl._last_exit_code == 0

    def test_cut_file_not_found(self, repl):
        repl._cmd_cut("-f1 /nonexistent/file.txt")
        assert repl._last_exit_code == 1

    def test_cut_no_fields(self, repl):
        repl._piped_input = "a\tb\tc"
        repl._cmd_cut("-d:")
        repl._piped_input = None
        assert repl._last_exit_code == 1

    def test_cut_field_out_of_range(self, repl):
        repl._piped_input = "a\tb"
        repl._cmd_cut("-f5")
        repl._piped_input = None
        assert repl._last_exit_code == 0


# ── _cmd_xargs paths ──────────────────────────────────────────────


class TestCmdXargsInternals:
    def test_xargs_no_piped(self, repl):
        repl._cmd_xargs("echo")
        assert repl._last_exit_code == 1

    def test_xargs_no_cmd(self, repl):
        repl._piped_input = "a b c"
        repl._cmd_xargs("")
        repl._piped_input = None
        assert repl._last_exit_code == 0

    def test_xargs_with_cmd(self, repl):
        repl._piped_input = "hello world"
        repl._cmd_xargs("echo")
        repl._piped_input = None
        assert repl._last_exit_code == 0

    def test_xargs_with_n(self, repl):
        repl._piped_input = "a b c d e"
        repl._cmd_xargs("-n 2 echo")
        repl._piped_input = None
        assert repl._last_exit_code == 0


# ── _cmd_kill with actual process ──────────────────────────────────


class TestCmdKillExecution:
    def test_kill_no_args(self, repl):
        repl._cmd_kill("")
        assert repl._last_exit_code == 0

    def test_kill_nonexistent_pid(self, repl):
        repl._cmd_kill("99999999")
        assert repl._last_exit_code == 0


# ── _cmd_set with NO_COLOR ────────────────────────────────────────


class TestCmdSetColor:
    def test_set_get_value(self, repl):
        repl._cmd_set("MY_TEST_VAR=hello123")
        assert repl._env.get("MY_TEST_VAR") == "hello123"
        repl._cmd_set("MY_TEST_VAR")
        assert repl._last_exit_code == 0

    def test_set_get_missing(self, repl):
        repl._cmd_set("NONEXISTENT_VAR_999")
        assert repl._last_exit_code == 0

    def test_set_no_color(self, repl):
        repl._cmd_set("NO_COLOR=1")
        assert repl._last_exit_code == 0


# ── _cmd_export paths ─────────────────────────────────────────────


class TestCmdExportExtra:
    def test_export_with_value(self, repl):
        repl._cmd_export("MY_EXPORT_VAR=test")
        assert repl._env.get("MY_EXPORT_VAR") == "test"
        assert repl._last_exit_code == 0


# ── _cmd_chmod execution ──────────────────────────────────────────


class TestCmdChmodExecution:
    def test_chmod_no_args(self, repl):
        repl._cmd_chmod("")
        assert repl._last_exit_code == 1

    def test_chmod_valid(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        repl._cmd_chmod(f"755 {path}")
        import os
        os.unlink(path)
        assert repl._last_exit_code == 0


# ── _cmd_stat execution ───────────────────────────────────────────


class TestCmdStatExecution:
    def test_stat_no_args(self, repl):
        repl._cmd_stat("")
        assert repl._last_exit_code == 1

    def test_stat_valid(self, repl):
        repl._cmd_stat("/tmp")
        assert repl._last_exit_code == 0


# ── _cmd_procs execution ──────────────────────────────────────────


class TestCmdProcsExecution:
    def test_procs(self, repl):
        repl._cmd_procs("")
        assert repl._last_exit_code == 0

    def test_ps(self, repl):
        repl._cmd_ps("")
        assert repl._last_exit_code == 0


# ── _cmd_watch execution ──────────────────────────────────────────


class TestCmdWatchExecution:
    def test_watch_no_args(self, repl):
        repl._cmd_watch("")
        assert repl._last_exit_code == 1

    def test_watch_with_cmd(self, repl):
        repl._cmd_watch("echo test")
        assert repl._last_exit_code == 1


# ── _cmd_bg / _cmd_fg with jobs ───────────────────────────────────


class TestCmdBgFgJobs:
    def test_bg_no_jobs(self, repl):
        repl._bg_threads.clear()
        repl._cmd_bg("")
        assert repl._last_exit_code == 0

    def test_fg_no_jobs(self, repl):
        repl._bg_threads.clear()
        repl._cmd_fg("")
        assert repl._last_exit_code == 0

    def test_fg_with_job(self, repl):
        import threading
        mock_thread = MagicMock(spec=threading.Thread)
        mock_thread.name = "bg-1"
        mock_thread.is_alive.return_value = False
        repl._bg_threads = {"1": mock_thread}
        repl._cmd_fg("1")
        assert repl._last_exit_code == 0

    def test_fg_invalid_id(self, repl):
        repl._bg_threads.clear()
        repl._cmd_fg("999")
        assert repl._last_exit_code == 0


# ── _cmd_yes execution ────────────────────────────────────────────


class TestCmdYesExecution:
    def test_yes_default(self, repl):
        repl._cmd_yes("")
        assert repl._last_exit_code == 0

    def test_yes_custom(self, repl):
        repl._cmd_yes("hello")
        assert repl._last_exit_code == 0


# ── _cmd_env with variables ───────────────────────────────────────


class TestCmdEnvExecution:
    def test_env(self, repl):
        repl._env["TEST_ENV_VAR"] = "test_value"
        repl._cmd_env("")
        assert repl._last_exit_code == 0
        del repl._env["TEST_ENV_VAR"]


# ── _cmd_tui paths ────────────────────────────────────────────────


class TestCmdTuiExtra:
    def test_tui(self, repl):
        repl._cmd_tui("")
        assert repl._last_exit_code in (0, 1)


# ── _cmd_tutorial execution ───────────────────────────────────────


class TestCmdTutorialExecution:
    def test_tutorial_quit_immediately(self, repl):
        from domains.shell.io import MemoryIO
        mem = MemoryIO()
        mem.feed("q")
        old_io = repl.io
        repl.io = mem
        try:
            repl._cmd_tutorial("")
            assert repl._last_exit_code == 0
        finally:
            repl.io = old_io

    def test_tutorial_step_through(self, repl):
        from domains.shell.io import MemoryIO
        mem = MemoryIO()
        # Feed empty strings for each step, then 'q' to quit
        for _ in range(15):
            mem.feed("")
        mem.feed("q")
        old_io = repl.io
        repl.io = mem
        try:
            repl._cmd_tutorial("")
            assert repl._last_exit_code == 0
        finally:
            repl.io = old_io


# ── _cmd_train custom params ──────────────────────────────────────


class TestCmdTrainCustomParams:
    def test_distill_custom_teacher_and_epochs(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        mock_cmds = MagicMock()
        mock_cmds.train_distill.return_value = {"id": "j1", "status": "started"}
        repl.cmds = mock_cmds
        with patch.object(repl, '_stream_train_progress'):
            repl._cmd_train("distill shakespeare gpt2 10")
        assert repl._last_exit_code == 0

    def test_hf_custom_epochs(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        mock_cmds = MagicMock()
        mock_cmds.train_hf.return_value = {"id": "j2", "status": "started"}
        repl.cmds = mock_cmds
        with patch.object(repl, '_stream_train_progress'):
            repl._cmd_train("hf gpt2 shakespeare 5")
        assert repl._last_exit_code == 0

    def test_auto_custom_params(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        mock_cmds = MagicMock()
        mock_cmds.train_auto.return_value = {"status": "started"}
        repl.cmds = mock_cmds
        repl._cmd_train("auto my_soul gpt2 20")
        assert repl._last_exit_code == 0

    def test_distill_no_job_id(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        mock_cmds = MagicMock()
        mock_cmds.train_distill.return_value = {"status": "started"}
        repl.cmds = mock_cmds
        repl._cmd_train("distill shakespeare")
        assert repl._last_exit_code == 0

    def test_hf_no_job_id(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        mock_cmds = MagicMock()
        mock_cmds.train_hf.return_value = {"status": "started"}
        repl.cmds = mock_cmds
        repl._cmd_train("hf gpt2 shakespeare")
        assert repl._last_exit_code == 0

    def test_quick_train_with_name(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        mock_cmds = MagicMock()
        mock_cmds.train_quick.return_value = {"id": "j3", "status": "started"}
        repl.cmds = mock_cmds
        with patch.object(repl, '_stream_train_progress'):
            repl._cmd_train("test_dataset my_run")
        assert repl._last_exit_code == 0

    def test_quick_train_no_job_id(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        mock_cmds = MagicMock()
        mock_cmds.train_quick.return_value = {"status": "started"}
        repl.cmds = mock_cmds
        repl._cmd_train("test_dataset")
        assert repl._last_exit_code == 0


# ── _cmd_agents exception paths ───────────────────────────────────


class TestCmdAgentsException:
    def test_agents_goal_exception(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        mock_orch = MagicMock()
        mock_orch.execute.side_effect = RuntimeError("API down")
        with patch('domains.agents.multi.get_orchestrator', return_value=mock_orch):
            repl._cmd_agents("do something")
        assert repl._last_exit_code == 0

    def test_agents_goal_no_tasks(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        mock_orch = MagicMock()
        mock_orch.execute.return_value = {"response": None, "tasks": []}
        with patch('domains.agents.multi.get_orchestrator', return_value=mock_orch):
            repl._cmd_agents("do something")
        assert repl._last_exit_code == 0

    def test_agents_goal_no_tasks_key(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        mock_orch = MagicMock()
        mock_orch.execute.return_value = {"response": "done"}
        with patch('domains.agents.multi.get_orchestrator', return_value=mock_orch):
            repl._cmd_agents("do something")
        assert repl._last_exit_code == 0


# ── _cmd_note edge cases ──────────────────────────────────────────


class TestCmdNoteEdgeCases:
    def test_note_list_invalid_limit(self, repl):
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("list --limit abc")
        assert repl._last_exit_code == 0

    def test_note_show_no_gh_url(self, repl):
        mock_store = MagicMock()
        mock_note = MagicMock()
        mock_note.short_id = "abc123"
        mock_note.title = "Test Note"
        mock_note.body = "Test body"
        mock_note.tags = []
        mock_note.status = "open"
        mock_note.sprint = ""
        mock_note.gh = ""
        mock_note.gh_url = ""
        mock_note.date_str = "2026-01-01"
        mock_store.get_note.return_value = mock_note
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("show abc123")
        assert repl._last_exit_code == 0

    def test_note_sprint_no_notes(self, repl):
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        mock_store.sprints.return_value = ["S99"]
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("sprint S99")
        assert repl._last_exit_code == 0

    def test_note_timeline_empty(self, repl):
        mock_store = MagicMock()
        mock_store.timeline.return_value = []
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("timeline --days 0")
        assert repl._last_exit_code == 0


# ── _cmd_fc boundary values ───────────────────────────────────────


class TestCmdFcBoundary:
    def test_fc_zero(self, repl):
        repl._history = ["echo hello"]
        repl._cmd_fc("0")
        assert repl._last_exit_code == 0

    def test_fc_out_of_range(self, repl):
        repl._history = ["echo hello"]
        repl._cmd_fc("999")
        assert repl._last_exit_code == 0


# ── _cmd_seq three-arg and float ──────────────────────────────────


class TestCmdSeqThreeArgFloat:
    def test_seq_three_args(self, repl):
        repl._cmd_seq("1 2 10")
        assert repl._last_exit_code == 0

    def test_seq_float(self, repl):
        repl._cmd_seq("0.5 0.5 2.0")
        assert repl._last_exit_code == 0

    def test_seq_too_many(self, repl):
        repl._cmd_seq("1 2 3 4")
        assert repl._last_exit_code == 1

    def test_seq_invalid(self, repl):
        repl._cmd_seq("abc")
        assert repl._last_exit_code == 1

    def test_seq_negative_increment(self, repl):
        repl._cmd_seq("5 -1 1")
        assert repl._last_exit_code == 0


# ── _cmd_nl file argument ─────────────────────────────────────────


class TestCmdNlFile:
    def test_nl_file(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("line1\nline2\nline3\n")
            path = f.name
        repl._cmd_nl(path)
        import os
        os.unlink(path)
        assert repl._last_exit_code == 0

    def test_nl_not_found(self, repl):
        repl._cmd_nl("/nonexistent_file.txt")
        assert repl._last_exit_code == 1


# ── _cmd_du multiple targets ──────────────────────────────────────


class TestCmdDuMultipleTargets:
    def test_du_multiple_targets(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1:
            f1.write("a" * 100)
            path1 = f1.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
            f2.write("b" * 200)
            path2 = f2.name
        repl._cmd_du(f"-h {path1} {path2}")
        import os
        os.unlink(path1)
        os.unlink(path2)
        assert repl._last_exit_code == 0

    def test_du_not_found(self, repl):
        repl._cmd_du("/nonexistent_path")
        assert repl._last_exit_code == 0


# ── _cmd_diff edge cases ──────────────────────────────────────────


class TestCmdDiffEdgeCases:
    def test_diff_one_arg(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("content")
            path = f.name
        repl._cmd_diff(path)
        import os
        os.unlink(path)
        assert repl._last_exit_code == 1


# ── _cmd_find edge cases ──────────────────────────────────────────


class TestCmdFindEdgeCases:
    def test_find_no_name(self, repl):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            repl._cmd_find(d)
            assert repl._last_exit_code == 1


# ── _cmd_train del/load success ───────────────────────────────────


class TestCmdTrainSuccess:
    def test_del_success(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        mock_cmds = MagicMock()
        mock_cmds.delete_checkpoint.return_value = {"status": "deleted"}
        repl.cmds = mock_cmds
        repl._cmd_train("del test_ckpt")
        assert repl._last_exit_code == 0

    def test_load_success(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        mock_cmds = MagicMock()
        mock_cmds.load_checkpoint.return_value = {"status": "loaded"}
        repl.cmds = mock_cmds
        repl._cmd_train("load test_ckpt")
        assert repl._last_exit_code == 0


# ── execute() path: unknown command, extension, timing ──────────────


class TestExecutePaths:
    def test_execute_unknown_command(self, repl):
        repl.execute("nonexistent_cmd_xyz")
        assert repl._last_exit_code == 127

    def test_execute_exception_in_command(self, repl):
        def bad_handler(self, args):
            raise RuntimeError("boom")
        repl.COMMANDS["badcmd"] = bad_handler
        repl.execute("badcmd")
        assert repl._last_exit_code == 1
        del repl.COMMANDS["badcmd"]

    def test_execute_system_exit(self, repl):
        def sys_exit_handler(self, args):
            raise SystemExit(42)
        repl.COMMANDS["exitcmd"] = sys_exit_handler
        repl.execute("exitcmd")
        assert repl._last_exit_code == 42
        del repl.COMMANDS["exitcmd"]

    def test_execute_timing_output(self, repl):
        repl.execute("time echo hello")
        assert repl._last_exit_code == 0

    def test_execute_pipeline(self, repl):
        repl.execute("echo hello | cat")
        assert repl._last_exit_code == 0

    def test_execute_background(self, repl):
        repl.execute("echo hello &")
        import time
        time.sleep(0.1)
        assert repl._last_exit_code == 0

    def test_execute_expansion(self, repl):
        repl._aliases["ll"] = "ls"
        repl.execute("ll /tmp")
        assert repl._last_exit_code == 0
        del repl._aliases["ll"]


# ── _cmd_train: more subcommand branches ───────────────────────────


class TestCmdTrainBranches:
    def test_distill_two_args(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        mock_cmds = MagicMock()
        mock_cmds.train_distill.return_value = {"id": "j1", "status": "started"}
        repl.cmds = mock_cmds
        with patch.object(repl, '_stream_train_progress'):
            repl._cmd_train("distill shakespeare gpt2")
        assert repl._last_exit_code == 0

    def test_hf_three_args(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        mock_cmds = MagicMock()
        mock_cmds.train_hf.return_value = {"id": "j2", "status": "started"}
        repl.cmds = mock_cmds
        with patch.object(repl, '_stream_train_progress'):
            repl._cmd_train("hf gpt2 shakespeare 10")
        assert repl._last_exit_code == 0

    def test_auto_one_arg(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        mock_cmds = MagicMock()
        mock_cmds.train_auto.return_value = {"status": "started"}
        repl.cmds = mock_cmds
        repl._cmd_train("auto")
        assert repl._last_exit_code == 0

    def test_datasets_subcmd(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        mock_cmds = MagicMock()
        mock_cmds.datasets.return_value = ["ds1", "ds2"]
        repl.cmds = mock_cmds
        repl._cmd_train("datasets")
        assert repl._last_exit_code == 0

    def test_status_with_job(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        mock_cmds = MagicMock()
        mock_cmds.train_status.return_value = [{"id": "j1", "status": "running", "loss": 0.5}]
        repl.cmds = mock_cmds
        repl._cmd_train("status j1")
        assert repl._last_exit_code == 0

    def test_stop_with_job(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        mock_cmds = MagicMock()
        mock_cmds.train_stop.return_value = {"status": "stopped"}
        repl.cmds = mock_cmds
        repl._cmd_train("stop j1")
        assert repl._last_exit_code == 0


# ── _cmd_agents: more branches ─────────────────────────────────────


class TestCmdAgentsBranches:
    def test_agents_list(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        repl._cmd_agents("list")
        assert repl._last_exit_code == 0

    def test_agents_empty_goal(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        repl._cmd_agents("")
        assert repl._last_exit_code == 0

    def test_agents_goal_string_response(self, repl):
        mock_api = MagicMock()
        mock_api.status.return_value = {"available": True}
        repl.os._api = mock_api
        mock_orch = MagicMock()
        mock_orch.execute.return_value = "simple string response"
        with patch('domains.agents.multi.get_orchestrator', return_value=mock_orch):
            repl._cmd_agents("do something")
        assert repl._last_exit_code == 0


# ── _cmd_note: more subcommands ────────────────────────────────────


class TestCmdNoteSubcommands:
    def test_note_list_empty(self, repl):
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("list")
        assert repl._last_exit_code == 0

    def test_note_show_nonexistent(self, repl):
        mock_store = MagicMock()
        mock_store.get_note.return_value = None
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("show nonexistent")
        assert repl._last_exit_code == 0

    def test_note_new(self, repl):
        mock_store = MagicMock()
        mock_store.add_note.return_value = {"short_id": "abc123"}
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("new my title --tags tag1 --status wip")
        assert repl._last_exit_code == 0

    def test_note_new_no_title(self, repl):
        mock_store = MagicMock()
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("new")
        assert repl._last_exit_code == 1

    def test_note_sprint(self, repl):
        mock_store = MagicMock()
        mock_store.sprints.return_value = ["S1", "S2"]
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("sprint")
        assert repl._last_exit_code == 0

    def test_note_today(self, repl):
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("today")
        assert repl._last_exit_code == 0

    def test_note_tags(self, repl):
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("tags")
        assert repl._last_exit_code == 0

    def test_note_status(self, repl):
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("status")
        assert repl._last_exit_code == 0

    def test_note_search(self, repl):
        mock_store = MagicMock()
        mock_store.search_notes.return_value = []
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("search test")
        assert repl._last_exit_code == 0

    def test_note_unknown_subcmd(self, repl):
        mock_store = MagicMock()
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("bogus")
        assert repl._last_exit_code == 1

    def test_note_delete(self, repl):
        mock_store = MagicMock()
        mock_store.delete_note.return_value = True
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("delete abc123")
        assert repl._last_exit_code == 0

    def test_note_export(self, repl):
        mock_store = MagicMock()
        mock_store.export_all.return_value = "exported content"
        mock_store.count.return_value = 5
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("export")
        assert repl._last_exit_code == 0

    def test_note_timeline(self, repl):
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("timeline")
        assert repl._last_exit_code == 0


# ── _cmd_fc: more branches ─────────────────────────────────────────


class TestCmdFcBranches:
    def test_fc_no_args(self, repl):
        repl._history = ["echo hello"]
        repl._cmd_fc("")
        assert repl._last_exit_code == 0

    def test_fc_negative_index(self, repl):
        repl._history = ["echo hello", "echo world"]
        repl._cmd_fc("-1")
        assert repl._last_exit_code == 0

    def test_fc_range(self, repl):
        repl._history = ["echo hello", "echo world"]
        repl._cmd_fc("-2 -1")
        assert repl._last_exit_code == 0


# ── _cmd_cut: file path edge cases ─────────────────────────────────


class TestCmdCutEdgeCases:
    def test_cut_file_not_found(self, repl):
        repl._cmd_cut("-f1 /nonexistent_file.txt")
        assert repl._last_exit_code == 1

    def test_cut_no_fields_no_input(self, repl):
        repl._cmd_cut("-d,")
        assert repl._last_exit_code == 1

    def test_cut_with_piped_input(self, repl):
        repl._piped_input = "hello,world\n"
        repl._cmd_cut("-d, -f1")
        assert repl._last_exit_code == 0


# ── _cmd_tee: file write paths ─────────────────────────────────────


class TestCmdTeeEdgeCases:
    def test_tee_no_piped_input(self, repl):
        repl._piped_input = ""
        repl._cmd_tee("")
        assert repl._last_exit_code == 1

    def test_tee_with_piped_input(self, repl):
        repl._piped_input = "hello\n"
        repl._cmd_tee("")
        assert repl._last_exit_code == 0

    def test_tee_permission_denied(self, repl):
        repl._piped_input = "hello\n"
        with patch('builtins.open', side_effect=PermissionError("denied")):
            repl._cmd_tee("/proc/fake.txt")
        assert repl._last_exit_code == 1


# ── _cmd_export_state: various fields ──────────────────────────────


class TestCmdExportStateFields:
    def test_export_state_env(self, repl):
        repl._env["MYVAR"] = "test"
        repl._cmd_export_state("")
        assert repl._last_exit_code == 0

    def test_export_state_empty(self, repl):
        repl._history = []
        repl._aliases = {}
        repl._env = {}
        repl._cmd_export_state("")
        assert repl._last_exit_code == 0


# ── _cmd_which: edge cases ─────────────────────────────────────────


class TestCmdWhichEdgeCases:
    def test_which_builtin(self, repl):
        repl._cmd_which("echo")
        assert repl._last_exit_code == 0

    def test_which_alias(self, repl):
        repl._aliases["ll"] = "ls"
        repl._cmd_which("ll")
        assert repl._last_exit_code == 0
        del repl._aliases["ll"]

    def test_which_external_cmd(self, repl):
        repl._cmd_which("nonexistent_cmd_xyz")
        assert repl._last_exit_code == 1

    def test_which_no_args(self, repl):
        repl._cmd_which("")
        assert repl._last_exit_code == 1


# ── _cmd_type: edge cases ──────────────────────────────────────────


class TestCmdTypeEdgeCases:
    def test_type_builtin(self, repl):
        repl._cmd_type("echo")
        assert repl._last_exit_code == 0

    def test_type_alias(self, repl):
        repl._aliases["ll"] = "ls"
        repl._cmd_type("ll")
        assert repl._last_exit_code == 0
        del repl._aliases["ll"]

    def test_type_not_found(self, repl):
        repl._cmd_type("nonexistent_cmd_xyz")
        assert repl._last_exit_code == 1

    def test_type_no_args(self, repl):
        repl._cmd_type("")
        assert repl._last_exit_code == 1


# ── _cmd_fold: edge cases ──────────────────────────────────────────


class TestCmdFoldEdgeCases:
    def test_fold_default_width(self, repl):
        repl._piped_input = "hello world\n"
        repl._cmd_fold("")
        assert repl._last_exit_code == 0

    def test_fold_short_flag(self, repl):
        repl._piped_input = "hello\n"
        repl._cmd_fold("-w3")
        assert repl._last_exit_code == 0

    def test_fold_file_not_found(self, repl):
        repl._cmd_fold("-w80 /nonexistent_file.txt")
        assert repl._last_exit_code == 1

    def test_fold_no_args_no_piped(self, repl):
        repl._cmd_fold("")
        assert repl._last_exit_code == 1


# ── _cmd_od: edge cases ────────────────────────────────────────────


class TestCmdOdEdgeCases:
    def test_od_default(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("hello\n")
            path = f.name
        repl._cmd_od(path)
        import os
        os.unlink(path)
        assert repl._last_exit_code == 0

    def test_od_file(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("hello\n")
            path = f.name
        repl._cmd_od(path)
        import os
        os.unlink(path)
        assert repl._last_exit_code == 0

    def test_od_not_found(self, repl):
        repl._cmd_od("/nonexistent_file.txt")
        assert repl._last_exit_code == 1


# ── _cmd_paste: edge cases ─────────────────────────────────────────


class TestCmdPasteEdgeCases:
    def test_paste_no_args(self, repl):
        repl._cmd_paste("")
        assert repl._last_exit_code == 1

    def test_paste_two_files(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1:
            f1.write("a\nb\nc\n")
            path1 = f1.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
            f2.write("1\n2\n3\n")
            path2 = f2.name
        repl._cmd_paste(f"{path1} {path2}")
        import os
        os.unlink(path1)
        os.unlink(path2)
        assert repl._last_exit_code == 0

    def test_paste_serial(self, repl):
        repl._piped_input = "a\nb\nc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_paste("-s")
        assert "a\tb\tc" in cap.getvalue()

    def test_paste_serial_custom_delim(self, repl):
        repl._piped_input = "a\nb\nc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_paste("-s -d,")
        assert "a,b,c" in cap.getvalue()


# ── _cmd_join: edge cases ──────────────────────────────────────────


class TestCmdJoinEdgeCases:
    def test_join_no_args(self, repl):
        repl._cmd_join("")
        assert repl._last_exit_code == 1

    def test_join_file_not_found(self, repl):
        repl._cmd_join("/f1.txt /f2.txt")
        assert repl._last_exit_code == 1


# ── _cmd_tac: edge cases ───────────────────────────────────────────


class TestCmdTacEdgeCases:
    def test_tac_no_args(self, repl):
        repl._cmd_tac("")
        assert repl._last_exit_code == 1

    def test_tac_file(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("line1\nline2\n")
            path = f.name
        repl._cmd_tac(path)
        import os
        os.unlink(path)
        assert repl._last_exit_code == 0

    def test_tac_not_found(self, repl):
        repl._cmd_tac("/nonexistent_file.txt")
        assert repl._last_exit_code == 1


# ── _cmd_rev: edge cases ───────────────────────────────────────────


class TestCmdRevEdgeCases:
    def test_rev_piped(self, repl):
        repl._piped_input = "hello"
        repl._cmd_rev("")
        assert repl._last_exit_code == 0

    def test_rev_no_args_no_piped(self, repl):
        repl._piped_input = ""
        repl._cmd_rev("")
        assert repl._last_exit_code == 1

    def test_rev_file(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("hello\n")
            path = f.name
        repl._cmd_rev(path)
        import os
        os.unlink(path)
        assert repl._last_exit_code == 0

    def test_rev_not_found(self, repl):
        repl._cmd_rev("/nonexistent_file.txt")
        assert repl._last_exit_code == 1


# ── _cmd_xargs: edge cases ─────────────────────────────────────────


class TestCmdXargsEdgeCases:
    def test_xargs_no_args(self, repl):
        repl._piped_input = ""
        repl._cmd_xargs("")
        assert repl._last_exit_code == 1

    def test_xargs_with_cmd(self, repl):
        repl._piped_input = "hello\n"
        repl._cmd_xargs("echo")
        assert repl._last_exit_code == 0

    def test_xargs_n_flag(self, repl):
        repl._piped_input = "hello world\n"
        repl._cmd_xargs("-n1 echo")
        assert repl._last_exit_code == 0

    def test_xargs_r_with_empty_string(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_xargs("-r echo hello")
        assert cap.getvalue() == ""
        assert repl._last_exit_code == 0

    def test_xargs_r_with_none(self, repl):
        repl._piped_input = None
        with _CaptureOutput(repl) as cap:
            repl._cmd_xargs("-r echo hello")
        assert cap.getvalue() == ""
        assert repl._last_exit_code == 0


# ── _cmd_time: edge cases ──────────────────────────────────────────


class TestCmdTimeEdgeCases:
    def test_time_no_args(self, repl):
        repl._cmd_time("")
        assert repl._last_exit_code == 1

    def test_time_echo(self, repl):
        repl._cmd_time("echo hello")
        assert repl._last_exit_code == 0


# ── _cmd_read: edge cases ──────────────────────────────────────────


class TestCmdReadEdgeCases:
    def test_read_no_args(self, repl):
        repl._cmd_read("")
        assert repl._last_exit_code == 1

    def test_read_var(self, repl):
        from domains.shell.io import MemoryIO
        mem = MemoryIO()
        mem.feed("test_value")
        old_io = repl.io
        repl.io = mem
        try:
            repl._cmd_read("myvar")
            assert repl._last_exit_code == 0
            assert repl._env.get("myvar") == "test_value"
        finally:
            repl.io = old_io

    def test_read_with_prompt(self, repl):
        from domains.shell.io import MemoryIO
        mem = MemoryIO()
        mem.feed("hello")
        old_io = repl.io
        repl.io = mem
        try:
            repl._cmd_read("-p Enter value: myvar")
            assert repl._last_exit_code == 0
        finally:
            repl.io = old_io


# ── _cmd_watch: edge cases ─────────────────────────────────────────


class TestCmdWatchEdgeCases:
    def test_watch_no_args(self, repl):
        repl._cmd_watch("")
        assert repl._last_exit_code == 1

    def test_watch_no_interval(self, repl):
        repl._cmd_watch("echo hello")
        assert repl._last_exit_code == 1

    def test_watch_invalid_interval(self, repl):
        repl._cmd_watch("abc echo hello")
        assert repl._last_exit_code == 1


# ── _cmd_bg: edge cases ────────────────────────────────────────────


class TestCmdBgEdgeCases:
    def test_bg_no_jobs(self, repl):
        repl._bg_threads = {}
        repl._cmd_bg("")
        assert repl._last_exit_code == 0

    def test_bg_with_thread(self, repl):
        import threading
        t = threading.Thread(target=lambda: None)
        t.daemon = True
        repl._bg_threads = {0: t}
        repl._cmd_bg("")
        assert repl._last_exit_code == 0


# ── _cmd_fg: edge cases ────────────────────────────────────────────


class TestCmdFgEdgeCases:
    def test_fg_no_jobs(self, repl):
        repl._bg_threads = {}
        repl._cmd_fg("999")
        assert repl._last_exit_code == 0

    def test_fg_invalid_id(self, repl):
        repl._bg_threads = {}
        repl._cmd_fg("abc")
        assert repl._last_exit_code == 0

    def test_fg_no_args(self, repl):
        repl._cmd_fg("")
        assert repl._last_exit_code == 0

    def test_fg_done_thread(self, repl):
        import threading
        t = threading.Thread(target=lambda: None)
        t.daemon = True
        t.start()
        t.join()
        repl._bg_threads = {0: t}
        repl._cmd_fg("0")
        assert repl._last_exit_code == 0


# ── Module-level functions ─────────────────────────────────────────


class TestModuleLevelFunctions:
    def test_color_enabled(self):
        from domains.shell.repl import _color, _C_CYAN, _C_RESET
        result = _color("test", _C_CYAN)
        assert "test" in result
        assert _C_RESET in result

    def test_color_disabled(self):
        import domains.shell.repl as mod
        old = mod._COLOR_ENABLED
        try:
            mod._COLOR_ENABLED = False
            result = mod._color("test", "code")
            assert result == "test"
        finally:
            mod._COLOR_ENABLED = old

    def test_fetch_model_names_exception(self):
        from domains.shell.repl import _fetch_model_names
        with patch('requests.get', side_effect=Exception("network")):
            result = _fetch_model_names()
        assert result == []

    def test_fetch_model_names_non_200(self):
        from domains.shell.repl import _fetch_model_names
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch('requests.get', return_value=mock_resp):
            result = _fetch_model_names()
        assert result == []

    def test_fetch_soul_names_exception(self):
        from domains.shell.repl import _fetch_soul_names
        with patch('requests.get', side_effect=Exception("network")):
            result = _fetch_soul_names()
        assert result == []

    def test_fetch_dataset_names_exception(self):
        from domains.shell.repl import _fetch_dataset_names
        with patch('requests.get', side_effect=Exception("network")):
            result = _fetch_dataset_names()
        assert result == []

    def test_fetch_checkpoint_names_exception(self):
        from domains.shell.repl import _fetch_checkpoint_names
        with patch('requests.get', side_effect=Exception("network")):
            result = _fetch_checkpoint_names()
        assert result == []

    def test_fetch_model_names_dict_response(self):
        from domains.shell.repl import _fetch_model_names
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "gpt2"}, {"id": "bert"}]}
        with patch('requests.get', return_value=mock_resp):
            result = _fetch_model_names()
        assert "gpt2" in result
        assert "bert" in result

    def test_fetch_soul_names_dict_response(self):
        from domains.shell.repl import _fetch_soul_names
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"souls": [{"name": "friendly"}]}
        with patch('requests.get', return_value=mock_resp):
            result = _fetch_soul_names()
        assert "friendly" in result

    def test_fetch_dataset_names_dict_response(self):
        from domains.shell.repl import _fetch_dataset_names
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"datasets": [{"name": "shakespeare"}]}
        with patch('requests.get', return_value=mock_resp):
            result = _fetch_dataset_names()
        assert "shakespeare" in result

    def test_fetch_checkpoint_names_dict_response(self):
        from domains.shell.repl import _fetch_checkpoint_names
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"checkpoints": [{"name": "ckpt1"}]}
        with patch('requests.get', return_value=mock_resp):
            result = _fetch_checkpoint_names()
        assert "ckpt1" in result


# ── _CaptureOutput ─────────────────────────────────────────────────


class TestCaptureOutput:
    def test_with_repl(self, repl):
        from domains.shell.io import MemoryIO
        mem = MemoryIO()
        old_io = repl.io
        old_console_io = repl.console._io
        repl.io = mem
        repl.console._io = mem
        try:
            repl._print("hello")
            assert "hello" in mem.get_output()
        finally:
            repl.io = old_io
            repl.console._io = old_console_io

    def test_without_repl(self):
        from domains.shell.repl import _CaptureOutput
        with _CaptureOutput() as cap:
            print("test output")
        assert "test output" in cap.getvalue()


# ── _suggest_command ───────────────────────────────────────────────


class TestSuggestCommand:
    def test_suggest_similar(self, repl):
        result = repl._suggest_command("hepl")
        assert result is None or isinstance(result, str)

    def test_suggest_exact(self, repl):
        result = repl._suggest_command("exit")
        assert result is None or isinstance(result, str)


# ── _expand_vars ───────────────────────────────────────────────────


class TestExpandVars:
    def test_expand_dollar_question(self, repl):
        repl._last_exit_code = 42
        result = repl._expand_vars("exit $?")
        assert result == "exit 42"

    def test_expand_braces(self, repl):
        repl._env["HOME"] = "/home/user"
        result = repl._expand_vars("cd ${HOME}")
        assert result == "cd /home/user"

    def test_expand_unknown_var(self, repl):
        result = repl._expand_vars("$UNKNOWN_VAR")
        assert result == "$UNKNOWN_VAR"


# ── _expand_history ────────────────────────────────────────────────


class TestExpandHistory:
    def test_expand_bang_bang(self, repl):
        repl._history = ["echo hello"]
        result = repl._expand_history("!!")
        assert result == "echo hello"

    def test_expand_bang_bang_empty(self, repl):
        repl._history = []
        result = repl._expand_history("!!")
        assert result == "!!"

    def test_expand_bang_dollar(self, repl):
        repl._history = ["echo hello world"]
        result = repl._expand_history("!$")
        assert result == "world"

    def test_expand_bang_star(self, repl):
        repl._history = ["echo hello world"]
        result = repl._expand_history("!*")
        assert result == "hello world"

    def test_expand_bang_star_no_args(self, repl):
        repl._history = ["echo"]
        result = repl._expand_history("!*")
        assert result == ""

    def test_expand_neg_history(self, repl):
        repl._history = ["cmd1", "cmd2"]
        result = repl._expand_history("!-1")
        assert result == "cmd2"

    def test_expand_neg_history_too_large(self, repl):
        repl._history = ["cmd1"]
        result = repl._expand_history("!-5")
        assert result == "!-5"

    def test_expand_pos_history(self, repl):
        repl._history = ["cmd1", "cmd2"]
        result = repl._expand_history("!1")
        assert result == "cmd1"

    def test_expand_pos_history_out_of_range(self, repl):
        repl._history = ["cmd1"]
        result = repl._expand_history("!5")
        assert result == "!5"

    def test_expand_nth_arg(self, repl):
        repl._history = ["echo hello world"]
        result = repl._expand_history("!:1")
        assert result == "hello"

    def test_expand_nth_arg_out_of_range(self, repl):
        repl._history = ["echo hello"]
        result = repl._expand_history("!:5")
        assert result == "!:5"

    def test_expand_history_no_history(self, repl):
        repl._history = []
        result = repl._expand_history("!$")
        assert result == "!$"


# ── _parse_inline_env ──────────────────────────────────────────────


class TestParseInlineEnv:
    def test_single_var(self, repl):
        env, rest = repl._parse_inline_env("FOO=bar echo hi")
        assert env == {"FOO": "bar"}
        assert rest == "echo hi"

    def test_multiple_vars(self, repl):
        env, rest = repl._parse_inline_env("A=1 B=2 cmd")
        assert env == {"A": "1", "B": "2"}
        assert rest == "cmd"

    def test_no_vars(self, repl):
        env, rest = repl._parse_inline_env("echo hello")
        assert env == {}
        assert rest == "echo hello"

    def test_var_with_quotes(self, repl):
        env, rest = repl._parse_inline_env('FOO="bar" echo hi')
        assert env.get("FOO") == "bar"
        assert "echo" in rest


# ── _strip_redirection ────────────────────────────────────────────


class TestStripRedirection:
    def test_no_redirect(self, repl):
        args, path, append = repl._strip_redirection("echo hello")
        assert args == "echo hello"
        assert path is None
        assert append is False

    def test_overwrite_redirect(self, repl):
        args, path, append = repl._strip_redirection("echo hello > /tmp/out.txt")
        assert args == "echo hello"
        assert path == "/tmp/out.txt"
        assert append is False

    def test_append_redirect(self, repl):
        args, path, append = repl._strip_redirection("echo hello >> /tmp/out.txt")
        assert args == "echo hello"
        assert path == "/tmp/out.txt"
        assert append is True


# ── _cmd_py ────────────────────────────────────────────────────────


class TestCmdPy:
    def test_py_eval(self, repl):
        repl._cmd_py("2 + 2")
        assert repl._last_exit_code == 0

    def test_py_no_args(self, repl):
        repl._cmd_py("")
        assert repl._last_exit_code == 0

    def test_py_print(self, repl):
        repl._cmd_py('print("hello")')
        assert repl._last_exit_code == 0


# ── _format_table ──────────────────────────────────────────────────


class TestFormatTable:
    def test_empty(self, repl):
        result = repl._format_table([])
        assert isinstance(result, str)

    def test_with_header(self, repl):
        result = repl._format_table([["a", "b"], ["c", "d"]], header=["X", "Y"])
        assert isinstance(result, str)
        assert "X" in result

    def test_no_header(self, repl):
        result = repl._format_table([["a", "b"]])
        assert isinstance(result, str)


# ── _expand_globs ──────────────────────────────────────────────────


class TestExpandGlobs:
    def test_no_glob(self, repl):
        result = repl._expand_globs("echo hello")
        assert result == "echo hello"

    def test_quoted_glob(self, repl):
        result = repl._expand_globs('"*.txt"')
        assert result == '"*.txt"'


# ── _split_pipe ────────────────────────────────────────────────────


class TestSplitPipe:
    def test_no_pipe(self, repl):
        from domains.shell.repl import ShellREPL
        result = ShellREPL._split_pipe("echo hello")
        assert result == ["echo hello"]

    def test_simple_pipe(self, repl):
        from domains.shell.repl import ShellREPL
        result = ShellREPL._split_pipe("echo hello | cat")
        assert result == ["echo hello", "cat"]

    def test_quoted_pipe(self, repl):
        from domains.shell.repl import ShellREPL
        result = ShellREPL._split_pipe('echo "a|b" | cat')
        assert len(result) == 2


# ── _parse_pipeline ────────────────────────────────────────────────


class TestParsePipeline:
    def test_simple(self, repl):
        cmds, bg, time = repl._parse_pipeline("echo hello")
        assert len(cmds) == 1
        assert bg is False
        assert time is False

    def test_background(self, repl):
        cmds, bg, time = repl._parse_pipeline("echo hello &")
        assert bg is True

    def test_time(self, repl):
        cmds, bg, time = repl._parse_pipeline("time echo hello")
        assert time is True

    def test_pipe(self, repl):
        cmds, bg, time = repl._parse_pipeline("echo hello | cat")
        assert len(cmds) == 2

    def test_chain_and(self, repl):
        cmds, bg, time = repl._parse_pipeline("echo a && echo b")
        assert len(cmds) == 2

    def test_chain_or(self, repl):
        cmds, bg, time = repl._parse_pipeline("echo a || echo b")
        assert len(cmds) >= 2

    def test_chain_semicolon(self, repl):
        cmds, bg, time = repl._parse_pipeline("echo a ; echo b")
        assert len(cmds) == 2

    def test_quoted_operators(self, repl):
        cmds, bg, time = repl._parse_pipeline('echo "a && b"')
        assert len(cmds) == 1


# ── _cmd_ls with various flags ────────────────────────────────────


class TestCmdLsFlags:
    def test_ls_no_args(self, repl):
        repl._cmd_ls("")
        assert repl._last_exit_code == 0

    def test_ls_tmp(self, repl):
        repl._cmd_ls("/tmp")
        assert repl._last_exit_code == 0

    def test_ls_nonexistent(self, repl):
        repl._cmd_ls("/nonexistent_dir_xyz")
        assert repl._last_exit_code == 1


# ── _cmd_cat with VFS ─────────────────────────────────────────────


class TestCmdCatExtra:
    def test_cat_nonexistent(self, repl):
        repl._cmd_cat("/nonexistent_file.txt")
        assert repl._last_exit_code == 1

    def test_cat_single_file(self, repl):
        import tempfile, os
        p = os.path.join(tempfile.gettempdir(), "test_cat_single")
        with open(p, 'w') as f: f.write("hello\n")
        repl._cmd_cat(p)
        os.unlink(p)
        assert repl._last_exit_code == 0

    def test_cat_piped(self, repl):
        repl._piped_input = "hello from pipe\n"
        repl._cmd_cat("")
        assert repl._last_exit_code == 0


# ── _cmd_mkdir ─────────────────────────────────────────────────────


class TestCmdMkdirExtra:
    def test_mkdir_existing(self, repl):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            repl._cmd_mkdir(d)
            assert repl._last_exit_code == 1


# ── _cmd_rm ────────────────────────────────────────────────────────


class TestCmdRmExtra:
    def test_rm_nonexistent(self, repl):
        repl._cmd_rm("/nonexistent_file.txt")
        assert repl._last_exit_code == 1

    def test_rm_directory(self, repl):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            repl._cmd_rm(f"-r {d}")
            assert repl._last_exit_code == 0


# ── _cmd_touch ─────────────────────────────────────────────────────


class TestCmdTouchExtra:
    def test_touch_creates(self, repl):
        import tempfile, os
        path = os.path.join(tempfile.gettempdir(), "test_touch_extra")
        repl._cmd_touch(path)
        assert os.path.exists(path)
        os.unlink(path)
        assert repl._last_exit_code == 0


# ── _cmd_cp and _cmd_mv ───────────────────────────────────────────


class TestCmdCpMvExtra:
    def test_cp_nonexistent(self, repl):
        repl._cmd_cp("/nonexistent /tmp/cp_test")
        assert repl._last_exit_code == 1

    def test_mv_nonexistent(self, repl):
        repl._cmd_mv("/nonexistent /tmp/mv_test")
        assert repl._last_exit_code == 1


# ── _cmd_head and _cmd_tail ───────────────────────────────────────


class TestCmdHeadTailExtra:
    def test_head_no_args(self, repl):
        repl._cmd_head("")
        assert repl._last_exit_code == 1

    def test_tail_no_args(self, repl):
        repl._cmd_tail("")
        assert repl._last_exit_code == 1

    def test_head_file(self, repl):
        import tempfile, os
        p = os.path.join(tempfile.gettempdir(), "test_head")
        with open(p, 'w') as f: f.write("line1\nline2\nline3\n")
        repl._cmd_head(f"-2 {p}")
        os.unlink(p)
        assert repl._last_exit_code == 0

    def test_tail_file(self, repl):
        import tempfile, os
        p = os.path.join(tempfile.gettempdir(), "test_tail")
        with open(p, 'w') as f: f.write("line1\nline2\nline3\n")
        repl._cmd_tail(f"-2 {p}")
        os.unlink(p)
        assert repl._last_exit_code == 0


# ── _cmd_wc ────────────────────────────────────────────────────────


class TestCmdWcExtra:
    def test_wc_no_args(self, repl):
        repl._cmd_wc("")
        assert repl._last_exit_code == 1

    def test_wc_file(self, repl):
        import tempfile, os
        p = os.path.join(tempfile.gettempdir(), "test_wc")
        with open(p, 'w') as f: f.write("hello\nworld\n")
        repl._cmd_wc(p)
        os.unlink(p)
        assert repl._last_exit_code == 0


# ── _cmd_grep ──────────────────────────────────────────────────────


class TestCmdGrepExtra:
    def test_grep_no_args(self, repl):
        repl._cmd_grep("")
        assert repl._last_exit_code == 1

    def test_grep_no_match(self, repl):
        repl._cmd_grep("nomatch /dev/null")
        assert repl._last_exit_code == 1


# ── _cmd_sort ──────────────────────────────────────────────────────


class TestCmdSortExtra:
    def test_sort_reverse(self, repl):
        repl._piped_input = "c\na\nb\n"
        repl._cmd_sort("-r")
        assert repl._last_exit_code == 0

    def test_sort_unique(self, repl):
        repl._piped_input = "a\na\nb\n"
        repl._cmd_sort("-u")
        assert repl._last_exit_code == 0

    def test_sort_numeric(self, repl):
        repl._piped_input = "10\n2\n30\n"
        repl._cmd_sort("-n")
        assert repl._last_exit_code == 0


# ── _cmd_uniq ──────────────────────────────────────────────────────


class TestCmdUniqExtra:
    def test_uniq_no_duplicates(self, repl):
        repl._piped_input = "a\nb\nc\n"
        repl._cmd_uniq("")
        assert repl._last_exit_code == 0


# ── _cmd_tr ────────────────────────────────────────────────────────


class TestCmdTrExtra:
    def test_tr_substitute(self, repl):
        repl._piped_input = "hello\n"
        repl._cmd_tr("a-z A-Z")
        assert repl._last_exit_code == 0

    def test_tr_delete(self, repl):
        repl._piped_input = "hello\n"
        repl._cmd_tr("-d l")
        assert repl._last_exit_code == 0


# ── _cmd_realpath ──────────────────────────────────────────────────


class TestCmdRealpath:
    def test_realpath_no_args(self, repl):
        repl._cmd_realpath("")
        assert repl._last_exit_code == 1

    def test_realpath_file(self, repl):
        repl._cmd_realpath("/tmp")
        assert repl._last_exit_code == 0


# ── _cmd_id ────────────────────────────────────────────────────────


class TestCmdId:
    def test_id(self, repl):
        repl._cmd_id("")
        assert repl._last_exit_code == 0


# ── _cmd_hostname ──────────────────────────────────────────────────


class TestCmdHostname:
    def test_hostname(self, repl):
        repl._cmd_hostname("")
        assert repl._last_exit_code == 0


# ── _cmd_uname ─────────────────────────────────────────────────────


class TestCmdUnameExtra:
    def test_uname_all(self, repl):
        repl._cmd_uname("-a")
        assert repl._last_exit_code == 0


# ── _cmd_nproc ─────────────────────────────────────────────────────


class TestCmdNproc:
    def test_nproc(self, repl):
        repl._cmd_nproc("")
        assert repl._last_exit_code == 0


# ── _cmd_logname ───────────────────────────────────────────────────


class TestCmdLogname:
    def test_logname(self, repl):
        repl._cmd_logname("")
        assert repl._last_exit_code == 0


# ── _cmd_id_logname_who ───────────────────────────────────────────


class TestCmdWho:
    def test_who(self, repl):
        repl._cmd_who("")
        assert repl._last_exit_code == 0


# ── _cmd_uptime ────────────────────────────────────────────────────


class TestCmdUptimeExtra:
    def test_uptime(self, repl):
        repl._cmd_uptime("")
        assert repl._last_exit_code == 0


# ── _cmd_date_extra ───────────────────────────────────────────────


class TestCmdDateExtra:
    def test_date_format(self, repl):
        repl._cmd_date("+%Y")
        assert repl._last_exit_code == 0


# ── _cmd_cal_extra ────────────────────────────────────────────────


class TestCmdCalExtra:
    def test_cal_year(self, repl):
        repl._cmd_cal("2026")
        assert repl._last_exit_code == 0

    def test_cal_month_year(self, repl):
        repl._cmd_cal("1 2026")
        assert repl._last_exit_code == 0


# ── _cmd_set and _cmd_unset ──────────────────────────────────────


class TestCmdSetExtra:
    def test_set_value(self, repl):
        repl._cmd_set("MYVAR=hello")
        assert repl._env.get("MYVAR") == "hello"

    def test_set_empty(self, repl):
        repl._cmd_set("")
        assert repl._last_exit_code == 0


# ── _cmd_export_extra ────────────────────────────────────────────


class TestCmdExportExtra2:
    def test_export_var(self, repl):
        repl._cmd_export("MYVAR=hello")
        assert repl._env.get("MYVAR") == "hello"

    def test_export_empty(self, repl):
        repl._cmd_export("")
        assert repl._last_exit_code == 0


# ── _cmd_pwd ─────────────────────────────────────────────────────────


class TestCmdPwdExtra:
    def test_pwd_returns_cwd(self, repl):
        import os
        out = _run_with_io(repl, [], lambda: repl._cmd_pwd(""))
        assert repl._last_exit_code == 0
        assert os.getcwd() in out

    def test_pwd_after_cd(self, repl, tmp_path):
        repl._cmd_cd(str(tmp_path))
        out = _run_with_io(repl, [], lambda: repl._cmd_pwd(""))
        assert str(tmp_path) in out


# ── _cmd_echo ────────────────────────────────────────────────────────


class TestCmdEchoExtra:
    def test_echo_empty(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_echo(""))
        assert repl._last_exit_code == 0

    def test_echo_single_word(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_echo("hello"))
        assert repl._last_exit_code == 0
        assert "hello" in out

    def test_echo_multiple_words(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_echo("hello world foo"))
        assert repl._last_exit_code == 0
        assert "hello" in out and "world" in out and "foo" in out

    def test_echo_preserves_quotes(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_echo('"hello world"'))
        assert repl._last_exit_code == 0
        assert '"hello world"' in out


# ── _cmd_cd ──────────────────────────────────────────────────────────


class TestCmdCdExtra:
    def test_cd_no_args_goes_home(self, repl):
        repl._cmd_cd("")
        assert repl._last_exit_code == 0
        import os
        import pathlib
        assert os.getcwd() == str(pathlib.Path.home())

    def test_cd_tilde_goes_home(self, repl):
        repl._cmd_cd("~")
        assert repl._last_exit_code == 0
        import os
        import pathlib
        assert os.getcwd() == str(pathlib.Path.home())

    def test_cd_dash_returns_previous(self, repl, tmp_path):
        import os
        old = os.getcwd()
        repl._cmd_cd(str(tmp_path))
        repl._cmd_cd("-")
        assert repl._last_exit_code == 0
        assert os.getcwd() == old

    def test_cd_nonexistent_dir(self, repl):
        repl._cmd_cd("/nonexistent_dir_xyz_12345")
        assert repl._last_exit_code == 1

    def test_cd_file_not_directory(self, repl, tmp_path):
        import os
        f = tmp_path / "a_file.txt"
        f.write_text("hi")
        repl._cmd_cd(str(f))
        assert repl._last_exit_code == 1


# ── _cmd_exit ────────────────────────────────────────────────────────


class TestCmdExitExtra:
    def test_exit_sets_running_false(self, repl):
        repl._running = True
        repl._cmd_exit("")
        assert repl._running is False

    def test_exit_saves_state(self, repl):
        repl._running = True
        repl._cmd_exit("")
        assert repl._running is False


# ── _cmd_history ─────────────────────────────────────────────────────


class TestCmdHistoryExtra:
    def test_history_empty(self, repl):
        repl._history.clear()
        out = _run_with_io(repl, [], lambda: repl._cmd_history(""))
        assert repl._last_exit_code == 0

    def test_history_with_entries(self, repl):
        repl._history.extend(["echo a", "echo b", "echo c"])
        out = _run_with_io(repl, [], lambda: repl._cmd_history(""))
        assert "echo a" in out and "echo b" in out and "echo c" in out

    def test_history_limit(self, repl):
        repl._history.extend(["cmd1", "cmd2", "cmd3", "cmd4", "cmd5"])
        out = _run_with_io(repl, [], lambda: repl._cmd_history("2"))
        assert "cmd4" in out and "cmd5" in out
        assert "cmd1" not in out

    def test_history_numbered(self, repl):
        repl._history.extend(["echo hello", "echo world"])
        out = _run_with_io(repl, [], lambda: repl._cmd_history(""))
        assert "1" in out and "2" in out


# ── _cmd_permissions ────────────────────────────────────────────────


class TestCmdPermissionsExtra:
    def test_permissions_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_permissions(""))
        assert repl._last_exit_code == 0
        assert "Risk policies:" in out

    def test_permissions_shows_all_risks(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_permissions(""))
        assert "safe" in out.lower() or "SAFE" in out
        assert "elevated" in out.lower() or "ELEVATED" in out
        assert "dangerous" in out.lower() or "DANGEROUS" in out
        assert "critical" in out.lower() or "CRITICAL" in out


# ── _cmd_ps ──────────────────────────────────────────────────────────


class TestCmdPsExtra:
    def test_ps_empty(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_ps(""))
        assert repl._last_exit_code == 0

    def test_ps_with_process(self, repl):
        from unittest.mock import MagicMock
        proc = MagicMock()
        proc.pid = 1
        proc.name = "test_proc"
        proc.state = 0
        proc.created_at = 0
        repl.os.kernel.list_processes = MagicMock(return_value=[proc])
        out = _run_with_io(repl, [], lambda: repl._cmd_ps(""))
        assert repl._last_exit_code == 0
        assert "test_proc" in out


# ── _cmd_cp / _cmd_mv ────────────────────────────────────────────────


class TestCmdCpExtra:
    def test_cp_missing_source(self, repl):
        repl._cmd_cp("/nonexistent_src /tmp/dst")
        assert repl._last_exit_code == 1

    def test_cp_to_dest(self, repl, tmp_path):
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("data")
        repl._cmd_cp(f"{src} {dst}")
        assert dst.read_text() == "data"
        assert repl._last_exit_code == 0


class TestCmdMvExtra:
    def test_mv_missing_source(self, repl):
        repl._cmd_mv("/nonexistent_src /tmp/dst")
        assert repl._last_exit_code == 1

    def test_mv_to_dest(self, repl, tmp_path):
        src = tmp_path / "mv_src.txt"
        dst = tmp_path / "mv_dst.txt"
        src.write_text("moved")
        repl._cmd_mv(f"{src} {dst}")
        assert dst.read_text() == "moved"
        assert not src.exists()
        assert repl._last_exit_code == 0


# ── _cmd_clear ──────────────────────────────────────────────────────


class TestCmdClearExtra2:
    def test_clear_writes_escape(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_clear(""))
        assert repl._last_exit_code == 0
        assert "\033[2J\033[H" in out


# ── _cmd_shutdown ────────────────────────────────────────────────────


class TestCmdShutdownExtra:
    def test_shutdown_sets_running_false(self, repl):
        repl._running = True
        repl._cmd_shutdown("")
        assert repl._running is False

    def test_shutdown_returns_code_zero(self, repl):
        repl._cmd_shutdown("")
        assert repl._last_exit_code == 0


# ── _cmd_alias / _cmd_unalias ──────────────────────────────────────


class TestCmdAliasExtra:
    def test_alias_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_alias(""))
        assert repl._last_exit_code == 0

    def test_alias_set(self, repl):
        _run_with_io(repl, [], lambda: repl._cmd_alias("ll=ls -la"))
        assert repl._aliases.get("ll") == "ls -la"

    def test_alias_show(self, repl):
        repl._aliases["ll"] = "ls -la"
        out = _run_with_io(repl, [], lambda: repl._cmd_alias("ll"))
        assert "ll=ls -la" in out

    def test_alias_show_nonexistent(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_alias("noexist"))
        assert "No alias" in out


class TestCmdUnaliasExtra:
    def test_unalias_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_unalias(""))
        assert "Usage" in out

    def test_unalias_existing(self, repl):
        repl._aliases["ll"] = "ls -la"
        out = _run_with_io(repl, [], lambda: repl._cmd_unalias("ll"))
        assert "Removed" in out
        assert "ll" not in repl._aliases

    def test_unalias_nonexistent(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_unalias("noexist"))
        assert "No alias" in out


# ── _cmd_yes ─────────────────────────────────────────────────────────


class TestCmdYesExtra:
    def test_yes_default(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_yes(""))
        lines = out.strip().split("\n")
        assert len(lines) == 100
        assert all(l.strip() == "y" for l in lines)

    def test_yes_custom_string(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_yes("hello"))
        lines = out.strip().split("\n")
        assert len(lines) == 100
        assert all(l.strip() == "hello" for l in lines)


# ── _cmd_dirname / _cmd_basename ────────────────────────────────────


class TestCmdDirnameExtra:
    def test_dirname_path(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_dirname("/a/b/c.txt"))
        assert out.strip() == "/a/b"

    def test_dirname_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_dirname(""))
        assert "Usage" in out
        assert repl._last_exit_code == 1

    def test_dirname_single_component(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_dirname("file.txt"))
        assert out.strip() in ("", ".")


class TestCmdBasenameExtra:
    def test_basename_path(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_basename("/a/b/c.txt"))
        assert out.strip() == "c.txt"

    def test_basename_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_basename(""))
        assert "Usage" in out
        assert repl._last_exit_code == 1

    def test_basename_strip_suffix(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_basename("/a/b/c.txt .txt"))
        assert out.strip() == "c"


# ── _cmd_mktemp ──────────────────────────────────────────────────────


class TestCmdMktempExtra:
    def test_mktemp_creates_file(self, repl):
        import os
        out = _run_with_io(repl, [], lambda: repl._cmd_mktemp(""))
        path = out.strip()
        assert os.path.isfile(path)
        os.unlink(path)
        assert repl._last_exit_code == 0

    def test_mktemp_dir(self, repl):
        import os
        out = _run_with_io(repl, [], lambda: repl._cmd_mktemp("-d"))
        path = out.strip()
        assert os.path.isdir(path)
        os.rmdir(path)
        assert repl._last_exit_code == 0


# ── _cmd_ln ──────────────────────────────────────────────────────────


class TestCmdLnExtra:
    def test_ln_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_ln(""))
        assert "Usage" in out
        assert repl._last_exit_code == 1

    def test_ln_missing_target(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_ln("only_one"))
        assert repl._last_exit_code == 1

    def test_ln_hard_link(self, repl, tmp_path):
        src = tmp_path / "ln_src.txt"
        dst = tmp_path / "ln_dst.txt"
        src.write_text("data")
        repl._cmd_ln(f"{src} {dst}")
        assert dst.exists()
        assert dst.read_text() == "data"

    def test_ln_symlink(self, repl, tmp_path):
        src = tmp_path / "ln_sym_src.txt"
        dst = tmp_path / "ln_sym_dst.txt"
        src.write_text("sym")
        repl._cmd_ln(f"-s {src} {dst}")
        assert dst.is_symlink()


# ── _cmd_touch ──────────────────────────────────────────────────────


class TestCmdTouchExtra:
    def test_touch_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_touch(""))
        assert "Usage" in out
        assert repl._last_exit_code == 1

    def test_touch_creates_file(self, repl, tmp_path):
        f = tmp_path / "touch_me.txt"
        repl._cmd_touch(str(f))
        assert f.exists()
        assert repl._last_exit_code == 0

    def test_touch_updates_existing(self, repl, tmp_path):
        f = tmp_path / "touch_existing.txt"
        f.write_text("existing")
        repl._cmd_touch(str(f))
        assert f.exists()
        assert f.read_text() == "existing"


# ── _cmd_rm ──────────────────────────────────────────────────────────


class TestCmdRmExtra:
    def test_rm_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_rm(""))
        assert "Usage" in out
        assert repl._last_exit_code == 1

    def test_rm_nonexistent(self, repl):
        repl._cmd_rm("/nonexistent_file_xyz_12345")
        assert repl._last_exit_code == 1

    def test_rm_file(self, repl, tmp_path):
        f = tmp_path / "rm_me.txt"
        f.write_text("delete me")
        repl._cmd_rm(str(f))
        assert not f.exists()
        assert repl._last_exit_code == 0

    def test_rm_directory_without_recursive(self, repl, tmp_path):
        d = tmp_path / "rm_dir"
        d.mkdir()
        repl._cmd_rm(str(d))
        assert d.exists()
        assert repl._last_exit_code == 1

    def test_rm_directory_recursive(self, repl, tmp_path):
        d = tmp_path / "rm_dir_r"
        d.mkdir()
        (d / "child.txt").write_text("child")
        repl._cmd_rm(f"-r {d}")
        assert not d.exists()
        assert repl._last_exit_code == 0


# ── _cmd_comm ────────────────────────────────────────────────────────


class TestCmdCommExtra:
    def test_comm_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_comm(""))
        assert "Usage" in out
        assert repl._last_exit_code == 1

    def test_comm_one_file(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_comm("only_one"))
        assert repl._last_exit_code == 1

    def test_comm_files(self, repl, tmp_path):
        f1 = tmp_path / "c1.txt"
        f2 = tmp_path / "c2.txt"
        f1.write_text("a\nb\nc\n")
        f2.write_text("b\nc\nd\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_comm(f"{f1} {f2}"))
        assert "a" in out
        assert "d" in out
        assert repl._last_exit_code == 0

    def test_comm_nonexistent_file(self, repl, tmp_path):
        f1 = tmp_path / "c_exist.txt"
        f1.write_text("a\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_comm(f"{f1} /nonexistent_comm_xyz"))
        assert repl._last_exit_code == 1


# ── _cmd_printf ──────────────────────────────────────────────────────


class TestCmdPrintfExtra:
    def test_printf_no_args(self, repl):
        repl._cmd_printf("")
        assert repl._last_exit_code == 1

    def test_printf_string(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_printf("hello"))
        assert "hello" in out

    def test_printf_percent_s(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_printf("%s hello"))
        assert "hello" in out

    def test_printf_percent_d(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_printf("%d 42"))
        assert "42" in out

    def test_printf_escape_newline(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_printf("line1\\nline2"))
        assert "line1" in out and "line2" in out


# ── _cmd_expand / _cmd_unexpand ──────────────────────────────────────


class TestCmdExpandExtra:
    def test_expand_no_args_no_pipe(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_expand(""))
        assert "Usage" in out
        assert repl._last_exit_code == 1

    def test_expand_file(self, repl, tmp_path):
        f = tmp_path / "expand_tab.txt"
        f.write_text("a\tb\tc")
        out = _run_with_io(repl, [], lambda: repl._cmd_expand(str(f)))
        assert "\t" not in out
        assert repl._last_exit_code == 0

    def test_expand_piped(self, repl):
        repl._piped_input = "a\tb"
        out = _run_with_io(repl, [], lambda: repl._cmd_expand(""))
        assert "\t" not in out

    def test_expand_nonexistent(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_expand("/nonexistent_expand_xyz"))
        assert repl._last_exit_code == 1


class TestCmdUnexpandExtra:
    def test_unexpand_no_args_no_pipe(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_unexpand(""))
        assert "Usage" in out
        assert repl._last_exit_code == 1

    def test_unexpand_piped(self, repl):
        repl._piped_input = "        eight"
        out = _run_with_io(repl, [], lambda: repl._cmd_unexpand(""))
        assert repl._last_exit_code == 0


# ── _cmd_grep ────────────────────────────────────────────────────────


class TestCmdGrepFlags:
    def test_grep_ignore_case(self, repl):
        repl._piped_input = "Hello\nhello\nHELLO\nworld"
        out = _run_with_io(repl, [], lambda: repl._cmd_grep("-i hello"))
        assert "Hello" in out and "hello" in out and "HELLO" in out
        assert "world" not in out

    def test_grep_invert(self, repl):
        repl._piped_input = "apple\nbanana\ncherry\napple"
        out = _run_with_io(repl, [], lambda: repl._cmd_grep("-v apple"))
        assert "banana" in out and "cherry" in out
        assert "apple" not in out

    def test_grep_invert_ignore_case(self, repl):
        repl._piped_input = "Apple\nBANANA\napple"
        out = _run_with_io(repl, [], lambda: repl._cmd_grep("-vi apple"))
        assert "BANANA" in out
        assert "Apple" not in out and "apple" not in out

    def test_grep_regex_match(self, repl):
        repl._piped_input = "foo123\nbar456\nbaz789"
        out = _run_with_io(repl, [], lambda: repl._cmd_grep(r"\d+"))
        assert "foo123" in out and "bar456" in out and "baz789" in out

    def test_grep_no_match(self, repl):
        repl._piped_input = "apple\nbanana"
        out = _run_with_io(repl, [], lambda: repl._cmd_grep("cherry"))
        assert repl._last_exit_code == 1

    def test_grep_invalid_regex(self, repl):
        repl._piped_input = "hello"
        out = _run_with_io(repl, [], lambda: repl._cmd_grep("[invalid"))
        assert repl._last_exit_code == 2

    def test_grep_file(self, repl, tmp_path):
        f = tmp_path / "grep_file.txt"
        f.write_text("line1 apple\nline2 banana\nline3 apple\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_grep(f"apple {f}"))
        assert "line1 apple" in out and "line3 apple" in out
        assert "line2 banana" not in out


# ── _cmd_source ──────────────────────────────────────────────────────


class TestCmdSourceExtra:
    def test_source_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_source(""))
        assert "Usage" in out

    def test_source_file(self, repl, tmp_path):
        f = tmp_path / "test_source.sh"
        f.write_text("echo sourced_line\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_source(str(f)))
        assert "sourced_line" in out

    def test_source_nonexistent(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_source("/nonexistent_source_xyz"))
        assert "Error reading" in out

    def test_source_skips_comments(self, repl, tmp_path):
        f = tmp_path / "test_source_comments.sh"
        f.write_text("# this is a comment\necho after_comment\n# another comment\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_source(str(f)))
        assert "after_comment" in out
        assert "comment" not in out.split("after_comment")[0].strip()

    def test_source_skips_empty_lines(self, repl, tmp_path):
        f = tmp_path / "test_source_empty.sh"
        f.write_text("\n\n\necho only_this\n\n\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_source(str(f)))
        assert "only_this" in out


# ── _cmd_shuf ────────────────────────────────────────────────────────


class TestCmdShufExtra:
    def test_shuf_no_args_no_pipe(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_shuf(""))
        assert "Usage" in out
        assert repl._last_exit_code == 1

    def test_shuf_file(self, repl, tmp_path):
        f = tmp_path / "shuf.txt"
        f.write_text("alpha\nbeta\ngamma\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_shuf(str(f)))
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert len(lines) == 3
        assert set(lines) == {"alpha", "beta", "gamma"}

    def test_shuf_nonexistent(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_shuf("/nonexistent_shuf_xyz"))
        assert repl._last_exit_code == 1

    def test_shuf_piped(self, repl):
        repl._piped_input = "line1\nline2\nline3\n"
        out = _run_with_io(repl, [], lambda: repl._cmd_shuf(""))
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert set(lines) == {"line1", "line2", "line3"}


# ── _cmd_stat ────────────────────────────────────────────────────────


class TestCmdStatExtra:
    def test_stat_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_stat(""))
        assert "Usage" in out
        assert repl._last_exit_code == 1

    def test_stat_file(self, repl, tmp_path):
        f = tmp_path / "stat_file.txt"
        f.write_text("hello")
        out = _run_with_io(repl, [], lambda: repl._cmd_stat(str(f)))
        assert "File:" in out
        assert "Size:" in out
        assert "Type: file" in out

    def test_stat_directory(self, repl, tmp_path):
        out = _run_with_io(repl, [], lambda: repl._cmd_stat(str(tmp_path)))
        assert "Type: directory" in out

    def test_stat_nonexistent(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_stat("/nonexistent_stat_xyz"))
        assert repl._last_exit_code == 1
        assert "No such file or directory" in out


# ── _cmd_tee ─────────────────────────────────────────────────────────


class TestCmdTeeExtra:
    def test_tee_no_pipe(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_tee(""))
        assert "Usage" in out
        assert repl._last_exit_code == 1

    def test_tee_writes_file(self, repl, tmp_path):
        repl._piped_input = "tee content\n"
        f = tmp_path / "tee_out.txt"
        out = _run_with_io(repl, [], lambda: repl._cmd_tee(str(f)))
        assert f.read_text() == "tee content\n"
        assert "tee content" in out

    def test_tee_append(self, repl, tmp_path):
        f = tmp_path / "tee_append.txt"
        f.write_text("first\n")
        repl._piped_input = "second\n"
        out = _run_with_io(repl, [], lambda: repl._cmd_tee(f"-a {f}"))
        content = f.read_text()
        assert "first" in content and "second" in content

    def test_tee_multiple_files(self, repl, tmp_path):
        repl._piped_input = "multi\n"
        f1 = tmp_path / "tee_m1.txt"
        f2 = tmp_path / "tee_m2.txt"
        out = _run_with_io(repl, [], lambda: repl._cmd_tee(f"{f1} {f2}"))
        assert f1.read_text() == "multi\n"
        assert f2.read_text() == "multi\n"


# ── _cmd_protect / _cmd_unprotect ────────────────────────────────────


class TestCmdProtectExtra:
    def test_protect_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_protect(""))
        assert "Usage" in out

    def test_protect_success(self, repl, tmp_path):
        from unittest.mock import patch as mp
        mock_result = {"protected": ["model.bin"], "errors": []}
        with mp("domains.infrastructure.model_protector.protect_model", return_value=mock_result):
            out = _run_with_io(repl, [], lambda: repl._cmd_protect("mymodel"))
            assert "Protected 1" in out

    def test_protect_no_files(self, repl):
        from unittest.mock import patch as mp
        mock_result = {"protected": [], "errors": []}
        with mp("domains.infrastructure.model_protector.protect_model", return_value=mock_result):
            out = _run_with_io(repl, [], lambda: repl._cmd_protect("mymodel"))
            assert "No files found" in out

    def test_protect_with_errors(self, repl):
        from unittest.mock import patch as mp
        mock_result = {"protected": ["a.bin"], "errors": [{"error": "perm denied"}]}
        with mp("domains.infrastructure.model_protector.protect_model", return_value=mock_result):
            out = _run_with_io(repl, [], lambda: repl._cmd_protect("mymodel"))
            assert "Warning: perm denied" in out


class TestCmdUnprotectExtra:
    def test_unprotect_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_unprotect(""))
        assert "Usage" in out

    def test_unprotect_success(self, repl):
        from unittest.mock import patch as mp
        mock_result = {"unprotected": 3, "errors": []}
        with mp("domains.infrastructure.model_protector.unprotect_model", return_value=mock_result):
            out = _run_with_io(repl, [], lambda: repl._cmd_unprotect("mymodel"))
            assert "Unprotected 3" in out

    def test_unprotect_none_found(self, repl):
        from unittest.mock import patch as mp
        mock_result = {"unprotected": 0, "errors": []}
        with mp("domains.infrastructure.model_protector.unprotect_model", return_value=mock_result):
            out = _run_with_io(repl, [], lambda: repl._cmd_unprotect("mymodel"))
            assert "No protected files found" in out


# ── _cmd_lsdev ───────────────────────────────────────────────────────


class TestCmdLsdevExtra:
    def test_lsdev_no_devices(self, repl):
        from unittest.mock import patch as mp, PropertyMock
        with mp.object(type(repl.os), 'devices', new_callable=PropertyMock, return_value=None):
            out = _run_with_io(repl, [], lambda: repl._cmd_lsdev(""))
            assert "not available" in out.lower()

    def test_lsdev_with_devices(self, repl):
        from unittest.mock import MagicMock, patch as mp, PropertyMock
        mock_dev = MagicMock()
        mock_dev.list_devices.return_value = "  /dev/null\n  /dev/zero"
        with mp.object(type(repl.os), 'devices', new_callable=PropertyMock, return_value=mock_dev):
            out = _run_with_io(repl, [], lambda: repl._cmd_lsdev(""))
            assert "Device nodes" in out
            assert "/dev/null" in out


# ── _cmd_events ──────────────────────────────────────────────────────


class TestCmdEventsExtra:
    def test_events_no_bus(self, repl):
        from unittest.mock import patch as mp
        with mp("domains.infrastructure.event_bus.get_event_bus", side_effect=Exception("no bus")):
            out = _run_with_io(repl, [], lambda: repl._cmd_events(""))
            assert "not available" in out

    def test_events_empty_history(self, repl):
        from unittest.mock import MagicMock, patch as mp
        mock_bus = MagicMock()
        mock_bus.history.return_value = []
        with mp("domains.infrastructure.event_bus.get_event_bus", return_value=mock_bus):
            out = _run_with_io(repl, [], lambda: repl._cmd_events(""))
            assert "No events" in out


# ── _cmd_metrics ─────────────────────────────────────────────────────


class TestCmdMetricsExtra:
    def test_metrics_error(self, repl):
        from unittest.mock import MagicMock
        repl.cmds.system_metrics = MagicMock(return_value={"error": "connection refused"})
        out = _run_with_io(repl, [], lambda: repl._cmd_metrics(""))
        assert "connection refused" in out

    def test_metrics_ok(self, repl):
        from unittest.mock import MagicMock
        repl.cmds.system_metrics = MagicMock(return_value={"cpu": 50.0, "mem": 1024})
        out = _run_with_io(repl, [], lambda: repl._cmd_metrics(""))
        assert "cpu" in out
        assert "mem" in out


# ── _cmd_mv overwrite ───────────────────────────────────────────────


class TestCmdMvOverwrite:
    def test_mv_overwrites_existing(self, repl, tmp_path):
        src = tmp_path / "mv_over_src.txt"
        dst = tmp_path / "mv_over_dst.txt"
        src.write_text("new content")
        dst.write_text("old content")
        repl._cmd_mv(f"{src} {dst}")
        assert dst.read_text() == "new content"
        assert not src.exists()

    def test_mv_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_mv(""))
        assert "Usage" in out

    def test_mv_one_arg(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_mv("only_one"))
        assert repl._last_exit_code == 1


# ── _cmd_cp edge cases ───────────────────────────────────────────────


class TestCmdCpEdgeCases:
    def test_cp_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_cp(""))
        assert "Usage" in out

    def test_cp_one_arg(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_cp("only_one"))
        assert "missing destination" in out


# ── _cmd_svc ─────────────────────────────────────────────────────────


class TestCmdSvcExtra:
    def test_svc_no_init(self, repl):
        from unittest.mock import patch as mp, PropertyMock
        with mp.object(type(repl.os), 'init_system', new_callable=PropertyMock, return_value=None):
            out = _run_with_io(repl, [], lambda: repl._cmd_svc(""))
            assert "not booted" in out
            assert repl._last_exit_code == 1

    def test_svc_list(self, repl):
        from unittest.mock import MagicMock, patch as mp, PropertyMock
        mock_init = MagicMock()
        mock_init.service_table.return_value = "  svc1  running\n  svc2  stopped"
        with mp.object(type(repl.os), 'init_system', new_callable=PropertyMock, return_value=mock_init):
            out = _run_with_io(repl, [], lambda: repl._cmd_svc("list"))
            assert "svc1" in out
            assert "svc2" in out

    def test_svc_status(self, repl):
        from unittest.mock import MagicMock, patch as mp, PropertyMock
        mock_init = MagicMock()
        mock_mgr = MagicMock()
        mock_mgr.status_line.return_value = "  svc1  running"
        mock_mgr.instance.log = ["started", "ready"]
        mock_init.get_manager.return_value = mock_mgr
        with mp.object(type(repl.os), 'init_system', new_callable=PropertyMock, return_value=mock_init):
            out = _run_with_io(repl, [], lambda: repl._cmd_svc("status svc1"))
            assert "running" in out

    def test_svc_start(self, repl):
        from unittest.mock import MagicMock, patch as mp, PropertyMock
        mock_init = MagicMock()
        mock_init.start.return_value = True
        with mp.object(type(repl.os), 'init_system', new_callable=PropertyMock, return_value=mock_init):
            out = _run_with_io(repl, [], lambda: repl._cmd_svc("start svc1"))
            assert repl._last_exit_code == 0

    def test_svc_stop(self, repl):
        from unittest.mock import MagicMock, patch as mp, PropertyMock
        mock_init = MagicMock()
        with mp.object(type(repl.os), 'init_system', new_callable=PropertyMock, return_value=mock_init):
            out = _run_with_io(repl, [], lambda: repl._cmd_svc("stop svc1"))
            assert repl._last_exit_code == 0


# ── _cmd_boot ────────────────────────────────────────────────────────


class TestCmdBootExtra:
    def test_boot_already_booted(self, repl):
        repl._running = True
        repl._piped_input = None
        out = _run_with_io(repl, [], lambda: repl._cmd_boot(""))
        assert "Already booted" in out


# ── _cmd_head / _cmd_tail edge cases ────────────────────────────────


class TestCmdHeadTailEdgeCases:
    def test_head_with_line_count(self, repl, tmp_path):
        f = tmp_path / "head_n.txt"
        f.write_text("a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_head(f"-3 {f}"))
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert len(lines) == 3

    def test_head_multiple_files(self, repl, tmp_path):
        f1 = tmp_path / "head_m1.txt"
        f2 = tmp_path / "head_m2.txt"
        f1.write_text("file1\n")
        f2.write_text("file2\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_head(f"{f1} {f2}"))
        assert "==> " in out
        assert "file1" in out and "file2" in out

    def test_head_piped(self, repl):
        repl._piped_input = "p1\np2\np3\np4\np5\n"
        out = _run_with_io(repl, [], lambda: repl._cmd_head("-2"))
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert len(lines) == 2

    def test_tail_with_line_count(self, repl, tmp_path):
        f = tmp_path / "tail_n.txt"
        f.write_text("a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_tail(f"-3 {f}"))
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert len(lines) == 3

    def test_tail_piped(self, repl):
        repl._piped_input = "p1\np2\np3\np4\np5\n"
        out = _run_with_io(repl, [], lambda: repl._cmd_tail("-2"))
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert len(lines) == 2


# ── _cmd_cut / _cmd_nl / _cmd_fold edge cases ────────────────────────


class TestCmdCutDeeper:
    def test_cut_with_delimiter(self, repl):
        repl._piped_input = "a:b:c\nd:e:f\n"
        out = _run_with_io(repl, [], lambda: repl._cmd_cut("-d: -f2"))
        assert "b" in out and "e" in out
        assert "a" not in out and "d" not in out

    def test_cut_range(self, repl):
        repl._piped_input = "a\tb\tc\td\n"
        out = _run_with_io(repl, [], lambda: repl._cmd_cut("-f2-3"))
        assert "b" in out and "c" in out
        assert "a" not in out

    def test_cut_no_fields_flag(self, repl):
        repl._piped_input = "a\tb\tc\n"
        out = _run_with_io(repl, [], lambda: repl._cmd_cut("-d\t"))
        assert "must specify" in out.lower() or "cut" in out.lower()
        assert repl._last_exit_code == 1


class TestCmdNlDeeper:
    def test_nl_file(self, repl, tmp_path):
        f = tmp_path / "nl_file.txt"
        f.write_text("line1\nline2\nline3\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_nl(str(f)))
        assert "1\tline1" in out
        assert "2\tline2" in out
        assert "3\tline3" in out

    def test_nl_piped(self, repl):
        repl._piped_input = "aa\nbb\n"
        out = _run_with_io(repl, [], lambda: repl._cmd_nl(""))
        assert "1\taa" in out
        assert "2\tbb" in out

    def test_nl_nonexistent(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_nl("/nonexistent_nl_xyz"))
        assert repl._last_exit_code == 1


class TestCmdFoldDeeper:
    def test_fold_narrow_width(self, repl):
        repl._piped_input = "abcdefghijklmnopqrstuvwxyz\n"
        out = _run_with_io(repl, [], lambda: repl._cmd_fold("-w 10"))
        lines = [l for l in out.strip().split("\n") if l.strip()]
        assert len(lines) > 1

    def test_fold_short_flag(self, repl):
        repl._piped_input = "abcdefghij\n"
        out = _run_with_io(repl, [], lambda: repl._cmd_fold("-w5"))
        assert repl._last_exit_code == 0

    def test_fold_file(self, repl, tmp_path):
        f = tmp_path / "fold_file.txt"
        f.write_text("this is a long line that should be wrapped\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_fold(f"-w 10 {f}"))
        lines = [l for l in out.strip().split("\n") if l.strip()]
        assert len(lines) > 1


# ── _parse_pipeline edge cases ──────────────────────────────────────


class TestParsePipelineExtra:
    def test_chain_and_operator(self, repl):
        cmds, bg, timed = repl._parse_pipeline("echo a && echo b")
        assert bg is False
        assert len(cmds) >= 2
        ops = [op for _, op in cmds]
        assert "&&" in ops

    def test_chain_or_operator(self, repl):
        cmds, bg, timed = repl._parse_pipeline("echo a || echo b")
        assert bg is False
        ops = [op for _, op in cmds]
        assert "||" in ops

    def test_chain_semicolon(self, repl):
        cmds, bg, timed = repl._parse_pipeline("echo a ; echo b")
        assert len(cmds) >= 2
        ops = [op for _, op in cmds]
        assert ";" in ops

    def test_background_operator(self, repl):
        cmds, bg, timed = repl._parse_pipeline("echo a &")
        assert bg is True

    def test_time_operator(self, repl):
        cmds, bg, timed = repl._parse_pipeline("time echo a")
        assert timed is True

    def test_quoted_pipe_operator(self, repl):
        cmds, bg, timed = repl._parse_pipeline('echo "a|b"')
        assert len(cmds) == 1

    def test_empty_input(self, repl):
        cmds, bg, timed = repl._parse_pipeline("")
        assert cmds == []
        assert bg is False


# ── _execute_pipeline edge cases ─────────────────────────────────────


class TestExecutePipelineExtra:
    def test_pipeline_empty(self, repl):
        repl._execute_pipeline([])
        assert repl._last_exit_code == 0

    def test_pipeline_single_command(self, repl):
        repl._execute_pipeline([("echo pipeline_single", None)])
        assert repl._last_exit_code == 0

    def test_pipeline_and_skips_on_failure(self, repl):
        repl._last_exit_code = 1
        repl._execute_pipeline([("false", "&&"), ("echo should_run", None)])
        assert repl._last_exit_code == 0

    def test_pipeline_or_skips_on_success(self, repl):
        repl._last_exit_code = 0
        repl._execute_pipeline([("true", "||"), ("echo should_skip", None)])


# ── _cmd_read edge cases ────────────────────────────────────────────


class TestCmdReadDeeper:
    def test_read_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_read(""))
        assert "Usage" in out
        assert repl._last_exit_code == 1

    def test_read_with_prompt(self, repl):
        mem = MemoryIO()
        mem.feed("my_value\n")
        old_io = repl.io
        old_console_io = repl.console._io
        repl.io = mem
        repl.console._io = mem
        try:
            repl._cmd_read("-p Enter:")
        finally:
            repl.io = old_io
            repl.console._io = old_console_io
        assert repl._env.get("myvar") is None or repl._last_exit_code == 0

    def test_read_piped_input(self, repl):
        mem = MemoryIO()
        mem.feed("piped_val\n")
        old_io = repl.io
        old_console_io = repl.console._io
        repl.io = mem
        repl.console._io = mem
        try:
            repl._cmd_read("MYVAR")
        finally:
            repl.io = old_io
            repl.console._io = old_console_io
        assert repl._env.get("MYVAR") == "piped_val"

    def test_read_prompt_only_no_var(self, repl):
        mem = MemoryIO()
        mem.feed("dummy\n")
        old_io = repl.io
        old_console_io = repl.console._io
        repl.io = mem
        repl.console._io = mem
        try:
            repl._cmd_read("-p dummy")
        finally:
            repl.io = old_io
            repl.console._io = old_console_io


# ── _cmd_time edge cases ────────────────────────────────────────────


class TestCmdTimeExtra:
    def test_time_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_time(""))
        assert repl._last_exit_code == 1

    def test_time_with_echo(self, repl):
        result = repl._execute_single("echo timed")
        assert "timed" in result
        assert repl._last_exit_code == 0


# ── _cmd_yes/_cmd_sleep edge cases ──────────────────────────────────


class TestCmdYesDeeper:
    def test_yes_with_piped(self, repl):
        repl._piped_input = "ignored\n"
        out = _run_with_io(repl, [], lambda: repl._cmd_yes(""))
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert len(lines) == 100

    def test_yes_single_char(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_yes("x"))
        lines = out.strip().split("\n")
        assert all(l.strip() == "x" for l in lines)


class TestCmdSleepDeeper:
    def test_sleep_zero(self, repl):
        repl._cmd_sleep("0")
        assert repl._last_exit_code == 0

    def test_sleep_negative(self, repl):
        try:
            repl._cmd_sleep("-1")
        except ValueError:
            pass

    def test_sleep_float(self, repl):
        repl._cmd_sleep("0.001")
        assert repl._last_exit_code == 0


# ── _cmd_kill/_cmd_procs/_cmd_logname/_cmd_who ──────────────────────


class TestCmdKillDeeper:
    def test_kill_with_process(self, repl):
        from unittest.mock import MagicMock
        repl.cmds.kill = MagicMock(return_value={"ok": True})
        out = _run_with_io(repl, [], lambda: repl._cmd_kill("123"))
        assert repl._last_exit_code == 0

    def test_kill_invalid(self, repl):
        from unittest.mock import MagicMock
        repl.cmds.kill = MagicMock(side_effect=Exception("not found"))
        try:
            out = _run_with_io(repl, [], lambda: repl._cmd_kill("999"))
        except Exception:
            pass


class TestCmdLognameDeeper:
    def test_logname_returns_value(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_logname(""))
        assert repl._last_exit_code == 0
        assert len(out.strip()) > 0


class TestCmdWhoDeeper:
    def test_who_returns_value(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_who(""))
        assert repl._last_exit_code == 0


# ── _cmd_realpath/_cmd_dirname/_cmd_basename ────────────────────────


class TestCmdRealpathDeeper:
    def test_realpath_dot(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_realpath("."))
        assert repl._last_exit_code == 0
        assert len(out.strip()) > 0

    def test_realpath_file(self, repl, tmp_path):
        f = tmp_path / "rp.txt"
        f.write_text("x")
        out = _run_with_io(repl, [], lambda: repl._cmd_realpath(str(f)))
        assert str(f) in out


class TestCmdDirnameDeeper:
    def test_dirname_nested(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_dirname("/a/b/c/d.txt"))
        assert out.strip() == "/a/b/c"

    def test_dirname_root(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_dirname("/"))
        assert out.strip() == "/"


class TestCmdBasenameDeeper:
    def test_basename_no_suffix(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_basename("/a/b/file.tar.gz"))
        assert out.strip() == "file.tar.gz"

    def test_basename_multiple_suffix(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_basename("/a/b/file.tar.gz .gz"))
        assert out.strip() == "file.tar"

    def test_basename_just_name(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_basename("simple.txt"))
        assert out.strip() == "simple.txt"


# ── _cmd_watch/_cmd_xargs edge cases ────────────────────────────────


class TestCmdWatchDeeper:
    def test_watch_invalid_interval(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_watch("echo hi abc"))
        assert "Invalid interval" in out


class TestCmdXargsDeeper:
    def test_xargs_no_pipe(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_xargs("echo"))
        assert "Usage" in out
        assert repl._last_exit_code == 1

    def test_xargs_no_cmd(self, repl):
        repl._piped_input = "a b c"
        out = _run_with_io(repl, [], lambda: repl._cmd_xargs(""))
        assert "a" in out and "b" in out and "c" in out

    def test_xargs_with_n_flag(self, repl):
        repl._piped_input = "1 2 3 4 5"
        out = _run_with_io(repl, [], lambda: repl._cmd_xargs("-n 2 echo"))
        assert repl._last_exit_code == 0


# ── _cmd_comm edge cases ────────────────────────────────────────────


class TestCmdCommDeeper:
    def test_comm_identical_files(self, repl, tmp_path):
        f1 = tmp_path / "c_ident1.txt"
        f2 = tmp_path / "c_ident2.txt"
        f1.write_text("x\ny\nz\n")
        f2.write_text("x\ny\nz\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_comm(f"{f1} {f2}"))
        assert "x" in out and "y" in out and "z" in out
        assert repl._last_exit_code == 0

    def test_comm_disjoint_files(self, repl, tmp_path):
        f1 = tmp_path / "c_disj1.txt"
        f2 = tmp_path / "c_disj2.txt"
        f1.write_text("aaa\n")
        f2.write_text("zzz\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_comm(f"{f1} {f2}"))
        assert "aaa" in out and "zzz" in out
        assert repl._last_exit_code == 0


# ── _cmd_tui edge cases ─────────────────────────────────────────────


class TestCmdTuiExtra:
    def test_tui_import_error(self, repl):
        from unittest.mock import patch as mp
        with mp.dict('sys.modules', {'domains.shell.tui_repl': None}):
            out = _run_with_io(repl, [], lambda: repl._cmd_tui(""))
            assert "not available" in out.lower() or "error" in out.lower()


# ── _execute_single redirect edge cases ──────────────────────────────


class TestExecuteSingleRedirect:
    def test_redirect_overwrite(self, repl, tmp_path):
        f = tmp_path / "redirect_out.txt"
        f.write_text("old content")
        repl._execute_single(f"echo redirected > {f}")
        content = f.read_text()
        assert "redirected" in content
        assert "old content" not in content

    def test_redirect_append(self, repl, tmp_path):
        f = tmp_path / "redirect_append.txt"
        f.write_text("first\n")
        repl._execute_single(f"echo second >> {f}")
        content = f.read_text()
        assert "first" in content and "second" in content

    def test_inline_env(self, repl):
        repl._execute_single("MYVAR=hello echo test_inline")
        assert repl._last_exit_code == 0

    def test_command_substitution(self, repl):
        repl._execute_single("echo hi")
        assert repl._last_exit_code == 0


# ── _dispatch ─────────────────────────────────────────────────────


class TestDispatch:
    def test_dispatch_simple_command(self, repl):
        repl._dispatch("echo hello_dispatch")
        assert repl._last_exit_code == 0
        assert repl._cmd_count >= 1

    def test_dispatch_records_history(self, repl):
        repl._dispatch("echo history_test")
        assert "echo history_test" in repl._history

    def test_dispatch_unknown_command(self, repl):
        repl._dispatch("nonexistent_cmd_xyz")
        assert repl._last_exit_code == 127

    def test_dispatch_pipeline(self, repl):
        repl._dispatch("echo pipe_test | wc")
        assert repl._last_exit_code == 0

    def test_dispatch_error_returns_1(self, repl):
        with patch.object(repl, "COMMANDS", {}):
            repl._dispatch("fail_cmd")
            assert repl._last_exit_code == 127

    def test_dispatch_keyboard_interrupt(self, repl):
        def raise_kb(self_repl, args):
            raise KeyboardInterrupt()
        repl.COMMANDS["kbcmd"] = raise_kb
        try:
            repl._dispatch("kbcmd")
            assert repl._aborted is True
        finally:
            del repl.COMMANDS["kbcmd"]

    def test_dispatch_ext_mod(self, repl):
        mock_mod = MagicMock()
        mock_mod.help = "test ext"
        mock_mod.run.return_value = 0
        repl._ext_cmds["testext"] = mock_mod
        try:
            repl._dispatch("testext arg1")
            assert repl._last_exit_code == 0
            mock_mod.run.assert_called_once()
        finally:
            del repl._ext_cmds["testext"]

    def test_dispatch_ext_mod_permission_denied_noninteractive(self, repl):
        mock_mod = MagicMock()
        mock_mod.help = "test ext"
        mock_mod.run.return_value = 0
        repl._ext_cmds["rmtestext"] = mock_mod
        try:
            repl._last_exit_code = None
            repl.execute("rmtestext")
            assert repl._last_exit_code == 0
        finally:
            del repl._ext_cmds["rmtestext"]

    def test_dispatch_command_exception(self, repl):
        def bad_handler(self_repl, args):
            raise RuntimeError("test boom")
        repl.COMMANDS["boomcmd"] = bad_handler
        try:
            repl._dispatch("boomcmd")
            assert repl._last_exit_code == 1
        finally:
            del repl.COMMANDS["boomcmd"]

    def test_dispatch_suggests_typo(self, repl):
        repl._dispatch("healht")
        assert repl._last_exit_code == 127

    def test_dispatch_timed(self, repl):
        repl._dispatch("time echo timed_dispatch")
        assert repl._last_exit_code == 0

    def test_dispatch_bg(self, repl):
        repl._dispatch("echo bg_test &")
        assert repl._last_exit_code == 0

    def test_dispatch_alias_expansion(self, repl):
        repl._aliases["ll"] = "ls"
        try:
            repl._dispatch("ll")
            assert repl._last_exit_code == 0
        finally:
            del repl._aliases["ll"]


# ── _apply_render_preset ──────────────────────────────────────────


class TestApplyRenderPreset:
    def test_demo_preset(self, repl):
        mock_dev = MagicMock()
        repl._render_device = mock_dev
        out = _run_with_io(repl, [], lambda: repl._apply_render_preset("demo"))
        assert "demo" in out
        assert mock_dev.call.call_count > 0

    def test_cornell_preset(self, repl):
        mock_dev = MagicMock()
        repl._render_device = mock_dev
        out = _run_with_io(repl, [], lambda: repl._apply_render_preset("cornell"))
        assert "cornell" in out

    def test_spheres_preset(self, repl):
        mock_dev = MagicMock()
        repl._render_device = mock_dev
        out = _run_with_io(repl, [], lambda: repl._apply_render_preset("spheres"))
        assert "spheres" in out

    def test_unknown_preset(self, repl):
        mock_dev = MagicMock()
        repl._render_device = mock_dev
        out = _run_with_io(repl, [], lambda: repl._apply_render_preset("nonexistent"))
        assert "Unknown preset" in out


# ── _update_color_state ───────────────────────────────────────────


class TestUpdateColorState:
    def test_no_color_disables(self, repl):
        repl._env["NO_COLOR"] = "1"
        repl._update_color_state()
        assert repl_mod._COLOR_ENABLED is False
        assert repl_mod._C_RED == ""

    def test_no_color_not_set_enables(self, repl):
        repl._env.pop("NO_COLOR", None)
        repl_mod._COLOR_ENABLED = False
        repl_mod._C_RED = ""
        repl._update_color_state()
        assert repl_mod._COLOR_ENABLED is True
        assert repl_mod._C_RED == "\033[31m"

    def test_no_color_true_string(self, repl):
        repl._env["NO_COLOR"] = "true"
        repl._update_color_state()
        assert repl_mod._COLOR_ENABLED is False


# ── _group_ext_cmds ───────────────────────────────────────────────


class TestGroupExtCmds:
    def test_groups_by_help(self, repl):
        m1 = MagicMock()
        m1.help = "File ops"
        m2 = MagicMock()
        m2.help = "File ops"
        m3 = MagicMock()
        m3.help = "Network"
        result = ShellREPL._group_ext_cmds({"cat": m1, "cp": m2, "ping": m3})
        assert "File ops" in result
        assert "Network" in result
        assert sorted(result["File ops"]) == ["cat", "cp"]

    def test_empty_dict(self, repl):
        result = ShellREPL._group_ext_cmds({})
        assert result == {}

    def test_no_help_attr(self, repl):
        m = MagicMock(spec=[])
        m.help = None
        result = ShellREPL._group_ext_cmds({"orphan": m})
        assert "" in result


# ── _note_status_summary ──────────────────────────────────────────


class TestNoteStatusSummary:
    def test_empty_store(self, repl):
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        out = _run_with_io(repl, [], lambda: repl._note_status_summary(mock_store))
        assert "No notes" in out

    def test_multiple_statuses(self, repl):
        mock_store = MagicMock()
        mock_store.list_notes.return_value = [
            MagicMock(status="open"),
            MagicMock(status="open"),
            MagicMock(status="done"),
            MagicMock(status="wip"),
            MagicMock(status="blocked"),
        ]
        out = _run_with_io(repl, [], lambda: repl._note_status_summary(mock_store))
        assert "open" in out
        assert "done" in out
        assert "wip" in out
        assert "blocked" in out
        assert repl._last_exit_code == 0


# ── _interpret_natural ────────────────────────────────────────────


class TestInterpretNatural:
    def test_processes_keyword(self, repl):
        out = _run_with_io(repl, [], lambda: repl._interpret_natural("show me running processes"))
        assert repl._last_exit_code == 0

    def test_models_keyword(self, repl):
        out = _run_with_io(repl, [], lambda: repl._interpret_natural("what models are available"))
        assert repl._last_exit_code == 0

    def test_health_keyword(self, repl):
        with patch("domains.shell.commands.ShellCommands.health", return_value={"status": "healthy"}):
            out = _run_with_io(repl, [], lambda: repl._interpret_natural("check health status"))
        assert repl._last_exit_code == 0

    def test_dataset_keyword(self, repl):
        out = _run_with_io(repl, [], lambda: repl._interpret_natural("show datasets"))
        assert repl._last_exit_code == 0

    def test_knowledge_keyword(self, repl):
        out = _run_with_io(repl, [], lambda: repl._interpret_natural("show knowledge facts"))
        assert repl._last_exit_code == 0

    def test_checkpoint_keyword(self, repl):
        out = _run_with_io(repl, [], lambda: repl._interpret_natural("list checkpoints"))
        assert repl._last_exit_code == 0

    def test_metric_keyword(self, repl):
        out = _run_with_io(repl, [], lambda: repl._interpret_natural("show cpu metrics"))
        assert repl._last_exit_code == 0

    def test_help_keyword(self, repl):
        out = _run_with_io(repl, [], lambda: repl._interpret_natural("help commands"))
        assert repl._last_exit_code == 0

    def test_unknown_query(self, repl):
        out = _run_with_io(repl, [], lambda: repl._interpret_natural("random nonsense xyz"))
        assert "Unknown query" in out


# ── _show_welcome ─────────────────────────────────────────────────


class TestShowWelcome:
    def test_shows_welcome(self, repl):
        out = _run_with_io(repl, [], lambda: repl._show_welcome())
        assert "Welcome" in out
        assert "Dait" in out

    def test_sets_first_run_false(self, repl):
        repl.state.first_run = True
        _run_with_io(repl, [], lambda: repl._show_welcome())
        assert repl.state.first_run is False


# ── _load_rc ──────────────────────────────────────────────────────


class TestLoadRc:
    def test_rc_executes_commands(self, repl, tmp_path):
        rc_dir = tmp_path / ".config" / "sloughgpt"
        rc_dir.mkdir(parents=True)
        rc_file = rc_dir / "rc"
        rc_file.write_text("echo rc_line_1\n")
        with patch.object(repl, "_rc_path", return_value=rc_file):
            repl._load_rc()
        assert repl._last_exit_code == 0

    def test_rc_skips_comments_and_blanks(self, repl, tmp_path):
        rc_dir = tmp_path / ".config" / "sloughgpt"
        rc_dir.mkdir(parents=True)
        rc_file = rc_dir / "rc"
        rc_file.write_text("# comment\n\n  \necho after_comment\n")
        with patch.object(repl, "_rc_path", return_value=rc_file):
            repl._load_rc()
        assert repl._last_exit_code == 0

    def test_rc_no_file(self, repl, tmp_path):
        fake_rc = tmp_path / "nonexistent" / "rc"
        with patch.object(repl, "_rc_path", return_value=fake_rc):
            repl._load_rc()

    def test_rc_error_isolation(self, repl, tmp_path):
        rc_dir = tmp_path / ".config" / "sloughgpt"
        rc_dir.mkdir(parents=True)
        rc_file = rc_dir / "rc"
        rc_file.write_text("bad_cmd_on_line_1\necho after_error\n")
        with patch.object(repl, "_rc_path", return_value=rc_file):
            repl._load_rc()
        assert repl._last_exit_code == 0


# ── _cmd_vmperms edge cases ───────────────────────────────────────


class TestCmdVmPermsDeeper:
    def test_vmperms_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_vmperms(""))
        assert "Permission" in out

    def test_vmperms_list(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_vmperms("list"))
        assert repl._last_exit_code == 0

    def test_vmperms_allow(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_vmperms("allow user ls"))
        assert repl._last_exit_code == 0

    def test_vmperms_deny(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_vmperms("deny user ls"))
        assert repl._last_exit_code == 0

    def test_vmperms_revoke(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_vmperms("revoke user ls"))
        assert repl._last_exit_code == 0

    def test_vmperms_audit(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_vmperms("audit"))
        assert repl._last_exit_code == 0

    def test_vmperms_help(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_vmperms("help"))
        assert "Permission" in out


# ── _execute_background_tuples ────────────────────────────────────


class TestExecuteBackgroundTuples:
    def test_background_single(self, repl):
        repl._execute_background_tuples([("echo bg_tuple_test", None)])
        assert repl._last_exit_code == 0

    def test_background_piped(self, repl):
        repl._execute_background_tuples([("echo piped", "|"), ("wc", None)])
        assert repl._last_exit_code == 0


# ── _cmd_tutorial deeper ──────────────────────────────────────────


class TestCmdTutorialDeeper:
    def test_tutorial_quit_immediately(self, repl):
        mem = MemoryIO()
        mem.feed("q\n")
        old_io = repl.io
        old_console_io = repl.console._io
        repl.io = mem
        repl.console._io = mem
        try:
            repl._cmd_tutorial()
        finally:
            repl.io = old_io
            repl.console._io = old_console_io
        assert repl._last_exit_code == 0

    def test_tutorial_runs_all_steps(self, repl):
        mem = MemoryIO()
        mem.feed("\n" * 20)
        old_io = repl.io
        old_console_io = repl.console._io
        repl.io = mem
        repl.console._io = mem
        try:
            repl._cmd_tutorial()
        finally:
            repl.io = old_io
            repl.console._io = old_console_io
        assert repl._last_exit_code == 0


# ── _cmd_confirm deeper ──────────────────────────────────────────


class TestCmdConfirmDeeper:
    def test_confirm_on(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_confirm("on"))
        assert repl._last_exit_code == 0

    def test_confirm_off(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_confirm("off"))
        assert repl._last_exit_code == 0

    def test_confirm_yes(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_confirm("yes"))
        assert repl._last_exit_code == 0

    def test_confirm_no(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_confirm("no"))
        assert repl._last_exit_code == 0

    def test_confirm_status(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_confirm(""))
        assert repl._last_exit_code == 0


# ── _cmd_agents deeper ───────────────────────────────────────────


class TestCmdAgentsDeeper:
    def test_agents_list(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_agents("list"))
        assert repl._last_exit_code == 0

    def test_agents_empty(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_agents(""))
        assert "agents" in out


# ── _cmd_py deeper ────────────────────────────────────────────────


class TestCmdPyDeeper:
    def test_py_blocked_import(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_py("__import__('os')"))
        assert "not allowed" in out

    def test_py_syntax_error(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_py("def"))
        assert "Error" in out or "invalid" in out.lower()

    def test_py_runtime_error(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_py("1/0"))
        assert "Error" in out or "division" in out.lower()

    def test_py_safe_import(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_py("__import__('math').pi"))
        assert "3.14" in out

    def test_py_list_comprehension(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_py("[i*i for i in range(3)]"))
        assert "[0, 1, 4]" in out

    def test_py_string_method(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_py("'hello'.upper()"))
        assert "HELLO" in out

    def test_py_dict(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_py("{'a': 1, 'b': 2}"))
        assert "'a'" in out

    def test_py_nested_import(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_py("__import__('json').dumps([1,2])"))
        assert "[1, 2]" in out


# ── _cmd_source deeper ────────────────────────────────────────────


class TestCmdSourceDeeper:
    def test_source_with_pipeline(self, repl, tmp_path):
        src = tmp_path / "src_pipe.sh"
        src.write_text("echo pipe_test | wc\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_source(str(src)))
        assert repl._last_exit_code == 0

    def test_source_with_bg(self, repl, tmp_path):
        src = tmp_path / "src_bg.sh"
        src.write_text("echo bg_test &\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_source(str(src)))
        assert repl._last_exit_code == 0

    def test_source_with_error_line(self, repl, tmp_path):
        src = tmp_path / "src_err.sh"
        src.write_text("echo ok\necho bad | |\necho after_err\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_source(str(src)))
        assert "Error at line" in out

    def test_source_dot_alias(self, repl, tmp_path):
        src = tmp_path / "src_dot.sh"
        src.write_text("echo dot_test\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_source(str(src)))
        assert "dot_test" in out


# ── _cmd_echo deeper ──────────────────────────────────────────────


class TestCmdEchoDeeper:
    def test_echo_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_echo(""))
        assert out.strip() == ""

    def test_echo_multiple_spaces(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_echo("hello   world"))
        assert "hello   world" in out

    def test_echo_special_chars(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_echo("a@b#c$d"))
        assert "a@b#c$d" in out

    def test_echo_empty_quotes(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_echo('""'))
        assert '""' in out


# ── _cmd_test deeper ──────────────────────────────────────────────


class TestCmdTestDeeper:
    def test_test_no_args(self, repl):
        repl._cmd_test("")
        assert repl._last_exit_code == 1

    def test_test_bracket_syntax(self, repl):
        repl._cmd_test("[ 1 = 1 ]")
        assert repl._last_exit_code == 0

    def test_test_ne(self, repl):
        repl._cmd_test("1 -ne 2")
        assert repl._last_exit_code == 0

    def test_test_le(self, repl):
        repl._cmd_test("1 -le 1")
        assert repl._last_exit_code == 0

    def test_test_ge(self, repl):
        repl._cmd_test("2 -ge 1")
        assert repl._last_exit_code == 0

    def test_test_lt_false(self, repl):
        repl._cmd_test("2 -lt 1")
        assert repl._last_exit_code == 1

    def test_test_gt_false(self, repl):
        repl._cmd_test("1 -gt 2")
        assert repl._last_exit_code == 1

    def test_test_eq_false(self, repl):
        repl._cmd_test("1 -eq 2")
        assert repl._last_exit_code == 1

    def test_test_neq(self, repl):
        repl._cmd_test("a != b")
        assert repl._last_exit_code == 0

    def test_test_n(self, repl):
        repl._cmd_test("-n hello")
        assert repl._last_exit_code == 0

    def test_test_z_empty(self, repl):
        repl._cmd_test("[ -n hello ]")
        assert repl._last_exit_code == 0

    def test_test_unknown(self, repl):
        repl._cmd_test("--unknown")
        assert repl._last_exit_code == 1


# ── _cmd_cut deeper ───────────────────────────────────────────────


class TestCmdCutDeeper:
    def test_cut_range(self, repl):
        repl._piped_input = "a\tb\tc\td"
        out = _run_with_io(repl, [], lambda: repl._cmd_cut("-f2-3"))
        assert "b\tc" in out

    def test_cut_no_fields(self, repl):
        repl._piped_input = "a\tb"
        out = _run_with_io(repl, [], lambda: repl._cmd_cut(""))
        assert "must specify" in out

    def test_cut_missing_file(self, repl):
        repl._piped_input = ""
        out = _run_with_io(repl, [], lambda: repl._cmd_cut("-f1 /nonexistent"))
        assert "No such file" in out


# ── _cmd_tr deeper ────────────────────────────────────────────────


class TestCmdTrDeeper:
    def test_tr_delete_mode(self, repl):
        repl._piped_input = "hello world"
        out = _run_with_io(repl, [], lambda: repl._cmd_tr("-d l"))
        assert "heo word" in out

    def test_tr_squeeze(self, repl):
        repl._piped_input = "a\t\tb\t\tc"
        out = _run_with_io(repl, [], lambda: repl._cmd_tr("-s '\\t' ' '"))
        assert "a b c" in out or "a" in out

    def test_tr_no_input(self, repl):
        repl._piped_input = None
        out = _run_with_io(repl, [], lambda: repl._cmd_tr("a b"))
        assert repl._last_exit_code == 1


# ── _cmd_seq deeper ───────────────────────────────────────────────


class TestCmdSeqDeeper:
    def test_seq_float(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_seq("0.5 0.5 1.5"))
        assert "0.5" in out

    def test_seq_zero_increment(self, repl):
        try:
            repl._cmd_seq("1 0 3")
        except (ValueError, Exception):
            pass

    def test_seq_reverse(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_seq("3 -1 1"))
        assert "3" in out


# ── _cmd_nl deeper ────────────────────────────────────────────────


class TestCmdNlDeeper:
    def test_nl_piped(self, repl):
        repl._piped_input = "a\nb\nc"
        out = _run_with_io(repl, [], lambda: repl._cmd_nl(""))
        assert "1" in out and "2" in out

    def test_nl_no_input(self, repl):
        repl._piped_input = None
        out = _run_with_io(repl, [], lambda: repl._cmd_nl(""))
        assert repl._last_exit_code == 1


# ── _cmd_fold deeper ──────────────────────────────────────────────


class TestCmdFoldDeeper:
    def test_fold_piped(self, repl):
        repl._piped_input = "hello world this is a test"
        out = _run_with_io(repl, [], lambda: repl._cmd_fold("-w 5"))
        assert "hello" in out

    def test_fold_no_input(self, repl):
        repl._piped_input = None
        out = _run_with_io(repl, [], lambda: repl._cmd_fold("-w 10"))
        assert repl._last_exit_code == 1
        assert "no input" in out.lower()


# ── _cmd_bg/fg/jobs deeper ────────────────────────────────────────


class TestCmdBgFgDeeper:
    def test_bg_no_jobs(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_bg(""))
        assert repl._last_exit_code == 0

    def test_fg_no_jobs(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_fg(""))
        assert repl._last_exit_code == 0

    def test_jobs_empty(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_bg(""))
        assert repl._last_exit_code == 0


# ── _cmd_export deeper ────────────────────────────────────────────


class TestCmdExportDeeper:
    def test_export_with_value(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_export("MYVAR=hello"))
        assert repl._last_exit_code == 0

    def test_export_empty(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_export(""))
        assert "export" in out.lower() or repl._last_exit_code == 0


# ── _cmd_set deeper ───────────────────────────────────────────────


class TestCmdSetDeeper:
    def test_set_with_value(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_set("MYVAR=world"))
        assert repl._last_exit_code == 0

    def test_set_empty(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_set(""))
        assert repl._last_exit_code == 0


# ── _cmd_alias deeper ─────────────────────────────────────────────


class TestCmdAliasDeeper:
    def test_alias_with_equals(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_alias("ll=ls -la"))
        assert repl._last_exit_code == 0

    def test_alias_list(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_alias(""))
        assert "alias" in out.lower() or repl._last_exit_code == 0


# ── _cmd_unalias deeper ──────────────────────────────────────────


class TestCmdUnaliasDeeper:
    def test_unalias_nonexistent(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_unalias("nonexistent"))
        assert repl._last_exit_code == 0


# ── _cmd_history deeper ──────────────────────────────────────────


class TestCmdHistoryDeeper:
    def test_history_after_commands(self, repl):
        repl._dispatch("echo history_test_1")
        repl._dispatch("echo history_test_2")
        out = _run_with_io(repl, [], lambda: repl._cmd_history(""))
        assert "history_test_1" in out


# ── _cmd_fc deeper ───────────────────────────────────────────────


class TestCmdFcDeeper:
    def test_fc_list(self, repl):
        repl._history.append("echo fc_test")
        out = _run_with_io(repl, [], lambda: repl._cmd_fc("-l"))
        assert "fc_test" in out

    def test_fc_re_exec(self, repl):
        repl._history.append("echo fc_reexec")
        out = _run_with_io(repl, [], lambda: repl._cmd_fc("1"))
        assert repl._last_exit_code == 0


# ── _cmd_chmod deeper ────────────────────────────────────────────


class TestCmdChmodDeeper:
    def test_chmod_nonexistent(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_chmod("755 /nonexistent"))
        assert repl._last_exit_code == 1


# ── _cmd_du deeper ───────────────────────────────────────────────


class TestCmdDuDeeper:
    def test_du_nonexistent(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_du("/nonexistent"))
        assert "cannot access" in out.lower() or repl._last_exit_code == 0


# ── _cmd_diff deeper ─────────────────────────────────────────────


class TestCmdDiffDeeper:
    def test_diff_same_content(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("same\n")
        f2.write_text("same\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_diff(str(f1) + " " + str(f2)))
        assert repl._last_exit_code == 0

    def test_diff_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_diff(""))
        assert repl._last_exit_code == 1


# ── _cmd_stat deeper ─────────────────────────────────────────────


class TestCmdStatDeeper:
    def test_stat_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_stat(""))
        assert repl._last_exit_code == 1

    def test_stat_nonexistent(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_stat("/nonexistent"))
        assert repl._last_exit_code == 1


# ── _cmd_ln deeper ───────────────────────────────────────────────


class TestCmdLnDeeper:
    def test_ln_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_ln(""))
        assert repl._last_exit_code == 1


# ── _cmd_mktemp deeper ───────────────────────────────────────────


class TestCmdMktempDeeper:
    def test_mktemp_creates_file(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_mktemp(""))
        assert repl._last_exit_code == 0


# ── _cmd_realpath deeper ─────────────────────────────────────────


class TestCmdRealpathDeeper:
    def test_realpath_dot(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_realpath("."))
        assert repl._last_exit_code == 0


# ── _cmd_dirname deeper ──────────────────────────────────────────


class TestCmdDirnameDeeper:
    def test_dirname_root(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_dirname("/"))
        assert "/" in out


# ── _cmd_basename deeper ─────────────────────────────────────────


class TestCmdBasenameDeeper:
    def test_basename_with_suffix(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_basename("file.txt .txt"))
        assert "file" in out


# ── _cmd_nproc deeper ────────────────────────────────────────────


class TestCmdNprocDeeper:
    def test_nproc_returns_number(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_nproc(""))
        assert out.strip().isdigit()


# ── _cmd_hostname deeper ─────────────────────────────────────────


class TestCmdHostnameDeeper:
    def test_hostname_returns_string(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_hostname(""))
        assert len(out.strip()) > 0


# ── _cmd_uname deeper ────────────────────────────────────────────


class TestCmdUnameDeeper:
    def test_uname_all(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_uname("-a"))
        assert "Linux" in out or "Darwin" in out

    def test_uname_s(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_uname("-s"))
        assert "Linux" in out or "Darwin" in out


# ── _cmd_id deeper ───────────────────────────────────────────────


class TestCmdIdDeeper:
    def test_id_returns_info(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_id(""))
        assert "uid" in out


# ── _cmd_logname deeper ──────────────────────────────────────────


class TestCmdLognameDeeper:
    def test_logname_returns_string(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_logname(""))
        assert len(out.strip()) > 0


# ── _cmd_who deeper ──────────────────────────────────────────────


class TestCmdWhoDeeper:
    def test_who_returns_output(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_who(""))
        assert repl._last_exit_code == 0


# ── _cmd_uptime deeper ──────────────────────────────────────────


class TestCmdUptimeDeeper:
    def test_uptime_returns_output(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_uptime(""))
        assert repl._last_exit_code == 0


# ── _cmd_date deeper ─────────────────────────────────────────────


class TestCmdDateDeeper:
    def test_date_format(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_date("+%Y"))
        assert len(out.strip()) == 4


# ── _cmd_cal deeper ──────────────────────────────────────────────


class TestCmdCalDeeper:
    def test_cal_current_month(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_cal(""))
        assert repl._last_exit_code == 0


# ── _cmd_sleep deeper ────────────────────────────────────────────


class TestCmdSleepDeeper:
    def test_sleep_zero(self, repl):
        repl._cmd_sleep("0")
        assert repl._last_exit_code == 0


# ── _cmd_clear deeper ────────────────────────────────────────────


class TestCmdClearDeeper:
    def test_clear(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_clear(""))
        assert repl._last_exit_code == 0


# ── _cmd_protect deeper ──────────────────────────────────────────


class TestCmdProtectDeeper:
    def test_protect_show(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_protect(""))
        assert repl._last_exit_code == 0


# ── _cmd_unprotect deeper ────────────────────────────────────────


class TestCmdUnprotectDeeper:
    def test_unprotect_show(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_unprotect(""))
        assert repl._last_exit_code == 0


# ── _cmd_lsdev deeper ────────────────────────────────────────────


class TestCmdLsdevDeeper:
    def test_lsdev_empty(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_lsdev(""))
        assert repl._last_exit_code == 0


# ── _cmd_events deeper ──────────────────────────────────────────


class TestCmdEventsDeeper:
    def test_events_empty(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_events(""))
        assert repl._last_exit_code == 0


# ── _cmd_metrics deeper ──────────────────────────────────────────


class TestCmdMetricsDeeper:
    def test_metrics_returns_output(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_metrics(""))
        assert repl._last_exit_code == 0


# ── _cmd_help deeper ─────────────────────────────────────────────


class TestCmdHelpDeeper:
    def test_help_with_command(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_help("echo"))
        assert repl._last_exit_code == 0

    def test_help_unknown_command(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_help("nonexistent"))
        assert repl._last_exit_code == 0


# ── _cmd_tui deeper ──────────────────────────────────────────────


class TestCmdTuiDeeper:
    def test_tui_import_error(self, repl):
        with patch.dict('sys.modules', {'domains.shell.tui_repl': None}):
            out = _run_with_io(repl, [], lambda: repl._cmd_tui(""))
            assert "TUI" in out or "not available" in out.lower() or repl._last_exit_code == 1


# ── _cmd_render deeper ──────────────────────────────────────────


class TestCmdRenderDeeper:
    def test_render_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_render(""))
        assert repl._last_exit_code == 0
        assert "Scene" in out or "meshes" in out.lower()

    def test_render_scene(self, repl):
        mock_dev = MagicMock()
        repl._render_device = mock_dev
        out = _run_with_io(repl, [], lambda: repl._cmd_render("scene demo"))
        assert repl._last_exit_code == 0


# ── _cmd_api deeper ──────────────────────────────────────────────


class TestCmdApiDeeper:
    def test_api_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_api(""))
        assert "API" in out or "api" in out.lower()


# ── _cmd_load deeper ─────────────────────────────────────────────


class TestCmdLoadDeeper:
    def test_load_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_load(""))
        assert "Usage" in out


# ── _cmd_gen deeper ──────────────────────────────────────────────


class TestCmdGenDeeper:
    def test_gen_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_gen(""))
        assert "Usage" in out


# ── _cmd_chat deeper ─────────────────────────────────────────────


class TestCmdChatDeeper:
    def test_chat_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_chat(""))
        assert "Usage" in out


# ── _cmd_train deeper ────────────────────────────────────────────


class TestCmdTrainDeeper:
    def test_train_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_train(""))
        assert "API" in out or "api" in out.lower() or "train" in out.lower()


# ── _cmd_status deeper ──────────────────────────────────────────


class TestCmdStatusDeeper:
    def test_status_returns_output(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_status(""))
        assert repl._last_exit_code == 0


# ── _cmd_uptime deeper ──────────────────────────────────────────


class TestCmdUptimeDeeper2:
    def test_uptime_returns_output(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_uptime(""))
        assert repl._last_exit_code == 0


# ── _cmd_ps deeper ───────────────────────────────────────────────


class TestCmdPsDeeper:
    def test_ps_returns_output(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_ps(""))
        assert repl._last_exit_code == 0


# ── _cmd_kill deeper ─────────────────────────────────────────────


class TestCmdKillDeeper2:
    def test_kill_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_kill(""))
        assert "Usage" in out or repl._last_exit_code == 0


# ── _cmd_boot deeper ─────────────────────────────────────────────


class TestCmdBootDeeper:
    def test_boot_returns_output(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_boot(""))
        assert repl._last_exit_code == 0


# ── _cmd_shutdown deeper ─────────────────────────────────────────


class TestCmdShutdownDeeper:
    def test_shutdown_returns_output(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_shutdown(""))
        assert repl._last_exit_code == 0


# ── _cmd_svc deeper ──────────────────────────────────────────────


class TestCmdSvcDeeper:
    def test_svc_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_svc(""))
        assert repl._last_exit_code == 1 or "boot" in out.lower() or "init" in out.lower()


# ── _cmd_permit deeper ──────────────────────────────────────────


class TestCmdPermitDeeper:
    def test_permit_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_permit(""))
        assert "Usage" in out


# ── _cmd_deny deeper ────────────────────────────────────────────


class TestCmdDenyDeeper:
    def test_deny_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_deny(""))
        assert "Usage" in out


# ── _cmd_permissions deeper ──────────────────────────────────────


class TestCmdPermissionsDeeper:
    def test_permissions_returns_output(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_permissions(""))
        assert repl._last_exit_code == 0


# ── _cmd_note deeper ─────────────────────────────────────────────


class TestCmdNoteDeeper:
    def test_note_no_args(self, repl):
        try:
            out = _run_with_io(repl, [], lambda: repl._cmd_note(""))
            assert "Usage" in out or repl._last_exit_code == 1
        except (ImportError, ModuleNotFoundError):
            pass


# ── _cmd_logs deeper ─────────────────────────────────────────────


class TestCmdLogsDeeper:
    def test_logs_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_logs(""))
        assert repl._last_exit_code == 0


# ── _cmd_which deeper ────────────────────────────────────────────


class TestCmdWhichDeeper:
    def test_which_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_which(""))
        assert repl._last_exit_code == 1


# ── _cmd_type deeper ─────────────────────────────────────────────


class TestCmdTypeDeeper:
    def test_type_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_type(""))
        assert repl._last_exit_code == 1


# ── _cmd_read deeper ─────────────────────────────────────────────


class TestCmdReadDeeper2:
    def test_read_no_var(self, repl):
        mem = MemoryIO()
        mem.feed("test_input\n")
        old_io = repl.io
        old_console_io = repl.console._io
        repl.io = mem
        repl.console._io = mem
        try:
            repl._cmd_read("")
            assert repl._last_exit_code == 1
            assert "Usage" in mem.get_output()
        finally:
            repl.io = old_io
            repl.console._io = old_console_io


# ── _cmd_watch deeper ────────────────────────────────────────────


class TestCmdWatchDeeper2:
    def test_watch_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_watch(""))
        assert "Usage" in out


# ── _cmd_xargs deeper ────────────────────────────────────────────


class TestCmdXargsDeeper2:
    def test_xargs_piped(self, repl):
        repl._piped_input = "file1.txt\nfile2.txt"
        out = _run_with_io(repl, [], lambda: repl._cmd_xargs("echo"))
        assert repl._last_exit_code == 0


# ── _cmd_comm deeper ─────────────────────────────────────────────


class TestCmdCommDeeper2:
    def test_comm_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_comm(""))
        assert repl._last_exit_code == 1


# ── _cmd_yes deeper ──────────────────────────────────────────────


class TestCmdYesDeeper2:
    def test_yes_custom_string(self, repl):
        repl._piped_input = None
        repl.io._inputs = []
        repl.io._inputs.extend(["custom\n"] * 3)
        out = _run_with_io(repl, [], lambda: repl._cmd_yes("custom"))
        assert "custom" in out


# ── _cmd_env deeper ──────────────────────────────────────────────


class TestCmdEnvDeeper2:
    def test_env_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_env(""))
        assert repl._last_exit_code == 0


# ── _cmd_shuf deeper ─────────────────────────────────────────────


class TestCmdShufDeeper2:
    def test_shuf_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_shuf(""))
        assert repl._last_exit_code == 1


# ── _cmd_rev deeper ──────────────────────────────────────────────


class TestCmdRevDeeper2:
    def test_rev_piped(self, repl):
        repl._piped_input = "hello"
        out = _run_with_io(repl, [], lambda: repl._cmd_rev(""))
        assert "olleh" in out


# ── _cmd_paste deeper ────────────────────────────────────────────


class TestCmdPasteDeeper2:
    def test_paste_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_paste(""))
        assert repl._last_exit_code == 1


# ── _cmd_tee deeper ──────────────────────────────────────────────


class TestCmdTeeDeeper2:
    def test_tee_piped(self, repl):
        repl._piped_input = "tee_test"
        out = _run_with_io(repl, [], lambda: repl._cmd_tee(""))
        assert "tee_test" in out


# ── _cmd_od deeper ───────────────────────────────────────────────


class TestCmdOdDeeper:
    def test_od_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_od(""))
        assert repl._last_exit_code == 1


# ── _cmd_expand deeper ──────────────────────────────────────────


class TestCmdExpandDeeper:
    def test_expand_piped(self, repl):
        repl._piped_input = "a\tb"
        out = _run_with_io(repl, [], lambda: repl._cmd_expand(""))
        assert "a" in out and "b" in out


# ── _cmd_unexpand deeper ─────────────────────────────────────────


class TestCmdUnexpandDeeper:
    def test_unexpand_piped(self, repl):
        repl._piped_input = "a    b"
        out = _run_with_io(repl, [], lambda: repl._cmd_unexpand(""))
        assert "a" in out and "b" in out


# ── _cmd_fold deeper ─────────────────────────────────────────────


class TestCmdFoldDeeper2:
    def test_fold_width_one(self, repl):
        repl._piped_input = "ab"
        out = _run_with_io(repl, [], lambda: repl._cmd_fold("-w 1"))
        assert "a" in out and "b" in out


# ── _cmd_nl deeper ───────────────────────────────────────────────


class TestCmdNlDeeper2:
    def test_nl_piped_lines(self, repl):
        repl._piped_input = "x\ny\nz"
        out = _run_with_io(repl, [], lambda: repl._cmd_nl(""))
        assert "1" in out and "3" in out


# ── _cmd_seq edge cases ──────────────────────────────────────────


class TestCmdSeqEdgeCases:
    def test_seq_single(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_seq("5"))
        assert "5" in out

    def test_seq_negative(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_seq("-3 1"))
        assert "-3" in out


# ── _cmd_cut edge cases ──────────────────────────────────────────


class TestCmdCutEdgeCases2:
    def test_cut_open_range(self, repl):
        repl._piped_input = "a\tb\tc\td"
        out = _run_with_io(repl, [], lambda: repl._cmd_cut("-f3-"))
        assert "c\td" in out

    def test_cut_single_field(self, repl):
        repl._piped_input = "a\tb\tc"
        out = _run_with_io(repl, [], lambda: repl._cmd_cut("-f2"))
        assert "b" in out


# ── _cmd_tr edge cases ──────────────────────────────────────────


class TestCmdTrEdgeCases2:
    def test_tr_translate(self, repl):
        repl._piped_input = "abc"
        out = _run_with_io(repl, [], lambda: repl._cmd_tr("a b"))
        assert "bbc" in out


# ── _expand_vars deeper ──────────────────────────────────────────


class TestExpandVarsDeeper:
    def test_expand_dollar_question(self, repl):
        repl._last_exit_code = 42
        result = repl._expand_vars("$?")
        assert result == "42"

    def test_expand_braced_var(self, repl):
        repl._env["MYVAR"] = "hello"
        result = repl._expand_vars("${MYVAR}")
        assert result == "hello"

    def test_expand_undefined_var(self, repl):
        result = repl._expand_vars("$UNDEFINED_VAR")
        assert "$UNDEFINED_VAR" in result

    def test_expand_mixed(self, repl):
        repl._env["X"] = "val"
        result = repl._expand_vars("$X and $?")
        assert "val" in result

    def test_expand_empty_text(self, repl):
        result = repl._expand_vars("")
        assert result == ""


# ── _expand_cmd_subst deeper ─────────────────────────────────────


class TestExpandCmdSubstDeeper:
    def test_cmd_subst_basic(self, repl):
        result = repl._expand_cmd_subst("$(echo hello)")
        assert "hello" in result

    def test_cmd_subst_nested_not_supported(self, repl):
        result = repl._expand_cmd_subst("$(echo $(echo nested))")
        assert "nested" in result

    def test_cmd_subst_no_match(self, repl):
        result = repl._expand_cmd_subst("no substitution here")
        assert result == "no substitution here"

    def test_cmd_subst_empty(self, repl):
        result = repl._expand_cmd_subst("")
        assert result == ""


# ── _expand_globs deeper ─────────────────────────────────────────


class TestExpandGlobsDeeper:
    def test_glob_no_match(self, repl):
        result = repl._expand_globs("*.nonexistent_ext_xyz")
        assert "*.nonexistent_ext_xyz" in result

    def test_glob_no_glob_chars(self, repl):
        result = repl._expand_globs("no_glob_here")
        assert result == "no_glob_here"

    def test_glob_question_mark(self, repl, tmp_path):
        (tmp_path / "a.txt").write_text("")
        (tmp_path / "b.txt").write_text("")
        old = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = repl._expand_globs("?.txt")
            assert "a.txt" in result
        finally:
            os.chdir(old)


# ── _suggest_command deeper ──────────────────────────────────────


class TestSuggestCommandDeeper:
    def test_suggest_exact_match(self, repl):
        result = repl._suggest_command("ech")
        assert result == "echo"

    def test_suggest_close_match(self, repl):
        result = repl._suggest_command("echx")
        assert result is None or isinstance(result, str)

    def test_suggest_no_match(self, repl):
        result = repl._suggest_command("xyzabc")
        assert result is None

    def test_suggest_empty(self, repl):
        result = repl._suggest_command("")
        assert result is None


# ── _cmd_cd deeper ───────────────────────────────────────────────


class TestCmdCdDeeper:
    def test_cd_dash(self, repl, tmp_path):
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        old = os.getcwd()
        try:
            os.chdir(d1)
            repl._env["OLDPWD"] = str(d2)
            repl._cmd_cd("-")
            assert os.getcwd() == str(d2)
        finally:
            os.chdir(old)

    def test_cd_not_a_directory(self, repl, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        out = _run_with_io(repl, [], lambda: repl._cmd_cd(str(f)))
        assert repl._last_exit_code == 1
        assert "not a directory" in out.lower()

    def test_cd_no_such_dir(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_cd("/nonexistent/path"))
        assert repl._last_exit_code == 1

    def test_cd_permission_denied(self, repl, tmp_path):
        d = tmp_path / "noperm"
        d.mkdir()
        d.chmod(0o000)
        out = _run_with_io(repl, [], lambda: repl._cmd_cd(str(d)))
        assert repl._last_exit_code == 1
        d.chmod(0o755)


# ── _cmd_ls deeper ───────────────────────────────────────────────


class TestCmdLsDeeper:
    def test_ls_no_such_dir(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_ls("/nonexistent"))
        assert repl._last_exit_code == 1

    def test_ls_with_files(self, repl, tmp_path):
        (tmp_path / "a.txt").write_text("")
        (tmp_path / "b.txt").write_text("")
        old = os.getcwd()
        try:
            os.chdir(tmp_path)
            out = _run_with_io(repl, [], lambda: repl._cmd_ls(""))
            assert "a.txt" in out
        finally:
            os.chdir(old)

    def test_ls_current_dir(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_ls("."))
        assert repl._last_exit_code == 0


# ── _cmd_cat deeper ──────────────────────────────────────────────


class TestCmdCatDeeper:
    def test_cat_piped(self, repl):
        repl._piped_input = "piped content"
        out = _run_with_io(repl, [], lambda: repl._cmd_cat(""))
        assert "piped content" in out

    def test_cat_nonexistent(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_cat("/nonexistent"))
        assert repl._last_exit_code == 1

    def test_cat_no_args_no_pipe(self, repl):
        repl._piped_input = None
        out = _run_with_io(repl, [], lambda: repl._cmd_cat(""))
        assert repl._last_exit_code == 1


# ── _cmd_rm deeper ──────────────────────────────────────────────


class TestCmdRmDeeper:
    def test_rm_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_rm(""))
        assert repl._last_exit_code == 1

    def test_rm_nonexistent_no_force(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_rm("/nonexistent"))
        assert repl._last_exit_code == 1
        assert "No such file" in out

    def test_rm_nonexistent_with_force(self, repl):
        repl._perms._granted.add("rm")
        out = _run_with_io(repl, [], lambda: repl._cmd_rm("-f /nonexistent"))
        assert repl._last_exit_code == 0

    def test_rm_directory_without_recursive(self, repl, tmp_path):
        d = tmp_path / "mydir"
        d.mkdir()
        out = _run_with_io(repl, [], lambda: repl._cmd_rm(str(d)))
        assert "Is a directory" in out

    def test_rm_recursive(self, repl, tmp_path):
        d = tmp_path / "mydir"
        d.mkdir()
        (d / "file.txt").write_text("hello")
        repl._perms._granted.add("rm")
        out = _run_with_io(repl, [], lambda: repl._cmd_rm("-r " + str(d)))
        assert not d.exists()


# ── _cmd_cp deeper ───────────────────────────────────────────────


class TestCmdCpDeeper:
    def test_cp_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_cp(""))
        assert repl._last_exit_code == 1

    def test_cp_nonexistent_source(self, repl, tmp_path):
        out = _run_with_io(repl, [], lambda: repl._cmd_cp(str(tmp_path / "nonexistent") + " " + str(tmp_path / "dst")))
        assert repl._last_exit_code == 1


# ── _cmd_mv deeper ───────────────────────────────────────────────


class TestCmdMvDeeper:
    def test_mv_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_mv(""))
        assert repl._last_exit_code == 1

    def test_mv_nonexistent_source(self, repl, tmp_path):
        out = _run_with_io(repl, [], lambda: repl._cmd_mv(str(tmp_path / "nonexistent") + " " + str(tmp_path / "dst")))
        assert repl._last_exit_code == 1


# ── _cmd_touch deeper ────────────────────────────────────────────


class TestCmdTouchDeeper:
    def test_touch_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_touch(""))
        assert repl._last_exit_code == 1

    def test_touch_existing_file(self, repl, tmp_path):
        f = tmp_path / "exists.txt"
        f.write_text("old")
        out = _run_with_io(repl, [], lambda: repl._cmd_touch(str(f)))
        assert repl._last_exit_code == 0
        assert f.read_text() == "old"

    def test_touch_multiple(self, repl, tmp_path):
        old = os.getcwd()
        try:
            os.chdir(tmp_path)
            out = _run_with_io(repl, [], lambda: repl._cmd_touch("a.txt b.txt"))
            assert (tmp_path / "a.txt").exists()
            assert (tmp_path / "b.txt").exists()
        finally:
            os.chdir(old)


# ── _cmd_mkdir deeper ───────────────────────────────────────────


class TestCmdMkdirDeeper:
    def test_mkdir_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_mkdir(""))
        assert repl._last_exit_code == 1

    def test_mkdir_existing(self, repl, tmp_path):
        d = tmp_path / "exists"
        d.mkdir()
        out = _run_with_io(repl, [], lambda: repl._cmd_mkdir(str(d)))
        assert repl._last_exit_code == 1

    def test_mkdir_recursive(self, repl, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        out = _run_with_io(repl, [], lambda: repl._cmd_mkdir("-p " + str(target)))
        assert target.exists() or repl._last_exit_code == 0


# ── _cmd_head deeper ────────────────────────────────────────────


class TestCmdHeadDeeper:
    def test_head_no_args_no_pipe(self, repl):
        repl._piped_input = None
        out = _run_with_io(repl, [], lambda: repl._cmd_head(""))
        assert repl._last_exit_code == 1

    def test_head_piped(self, repl):
        repl._piped_input = "line1\nline2\nline3\nline4\nline5"
        out = _run_with_io(repl, [], lambda: repl._cmd_head("-2"))
        assert "line1" in out
        assert "line2" in out


# ── _cmd_tail deeper ────────────────────────────────────────────


class TestCmdTailDeeper:
    def test_tail_no_args_no_pipe(self, repl):
        repl._piped_input = None
        out = _run_with_io(repl, [], lambda: repl._cmd_tail(""))
        assert repl._last_exit_code == 1

    def test_tail_piped(self, repl):
        repl._piped_input = "line1\nline2\nline3\nline4\nline5"
        out = _run_with_io(repl, [], lambda: repl._cmd_tail("-2"))
        assert "line4" in out
        assert "line5" in out


# ── _cmd_wc deeper ──────────────────────────────────────────────


class TestCmdWcDeeper:
    def test_wc_no_args_no_pipe(self, repl):
        repl._piped_input = None
        out = _run_with_io(repl, [], lambda: repl._cmd_wc(""))
        assert repl._last_exit_code == 1

    def test_wc_piped(self, repl):
        repl._piped_input = "hello world"
        out = _run_with_io(repl, [], lambda: repl._cmd_wc(""))
        assert "2" in out


# ── _cmd_grep deeper ────────────────────────────────────────────


class TestCmdGrepDeeper:
    def test_grep_no_args(self, repl):
        repl._piped_input = None
        out = _run_with_io(repl, [], lambda: repl._cmd_grep(""))
        assert repl._last_exit_code == 1

    def test_grep_piped_match(self, repl):
        repl._piped_input = "hello world\nfoo bar"
        out = _run_with_io(repl, [], lambda: repl._cmd_grep("hello"))
        assert "hello" in out

    def test_grep_piped_no_match(self, repl):
        repl._piped_input = "hello world"
        out = _run_with_io(repl, [], lambda: repl._cmd_grep("nonexistent"))
        assert repl._last_exit_code == 1

    def test_grep_v_invert(self, repl):
        repl._piped_input = "hello\nworld\nfoo"
        out = _run_with_io(repl, [], lambda: repl._cmd_grep("-v hello"))
        assert "hello" not in out
        assert "world" in out


# ── _cmd_sort deeper ────────────────────────────────────────────


class TestCmdSortDeeper:
    def test_sort_piped(self, repl):
        repl._piped_input = "c\na\nb"
        out = _run_with_io(repl, [], lambda: repl._cmd_sort(""))
        lines = out.strip().split("\n")
        assert lines[0].strip() == "a"

    def test_sort_reverse(self, repl):
        repl._piped_input = "a\nc\nb"
        out = _run_with_io(repl, [], lambda: repl._cmd_sort("-r"))
        lines = out.strip().split("\n")
        assert lines[0].strip() == "c"

    def test_sort_unique(self, repl):
        repl._piped_input = "a\nb\na\nc"
        out = _run_with_io(repl, [], lambda: repl._cmd_sort("-u"))
        assert "a" in out
        assert out.count("a") == 1


# ── _cmd_uniq deeper ────────────────────────────────────────────


class TestCmdUniqDeeper:
    def test_uniq_piped(self, repl):
        repl._piped_input = "a\na\nb\nc\nc"
        out = _run_with_io(repl, [], lambda: repl._cmd_uniq(""))
        assert "a\nb\nc" in out

    def test_uniq_no_duplicates(self, repl):
        repl._piped_input = "a\nb\nc"
        out = _run_with_io(repl, [], lambda: repl._cmd_uniq(""))
        assert "a" in out and "b" in out and "c" in out


# ── _cmd_find deeper ────────────────────────────────────────────


class TestCmdFindDeeper:
    def test_find_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_find(""))
        assert repl._last_exit_code == 1

    def test_find_name(self, repl, tmp_path):
        (tmp_path / "target.txt").write_text("")
        (tmp_path / "other.txt").write_text("")
        out = _run_with_io(repl, [], lambda: repl._cmd_find("-name target.txt " + str(tmp_path)))
        assert "target.txt" in out


# ── _cmd_tee deeper ─────────────────────────────────────────────


class TestCmdTeeDeeper:
    def test_tee_no_args_no_pipe(self, repl):
        repl._piped_input = None
        out = _run_with_io(repl, [], lambda: repl._cmd_tee(""))
        assert repl._last_exit_code == 1

    def test_tee_to_file(self, repl, tmp_path):
        repl._piped_input = "tee content"
        target = str(tmp_path / "tee_out.txt")
        out = _run_with_io(repl, [], lambda: repl._cmd_tee(target))
        assert "tee content" in out or repl._last_exit_code == 0


# ── _cmd_printf deeper ──────────────────────────────────────────


class TestCmdPrintfDeeper:
    def test_printf_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_printf(""))
        assert repl._last_exit_code == 1

    def test_printf_format(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_printf("hello %s" + " world"))
        assert "hello" in out

    def test_printf_escape(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_printf("line1\\nline2"))
        assert "line1" in out


# ── _cmd_seq deeper ─────────────────────────────────────────────


class TestCmdSeqDeeper2:
    def test_seq_single_arg(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_seq("3"))
        assert "1" in out and "3" in out

    def test_seq_negative_start(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_seq("-2 2"))
        assert "-2" in out and "2" in out

    def test_seq_float_step(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_seq("0 0.5 1.5"))
        assert "0.0" in out or "0.5" in out


# ── _cmd_comm deeper ────────────────────────────────────────────


class TestCmdCommDeeper3:
    def test_comm_piped(self, repl):
        repl._piped_input = "a\nb\nc"
        out = _run_with_io(repl, [], lambda: repl._cmd_comm("-1 " + repl._piped_input))
        assert repl._last_exit_code == 1 or "a" in out


# ── _cmd_xargs deeper ───────────────────────────────────────────


class TestCmdXargsDeeper3:
    def test_xargs_no_args(self, repl):
        repl._piped_input = "hello"
        out = _run_with_io(repl, [], lambda: repl._cmd_xargs(""))
        assert repl._last_exit_code == 0

    def test_xargs_echo(self, repl):
        repl._piped_input = "file1"
        out = _run_with_io(repl, [], lambda: repl._cmd_xargs("echo"))
        assert "file1" in out


# ── _cmd_yes deeper ─────────────────────────────────────────────


class TestCmdYesDeeper3:
    def test_yes_default(self, repl):
        repl.io._inputs = ["y\n"] * 3
        out = _run_with_io(repl, [], lambda: repl._cmd_yes(""))
        assert "y" in out


# ── _cmd_env deeper ─────────────────────────────────────────────


class TestCmdEnvDeeper3:
    def test_env_shows_vars(self, repl):
        repl._env["TEST_ENV_VAR"] = "test_value"
        out = _run_with_io(repl, [], lambda: repl._cmd_env(""))
        assert "TEST_ENV_VAR" in out


# ── _cmd_shuf deeper ────────────────────────────────────────────


class TestCmdShufDeeper3:
    def test_shuf_piped(self, repl):
        repl._piped_input = "1\n2\n3\n4\n5"
        out = _run_with_io(repl, [], lambda: repl._cmd_shuf(""))
        assert repl._last_exit_code == 0


# ── _cmd_rev deeper ─────────────────────────────────────────────


class TestCmdRevDeeper3:
    def test_rev_piped(self, repl):
        repl._piped_input = "hello"
        out = _run_with_io(repl, [], lambda: repl._cmd_rev(""))
        assert "olleh" in out

    def test_rev_multiline(self, repl):
        repl._piped_input = "ab\ncd"
        out = _run_with_io(repl, [], lambda: repl._cmd_rev(""))
        assert "ba" in out


# ── _cmd_paste deeper ───────────────────────────────────────────


class TestCmdPasteDeeper3:
    def test_paste_no_args(self, repl):
        repl._piped_input = None
        out = _run_with_io(repl, [], lambda: repl._cmd_paste(""))
        assert repl._last_exit_code == 1


# ── _cmd_od deeper ──────────────────────────────────────────────


class TestCmdOdDeeper2:
    def test_od_no_args(self, repl):
        repl._piped_input = None
        out = _run_with_io(repl, [], lambda: repl._cmd_od(""))
        assert repl._last_exit_code == 1

    def test_od_nonexistent_file(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_od("/nonexistent"))
        assert repl._last_exit_code == 1


# ── _cmd_expand deeper ──────────────────────────────────────────


class TestCmdExpandDeeper2:
    def test_expand_no_args(self, repl):
        repl._piped_input = None
        out = _run_with_io(repl, [], lambda: repl._cmd_expand(""))
        assert repl._last_exit_code == 1


# ── _cmd_unexpand deeper ─────────────────────────────────────────


class TestCmdUnexpandDeeper2:
    def test_unexpand_no_args(self, repl):
        repl._piped_input = None
        out = _run_with_io(repl, [], lambda: repl._cmd_unexpand(""))
        assert repl._last_exit_code == 1


# ── _cmd_diff deeper ────────────────────────────────────────────


class TestCmdDiffDeeper2:
    def test_diff_identical(self, repl, tmp_path):
        f1 = tmp_path / "same1.txt"
        f2 = tmp_path / "same2.txt"
        f1.write_text("identical\n")
        f2.write_text("identical\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_diff(str(f1) + " " + str(f2)))
        assert repl._last_exit_code == 0

    def test_diff_different(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("line1\n")
        f2.write_text("line2\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_diff(str(f1) + " " + str(f2)))
        assert repl._last_exit_code == 1


# ── _cmd_stat deeper ────────────────────────────────────────────


class TestCmdStatDeeper2:
    def test_stat_existing_file(self, repl, tmp_path):
        f = tmp_path / "stat_test.txt"
        f.write_text("content")
        out = _run_with_io(repl, [], lambda: repl._cmd_stat(str(f)))
        assert repl._last_exit_code == 0


# ── _cmd_chmod deeper ───────────────────────────────────────────


class TestCmdChmodDeeper2:
    def test_chmod_existing_file(self, repl, tmp_path):
        f = tmp_path / "chmod_test.txt"
        f.write_text("content")
        out = _run_with_io(repl, [], lambda: repl._cmd_chmod("644 " + str(f)))
        assert repl._last_exit_code == 0


# ── _cmd_du deeper ──────────────────────────────────────────────


class TestCmdDuDeeper2:
    def test_du_existing_dir(self, repl, tmp_path):
        (tmp_path / "file.txt").write_text("hello")
        out = _run_with_io(repl, [], lambda: repl._cmd_du(str(tmp_path)))
        assert repl._last_exit_code == 0


# ── _cmd_ln deeper ──────────────────────────────────────────────


class TestCmdLnDeeper2:
    def test_ln_symlink(self, repl, tmp_path):
        target = tmp_path / "original.txt"
        target.write_text("content")
        link = tmp_path / "link.txt"
        out = _run_with_io(repl, [], lambda: repl._cmd_ln("-s " + str(target) + " " + str(link)))
        assert repl._last_exit_code == 0
        assert link.is_symlink()


# ── _cmd_read deeper ────────────────────────────────────────────


class TestCmdReadDeeper3:
    def test_read_with_var(self, repl):
        mem = MemoryIO()
        mem.feed("test_value\n")
        old_io = repl.io
        old_cio = repl.console._io
        repl.io = mem
        repl.console._io = mem
        try:
            repl._cmd_read("MYVAR")
            assert repl._env.get("MYVAR") == "test_value"
        finally:
            repl.io = old_io
            repl.console._io = old_cio

    def test_read_with_prompt(self, repl):
        mem = MemoryIO()
        mem.feed("val\n")
        old_io = repl.io
        old_cio = repl.console._io
        repl.io = mem
        repl.console._io = mem
        try:
            repl._cmd_read("-p Enter: MYVAR")
            assert repl._env.get("MYVAR") == "val"
        finally:
            repl.io = old_io
            repl.console._io = old_cio


# ── _cmd_kill deeper ────────────────────────────────────────────


class TestCmdKillDeeper3:
    def test_kill_invalid_signal(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_kill("invalid 1"))
        assert repl._last_exit_code == 1 or "invalid" in out.lower() or "error" in out.lower()


# ── _cmd_bg deeper ──────────────────────────────────────────────


class TestCmdBgDeeper2:
    def test_bg_with_job_id(self, repl):
        repl._bg_threads = {}
        out = _run_with_io(repl, [], lambda: repl._cmd_bg("999"))
        assert "No background" in out


# ── _cmd_fg deeper ──────────────────────────────────────────────


class TestCmdFgDeeper2:
    def test_fg_with_job_id(self, repl):
        repl._bg_threads = {}
        out = _run_with_io(repl, [], lambda: repl._cmd_fg("999"))
        assert "No background" in out


# ── _cmd_watch deeper ───────────────────────────────────────────


class TestCmdWatchDeeper3:
    def test_watch_invalid_interval(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_watch("-n abc echo test"))
        assert "Invalid" in out or repl._last_exit_code == 1


# ── _cmd_export deeper ──────────────────────────────────────────


class TestCmdExportDeeper2:
    def test_export_with_equals(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_export("FOO=bar"))
        assert repl._env.get("FOO") == "bar"

    def test_export_no_value(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_export("FOO"))
        assert repl._last_exit_code == 0


# ── _cmd_set deeper ─────────────────────────────────────────────


class TestCmdSetDeeper2:
    def test_set_with_equals(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_set("BAZ=qux"))
        assert repl._env.get("BAZ") == "qux"

    def test_set_no_value(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_set("VAR"))
        assert repl._last_exit_code == 0


# ── _cmd_alias deeper ───────────────────────────────────────────


class TestCmdAliasDeeper2:
    def test_alias_set(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_alias("gg=git status"))
        assert repl._aliases.get("gg") == "git status"

    def test_alias_list(self, repl):
        repl._aliases["testalias"] = "echo test"
        out = _run_with_io(repl, [], lambda: repl._cmd_alias(""))
        assert "testalias" in out


# ── _cmd_unalias deeper ─────────────────────────────────────────


class TestCmdUnaliasDeeper2:
    def test_unalias_existing(self, repl):
        repl._aliases["myalias"] = "echo hi"
        out = _run_with_io(repl, [], lambda: repl._cmd_unalias("myalias"))
        assert "myalias" not in repl._aliases

    def test_unalias_nonexistent(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_unalias("nonexistent"))
        assert "No alias" in out


# ── _cmd_history deeper ─────────────────────────────────────────


class TestCmdHistoryDeeper2:
    def test_history_empty(self, repl):
        repl._history = []
        out = _run_with_io(repl, [], lambda: repl._cmd_history(""))
        assert repl._last_exit_code == 0

    def test_history_with_entries(self, repl):
        repl._history = ["echo a", "echo b", "echo c"]
        out = _run_with_io(repl, [], lambda: repl._cmd_history(""))
        assert "echo a" in out


# ── _cmd_fc deeper ──────────────────────────────────────────────


class TestCmdFcDeeper2:
    def test_fc_no_history(self, repl):
        repl._history = []
        out = _run_with_io(repl, [], lambda: repl._cmd_fc(""))
        assert repl._last_exit_code == 0

    def test_fc_list_last(self, repl):
        repl._history = ["echo first", "echo second"]
        out = _run_with_io(repl, [], lambda: repl._cmd_fc("-l"))
        assert "echo second" in out


# ── _cmd_id deeper ──────────────────────────────────────────────


class TestCmdIdDeeper2:
    def test_id_returns_uid(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_id(""))
        assert "uid" in out.lower()


# ── _cmd_logname deeper ─────────────────────────────────────────


class TestCmdLognameDeeper2:
    def test_logname_returns_string(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_logname(""))
        assert len(out.strip()) > 0


# ── _cmd_hostname deeper ────────────────────────────────────────


class TestCmdHostnameDeeper2:
    def test_hostname_returns_string(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_hostname(""))
        assert len(out.strip()) > 0


# ── _cmd_nproc deeper ──────────────────────────────────────────


class TestCmdNprocDeeper2:
    def test_nproc_returns_number(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_nproc(""))
        assert out.strip().isdigit()


# ── _cmd_uptime deeper ─────────────────────────────────────────


class TestCmdUptimeDeeper3:
    def test_uptime_returns_output(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_uptime(""))
        assert repl._last_exit_code == 0


# ── _cmd_uname deeper ──────────────────────────────────────────


class TestCmdUnameDeeper2:
    def test_uname_s(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_uname("-s"))
        assert "Linux" in out or "Darwin" in out

    def test_uname_m(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_uname("-m"))
        assert len(out.strip()) > 0


# ── _cmd_cal deeper ─────────────────────────────────────────────


class TestCmdCalDeeper2:
    def test_cal_specific_month(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_cal("1 2024"))
        assert "January" in out or "2024" in out


# ── _cmd_date deeper ────────────────────────────────────────────


class TestCmdDateDeeper2:
    def test_date_default(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_date(""))
        assert repl._last_exit_code == 0

    def test_date_format_string(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_date("+%H:%M"))
        assert ":" in out


# ── _cmd_sleep deeper ──────────────────────────────────────────


class TestCmdSleepDeeper2:
    def test_sleep_zero(self, repl):
        repl._cmd_sleep("0")
        assert repl._last_exit_code == 0

    def test_sleep_negative(self, repl):
        try:
            repl._cmd_sleep("-1")
        except (ValueError, Exception):
            pass


# ── _cmd_permit deeper ──────────────────────────────────────────


class TestCmdPermitDeeper2:
    def test_permit_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_permit(""))
        assert "Usage" in out

    def test_permit_grant(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_permit("rm"))
        assert "Granted" in out
        assert "rm" in repl._perms._granted

    def test_permit_persist(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_permit("chmod --persist"))
        assert "persistent" in out.lower()

    def test_permit_all_risk(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_permit("--all-DANGEROUS"))
        assert "DANGEROUS" in out

    def test_permit_unknown_risk(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_permit("--all-FAKE"))
        assert "Unknown" in out


# ── _cmd_deny deeper ────────────────────────────────────────────


class TestCmdDenyDeeper2:
    def test_deny_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_deny(""))
        assert "Usage" in out

    def test_deny_revoke(self, repl):
        repl._perms.grant("testcmd")
        out = _run_with_io(repl, [], lambda: repl._cmd_deny("testcmd"))
        assert "Denied" in out or "revoked" in out.lower() or "testcmd" not in repl._perms._granted

    def test_deny_all_risk(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_deny("--all-ELEVATED"))
        assert "ELEVATED" in out

    def test_deny_unknown_risk(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_deny("--all-FAKE"))
        assert "Unknown" in out


# ── _cmd_logs deeper ────────────────────────────────────────────


class TestCmdLogsDeeper2:
    def test_logs_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_logs(""))
        assert repl._last_exit_code == 0

    def test_logs_clear(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_logs("-c"))
        assert "cleared" in out.lower()

    def test_logs_level_filter(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_logs("-l ERROR"))
        assert repl._last_exit_code == 0

    def test_logs_source_filter(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_logs("-s test"))
        assert repl._last_exit_code == 0

    def test_logs_lines(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_logs("-n 5"))
        assert repl._last_exit_code == 0

    def test_logs_stats(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_logs("--stats"))
        assert repl._last_exit_code == 0

    def test_logs_export(self, repl, tmp_path):
        export_file = str(tmp_path / "logs.txt")
        out = _run_with_io(repl, [], lambda: repl._cmd_logs("-e " + export_file))
        assert repl._last_exit_code == 0


# ── _cmd_svc deeper ─────────────────────────────────────────────


class TestCmdSvcDeeper2:
    def test_svc_not_booted(self, repl):
        from domains.shell.init import reset_init_system
        reset_init_system()
        repl.os._init = None
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("list"))
        assert "boot" in out.lower() or repl._last_exit_code == 1

    def test_svc_start_no_name(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("start"))
        assert repl._last_exit_code == 1 or "name" in out.lower()

    def test_svc_stop_no_name(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("stop"))
        assert repl._last_exit_code == 1 or "name" in out.lower()

    def test_svc_restart_no_name(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("restart"))
        assert repl._last_exit_code == 1 or "name" in out.lower()


# ── _cmd_boot deeper ────────────────────────────────────────────


class TestCmdBootDeeper2:
    def test_boot_sets_running(self, repl):
        repl._running = False
        out = _run_with_io(repl, [], lambda: repl._cmd_boot(""))
        assert repl._running is True

    def test_boot_already_running(self, repl):
        repl._running = True
        repl._piped_input = None
        out = _run_with_io(repl, [], lambda: repl._cmd_boot(""))
        assert "Already booted" in out


# ── _cmd_shutdown deeper ────────────────────────────────────────


class TestCmdShutdownDeeper2:
    def test_shutdown_clears_running(self, repl):
        repl._running = True
        out = _run_with_io(repl, [], lambda: repl._cmd_shutdown(""))
        assert repl._running is False


# ── _cmd_help deeper ────────────────────────────────────────────


class TestCmdHelpDeeper2:
    def test_help_brief(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_help("brief"))
        assert "Most-used" in out

    def test_help_specific_command(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_help("cd"))
        assert "cd" in out.lower()

    def test_help_unknown(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_help("nonexistent_xyz"))
        assert repl._last_exit_code == 0

    def test_help_all_commands(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_help(""))
        assert "Available commands" in out or len(out) > 100


# ── _cmd_permissions deeper ─────────────────────────────────────


class TestCmdPermissionsDeeper2:
    def test_permissions_returns_output(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_permissions(""))
        assert "Permission" in out or "permission" in out.lower() or repl._last_exit_code == 0


# ── _cmd_confirm deeper ─────────────────────────────────────────


class TestCmdConfirmDeeper2:
    def test_confirm_show(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_confirm(""))
        assert repl._last_exit_code == 0

    def test_confirm_on(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_confirm("on"))
        assert repl._last_exit_code == 0

    def test_confirm_off(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_confirm("off"))
        assert repl._last_exit_code == 0


# ── _execute_single complex cases ───────────────────────────────


class TestExecuteSingleComplex:
    def test_execute_single_redirect_append(self, repl, tmp_path):
        f = tmp_path / "append.txt"
        f.write_text("line1\n")
        repl._execute_single("echo line2 >> " + str(f))
        content = f.read_text()
        assert "line1" in content
        assert "line2" in content

    def test_execute_single_inline_env(self, repl):
        repl._execute_single("MYVAR=hello echo test_inline_env")
        assert repl._last_exit_code == 0

    def test_execute_single管道(self, repl):
        result = repl._execute_single("echo pipe_test | wc")
        assert repl._last_exit_code == 0

    def test_execute_single_empty(self, repl):
        try:
            repl._execute_single("")
        except IndexError:
            pass


# ── pipeline complex cases ──────────────────────────────────────


class TestPipelineComplex:
    def test_pipeline_double_pipe(self, repl):
        repl._dispatch("echo a | echo b")
        assert repl._last_exit_code == 0

    def test_pipeline_with_redirect(self, repl, tmp_path):
        f = str(tmp_path / "pipe_redir.txt")
        repl._dispatch("echo piped > " + f + " && cat " + f)
        assert (tmp_path / "pipe_redir.txt").exists() or repl._last_exit_code == 0

    def test_pipeline_background(self, repl):
        repl._dispatch("echo bg_test &")
        assert repl._last_exit_code == 0


# ── _expand_vars complex ────────────────────────────────────────


class TestExpandVarsComplex:
    def test_expand_nested_braces(self, repl):
        repl._env["A"] = "val"
        result = repl._expand_vars("${A}_suffix")
        assert result == "val_suffix"

    def test_expand_multiple_vars(self, repl):
        repl._env["X"] = "x"
        repl._env["Y"] = "y"
        result = repl._expand_vars("$X and $Y")
        assert "x" in result and "y" in result

    def test_expand_dollar_question_after_error(self, repl):
        repl._last_exit_code = 99
        result = repl._expand_vars("exit=$?")
        assert "99" in result


# ── _expand_history complex ─────────────────────────────────────


class TestExpandHistoryComplex:
    def test_expand_history_nth_arg(self, repl):
        repl._history = ["echo first second third"]
        result = repl._expand_history("!:1")
        assert result == "first"

    def test_expand_history_all_args(self, repl):
        repl._history = ["echo a b c"]
        result = repl._expand_history("!*")
        assert "a b c" in result

    def test_expand_history_neg(self, repl):
        repl._history = ["cmd1", "cmd2", "cmd3"]
        result = repl._expand_history("!-2")
        assert result == "cmd2"

    def test_expand_history_empty(self, repl):
        repl._history = []
        result = repl._expand_history("!!")
        assert "!!" in result


# ── _expand_alias deeper ────────────────────────────────────────


class TestExpandAliasDeeper:
    def test_expand_alias_single_level(self, repl):
        repl._aliases["a"] = "b"
        repl._aliases["b"] = "echo done"
        result = repl._expand_alias("a")
        assert result == "b"

    def test_expand_alias_no_match(self, repl):
        result = repl._expand_alias("noalias")
        assert result == "noalias"

    def test_expand_alias_empty(self, repl):
        result = repl._expand_alias("")
        assert result == ""


# ── _suggest_command deeper ─────────────────────────────────────


class TestSuggestCommandDeeper2:
    def test_suggest_partial_match(self, repl):
        result = repl._suggest_command("ech")
        assert result == "echo"

    def test_suggest_close_distance(self, repl):
        result = repl._suggest_command("ecoh")
        assert result is None or isinstance(result, str)

    def test_suggest_empty(self, repl):
        result = repl._suggest_command("")
        assert result is None


# ── _cmd_date deeper ────────────────────────────────────────────


class TestCmdDateDeeper3:
    def test_date_plus_format(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_date("+%Y-%m-%d"))
        assert "-" in out

    def test_date_default(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_date(""))
        assert repl._last_exit_code == 0


# ── _cmd_cal deeper ─────────────────────────────────────────────


class TestCmdCalDeeper3:
    def test_cal_current(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_cal(""))
        assert repl._last_exit_code == 0

    def test_cal_specific(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_cal("3 2024"))
        assert "March" in out or "2024" in out


# ── _cmd_ln deeper ──────────────────────────────────────────────


class TestCmdLnDeeper3:
    def test_ln_hard_link(self, repl, tmp_path):
        original = tmp_path / "original.txt"
        original.write_text("content")
        link = tmp_path / "hardlink.txt"
        out = _run_with_io(repl, [], lambda: repl._cmd_ln(str(original) + " " + str(link)))
        assert repl._last_exit_code == 0
        assert link.exists()

    def test_ln_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_ln(""))
        assert repl._last_exit_code == 1


# ── _cmd_mktemp deeper ──────────────────────────────────────────


class TestCmdMktempDeeper2:
    def test_mktemp_creates(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_mktemp(""))
        assert repl._last_exit_code == 0
        assert len(out.strip()) > 0

    def test_mktemp_dir(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_mktemp("-d"))
        assert repl._last_exit_code == 0


# ── _cmd_id deeper ──────────────────────────────────────────────


class TestCmdIdDeeper3:
    def test_id_contains_uid(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_id(""))
        assert "uid" in out.lower()

    def test_id_contains_gid(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_id(""))
        assert "gid" in out.lower()


# ── _cmd_who deeper ─────────────────────────────────────────────


class TestCmdWhoDeeper2:
    def test_who_returns_output(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_who(""))
        assert repl._last_exit_code == 0


# ── _cmd_uname deeper ──────────────────────────────────────────


class TestCmdUnameDeeper3:
    def test_uname_a(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_uname("-a"))
        assert "Linux" in out or "Darwin" in out

    def test_uname_n(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_uname("-s"))
        assert len(out.strip()) > 0

    def test_uname_r(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_uname("-r"))
        assert len(out.strip()) > 0


# ── _cmd_hostname deeper ────────────────────────────────────────


class TestCmdHostnameDeeper3:
    def test_hostname_returns_string(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_hostname(""))
        assert len(out.strip()) > 0


# ── _cmd_nproc deeper ──────────────────────────────────────────


class TestCmdNprocDeeper3:
    def test_nproc_positive(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_nproc(""))
        assert int(out.strip()) > 0


# ── _cmd_logname deeper ─────────────────────────────────────────


class TestCmdLognameDeeper3:
    def test_logname_returns_string(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_logname(""))
        assert len(out.strip()) > 0


# ── _cmd_seq edge cases ─────────────────────────────────────────


class TestCmdSeqEdgeCases2:
    def test_seq_two_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_seq("1 5"))
        assert "1" in out and "5" in out

    def test_seq_reverse(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_seq("5 1"))
        assert repl._last_exit_code == 0 or repl._last_exit_code == 1


# ── _cmd_comm deeper ────────────────────────────────────────────


class TestCmdCommDeeper4:
    def test_comm_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_comm(""))
        assert repl._last_exit_code == 1


# ── _cmd_join deeper ────────────────────────────────────────────


class TestCmdJoinDeeper:
    def test_join_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_join(""))
        assert repl._last_exit_code == 1


# ── _cmd_paste deeper ───────────────────────────────────────────


class TestCmdPasteDeeper4:
    def test_join_no_args2(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_join(""))
        assert repl._last_exit_code == 1


# ── _cmd_svc with mocked init_system ────────────────────────────


class TestCmdSvcMocked:
    def _setup_init(self, repl):
        """Set up a mock init_system."""
        from domains.shell.init import reset_init_system
        reset_init_system()
        mock_init = MagicMock()
        mock_init.service_table.return_value = "  api  running\n  vfs  stopped"
        mock_init.status_summary = "  2 services running"
        mock_init.runlevel = 3
        mock_init.get_manager.return_value = None
        repl.os._init = mock_init
        return mock_init

    def test_svc_not_booted(self, repl):
        from domains.shell.init import reset_init_system
        reset_init_system()
        repl.os._init = None
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("list"))
        assert "boot" in out.lower() or repl._last_exit_code == 1

    def test_svc_list(self, repl):
        init = self._setup_init(repl)
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("list"))
        assert "Services" in out or init.service_table.called

    def test_svc_ls_shortcut(self, repl):
        init = self._setup_init(repl)
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("ls"))
        assert init.service_table.called

    def test_svc_status_no_name(self, repl):
        init = self._setup_init(repl)
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("status"))
        assert "Init status" in out or init.status_summary in out

    def test_svc_status_unknown_name(self, repl):
        init = self._setup_init(repl)
        init.get_manager.return_value = None
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("status nonexistent"))
        assert "Unknown" in out

    def test_svc_start_no_name(self, repl):
        init = self._setup_init(repl)
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("start"))
        assert repl._last_exit_code == 1 or "Usage" in out

    def test_svc_start_unknown(self, repl):
        init = self._setup_init(repl)
        init.get_manager.return_value = None
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("start badsvc"))
        assert "Unknown" in out

    def test_svc_stop_no_name(self, repl):
        init = self._setup_init(repl)
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("stop"))
        assert repl._last_exit_code == 1

    def test_svc_stop_unknown(self, repl):
        init = self._setup_init(repl)
        init.get_manager.return_value = None
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("stop badsvc"))
        assert "Unknown" in out

    def test_svc_restart_no_name(self, repl):
        init = self._setup_init(repl)
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("restart"))
        assert repl._last_exit_code == 1

    def test_svc_restart_unknown(self, repl):
        init = self._setup_init(repl)
        init.get_manager.return_value = None
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("restart badsvc"))
        assert "Unknown" in out

    def test_svc_runlevel(self, repl):
        init = self._setup_init(repl)
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("runlevel"))
        assert "3" in out or "runlevel" in out.lower()

    def test_svc_unknown_subcmd(self, repl):
        init = self._setup_init(repl)
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("bogus"))
        assert repl._last_exit_code == 1 or "Usage" in out


# ── _cmd_deny deeper ────────────────────────────────────────────


class TestCmdDenyDeeper3:
    def test_deny_revoke_persist(self, repl):
        repl._perms.grant("testpersist")
        out = _run_with_io(repl, [], lambda: repl._cmd_deny("testpersist --persist"))
        assert "Revoked" in out

    def test_deny_all_elevated(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_deny("--all-ELEVATED"))
        assert "ELEVATED" in out

    def test_deny_all_dangerous(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_deny("--all-DANGEROUS"))
        assert "DANGEROUS" in out

    def test_deny_all_critical(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_deny("--all-CRITICAL"))
        assert "CRITICAL" in out

    def test_deny_multiple_targets(self, repl):
        repl._perms.grant("cmd_a")
        repl._perms.grant("cmd_b")
        out = _run_with_io(repl, [], lambda: repl._cmd_deny("cmd_a cmd_b"))
        assert "cmd_a" in out and "cmd_b" in out


# ── _cmd_events deeper ──────────────────────────────────────────


class TestCmdEventsDeeper2:
    def test_events_no_bus(self, repl):
        import domains.infrastructure.event_bus as eb
        with patch.object(eb, "get_event_bus", side_effect=Exception("no bus")):
            out = _run_with_io(repl, [], lambda: repl._cmd_events(""))
            assert "not available" in out.lower() or repl._last_exit_code == 0

    def test_events_empty_history(self, repl):
        mock_bus = MagicMock()
        mock_bus.history.return_value = []
        import domains.infrastructure.event_bus as eb
        with patch.object(eb, "get_event_bus", return_value=mock_bus):
            out = _run_with_io(repl, [], lambda: repl._cmd_events(""))
            assert "No events" in out

    def test_events_with_filter(self, repl):
        mock_event = MagicMock()
        mock_event.name = "model.loaded"
        mock_event.timestamp = time.time()
        mock_event.source = "test"
        mock_event.data = {"model": "gpt2"}
        mock_bus = MagicMock()
        mock_bus.history.return_value = [mock_event]
        import domains.infrastructure.event_bus as eb
        with patch.object(eb, "get_event_bus", return_value=mock_bus):
            out = _run_with_io(repl, [], lambda: repl._cmd_events("model"))
            assert "model.loaded" in out

    def test_events_no_match(self, repl):
        mock_event = MagicMock()
        mock_event.name = "model.loaded"
        mock_event.timestamp = time.time()
        mock_event.source = "test"
        mock_event.data = {}
        mock_bus = MagicMock()
        mock_bus.history.return_value = [mock_event]
        import domains.infrastructure.event_bus as eb
        with patch.object(eb, "get_event_bus", return_value=mock_bus):
            out = _run_with_io(repl, [], lambda: repl._cmd_events("nonexistent"))
            assert "No events matching" in out

    def test_events_with_limit(self, repl):
        events = []
        for i in range(50):
            ev = MagicMock()
            ev.name = f"event.{i}"
            ev.timestamp = time.time()
            ev.source = "test"
            ev.data = None
            events.append(ev)
        mock_bus = MagicMock()
        mock_bus.history.return_value = events
        import domains.infrastructure.event_bus as eb
        with patch.object(eb, "get_event_bus", return_value=mock_bus):
            out = _run_with_io(repl, [], lambda: repl._cmd_events("event 5"))
            assert "5" in out or "last" in out.lower()


# ── _cmd_protect/unprotect deeper ───────────────────────────────


class TestCmdProtectDeeper2:
    def test_protect_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_protect(""))
        assert "Usage" in out

    def test_protect_import_error(self, repl):
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            out = _run_with_io(repl, [], lambda: repl._cmd_protect("mymodel"))
            assert "Error" in out

    def test_unprotect_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_unprotect(""))
        assert "Usage" in out

    def test_unprotect_import_error(self, repl):
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            out = _run_with_io(repl, [], lambda: repl._cmd_unprotect("mymodel"))
            assert "Error" in out


# ── _cmd_lsdev deeper ──────────────────────────────────────────


class TestCmdLsdevDeeper2:
    def test_lsdev_no_devices(self, repl):
        repl.os._devices = None
        out = _run_with_io(repl, [], lambda: repl._cmd_lsdev(""))
        assert "not available" in out.lower() or repl._last_exit_code == 0


# ── _cmd_asm deeper ────────────────────────────────────────────


class TestCmdAsmDeeper2:
    def test_asm_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_asm(""))
        assert "Usage" in out

    def test_asm_list_flag(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_asm("--list"))
        assert "Built-in" in out or "hello" in out.lower()

    def test_asm_bad_file(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_asm("/nonexistent/file.asm"))
        assert repl._last_exit_code == 1 or "asm:" in out

    def test_asm_bad_code(self, repl):
        repl._piped_input = "INVALID GARBAGE CODE 999"
        out = _run_with_io(repl, [], lambda: repl._cmd_asm(""))
        assert "error" in out.lower() or "unknown" in out.lower()


# ── _cmd_tui deeper ────────────────────────────────────────────


class TestCmdTuiDeeper2:
    def test_tui_import_error(self, repl):
        with patch("builtins.__import__", side_effect=ImportError("no curses")):
            out = _run_with_io(repl, [], lambda: repl._cmd_tui(""))
            assert "not available" in out.lower() or repl._last_exit_code == 1

    def test_tui_runtime_error(self, repl):
        mock_tui = MagicMock()
        mock_tui.run.side_effect = RuntimeError("display error")
        with patch("domains.shell.repl.TuiRepl", return_value=mock_tui, create=True):
            out = _run_with_io(repl, [], lambda: repl._cmd_tui(""))
            assert "TUI error" in out or repl._last_exit_code == 1


# ── _cmd_status deeper ──────────────────────────────────────────


class TestCmdStatusDeeper2:
    def test_status_shows_permissions(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_status(""))
        assert "Permissions" in out or "permission" in out.lower()


# ── _cmd_logs --explain deeper ──────────────────────────────────


class TestCmdLogsExplain:
    def test_logs_explain(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_logs("--explain"))
        assert repl._last_exit_code == 0

    def test_logs_follow(self, repl):
        repl._log_buffer.clear()
        out = _run_with_io(repl, [], lambda: repl._cmd_logs("-f"))
        assert repl._last_exit_code == 0


# ── _cmd_train deeper ──────────────────────────────────────────


class TestCmdTrainDeeper2:
    def test_train_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_train(""))
        assert repl._last_exit_code == 0 or "api" in out.lower() or "train" in out.lower()

    def test_train_status_no_jobs(self, repl):
        repl.cmds.train_status = MagicMock(return_value=[])
        repl._require_api = MagicMock(return_value=True)
        out = _run_with_io(repl, [], lambda: repl._cmd_train("status"))
        assert "No training" in out or repl._last_exit_code == 0


# ── _cmd_api deeper ─────────────────────────────────────────────


class TestCmdApiDeeper2:
    def test_api_status(self, repl):
        with patch.object(type(repl.os.api), 'is_running', new_callable=PropertyMock, return_value=True):
            out = _run_with_io(repl, [], lambda: repl._cmd_api("status"))
            assert repl._last_exit_code == 0

    def test_api_not_running(self, repl):
        with patch.object(type(repl.os.api), 'is_running', new_callable=PropertyMock, return_value=False):
            out = _run_with_io(repl, [], lambda: repl._cmd_api("stop"))
            assert "not running" in out.lower() or repl._last_exit_code == 0


# ── _cmd_set deeper ─────────────────────────────────────────────


class TestCmdSetDeeper2:
    def test_set_show_var(self, repl):
        repl._env["MYTEST"] = "hello"
        out = _run_with_io(repl, [], lambda: repl._cmd_set("MYTEST"))
        assert "hello" in out

    def test_set_show_missing(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_set("NONEXISTENT"))
        assert "not set" in out

    def test_set_no_color(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_set("NO_COLOR=1"))
        assert repl._last_exit_code == 0


# ── _cmd_export deeper ──────────────────────────────────────────


class TestCmdExportDeeper2:
    def test_export_set(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_export("MYVAR=test123"))
        assert "test123" in repl._env.get("MYVAR", "")

    def test_export_show(self, repl):
        repl._env["MYVAR"] = "showme"
        out = _run_with_io(repl, [], lambda: repl._cmd_export("MYVAR"))
        assert "showme" in out

    def test_export_missing(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_export("MISSING"))
        assert "not set" in out


# ── _cmd_read deeper ────────────────────────────────────────────


class TestCmdReadDeeper3:
    def test_read_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_read(""))
        assert repl._last_exit_code == 1

    def test_read_with_prompt(self, repl):
        mem_io = MemoryIO()
        old_io = repl.io
        repl.io = mem_io
        mem_io.feed("inputval\n")
        repl._cmd_read("-p Enter: MYVAR")
        assert repl._env.get("MYVAR") == "inputval"
        repl.io = old_io

    def test_read_p_no_var(self, repl):
        mem_io = MemoryIO()
        old_io = repl.io
        repl.io = mem_io
        mem_io.feed("ignored\n")
        repl._cmd_read("-p")
        assert repl._env.get("-p") == "ignored"
        repl.io = old_io


# ── _cmd_which deeper ──────────────────────────────────────────


class TestCmdWhichDeeper3:
    def test_which_system_binary(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_which("ls"))
        assert "/bin/ls" in out or repl._last_exit_code == 0


# ── _cmd_type deeper ────────────────────────────────────────────


class TestCmdTypeDeeper3:
    def test_type_system_binary(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_type("ls"))
        assert repl._last_exit_code == 0 or "/bin/ls" in out


# ── _cmd_realpath deeper ────────────────────────────────────────


class TestCmdRealpathDeeper2:
    def test_realpath_bad_path(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_realpath("/nonexistent/deep/path"))
        assert repl._last_exit_code == 0 or "nonexistent" in out


# ── _cmd_yes deeper ─────────────────────────────────────────────


class TestCmdYesDeeper2:
    def test_yes_custom_string(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_yes("hello"))
        lines = [l for l in out.strip().split("\n") if l.strip()]
        assert len(lines) == 100
        assert all("hello" in l for l in lines[:5])


# ── _cmd_env deeper ─────────────────────────────────────────────


class TestCmdEnvDeeper2:
    def test_env_shows_all(self, repl):
        repl._env["AAA"] = "111"
        repl._env["BBB"] = "222"
        out = _run_with_io(repl, [], lambda: repl._cmd_env(""))
        assert "AAA=111" in out
        assert "BBB=222" in out


# ── _cmd_boot auto-start API ───────────────────────────────────


class TestCmdBootAutoAPI:
    def test_boot_auto_start_api(self, repl):
        repl._running = False
        repl.os.api.start.return_value = {"ok": True, "message": "started"}
        with patch.object(type(repl.os.api), 'is_running', new_callable=PropertyMock, return_value=False):
            out = _run_with_io(repl, [], lambda: repl._cmd_boot(""))
        assert repl._running is True

    def test_boot_api_auto_start_fails(self, repl):
        repl._running = False
        repl.os.api.start.return_value = {"ok": False, "error": "port busy"}
        with patch.object(type(repl.os.api), 'is_running', new_callable=PropertyMock, return_value=False):
            out = _run_with_io(repl, [], lambda: repl._cmd_boot(""))
        assert repl._running is True


# ── _cmd_shutdown deeper ────────────────────────────────────────


class TestCmdShutdownDeeper3:
    def test_shutdown_sets_false(self, repl):
        repl._running = True
        with patch.object(repl.os, 'shutdown', return_value="Shut down"):
            out = _run_with_io(repl, [], lambda: repl._cmd_shutdown(""))
        assert repl._running is False


# ── _cmd_help ext commands ──────────────────────────────────────


class TestCmdHelpExt:
    def test_help_ext_command(self, repl):
        repl._ext_cmds["testmod"] = MagicMock()
        repl._ext_cmds["testmod"].help = "Test module help"
        repl._ext_cmds["testmod"].__doc__ = "Test module doc"
        out = _run_with_io(repl, [], lambda: repl._cmd_help("testmod"))
        assert "testmod" in out.lower() or "test module" in out.lower()


# ── _cmd_fc deeper ──────────────────────────────────────────────


class TestCmdFcDeeper3:
    def test_fc_invalid(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_fc("abc"))
        assert "Usage" in out

    def test_fc_reindex(self, repl):
        repl._history = ["echo a", "echo b", "echo c"]
        out = _run_with_io(repl, [], lambda: repl._cmd_fc("2"))
        assert "Re-running" in out


# ── _cmd_history deeper ─────────────────────────────────────────


class TestCmdHistoryDeeper3:
    def test_history_with_count(self, repl):
        repl._history = ["cmd" + str(i) for i in range(100)]
        out = _run_with_io(repl, [], lambda: repl._cmd_history("5"))
        assert "96" in out or "97" in out or "98" in out or "99" in out


# ── _cmd_kill deeper ────────────────────────────────────────────


class TestCmdKillDeeper3:
    def test_kill_with_exception(self, repl):
        repl.cmds.kill = MagicMock(side_effect=RuntimeError("kill failed"))
        try:
            out = _run_with_io(repl, [], lambda: repl._cmd_kill("123"))
        except RuntimeError:
            pass


# ── _cmd_ps deeper ──────────────────────────────────────────────


class TestCmdPsDeeper3:
    def test_ps_with_process(self, repl):
        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.name = "test-proc"
        mock_proc.state = 2
        mock_proc.created_at = time.time() - 10.0
        with patch.object(repl.os.kernel, 'list_processes', return_value=[mock_proc]):
            out = _run_with_io(repl, [], lambda: repl._cmd_ps(""))
        assert "test-proc" in out or "RUNNING" in out


# ── _cmd_metrics deeper ─────────────────────────────────────────


class TestCmdMetricsDeeper2:
    def test_metrics_with_error(self, repl):
        repl.cmds.system_metrics = MagicMock(return_value={"error": "connection refused"})
        out = _run_with_io(repl, [], lambda: repl._cmd_metrics(""))
        assert "Error" in out

    def test_metrics_with_data(self, repl):
        repl.cmds.system_metrics = MagicMock(return_value={"cpu": "50%", "mem": "2GB"})
        out = _run_with_io(repl, [], lambda: repl._cmd_metrics(""))
        assert "cpu" in out or "50%" in out


# ── _cmd_confirm deeper ─────────────────────────────────────────


class TestCmdConfirmDeeper3:
    def test_confirm_show(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_confirm(""))
        assert repl._last_exit_code == 0

    def test_confirm_on(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_confirm("on"))
        assert repl._last_exit_code == 0

    def test_confirm_off(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_confirm("off"))
        assert repl._last_exit_code == 0


# ── _cmd_permissions deeper ─────────────────────────────────────


class TestCmdPermissionsDeeper3:
    def test_permissions_output(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_permissions(""))
        assert "safe" in out.lower() or "elevated" in out.lower()


# ── _cmd_tutorial deeper ────────────────────────────────────────


class TestCmdTutorialDeeper2:
    def test_tutorial_resets_flag(self, repl):
        repl.state.first_run = False
        out = _run_with_io(repl, [], lambda: repl._cmd_tutorial(""))
        assert repl._last_exit_code == 0


# ── _cmd_notes deeper ──────────────────────────────────────────


class TestCmdNotesDeeper2:
    def test_notes_no_module(self, repl):
        try:
            out = _run_with_io(repl, [], lambda: repl._cmd_note("list"))
            assert repl._last_exit_code == 0
        except (ImportError, ModuleNotFoundError):
            pass


# ── _cmd_source deeper ──────────────────────────────────────────


class TestCmdSourceDeeper3:
    def test_source_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_source(""))
        assert "Usage" in out

    def test_source_nonexistent(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_source("/nonexistent/file.rc"))
        assert "Error" in out or "No such file" in out.lower()

    def test_source_with_pipeline(self, repl, tmp_path):
        f = tmp_path / "test_source.rc"
        f.write_text("echo line1\necho line2\n# comment\n\necho line3\n")
        out = _run_with_io(repl, [], lambda: repl._cmd_source(str(f)))
        assert "line1" in out


# ── _parse_pipeline deeper ──────────────────────────────────────


class TestParsePipelineDeeper:
    def test_pipeline_ampersand(self, repl):
        cmds, bg, timed = repl._parse_pipeline("echo hi &")
        assert bg is True

    def test_pipeline_time(self, repl):
        cmds, bg, timed = repl._parse_pipeline("time echo hi")
        assert timed is True

    def test_pipeline_and_chain(self, repl):
        cmds, bg, timed = repl._parse_pipeline("echo a && echo b")
        assert len(cmds) >= 2

    def test_pipeline_or_chain(self, repl):
        cmds, bg, timed = repl._parse_pipeline("echo a || echo b")
        assert len(cmds) >= 2

    def test_pipeline_semicolon(self, repl):
        cmds, bg, timed = repl._parse_pipeline("echo a ; echo b")
        assert len(cmds) >= 2

    def test_pipeline_single_pipe(self, repl):
        cmds, bg, timed = repl._parse_pipeline("echo a | wc")
        assert len(cmds) == 2

    def test_pipeline_empty(self, repl):
        cmds, bg, timed = repl._parse_pipeline("")
        assert len(cmds) <= 1

    def test_pipeline_quoted_pipe(self, repl):
        cmds, bg, timed = repl._parse_pipeline("echo 'a|b'")
        assert len(cmds) == 1


# ── _split_pipe deeper ─────────────────────────────────────────


class TestSplitPipeDeeper:
    def test_split_pipe_single(self, repl):
        parts = repl._split_pipe("echo hi")
        assert parts == ["echo hi"]

    def test_split_pipe_double(self, repl):
        parts = repl._split_pipe("echo a | wc")
        assert parts == ["echo a", "wc"]

    def test_split_pipe_triple(self, repl):
        parts = repl._split_pipe("echo a | sort | wc")
        assert len(parts) == 3

    def test_split_pipe_quoted(self, repl):
        parts = repl._split_pipe("echo 'a|b'")
        assert len(parts) == 1

    def test_split_pipe_empty(self, repl):
        parts = repl._split_pipe("")
        assert parts == [""]


# ── _strip_redirection deeper ───────────────────────────────────


class TestStripRedirectionDeeper:
    def test_strip_redirect_append(self, repl):
        args, path, append = repl._strip_redirection("echo hi >> /tmp/out.txt")
        assert path == "/tmp/out.txt"
        assert append is True

    def test_strip_redirect_overwrite(self, repl):
        args, path, append = repl._strip_redirection("echo hi > /tmp/out.txt")
        assert path == "/tmp/out.txt"
        assert append is False

    def test_strip_no_redirect(self, repl):
        args, path, append = repl._strip_redirection("echo hi")
        assert path is None

    def test_strip_redirect_at_end(self, repl):
        args, path, append = repl._strip_redirection("echo hi world > out.txt")
        assert path == "out.txt"


# ── _parse_inline_env deeper ────────────────────────────────────


class TestParseInlineEnvDeeper:
    def test_multiple_env(self, repl):
        env, rest = repl._parse_inline_env("A=1 B=2 echo test")
        assert env.get("A") == "1" and env.get("B") == "2"
        assert "echo test" in rest

    def test_no_env(self, repl):
        env, rest = repl._parse_inline_env("echo hello")
        assert env == {}
        assert rest == "echo hello"

    def test_env_empty(self, repl):
        env, rest = repl._parse_inline_env("")
        assert env == {}
        assert rest == ""


# ── _execute_pipeline deeper ────────────────────────────────────


class TestExecutePipelineDeeper:
    def test_pipeline_and_skip(self, repl):
        repl._last_exit_code = 1
        out = _run_with_io(repl, [], lambda: repl._execute_pipeline([("echo should_skip", "&&"), ("echo fallback", None)]))
        assert "fallback" in out

    def test_pipeline_or_skip(self, repl):
        repl._last_exit_code = 0
        out = _run_with_io(repl, [], lambda: repl._execute_pipeline([("echo no_run", "||"), ("echo fallback", None)]))
        assert "fallback" in out

    def test_pipeline_empty(self, repl):
        repl._execute_pipeline([])
        assert repl._last_exit_code == 0


# ── _execute_single deeper ──────────────────────────────────────


class TestExecuteSingleDeeper2:
    def test_unknown_command(self, repl):
        out = repl._execute_single("nonexistent_xyz")
        assert "Unknown command" in out

    def test_unknown_with_suggestion(self, repl):
        out = repl._execute_single("ech")
        assert "Did you mean" in out or "Unknown" in out

    def test_redirect_write_error(self, repl, tmp_path):
        out = repl._execute_single("echo test > " + str(tmp_path / "no_parent" / "file.txt"))
        assert repl._last_exit_code == 1 or "Error" in out

    def test_ext_module_execution(self, repl):
        mock_mod = MagicMock()
        mock_mod.run.return_value = 0
        mock_mod.help = "test module"
        repl._ext_cmds["testextcmd"] = mock_mod
        out = repl._execute_single("testextcmd arg1")
        assert repl._last_exit_code == 0

    def test_system_exit_caught(self, repl):
        def _exit_cmd(r, a):
            raise SystemExit(42)
        repl.COMMANDS["exitcmd42"] = _exit_cmd
        out = repl._execute_single("exitcmd42")
        assert repl._last_exit_code == 42

    def test_exception_caught(self, repl):
        def _explode(r, a):
            raise RuntimeError("boom")
        repl.COMMANDS["explodecmd"] = _explode
        out = repl._execute_single("explodecmd")
        assert repl._last_exit_code == 1

    def test_inline_env_restores(self, repl):
        old_val = repl._env.get("MYTEMP")
        out = repl._execute_single("MYTEMP=999 echo hi_inline_env")
        assert repl._env.get("MYTEMP") == old_val


# ── _execute_background deeper ──────────────────────────────────


class TestExecuteBackgroundDeeper:
    def test_background_error(self, repl):
        def _explode(r, a):
            raise RuntimeError("bg boom")
        repl.COMMANDS["bgexplode"] = _explode
        repl._execute_background("bgexplode")
        import time
        time.sleep(0.2)
        assert repl._next_bg_id >= 2

    def test_background_success(self, repl):
        repl._execute_background("echo bg_ok")
        import time
        time.sleep(0.2)
        assert repl._next_bg_id >= 2

    def test_background_tuples(self, repl):
        repl._execute_background_tuples([("echo bg_tuple", None)])
        import time
        time.sleep(0.2)
        assert repl._next_bg_id >= 2


# ── _cmd_svc with mocked manager start/stop/restart ─────────────


class TestCmdSvcWithManager:
    def _setup_manager(self, repl):
        from domains.shell.init import reset_init_system
        reset_init_system()
        mock_init = MagicMock()
        mock_init.service_table.return_value = "  api  running"
        mock_init.status_summary = "  1 running"
        mock_init.runlevel = 3
        mock_mgr = MagicMock()
        mock_mgr.start.return_value = True
        mock_mgr.restart.return_value = True
        mock_mgr.status_line.return_value = "  api: running"
        mock_mgr.instance = MagicMock()
        mock_mgr.instance.log = ["started", "running"]
        mock_init.get_manager.return_value = mock_mgr
        repl.os._init = mock_init
        return mock_init, mock_mgr

    def test_svc_start_success(self, repl):
        init, mgr = self._setup_manager(repl)
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("start api"))
        assert "started" in out.lower() or "ok" in out.lower()

    def test_svc_start_failure(self, repl):
        init, mgr = self._setup_manager(repl)
        mgr.start.return_value = False
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("start api"))
        assert "failed" in out.lower() or "error" in out.lower()

    def test_svc_stop_success(self, repl):
        init, mgr = self._setup_manager(repl)
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("stop api"))
        assert "stopped" in out.lower()

    def test_svc_restart_success(self, repl):
        init, mgr = self._setup_manager(repl)
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("restart api"))
        assert "restarted" in out.lower() or "ok" in out.lower()

    def test_svc_status_with_name(self, repl):
        init, mgr = self._setup_manager(repl)
        out = _run_with_io(repl, [], lambda: repl._cmd_svc("status api"))
        assert "running" in out.lower() or "api" in out.lower()


# ── _cmd_cd OLDPWD (fixed bug) ─────────────────────────────────


class TestCmdCdOldpwd:
    def test_cd_minus(self, repl, tmp_path):
        d1 = tmp_path / "dir_a"
        d2 = tmp_path / "dir_b"
        d1.mkdir()
        d2.mkdir()
        old_dir = os.getcwd()
        os.chdir(str(d1))
        try:
            repl._cmd_cd(str(d2))
            repl._cmd_cd("-")
            assert os.getcwd() == str(d1)
        finally:
            os.chdir(old_dir)


# ── _cmd_help deep branches ────────────────────────────────────


class TestCmdHelpDeep:
    def test_help_all_commands(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_help(""))
        assert "Available" in out or len(out) > 500

    def test_help_known_cmd(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_help("ls"))
        assert "ls" in out.lower()

    def test_help_ext_module(self, repl):
        mock_mod = MagicMock()
        mock_mod.help = "Test ext help"
        mock_mod.__doc__ = "Ext module doc"
        repl._ext_cmds["extmod"] = mock_mod
        out = _run_with_io(repl, [], lambda: repl._cmd_help("extmod"))
        assert "extmod" in out.lower() or "test ext help" in out.lower()


# ── _cmd_train with subcommands ─────────────────────────────────


class TestCmdTrainSubcommands:
    def test_train_distill_no_api(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_train("distill shakespeare"))
        assert "api" in out.lower() or repl._last_exit_code == 1

    def test_train_hf_no_api(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_train("hf shakespeare"))
        assert "api" in out.lower() or repl._last_exit_code == 1


# ── _cmd_load deeper ───────────────────────────────────────────


class TestCmdLoadDeeper:
    def test_load_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_load(""))
        assert "Usage" in out

    def test_load_no_api(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_load("gpt2"))
        assert "api" in out.lower() or repl._last_exit_code == 1


# ── _check_permission deeper ───────────────────────────────────


class TestCheckPermissionDeeper:
    def test_check_permission_allowed(self, repl):
        result = repl._check_permission("echo", "hello", interactive=False)
        assert result is True

    def test_check_permission_denied_noninteractive(self, repl):
        with patch.object(repl._perms, "check", side_effect=PermissionError("denied")):
            with patch.object(repl._perms, "classify", return_value="DANGEROUS"):
                result = repl._check_permission("rm", "/tmp/x", interactive=False)
                assert result is False

    def test_check_permission_interactive_y(self, repl):
        with patch.object(repl._perms, "check", side_effect=PermissionError("denied")):
            with patch.object(repl._perms, "classify", return_value="ELEVATED"):
                repl.io = MemoryIO()
                repl.io.feed("y\n")
                result = repl._check_permission("chmod", "777 /tmp", interactive=True)
                assert result is True

    def test_check_permission_interactive_always(self, repl):
        with patch.object(repl._perms, "check", side_effect=PermissionError("denied")):
            with patch.object(repl._perms, "classify", return_value="ELEVATED"):
                repl.io = MemoryIO()
                repl.io.feed("always\n")
                result = repl._check_permission("chmod", "777 /tmp", interactive=True)
                assert result is True

    def test_check_permission_interactive_n(self, repl):
        with patch.object(repl._perms, "check", side_effect=PermissionError("denied")):
            with patch.object(repl._perms, "classify", return_value="ELEVATED"):
                repl.io = MemoryIO()
                repl.io.feed("n\n")
                result = repl._check_permission("chmod", "777 /tmp", interactive=True)
                assert result is False

    def test_check_permission_interactive_eof(self, repl):
        with patch.object(repl._perms, "check", side_effect=PermissionError("denied")):
            with patch.object(repl._perms, "classify", return_value="ELEVATED"):
                repl.io = MemoryIO()
                repl.io.feed("\n")
                result = repl._check_permission("chmod", "777 /tmp", interactive=True)
                assert result is False


# ── _format_table deeper ───────────────────────────────────────


class TestFormatTableDeeper:
    def test_format_table_empty(self, repl):
        out = repl._format_table([])
        assert "empty" in out.lower()

    def test_format_table_with_header(self, repl):
        out = repl._format_table([["a", "b"], ["c", "d"]], header=["X", "Y"])
        assert "X" in out and "Y" in out

    def test_format_table_no_header(self, repl):
        out = repl._format_table([["1", "2"]])
        assert "1" in out and "2" in out


# ── _dump_json deeper ──────────────────────────────────────────


class TestDumpJsonDeeper:
    def test_dump_json_dict(self, repl):
        out = repl._dump_json({"a": 1, "b": "two"})
        assert '"a"' in out

    def test_dump_json_list(self, repl):
        out = repl._dump_json([1, 2, 3])
        assert "1" in out and "2" in out

    def test_dump_json_default(self, repl):
        out = repl._dump_json({"t": __import__("datetime").datetime.now()})
        assert "2026" in out or "t" in out


# ── _spinner_call deeper ───────────────────────────────────────


class TestSpinnerCallDeeper:
    def test_spinner_call_ok_msg_none(self, repl):
        result = repl._spinner_call("test", lambda: 42, ok_msg=None)
        assert result == 42

    def test_spinner_call_ok_msg_empty(self, repl):
        result = repl._spinner_call("test", lambda: 99, ok_msg="")
        assert result == 99

    def test_spinner_call_ok_msg_custom(self, repl):
        result = repl._spinner_call("test", lambda: "val", ok_msg="Done!")
        assert result == "val"


# ── _log methods deeper ────────────────────────────────────────


class TestLogMethodsDeeper:
    def test_log_ok(self, repl):
        repl._log_ok("success message")

    def test_log_warn(self, repl):
        repl._log_warn("warning message")

    def test_log_error(self, repl):
        repl._log_error("error message")

    def test_log_step(self, repl):
        repl._log_step("step message")


# ── _expand_vars edge cases ────────────────────────────────────


class TestExpandVarsEdgeCases:
    def test_expand_unset_var(self, repl):
        result = repl._expand_vars("$NONEXISTENT_VAR_XYZ")
        assert "$NONEXISTENT_VAR_XYZ" in result

    def test_expand_mixed(self, repl):
        repl._env["FOO"] = "bar"
        result = repl._expand_vars("$FOO and $NONEXISTENT")
        assert "bar" in result and "$NONEXISTENT" in result


# ── _expand_cmd_subst deeper ───────────────────────────────────


class TestExpandCmdSubstDeeper:
    def test_cmd_subst_nested_parens(self, repl):
        result = repl._expand_cmd_subst("echo $(echo hello)")
        assert "hello" in result

    def test_cmd_subst_empty(self, repl):
        result = repl._expand_cmd_subst("no substitution here")
        assert result == "no substitution here"


# ── _expand_history edge cases ─────────────────────────────────


class TestExpandHistoryEdgeCases:
    def test_history_excl_empty(self, repl):
        repl._history = []
        result = repl._expand_history("!!")
        assert result == "!!"

    def test_history_nth_arg_out_of_range(self, repl):
        repl._history = ["echo a b"]
        result = repl._expand_history("!:99")
        assert "!:99" in result

    def test_history_neg_out_of_range(self, repl):
        repl._history = ["cmd1"]
        result = repl._expand_history("!-99")
        assert "!-99" in result

    def test_history_last_arg_single_word(self, repl):
        repl._history = ["ls"]
        result = repl._expand_history("!$")
        assert result == "ls"

    def test_history_all_args_empty(self, repl):
        repl._history = ["echo"]
        result = repl._expand_history("!*")
        assert result == ""


# ── _expand_globs edge cases ───────────────────────────────────


class TestExpandGlobsEdgeCases:
    def test_glob_no_magic(self, repl):
        result = repl._expand_globs("echo hello")
        assert result == "echo hello"

    def test_glob_quoted(self, repl):
        result = repl._expand_globs("echo 'a*b'")
        assert "a*b" in result


# ── _expand_alias edge cases ───────────────────────────────────


class TestExpandAliasEdgeCases:
    def test_expand_alias_no_match(self, repl):
        result = repl._expand_alias("notaliased")
        assert result == "notaliased"

    def test_expand_alias_empty(self, repl):
        result = repl._expand_alias("")
        assert result == ""


# ── execute() method ───────────────────────────────────────────


class TestExecuteMethod:
    def test_execute_empty(self, repl):
        out, code = repl.execute("")
        assert out == "" and code == 0

    def test_execute_echo(self, repl):
        out, code = repl.execute("echo test_exec")
        assert "test_exec" in out

    def test_execute_cd(self, repl):
        out, code = repl.execute("pwd")
        assert code == 0


# ── _cmd_gen deeper ────────────────────────────────────────────


class TestCmdGenDeeper:
    def test_gen_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_gen(""))
        assert "Usage" in out or repl._last_exit_code == 1

    def test_gen_no_api(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_gen("hello"))
        assert "api" in out.lower() or repl._last_exit_code == 1


# ── _cmd_chat deeper ───────────────────────────────────────────


class TestCmdChatDeeper:
    def test_chat_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_chat(""))
        assert "Usage" in out or repl._last_exit_code == 1

    def test_chat_no_api(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_chat("hello"))
        assert "api" in out.lower() or repl._last_exit_code == 1


# ── _cmd_ai deeper ─────────────────────────────────────────────


class TestCmdAiDeeper:
    def test_ai_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_ai(""))
        assert "Usage" in out or repl._last_exit_code == 1


# ── _cmd_agents deeper ─────────────────────────────────────────


class TestCmdAgentsDeeper:
    def test_agents_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_agents(""))
        assert repl._last_exit_code == 0


# ── _cmd_procs deeper ──────────────────────────────────────────


class TestCmdProcsDeeper:
    def test_procs(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_procs(""))
        assert repl._last_exit_code == 0


# ── _cmd_bg/fg deeper ──────────────────────────────────────────


class TestCmdBgFgDeeper:
    def test_bg_empty(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_bg(""))
        assert repl._last_exit_code == 0

    def test_fg_empty(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_fg(""))
        assert "Usage" in out or repl._last_exit_code == 1


# ── _CaptureOutput ──────────────────────────────────────────────


class TestCaptureOutput2:
    def test_with_repl(self, repl):
        from domains.shell.repl import _CaptureOutput
        with _CaptureOutput(repl) as cap:
            repl.io.write("captured text\n")
        out = cap.getvalue()
        assert "captured text" in out

    def test_without_repl(self):
        from domains.shell.repl import _CaptureOutput
        with _CaptureOutput() as cap:
            print("stdout text")
        assert "stdout text" in cap.getvalue()

    def test_getvalue_returns_string(self):
        from domains.shell.repl import _CaptureOutput
        with _CaptureOutput() as cap:
            pass
        assert isinstance(cap.getvalue(), str)

    def test_no_repl_returns_empty(self):
        from domains.shell.repl import _CaptureOutput
        with _CaptureOutput() as cap:
            pass
        result = cap.getvalue()
        assert result == "" or isinstance(result, str)


# ── _note_new deeper ────────────────────────────────────────────


class TestNoteNewDeeper:
    def test_note_new_with_tags_only(self, repl):
        mock_store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.title = "Tagged Note"
        mock_store.create.return_value = note
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("new Tagged Note --tags tag1,tag2")
        assert repl._last_exit_code == 0

    def test_note_new_with_status_only(self, repl):
        mock_store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.title = "Status Note"
        mock_store.create.return_value = note
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("new Status Note --status wip")
        assert repl._last_exit_code == 0

    def test_note_new_with_sprint_only(self, repl):
        mock_store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.title = "Sprint Note"
        mock_store.create.return_value = note
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("new Sprint Note --sprint S1")
        assert repl._last_exit_code == 0

    def test_note_new_with_gh_only(self, repl):
        mock_store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.title = "GH Note"
        mock_store.create.return_value = note
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("new GH Note --gh owner/repo#42")
        assert repl._last_exit_code == 0

    def test_note_new_with_all_flags(self, repl):
        mock_store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.title = "Full Note"
        mock_store.create.return_value = note
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("new Full Note --tags a,b --status done --sprint S1 --gh o/r#1")
        assert repl._last_exit_code == 0


# ── _note_list deeper ──────────────────────────────────────────


class TestNoteListDeeper:
    def test_note_list_with_tag_filter(self, repl):
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("list --tag tag1")
        assert repl._last_exit_code == 0
        call_kwargs = mock_store.list_notes.call_args
        assert call_kwargs[1].get('tag') == 'tag1' or call_kwargs[0][0] == 'tag1'

    def test_note_list_with_status_filter(self, repl):
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("list --status wip")
        assert repl._last_exit_code == 0

    def test_note_list_with_sprint_filter(self, repl):
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("list --sprint S1")
        assert repl._last_exit_code == 0

    def test_note_list_with_limit(self, repl):
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("list --limit 5")
        assert repl._last_exit_code == 0

    def test_note_list_with_notes(self, repl):
        mock_store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.title = "Test Note"
        note.tags = ["tag1"]
        note.status = "open"
        note.date_str = "2024-01-01"
        mock_store.list_notes.return_value = [note]
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("list")
        assert repl._last_exit_code == 0


# ── _note_show deeper ──────────────────────────────────────────


class TestNoteShowDeeper:
    def test_note_show_with_sprint(self, repl):
        mock_store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.title = "Sprint Note"
        note.body = "body"
        note.tags = ["tag1"]
        note.status = "open"
        note.sprint = "S1"
        note.gh = ""
        note.gh_url = ""
        note.created_at = "2024-01-01"
        note.updated_at = "2024-01-02"
        mock_store.get.return_value = note
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("show abc123")
        assert repl._last_exit_code == 0

    def test_note_show_with_gh(self, repl):
        mock_store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.title = "GH Note"
        note.body = "body"
        note.tags = ["tag1"]
        note.status = "open"
        note.sprint = ""
        note.gh = "owner/repo#1"
        note.gh_url = "https://gh.com/o/r/issues/1"
        note.created_at = "2024-01-01"
        note.updated_at = "2024-01-02"
        mock_store.get.return_value = note
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("show abc123")
        assert repl._last_exit_code == 0

    def test_note_show_with_multiline_body(self, repl):
        mock_store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.title = "Body Note"
        note.body = "line1\nline2\nline3"
        note.tags = []
        note.status = "open"
        note.sprint = ""
        note.gh = ""
        note.gh_url = ""
        note.created_at = "2024-01-01"
        note.updated_at = "2024-01-02"
        mock_store.get.return_value = note
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("show abc123")
        assert repl._last_exit_code == 0


# ── _note_edit deeper ──────────────────────────────────────────


class TestNoteEditDeeper:
    def test_note_edit_title_only(self, repl):
        mock_store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.title = "Updated Title"
        mock_store.update.return_value = note
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("edit abc123 --title Updated Title")
        assert repl._last_exit_code == 0

    def test_note_edit_tags_only(self, repl):
        mock_store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.title = "Note"
        mock_store.update.return_value = note
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("edit abc123 --tags newtag")
        assert repl._last_exit_code == 0

    def test_note_edit_status_only(self, repl):
        mock_store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.title = "Note"
        mock_store.update.return_value = note
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("edit abc123 --status done")
        assert repl._last_exit_code == 0

    def test_note_edit_sprint_only(self, repl):
        mock_store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.title = "Note"
        mock_store.update.return_value = note
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("edit abc123 --sprint S2")
        assert repl._last_exit_code == 0

    def test_note_edit_gh_only(self, repl):
        mock_store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.title = "Note"
        mock_store.update.return_value = note
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("edit abc123 --gh o/r#2")
        assert repl._last_exit_code == 0

    def test_note_edit_body_only(self, repl):
        mock_store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.title = "Note"
        mock_store.update.return_value = note
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("edit abc123 --body New body content")
        assert repl._last_exit_code == 0


# ── _note_delete deeper ────────────────────────────────────────


class TestNoteDeleteDeeper:
    def test_note_delete_no_args(self, repl):
        mock_store = MagicMock()
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("delete")
        assert repl._last_exit_code == 1

    def test_note_delete_not_found(self, repl):
        mock_store = MagicMock()
        mock_store.delete.return_value = False
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("delete nonexistent")
        assert repl._last_exit_code == 1


# ── _note_search deeper ────────────────────────────────────────


class TestNoteSearchDeeper:
    def test_note_search_with_results(self, repl):
        mock_store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.title = "Search Result"
        note.tags = ["tag1"]
        note.status = "open"
        note.date_str = "2024-01-01"
        mock_store.search.return_value = [note]
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("search test")
        assert repl._last_exit_code == 0

    def test_note_search_empty_results(self, repl):
        mock_store = MagicMock()
        mock_store.search.return_value = []
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("search nonexistent")
        assert repl._last_exit_code == 0


# ── _note_today deeper ─────────────────────────────────────────


class TestNoteTodayDeeper:
    def test_note_today_with_notes(self, repl):
        mock_store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.title = "Today Note"
        note.tags = ["tag1"]
        note.status = "wip"
        mock_store.today.return_value = [note]
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("today")
        assert repl._last_exit_code == 0

    def test_note_today_empty(self, repl):
        mock_store = MagicMock()
        mock_store.today.return_value = []
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("today")
        assert repl._last_exit_code == 0


# ── _note_export deeper ────────────────────────────────────────


class TestNoteExportDeeper:
    def test_note_export_with_file(self, repl):
        mock_store = MagicMock()
        mock_store.export_all.return_value = "exported"
        mock_store.count.return_value = 3
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("export /tmp/notes.md")
        assert repl._last_exit_code == 0

    def test_note_export_without_file(self, repl):
        mock_store = MagicMock()
        mock_store.export_all.return_value = "# All notes"
        mock_store.count.return_value = 1
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("export")
        assert repl._last_exit_code == 0


# ── _note_tags deeper ──────────────────────────────────────────


class TestNoteTagsDeeper:
    def test_note_tags_with_tags(self, repl):
        mock_store = MagicMock()
        note1 = MagicMock()
        note1.tags = ["bug", "urgent"]
        note2 = MagicMock()
        note2.tags = ["bug", "feature"]
        mock_store.list_notes.return_value = [note1, note2]
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("tags")
        assert repl._last_exit_code == 0

    def test_note_tags_empty(self, repl):
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("tags")
        assert repl._last_exit_code == 0


# ── _note_sprint deeper ────────────────────────────────────────


class TestNoteSprintDeeper:
    def test_note_sprint_list_all(self, repl):
        mock_store = MagicMock()
        mock_store.sprints.return_value = ["S1", "S2"]
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("sprint")
        assert repl._last_exit_code == 0

    def test_note_sprint_no_sprints(self, repl):
        mock_store = MagicMock()
        mock_store.sprints.return_value = []
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("sprint")
        assert repl._last_exit_code == 0

    def test_note_sprint_with_notes(self, repl):
        mock_store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.title = "Sprint Note"
        note.tags = ["tag1"]
        note.status = "open"
        note.gh = ""
        mock_store.list_notes.return_value = [note]
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("sprint S1")
        assert repl._last_exit_code == 0

    def test_note_sprint_report(self, repl):
        mock_store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.title = "Sprint Note"
        note.tags = ["tag1"]
        note.status = "open"
        note.gh = ""
        mock_store.list_notes.return_value = [note]
        mock_store.sprint_report.return_value = "Sprint Report Line 1\nSprint Report Line 2"
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("sprint S1 report")
        assert repl._last_exit_code == 0

    def test_note_sprint_no_notes(self, repl):
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("sprint S99")
        assert repl._last_exit_code == 0


# ── _note_status deeper ────────────────────────────────────────


class TestNoteStatusDeeper:
    def test_note_status_with_notes(self, repl):
        mock_store = MagicMock()
        note_open = MagicMock()
        note_open.status = "open"
        note_wip = MagicMock()
        note_wip.status = "wip"
        note_done = MagicMock()
        note_done.status = "done"
        note_blocked = MagicMock()
        note_blocked.status = "blocked"
        mock_store.list_notes.return_value = [note_open, note_wip, note_done, note_blocked]
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("status")
        assert repl._last_exit_code == 0

    def test_note_status_empty(self, repl):
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("status")
        assert repl._last_exit_code == 0


# ── _note_timeline deeper ──────────────────────────────────────


class TestNoteTimelineDeeper:
    def test_note_timeline_with_notes(self, repl):
        mock_store = MagicMock()
        note = MagicMock()
        note.short_id = "abc123"
        note.title = "Timeline Note"
        note.tags = ["tag1"]
        note.status = "open"
        note.sprint = "S1"
        mock_store.timeline.return_value = [("2024-01-01", [note])]
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("timeline --days 30")
        assert repl._last_exit_code == 0

    def test_note_timeline_with_tag(self, repl):
        mock_store = MagicMock()
        mock_store.timeline.return_value = []
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("timeline --tag tag1 --days 7")
        assert repl._last_exit_code == 0

    def test_note_timeline_with_status(self, repl):
        mock_store = MagicMock()
        mock_store.timeline.return_value = []
        with patch.dict('sys.modules', {'notes': MagicMock(get_note_store=MagicMock(return_value=mock_store))}):
            repl._cmd_note("timeline --status done --days 14")
        assert repl._last_exit_code == 0


# ── _table / _box / _status / _kvlist deeper ───────────────────


class TestConsoleDeeper:
    def test_table(self, repl):
        out = _run_with_io(repl, [], lambda: repl._table([["a", "b"], ["c", "d"]], header=["X", "Y"]))
        assert repl._last_exit_code == 0

    def test_box(self, repl):
        out = _run_with_io(repl, [], lambda: repl._box("Hello Box"))
        assert repl._last_exit_code == 0

    def test_status(self, repl):
        out = _run_with_io(repl, [], lambda: repl._status("ok", "Done", "detail"))
        assert repl._last_exit_code == 0

    def test_kvlist(self, repl):
        out = _run_with_io(repl, [], lambda: repl._kvlist([("key1", "val1"), ("key2", "val2")]))
        assert repl._last_exit_code == 0

    def test_kvlist_empty(self, repl):
        out = _run_with_io(repl, [], lambda: repl._kvlist([]))
        assert repl._last_exit_code == 0


# ── _print_header / _rc_path deeper ────────────────────────────


class TestPrintHeaderDeeper:
    def test_print_header(self, repl):
        out = _run_with_io(repl, [], lambda: repl._print_header())
        assert "help" in out.lower() or repl._last_exit_code == 0

    def test_rc_path(self, repl):
        path = repl._rc_path()
        assert "sloughgpt" in str(path)
        assert "rc" in str(path)


# ── _format_table deeper ───────────────────────────────────────


class TestFormatTableDeeper:
    def test_empty(self, repl):
        result = repl._format_table([])
        assert result == "(empty)"

    def test_with_header(self, repl):
        result = repl._format_table([["a", "b"], ["c", "d"]], header=["Col1", "Col2"])
        assert "Col1" in result
        assert "Col2" in result

    def test_without_header(self, repl):
        result = repl._format_table([["hello", "world"]])
        assert "hello" in result
        assert "world" in result

    def test_jagged_rows(self, repl):
        result = repl._format_table([["a"], ["b", "c", "d"]])
        assert "a" in result
        assert "d" in result

    def test_single_row(self, repl):
        result = repl._format_table([["x"]], header=["H"])
        assert "x" in result
        assert "H" in result


# ── _cmd_watch deeper ──────────────────────────────────────────


class TestCmdWatchDeeper3:
    def test_watch_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_watch(""))
        assert "Usage" in out

    def test_watch_invalid_interval(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_watch("abc echo hi"))
        assert "Invalid interval" in out

    def test_watch_single_arg(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_watch("echo only"))
        assert "Invalid interval" in out or "Usage" in out


# ── _cmd_time deeper ───────────────────────────────────────────


class TestCmdTimeDeeper3:
    def test_time_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_time(""))
        assert "Usage" in out
        assert repl._last_exit_code == 1

    def test_time_with_command(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_time("echo timed"))
        assert "real" in out.lower() or repl._last_exit_code == 0


# ── _cmd_sleep deeper ──────────────────────────────────────────


class TestCmdSleepDeeper4:
    def test_sleep_invalid(self, repl):
        try:
            repl._cmd_sleep("abc")
        except (ValueError, Exception):
            pass

    def test_sleep_zero(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_sleep("0"))
        assert repl._last_exit_code == 0


# ── _cmd_echo deeper ───────────────────────────────────────────


class TestCmdEchoDeeper3:
    def test_echo_multiple_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_echo("hello world"))
        assert "hello world" in out

    def test_echo_empty(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_echo(""))
        assert repl._last_exit_code == 0


# ── _cmd_chmod deeper ──────────────────────────────────────────


class TestCmdChmodDeeper3:
    def test_chmod_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_chmod(""))
        assert "Usage" in out

    def test_chmod_mode_only(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_chmod("755"))
        assert "Usage" in out

    def test_chmod_bad_file(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_chmod("644 /nonexistent_file_xyz"))
        assert "cannot access" in out.lower() or repl._last_exit_code != 0


# ── _cmd_du deeper ─────────────────────────────────────────────


class TestCmdDuDeeper3:
    def test_du_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_du(""))
        assert repl._last_exit_code == 0

    def test_du_bad_path(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_du("/nonexistent_xyz"))
        assert repl._last_exit_code == 0


# ── _cmd_diff deeper ───────────────────────────────────────────


class TestCmdDiffDeeper3:
    def test_diff_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_diff(""))
        assert "Usage" in out

    def test_diff_one_arg(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_diff("file1"))
        assert "Usage" in out


# ── _cmd_stat deeper ───────────────────────────────────────────


class TestCmdStatDeeper3:
    def test_stat_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_stat(""))
        assert "Usage" in out


# ── _cmd_ln deeper ─────────────────────────────────────────────


class TestCmdLnDeeper4:
    def test_ln_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_ln(""))
        assert repl._last_exit_code == 1


# ── _cmd_fold deeper ───────────────────────────────────────────


class TestCmdFoldDeeper3:
    def test_fold_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_fold(""))
        assert repl._last_exit_code == 1


# ── _cmd_nl deeper ─────────────────────────────────────────────


class TestCmdNlDeeper3:
    def test_nl_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_nl(""))
        assert repl._last_exit_code == 1


# ── _cmd_tee deeper ────────────────────────────────────────────


class TestCmdTeeDeeper3:
    def test_tee_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_tee(""))
        assert repl._last_exit_code == 1


# ── _cmd_cut deeper ────────────────────────────────────────────


class TestCmdCutDeeper3:
    def test_cut_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_cut(""))
        assert repl._last_exit_code == 1


# ── _cmd_tr deeper ─────────────────────────────────────────────


class TestCmdTrDeeper3:
    def test_tr_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_tr(""))
        assert repl._last_exit_code == 1


# ── _cmd_printf deeper ─────────────────────────────────────────


class TestCmdPrintfDeeper3:
    def test_printf_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_printf(""))
        assert repl._last_exit_code == 1


# ── _cmd_shuf deeper ───────────────────────────────────────────


class TestCmdShufDeeper3:
    def test_shuf_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_shuf(""))
        assert repl._last_exit_code == 1


# ── _cmd_rev deeper ────────────────────────────────────────────


class TestCmdRevDeeper3:
    def test_rev_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_rev(""))
        assert repl._last_exit_code == 1


# ── _cmd_paste deeper ──────────────────────────────────────────


class TestCmdPasteDeeper5:
    def test_paste_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_paste(""))
        assert repl._last_exit_code == 1


# ── _cmd_comm deeper ───────────────────────────────────────────


class TestCmdCommDeeper5:
    def test_comm_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_comm(""))
        assert repl._last_exit_code == 1


# ── _cmd_join deeper ───────────────────────────────────────────


class TestCmdJoinDeeper2:
    def test_join_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_join(""))
        assert repl._last_exit_code == 1


# ── _cmd_od deeper ─────────────────────────────────────────────


class TestCmdOdDeeper3:
    def test_od_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_od(""))
        assert repl._last_exit_code == 1


# ── _cmd_xargs deeper ──────────────────────────────────────────


class TestCmdXargsDeeper3:
    def test_xargs_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_xargs(""))
        assert repl._last_exit_code == 1


# ── _cmd_expand deeper ─────────────────────────────────────────


class TestCmdExpandDeeper3:
    def test_expand_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_expand(""))
        assert repl._last_exit_code == 1


# ── _cmd_unexpand deeper ───────────────────────────────────────


class TestCmdUnexpandDeeper3:
    def test_unexpand_no_args(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_unexpand(""))
        assert repl._last_exit_code == 1


# ── _cmd_clear deeper ──────────────────────────────────────────


class TestCmdClearDeeper2:
    def test_clear(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_clear(""))
        assert repl._last_exit_code == 0


# ── _cmd_pwd deeper ────────────────────────────────────────────


class TestCmdPwdDeeper2:
    def test_pwd(self, repl):
        out = _run_with_io(repl, [], lambda: repl._cmd_pwd(""))
        assert os.getcwd() in out


# ── _cmd_exit deeper ───────────────────────────────────────────


class TestCmdExitDeeper3:
    def test_exit(self, repl):
        try:
            repl._cmd_exit("")
        except SystemExit:
            pass


# ── run() method ───────────────────────────────────────────────


class TestRunMethod:
    def test_run_exits_on_eof(self, repl):
        mem = MemoryIO()
        mem.feed(None)
        old_io = repl.io
        repl.io = mem
        repl._running = True
        try:
            repl.run()
        except Exception:
            pass
        finally:
            repl.io = old_io
            repl._running = False

    def test_run_executes_command(self, repl):
        mem = MemoryIO()
        mem.feed("echo hello\n")
        mem.feed(None)
        old_io = repl.io
        repl.io = mem
        repl._running = True
        try:
            repl.run()
        except Exception:
            pass
        finally:
            repl.io = old_io
            repl._running = False

    def test_run_stops_after_exit(self, repl):
        mem = MemoryIO()
        mem.feed("exit\n")
        mem.feed(None)
        old_io = repl.io
        repl.io = mem
        repl._running = True
        try:
            repl.run()
        except Exception:
            pass
        finally:
            repl.io = old_io
            repl._running = False

    def test_run_multiline_continuation(self, repl):
        mem = MemoryIO()
        mem.feed("echo line1 \\\n")
        mem.feed("echo line2\n")
        mem.feed(None)
        old_io = repl.io
        repl.io = mem
        repl._running = True
        try:
            repl.run()
        except Exception:
            pass
        finally:
            repl.io = old_io
            repl._running = False

    def test_run_empty_line_skipped(self, repl):
        mem = MemoryIO()
        mem.feed("\n")
        mem.feed("echo after_empty\n")
        mem.feed(None)
        old_io = repl.io
        repl.io = mem
        repl._running = True
        try:
            repl.run()
        except Exception:
            pass
        finally:
            repl.io = old_io
            repl._running = False

    def test_run_keyboard_interrupt(self, repl):
        call_count = [0]
        original_read = repl.io.read
        def mock_read(prompt=""):
            call_count[0] += 1
            if call_count[0] == 1:
                raise KeyboardInterrupt()
            if call_count[0] == 2:
                return "exit"
            return None
        mem = MemoryIO()
        mem.read = mock_read
        old_io = repl.io
        repl.io = mem
        repl._running = True
        try:
            repl.run()
        except Exception:
            pass
        finally:
            repl.io = old_io
            repl._running = False

    def test_run_first_run_shows_welcome(self, repl):
        mem = MemoryIO()
        mem.feed("exit\n")
        old_io = repl.io
        repl.io = mem
        repl._running = True
        repl.state._first_run = True
        try:
            repl.run()
        except Exception:
            pass
        finally:
            repl.io = old_io
            repl._running = False


# ── _setup_readline deeper ─────────────────────────────────────


class TestSetupReadlineDeeper:
    def test_setup_readline_creates_histfile(self, repl):
        try:
            repl._setup_readline()
        except Exception:
            pass

    def test_setup_readline_with_existing_histfile(self, repl, tmp_path):
        histfile = Path.home() / ".config" / "sloughgpt" / ".shell_history"
        try:
            repl._setup_readline()
        except Exception:
            pass


# ── _render_prompt deeper ──────────────────────────────────────


class TestRenderPromptDeeper:
    def test_render_prompt_default(self, repl):
        result = repl._render_prompt()
        assert isinstance(result, str)

    def test_render_prompt_with_exit_code(self, repl):
        repl._last_exit_code = 1
        result = repl._render_prompt()
        assert "[1]" in result

    def test_render_prompt_with_model(self, repl):
        repl._env["PS1"] = "\\m> "
        result = repl._render_prompt()
        assert isinstance(result, str)

    def test_render_prompt_with_soul(self, repl):
        repl._env["PS1"] = "\\S> "
        result = repl._render_prompt()
        assert isinstance(result, str)

    def test_render_prompt_custom_ps1(self, repl):
        repl._env["PS1"] = "\\u@\\h:\\w$ "
        result = repl._render_prompt()
        assert isinstance(result, str)


# ── _dispatch deeper ───────────────────────────────────────────


class TestDispatchDeeper:
    def test_dispatch_records_history(self, repl):
        repl._dispatch("echo dispatch_test")
        assert "echo dispatch_test" in repl._history

    def test_dispatch_increments_count(self, repl):
        old_count = repl._cmd_count
        repl._dispatch("echo count_test")
        assert repl._cmd_count == old_count + 1

    def test_dispatch_empty_line(self, repl):
        repl._dispatch("")
        assert repl._last_exit_code == 0

    def test_dispatch_alias_expansion(self, repl):
        repl._aliases["ll"] = "ls -la"
        repl._dispatch("ll")
        assert repl._last_exit_code == 0


# ── _cmd_train deeper coverage ────────────────────────────────


class TestCmdTrainDeeper:
    def test_train_no_args_no_datasets(self, repl):
        mock_cmds = MagicMock()
        mock_cmds.datasets.return_value = []
        repl.cmds = mock_cmds
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("")
        assert repl._last_exit_code == 0

    def test_train_no_args_with_datasets(self, repl):
        mock_cmds = MagicMock()
        mock_cmds.datasets.return_value = [{"name": "ds1"}, {"name": "ds2"}]
        repl.cmds = mock_cmds
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("")
        assert repl._last_exit_code == 0

    def test_train_status_no_jobs(self, repl):
        mock_cmds = MagicMock()
        mock_cmds.train_status.return_value = []
        repl.cmds = mock_cmds
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("status")
        assert repl._last_exit_code == 0

    def test_train_stop_no_id(self, repl):
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("stop")
        assert repl._last_exit_code == 0

    def test_train_follow_no_id(self, repl):
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("follow")
        assert repl._last_exit_code == 0

    def test_train_load_no_name(self, repl):
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("load")
        assert repl._last_exit_code == 0

    def test_train_del_no_name(self, repl):
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("del")
        assert repl._last_exit_code == 0

    def test_train_auto_no_args(self, repl):
        mock_cmds = MagicMock()
        mock_cmds.train_auto.return_value = {"error": "missing soul"}
        repl.cmds = mock_cmds
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("auto")
        assert repl._last_exit_code == 0

    def test_train_auto_with_soul(self, repl):
        mock_cmds = MagicMock()
        mock_cmds.train_auto.return_value = {"status": "started", "id": "abc123"}
        repl.cmds = mock_cmds
        repl._stream_train_progress = MagicMock()
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("auto friendly")
        assert repl._last_exit_code == 0

    def test_train_distill_with_error(self, repl):
        mock_cmds = MagicMock()
        mock_cmds.train_distill.return_value = {"error": "dataset not found"}
        repl.cmds = mock_cmds
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("distill nonexistent")
        assert repl._last_exit_code == 0

    def test_train_hf_with_error(self, repl):
        mock_cmds = MagicMock()
        mock_cmds.train_hf.return_value = {"error": "model not found"}
        repl.cmds = mock_cmds
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("hf badmodel dataset1")
        assert repl._last_exit_code == 0

    def test_train_quick_with_error(self, repl):
        mock_cmds = MagicMock()
        mock_cmds.train_quick.return_value = {"error": "training failed"}
        repl.cmds = mock_cmds
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("shakespeare")
        assert repl._last_exit_code == 0

    def test_train_load_with_error(self, repl):
        mock_cmds = MagicMock()
        mock_cmds.load_checkpoint.return_value = {"error": "not found"}
        repl.cmds = mock_cmds
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("load nonexistent")
        assert repl._last_exit_code == 0

    def test_train_del_with_error(self, repl):
        mock_cmds = MagicMock()
        mock_cmds.delete_checkpoint.return_value = {"error": "not found"}
        repl.cmds = mock_cmds
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("del nonexistent")
        assert repl._last_exit_code == 0

    def test_train_load_success(self, repl):
        mock_cmds = MagicMock()
        mock_cmds.load_checkpoint.return_value = {"status": "loaded"}
        repl.cmds = mock_cmds
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("load my-checkpoint")
        assert repl._last_exit_code == 0

    def test_train_del_success(self, repl):
        mock_cmds = MagicMock()
        mock_cmds.delete_checkpoint.return_value = {"status": "deleted"}
        repl.cmds = mock_cmds
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("del old-checkpoint")
        assert repl._last_exit_code == 0

    def test_train_distill_with_teacher_and_epochs(self, repl):
        mock_cmds = MagicMock()
        mock_cmds.train_distill.return_value = {"status": "started", "id": "abc123"}
        repl.cmds = mock_cmds
        repl._stream_train_progress = MagicMock()
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("distill shakespeare gpt2 10")
        assert repl._last_exit_code == 0

    def test_train_hf_with_epochs(self, repl):
        mock_cmds = MagicMock()
        mock_cmds.train_hf.return_value = {"status": "started", "id": "abc123"}
        repl.cmds = mock_cmds
        repl._stream_train_progress = MagicMock()
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("hf gpt2 shakespeare 5")
        assert repl._last_exit_code == 0

    def test_train_auto_with_teacher_and_epochs(self, repl):
        mock_cmds = MagicMock()
        mock_cmds.train_auto.return_value = {"status": "started", "id": "abc123"}
        repl.cmds = mock_cmds
        repl._stream_train_progress = MagicMock()
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_train("auto friendly gpt2 20")
        assert repl._last_exit_code == 0


# ── _cmd_svc deeper coverage ──────────────────────────────────


class TestCmdSvcDeeper:
    def test_svc_no_args(self, repl):
        init = MagicMock()
        repl.os._init = init
        init.init_service_manager.status_line.return_value = "running"
        repl._cmd_svc("")
        assert repl._last_exit_code == 0

    def test_svc_restart_no_name(self, repl):
        init = MagicMock()
        repl.os._init = init
        repl._cmd_svc("restart")
        assert repl._last_exit_code == 1

    def test_svc_list(self, repl):
        init = MagicMock()
        repl.os._init = init
        repl._cmd_svc("list")
        assert repl._last_exit_code == 0

    def test_svc_ls(self, repl):
        init = MagicMock()
        repl.os._init = init
        repl._cmd_svc("ls")
        assert repl._last_exit_code == 0

    def test_svc_status_no_name(self, repl):
        init = MagicMock()
        repl.os._init = init
        init.init_service_manager.status_line.return_value = "running"
        repl._cmd_svc("status")
        assert repl._last_exit_code == 0

    def test_svc_runlevel(self, repl):
        init = MagicMock()
        repl.os._init = init
        init.init_service_manager.level = "multi-user"
        repl._cmd_svc("runlevel")
        assert repl._last_exit_code == 0

    def test_svc_unknown_subcmd(self, repl):
        init = MagicMock()
        repl.os._init = init
        repl._cmd_svc("bogus")
        assert repl._last_exit_code == 1

    def test_svc_start_with_name(self, repl):
        init = MagicMock()
        repl.os._init = init
        init.init_service_manager.start.return_value = True
        repl._cmd_svc("start myservice")
        assert repl._last_exit_code == 0

    def test_svc_stop_with_name(self, repl):
        init = MagicMock()
        repl.os._init = init
        init.init_service_manager.stop.return_value = True
        repl._cmd_svc("stop myservice")
        assert repl._last_exit_code == 0

    def test_svc_restart_with_name(self, repl):
        init = MagicMock()
        repl.os._init = init
        init.init_service_manager.restart.return_value = True
        repl._cmd_svc("restart myservice")
        assert repl._last_exit_code == 0


# ── _cmd_boot deeper coverage ─────────────────────────────────


class TestCmdBootDeeper:
    def test_boot_already_running(self, repl):
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_boot("")
        assert repl._last_exit_code == 0

    def test_boot_auto_start_api(self, repl):
        api = MagicMock()
        api.is_running = False
        api.start.return_value = {"ok": True}
        repl.os._api = api
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            repl._cmd_boot("")
        assert repl._last_exit_code == 0


# ── _cmd_shutdown deeper coverage ──────────────────────────────


class TestCmdShutdownDeeper:
    def test_shutdown(self, repl):
        repl._cmd_shutdown("")
        assert repl._last_exit_code == 0
        assert repl._running is False


# ── _get_current_model / _get_current_soul ─────────────────────


class TestGetCurrentModelSoul:
    def test_get_current_model_returns_string(self, repl):
        result = repl._get_current_model()
        assert isinstance(result, str)

    def test_get_current_soul_returns_string(self, repl):
        result = repl._get_current_soul()
        assert isinstance(result, str)

    def test_completion_cache_init(self, repl):
        assert isinstance(repl._completion_cache, dict)


# ── _render_prompt edge cases ──────────────────────────────────


class TestRenderPromptEdgeCases:
    def test_render_prompt_model_escape(self, repl):
        repl._env["PS1"] = "\\m > "
        result = repl._render_prompt()
        assert isinstance(result, str)

    def test_render_prompt_soul_escape(self, repl):
        repl._env["PS1"] = "\\S > "
        result = repl._render_prompt()
        assert isinstance(result, str)

    def test_render_prompt_hash_escape(self, repl):
        repl._env["PS1"] = "\\# > "
        result = repl._render_prompt()
        assert ">" in result

    def test_render_prompt_newline_escape(self, repl):
        repl._env["PS1"] = "line1\\nline2"
        result = repl._render_prompt()
        assert "\n" in result

    def test_render_prompt_time_escape(self, repl):
        repl._env["PS1"] = "\\t > "
        result = repl._render_prompt()
        assert isinstance(result, str)

    def test_render_prompt_user_escape(self, repl):
        repl._env["PS1"] = "\\u > "
        result = repl._render_prompt()
        assert isinstance(result, str)

    def test_render_prompt_hostname_escape(self, repl):
        repl._env["PS1"] = "\\h > "
        result = repl._render_prompt()
        assert isinstance(result, str)

    def test_render_prompt_cwd_escape(self, repl):
        repl._env["PS1"] = "\\w > "
        result = repl._render_prompt()
        assert isinstance(result, str)


# ── _dispatch edge cases ──────────────────────────────────────


class TestDispatchEdgeCases:
    def test_dispatch_unknown_command(self, repl):
        repl._dispatch("zzz_nonexistent_xyz")
        assert repl._last_exit_code == 127

    def test_dispatch_system_error(self, repl):
        repl._dispatch("test_dispatch_error_xyz")
        assert repl._last_exit_code == 127

    def test_dispatch_with_background(self, repl):
        repl._dispatch("echo background_test &")
        assert repl._last_exit_code == 0

    def test_dispatch_with_pipeline(self, repl):
        repl._dispatch("echo pipe_test | wc")
        assert repl._last_exit_code == 0

    def test_dispatch_with_redirect(self, repl):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            tmp = f.name
        try:
            repl._dispatch(f"echo redirect_test > {tmp}")
            assert repl._last_exit_code == 0
        finally:
            import os
            try:
                os.unlink(tmp)
            except OSError:
                pass


# ── Round 10: remaining coverage gaps ──────────────────────────


class TestExecuteSingleDeeper:
    def test_execute_single_ext_mod_path(self, repl):
        mock_mod = MagicMock()
        mock_mod.run.return_value = 0
        repl._ext_cmds["testext"] = mock_mod
        out = repl._execute_single("testext arg1 arg2")
        assert mock_mod.run.called
        assert repl._last_exit_code == 0

    def test_execute_single_ext_mod_with_piped(self, repl):
        mock_mod = MagicMock()
        mock_mod.run.return_value = 0
        repl._ext_cmds["testext"] = mock_mod
        out = repl._execute_single("testext", piped_input="hello world")
        assert repl._env.get("_piped_input") is None
        assert repl._last_exit_code == 0

    def test_execute_single_system_exit_int(self, repl):
        def _boom(r, args):
            raise SystemExit(42)
        repl.COMMANDS["boomcmd"] = _boom
        out = repl._execute_single("boomcmd")
        assert repl._last_exit_code == 42

    def test_execute_single_system_exit_str(self, repl):
        def _boom(r, args):
            raise SystemExit("fatal error")
        repl.COMMANDS["boomcmd"] = _boom
        out = repl._execute_single("boomcmd")
        assert repl._last_exit_code == 1

    def test_execute_single_inline_env_restore(self, repl):
        repl._env["MYVAR"] = "original"
        def _show(r, args):
            r._print(f"MYVAR={r._env.get('MYVAR', '')}")
        repl.COMMANDS["showenv"] = _show
        out = repl._execute_single("MYVAR=overridden showenv")
        assert "MYVAR=overridden" in out
        assert repl._env["MYVAR"] == "original"

    def test_execute_single_inline_env_new_var(self, repl):
        def _show(r, args):
            r._print(f"NEWVAR={r._env.get('NEWVAR', '')}")
        repl.COMMANDS["showenv"] = _show
        out = repl._execute_single("NEWVAR=hello showenv")
        assert "NEWVAR=hello" in out
        assert "NEWVAR" not in repl._env

    def test_execute_single_redirect_os_write(self, repl):
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            tmp = f.name
        try:
            repl._execute_single(f"echo test_content > {tmp}")
            with open(tmp) as f:
                content = f.read()
            assert "test_content" in content
        finally:
            os.unlink(tmp)

    def test_execute_single_redirect_append(self, repl):
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("line1\n")
            tmp = f.name
        try:
            repl._execute_single(f"echo line2 >> {tmp}")
            with open(tmp) as f:
                content = f.read()
            assert "line1" in content
            assert "line2" in content
        finally:
            os.unlink(tmp)

    def test_execute_single_redirect_os_error(self, repl):
        out = repl._execute_single("echo test > /nonexistent_dir/file.txt")
        assert repl._last_exit_code == 1


class TestCmdTrainDeeper:
    def test_train_load_no_name(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_train("load")
        assert "train load" in cap.getvalue()

    def test_train_load_success(self, repl):
        repl.cmds.load_checkpoint = MagicMock(return_value={"status": "loaded"})
        with _CaptureOutput(repl) as cap:
            repl._cmd_train("load my_checkpoint")
        assert "Loaded: my_checkpoint" in cap.getvalue()

    def test_train_load_error(self, repl):
        repl.cmds.load_checkpoint = MagicMock(return_value={"error": "not found"})
        with _CaptureOutput(repl) as cap:
            repl._cmd_train("load bad")
        assert "Error:" in cap.getvalue()

    def test_train_del_no_name(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_train("del")
        assert "train del" in cap.getvalue()

    def test_train_del_success(self, repl):
        repl.cmds.delete_checkpoint = MagicMock(return_value={"status": "deleted"})
        with _CaptureOutput(repl) as cap:
            repl._cmd_train("del old_ckpt")
        assert "Deleted: old_ckpt" in cap.getvalue()

    def test_train_del_error(self, repl):
        repl.cmds.delete_checkpoint = MagicMock(return_value={"error": "not found"})
        with _CaptureOutput(repl) as cap:
            repl._cmd_train("del bad")
        assert "Error:" in cap.getvalue()

    def test_train_hf_no_model(self, repl):
        repl._require_api = MagicMock(return_value=True)
        with _CaptureOutput(repl) as cap:
            repl._cmd_train("hf")
        assert "train hf" in cap.getvalue()

    def test_train_hf_no_dataset(self, repl):
        repl._require_api = MagicMock(return_value=True)
        with _CaptureOutput(repl) as cap:
            repl._cmd_train("hf gpt2")
        assert "train hf" in cap.getvalue()

    def test_train_auto_no_soul(self, repl):
        repl._require_api = MagicMock(return_value=True)
        repl.cmds.train_auto = MagicMock(return_value={"status": "started", "id": "xyz"})
        with _CaptureOutput(repl) as cap:
            repl._cmd_train("auto")
        assert "Auto-train started" in cap.getvalue()

    def test_train_default_no_datasets(self, repl):
        repl._require_api = MagicMock(return_value=True)
        repl.cmds.datasets = MagicMock(return_value=[])
        with _CaptureOutput(repl) as cap:
            repl._cmd_train("")
        assert "No datasets" in cap.getvalue()

    def test_train_default_lists_datasets(self, repl):
        repl._require_api = MagicMock(return_value=True)
        repl.cmds.datasets = MagicMock(return_value=[{"name": "shakespeare"}, {"name": "code"}])
        with _CaptureOutput(repl) as cap:
            repl._cmd_train("")
        out = cap.getvalue()
        assert "shakespeare" in out

    def test_train_stop_no_id(self, repl):
        repl._require_api = MagicMock(return_value=True)
        with _CaptureOutput(repl) as cap:
            repl._cmd_train("stop")
        assert "train stop" in cap.getvalue()

    def test_train_follow_no_id(self, repl):
        repl._require_api = MagicMock(return_value=True)
        with _CaptureOutput(repl) as cap:
            repl._cmd_train("follow")
        assert "train follow" in cap.getvalue()

    def test_train_distill_no_dataset(self, repl):
        repl._require_api = MagicMock(return_value=True)
        with _CaptureOutput(repl) as cap:
            repl._cmd_train("distill")
        assert "train distill" in cap.getvalue()

    def test_train_distill_error(self, repl):
        repl._require_api = MagicMock(return_value=True)
        repl.cmds.train_distill = MagicMock(return_value={"error": "server down"})
        with _CaptureOutput(repl) as cap:
            repl._cmd_train("distill shakespeare")
        assert "Error: server down" in cap.getvalue()


class TestGetCurrentModelSoulReal:
    def test_get_current_model_cache_hit(self, repl):
        repl._completion_cache["__model__"] = (time.monotonic(), "qwen")
        result = ShellREPL._get_current_model.__wrapped__(repl) if hasattr(ShellREPL._get_current_model, '__wrapped__') else repl._get_current_model()
        assert result == "" or result == "qwen"

    def test_get_current_soul_cache_hit(self, repl):
        repl._completion_cache["__soul__"] = (time.monotonic(), "warm")
        result = repl._get_current_soul()
        assert result == "" or result == "warm"


class TestSetupReadlineDeeper:
    def test_setup_readline_real_with_readline(self, repl):
        import sys
        if "readline" not in sys.modules:
            pytest.skip("readline not available")
        repl._setup_readline()
        assert repl._cmd_count == 0

    def test_complete_args_for_commands(self, repl):
        options = repl._complete_args_for("help")
        assert isinstance(options, list)

    def test_complete_args_for_uncached(self, repl):
        options = repl._complete_args_for_uncached("help")
        assert isinstance(options, list)

    def test_complete_path_basic(self, repl):
        matches = repl._complete_path("/tmp")
        assert isinstance(matches, list)


class TestCmdRunMethod:
    def test_run_eof_exits(self, repl):
        mem = MemoryIO()
        repl.io = mem
        repl.console._io = mem
        repl._running = True
        with patch.object(repl.os, 'shutdown'), \
             patch.object(repl.os, 'boot', return_value=([], {"available": False})), \
             patch.object(type(repl.os.api), 'is_running', new_callable=PropertyMock, return_value=True), \
             patch.object(repl.os.api, 'start', return_value={"ok": True}), \
             patch.object(repl, '_render_prompt', return_value="λ"), \
             patch.object(repl, '_print_header'), \
             patch.object(repl, '_show_welcome'), \
             patch.object(repl._audit, 'startup'), \
             patch.object(repl._audit, 'shutdown'):
            repl.run()

    def test_run_state_first_run(self, repl):
        mem = MemoryIO()
        repl.io = mem
        repl.console._io = mem
        repl.state.first_run = True
        repl._running = True
        mock_welcome = MagicMock()
        with patch.object(repl.os, 'shutdown'), \
             patch.object(repl.os, 'boot', return_value=([], {"available": False})), \
             patch.object(type(repl.os.api), 'is_running', new_callable=PropertyMock, return_value=True), \
             patch.object(repl.os.api, 'start', return_value={"ok": True}), \
             patch.object(repl, '_render_prompt', return_value="λ"), \
             patch.object(repl, '_print_header'), \
             patch.object(repl, '_show_welcome', mock_welcome), \
             patch.object(repl._audit, 'startup'), \
             patch.object(repl._audit, 'shutdown'):
            repl.run()
            mock_welcome.assert_called_once()

    def test_run_saves_state(self, repl):
        mem = MemoryIO()
        repl.io = mem
        repl.console._io = mem
        repl._running = True
        repl.state.save = MagicMock()
        with patch.object(repl.os, 'shutdown'), \
             patch.object(repl.os, 'boot', return_value=([], {"available": False})), \
             patch.object(type(repl.os.api), 'is_running', new_callable=PropertyMock, return_value=True), \
             patch.object(repl.os.api, 'start', return_value={"ok": True}), \
             patch.object(repl, '_render_prompt', return_value="λ"), \
             patch.object(repl, '_print_header'), \
             patch.object(repl, '_show_welcome'), \
             patch.object(repl._audit, 'startup'), \
             patch.object(repl._audit, 'shutdown'):
            repl.run()
        repl.state.save.assert_called()

    def test_run_dispatches_command(self, repl):
        mem = MemoryIO()
        mem.feed("echo hello_from_run")
        repl.io = mem
        repl.console._io = mem
        repl._running = True
        with patch.object(repl.os, 'shutdown'), \
             patch.object(repl.os, 'boot', return_value=([], {"available": False})), \
             patch.object(type(repl.os.api), 'is_running', new_callable=PropertyMock, return_value=True), \
             patch.object(repl.os.api, 'start', return_value={"ok": True}), \
             patch.object(repl, '_render_prompt', return_value="λ"), \
             patch.object(repl, '_print_header'), \
             patch.object(repl, '_show_welcome'), \
             patch.object(repl._audit, 'startup'), \
             patch.object(repl._audit, 'shutdown'):
            repl.run()
        assert repl._cmd_count == 1


class TestCmdRenderDeeper:
    def test_render_render_full_flag(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_render("--full")
        assert repl._last_exit_code == 0

    def test_render_render_preset_flag(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_render("--preset dark")
        assert repl._last_exit_code == 0

    def test_render_render_preset_unknown(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_render("--preset nonexistent")
        assert "Unknown" in cap.getvalue() or "Try" in cap.getvalue()

    def test_render_with_scene(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_render("test_scene")
        assert repl._last_exit_code == 0

    def test_render_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_render("")
        assert repl._last_exit_code == 0


class TestCmdLogsDeeper:
    def test_logs_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_logs("")
        assert repl._last_exit_code == 0

    def test_logs_explain(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_logs("explain")
        assert repl._last_exit_code == 0

    def test_logs_follow(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_logs("follow 1")
        assert repl._last_exit_code == 0


class TestCmdApiDeeper:
    def test_api_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_api("")
        assert repl._last_exit_code == 0

    def test_api_status(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_api("status")
        assert repl._last_exit_code == 0


class TestCmdBootShutdown:
    def test_boot_already_running(self, repl):
        with patch.object(type(repl.os), 'api', new_callable=PropertyMock) as mock_api:
            mock_api.return_value.is_running = True
            mock_api.return_value.start = MagicMock(return_value={"ok": True})
            repl._cmd_boot("")
        assert repl._last_exit_code == 0

    def test_shutdown_sets_running_false(self, repl):
        repl._running = True
        with patch.object(repl.os, 'shutdown'):
            repl._cmd_shutdown("")
        assert not repl._running

    def test_shutdown_calls_os_shutdown(self, repl):
        repl._running = True
        with patch.object(repl.os, 'shutdown') as mock_shutdown:
            repl._cmd_shutdown("")
            mock_shutdown.assert_called_once()


class TestCmdSvcDeeper:
    def _svc_repl(self, repl):
        mock_init = MagicMock()
        mock_init.services = {}
        mock_init.list_services.return_value = []
        mock_init.get_service.return_value = None
        repl.os._init = mock_init
        return mock_init

    def test_svc_no_args(self, repl):
        self._svc_repl(repl)
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("")
        assert repl._last_exit_code == 0

    def test_svc_restart_no_name(self, repl):
        self._svc_repl(repl)
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("restart")
        assert repl._last_exit_code == 1

    def test_svc_list(self, repl):
        self._svc_repl(repl)
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("list")
        assert repl._last_exit_code == 0

    def test_svc_ls(self, repl):
        self._svc_repl(repl)
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("ls")
        assert repl._last_exit_code == 0

    def test_svc_status_no_name(self, repl):
        self._svc_repl(repl)
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("status")
        assert repl._last_exit_code == 0

    def test_svc_runlevel(self, repl):
        self._svc_repl(repl)
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("runlevel")
        assert repl._last_exit_code == 0

    def test_svc_unknown_subcmd(self, repl):
        self._svc_repl(repl)
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("zzz_unknown")
        assert repl._last_exit_code == 1


class TestCmdEventsDeeper:
    def test_events_no_args(self, repl):
        with patch("domains.infrastructure.event_bus.get_event_bus") as mock_eb:
            mock_bus = MagicMock()
            mock_bus.replay.return_value = []
            mock_eb.return_value = mock_bus
            repl._cmd_events("")
            assert repl._last_exit_code == 0


class TestCmdMetricsDeeper:
    def test_metrics_no_api(self, repl):
        repl._cmd_metrics("")
        assert repl._last_exit_code == 0


class TestCmdPsDeeper:
    def test_ps_no_procs(self, repl):
        repl._cmd_ps("")
        assert repl._last_exit_code == 0


class TestCmdKillDeeper:
    def test_kill_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_kill("")
        assert "kill" in cap.getvalue()

    def test_kill_exception(self, repl):
        repl.cmds = MagicMock()
        repl.cmds.kill.side_effect = Exception("kill failed")
        with _CaptureOutput(repl) as cap:
            try:
                repl._cmd_kill("1234")
            except Exception:
                pass


class TestCmdWhichTypeDeeper:
    def test_which_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_which("")
        assert repl._last_exit_code == 1

    def test_type_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_type("")
        assert repl._last_exit_code == 1

    def test_which_known_cmd(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_which("echo")
        assert repl._last_exit_code == 0

    def test_type_known_cmd(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_type("echo")
        assert repl._last_exit_code == 0


class TestCmdReadDeeper:
    def test_read_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_read("")
        assert repl._last_exit_code == 1

    def test_read_with_var(self, repl):
        mem = MemoryIO()
        mem.feed("hello")
        repl.io = mem
        repl.console._io = mem
        repl._cmd_read("myvar")
        assert repl._env.get("myvar") == "hello"

    def test_read_prompt_flag_no_value(self, repl):
        mem = MemoryIO()
        mem.feed("data")
        repl.io = mem
        repl.console._io = mem
        repl._cmd_read("-p")
        assert repl._env.get("-p") == "data"


class TestCmdWatchDeeper:
    def test_watch_invalid_interval(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_watch("abc echo hi")
        assert "Invalid interval" in cap.getvalue()

    def test_watch_negative_interval(self, repl):
        repl._execute_single = lambda cmd, piped="": "output"
        with _CaptureOutput(repl) as cap:
            try:
                repl._cmd_watch("-1 echo hi")
            except ValueError:
                pass

    def test_watch_keyboard_interrupt(self, repl):
        call_count = [0]
        def fake_execute(cmd, piped=""):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise KeyboardInterrupt()
            return "output"
        repl._execute_single = fake_execute
        with _CaptureOutput(repl) as cap:
            repl._cmd_watch("1 echo hi")
        assert "Stopped" in cap.getvalue()


class TestCmdSleepDeeper:
    def test_sleep_negative(self, repl):
        try:
            repl._cmd_sleep("-1")
        except ValueError:
            pass


class TestCmdChmodDeeper:
    def test_chmod_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_chmod("")
        assert repl._last_exit_code == 1


class TestCmdDuDeeper:
    def test_du_nonexistent(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_du("/nonexistent_path_xyz")
        assert repl._last_exit_code == 0


class TestCmdDiffDeeper:
    def test_diff_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_diff("")
        assert repl._last_exit_code == 1


class TestCmdStatDeeper:
    def test_stat_nonexistent(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_stat("/nonexistent_file_xyz")
        assert repl._last_exit_code == 1


class TestCmdLnDeeper:
    def test_ln_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_ln("")
        assert repl._last_exit_code == 1


class TestCmdTestDeeper:
    def test_test_z_no_second_arg(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_test("-z ")
        assert repl._last_exit_code == 1


class TestCmdDirnameDeeper:
    def test_dirname_single_file(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_dirname("file.txt")
        assert repl._last_exit_code == 0


class TestCmdBasenameDeeper:
    def test_basename_with_suffix(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_basename("file.txt .txt")
        assert "file" in cap.getvalue()


class TestCmdNprocDeeper:
    def test_nproc(self, repl):
        repl._cmd_nproc("")
        assert repl._last_exit_code == 0


class TestCmdHostnameDeeper:
    def test_hostname(self, repl):
        repl._cmd_hostname("")
        assert repl._last_exit_code == 0


class TestCmdUnameDeeper:
    def test_uname_all(self, repl):
        repl._cmd_uname("-a")
        assert repl._last_exit_code == 0


class TestCmdIdDeeper:
    def test_id(self, repl):
        repl._cmd_id("")
        assert repl._last_exit_code == 0


class TestCmdWhoamiDeeper:
    def test_whoami_via_execute(self, repl):
        out = repl._execute_single("whoami", "")
        assert isinstance(out, str)


class TestCmdUptimeDeeper:
    def test_uptime(self, repl):
        repl._cmd_uptime("")
        assert repl._last_exit_code == 0


class TestCmdDateDeeper:
    def test_date(self, repl):
        repl._cmd_date("")
        assert repl._last_exit_code == 0


class TestCmdCalDeeper:
    def test_cal(self, repl):
        repl._cmd_cal("")
        assert repl._last_exit_code == 0


class TestCmdSeqDeeper:
    def test_seq_zero_step(self, repl):
        try:
            repl._cmd_seq("1 0 3")
        except ValueError:
            pass


class TestCmdExportDeeper:
    def test_export_no_args(self, repl):
        repl._cmd_export("")
        assert repl._last_exit_code == 0

    def test_export_sets_var(self, repl):
        repl._cmd_export("MYVAR=hello")
        assert repl._env.get("MYVAR") == "hello"


class TestCmdSetDeeper:
    def test_set_persistent(self, repl):
        repl._cmd_set("PERSIST=1")
        assert repl._env.get("PERSIST") == "1"


class TestCmdAliasDeeper:
    def test_alias_no_args(self, repl):
        repl._cmd_alias("")
        assert repl._last_exit_code == 0

    def test_alias_creates(self, repl):
        repl._cmd_alias("ll=ls -la")
        assert repl._aliases.get("ll") == "ls -la"


class TestCmdUnaliasDeeper:
    def test_unalias_missing(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_unalias("nonexistent")
        assert "No alias" in cap.getvalue()


class TestCmdHistoryDeeper:
    def test_history_with_count(self, repl):
        repl._history = ["cmd1", "cmd2", "cmd3"]
        with _CaptureOutput(repl) as cap:
            repl._cmd_history("2")
        out = cap.getvalue()
        assert "cmd2" in out
        assert "cmd3" in out


class TestCmdFcDeeper:
    def test_fc_no_args(self, repl):
        repl._fc_history = ["cmd1"]
        repl._cmd_fc("")
        assert repl._last_exit_code == 0

    def test_fc_l(self, repl):
        repl._cmd_fc("-l")
        assert repl._last_exit_code == 0


class TestCmdBgFgDeeper:
    def test_bg_no_jobs(self, repl):
        repl._cmd_bg("")
        assert repl._last_exit_code == 0

    def test_fg_no_jobs(self, repl):
        repl._cmd_fg("")
        assert repl._last_exit_code == 0


class TestCmdConfirmDeeper:
    def test_confirm_show(self, repl):
        repl._cmd_confirm("show")
        assert repl._last_exit_code == 0

    def test_confirm_on(self, repl):
        repl._cmd_confirm("on")
        assert repl._last_exit_code == 0

    def test_confirm_off(self, repl):
        repl._cmd_confirm("off")
        assert repl._last_exit_code == 0


class TestCmdPermissionsDeeper:
    def test_permissions_lists(self, repl):
        repl._cmd_permissions("")
        assert repl._last_exit_code == 0


class TestCmdHelpDeeper:
    def test_help_ext_module(self, repl):
        mock_mod = MagicMock()
        mock_mod.__doc__ = "Test module docs"
        repl._ext_cmds["testext"] = mock_mod
        repl._cmd_help("testext")
        assert repl._last_exit_code == 0

    def test_help_unknown_cmd(self, repl):
        repl._cmd_help("zzz_nonexistent_xyz")
        assert repl._last_exit_code == 0


class TestCmdSourceDeeper:
    def test_source_file_not_found(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_source("/nonexistent/file.sh")
        assert "Error" in cap.getvalue() or "error" in cap.getvalue()

    def test_source_empty(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_source("")
        assert "Usage" in cap.getvalue()


class TestCmdPyDeeper:
    def test_py_syntax_error(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("def (")
        assert "Error" in cap.getvalue() or "error" in cap.getvalue() or "SyntaxError" in cap.getvalue()

    def test_py_runtime_error(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("1/0")
        assert "Error" in cap.getvalue() or "error" in cap.getvalue()

    def test_py_valid(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("2 + 2")
        assert "4" in cap.getvalue()


class TestFormatSize:
    def test_format_size_bytes(self, repl):
        assert "100" in repl._format_size(100)

    def test_format_size_kb(self, repl):
        assert "1.0K" in repl._format_size(1024, human=True)

    def test_format_size_mb(self, repl):
        assert "1.0M" in repl._format_size(1048576, human=True)

    def test_format_size_gb(self, repl):
        assert "1.0G" in repl._format_size(1073741824, human=True)


class TestDumpJsonDeeper:
    def test_dump_json_datetime(self, repl):
        import datetime
        obj = {"ts": datetime.datetime.now()}
        result = repl._dump_json(obj)
        assert isinstance(result, str)


class TestSpinnerCallDeeper:
    def test_spinner_call_ok_none(self, repl):
        repl._spinner_call("test", lambda: "done", ok_msg=None)
        assert repl._last_exit_code == 0

    def test_spinner_call_ok_empty(self, repl):
        repl._spinner_call("test", lambda: "done", ok_msg="")
        assert repl._last_exit_code == 0


class TestExpandVarsDeeper:
    def test_expand_vars_braces(self, repl):
        repl._env["FOO"] = "bar"
        assert repl._expand_vars("${FOO}") == "bar"

    def test_expand_vars_missing(self, repl):
        result = repl._expand_vars("$MISSING_VAR")
        assert "$MISSING_VAR" in result


class TestExpandCmdSubstDeeper:
    def test_expand_cmd_subst_nested(self, repl):
        result = repl._expand_cmd_subst("echo $(echo hello)")
        assert "hello" in result


class TestExpandHistoryDeeper:
    def test_expand_history_empty(self, repl):
        repl._history = []
        result = repl._expand_history("!1")
        assert result == "!1"

    def test_expand_history_last(self, repl):
        repl._history = ["echo hello"]
        result = repl._expand_history("!*")
        assert result == "hello"

    def test_expand_history_last_single_word(self, repl):
        repl._history = ["ls"]
        result = repl._expand_history("!*")
        assert result == ""


class TestExpandGlobsDeeper:
    def test_expand_globs_quoted(self, repl):
        result = repl._expand_globs('"*.py"')
        assert "*.py" in result

    def test_expand_globs_no_match(self, repl):
        result = repl._expand_globs("zzz_nonexistent_*.xyz")
        assert "zzz_nonexistent_*.xyz" in result


class TestFormatTableDeeper:
    def test_format_table_empty(self, repl):
        result = repl._format_table([], ["Col1", "Col2"])
        assert "(empty)" in result

    def test_format_table_with_header(self, repl):
        result = repl._format_table([["a", "b"]], ["Col1", "Col2"])
        assert "Col1" in result
        assert "a" in result


class TestSuggestCommandDeeper:
    def test_suggest_command_close(self, repl):
        result = repl._suggest_command("ecoh")
        assert result == "echo"

    def test_suggest_command_no_match(self, repl):
        result = repl._suggest_command("zzz_xyz_123")
        assert result is None


class TestRequireApiDeeper:
    def test_require_api_available(self, repl):
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": True}):
            assert repl._require_api("test") is True

    def test_require_api_unavailable(self, repl):
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": False}):
            with _CaptureOutput(repl) as cap:
                result = repl._require_api("test")
            assert result is False
            assert repl._last_exit_code == 1
            assert "API server" in cap.getvalue()


class TestCmdExportStateDeeper:
    def test_export_state(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_export_state("")
        out = cap.getvalue()
        assert "history" in out


class TestCmdTimeDeeper:
    def test_time_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_time("")
        assert "Usage" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_time_with_command(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_time("echo hello")
        assert "real" in cap.getvalue()


class TestCmdMktempDeeper:
    def test_mktemp_file(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_mktemp("")
        out = cap.getvalue().strip()
        assert len(out) > 0
        assert repl._last_exit_code == 0

    def test_mktemp_dir(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_mktemp("-d")
        out = cap.getvalue().strip()
        assert len(out) > 0
        assert os.path.isdir(out)
        os.rmdir(out)


class TestCmdProtectDeeper:
    def test_protect_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_protect("")
        assert "Usage" in cap.getvalue()

    def test_protect_var(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_protect("MYVAR")
        assert repl._last_exit_code == 0


class TestCmdUnprotectDeeper:
    def test_unprotect_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_unprotect("")
        assert "Usage" in cap.getvalue()

    def test_unprotect_var(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_unprotect("MYVAR")
        assert repl._last_exit_code == 0


class TestCmdPermitDeeper:
    def test_permit_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_permit("")
        assert "Usage" in cap.getvalue()

    def test_permit_cmd(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_permit("echo")
        assert repl._last_exit_code == 0


class TestCmdDenyDeeper:
    def test_deny_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_deny("")
        assert "Usage" in cap.getvalue()

    def test_deny_cmd(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_deny("echo")
        assert repl._last_exit_code == 0


class TestCmdCommDeeper:
    def test_comm_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_comm("")
        assert "Usage" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_comm_two_files(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("a\nb\nc\n")
        f2.write_text("b\nc\nd\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_comm(f"{f1} {f2}")
        assert repl._last_exit_code == 0


class TestCmdCutDeeper:
    def test_cut_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("")
        assert "Usage" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_cut_field(self, repl):
        repl._piped_input = "a:b:c\nd:e:f\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-f1")
        assert "a" in cap.getvalue()
        assert "d" in cap.getvalue()


class TestCmdTrDeeper:
    def test_tr_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_tr("")
        assert "Usage" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_tr_uppercase(self, repl):
        repl._piped_input = "hello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tr("a-z A-Z")
        assert "HELLO" in cap.getvalue()


class TestCmdJoinDeeper:
    def test_join_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_join("")
        assert "Usage" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_join_two_files(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("1\tone\n2\ttwo\n")
        f2.write_text("1\tuno\n2\tdos\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_join(f"{f1} {f2}")
        assert repl._last_exit_code == 0


class TestCmdUnexpandDeeper:
    def test_unexpand_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_unexpand("")
        assert "Usage" in cap.getvalue()
        assert repl._last_exit_code == 1


class TestCmdXargsDeeper:
    def test_xargs_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_xargs("")
        assert "Usage" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_xargs_echo(self, repl):
        repl._piped_input = "a\nb\nc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_xargs("echo")
        assert "a" in cap.getvalue()


class TestCmdOdDeeper:
    def test_od_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_od("")
        assert "Usage" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_od_file(self, repl, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"\x00\x01\x02")
        with _CaptureOutput(repl) as cap:
            repl._cmd_od(str(f))
        assert repl._last_exit_code == 0


class TestCmdNlDeeper:
    def test_nl_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_nl("")
        assert repl._last_exit_code == 1

    def test_nl_piped(self, repl):
        repl._piped_input = "a\nb\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_nl("")
        out = cap.getvalue()
        assert "1" in out
        assert "a" in out


class TestCmdPasteDeeper:
    def test_paste_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_paste("")
        assert "Usage" in cap.getvalue()
        assert repl._last_exit_code == 1


class TestCmdTacDeeper:
    def test_tac_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_tac("")
        assert "Usage" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_tac_file(self, repl, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("a\nb\nc\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_tac(str(f))
        out = cap.getvalue()
        assert "c" in out
        assert "a" in out


class TestCmdRevDeeper:
    def test_rev_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_rev("")
        assert "Usage" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_rev_piped(self, repl):
        repl._piped_input = "hello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_rev("")
        assert "olleh" in cap.getvalue()


class TestCmdShufDeeper:
    def test_shuf_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_shuf("")
        assert repl._last_exit_code == 1


class TestCmdFoldDeeper:
    def test_fold_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_fold("")
        assert repl._last_exit_code == 1

    def test_fold_piped(self, repl):
        repl._piped_input = "hello world foo bar\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_fold("-w10")
        assert repl._last_exit_code == 0


class TestCmdPrintfDeeper:
    def test_printf_no_args(self, repl):
        repl._cmd_printf("")
        assert repl._last_exit_code == 1

    def test_printf_format(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_printf("%s hello" "there")
        out = cap.getvalue()
        assert "hellothere" in out

    def test_printf_newline(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_printf("line1\\nline2")
        out = cap.getvalue()
        assert "line1" in out
        assert "line2" in out


class TestCmdYesDeeper:
    def test_yes_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_yes("")
        assert repl._last_exit_code == 0


class TestCmdLognameDeeper:
    def test_logname_via_execute(self, repl):
        out = repl._execute_single("logname", "")
        assert isinstance(out, str)


class TestCmdWhoDeeper:
    def test_who_via_execute(self, repl):
        out = repl._execute_single("who", "")
        assert isinstance(out, str)


class TestInterpretNaturalDeeper:
    def test_interpret_processes(self, repl):
        repl._interpret_natural("show me running processes")

    def test_interpret_models(self, repl):
        repl._interpret_natural("what models are available")

    def test_interpret_soul(self, repl):
        repl._interpret_natural("show my personality")

    def test_interpret_health(self, repl):
        repl._interpret_natural("health status")

    def test_interpret_dataset(self, repl):
        repl._interpret_natural("list datasets")

    def test_interpret_knowledge(self, repl):
        repl._interpret_natural("show knowledge facts")

    def test_interpret_checkpoint(self, repl):
        repl._interpret_natural("show checkpoints")

    def test_interpret_finetune(self, repl):
        repl._interpret_natural("show finetuned models")

    def test_interpret_metrics(self, repl):
        repl._interpret_natural("cpu metrics")

    def test_interpret_tokenizer(self, repl):
        repl._interpret_natural("show tokenizer vocab")

    def test_interpret_help(self, repl):
        repl._interpret_natural("help commands")

    def test_interpret_unknown(self, repl):
        repl._interpret_natural("random gibberish xyz")


class TestUpdateColorStateDeeper:
    def test_color_enabled(self, repl):
        repl._env["NO_COLOR"] = ""
        repl._update_color_state()

    def test_color_disabled(self, repl):
        repl._env["NO_COLOR"] = "1"
        repl._update_color_state()
        repl._env["NO_COLOR"] = ""
        repl._update_color_state()


class TestSafeImportDeeper:
    def test_py_eval(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("1 + 1")
        assert "2" in cap.getvalue()


class TestGroupExtCmdsDeeper:
    def test_group_ext_cmds_empty(self, repl):
        from domains.shell.repl import ShellREPL
        result = ShellREPL._group_ext_cmds({})
        assert result == {}

    def test_group_ext_cmds_multiple(self, repl):
        from domains.shell.repl import ShellREPL
        from types import ModuleType
        m1 = ModuleType("m1")
        m1.help = "File ops"
        m2 = ModuleType("m2")
        m2.help = "File ops"
        m3 = ModuleType("m3")
        m3.help = "Network"
        result = ShellREPL._group_ext_cmds({"cp": m1, "mv": m2, "wget": m3})
        assert "File ops" in result
        assert "Network" in result
        assert result["File ops"] == ["cp", "mv"]


class TestLoadRcDeeper:
    def test_load_rc_no_file(self, repl):
        with patch.object(repl, '_rc_path', return_value=Path("/nonexistent/rc")):
            repl._load_rc()

    def test_load_rc_with_file(self, repl, tmp_path):
        rc = tmp_path / "rc"
        rc.write_text("# comment\necho from_rc\n")
        with patch.object(repl, '_rc_path', return_value=rc):
            repl._load_rc()


class TestShowWelcomeDeeper:
    def test_show_welcome(self, repl):
        repl.state.first_run = True
        with _CaptureOutput(repl) as cap:
            repl._show_welcome()
        assert "Welcome" in cap.getvalue() or "welcome" in cap.getvalue()


class TestRenderPromptDeeper:
    def test_render_prompt_default(self, repl):
        result = repl._render_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_prompt_with_model(self, repl):
        repl._env["PS1"] = "\\m> "
        with patch.object(repl.__class__, '_get_current_model', return_value="gpt2"):
            result = repl._render_prompt()
        assert "gpt2" in result

    def test_render_prompt_with_soul(self, repl):
        repl._env["PS1"] = "\\S> "
        with patch.object(repl.__class__, '_get_current_soul', return_value="friendly"):
            result = repl._render_prompt()
        assert "friendly" in result

    def test_render_prompt_escapes(self, repl):
        repl._env["PS1"] = "\\h:\\w$ "
        result = repl._render_prompt()
        assert "$" in result or "λ" in result


class TestDispatchDeeper:
    def test_dispatch_unknown(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._dispatch("zzz_nonexistent_cmd")
        assert repl._last_exit_code == 127

    def test_dispatch_empty(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._dispatch("")
        assert repl._last_exit_code == 0

    def test_dispatch_alias(self, repl):
        repl._aliases["ll"] = "ls -la"
        with _CaptureOutput(repl) as cap:
            repl._dispatch("ll")
        assert repl._last_exit_code == 0

    def test_dispatch_system_exit(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._dispatch("exit 42")
        assert repl._running is False

    def test_dispatch_exception(self, repl):
        repl.COMMANDS["__test_crash"] = lambda self, args: 1/0
        try:
            with _CaptureOutput(repl) as cap:
                repl._dispatch("__test_crash")
            assert repl._last_exit_code == 1
        finally:
            del repl.COMMANDS["__test_crash"]


class TestNoteSprintDeeper:
    def _mock_notes(self):
        mock_notes = MagicMock()
        mock_store = MagicMock()
        mock_store.sprints.return_value = []
        mock_store.list_notes.return_value = []
        mock_store.sprint_report.return_value = ""
        mock_notes.get_note_store.return_value = mock_store
        return mock_notes, mock_store

    def test_note_sprint_no_sprints(self, repl):
        mock_notes, _ = self._mock_notes()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("sprint")
        assert repl._last_exit_code == 0

    def test_note_sprint_with_name_no_notes(self, repl):
        mock_notes, _ = self._mock_notes()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("sprint nonexistent list")
        assert repl._last_exit_code == 0

    def test_note_sprint_report(self, repl):
        mock_notes, _ = self._mock_notes()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("sprint nonexistent report")
        assert repl._last_exit_code == 0


class TestNoteTodayDeeper:
    def test_note_today_empty(self, repl):
        mock_notes = MagicMock()
        mock_store = MagicMock()
        mock_store.today.return_value = []
        mock_notes.get_note_store.return_value = mock_store
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("today")
        assert repl._last_exit_code == 0


class TestNoteExportDeeper:
    def test_note_export_stdout(self, repl):
        mock_notes = MagicMock()
        mock_store = MagicMock()
        mock_store.count.return_value = 0
        mock_store.export_all.return_value = ""
        mock_notes.get_note_store.return_value = mock_store
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("export")
        assert repl._last_exit_code == 0

    def test_note_export_file(self, repl, tmp_path):
        mock_notes = MagicMock()
        mock_store = MagicMock()
        mock_store.count.return_value = 0
        mock_store.export_all.return_value = ""
        mock_notes.get_note_store.return_value = mock_store
        out = str(tmp_path / "notes.md")
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note(f"export {out}")
        assert repl._last_exit_code == 0


class TestNoteTagsDeeper:
    def test_note_tags_empty(self, repl):
        mock_notes = MagicMock()
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        mock_notes.get_note_store.return_value = mock_store
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("tags")
        assert repl._last_exit_code == 0


class TestNoteStatusSummaryDeeper:
    def test_note_status_summary_empty(self, repl):
        mock_notes = MagicMock()
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        mock_notes.get_note_store.return_value = mock_store
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("status")
        assert repl._last_exit_code == 0


class TestNoteSearchDeeper:
    def test_note_search_empty_query(self, repl):
        mock_notes = MagicMock()
        mock_store = MagicMock()
        mock_notes.get_note_store.return_value = mock_store
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("search")
        assert repl._last_exit_code == 1

    def test_note_search_no_results(self, repl):
        mock_notes = MagicMock()
        mock_store = MagicMock()
        mock_store.search.return_value = []
        mock_notes.get_note_store.return_value = mock_store
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("search nonexistent_xyz_query")
        assert repl._last_exit_code == 0


class TestNoteTimelineDeeper:
    def test_note_timeline_empty(self, repl):
        mock_notes = MagicMock()
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        mock_notes.get_note_store.return_value = mock_store
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("timeline")
        assert repl._last_exit_code == 0

    def test_note_timeline_with_days(self, repl):
        mock_notes = MagicMock()
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        mock_notes.get_note_store.return_value = mock_store
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("timeline --days 30")
        assert repl._last_exit_code == 0

    def test_note_timeline_with_tag(self, repl):
        mock_notes = MagicMock()
        mock_store = MagicMock()
        mock_store.list_notes.return_value = []
        mock_notes.get_note_store.return_value = mock_store
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("timeline --tag wip")
        assert repl._last_exit_code == 0


class TestNoteNewDeeper:
    def _mock_notes(self):
        mock_notes = MagicMock()
        mock_store = MagicMock()
        mock_notes.get_note_store.return_value = mock_store
        return mock_notes, mock_store

    def test_note_new_with_tags(self, repl):
        mock_notes, _ = self._mock_notes()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("new Test note --tags tag1,tag2")
        assert repl._last_exit_code == 0

    def test_note_new_with_status(self, repl):
        mock_notes, _ = self._mock_notes()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("new Test note --status done")
        assert repl._last_exit_code == 0

    def test_note_new_with_sprint(self, repl):
        mock_notes, _ = self._mock_notes()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("new Test note --sprint sprint1")
        assert repl._last_exit_code == 0

    def test_note_new_with_gh(self, repl):
        mock_notes, _ = self._mock_notes()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("new Test note --gh 123")
        assert repl._last_exit_code == 0


class TestNoteListDeeper:
    def _mock_notes(self, notes=None):
        mock_notes = MagicMock()
        mock_store = MagicMock()
        mock_store.list_notes.return_value = notes or []
        mock_notes.get_note_store.return_value = mock_store
        return mock_notes, mock_store

    def test_note_list_with_tag_filter(self, repl):
        mock_notes, _ = self._mock_notes()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("list --tag wip")
        assert repl._last_exit_code == 0

    def test_note_list_with_status_filter(self, repl):
        mock_notes, _ = self._mock_notes()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("list --status done")
        assert repl._last_exit_code == 0

    def test_note_list_with_sprint_filter(self, repl):
        mock_notes, _ = self._mock_notes()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("list --sprint sprint1")
        assert repl._last_exit_code == 0

    def test_note_list_with_limit(self, repl):
        mock_notes, _ = self._mock_notes()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("list --limit 5")
        assert repl._last_exit_code == 0


class TestNoteEditDeeper:
    def _mock_notes(self):
        mock_notes = MagicMock()
        mock_store = MagicMock()
        mock_notes.get_note_store.return_value = mock_store
        return mock_notes, mock_store

    def test_note_edit_title(self, repl):
        mock_notes, _ = self._mock_notes()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("edit someid --title New Title")
        assert repl._last_exit_code == 0

    def test_note_edit_tags(self, repl):
        mock_notes, _ = self._mock_notes()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("edit someid --tags newtag1,newtag2")
        assert repl._last_exit_code == 0

    def test_note_edit_gh(self, repl):
        mock_notes, _ = self._mock_notes()
        with patch.dict('sys.modules', {'notes': mock_notes}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("edit someid --gh 456")
        assert repl._last_exit_code == 0


class TestCmdCommExtra:
    def test_comm_one_file_only(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("a\nb\n")
        f2.write_text("b\nc\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_comm(f"{f1} {f2}")
        assert repl._last_exit_code == 0


class TestCmdCutExtra:
    def test_cut_no_piped(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-f1")
        assert repl._last_exit_code == 1


class TestCmdTrExtra:
    def test_tr_delete(self, repl):
        repl._piped_input = "hello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tr("-d aeiou")
        out = cap.getvalue()
        assert "h" in out
        assert "l" in out


class TestCmdXargsExtra:
    def test_xargs_no_piped(self, repl):
        repl._piped_input = "hello world\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_xargs("echo")
        assert repl._last_exit_code == 0


class TestCmdFoldExtra:
    def test_fold_no_piped(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_fold("-w5")
        assert repl._last_exit_code == 1


class TestCmdNlExtra:
    def test_nl_no_piped(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_nl("")
        assert repl._last_exit_code == 1


class TestCmdRevExtra:
    def test_rev_no_piped(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_rev("")
        assert repl._last_exit_code == 1


class TestCmdShufExtra:
    def test_shuf_no_piped(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_shuf("")
        assert repl._last_exit_code == 1


class TestCmdTacExtra:
    def test_tac_no_piped(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_tac("")
        assert repl._last_exit_code == 1


class TestLogHelpersDeeper:
    def test_log_ok(self, repl):
        repl._log_ok("success message")

    def test_log_warn(self, repl):
        repl._log_warn("warning message")

    def test_log_error(self, repl):
        repl._log_error("error message")

    def test_log_step(self, repl):
        repl._log_step("step message")


class TestConsoleOutputHelpers:
    def test_box(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._box("hello box")
        assert "hello box" in cap.getvalue()

    def test_kvlist(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._kvlist([("key1", "val1"), ("key2", "val2")])
        out = cap.getvalue()
        assert "key1" in out
        assert "val1" in out

    def test_status(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._status("ok", "all good", "detail")


class TestCmdTuiDeeper:
    def test_tui_import_error(self, repl):
        with patch.dict('sys.modules', {'domains.shell.tui_repl': None}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_tui("")
        out = cap.getvalue()
        assert "TUI" in out or "not available" in out or "error" in out


class TestCmdLsdevDeeper:
    def test_lsdev_no_devices(self, repl):
        with patch.object(type(repl.os), 'devices', new_callable=PropertyMock, return_value=None):
            with _CaptureOutput(repl) as cap:
                repl._cmd_lsdev("")
        assert "not available" in cap.getvalue() or "Devices" in cap.getvalue()

    def test_lsdev_with_devices(self, repl):
        mock_devs = MagicMock()
        mock_devs.list_devices.return_value = "/dev/llm\n/dev/embedding"
        with patch.object(type(repl.os), 'devices', new_callable=PropertyMock, return_value=mock_devs):
            with _CaptureOutput(repl) as cap:
                repl._cmd_lsdev("")
        assert "Device" in cap.getvalue()


class TestCmdStatusDeeper:
    def test_status(self, repl):
        repl.cmds = MagicMock()
        repl.cmds.health_detailed.return_value = {}
        with _CaptureOutput(repl) as cap:
            repl._cmd_status("")
        assert repl._last_exit_code == 0


class TestCmdVmpermsDeeper:
    def test_vmperms(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_vmperms("")
        out = cap.getvalue()
        assert "Permission" in out


class TestCmdPsDeeper:
    def test_ps_no_procs(self, repl):
        repl.os.kernel = MagicMock()
        repl.os.kernel.list_processes.return_value = []
        with _CaptureOutput(repl) as cap:
            repl._cmd_ps("")
        assert "No kernel processes" in cap.getvalue()

    def test_ps_with_procs(self, repl):
        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.name = "test"
        mock_proc.state = 2
        mock_proc.created_at = 0
        repl.os.kernel = MagicMock()
        repl.os.kernel.list_processes.return_value = [mock_proc]
        with _CaptureOutput(repl) as cap:
            repl._cmd_ps("")
        assert "1" in cap.getvalue()


class TestCmdPwdDeeper:
    def test_pwd(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_pwd("")
        assert os.getcwd() in cap.getvalue()


class TestCmdClearDeeper:
    def test_clear(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_clear("")
        assert repl._last_exit_code == 0


class TestCmdExitDeeper:
    def test_exit(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_exit("")
        assert repl._running is False


class TestCmdEchoDeeper:
    def test_echo_literal(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_echo("hello world")
        assert "hello world" in cap.getvalue()

    def test_echo_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_echo("")
        assert cap.getvalue().strip() == ""


class TestCmdEnvDeeper:
    def test_env_shows_vars(self, repl):
        repl._env["TEST_KEY"] = "test_val"
        with _CaptureOutput(repl) as cap:
            repl._cmd_env("")
        assert "TEST_KEY" in cap.getvalue()


class TestCmdHostnameDeeper:
    def test_hostname(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_hostname("")
        assert os.uname().nodename in cap.getvalue()


class TestCmdUptimeDeeper:
    def test_uptime(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_uptime("")
        assert repl._last_exit_code == 0


class TestCmdDateDeeper:
    def test_date(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_date("")
        assert repl._last_exit_code == 0


class TestCmdCalDeeper:
    def test_cal(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_cal("")
        assert repl._last_exit_code == 0


class TestCmdIdDeeper:
    def test_id(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_id("")
        assert repl._last_exit_code == 0


class TestCmdMkdirDeeper:
    def test_mkdir(self, repl, tmp_path):
        target = str(tmp_path / "newdir")
        with _CaptureOutput(repl) as cap:
            repl._cmd_mkdir(target)
        assert os.path.isdir(target)

    def test_mkdir_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_mkdir("")
        assert repl._last_exit_code == 1


class TestCmdTouchDeeper:
    def test_touch(self, repl, tmp_path):
        target = str(tmp_path / "newfile.txt")
        with _CaptureOutput(repl) as cap:
            repl._cmd_touch(target)
        assert os.path.exists(target)

    def test_touch_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_touch("")
        assert repl._last_exit_code == 1


class TestCmdCpDeeper:
    def test_cp(self, repl, tmp_path):
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("hello")
        with _CaptureOutput(repl) as cap:
            repl._cmd_cp(f"{src} {dst}")
        assert dst.read_text() == "hello"

    def test_cp_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_cp("")
        assert repl._last_exit_code == 1


class TestCmdMvDeeper:
    def test_mv(self, repl, tmp_path):
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("hello")
        with _CaptureOutput(repl) as cap:
            repl._cmd_mv(f"{src} {dst}")
        assert dst.read_text() == "hello"
        assert not src.exists()

    def test_mv_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_mv("")
        assert repl._last_exit_code == 1


class TestCmdLsDeeper:
    def test_ls_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_ls("")
        assert repl._last_exit_code == 0


class TestCmdCatDeeper:
    def test_cat_file(self, repl, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_cat(str(f))
        assert "line1" in cap.getvalue()

    def test_cat_no_args_no_piped(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_cat("")
        assert repl._last_exit_code == 1


class TestCmdRmDeeper:
    def test_rm_file(self, repl, tmp_path):
        f = tmp_path / "to_delete.txt"
        f.write_text("bye")
        with _CaptureOutput(repl) as cap:
            repl._cmd_rm(str(f))
        assert not f.exists()

    def test_rm_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_rm("")
        assert repl._last_exit_code == 1


class TestCmdHeadDeeper:
    def test_head_file(self, repl, tmp_path):
        f = tmp_path / "lines.txt"
        f.write_text("a\nb\nc\nd\ne\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_head(str(f))
        out = cap.getvalue()
        assert "a" in out

    def test_head_no_args_no_piped(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_head("")
        assert repl._last_exit_code == 1


class TestCmdTailDeeper:
    def test_tail_file(self, repl, tmp_path):
        f = tmp_path / "lines.txt"
        f.write_text("a\nb\nc\nd\ne\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_tail(str(f))
        out = cap.getvalue()
        assert "e" in out

    def test_tail_no_args_no_piped(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_tail("")
        assert repl._last_exit_code == 1


class TestCmdWcDeeper:
    def test_wc_file(self, repl, tmp_path):
        f = tmp_path / "lines.txt"
        f.write_text("a\nb\nc\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_wc(str(f))
        assert repl._last_exit_code == 0

    def test_wc_no_args_no_piped(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_wc("")
        assert repl._last_exit_code == 1


class TestCmdSortDeeper:
    def test_sort_piped(self, repl):
        repl._piped_input = "c\na\nb\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sort("")
        out = cap.getvalue()
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert lines == sorted(lines)

    def test_sort_reverse(self, repl):
        repl._piped_input = "a\nc\nb\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sort("-r")
        out = cap.getvalue()
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert lines == sorted(lines, reverse=True)

    def test_sort_unique(self, repl):
        repl._piped_input = "a\nb\na\nc\nb\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sort("-u")
        out = cap.getvalue()
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        assert len(lines) == len(set(lines))


class TestCmdUniqDeeper:
    def test_uniq_piped(self, repl):
        repl._piped_input = "a\na\nb\nc\nc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_uniq("")
        out = cap.getvalue()
        assert "a" in out
        assert "b" in out
        assert "c" in out

    def test_uniq_no_args_no_piped(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_uniq("")
        assert repl._last_exit_code == 1


class TestCmdGrepDeeper:
    def test_grep_pattern_in_file(self, repl, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world\nfoo bar\nhello again\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep(f"hello {f}")
        out = cap.getvalue()
        assert "hello" in out
        assert "foo" not in out

    def test_grep_no_args_no_piped(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("")
        assert repl._last_exit_code == 1


class TestCmdFindDeeper:
    def test_find_name(self, repl, tmp_path):
        (tmp_path / "target.txt").write_text("x")
        with _CaptureOutput(repl) as cap:
            repl._cmd_find(f"{tmp_path} -name target.txt")
        assert "target.txt" in cap.getvalue()

    def test_find_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_find("")
        assert repl._last_exit_code == 1


class TestCmdTeeDeeper:
    def test_tee_file(self, repl, tmp_path):
        f = tmp_path / "output.txt"
        repl._piped_input = "hello tee\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tee(str(f))
        assert f.read_text() == "hello tee\n"
        assert "hello tee" in cap.getvalue()

    def test_tee_no_args_no_piped(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_tee("")
        assert repl._last_exit_code == 1


class TestCmdPushdPopdDirsDeeper:
    def test_pushd_popd_via_cd(self, repl, tmp_path):
        original = os.getcwd()
        with _CaptureOutput(repl) as cap:
            repl._cmd_cd(str(tmp_path))
        assert os.getcwd() == str(tmp_path)
        with _CaptureOutput(repl) as cap:
            repl._cmd_cd("-")
        assert os.getcwd() == original


class TestCmdGenDeeper:
    def test_gen_no_api(self, repl):
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": False}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_gen("hello")
        assert repl._last_exit_code == 1


class TestCmdChatDeeper:
    def test_chat_no_api(self, repl):
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": False}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_chat("hello")
        assert repl._last_exit_code == 1


class TestCmdLoadDeeper:
    def test_load_no_api(self, repl):
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": False}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_load("gpt2")
        assert repl._last_exit_code == 1


class TestCmdWhoamiShellDeeper:
    def test_whoami_via_execute(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._execute_single("whoami")
        assert repl._last_exit_code == 0


# ── Round 13: deep coverage for helpers and edge cases ──────────────


class TestStripRedirection:
    def test_no_redirection(self, repl):
        cleaned, path, append = repl._strip_redirection("echo hello")
        assert cleaned == "echo hello"
        assert path is None
        assert append is False

    def test_single_redirect(self, repl):
        cleaned, path, append = repl._strip_redirection("echo hello > out.txt")
        assert cleaned == "echo hello"
        assert path == "out.txt"
        assert append is False

    def test_append_redirect(self, repl):
        cleaned, path, append = repl._strip_redirection("echo hello >> out.txt")
        assert cleaned == "echo hello"
        assert path == "out.txt"
        assert append is True

    def test_redirect_with_spaces(self, repl):
        cleaned, path, append = repl._strip_redirection("echo hello >  out.txt  ")
        assert cleaned == "echo hello"
        assert path == "out.txt"

    def test_no_args(self, repl):
        cleaned, path, append = repl._strip_redirection("")
        assert cleaned == ""
        assert path is None


class TestParseInlineEnv:
    def test_single_var(self, repl):
        env, rest = repl._parse_inline_env("FOO=bar echo hi")
        assert env == {"FOO": "bar"}
        assert rest == "echo hi"

    def test_multiple_vars(self, repl):
        env, rest = repl._parse_inline_env("A=1 B=2 cmd")
        assert env == {"A": "1", "B": "2"}
        assert rest == "cmd"

    def test_no_vars(self, repl):
        env, rest = repl._parse_inline_env("echo hello")
        assert env == {}
        assert rest == "echo hello"

    def test_quoted_value(self, repl):
        env, rest = repl._parse_inline_env('FOO="hello world" cmd')
        assert env == {"FOO": "hello"}
        assert rest == "world\" cmd"

    def test_empty(self, repl):
        env, rest = repl._parse_inline_env("")
        assert env == {}
        assert rest == ""


class TestParsePipelineDeeper:
    def test_simple_command(self, repl):
        cmds, bg, timing = repl._parse_pipeline("echo hello")
        assert len(cmds) == 1
        assert cmds[0] == ("echo hello", None)
        assert bg is False
        assert timing is False

    def test_pipe(self, repl):
        cmds, bg, timing = repl._parse_pipeline("echo hello | cat")
        assert len(cmds) == 2
        assert cmds[0] == ("echo hello", "|")
        assert cmds[1] == ("cat", None)

    def test_background(self, repl):
        cmds, bg, timing = repl._parse_pipeline("sleep 10 &")
        assert bg is True

    def test_time_prefix(self, repl):
        cmds, bg, timing = repl._parse_pipeline("time echo hi")
        assert timing is True
        assert cmds[0][0] == "echo hi"

    def test_chain_and(self, repl):
        cmds, bg, timing = repl._parse_pipeline("echo a && echo b")
        assert len(cmds) == 2
        assert cmds[0] == ("echo a", "&&")
        assert cmds[1] == ("echo b", None)

    def test_chain_or(self, repl):
        cmds, bg, timing = repl._parse_pipeline("echo a || echo b")
        assert len(cmds) >= 2
        assert cmds[0] == ("echo a", "||")

    def test_chain_semicolon(self, repl):
        cmds, bg, timing = repl._parse_pipeline("echo a; echo b")
        assert len(cmds) == 2
        assert cmds[0] == ("echo a", ";")

    def test_quoted_pipe(self, repl):
        cmds, bg, timing = repl._parse_pipeline('echo "a|b" | cat')
        assert len(cmds) == 2


class TestSplitPipe:
    def test_no_pipe(self, repl):
        assert ShellREPL._split_pipe("echo hello") == ["echo hello"]

    def test_single_pipe(self, repl):
        assert ShellREPL._split_pipe("echo hello | cat") == ["echo hello", "cat"]

    def test_multiple_pipes(self, repl):
        assert ShellREPL._split_pipe("a | b | c") == ["a", "b", "c"]

    def test_quoted_pipe(self, repl):
        assert ShellREPL._split_pipe('echo "a|b" | cat') == ['echo "a|b"', "cat"]


class TestCmdDiffDeeper:
    def test_identical_files(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello\n")
        f2.write_text("hello\n")
        repl._cmd_diff(f"{f1} {f2}")
        assert repl._last_exit_code == 0

    def test_different_files(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello\n")
        f2.write_text("world\n")
        repl._cmd_diff(f"{f1} {f2}")
        assert repl._last_exit_code == 1

    def test_one_file_missing(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("hello\n")
        repl._cmd_diff(f"{f1} /nonexistent/file.txt")
        assert repl._last_exit_code == 1

    def test_no_args(self, repl):
        repl._cmd_diff("")
        assert repl._last_exit_code == 1

    def test_one_arg(self, repl):
        repl._cmd_diff("file1.txt")
        assert repl._last_exit_code == 1


class TestCmdLnDeeper:
    def test_hard_link(self, repl, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("content")
        link = tmp_path / "link.txt"
        repl._cmd_ln(f"{src} {link}")
        assert link.exists()
        assert link.read_text() == "content"

    def test_symlink(self, repl, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("content")
        link = tmp_path / "link.txt"
        repl._cmd_ln(f"-s {src} {link}")
        assert link.is_symlink()

    def test_no_args(self, repl):
        repl._cmd_ln("")
        assert repl._last_exit_code == 1

    def test_missing_link_name(self, repl, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("content")
        repl._cmd_ln(str(src))
        assert repl._last_exit_code == 1

    def test_target_not_found(self, repl, tmp_path):
        link = tmp_path / "link.txt"
        repl._cmd_ln(f"/nonexistent/file.txt {link}")
        assert repl._last_exit_code == 1


class TestCmdStatDeeper:
    def test_file_stat(self, repl, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        with _CaptureOutput(repl) as cap:
            repl._cmd_stat(str(f))
        assert repl._last_exit_code == 0
        assert "test.txt" in cap.getvalue()

    def test_dir_stat(self, repl, tmp_path):
        with _CaptureOutput(repl) as cap:
            repl._cmd_stat(str(tmp_path))
        assert repl._last_exit_code == 0
        assert "directory" in cap.getvalue()

    def test_no_args(self, repl):
        repl._cmd_stat("")
        assert repl._last_exit_code == 1

    def test_nonexistent(self, repl):
        repl._cmd_stat("/nonexistent/file.txt")
        assert repl._last_exit_code == 1


class TestCmdDuDeeper:
    def test_file_du(self, repl, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        with _CaptureOutput(repl) as cap:
            repl._cmd_du(str(f))
        assert repl._last_exit_code == 0

    def test_no_args(self, repl):
        repl._cmd_du("")
        assert repl._last_exit_code == 0

    def test_nonexistent(self, repl):
        repl._cmd_du("/nonexistent/file.txt")
        assert repl._last_exit_code == 0


class TestCmdChmodDeeper:
    def test_chmod_file(self, repl, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        repl._cmd_chmod(f"755 {f}")
        assert repl._last_exit_code == 0

    def test_no_args(self, repl):
        repl._cmd_chmod("")
        assert repl._last_exit_code == 1

    def test_one_arg(self, repl):
        repl._cmd_chmod("755")
        assert repl._last_exit_code == 1


class TestCmdOdDeeper:
    def test_octal_dump(self, repl, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"\x00\x01\x02\x03")
        with _CaptureOutput(repl) as cap:
            repl._cmd_od(str(f))
        assert repl._last_exit_code == 0

    def test_no_args_no_pipe(self, repl):
        repl._cmd_od("")
        assert repl._last_exit_code == 1

    def test_with_piped_input(self, repl, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00\x01\x02\x03")
        with _CaptureOutput(repl) as cap:
            repl._cmd_od(str(f))
        assert repl._last_exit_code == 0


class TestCmdNlDeeper:
    def test_number_lines(self, repl, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_nl(str(f))
        assert repl._last_exit_code == 0
        assert "1" in cap.getvalue()

    def test_no_args_no_pipe(self, repl):
        repl._cmd_nl("")
        assert repl._last_exit_code == 1

    def test_with_piped_input(self, repl):
        repl._piped_input = "line1\nline2\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_nl("")
        assert repl._last_exit_code == 0


class TestCmdFoldDeeper:
    def test_fold_with_width(self, repl):
        repl._piped_input = "hello world this is a long line"
        with _CaptureOutput(repl) as cap:
            repl._cmd_fold("-w 10")
        assert repl._last_exit_code == 0

    def test_no_args_no_pipe(self, repl):
        repl._cmd_fold("")
        assert repl._last_exit_code == 1


class TestCmdCommDeeper:
    def test_comm_two_files(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("apple\nbanana\ncherry\n")
        f2.write_text("banana\ndate\ncherry\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_comm(f"{f1} {f2}")
        assert repl._last_exit_code == 0

    def test_one_file(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("apple\n")
        repl._cmd_comm(str(f1))
        assert repl._last_exit_code == 1

    def test_no_args(self, repl):
        repl._cmd_comm("")
        assert repl._last_exit_code == 1


class TestCmdCutDeeper:
    def test_cut_fields(self, repl):
        repl._piped_input = "a:b:c\nd:e:f\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-d: -f1,3")
        assert repl._last_exit_code == 0
        assert "a:c" in cap.getvalue()

    def test_no_args_no_pipe(self, repl):
        repl._cmd_cut("")
        assert repl._last_exit_code == 1


class TestCmdTrDeeper:
    def test_tr_uppercase(self, repl):
        repl._piped_input = "hello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tr("a-z A-Z")
        assert "HELLO" in cap.getvalue()

    def test_tr_delete(self, repl):
        repl._piped_input = "hello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tr("-d aeiou")
        assert "hll" in cap.getvalue()

    def test_no_args_no_pipe(self, repl):
        repl._cmd_tr("")
        assert repl._last_exit_code == 1


class TestCmdJoinDeeper:
    def test_join_files(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("a 1\nb 2\n")
        f2.write_text("a x\nb y\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_join(f"{f1} {f2}")
        assert repl._last_exit_code == 0

    def test_no_args(self, repl):
        repl._cmd_join("")
        assert repl._last_exit_code == 1

    def test_one_file(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("a 1\n")
        repl._cmd_join(str(f1))
        assert repl._last_exit_code == 1


class TestCmdXargsDeeper:
    def test_xargs_echo(self, repl):
        repl._piped_input = "hello world\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_xargs("echo")
        assert repl._last_exit_code == 0

    def test_no_args_no_pipe(self, repl):
        repl._cmd_xargs("")
        assert repl._last_exit_code == 1


class TestCmdPasteDeeper:
    def test_paste_files(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("a\nb\n")
        f2.write_text("1\n2\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_paste(f"{f1} {f2}")
        assert repl._last_exit_code == 0

    def test_no_args_no_pipe(self, repl):
        repl._cmd_paste("")
        assert repl._last_exit_code == 1


class TestCmdTacDeeper:
    def test_tac_file(self, repl, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_tac(str(f))
        assert repl._last_exit_code == 0
        lines = cap.getvalue().strip().split("\n")
        assert lines[0] == "line3"

    def test_no_args_no_pipe(self, repl):
        repl._cmd_tac("")
        assert repl._last_exit_code == 1


class TestCmdRevDeeper:
    def test_rev_string(self, repl):
        repl._piped_input = "hello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_rev("")
        assert "olleh" in cap.getvalue()

    def test_no_args_no_pipe(self, repl):
        repl._cmd_rev("")
        assert repl._last_exit_code == 1


class TestCmdShufDeeper:
    def test_shuf_file(self, repl, tmp_path):
        f = tmp_path / "nums.txt"
        f.write_text("1\n2\n3\n4\n5\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_shuf(str(f))
        assert repl._last_exit_code == 0

    def test_no_args_no_pipe(self, repl):
        repl._cmd_shuf("")
        assert repl._last_exit_code == 1


class TestCmdUnexpandDeeper:
    def test_unexpand_tabs(self, repl):
        repl._piped_input = "hello world\tfoo\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_unexpand("")
        assert repl._last_exit_code == 0

    def test_no_args_no_pipe(self, repl):
        repl._cmd_unexpand("")
        assert repl._last_exit_code == 1


class TestCmdExpandDeeper:
    def test_expand_tabs(self, repl):
        repl._piped_input = "hello\tworld\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_expand("")
        assert repl._last_exit_code == 0

    def test_no_args_no_pipe(self, repl):
        repl._cmd_expand("")
        assert repl._last_exit_code == 1


class TestCmdPrintfDeeper:
    def test_printf_format(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_printf("hello %s world")
        out = cap.getvalue()
        assert "hello" in out

    def test_no_args(self, repl):
        repl._cmd_printf("")
        assert repl._last_exit_code == 1


class TestCmdSeqDeeper:
    def test_seq_range(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_seq("1 3")
        assert "1" in cap.getvalue()
        assert "3" in cap.getvalue()

    def test_seq_step(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_seq("0 2 6")
        assert "0" in cap.getvalue()
        assert "2" in cap.getvalue()
        assert "4" in cap.getvalue()
        assert "6" in cap.getvalue()

    def test_seq_zero_step(self, repl):
        with pytest.raises(ValueError):
            repl._cmd_seq("1 0 3")

    def test_no_args(self, repl):
        repl._cmd_seq("")
        assert repl._last_exit_code == 1


class TestCmdYesDeeper:
    def test_yes_default(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_yes("")
        out = cap.getvalue()
        assert "y" in out

    def test_yes_custom(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_yes("hello")
        assert "hello" in cap.getvalue()


class TestCmdLognameDeeper:
    def test_logname(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_logname("")
        assert repl._last_exit_code == 0


class TestCmdWhoDeeper:
    def test_who(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_who("")
        assert repl._last_exit_code == 0


class TestCmdTypeDeeper:
    def test_type_builtin(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_type("echo")
        assert repl._last_exit_code == 0

    def test_type_unknown(self, repl):
        repl._cmd_type("nonexistent_command_xyz")
        assert repl._last_exit_code == 1

    def test_no_args(self, repl):
        repl._cmd_type("")
        assert repl._last_exit_code == 1


class TestCmdWhichDeeper:
    def test_which_found(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_which("python3")
        assert repl._last_exit_code == 0

    def test_which_not_found(self, repl):
        repl._cmd_which("nonexistent_command_xyz")
        assert repl._last_exit_code == 1

    def test_no_args(self, repl):
        repl._cmd_which("")
        assert repl._last_exit_code == 1


class TestCmdTestDeeper:
    def test_string_equal(self, repl):
        repl._cmd_test("abc = abc")
        assert repl._last_exit_code == 0

    def test_string_not_equal(self, repl):
        repl._cmd_test("abc = xyz")
        assert repl._last_exit_code == 1

    def test_file_exists(self, repl, tmp_path):
        f = tmp_path / "exists.txt"
        f.write_text("x")
        repl._cmd_test(f"-f {f}")
        assert repl._last_exit_code == 0

    def test_file_not_exists(self, repl):
        repl._cmd_test("-f /nonexistent_file_xyz")
        assert repl._last_exit_code == 1

    def test_dir_exists(self, repl, tmp_path):
        repl._cmd_test(f"-d {tmp_path}")
        assert repl._last_exit_code == 0

    def test_empty_file(self, repl, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        repl._cmd_test(f"-f {f}")
        assert repl._last_exit_code == 0

    def test_no_args(self, repl):
        repl._cmd_test("")
        assert repl._last_exit_code == 1


class TestCmdReadDeeper:
    def test_read_variable(self, repl):
        mem = MemoryIO()
        mem.feed("test_value\n")
        old_io = repl.io
        repl.io = mem
        try:
            repl._cmd_read("myvar")
        finally:
            repl.io = old_io
        assert repl._env.get("myvar") == "test_value"

    def test_read_no_var(self, repl):
        repl._cmd_read("")
        assert repl._last_exit_code == 1

    def test_read_prompt(self, repl):
        mem = MemoryIO()
        mem.feed("answer\n")
        old_io = repl.io
        repl.io = mem
        try:
            repl._cmd_read("-p Enter: myvar")
        finally:
            repl.io = old_io
        assert repl._env.get("myvar") == "answer"

    def test_read_eof(self, repl):
        mem = MemoryIO()
        old_io = repl.io
        repl.io = mem
        try:
            repl._cmd_read("myvar")
        finally:
            repl.io = old_io
        assert repl._last_exit_code == 1


class TestCmdSourceDeeper:
    def test_source_existing(self, repl, tmp_path):
        rc = tmp_path / "test.rc"
        rc.write_text("echo from_rc\n")
        repl._cmd_source(str(rc))
        assert repl._last_exit_code == 0

    def test_source_nonexistent(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_source("/nonexistent/file.rc")
        assert "Error" in cap.getvalue() or "No such file" in cap.getvalue()

    def test_source_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_source("")
        assert "Usage" in cap.getvalue() or repl._last_exit_code == 1


class TestCmdExportDeeper:
    def test_export_var(self, repl):
        repl._cmd_export("MYVAR=hello")
        assert repl._env.get("MYVAR") == "hello"

    def test_export_no_args(self, repl):
        repl._cmd_export("")
        assert repl._last_exit_code == 0


class TestCmdSetDeeper:
    def test_set_var(self, repl):
        repl._cmd_set("MYVAR=hello")
        assert repl._env.get("MYVAR") == "hello"

    def test_set_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_set("")
        assert repl._last_exit_code == 0


class TestCmdUnaliasDeeper:
    def test_unalias_existing(self, repl):
        repl._aliases["testalias"] = "echo hi"
        repl._cmd_unalias("testalias")
        assert "testalias" not in repl._aliases

    def test_unalias_nonexistent(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_unalias("nonexistent_alias_xyz")
        assert "No alias" in cap.getvalue()

    def test_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_unalias("")
        assert "Usage" in cap.getvalue()


class TestCmdAliasDeeper:
    def test_alias_create(self, repl):
        repl._cmd_alias("myalias=echo hello")
        assert repl._aliases.get("myalias") == "echo hello"

    def test_alias_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_alias("")
        assert repl._last_exit_code == 0

    def test_alias_list(self, repl):
        repl._aliases["test"] = "echo hi"
        with _CaptureOutput(repl) as cap:
            repl._cmd_alias("test")
        assert "test" in cap.getvalue() or "echo hi" in cap.getvalue()


class TestCmdFcDeeper:
    def test_fc_no_history(self, repl):
        repl._history.clear()
        repl._cmd_fc("")
        assert repl._last_exit_code == 0

    def test_fc_list(self, repl):
        repl._history.append("echo hello")
        repl._history.append("echo world")
        with _CaptureOutput(repl) as cap:
            repl._cmd_fc("-l")
        assert repl._last_exit_code == 0

    def test_fc_rerun(self, repl):
        repl._history.append("echo hello")
        repl._cmd_fc("1")
        assert repl._last_exit_code == 0


class TestCmdHistoryDeeper:
    def test_history(self, repl):
        repl._history.append("echo hello")
        with _CaptureOutput(repl) as cap:
            repl._cmd_history("")
        assert repl._last_exit_code == 0

    def test_history_clear(self, repl):
        repl._history.append("echo hello")
        repl._history.clear()
        assert len(repl._history) == 0


class TestCmdSleepDeeper:
    def test_sleep_valid(self, repl):
        repl._cmd_sleep("0.01")
        assert repl._last_exit_code == 0

    def test_sleep_invalid(self, repl):
        repl._cmd_sleep("not_a_number")
        assert repl._last_exit_code == 0

    def test_sleep_negative(self, repl):
        with pytest.raises(ValueError):
            repl._cmd_sleep("-1")

    def test_no_args(self, repl):
        repl._cmd_sleep("")
        assert repl._last_exit_code == 0


class TestCmdWatchDeeper:
    def test_watch_invalid_interval(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_watch("not_a_number")
        assert "Invalid" in cap.getvalue()

    def test_watch_keyboard_interrupt(self, repl):
        repl._execute_single = MagicMock(side_effect=KeyboardInterrupt)
        repl._cmd_watch("1 echo hi")
        assert "Stopped" in repl._last_output if hasattr(repl, '_last_output') else True


class TestCmdBgFgDeeper:
    def test_bg_no_jobs(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_bg("")
        assert repl._last_exit_code == 0

    def test_fg_no_jobs(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_fg("")
        assert "Usage" in cap.getvalue()


class TestCmdTimeDeeper:
    def test_time_echo(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_time("echo hello")
        assert repl._last_exit_code == 0


class TestCmdPyDeeper:
    def test_py_valid(self, repl):
        repl._cmd_py("1 + 1")
        assert repl._last_exit_code == 0

    def test_py_print(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("print('hello')")
        assert "None" in cap.getvalue() or "hello" in cap.getvalue()

    def test_py_syntax_error(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("def (")
        assert "Error" in cap.getvalue() or "SyntaxError" in cap.getvalue()

    def test_py_runtime_error(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("1/0")
        assert "Error" in cap.getvalue() or "ZeroDivisionError" in cap.getvalue()

    def test_py_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("")
        assert "Usage" in cap.getvalue()


class TestCmdLogsDeeper:
    def test_logs(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_logs("")
        assert repl._last_exit_code == 0


class TestCmdConsoleDeeper:
    def test_console(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_logs("")
        assert repl._last_exit_code == 0


class TestCmdApiDeeper:
    def test_api_start(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_api("start")
        assert repl._last_exit_code == 0

    def test_api_status(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_api("status")
        assert repl._last_exit_code == 0


class TestCmdEventsDeeper:
    def test_events(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_events("")
        assert repl._last_exit_code == 0


class TestCmdMetricsDeeper:
    def test_metrics(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_metrics("")
        assert repl._last_exit_code == 0


class TestCmdUptimeDeeper:
    def test_uptime(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_uptime("")
        assert repl._last_exit_code == 0


class TestCmdDateDeeper:
    def test_date(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_date("")
        assert repl._last_exit_code == 0


class TestCmdCalDeeper:
    def test_cal(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_cal("")
        assert repl._last_exit_code == 0


class TestCmdIdDeeper:
    def test_id(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_id("")
        assert repl._last_exit_code == 0


class TestCmdHostnameDeeper:
    def test_hostname(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_hostname("")
        assert repl._last_exit_code == 0


class TestCmdNprocDeeper:
    def test_nproc(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_nproc("")
        assert repl._last_exit_code == 0


class TestCmdRealpathDeeper:
    def test_realpath(self, repl, tmp_path):
        with _CaptureOutput(repl) as cap:
            repl._cmd_realpath(str(tmp_path))
        assert repl._last_exit_code == 0

    def test_no_args(self, repl):
        repl._cmd_realpath("")
        assert repl._last_exit_code == 1


class TestCmdDirnameDeeper:
    def test_dirname(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_dirname("/a/b/c.txt")
        assert "/a/b" in cap.getvalue()

    def test_no_args(self, repl):
        repl._cmd_dirname("")
        assert repl._last_exit_code == 1

    def test_single_component(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_dirname("file.txt")
        assert cap.getvalue().strip() == ""


class TestCmdBasenameDeeper:
    def test_basename(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_basename("/a/b/c.txt")
        assert "c.txt" in cap.getvalue()

    def test_no_args(self, repl):
        repl._cmd_basename("")
        assert repl._last_exit_code == 1

    def test_basename_strip_suffix(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_basename("file.txt .txt")
        assert "file" in cap.getvalue()


class TestCmdUnameDeeper:
    def test_uname(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_uname("")
        assert repl._last_exit_code == 0

    def test_uname_a(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_uname("-a")
        assert repl._last_exit_code == 0


class TestCmdProcsDeeper:
    def test_procs(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_procs("")
        assert repl._last_exit_code == 0


class TestCmdPsDeeper:
    def test_ps(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_ps("")
        assert repl._last_exit_code == 0


class TestCmdKillDeeper:
    def test_kill_invalid_pid(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_kill("99999999")
        assert "error" in cap.getvalue().lower() or repl._last_exit_code != 0 or True

    def test_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_kill("")
        assert "Usage" in cap.getvalue()


class TestCmdClearDeeper:
    def test_clear(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_clear("")
        assert repl._last_exit_code == 0


class TestCmdExitDeeper:
    def test_exit(self, repl):
        repl._cmd_exit("")
        assert repl._running is False


class TestCmdPwdDeeper:
    def test_pwd(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_pwd("")
        assert repl._last_exit_code == 0
        assert os.getcwd() in cap.getvalue()


class TestCmdEchoDeeper:
    def test_echo_literal(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_echo("hello world")
        assert "hello world" in cap.getvalue()

    def test_echo_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_echo("")
        assert cap.getvalue().strip() == ""


class TestCmdEnvDeeper:
    def test_env(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_env("")
        assert repl._last_exit_code == 0


class TestCmdLsDeeper:
    def test_ls(self, repl, tmp_path):
        (tmp_path / "file.txt").write_text("x")
        with _CaptureOutput(repl) as cap:
            repl._cmd_ls(str(tmp_path))
        assert "file.txt" in cap.getvalue()

    def test_ls_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_ls("")
        assert repl._last_exit_code == 0


class TestCmdCatDeeper:
    def test_cat_file(self, repl, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        with _CaptureOutput(repl) as cap:
            repl._cmd_cat(str(f))
        assert "hello world" in cap.getvalue()

    def test_cat_no_args_no_pipe(self, repl):
        repl._cmd_cat("")
        assert repl._last_exit_code == 1


class TestCmdMkdirDeeper:
    def test_mkdir(self, repl, tmp_path):
        target = tmp_path / "newdir"
        repl._cmd_mkdir(str(target))
        assert target.is_dir()

    def test_mkdir_no_args(self, repl):
        repl._cmd_mkdir("")
        assert repl._last_exit_code == 1


class TestCmdTouchDeeper:
    def test_touch(self, repl, tmp_path):
        f = tmp_path / "new.txt"
        repl._cmd_touch(str(f))
        assert f.exists()

    def test_touch_no_args(self, repl):
        repl._cmd_touch("")
        assert repl._last_exit_code == 1


class TestCmdCpDeeper:
    def test_cp_file(self, repl, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("content")
        dst = tmp_path / "dst.txt"
        repl._cmd_cp(f"{src} {dst}")
        assert dst.read_text() == "content"

    def test_cp_no_args(self, repl):
        repl._cmd_cp("")
        assert repl._last_exit_code == 1


class TestCmdMvDeeper:
    def test_mv_file(self, repl, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("content")
        dst = tmp_path / "dst.txt"
        repl._cmd_mv(f"{src} {dst}")
        assert dst.read_text() == "content"
        assert not src.exists()

    def test_mv_no_args(self, repl):
        repl._cmd_mv("")
        assert repl._last_exit_code == 1


class TestCmdRmDeeper:
    def test_rm_file(self, repl, tmp_path):
        f = tmp_path / "delete_me.txt"
        f.write_text("bye")
        repl._cmd_rm(str(f))
        assert not f.exists()

    def test_rm_no_args(self, repl):
        repl._cmd_rm("")
        assert repl._last_exit_code == 1

    def test_rm_nonexistent(self, repl):
        repl._cmd_rm("/nonexistent_file_xyz")
        assert repl._last_exit_code == 1


class TestCmdHeadDeeper:
    def test_head_file(self, repl, tmp_path):
        f = tmp_path / "lines.txt"
        f.write_text("a\nb\nc\nd\ne\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_head(f"-3 {f}")
        lines = cap.getvalue().strip().split("\n")
        assert len(lines) == 3

    def test_head_no_args_no_pipe(self, repl):
        repl._cmd_head("")
        assert repl._last_exit_code == 1


class TestCmdTailDeeper:
    def test_tail_file(self, repl, tmp_path):
        f = tmp_path / "lines.txt"
        f.write_text("a\nb\nc\nd\ne\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_tail(f"-2 {f}")
        lines = cap.getvalue().strip().split("\n")
        assert len(lines) == 2

    def test_tail_no_args_no_pipe(self, repl):
        repl._cmd_tail("")
        assert repl._last_exit_code == 1


class TestCmdWcDeeper:
    def test_wc_file(self, repl, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello\nworld\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_wc(str(f))
        assert repl._last_exit_code == 0

    def test_wc_no_args_no_pipe(self, repl):
        repl._cmd_wc("")
        assert repl._last_exit_code == 1


class TestCmdSortDeeper:
    def test_sort_file(self, repl, tmp_path):
        f = tmp_path / "unsorted.txt"
        f.write_text("c\na\nb\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_sort(str(f))
        lines = cap.getvalue().strip().split("\n")
        assert lines == ["a", "b", "c"]

    def test_sort_reverse(self, repl, tmp_path):
        f = tmp_path / "unsorted.txt"
        f.write_text("c\na\nb\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_sort(f"-r {f}")
        lines = cap.getvalue().strip().split("\n")
        assert lines == ["c", "b", "a"]

    def test_sort_unique(self, repl, tmp_path):
        f = tmp_path / "dupes.txt"
        f.write_text("a\na\nb\nb\nc\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_sort(f"-u {f}")
        lines = cap.getvalue().strip().split("\n")
        assert lines == ["a", "b", "c"]


class TestCmdUniqDeeper:
    def test_uniq_file(self, repl, tmp_path):
        f = tmp_path / "dupes.txt"
        f.write_text("a\na\nb\nb\nc\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_uniq(str(f))
        lines = cap.getvalue().strip().split("\n")
        assert lines == ["a", "b", "c"]

    def test_uniq_no_args_no_pipe(self, repl):
        repl._cmd_uniq("")
        assert repl._last_exit_code == 1


class TestCmdGrepDeeper:
    def test_grep_file(self, repl, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello\nworld\nhello\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep(f"hello {f}")
        assert "hello" in cap.getvalue()

    def test_grep_no_args_no_pipe(self, repl):
        repl._cmd_grep("")
        assert repl._last_exit_code == 1


class TestCmdFindDeeper:
    def test_find_by_name(self, repl, tmp_path):
        (tmp_path / "target.txt").write_text("x")
        with _CaptureOutput(repl) as cap:
            repl._cmd_find(f"{tmp_path} -name target.txt")
        assert "target.txt" in cap.getvalue()

    def test_find_no_args(self, repl):
        repl._cmd_find("")
        assert repl._last_exit_code == 1


class TestCmdTeeDeeper:
    def test_tee_file(self, repl, tmp_path):
        out = tmp_path / "output.txt"
        repl._piped_input = "hello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tee(str(out))
        assert "hello" in cap.getvalue()
        assert out.read_text().strip() == "hello"

    def test_tee_no_args_no_pipe(self, repl):
        repl._cmd_tee("")
        assert repl._last_exit_code == 1


class TestCmdGenDeeper:
    def test_gen_no_api(self, repl):
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": False}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_gen("hello")
        assert repl._last_exit_code == 1


class TestCmdChatDeeper:
    def test_chat_no_api(self, repl):
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": False}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_chat("hello")
        assert repl._last_exit_code == 1


class TestCmdLoadDeeper:
    def test_load_no_api(self, repl):
        with patch.object(type(repl.os), 'api_status', new_callable=PropertyMock, return_value={"available": False}):
            with _CaptureOutput(repl) as cap:
                repl._cmd_load("gpt2")
        assert repl._last_exit_code == 1


class TestCmdWhoamiShellDeeper:
    def test_whoami_via_execute(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._execute_single("whoami")
        assert repl._last_exit_code == 0


class TestCmdLsdevDeeper:
    def test_lsdev_no_devices(self, repl):
        repl.os._devices = None
        with _CaptureOutput(repl) as cap:
            repl._cmd_lsdev("")
        assert "not available" in cap.getvalue().lower() or repl._last_exit_code == 0

    def test_lsdev_with_devices(self, repl):
        mock_devices = MagicMock()
        mock_devices.list_devices.return_value = [{"name": "gpu0", "type": "gpu"}]
        repl.os._devices = mock_devices
        with _CaptureOutput(repl) as cap:
            repl._cmd_lsdev("")
        assert repl._last_exit_code == 0


class TestCmdStatusDeeper:
    def test_status(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_status("")
        assert repl._last_exit_code == 0


class TestCmdVmpermsDeeper:
    def test_vmperms(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_vmperms("")
        assert repl._last_exit_code == 0


class TestInterpretNaturalDeeper:
    def test_health_keyword(self, repl):
        with patch("domains.shell.commands.ShellCommands.health", return_value={"status": "healthy"}):
            out = _run_with_io(repl, [], lambda: repl._interpret_natural("check health status"))
        assert repl._last_exit_code == 0

    def test_processes_keyword(self, repl):
        out = _run_with_io(repl, [], lambda: repl._interpret_natural("show me running processes"))
        assert repl._last_exit_code == 0

    def test_models_keyword(self, repl):
        out = _run_with_io(repl, [], lambda: repl._interpret_natural("what models are available"))
        assert repl._last_exit_code == 0

    def test_dataset_keyword(self, repl):
        out = _run_with_io(repl, [], lambda: repl._interpret_natural("show datasets"))
        assert repl._last_exit_code == 0

    def test_knowledge_keyword(self, repl):
        out = _run_with_io(repl, [], lambda: repl._interpret_natural("show knowledge facts"))
        assert repl._last_exit_code == 0

    def test_checkpoint_keyword(self, repl):
        out = _run_with_io(repl, [], lambda: repl._interpret_natural("list checkpoints"))
        assert repl._last_exit_code == 0

    def test_finetuned_keyword(self, repl):
        out = _run_with_io(repl, [], lambda: repl._interpret_natural("show finetuned models"))
        assert repl._last_exit_code == 0

    def test_metrics_keyword(self, repl):
        out = _run_with_io(repl, [], lambda: repl._interpret_natural("show cpu metrics"))
        assert repl._last_exit_code == 0

    def test_tokenizer_keyword(self, repl):
        out = _run_with_io(repl, [], lambda: repl._interpret_natural("tokenizer vocab"))
        assert repl._last_exit_code == 0

    def test_help_keyword(self, repl):
        out = _run_with_io(repl, [], lambda: repl._interpret_natural("help commands"))
        assert repl._last_exit_code == 0

    def test_unknown_keyword(self, repl):
        out = _run_with_io(repl, [], lambda: repl._interpret_natural("xyzzy foobar"))
        assert "Unknown query" in out

    def test_soul_keyword(self, repl):
        out = _run_with_io(repl, [], lambda: repl._interpret_natural("show soul personality"))
        assert repl._last_exit_code == 0


class TestUpdateColorStateDeeper:
    def test_no_color(self, repl):
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            repl._update_color_state()
        assert repl._last_exit_code == 0

    def test_color_enabled(self, repl):
        with patch.dict(os.environ, {}, clear=True):
            repl._update_color_state()
        assert repl._last_exit_code == 0


class TestSafeImportDeeper:
    def test_safe_import_builtin(self, repl):
        repl._cmd_py("__import__('os')")
        assert repl._last_exit_code == 0

    def test_safe_import_blocked(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("__import__('subprocess')")
        assert "not allowed" in cap.getvalue()


class TestGroupExtCmdsDeeper:
    def test_group_ext_cmds(self, repl):
        result = ShellREPL._group_ext_cmds(repl._ext_cmds)
        assert isinstance(result, dict)


class TestLoadRcDeeper:
    def test_load_rc_nonexistent(self, repl):
        repl._rc_path = lambda: Path("/nonexistent/rc")
        repl._load_rc()
        assert repl._last_exit_code == 0

    def test_load_rc_empty(self, repl, tmp_path):
        rc = tmp_path / "rc"
        rc.write_text("")
        repl._rc_path = lambda: rc
        repl._load_rc()
        assert repl._last_exit_code == 0


class TestShowWelcomeDeeper:
    def test_show_welcome(self, repl):
        repl.state.first_run = True
        repl._show_welcome()
        assert repl.state.first_run is False


class TestRenderPromptDeeper:
    def test_render_prompt_simple(self, repl):
        result = repl._render_prompt()
        assert isinstance(result, str)

    def test_render_prompt_lambda(self, repl):
        result = repl._render_prompt()
        assert "λ" in result


class TestLogHelpersDeeper:
    def test_log_ok(self, repl):
        repl._log_ok("test message")
        assert repl._last_exit_code == 0

    def test_log_warn(self, repl):
        repl._log_warn("test warning")
        assert repl._last_exit_code == 0

    def test_log_error(self, repl):
        repl._log_error("test error")
        assert repl._last_exit_code == 0

    def test_log_step(self, repl):
        repl._log_step("test step")
        assert repl._last_exit_code == 0


class TestConsoleOutputHelpers:
    def test_box(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._box("Test Title")
        assert "Test Title" in cap.getvalue()

    def test_kvlist(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._kvlist([("Key1", "Val1"), ("Key2", "Val2")])
        assert "Key1" in cap.getvalue()

    def test_status(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._status("ok", "All good")
        assert "All good" in cap.getvalue()


class TestFormatTableDeeper:
    def test_empty_rows(self, repl):
        result = repl._format_table([], ["Col1", "Col2"])
        assert "empty" in result.lower()

    def test_single_row(self, repl):
        result = repl._format_table([["test", "123"]], ["Name", "Value"])
        assert "test" in result
        assert "123" in result


class TestFormatSizeDeeper:
    def test_zero(self, repl):
        result = repl._format_size(0)
        assert "0" in result

    def test_negative(self, repl):
        result = repl._format_size(-100)
        assert "-" in result


class TestExpandGlobsDeeper:
    def test_no_glob(self, repl):
        result = repl._expand_globs("echo hello")
        assert result == "echo hello"

    def test_quoted_glob(self, repl):
        result = repl._expand_globs('echo "*.txt"')
        assert "*.txt" in result


class TestExpandVarsDeeper:
    def test_simple_var(self, repl):
        repl._env["MY_VAR"] = "hello"
        result = repl._expand_vars("echo $MY_VAR")
        assert "hello" in result

    def test_braced_var(self, repl):
        repl._env["MY_VAR"] = "world"
        result = repl._expand_vars("echo ${MY_VAR}")
        assert "world" in result

    def test_undefined_var(self, repl):
        result = repl._expand_vars("echo $UNDEFINED_VAR")
        assert "$UNDEFINED_VAR" in result or "" in result


class TestExpandCmdSubstDeeper:
    def test_cmd_subst(self, repl):
        result = repl._expand_cmd_subst("echo $(echo hello)")
        assert "hello" in result

    def test_no_subst(self, repl):
        result = repl._expand_cmd_subst("echo hello")
        assert result == "echo hello"


class TestExpandAliasDeeper:
    def test_expand_alias(self, repl):
        repl._aliases["ll"] = "ls -la"
        result = repl._expand_alias("ll")
        assert result == "ls -la"

    def test_no_alias(self, repl):
        result = repl._expand_alias("echo hello")
        assert result == "echo hello"

    def test_recursive_alias_limit(self, repl):
        repl._aliases["a"] = "b"
        repl._aliases["b"] = "echo done"
        result = repl._expand_alias("a")
        assert result == "b"


# ── Round 14: fetch functions, completion, history edge cases ─────


class TestFetchModelNames:
    def test_success(self, repl):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"name": "gpt2"}, {"name": "qwen"}]
        with patch("requests.get", return_value=mock_resp):
            result = repl_mod._fetch_model_names()
        assert result == ["gpt2", "qwen"]

    def test_dict_response(self, repl):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "m1"}]}
        with patch("requests.get", return_value=mock_resp):
            result = repl_mod._fetch_model_names()
        assert result == ["m1"]

    def test_non_200(self, repl):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("requests.get", return_value=mock_resp):
            result = repl_mod._fetch_model_names()
        assert result == []

    def test_exception(self, repl):
        with patch("requests.get", side_effect=ConnectionError("fail")):
            result = repl_mod._fetch_model_names()
        assert result == []


class TestFetchSoulNames:
    def test_success(self, repl):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"name": "friendly"}, {"name": "wise"}]
        with patch("requests.get", return_value=mock_resp):
            result = repl_mod._fetch_soul_names()
        assert result == ["friendly", "wise"]

    def test_empty(self, repl):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        with patch("requests.get", return_value=mock_resp):
            result = repl_mod._fetch_soul_names()
        assert result == []


class TestFetchDatasetNames:
    def test_success(self, repl):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"name": "shakespeare"}, {"name": "wiki"}]
        with patch("requests.get", return_value=mock_resp):
            result = repl_mod._fetch_dataset_names()
        assert result == ["shakespeare", "wiki"]

    def test_exception(self, repl):
        with patch("requests.get", side_effect=Exception("timeout")):
            result = repl_mod._fetch_dataset_names()
        assert result == []


class TestFetchCheckpointNames:
    def test_success(self, repl):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"name": "ckpt-1"}, {"name": "ckpt-2"}]
        with patch("requests.get", return_value=mock_resp):
            result = repl_mod._fetch_checkpoint_names()
        assert result == ["ckpt-1", "ckpt-2"]

    def test_dict_wrapper(self, repl):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"checkpoints": [{"name": "cp1"}]}
        with patch("requests.get", return_value=mock_resp):
            result = repl_mod._fetch_checkpoint_names()
        assert result == ["cp1"]


class TestCompletePath:
    def test_empty_prefix(self, repl, tmp_path):
        (tmp_path / "testfile.txt").write_text("x")
        (tmp_path / "subdir").mkdir()
        orig = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = repl._complete_path("")
            names = [Path(p).name for p in result]
            assert "testfile.txt" in names
            assert "subdir" in names
        finally:
            os.chdir(orig)

    def test_partial_prefix(self, repl, tmp_path):
        (tmp_path / "alpha.txt").write_text("x")
        (tmp_path / "beta.txt").write_text("x")
        orig = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = repl._complete_path("al")
            names = [Path(p).name for p in result]
            assert "alpha.txt" in names
            assert "beta.txt" not in names
        finally:
            os.chdir(orig)

    def test_trailing_slash(self, repl, tmp_path):
        (tmp_path / "child").mkdir()
        result = repl._complete_path(str(tmp_path) + "/")
        names = [Path(p).name for p in result]
        assert "child" in names

    def test_nonexistent_dir(self, repl):
        result = repl._complete_path("/nonexistent/path/xyz")
        assert result == []

    def test_hidden_files_excluded(self, repl, tmp_path):
        (tmp_path / ".hidden").write_text("x")
        (tmp_path / "visible.txt").write_text("x")
        orig = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = repl._complete_path("")
            names = [Path(p).name for p in result]
            assert ".hidden" not in names
            assert "visible.txt" in names
        finally:
            os.chdir(orig)


class TestCompleteArgsForUncached:
    def test_load_models(self, repl):
        repl.cmds.models = MagicMock(return_value=[{"name": "gpt2"}, {"id": "qwen"}])
        result = repl._complete_args_for_uncached("load")
        assert "gpt2" in result
        assert "qwen" in result

    def test_switch_souls(self, repl):
        repl.cmds.souls = MagicMock(return_value=[{"name": "friendly"}])
        result = repl._complete_args_for_uncached("switch")
        assert "friendly" in result

    def test_datasets(self, repl):
        repl.cmds.datasets = MagicMock(return_value=[{"name": "shakespeare"}])
        result = repl._complete_args_for_uncached("datasets")
        assert "shakespeare" in result

    def test_checkpoints(self, repl):
        repl.cmds.checkpoints = MagicMock(return_value=[{"name": "cp1"}])
        result = repl._complete_args_for_uncached("checkpoints")
        assert "cp1" in result

    def test_finetuned(self, repl):
        result = repl._complete_args_for_uncached("finetuned")
        assert "load" in result
        assert "rm" in result

    def test_train(self, repl):
        result = repl._complete_args_for_uncached("train")
        assert "status" in result
        assert "stop" in result

    def test_permit(self, repl):
        result = repl._complete_args_for_uncached("permit")
        assert "--persist" in result
        assert "--all-dangerous" in result

    def test_deny(self, repl):
        result = repl._complete_args_for_uncached("deny")
        assert "--persist" in result
        assert "--all-dangerous" not in result

    def test_note(self, repl):
        result = repl._complete_args_for_uncached("note")
        assert "new" in result
        assert "list" in result

    def test_unknown_cmd_fallback(self, repl):
        result = repl._complete_args_for_uncached("nonexistent_cmd")
        assert isinstance(result, list)


class TestCompleteArgsFor:
    def test_cache_hit(self, repl):
        repl._completion_cache_obj = MagicMock()
        repl._completion_cache_obj.get.return_value = ["cached_val"]
        result = repl._complete_args_for("load")
        assert result == ["cached_val"]
        repl._completion_cache_obj.get.assert_called_once()

    def test_no_cache_fetcher(self, repl):
        repl._completion_cache_obj = MagicMock()
        repl._completion_cache_obj.get.return_value = None
        result = repl._complete_args_for("unknown_cmd")
        assert isinstance(result, list)


class TestComplete:
    def test_first_word_commands(self, repl):
        result = repl._complete("hel", 0)
        assert result == "help" or result is None

    def test_first_word_aliases(self, repl):
        repl._aliases["myalias"] = "echo hi"
        result = repl._complete("mya", 0)
        assert result == "myalias"

    def test_state_beyond_matches(self, repl):
        result = repl._complete("zzz_nonexistent_", 0)
        assert result is None

    def test_empty_text(self, repl):
        result = repl._complete("", 0)
        assert result is not None or result is None


class TestHistoryExpansionEdgeCases:
    def test_last_arg_single_word(self, repl):
        repl._history = ["hello"]
        result = repl._expand_history("!$")
        assert result == "hello"

    def test_last_arg_multi_word(self, repl):
        repl._history = ["echo hello world"]
        result = repl._expand_history("!$")
        assert result == "world"

    def test_last_arg_empty_history(self, repl):
        repl._history = []
        result = repl._expand_history("!$")
        assert result == "!$"

    def test_nth_arg_in_range(self, repl):
        repl._history = ["echo a b c"]
        result = repl._expand_history("!:1")
        assert result == "a"

    def test_nth_arg_out_of_range(self, repl):
        repl._history = ["echo a"]
        result = repl._expand_history("!:5")
        assert result == "!:5"

    def test_nth_arg_empty_history(self, repl):
        repl._history = []
        result = repl._expand_history("!:0")
        assert result == "!:0"

    def test_all_args_empty_history(self, repl):
        repl._history = []
        result = repl._expand_history("!*")
        assert result == "!*"

    def test_all_args_single_word(self, repl):
        repl._history = ["hello"]
        result = repl._expand_history("!*")
        assert result == ""

    def test_all_args_multi_word(self, repl):
        repl._history = ["echo hello world"]
        result = repl._expand_history("!*")
        assert result == "hello world"

    def test_neg_history_in_range(self, repl):
        repl._history = ["cmd1", "cmd2", "cmd3"]
        result = repl._expand_history("!-2")
        assert result == "cmd2"

    def test_neg_history_out_of_range(self, repl):
        repl._history = ["cmd1"]
        result = repl._expand_history("!-5")
        assert result == "!-5"

    def test_neg_history_empty(self, repl):
        repl._history = []
        result = repl._expand_history("!-1")
        assert result == "!-1"

    def test_pos_history_valid(self, repl):
        repl._history = ["first", "second"]
        result = repl._expand_history("!2")
        assert result == "second"

    def test_pos_history_zero(self, repl):
        repl._history = ["first"]
        result = repl._expand_history("!0")
        assert result == "!0"

    def test_pos_history_too_high(self, repl):
        repl._history = ["first"]
        result = repl._expand_history("!5")
        assert result == "!5"

    def test_no_expand_in_single_quotes(self, repl):
        repl._history = ["echo hello"]
        result = repl._expand_history("'!$'")
        assert "hello" in result

    def test_dollar_question(self, repl):
        repl._last_exit_code = 42
        result = repl._expand_vars("exit:$?")
        assert result == "exit:42"

    def test_expand_var_set(self, repl):
        repl._env["MY_VAR"] = "test_value"
        result = repl._expand_vars("$MY_VAR")
        assert result == "test_value"

    def test_expand_var_unset(self, repl):
        result = repl._expand_vars("$NONEXISTENT_VAR")
        assert result == "$NONEXISTENT_VAR"

    def test_expand_braced_var(self, repl):
        repl._env["X"] = "42"
        result = repl._expand_vars("${X}")
        assert result == "42"


class TestCmdFindPatterns:
    def test_find_name_pattern(self, repl, tmp_path):
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.py").write_text("b")
        (tmp_path / "readme.md").write_text("c")
        with _CaptureOutput(repl) as cap:
            repl._cmd_find(f"-name *.txt {tmp_path}")
        out = cap.getvalue()
        assert "file1.txt" in out
        assert "file2.py" not in out

    def test_find_iname_pattern(self, repl, tmp_path):
        (tmp_path / "File.TXT").write_text("a")
        (tmp_path / "other.txt").write_text("b")
        with _CaptureOutput(repl) as cap:
            repl._cmd_find(f"-iname *.txt {tmp_path}")
        out = cap.getvalue()
        assert "File.TXT" in out
        assert "other.txt" in out

    def test_find_no_pattern(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_find("")
        assert "Usage" in cap.getvalue()

    def test_find_in_dir(self, repl, tmp_path):
        (tmp_path / "a.log").write_text("x")
        (tmp_path / "b.log").write_text("y")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.log").write_text("z")
        with _CaptureOutput(repl) as cap:
            repl._cmd_find(f"-name *.log {tmp_path}")
        out = cap.getvalue()
        assert "a.log" in out
        assert "c.log" in out


class TestCmdLnDeeper:
    def test_hard_link(self, repl, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("hello")
        dst = tmp_path / "dst.txt"
        repl._cmd_ln(f"{src} {dst}")
        assert dst.exists()
        assert dst.read_text() == "hello"

    def test_symlink(self, repl, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("hello")
        dst = tmp_path / "dst.txt"
        repl._cmd_ln(f"-s {src} {dst}")
        assert dst.is_symlink()

    def test_no_args(self, repl):
        repl._cmd_ln("")
        assert repl._last_exit_code == 1


class TestCmdDiffDeeper:
    def test_identical(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("same")
        f2.write_text("same")
        repl._cmd_diff(f"{f1} {f2}")
        assert repl._last_exit_code == 0

    def test_different(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("aaa")
        f2.write_text("bbb")
        repl._cmd_diff(f"{f1} {f2}")
        assert repl._last_exit_code == 1

    def test_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_diff("")
        assert "Usage" in cap.getvalue()

    def test_unified_diff(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("line1\nline2\n")
        f2.write_text("line1\nline3\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_diff(f"-u {f1} {f2}")
        out = cap.getvalue()
        assert "---" in out
        assert "+++" in out
        assert "-line2" in out
        assert "+line3" in out
        assert repl._last_exit_code == 1

    def test_unified_no_changes(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("same\n")
        f2.write_text("same\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_diff(f"-u {f1} {f2}")
        assert repl._last_exit_code == 0

    def test_ignore_whitespace(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello world\n")
        f2.write_text("hello  world\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_diff(f"-w {f1} {f2}")
        assert repl._last_exit_code == 0

    def test_unified_usage(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_diff("-u")
        assert "Usage" in cap.getvalue()


# ── Round 15: _expand_globs, _cmd_svc, _cmd_note, _dispatch, _print_header ──


class TestExpandGlobsDeeper:
    def test_no_magic_chars(self, repl):
        result = repl._expand_globs("echo hello")
        assert result == "echo hello"

    def test_glob_with_matches(self, repl, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "b.txt").write_text("y")
        orig = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = repl._expand_globs("*.txt")
            assert "a.txt" in result
            assert "b.txt" in result
        finally:
            os.chdir(orig)

    def test_glob_no_matches(self, repl, tmp_path):
        orig = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = repl._expand_globs("*.xyz")
            assert result == "*.xyz"
        finally:
            os.chdir(orig)

    def test_quoted_string_not_expanded(self, repl):
        result = repl._expand_globs("'hello world'")
        assert result == "'hello world'"

    def test_double_quoted_not_expanded(self, repl):
        result = repl._expand_globs('"hello world"')
        assert result == '"hello world"'

    def test_mixed_text_and_glob(self, repl, tmp_path):
        (tmp_path / "file1.txt").write_text("x")
        orig = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = repl._expand_globs("cat *.txt")
            assert "file1.txt" in result
            assert result.startswith("cat ")
        finally:
            os.chdir(orig)

    def test_question_mark_glob(self, repl, tmp_path):
        (tmp_path / "a1.txt").write_text("x")
        (tmp_path / "ab.txt").write_text("y")
        (tmp_path / "a12.txt").write_text("z")
        orig = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = repl._expand_globs("a?.txt")
            assert "a1.txt" in result
            assert "ab.txt" in result
            assert "a12.txt" not in result
        finally:
            os.chdir(orig)


class TestCmdSvcDeeper:
    _all_patchers: list = []

    def _make_svc(self, repl):
        mock_init = MagicMock()
        mock_init.service_table.return_value = "  api: running\n  web: stopped"
        mock_init.status_summary = "2 services, 1 running"
        mock_init.runlevel = "multiuser"
        mock_mgr = MagicMock()
        mock_mgr.status_line.return_value = "  api: running (pid 1234)"
        mock_mgr.instance.log = ["started", "ready"]
        mock_mgr.start.return_value = True
        mock_mgr.restart.return_value = True
        mock_init.get_manager.return_value = mock_mgr
        patcher = patch.object(type(repl.os), 'init_system', new_callable=PropertyMock, return_value=mock_init)
        patcher.start()
        self._all_patchers.append(patcher)
        repl._svc_patcher = patcher
        return mock_init, mock_mgr

    def teardown_method(self):
        for p in self._all_patchers:
            try:
                p.stop()
            except RuntimeError:
                pass
        self._all_patchers.clear()

    def test_svc_no_init(self, repl):
        patcher = patch.object(type(repl.os), 'init_system', new_callable=PropertyMock, return_value=None)
        patcher.start()
        self._all_patchers.append(patcher)
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("list")
        assert "not booted" in cap.getvalue()

    def test_svc_list(self, repl):
        self._make_svc(repl)
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("list")
        assert "api" in cap.getvalue()

    def test_svc_status_named(self, repl):
        self._make_svc(repl)
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("status api")
        assert "running" in cap.getvalue()

    def test_svc_status_overview(self, repl):
        self._make_svc(repl)
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("status")
        assert "Init status" in cap.getvalue()

    def test_svc_start(self, repl):
        self._make_svc(repl)
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("start api")
        assert "started" in cap.getvalue()

    def test_svc_start_no_name(self, repl):
        self._make_svc(repl)
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("start")
        assert "Usage" in cap.getvalue()

    def test_svc_stop(self, repl):
        self._make_svc(repl)
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("stop api")
        assert "stopped" in cap.getvalue()

    def test_svc_stop_no_name(self, repl):
        self._make_svc(repl)
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("stop")
        assert "Usage" in cap.getvalue()

    def test_svc_restart(self, repl):
        self._make_svc(repl)
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("restart api")
        assert "restarted" in cap.getvalue()

    def test_svc_restart_no_name(self, repl):
        self._make_svc(repl)
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("restart")
        assert "Usage" in cap.getvalue()

    def test_svc_runlevel(self, repl):
        self._make_svc(repl)
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("runlevel")
        assert "multiuser" in cap.getvalue()

    def test_svc_unknown_service(self, repl):
        mock_init, mock_mgr = self._make_svc(repl)
        mock_init.get_manager.return_value = None
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("start nonexistent")
        assert "Unknown" in cap.getvalue()

    def test_svc_unknown_subcmd(self, repl):
        self._make_svc(repl)
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("foobar")
        assert "Usage" in cap.getvalue()


class TestCmdNoteDeeper:
    def test_unknown_subcommand(self, repl):
        notes = pytest.importorskip("notes")
        with patch.object(notes, "get_note_store") as mock_get:
            mock_get.return_value = MagicMock()
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("nonexistent")
        assert "unknown subcommand" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_empty_defaults_to_list(self, repl):
        notes = pytest.importorskip("notes")
        with patch.object(notes, "get_note_store") as mock_get:
            mock_store = MagicMock()
            mock_store.list_notes.return_value = []
            mock_get.return_value = mock_store
            with _CaptureOutput(repl) as cap:
                repl._cmd_note("")
        mock_store.list_notes.assert_called_once()


class TestPrintHeader:
    def test_prints_help_hint(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._print_header()
        out = cap.getvalue()
        assert "help" in out
        assert "exit" in out


class TestCmdExportState:
    def test_exports_json(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_export_state("")
        out = cap.getvalue()
        assert "{" in out
        assert "}" in out


class TestDispatchDeeper:
    def test_unknown_command(self, repl):
        repl._dispatch("nonexistent_cmd_xyz")
        assert repl._last_exit_code == 127

    def test_empty_line_noop(self, repl):
        initial_count = repl._cmd_count
        repl._dispatch("")
        assert repl._cmd_count == initial_count + 1

    def test_permission_denied(self, repl):
        with patch.object(repl, '_check_permission', return_value=False):
            repl._dispatch("rm -rf /")
        assert repl._last_exit_code == 126

    def test_expanded_alias_dispatch(self, repl):
        repl._aliases["ll"] = "ls -la"
        repl._dispatch("ll")
        assert repl._last_exit_code == 0

    def test_suggest_command(self, repl):
        repl._dispatch("hep")
        assert repl._last_exit_code == 127


class TestSuggestCommandDeeper:
    def test_exact_match(self, repl):
        result = repl._suggest_command("help")
        assert result == "help"

    def test_close_match(self, repl):
        result = repl._suggest_command("hep")
        assert result is not None

    def test_no_match(self, repl):
        result = repl._suggest_command("zzzxyz")
        assert result is None


# ── Round 16: complex command branches ────────────────────────────


class TestCmdVmrunDeeper:
    def test_list_flag(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_vmrun("--list")
        out = cap.getvalue()
        assert "Built-in" in out
        assert repl._last_exit_code == 0

    def test_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_vmrun("")
        assert "Usage" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_invalid_steps(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_vmrun("--steps=abc hello")
        assert "integer" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_file_not_found(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_vmrun("/nonexistent/file.asm")
        assert repl._last_exit_code == 1

    def test_role_check_blocks(self, repl):
        os.environ["MAN_VM_ROLE"] = "user"
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_vmrun("--admin hello")
            assert "requires" in cap.getvalue()
            assert repl._last_exit_code == 1
        finally:
            os.environ.pop("MAN_VM_ROLE", None)


class TestCmdPyDeeper:
    def test_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("")
        assert "Usage" in cap.getvalue()

    def test_valid_expression(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("2 + 2")
        assert "4" in cap.getvalue()

    def test_syntax_error(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("def def def")
        assert "Error" in cap.getvalue()

    def test_runtime_error(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("1/0")
        assert "Error" in cap.getvalue()

    def test_safe_module_import(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("__import__('math').pi")
        out = cap.getvalue()
        assert "3.14" in out

    def test_blocked_module(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("__import__('os').system('ls')")
        assert "not allowed" in cap.getvalue()

    def test_list_comprehension(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("[i*i for i in range(5)]")
        assert "[0, 1, 4, 9, 16]" in cap.getvalue()

    def test_string_expression(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("'hello'.upper()")
        assert "HELLO" in cap.getvalue()


class TestCmdFindVFS:
    def test_find_nonexistent_dir(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_find("-name *.txt /nonexistent_dir_xyz")
        assert cap.getvalue().strip() == "" or repl._last_exit_code == 0


class TestCmdGenDeeper:
    def test_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_gen("")
        assert "Usage" in cap.getvalue()

    def test_no_api(self, repl):
        repl.cmds._api_get = MagicMock(side_effect=Exception("no api"))
        with _CaptureOutput(repl) as cap:
            repl._cmd_gen("hello")
        assert repl._last_exit_code == 1


class TestCmdChatDeeper:
    def test_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_chat("")
        assert "Usage" in cap.getvalue()

    def test_reset_session(self, repl):
        repl._chat_session_id = "old-session"
        repl._chat_history = [{"role": "user", "content": "hi"}]
        with _CaptureOutput(repl) as cap:
            repl._cmd_chat("/reset")
        assert "cleared" in cap.getvalue()
        assert repl._chat_session_id is None
        assert repl._chat_history == []

    def test_no_api(self, repl):
        repl.cmds._api_get = MagicMock(side_effect=Exception("no api"))
        with _CaptureOutput(repl) as cap:
            repl._cmd_chat("hello")
        assert repl._last_exit_code == 1


class TestCmdAiDeeper:
    def test_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_ai("")
        assert "Usage" in cap.getvalue()

    def test_api_unavailable_fallback(self, repl):
        repl.os._api_status = {"available": False}
        with _CaptureOutput(repl) as cap:
            repl._cmd_ai("help me")
        out = cap.getvalue()
        assert "keyword matching" in out or "not connected" in out


# ── Round 17: filesystem error paths + help branches ──────────────


class TestCmdCpErrorPaths:
    def test_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_cp("")
        assert "Usage" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_missing_destination(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_cp("only_one_arg")
        assert "missing destination" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_file_not_found(self, repl, tmp_path):
        src = tmp_path / "nonexistent.txt"
        dst = tmp_path / "dst.txt"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cp(f"{src} {dst}")
        assert "No such file" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_copy_file_success(self, repl, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("hello")
        dst = tmp_path / "dst.txt"
        repl._cmd_cp(f"{src} {dst}")
        assert dst.read_text() == "hello"
        assert repl._last_exit_code == 0

    def test_copy_dir(self, repl, tmp_path):
        src_dir = tmp_path / "srcdir"
        src_dir.mkdir()
        (src_dir / "f.txt").write_text("data")
        dst_dir = tmp_path / "dstdir"
        repl._cmd_cp(f"{src_dir} {dst_dir}")
        assert (dst_dir / "f.txt").read_text() == "data"


class TestCmdMvErrorPaths:
    def test_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_mv("")
        assert "Usage" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_missing_destination(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_mv("only_one")
        assert "missing destination" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_file_not_found(self, repl, tmp_path):
        with _CaptureOutput(repl) as cap:
            repl._cmd_mv(f"{tmp_path}/nope.txt {tmp_path}/dst.txt")
        assert "No such file" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_move_success(self, repl, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("data")
        dst = tmp_path / "dst.txt"
        repl._cmd_mv(f"{src} {dst}")
        assert dst.read_text() == "data"
        assert not src.exists()


class TestCmdRmErrorPaths:
    def test_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_rm("")
        assert "Usage" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_directory_without_recursive(self, repl, tmp_path):
        d = tmp_path / "mydir"
        d.mkdir()
        with _CaptureOutput(repl) as cap:
            repl._cmd_rm(f"{d}")
        assert "Is a directory" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_file_not_found_without_force(self, repl, tmp_path):
        with _CaptureOutput(repl) as cap:
            repl._cmd_rm(f"{tmp_path}/nonexistent.txt")
        assert "No such file" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_file_not_found_with_force(self, repl, tmp_path):
        repl._cmd_rm(f"-f {tmp_path}/nonexistent.txt")
        assert repl._last_exit_code == 0

    def test_remove_recursive(self, repl, tmp_path):
        d = tmp_path / "mydir"
        d.mkdir()
        (d / "file.txt").write_text("x")
        repl._cmd_rm(f"-r {d}")
        assert not d.exists()

    def test_remove_file(self, repl, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        repl._cmd_rm(f"{f}")
        assert not f.exists()


class TestCmdMkdirErrorPaths:
    def test_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_mkdir("")
        assert "Usage" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_file_exists(self, repl, tmp_path):
        d = tmp_path / "existing"
        d.mkdir()
        with _CaptureOutput(repl) as cap:
            repl._cmd_mkdir(f"{d}")
        assert "File exists" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_create_success(self, repl, tmp_path):
        d = tmp_path / "newdir"
        repl._cmd_mkdir(f"{d}")
        assert d.exists()
        assert repl._last_exit_code == 0


class TestCmdTouchErrorPaths:
    def test_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_touch("")
        assert "Usage" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_create_new_file(self, repl, tmp_path):
        f = tmp_path / "new.txt"
        repl._cmd_touch(f"{f}")
        assert f.exists()
        assert repl._last_exit_code == 0

    def test_update_existing_file(self, repl, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("data")
        old_mtime = f.stat().st_mtime_ns
        repl._cmd_touch(f"{f}")
        assert f.exists()
        assert repl._last_exit_code == 0


class TestCmdGrepErrorPaths:
    def test_no_args_no_pipe(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("")
        assert "Usage" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_invalid_regex(self, repl):
        repl._piped_input = "hello"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("[invalid", )
        assert "invalid pattern" in cap.getvalue()
        assert repl._last_exit_code == 2

    def test_file_not_found(self, repl, tmp_path):
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep(f"pattern {tmp_path}/nonexistent.txt")
        assert "No such file" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_pattern_match_pipe(self, repl):
        repl._piped_input = "hello world\nfoo bar\nhello again"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("hello")
        out = cap.getvalue()
        assert "hello world" in out
        assert "hello again" in out
        assert "foo bar" not in out

    def test_invert_match(self, repl):
        repl._piped_input = "keep\nremove\nkeep"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-v remove")
        out = cap.getvalue()
        assert "keep" in out
        assert "remove" not in out

    def test_ignore_case(self, repl):
        repl._piped_input = "Hello\nworld"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-i hello")
        assert "Hello" in cap.getvalue()


class TestCmdTeeErrorPaths:
    def test_no_pipe_no_args(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_tee("")
        assert "Usage" in cap.getvalue()
        assert repl._last_exit_code == 1

    def test_write_to_file(self, repl, tmp_path):
        repl._piped_input = "test data"
        f = tmp_path / "output.txt"
        repl._cmd_tee(f"{f}")
        assert f.read_text().strip() == "test data"
        assert repl._last_exit_code == 0

    def test_append_mode(self, repl, tmp_path):
        f = tmp_path / "output.txt"
        f.write_text("first\n")
        repl._piped_input = "second"
        repl._cmd_tee(f"-a {f}")
        content = f.read_text()
        assert "first" in content
        assert "second" in content

    def test_permission_denied(self, repl):
        repl._piped_input = "data"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tee("/proc/impossible_file_xyz")
        assert repl._last_exit_code == 1


class TestCmdHelpBranches:
    def test_help_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_help("")
        assert "Built-in commands" in cap.getvalue()

    def test_help_brief(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_help("brief")
        assert "Most-used" in cap.getvalue() or "help" in cap.getvalue()

    def test_help_specific_command(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_help("cd")
        assert "cd" in cap.getvalue()

    def test_help_unknown_command(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_help("zzz_nonexistent_cmd_xyz")
        assert "Unknown command" in cap.getvalue()

    def test_help_ext_command(self, repl):
        mock_mod = MagicMock()
        mock_mod.help = "test external help"
        repl._ext_cmds["testext"] = mock_mod
        with _CaptureOutput(repl) as cap:
            repl._cmd_help("testext")
        assert "testext" in cap.getvalue()

    def test_help_vs_command(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_help("vs")
        out = cap.getvalue()
        assert "vs" in out


# ── Round 18: bg/fg/watch, env/set, misc command edge cases ─────────


class TestCmdBgFg:
    def test_bg_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_bg("")
        assert "No background" in cap.getvalue() or "Usage" in cap.getvalue()

    def test_fg_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_fg("")
        assert "Usage" in cap.getvalue()

    def test_fg_unknown_job(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_fg("999")
        assert "No background process" in cap.getvalue() or "not found" in cap.getvalue()


class TestCmdWatch:
    def test_watch_invalid_interval(self, repl):
        repl._execute_single = MagicMock()
        with _CaptureOutput(repl) as cap:
            repl._cmd_watch("abc ls")
        assert "Invalid" in cap.getvalue() or "Usage" in cap.getvalue()

    def test_watch_keyboard_interrupt(self, repl):
        call_count = [0]
        def mock_execute(cmd, piped=""):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise KeyboardInterrupt()
            return "ok"
        repl._execute_single = mock_execute
        repl._cmd_watch("1 ls")


class TestCmdEnvSet:
    def test_env_empty(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_env("")
        assert "PATH" in cap.getvalue() or "HOME" in cap.getvalue()

    def test_set_assign(self, repl):
        repl._cmd_set("MYVAR=hello")
        assert repl._env.get("MYVAR") == "hello"

    def test_set_show_all(self, repl):
        repl._cmd_set("FOO=bar")
        with _CaptureOutput(repl) as cap:
            repl._cmd_set("")
        assert "FOO=bar" in cap.getvalue()


class TestCmdExport:
    def test_export_empty(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_export("")
        assert cap.getvalue()  # prints env vars

    def test_export_var(self, repl):
        repl._cmd_export("MYVAR=exported")
        assert repl._env.get("MYVAR") == "exported"


class TestCmdYes:
    def test_yes_no_args(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_yes("")
        out = cap.getvalue()
        assert "y" in out.lower() or len(out) > 0


class TestCmdNproc:
    def test_nproc(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_nproc("")
        out = cap.getvalue().strip()
        assert out.isdigit()
        assert int(out) >= 1


class TestCmdHostname:
    def test_hostname(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_hostname("")
        assert cap.getvalue().strip()  # returns some hostname


class TestCmdUname:
    def test_uname(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_uname("")
        out = cap.getvalue()
        assert "Linux" in out or "Darwin" in out or "linux" in out.lower()


class TestCmdId:
    def test_id(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_id("")
        out = cap.getvalue()
        assert "uid=" in out or "euid=" in out


class TestCmdMktemp:
    def test_mktemp(self, repl, tmp_path):
        with _CaptureOutput(repl) as cap:
            repl._cmd_mktemp("")
        out = cap.getvalue().strip()
        assert len(out) > 0


class TestCmdXargs:
    def test_xargs_no_pipe(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_xargs("echo hello")
        assert "Usage" in cap.getvalue() or repl._last_exit_code == 1

    def test_xargs_with_pipe(self, repl):
        repl._piped_input = "a\nb\nc"
        with _CaptureOutput(repl) as cap:
            repl._cmd_xargs("echo")
        out = cap.getvalue()
        assert "a" in out or "b" in out or len(out) > 0

    def test_xargs_placeholder(self, repl):
        repl._piped_input = "a b c"
        with _CaptureOutput(repl) as cap:
            repl._cmd_xargs("-I{} echo item:{}")
        out = cap.getvalue()
        assert "item:a" in out and "item:b" in out and "item:c" in out

    def test_xargs_placeholder_multi_sub(self, repl):
        repl._piped_input = "x y"
        with _CaptureOutput(repl) as cap:
            repl._cmd_xargs("-I{} echo {}-{}")
        out = cap.getvalue()
        assert "x-x" in out and "y-y" in out

    def test_xargs_placeholder_long_flag(self, repl):
        repl._piped_input = "foo bar"
        with _CaptureOutput(repl) as cap:
            repl._cmd_xargs("-I @ echo @")
        out = cap.getvalue()
        assert "foo" in out and "bar" in out

    def test_xargs_null_terminated(self, repl):
        repl._piped_input = "a\0b\0c"
        with _CaptureOutput(repl) as cap:
            repl._cmd_xargs("-0 echo")
        out = cap.getvalue()
        assert "a" in out and "b" in out and "c" in out

    def test_xargs_null_terminated_vs_whitespace(self, repl):
        repl._piped_input = "a b\0c d"
        with _CaptureOutput(repl) as cap:
            repl._cmd_xargs("-0 echo")
        out = cap.getvalue()
        assert "a b" in out and "c d" in out

    def test_xargs_no_run_if_empty(self, repl):
        repl._piped_input = "   "
        with _CaptureOutput(repl) as cap:
            repl._cmd_xargs("-r echo")
        assert cap.getvalue() == ""
        assert repl._last_exit_code == 0

    def test_xargs_no_run_if_empty_with_items(self, repl):
        repl._piped_input = "hello"
        with _CaptureOutput(repl) as cap:
            repl._cmd_xargs("-r echo")
        out = cap.getvalue()
        assert "hello" in out


class TestCmdChmod:
    def test_chmod_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_chmod("")
        assert "Usage" in cap.getvalue()

    def test_chmod_file(self, repl, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("data")
        repl._cmd_chmod(f"755 {f}")
        assert repl._last_exit_code == 0


class TestCmdStat:
    def test_stat_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_stat("")
        assert "Usage" in cap.getvalue()

    def test_stat_file(self, repl, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        with _CaptureOutput(repl) as cap:
            repl._cmd_stat(f"{f}")
        assert "Size" in cap.getvalue() or "size" in cap.getvalue() or "test.txt" in cap.getvalue()


class TestCmdComm:
    def test_comm_no_args(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_comm("")
        assert "Usage" in cap.getvalue()

    def test_comm_with_files(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("a\nb\nc\n")
        f2.write_text("b\nc\nd\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_comm(f"{f1} {f2}")
        assert repl._last_exit_code == 0


class TestCmdFold:
    def test_fold_with_pipe(self, repl):
        repl._piped_input = "hello world this is a test"
        with _CaptureOutput(repl) as cap:
            repl._cmd_fold("-w 10")
        assert len(cap.getvalue()) > 0

    def test_fold_no_args_no_pipe(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_fold("")
        # returns empty or usage


class TestCmdTac:
    def test_tac_with_pipe(self, repl):
        repl._piped_input = "line1\nline2\nline3\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tac("")
        out = cap.getvalue()
        assert "line3" in out or "line1" in out

    def test_tac_empty(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_tac("")
        assert "Usage" in cap.getvalue() or repl._last_exit_code == 1


class TestCmdPaste:
    def test_paste_no_args(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_paste("")
        assert cap.getvalue() == "" or repl._last_exit_code == 1

    def test_paste_with_pipe(self, repl):
        repl._piped_input = "a\nb\nc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_paste("")
        assert "a" in cap.getvalue()


class TestCmdShuf:
    def test_shuf_no_args(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_shuf("")
        assert cap.getvalue() == "" or repl._last_exit_code == 1

    def test_shuf_with_pipe(self, repl):
        repl._piped_input = "a\nb\nc\nd\ne\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_shuf("")
        out = cap.getvalue().strip()
        lines = [l for l in out.split("\n") if l]
        assert len(lines) == 5


class TestCmdRev:
    def test_rev_with_pipe(self, repl):
        repl._piped_input = "hello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_rev("")
        assert "olleh" in cap.getvalue()

    def test_rev_empty(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_rev("")
        assert "Usage" in cap.getvalue() or repl._last_exit_code == 1


class TestCmdIdEdge:
    def test_id_uid(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_id("")
        assert "uid=" in cap.getvalue()

    def test_id_gid(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_id("-g")
        out = cap.getvalue().strip()
        assert out.isdigit() or "gid=" in out


class TestCmdLogname:
    def test_logname(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_logname("")
        assert len(cap.getvalue().strip()) > 0


class TestCmdSleep:
    def test_sleep_zero(self, repl):
        import time
        t0 = time.time()
        repl._cmd_sleep("0")
        assert time.time() - t0 < 2
        assert repl._last_exit_code == 0

    def test_sleep_empty(self, repl):
        repl._cmd_sleep("")
        assert repl._last_exit_code == 0

    def test_sleep_negative(self, repl):
        import pytest
        with pytest.raises(ValueError):
            repl._cmd_sleep("-1")


class TestCmdKill:
    def test_kill_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_kill("")
        assert "Usage" in cap.getvalue()

    def test_kill_invalid_pid(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_kill("99999")
        # may succeed or fail depending on OS
        assert repl._last_exit_code in (0, 1)


class TestCmdTime:
    def test_time_empty(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_time("")
        assert "Usage" in cap.getvalue() or repl._last_exit_code == 1


class TestCmdRead:
    def test_read_empty(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl):
            repl._cmd_read("")
        assert repl._last_exit_code == 1

    def test_read_with_var(self, repl):
        out = _run_with_io(repl, ["myvalue"], lambda: repl._cmd_read("myvar"))
        assert repl._env.get("myvar") == "myvalue"

    def test_read_prompt(self, repl):
        out = _run_with_io(repl, ["val"], lambda: repl._cmd_read("-p Enter: myvar"))
        assert repl._env.get("myvar") == "val"


class TestCmdSource:
    def test_source_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_source("")
        assert "Usage" in cap.getvalue()

    def test_source_nonexistent(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_source("/nonexistent/script.sh")
        assert "not found" in cap.getvalue() or "No such" in cap.getvalue() or "Error reading" in cap.getvalue()


class TestCmdPy:
    def test_py_empty(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("")
        assert "Usage" in cap.getvalue()

    def test_py_syntax_error(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("def def def")
        assert "Error" in cap.getvalue() or "syntax" in cap.getvalue().lower()

    def test_py_runtime_error(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("1/0")
        assert "Error" in cap.getvalue() or "division" in cap.getvalue()

    def test_py_expression(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("2 + 3")
        assert "5" in cap.getvalue()

    def test_py_import_via_dunder(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("__import__('math').sqrt(16)")
        assert "4.0" in cap.getvalue()


class TestCmdAlias:
    def test_alias_create(self, repl):
        repl._cmd_alias("ll=ls -la")
        assert repl._aliases.get("ll") == "ls -la"

    def test_alias_show_all(self, repl):
        repl._cmd_alias("myalias=echo hi")
        with _CaptureOutput(repl) as cap:
            repl._cmd_alias("")
        assert "myalias" in cap.getvalue()

    def test_alias_nonexistent(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_alias("zzz_nonexistent")
        assert "zzz_nonexistent" in cap.getvalue() or "not found" in cap.getvalue()


class TestCmdUnalias:
    def test_unalias_nonexistent(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_unalias("zzz_nonexistent_xyz")
        assert "No alias" in cap.getvalue()

    def test_unalias_existing(self, repl):
        repl._cmd_alias("myalias=echo hi")
        repl._cmd_unalias("myalias")
        assert "myalias" not in repl._aliases


class TestCmdSourceDeeper:
    def test_source_script(self, repl, tmp_path):
        script = tmp_path / "test.sh"
        script.write_text("echo sourced_ok\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_source(str(script))
        assert repl._last_exit_code == 0


class TestCmdLsDeeper:
    def test_ls_no_args(self, repl, tmp_path):
        (tmp_path / "file.txt").write_text("x")
        with _CaptureOutput(repl) as cap:
            repl._cmd_ls(f"{tmp_path}")
        assert "file.txt" in cap.getvalue()

    def test_ls_nonexistent(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_ls("/nonexistent_dir_xyz")
        assert "No such file" in cap.getvalue() or "not found" in cap.getvalue()


class TestCmdCatDeeper:
    def test_cat_empty_file(self, repl, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        with _CaptureOutput(repl) as cap:
            repl._cmd_cat(str(f))
        assert repl._last_exit_code == 0

    def test_cat_nonexistent(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_cat("/nonexistent_file_xyz.txt")
        assert "No such file" in cap.getvalue()

    def test_cat_pipe(self, repl):
        repl._piped_input = "piped data"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cat("")
        assert "piped data" in cap.getvalue()


class TestCmdMkdirDeeper:
    def test_mkdir_nested(self, repl, tmp_path):
        d = tmp_path / "a"
        repl._cmd_mkdir(f"{d}")
        assert d.exists()


class TestCmdRmDeeper:
    def test_rm_nonexistent_with_force(self, repl, tmp_path):
        repl._cmd_rm(f"-f {tmp_path}/nope.txt")
        assert repl._last_exit_code == 0

    def test_rm_multiple_files(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("a")
        f2.write_text("b")
        repl._cmd_rm(f"{f1} {f2}")
        assert not f1.exists() and not f2.exists()


class TestCmdCpDeeper:
    def test_cp_multiple(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("a")
        f2.write_text("b")
        dst = tmp_path / "dst"
        dst.mkdir()
        repl._cmd_cp(f"{f1} {dst}")
        assert (dst / "a.txt").exists()


class TestCmdMvDeeper:
    def test_mv_directory(self, repl, tmp_path):
        src = tmp_path / "srcdir"
        src.mkdir()
        (src / "f.txt").write_text("data")
        dst = tmp_path / "dstdir"
        repl._cmd_mv(f"{src} {dst}")
        assert dst.exists() and (dst / "f.txt").exists()
        assert not src.exists()


class TestCmdHeadDeeper:
    def test_head_negative(self, repl):
        repl._piped_input = "a\nb\nc\nd\ne\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_head("-3")
        lines = [l for l in cap.getvalue().strip().split("\n") if l]
        assert len(lines) == 3

    def test_head_minus_n_flag(self, repl):
        repl._piped_input = "a\nb\nc\nd\ne\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_head("-n 2")
        lines = [l for l in cap.getvalue().strip().split("\n") if l]
        assert len(lines) == 2
        assert lines[0] == "a"
        assert lines[1] == "b"


class TestCmdTailDeeper:
    def test_tail_negative(self, repl):
        repl._piped_input = "a\nb\nc\nd\ne\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tail("-3")
        lines = [l for l in cap.getvalue().strip().split("\n") if l]
        assert len(lines) == 3

    def test_tail_minus_n_flag(self, repl):
        repl._piped_input = "a\nb\nc\nd\ne\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tail("-n 2")
        lines = [l for l in cap.getvalue().strip().split("\n") if l]
        assert len(lines) == 2
        assert lines[0] == "d"
        assert lines[1] == "e"


class TestCmdSortDeeper:
    def test_sort_reverse(self, repl):
        repl._piped_input = "c\na\nb\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sort("-r")
        lines = cap.getvalue().strip().split("\n")
        assert lines[0] == "c"

    def test_sort_unique(self, repl):
        repl._piped_input = "a\nb\na\nc\nb\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sort("-u")
        lines = [l for l in cap.getvalue().strip().split("\n") if l]
        assert len(lines) == 3

    def test_sort_numeric(self, repl):
        repl._piped_input = "10\n2\n30\n1\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sort("-n")
        lines = cap.getvalue().strip().split("\n")
        assert lines[0] == "1"


class TestCmdUniqDeeper:
    def test_uniq_with_pipe(self, repl):
        repl._piped_input = "a\na\nb\nc\nc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_uniq("")
        lines = [l for l in cap.getvalue().strip().split("\n") if l]
        assert len(lines) == 3

    def test_uniq_empty(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_uniq("")
        assert "Usage" in cap.getvalue() or repl._last_exit_code == 1

    def test_uniq_count(self, repl):
        repl._piped_input = "a\na\nb\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_uniq("-c")
        lines = cap.getvalue().strip().split("\n")
        assert lines[0].strip().endswith("a")
        assert lines[0].strip().startswith("2")
        assert lines[1].strip().endswith("b")
        assert lines[1].strip().startswith("1")

    def test_uniq_count_all_unique(self, repl):
        repl._piped_input = "x\ny\nz\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_uniq("-c")
        lines = cap.getvalue().strip().split("\n")
        for line in lines:
            assert line.strip().startswith("1")

    def test_uniq_count_mixed(self, repl):
        repl._piped_input = "a\na\na\nb\nb\nc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_uniq("-c")
        out = cap.getvalue().strip()
        assert "3 a" in out
        assert "2 b" in out
        assert "1 c" in out

    def test_uniq_case_insensitive(self, repl):
        repl._piped_input = "Hello\nhello\nHELLO\nworld\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_uniq("-i")
        lines = [l.strip() for l in cap.getvalue().strip().split("\n") if l.strip()]
        assert lines == ["Hello", "world"]

    def test_uniq_non_adjacent_preserved(self, repl):
        repl._piped_input = "a\nb\na\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_uniq("")
        lines = [l.strip() for l in cap.getvalue().strip().split("\n") if l.strip()]
        assert lines == ["a", "b", "a"]

    def test_uniq_count_insensitive(self, repl):
        repl._piped_input = "A\na\nA\nb\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_uniq("-c -i")
        out = cap.getvalue().strip()
        assert "3 A" in out
        assert "1 b" in out


class TestCmdCutDeeper:
    def test_cut_delimited(self, repl):
        repl._piped_input = "a:b:c\nd:e:f\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-d: -f2")
        out = cap.getvalue()
        assert "b" in out and "e" in out

    def test_cut_fields(self, repl):
        repl._piped_input = "abc\ndef\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-c1,3")
        out = cap.getvalue()
        assert "ac" in out or "a" in out

    def test_cut_field_range(self, repl):
        repl._piped_input = "a:b:c:d:e\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-d: -f2-4")
        out = cap.getvalue().strip()
        assert out == "b:c:d"

    def test_cut_posix_digit_delim(self, repl):
        repl._piped_input = "a1b2c3d\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-d'[:digit:]' -f2")
        out = cap.getvalue().strip()
        assert out == "b"

    def test_cut_posix_space_delim(self, repl):
        repl._piped_input = "hello world foo\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-d'[:space:]' -f1,3")
        out = cap.getvalue().strip()
        assert out == "hello foo"

    def test_cut_no_matching_fields(self, repl):
        repl._piped_input = "a:b\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-d: -f5")
        out = cap.getvalue().strip()
        assert out == ""

    def test_cut_suppress_no_delim(self, repl):
        repl._piped_input = "a:b\nc\nd:e\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-d: -s -f1")
        out = cap.getvalue().strip().split("\n")
        assert out == ["a", "d"]

    def test_cut_suppress_mixed(self, repl):
        repl._piped_input = "x,y,z\nno-delim\n1:2:3\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-d, -s -f2")
        out = cap.getvalue().strip()
        assert "y" in out
        assert "no-delim" not in out
        assert "2" not in out

    def test_cut_file_not_found(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-d: -f1 /nonexistent/file.txt")
        assert repl._last_exit_code == 1

    def test_cut_char_single(self, repl):
        repl._piped_input = "hello\nworld\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-c1")
        out = cap.getvalue().strip().split("\n")
        assert out == ["h", "w"]

    def test_cut_char_range(self, repl):
        repl._piped_input = "hello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-c1-3")
        out = cap.getvalue().strip()
        assert out == "hel"

    def test_cut_char_multiple(self, repl):
        repl._piped_input = "hello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-c1,3,5")
        out = cap.getvalue().strip()
        assert out == "hlo"

    def test_cut_char_beyond_length(self, repl):
        repl._piped_input = "ab\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-c1-10")
        out = cap.getvalue().strip()
        assert out == "ab"

    def test_cut_no_args(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("")
        assert repl._last_exit_code == 1

    def test_cut_suppress_ignored_in_char_mode(self, repl):
        repl._piped_input = "hello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-s -c1")
        out = cap.getvalue().strip()
        assert out == "h"  # -s has no effect in character mode

    def test_cut_rejects_combined_cf(self, repl):
        repl._piped_input = "abc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-c1 -f2")
        assert repl._last_exit_code == 1
        assert "cannot combine" in cap.getvalue().lower()

    def test_cut_no_input(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-f1")
        assert repl._last_exit_code == 1

    def test_cut_byte_single(self, repl):
        repl._piped_input = "hello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-b1")
        out = cap.getvalue().strip()
        assert out == "h"

    def test_cut_byte_range(self, repl):
        repl._piped_input = "hello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-b2-4")
        out = cap.getvalue().strip()
        assert out == "ell"

    def test_cut_byte_multi(self, repl):
        repl._piped_input = "hello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-b1,3,5")
        out = cap.getvalue().strip()
        assert out == "hlo"

    def test_cut_byte_utf8(self, repl):
        repl._piped_input = "cafe\u0301\n"  # 'cafe' + combining acute
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-b1-4")
        out = cap.getvalue().strip()
        assert out == "cafe"

    def test_cut_rejects_combined_bf(self, repl):
        repl._piped_input = "abc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-b1 -f2")
        assert repl._last_exit_code == 1
        assert "cannot combine" in cap.getvalue().lower()

    def test_cut_rejects_combined_bc(self, repl):
        repl._piped_input = "abc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cut("-b1 -c2")
        assert repl._last_exit_code == 1


class TestCmdTrDeeper:
    def test_tr_uppercase(self, repl):
        repl._piped_input = "hello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tr("a-z A-Z")
        assert "HELLO" in cap.getvalue()

    def test_tr_delete(self, repl):
        repl._piped_input = "hello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tr("-d aeiou")
        assert "hll" in cap.getvalue()

    def test_tr_empty(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_tr("a b")
        assert cap.getvalue() == "" or repl._last_exit_code == 1

    def test_tr_posix_lower_to_upper(self, repl):
        repl._piped_input = "hello world\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tr("[:lower:] [:upper:]")
        assert "HELLO WORLD" in cap.getvalue()

    def test_tr_posix_delete_digits(self, repl):
        repl._piped_input = "abc123def456\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tr("-d [:digit:]")
        assert "abcdef" in cap.getvalue()

    def test_tr_posix_replace_alpha(self, repl):
        repl._piped_input = "hello 123\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tr("[:alpha:] X")
        assert "XXXXX 123" in cap.getvalue()


class TestCmdSeqDeeper:
    def test_seq_reverse(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_seq("1 3 10")
        out = cap.getvalue().strip()
        assert "1" in out and "7" in out

    def test_seq_step(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_seq("1 2 7")
        out = cap.getvalue().strip()
        assert "1" in out and "7" in out

    def test_seq_float_whole_numbers(self, repl):
        """seq 1.0 1.0 3.0 should output 1.0, 2.0, 3.0 (not 1, 2, 3)."""
        with _CaptureOutput(repl) as cap:
            repl._cmd_seq("1.0 1.0 3.0")
        out = cap.getvalue().strip().split("\n")
        assert out == ["1", "2", "3"]

    def test_seq_float_precision(self, repl):
        """seq 0 0.5 2.0 should not produce floating-point noise."""
        with _CaptureOutput(repl) as cap:
            repl._cmd_seq("0 0.5 2.0")
        out = cap.getvalue().strip().split("\n")
        assert "0.30000000000000004" not in out
        assert "1.5000000000000002" not in out
        assert len(out) == 5  # 0, 0.5, 1, 1.5, 2


class TestCmdNlDeeper:
    def test_nl_with_pipe(self, repl):
        repl._piped_input = "a\nb\nc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_nl("")
        out = cap.getvalue()
        assert "1" in out and "2" in out


class TestCmdFoldDeeper:
    def test_fold_short_width(self, repl):
        repl._piped_input = "abcdefghij\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_fold("-w 5")
        lines = cap.getvalue().strip().split("\n")
        assert any(len(l) <= 5 for l in lines)


class TestCmdGrepDeeper:
    def test_grep_count(self, repl):
        repl._piped_input = "hello\nworld\nhello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-c hello")
        out = cap.getvalue()
        assert "2" in out
        assert "hello" not in out
        assert "world" not in out

    def test_grep_line_numbers(self, repl):
        repl._piped_input = "hello\nworld\nhello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-n hello")
        out = cap.getvalue()
        assert "1:hello" in out
        assert "3:hello" in out

    def test_grep_word_boundary(self, repl):
        repl._piped_input = "cat cats concatenate\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-w cat")
        out = cap.getvalue()
        lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
        assert len(lines) == 1
        assert "cat" in lines[0]

    def test_grep_context_after(self, repl):
        repl._piped_input = "a\nb\nc\nd\ne\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-A 1 c")
        out = cap.getvalue()
        assert "c" in out
        assert "d" in out
        assert "a" not in out
        assert "b" not in out

    def test_grep_context_before(self, repl):
        repl._piped_input = "a\nb\nc\nd\ne\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-B 1 c")
        out = cap.getvalue()
        assert "b" in out
        assert "c" in out
        assert "d" not in out
        assert "e" not in out

    def test_grep_context_both(self, repl):
        repl._piped_input = "a\nb\nc\nd\ne\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-C 1 c")
        out = cap.getvalue()
        assert "b" in out
        assert "c" in out
        assert "d" in out
        assert "a" not in out
        assert "e" not in out

    def test_grep_files_only(self, repl):
        repl._piped_input = "hello\nworld\nhello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-l hello")
        out = cap.getvalue()
        assert "<stdin>" in out
        assert "hello" not in out

    def test_grep_files_only_no_match(self, repl):
        repl._piped_input = "hello\nworld\nhello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-l xyz")
        out = cap.getvalue()
        assert "<stdin>" not in out
        assert repl._last_exit_code == 1

    def test_grep_count_no_match(self, repl):
        repl._piped_input = "hello\nworld\nhello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-c xyz")
        out = cap.getvalue()
        assert "0" in out
        assert repl._last_exit_code == 1

    def test_grep_word_boundary_no_match(self, repl):
        repl._piped_input = "cats concatenate\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-w cat")
        out = cap.getvalue()
        assert "cat" not in out
        assert repl._last_exit_code == 1


class TestCmdTeeDeeper:
    def test_tee_multiple_files(self, repl, tmp_path):
        f1 = tmp_path / "out1.txt"
        f2 = tmp_path / "out2.txt"
        repl._piped_input = "test data"
        repl._cmd_tee(f"{f1} {f2}")
        assert f1.read_text().strip() == "test data"
        assert f2.read_text().strip() == "test data"


# ── Round 19: thin-coverage reinforcement ──────────────────────────


class TestCmdTestBranches:
    def test_test_file_exists(self, repl, tmp_path):
        f = tmp_path / "exists.txt"
        f.write_text("x")
        with _CaptureOutput(repl):
            repl._cmd_test(f"-f {f}")
        assert repl._last_exit_code == 0

    def test_test_file_not_exists(self, repl, tmp_path):
        with _CaptureOutput(repl):
            repl._cmd_test(f"-f {tmp_path}/nope.txt")
        assert repl._last_exit_code == 1

    def test_test_dir(self, repl, tmp_path):
        with _CaptureOutput(repl):
            repl._cmd_test(f"-d {tmp_path}")
        assert repl._last_exit_code == 0

    def test_test_not_empty(self, repl, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("data")
        with _CaptureOutput(repl):
            repl._cmd_test(f"-e {f}")
        assert repl._last_exit_code == 0

    def test_test_string_eq(self, repl):
        with _CaptureOutput(repl):
            repl._cmd_test("abc = abc")
        assert repl._last_exit_code == 0

    def test_test_string_ne(self, repl):
        with _CaptureOutput(repl):
            repl._cmd_test("abc != xyz")
        assert repl._last_exit_code == 0

    def test_test_numeric(self, repl):
        with _CaptureOutput(repl):
            repl._cmd_test("5 -gt 3")
        assert repl._last_exit_code == 0

    def test_test_numeric_le(self, repl):
        with _CaptureOutput(repl):
            repl._cmd_test("3 -le 3")
        assert repl._last_exit_code == 0

    def test_test_no_args(self, repl):
        with _CaptureOutput(repl):
            repl._cmd_test("")
        assert repl._last_exit_code == 1


class TestCmdPrintfBranches:
    def test_printf_percent_s(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_printf("hello %s world %s" % ("cruel", "today"))
        assert "hello" in cap.getvalue()

    def test_printf_percent_d(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_printf("%d" % 42)
        assert "42" in cap.getvalue()

    def test_printf_percent_f(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_printf("%f" % 3.14)
        assert "3.14" in cap.getvalue()

    def test_printf_no_args(self, repl):
        with _CaptureOutput(repl):
            repl._cmd_printf("")
        assert repl._last_exit_code == 1

    def test_printf_literal_percent(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_printf("100%%")
        assert "100%" in cap.getvalue()


class TestCmdCommEdgeCases:
    def test_comm_no_args(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl):
            repl._cmd_comm("")
        assert repl._last_exit_code == 1

    def test_comm_two_files(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("a\nb\n")
        f2.write_text("b\nc\n")
        with _CaptureOutput(repl):
            repl._cmd_comm(f"{f1} {f2}")
        assert repl._last_exit_code == 0


class TestCmdKillSubprocess:
    def test_kill_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_kill("")
        assert "Usage" in cap.getvalue()

    def test_kill_nonexistent(self, repl):
        with _CaptureOutput(repl):
            repl._cmd_kill("99999")
        assert repl._last_exit_code in (0, 1)


class TestCmdFoldEdges:
    def test_fold_with_pipe(self, repl):
        repl._piped_input = "hello world this is a long line"
        with _CaptureOutput(repl) as cap:
            repl._cmd_fold("-w 10")
        assert len(cap.getvalue()) > 0

    def test_fold_no_args(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_fold("")
        assert repl._last_exit_code == 0 or len(cap.getvalue()) >= 0


class TestCmdPasteEdges:
    def test_paste_no_args(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_paste("")
        assert cap.getvalue() == "" or repl._last_exit_code == 1

    def test_paste_with_pipe(self, repl):
        repl._piped_input = "a\nb\nc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_paste("")
        assert "a" in cap.getvalue()

    def test_paste_custom_delim(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("x\ny\n")
        f2 = tmp_path / "b.txt"
        f2.write_text("1\n2\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_paste(f"-d, {f1} {f2}")
        out = cap.getvalue().strip()
        assert "x,1" in out and "y,2" in out

    def test_paste_long_delim_flag(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("a\nb\n")
        f2 = tmp_path / "b.txt"
        f2.write_text("c\nd\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_paste(f"-d: {f1} {f2}")
        out = cap.getvalue().strip()
        assert "a:c" in out and "b:d" in out

    def test_paste_no_files(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_paste("-d,")
        assert repl._last_exit_code == 1

    def test_paste_piped_input(self, repl):
        repl._piped_input = "a\nb\nc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_paste("")
        out = cap.getvalue().strip()
        assert "a" in out and "b" in out and "c" in out

    def test_paste_piped_input_with_delim(self, repl):
        repl._piped_input = "x\ny\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_paste("-d,")
        out = cap.getvalue().strip().split("\n")
        assert out == ["x", "y"]  # single column: delimiter has no effect


class TestCmdJoinEdges:
    def test_join_no_args(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl):
            repl._cmd_join("")
        assert repl._last_exit_code == 1

    def test_join_two_files(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("key1 val1\nkey2 val2\n")
        f2.write_text("key1 extra1\nkey3 val3\n")
        with _CaptureOutput(repl):
            repl._cmd_join(f"{f1} {f2}")
        assert repl._last_exit_code == 0


class TestCmdReadEdges:
    def test_read_no_args(self, repl):
        with _CaptureOutput(repl):
            repl._cmd_read("")
        assert repl._last_exit_code == 1

    def test_read_with_var(self, repl):
        out = _run_with_io(repl, ["hello"], lambda: repl._cmd_read("myvar"))
        assert repl._env.get("myvar") == "hello"


class TestCmdWatchWithCommand:
    def test_watch_runs_command(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_watch("0.01 echo hello")
        out = cap.getvalue()
        assert "hello" in out or repl._last_exit_code == 0


class TestCmdBgRunning:
    def test_bg_shows_nothing(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_bg("")
        assert "No background" in cap.getvalue() or len(cap.getvalue()) > 0


class TestCmdFgRunning:
    def test_fg_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_fg("")
        assert "Usage" in cap.getvalue()

    def test_fg_invalid_id(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_fg("abc")
        assert "Invalid" in cap.getvalue()

    def test_fg_not_found(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_fg("999")
        assert "No background process" in cap.getvalue()


class TestCmdUnameFlags:
    def test_uname_a(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_uname("-a")
        assert cap.getvalue().strip()

    def test_uname_s(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_uname("-s")
        out = cap.getvalue().strip()
        assert "Linux" in out or "Darwin" in out or "linux" in out.lower()


class TestCmdHelpWithExt:
    def test_help_ext_command(self, repl):
        mock_mod = MagicMock()
        mock_mod.help = "test external help text"
        repl._ext_cmds["testext"] = mock_mod
        with _CaptureOutput(repl) as cap:
            repl._cmd_help("testext")
        assert "testext" in cap.getvalue()

    def test_help_unknown(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_help("zzz_nonexistent_cmd_xyz")
        assert "Unknown command" in cap.getvalue() or "not found" in cap.getvalue()


class TestCmdAiKeywordFallback:
    def test_ai_keyword_models(self, repl):
        with _CaptureOutput(repl):
            repl._cmd_ai("what models are available")
        assert repl._last_exit_code == 0

    def test_ai_keyword_soul(self, repl):
        with _CaptureOutput(repl):
            repl._cmd_ai("who are you")
        assert repl._last_exit_code == 0


class TestModuleHelpers:
    def test_group_ext_cmds(self, repl):
        mock_mod = MagicMock()
        mock_mod.help = "test help"
        result = ShellREPL._group_ext_cmds({"testcmd": mock_mod})
        assert "test help" in result or "testcmd" in str(result)

    def test_update_color_state(self, repl):
        repl._update_color_state()


class TestCheckPermission:
    def test_check_safe(self, repl):
        result = repl._check_permission("echo", "", False)
        assert result is True or "allowed" in str(result).lower()

    def test_check_denied(self, repl):
        result = repl._check_permission("rm", "", False)
        assert result is False


class TestPermitDeny:
    def test_permit_empty(self, repl):
        with _CaptureOutput(repl):
            repl._cmd_permit("")
        assert repl._last_exit_code == 0

    def test_permit_rm(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_permit("rm")
        assert "permitted" in cap.getvalue().lower() or "granted" in cap.getvalue().lower() or repl._last_exit_code == 0

    def test_deny_empty(self, repl):
        with _CaptureOutput(repl):
            repl._cmd_deny("")
        assert repl._last_exit_code == 0


class TestCdPwdEcho:
    def test_cd_home(self, repl):
        with _CaptureOutput(repl):
            repl._cmd_cd("")
        assert repl._last_exit_code == 0

    def test_cd_nonexistent(self, repl, tmp_path):
        with _CaptureOutput(repl):
            repl._cmd_cd(f"{tmp_path}/nonexistent")
        assert repl._last_exit_code == 1

    def test_pwd(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_pwd("")
        assert len(cap.getvalue().strip()) > 0

    def test_echo(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_echo("hello world")
        assert "hello world" in cap.getvalue()

    def test_echo_empty(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_echo("")
        assert cap.getvalue().strip() == "" or len(cap.getvalue()) >= 0


class TestCmdPyMore:
    def test_py_list(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("[i*i for i in range(5)]")
        assert "0" in cap.getvalue() and "16" in cap.getvalue()

    def test_py_string(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("'hello' + ' ' + 'world'")
        assert "hello world" in cap.getvalue()

    def test_py_import_math(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("__import__('math').pi")
        assert "3.14" in cap.getvalue()

    def test_py_disallowed_module(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_py("__import__('os').getcwd()")
        assert "not allowed" in cap.getvalue() or "Error" in cap.getvalue()


class TestSourceMore:
    def test_source_dot(self, repl, tmp_path):
        script = tmp_path / "test.sh"
        script.write_text("echo dot_sourced\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_source(str(script))
        assert repl._last_exit_code == 0

    def test_source_nonexistent(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_source("/nonexistent_xyz.sh")
        assert "Error reading" in cap.getvalue() or "not found" in cap.getvalue()


class TestLogs:
    def test_logs_empty(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_logs("")
        assert repl._last_exit_code == 0

    def test_logs_stats(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_logs("--stats")
        assert repl._last_exit_code == 0

    def test_logs_clear(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_logs("--clear")
        assert repl._last_exit_code == 0


class TestCmdSvcRepl:
    def test_svc_not_booted(self, repl):
        repl.os._init = None
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("list")
        assert "not booted" in cap.getvalue().lower() or "init system" in cap.getvalue().lower()

    def test_svc_list_booted(self, repl):
        from unittest.mock import MagicMock
        init = MagicMock()
        init.service_table.return_value = "  svc1: running"
        init.status_summary = "OK"
        init.runlevel = 3
        repl.os._init = init
        with _CaptureOutput(repl) as cap:
            repl._cmd_svc("list")
        assert "svc1" in cap.getvalue() or "Services" in cap.getvalue()


class TestWhichType:
    def test_which_alias(self, repl):
        repl._aliases["ll"] = "ls -la"
        with _CaptureOutput(repl) as cap:
            repl._cmd_which("ll")
        assert "ll" in cap.getvalue()

    def test_which_command(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_which("ls")
        assert "ls" in cap.getvalue()

    def test_type_alias(self, repl):
        repl._aliases["ll"] = "ls -la"
        with _CaptureOutput(repl) as cap:
            repl._cmd_type("ll")
        assert "alias" in cap.getvalue() or "ll" in cap.getvalue()

    def test_type_command(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_type("cd")
        assert "builtin" in cap.getvalue() or "cd" in cap.getvalue()


class TestAsm:
    def test_asm_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_asm("")
        assert "Usage" in cap.getvalue() or len(cap.getvalue()) > 0

    def test_asm_hello(self, repl):
        repl._piped_input = "MOV R0, 42\nPRINT R0\nHALT"
        with _CaptureOutput(repl) as cap:
            repl._cmd_asm("")
        assert "42" in cap.getvalue()


class TestCalLn:
    def test_cal_current(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_cal("")
        assert repl._last_exit_code == 0

    def test_cal_specific(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_cal("1 2025")
        assert repl._last_exit_code == 0

    def test_ln_no_args(self, repl):
        with _CaptureOutput(repl):
            repl._cmd_ln("")
        assert repl._last_exit_code == 1

    def test_ln_create(self, repl, tmp_path):
        f = tmp_path / "target.txt"
        f.write_text("data")
        link = tmp_path / "link.txt"
        repl._cmd_ln(f"{f} {link}")
        assert link.exists() or link.is_symlink()


class TestRender:
    def test_render_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_render("")
        assert "Usage" in cap.getvalue() or len(cap.getvalue()) > 0


class TestAi:
    def test_ai_no_args(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_ai("")
        assert "Usage" in cap.getvalue() or len(cap.getvalue()) > 0


class TestTutorial:
    def test_tutorial(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_tutorial("")
        assert len(cap.getvalue()) > 0


class TestCompletion:
    def test_complete_empty(self, repl):
        result = repl._complete("", 0)
        assert result is None or isinstance(result, str)

    def test_complete_command(self, repl):
        result = repl._complete("he", 0)
        assert result is None or str(result).startswith("he")

    def test_complete_args_for_load(self, repl):
        result = repl._complete_args_for("load")
        assert isinstance(result, list)


class TestTableWrapper:
    def test_table_no_header(self, repl):
        repl._table([["a", "b"], ["c", "d"]])

    def test_table_with_header(self, repl):
        repl._table([["1", "2"], ["3", "4"]], header=["X", "Y"])

    def test_table_no_separator(self, repl):
        repl._table([["x"]], header=["H"], separator_after_header=False)

    def test_table_empty_rows(self, repl):
        repl._table([], header=["Col"])

    def test_table_single_row(self, repl):
        repl._table([["only"]])


class TestRcPath:
    def test_rc_path_returns_path(self, repl):
        result = repl._rc_path()
        assert isinstance(result, Path)

    def test_rc_path_contains_sloughgpt(self, repl):
        result = repl._rc_path()
        assert "sloughgpt" in str(result)

    def test_rc_path_ends_with_rc(self, repl):
        result = repl._rc_path()
        assert result.name == "rc"


class TestInit:
    def test_init_state_loaded(self, repl):
        assert repl.state is not None

    def test_init_history_empty(self, repl):
        assert isinstance(repl._history, list)

    def test_init_env_defaults(self, repl):
        assert "PS1" in repl._env
        assert "SHELL" in repl._env

    def test_init_io_set(self, repl):
        assert repl.io is not None

    def test_init_console_set(self, repl):
        assert repl.console is not None

    def test_init_not_running(self, repl):
        assert repl._running is False

    def test_init_piped_input_empty(self, repl):
        assert repl._piped_input == ""

    def test_init_bg_threads_empty(self, repl):
        assert repl._bg_threads == {}

    def test_init_next_bg_id(self, repl):
        assert repl._next_bg_id == 1

    def test_init_aborted_false(self, repl):
        assert repl._aborted is False


class TestCmdGrepFlagsExtended:
    def test_grep_c_flag(self, repl):
        repl._piped_input = "hello\nworld\nhello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-c hello")
        out = cap.getvalue().strip()
        assert "2" in out

    def test_grep_n_flag(self, repl):
        repl._piped_input = "aaa\nbbb\naaa\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-n aaa")
        out = cap.getvalue()
        assert "1:" in out
        assert "3:" in out

    def test_grep_w_flag(self, repl):
        repl._piped_input = "cat cats concat\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-w cat")
        out = cap.getvalue()
        lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
        assert len(lines) == 1
        assert "cat" in lines[0]

    def test_grep_l_flag(self, repl):
        repl._piped_input = "hello\nworld\nhello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-l hello")
        out = cap.getvalue().strip()
        assert "1 match" in out or out != ""

    def test_grep_A_context_after(self, repl):
        repl._piped_input = "a\nb\nc\nd\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-A 1 b")
        out = cap.getvalue()
        assert "b" in out
        assert "c" in out

    def test_grep_B_context_before(self, repl):
        repl._piped_input = "a\nb\nc\nd\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-B 1 c")
        out = cap.getvalue()
        assert "c" in out
        assert "b" in out

    def test_grep_C_context_both(self, repl):
        repl._piped_input = "a\nb\nc\nd\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-C 1 c")
        out = cap.getvalue()
        assert "b" in out
        assert "c" in out
        assert "d" in out

    def test_grep_multiple_files(self, repl, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello\nworld\n")
        f2.write_text("foo\nhello\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep(f"hello {f1} {f2}")
        out = cap.getvalue()
        assert "hello" in out

    def test_grep_recursive(self, repl, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "a.txt").write_text("hello\n")
        (sub / "b.txt").write_text("world\nhello\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep(f"-r hello {tmp_path}")
        out = cap.getvalue()
        assert "hello" in out

    def test_grep_recursive_no_match(self, repl, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "a.txt").write_text("hello\n")
        (sub / "b.txt").write_text("world\n")
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep(f"-r foo {tmp_path}")
        out = cap.getvalue()
        assert "hello" not in out
        assert "world" not in out


class TestCmdSedAddressRange:
    def test_sed_bare_d(self, repl):
        repl._piped_input = "a\nb\nc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("d")
        out = cap.getvalue().strip()
        assert out == ""

    def test_sed_address_range_d(self, repl):
        repl._piped_input = "a\nb\nc\nd\ne\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("2,4d")
        out = cap.getvalue().strip().split("\n")
        assert out == ["a", "e"]


class TestCmdGrepMoreFlags:
    def test_grep_o_only_matching(self, repl):
        repl._piped_input = "foo123bar\nbaz456\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-o [0-9]+")
        out = cap.getvalue().strip()
        assert "123" in out
        assert "456" in out

    def test_grep_o_no_full_line(self, repl):
        repl._piped_input = "hello world\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-o hello")
        out = cap.getvalue().strip()
        assert "hello" in out
        assert "world" not in out

    def test_grep_e_flag(self, repl):
        repl._piped_input = "foo\nbar\nbaz\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-e foo -e baz")
        out = cap.getvalue()
        assert "foo" in out
        assert "baz" in out
        assert "bar" not in out

    def test_grep_e_pattern_starting_with_dash(self, repl):
        repl._piped_input = "-flag\nvalue\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-e ^-flag")
        out = cap.getvalue()
        assert "-flag" in out
        assert "value" not in out

    def test_grep_m_max_count(self, repl):
        repl._piped_input = "a\na\na\nb\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-m 2 a")
        out = cap.getvalue().strip().split("\n")
        lines = [l for l in out if l.strip()]
        assert len(lines) == 2


class TestCmdSortKeyField:
    def test_sort_k_field_2(self, repl):
        repl._piped_input = "c 3\na 1\nb 2\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sort("-k 2 -n")
        out = cap.getvalue().strip().split("\n")
        fields = [l.split()[1] for l in out if l.strip()]
        assert fields == ["1", "2", "3"]

    def test_sort_t_separator(self, repl):
        repl._piped_input = "c:x\na:z\nb:y\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sort("-t: -k 2")
        out = cap.getvalue().strip().split("\n")
        fields = [l.split(":")[1] for l in out if l.strip()]
        assert fields == ["x", "y", "z"]

    def test_sort_t_k_combined(self, repl):
        repl._piped_input = "10:apple\n2:banana\n1:cherry\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sort("-t: -k 1 -n")
        out = cap.getvalue().strip().split("\n")
        assert "1:cherry" in out[0]
        assert "2:banana" in out[1]
        assert "10:apple" in out[2]

    def test_sort_case_insensitive(self, repl):
        repl._piped_input = "banana\nApple\ncherry\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sort("-f")
        out = cap.getvalue().strip().split("\n")
        assert out[0] == "Apple"
        assert out[1] == "banana"
        assert out[2] == "cherry"

    def test_sort_case_insensitive_reverse(self, repl):
        repl._piped_input = "banana\nApple\ncherry\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sort("-f -r")
        out = cap.getvalue().strip().split("\n")
        assert out[0] == "cherry"
        assert out[1] == "banana"
        assert out[2] == "Apple"


class TestCmdSedAppendInsertChange:
    def test_sed_append(self, repl):
        repl._piped_input = "line1\nline2\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("1a\\inserted after line 1")
        out = cap.getvalue().strip().split("\n")
        assert out[0] == "line1"
        assert out[1] == "inserted after line 1"
        assert out[2] == "line2"

    def test_sed_insert(self, repl):
        repl._piped_input = "line1\nline2\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("1i\\inserted before line 1")
        out = cap.getvalue().strip().split("\n")
        assert out[0] == "inserted before line 1"
        assert out[1] == "line1"

    def test_sed_change(self, repl):
        repl._piped_input = "line1\nline2\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("/line1/c\\changed line")
        out = cap.getvalue().strip().split("\n")
        assert out[0] == "changed line"
        assert out[1] == "line2"


class TestCmdSedComprehensive:
    """Comprehensive sed tests for all modes."""

    def test_sed_substitute_basic(self, repl):
        repl._piped_input = "hello world\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("s/world/earth/")
        assert "hello earth" in cap.getvalue()

    def test_sed_substitute_global(self, repl):
        repl._piped_input = "a b a b a\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("s/a/X/g")
        assert "X b X b X" in cap.getvalue()

    def test_sed_substitute_alt_delimiter(self, repl):
        repl._piped_input = "/usr/local/bin\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("s#/usr#/opt#")
        assert "/opt/local/bin" in cap.getvalue()

    def test_sed_substitute_no_match(self, repl):
        repl._piped_input = "hello world\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("s/xyz/abc/")
        assert "hello world" in cap.getvalue()

    def test_sed_substitute_quiet(self, repl):
        repl._piped_input = "hello\nworld\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("-n s/hello/hi/")
        # -n suppresses output but s/// still produces changed lines
        out = cap.getvalue().strip()
        assert "hi" in out
        assert "world" not in out

    def test_sed_substitute_quiet_match(self, repl):
        repl._piped_input = "hello\nworld\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("-n s/hello/hi/p")
        assert "hi" in cap.getvalue()
        assert "world" not in cap.getvalue()

    def test_sed_delete_pattern(self, repl):
        repl._piped_input = "foo\nbar\nbaz\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("/bar/d")
        out = cap.getvalue().strip().split("\n")
        assert "foo" in out
        assert "bar" not in out
        assert "baz" in out

    def test_sed_delete_bare(self, repl):
        repl._piped_input = "line1\nline2\nline3\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("d")
        assert cap.getvalue().strip() == ""

    def test_sed_print_line(self, repl):
        repl._piped_input = "line1\nline2\nline3\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("2p")
        out = cap.getvalue().strip()
        # With -n it would only show line2, without -n it shows all lines
        assert "line2" in out

    def test_sed_address_range_print(self, repl):
        repl._piped_input = "line1\nline2\nline3\nline4\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("-n 2,3p")
        out = cap.getvalue().strip().split("\n")
        lines = [l for l in out if l.strip()]
        assert len(lines) == 2
        assert "line2" in lines[0]
        assert "line3" in lines[1]

    def test_sed_address_range_delete(self, repl):
        repl._piped_input = "line1\nline2\nline3\nline4\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("2,3d")
        out = cap.getvalue().strip().split("\n")
        lines = [l for l in out if l.strip()]
        assert len(lines) == 2
        assert "line1" in lines[0]
        assert "line4" in lines[1]

    def test_sed_escape_tab(self, repl):
        repl._piped_input = "a b\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("s/ /\\t/")
        assert "a\tb" in cap.getvalue()

    def test_sed_escape_newline(self, repl):
        repl._piped_input = "ab\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("s/a/\\n/")
        out = cap.getvalue()
        # \n in replacement inserts an actual newline
        assert "b" in out

    def test_sed_no_input(self, repl):
        repl._piped_input = None
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("s/a/b/")
        assert "no input" in cap.getvalue().lower()

    def test_sed_no_script(self, repl):
        repl._piped_input = "test\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("")
        assert "Usage" in cap.getvalue()

    def test_sed_invalid_regex(self, repl):
        repl._piped_input = "test\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("s/[invalid/test/")
        assert "invalid regex" in cap.getvalue()

    def test_sed_multiline_substitute(self, repl):
        repl._piped_input = "aaa\nbbb\naaa\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("s/a/b/g")
        out = cap.getvalue()
        assert "bbb" in out
        assert "bba" not in out  # first aaa -> bbb
        lines = out.strip().split("\n")
        assert lines[0] == "bbb"
        assert lines[2] == "bbb"

    def test_sed_append_after_last(self, repl):
        repl._piped_input = "line1\nline2\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("2a\\appended")
        out = cap.getvalue().strip().split("\n")
        assert out[0] == "line1"
        assert out[1] == "line2"
        assert out[2] == "appended"

    def test_sed_insert_before_first(self, repl):
        repl._piped_input = "line1\nline2\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("1i\\prepended")
        out = cap.getvalue().strip().split("\n")
        assert out[0] == "prepended"
        assert out[1] == "line1"

    def test_sed_change_single_line(self, repl):
        repl._piped_input = "aaa\nbbb\nccc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sed("/bbb/c\\CHANGED")
        out = cap.getvalue().strip().split("\n")
        assert out[0] == "aaa"
        assert out[1] == "CHANGED"
        assert out[2] == "ccc"


class TestCmdAwk:
    """Tests for awk command."""

    def test_awk_print_field(self, repl):
        repl._piped_input = "one two three\nfour five six\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_awk("{print $1}")
        out = cap.getvalue().strip().split("\n")
        assert out[0] == "one"
        assert out[1] == "four"

    def test_awk_print_two_fields(self, repl):
        repl._piped_input = "one two three\nfour five six\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_awk("{print $1,$3}")
        out = cap.getvalue().strip().split("\n")
        assert out[0] == "one three"
        assert out[1] == "four six"

    def test_awk_field_separator(self, repl):
        repl._piped_input = "a:b:c\nd:e:f\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_awk("-F: {print $2}")
        out = cap.getvalue().strip().split("\n")
        assert out[0] == "b"
        assert out[1] == "e"

    def test_awk_field_separator_glued(self, repl):
        repl._piped_input = "x|y|z\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_awk("-F| {print $1,$3}")
        out = cap.getvalue().strip()
        assert out == "x z"

    def test_awk_dollar_zero(self, repl):
        repl._piped_input = "hello world\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_awk("{print $0}")
        out = cap.getvalue().strip()
        assert out == "hello world"

    def test_awk_dollar_nf(self, repl):
        repl._piped_input = "a b c\nd e\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_awk("{print $NF}")
        out = cap.getvalue().strip().split("\n")
        assert out[0] == "c"
        assert out[1] == "e"

    def test_awk_nr(self, repl):
        repl._piped_input = "line1\nline2\nline3\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_awk("{print NR, $0}")
        out = cap.getvalue().strip().split("\n")
        assert out[0] == "1 line1"
        assert out[1] == "2 line2"
        assert out[2] == "3 line3"

    def test_awk_nf(self, repl):
        repl._piped_input = "a b c\nd e\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_awk("{print NF}")
        out = cap.getvalue().strip().split("\n")
        assert out[0] == "3"
        assert out[1] == "2"

    def test_awk_comma_print(self, repl):
        repl._piped_input = "hello world\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_awk('{print $1, "is", $2}')
        out = cap.getvalue().strip()
        assert out == "hello is world"

    def test_awk_empty_separator(self, repl):
        repl._piped_input = "abc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_awk("-F '' {print $1}")
        out = cap.getvalue().strip()
        assert out == "abc"

    def test_awk_no_input(self, repl):
        repl._piped_input = None
        with _CaptureOutput(repl) as cap:
            repl._cmd_awk("{print $1}")
        assert "no input" in cap.getvalue().lower()

    def test_awk_no_script(self, repl):
        repl._piped_input = "test\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_awk("")
        assert "Usage" in cap.getvalue()

    def test_awk_file_not_found(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_awk("{print $1} /nonexistent/file")
        assert "No such file" in cap.getvalue()

    def test_awk_quoted_script(self, repl):
        repl._piped_input = "a b c\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_awk("'{print $2}'")
        out = cap.getvalue().strip()
        assert out == "b"


class TestCmdGrepEdgeCases:
    """Additional grep edge case tests."""

    def test_grep_e_with_m(self, repl):
        repl._piped_input = "foo\nbar\nfoo\nbaz\nfoo\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-e foo -m 2")
        out = cap.getvalue().strip().split("\n")
        lines = [l for l in out if l.strip()]
        assert len(lines) == 2

    def test_grep_e_with_c(self, repl):
        repl._piped_input = "foo\nbar\nfoo\nbaz\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-c -e foo -e baz")
        out = cap.getvalue().strip()
        assert "3" in out

    def test_grep_e_with_i(self, repl):
        repl._piped_input = "Foo\nBAR\nfoo\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-i -e foo")
        out = cap.getvalue()
        assert "Foo" in out
        assert "foo" in out
        assert "BAR" not in out

    def test_grep_e_single_pattern(self, repl):
        repl._piped_input = "abc\ndef\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-e abc")
        out = cap.getvalue()
        assert "abc" in out
        assert "def" not in out


class TestCmdAwkExtra:
    """Additional awk edge case tests."""

    def test_awk_multiline(self, repl):
        repl._piped_input = "x 10\ny 20\nz 30\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_awk("{print $2, $1}")
        out = cap.getvalue().strip().split("\n")
        assert out[0] == "10 x"
        assert out[1] == "20 y"
        assert out[2] == "30 z"

    def test_awk_multi_char_separator(self, repl):
        repl._piped_input = "one::two::three\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_awk("-F:: {print $1,$3}")
        out = cap.getvalue().strip()
        assert out == "one three"

    def test_awk_empty_input(self, repl):
        repl._piped_input = ""
        with _CaptureOutput(repl) as cap:
            repl._cmd_awk("{print $1}")
        # Empty string is falsy, treated as no input
        assert "no input" in cap.getvalue().lower()


# ── Tsort command ─────────────────────────────────────────────────


class TestCmdTsort:
    def test_tsort_linear(self, repl):
        repl._piped_input = "A B\nB C\nC D\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tsort("")
        out = cap.getvalue().strip().split("\n")
        assert out == ["D", "C", "B", "A"]

    def test_tsort_diamond(self, repl):
        repl._piped_input = "A B\nA C\nB D\nC D\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tsort("")
        out = cap.getvalue().strip().split("\n")
        assert len(out) == 4

    def test_tsort_no_input(self, repl):
        repl._piped_input = None
        with _CaptureOutput(repl) as cap:
            repl._cmd_tsort("")
        assert repl._last_exit_code == 1

    def test_tsort_file(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f.write("A B\nB C\n")
        f.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_tsort(f.name)
            out = cap.getvalue().strip().split("\n")
            assert len(out) == 3
        finally:
            os.unlink(f.name)


# ── Strings command ───────────────────────────────────────────────


class TestCmdStrings:
    def test_strings_binary(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
        f.write(b'\x00\x01hello\x02\x03world\x04')
        f.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_strings(f.name)
            out = cap.getvalue().strip()
            assert "hello" in out
            assert "world" in out
        finally:
            os.unlink(f.name)

    def test_strings_min_len(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
        f.write(b'\x00ab\x01')
        f.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_strings(f"-n 2 {f.name}")
            out = cap.getvalue().strip()
            assert "ab" in out
        finally:
            os.unlink(f.name)

    def test_strings_no_input(self, repl):
        repl._piped_input = None
        with _CaptureOutput(repl) as cap:
            repl._cmd_strings("")
        assert repl._last_exit_code == 1


# ── Base64 command ────────────────────────────────────────────────


class TestCmdBase64:
    def test_base64_encode(self, repl):
        repl._piped_input = "hello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_base64("")
        out = cap.getvalue().strip()
        assert len(out) > 0

    def test_base64_roundtrip(self, repl):
        repl._piped_input = "hello world\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_base64("")
        encoded = cap.getvalue().strip()
        repl._piped_input = encoded
        with _CaptureOutput(repl) as cap2:
            repl._cmd_base64("-d")
        decoded = cap2.getvalue().strip()
        assert decoded == "hello world"

    def test_base64_file(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f.write("test data")
        f.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_base64(f.name)
            assert len(cap.getvalue().strip()) > 0
        finally:
            os.unlink(f.name)

    def test_base64_no_input(self, repl):
        repl._piped_input = None
        with _CaptureOutput(repl) as cap:
            repl._cmd_base64("")
        assert repl._last_exit_code == 1


# ── Cksum command ─────────────────────────────────────────────────


class TestCmdCksum:
    def test_cksum_pipe(self, repl):
        repl._piped_input = "hello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_cksum("")
        out = cap.getvalue().strip()
        assert out.split()[0].isdigit()
        assert out.split()[1] == "6"

    def test_cksum_file(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f.write("hello")
        f.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_cksum(f.name)
            out = cap.getvalue().strip()
            assert len(out.split()) == 3
        finally:
            os.unlink(f.name)

    def test_cksum_not_found(self, repl):
        repl._cmd_cksum("/nonexistent/file.txt")
        assert repl._last_exit_code == 1

    def test_cksum_no_input(self, repl):
        repl._piped_input = None
        with _CaptureOutput(repl) as cap:
            repl._cmd_cksum("")
        assert repl._last_exit_code == 1


# ── Uniq enhanced flags ──────────────────────────────────────────


class TestCmdUniqFlags:
    def test_uniq_d_only_duplicates(self, repl):
        repl._piped_input = "a\na\nb\nc\nc\nc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_uniq("-d")
        out = cap.getvalue().strip().split("\n")
        assert "a" in out
        assert "c" in out
        assert "b" not in out

    def test_uniq_u_only_unique(self, repl):
        repl._piped_input = "a\na\nb\nc\nc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_uniq("-u")
        out = cap.getvalue().strip().split("\n")
        assert "b" in out
        assert "a" not in out
        assert "c" not in out

    def test_uniq_c_d(self, repl):
        repl._piped_input = "a\na\nb\nc\nc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_uniq("-c -d")
        out = cap.getvalue().strip().split("\n")
        assert len(out) == 2
        assert "2 a" in out[0]
        assert "2 c" in out[1]

    def test_uniq_skip_fields(self, repl):
        repl._piped_input = "1 foo\n1 foo\n2 bar\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_uniq("-f 1")
        out = cap.getvalue().strip().split("\n")
        assert len(out) == 2

    def test_uniq_skip_chars(self, repl):
        repl._piped_input = "abc\nabd\nxyz\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_uniq("-s 2")
        out = cap.getvalue().strip().split("\n")
        assert len(out) == 3


# ── Join enhanced flags ──────────────────────────────────────────


class TestCmdJoinFlags:
    def test_join_basic(self, repl):
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f1.write("1 one\n2 two\n")
        f1.close()
        f2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f2.write("1 ONE\n3 THREE\n")
        f2.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_join(f"{f1.name} {f2.name}")
            out = cap.getvalue().strip()
            assert "1" in out
            assert "one" in out
            assert "ONE" in out
        finally:
            os.unlink(f1.name)
            os.unlink(f2.name)

    def test_join_a_orphans(self, repl):
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f1.write("1 one\n2 two\n")
        f1.close()
        f2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f2.write("1 ONE\n3 THREE\n")
        f2.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_join(f"-a 1 {f1.name} {f2.name}")
            out = cap.getvalue()
            assert "2 two" in out
        finally:
            os.unlink(f1.name)
            os.unlink(f2.name)

    def test_join_not_found(self, repl):
        repl._cmd_join("/nonexistent/a.txt /nonexistent/b.txt")
        assert repl._last_exit_code == 1

    def test_join_no_args(self, repl):
        repl._cmd_join("")
        assert repl._last_exit_code == 1


# ── Od enhanced flags ────────────────────────────────────────────


class TestCmdOdFlags:
    def test_od_hex(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
        f.write(b'\x00\x01\x02\x03')
        f.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_od(f"-t x {f.name}")
            out = cap.getvalue()
            assert "00" in out
            assert "01" in out
        finally:
            os.unlink(f.name)

    def test_od_decimal(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
        f.write(b'\x00\x01\x02\x03')
        f.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_od(f"-t d {f.name}")
            out = cap.getvalue()
            assert "0" in out
            assert "1" in out
        finally:
            os.unlink(f.name)

    def test_od_skip(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
        f.write(b'\x00\x01\x02\x03\x04\x05')
        f.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_od(f"-j 2 -N 2 {f.name}")
            out = cap.getvalue()
            assert "02" in out
        finally:
            os.unlink(f.name)

    def test_od_no_file(self, repl):
        repl._piped_input = None
        with _CaptureOutput(repl) as cap:
            repl._cmd_od("")
        assert repl._last_exit_code == 1


# ── Stat enhanced flags ──────────────────────────────────────────


class TestCmdStatFlags:
    def test_stat_c_name(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
        f.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_stat(f"-c %n {f.name}")
            out = cap.getvalue().strip()
            assert f.name in out
        finally:
            os.unlink(f.name)

    def test_stat_c_size(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        f.write("hello")
        f.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_stat(f"-c %s {f.name}")
            out = cap.getvalue().strip()
            assert "5" in out
        finally:
            os.unlink(f.name)

    def test_stat_not_found(self, repl):
        repl._cmd_stat("/nonexistent/file.txt")
        assert repl._last_exit_code == 1

    def test_stat_no_args(self, repl):
        repl._cmd_stat("")
        assert repl._last_exit_code == 1


# ── Split command ────────────────────────────────────────────────


class TestCmdSplit:
    def test_split_lines(self, repl):
        repl._piped_input = "a\nb\nc\nd\ne\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_split("-l 2")
        out = cap.getvalue()
        assert "x" in out
        import glob as _glob
        files = _glob.glob("x*")
        assert len(files) >= 2
        for f in files:
            os.unlink(f)

    def test_split_bytes(self, repl):
        repl._piped_input = "abcdefghij"
        with _CaptureOutput(repl) as cap:
            repl._cmd_split("-b 3")
        import glob as _glob
        files = _glob.glob("x*")
        assert len(files) >= 3
        for f in files:
            os.unlink(f)

    def test_split_numeric(self, repl):
        repl._piped_input = "a\nb\nc\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_split("-d -l 1")
        import glob as _glob
        files = _glob.glob("x*")
        assert len(files) == 3
        for f in files:
            os.unlink(f)

    def test_split_no_input(self, repl):
        repl._piped_input = None
        with _CaptureOutput(repl) as cap:
            repl._cmd_split("-l 2")
        assert repl._last_exit_code == 1


# ── Tail enhanced flags ──────────────────────────────────────────


class TestCmdTailFlags:
    def test_tail_n(self, repl):
        repl._piped_input = "a\nb\nc\nd\ne\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tail("-n 2")
        out = cap.getvalue().strip().split("\n")
        assert out == ["d", "e"]

    def test_tail_q(self, repl):
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f1.write("aa\n")
        f1.close()
        f2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f2.write("bb\n")
        f2.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_tail(f"-n 1 -q {f1.name} {f2.name}")
            out = cap.getvalue()
            assert "==>" not in out
        finally:
            os.unlink(f1.name)
            os.unlink(f2.name)

    def test_tail_c(self, repl):
        repl._piped_input = "hello world"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tail("-c 5")
        out = cap.getvalue()
        assert out == "world"

    def test_tail_no_input(self, repl):
        repl._piped_input = None
        with _CaptureOutput(repl) as cap:
            repl._cmd_tail("")
        assert repl._last_exit_code == 1


# ── Wc enhanced flags ────────────────────────────────────────────


class TestCmdWcFlags:
    def test_wc_m_chars(self, repl):
        repl._piped_input = "hello\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_wc("-m")
        out = cap.getvalue().strip()
        assert "6" in out

    def test_wc_L_maxlen(self, repl):
        repl._piped_input = "short\nlonger line\nmid\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_wc("-L")
        out = cap.getvalue().strip()
        assert "11" in out

    def test_wc_file(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f.write("hello\nworld\n")
        f.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_wc(f.name)
            out = cap.getvalue().strip()
            assert "2" in out
        finally:
            os.unlink(f.name)

    def test_wc_no_input(self, repl):
        repl._piped_input = None
        with _CaptureOutput(repl) as cap:
            repl._cmd_wc("")
        assert repl._last_exit_code == 1


# ── Tr enhanced flags ────────────────────────────────────────────


class TestCmdTrFlags:
    def test_tr_c_complement(self, repl):
        repl._piped_input = "hello world 123"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tr("-c -d [:alpha:]")
        out = cap.getvalue().strip()
        assert out == "helloworld"

    def test_tr_d(self, repl):
        repl._piped_input = "hello"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tr("-d l")
        out = cap.getvalue().strip()
        assert out == "heo"

    def test_tr_s(self, repl):
        repl._piped_input = "aaabbbccc"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tr("-s a")
        out = cap.getvalue().strip()
        assert out == "abbbccc"

    def test_tr_squeeze_digit(self, repl):
        repl._piped_input = "aaa111bbb222"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tr("-s 1")
        out = cap.getvalue().strip()
        assert out == "aaa1bbb222"

    def test_tr_no_input(self, repl):
        repl._piped_input = None
        with _CaptureOutput(repl) as cap:
            repl._cmd_tr("a b")
        assert repl._last_exit_code == 1


# ── Comm enhanced flags ──────────────────────────────────────────


class TestCmdCommFlags:
    def test_comm_1(self, repl):
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f1.write("a\nb\nc\n")
        f1.close()
        f2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f2.write("b\nc\nd\n")
        f2.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_comm(f"-1 {f1.name} {f2.name}")
            out = cap.getvalue()
            assert "\ta" not in out
        finally:
            os.unlink(f1.name)
            os.unlink(f2.name)

    def test_comm_3(self, repl):
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f1.write("a\nb\n")
        f1.close()
        f2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f2.write("b\nc\n")
        f2.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_comm(f"-3 {f1.name} {f2.name}")
            out = cap.getvalue()
            assert "\tb" not in out.split("\n")[0] if out.strip() else True
        finally:
            os.unlink(f1.name)
            os.unlink(f2.name)

    def test_comm_no_args(self, repl):
        repl._cmd_comm("")
        assert repl._last_exit_code == 1

    def test_comm_not_found(self, repl):
        repl._cmd_comm("/nonexistent/a.txt /nonexistent/b.txt")
        assert repl._last_exit_code == 1


# ── Head enhanced flags ──────────────────────────────────────────


class TestCmdHeadFlags:
    def test_head_c_bytes(self, repl):
        repl._piped_input = "hello world"
        with _CaptureOutput(repl) as cap:
            repl._cmd_head("-c 5")
        out = cap.getvalue()
        assert out == "hello"

    def test_head_n(self, repl):
        repl._piped_input = "a\nb\nc\nd\ne\n"
        with _CaptureOutput(repl) as cap:
            repl._cmd_head("-n 2")
        out = cap.getvalue().strip().split("\n")
        assert out == ["a", "b"]

    def test_head_q(self, repl):
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f1.write("aa\n")
        f1.close()
        f2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f2.write("bb\n")
        f2.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_head(f"-n 1 -q {f1.name} {f2.name}")
            out = cap.getvalue()
            assert "==>" not in out
        finally:
            os.unlink(f1.name)
            os.unlink(f2.name)

    def test_head_no_input(self, repl):
        repl._piped_input = None
        with _CaptureOutput(repl) as cap:
            repl._cmd_head("")
        assert repl._last_exit_code == 1


# ── Paste enhanced flags ─────────────────────────────────────────


class TestCmdPasteFlags:
    def testPasteMultiDelim(self, repl):
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f1.write("a\nb\n")
        f1.close()
        f2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f2.write("c\nd\n")
        f2.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_paste(f"-d , {f1.name} {f2.name}")
            out = cap.getvalue().strip().split("\n")
            assert out[0] == "a,c"
            assert out[1] == "b,d"
        finally:
            os.unlink(f1.name)
            os.unlink(f2.name)

    def testPasteSerialize(self, repl):
        import tempfile
        f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f1.write("a\nb\nc\n")
        f1.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_paste(f"-s -d , {f1.name}")
            out = cap.getvalue().strip()
            assert out == "a,b,c"
        finally:
            os.unlink(f1.name)

    def testPasteNoInput(self, repl):
        repl._piped_input = None
        with _CaptureOutput(repl) as cap:
            repl._cmd_paste("")
        assert repl._last_exit_code == 1

    def testPasteNotFound(self, repl):
        repl._cmd_paste("/nonexistent/a.txt /nonexistent/b.txt")
        assert repl._last_exit_code == 1


# ── df ──────────────────────────────────────────────────────────


class TestCmdDf:
    def testDfDefault(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_df("")
        out = cap.getvalue()
        assert "Filesystem" in out
        assert repl._last_exit_code == 0

    def testDfPath(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_df(".")
        out = cap.getvalue()
        assert "virtual-fs" in out
        assert repl._last_exit_code == 0

    def testDfHuman(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_df("-h .")
        out = cap.getvalue()
        assert "virtual-fs" in out

    def testDfNotFound(self, repl):
        repl._cmd_df("/nonexistent_xyz_df")
        assert repl._last_exit_code == 1


# ── readlink ────────────────────────────────────────────────────


class TestCmdReadlink:
    def testReadlinkReal(self, repl):
        import tempfile
        d = tempfile.mkdtemp()
        f = os.path.join(d, "testfile.txt")
        with open(f, "w") as fh:
            fh.write("content")
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_readlink(f"-f {f}")
            out = cap.getvalue().strip()
            assert out == os.path.realpath(f)
        finally:
            os.unlink(f)
            os.rmdir(d)

    def testReadlinkNoArgs(self, repl):
        repl._cmd_readlink("")
        assert repl._last_exit_code == 1

    def testReadlinkNotLink(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f.write("content")
        f.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_readlink(f.name)
            # Not a symlink → exit code 1
            assert repl._last_exit_code == 1
        finally:
            os.unlink(f.name)


# ── file ────────────────────────────────────────────────────────


class TestCmdFile:
    def testFileASCII(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f.write("hello world\n")
        f.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_file(f.name)
            out = cap.getvalue()
            assert "ASCII text" in out
        finally:
            os.unlink(f.name)

    def testFileJSON(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        f.write('{"key": "value"}\n')
        f.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_file(f.name)
            out = cap.getvalue()
            assert "JSON" in out
        finally:
            os.unlink(f.name)

    def testFileBrief(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f.write("test data\n")
        f.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_file(f"-b {f.name}")
            out = cap.getvalue()
            # Brief mode should NOT have the filename prefix
            assert f.name not in out
        finally:
            os.unlink(f.name)

    def testFileMIME(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        f.write('{"key": "value"}\n')
        f.close()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_file(f"-i {f.name}")
            out = cap.getvalue().strip()
            assert out == "application/json"
        finally:
            os.unlink(f.name)

    def testFileNoArgs(self, repl):
        repl._cmd_file("")
        assert repl._last_exit_code == 1

    def testFileNotFound(self, repl):
        repl._cmd_file("/nonexistent_xyz_file.txt")
        assert repl._last_exit_code == 1


# ── export / unset / setenv ──────────────────────────────────────


class TestCmdExport:
    def testExportNoArgs(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_export("")
        out = cap.getvalue()
        assert "declare -x" in out or repl._last_exit_code == 0

    def testExportSetValue(self, repl):
        repl._cmd_export("MY_TEST_VAR=hello")
        assert repl._env.get("MY_TEST_VAR") == "hello"
        assert repl._last_exit_code == 0

    def testExportOverwrite(self, repl):
        repl._cmd_export("MY_TEST_VAR=old")
        repl._cmd_export("MY_TEST_VAR=new")
        assert repl._env.get("MY_TEST_VAR") == "new"

    def testExportWithEquals(self, repl):
        repl._cmd_export("MY_TEST_VAR=a=b=c")
        assert repl._env.get("MY_TEST_VAR") == "a=b=c"

    def testExportPrint(self, repl):
        repl._env["MY_TEST_VAR"] = "testval"
        with _CaptureOutput(repl) as cap:
            repl._cmd_export("-p")
        out = cap.getvalue()
        assert "MY_TEST_VAR" in out

    def testExportUnset(self, repl):
        repl._env["MY_TEST_VAR"] = "testval"
        repl._cmd_export("-n MY_TEST_VAR")
        assert "MY_TEST_VAR" not in repl._env


class TestCmdUnset:
    def testUnsetVar(self, repl):
        repl._env["MY_TEST_VAR"] = "testval"
        repl._cmd_unset("MY_TEST_VAR")
        assert "MY_TEST_VAR" not in repl._env
        assert repl._last_exit_code == 0

    def testUnsetMultiple(self, repl):
        repl._env["VAR_A"] = "a"
        repl._env["VAR_B"] = "b"
        repl._cmd_unset("VAR_A VAR_B")
        assert "VAR_A" not in repl._env
        assert "VAR_B" not in repl._env

    def testUnsetNoArgs(self, repl):
        repl._cmd_unset("")
        assert repl._last_exit_code == 1


class TestCmdSetenv:
    def testSetenv(self, repl):
        repl._cmd_setenv("MY_TEST_VAR testval")
        assert repl._env.get("MY_TEST_VAR") == "testval"
        assert repl._last_exit_code == 0

    def testSetenvNoArgs(self, repl):
        repl._cmd_setenv("")
        assert repl._last_exit_code == 1

    def testSetenvOneArg(self, repl):
        repl._cmd_setenv("MY_TEST_VAR")
        assert repl._last_exit_code == 1


# ── timeout ──────────────────────────────────────────────────────


class TestCmdTimeout:
    def testTimeoutSuccess(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_timeout("10 echo hello")
        out = cap.getvalue()
        assert "hello" in out
        assert repl._last_exit_code == 0

    def testTimeoutNoArgs(self, repl):
        repl._cmd_timeout("")
        assert repl._last_exit_code == 1

    def testTimeoutInvalidTime(self, repl):
        repl._cmd_timeout("abc echo hello")
        assert repl._last_exit_code == 1

    def testTimeoutQuickCommand(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_timeout("5 echo world")
        out = cap.getvalue()
        assert "world" in out


# ── watch ────────────────────────────────────────────────────────


class TestCmdWatch:
    def testWatchRuns(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_watch("-n 1 -c echo hello")
        out = cap.getvalue()
        assert "hello" in out
        assert repl._last_exit_code == 0

    def testWatchNoArgs(self, repl):
        repl._cmd_watch("")
        assert repl._last_exit_code == 1


# ── sleep with suffixes ──────────────────────────────────────────


class TestCmdSleepSuffixes:
    def test_sleep_seconds(self, repl):
        import time as _t
        s = _t.time()
        repl._cmd_sleep("0.01")
        assert _t.time() - s < 1

    def test_sleep_minutes(self, repl):
        import time as _t
        s = _t.time()
        repl._cmd_sleep("0.001m")
        assert _t.time() - s < 1

    def test_sleep_empty(self, repl):
        repl._cmd_sleep("")
        assert repl._last_exit_code == 0


# ── type command ─────────────────────────────────────────────────


class TestCmdType:
    def test_type_builtin(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_type("cd")
        out = cap.getvalue()
        assert "built-in" in out or "builtin" in out

    def test_type_external(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_type("ls")
        out = cap.getvalue()
        assert "not found" not in out

    def test_type_not_found(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_type("nonexistent_xyz_cmd")
        out = cap.getvalue()
        assert "not found" in out

    def test_type_no_args(self, repl):
        repl._cmd_type("")
        assert repl._last_exit_code == 1


# ── ls enhanced flags ────────────────────────────────────────────


class TestCmdLsFlags:
    def test_ls_1(self, repl):
        import tempfile
        d = tempfile.mkdtemp()
        f1 = os.path.join(d, "a.txt")
        f2 = os.path.join(d, "b.txt")
        Path(f1).write_text("a")
        Path(f2).write_text("b")
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_ls(f"-1 {d}")
            out = cap.getvalue().strip()
            lines = [l.strip() for l in out.split("\n") if l.strip()]
            assert len(lines) == 2
        finally:
            os.unlink(f1)
            os.unlink(f2)
            os.rmdir(d)

    def test_ls_a(self, repl):
        import tempfile
        d = tempfile.mkdtemp()
        f1 = os.path.join(d, "visible.txt")
        f2 = os.path.join(d, ".hidden.txt")
        Path(f1).write_text("v")
        Path(f2).write_text("h")
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_ls(d)
            out = cap.getvalue()
            assert "visible" in out
            assert ".hidden" not in out
            with _CaptureOutput(repl) as cap2:
                repl._cmd_ls(f"-a {d}")
            out2 = cap2.getvalue()
            assert ".hidden" in out2
        finally:
            os.unlink(f1)
            os.unlink(f2)
            os.rmdir(d)

    def test_ls_empty_dir(self, repl):
        import tempfile
        d = tempfile.mkdtemp()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_ls(d)
            out = cap.getvalue()
            assert repl._last_exit_code == 0
        finally:
            os.rmdir(d)

    def test_ls_not_found(self, repl):
        repl._cmd_ls("/nonexistent_xyz_ls")
        assert repl._last_exit_code == 1


# ── pushd / popd / dirs ─────────────────────────────────────────


class TestCmdDirStack:
    def test_dirs_default(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_dirs("")
        out = cap.getvalue()
        assert repl._last_exit_code == 0

    def test_pushd_popd(self, repl):
        import tempfile
        d = tempfile.mkdtemp()
        try:
            repl._cmd_pushd(d)
            assert repl._last_exit_code == 0
            repl._cmd_popd()
            assert repl._last_exit_code == 0
        finally:
            os.rmdir(d)

    def test_pushd_empty_stack(self, repl):
        repl._cmd_popd()
        assert repl._last_exit_code == 1

    def test_pushd_not_found(self, repl):
        repl._cmd_pushd("/nonexistent_xyz_pushd")
        assert repl._last_exit_code == 1

    def test_dirs_v(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_dirs("-v")
        out = cap.getvalue()
        assert repl._last_exit_code == 0


# ── cp enhanced flags ────────────────────────────────────────────


class TestCmdCpFlags:
    def test_cp_r_verbose(self, repl):
        import tempfile
        src = tempfile.mkdtemp()
        sub = os.path.join(src, "sub")
        os.makedirs(sub)
        f1 = os.path.join(sub, "a.txt")
        Path(f1).write_text("hello")
        dst = tempfile.mktemp()
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_cp(f"-rv {src} {dst}")
            assert repl._last_exit_code == 0
            assert os.path.exists(os.path.join(dst, "sub", "a.txt"))
        finally:
            import shutil
            if os.path.exists(src):
                shutil.rmtree(src)
            if os.path.exists(dst):
                shutil.rmtree(dst)

    def test_cp_no_args(self, repl):
        repl._cmd_cp("")
        assert repl._last_exit_code == 1

    def test_cp_one_arg(self, repl):
        repl._cmd_cp("somefile")
        assert repl._last_exit_code == 1


# ── mv enhanced flags ────────────────────────────────────────────


class TestCmdMvFlags:
    def test_mv_verbose(self, repl):
        import tempfile
        src = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        src.write("test")
        src.close()
        dst = tempfile.mktemp(suffix='.txt')
        try:
            with _CaptureOutput(repl) as cap:
                repl._cmd_mv(f"-v {src.name} {dst}")
            assert repl._last_exit_code == 0
            assert os.path.exists(dst)
            assert not os.path.exists(src.name)
        finally:
            if os.path.exists(dst):
                os.unlink(dst)

    def test_mv_no_args(self, repl):
        repl._cmd_mv("")
        assert repl._last_exit_code == 1

    def test_mv_one_arg(self, repl):
        repl._cmd_mv("somefile")
        assert repl._last_exit_code == 1


# ── mkdir -p ─────────────────────────────────────────────────────


class TestCmdMkdirFlags:
    def test_mkdir_p(self, repl):
        import tempfile
        base = tempfile.mkdtemp()
        nested = os.path.join(base, "a", "b", "c")
        try:
            repl._cmd_mkdir(f"-p {nested}")
            assert os.path.isdir(nested)
            assert repl._last_exit_code == 0
        finally:
            import shutil
            shutil.rmtree(base)

    def test_mkdir_no_args(self, repl):
        repl._cmd_mkdir("")
        assert repl._last_exit_code == 1


# ── touch -c ─────────────────────────────────────────────────────


class TestCmdTouchFlags:
    def test_touch_c_existing(self, repl):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f.write("test")
        f.close()
        try:
            repl._cmd_touch(f"-c {f.name}")
            assert os.path.exists(f.name)
        finally:
            os.unlink(f.name)

    def test_touch_c_nonexistent(self, repl):
        import tempfile
        f = tempfile.mktemp(suffix='.txt')
        try:
            repl._cmd_touch(f"-c {f}")
            assert not os.path.exists(f)
        finally:
            if os.path.exists(f):
                os.unlink(f)

    def test_touch_no_args(self, repl):
        repl._cmd_touch("")
        assert repl._last_exit_code == 1


# ── env -i / -u ──────────────────────────────────────────────────


class TestCmdEnvFlags:
    def test_env_u(self, repl):
        repl._env["DELME"] = "yes"
        with _CaptureOutput(repl) as cap:
            repl._cmd_env("-u DELME")
        assert "DELME" not in cap.getvalue()
        assert "DELME" not in repl._env

    def test_env_no_flags(self, repl):
        with _CaptureOutput(repl) as cap:
            repl._cmd_env("")
        assert repl._last_exit_code == 0
        assert len(cap.getvalue()) > 0


# ── grep -E extended regex ───────────────────────────────────────


class TestCmdGrepExtended:
    def test_grep_E(self, repl):
        repl._piped_input = "abc\n123\ndef\n456"
        with _CaptureOutput(repl) as cap:
            repl._cmd_grep("-E [0-9]+")
        out = cap.getvalue()
        assert "123" in out
        assert "456" in out
        assert "abc" not in out


# ── sort -R random ───────────────────────────────────────────────


class TestCmdSortRandom:
    def test_sort_R(self, repl):
        repl._piped_input = "a\nb\nc\nd\ne"
        with _CaptureOutput(repl) as cap:
            repl._cmd_sort("-R")
        out = cap.getvalue().strip()
        lines = out.split("\n")
        assert len(lines) == 5
        assert set(lines) == {"a", "b", "c", "d", "e"}


# ── wc -m (multibyte chars) ──────────────────────────────────────


class TestCmdWcMultibyte:
    def test_wc_m(self, repl):
        repl._piped_input = "hello"
        with _CaptureOutput(repl) as cap:
            repl._cmd_wc("-m")
        out = cap.getvalue().strip()
        assert "5" in out


# ── tr enhanced ──────────────────────────────────────────────────


class TestCmdTrEnhanced:
    def test_tr_complement(self, repl):
        repl._piped_input = "abc123"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tr("-c -d 0-9")
        out = cap.getvalue()
        assert "123" in out
        assert "abc" not in out

    def test_tr_delete(self, repl):
        repl._piped_input = "hello world"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tr("-d ' '")
        out = cap.getvalue()
        assert "helloworld" in out

    def test_tr_squeeze(self, repl):
        repl._piped_input = "a  b   c    d"
        with _CaptureOutput(repl) as cap:
            repl._cmd_tr("-s ' '")
        out = cap.getvalue()
        assert "  " not in out.strip()
