"""Tests for domains.shell.cmds.linux — LinuxCommandsMixin (static helpers + key commands)."""

from __future__ import annotations

import io
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from domains.shell.cmds.linux import LinuxCommandsMixin


# ── Host stub ─────────────────────────────────────────────────────────────────

class FakeHost(LinuxCommandsMixin):
    """Minimal host that provides the attributes LinuxCommandsMixin methods access."""

    def __init__(self):
        self._output = []
        self._last_exit_code = 0
        self._piped_input = None
        self._env = {"HOME": "/home/test"}
        self.os = MagicMock()

    def _print(self, text="", end="\n"):
        self._output.append(text + end if end else text)

    def _get_output(self):
        return "".join(self._output)


@pytest.fixture
def host():
    return FakeHost()


# ══════════════════════════════════════════════════════════════════════════════
# Static helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatSize:
    def test_bytes_no_human(self):
        assert LinuxCommandsMixin._format_size(12345) == "   12345"

    def test_bytes_human(self):
        assert LinuxCommandsMixin._format_size(500, human=True) == "500.0B"

    def test_kilobytes(self):
        assert LinuxCommandsMixin._format_size(2048, human=True) == " 2.0K"

    def test_megabytes(self):
        assert LinuxCommandsMixin._format_size(5 * 1048576, human=True) == " 5.0M"

    def test_gigabytes(self):
        assert LinuxCommandsMixin._format_size(3 * 1073741824, human=True) == " 3.0G"

    def test_terabytes(self):
        assert LinuxCommandsMixin._format_size(2 * 1099511627776, human=True) == " 2.0T"

    def test_petabytes(self):
        assert LinuxCommandsMixin._format_size(1024**5, human=True) == " 1.0P"

    def test_zero(self):
        assert LinuxCommandsMixin._format_size(0, human=True) == " 0.0B"

    def test_exact_1024(self):
        assert LinuxCommandsMixin._format_size(1024, human=True) == " 1.0K"


class TestFmtError:
    def test_file_not_found(self):
        e = FileNotFoundError("missing.txt")
        result = LinuxCommandsMixin._fmt_error(e, "cat")
        assert "not found" in result.lower()
        assert "cat" in result

    def test_permission_error(self):
        e = PermissionError()
        result = LinuxCommandsMixin._fmt_error(e, "ls")
        assert "permission denied" in result.lower()

    def test_os_error(self):
        e = OSError("disk full")
        result = LinuxCommandsMixin._fmt_error(e, "write")
        assert "disk full" in result

    def test_generic_error(self):
        e = ValueError("bad value")
        result = LinuxCommandsMixin._fmt_error(e, "test")
        assert "ValueError" in result
        assert "bad value" in result

    def test_no_cmd_prefix(self):
        e = RuntimeError("oops")
        result = LinuxCommandsMixin._fmt_error(e)
        assert "RuntimeError" in result


# ══════════════════════════════════════════════════════════════════════════════
# Commands — pwd, echo, cat, head, tail, sort
# ══════════════════════════════════════════════════════════════════════════════

class TestCmdPwd:
    def test_pwd(self, host):
        host._cmd_pwd()
        assert host._last_exit_code == 0
        assert os.getcwd() in host._get_output()


class TestCmdEcho:
    def test_echo_text(self, host):
        host._cmd_echo("hello world")
        assert "hello world" in host._get_output()
        assert host._last_exit_code == 0

    def test_echo_empty(self, host):
        host._cmd_echo("")
        assert host._last_exit_code == 0


class TestCmdCat:
    def test_cat_no_args_no_pipe(self, host):
        host._cmd_cat("")
        assert host._last_exit_code == 1
        assert "Usage" in host._get_output()

    def test_cat_with_piped_input(self, host):
        host._piped_input = "piped data"
        host._cmd_cat("")
        assert "piped data" in host._get_output()
        assert host._last_exit_code == 0

    def test_cat_real_file(self, host, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("file content here")
        host._cmd_cat(str(f))
        assert "file content here" in host._get_output()
        assert host._last_exit_code == 0

    def test_cat_missing_file(self, host, tmp_path):
        host._cmd_cat(str(tmp_path / "nope.txt"))
        assert host._last_exit_code == 1
        assert "No such file" in host._get_output()

    def test_cat_directory(self, host, tmp_path):
        host._cmd_cat(str(tmp_path))
        assert host._last_exit_code == 1
        assert "Is a directory" in host._get_output()


class TestCmdHead:
    def test_head_no_args_no_pipe(self, host):
        host._cmd_head("")
        assert host._last_exit_code == 1

    def test_head_piped(self, host):
        lines = "\n".join(f"line {i}" for i in range(20))
        host._piped_input = lines
        host._cmd_head("")
        output = host._get_output()
        assert "line 0" in output
        assert "line 9" in output
        assert "line 10" not in output

    def test_head_n5(self, host, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("\n".join(f"line{i}" for i in range(10)))
        host._cmd_head(f"-n 5 {f}")
        output = host._get_output()
        assert "line0" in output
        assert "line4" in output
        assert "line5" not in output

    def test_head_c_mode(self, host, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("abcdefghij")
        host._cmd_head(f"-c 3 {f}")
        assert "abc" in host._get_output()

    def test_head_missing_file(self, host):
        host._cmd_head("/nonexistent/file")
        assert host._last_exit_code == 1


class TestCmdTail:
    def test_tail_no_args_no_pipe(self, host):
        host._cmd_tail("")
        assert host._last_exit_code == 1

    def test_tail_piped(self, host):
        lines = "\n".join(f"line {i}" for i in range(20))
        host._piped_input = lines
        host._cmd_tail("")
        output = host._get_output()
        assert "line 19" in output
        assert "line 0" not in output

    def test_tail_n5(self, host, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("\n".join(f"line{i}" for i in range(10)))
        host._cmd_tail(f"-n 5 {f}")
        output = host._get_output()
        assert "line5" in output
        assert "line9" in output
        assert "line0" not in output


class TestCmdSort:
    def test_sort_piped(self, host):
        host._piped_input = "cherry\napple\nbanana"
        host._cmd_sort("")
        output = host._get_output()
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        assert lines == ["apple", "banana", "cherry"]

    def test_sort_reverse(self, host):
        host._piped_input = "cherry\napple\nbanana"
        host._cmd_sort("-r")
        output = host._get_output()
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        assert lines == ["cherry", "banana", "apple"]

    def test_sort_numeric(self, host):
        host._piped_input = "10\n2\n1\n20"
        host._cmd_sort("-n")
        output = host._get_output()
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        assert lines == ["1", "2", "10", "20"]

    def test_sort_unique(self, host):
        host._piped_input = "a\nb\na\nc\nb"
        host._cmd_sort("-u")
        output = host._get_output()
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        assert lines == ["a", "b", "c"]

    def test_sort_no_input(self, host):
        host._cmd_sort("")
        assert host._last_exit_code == 1

    def test_sort_file(self, host, tmp_path):
        f = tmp_path / "sort.txt"
        f.write_text("c\na\nb")
        host._cmd_sort(str(f))
        output = host._get_output()
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        assert lines == ["a", "b", "c"]

    def test_sort_field_sep(self, host):
        host._piped_input = "3:foo\n1:bar\n2:baz"
        host._cmd_sort("-t : -k 1 -n")
        output = host._get_output()
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        assert "1:bar" in lines[0]


# ══════════════════════════════════════════════════════════════════════════════
# Commands — mkdir, rm, touch, cp, mv
# ══════════════════════════════════════════════════════════════════════════════

class TestCmdMkdir:
    def test_mkdir_no_args(self, host):
        host._cmd_mkdir("")
        assert host._last_exit_code == 1
        assert "Usage" in host._get_output()

    def test_mkdir_simple(self, host, tmp_path):
        target = str(tmp_path / "newdir")
        host._cmd_mkdir(target)
        assert os.path.isdir(target)
        assert host._last_exit_code == 0

    def test_mkdir_verbose(self, host, tmp_path):
        target = str(tmp_path / "vdir")
        host._cmd_mkdir(f"-v {target}")
        assert "created" in host._get_output().lower()

    def test_mkdir_already_exists(self, host, tmp_path):
        target = str(tmp_path / "existing")
        os.makedirs(target)
        host._cmd_mkdir(target)
        assert host._last_exit_code == 1
        assert "File exists" in host._get_output()

    def test_mkdir_parents(self, host, tmp_path):
        target = str(tmp_path / "a" / "b" / "c")
        host._cmd_mkdir(f"-p {target}")
        assert os.path.isdir(target)


class TestCmdTouch:
    def test_touch_creates_file(self, host, tmp_path):
        target = str(tmp_path / "new.txt")
        host._cmd_touch(target)
        assert os.path.isfile(target)
        assert host._last_exit_code == 0

    def test_touch_existing_file(self, host, tmp_path):
        f = tmp_path / "exists.txt"
        f.write_text("old")
        mtime = f.stat().st_mtime
        host._cmd_touch(str(f))
        assert f.read_text() == "old"


class TestCmdRm:
    def test_rm_no_args(self, host):
        host._cmd_rm("")
        assert host._last_exit_code == 1

    def test_rm_file(self, host, tmp_path):
        f = tmp_path / "deleteme.txt"
        f.write_text("bye")
        host._cmd_rm(str(f))
        assert not f.exists()
        assert host._last_exit_code == 0

    def test_rm_directory_fails(self, host, tmp_path):
        d = tmp_path / "dir"
        d.mkdir()
        host._cmd_rm(str(d))
        assert d.exists()
        assert host._last_exit_code == 1

    def test_rm_recursive(self, host, tmp_path):
        d = tmp_path / "tree"
        d.mkdir()
        (d / "file.txt").write_text("x")
        host._cmd_rm(f"-r {d}")
        assert not d.exists()

    def test_rm_missing_file(self, host, tmp_path):
        host._cmd_rm(str(tmp_path / "nope.txt"))
        assert host._last_exit_code == 1


class TestCmdCp:
    def test_cp_file(self, host, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("content")
        dst = tmp_path / "dst.txt"
        host._cmd_cp(f"{src} {dst}")
        assert dst.read_text() == "content"
        assert host._last_exit_code == 0

    def test_cp_missing_source(self, host, tmp_path):
        host._cmd_cp(f"{tmp_path / 'nope.txt'} {tmp_path / 'dst.txt'}")
        assert host._last_exit_code == 1


class TestCmdMv:
    def test_mv_file(self, host, tmp_path):
        src = tmp_path / "old.txt"
        src.write_text("data")
        dst = tmp_path / "new.txt"
        host._cmd_mv(f"{src} {dst}")
        assert dst.read_text() == "data"
        assert not src.exists()
        assert host._last_exit_code == 0


# ══════════════════════════════════════════════════════════════════════════════
# Commands — grep
# ══════════════════════════════════════════════════════════════════════════════

class TestCmdGrep:
    def test_grep_no_args(self, host):
        host._cmd_grep("")
        assert host._last_exit_code == 1
        assert "Usage" in host._get_output()

    def test_grep_piped(self, host):
        host._piped_input = "hello world\nfoo bar\nhello again"
        host._cmd_grep("hello")
        output = host._get_output()
        assert "hello world" in output
        assert "hello again" in output
        assert "foo bar" not in output

    def test_grep_ignore_case(self, host):
        host._piped_input = "Hello\nWORLD\nhello"
        host._cmd_grep("-i hello")
        output = host._get_output()
        assert "Hello" in output
        assert "hello" in output

    def test_grep_invert(self, host):
        host._piped_input = "apple\nbanana\ncherry"
        host._cmd_grep("-v banana")
        output = host._get_output()
        assert "apple" in output
        assert "cherry" in output
        assert "banana" not in output

    def test_grep_count(self, host):
        host._piped_input = "a\nb\na\nc\na"
        host._cmd_grep("-c a")
        assert "3" in host._get_output()

    def test_grep_line_numbers(self, host):
        host._piped_input = "aaa\nbbb\naaa"
        host._cmd_grep("-n aaa")
        output = host._get_output()
        assert "1:" in output
        assert "3:" in output

    def test_grep_file(self, host, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("line one\nline two\nline three")
        host._cmd_grep(f"one {f}")
        assert "line one" in host._get_output()

    def test_grep_max_count(self, host):
        host._piped_input = "a\na\na\na"
        host._cmd_grep("-m 2 a")
        output = host._get_output()
        count = output.count("a")
        assert count <= 3

    def test_grep_word_boundary(self, host):
        host._piped_input = "cat\ncategory\ndog"
        host._cmd_grep("-w cat")
        output = host._get_output()
        assert "cat" in output
        assert "category" not in output


# ══════════════════════════════════════════════════════════════════════════════
# Commands — wc
# ══════════════════════════════════════════════════════════════════════════════

class TestCmdWc:
    def test_wc_piped(self, host):
        host._piped_input = "line1\nline2\nline3"
        host._cmd_wc("")
        output = host._get_output()
        assert "3" in output

    def test_wc_no_args(self, host):
        host._cmd_wc("")
        assert host._last_exit_code == 1

    def test_wc_file(self, host, tmp_path):
        f = tmp_path / "wc.txt"
        f.write_text("a\nb\nc\nd")
        host._cmd_wc(str(f))
        output = host._get_output()
        assert "4" in output


# ══════════════════════════════════════════════════════════════════════════════
# Commands — find
# ══════════════════════════════════════════════════════════════════════════════

class TestCmdFind:
    def test_find_no_args(self, host):
        host._cmd_find("")
        assert host._last_exit_code == 1

    def test_find_in_dir(self, host, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.txt").write_text("c")
        host._cmd_find(f"-type f {tmp_path}")
        output = host._get_output()
        assert "a.txt" in output
        assert "b.txt" in output

    def test_find_by_name(self, host, tmp_path):
        (tmp_path / "target.py").write_text("x")
        (tmp_path / "other.txt").write_text("y")
        host._cmd_find(f"-name *.py -type f {tmp_path}")
        output = host._get_output()
        assert "target.py" in output
        assert "other.txt" not in output


# ══════════════════════════════════════════════════════════════════════════════
# Commands — ls (real filesystem)
# ══════════════════════════════════════════════════════════════════════════════

class TestCmdLs:
    def test_ls_empty_dir(self, host, tmp_path):
        host.os.vfs = None
        host._cmd_ls(str(tmp_path))
        assert host._last_exit_code == 0

    def test_ls_with_files(self, host, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        host.os.vfs = None
        host._cmd_ls(str(tmp_path))
        output = host._get_output()
        assert "a.txt" in output
        assert "b.txt" in output

    def test_ls_one_per_line(self, host, tmp_path):
        (tmp_path / "x.txt").write_text("x")
        host.os.vfs = None
        host._cmd_ls(f"-1 {tmp_path}")
        assert "x.txt" in host._get_output()

    def test_ls_long_format(self, host, tmp_path):
        (tmp_path / "f.txt").write_text("content")
        host.os.vfs = None
        host._cmd_ls(f"-l {tmp_path}")
        output = host._get_output()
        assert "f.txt" in output
        assert "7" in output

    def test_ls_no_such_dir(self, host, tmp_path):
        host.os.vfs = None
        host._cmd_ls(str(tmp_path / "nonexistent"))
        assert host._last_exit_code == 1


# ══════════════════════════════════════════════════════════════════════════════
# Commands — chmod, stat, du, ln
# ══════════════════════════════════════════════════════════════════════════════

class TestCmdChmod:
    def test_chmod_file(self, host, tmp_path):
        f = tmp_path / "perms.txt"
        f.write_text("x")
        host._cmd_chmod(f"755 {f}")
        assert host._last_exit_code == 0

    def test_chmod_no_args(self, host):
        host._cmd_chmod("")
        assert host._last_exit_code == 1


class TestCmdLn:
    def test_symlink(self, host, tmp_path):
        target = tmp_path / "target.txt"
        target.write_text("link me")
        link = tmp_path / "link.txt"
        host._cmd_ln(f"-s {target} {link}")
        assert link.is_symlink()
        assert host._last_exit_code == 0

    def test_ln_no_args(self, host):
        host._cmd_ln("")
        assert host._last_exit_code == 1


class TestCmdDu:
    def test_du_file(self, host, tmp_path):
        f = tmp_path / "du.txt"
        f.write_text("x" * 100)
        host._cmd_du(str(f))
        assert host._last_exit_code == 0
        assert "100" in host._get_output() or "du" in host._get_output().lower()


class TestCmdStat:
    def test_stat_file(self, host, tmp_path):
        f = tmp_path / "stat.txt"
        f.write_text("data")
        host._cmd_stat(str(f))
        assert host._last_exit_code == 0
        assert "stat.txt" in host._get_output()

    def test_stat_missing(self, host, tmp_path):
        host._cmd_stat(str(tmp_path / "nope"))
        assert host._last_exit_code == 1
