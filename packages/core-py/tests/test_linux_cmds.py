"""Tests for domains.shell.cmds.linux — LinuxCommandsMixin static helpers and key commands."""

from __future__ import annotations

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from domains.shell.cmds.linux import LinuxCommandsMixin


# ── Fake host for mixin ───────────────────────────────────────────────────────

class FakeHost(LinuxCommandsMixin):
    """Minimal host object satisfying LinuxCommandsMixin's self.* access."""

    def __init__(self):
        self._printed = []
        self._last_exit_code = 0
        self._env = {}
        self._piped_input = None
        self.os = MagicMock()
        self.os.vfs = None

    def _print(self, s="", **kwargs):
        self._printed.append(s)


@pytest.fixture
def host(tmp_path):
    h = FakeHost()
    os.chdir(tmp_path)
    return h


# ── Static helpers ────────────────────────────────────────────────────────────

class TestFormatSize:
    def test_exact_bytes(self):
        assert LinuxCommandsMixin._format_size(0) == "       0"
        assert LinuxCommandsMixin._format_size(1) == "       1"
        assert LinuxCommandsMixin._format_size(999) == "     999"

    def test_human_kb(self):
        result = LinuxCommandsMixin._format_size(1024, human=True)
        assert "1.0K" in result

    def test_human_mb(self):
        result = LinuxCommandsMixin._format_size(1048576, human=True)
        assert "1.0M" in result

    def test_human_gb(self):
        result = LinuxCommandsMixin._format_size(1073741824, human=True)
        assert "1.0G" in result

    def test_human_tb(self):
        result = LinuxCommandsMixin._format_size(1099511627776, human=True)
        assert "1.0T" in result

    def test_human_large(self):
        result = LinuxCommandsMixin._format_size(1024 ** 5, human=True)
        assert "1.0P" in result

    def test_human_small(self):
        result = LinuxCommandsMixin._format_size(500, human=True)
        assert "500.0B" in result

    def test_non_human_default(self):
        assert LinuxCommandsMixin._format_size(4096, human=False) == "    4096"


class TestFmtError:
    def test_file_not_found(self):
        e = FileNotFoundError("missing.txt")
        result = LinuxCommandsMixin._fmt_error(e, "cat")
        assert "cat" in result
        assert "not found" in result

    def test_permission_error(self):
        e = PermissionError("denied")
        result = LinuxCommandsMixin._fmt_error(e, "rm")
        assert "permission denied" in result

    def test_os_error(self):
        e = OSError("disk full")
        result = LinuxCommandsMixin._fmt_error(e, "write")
        assert "disk full" in result

    def test_generic_error(self):
        e = ValueError("bad value")
        result = LinuxCommandsMixin._fmt_error(e, "test")
        assert "ValueError" in result
        assert "bad value" in result

    def test_no_cmd(self):
        e = RuntimeError("oops")
        result = LinuxCommandsMixin._fmt_error(e)
        assert "RuntimeError" in result


# ── _cmd_cd ───────────────────────────────────────────────────────────────────

class TestCd:
    def test_cd_home(self, host):
        host._cmd_cd("")
        assert host._last_exit_code == 0

    def test_cd_tilde(self, host):
        host._cmd_cd("~")
        assert host._last_exit_code == 0

    def test_cd_nonexistent(self, host):
        host._cmd_cd("/nonexistent_dir_xyz_123")
        assert host._last_exit_code == 1
        assert any("no such file" in p.lower() for p in host._printed)

    def test_cd_oldpwd(self, host):
        host._env = {"OLDPWD": str(Path.cwd())}
        host._cmd_cd("-")
        assert host._last_exit_code == 0

    def test_cd_not_a_directory(self, host):
        tmp_file = Path.cwd() / "not_a_dir.txt"
        tmp_file.write_text("hello")
        host._cmd_cd(str(tmp_file))
        assert host._last_exit_code == 1


# ── _cmd_pwd ──────────────────────────────────────────────────────────────────

class TestPwd:
    def test_pwd(self, host):
        host._cmd_pwd()
        assert host._last_exit_code == 0
        assert len(host._printed) == 1
        assert os.getcwd() in host._printed[0]


# ── _cmd_echo ─────────────────────────────────────────────────────────────────

class TestEcho:
    def test_echo(self, host):
        host._last_exit_code = 0
        host._cmd_echo("hello world")
        assert host._printed[-1] == "hello world"
        assert host._last_exit_code == 0

    def test_echo_empty(self, host):
        host._last_exit_code = 0
        host._cmd_echo("")
        assert host._last_exit_code == 0


# ── _cmd_mkdir ────────────────────────────────────────────────────────────────

class TestMkdir:
    def test_mkdir_basic(self, host):
        host._last_exit_code = 0
        host._cmd_mkdir("testdir")
        assert Path("testdir").exists()
        assert host._last_exit_code == 0

    def test_mkdir_no_args(self, host):
        host._last_exit_code = 0
        host._cmd_mkdir("")
        assert host._last_exit_code == 1
        assert any("Usage" in p for p in host._printed)

    def test_mkdir_exists(self, host):
        host._last_exit_code = 0
        host._cmd_mkdir("existing")
        host._cmd_mkdir("existing")
        assert host._last_exit_code == 1
        assert any("File exists" in p for p in host._printed)

    def test_mkdir_verbose(self, host):
        host._last_exit_code = 0
        host._cmd_mkdir("-v vdir")
        assert Path("vdir").exists()
        assert any("created" in p for p in host._printed)

    def test_mkdir_parents(self, host):
        host._last_exit_code = 0
        host._cmd_mkdir("-p a/b/c")
        assert Path("a/b/c").exists()


# ── _cmd_touch ────────────────────────────────────────────────────────────────

class TestTouch:
    def test_touch_creates(self, host):
        host._last_exit_code = 0
        host._cmd_touch("newfile.txt")
        assert Path("newfile.txt").exists()
        assert host._last_exit_code == 0

    def test_touch_no_args(self, host):
        host._last_exit_code = 0
        host._cmd_touch("")
        assert host._last_exit_code == 1

    def test_touch_existing(self, host):
        host._last_exit_code = 0
        Path("existing.txt").write_text("content")
        host._last_exit_code = 0
        host._cmd_touch("existing.txt")
        assert Path("existing.txt").exists()
        assert host._last_exit_code == 0


# ── _cmd_cat ──────────────────────────────────────────────────────────────────

class TestCat:
    def test_cat_file(self, host):
        host._last_exit_code = 0
        Path("hello.txt").write_text("hello world")
        host._cmd_cat("hello.txt")
        assert "hello world" in host._printed[-1]
        assert host._last_exit_code == 0

    def test_cat_no_args(self, host):
        host._last_exit_code = 0
        host._piped_input = None
        host._last_exit_code = 0
        host._cmd_cat("")
        assert host._last_exit_code == 1

    def test_cat_piped(self, host):
        host._last_exit_code = 0
        host._piped_input = "piped data\n"
        host._last_exit_code = 0
        host._cmd_cat("")
        assert "piped data" in host._printed[-1]
        assert host._last_exit_code == 0

    def test_cat_not_found(self, host):
        host._last_exit_code = 0
        host._cmd_cat("nonexistent.txt")
        assert host._last_exit_code == 1
        assert any("not found" in p.lower() or "no such" in p.lower() for p in host._printed)

    def test_cat_directory(self, host):
        host._last_exit_code = 0
        Path("dir").mkdir()
        host._cmd_cat("dir")
        assert host._last_exit_code == 1


# ── _cmd_head ─────────────────────────────────────────────────────────────────

class TestHead:
    def test_head_default(self, host):
        lines = "\n".join(f"line {i}" for i in range(20))
        Path("many.txt").write_text(lines)
        host._cmd_head("many.txt")
        assert host._last_exit_code == 0
        output = "\n".join(host._printed)
        for i in range(10):
            assert f"line {i}" in output
        assert "line 10" not in output

    def test_head_n(self, host):
        lines = "\n".join(f"line {i}" for i in range(20))
        Path("many.txt").write_text(lines)
        host._cmd_head("-n 3 many.txt")
        output = "\n".join(host._printed)
        for i in range(3):
            assert f"line {i}" in output
        assert "line 3" not in output

    def test_head_no_args(self, host):
        host._cmd_head("")
        assert host._last_exit_code == 1


# ── _cmd_tail ─────────────────────────────────────────────────────────────────

class TestTail:
    def test_tail_default(self, host):
        lines = "\n".join(f"line {i}" for i in range(20))
        Path("many.txt").write_text(lines)
        host._cmd_tail("many.txt")
        assert host._last_exit_code == 0
        output = "\n".join(host._printed)
        for i in range(10, 20):
            assert f"line {i}" in output
        assert "line 9" not in output

    def test_tail_n(self, host):
        lines = "\n".join(f"line {i}" for i in range(20))
        Path("many.txt").write_text(lines)
        host._cmd_tail("-n 3 many.txt")
        output = "\n".join(host._printed)
        for i in range(17, 20):
            assert f"line {i}" in output

    def test_tail_not_found(self, host):
        host._cmd_tail("nope.txt")
        assert host._last_exit_code == 1


# ── _cmd_wc ───────────────────────────────────────────────────────────────────

class TestWc:
    def test_wc_lines(self, host):
        host._last_exit_code = 0
        Path("three.txt").write_text("a\nb\nc\n")
        host._cmd_wc("three.txt")
        output = host._printed[-1]
        assert "3" in output
        assert "3" in output  # 3 lines

    def test_wc_no_args(self, host):
        host._last_exit_code = 0
        host._cmd_wc("")
        assert host._last_exit_code == 1

    def test_wc_bytes_flag(self, host):
        host._last_exit_code = 0
        Path("data.txt").write_text("hello")
        host._cmd_wc("-c data.txt")
        assert "5" in host._printed[-1]


# ── _cmd_grep ─────────────────────────────────────────────────────────────────

class TestGrep:
    def test_grep_match(self, host):
        host._last_exit_code = 0
        Path("log.txt").write_text("error\nok\nerror again\nok2\n")
        host._cmd_grep("error log.txt")
        assert any("error" in p for p in host._printed)
        assert host._last_exit_code == 0

    def test_grep_no_match(self, host):
        host._last_exit_code = 0
        Path("log.txt").write_text("ok\nok2\n")
        host._cmd_grep("error log.txt")
        assert host._last_exit_code == 1

    def test_grep_no_args(self, host):
        host._last_exit_code = 0
        host._cmd_grep("")
        assert host._last_exit_code == 1

    def test_grep_invert(self, host):
        host._last_exit_code = 0
        Path("log.txt").write_text("error\nok\nerror again\n")
        host._cmd_grep("-v error log.txt")
        output = "\n".join(host._printed)
        assert "ok" in output
        assert "error" not in output

    def test_grep_count(self, host):
        host._last_exit_code = 0
        Path("log.txt").write_text("error\nok\nerror\n")
        host._cmd_grep("-c error log.txt")
        assert "2" in host._printed[-1]


# ── _cmd_sort ─────────────────────────────────────────────────────────────────

class TestSort:
    def test_sort_basic(self, host):
        host._last_exit_code = 0
        Path("nums.txt").write_text("c\na\nb\n")
        host._cmd_sort("nums.txt")
        output = "\n".join(host._printed)
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        assert lines == ["a", "b", "c"]

    def test_sort_reverse(self, host):
        host._last_exit_code = 0
        Path("nums.txt").write_text("c\na\nb\n")
        host._cmd_sort("-r nums.txt")
        output = "\n".join(host._printed)
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        assert lines == ["c", "b", "a"]

    def test_sort_numeric(self, host):
        host._last_exit_code = 0
        Path("nums.txt").write_text("10\n2\n30\n1\n")
        host._cmd_sort("-n nums.txt")
        output = "\n".join(host._printed)
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        assert lines == ["1", "2", "10", "30"]


# ── _cmd_find ─────────────────────────────────────────────────────────────────

class TestFind:
    def test_find_files(self, host):
        Path("a.txt").write_text("x")
        Path("b.txt").write_text("y")
        host._cmd_find(". -name *.txt")
        output = "\n".join(host._printed)
        assert "a.txt" in output
        assert "b.txt" in output

    def test_find_no_match(self, host):
        Path("a.txt").write_text("x")
        host._cmd_find(". -name *.xyz")
        output = "\n".join(host._printed)
        assert "a.txt" not in output

    def test_find_no_args(self, host):
        host._cmd_find("")
        assert host._last_exit_code == 1

    def test_find_no_pattern(self, host):
        host._cmd_find(".")
        assert host._last_exit_code == 1


# ── _cmd_diff ─────────────────────────────────────────────────────────────────

class TestDiff:
    def test_diff_identical(self, host):
        host._last_exit_code = 0
        Path("a.txt").write_text("same\n")
        Path("b.txt").write_text("same\n")
        host._cmd_diff("a.txt b.txt")
        assert host._last_exit_code == 0

    def test_diff_different(self, host):
        host._last_exit_code = 0
        Path("a.txt").write_text("line1\n")
        Path("b.txt").write_text("line2\n")
        host._cmd_diff("a.txt b.txt")
        assert host._last_exit_code == 1

    def test_diff_no_args(self, host):
        host._last_exit_code = 0
        host._cmd_diff("")
        assert host._last_exit_code == 1


# ── _cmd_cp ───────────────────────────────────────────────────────────────────

class TestCp:
    def test_cp_basic(self, host):
        host._last_exit_code = 0
        Path("src.txt").write_text("content")
        host._cmd_cp("src.txt dst.txt")
        assert Path("dst.txt").read_text() == "content"
        assert host._last_exit_code == 0

    def test_cp_no_args(self, host):
        host._last_exit_code = 0
        host._cmd_cp("")
        assert host._last_exit_code == 1

    def test_cp_missing_source(self, host):
        host._last_exit_code = 0
        host._cmd_cp("nonexistent.txt dst.txt")
        assert host._last_exit_code == 1


# ── _cmd_mv ───────────────────────────────────────────────────────────────────

class TestMv:
    def test_mv_basic(self, host):
        host._last_exit_code = 0
        Path("old.txt").write_text("data")
        host._cmd_mv("old.txt new.txt")
        assert Path("new.txt").read_text() == "data"
        assert not Path("old.txt").exists()
        assert host._last_exit_code == 0

    def test_mv_no_args(self, host):
        host._last_exit_code = 0
        host._cmd_mv("")
        assert host._last_exit_code == 1


# ── _cmd_rm ───────────────────────────────────────────────────────────────────

class TestRm:
    def test_rm_file(self, host):
        host._last_exit_code = 0
        Path("delete_me.txt").write_text("bye")
        host._cmd_rm("delete_me.txt")
        assert not Path("delete_me.txt").exists()
        assert host._last_exit_code == 0

    def test_rm_dir_recursive(self, host):
        host._last_exit_code = 0
        Path("mydir/sub").mkdir(parents=True)
        Path("mydir/sub/file.txt").write_text("x")
        host._cmd_rm("-r mydir")
        assert not Path("mydir").exists()

    def test_rm_no_args(self, host):
        host._last_exit_code = 0
        host._cmd_rm("")
        assert host._last_exit_code == 1

    def test_rm_nonexistent(self, host):
        host._last_exit_code = 0
        host._cmd_rm("ghost.txt")
        assert host._last_exit_code == 1


# ── _cmd_chmod ────────────────────────────────────────────────────────────────

class TestChmod:
    def test_chmod_basic(self, host):
        host._last_exit_code = 0
        Path("perms.txt").write_text("x")
        host._cmd_chmod("755 perms.txt")
        assert host._last_exit_code == 0

    def test_chmod_no_args(self, host):
        host._last_exit_code = 0
        host._cmd_chmod("")
        assert host._last_exit_code == 1


# ── _cmd_ln ───────────────────────────────────────────────────────────────────

class TestLn:
    def test_ln_symlink(self, host):
        host._last_exit_code = 0
        Path("target.txt").write_text("data")
        host._cmd_ln("-s target.txt link.txt")
        assert Path("link.txt").is_symlink()
        assert host._last_exit_code == 0

    def test_ln_no_args(self, host):
        host._last_exit_code = 0
        host._cmd_ln("")
        assert host._last_exit_code == 1


# ── _cmd_du ───────────────────────────────────────────────────────────────────

class TestDu:
    def test_du_file(self, host):
        host._last_exit_code = 0
        Path("file.txt").write_text("hello")
        host._cmd_du("file.txt")
        assert host._last_exit_code == 0
        assert len(host._printed) >= 1

    def test_du_human(self, host):
        host._last_exit_code = 0
        Path("big.txt").write_text("x" * 2048)
        host._cmd_du("-h big.txt")
        output = host._printed[-1]
        assert "K" in output or "2" in output


# ── _cmd_stat ─────────────────────────────────────────────────────────────────

class TestStat:
    def test_stat_file(self, host):
        host._last_exit_code = 0
        Path("statme.txt").write_text("content")
        host._cmd_stat("statme.txt")
        assert host._last_exit_code == 0
        output = "\n".join(host._printed)
        assert "statme.txt" in output

    def test_stat_not_found(self, host):
        host._last_exit_code = 0
        host._cmd_stat("ghost.txt")
        assert host._last_exit_code == 1


# ── _cmd_seq ──────────────────────────────────────────────────────────────────

class TestSeq:
    def test_seq_basic(self, host):
        host._last_exit_code = 0
        host._cmd_seq("5")
        output = "\n".join(host._printed)
        for i in range(1, 6):
            assert str(i) in output

    def test_seq_range(self, host):
        host._last_exit_code = 0
        host._cmd_seq("2 5")
        output = "\n".join(host._printed)
        for i in range(2, 6):
            assert str(i) in output

    def test_seq_no_args(self, host):
        host._last_exit_code = 0
        host._cmd_seq("")
        assert host._last_exit_code == 1


# ── _cmd_cut ──────────────────────────────────────────────────────────────────

class TestCut:
    def test_cut_fields(self, host):
        host._last_exit_code = 0
        Path("csv.txt").write_text("a,b,c\nd,e,f\n")
        host._cmd_cut("-f 2 -d , csv.txt")
        output = "\n".join(host._printed)
        assert "b" in output
        assert "e" in output

    def test_cut_no_args(self, host):
        host._last_exit_code = 0
        host._cmd_cut("")
        assert host._last_exit_code == 1


# ── _cmd_tr ───────────────────────────────────────────────────────────────────

class TestTr:
    def test_tr_lowercase(self, host):
        host._last_exit_code = 0
        host._piped_input = "HELLO WORLD"
        host._cmd_tr("a-z A-Z")
        assert "HELLO WORLD" in host._printed[-1].upper()

    def test_tr_no_args(self, host):
        host._last_exit_code = 0
        host._piped_input = "hello"
        host._last_exit_code = 0
        host._cmd_tr("")
        assert host._last_exit_code == 1


# ── _cmd_nl ───────────────────────────────────────────────────────────────────

class TestNl:
    def test_nl_numbers(self, host):
        host._last_exit_code = 0
        Path("lines.txt").write_text("a\nb\nc\n")
        host._cmd_nl("lines.txt")
        output = "\n".join(host._printed)
        assert "1" in output
        assert "a" in output

    def test_nl_no_args(self, host):
        host._last_exit_code = 0
        host._cmd_nl("")
        assert host._last_exit_code == 1


# ── _cmd_uniq ─────────────────────────────────────────────────────────────────

class TestUniq:
    def test_uniq_dedup(self, host):
        host._last_exit_code = 0
        Path("dupes.txt").write_text("a\na\nb\nb\nb\nc\n")
        host._cmd_uniq("dupes.txt")
        output = "\n".join(host._printed)
        assert output.count("a") == 1
        assert output.count("b") == 1
        assert output.count("c") == 1

    def test_uniq_count(self, host):
        host._last_exit_code = 0
        Path("dupes.txt").write_text("a\na\nb\n")
        host._cmd_uniq("-c dupes.txt")
        output = "\n".join(host._printed)
        assert "2" in output


# ── _cmd_tac ──────────────────────────────────────────────────────────────────

class TestTac:
    def test_tac_reverse(self, host):
        host._last_exit_code = 0
        Path("ordered.txt").write_text("1\n2\n3\n")
        host._cmd_tac("ordered.txt")
        output = "\n".join(host._printed)
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        assert lines == ["3", "2", "1"]


# ── _cmd_rev ──────────────────────────────────────────────────────────────────

class TestRev:
    def test_rev_string(self, host):
        host._last_exit_code = 0
        Path("word.txt").write_text("hello\n")
        host._cmd_rev("word.txt")
        assert host._printed[-1] == "olleh"


# ── _cmd_fold ─────────────────────────────────────────────────────────────────

class TestFold:
    def test_fold_width(self, host):
        host._last_exit_code = 0
        Path("long.txt").write_text("abcdefghij\n")
        host._cmd_fold("-w 5 long.txt")
        output = "\n".join(host._printed)
        assert "abcde" in output


# ── _cmd_date ─────────────────────────────────────────────────────────────────

class TestDate:
    def test_date(self, host):
        host._last_exit_code = 0
        host._cmd_date()
        assert len(host._printed) == 1
        assert host._last_exit_code == 0


# ── _cmd_sleep ────────────────────────────────────────────────────────────────

class TestSleep:
    def test_sleep_short(self, host):
        import time
        host._last_exit_code = 0
        start = time.monotonic()
        host._cmd_sleep("0.01")
        elapsed = time.monotonic() - start
        assert elapsed >= 0.009
        assert host._last_exit_code == 0

    def test_sleep_no_args(self, host):
        host._last_exit_code = 0
        host._cmd_sleep("")
        assert host._last_exit_code == 0  # no-op, not an error


# ── _cmd_clear ────────────────────────────────────────────────────────────────

class TestClear:
    def test_clear(self, host):
        host._last_exit_code = 0
        host._cmd_clear()
        assert host._last_exit_code == 0


# ── _cmd_hostname ─────────────────────────────────────────────────────────────

class TestHostname:
    def test_hostname(self, host):
        host._last_exit_code = 0
        host._cmd_hostname()
        assert len(host._printed) == 1
        assert host._last_exit_code == 0


# ── _cmd_nproc ────────────────────────────────────────────────────────────────

class TestNproc:
    def test_nproc(self, host):
        host._last_exit_code = 0
        host._cmd_nproc()
        assert len(host._printed) == 1
        assert host._printed[0].strip().isdigit()
        assert host._last_exit_code == 0


# ── _cmd_uname ────────────────────────────────────────────────────────────────

class TestUname:
    def test_uname(self, host):
        host._last_exit_code = 0
        host._cmd_uname()
        assert len(host._printed) == 1
        assert host._last_exit_code == 0

    def test_uname_s(self, host):
        host._last_exit_code = 0
        host._cmd_uname("-s")
        output = host._printed[-1].lower()
        assert "linux" in output or "darwin" in output


# ── _cmd_id ───────────────────────────────────────────────────────────────────

class TestId:
    def test_id(self, host):
        host._last_exit_code = 0
        host._cmd_id()
        output = host._printed[-1].lower()
        assert "uid" in output
        assert host._last_exit_code == 0


# ── _cmd_logname ──────────────────────────────────────────────────────────────

class TestLogname:
    def test_logname(self, host):
        host._last_exit_code = 0
        host._cmd_logname()
        assert len(host._printed) == 1
        assert host._last_exit_code == 0


# ── _cmd_who ──────────────────────────────────────────────────────────────────

class TestWho:
    def test_who(self, host):
        host._last_exit_code = 0
        host._cmd_who()
        assert host._last_exit_code == 0


# ── _cmd_cal ──────────────────────────────────────────────────────────────────

class TestCal:
    def test_cal(self, host):
        host._last_exit_code = 0
        host._cmd_cal()
        assert host._last_exit_code == 0
        output = "\n".join(host._printed)
        assert "Mon" in output or "Su" in output

    def test_cal_specific_month(self, host):
        host._last_exit_code = 0
        host._cmd_cal("3 2026")
        assert host._last_exit_code == 0


# ── _cmd_yes ──────────────────────────────────────────────────────────────────

class TestYes:
    def test_yes_default(self, host):
        host._cmd_yes("")
        output = "\n".join(host._printed)
        lines = [l for l in output.strip().split("\n") if l.strip()]
        assert len(lines) == 100
        assert all(l == "y" for l in lines)

    def test_yes_custom_string(self, host):
        host._cmd_yes("hello")
        output = "\n".join(host._printed)
        lines = [l for l in output.strip().split("\n") if l.strip()]
        assert len(lines) == 100
        assert all(l == "hello" for l in lines)


# ── _cmd_realpath ─────────────────────────────────────────────────────────────

class TestRealpath:
    def test_realpath(self, host):
        host._last_exit_code = 0
        host._cmd_realpath(".")
        assert os.getcwd() in host._printed[-1]


# ── _cmd_dirname ──────────────────────────────────────────────────────────────

class TestDirname:
    def test_dirname(self, host):
        host._last_exit_code = 0
        host._cmd_dirname("/a/b/c.txt")
        assert host._printed[-1] == "/a/b"

    def test_dirname_no_slash(self, host):
        host._last_exit_code = 0
        host._cmd_dirname("file.txt")
        assert host._printed[-1] == ""


# ── _cmd_basename ─────────────────────────────────────────────────────────────

class TestBasename:
    def test_basename(self, host):
        host._last_exit_code = 0
        host._cmd_basename("/a/b/c.txt")
        assert host._printed[-1] == "c.txt"

    def test_basename_strip_ext(self, host):
        host._last_exit_code = 0
        host._cmd_basename("/a/b/c.txt .txt")
        assert host._printed[-1] == "c"


# ── _cmd_env ──────────────────────────────────────────────────────────────────

class TestEnv:
    def test_env(self, host):
        host._last_exit_code = 0
        host._env = {"HOME": "/home/test", "PATH": "/usr/bin"}
        host._cmd_env()
        output = "\n".join(host._printed)
        assert "HOME=/home/test" in output


# ── _cmd_time ─────────────────────────────────────────────────────────────────

class TestTime:
    def test_time_no_args(self, host):
        host._last_exit_code = 0
        host._cmd_time("")
        assert host._last_exit_code == 1
