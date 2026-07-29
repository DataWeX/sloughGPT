"""
Tests for Tput — structured output with input-buffer preservation.
"""

from __future__ import annotations

from domains.shell.io import MemoryIO
from domains.shell.tput import Tput


def _make_tput() -> tuple[Tput, MemoryIO]:
    io = MemoryIO()
    tput = Tput(io, has_readline=False)
    return tput, io


class TestPrint:
    def test_print_basic(self):
        tput, io = _make_tput()
        tput.print("hello")
        assert io.get_output().strip() == "hello"

    def test_print_multiple_args(self):
        tput, io = _make_tput()
        tput.print("a", "b", "c")
        assert io.get_output().strip() == "a b c"

    def test_print_custom_end(self):
        tput, io = _make_tput()
        tput.print("hello", end="")
        assert io.get_output() == "hello"


class TestSeparator:
    def test_separator_default(self):
        tput, io = _make_tput()
        result = tput.separator()
        assert len(result) > 0
        assert result in io.get_output()

    def test_separator_custom_char(self):
        tput, io = _make_tput()
        result = tput.separator(char="=")
        assert "=" in result


class TestSection:
    def test_section_basic(self):
        tput, io = _make_tput()
        tput.section("Test")
        out = io.get_output()
        assert "Test" in out

    def test_section_width(self):
        tput, io = _make_tput()
        tput.section("X", width=20)
        out = io.get_output().strip()
        import re
        visible = re.sub(r"\033\[[0-9;]*m", "", out)
        dashes = visible.count("─")
        assert dashes > 0
        assert " X " in visible


class TestBox:
    def test_box_basic(self):
        tput, io = _make_tput()
        tput.box("Hello")
        out = io.get_output()
        assert "┌" in out
        assert "┐" in out
        assert "└" in out
        assert "┘" in out

    def test_box_multi_line(self):
        tput, io = _make_tput()
        tput.box("Line1\nLine2")
        out = io.get_output()
        assert out.count("\n") >= 4


class TestStatus:
    def test_status_ok(self):
        tput, io = _make_tput()
        tput.status("ok", "done")
        out = io.get_output()
        assert "done" in out

    def test_status_error(self):
        tput, io = _make_tput()
        tput.status("error", "fail")
        out = io.get_output()
        assert "fail" in out

    def test_status_with_detail(self):
        tput, io = _make_tput()
        tput.status("info", "msg", "detail")
        out = io.get_output()
        assert "detail" in out


class TestTable:
    def test_table_empty(self):
        tput, io = _make_tput()
        tput.table([])
        assert "(empty)" in io.get_output()

    def test_table_simple(self):
        tput, io = _make_tput()
        tput.table([["a", "1"], ["bb", "22"]])
        out = io.get_output()
        assert "a " in out
        assert "bb" in out

    def test_table_with_header(self):
        tput, io = _make_tput()
        tput.table([["a", "1"]], header=["Name", "Val"])
        out = io.get_output()
        assert "Name" in out
        assert "Val" in out

    def test_table_no_header_separator(self):
        tput, io = _make_tput()
        tput.table([["a", "1"]], header=["Name", "Val"], separator_after_header=False)
        out = io.get_output()
        assert "─" not in out.split("\n")[0] if len(out.split("\n")) > 1 else True


class TestKvList:
    def test_kvlist_empty(self):
        tput, io = _make_tput()
        tput.kvlist([])
        assert io.get_output() == ""

    def test_kvlist_basic(self):
        tput, io = _make_tput()
        tput.kvlist([("key", "value"), ("name", "test")])
        out = io.get_output()
        assert "value" in out
        assert "test" in out


class TestProgress:
    def test_progress_at_start(self):
        tput, io = _make_tput()
        tput.progress("Train", 0, 100)
        out = io.get_output()
        assert "0.0%" in out
        assert "Train" in out

    def test_progress_complete(self):
        tput, io = _make_tput()
        tput.progress("Train", 100, 100)
        out = io.get_output()
        assert "100.0%" in out


class TestError:
    def test_error_basic(self):
        tput, io = _make_tput()
        tput.error("something broke")
        out = io.get_output()
        assert "Error" in out
        assert "broke" in out

    def test_error_with_hint(self):
        tput, io = _make_tput()
        tput.error("fail", "try again")
        out = io.get_output()
        assert "try again" in out


class TestSuccess:
    def test_success_basic(self):
        tput, io = _make_tput()
        tput.success("done")
        out = io.get_output()
        assert "done" in out


class TestInfo:
    def test_info_basic(self):
        tput, io = _make_tput()
        tput.info("message")
        out = io.get_output()
        assert "message" in out
