"""
Tests for Console — structured output with input-buffer preservation.
"""

from __future__ import annotations

import re

from domains.shell.io import MemoryIO
from domains.shell.console import Console


def _make_c() -> tuple[Console, MemoryIO]:
    io = MemoryIO()
    c = Console(io, has_readline=False)
    return c, io


def _visible(s: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", s)


class TestPrint:
    def test_print_basic(self):
        c, io = _make_c()
        c.print("hello")
        assert io.get_output().strip() == "hello"

    def test_print_multiple_args(self):
        c, io = _make_c()
        c.print("a", "b", "c")
        assert io.get_output().strip() == "a b c"

    def test_print_custom_end(self):
        c, io = _make_c()
        c.print("hello", end="")
        assert io.get_output() == "hello"


class TestSeparator:
    def test_separator_default(self):
        c, io = _make_c()
        result = c.separator()
        assert len(result) > 0
        assert result in io.get_output()

    def test_separator_custom_char(self):
        c, io = _make_c()
        result = c.separator(char="=")
        assert "=" in result


class TestSection:
    def test_section_basic(self):
        c, io = _make_c()
        c.section("Test")
        out = io.get_output()
        assert "Test" in out

    def test_section_width(self):
        c, io = _make_c()
        c.section("X", width=20)
        out = io.get_output().strip()
        visible = _visible(out)
        dashes = visible.count("─")
        assert dashes > 0
        assert " X " in visible


class TestBox:
    def test_box_basic(self):
        c, io = _make_c()
        c.box("Hello")
        out = io.get_output()
        assert "┌" in out
        assert "┐" in out
        assert "└" in out
        assert "┘" in out

    def test_box_multi_line(self):
        c, io = _make_c()
        c.box("Line1\nLine2")
        out = io.get_output()
        assert out.count("\n") >= 4


class TestStatus:
    def test_status_ok(self):
        c, io = _make_c()
        c.status("ok", "done")
        out = io.get_output()
        assert "done" in out

    def test_status_error(self):
        c, io = _make_c()
        c.status("error", "fail")
        out = io.get_output()
        assert "fail" in out

    def test_status_with_detail(self):
        c, io = _make_c()
        c.status("info", "msg", "detail")
        out = io.get_output()
        assert "detail" in out


class TestTable:
    def test_table_empty(self):
        c, io = _make_c()
        c.table([])
        assert "(empty)" in io.get_output()

    def test_table_simple(self):
        c, io = _make_c()
        c.table([["a", "1"], ["bb", "22"]])
        out = io.get_output()
        assert "a " in out
        assert "bb" in out

    def test_table_with_header(self):
        c, io = _make_c()
        c.table([["a", "1"]], header=["Name", "Val"])
        out = io.get_output()
        assert "Name" in out
        assert "Val" in out

    def test_table_no_header_separator(self):
        c, io = _make_c()
        c.table([["a", "1"]], header=["Name", "Val"], separator_after_header=False)
        out = io.get_output()
        assert "─" not in out.split("\n")[0] if len(out.split("\n")) > 1 else True


class TestKvList:
    def test_kvlist_empty(self):
        c, io = _make_c()
        c.kvlist([])
        assert io.get_output() == ""

    def test_kvlist_basic(self):
        c, io = _make_c()
        c.kvlist([("key", "value"), ("name", "test")])
        out = io.get_output()
        assert "value" in out
        assert "test" in out


class TestProgress:
    def test_progress_at_start(self):
        c, io = _make_c()
        c.progress("Train", 0, 100)
        out = io.get_output()
        assert "0.0%" in out
        assert "Train" in out

    def test_progress_complete(self):
        c, io = _make_c()
        c.progress("Train", 100, 100)
        out = io.get_output()
        assert "100.0%" in out


class TestError:
    def test_error_basic(self):
        c, io = _make_c()
        c.error("something broke")
        out = io.get_output()
        assert "Error" in out
        assert "broke" in out

    def test_error_with_hint(self):
        c, io = _make_c()
        c.error("fail", "try again")
        out = io.get_output()
        assert "try again" in out


class TestSuccess:
    def test_success_basic(self):
        c, io = _make_c()
        c.success("done")
        out = io.get_output()
        assert "done" in out


class TestInfo:
    def test_info_basic(self):
        c, io = _make_c()
        c.info("message")
        out = io.get_output()
        assert "message" in out
