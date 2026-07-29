"""
Tests for Fmt — formatted terminal output.
"""

from __future__ import annotations

import re

from domains.shell.io import MemoryIO
from domains.shell.fmt import Fmt


def _make() -> tuple[Fmt, MemoryIO]:
    io = MemoryIO()
    return Fmt(io, has_readline=False), io


class TestPrintln:
    def test_basic(self):
        f, io = _make()
        f.println("hello")
        assert io.get_output().strip() == "hello"

    def test_multiple_args(self):
        f, io = _make()
        f.println("a", "b", "c")
        assert io.get_output().strip() == "a b c"

    def test_custom_end(self):
        f, io = _make()
        f.println("hello", end="")
        assert io.get_output() == "hello"


class TestSep:
    def test_default(self):
        f, io = _make()
        result = f.sep()
        assert len(result) > 0
        assert result in io.get_output()

    def test_custom_char(self):
        f, io = _make()
        result = f.sep(char="=")
        assert "=" in result


class TestSection:
    def test_basic(self):
        f, io = _make()
        f.section("Test")
        assert "Test" in io.get_output()

    def test_width(self):
        f, io = _make()
        f.section("X", width=20)
        visible = re.sub(r"\033\[[0-9;]*m", "", io.get_output().strip())
        assert " X " in visible
        assert visible.count("─") > 0


class TestBox:
    def test_basic(self):
        f, io = _make()
        f.box("Hello")
        out = io.get_output()
        assert "┌" in out and "┐" in out and "└" in out and "┘" in out

    def test_multi_line(self):
        f, io = _make()
        f.box("Line1\nLine2")
        assert io.get_output().count("\n") >= 4


class TestStatus:
    def test_ok(self):
        f, io = _make()
        f.status("ok", "done")
        assert "done" in io.get_output()

    def test_error(self):
        f, io = _make()
        f.status("error", "fail")
        assert "fail" in io.get_output()

    def test_with_detail(self):
        f, io = _make()
        f.status("info", "msg", "detail")
        assert "detail" in io.get_output()


class TestTable:
    def test_empty(self):
        f, io = _make()
        f.table([])
        assert "(empty)" in io.get_output()

    def test_simple(self):
        f, io = _make()
        f.table([["a", "1"], ["bb", "22"]])
        out = io.get_output()
        assert "a " in out and "bb" in out

    def test_with_header(self):
        f, io = _make()
        f.table([["a", "1"]], header=["Name", "Val"])
        assert "Name" in io.get_output() and "Val" in io.get_output()

    def test_no_header_separator(self):
        f, io = _make()
        f.table([["a", "1"]], header=["Name", "Val"], sep_header=False)
        lines = io.get_output().strip().split("\n")
        if len(lines) > 1:
            assert "─" not in lines[1]


class TestKv:
    def test_empty(self):
        f, io = _make()
        f.kv([])
        assert io.get_output() == ""

    def test_basic(self):
        f, io = _make()
        f.kv([("key", "value"), ("name", "test")])
        out = io.get_output()
        assert "value" in out and "test" in out


class TestProgress:
    def test_start(self):
        f, io = _make()
        f.progress("Train", 0, 100)
        assert "0.0%" in io.get_output() and "Train" in io.get_output()

    def test_complete(self):
        f, io = _make()
        f.progress("Train", 100, 100)
        assert "100.0%" in io.get_output()


class TestErr:
    def test_basic(self):
        f, io = _make()
        f.err("something broke")
        assert "Error" in io.get_output() and "broke" in io.get_output()

    def test_with_hint(self):
        f, io = _make()
        f.err("fail", "try again")
        assert "try again" in io.get_output()


class TestOk:
    def test_basic(self):
        f, io = _make()
        f.ok("done")
        assert "done" in io.get_output()


class TestInfo:
    def test_basic(self):
        f, io = _make()
        f.info("message")
        assert "message" in io.get_output()
