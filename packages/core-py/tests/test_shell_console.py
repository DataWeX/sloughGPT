"""Tests for domains.shell.console — Console output, TUI-aware methods."""

import pytest
from domains.shell.console import Console, _TuiSpinner, _Spinner, _Live
from domains.shell.io import MemoryIO


def _make_console(tui: bool = False) -> tuple[Console, MemoryIO]:
    """Create a Console with MemoryIO. If tui=True, simulate TUI mode."""
    io = MemoryIO()
    c = Console(io)
    if tui:
        c._tui_repl = object()  # non-None sentinel
    return c, io


class TestWrite:
    def test_write_default(self):
        c, io = _make_console()
        c.write("hello")
        assert io.get_output() == "hello\n"

    def test_write_no_newline(self):
        c, io = _make_console()
        c.write("hello", end="")
        assert io.get_output() == "hello"

    def test_write_empty(self):
        c, io = _make_console()
        c.write("")
        assert io.get_output() == "\n"


class TestPrint:
    def test_print_default(self):
        c, io = _make_console()
        c.print("a", "b")
        assert io.get_output() == "a b\n"

    def test_print_no_newline(self):
        c, io = _make_console()
        c.print("x", end="")
        assert io.get_output() == "x"


class TestRule:
    def test_rule_no_label(self):
        c, io = _make_console()
        c.rule(width=20)
        output = io.get_output()
        assert "\u2500" in output  # horizontal line char
        assert len(output.strip()) >= 20


class TestStatus:
    def test_status_ok(self):
        c, io = _make_console()
        c.status("ok", "done")
        output = io.get_output()
        assert "\u2713" in output  # checkmark
        assert "done" in output

    def test_status_error(self):
        c, io = _make_console()
        c.status("error", "failed")
        output = io.get_output()
        assert "\u2717" in output  # cross
        assert "failed" in output


class TestPanel:
    def test_panel_basic(self):
        c, io = _make_console()
        c.panel("hello", width=40)
        output = io.get_output()
        assert "\u250c" in output  # top-left corner
        assert "\u2514" in output  # bottom-left corner
        assert "hello" in output

    def test_panel_with_title(self):
        c, io = _make_console()
        c.panel("text", title="Title", width=40)
        output = io.get_output()
        assert "Title" in output


class TestProgress:
    def test_progress_cli_mode(self):
        c, io = _make_console()
        c.progress("dl", 50, 100)
        output = io.get_output()
        assert "dl" in output
        assert "50.0%" in output

    def test_progress_tui_mode(self):
        c, io = _make_console(tui=True)
        c.progress("dl", 50, 100)
        output = io.get_output()
        assert "dl" in output
        assert "50.0%" in output
        # TUI mode: no \r in output
        assert "\r" not in output

    def test_progress_complete(self):
        c, io = _make_console()
        c.progress("dl", 100, 100)
        output = io.get_output()
        assert "100.0%" in output


class TestDownloadBar:
    def test_download_bar_cli(self):
        c, io = _make_console()
        c.download_bar("file", 50, 100, bytes_done=500, bytes_total=1000)
        output = io.get_output()
        assert "file" in output
        assert "50.0%" in output

    def test_download_bar_tui(self):
        c, io = _make_console(tui=True)
        c.download_bar("file", 50, 100, bytes_done=500, bytes_total=1000)
        output = io.get_output()
        assert "file" in output
        assert "\r" not in output


class TestSpinner:
    def test_spinner_cli_returns_spinner(self):
        c, _ = _make_console()
        s = c.spinner("loading")
        assert isinstance(s, _Spinner)

    def test_spinner_tui_returns_tui_spinner(self):
        c, _ = _make_console(tui=True)
        s = c.spinner("loading")
        assert isinstance(s, _TuiSpinner)

    def test_tui_spinner_writes_static_line(self):
        c, io = _make_console(tui=True)
        with c.spinner("working") as s:
            pass
        output = io.get_output()
        assert "working" in output

    def test_tui_spinner_ok(self):
        c, io = _make_console(tui=True)
        with c.spinner("task") as s:
            pass
        s.ok("done")
        output = io.get_output()
        assert "done" in output

    def test_tui_spinner_fail(self):
        c, io = _make_console(tui=True)
        with c.spinner("task") as s:
            pass
        s.fail("error")
        output = io.get_output()
        assert "error" in output


class TestClear:
    def test_clear_cli(self):
        c, io = _make_console()
        c.clear()
        output = io.get_output()
        assert "\033[2J\033[H" in output

    def test_clear_tui(self):
        c, io = _make_console(tui=True)
        c.clear()
        output = io.get_output()
        assert "\u2500" in output  # separator line
        assert "\033" not in output  # no ANSI


class TestCursorControl:
    def test_hide_cursor_cli(self):
        c, io = _make_console()
        c.hide_cursor()
        assert "\033[?25l" in io.get_output()

    def test_show_cursor_cli(self):
        c, io = _make_console()
        c.show_cursor()
        assert "\033[?25h" in io.get_output()

    def test_hide_cursor_tui_noop(self):
        c, io = _make_console(tui=True)
        c.hide_cursor()
        assert io.get_output() == ""  # no output

    def test_show_cursor_tui_noop(self):
        c, io = _make_console(tui=True)
        c.show_cursor()
        assert io.get_output() == ""  # no output


class TestPaginate:
    def test_paginate_tui_writes_all(self):
        c, io = _make_console(tui=True)
        c.paginate(["line1", "line2", "line3"])
        output = io.get_output()
        assert "line1" in output
        assert "line2" in output
        assert "line3" in output

    def test_paginate_cli_writes_all(self):
        c, io = _make_console()
        c.paginate(["a", "b"])
        output = io.get_output()
        assert "a" in output
        assert "b" in output


class TestLive:
    def test_live_tui_writes_separator(self):
        c, io = _make_console(tui=True)
        with _Live(c) as live:
            live.update("v1")
            live.update("v2")
        output = io.get_output()
        assert "v1" in output
        assert "v2" in output
        assert "\u2500" in output  # separator between updates

    def test_live_cli_overwrites(self):
        c, io = _make_console()
        with _Live(c) as live:
            live.update("first")
            live.update("second")
        output = io.get_output()
        assert "first" in output
        assert "second" in output


class TestBlockRecording:
    def test_emit_records_block(self):
        c, _ = _make_console()
        c.write("test")
        blocks = c.get_blocks()
        assert len(blocks) >= 1
        assert blocks[-1]["type"] == "write"

    def test_clear_blocks(self):
        c, _ = _make_console()
        c.write("x")
        c.clear_blocks()
        assert c.get_blocks() == []

    def test_last_block(self):
        c, _ = _make_console()
        assert c.last_block() is None
        c.write("a")
        c.write("b")
        assert c.last_block()["type"] == "write"
        assert c.last_block()["data"]["text"] == "b"
