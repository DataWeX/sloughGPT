"""
Comprehensive tests for domains.shell.console.

Pure-logic tests for Console formatting, rendering, block recording,
helper classes (_Capture, _Indent, _Live, _Spinner), and utility functions.
No mocks — uses MemoryIO for all output capture.
"""

from __future__ import annotations

import json
import os
import threading
import time
import pytest

os.environ["NO_COLOR"] = "1"

from domains.shell.io import MemoryIO
from domains.shell.console import (
    Console,
    Block,
    _color,
    _human_size,
    _render_inline,
    SPINNER_FRAMES,
    _Capture,
    _Indent,
    _Live,
    _Spinner,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_console(has_readline=False) -> tuple[Console, MemoryIO]:
    io = MemoryIO()
    c = Console(io, has_readline=has_readline)
    return c, io


def _output(io: MemoryIO) -> str:
    return io.get_output()


# =============================================================================
# Utility functions
# =============================================================================

class TestColor:
    def test_color_disabled_returns_plain_text(self, monkeypatch):
        import domains.shell.console as _mod
        monkeypatch.setattr(_mod, "_COLOR_ENABLED", False)
        result = _color("hello", "\033[36m")
        assert result == "hello"

    def test_color_empty_code_returns_text(self):
        result = _color("hello", "")
        assert result == "hello"

    def test_color_none_code_returns_text(self):
        result = _color("hello", None)
        assert result == "hello"


class TestHumanSize:
    def test_bytes(self):
        assert _human_size(0) == "0.0 B"
        assert _human_size(512) == "512.0 B"

    def test_kilobytes(self):
        assert _human_size(1024) == "1.0 KB"
        assert _human_size(1536) == "1.5 KB"

    def test_megabytes(self):
        assert _human_size(1024 * 1024) == "1.0 MB"

    def test_gigabytes(self):
        assert _human_size(1024 ** 3) == "1.0 GB"

    def test_terabytes(self):
        assert _human_size(1024 ** 4) == "1.0 TB"

    def test_petabytes(self):
        assert _human_size(1024 ** 5) == "1.0 PB"

    def test_exact_boundary(self):
        assert _human_size(1023) == "1023.0 B"
        assert _human_size(1024) == "1.0 KB"


class TestRenderInline:
    def test_bold(self):
        result = _render_inline("**bold**")
        assert "bold" in result

    def test_italic(self):
        result = _render_inline("*italic*")
        assert "italic" in result

    def test_code(self):
        result = _render_inline("`code`")
        assert "code" in result

    def test_link(self):
        result = _render_inline("[text](url)")
        assert "text" in result
        assert "url" in result

    def test_mixed(self):
        result = _render_inline("**bold** and `code`")
        assert "bold" in result
        assert "code" in result

    def test_plain_text_unchanged(self):
        result = _render_inline("just plain text")
        assert result == "just plain text"


class TestSpinnerFrames:
    def test_frame_count(self):
        assert len(SPINNER_FRAMES) == 10

    def test_all_braille(self):
        for f in SPINNER_FRAMES:
            assert len(f) == 1


# =============================================================================
# Block dataclass
# =============================================================================

class TestBlock:
    def test_creation(self):
        b = Block(type="test", data={"key": "value"})
        assert b.type == "test"
        assert b.data == {"key": "value"}
        assert b.meta == {}

    def test_with_meta(self):
        b = Block(type="x", data={}, meta={"a": 1})
        assert b.meta == {"a": 1}


# =============================================================================
# Console — Block recording
# =============================================================================

class TestConsoleBlocks:
    def test_emit_records_block(self):
        c, _ = _make_console()
        c._emit("test", {"k": 1})
        blocks = c.get_blocks()
        assert len(blocks) == 1
        assert blocks[0]["type"] == "test"
        assert blocks[0]["data"] == {"k": 1}

    def test_get_blocks_returns_dicts(self):
        c, _ = _make_console()
        c._emit("a", {})
        c._emit("b", {"x": 2})
        blocks = c.get_blocks()
        assert len(blocks) == 2
        assert all(isinstance(b, dict) for b in blocks)

    def test_get_json(self):
        c, _ = _make_console()
        c._emit("t", {"v": 1})
        j = c.get_json()
        parsed = json.loads(j)
        assert len(parsed) == 1
        assert parsed[0]["type"] == "t"

    def test_clear_blocks(self):
        c, _ = _make_console()
        c._emit("a", {})
        c.clear_blocks()
        assert c.get_blocks() == []

    def test_last_block_none_when_empty(self):
        c, _ = _make_console()
        assert c.last_block() is None

    def test_last_block(self):
        c, _ = _make_console()
        c._emit("first", {})
        c._emit("second", {})
        last = c.last_block()
        assert last["type"] == "second"

    def test_meta_passed_through(self):
        c, _ = _make_console()
        c._emit("t", {}, foo="bar")
        b = c.get_blocks()[0]
        assert b["meta"] == {"foo": "bar"}


# =============================================================================
# Console — write / print
# =============================================================================

class TestConsoleWrite:
    def test_write_basic(self):
        c, io = _make_console()
        c.write("hello")
        assert "hello" in _output(io)

    def test_write_end_default_newline(self):
        c, io = _make_console()
        c.write("x")
        out = _output(io)
        assert out.endswith("\n")

    def test_write_end_empty(self):
        c, io = _make_console()
        c.write("x", end="")
        out = _output(io)
        assert "x" in out

    def test_write_empty_text(self):
        c, io = _make_console()
        c.write("")
        assert _output(io) != ""

    def test_write_emits_block(self):
        c, _ = _make_console()
        c.write("data")
        b = c.last_block()
        assert b["type"] == "write"
        assert b["data"]["text"] == "data"


class TestConsolePrint:
    def test_print_single_arg(self):
        c, io = _make_console()
        c.print("hello")
        assert "hello" in _output(io)

    def test_print_multiple_args(self):
        c, io = _make_console()
        c.print("a", "b", "c")
        out = _output(io)
        assert "a b c" in out

    def test_print_end_override(self):
        c, io = _make_console()
        c.print("x", end="")
        assert _output(io).endswith("x")

    def test_print_int(self):
        c, io = _make_console()
        c.print(42)
        assert "42" in _output(io)

    def test_print_emits_block(self):
        c, _ = _make_console()
        c.print("test")
        blocks = c.get_blocks()
        types = [b["type"] for b in blocks]
        assert "print" in types


# =============================================================================
# Console — rule / separator / section
# =============================================================================

class TestConsoleRule:
    def test_rule_no_label(self):
        c, io = _make_console()
        c.rule(width=40)
        out = _output(io)
        assert len(out.strip()) > 0

    def test_rule_with_label(self):
        c, io = _make_console()
        c.rule("Title", width=40)
        assert "Title" in _output(io)

    def test_rule_emits_block(self):
        c, _ = _make_console()
        c.rule(width=40)
        b = c.last_block()
        assert b["type"] == "rule"


class TestConsoleSeparator:
    def test_separator_returns_string(self):
        c, io = _make_console()
        result = c.separator(char="=")
        assert "==" in result
        assert _output(io) != ""

    def test_separator_emits_block(self):
        c, _ = _make_console()
        c.separator()
        b = c.last_block()
        assert b["type"] == "separator"


class TestConsoleSection:
    def test_section_delegates_to_rule(self):
        c, io = _make_console()
        c.section("My Section", width=40)
        assert "My Section" in _output(io)


# =============================================================================
# Console — panel / box
# =============================================================================

class TestConsolePanel:
    def test_panel_basic(self):
        c, io = _make_console()
        c.panel("Hello World", width=40)
        out = _output(io)
        assert "Hello World" in out

    def test_panel_with_title(self):
        c, io = _make_console()
        c.panel("content", title="My Title", width=40)
        assert "My Title" in _output(io)

    def test_panel_title_centered(self):
        c, io = _make_console()
        c.panel("x", title="T", width=40, title_align="center")
        assert "T" in _output(io)

    def test_panel_title_right(self):
        c, io = _make_console()
        c.panel("x", title="T", width=40, title_align="right")
        assert "T" in _output(io)

    def test_panel_multiline(self):
        c, io = _make_console()
        c.panel("line1\nline2\nline3", width=40)
        out = _output(io)
        assert "line1" in out
        assert "line2" in out
        assert "line3" in out

    def test_panel_emits_block(self):
        c, _ = _make_console()
        c.panel("x", width=40)
        b = c.last_block()
        assert b["type"] == "panel"


class TestConsoleBox:
    def test_box_delegates_to_panel(self):
        c, io = _make_console()
        c.box("Hello", width=40)
        assert "Hello" in _output(io)


# =============================================================================
# Console — status
# =============================================================================

class TestConsoleStatus:
    @pytest.mark.parametrize("kind", ["ok", "warn", "error", "info", "step"])
    def test_status_kinds(self, kind):
        c, io = _make_console()
        c.status(kind, f"msg-{kind}")
        assert f"msg-{kind}" in _output(io)

    def test_status_with_detail(self):
        c, io = _make_console()
        c.status("ok", "done", detail="2s")
        out = _output(io)
        assert "done" in out
        assert "2s" in out

    def test_status_unknown_kind(self):
        c, io = _make_console()
        c.status("custom", "msg")
        assert "msg" in _output(io)

    def test_status_emits_block(self):
        c, _ = _make_console()
        c.status("ok", "msg")
        b = c.last_block()
        assert b["type"] == "status"
        assert b["data"]["kind"] == "ok"


# =============================================================================
# Console — table
# =============================================================================

class TestConsoleTable:
    def test_empty_table(self):
        c, io = _make_console()
        c.table([])
        assert "(empty)" in _output(io)

    def test_table_with_rows(self):
        c, io = _make_console()
        c.table([["a", "b"], ["c", "d"]])
        out = _output(io)
        assert "a" in out
        assert "d" in out

    def test_table_with_header(self):
        c, io = _make_console()
        c.table([["1", "2"]], header=["X", "Y"])
        out = _output(io)
        assert "X" in out
        assert "Y" in out

    def test_table_no_header_separator(self):
        c, io = _make_console()
        c.table([["a"]], header=["H"], separator_after_header=False)
        out = _output(io)
        assert "H" in out

    def test_table_uneven_rows(self):
        c, io = _make_console()
        c.table([["a", "b"], ["c"]])
        out = _output(io)
        assert "a" in out
        assert "c" in out

    def test_table_emits_block(self):
        c, _ = _make_console()
        c.table([["x"]])
        b = c.last_block()
        assert b["type"] == "table"


class TestConsoleTableFromDicts:
    def test_empty(self):
        c, io = _make_console()
        c.table_from_dicts([])
        assert "(empty)" in _output(io)

    def test_with_data(self):
        c, io = _make_console()
        c.table_from_dicts([{"a": "1", "b": "2"}, {"a": "3", "b": "4"}])
        out = _output(io)
        assert "1" in out
        assert "4" in out

    def test_missing_key(self):
        c, io = _make_console()
        c.table_from_dicts([{"a": "1"}, {"a": "2", "b": "x"}])
        out = _output(io)
        assert "1" in out


# =============================================================================
# Console — kvlist
# =============================================================================

class TestConsoleKvlist:
    def test_empty(self):
        c, io = _make_console()
        c.kvlist([])
        assert _output(io) == ""

    def test_items(self):
        c, io = _make_console()
        c.kvlist([("name", "Alice"), ("age", "30")])
        out = _output(io)
        assert "name" in out
        assert "Alice" in out
        assert "age" in out

    def test_alignment(self):
        c, io = _make_console()
        c.kvlist([("a", "1"), ("bb", "22")], indent=0)
        lines = _output(io).strip().split("\n")
        assert len(lines) == 2


# =============================================================================
# Console — progress
# =============================================================================

class TestConsoleProgress:
    def test_progress_zero(self):
        c, io = _make_console()
        c.progress("DL", 0, 100)
        out = _output(io)
        assert "0.0%" in out

    def test_progress_full(self):
        c, io = _make_console()
        c.progress("DL", 100, 100)
        out = _output(io)
        assert "100.0%" in out

    def test_progress_emits_block(self):
        c, _ = _make_console()
        c.progress("DL", 50, 100)
        b = c.last_block()
        assert b["type"] == "progress"
        assert b["data"]["current"] == 50


# =============================================================================
# Console — json
# =============================================================================

class TestConsoleJson:
    def test_json_dict(self):
        c, io = _make_console()
        c.json({"key": "value"})
        out = _output(io)
        assert "key" in out
        assert "value" in out

    def test_json_list(self):
        c, io = _make_console()
        c.json([1, 2, 3])
        assert "1" in _output(io)

    def test_json_emits_block(self):
        c, _ = _make_console()
        c.json({"a": 1})
        b = c.last_block()
        assert b["type"] == "json"


# =============================================================================
# Console — error / success / info / warn / note
# =============================================================================

class TestConsoleStatusMethods:
    def test_error(self):
        c, io = _make_console()
        c.error("something broke")
        assert "something broke" in _output(io)

    def test_error_with_hint(self):
        c, io = _make_console()
        c.error("fail", hint="try again")
        out = _output(io)
        assert "fail" in out
        assert "try again" in out

    def test_success(self):
        c, io = _make_console()
        c.success("done")
        assert "done" in _output(io)

    def test_info(self):
        c, io = _make_console()
        c.info("fyi")
        assert "fyi" in _output(io)

    def test_warn(self):
        c, io = _make_console()
        c.warn("careful")
        assert "careful" in _output(io)

    def test_note(self):
        c, io = _make_console()
        c.note("remember this")
        assert "remember this" in _output(io)

    def test_error_emits_block(self):
        c, _ = _make_console()
        c.error("x")
        assert c.last_block()["type"] == "error"

    def test_success_emits_block(self):
        c, _ = _make_console()
        c.success("x")
        assert c.last_block()["type"] == "success"

    def test_info_emits_block(self):
        c, _ = _make_console()
        c.info("x")
        assert c.last_block()["type"] == "info"

    def test_warn_emits_block(self):
        c, _ = _make_console()
        c.warn("x")
        assert c.last_block()["type"] == "warn"

    def test_note_emits_block(self):
        c, _ = _make_console()
        c.note("x")
        assert c.last_block()["type"] == "note"


# =============================================================================
# Console — columns
# =============================================================================

class TestConsoleColumns:
    def test_empty(self):
        c, io = _make_console()
        c.columns([])
        assert _output(io) == ""

    def test_single_column(self):
        c, io = _make_console()
        c.columns(["a", "b", "c"], col_count=1)
        out = _output(io)
        assert "a" in out
        assert "c" in out

    def test_multi_column(self):
        c, io = _make_console()
        c.columns(["a", "b", "c", "d"], col_count=2)
        out = _output(io)
        assert "a" in out
        assert "d" in out

    def test_emits_block(self):
        c, _ = _make_console()
        c.columns(["x"])
        b = c.last_block()
        assert b["type"] == "columns"


# =============================================================================
# Console — tree
# =============================================================================

class TestConsoleTree:
    def test_flat(self):
        c, io = _make_console()
        c.tree({"root": ["a", "b"]})
        out = _output(io)
        assert "root" in out
        assert "a" in out

    def test_nested(self):
        c, io = _make_console()
        c.tree({"parent": {"child": ["leaf"]}})
        out = _output(io)
        assert "parent" in out
        assert "child" in out
        assert "leaf" in out

    def test_emits_block(self):
        c, _ = _make_console()
        c.tree({"r": []})
        assert c.last_block()["type"] == "tree"


# =============================================================================
# Console — log
# =============================================================================

class TestConsoleLog:
    @pytest.mark.parametrize("level", ["info", "warn", "error", "debug"])
    def test_log_levels(self, level):
        c, io = _make_console()
        c.log("msg", level=level)
        assert "msg" in _output(io)

    def test_log_emits_block(self):
        c, _ = _make_console()
        c.log("test")
        b = c.last_block()
        assert b["type"] == "log"
        assert b["data"]["level"] == "info"


# =============================================================================
# Console — markdown
# =============================================================================

class TestConsoleMarkdown:
    def test_heading(self):
        c, io = _make_console()
        c.markdown("# Title")
        assert "Title" in _output(io)

    def test_bold_in_heading(self):
        c, io = _make_console()
        c.markdown("## **Bold**")
        out = _output(io)
        assert "Bold" in out

    def test_code_block(self):
        c, io = _make_console()
        c.markdown("```\nprint('hi')\n```")
        assert "print" in _output(io)

    def test_code_block_with_lang(self):
        c, io = _make_console()
        c.markdown("```python\ncode\n```")
        assert "python" in _output(io)

    def test_blockquote(self):
        c, io = _make_console()
        c.markdown("> quoted text")
        assert "quoted text" in _output(io)

    def test_unordered_list(self):
        c, io = _make_console()
        c.markdown("- item1\n- item2")
        out = _output(io)
        assert "item1" in out
        assert "item2" in out

    def test_ordered_list(self):
        c, io = _make_console()
        c.markdown("1. first\n2. second")
        out = _output(io)
        assert "first" in out

    def test_horizontal_rule(self):
        c, io = _make_console()
        c.markdown("---")
        assert _output(io) != ""

    def test_emits_block(self):
        c, _ = _make_console()
        c.markdown("text")
        assert c.last_block()["type"] == "markdown"


# =============================================================================
# Console — badge
# =============================================================================

class TestConsoleBadge:
    @pytest.mark.parametrize("color", ["info", "ok", "warn", "error"])
    def test_badge_colors(self, color):
        c, io = _make_console()
        c.badge("tag", color=color)
        assert "tag" in _output(io)

    def test_badge_unknown_color(self):
        c, io = _make_console()
        c.badge("x", color="purple")
        assert "x" in _output(io)


# =============================================================================
# Console — summary / header
# =============================================================================

class TestConsoleSummary:
    def test_summary(self):
        c, io = _make_console()
        c.summary("Report", [("k1", "v1"), ("k2", "v2")], width=50)
        out = _output(io)
        assert "Report" in out
        assert "k1" in out
        assert "v1" in out


class TestConsoleHeader:
    def test_header_with_title(self):
        c, io = _make_console()
        c.header("Page Title")
        assert "Page Title" in _output(io)

    def test_header_with_subtitle(self):
        c, io = _make_console()
        c.header("Title", subtitle="sub")
        out = _output(io)
        assert "Title" in out
        assert "sub" in out


# =============================================================================
# Console — styled
# =============================================================================

class TestConsoleStyled:
    def test_styled_bold(self):
        c, _ = _make_console()
        result = c.styled("hello", "bold")
        assert "hello" in result

    def test_styled_unknown(self):
        c, _ = _make_console()
        result = c.styled("hello", "unknownstyle")
        assert result == "hello"


# =============================================================================
# Console — cursor control / clear
# =============================================================================

class TestConsoleCursorClear:
    def test_hide_cursor(self):
        c, io = _make_console()
        c.hide_cursor()
        assert "\033[?25l" in _output(io)

    def test_show_cursor(self):
        c, io = _make_console()
        c.show_cursor()
        assert "\033[?25h" in _output(io)

    def test_clear(self):
        c, io = _make_console()
        c.clear()
        assert "\033[2J\033[H" in _output(io)


# =============================================================================
# Console — download_bar
# =============================================================================

class TestConsoleDownloadBar:
    def test_download_bar_basic(self):
        c, io = _make_console()
        c.download_bar("file.zip", 50, 100)
        out = _output(io)
        assert "file.zip" in out
        assert "50.0%" in out

    def test_download_bar_complete(self):
        c, io = _make_console()
        c.download_bar("f", 100, 100, bytes_done=1024, bytes_total=1024, speed=512)
        assert _output(io) != ""


# =============================================================================
# _Capture
# =============================================================================

class TestCapture:
    def test_capture_writes(self):
        c, io = _make_console()
        with c.capture() as cap:
            c.write("hello")
        text = cap.get()
        assert "hello" in text

    def test_capture_restores_write(self):
        c, io = _make_console()
        with c.capture():
            pass
        io.write("test")
        assert "test" in _output(io)

    def test_capture_multiple_writes(self):
        c, io = _make_console()
        with c.capture() as cap:
            c.write("a")
            c.write("b")
        text = cap.get()
        assert "a" in text
        assert "b" in text


# =============================================================================
# _Indent
# =============================================================================

class TestIndent:
    def test_indent_adds_prefix(self):
        c, io = _make_console()
        with c.indent(4):
            c.write("text")
        out = _output(io)
        assert "    text" in out

    def test_indent_restores(self):
        c, io = _make_console()
        with c.indent(2):
            pass
        c.write("plain")
        out = _output(io)
        assert "plain" in out
        assert "  plain" not in out

    def test_indent_custom_char(self):
        c, io = _make_console()
        with c.indent(3, char="."):
            c.write("x")
        out = _output(io)
        assert "..." in out


# =============================================================================
# _Live
# =============================================================================

class TestLive:
    def test_live_context(self):
        c, io = _make_console()
        with c.live() as live:
            live.update("hello")
        assert "hello" in _output(io)

    def test_live_update_replaces(self):
        c, io = _make_console()
        with c.live() as live:
            live.update("first")
            live.update("second")
        out = _output(io)
        assert "first" in out
        assert "second" in out


# =============================================================================
# _Spinner
# =============================================================================

class TestSpinner:
    def test_spinner_enter_exit(self):
        c, io = _make_console()
        s = c.spinner("Loading", rate=0.01)
        with s:
            pass
        assert isinstance(s, _Spinner)

    def test_spinner_ok(self):
        c, io = _make_console()
        s = c.spinner("Working", rate=0.01)
        with s:
            pass
        s.ok("done")
        assert "done" in _output(io)

    def test_spinner_fail(self):
        c, io = _make_console()
        s = c.spinner("Working", rate=0.01)
        with s:
            pass
        s.fail("failed")
        assert "failed" in _output(io)


# =============================================================================
# Console — paginate
# =============================================================================

class TestConsolePaginate:
    def test_paginate_single_page(self):
        c, io = _make_console()
        c.paginate(["a", "b", "c"], page_size=10)
        out = _output(io)
        assert "a" in out
        assert "c" in out

    def test_paginate_emits_block(self):
        c, _ = _make_console()
        c.paginate(["x"], page_size=10)
        b = c.last_block()
        assert b["type"] == "paginate"
        assert b["data"]["line_count"] == 1


# =============================================================================
# Console — dispatch delegate methods (emit only, delegate to InteractivePrompt)
# =============================================================================

class TestConsoleDelegates:
    def test_confirm_emits_block(self):
        c, _ = _make_console()
        c._tui_repl = None
        result = c.confirm("yes?", default=True)
        assert isinstance(result, bool)
        assert c.last_block()["type"] == "confirm"

    def test_ask_emits_block(self):
        c, io = _make_console()
        io.feed("answer")
        c._tui_repl = None
        result = c.ask("question?")
        assert c.last_block()["type"] == "ask"

    def test_select_emits_block(self):
        c, io = _make_console()
        io.feed("1")
        c._tui_repl = None
        c.select("pick", ["a", "b"])
        assert c.last_block()["type"] == "select"


# =============================================================================
# Console — constructor
# =============================================================================

class TestConsoleInit:
    def test_init_default(self):
        c, io = _make_console()
        assert c._io is io
        assert c._has_readline is False
        assert c._blocks == []

    def test_init_readline_flag(self):
        c, io = _make_console(has_readline=True)
        assert isinstance(c._has_readline, bool)

    def test_tui_repl_none(self):
        c, _ = _make_console()
        assert c._tui_repl is None
