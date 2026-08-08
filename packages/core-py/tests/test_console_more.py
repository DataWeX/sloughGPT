"""Coverage tests for Console (domains.shell.console)."""

import importlib
import sys

import pytest

from domains.shell import console as console_mod
from domains.shell.console import Console, _human_size, _render_inline
from domains.shell.io import MemoryIO


def _mk(feeds=None, has_readline=False):
    io = MemoryIO()
    for f in feeds or []:
        io.feed(f)
    return io, Console(io, has_readline=has_readline)


class TestHelpers:
    def test_human_size(self):
        assert _human_size(500) == "500.0 B"
        assert _human_size(2048) == "2.0 KB"
        assert _human_size(2 * 1024 * 1024) == "2.0 MB"
        assert _human_size(3 * 1024 ** 3) == "3.0 GB"
        assert _human_size(4 * 1024 ** 4) == "4.0 TB"
        assert _human_size(5 * 1024 ** 5) == "5.0 PB"

    def test_render_inline(self):
        out = _render_inline("**bold** *it* `code` [label](url)")
        assert "bold" in out
        assert "it" in out
        assert "code" in out
        assert "label" in out
        assert "url" in out

    def test_color_disabled_branch(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        importlib.reload(console_mod)
        assert console_mod._COLOR_ENABLED is False
        assert console_mod._color("x", "\033[1m") == "x"
        monkeypatch.delenv("NO_COLOR")
        importlib.reload(console_mod)
        assert console_mod._COLOR_ENABLED is True


class TestBlocks:
    def test_blocks_recorded(self):
        _, c = _mk()
        c.write("a")
        c.print("b")
        c.status("ok", "fine")
        blocks = c.get_blocks()
        assert [b["type"] for b in blocks] == ["write", "print", "write", "status"]
        assert blocks[0]["data"]["text"] == "a"

    def test_get_json(self):
        _, c = _mk()
        c.write("hi")
        assert '"type"' in c.get_json()

    def test_get_json_with_bad_object(self):
        _, c = _mk()
        c.json(object())
        assert "{}" in c.get_json() or "null" in c.get_json()

    def test_clear_and_last_block(self):
        _, c = _mk()
        assert c.last_block() is None
        c.write("a")
        assert c.last_block()["type"] == "write"
        c.clear_blocks()
        assert c.get_blocks() == []
        assert c.last_block() is None


class TestWrite:
    def test_write_no_readline(self):
        io, c = _mk()
        c.write("hello")
        assert "hello" in io.get_output()

    def test_write_empty_text(self):
        io, c = _mk(has_readline=True)
        c.write("  ")
        assert io.get_output() == "  \n"

    def test_write_readline_empty_buffer(self):
        io, c = _mk(has_readline=True)
        c.write("x")
        assert "x" in io.get_output()

    def test_write_readline_buffer_save_restore(self, monkeypatch):
        import readline
        monkeypatch.setattr(readline, "get_line_buffer", lambda: "abc")
        io, c = _mk(has_readline=True)
        c.write("line1\nline2")
        out = io.get_output()
        assert "\033[s" in out
        assert "\033[u" in out
        assert "abc" in out

    def test_write_readline_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "readline", None)
        io, c = _mk(has_readline=True)
        c.write("ok")
        assert "ok" in io.get_output()

    def test_write_readline_get_line_buffer_raises(self, monkeypatch):
        import readline

        def boom():
            raise RuntimeError("nope")

        monkeypatch.setattr(readline, "get_line_buffer", boom)
        io, c = _mk(has_readline=True)
        c.write("ok")
        assert "ok" in io.get_output()


class TestPrintAndRule:
    def test_print(self):
        io, c = _mk()
        c.print("a", "b", sep=",")
        assert "a b" in io.get_output()

    def test_rule_no_label(self):
        io, c = _mk()
        c.rule()
        assert "─" in io.get_output()

    def test_rule_label(self):
        io, c = _mk()
        c.rule("Section")
        assert "Section" in io.get_output()

    def test_rule_custom_width(self):
        io, c = _mk()
        c.rule("x", width=20)
        assert "x" in io.get_output()

    def test_separator(self):
        io, c = _mk()
        line = c.separator()
        assert line and "─" in io.get_output()

    def test_separator_no_color(self):
        io, c = _mk()
        line = c.separator(color="")
        assert line

    def test_section(self):
        io, c = _mk()
        c.section("T")
        assert "T" in io.get_output()


class TestPanel:
    def test_panel_no_title(self):
        io, c = _mk()
        c.panel("body")
        out = io.get_output()
        assert "┌" in out and "┐" in out and "body" in out

    def test_panel_title_left(self):
        io, c = _mk()
        c.panel("body", title="T")
        assert "T" in io.get_output()

    def test_panel_title_center(self):
        io, c = _mk()
        c.panel("body", title="T", title_align="center")
        assert "T" in io.get_output()

    def test_panel_title_right(self):
        io, c = _mk()
        c.panel("body", title="T", title_align="right")
        assert "T" in io.get_output()

    def test_panel_multiline_and_truncation(self):
        io, c = _mk()
        c.panel("l1\nl2", title="x" * 200)
        assert "l1" in io.get_output()

    def test_box(self):
        io, c = _mk()
        c.box("hi")
        assert "hi" in io.get_output()


class TestStatus:
    def test_status_kinds(self):
        for kind in ("ok", "warn", "error", "info", "step"):
            io, c = _mk()
            c.status(kind, "msg")
            assert "msg" in io.get_output()

    def test_status_unknown_kind(self):
        io, c = _mk()
        c.status("weird", "msg")
        assert "weird" in io.get_output()

    def test_status_detail(self):
        io, c = _mk()
        c.status("ok", "msg", "detail")
        assert "detail" in io.get_output()


class TestTable:
    def test_table_empty(self):
        io, c = _mk()
        c.table([])
        assert "empty" in io.get_output()

    def test_table_with_header_and_separator(self):
        io, c = _mk()
        c.table([["1", "2"], ["3", "4"]], header=["A", "B"])
        assert "A" in io.get_output()
        assert "1" in io.get_output()

    def test_table_no_header(self):
        io, c = _mk()
        c.table([["x", "y"]])
        assert "x" in io.get_output()

    def test_table_no_separator(self):
        io, c = _mk()
        c.table([["x"]], header=["A"], separator_after_header=False)
        assert "x" in io.get_output()

    def test_table_ragged_rows(self):
        io, c = _mk()
        c.table([["a"], ["b", "c"]])
        assert "a" in io.get_output()


class TestKvlistProgress:
    def test_kvlist(self):
        io, c = _mk()
        c.kvlist([("a", "1"), ("longer", "2")])
        assert "a" in io.get_output()

    def test_kvlist_empty(self):
        io, c = _mk()
        c.kvlist([])
        assert io.get_output() == ""

    def test_progress_partial(self):
        io, c = _mk()
        c.progress("work", 5, 10)
        assert "work" in io.get_output()

    def test_progress_complete(self):
        io, c = _mk()
        c.progress("work", 10, 10)
        assert "100.0%" in io.get_output()

    def test_progress_zero_total(self):
        io, c = _mk()
        c.progress("work", 0, 0)
        assert "work" in io.get_output()


class TestPrompts:
    def test_confirm_default(self):
        io, c = _mk(feeds=[""])
        assert c.confirm("go?", default=True) is True

    def test_confirm_yes(self):
        io, c = _mk(feeds=["yes"])
        assert c.confirm("go?") is True

    def test_confirm_no(self):
        io, c = _mk(feeds=["no"])
        assert c.confirm("go?", default=True) is False

    def test_ask_input(self):
        io, c = _mk(feeds=["value"])
        assert c.ask("name") == "value"

    def test_ask_default(self):
        io, c = _mk(feeds=[""])
        assert c.ask("name", "def") == "def"


class TestJsonPaginate:
    def test_json(self):
        io, c = _mk()
        c.json({"a": 1, "b": [2, 3]})
        assert '"a"' in io.get_output()

    def test_json_side_effect(self):
        io, c = _mk()
        c.json({"a": 1})
        blocks = c.get_blocks()
        assert blocks[-1]["type"] == "json"

    def test_paginate_single_page(self):
        io, c = _mk()
        c.paginate(["a", "b"], page_size=10)
        assert "a" in io.get_output()

    def test_paginate_multiple_pages(self):
        io, c = _mk(feeds=["", ""])
        c.paginate(list("abcdefgh"), page_size=3)
        assert "More" in io.get_output()

    def test_paginate_default_size(self):
        io, c = _mk(feeds=[""])
        c.paginate(["a"], page_size=None)
        assert "a" in io.get_output()


class TestStatusMessages:
    def test_error_no_hint(self):
        io, c = _mk()
        c.error("boom")
        assert "boom" in io.get_output()

    def test_error_with_hint(self):
        io, c = _mk()
        c.error("boom", "try again")
        assert "try again" in io.get_output()

    def test_success(self):
        io, c = _mk()
        c.success("done")
        assert "done" in io.get_output()

    def test_info(self):
        io, c = _mk()
        c.info("note")
        assert "note" in io.get_output()

    def test_warn(self):
        io, c = _mk()
        c.warn("careful")
        assert "careful" in io.get_output()


class TestTableFromDicts:
    def test_table_from_dicts(self):
        io, c = _mk()
        c.table_from_dicts([{"a": "1", "b": "2"}])
        assert "a" in io.get_output()

    def test_table_from_dicts_empty(self):
        io, c = _mk()
        c.table_from_dicts([])
        assert "empty" in io.get_output()


class TestCaptureIndent:
    def test_capture(self):
        _, c = _mk()
        with c.capture() as cap:
            c.print("hello")
            c.write(" world")
        text = cap.get()
        assert "hello" in text and "world" in text
        assert "hello\n world" in text

    def test_indent(self):
        io, c = _mk()
        with c.indent(4):
            c.print("x")
        assert "    x" in io.get_output()

    def test_indent_multiline(self):
        io, c = _mk()
        with c.indent(2):
            c.print("a\nb")
        assert "  a\n  b" in io.get_output()


class TestColumnsTree:
    def test_columns_empty(self):
        io, c = _mk()
        c.columns([])
        assert io.get_output() == ""

    def test_columns_explicit_count(self):
        io, c = _mk()
        c.columns(["a", "b", "c", "d"], col_count=2)
        assert "a" in io.get_output()

    def test_columns_auto_count(self):
        io, c = _mk()
        c.columns(["aa", "bb", "cc"])
        assert "aa" in io.get_output()

    def test_tree(self):
        io, c = _mk()
        c.tree({"root": {"sub": ["x", "y"]}, "leaf": ["1", "2"]})
        out = io.get_output()
        assert "root" in out
        assert "sub" in out
        assert "leaf" in out


class TestLogBadge:
    def test_log_levels(self):
        for level in ("info", "warn", "error", "debug"):
            io, c = _mk()
            c.log("msg", level)
            assert "msg" in io.get_output()

    def test_log_unknown_level(self):
        io, c = _mk()
        c.log("msg", "trace")
        assert "msg" in io.get_output()

    def test_badge_known(self):
        for color in ("info", "ok", "warn", "error"):
            io, c = _mk()
            c.badge("L", color)
            assert "L" in io.get_output()

    def test_badge_unknown(self):
        io, c = _mk()
        c.badge("L", "nope")
        assert "L" in io.get_output()


class TestMarkdown:
    def test_markdown_code_fence(self):
        io, c = _mk()
        c.markdown("```\n\nx = 1\n\n```")
        assert "x = 1" in io.get_output()

    def test_markdown_code_fence_with_lang(self):
        io, c = _mk()
        c.markdown("```py\n\nx\n\n```")
        assert "py" in io.get_output()

    def test_markdown_hr(self):
        io, c = _mk()
        c.markdown("---")
        assert "─" in io.get_output()

    def test_markdown_blockquote(self):
        io, c = _mk()
        c.markdown("> quote")
        assert "quote" in io.get_output()

    def test_markdown_heading1(self):
        io, c = _mk()
        c.markdown("# Title")
        assert "Title" in io.get_output()

    def test_markdown_heading2(self):
        io, c = _mk()
        c.markdown("## Sub")
        assert "Sub" in io.get_output()

    def test_markdown_bullets(self):
        io, c = _mk()
        c.markdown("- one\n- two")
        assert "one" in io.get_output()

    def test_markdown_numbered(self):
        io, c = _mk()
        c.markdown("1. first")
        assert "first" in io.get_output()

    def test_markdown_plain(self):
        io, c = _mk()
        c.markdown("just text")
        assert "just text" in io.get_output()

    def test_markdown_empty_block_skipped(self):
        io, c = _mk()
        c.markdown("a\n\n")
        assert "a" in io.get_output()


class TestMisc:
    def test_summary(self):
        io, c = _mk()
        c.summary("Title", [("k", "v")])
        assert "Title" in io.get_output()

    def test_header_with_subtitle(self):
        io, c = _mk()
        c.header("Title", "sub")
        assert "Title" in io.get_output()

    def test_header_no_subtitle(self):
        io, c = _mk()
        c.header("Title")
        assert "Title" in io.get_output()

    def test_note(self):
        io, c = _mk()
        c.note("n")
        assert "n" in io.get_output()

    def test_select_valid(self):
        io, c = _mk(feeds=["2"])
        assert c.select("pick", ["a", "b"]) == "b"

    def test_select_invalid_then_valid(self):
        io, c = _mk(feeds=["9", "1"])
        assert c.select("pick", ["a", "b"]) == "a"

    def test_select_eof(self):
        io, c = _mk()
        assert c.select("pick", ["a", "b"]) == "a"

    def test_select_eof_empty_options(self):
        io, c = _mk()
        assert c.select("pick", []) == ""

    def test_hide_show_cursor(self):
        io, c = _mk()
        c.hide_cursor()
        c.show_cursor()
        assert "\033[?25l" in io.get_output()
        assert "\033[?25h" in io.get_output()

    def test_clear(self):
        io, c = _mk()
        c.clear()
        assert "\033[2J" in io.get_output()

    def test_styled(self):
        io, c = _mk()
        for style in ("bold", "dim", "italic", "underline", "cyan", "green", "yellow", "red"):
            assert c.styled("x", style)
        assert c.styled("x", "unknown") == "x"

    def test_download_bar_partial(self):
        io, c = _mk()
        c.download_bar("dl", 5, 10, 500, 1000, speed=100)
        assert "dl" in io.get_output()

    def test_download_bar_no_speed(self):
        io, c = _mk()
        c.download_bar("dl", 5, 10)
        assert "dl" in io.get_output()

    def test_download_bar_eta_minutes(self):
        io, c = _mk()
        c.download_bar("dl", 0, 10, 0, 2000, speed=1)
        assert "m" in io.get_output()

    def test_download_bar_complete(self):
        io, c = _mk()
        c.download_bar("dl", 10, 10, 1000, 1000, speed=100)
        assert "100.0%" in io.get_output()


class TestSpinnerLive:
    def test_spinner_enter_exit(self):
        io, c = _mk()
        with c.spinner("working", rate=0.01):
            pass
        assert "working" in io.get_output()

    def test_spinner_ok(self):
        io, c = _mk()
        with c.spinner("working", rate=0.01) as s:
            pass
        s.ok("finished")
        assert "finished" in io.get_output()

    def test_spinner_fail(self):
        io, c = _mk()
        with c.spinner("working", rate=0.01) as s:
            pass
        s.fail("broke")
        assert "broke" in io.get_output()

    def test_live_update(self):
        io, c = _mk()
        with c.live() as lv:
            lv.update("one")
            lv.update("two\nthree")
        out = io.get_output()
        assert "one" in out
        assert "two" in out
