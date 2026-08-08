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


class TestRule:
    def test_rule_empty_label(self):
        c, io = _make_c()
        c.rule()
        out = io.get_output()
        assert len(out.strip()) > 0

    def test_rule_with_label(self):
        c, io = _make_c()
        c.rule("Section")
        out = io.get_output()
        assert "Section" in out

    def test_rule_custom_char(self):
        c, io = _make_c()
        c.rule("X", char="=", width=20)
        out = _visible(io.get_output())
        assert " X " in out
        assert "=" in out


class TestPanel:
    def test_panel_basic(self):
        c, io = _make_c()
        c.panel("Hello")
        out = io.get_output()
        assert "┌" in out
        assert "┐" in out
        assert "└" in out
        assert "┘" in out

    def test_panel_with_title(self):
        c, io = _make_c()
        c.panel("Body", title="Title")
        out = io.get_output()
        assert "Title" in out
        assert "Body" in out
        assert "┌" in out

    def test_panel_title_align_left(self):
        c, io = _make_c()
        c.panel("Body", title="T", width=20, title_align="left")
        out = io.get_output()
        assert "T" in out


class TestConfirm:
    def test_confirm_default_true(self):
        io = MemoryIO()
        io.feed("")
        c = Console(io)
        result = c.confirm("Are you sure?", default=True)
        assert result is True

    def test_confirm_default_false(self):
        io = MemoryIO()
        io.feed("")
        c = Console(io)
        result = c.confirm("Are you sure?", default=False)
        assert result is False

    def test_confirm_y(self):
        io = MemoryIO()
        io.feed("y")
        c = Console(io)
        result = c.confirm("Sure?", default=False)
        assert result is True

    def test_confirm_n(self):
        io = MemoryIO()
        io.feed("n")
        c = Console(io)
        result = c.confirm("Sure?", default=True)
        assert result is False


class TestAsk:
    def test_ask_returns_input(self):
        io = MemoryIO()
        io.feed("hello")
        c = Console(io)
        result = c.ask("Name?")
        assert result == "hello"

    def test_ask_returns_default(self):
        io = MemoryIO()
        io.feed("")
        c = Console(io)
        result = c.ask("Name?", default="Alice")
        assert result == "Alice"


class TestJson:
    def test_json_basic(self):
        c, io = _make_c()
        c.json({"a": 1, "b": [2, 3]})
        out = io.get_output()
        assert '"a"' in out
        assert "1" in out

    def test_json_non_serializable_fallback(self):
        c, io = _make_c()
        c.json({"x": object()})
        out = io.get_output()
        assert '"x"' in out


class TestPaginate:
    def test_paginate_single_page(self):
        c, io = _make_c()
        c.paginate(["a", "b"], page_size=5)
        out = io.get_output()
        assert "a" in out
        assert "b" in out

    def test_paginate_multiple_pages(self):
        io = MemoryIO()
        io.feed("")
        c = Console(io)
        c.paginate(["a", "b", "c", "d", "e", "f"], page_size=3)
        out = io.get_output()
        assert "c" in out
        assert "f" in out
        assert "More" not in out.split("\n")[-1] if io.get_output().strip() else True


class TestSpinner:
    def test_spinner_context_manager(self):
        c, io = _make_c()
        with c.spinner("loading"):
            pass
        out = io.get_output()
        assert out.endswith("\r\033[K")  # line cleared on exit

    def test_spinner_ok(self):
        c, io = _make_c()
        with c.spinner("task") as s:
            pass
        s.ok("done!")
        out = io.get_output()
        assert "done!" in out

    def test_spinner_fail(self):
        c, io = _make_c()
        with c.spinner("task") as s:
            pass
        s.fail("broken")
        out = io.get_output()
        assert "broken" in out


class TestColumns:
    def test_columns_empty(self):
        c, io = _make_c()
        c.columns([])
        assert io.get_output() == ""

    def test_columns_basic(self):
        c, io = _make_c()
        c.columns(["a", "b", "c"], col_count=2)
        out = io.get_output()
        assert "a" in out
        assert "b" in out
        assert "c" in out

    def test_columns_single_row(self):
        c, io = _make_c()
        c.columns(["x"], col_count=3)
        out = io.get_output()
        assert "x" in out


class TestTree:
    def test_tree_flat_list(self):
        c, io = _make_c()
        c.tree({"root": ["a", "b"]})
        out = io.get_output()
        assert "root" in out
        assert "├──" in out
        assert "└──" in out

    def test_tree_nested(self):
        c, io = _make_c()
        c.tree({"top": {"mid": ["leaf"]}})
        out = io.get_output()
        assert "top" in out
        assert "mid" in out
        assert "leaf" in out


class TestLog:
    def test_log_info(self):
        c, io = _make_c()
        c.log("started")
        out = io.get_output()
        assert "started" in out
        assert "INFO" in _visible(out)

    def test_log_warn(self):
        c, io = _make_c()
        c.log("careful", level="warn")
        out = io.get_output()
        assert "careful" in out
        assert "WARN" in _visible(out)

    def test_log_error(self):
        c, io = _make_c()
        c.log("fail", level="error")
        out = io.get_output()
        assert "fail" in out
        assert "ERROR" in _visible(out)

    def test_log_debug(self):
        c, io = _make_c()
        c.log("verbose", level="debug")
        out = io.get_output()
        assert "verbose" in out
        assert "DEBUG" in _visible(out)

    def test_log_custom_level(self):
        c, io = _make_c()
        c.log("custom", level="trace")
        out = io.get_output()
        assert "custom" in out
        assert "TRACE" in _visible(out)


class TestSummary:
    def test_summary_basic(self):
        c, io = _make_c()
        c.summary("Results", [("key1", "val1"), ("key2", "val2")], width=40)
        out = io.get_output()
        assert "Results" in out
        assert "key1" in out
        assert "val1" in out
        assert "┌" in out
        assert "└" in out

    def test_summary_empty_items(self):
        c, io = _make_c()
        c.summary("Empty", [], width=30)
        out = io.get_output()
        assert "Empty" in out


class TestLive:
    def test_live_single_update(self):
        c, io = _make_c()
        with c.live() as live:
            live.update("hello")
        out = io.get_output()
        assert "hello" in out

    def test_live_multiple_updates(self):
        c, io = _make_c()
        with c.live() as live:
            live.update("first")
            live.update("second")
        out = io.get_output()
        assert "second" in out

    def test_live_multi_line(self):
        c, io = _make_c()
        with c.live() as live:
            live.update("a\nb\nc")
        out = io.get_output()
        assert "a" in out
        assert "b" in out
        assert "c" in out


class TestMarkdown:
    def test_md_heading(self):
        c, io = _make_c()
        c.markdown("# Title")
        out = io.get_output()
        assert "Title" in out

    def test_md_bold(self):
        c, io = _make_c()
        c.markdown("**bold** text")
        out = io.get_output()
        assert "bold" in out

    def test_md_italic(self):
        c, io = _make_c()
        c.markdown("*italic* text")
        out = io.get_output()
        assert "italic" in out

    def test_md_code(self):
        c, io = _make_c()
        c.markdown("use `code` here")
        out = io.get_output()
        assert "code" in out

    def test_md_code_block(self):
        c, io = _make_c()
        c.markdown("```\nprint(1)\n```")
        out = io.get_output()
        assert "print(1)" in out

    def test_md_bullet_list(self):
        c, io = _make_c()
        c.markdown("- a\n- b\n- c")
        out = io.get_output()
        assert "• a" in out
        assert "• b" in out

    def test_md_numbered_list(self):
        c, io = _make_c()
        c.markdown("1. first\n2. second")
        out = io.get_output()
        assert "first" in out
        assert "second" in out

    def test_md_blockquote(self):
        c, io = _make_c()
        c.markdown("> quoted text")
        out = io.get_output()
        assert "quoted" in out
        assert "│" in out

    def test_md_hr(self):
        c, io = _make_c()
        c.markdown("---")
        out = io.get_output()
        assert "─" in out

    def test_md_link(self):
        c, io = _make_c()
        c.markdown("[text](http://url)")
        out = io.get_output()
        assert "text" in out
        assert "url" in out

    def test_md_multiline_paragraph(self):
        c, io = _make_c()
        c.markdown("hello\nworld")
        out = io.get_output()
        assert "hello" in out
        assert "world" in out


class TestBadge:
    def test_badge_info(self):
        c, io = _make_c()
        c.badge("INFO")
        out = io.get_output()
        assert "INFO" in out

    def test_badge_ok(self):
        c, io = _make_c()
        c.badge("OK", color="ok")
        out = io.get_output()
        assert "OK" in out

    def test_badge_warn(self):
        c, io = _make_c()
        c.badge("WARN", color="warn")
        out = io.get_output()
        assert "WARN" in out

    def test_badge_error(self):
        c, io = _make_c()
        c.badge("ERR", color="error")
        out = io.get_output()
        assert "ERR" in out


class TestWarn:
    def test_warn_basic(self):
        c, io = _make_c()
        c.warn("caution")
        out = io.get_output()
        assert "caution" in out


class TestTableFromDicts:
    def test_tfd_empty(self):
        c, io = _make_c()
        c.table_from_dicts([])
        assert "(empty)" in io.get_output()

    def test_tfd_basic(self):
        c, io = _make_c()
        c.table_from_dicts([{"a": "1", "b": "2"}, {"a": "3", "b": "4"}])
        out = io.get_output()
        assert "a" in out
        assert "b" in out
        assert "1" in out
        assert "4" in out


class TestCapture:
    def test_capture_basic(self):
        c, io = _make_c()
        with c.capture() as cap:
            c.print("hello")
        assert cap.get() == "hello\n"

    def test_capture_multiple_writes(self):
        c, io = _make_c()
        with c.capture() as cap:
            c.print("a")
            c.print("b")
        text = cap.get()
        assert "a" in text
        assert "b" in text

    def test_capture_restores_write(self):
        c, io = _make_c()
        with c.capture():
            pass
        c.print("after")
        assert io.get_output().strip() == "after"


class TestIndent:
    def test_indent_basic(self):
        c, io = _make_c()
        with c.indent(4):
            c.print("hello")
        out = io.get_output()
        assert out.startswith("    hello")

    def test_indent_restores_write(self):
        c, io = _make_c()
        with c.indent(2):
            pass
        c.print("noindent")
        out = io.get_output()
        assert out.startswith("noindent")


class TestHeader:
    def test_header_title_only(self):
        c, io = _make_c()
        c.header("Title")
        out = io.get_output()
        assert "Title" in out

    def test_header_with_subtitle(self):
        c, io = _make_c()
        c.header("Title", "Subtitle")
        out = io.get_output()
        assert "Subtitle" in out


class TestNote:
    def test_note_basic(self):
        c, io = _make_c()
        c.note("annotation")
        out = io.get_output()
        assert "annotation" in out


class TestSelect:
    def test_select_returns_chosen(self):
        io = MemoryIO()
        io.feed("2")
        c = Console(io)
        result = c.select("Pick:", ["a", "b", "c"])
        assert result == "b"

    def test_select_first_on_eof(self):
        io = MemoryIO()
        c = Console(io)
        result = c.select("Pick:", ["x", "y"])
        assert result == "x"

    def test_select_invalid_then_valid(self):
        io = MemoryIO()
        io.feed("99")
        io.feed("1")
        c = Console(io)
        result = c.select("Pick:", ["only"])
        assert result == "only"


class TestCursor:
    def test_hide_cursor(self):
        c, io = _make_c()
        c.hide_cursor()
        out = io.get_output()
        assert "?25l" in out

    def test_show_cursor(self):
        c, io = _make_c()
        c.show_cursor()
        out = io.get_output()
        assert "?25h" in out


class TestClear:
    def test_clear(self):
        c, io = _make_c()
        c.clear()
        out = io.get_output()
        assert "2J" in out
        assert "H" in out


class TestStyled:
    def test_styled_bold(self):
        c, io = _make_c()
        result = c.styled("text", "bold")
        assert "text" in result

    def test_styled_unknown_style(self):
        c, io = _make_c()
        result = c.styled("text", "nonexistent")
        assert result == "text"


class TestDownloadBar:
    def test_download_bar_start(self):
        c, io = _make_c()
        c.download_bar("file.bin", 0, 100, 0, 1000)
        out = io.get_output()
        assert "0.0%" in out
        assert "file.bin" in out

    def test_download_bar_complete(self):
        c, io = _make_c()
        c.download_bar("file.bin", 100, 100, 1000, 1000)
        out = io.get_output()
        assert "100.0%" in out

    def test_download_bar_with_speed(self):
        c, io = _make_c()
        c.download_bar("f", 50, 100, 500, 1000, speed=2048)
        out = io.get_output()
        assert "KB/s" in out or "B/s" in out

    def test_human_size(self):
        from domains.shell.console import _human_size
        assert "B" in _human_size(0)
        assert "KB" in _human_size(2048)
        assert "MB" in _human_size(2**20 * 5)


class TestBlockEmit:
    """Every Console method emits a block that an LLM can parse and render."""

    def _emit_all(self, c, io):
        io.feed("y", "answer")  # confirm + ask
        c.write("hello")
        c.print("world")
        c.rule()
        c.separator()
        c.panel("panel text", title="Panel")
        c.box("box text")
        c.status("ok", "done")
        c.table([["a", "1"], ["b", "2"]], header=["K", "V"])
        c.table_from_dicts([{"x": "1"}, {"x": "2"}])
        c.kvlist([("key", "value")])
        c.progress("p", 5, 10)
        c.confirm("y?", default=True)
        c.ask("name?", default="me")
        c.json({"a": 1})
        c.paginate(["line1", "line2"], page_size=10)
        c.error("err", hint="fix")
        c.success("ok")
        c.info("info msg")
        c.warn("warn msg")
        c.note("note msg")
        c.badge("badge", color="warn")
        c.columns(["a", "b", "c"], col_count=2)
        c.tree({"root": {"sub": ["leaf"]}})
        c.log("log msg", level="warn")
        c.markdown("# Hello\n\nThis is **bold**")
        c.summary("Summary", [("k", "v")])
        c.header("Title", "sub")
        c.select("Pick", ["a", "b"])
        c.hide_cursor()
        c.show_cursor()
        c.clear()
        c.download_bar("dl", 50, 100)
        c.live()

    EXPECTED_TYPES = {
        "write", "print", "rule", "separator", "panel",
        "status", "table", "kvlist", "progress", "confirm",
        "ask", "json", "paginate", "error", "success",
        "info", "warn", "note", "badge", "columns", "tree",
        "log", "markdown", "summary", "header", "select",
        "hide_cursor", "show_cursor", "clear", "download_bar",
        "live",
    }

    def test_all_blocks_have_types(self):
        c, io = _make_c()
        self._emit_all(c, io)
        blocks = c.get_blocks()
        types = {b["type"] for b in blocks}
        missing = self.EXPECTED_TYPES - types
        assert not missing, f"missing block types: {missing}"

    def test_json_is_parseable(self):
        c, io = _make_c()
        self._emit_all(c, io)
        raw = c.get_json()
        import json as _json
        parsed = _json.loads(raw)
        assert isinstance(parsed, list)
        assert len(parsed) >= len(self.EXPECTED_TYPES)

    def test_each_block_has_type_and_data(self):
        c, io = _make_c()
        self._emit_all(c, io)
        for b in c.get_blocks():
            assert "type" in b, f"block missing type: {b}"
            assert "data" in b, f"block {b['type']} missing data"
            assert isinstance(b["data"], dict)

    def test_block_clear(self):
        c, io = _make_c()
        c.print("hello")
        assert len(c.get_blocks()) > 0
        c.clear_blocks()
        assert c.get_blocks() == []

    def test_last_block(self):
        c, io = _make_c()
        assert c.last_block() is None
        c.write("hello")
        last = c.last_block()
        assert last is not None
        assert last["type"] == "write"

    def test_panel_block_contains_text_and_title(self):
        c, io = _make_c()
        c.panel("content", title="T")
        b = c.last_block()
        assert b["type"] == "panel"
        assert b["data"]["text"] == "content"
        assert b["data"]["title"] == "T"

    def test_table_block_contains_header_and_rows(self):
        c, io = _make_c()
        c.table([["a", "b"]], header=["X", "Y"])
        b = c.last_block()
        assert b["type"] == "table"
        assert b["data"]["header"] == ["X", "Y"]
        assert b["data"]["rows"] == [["a", "b"]]

    def test_json_block_contains_data(self):
        c, io = _make_c()
        c.json({"nested": {"key": 42}})
        b = c.last_block()
        assert b["type"] == "json"
        assert b["data"]["data"] == {"nested": {"key": 42}}

    def test_confirm_block_contains_result(self):
        c, io = _make_c()
        io.feed("y")
        c.confirm("proceed?")
        b = c.last_block()
        assert b["type"] == "confirm"
        assert b["data"]["result"] is True


class TestEdgeBranches:
    def test_human_size_reaches_pb(self):
        from domains.shell.console import _human_size
        assert _human_size(1024 ** 5) == "1.0 PB"

    def test_panel_title_center(self):
        c, io = _make_c()
        c.panel("x", title="T", title_align="center")
        assert any("\u250c" in _visible(s) for s in io._output)

    def test_panel_title_right(self):
        c, io = _make_c()
        c.panel("x", title="T", title_align="right")
        assert any("\u250c" in _visible(s) for s in io._output)

    def test_columns_auto_fit(self):
        c, io = _make_c()
        c.columns(["a", "bb", "ccc"])
        assert "ccc" in io.get_output()

    def test_paginate_default_page_size(self):
        c, io = _make_c()
        c.paginate(["line-%d" % i for i in range(5)])
        assert "line-0" in io.get_output()

    def test_markdown_skips_empty_block(self):
        c, io = _make_c()
        c.markdown("\n\ntext")
        assert "text" in io.get_output()

    def test_markdown_code_fence_roundtrip(self):
        c, io = _make_c()
        c.markdown("```py\n\nprint(1)\n\n```")
        out = _visible(io.get_output())
        assert "print(1)" in out

    def test_markdown_heading_level_two(self):
        c, io = _make_c()
        c.markdown("## Sub")
        assert "Sub" in _visible(io.get_output())

    def test_download_bar_minute_eta(self):
        c, io = _make_c()
        c.download_bar("dl", current=5, total=10,
                       bytes_done=10_000, bytes_total=100_000, speed=30)
        out = _visible(io.get_output())
        assert "m" in out

    def test_console_has_readline(self):
        io = MemoryIO()
        c = Console(io, has_readline=True)
        assert c._has_readline is True

    def test_write_preserves_readline_buffer(self, monkeypatch):
        import readline as _rl
        monkeypatch.setattr(_rl, "get_line_buffer", lambda: "typing...")
        io = MemoryIO()
        c = Console(io, has_readline=True)
        c.write("line\n")
        out = io.get_output()
        assert "\033[s" in out
        assert "\033[u" in out

    def test_write_falls_back_when_readline_raises(self, monkeypatch):
        import readline as _rl

        def _boom():
            raise RuntimeError()

        monkeypatch.setattr(_rl, "get_line_buffer", _boom)
        io = MemoryIO()
        c = Console(io, has_readline=True)
        c.write("msg")
        assert "msg" in io.get_output()

    def test_write_with_empty_readline_buffer(self, monkeypatch):
        import readline as _rl
        monkeypatch.setattr(_rl, "get_line_buffer", lambda: "")
        io = MemoryIO()
        c = Console(io, has_readline=True)
        c.write("plain")
        assert "plain" in io.get_output()

    def test_no_color_import_path(self, monkeypatch):
        import importlib
        from domains.shell import console as _console
        monkeypatch.setenv("NO_COLOR", "1")
        importlib.reload(_console)
        assert _console._C_RED == ""
        assert _console._C_RESET == ""
        monkeypatch.delenv("NO_COLOR")
        importlib.reload(_console)
        assert _console._C_RED == "\033[31m"

    def test_console_without_readline(self, monkeypatch):
        import builtins
        import sys
        from domains.shell import console as _console
        _orig = builtins.__import__

        def _no_readline(name, *args, **kwargs):
            if name == "readline":
                raise ImportError("no readline")
            return _orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_readline)
        monkeypatch.delitem(sys.modules, "readline", raising=False)
        io = MemoryIO()
        c = _console.Console(io, has_readline=True)
        assert c._has_readline is False
