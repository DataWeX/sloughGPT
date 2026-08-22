"""
Tests for InteractivePrompt — arrow-key selection, confirm, and ask prompts.

Tests the fallback (non-TTY) path since raw terminal input requires a real
terminal.  The raw-mode codepaths are covered by the live TUI pty tests.
"""

from __future__ import annotations

import pytest

from domains.shell.io import MemoryIO
from domains.shell.interactive import InteractivePrompt, _RawKey, _read_raw_key, _KEY_UP, _KEY_DOWN, _KEY_ENTER, _KEY_ESC, _KEY_BACKSPACE, _KEY_CTRL_C, _KEY_CHAR


def _make_prompt(feeds: list[str] | None = None) -> tuple[InteractivePrompt, MemoryIO]:
    io = MemoryIO()
    if feeds:
        io.feed(*feeds)
    p = InteractivePrompt(io)
    # Force fallback mode (non-TTY)
    p._is_tty = False
    return p, io


# ── _RawKey ───────────────────────────────────────────────────────────────────

class TestRawKey:
    def test_repr_with_char(self):
        k = _RawKey(_KEY_CHAR, "a")
        assert "a" in repr(k)

    def test_repr_without_char(self):
        k = _RawKey(_KEY_ENTER)
        assert "enter" in repr(k)


# ── select fallback ───────────────────────────────────────────────────────────

class TestSelectFallback:
    def test_select_returns_chosen(self):
        p, io = _make_prompt(feeds=["2"])
        result = p.select("Pick:", ["a", "b", "c"])
        assert result == "b"

    def test_select_first_on_eof(self):
        p, io = _make_prompt()
        result = p.select("Pick:", ["x", "y"])
        assert result == "x"

    def test_select_invalid_then_valid(self):
        p, io = _make_prompt(feeds=["99", "1"])
        result = p.select("Pick:", ["only"])
        assert result == "only"

    def test_select_empty_options(self):
        p, io = _make_prompt()
        result = p.select("Pick:", [])
        assert result == ""

    def test_select_single_option(self):
        p, io = _make_prompt()
        result = p.select("Pick:", ["only-one"])
        assert result == "only-one"

    def test_select_displays_numbered_menu(self):
        p, io = _make_prompt(feeds=["1"])
        p.select("Choose:", ["alpha", "beta"])
        out = io.get_output()
        assert "1." in out
        assert "2." in out
        assert "alpha" in out
        assert "beta" in out


# ── confirm fallback ──────────────────────────────────────────────────────────

class TestConfirmFallback:
    def test_confirm_yes(self):
        p, io = _make_prompt(feeds=["y"])
        assert p.confirm("Continue?") is True

    def test_confirm_no(self):
        p, io = _make_prompt(feeds=["n"])
        assert p.confirm("Continue?") is False

    def test_confirm_default_true(self):
        p, io = _make_prompt(feeds=[""])
        assert p.confirm("Continue?", default=True) is True

    def test_confirm_default_false(self):
        p, io = _make_prompt(feeds=[""])
        assert p.confirm("Continue?", default=False) is False

    def test_confirm_eof_returns_default(self):
        p, io = _make_prompt()
        assert p.confirm("Continue?", default=True) is True

    def test_confirm_capital_yes(self):
        p, io = _make_prompt(feeds=["Y"])
        assert p.confirm("Continue?") is True

    def test_confirm_full_word(self):
        p, io = _make_prompt(feeds=["yes"])
        assert p.confirm("Continue?") is True


# ── ask fallback ──────────────────────────────────────────────────────────────

class TestAskFallback:
    def test_ask_returns_input(self):
        p, io = _make_prompt(feeds=["hello"])
        assert p.ask("Name:") == "hello"

    def test_ask_returns_default_on_empty(self):
        p, io = _make_prompt(feeds=[""])
        assert p.ask("Name:", default="fallback") == "fallback"

    def test_ask_eof_returns_default(self):
        p, io = _make_prompt()
        assert p.ask("Name:", default="fallback") == "fallback"

    def test_ask_strips_whitespace(self):
        p, io = _make_prompt(feeds=["  spaced  "])
        assert p.ask("Name:") == "spaced"

    def test_ask_displays_default_in_prompt(self):
        p, io = _make_prompt(feeds=[""])
        p.ask("Name:", default="fallback")
        out = io.get_output()
        assert "[fallback]" in out


# ── Integration with Console ──────────────────────────────────────────────────

class TestConsoleInteractiveIntegration:
    """Verify Console.select/confirm/ask delegate to InteractivePrompt."""

    def test_console_select_uses_interactive(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("2")
        c = Console(io, has_readline=False)
        result = c.select("Pick:", ["a", "b", "c"])
        assert result == "b"

    def test_console_confirm_uses_interactive(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("y")
        c = Console(io, has_readline=False)
        assert c.confirm("Go?") is True

    def test_console_ask_uses_interactive(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("test input")
        c = Console(io, has_readline=False)
        assert c.ask("Value:") == "test input"


# ── select_multi fallback ─────────────────────────────────────────────────────

class TestSelectMultiFallback:
    def test_select_multi_comma_separated(self):
        p, io = _make_prompt(feeds=["1,3"])
        result = p.select_multi("Pick:", ["a", "b", "c"])
        assert result == ["a", "c"]

    def test_select_multi_single(self):
        p, io = _make_prompt(feeds=["2"])
        result = p.select_multi("Pick:", ["a", "b", "c"])
        assert result == ["b"]

    def test_select_multi_empty_input(self):
        p, io = _make_prompt(feeds=[""])
        result = p.select_multi("Pick:", ["a", "b"])
        assert result == []

    def test_select_multi_eof(self):
        p, io = _make_prompt()
        result = p.select_multi("Pick:", ["a", "b"])
        assert result == []

    def test_select_multi_empty_options(self):
        p, io = _make_prompt()
        result = p.select_multi("Pick:", [])
        assert result == []

    def test_select_multi_invalid_numbers(self):
        p, io = _make_prompt(feeds=["99,abc,1"])
        result = p.select_multi("Pick:", ["a", "b", "c"])
        assert result == ["a"]

    def test_console_select_multi(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("1,2")
        c = Console(io, has_readline=False)
        result = c.select_multi("Pick:", ["x", "y", "z"])
        assert result == ["x", "y"]

    def test_select_multi_duplicates_preserved(self):
        p, io = _make_prompt(feeds=["1,1,2"])
        result = p.select_multi("Pick:", ["a", "b", "c"])
        assert result == ["a", "a", "b"]


# ── Module export ─────────────────────────────────────────────────────────────

class TestModuleExport:
    def test_interactive_prompt_importable(self):
        from domains.shell import InteractivePrompt
        assert InteractivePrompt is not None

    def test_interactive_prompt_in_all(self):
        from domains.shell import __all__
        assert "InteractivePrompt" in __all__


# ── select_with_details fallback ──────────────────────────────────────────────

class TestSelectWithDetailsFallback:
    def test_select_with_details_returns_chosen(self):
        p, io = _make_prompt(feeds=["2"])
        result = p.select_with_details("Pick:", ["a", "b"], ["desc1", "desc2"])
        assert result == "b"

    def test_select_with_details_single_option(self):
        p, io = _make_prompt()
        result = p.select_with_details("Pick:", ["only"], ["desc"])
        assert result == "only"

    def test_select_with_details_empty(self):
        p, io = _make_prompt()
        result = p.select_with_details("Pick:", [], [])
        assert result == ""

    def test_select_with_details_eof(self):
        p, io = _make_prompt()
        result = p.select_with_details("Pick:", ["a", "b"], ["d1", "d2"])
        assert result == "a"

    def test_console_select_with_details(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.select_with_details("Pick:", ["x", "y"], ["dx", "dy"])
        assert result == "x"


# ── confirm_multi fallback ────────────────────────────────────────────────────

class TestConfirmMultiFallback:
    def test_confirm_multi_yes_all(self):
        p, io = _make_prompt(feeds=["y"])
        result = p.confirm_multi("Confirm:", ["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_confirm_multi_no(self):
        p, io = _make_prompt(feeds=["n"])
        result = p.confirm_multi("Confirm:", ["a", "b"])
        assert result == []

    def test_confirm_multi_empty(self):
        p, io = _make_prompt()
        result = p.confirm_multi("Confirm:", [])
        assert result == []

    def test_confirm_multi_eof(self):
        p, io = _make_prompt()
        result = p.confirm_multi("Confirm:", ["a", "b"])
        assert result == []

    def test_confirm_multi_default_yes(self):
        p, io = _make_prompt(feeds=[""])
        result = p.confirm_multi("Confirm:", ["a"], default=True)
        assert result == ["a"]

    def test_console_confirm_multi(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("y")
        c = Console(io, has_readline=False)
        result = c.confirm_multi("Confirm:", ["x", "y"])
        assert result == ["x", "y"]


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_select_single_option_returns_immediately(self):
        p, io = _make_prompt()
        result = p.select("Pick:", ["only-one"])
        assert result == "only-one"

    def test_select_two_options(self):
        p, io = _make_prompt(feeds=["2"])
        result = p.select("Pick:", ["first", "second"])
        assert result == "second"

    def test_confirm_default_true_eof(self):
        p, io = _make_prompt()
        assert p.confirm("Go?", default=True) is True

    def test_confirm_default_false_eof(self):
        p, io = _make_prompt()
        assert p.confirm("Go?", default=False) is False

    def test_ask_default_used_on_empty(self):
        p, io = _make_prompt(feeds=[""])
        assert p.ask("Name:", default="fallback") == "fallback"

    def test_ask_empty_default_returns_empty(self):
        p, io = _make_prompt(feeds=[""])
        assert p.ask("Name:") == ""

    def test_select_multi_empty_input_returns_empty(self):
        p, io = _make_prompt(feeds=[""])
        assert p.select_multi("Pick:", ["a", "b"]) == []

    def test_progress_does_not_crash(self):
        p, io = _make_prompt()
        p.progress("test", 0, 100)
        p.progress("test", 50, 100)
        p.progress("test", 100, 100)

    def test_status_does_not_crash(self):
        p, io = _make_prompt()
        p.status("ok", "done")
        p.status("error", "failed")
        p.status("info", "note")


# ── select_with_preview fallback ──────────────────────────────────────────────

class TestSelectWithPreviewFallback:
    def test_select_with_preview_returns_chosen(self):
        p, io = _make_prompt(feeds=["2"])
        result = p.select_with_preview("Pick:", ["a", "b"], lambda x: f"preview: {x}")
        assert result == "b"

    def test_select_with_preview_single(self):
        p, io = _make_prompt()
        result = p.select_with_preview("Pick:", ["only"], lambda x: "p")
        assert result == "only"

    def test_select_with_preview_empty(self):
        p, io = _make_prompt()
        result = p.select_with_preview("Pick:", [], lambda x: "p")
        assert result == ""

    def test_select_with_preview_exception_in_fn(self):
        p, io = _make_prompt(feeds=["1"])
        def bad_fn(x):
            raise RuntimeError("boom")
        result = p.select_with_preview("Pick:", ["a", "b"], bad_fn)
        assert result == "a"


# ── edit fallback ─────────────────────────────────────────────────────────────

class TestEditFallback:
    def test_edit_returns_input(self):
        p, io = _make_prompt(feeds=["hello"])
        assert p.edit("Name:") == "hello"

    def test_edit_returns_default_on_empty(self):
        p, io = _make_prompt(feeds=[""])
        assert p.edit("Name:", default="fallback") == "fallback"

    def test_edit_eof_returns_default(self):
        p, io = _make_prompt()
        assert p.edit("Name:", default="fallback") == "fallback"

    def test_edit_with_valid_validator(self):
        p, io = _make_prompt(feeds=["valid"])
        v = lambda s: None if s.isalpha() else "Letters only"
        assert p.edit("Name:", validator=v) == "valid"

    def test_edit_with_invalid_validator_still_returns(self):
        p, io = _make_prompt(feeds=["123"])
        v = lambda s: None if s.isalpha() else "Letters only"
        assert p.edit("Name:", validator=v) == "123"

    def test_console_edit(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("test")
        c = Console(io, has_readline=False)
        assert c.edit("Value:") == "test"

    def test_console_edit_with_validator(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("abc")
        c = Console(io, has_readline=False)
        v = lambda s: None if s.isalpha() else "Letters only"
        assert c.edit("Value:", validator=v) == "abc"


# ── pager fallback ────────────────────────────────────────────────────────────

class TestPagerFallback:
    def test_pager_fallback_writes_content(self):
        p, io = _make_prompt()
        p.pager("line1\nline2\nline3")
        assert any("line1" in line for line in io._output)

    def test_pager_fallback_single_line(self):
        p, io = _make_prompt()
        p.pager("just one line", title="T")
        assert any("just one line" in line for line in io._output)

    def test_pager_empty(self):
        p, io = _make_prompt()
        p.pager("")
        assert len(io._output) == 1

    def test_console_pager(self):
        from domains.shell.console import Console
        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.pager("test content")
        assert any("test content" in line for line in io._output)

    def test_pager_does_not_crash(self):
        p, io = _make_prompt()
        p.pager("a\nb\nc\nd\ne", title="Test")


# ── tree_multi console method ────────────────────────────────────────────────

class TestTreeMultiConsole:
    def test_tree_multi_does_not_crash(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        data = {"root": ["a", "b", "c"]}
        result = c.tree_multi(data, "Pick:")
        assert isinstance(result, list)

    def test_tree_multi_renders_tree(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        data = {"root": ["a", "b"]}
        c.tree_multi(data, "Pick:")
        assert any("a" in line for line in io._output)
        assert any("b" in line for line in io._output)


# ── _terminal_width / _terminal_height ───────────────────────────────────────

class TestTerminalHelpers:
    def test_terminal_width_returns_int(self):
        from domains.shell.interactive import _terminal_width
        w = _terminal_width()
        assert isinstance(w, int)
        assert w > 0

    def test_terminal_height_returns_int(self):
        from domains.shell.interactive import _terminal_height
        h = _terminal_height()
        assert isinstance(h, int)
        assert h > 0


# ── Page Up/Down key decoding ───────────────────────────────────────────────

class TestPageKeys:
    def test_key_constants_exist(self):
        from domains.shell.interactive import _KEY_PAGE_UP, _KEY_PAGE_DOWN
        assert _KEY_PAGE_UP == "page_up"
        assert _KEY_PAGE_DOWN == "page_down"


# ── table display ─────────────────────────────────────────────────────────────

class TestTable:
    def test_table_does_not_crash(self):
        p, io = _make_prompt()
        p.table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"]])
        assert len(io._output) > 0

    def test_table_with_title(self):
        p, io = _make_prompt()
        p.table(["Col"], [["val"]], title="My Table")
        assert any("My Table" in line for line in io._output)

    def test_table_empty_rows(self):
        p, io = _make_prompt()
        p.table(["A", "B"], [])
        assert len(io._output) > 0

    def test_table_empty_headers_returns_early(self):
        p, io = _make_prompt()
        p.table([], [])
        assert len(io._output) == 0


# ── diff display ──────────────────────────────────────────────────────────────

class TestDiff:
    def test_diff_does_not_crash(self):
        p, io = _make_prompt()
        p.diff("left", ["a", "b"], "right", ["a", "c"])
        assert len(io._output) > 0

    def test_diff_with_title(self):
        p, io = _make_prompt()
        p.diff("L", ["x"], "R", ["y"], title="Comparison")
        assert any("Comparison" in line for line in io._output)

    def test_diff_identical(self):
        p, io = _make_prompt()
        p.diff("L", ["same"], "R", ["same"])
        assert len(io._output) > 0

    def test_diff_empty_sides(self):
        p, io = _make_prompt()
        p.diff("L", [], "R", [])
        assert len(io._output) > 0

    def test_console_diff(self):
        from domains.shell.console import Console
        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.diff("old", ["line1"], "new", ["line2"])
        assert len(io._output) > 0


# ── password fallback ─────────────────────────────────────────────────────────

class TestPasswordFallback:
    def test_password_returns_input(self):
        p, io = _make_prompt(feeds=["secret"])
        assert p.password("Password:") == "secret"

    def test_password_eof_returns_empty(self):
        p, io = _make_prompt()
        assert p.password("Password:") == ""

    def test_console_password(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("pass123")
        c = Console(io, has_readline=False)
        assert c.password("Password:") == "pass123"


# ── confirm_action fallback ───────────────────────────────────────────────────

class TestConfirmActionFallback:
    def test_confirm_action_default_no(self):
        p, io = _make_prompt(feeds=[""])
        assert p.confirm_action("Delete file") is False

    def test_confirm_action_y(self):
        p, io = _make_prompt(feeds=["y"])
        assert p.confirm_action("Delete file") is True

    def test_confirm_action_n(self):
        p, io = _make_prompt(feeds=["n"])
        assert p.confirm_action("Delete file") is False

    def test_confirm_action_danger(self):
        p, io = _make_prompt(feeds=["y"])
        assert p.confirm_action("Destroy data", danger=True) is True

    def test_console_confirm_action(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("y")
        c = Console(io, has_readline=False)
        assert c.confirm_action("Proceed") is True


# ── countdown fallback ────────────────────────────────────────────────────────

class TestCountdownFallback:
    def test_countdown_zero(self):
        p, io = _make_prompt()
        assert p.countdown(0) is True

    def test_countdown_writes_output(self):
        p, io = _make_prompt()
        p.countdown(2)
        assert len(io._output) > 0

    def test_console_countdown(self):
        from domains.shell.console import Console
        io = MemoryIO()
        c = Console(io, has_readline=False)
        assert c.countdown(1) is True


# ── banner ────────────────────────────────────────────────────────────────────

class TestBanner:
    def test_banner_does_not_crash(self):
        p, io = _make_prompt()
        p.banner("Hello World")
        assert len(io._output) > 0

    def test_banner_single_style(self):
        p, io = _make_prompt()
        p.banner("Test", style="single")
        assert any("Test" in line for line in io._output)

    def test_banner_thick_style(self):
        p, io = _make_prompt()
        p.banner("Thick", style="thick")
        assert any("Thick" in line for line in io._output)

    def test_banner_dashed_style(self):
        p, io = _make_prompt()
        p.banner("Dashed", style="dashed")
        assert any("Dashed" in line for line in io._output)

    def test_banner_unknown_style_fallback(self):
        p, io = _make_prompt()
        p.banner("Fallback", style="unknown")
        assert any("Fallback" in line for line in io._output)

    def test_console_banner(self):
        from domains.shell.console import Console
        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.banner("Console Banner")
        assert any("Console Banner" in line for line in io._output)


# ── slider fallback ───────────────────────────────────────────────────────────

class TestSliderFallback:
    def test_slider_default(self):
        p, io = _make_prompt(feeds=[""])
        assert p.slider("Volume:", 0, 100, 50) == 50

    def test_slider_returns_number(self):
        p, io = _make_prompt(feeds=["75"])
        result = p.slider("Level:", 0, 100, 50)
        assert isinstance(result, int)

    def test_console_slider(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        assert c.slider("Val:", 0, 10, 5) == 5


# ── toggle fallback ───────────────────────────────────────────────────────────

class TestToggleFallback:
    def test_toggle_default_false(self):
        p, io = _make_prompt(feeds=[""])
        assert p.toggle("Dark mode:") is False

    def test_toggle_y(self):
        p, io = _make_prompt(feeds=["y"])
        assert p.toggle("Dark mode:") is True

    def test_toggle_n(self):
        p, io = _make_prompt(feeds=["n"])
        assert p.toggle("Feature:", default=True) is False

    def test_console_toggle(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        assert c.toggle("Switch:") is True


# ── tag_input fallback ────────────────────────────────────────────────────────

class TestTagInputFallback:
    def test_tag_input_empty(self):
        p, io = _make_prompt(feeds=[""])
        assert p.tag_input("Tags:") == []

    def test_tag_input_comma_separated(self):
        p, io = _make_prompt(feeds=["a, b, c"])
        result = p.tag_input("Tags:")
        assert result == ["a", "b", "c"]

    def test_tag_input_with_defaults(self):
        p, io = _make_prompt(feeds=[""])
        result = p.tag_input("Tags:", defaults=["existing"])
        assert result == ["existing"]

    def test_console_tag_input(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("x, y")
        c = Console(io, has_readline=False)
        assert c.tag_input("Tags:") == ["x", "y"]


# ── select_tree fallback ──────────────────────────────────────────────────────

class TestSelectTreeFallback:
    def test_select_tree_returns_leaf(self):
        p, io = _make_prompt(feeds=["1"])
        tree = {"root": ["a", "b", "c"]}
        result = p.select_tree("Pick:", tree)
        assert result in ["a", "b", "c"]

    def test_select_tree_nested(self):
        p, io = _make_prompt(feeds=["2"])
        tree = {"pkg": {"src": ["main.py", "utils.py"]}}
        result = p.select_tree("Pick:", tree)
        assert result in ["main.py", "utils.py"]

    def test_select_tree_empty(self):
        p, io = _make_prompt(feeds=[""])
        assert p.select_tree("Pick:", {}) is None

    def test_console_select_tree(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.select_tree("Pick:", {"r": ["x", "y"]})
        assert result in ["x", "y"]


# ── spin_wait fallback ────────────────────────────────────────────────────────

class TestSpinWaitFallback:
    def test_spin_wait_immediate(self):
        p, io = _make_prompt()
        assert p.spin_wait("Loading...", lambda: True) is True

    def test_spin_wait_with_retries(self):
        p, io = _make_prompt()
        counter = [0]
        def check():
            counter[0] += 1
            return counter[0] >= 3
        assert p.spin_wait("Working...", check, interval=0) is True
        assert counter[0] >= 3

    def test_spin_wait_timeout(self):
        p, io = _make_prompt()
        result = p.spin_wait("Wait...", lambda: False, interval=0, timeout=0.05)
        assert result is False

    def test_console_spin_wait(self):
        from domains.shell.console import Console
        io = MemoryIO()
        c = Console(io, has_readline=False)
        assert c.spin_wait("Done", lambda: True) is True


# ── confirm_dangerous fallback ────────────────────────────────────────────────

class TestConfirmDangerousFallback:
    def test_confirm_dangerous_correct_phrase(self):
        p, io = _make_prompt(feeds=["yes, I am sure"])
        assert p.confirm_dangerous("Delete everything") is True

    def test_confirm_dangerous_wrong_phrase(self):
        p, io = _make_prompt(feeds=["nope"])
        assert p.confirm_dangerous("Destroy data") is False

    def test_confirm_dangerous_case_insensitive(self):
        p, io = _make_prompt(feeds=["Yes, I Am Sure"])
        assert p.confirm_dangerous("Action") is True

    def test_confirm_dangerous_custom_phrase(self):
        p, io = _make_prompt(feeds=["confirm"])
        assert p.confirm_dangerous("Drop DB", phrase="confirm") is True

    def test_confirm_dangerous_empty(self):
        p, io = _make_prompt(feeds=[""])
        assert p.confirm_dangerous("Action") is False

    def test_console_confirm_dangerous(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("yes, I am sure")
        c = Console(io, has_readline=False)
        assert c.confirm_dangerous("Nuke") is True


# ── file_browser fallback ─────────────────────────────────────────────────────

class TestFileBrowserFallback:
    def test_file_browser_no_tty(self):
        p, io = _make_prompt(feeds=["1"])
        result = p.file_browser("Pick file:", "/tmp")
        assert result is None or isinstance(result, str)

    def test_file_browser_nonexistent(self):
        p, io = _make_prompt(feeds=[""])
        result = p.file_browser("Pick:", "/nonexistent")
        assert result is None or isinstance(result, str)

    def test_console_file_browser(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.file_browser("Pick:", "/tmp")
        assert result is None or isinstance(result, str)


# ── progress_step ─────────────────────────────────────────────────────────────

class TestProgressStep:
    def test_progress_step_does_not_crash(self):
        p, io = _make_prompt()
        p.progress_step(["Build", "Test", "Deploy"], 1)
        assert len(io._output) > 0

    def test_progress_step_done(self):
        p, io = _make_prompt()
        p.progress_step(["A", "B"], 0, done=True)
        assert len(io._output) > 0

    def test_progress_step_first(self):
        p, io = _make_prompt()
        p.progress_step(["X", "Y", "Z"], 0)
        assert len(io._output) == 3

    def test_console_progress_step(self):
        from domains.shell.console import Console
        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.progress_step(["Step 1", "Step 2"], 0)
        assert len(io._output) == 2


# ── multi_choice fallback ─────────────────────────────────────────────────────

class TestMultiChoiceFallback:
    def test_multi_choice_empty(self):
        p, io = _make_prompt(feeds=[""])
        assert p.multi_choice("Pick:", []) == []

    def test_multi_choice_returns_list(self):
        p, io = _make_prompt(feeds=[""])
        result = p.multi_choice("Pick:", ["a", "b", "c"])
        assert isinstance(result, list)

    def test_multi_choice_with_defaults(self):
        p, io = _make_prompt(feeds=[""])
        result = p.multi_choice("Pick:", ["a", "b"], defaults=[0])
        assert isinstance(result, list)

    def test_console_multi_choice(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.multi_choice("Pick:", ["x", "y"])
        assert isinstance(result, list)


# ── date_picker fallback ──────────────────────────────────────────────────────

class TestDatePickerFallback:
    def test_date_picker_returns_string(self):
        p, io = _make_prompt(feeds=[""])
        result = p.date_picker("Date:")
        assert isinstance(result, str)
        assert len(result) == 10  # YYYY-MM-DD

    def test_date_picker_with_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.date_picker("Date:", default="2025-06-15")
        assert result == "2025-06-15"

    def test_console_date_picker(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.date_picker("Pick date:")
        assert isinstance(result, str)


# ── color_picker fallback ─────────────────────────────────────────────────────

class TestColorPickerFallback:
    def test_color_picker_returns_hex(self):
        p, io = _make_prompt(feeds=[""])
        result = p.color_picker("Color:")
        assert result.startswith("#")
        assert len(result) == 7

    def test_color_picker_with_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.color_picker("Color:", default="#ff0000")
        assert result == "#ff0000"

    def test_color_picker_short_hex(self):
        p, io = _make_prompt()
        r, g, b = p._hex_to_rgb("#abc")
        assert r == 0xaa
        assert g == 0xbb
        assert b == 0xcc

    def test_console_color_picker(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.color_picker("Pick:")
        assert result.startswith("#")


# ── confirm_timeout fallback ──────────────────────────────────────────────────

class TestConfirmTimeoutFallback:
    def test_confirm_timeout_default(self):
        p, io = _make_prompt(feeds=[""])
        assert p.confirm_timeout("Proceed?") is True

    def test_confirm_timeout_custom_default(self):
        p, io = _make_prompt(feeds=[""])
        assert p.confirm_timeout("Proceed?", default=False) is False

    def test_confirm_timeout_y(self):
        p, io = _make_prompt(feeds=["y"])
        assert p.confirm_timeout("Go?") is True

    def test_confirm_timeout_n(self):
        p, io = _make_prompt(feeds=["n"])
        assert p.confirm_timeout("Go?") is False

    def test_console_confirm_timeout(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("y")
        c = Console(io, has_readline=False)
        assert c.confirm_timeout("OK?") is True


# ── spin_until fallback ───────────────────────────────────────────────────────

class TestSpinUntilFallback:
    def test_spin_until_immediate(self):
        p, io = _make_prompt()
        result = p.spin_until("Loading...", lambda: 42, lambda x: x == 42)
        assert result == 42

    def test_spin_until_timeout(self):
        p, io = _make_prompt()
        result = p.spin_until("Wait...", lambda: None, lambda x: x is not None,
                              interval=0, timeout=0.05)
        assert result is None

    def test_spin_until_retries(self):
        p, io = _make_prompt()
        counter = [0]
        def gen():
            counter[0] += 1
            return counter[0]
        result = p.spin_until("Work...", gen, lambda x: x >= 3, interval=0)
        assert result == 3

    def test_console_spin_until(self):
        from domains.shell.console import Console
        io = MemoryIO()
        c = Console(io, has_readline=False)
        result = c.spin_until("Done", lambda: "ok", lambda x: x == "ok")
        assert result == "ok"


# ── progress_multi ────────────────────────────────────────────────────────────

class TestProgressMulti:
    def test_progress_multi_does_not_crash(self):
        p, io = _make_prompt()
        p.progress_multi([("Build", 50, 100), ("Test", 0, 10)])
        assert len(io._output) == 2

    def test_progress_multi_zero_total(self):
        p, io = _make_prompt()
        p.progress_multi([("Task", 0, 0)])
        assert len(io._output) == 1

    def test_progress_multi_complete(self):
        p, io = _make_prompt()
        p.progress_multi([("Done", 100, 100)])
        assert len(io._output) == 1

    def test_console_progress_multi(self):
        from domains.shell.console import Console
        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.progress_multi([("A", 10, 20), ("B", 5, 5)])
        assert len(io._output) == 2


# ── time_picker fallback ──────────────────────────────────────────────────────

class TestTimePickerFallback:
    def test_time_picker_returns_string(self):
        p, io = _make_prompt(feeds=[""])
        result = p.time_picker("Time:")
        assert isinstance(result, str)
        assert ":" in result

    def test_time_picker_with_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.time_picker("Time:", default="03:30 PM")
        assert result == "03:30 PM"

    def test_console_time_picker(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.time_picker("Pick time:")
        assert isinstance(result, str)


# ── progress_eta ──────────────────────────────────────────────────────────────

class TestProgressETA:
    def test_progress_eta_does_not_crash(self):
        p, io = _make_prompt()
        p.progress_eta("Build", 50, 100, elapsed=10.0)
        assert len(io._output) > 0

    def test_progress_eta_no_elapsed(self):
        p, io = _make_prompt()
        p.progress_eta("Test", 0, 100)
        assert len(io._output) > 0

    def test_progress_eta_complete(self):
        p, io = _make_prompt()
        p.progress_eta("Done", 100, 100, elapsed=5.0)
        assert len(io._output) > 0

    def test_progress_eta_zero(self):
        p, io = _make_prompt()
        p.progress_eta("Wait", 0, 100, elapsed=0)
        assert len(io._output) > 0

    def test_console_progress_eta(self):
        from domains.shell.console import Console
        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.progress_eta("Task", 30, 60, elapsed=3.0)
        assert len(io._output) > 0


# ── select_with_search fallback ───────────────────────────────────────────────

class TestSelectWithSearchFallback:
    def test_select_with_search_returns_chosen(self):
        p, io = _make_prompt(feeds=["2"])
        result = p.select_with_search("Pick:", ["alpha", "beta", "gamma"])
        assert result == "beta"

    def test_select_with_search_single(self):
        p, io = _make_prompt()
        result = p.select_with_search("Pick:", ["only-one"])
        assert result == "only-one"

    def test_select_with_search_empty(self):
        p, io = _make_prompt()
        result = p.select_with_search("Pick:", [])
        assert result == ""

    def test_console_select_with_search(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.select_with_search("Pick:", ["x", "y", "z"])
        assert result in ["x", "y", "z"]


# ── table_select fallback ─────────────────────────────────────────────────────

class TestTableSelectFallback:
    def test_table_select_empty(self):
        p, io = _make_prompt()
        assert p.table_select(["A"], []) is None

    def test_table_select_returns_index(self):
        p, io = _make_prompt(feeds=[""])
        result = p.table_select(["Name"], [["a"], ["b"]])
        assert isinstance(result, int)

    def test_table_select_no_tty(self):
        p, io = _make_prompt(feeds=[""])
        result = p.table_select(["Col"], [["x"]])
        assert result == 0

    def test_console_table_select(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.table_select(["H"], [["a"], ["b"]])
        assert isinstance(result, int)


# ── year_picker fallback ──────────────────────────────────────────────────────

class TestYearPickerFallback:
    def test_year_picker_returns_int(self):
        p, io = _make_prompt(feeds=[""])
        result = p.year_picker("Year:")
        assert isinstance(result, int)
        assert 1900 <= result <= 2100

    def test_year_picker_with_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.year_picker("Year:", default=2025)
        assert result == 2025

    def test_console_year_picker(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.year_picker("Year:", default=2030)
        assert result == 2030


# ── month_picker fallback ─────────────────────────────────────────────────────

class TestMonthPickerFallback:
    def test_month_picker_returns_int(self):
        p, io = _make_prompt(feeds=[""])
        result = p.month_picker("Month:")
        assert isinstance(result, int)
        assert 1 <= result <= 12

    def test_month_picker_with_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.month_picker("Month:", default=6)
        assert result == 6

    def test_console_month_picker(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.month_picker("Month:", default=3)
        assert result == 3


# ── confirm_list fallback ─────────────────────────────────────────────────────

class TestConfirmListFallback:
    def test_confirm_list_empty(self):
        p, io = _make_prompt(feeds=[""])
        assert p.confirm_list("Confirm:", []) == []

    def test_confirm_list_returns_list(self):
        p, io = _make_prompt(feeds=[""])
        result = p.confirm_list("Confirm:", ["a", "b"])
        assert isinstance(result, list)

    def test_confirm_list_y(self):
        p, io = _make_prompt(feeds=["y"])
        result = p.confirm_list("Confirm:", ["item1"], default=False)
        assert isinstance(result, list)

    def test_console_confirm_list(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.confirm_list("Confirm:", ["x", "y"])
        assert isinstance(result, list)


# ── table_edit fallback ───────────────────────────────────────────────────────

class TestTableEditFallback:
    def test_table_edit_empty(self):
        p, io = _make_prompt()
        assert p.table_edit(["A"], []) == []

    def test_table_edit_returns_rows(self):
        p, io = _make_prompt(feeds=[""])
        result = p.table_edit(["Name"], [["value"]])
        assert result == [["value"]]

    def test_table_edit_no_tty(self):
        p, io = _make_prompt(feeds=[""])
        result = p.table_edit(["Col"], [["x"]])
        assert result == [["x"]]

    def test_console_table_edit(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.table_edit(["H"], [["val"]])
        assert result == [["val"]]


# ── duration_picker fallback ──────────────────────────────────────────────────

class TestDurationPickerFallback:
    def test_duration_picker_returns_int(self):
        p, io = _make_prompt(feeds=[""])
        result = p.duration_picker("Duration:")
        assert isinstance(result, int)
        assert result >= 0

    def test_duration_picker_with_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.duration_picker("Duration:", default=120)
        assert result == 120

    def test_duration_picker_zero(self):
        p, io = _make_prompt(feeds=[""])
        result = p.duration_picker("Duration:", default=0)
        assert result == 0

    def test_console_duration_picker(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.duration_picker("Time:", default=3600)
        assert result == 3600


# ── confirm_text fallback ─────────────────────────────────────────────────────

class TestConfirmTextFallback:
    def test_confirm_text_correct(self):
        p, io = _make_prompt(feeds=["deploy"])
        assert p.confirm_text("Confirm:", "deploy") is True

    def test_confirm_text_wrong(self):
        p, io = _make_prompt(feeds=["wrong"])
        assert p.confirm_text("Confirm:", "deploy") is False

    def test_confirm_text_empty(self):
        p, io = _make_prompt(feeds=[""])
        assert p.confirm_text("Confirm:", "yes") is False

    def test_confirm_text_with_hint(self):
        p, io = _make_prompt(feeds=["go"])
        assert p.confirm_text("Proceed?", "go", hint="Type 'go' to proceed") is True

    def test_console_confirm_text(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("ok")
        c = Console(io, has_readline=False)
        assert c.confirm_text("Confirm:", "ok") is True


# ── table_sort fallback ───────────────────────────────────────────────────────

class TestTableSortFallback:
    def test_table_sort_empty(self):
        p, io = _make_prompt()
        assert p.table_sort(["A"], []) == []

    def test_table_sort_returns_rows(self):
        p, io = _make_prompt(feeds=[""])
        result = p.table_sort(["Name"], [["z"], ["a"], ["m"]])
        assert len(result) == 3

    def test_table_sort_no_tty(self):
        p, io = _make_prompt(feeds=[""])
        result = p.table_sort(["Col"], [["c"], ["a"], ["b"]])
        assert len(result) == 3

    def test_console_table_sort(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.table_sort(["H"], [["b"], ["a"]])
        assert len(result) == 2


# ── notify ────────────────────────────────────────────────────────────────────

class TestNotify:
    def test_notify_info(self):
        p, io = _make_prompt()
        p.notify("Title", "message", level="info")
        assert len(io._output) > 0
        assert any("Title" in line for line in io._output)

    def test_notify_success(self):
        p, io = _make_prompt()
        p.notify("Done", "completed", level="success")
        assert len(io._output) > 0

    def test_notify_warn(self):
        p, io = _make_prompt()
        p.notify("Warning", "low space", level="warn")
        assert len(io._output) > 0

    def test_notify_error(self):
        p, io = _make_prompt()
        p.notify("Error", "failed", level="error")
        assert len(io._output) > 0

    def test_notify_no_message(self):
        p, io = _make_prompt()
        p.notify("Title", "")
        assert len(io._output) > 0

    def test_console_notify(self):
        from domains.shell.console import Console
        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.notify("Info", "test", level="info")
        assert len(io._output) > 0


# ── week_picker fallback ──────────────────────────────────────────────────────

class TestWeekPickerFallback:
    def test_week_picker_returns_int(self):
        p, io = _make_prompt(feeds=[""])
        result = p.week_picker("Week:")
        assert isinstance(result, int)
        assert 1 <= result <= 52

    def test_week_picker_with_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.week_picker("Week:", default=20)
        assert result == 20

    def test_console_week_picker(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.week_picker("Week:", default=10)
        assert result == 10


# ── quarter_picker fallback ───────────────────────────────────────────────────

class TestQuarterPickerFallback:
    def test_quarter_picker_returns_int(self):
        p, io = _make_prompt(feeds=[""])
        result = p.quarter_picker("Quarter:")
        assert isinstance(result, int)
        assert 1 <= result <= 4

    def test_quarter_picker_with_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.quarter_picker("Quarter:", default=3)
        assert result == 3

    def test_console_quarter_picker(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.quarter_picker("Quarter:", default=2)
        assert result == 2


# ── confirm_delete ────────────────────────────────────────────────────────────

class TestConfirmDeleteFallback:
    def test_confirm_delete_wrong(self):
        p, io = _make_prompt(feeds=["no"])
        assert p.confirm_delete("file.txt") is False

    def test_confirm_delete_right(self):
        p, io = _make_prompt(feeds=["delete"])
        assert p.confirm_delete("file.txt") is True

    def test_confirm_delete_count(self):
        p, io = _make_prompt(feeds=["delete"])
        assert p.confirm_delete("files", count=5) is True

    def test_console_confirm_delete(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("delete")
        c = Console(io, has_readline=False)
        assert c.confirm_delete("data.csv") is True


# ── confirm_overwrite ─────────────────────────────────────────────────────────

class TestConfirmOverwriteFallback:
    def test_confirm_overwrite_wrong(self):
        p, io = _make_prompt(feeds=["no"])
        assert p.confirm_overwrite("/path/file.txt") is False

    def test_confirm_overwrite_right(self):
        p, io = _make_prompt(feeds=["overwrite"])
        assert p.confirm_overwrite("/path/file.txt") is True

    def test_console_confirm_overwrite(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("overwrite")
        c = Console(io, has_readline=False)
        assert c.confirm_overwrite("out.txt") is True


# ── progress_ring ─────────────────────────────────────────────────────────────

class TestProgressRing:
    def test_progress_ring_does_not_crash(self):
        p, io = _make_prompt()
        p.progress_ring("Build", 50, 100)
        assert len(io._output) > 0

    def test_progress_ring_zero(self):
        p, io = _make_prompt()
        p.progress_ring("Test", 0, 100)
        assert len(io._output) > 0

    def test_progress_ring_complete(self):
        p, io = _make_prompt()
        p.progress_ring("Done", 100, 100)
        assert len(io._output) > 0

    def test_console_progress_ring(self):
        from domains.shell.console import Console
        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.progress_ring("Task", 75, 100)
        assert len(io._output) > 0


# ── timezone_picker fallback ──────────────────────────────────────────────────

class TestTimezonePickerFallback:
    def test_timezone_picker_returns_string(self):
        p, io = _make_prompt(feeds=[""])
        result = p.timezone_picker("TZ:")
        assert isinstance(result, str)

    def test_timezone_picker_with_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.timezone_picker("TZ:", default="US/Eastern")
        assert result == "US/Eastern"

    def test_console_timezone_picker(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.timezone_picker("TZ:")
        assert isinstance(result, str)


# ── currency_picker fallback ──────────────────────────────────────────────────

class TestCurrencyPickerFallback:
    def test_currency_picker_returns_3letters(self):
        p, io = _make_prompt(feeds=[""])
        result = p.currency_picker("Currency:")
        assert isinstance(result, str)
        assert len(result) == 3

    def test_currency_picker_with_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.currency_picker("Currency:", default="EUR")
        assert result == "EUR"

    def test_console_currency_picker(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.currency_picker("Currency:")
        assert isinstance(result, str)


# ── language_picker fallback ──────────────────────────────────────────────────

class TestLanguagePickerFallback:
    def test_language_picker_returns_2letters(self):
        p, io = _make_prompt(feeds=[""])
        result = p.language_picker("Language:")
        assert isinstance(result, str)
        assert len(result) == 2

    def test_language_picker_with_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.language_picker("Language:", default="es")
        assert result == "es"

    def test_console_language_picker(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.language_picker("Language:")
        assert isinstance(result, str)


# ── confirm_with_preview fallback ─────────────────────────────────────────────

class TestConfirmWithPreviewFallback:
    def test_confirm_with_preview_default(self):
        p, io = _make_prompt(feeds=[""])
        assert p.confirm_with_preview("Apply?", "diff output here") is False

    def test_confirm_with_preview_y(self):
        p, io = _make_prompt(feeds=["y"])
        assert p.confirm_with_preview("Apply?", "changes") is True

    def test_confirm_with_preview_n(self):
        p, io = _make_prompt(feeds=["n"])
        assert p.confirm_with_preview("Apply?", "changes", default=True) is False

    def test_console_confirm_with_preview(self):
        from domains.shell.console import Console
        io = MemoryIO()
        io.feed("y")
        c = Console(io, has_readline=False)
        assert c.confirm_with_preview("Apply?", "preview text") is True
