"""
Tests for InteractivePrompt — arrow-key selection, confirm, and ask prompts.

Tests the fallback (non-TTY) path since raw terminal input requires a real
terminal.  The raw-mode codepaths are covered by the live TUI pty tests.
"""

from __future__ import annotations

from domain.shell._internal.interactive import (
    _KEY_CHAR,
    _KEY_ENTER,
    InteractivePrompt,
    _RawKey,
)

# The raw-mode key vocabulary. `interactive` re-exports only the subset its own
# fallback paths reach for; the filter-box tests drive the raw paths, which see
# the full set.
from domain.shell._internal.interactive_term import (
    _CURSOR_UP,
    _ERASE_LINE,
    _KEY_BACKSPACE,
    _KEY_DELETE,
    _KEY_ENTER,
    _KEY_ESC,
    _KEY_HOME,
    _KEY_LEFT,
)
from domain.shell._internal.io import MemoryIO


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
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("2")
        c = Console(io, has_readline=False)
        result = c.select("Pick:", ["a", "b", "c"])
        assert result == "b"

    def test_console_confirm_uses_interactive(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("y")
        c = Console(io, has_readline=False)
        assert c.confirm("Go?") is True

    def test_console_ask_uses_interactive(self):
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell import InteractivePrompt

        assert InteractivePrompt is not None

    def test_interactive_prompt_in_all(self):
        from domain.shell import __all__

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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

        def v(s):
            return None if s.isalpha() else "Letters only"

        assert p.edit("Name:", validator=v) == "valid"

    def test_edit_with_invalid_validator_still_returns(self):
        p, io = _make_prompt(feeds=["123"])

        def v(s):
            return None if s.isalpha() else "Letters only"

        assert p.edit("Name:", validator=v) == "123"

    def test_console_edit(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("test")
        c = Console(io, has_readline=False)
        assert c.edit("Value:") == "test"

    def test_console_edit_with_validator(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("abc")
        c = Console(io, has_readline=False)

        def v(s):
            return None if s.isalpha() else "Letters only"

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        data = {"root": ["a", "b", "c"]}
        result = c.tree_multi(data, "Pick:")
        assert isinstance(result, list)

    def test_tree_multi_renders_tree(self):
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.interactive import _terminal_width

        w = _terminal_width()
        assert isinstance(w, int)
        assert w > 0

    def test_terminal_height_returns_int(self):
        from domain.shell._internal.interactive import _terminal_height

        h = _terminal_height()
        assert isinstance(h, int)
        assert h > 0


# ── Page Up/Down key decoding ───────────────────────────────────────────────


class TestPageKeys:
    def test_key_constants_exist(self):
        from domain.shell._internal.interactive import _KEY_PAGE_DOWN, _KEY_PAGE_UP

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.date_picker("Pick date:")
        assert isinstance(result, str)


# ── color_picker_rgb fallback ──────────────────────────────────────────────────


class TestColorPickerRgbFallback:
    def test_color_picker_rgb_returns_hex(self):
        p, io = _make_prompt(feeds=[""])
        result = p.color_picker_rgb("Color:")
        assert result.startswith("#")
        assert len(result) == 7

    def test_color_picker_rgb_with_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.color_picker_rgb("Color:", default="#ff0000")
        assert result == "#ff0000"

    def test_color_picker_short_hex(self):
        p, io = _make_prompt()
        r, g, b = p._hex_to_rgb("#abc")
        assert r == 0xAA
        assert g == 0xBB
        assert b == 0xCC

    def test_console_color_picker_rgb(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.color_picker_rgb("Pick:")
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
        from domain.shell._internal.console import Console

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
        result = p.spin_until(
            "Wait...", lambda: None, lambda x: x is not None, interval=0, timeout=0.05
        )
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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

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
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("y")
        c = Console(io, has_readline=False)
        assert c.confirm_with_preview("Apply?", "preview text") is True


# ── select_with_preview fallback ──────────────────────────────────────────────


class TestSelectWithPreviewFallback:
    def test_select_with_preview_returns_string(self):
        def preview(opt):
            return f"Preview for {opt}"

        p, io = _make_prompt(feeds=["1"])
        result = p.select_with_preview("Pick:", ["A", "B", "C"], preview)
        assert result in ["A", "B", "C"]

    def test_select_with_preview_with_default(self):
        def preview(opt):
            return f"Detail: {opt}"

        p, io = _make_prompt(feeds=[""])
        result = p.select_with_preview("Pick:", ["X", "Y"], preview)
        assert result in ["X", "Y"]

    def test_console_select_with_preview(self):
        def preview(opt):
            return f"Preview: {opt}"

        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("2")
        c = Console(io, has_readline=False)
        result = c.select_with_preview("Pick:", ["A", "B"], preview)
        assert result in ["A", "B"]


# ── progress_bar ──────────────────────────────────────────────────────────────


class TestProgressBar:
    def test_progress_bar_zero(self):
        p, io = _make_prompt()
        p.progress_bar("Build", 0, 100)
        assert len(io._output) > 0

    def test_progress_bar_half(self):
        p, io = _make_prompt()
        p.progress_bar("Build", 50, 100)
        assert len(io._output) > 0

    def test_progress_bar_full(self):
        p, io = _make_prompt()
        p.progress_bar("Done", 100, 100)
        assert len(io._output) > 0

    def test_progress_bar_custom_width(self):
        p, io = _make_prompt()
        p.progress_bar("Test", 30, 100, width=50)
        assert len(io._output) > 0

    def test_console_progress_bar(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.progress_bar("Task", 75, 100)
        assert len(io._output) > 0


# ── date_range_picker fallback ────────────────────────────────────────────────


class TestDateRangePickerFallback:
    def test_date_range_picker_returns_tuple(self):
        p, io = _make_prompt(feeds=["2025-01-01", "2025-12-31"])
        result = p.date_range_picker("Range:")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_date_range_picker_with_defaults(self):
        p, io = _make_prompt(feeds=["", ""])
        result = p.date_range_picker("Range:", default_start="2025-06-01", default_end="2025-06-30")
        assert result[0] == "2025-06-01"
        assert result[1] == "2025-06-30"

    def test_console_date_range_picker(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("2025-03-01")
        io.feed("2025-03-31")
        c = Console(io, has_readline=False)
        result = c.date_range_picker("Range:")
        assert isinstance(result, tuple)


# ── color_picker fallback ─────────────────────────────────────────────────────


class TestColorPickerFallback:
    def test_color_picker_returns_string(self):
        p, io = _make_prompt(feeds=[""])
        result = p.color_picker("Color:")
        assert isinstance(result, str)
        assert result.startswith("#")

    def test_color_picker_with_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.color_picker("Color:", default="#ff0000")
        assert result == "#ff0000"

    def test_console_color_picker(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.color_picker("Color:")
        assert isinstance(result, str)


# ── time_range_picker fallback ────────────────────────────────────────────────


class TestTimeRangePickerFallback:
    def test_time_range_picker_returns_tuple(self):
        p, io = _make_prompt(feeds=["09:00", "17:00"])
        result = p.time_range_picker("Range:")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_time_range_picker_with_defaults(self):
        p, io = _make_prompt(feeds=["", ""])
        result = p.time_range_picker("Range:", default_start="08:00", default_end="18:00")
        assert result[0] == "08:00"
        assert result[1] == "18:00"

    def test_console_time_range_picker(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("09:00")
        io.feed("17:00")
        c = Console(io, has_readline=False)
        result = c.time_range_picker("Range:")
        assert isinstance(result, tuple)


# ── number_range_picker fallback ──────────────────────────────────────────────


class TestNumberRangePickerFallback:
    def test_number_range_picker_returns_int(self):
        p, io = _make_prompt(feeds=[""])
        result = p.number_range_picker("Number:")
        assert isinstance(result, int)

    def test_number_range_picker_with_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.number_range_picker("Number:", default=42)
        assert result == 42

    def test_number_range_picker_custom_range(self):
        p, io = _make_prompt(feeds=[""])
        result = p.number_range_picker("Num:", min_val=10, max_val=50, default=25)
        assert result == 25

    def test_console_number_range_picker(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.number_range_picker("Number:", default=10)
        assert result == 10


# ── confirm_with_details fallback ─────────────────────────────────────────────


class TestConfirmWithDetailsFallback:
    def test_confirm_with_details_default(self):
        p, io = _make_prompt(feeds=[""])
        details = {"file": "test.txt", "size": "100KB"}
        assert p.confirm_with_details("Apply?", details) is False

    def test_confirm_with_details_y(self):
        p, io = _make_prompt(feeds=["y"])
        details = {"name": "John", "age": "30"}
        assert p.confirm_with_details("Save?", details) is True

    def test_confirm_with_details_n(self):
        p, io = _make_prompt(feeds=["n"])
        details = {"action": "delete"}
        assert p.confirm_with_details("Proceed?", details, default=True) is False

    def test_console_confirm_with_details(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("y")
        c = Console(io, has_readline=False)
        details = {"key": "value"}
        assert c.confirm_with_details("OK?", details) is True


# ── spinner_with_status fallback ──────────────────────────────────────────────


class TestSpinnerWithStatusFallback:
    def test_spinner_with_status_does_not_crash(self):
        p, io = _make_prompt()
        p.spinner_with_status("Loading", "please wait")
        assert len(io._output) > 0

    def test_spinner_with_status_contains_message(self):
        p, io = _make_prompt()
        p.spinner_with_status("Building", "step 1/5")
        output = io._output[0]
        assert "Building" in output

    def test_console_spinner_with_status(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.spinner_with_status("Working", "in progress")
        assert len(io._output) > 0


# ── select_with_filter fallback ───────────────────────────────────────────────


class TestSelectWithFilterFallback:
    def test_select_with_filter_returns_string(self):
        p, io = _make_prompt(feeds=["1"])
        result = p.select_with_filter("Pick:", ["A", "B", "C"])
        assert result in ["A", "B", "C"]

    def test_select_with_filter_with_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.select_with_filter("Pick:", ["X", "Y"])
        assert result in ["X", "Y"]

    def test_console_select_with_filter(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.select_with_filter("Pick:", ["A", "B"])
        assert result in ["A", "B"]


# ── confirm_with_preview_and_edit fallback ────────────────────────────────────


class TestConfirmWithPreviewAndEditFallback:
    def test_confirm_with_preview_and_edit_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.confirm_with_preview_and_edit("Apply?", "diff here")
        assert isinstance(result, tuple)
        assert result[0] is False

    def test_confirm_with_preview_and_edit_y(self):
        p, io = _make_prompt(feeds=["y"])
        result = p.confirm_with_preview_and_edit("Apply?", "changes")
        assert result[0] is True
        assert result[1] == "changes"

    def test_confirm_with_preview_and_edit_e(self):
        p, io = _make_prompt(feeds=["e", "edited text"])
        result = p.confirm_with_preview_and_edit("Apply?", "original")
        assert isinstance(result, tuple)
        assert result[1] in ["original", "edited text"]

    def test_console_confirm_with_preview_and_edit(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("y")
        c = Console(io, has_readline=False)
        result = c.confirm_with_preview_and_edit("Apply?", "preview")
        assert result[0] is True


# ── progress_bar_colored ──────────────────────────────────────────────────────


class TestProgressBarColored:
    def test_progress_bar_colored_zero(self):
        p, io = _make_prompt()
        p.progress_bar_colored("Build", 0, 100)
        assert len(io._output) > 0

    def test_progress_bar_colored_half(self):
        p, io = _make_prompt()
        p.progress_bar_colored("Build", 50, 100)
        assert len(io._output) > 0

    def test_progress_bar_colored_full(self):
        p, io = _make_prompt()
        p.progress_bar_colored("Done", 100, 100)
        assert len(io._output) > 0

    def test_console_progress_bar_colored(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.progress_bar_colored("Task", 75, 100)
        assert len(io._output) > 0


# ── spinner_with_progress fallback ────────────────────────────────────────────


class TestSpinnerWithProgressFallback:
    def test_spinner_with_progress_does_not_crash(self):
        p, io = _make_prompt()
        p.spinner_with_progress("Loading", 50, 100)
        assert len(io._output) > 0

    def test_spinner_with_progress_contains_message(self):
        p, io = _make_prompt()
        p.spinner_with_progress("Building", 25, 100)
        output = io._output[0]
        assert "Building" in output

    def test_console_spinner_with_progress(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.spinner_with_progress("Working", 75, 100)
        assert len(io._output) > 0


# ── select_with_icons fallback ────────────────────────────────────────────────


class TestSelectWithIconsFallback:
    def test_select_with_icons_returns_string(self):
        p, io = _make_prompt(feeds=["1"])
        options = [("\u2605", "Favorite"), ("\u2606", "Not Favorite")]
        result = p.select_with_icons("Pick:", options)
        assert result in ["Favorite", "Not Favorite"]

    def test_select_with_icons_with_default(self):
        p, io = _make_prompt(feeds=[""])
        options = [("\u2714", "Yes"), ("\u2718", "No")]
        result = p.select_with_icons("Pick:", options)
        assert result in ["Yes", "No"]

    def test_console_select_with_icons(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        options = [("\u2605", "Star"), ("\u2606", "No Star")]
        result = c.select_with_icons("Pick:", options)
        assert result in ["Star", "No Star"]


# ── confirm_with_warning fallback ─────────────────────────────────────────────


class TestConfirmWithWarningFallback:
    def test_confirm_with_warning_default(self):
        p, io = _make_prompt(feeds=[""])
        assert p.confirm_with_warning("Delete?", "This cannot be undone") is False

    def test_confirm_with_warning_y(self):
        p, io = _make_prompt(feeds=["y"])
        assert p.confirm_with_warning("Proceed?", "Warning!") is True

    def test_confirm_with_warning_n(self):
        p, io = _make_prompt(feeds=["n"])
        assert p.confirm_with_warning("Continue?", "Caution", default=True) is False

    def test_console_confirm_with_warning(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("y")
        c = Console(io, has_readline=False)
        assert c.confirm_with_warning("Delete?", "Irreversible") is True


# ── progress_bar_eta ──────────────────────────────────────────────────────────


class TestProgressBarETA:
    def test_progress_bar_eta_zero(self):
        p, io = _make_prompt()
        p.progress_bar_eta("Build", 0, 100, 0.0)
        assert len(io._output) > 0

    def test_progress_bar_eta_with_time(self):
        p, io = _make_prompt()
        p.progress_bar_eta("Build", 50, 100, 10.0)
        assert len(io._output) > 0

    def test_progress_bar_eta_full(self):
        p, io = _make_prompt()
        p.progress_bar_eta("Done", 100, 100, 20.0)
        assert len(io._output) > 0

    def test_console_progress_bar_eta(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.progress_bar_eta("Task", 75, 100, 15.0)
        assert len(io._output) > 0


# ── spinner_with_dots fallback ────────────────────────────────────────────────


class TestSpinnerWithDotsFallback:
    def test_spinner_with_dots_does_not_crash(self):
        p, io = _make_prompt()
        p.spinner_with_dots("Loading")
        assert len(io._output) > 0

    def test_spinner_with_dots_contains_message(self):
        p, io = _make_prompt()
        p.spinner_with_dots("Building")
        output = io._output[0]
        assert "Building" in output

    def test_console_spinner_with_dots(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.spinner_with_dots("Working")
        assert len(io._output) > 0


# ── select_with_pagination fallback ───────────────────────────────────────────


class TestSelectWithPaginationFallback:
    def test_select_with_pagination_returns_string(self):
        p, io = _make_prompt(feeds=["1"])
        options = [f"Item {i}" for i in range(25)]
        result = p.select_with_pagination("Pick:", options, page_size=10)
        assert result in options

    def test_select_with_pagination_small_list(self):
        p, io = _make_prompt(feeds=[""])
        options = ["A", "B", "C"]
        result = p.select_with_pagination("Pick:", options, page_size=10)
        assert result in options

    def test_console_select_with_pagination(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        options = [f"Item {i}" for i in range(20)]
        result = c.select_with_pagination("Pick:", options, page_size=5)
        assert result in options


# ── select_with_search_and_preview fallback ───────────────────────────────────


class TestSelectWithSearchAndPreviewFallback:
    def test_select_with_search_and_preview_returns_string(self):
        def preview(opt):
            return f"Preview for {opt}"

        p, io = _make_prompt(feeds=["1"])
        options = ["Apple", "Banana", "Cherry"]
        result = p.select_with_search_and_preview("Pick:", options, preview)
        assert result in options

    def test_select_with_search_and_preview_empty(self):
        def preview(opt):
            return f"Detail: {opt}"

        p, io = _make_prompt(feeds=[""])
        result = p.select_with_search_and_preview("Pick:", ["X"], preview)
        assert result == "X"

    def test_console_select_with_search_and_preview(self):
        def preview(opt):
            return f"Info: {opt}"

        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.select_with_search_and_preview("Pick:", ["A", "B"], preview)
        assert result in ["A", "B"]


# ── progress_bar_with_status ──────────────────────────────────────────────────


class TestProgressBarWithStatus:
    def test_progress_bar_with_status_zero(self):
        p, io = _make_prompt()
        p.progress_bar_with_status("Build", 0, 100, "starting")
        assert len(io._output) > 0

    def test_progress_bar_with_status_half(self):
        p, io = _make_prompt()
        p.progress_bar_with_status("Build", 50, 100, "in progress")
        assert len(io._output) > 0

    def test_progress_bar_with_status_full(self):
        p, io = _make_prompt()
        p.progress_bar_with_status("Done", 100, 100, "complete")
        assert len(io._output) > 0

    def test_progress_bar_with_status_no_status(self):
        p, io = _make_prompt()
        p.progress_bar_with_status("Test", 25, 100)
        assert len(io._output) > 0

    def test_console_progress_bar_with_status(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.progress_bar_with_status("Task", 75, 100, "almost done")
        assert len(io._output) > 0


# ── spinner_with_eta fallback ─────────────────────────────────────────────────


class TestSpinnerWithEtaFallback:
    def test_spinner_with_eta_does_not_crash(self):
        p, io = _make_prompt()
        p.spinner_with_eta("Loading", 5.0)
        assert len(io._output) > 0

    def test_spinner_with_eta_with_progress(self):
        p, io = _make_prompt()
        p.spinner_with_eta("Building", 10.0, progress=0.5)
        assert len(io._output) > 0

    def test_spinner_with_eta_contains_message(self):
        p, io = _make_prompt()
        p.spinner_with_eta("Working", 2.0)
        output = io._output[0]
        assert "Working" in output

    def test_console_spinner_with_eta(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.spinner_with_eta("Task", 3.0, progress=0.25)
        assert len(io._output) > 0


# ── select_with_grouping fallback ─────────────────────────────────────────────


class TestSelectWithGroupingFallback:
    def test_select_with_grouping_returns_string(self):
        p, io = _make_prompt(feeds=["1"])
        groups = {"Fruits": ["Apple", "Banana"], "Veggies": ["Carrot"]}
        result = p.select_with_grouping("Pick:", groups)
        assert result in ["Apple", "Banana", "Carrot"]

    def test_select_with_grouping_with_default(self):
        p, io = _make_prompt(feeds=[""])
        groups = {"A": ["X", "Y"]}
        result = p.select_with_grouping("Pick:", groups, default="Y")
        assert result in ["X", "Y"]

    def test_console_select_with_grouping(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        groups = {"Cat1": ["A", "B"], "Cat2": ["C"]}
        result = c.select_with_grouping("Pick:", groups)
        assert result in ["A", "B", "C"]


# ── multi_select_with_preview fallback ────────────────────────────────────────


class TestMultiSelectWithPreviewFallback:
    def test_multi_select_with_preview_returns_list(self):
        def preview(opt):
            return f"Preview: {opt}"

        p, io = _make_prompt(feeds=[""])
        result = p.multi_select_with_preview("Pick:", ["A", "B", "C"], preview)
        assert isinstance(result, list)

    def test_multi_select_with_preview_empty(self):
        def preview(opt):
            return f"Detail: {opt}"

        p, io = _make_prompt(feeds=[""])
        result = p.multi_select_with_preview("Pick:", ["X"], preview)
        assert isinstance(result, list)

    def test_console_multi_select_with_preview(self):
        def preview(opt):
            return f"Info: {opt}"

        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.multi_select_with_preview("Pick:", ["A", "B"], preview)
        assert isinstance(result, list)


# ── progress_bar_indeterminate fallback ───────────────────────────────────────


class TestProgressBarIndeterminateFallback:
    def test_progress_bar_indeterminate_does_not_crash(self):
        p, io = _make_prompt()
        p.progress_bar_indeterminate("Loading")
        assert len(io._output) > 0

    def test_progress_bar_indeterminate_with_status(self):
        p, io = _make_prompt()
        p.progress_bar_indeterminate("Building", "please wait")
        assert len(io._output) > 0

    def test_console_progress_bar_indeterminate(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.progress_bar_indeterminate("Task", "working")
        assert len(io._output) > 0


# ── table_with_search fallback ────────────────────────────────────────────────


class TestTableWithSearchFallback:
    def test_table_with_search_returns_all(self):
        p, io = _make_prompt(feeds=[""])
        headers = ["Name", "Age"]
        rows = [["Alice", "30"], ["Bob", "25"]]
        result = p.table_with_search(headers, rows)
        assert len(result) == 2

    def test_table_with_search_with_title(self):
        p, io = _make_prompt(feeds=[""])
        result = p.table_with_search(["A"], [["1"]], title="Test")
        assert len(result) == 1

    def test_console_table_with_search(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.table_with_search(["Col"], [["val"]], title="T")
        assert len(result) == 1


# ── select_with_countdown fallback ────────────────────────────────────────────


class TestSelectWithCountdownFallback:
    def test_select_with_countdown_returns_string(self):
        p, io = _make_prompt(feeds=["1"])
        result = p.select_with_countdown("Pick:", ["A", "B", "C"], timeout=5)
        assert result in ["A", "B", "C"]

    def test_select_with_countdown_with_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.select_with_countdown("Pick:", ["X", "Y"], timeout=1, default=1)
        assert result in ["X", "Y"]

    def test_console_select_with_countdown(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.select_with_countdown("Pick:", ["A", "B"], timeout=5)
        assert result in ["A", "B"]


# ── confirm_with_countdown fallback ───────────────────────────────────────────


class TestConfirmWithCountdownFallback:
    def test_confirm_with_countdown_default(self):
        p, io = _make_prompt(feeds=[""])
        assert p.confirm_with_countdown("Proceed?", timeout=1) is False

    def test_confirm_with_countdown_y(self):
        p, io = _make_prompt(feeds=["y"])
        assert p.confirm_with_countdown("OK?", timeout=5) is True

    def test_confirm_with_countdown_n(self):
        p, io = _make_prompt(feeds=["n"])
        assert p.confirm_with_countdown("Go?", timeout=5, default=True) is False

    def test_console_confirm_with_countdown(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("y")
        c = Console(io, has_readline=False)
        assert c.confirm_with_countdown("Yes?", timeout=5) is True


# ── progress_bar_stripe ──────────────────────────────────────────────────────


class TestProgressBarStripe:
    def test_progress_bar_stripe_zero(self):
        p, io = _make_prompt()
        p.progress_bar_stripe("Build", 0, 100)
        assert len(io._output) > 0

    def test_progress_bar_stripe_half(self):
        p, io = _make_prompt()
        p.progress_bar_stripe("Build", 50, 100)
        assert len(io._output) > 0

    def test_progress_bar_stripe_full(self):
        p, io = _make_prompt()
        p.progress_bar_stripe("Done", 100, 100)
        assert len(io._output) > 0

    def test_console_progress_bar_stripe(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.progress_bar_stripe("Task", 75, 100)
        assert len(io._output) > 0


# ── spinner_with_dots_eta fallback ────────────────────────────────────────────


class TestSpinnerWithDotsEtaFallback:
    def test_spinner_with_dots_eta_does_not_crash(self):
        p, io = _make_prompt()
        p.spinner_with_dots_eta("Loading", 5.0)
        assert len(io._output) > 0

    def test_spinner_with_dots_eta_with_progress(self):
        p, io = _make_prompt()
        p.spinner_with_dots_eta("Building", 10.0, progress=0.5)
        assert len(io._output) > 0

    def test_spinner_with_dots_eta_contains_message(self):
        p, io = _make_prompt()
        p.spinner_with_dots_eta("Working", 2.0)
        output = io._output[0]
        assert "Working" in output

    def test_console_spinner_with_dots_eta(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.spinner_with_dots_eta("Task", 3.0, progress=0.25)
        assert len(io._output) > 0


# ── confirm_with_phrase fallback ──────────────────────────────────────────────


class TestConfirmWithPhraseFallback:
    def test_confirm_with_phrase_wrong(self):
        p, io = _make_prompt(feeds=["no"])
        assert p.confirm_with_phrase("Confirm:", phrase="yes") is False

    def test_confirm_with_phrase_right(self):
        p, io = _make_prompt(feeds=["delete"])
        assert p.confirm_with_phrase("Type to confirm:", phrase="delete") is True

    def test_confirm_with_phrase_empty(self):
        p, io = _make_prompt(feeds=[""])
        assert p.confirm_with_phrase("Confirm:") is False

    def test_console_confirm_with_phrase(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("yes")
        c = Console(io, has_readline=False)
        assert c.confirm_with_phrase("Confirm:", phrase="yes") is True


# ── progress_bar_gradient ─────────────────────────────────────────────────────


class TestProgressBarGradient:
    def test_progress_bar_gradient_zero(self):
        p, io = _make_prompt()
        p.progress_bar_gradient("Build", 0, 100)
        assert len(io._output) > 0

    def test_progress_bar_gradient_half(self):
        p, io = _make_prompt()
        p.progress_bar_gradient("Build", 50, 100)
        assert len(io._output) > 0

    def test_progress_bar_gradient_full(self):
        p, io = _make_prompt()
        p.progress_bar_gradient("Done", 100, 100)
        assert len(io._output) > 0

    def test_console_progress_bar_gradient(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.progress_bar_gradient("Task", 75, 100)
        assert len(io._output) > 0


# ── spinner_pulse fallback ────────────────────────────────────────────────────


class TestSpinnerPulseFallback:
    def test_spinner_pulse_does_not_crash(self):
        p, io = _make_prompt()
        p.spinner_pulse("Loading", duration=0.1)
        assert len(io._output) > 0

    def test_spinner_pulse_contains_message(self):
        p, io = _make_prompt()
        p.spinner_pulse("Building", duration=0.1)
        output = io._output[0]
        assert "Building" in output

    def test_console_spinner_pulse(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.spinner_pulse("Working", duration=0.1)
        assert len(io._output) > 0


# ── select_with_preview_and_icons fallback ────────────────────────────────────


class TestSelectWithPreviewAndIconsFallback:
    def test_select_with_preview_and_icons_returns_string(self):
        def preview(label):
            return f"Preview for {label}"

        p, io = _make_prompt(feeds=["1"])
        options = [("\u2605", "Favorite"), ("\u2606", "Not Favorite")]
        result = p.select_with_preview_and_icons("Pick:", options, preview)
        assert result in ["Favorite", "Not Favorite"]

    def test_select_with_preview_and_icons_empty(self):
        def preview(label):
            return f"Detail: {label}"

        p, io = _make_prompt(feeds=[""])
        options = [("\u2714", "Yes")]
        result = p.select_with_preview_and_icons("Pick:", options, preview)
        assert result == "Yes"

    def test_console_select_with_preview_and_icons(self):
        def preview(label):
            return f"Info: {label}"

        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        options = [("\u2605", "Star"), ("\u2606", "No Star")]
        result = c.select_with_preview_and_icons("Pick:", options, preview)
        assert result in ["Star", "No Star"]


# ── multi_confirm fallback ────────────────────────────────────────────────────


class TestMultiConfirmFallback:
    def test_multi_confirm_returns_dict(self):
        p, io = _make_prompt(feeds=[""])
        result = p.multi_confirm("Confirm:", ["A", "B", "C"])
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_multi_confirm_all_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.multi_confirm("Confirm:", ["X", "Y"], default=True)
        assert all(result.values())

    def test_console_multi_confirm(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.multi_confirm("Confirm:", ["A", "B"])
        assert isinstance(result, dict)


# ── progress_bar_segmented ────────────────────────────────────────────────────


class TestProgressBarSegmented:
    def test_progress_bar_segmented_does_not_crash(self):
        p, io = _make_prompt()
        p.progress_bar_segmented("Build", [("green", 50), ("yellow", 30), ("red", 20)])
        assert len(io._output) > 0

    def test_progress_bar_segmented_single(self):
        p, io = _make_prompt()
        p.progress_bar_segmented("Test", [("cyan", 100)])
        assert len(io._output) > 0

    def test_console_progress_bar_segmented(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.progress_bar_segmented("Task", [("green", 70), ("red", 30)])
        assert len(io._output) > 0


# ── spinner_wave fallback ─────────────────────────────────────────────────────


class TestSpinnerWaveFallback:
    def test_spinner_wave_does_not_crash(self):
        p, io = _make_prompt()
        p.spinner_wave("Loading", duration=0.1)
        assert len(io._output) > 0

    def test_spinner_wave_contains_message(self):
        p, io = _make_prompt()
        p.spinner_wave("Building", duration=0.1)
        output = io._output[0]
        assert "Building" in output

    def test_console_spinner_wave(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.spinner_wave("Working", duration=0.1)
        assert len(io._output) > 0


# ── select_with_tags fallback ─────────────────────────────────────────────────


class TestSelectWithTagsFallback:
    def test_select_with_tags_returns_string(self):
        p, io = _make_prompt(feeds=["1"])
        options = ["Apple", "Banana", "Cherry"]
        tags = {"Apple": ["fruit"], "Banana": ["fruit"], "Cherry": ["fruit", "red"]}
        result = p.select_with_tags("Pick:", options, tags)
        assert result in options

    def test_select_with_tags_no_tags(self):
        p, io = _make_prompt(feeds=[""])
        result = p.select_with_tags("Pick:", ["A", "B"], {})
        assert result in ["A", "B"]

    def test_console_select_with_tags(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.select_with_tags("Pick:", ["A", "B"], {"A": ["x"]})
        assert result in ["A", "B"]


# ── select_with_preview_and_grouping fallback ─────────────────────────────────


class TestSelectWithPreviewAndGroupingFallback:
    def test_select_with_preview_and_grouping_returns_string(self):
        def preview(opt):
            return f"Preview for {opt}"

        p, io = _make_prompt(feeds=["1"])
        groups = {"Fruits": ["Apple", "Banana"], "Veggies": ["Carrot"]}
        result = p.select_with_preview_and_grouping("Pick:", groups, preview)
        assert result in ["Apple", "Banana", "Carrot"]

    def test_console_select_with_preview_and_grouping(self):
        def preview(opt):
            return f"Info: {opt}"

        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        groups = {"Cat1": ["A", "B"]}
        result = c.select_with_preview_and_grouping("Pick:", groups, preview)
        assert result in ["A", "B"]


# ── confirm_list_with_preview fallback ────────────────────────────────────────


class TestConfirmListWithPreviewFallback:
    def test_confirm_list_with_preview_returns_list(self):
        def preview(item):
            return f"Detail: {item}"

        p, io = _make_prompt(feeds=[""])
        result = p.confirm_list_with_preview("Confirm:", ["A", "B"], preview)
        assert isinstance(result, list)

    def test_confirm_list_with_preview_default_false(self):
        def preview(item):
            return f"Info: {item}"

        p, io = _make_prompt(feeds=[""])
        result = p.confirm_list_with_preview("Confirm:", ["X"], preview, default=False)
        assert result == []

    def test_console_confirm_list_with_preview(self):
        def preview(item):
            return f"Preview: {item}"

        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.confirm_list_with_preview("Confirm:", ["A"], preview)
        assert isinstance(result, list)


# ── progress_bar_multi_segment ────────────────────────────────────────────────


class TestProgressBarMultiSegment:
    def test_progress_bar_multi_segment_does_not_crash(self):
        p, io = _make_prompt()
        p.progress_bar_multi_segment("Build", [("green", 50, "green"), ("red", 50, "red")])
        assert len(io._output) > 0

    def test_progress_bar_multi_segment_single(self):
        p, io = _make_prompt()
        p.progress_bar_multi_segment("Test", [("done", 100, "cyan")])
        assert len(io._output) > 0

    def test_console_progress_bar_multi_segment(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.progress_bar_multi_segment("Task", [("a", 30, "green"), ("b", 70, "red")])
        assert len(io._output) > 0


# ── spinner_bounce fallback ───────────────────────────────────────────────────


class TestSpinnerBounceFallback:
    def test_spinner_bounce_does_not_crash(self):
        p, io = _make_prompt()
        p.spinner_bounce("Loading", duration=0.1)
        assert len(io._output) > 0

    def test_spinner_bounce_contains_message(self):
        p, io = _make_prompt()
        p.spinner_bounce("Building", duration=0.1)
        output = io._output[0]
        assert "Building" in output

    def test_console_spinner_bounce(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.spinner_bounce("Working", duration=0.1)
        assert len(io._output) > 0


# ── select_with_confirm fallback ──────────────────────────────────────────────


class TestSelectWithConfirmFallback:
    def test_select_with_confirm_returns_string(self):
        p, io = _make_prompt(feeds=["1"])
        result = p.select_with_confirm("Pick:", ["A", "B", "C"])
        assert result in ["A", "B", "C"]

    def test_select_with_confirm_with_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.select_with_confirm("Pick:", ["X", "Y"], default="Y")
        assert result in ["X", "Y"]

    def test_console_select_with_confirm(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.select_with_confirm("Pick:", ["A", "B"])
        assert result in ["A", "B"]


# ── confirm_with_preview_and_timeout fallback ─────────────────────────────────


class TestConfirmWithPreviewAndTimeoutFallback:
    def test_confirm_with_preview_and_timeout_default(self):
        p, io = _make_prompt(feeds=[""])
        assert p.confirm_with_preview_and_timeout("Apply?", "diff here", timeout=1) is False

    def test_confirm_with_preview_and_timeout_y(self):
        p, io = _make_prompt(feeds=["y"])
        assert p.confirm_with_preview_and_timeout("OK?", "preview", timeout=5) is True

    def test_console_confirm_with_preview_and_timeout(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("y")
        c = Console(io, has_readline=False)
        assert c.confirm_with_preview_and_timeout("Apply?", "text", timeout=5) is True


# ── progress_bar_animated ─────────────────────────────────────────────────────


class TestProgressBarAnimated:
    def test_progress_bar_animated_zero(self):
        p, io = _make_prompt()
        p.progress_bar_animated("Build", 0, 100)
        assert len(io._output) > 0

    def test_progress_bar_animated_half(self):
        p, io = _make_prompt()
        p.progress_bar_animated("Build", 50, 100)
        assert len(io._output) > 0

    def test_progress_bar_animated_full(self):
        p, io = _make_prompt()
        p.progress_bar_animated("Done", 100, 100)
        assert len(io._output) > 0

    def test_console_progress_bar_animated(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.progress_bar_animated("Task", 75, 100)
        assert len(io._output) > 0


# ── spinner_clock fallback ────────────────────────────────────────────────────


class TestSpinnerClockFallback:
    def test_spinner_clock_does_not_crash(self):
        p, io = _make_prompt()
        p.spinner_clock("Loading", duration=0.1)
        assert len(io._output) > 0

    def test_spinner_clock_contains_message(self):
        p, io = _make_prompt()
        p.spinner_clock("Building", duration=0.1)
        output = io._output[0]
        assert "Building" in output

    def test_console_spinner_clock(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.spinner_clock("Working", duration=0.1)
        assert len(io._output) > 0


# ── select_with_preview_and_confirm fallback ──────────────────────────────────


class TestSelectWithPreviewAndConfirmFallback:
    def test_select_with_preview_and_confirm_returns_string(self):
        def preview(opt):
            return f"Preview: {opt}"

        p, io = _make_prompt(feeds=["1"])
        result = p.select_with_preview_and_confirm("Pick:", ["A", "B", "C"], preview)
        assert result in ["A", "B", "C"]

    def test_select_with_preview_and_confirm_with_default(self):
        def preview(opt):
            return f"Info: {opt}"

        p, io = _make_prompt(feeds=[""])
        result = p.select_with_preview_and_confirm("Pick:", ["X", "Y"], preview, default="Y")
        assert result in ["X", "Y"]

    def test_console_select_with_preview_and_confirm(self):
        def preview(opt):
            return f"Detail: {opt}"

        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.select_with_preview_and_confirm("Pick:", ["A", "B"], preview)
        assert result in ["A", "B"]


# ── confirm_with_preview_and_countdown fallback ───────────────────────────────


class TestConfirmWithPreviewAndCountdownFallback:
    def test_confirm_with_preview_and_countdown_default(self):
        p, io = _make_prompt(feeds=[""])
        assert p.confirm_with_preview_and_countdown("Apply?", "diff", timeout=1) is True

    def test_confirm_with_preview_and_countdown_n(self):
        p, io = _make_prompt(feeds=["n"])
        assert p.confirm_with_preview_and_countdown("OK?", "preview", timeout=5) is False

    def test_console_confirm_with_preview_and_countdown(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("y")
        c = Console(io, has_readline=False)
        assert c.confirm_with_preview_and_countdown("Apply?", "text", timeout=5) is True


# ── progress_bar_with_status_and_eta ──────────────────────────────────────────


class TestProgressBarWithStatusAndEta:
    def test_progress_bar_with_status_and_eta_does_not_crash(self):
        p, io = _make_prompt()
        p.progress_bar_with_status_and_eta("Build", 50, 100, "compiling")
        assert len(io._output) > 0

    def test_progress_bar_with_status_and_eta_zero(self):
        p, io = _make_prompt()
        p.progress_bar_with_status_and_eta("Build", 0, 100, "starting")
        assert len(io._output) > 0

    def test_console_progress_bar_with_status_and_eta(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.progress_bar_with_status_and_eta("Task", 75, 100, "done")
        assert len(io._output) > 0


# ── spinner_with_messages fallback ────────────────────────────────────────────


class TestSpinnerWithMessagesFallback:
    def test_spinner_with_messages_does_not_crash(self):
        p, io = _make_prompt()
        p.spinner_with_messages("Loading", ["one", "two", "three"], duration=0.1)
        assert len(io._output) > 0

    def test_spinner_with_messages_contains_message(self):
        p, io = _make_prompt()
        p.spinner_with_messages("Building", ["step 1", "step 2"], duration=0.1)
        output = io._output[0]
        assert "Building" in output

    def test_console_spinner_with_messages(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.spinner_with_messages("Working", ["a", "b"], duration=0.1)
        assert len(io._output) > 0


# ── select_with_filter_and_preview fallback ───────────────────────────────────


class TestSelectWithFilterAndPreviewFallback:
    def test_select_with_filter_and_preview_returns_string(self):
        def preview(opt):
            return f"Preview: {opt}"

        p, io = _make_prompt(feeds=["1"])
        result = p.select_with_filter_and_preview("Pick:", ["A", "B", "C"], preview)
        assert result in ["A", "B", "C"]

    def test_select_with_filter_and_preview_empty(self):
        def preview(opt):
            return f"Info: {opt}"

        p, io = _make_prompt(feeds=[""])
        result = p.select_with_filter_and_preview("Pick:", ["X", "Y"], preview)
        assert result in ["X", "Y"]

    def test_console_select_with_filter_and_preview(self):
        def preview(opt):
            return f"Detail: {opt}"

        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.select_with_filter_and_preview("Pick:", ["A", "B"], preview)
        assert result in ["A", "B"]


# ── select_table_with_preview fallback ────────────────────────────────────────


class TestSelectTableWithPreviewFallback:
    def test_select_table_with_preview_returns_row(self):
        def preview(row):
            return f"Details: {row[0]}"

        p, io = _make_prompt(feeds=["1"])
        result = p.select_table_with_preview(
            ["Name", "Age"], [["Alice", "30"], ["Bob", "25"]], preview
        )
        assert result in [["Alice", "30"], ["Bob", "25"]]

    def test_console_select_table_with_preview(self):
        def preview(row):
            return f"Info: {row[0]}"

        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.select_table_with_preview(["A"], [["X"], ["Y"]], preview)
        assert result in [["X"], ["Y"]]


# ── confirm_with_preview_and_edit_with_timeout fallback ──────────────────────


class TestConfirmWithPreviewAndEditWithTimeoutFallback:
    def test_confirm_with_preview_and_edit_with_timeout_default(self):
        p, io = _make_prompt(feeds=[""])
        ok, text = p.confirm_with_preview_and_edit_with_timeout("Apply?", "diff here", timeout=1)
        assert ok is True

    def test_confirm_with_preview_and_edit_with_timeout_n(self):
        p, io = _make_prompt(feeds=["n"])
        ok, text = p.confirm_with_preview_and_edit_with_timeout("OK?", "preview", timeout=5)
        assert ok is False

    def test_console_confirm_with_preview_and_edit_with_timeout(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("y")
        c = Console(io, has_readline=False)
        ok, text = c.confirm_with_preview_and_edit_with_timeout("Apply?", "text", timeout=5)
        assert ok is True


# ── progress_bar_with_eta_and_status ─────────────────────────────────────────


class TestProgressBarWithEtaAndStatus:
    def test_progress_bar_with_eta_and_status_does_not_crash(self):
        p, io = _make_prompt()
        p.progress_bar_with_eta_and_status("Build", 50, 100, "compiling", elapsed=10.0)
        assert len(io._output) > 0

    def test_progress_bar_with_eta_and_status_zero(self):
        p, io = _make_prompt()
        p.progress_bar_with_eta_and_status("Build", 0, 100, "starting")
        assert len(io._output) > 0

    def test_console_progress_bar_with_eta_and_status(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.progress_bar_with_eta_and_status("Task", 75, 100, "done", elapsed=5.0)
        assert len(io._output) > 0


# ── spinner_with_dots_and_status fallback ────────────────────────────────────


class TestSpinnerWithDotsAndStatusFallback:
    def test_spinner_with_dots_and_status_does_not_crash(self):
        p, io = _make_prompt()
        p.spinner_with_dots_and_status("Loading", "waiting", duration=0.1)
        assert len(io._output) > 0

    def test_spinner_with_dots_and_status_contains_message(self):
        p, io = _make_prompt()
        p.spinner_with_dots_and_status("Building", "step 1", duration=0.1)
        output = io._output[0]
        assert "Building" in output

    def test_console_spinner_with_dots_and_status(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.spinner_with_dots_and_status("Working", "busy", duration=0.1)
        assert len(io._output) > 0


# ── select_with_preview_and_countdown fallback ───────────────────────────────


class TestSelectWithPreviewAndCountdownFallback:
    def test_select_with_preview_and_countdown_returns_string(self):
        def preview(opt):
            return f"Preview: {opt}"

        p, io = _make_prompt(feeds=["1"])
        result = p.select_with_preview_and_countdown("Pick:", ["A", "B"], preview, timeout=5)
        assert result in ["A", "B"]

    def test_select_with_preview_and_countdown_with_default(self):
        def preview(opt):
            return f"Info: {opt}"

        p, io = _make_prompt(feeds=[""])
        result = p.select_with_preview_and_countdown(
            "Pick:", ["X", "Y"], preview, timeout=5, default="Y"
        )
        assert result in ["X", "Y"]

    def test_console_select_with_preview_and_countdown(self):
        def preview(opt):
            return f"Detail: {opt}"

        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.select_with_preview_and_countdown("Pick:", ["A", "B"], preview, timeout=5)
        assert result in ["A", "B"]


# ── multi_select_with_filter fallback ─────────────────────────────────────────


class TestMultiSelectWithFilterFallback:
    def test_multi_select_with_filter_returns_list(self):
        p, io = _make_prompt(feeds=["1"])
        result = p.multi_select_with_filter("Pick:", ["A", "B", "C"])
        assert isinstance(result, list)

    def test_multi_select_with_filter_with_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.multi_select_with_filter("Pick:", ["X", "Y"], default=["X"])
        assert isinstance(result, list)

    def test_console_multi_select_with_filter(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.multi_select_with_filter("Pick:", ["A", "B"])
        assert isinstance(result, list)


# ── confirm_with_countdown_and_preview fallback ──────────────────────────────


class TestConfirmWithCountdownAndPreviewFallback:
    def test_confirm_with_countdown_and_preview_default(self):
        p, io = _make_prompt(feeds=[""])
        assert p.confirm_with_countdown_and_preview("Apply?", "diff", timeout=1) is True

    def test_confirm_with_countdown_and_preview_n(self):
        p, io = _make_prompt(feeds=["n"])
        assert p.confirm_with_countdown_and_preview("OK?", "preview", timeout=5) is False

    def test_console_confirm_with_countdown_and_preview(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("y")
        c = Console(io, has_readline=False)
        assert c.confirm_with_countdown_and_preview("Apply?", "text", timeout=5) is True


# ── progress_bar_with_steps ──────────────────────────────────────────────────


class TestProgressBarWithSteps:
    def test_progress_bar_with_steps_does_not_crash(self):
        p, io = _make_prompt()
        p.progress_bar_with_steps("Build", ["compile", "link", "test"], 0)
        assert len(io._output) > 0

    def test_progress_bar_with_steps_middle(self):
        p, io = _make_prompt()
        p.progress_bar_with_steps("Build", ["compile", "link", "test"], 1)
        assert len(io._output) > 0

    def test_console_progress_bar_with_steps(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.progress_bar_with_steps("Task", ["a", "b", "c"], 2)
        assert len(io._output) > 0


# ── spinner_with_eta_message fallback ────────────────────────────────────────


class TestSpinnerWithEtaMessageFallback:
    def test_spinner_with_eta_message_does_not_crash(self):
        p, io = _make_prompt()
        p.spinner_with_eta_message("Loading", 100, duration=0.1)
        assert len(io._output) > 0

    def test_spinner_with_eta_message_contains_message(self):
        p, io = _make_prompt()
        p.spinner_with_eta_message("Building", 50, duration=0.1)
        output = io._output[0]
        assert "Building" in output

    def test_console_spinner_with_eta_message(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        c = Console(io, has_readline=False)
        c.spinner_with_eta_message("Working", 100, duration=0.1)
        assert len(io._output) > 0


# ── table_with_search_and_preview fallback ───────────────────────────────────


class TestTableWithSearchAndPreviewFallback:
    def test_table_with_search_and_preview_returns_row(self):
        def preview(row):
            return f"Details: {row[0]}"

        p, io = _make_prompt(feeds=["1"])
        result = p.table_with_search_and_preview(["Name"], [["A"], ["B"]], preview)
        assert result in [["A"], ["B"]]

    def test_console_table_with_search_and_preview(self):
        def preview(row):
            return f"Info: {row[0]}"

        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.table_with_search_and_preview(["A"], [["X"], ["Y"]], preview)
        assert result in [["X"], ["Y"]]


# ── select_with_filter_and_confirm fallback ──────────────────────────────────


class TestSelectWithFilterAndConfirmFallback:
    def test_select_with_filter_and_confirm_returns_string(self):
        p, io = _make_prompt(feeds=["1"])
        result = p.select_with_filter_and_confirm("Pick:", ["A", "B", "C"])
        assert result in ["A", "B", "C"]

    def test_select_with_filter_and_confirm_with_default(self):
        p, io = _make_prompt(feeds=[""])
        result = p.select_with_filter_and_confirm("Pick:", ["X", "Y"], default="Y")
        assert result in ["X", "Y"]

    def test_console_select_with_filter_and_confirm(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.select_with_filter_and_confirm("Pick:", ["A", "B"])
        assert result in ["A", "B"]


# ── history_search fallback ───────────────────────────────────────────────────


class TestHistorySearchFallback:
    def test_history_search_returns_string(self):
        p, io = _make_prompt(feeds=["1"])
        result = p.history_search(["ls", "cd", "pwd"])
        assert result in ["ls", "cd", "pwd"]

    def test_history_search_empty(self):
        p, io = _make_prompt(feeds=[""])
        result = p.history_search([])
        assert result is None

    def test_console_history_search(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.history_search(["cmd1", "cmd2"])
        assert result in ["cmd1", "cmd2"]


# ── process_manager fallback ─────────────────────────────────────────────────


class TestProcessManagerFallback:
    def test_process_manager_returns_dict(self):
        p, io = _make_prompt(feeds=["1"])
        result = p.process_manager([{"name": "train", "status": "running"}])
        assert result is not None
        assert result["name"] == "train"

    def test_process_manager_empty(self):
        p, io = _make_prompt(feeds=[""])
        result = p.process_manager([])
        assert result is None

    def test_console_process_manager(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.process_manager([{"name": "job1", "status": "running"}])
        assert result is not None


# ── log_viewer fallback ──────────────────────────────────────────────────────


class TestLogViewerFallback:
    def test_log_viewer_returns_string(self):
        p, io = _make_prompt(feeds=["1"])
        result = p.log_viewer(["INFO: started", "ERROR: failed"])
        assert result in ["INFO: started", "ERROR: failed"]

    def test_log_viewer_empty(self):
        p, io = _make_prompt(feeds=[""])
        result = p.log_viewer([])
        assert result is None

    def test_console_log_viewer(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.log_viewer(["log1", "log2"])
        assert result in ["log1", "log2"]


# ── config_editor fallback ────────────────────────────────────────────────────


class TestConfigEditorFallback:
    def test_config_editor_returns_dict(self):
        p, io = _make_prompt(feeds=[""])
        result = p.config_editor({"host": "localhost", "port": 8000})
        assert isinstance(result, dict)
        assert result["host"] == "localhost"

    def test_config_editor_empty(self):
        p, io = _make_prompt(feeds=[""])
        result = p.config_editor({})
        assert result == {}

    def test_console_config_editor(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.config_editor({"key": "val"})
        assert result["key"] == "val"


# ── diff_viewer fallback ─────────────────────────────────────────────────────


class TestDiffViewerFallback:
    def test_diff_viewer_returns_string(self):
        p, io = _make_prompt(feeds=[""])
        result = p.diff_viewer("old", "new")
        assert result == "new"

    def test_diff_viewer_same(self):
        p, io = _make_prompt(feeds=[""])
        result = p.diff_viewer("same", "same")
        assert result == "same"

    def test_console_diff_viewer(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.diff_viewer("a", "b")
        assert result == "b"


# ── interactive_search fallback ──────────────────────────────────────────────


class TestInteractiveSearchFallback:
    def test_interactive_search_returns_string(self):
        p, io = _make_prompt(feeds=["1"])
        result = p.interactive_search(["apple", "banana", "cherry"])
        assert result in ["apple", "banana", "cherry"]

    def test_interactive_search_with_preview(self):
        def preview(item):
            return f"Info: {item}"

        p, io = _make_prompt(feeds=["1"])
        result = p.interactive_search(["x", "y"], preview_fn=preview)
        assert result in ["x", "y"]

    def test_console_interactive_search(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("1")
        c = Console(io, has_readline=False)
        result = c.interactive_search(["a", "b"])
        assert result in ["a", "b"]


# ── wizard fallback ──────────────────────────────────────────────────────────


class TestWizardFallback:
    def test_wizard_returns_dict(self):
        p, io = _make_prompt(feeds=["hello", ""])
        result = p.wizard(
            [
                {"label": "Name", "type": "input", "default": ""},
                {"label": "OK", "type": "confirm", "default": "true"},
            ]
        )
        assert isinstance(result, dict)
        assert "Name" in result

    def test_wizard_with_select(self):
        p, io = _make_prompt(feeds=["1", ""])
        result = p.wizard(
            [
                {"label": "Choice", "type": "select", "options": ["A", "B"]},
            ]
        )
        assert isinstance(result, dict)

    def test_console_wizard(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("val")
        c = Console(io, has_readline=False)
        result = c.wizard([{"label": "X", "type": "input", "default": ""}])
        assert "X" in result


# ── spreadsheet_editor fallback ──────────────────────────────────────────────


class TestSpreadsheetEditorFallback:
    def test_spreadsheet_editor_returns_rows(self):
        p, io = _make_prompt(feeds=[""])
        result = p.spreadsheet_editor(["A", "B"], [["1", "2"], ["3", "4"]])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_spreadsheet_editor_empty(self):
        p, io = _make_prompt(feeds=[""])
        result = p.spreadsheet_editor(["Col"], [["val"]])
        assert result == [["val"]]

    def test_console_spreadsheet_editor(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.spreadsheet_editor(["X"], [["a"]])
        assert result == [["a"]]


# ── hierarchical_menu fallback ──────────────────────────────────────────────


class TestHierarchicalMenuFallback:
    def test_hierarchical_menu_returns_string(self):
        p, io = _make_prompt(feeds=[""])
        menu = {"Item 1": "action1", "Item 2": "action2"}
        result = p.hierarchical_menu(menu)
        assert result is None

    def test_hierarchical_menu_nested(self):
        p, io = _make_prompt(feeds=[""])
        menu = {"Group": {"Sub": "result"}}
        result = p.hierarchical_menu(menu)
        assert result is None

    def test_console_hierarchical_menu(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.hierarchical_menu({"A": "action"})
        assert result is None


# ── form fallback ────────────────────────────────────────────────────────────


class TestFormFallback:
    def test_form_returns_dict(self):
        p, io = _make_prompt(feeds=["value", ""])
        result = p.form(
            [
                {"label": "Name", "type": "text", "default": ""},
            ]
        )
        assert isinstance(result, dict)
        assert "Name" in result

    def test_form_with_select(self):
        p, io = _make_prompt(feeds=["1", ""])
        result = p.form(
            [
                {"label": "Choice", "type": "select", "options": ["A", "B"]},
            ]
        )
        assert isinstance(result, dict)

    def test_console_form(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("val")
        c = Console(io, has_readline=False)
        result = c.form([{"label": "X", "type": "text", "default": ""}])
        assert "X" in result


# ── playlist_manager fallback ────────────────────────────────────────────────


class TestPlaylistManagerFallback:
    def test_playlist_manager_returns_list(self):
        p, io = _make_prompt(feeds=[""])
        result = p.playlist_manager(["song1", "song2"])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_playlist_manager_empty(self):
        p, io = _make_prompt(feeds=[""])
        result = p.playlist_manager([])
        assert result == []

    def test_console_playlist_manager(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.playlist_manager(["a", "b"])
        assert result == ["a", "b"]


# ── kanban_board fallback ────────────────────────────────────────────────────


class TestKanbanBoardFallback:
    def test_kanban_board_returns_dict(self):
        p, io = _make_prompt(feeds=[""])
        result = p.kanban_board({"Todo": ["task1"], "Done": []})
        assert isinstance(result, dict)
        assert "Todo" in result

    def test_kanban_board_empty(self):
        p, io = _make_prompt(feeds=[""])
        result = p.kanban_board({"A": [], "B": []})
        assert result == {"A": [], "B": []}

    def test_console_kanban_board(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.kanban_board({"X": ["y"]})
        assert result == {"X": ["y"]}


# ── calendar_view fallback ───────────────────────────────────────────────────


class TestCalendarViewFallback:
    def test_calendar_view_returns_none(self):
        p, io = _make_prompt(feeds=[""])
        result = p.calendar_view(2024, 1)
        assert result is None

    def test_calendar_view_with_events(self):
        p, io = _make_prompt(feeds=[""])
        result = p.calendar_view(2024, 6, events={15: "Meeting"})
        assert result is None

    def test_console_calendar_view(self):
        from domain.shell._internal.console import Console

        io = MemoryIO()
        io.feed("")
        c = Console(io, has_readline=False)
        result = c.calendar_view(2024, 1)
        assert result is None


# ── type-to-filter caret ──────────────────────────────────────────────────────


import contextlib as _contextlib
import io as _io_mod
import re as _re

from domain.shell._internal.tui_session import Key as _Key

_CSI = _re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _plain(text: str) -> str:
    """Strip every CSI sequence and carriage return, leaving printable text."""
    return _CSI.sub("", text).replace("\r", "")


def _frames(raw: str) -> list[str]:
    """Split a captured run into the individual frames that painted it.

    The loop clears before painting, and `_clear` writes one `CURSOR_UP` per
    line it must move back over, so splitting there recovers the frames.

    Reading the whole capture instead is a trap: frame one is painted with an
    empty query, and an empty query renders *every* option. Any assertion about
    what matched would therefore pass on frame one no matter what the caret
    actually did — which is exactly how five of these tests came back green
    before a caret existed.
    """
    frames = []
    for part in raw.split(_CURSOR_UP):
        text = _CSI.sub("", part).replace("\r", "")
        if text.strip():
            frames.append(text)
    return frames


def _frame(raw: str) -> str:
    """The last frame painted before the prompt returned."""
    frames = _frames(raw)
    return frames[-1] if frames else ""


def _filter_text(raw: str, label: str = "Filter:") -> str:
    """The contents of the final frame's *label* line, caret marker included.

    Boxes do not agree on the word — most say `Filter:`, the search boxes say
    `Search:` — so the label is a parameter rather than a second helper.
    """
    for line in _frame(raw).splitlines():
        if label in line:
            return line.split(label, 1)[1].strip()
    return ""


class _ScriptedSession:
    """Replays a fixed key sequence through a prompt's raw-mode surface.

    `read_key` raises rather than exhausting, so a script that forgets its
    terminating key fails loudly instead of looping on an empty list.
    """

    def __init__(self, keys):
        self._keys = list(keys)

    def raw(self):
        return _contextlib.nullcontext()

    def read_key(self):
        if not self._keys:
            raise AssertionError("key script exhausted before the prompt returned")
        return self._keys.pop(0)


def _drive(p, options, keys, meth: str = "_select_raw") -> str:
    """Run *meth* over *keys*, returning everything it wrote to stdout.

    Always appends Esc, which is what every one of these prompts returns on, so
    the loop terminates deterministically rather than depending on a test to
    end it. The method name is a parameter because the family is seventeen
    boxes with one shape and seventeen names.
    """
    p._session = _ScriptedSession([*keys, _RawKey(_KEY_ESC)])
    buf = _io_mod.StringIO()
    with _contextlib.redirect_stdout(buf):
        getattr(p, meth)("Pick", options)
    return buf.getvalue()


def _form_frames(raw: str, message: str) -> list[str]:
    """Split a form-style capture into one entry per repaint.

    `_form_raw` repaints without ever returning to the top of the frame it is
    replacing: it erases the single line the cursor happens to sit on and writes
    a fresh copy of the whole form below. So the repaints run together, and
    `_frames` — which splits on the cursor-up these prompts never emit between
    paints — sees the entire capture as one frame.

    The message line is the only seam. Every `_render` body opens with
    `ERASE_LINE CR two-spaces message`, and after escape-stripping that is
    nothing but the message at the head of a line. Splitting on the message
    recovers the repaints, last one included.
    """
    parts = _re.split(rf"(?=\s*{_re.escape(message)})", _plain(raw))
    return [p for p in parts if p.strip()]


def _frame_with(raw: str, message: str, marker: str) -> str:
    """The last repaint that still shows *marker* — i.e. the frame we mean.

    The final repaint is always the submitted view, where the caret has already
    gone, so "last frame" is the wrong frame. Failing loudly when no repaint
    ever showed the marker is the point: a caret that never painted must not
    read as a caret that was somewhere else.
    """
    for part in reversed(_form_frames(raw, message)):
        if marker in part:
            return part
    raise AssertionError(f"no repaint ever showed {marker!r}")


def _chars(text: str):
    return [_RawKey(_KEY_CHAR, c) for c in text]


class TestSelectFilterCaret:
    """The type-to-filter box in `select` must behave like a normal input field.

    It edits a bare `str` one character at a time from the end, so until now it
    had no caret to move: left/right, home/end, word jumps and forward-delete
    were all dead keys inside it, and backspace could only ever take the last
    character off. Every other caret-bearing field in the shell already moves
    (see `interactive_prompt`), so this is the one that lagged.

    The assertions observe *what the filter matched*, not the markup around it,
    because the match is the user-visible consequence of the caret position: a
    caret that does not actually move insertion produces a different surviving
    option list, which is a thing a test can see without parsing ANSI.
    """

    def test_filter_renders_caret(self):
        p, _ = _make_prompt()
        out = _drive(p, ["abc"], _chars("a"))
        assert _filter_text(out) == "a|"

    def test_caret_starts_at_end_so_typing_is_unchanged(self):
        # Guard: the caret defaults to end-of-text, so plain typing must keep
        # appending exactly as it did before there was a caret at all.
        p, _ = _make_prompt()
        out = _drive(p, ["abc"], _chars("abc"))
        assert _filter_text(out).replace("|", "") == "abc"

    def test_home_then_type_inserts_at_start(self):
        # "abc" -> Home -> type X should make the query "Xabc". If Home is
        # ignored the X lands at the end and the query becomes "abcX".
        p, _ = _make_prompt()
        out = _drive(p, ["Xabc"], [*_chars("abc"), _RawKey(_KEY_HOME), _RawKey(_KEY_CHAR, "X")])
        assert _filter_text(out).replace("|", "") == "Xabc"

    def test_backspace_takes_the_character_before_the_caret(self):
        # "abc" -> Left -> Backspace should leave "ac". Backspacing from the
        # end instead leaves "ab".
        p, _ = _make_prompt()
        out = _drive(
            p,
            ["ac"],
            [*_chars("abc"), _RawKey(_KEY_LEFT), _RawKey(_KEY_BACKSPACE)],
        )
        assert _filter_text(out).replace("|", "") == "ac"

    def test_delete_takes_the_character_at_the_caret(self):
        # "abc" -> Home -> Delete should leave "bc". Today Delete is not bound
        # in this branch at all, so the query stays "abc".
        p, _ = _make_prompt()
        out = _drive(p, ["bc"], [*_chars("abc"), _RawKey(_KEY_HOME), _RawKey(_KEY_DELETE)])
        assert _filter_text(out).replace("|", "") == "bc"

    def test_ctrl_left_jumps_to_the_start_of_the_word(self):
        # "foo bar" -> Ctrl+Left lands before "bar"; typing X then makes
        # "foo Xbar". This also proves the plain-Left branch cannot be the one
        # firing: a one-character step from the end would give "foo baXr", and
        # a chord left unhandled would give "foo barX".
        p, _ = _make_prompt()
        out = _drive(
            p,
            ["foo Xbar"],
            [*_chars("foo bar"), _Key(_KEY_LEFT, ctrl=True), _RawKey(_KEY_CHAR, "X")],
        )
        assert _filter_text(out).replace("|", "") == "foo Xbar"

    def test_filter_still_matches_the_whole_query_from_a_mid_caret(self):
        # Moving the caret must not narrow the query: the filter always sees
        # every character, wherever the caret happens to sit.
        p, _ = _make_prompt()
        out = _drive(p, ["ab", "abc"], [*_chars("ab"), _RawKey(_KEY_HOME)])
        assert "ab" in _frame(out)
        assert "abc" in _frame(out)


# ── the rest of the filter-box family ─────────────────────────────────────────


class TestSearchBoxCaret:
    """A second box, in a different module, driven the same way.

    `_select_raw` proved the helper works. This proves the hand-off reaches a
    *different* file through a different chain — `interactive_select`'s search
    box, which labels itself `Search:` and draws its bar with `_RESET` rather
    than reverse video. A conversion verified at only one site is a conversion
    verified at exactly one point.
    """

    def test_search_box_renders_its_bar(self):
        p, _ = _make_prompt()
        out = _drive(p, ["abc"], _chars("a"), "_select_with_search_raw")
        assert _filter_text(out, "Search:") == "a│"

    def test_search_box_inserts_at_a_moved_caret(self):
        p, _ = _make_prompt()
        out = _drive(
            p,
            ["Xabc"],
            [*_chars("abc"), _RawKey(_KEY_HOME), _RawKey(_KEY_CHAR, "X")],
            "_select_with_search_raw",
        )
        query = _filter_text(out, "Search:").replace("|", "").replace("│", "")
        assert query == "Xabc"

    def test_search_box_word_jumps(self):
        p, _ = _make_prompt()
        out = _drive(
            p,
            ["foo Xbar"],
            [*_chars("foo bar"), _Key(_KEY_LEFT, ctrl=True), _RawKey(_KEY_CHAR, "X")],
            "_select_with_search_raw",
        )
        query = _filter_text(out, "Search:").replace("|", "").replace("│", "")
        assert query == "foo Xbar"


class TestFilterBoxWiring:
    """Sweep every type-to-filter box, so none can be left append-only.

    Eighteen boxes across four modules gained a caret in one mechanical pass.
    The risk in a pass like that is not that one box regressed — it is that one
    box was silently *missed* and stayed append-only while its siblings moved,
    which driving any single box would never show. This asserts the contract on
    all of them at once, and counts them, so a box added later fails until it
    is wired too.
    """

    MODULES = (
        "interactive_input",
        "interactive_prompt",
        "interactive_select",
        "interactive_table",
    )
    MIN_BOXES = 18

    @staticmethod
    def _boxes(module: str, needle: str = 'query = ""') -> list[tuple[str, str]]:
        """`(method, source)` for each method in *module* owning a `needle` init.

        The needle is a parameter rather than a second helper because there are
        two families of text-entry surface with two buffer names — `query` for a
        filter over options, `edit_buf` for a value being edited — and a sweep
        that only knew about one of them is what let the other go caret-less.
        """
        import importlib
        import inspect as _inspect

        pkg = importlib.import_module(f"domain.shell._internal.{module}")
        src = _inspect.getsource(pkg)
        out = []
        for m in _re.finditer(r"^    def (\w+)\(", src, _re.MULTILINE):
            nxt = _re.search(r"^    def ", src[m.end() :], _re.MULTILINE)
            body = src[m.start() : m.end() + (nxt.start() if nxt else len(src))]
            if needle in body:
                out.append((m.group(1), body))
        return out

    def test_every_filter_box_is_wired_to_the_shared_editor(self):
        boxes = [(mod, name, body) for mod in self.MODULES for name, body in self._boxes(mod)]
        assert len(boxes) >= self.MIN_BOXES, (
            f"expected at least {self.MIN_BOXES} filter boxes, found {len(boxes)}: "
            f"{[f'{m}.{n}' for m, n, _ in boxes]}"
        )
        for mod, name, body in boxes:
            where = f"{mod}.{name}"
            assert "qcur = 0" in body, f"{where}: never initialises its caret"
            assert "_filter_view(query, qcur" in body, (
                f"{where}: renders the query without a caret on it"
            )
            assert "self._filter_key(query, qcur, key)" in body, (
                f"{where}: its keys never reach the shared editor"
            )
            assert "query, qcur," in body, f"{where}: throws the caret away"

    def test_every_value_field_is_wired_to_the_shared_editor(self):
        """The same contract for the three fields the query sweep cannot see.

        `_form_raw`, `_config_editor_raw` and `_spreadsheet_editor_raw` edit a
        *value* on `edit_buf`, so they carry no `query` and the sweep above
        skips them entirely — which is how they came to be the last three
        caret-less surfaces in the shell. Sweeping by `edit_buf = ""` instead
        closes that hole, and keeps it closed if a fourth field is added.
        """
        boxes = [
            (mod, name, body)
            for mod in self.MODULES
            for name, body in self._boxes(mod, needle='edit_buf = ""')
        ]
        assert len(boxes) >= 3, (
            f"expected at least 3 value fields, found {len(boxes)}: "
            f"{[f'{m}.{n}' for m, n, _ in boxes]}"
        )
        for mod, name, body in boxes:
            where = f"{mod}.{name}"
            # Whitespace-flattened: the dispatch is written across three lines
            # because it does not fit in one, and an assertion that only saw
            # one spelling would fail on a re-indent rather than on a real
            # unwiring. The words are the contract, not where the breaks are.
            flat = " ".join(body.split())
            assert "edit_cur = 0" in body, f"{where}: never initialises its caret"
            # `source` is the form's password mask; the other two fields paint
            # the buffer itself. Either is the shared view doing its job.
            assert _re.search(r"_filter_view\((?:edit_buf|source), edit_cur", flat), (
                f"{where}: renders the value without a caret on it"
            )
            assert _re.search(r"self\._filter_key\(\s*edit_buf,\s*edit_cur,\s*key\s*\)", flat), (
                f"{where}: its keys never reach the shared editor"
            )
            assert "edit_cur = len(edit_buf)" in flat, (
                f"{where}: opens a field without parking the caret at the end"
            )


# ── the fields that were not filter boxes ────────────────────────────────────


class TestFormCaret:
    """`_form_raw`'s text fields, which the filter-box sweep does not cover.

    The sweep reached every box whose string is a *query* — a filter over an
    option list. A form edits a *value*: a different method, a different buffer
    (`edit_buf`, not `query`), and a line already running in reverse video with
    a `│` frozen at its end. Same caret convention, none of the motion.
    """

    FIELDS = [{"label": "Name", "type": "text", "default": ""}]
    MESSAGE = "Save?"

    @staticmethod
    def _drive(p, keys, fields=None) -> tuple[dict, str]:
        """Run `_form_raw` to completion, returning its result and stdout.

        Always sandwiches *keys* between an Esc-to-edit opener and the
        `ENTER`/`Esc` closer, because `_form_raw` only returns from its
        not-editing branch: an Esc pressed *inside* a field just leaves it
        without saving, which would drive the key script past its end.
        """
        p._session = _ScriptedSession(
            [_RawKey(_KEY_ENTER), *keys, _RawKey(_KEY_ENTER), _RawKey(_KEY_ESC)]
        )
        buf = _io_mod.StringIO()
        with _contextlib.redirect_stdout(buf):
            result = p._form_raw(
                fields if fields is not None else TestFormCaret.FIELDS,
                TestFormCaret.MESSAGE,
            )
        return result, buf.getvalue()

    @staticmethod
    def _editing(out: str, marker: str = "│") -> str:
        """The last repaint while a field was open, escapes stripped."""
        return _frame_with(out, TestFormCaret.MESSAGE, marker)

    @staticmethod
    def _field(out: str, label: str = "Name", marker: str = "│") -> str:
        """The cell contents of `<label>:` in the frame that showed *marker*.

        Asking for the marker on the same line is what makes this assert the
        caret's *position* rather than the text's identity: without it,
        `Name: Xabc` would pass whether the caret sat at column 0 or at the end.
        """
        for ln in TestFormCaret._editing(out, marker).splitlines():
            if f"{label}:" in ln and marker in ln:
                return ln.strip().split(f"{label}:", 1)[1].strip()
        raise AssertionError(f"no line showed {label} with {marker!r}")

    @staticmethod
    def _value(out: str, label: str = "Name") -> str:
        """The field's contents with the caret marker removed."""
        return TestFormCaret._field(out, label).replace("│", "")

    def test_editing_shows_the_caret_where_the_caret_is(self):
        p, _ = _make_prompt()
        _, out = self._drive(p, _chars("abc"))
        assert self._field(out) == "abc│"

    def test_home_makes_room_at_the_front(self):
        p, _ = _make_prompt()
        _, out = self._drive(p, [*_chars("abc"), _RawKey(_KEY_HOME), _RawKey(_KEY_CHAR, "X")])
        assert self._value(out) == "Xabc"
        # the marker proves the caret is genuinely at column 0, not merely that
        # the insert happened to land there
        assert self._field(out) == "X│abc"

    def test_word_jump_lands_between_the_words(self):
        p, _ = _make_prompt()
        _, out = self._drive(
            p,
            [
                *_chars("foo bar"),
                _Key(_KEY_LEFT, ctrl=True),
                _RawKey(_KEY_CHAR, "X"),
            ],
        )
        assert self._value(out) == "foo Xbar"
        assert self._field(out) == "foo X│bar"

    def test_the_saved_value_follows_the_caret(self):
        # A caret is only worth having if what it edits is what gets saved —
        # the render and `results` must agree, or the field lies to the user.
        p, _ = _make_prompt()
        result, _ = self._drive(
            p,
            [*_chars("abc"), _RawKey(_KEY_HOME), _RawKey(_KEY_CHAR, "X")],
        )
        assert result == {"Name": "Xabc"}

    def test_rubout_deletes_the_char_left_of_the_caret(self):
        p, _ = _make_prompt()
        _, out = self._drive(
            p,
            [*_chars("abc"), _RawKey(_KEY_LEFT), _RawKey(_KEY_BACKSPACE)],
        )
        assert self._value(out) == "ac"
        assert self._field(out) == "a│c"

    def test_password_fields_mask_but_still_track_the_caret(self):
        # The mask is one `*` per character, so index-for-index the caret sits
        # in the same place in mask space — but only if the render feeds the
        # mask through the shared view instead of painting stars and a static
        # bar separately, which is exactly the mistake this catches.
        p, _ = _make_prompt()
        result, out = self._drive(
            p,
            [
                *_chars("secret"),
                _RawKey(_KEY_HOME),
                _RawKey(_KEY_CHAR, "X"),
            ],
            fields=[{"label": "Pass", "type": "password", "default": ""}],
        )
        assert self._field(out, "Pass") == "*│******"
        assert result == {"Pass": "Xsecret"}


class TestSpreadsheetCaret:
    """The third text-entry surface, and the one a table lives or dies by.

    A spreadsheet cell is where a caret matters most — you are editing inside
    a fixed-width column, so a caret pinned to the end tells you nothing about
    where the next character lands. It was still a bare `str` with a `│` glued
    to the end of the cell.
    """

    HEADERS = ["Name"]
    ROWS = [[""]]

    @staticmethod
    def _drive(p, keys, rows=None) -> tuple[list[list[str]], str]:
        """Run `_spreadsheet_editor_raw`, returning the rows and stdout.

        Same sandwich as the form: enter edit mode, run *keys*, confirm the
        cell, then Esc — the only key that returns from the not-editing branch.
        """
        data = rows if rows is not None else [[""]]
        p._session = _ScriptedSession(
            [_RawKey(_KEY_ENTER), *keys, _RawKey(_KEY_ENTER), _RawKey(_KEY_ESC)]
        )
        buf = _io_mod.StringIO()
        with _contextlib.redirect_stdout(buf):
            result = p._spreadsheet_editor_raw(TestSpreadsheetCaret.HEADERS, data, "Sheet:")
        return result, buf.getvalue()

    @staticmethod
    def _cell(out: str) -> str:
        """The cell contents from the frame that showed the caret."""
        frame = _frame_with(out, "Sheet:", "│")
        for ln in frame.splitlines():
            if "│" in ln:
                return ln.strip()
        raise AssertionError("no cell was ever painted with a caret in it")

    @staticmethod
    def _value(out: str) -> str:
        return TestSpreadsheetCaret._cell(out).replace("│", "")

    def test_editing_shows_the_caret_at_the_end_of_the_cell(self):
        p, _ = _make_prompt()
        _, out = self._drive(p, _chars("abc"))
        assert self._cell(out) == "abc│"

    def test_home_makes_room_at_the_front_of_the_cell(self):
        p, _ = _make_prompt()
        _, out = self._drive(p, [*_chars("abc"), _RawKey(_KEY_HOME), _RawKey(_KEY_CHAR, "X")])
        assert self._cell(out) == "X│abc"

    def test_word_jump_lands_between_the_words(self):
        p, _ = _make_prompt()
        _, out = self._drive(
            p,
            [
                *_chars("prod-v2"),
                _Key(_KEY_LEFT, ctrl=True),
                _RawKey(_KEY_CHAR, "X"),
            ],
        )
        # `prod-v2` is one word — the delimiter is whitespace, so the jump goes
        # to the front of the whole token rather than to the hyphen.
        assert self._value(out) == "Xprod-v2"
        assert self._cell(out) == "X│prod-v2"

    def test_the_saved_cell_follows_the_caret(self):
        # The rendered cell and the row that gets returned must agree, or the
        # table shows one thing and saves another.
        p, _ = _make_prompt()
        result, _ = self._drive(p, [*_chars("abc"), _RawKey(_KEY_HOME), _RawKey(_KEY_CHAR, "X")])
        assert result == [["Xabc"]]

    def test_rubout_deletes_the_char_left_of_the_caret(self):
        p, _ = _make_prompt()
        _, out = self._drive(p, [*_chars("abc"), _RawKey(_KEY_LEFT), _RawKey(_KEY_BACKSPACE)])
        assert self._cell(out) == "a│c"


# ── what the terminal actually ends up showing ───────────────────────────────

_CSI_PARTS = _re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")


def _screen(raw: str, *, strip_trailing: bool = True) -> list[str]:
    """Feed a capture through a terminal and return the rows still visible.

    Only four sequences move or mark anything these prompts write: line feed,
    carriage return, cursor-up (`1A`) and erase-line (`2K`). Everything else is
    SGR or cursor visibility — it changes how a row looks or whether the caret
    blinks, never where the cursor is — so it is skipped rather than modelled.

    Rows are a dict keyed by index rather than a list, so a row erased and
    rewritten later reads as its final content and not as two rows. That is
    the whole point: a prompt that erases one line and writes the rest below
    leaves its earlier copy intact, and only a keyed grid shows it there.

    `strip_trailing=False` keeps the pad, because a cell padded past the
    terminal's right edge still moves the cursor past it. Trimming it by
    default reads a row comfortably inside the margin when the element
    actually drew one column over it.
    """
    if not raw:
        return []
    rows: dict[int, str] = {}
    row = col = 0
    i = 0
    while i < len(raw):
        if raw[i] == "\x1b":
            m = _CSI_PARTS.match(raw, i)
            if not m:
                i += 1
                continue
            params, final = m.group(1), m.group(2)
            if final == "A":  # cursor up — the only mover they use
                row = max(0, row - (int(params) if params.isdigit() else 1))
            elif final == "K":  # `_ERASE_LINE` is `2K`: the whole row
                rows[row] = ""
            i = m.end()
            continue
        if raw[i] == "\n":
            row += 1
            col = 0
        elif raw[i] == "\r":
            col = 0
        else:
            cur = rows.get(row, "")
            rows[row] = (cur + " " * col)[:col] + raw[i] + cur[col + 1 :]
            col += 1
        i += 1
    if not rows:
        return []
    drawn = [rows.get(r, "") for r in range(min(rows), max(rows) + 1) if rows.get(r, "").strip()]
    return [r.rstrip() for r in drawn] if strip_trailing else drawn


class TestRepaintLeavesNothingBehind:
    """A prompt must paint *over* its previous frame, not under it.

    Eleven of `interactive_prompt`'s prompts wrote the whole form again below
    the one already on screen:

        while True:
            sys.stdout.write(f"\\r{_ERASE_LINE}\\r{_render()}")

    `_ERASE_LINE` erases the single row the cursor happens to sit on — the
    *last* row of the frame just painted — and `_render` then starts writing
    the whole form from there, two rows further down each time. A handful of
    keystrokes scrolled a dozen stale copies past the user, and a caret drawn
    into the newest copy sat above four that still claimed to be live.

    `_file_browser_raw` and `_multi_choice_raw`, in the same file, already do
    it correctly: rewind to the top of the previous frame, then paint. This
    pins that as the contract for the whole module.
    """

    # Prompts that return on a plain Esc, so a character script alone drives
    # them all the way out. `_diff_viewer_raw`, `_wizard_raw` and
    # `_playlist_manager_raw` end on keys no character script reaches — they
    # are covered by the sweep below instead.
    DRIVES = [
        ("_history_search_raw", (["one", "two", "three"], "Search history:")),
        ("_config_editor_raw", ({"Host": "localhost"}, "Config:")),
        ("_form_raw", ([{"label": "Name", "type": "text", "default": ""}], "Save?")),
    ]

    @staticmethod
    def _run(meth: str, args: tuple) -> str:
        p = InteractivePrompt(MemoryIO())
        p._session = _ScriptedSession([*_chars("abcdef"), _RawKey(_KEY_ESC)])
        buf = _io_mod.StringIO()
        with _contextlib.redirect_stdout(buf):
            getattr(p, meth)(*args)
        return buf.getvalue()

    def test_it_paints_on_every_key_yet_keeps_no_copy(self):
        keys = "abcdef"
        for meth, args in self.DRIVES:
            raw = self._run(meth, args)

            # It really did repaint, once per key: the loop's own leading
            # erase plus the renderer's. Without this, the clean screen below
            # would pass just as happily on a prompt that drew nothing at all.
            assert raw.count(_ERASE_LINE) >= 2 * len(keys), (
                f"{meth}: did not repaint once per keystroke"
            )

            # And it really did paint something: everything before the first
            # cursor-up is the first frame, before any clear touched it.
            first = _screen(raw.split(_CURSOR_UP, 1)[0])
            assert first, f"{meth}: its first frame painted nothing at all"

            # The contract — having painted six times, it still leaves none
            # of it behind on the way out.
            assert _screen(raw) == [], f"{meth}: left a copy of itself on screen: {_screen(raw)!r}"

    def test_every_prompt_loop_rewinds_before_it_paints(self):
        """All eleven, including the three a character script cannot reach.

        The test above exercises three prompts by driving them. The other
        three end on keys no character script can produce, so nothing
        behavioural covers them. This asks the source the same question the
        terminal is asked above: does the loop return to the top of the frame
        it is replacing *before* it writes the next one?
        """
        import importlib
        import inspect as _inspect

        pkg = importlib.import_module("domain.shell._internal.interactive_prompt")
        src = _inspect.getsource(pkg)

        # Both spellings of the paint site: the one that measures nothing and
        # the one that does. Keying the sweep off the old spelling alone would
        # make it blind the moment the fix lands — every method would stop
        # matching and the count would drop to zero, which reads as a pass
        # only because of the assertion that follows it.
        unmeasured = "{_ERASE_LINE}\\r{_render()}"
        measured = "{_ERASE_LINE}\\r{frame}"

        found = []
        for m in _re.finditer(r"^    def (\w+)\(", src, _re.MULTILINE):
            nxt = _re.search(r"^    def ", src[m.end() :], _re.MULTILINE)
            body = src[m.start() : m.end() + (nxt.start() if nxt else len(src))]
            if unmeasured not in body and measured not in body:
                continue
            found.append(m.group(1))
            where = f"interactive_prompt.{m.group(1)}"
            assert unmeasured not in body, (
                f"{where}: still writes a frame it never measured, so the copy "
                "lands below the one it just drew"
            )
            assert "frame = _render()" in body, f"{where}: does not keep the frame"
            assert "prev_rows = 0" in body, f"{where}: never tracks its frame height"
            # Written with `\\n` so the search string holds a literal
            # backslash-n, the way the source file spells it.
            assert 'prev_rows = frame.count("\\n") + 1' in body, (
                f"{where}: does not measure the frame it just painted"
            )
            rewind = body.find("_rewind(prev_rows)")
            paint = body.find('sys.stdout.write(f"\\r{_ERASE_LINE}\\r{frame}"')
            assert rewind != -1, f"{where}: never rewinds to the top of its frame"
            assert paint != -1, f"{where}: no longer writes the frame the way it should"
            assert rewind < paint, f"{where}: paints before it rewinds, so the copy lands below"
        assert len(found) == 11, f"expected 11 repaint-shaped prompts, found {len(found)}: {found}"


class TestTheRestOfTheFamilyRepaintsAndClears:
    """The same contract for the five modules `interactive_prompt` did not cover.

    The class above pins eleven prompts in `interactive_prompt`. The defect
    itself never lived only there: driving all 121 elements through a real
    terminal grid found **58 methods in five other modules** carrying one or
    both halves of it — 50 paints and 45 clears, sharing 37.

    **Paint.** `f"\\r{_ERASE_LINE}\\r{_render()}"` erases the single row the
    cursor sits on — the frame's *last* row — and writes the whole frame from
    there, so every key leaves N-1 rows of the previous copy below the new
    one. This is measured rather than inferred: every element that grew grew
    by *exactly* 2*(N-1) for a frame N rows tall, which is pure stacking with
    no content change in any of them. `_select_with_grouping_raw` ran
    19 -> 55, `_kanban_board_raw` 17 -> 49, `_color_picker_raw` 2 -> 4.

    **Clear.** `for _ in range(n): UP; ERASE` walks up *before* it erases, so
    it clears the n rows above the cursor and never the one it is sitting on:
    the frame's own last row survives. 45 elements left rows behind after
    returning, and not one of `interactive_prompt`'s fixed prompts did — that
    comparison is what names this defect rather than some other.

    `interactive_input` is deliberately absent. Its two prompts paint the same
    way, but both are one row tall: erase-one-and-write-one is a correct
    repaint regardless of whether anything measured it, and neither stacked
    nor leaked when driven. Measured harmless, left alone.
    """

    MODULES = (
        "interactive_select",
        "interactive_confirm",
        "interactive_table",
        "interactive_widget",
        "interactive_picker",
    )

    # Return on a plain Esc, so a character script drives them all the way out.
    EXIT_DRIVES = [
        ("_select_with_filter_raw", ("Pick", ["alpha", "bravo", "charlie"], "")),
        ("_confirm_action_raw", ("Deploy", "to production", False)),
        ("_table_with_search_raw", (["Name"], [["alpha"]], "Table")),
        ("_multi_confirm_raw", ("Pick", ["alpha", "bravo"], False)),
        ("_color_picker_raw", ("Pick", "#ffffff")),
    ]

    # Stopped by an exhausted script rather than returned: the frame has to be
    # inspected *while still up*, which is the only place stacking shows. The
    # assertion compares the screen to the last frame alone, so it holds
    # whatever typing did to the element's content.
    GROWTH_DRIVES = [
        ("_select_with_filter_raw", ("Pick", ["alpha", "bravo", "charlie"], "")),
        ("_confirm_action_raw", ("Deploy", "to production", False)),
        (
            "_select_table_with_preview_raw",
            (["Name"], [["alpha"]], lambda rows: "preview"),
        ),
        ("_multi_confirm_raw", ("Pick", ["alpha", "bravo"], False)),
        ("_color_picker_raw", ("Pick", "#ffffff")),
    ]

    PAINT_PREFIX = "\r" + _ERASE_LINE + "\r"  # where every paint begins

    @staticmethod
    def _run(meth: str, args: tuple, keys: str, *, exits: bool) -> str:
        """Everything *meth* wrote over *keys*, stopping it either way.

        `exits=True` ends on Esc so the prompt clears and returns, which is
        what the exit contract is about. `exits=False` leaves it waiting for a
        key that never comes: the script runs out mid-loop, the frame stays on
        screen, and that is the state the stacking contract inspects. Either
        way the raise has to be the script running dry — anything else means
        the element broke, and a broken element must not read as a passing one.
        """
        p = InteractivePrompt(MemoryIO())
        tail = [_RawKey(_KEY_ESC)] if exits else []
        p._session = _ScriptedSession([*_chars(keys), *tail])
        buf = _io_mod.StringIO()
        with _contextlib.redirect_stdout(buf):
            try:
                getattr(p, meth)(*args)
            except AssertionError as e:
                assert "exhausted" in str(e), e
        return buf.getvalue()

    def test_repaint_replaces_its_frame_instead_of_piling_under_it(self):
        for meth, args in self.GROWTH_DRIVES:
            raw = self._run(meth, args, "ab", exits=False)

            # It really did repaint: an initial paint plus one per key. The
            # unfixed shape writes one erase per paint, the fixed one one more
            # per rewind, so this floor admits both and lets the contract
            # below be the thing that decides.
            assert raw.count(_ERASE_LINE) >= len("ab") + 1, (
                f"{meth}: did not repaint once per keystroke"
            )
            assert self.PAINT_PREFIX in raw, f"{meth}: never painted a frame"

            last = raw.rsplit(self.PAINT_PREFIX, 1)[-1]
            # The screen shows the frame that was last painted and nothing
            # else. A prompt stacking under itself fails this with the stale
            # rows still counted on screen; one that repainted in place passes
            # however its content changed while being driven.
            assert _screen(last), f"{meth}: its last frame painted nothing"
            stale = len(_screen(raw)) - len(_screen(last))
            assert stale <= 0, f"{meth}: left {stale} stale row(s) below its last frame"

    def test_the_contract_is_one_already_met_by_a_fixed_prompt(self):
        """The assertion above is satisfiable *today*, so a failure means the fix.

        `_history_search_raw` was among the eleven fixed first. If this could
        not pass while the other four fail, the green that follows the fix
        would prove nothing about the fix — it would only show the test had
        been shaped around a result.
        """
        raw = self._run("_history_search_raw", (["one", "two"], "Search:"), "ab", exits=False)
        assert self.PAINT_PREFIX in raw
        last = raw.rsplit(self.PAINT_PREFIX, 1)[-1]
        assert len(_screen(raw)) <= len(_screen(last)), (
            "_history_search_raw: a prompt already fixed does not pass this contract"
        )

    def test_returning_leaves_the_screen_empty(self):
        for meth, args in self.EXIT_DRIVES:
            raw = self._run(meth, args, "ab", exits=True)
            left = _screen(raw)
            assert left == [], f"{meth}: left rows behind on exit: {left!r}"

    def test_no_module_paints_a_frame_it_never_measured(self):
        """Every paint site in the five modules measures the frame it wrote.

        Keying the sweep off the unfixed spelling alone would make it blind the
        moment the fix lands — the count would fall to zero and read as a pass
        only because of the count that follows it, so methods are found by
        *either* spelling and judged on the source inside them.
        """
        import importlib
        import inspect as _inspect

        unmeasured = "{_ERASE_LINE}\\r{_render()}"
        measured = "{_ERASE_LINE}\\r{frame}"
        found: list[str] = []
        for mod in self.MODULES:
            src = _inspect.getsource(importlib.import_module(f"domain.shell._internal.{mod}"))
            for m in _re.finditer(r"^    def (\w+)\(", src, _re.MULTILINE):
                nxt = _re.search(r"^    def ", src[m.end() :], _re.MULTILINE)
                body = src[m.start() : m.end() + (nxt.start() if nxt else len(src))]
                if unmeasured not in body and measured not in body:
                    continue
                where = f"{mod}.{m.group(1)}"
                found.append(where)
                assert unmeasured not in body, (
                    f"{where}: still writes a frame it never measured, so the copy "
                    "lands below the one it just drew"
                )
                assert "frame = _render()" in body, f"{where}: does not keep the frame"
                assert "prev_rows = 0" in body, f"{where}: never tracks its frame height"
                # Written with `\\n` so the search string holds a literal
                # backslash-n, the way the source file spells it.
                assert 'prev_rows = frame.count("\\n") + 1' in body, (
                    f"{where}: does not measure the frame it just painted"
                )
                rewind = body.find("_rewind(prev_rows)")
                paint = body.find('sys.stdout.write(f"\\r{_ERASE_LINE}\\r{frame}"')
                assert rewind != -1, f"{where}: never rewinds to the top of its frame"
                assert paint != -1, f"{where}: no longer writes the frame as it should"
                assert rewind < paint, f"{where}: paints before it rewinds, so the copy lands below"
        # 51, not 50: `_color_picker_rgb_raw` already carried the repaint fix
        # from the colour-picker defect work, so it matches the shape without
        # being part of the 50 N1 converted. Counting only N1's own output
        # would make this assert fail the moment a second module is fixed.
        assert len(found) == 51, (
            f"expected 51 repaint-shaped prompts across the five modules, "
            f"found {len(found)}: {found}"
        )

    def test_no_module_clears_upwards_before_it_erases(self):
        """No method in the five modules walks up before it erases.

        The two lines are only ever adjacent when the erase is *inside* a
        `for _ in range(...)` loop, which is what makes this checkable from
        the source: a lone `up`+`erase` moves the cursor deliberately and is
        not the defect. The `_rewind` assertion on each module is what stops
        this passing vacuously — a module that never cleared at all would
        have no loop-clear either.
        """
        import importlib
        import inspect as _inspect

        checked: list[str] = []
        for mod in self.MODULES:
            src = _inspect.getsource(importlib.import_module(f"domain.shell._internal.{mod}"))
            lines = src.splitlines()
            offenders: list[str] = []
            for i, ln in enumerate(lines):
                if "{_CURSOR_UP}{_ERASE_LINE}" not in ln:
                    continue
                if not (i and lines[i - 1].lstrip().startswith("for _ in range")):
                    continue
                for j in range(i, 0, -1):
                    if lines[j].startswith("    def "):
                        offenders.append(lines[j].strip()[4:].split("(")[0])
                        break
            assert not offenders, (
                f"{mod}: clears up before it erases, so the frame's own last row "
                f"survives: {sorted(set(offenders))}"
            )
            assert "_rewind(" in src, (
                f"{mod}: never rewinds — the sweep above would pass on a module "
                "that does not clear at all"
            )
            checked.append(mod)
        assert checked == list(self.MODULES), "a module escaped the sweep"


class TestDiffFitsTheTerminal:
    """`diff` sets two columns side by side; both must fit the terminal.

    At width 80 it drew a 232-column separator: `"  ─" * col_w` repeats the
    two spaces along with the dash, so each half ran three times too wide.
    Header, separator and body rows then put their divider at four different
    columns — 41, 116, 51, 42 — because the body pads a cell that may already
    be wrapped in `ESC[31m`/`ESC[0m`, and those nine bytes count as content,
    so a coloured row gets nine fewer visible spaces than the plain one above
    it.

    One divider column for every row, and no row wider than the terminal that
    is showing it.
    """

    WIDTH = 80
    DIVIDER = "│"
    CROSS = "┼"

    @staticmethod
    def _render(
        left: list[str],
        right: list[str],
        left_label: str = "alpha bravo charlie",
        right_label: str = "alpha bravo charlie",
    ) -> list[str]:
        p = InteractivePrompt(MemoryIO())
        p._is_tty = True
        p.diff(left_label, left, right_label, right)
        raw = "".join(getattr(p._io, "_output", []))
        return _screen(raw, strip_trailing=False)

    @classmethod
    def _dividers(cls, rows: list[str]) -> list[int]:
        """Column of each row's divider — `│` in the header and body, `┼`
        in the separator. Two glyphs, one column between them."""
        cols = []
        for row in rows:
            if cls.DIVIDER in row:
                cols.append(row.index(cls.DIVIDER))
            elif cls.CROSS in row:
                cols.append(row.index(cls.CROSS))
        return cols

    def test_no_row_runs_past_the_terminal(self):
        rows = self._render(["alpha", "bravo"], ["alpha", "charlie"])
        assert rows, "diff painted nothing"
        over = [len(r) for r in rows if len(r) > self.WIDTH]
        assert not over, f"rows wider than {self.WIDTH} columns: {over}"

    def test_every_row_puts_its_divider_in_one_column(self):
        rows = self._render(["alpha", "bravo"], ["alpha", "charlie"])
        cols = self._dividers(rows)
        assert len(cols) >= 2, f"only {len(cols)} row(s) carry a divider"
        assert len(set(cols)) == 1, (
            f"dividers at columns {sorted(set(cols))} — a table whose columns do not line up"
        )

    def test_a_coloured_row_does_not_shift_its_divider(self):
        # Left and right differ only on line two, so the first body row comes
        # back plain and the second wrapped in colour. Padding after colouring
        # is what makes those two disagree by exactly the escape length.
        rows = self._render(["same", "alpha"], ["same", "bravo"])
        cols = self._dividers(rows)
        assert len(set(cols)) == 1, f"dividers at columns {sorted(set(cols))}"

    def test_a_long_label_still_fits(self):
        rows = self._render(["alpha"], ["bravo"], left_label="x" * 200, right_label="y" * 200)
        assert max(len(r) for r in rows) <= self.WIDTH


class TestColorPickerRepaintsInPlace:
    """The RGB colour picker must sit on the same rows after every key.

    It painted a five-row frame behind a one-row erase: `\\r{_ERASE_LINE}`
    clears the line the cursor is on, and after painting, that is the frame's
    *last* line. Each keystroke therefore overwrote the old footer and dropped
    the next copy four rows lower — measured heights 5, 9, 13, 17 across four
    keys.

    Its own `_clear()` already rewinds `range(5)`, so the height was known; the
    repaint simply never used it. `_clear()` is wrong in the opposite
    direction for the same reason: `up` then `erase` erases the row *above*
    the cursor and leaves the footer the cursor is sitting on, so Esc stranded
    that one row.
    """

    FRAME = 5
    MESSAGE = "Deploy to production?"

    @classmethod
    def _run(cls, n_keys: int, tail: tuple = ()) -> str:
        p = InteractivePrompt(MemoryIO())
        p._is_tty = True
        p._session = _ScriptedSession([*_chars("a" * n_keys), *tail])
        buf = _io_mod.StringIO()
        try:
            with _contextlib.redirect_stdout(buf):
                p._color_picker_rgb_raw(cls.MESSAGE, "#ffffff")
        except AssertionError:
            pass  # script ran dry — the prompt is waiting for another key
        return buf.getvalue()

    def test_it_holds_its_height_across_every_keypress(self):
        heights = [len(_screen(self._run(n))) for n in range(4)]
        assert heights[0] == self.FRAME, f"first frame is {heights[0]} rows, expected {self.FRAME}"
        assert len(set(heights)) == 1, (
            f"heights {heights} after 0,1,2,3 keys — it stacks a fresh "
            f"copy per keypress instead of repainting in place"
        )

    def test_esc_leaves_nothing_behind(self):
        raw = self._run(0, tail=(_RawKey(_KEY_ESC),))
        left = _screen(raw)
        assert left == [], f"Esc left {left} row(s) on screen"
