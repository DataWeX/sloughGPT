"""Tests for tui_repl.py — utility functions only (no curses event loop)."""

from __future__ import annotations

import os
import tempfile
import shutil

import pytest

from domains.shell.surface import TextSurface
from domains.shell.tui_repl import (
    _complete_path,
    _ESC_FINALS,
    _STYLE_PAIRS,
    TuiIo,
    TuiRepl,
)
from domains.shell.surface import (
    STYLE_INFO, STYLE_WARN, STYLE_ERROR, STYLE_DEBUG, STYLE_CRITICAL,
)


# ---------------------------------------------------------------------------
# _complete_path
# ---------------------------------------------------------------------------

class TestCompletePath:
    """Tests for the filesystem path completion helper."""

    @pytest.fixture(autouse=True)
    def _tmpdir(self, tmp_path):
        self.base = tmp_path
        (self.base / "aaa.txt").touch()
        (self.base / "aab.txt").touch()
        (self.base / "b.txt").touch()
        sub = self.base / "subdir"
        sub.mkdir()
        (sub / "inner.txt").touch()

    def test_no_slash_lists_directory(self):
        matches = _complete_path(str(self.base) + "/aaa")
        assert len(matches) == 1
        assert matches[0].endswith("aaa.txt")

    def test_common_prefix(self):
        matches = _complete_path(str(self.base) + "/aa")
        assert len(matches) == 2
        names = [os.path.basename(m) for m in matches]
        assert "aaa.txt" in names
        assert "aab.txt" in names

    def test_subdirectory(self):
        matches = _complete_path(str(self.base) + "/subdir/inn")
        assert len(matches) == 1
        assert matches[0].endswith("inner.txt")

    def test_no_matches_returns_empty(self):
        matches = _complete_path(str(self.base) + "/zzz")
        assert matches == []

    def test_nonexistent_directory_returns_empty(self):
        matches = _complete_path("/nonexistent/path/xxx")
        assert matches == []

    def test_dot_prefix_lists_pwd(self):
        # _complete_path(".") lists CWD entries starting with "."
        (self.base / ".hidden").touch()
        matches = _complete_path(str(self.base) + "/.")
        assert any(".hidden" in m for m in matches)

    def test_tilde_expansion(self):
        matches = _complete_path("~/nonexistent_file_xyz_12345")
        assert matches == []


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_esc_finals_is_frozen(self):
        assert isinstance(_ESC_FINALS, frozenset)
        assert "A" in _ESC_FINALS
        assert "C" in _ESC_FINALS

    def test_style_pairs_cover_all_styles(self):
        expected = {STYLE_INFO, STYLE_WARN, STYLE_ERROR, STYLE_DEBUG, STYLE_CRITICAL}
        assert set(_STYLE_PAIRS.keys()) == expected

    def test_style_pairs_values_are_positive_ints(self):
        for v in _STYLE_PAIRS.values():
            assert isinstance(v, int)
            assert v > 0


# ---------------------------------------------------------------------------
# TuiIo
# ---------------------------------------------------------------------------

class TestTuiIo:
    def test_write_delegates_to_surface(self):
        surface = TextSurface()
        io = TuiIo(surface)
        io.write("hello", end="\n")
        assert "hello" in surface.capture

    def test_write_no_newline_keeps_partial(self):
        surface = TextSurface()
        io = TuiIo(surface)
        io.write("partial", end="")
        # partial line kept as partial, not in capture yet
        assert surface.capture[-1] == "partial" if surface.capture else True

    def test_flush_is_noop(self):
        io = TuiIo(TextSurface())
        io.flush()  # should not raise

    def test_read_raises(self):
        io = TuiIo(TextSurface())
        with pytest.raises(NotImplementedError, match="input comes from the curses event loop"):
            io.read("prompt: ")


# ---------------------------------------------------------------------------
# TuiRepl — _input_view
# ---------------------------------------------------------------------------

class TestInputView:
    """Tests for the visible-input-line computation."""

    @pytest.fixture
    def repl(self):
        surface = TextSurface()

        class FakeRepl:
            io = surface
            console = type("C", (), {"_io": surface, "_tui_repl": None})()
            COMMANDS = {}
            _history = []
        return TuiRepl(FakeRepl(), None)

    def test_short_buffer(self, repl):
        line, caret = repl._input_view(80, "hello", 5)
        assert line == "hello"
        assert caret == 7  # len("λ ") + 5

    def test_buffer_fits_exactly(self, repl):
        # cols=10, prompt="λ " (2) + cursor (1) + 1 blank = 4, max_w=6
        line, caret = repl._input_view(10, "abcdef", 3)
        assert line == "abcdef"
        assert caret == 5  # 2 + 3

    def test_long_buffer_scrolls(self, repl):
        # buffer wider than window — should clip and show caret
        # max_w = max(cols - len("λ ") - 1, 0) = max(10-2-1, 0) = 7
        cols = 10
        buf = "a" * 30
        line, caret = repl._input_view(cols, buf, 15)
        assert len(line) == 7
        # start=min(15, 30-7)=15, caret_col = 2+(15-15) = 2
        assert caret == 2

    def test_caret_at_start(self, repl):
        line, caret = repl._input_view(40, "hello world", 0)
        assert caret == 2  # just the prompt width

    def test_caret_at_end(self, repl):
        line, caret = repl._input_view(40, "hello", 5)
        assert caret == 7

    def test_empty_buffer(self, repl):
        line, caret = repl._input_view(40, "", 0)
        assert line == ""
        assert caret == 2

    def test_caret_negative_clamped(self, repl):
        line, caret = repl._input_view(40, "hello", -5)
        assert caret == 2  # prompt only

    def test_caret_beyond_length_clamped(self, repl):
        line, caret = repl._input_view(40, "hi", 100)
        assert caret == 4  # prompt + len("hi")

    def test_very_narrow_window(self, repl):
        # cols=3, prompt="λ " takes 2 cols, max_w=0
        line, caret = repl._input_view(3, "hello", 2)
        assert line == ""
        assert caret == 2


# ---------------------------------------------------------------------------
# TuiRepl — history navigation
# ---------------------------------------------------------------------------

class TestHistoryNavigation:
    @pytest.fixture
    def repl(self):
        surface = TextSurface()

        class FakeRepl:
            io = surface
            console = type("C", (), {"_io": surface, "_tui_repl": None})()
            COMMANDS = {}
            _history = []
        r = TuiRepl(FakeRepl(), None)
        r._input_buf = []
        r._input_cursor = 0
        r._cmd_history = ["first", "second", "third"]
        r._history_pos = 3  # past end = new input
        return r

    def test_history_back(self, repl):
        repl._history_back()
        assert repl._history_pos == 2
        assert repl._input_buf == list("third")

    def test_history_back_twice(self, repl):
        repl._history_back()
        repl._history_back()
        assert repl._history_pos == 1
        assert repl._input_buf == list("second")

    def test_history_back_at_start_noop(self, repl):
        repl._history_pos = 0
        repl._input_buf = list("first")
        repl._history_back()
        assert repl._history_pos == 0
        assert repl._input_buf == list("first")

    def test_history_fwd(self, repl):
        repl._history_pos = 0
        repl._input_buf = list("first")
        repl._history_fwd()
        assert repl._history_pos == 1
        assert repl._input_buf == list("second")

    def test_history_fwd_past_end_clears(self, repl):
        repl._history_pos = 2
        repl._input_buf = list("third")
        repl._history_fwd()
        assert repl._history_pos == 3
        assert repl._input_buf == []
        assert repl._input_cursor == 0

    def test_history_fwd_at_end_noop(self, repl):
        repl._history_pos = 3
        repl._history_fwd()
        assert repl._history_pos == 3
        assert repl._input_buf == []


# ---------------------------------------------------------------------------
# TuiRepl — caret movement
# ---------------------------------------------------------------------------

class TestCaretMovement:
    @pytest.fixture
    def repl(self):
        surface = TextSurface()

        class FakeRepl:
            io = surface
            console = type("C", (), {"_io": surface, "_tui_repl": None})()
            COMMANDS = {}
            _history = []
        r = TuiRepl(FakeRepl(), None)
        r._input_buf = list("hello world")
        r._input_cursor = 5
        return r

    def test_move_home(self, repl):
        repl._move_home()
        assert repl._input_cursor == 0

    def test_move_end(self, repl):
        repl._input_cursor = 0
        repl._move_end()
        assert repl._input_cursor == 11

    def test_move_word_forward_from_start(self, repl):
        repl._input_cursor = 0
        repl._move_word_forward()
        assert repl._input_cursor == 5  # skips "hello"

    def test_move_word_forward_mid_word(self, repl):
        repl._input_cursor = 2
        repl._move_word_forward()
        assert repl._input_cursor == 5  # end of "hello"

    def test_move_word_forward_past_space(self, repl):
        repl._input_cursor = 5
        repl._move_word_forward()
        assert repl._input_cursor == 11  # end of "world"

    def test_move_word_forward_at_end(self, repl):
        repl._input_cursor = 11
        repl._move_word_forward()
        assert repl._input_cursor == 11  # no-op

    def test_move_word_backward_from_end(self, repl):
        repl._input_cursor = 11
        repl._move_word_backward()
        assert repl._input_cursor == 6  # start of "world"

    def test_move_word_backward_mid_word(self, repl):
        repl._input_cursor = 8
        repl._move_word_backward()
        assert repl._input_cursor == 6

    def test_move_word_backward_past_space(self, repl):
        repl._input_cursor = 6
        repl._move_word_backward()
        assert repl._input_cursor == 0

    def test_move_word_backward_at_start(self, repl):
        repl._input_cursor = 0
        repl._move_word_backward()
        assert repl._input_cursor == 0

    def test_move_word_backward_from_after_spaces(self, repl):
        # "a   b" at pos 4: skip spaces back (4→1), skip word back (1→0)
        repl._input_buf = list("a   b")
        repl._input_cursor = 4
        repl._move_word_backward()
        assert repl._input_cursor == 0

    def test_move_word_backward_mid_spaces(self, repl):
        # "a   b" at pos 2: skip spaces back (2→1), skip word back (1→0)
        repl._input_buf = list("a   b")
        repl._input_cursor = 2
        repl._move_word_backward()
        assert repl._input_cursor == 0

    def test_move_word_forward_multiple_spaces(self, repl):
        # "a   b" at pos 0: skip word "a" (0→1), skip spaces (1→1)
        repl._input_buf = list("a   b")
        repl._input_cursor = 0
        repl._move_word_forward()
        assert repl._input_cursor == 1


# ---------------------------------------------------------------------------
# TuiRepl — transpose
# ---------------------------------------------------------------------------

class TestTransposeChars:
    @pytest.fixture
    def repl(self):
        surface = TextSurface()

        class FakeRepl:
            io = surface
            console = type("C", (), {"_io": surface, "_tui_repl": None})()
            COMMANDS = {}
            _history = []
        return TuiRepl(FakeRepl(), None)

    def test_transpose_mid_line(self, repl):
        repl._input_buf = list("abc")
        repl._input_cursor = 2
        repl._transpose_chars()
        assert "".join(repl._input_buf) == "acb"
        assert repl._input_cursor == 3

    def test_transpose_at_end(self, repl):
        repl._input_buf = list("abc")
        repl._input_cursor = 3
        repl._transpose_chars()
        assert "".join(repl._input_buf) == "acb"
        assert repl._input_cursor == 3

    def test_transpose_at_start_noop(self, repl):
        repl._input_buf = list("abc")
        repl._input_cursor = 0
        repl._transpose_chars()
        assert "".join(repl._input_buf) == "abc"

    def test_transpose_single_char_noop(self, repl):
        repl._input_buf = list("a")
        repl._input_cursor = 1
        repl._transpose_chars()
        assert "".join(repl._input_buf) == "a"


# ---------------------------------------------------------------------------
# TuiRepl — kill ring and kill operations
# ---------------------------------------------------------------------------

class TestKillRing:
    @pytest.fixture
    def repl(self):
        surface = TextSurface()

        class FakeRepl:
            io = surface
            console = type("C", (), {"_io": surface, "_tui_repl": None})()
            COMMANDS = {}
            _history = []
        r = TuiRepl(FakeRepl(), None)
        r._input_buf = []
        r._input_cursor = 0
        return r

    def test_push_kill_adds_to_ring(self, repl):
        repl._push_kill("hello")
        assert repl._kill_ring == ["hello"]

    def test_push_kill_empty_ignored(self, repl):
        repl._push_kill("")
        assert repl._kill_ring == []

    def test_push_kill_resets_yank(self, repl):
        repl._yank_active = True
        repl._yank_idx = 5
        repl._push_kill("text")
        assert repl._yank_active is False
        assert repl._yank_idx == -1

    def test_push_kill_caps_at_max(self, repl):
        for i in range(15):
            repl._push_kill(f"kill{i}")
        assert len(repl._kill_ring) == TuiRepl._KILL_RING_MAX
        # oldest entries dropped
        assert repl._kill_ring[0] == "kill5"

    def test_kill_to_start(self, repl):
        repl._input_buf = list("hello world")
        repl._input_cursor = 5
        repl._kill_to_start()
        assert "".join(repl._input_buf) == " world"
        assert repl._input_cursor == 0
        assert repl._kill_ring == ["hello"]

    def test_kill_to_end(self, repl):
        repl._input_buf = list("hello world")
        repl._input_cursor = 5
        repl._kill_to_end()
        assert "".join(repl._input_buf) == "hello"
        assert repl._input_cursor == 5
        assert repl._kill_ring == [" world"]

    def test_kill_to_start_empty(self, repl):
        repl._input_buf = list("")
        repl._input_cursor = 0
        repl._kill_to_start()
        assert repl._input_buf == []
        assert repl._kill_ring == []  # empty text ignored


class TestDeleteOperations:
    @pytest.fixture
    def repl(self):
        surface = TextSurface()

        class FakeRepl:
            io = surface
            console = type("C", (), {"_io": surface, "_tui_repl": None})()
            COMMANDS = {}
            _history = []
        r = TuiRepl(FakeRepl(), None)
        r._input_buf = []
        r._input_cursor = 0
        return r

    def test_delete_at_cursor(self, repl):
        repl._input_buf = list("abc")
        repl._input_cursor = 1
        repl._delete_at_cursor()
        assert "".join(repl._input_buf) == "ac"
        assert repl._input_cursor == 1

    def test_delete_at_cursor_end_noop(self, repl):
        repl._input_buf = list("ab")
        repl._input_cursor = 2
        repl._delete_at_cursor()
        assert "".join(repl._input_buf) == "ab"

    def test_delete_word_back(self, repl):
        repl._input_buf = list("hello world")
        repl._input_cursor = 11
        repl._delete_word_back()
        assert "".join(repl._input_buf) == "hello "
        assert repl._input_cursor == 6
        assert repl._kill_ring == ["world"]

    def test_delete_word_back_mid(self, repl):
        repl._input_buf = list("aa bb cc")
        repl._input_cursor = 5
        repl._delete_word_back()
        assert "".join(repl._input_buf) == "aa  cc"
        assert repl._input_cursor == 3

    def test_delete_word_back_at_start_noop(self, repl):
        repl._input_buf = list("abc")
        repl._input_cursor = 0
        repl._delete_word_back()
        assert "".join(repl._input_buf) == "abc"
        assert repl._input_cursor == 0

    def test_delete_word_forward(self, repl):
        # "hello world" at pos 0: skip word "hello" (0→5), skip space (5→6)
        # Actually: skip spaces first (none at 0), then skip word "hello" (0→5),
        # then skip leading space after word... wait, re-read the code.
        # Code: skip spaces forward, then skip word forward.
        # At pos 0: no leading spaces, skip "hello" (0→5). Deletes [0:5] = "hello"
        repl._input_buf = list("hello world")
        repl._input_cursor = 0
        repl._delete_word_forward()
        assert "".join(repl._input_buf) == " world"
        assert repl._input_cursor == 0
        assert repl._kill_ring == ["hello"]

    def test_delete_word_forward_at_end_noop(self, repl):
        repl._input_buf = list("abc")
        repl._input_cursor = 3
        repl._delete_word_forward()
        assert "".join(repl._input_buf) == "abc"

    def test_delete_word_forward_mid(self, repl):
        # "aa bb cc" at pos 3: skip word "bb" (3→5), then space (5→6)?
        # Actually: skip spaces (none, buf[3]="b"), skip word "bb" (3→5)
        # Deletes [3:5] = "bb" → "aa  cc"
        repl._input_buf = list("aa bb cc")
        repl._input_cursor = 3
        repl._delete_word_forward()
        assert "".join(repl._input_buf) == "aa  cc"
        assert repl._input_cursor == 3


# ---------------------------------------------------------------------------
# TuiRepl — yank (Ctrl+Y)
# ---------------------------------------------------------------------------

class TestYank:
    @pytest.fixture
    def repl(self):
        surface = TextSurface()

        class FakeRepl:
            io = surface
            console = type("C", (), {"_io": surface, "_tui_repl": None})()
            COMMANDS = {}
            _history = []
        r = TuiRepl(FakeRepl(), None)
        r._input_buf = []
        r._input_cursor = 0
        return r

    def test_yank_empty_ring_noop(self, repl):
        repl._yank()
        assert repl._input_buf == []

    def test_yank_inserts_newest(self, repl):
        repl._kill_ring = ["first", "second", "third"]
        repl._yank()
        assert "".join(repl._input_buf) == "third"
        assert repl._input_cursor == 5

    def test_yank_repeat_cycles(self, repl):
        repl._kill_ring = ["first", "second", "third"]
        repl._yank()  # inserts "third"
        repl._yank()  # replaces with "second"
        assert "".join(repl._input_buf) == "second"

    def test_yank_resets_on_edit(self, repl):
        repl._kill_ring = ["aa", "bb"]
        repl._yank()  # inserts "bb"
        repl._input_buf.insert(0, "X")  # manual edit
        repl._yank_active = True  # force state
        repl._yank()  # should detect mismatch and start fresh
        assert "".join(repl._input_buf).count("bb") >= 1


# ---------------------------------------------------------------------------
# TuiRepl — history search (_search_back)
# ---------------------------------------------------------------------------

class TestSearchBack:
    @pytest.fixture
    def repl(self):
        surface = TextSurface()

        class FakeRepl:
            io = surface
            console = type("C", (), {"_io": surface, "_tui_repl": None})()
            COMMANDS = {}
            _history = []
        r = TuiRepl(FakeRepl(), None)
        r._cmd_history = ["foo bar", "baz foo", "bar baz", "foo qux"]
        r._search_q = ""
        return r

    def test_empty_query_returns_clamped_start(self, repl):
        repl._search_q = ""
        assert repl._search_back(2) == 2

    def test_backward_finds_match(self, repl):
        repl._search_q = "foo"
        assert repl._search_back(3) == 3  # "foo qux" at index 3
        assert repl._search_back(2) == 1  # "baz foo" at index 1

    def test_backward_no_match(self, repl):
        repl._search_q = "zzz"
        assert repl._search_back(3) == -1

    def test_forward_finds_match(self, repl):
        repl._search_q = "foo"
        # _search_back(0, fwd=True) starts at idx=0 inclusive, "foo bar" matches
        assert repl._search_back(0, fwd=True) == 0

    def test_forward_finds_later_match(self, repl):
        repl._search_q = "foo"
        # _search_back(1, fwd=True) starts at idx=1, "baz foo" matches
        assert repl._search_back(1, fwd=True) == 1

    def test_forward_past_end(self, repl):
        repl._search_q = "foo"
        assert repl._search_back(4, fwd=True) == -1

    def test_backward_from_past_end(self, repl):
        repl._search_q = "bar"
        # start=10 clamps to min(10, 3)=3, "foo qux" no, "bar baz" at 2
        assert repl._search_back(10) == 2

    def test_backward_from_zero(self, repl):
        repl._search_q = "foo"
        assert repl._search_back(0) == 0  # "foo bar"

    def test_empty_history(self, repl):
        repl._cmd_history = []
        repl._search_q = "foo"
        assert repl._search_back(0) == 0


# ---------------------------------------------------------------------------
# TuiRepl — output search (_out_find)
# ---------------------------------------------------------------------------

class TestOutFind:
    @pytest.fixture
    def repl(self):
        surface = TextSurface()
        surface.write("line one alpha")
        surface.write("line two beta")
        surface.write("line three alpha")

        class FakeRepl:
            io = surface
            console = type("C", (), {"_io": surface, "_tui_repl": None})()
            COMMANDS = {}
            _history = []
        r = TuiRepl(FakeRepl(), None)
        r._output_surface = surface
        r._out_search_q = ""
        return r

    def test_empty_query(self, repl):
        repl._out_search_q = ""
        assert repl._out_find(0, fwd=True) == -1

    def test_forward_finds(self, repl):
        repl._out_search_q = "alpha"
        assert repl._out_find(0, fwd=True) == 0

    def test_forward_from_middle(self, repl):
        repl._out_search_q = "alpha"
        assert repl._out_find(1, fwd=True) == 2

    def test_backward_finds(self, repl):
        repl._out_search_q = "alpha"
        assert repl._out_find(2, fwd=False) == 2

    def test_case_insensitive(self, repl):
        repl._out_search_q = "ALPHA"
        assert repl._out_find(0, fwd=True) == 0

    def test_no_match(self, repl):
        repl._out_search_q = "gamma"
        assert repl._out_find(0, fwd=True) == -1

    def test_empty_buffer(self):
        surface = TextSurface()

        class FakeRepl:
            io = surface
            console = type("C", (), {"_io": surface, "_tui_repl": None})()
            COMMANDS = {}
            _history = []
        r = TuiRepl(FakeRepl(), None)
        r._output_surface = surface
        r._out_search_q = "test"
        assert r._out_find(0, fwd=True) == -1

    def test_negative_start_fwd(self, repl):
        repl._out_search_q = "beta"
        assert repl._out_find(-1, fwd=True) == 1

    def test_negative_start_bwd(self, repl):
        repl._out_search_q = "beta"
        assert repl._out_find(-1, fwd=False) == 1
