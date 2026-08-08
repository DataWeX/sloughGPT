"""
Tests for the curses-free logic of the split-pane TUI shell: path
completion, token completion, and reverse history search.

``TuiRepl.__init__`` builds only surfaces and the layout (no curses), so the
completion/search helpers can be exercised without a terminal.
"""

import curses
import ctypes
import os
import threading
import time
import types
from unittest.mock import patch

import pytest

import domains.shell.tui_repl as tui_mod
from domains.shell.tui_repl import TuiIo, TuiRepl, _complete_path, _read_escape_remainder


class _FakeRepl:
    COMMANDS = ["ai", "alias", "about", "bg", "cd", "echo", "exit"]


@pytest.fixture()
def repl():
    return TuiRepl(_FakeRepl(), None)


def set_buf(repl, text, caret=None):
    repl._input_buf = list(text)
    repl._input_cursor = len(text) if caret is None else caret


# ── _complete_path ────────────────────────────────────────────────────────

def test_complete_path_single_match(tmp_path):
    (tmp_path / "alpha.txt").write_text("")
    (tmp_path / "beta.txt").write_text("")
    matches = _complete_path(str(tmp_path / "a"))
    assert matches == [str(tmp_path / "alpha.txt")]


def test_complete_path_multiple_matches(tmp_path):
    (tmp_path / "alpha.txt").write_text("")
    (tmp_path / "alpine.txt").write_text("")
    matches = _complete_path(str(tmp_path / "al"))
    assert len(matches) == 2


def test_complete_path_no_match(tmp_path):
    assert _complete_path(str(tmp_path / "zzz")) == []


def test_complete_path_bare_name_no_slash(tmp_path, monkeypatch):
    (tmp_path / "alpha.txt").write_text("")
    (tmp_path / "beta.txt").write_text("")
    monkeypatch.chdir(tmp_path)
    assert _complete_path("al") == ["./alpha.txt"]


def test_complete_path_expands_tilde():
    home = os.path.expanduser("~")
    matches = _complete_path("~")
    assert any(os.path.realpath(m) == home for m in matches)


# ── token completion ──────────────────────────────────────────────────────

def test_complete_single_command(repl):
    set_buf(repl, "ai")
    repl._complete()
    assert "".join(repl._input_buf) == "ai "


def test_complete_command_common_prefix(repl):
    set_buf(repl, "a")
    repl._complete()
    assert "".join(repl._input_buf) == "a"
    lines = [r.text for r in repl._output_surface.render(20)]
    assert any("alias" in ln and "about" in ln for ln in lines)


def test_complete_path_after_argument(repl):
    set_buf(repl, "echo src/")
    repl._complete()
    assert "".join(repl._input_buf) == "echo src/"


def test_complete_dir_appends_slash(repl, tmp_path):
    sub = tmp_path / "mydir"
    sub.mkdir()
    set_buf(repl, "echo " + str(tmp_path / "my"))
    repl._complete()
    assert "".join(repl._input_buf) == "echo " + str(tmp_path / "mydir") + "/"


def test_complete_command_extends_common_prefix(repl, monkeypatch):
    monkeypatch.setattr(repl._repl, "COMMANDS", ["echo", "echidna", "exit"])
    set_buf(repl, "ec")
    repl._complete()
    assert "".join(repl._input_buf) == "ech"


def test_complete_path_token_common_prefix(repl, tmp_path):
    (tmp_path / "alpine.txt").write_text("")
    (tmp_path / "alps.txt").write_text("")
    set_buf(repl, "")
    repl._complete_path_token(0, 0, str(tmp_path / "al"))
    assert "".join(repl._input_buf) == str(tmp_path / "alp")


def test_complete_path_token_multiple_distinct(repl, tmp_path):
    (tmp_path / "alpha.txt").write_text("")
    (tmp_path / "beta.txt").write_text("")
    set_buf(repl, "")
    repl._complete_path_token(0, 0, str(tmp_path) + "/")
    cap = "".join(repl._output_surface.capture)
    assert "alpha.txt" in cap and "beta.txt" in cap


# ── reverse history search ────────────────────────────────────────────────

def test_search_back_finds_match(repl):
    repl._cmd_history = ["models", "ai hello", "gen"]
    repl._search_q = "ai"
    assert repl._search_back(2) == 1


def test_search_back_no_match_returns_minus_one(repl):
    repl._cmd_history = ["models", "gen"]
    repl._search_q = "zzz"
    assert repl._search_back(1) == -1


def test_search_back_empty_query(repl):
    repl._cmd_history = ["models"]
    repl._search_q = ""
    assert repl._search_back(0) == 0


def test_apply_search_loads_match(repl):
    repl._cmd_history = ["models", "ai hello", "gen"]
    repl._search_q = "ai"
    repl._search_idx = 2
    repl._search_save = (list("abc"), 1)
    repl._apply_search()
    assert "".join(repl._input_buf) == "ai hello"
    assert repl._input_cursor == len("ai hello")


def test_end_search_restores_saved(repl):
    repl._cmd_history = ["models"]
    repl._search_q = "zzz"
    repl._search_save = (list("abc"), 1)
    repl._input_buf = ["a", "i"]
    repl._input_cursor = 2
    repl._end_search(restore=True)
    assert "".join(repl._input_buf) == "abc"
    assert repl._input_cursor == 1
    assert not repl._searching


def test_end_search_accepts_keeps_match(repl):
    repl._cmd_history = ["ai hello"]
    repl._search_q = "ai"
    repl._search_idx = 0
    repl._search_save = (list("abc"), 1)
    repl._apply_search()
    repl._end_search(restore=False)
    assert "".join(repl._input_buf) == "ai hello"
    assert not repl._searching


def test_search_back_forward_finds_oldest(repl):
    repl._cmd_history = ["gen", "ai one", "echo", "ai two"]
    repl._search_q = "ai"
    assert repl._search_back(0, fwd=True) == 1


def test_search_back_forward_past_end_returns_minus_one(repl):
    repl._cmd_history = ["ai one", "ai two"]
    repl._search_q = "ai"
    assert repl._search_back(2, fwd=True) == -1


def test_search_back_backward_negative_start_returns_minus_one(repl):
    repl._cmd_history = ["ai one", "ai two"]
    repl._search_q = "ai"
    assert repl._search_back(-1, fwd=False) == -1


def test_search_back_empty_history_clamps(repl):
    repl._cmd_history = []
    repl._search_q = "ai"
    assert repl._search_back(5, fwd=False) == 0
    assert repl._search_back(0, fwd=True) == 0


def test_kill_ring_trims_to_max(repl):
    set_buf(repl, "")
    for _ in range(12):
        set_buf(repl, "word")
        repl._kill_to_start()
    assert len(repl._kill_ring) == repl._KILL_RING_MAX
    assert repl._kill_ring[-1] == "word"


def test_tui_io_flush_and_read():
    from domains.shell.surface import TextSurface

    io = TuiIo(TextSurface())
    io.flush()
    with pytest.raises(NotImplementedError):
        io.read("prompt")


def test_apply_search_forward_sentinel_finds_oldest(repl):
    repl._cmd_history = ["gen", "ai one", "echo", "ai two"]
    repl._search_q = "ai"
    repl._search_fwd = True
    repl._search_idx = -1  # fresh forward query — scan from the oldest entry
    repl._apply_search()
    assert "".join(repl._input_buf) == "ai one"
    assert repl._search_idx == 1


def test_apply_search_forward_advances(repl):
    repl._cmd_history = ["gen", "ai one", "echo", "ai two"]
    repl._search_q = "ai"
    repl._search_fwd = True
    repl._search_idx = 1
    repl._apply_search()
    assert "".join(repl._input_buf) == "ai one"
    repl._search_idx += 1  # navigation restarts just past the current match
    repl._apply_search()
    assert "".join(repl._input_buf) == "ai two"
    assert repl._search_idx == 3


def test_apply_search_forward_no_match_keeps_buffer(repl):
    repl._cmd_history = ["gen", "ai one"]
    repl._search_q = "zzz"
    repl._search_fwd = True
    repl._search_idx = -1
    repl._input_buf = list("untouched")
    repl._apply_search()
    assert "".join(repl._input_buf) == "untouched"


def test_apply_search_switch_direction_from_reverse(repl):
    repl._cmd_history = ["ai one", "echo", "ai two"]
    repl._search_q = "ai"
    repl._search_fwd = False
    repl._search_idx = 2
    repl._apply_search()  # reverse: newest match is index 2
    assert repl._search_idx == 2
    repl._search_fwd = True  # switch to forward at the current match
    repl._search_idx += 1
    repl._apply_search()
    assert repl._search_idx == 3  # past the end — buffer unchanged
    assert "".join(repl._input_buf) == "ai two"


# ── line editing helpers ─────────────────────────────────────────────────

def test_move_home_and_end(repl):
    set_buf(repl, "hello", caret=3)
    repl._move_home()
    assert repl._input_cursor == 0
    repl._move_end()
    assert repl._input_cursor == 5


def test_kill_to_start(repl):
    set_buf(repl, "hello world", caret=5)
    repl._kill_to_start()
    assert "".join(repl._input_buf) == " world"
    assert repl._input_cursor == 0


def test_kill_to_start_at_home_keeps_line(repl):
    set_buf(repl, "hello", caret=0)
    repl._kill_to_start()
    assert "".join(repl._input_buf) == "hello"


def test_kill_to_end(repl):
    set_buf(repl, "hello world", caret=5)
    repl._kill_to_end()
    assert "".join(repl._input_buf) == "hello"
    assert repl._input_cursor == 5


def test_kill_to_end_at_end_keeps_line(repl):
    set_buf(repl, "hello", caret=5)
    repl._kill_to_end()
    assert "".join(repl._input_buf) == "hello"


def test_delete_at_cursor(repl):
    set_buf(repl, "hello", caret=1)
    repl._delete_at_cursor()
    assert "".join(repl._input_buf) == "hllo"
    assert repl._input_cursor == 1


def test_delete_at_cursor_noop_at_end(repl):
    set_buf(repl, "hello", caret=5)
    repl._delete_at_cursor()
    assert "".join(repl._input_buf) == "hello"
    assert repl._input_cursor == 5


def test_delete_word_back_end_of_line(repl):
    set_buf(repl, "echo foo bar", caret=len("echo foo bar"))
    repl._delete_word_back()
    assert "".join(repl._input_buf) == "echo foo "
    assert repl._input_cursor == len("echo foo ")


def test_delete_word_back_mid_line(repl):
    set_buf(repl, "one two three", caret=7)
    repl._delete_word_back()
    assert "".join(repl._input_buf) == "one  three"
    assert repl._input_cursor == 4


def test_delete_word_back_trailing_spaces(repl):
    set_buf(repl, "a b  ", caret=5)
    repl._delete_word_back()
    assert "".join(repl._input_buf) == "a "
    assert repl._input_cursor == 2


def test_delete_word_back_at_start_noop(repl):
    set_buf(repl, "hello", caret=0)
    repl._delete_word_back()
    assert "".join(repl._input_buf) == "hello"
    assert repl._input_cursor == 0


# ── delete word forward (Alt+D) ───────────────────────────────────────────

def test_delete_word_forward_at_start(repl):
    set_buf(repl, "echo foo bar", caret=0)
    repl._delete_word_forward()
    assert "".join(repl._input_buf) == " foo bar"
    assert repl._input_cursor == 0


def test_delete_word_forward_mid_word(repl):
    set_buf(repl, "one two three", caret=4)
    repl._delete_word_forward()
    assert "".join(repl._input_buf) == "one  three"
    assert repl._input_cursor == 4


def test_delete_word_forward_at_whitespace(repl):
    set_buf(repl, "a b", caret=1)
    repl._delete_word_forward()
    assert "".join(repl._input_buf) == "a"
    assert repl._input_cursor == 1


def test_delete_word_forward_at_end_noop(repl):
    set_buf(repl, "hello", caret=5)
    repl._delete_word_forward()
    assert "".join(repl._input_buf) == "hello"
    assert repl._input_cursor == 5


def test_delete_word_forward_pushes_ring(repl):
    set_buf(repl, "rm cache", caret=3)
    repl._delete_word_forward()
    assert repl._kill_ring == ["cache"]
    set_buf(repl, "another", caret=0)
    repl._delete_word_forward()
    assert repl._kill_ring == ["cache", "another"]


# ── kill ring + yank (Ctrl+Y) ─────────────────────────────────────────────

def test_kill_to_end_pushes_ring(repl):
    set_buf(repl, "abc def", caret=4)
    repl._kill_to_end()
    assert "".join(repl._input_buf) == "abc "
    assert repl._kill_ring == ["def"]


def test_kill_to_start_pushes_ring(repl):
    set_buf(repl, "abc def", caret=4)
    repl._kill_to_start()
    assert "".join(repl._input_buf) == "def"
    assert repl._kill_ring == ["abc "]


def test_delete_word_back_pushes_ring(repl):
    set_buf(repl, "one two three", caret=7)
    repl._delete_word_back()
    assert repl._kill_ring == ["two"]


def test_delete_at_cursor_not_in_ring(repl):
    set_buf(repl, "abcdef", caret=3)
    repl._delete_at_cursor()
    assert repl._kill_ring == []


def test_yank_inserts_latest_kill(repl):
    set_buf(repl, "echo ", caret=5)
    repl._kill_ring = ["hello"]
    repl._yank()
    assert "".join(repl._input_buf) == "echo hello"
    assert repl._input_cursor == len("echo hello")


def test_yank_repeat_cycles_ring(repl):
    set_buf(repl, "", caret=0)
    repl._kill_ring = ["one", "two", "three"]
    repl._yank()
    assert "".join(repl._input_buf) == "three"
    repl._yank()
    assert "".join(repl._input_buf) == "two"
    repl._yank()
    assert "".join(repl._input_buf) == "one"
    repl._yank()
    assert "".join(repl._input_buf) == "one"  # oldest entry repeats


def test_yank_empty_ring_noop(repl):
    set_buf(repl, "abc", caret=3)
    repl._kill_ring = []
    repl._yank()
    assert "".join(repl._input_buf) == "abc"


def test_yank_after_edit_starts_fresh(repl):
    set_buf(repl, "keep", caret=4)
    repl._kill_ring = ["x", "yy"]
    repl._yank()
    assert "".join(repl._input_buf) == "keepyy"
    repl._input_buf.insert(0, "Z")  # edit the line between yanks
    repl._yank()
    assert "".join(repl._input_buf) == "Zkeepyyyy"  # fresh yank of newest entry
    assert repl._input_cursor == 8  # caret right after the fresh yank


def test_kill_ring_capped(repl):
    set_buf(repl, "", caret=0)
    for i in range(12):
        repl._push_kill(f"k{i}")
    assert len(repl._kill_ring) == 10
    assert repl._kill_ring[0] == "k2"
    assert repl._kill_ring[-1] == "k11"


def test_push_kill_resets_yank_cycle(repl):
    set_buf(repl, "", caret=0)
    repl._kill_ring = ["one", "two"]
    repl._yank()
    assert repl._yank_active
    repl._kill_to_start()
    assert not repl._yank_active


# ── word movement (Alt+F / Alt+B / Ctrl+arrows) ───────────────────────────

def test_move_word_forward_next_word(repl):
    set_buf(repl, "one two three", caret=0)
    repl._move_word_forward()
    assert repl._input_cursor == 3
    repl._move_word_forward()
    assert repl._input_cursor == 7
    repl._move_word_forward()
    assert repl._input_cursor == len("one two three")


def test_move_word_forward_skips_whitespace(repl):
    set_buf(repl, "one   two", caret=3)
    repl._move_word_forward()
    assert repl._input_cursor == 9  # past the end of "two"


def test_move_word_forward_from_mid_word(repl):
    set_buf(repl, "one two", caret=2)
    repl._move_word_forward()
    assert repl._input_cursor == 3  # end of the current word


def test_move_word_forward_at_end_noop(repl):
    set_buf(repl, "one two", caret=7)
    repl._move_word_forward()
    assert repl._input_cursor == 7


def test_move_word_backward_previous_word(repl):
    set_buf(repl, "one two three", caret=len("one two three"))
    repl._move_word_backward()
    assert repl._input_cursor == 8  # start of "three"
    repl._move_word_backward()
    assert repl._input_cursor == 4  # start of "two"
    repl._move_word_backward()
    assert repl._input_cursor == 0  # start of "one"


def test_move_word_backward_from_mid_word(repl):
    set_buf(repl, "one two", caret=6)
    repl._move_word_backward()
    assert repl._input_cursor == 4


def test_move_word_backward_skips_whitespace(repl):
    set_buf(repl, "one   two", caret=7)
    repl._move_word_backward()
    assert repl._input_cursor == 6  # start of "two"


def test_move_word_backward_at_start_noop(repl):
    set_buf(repl, "one two", caret=0)
    repl._move_word_backward()
    assert repl._input_cursor == 0


class _FakeKeyWin:
    def __init__(self, keys):
        self._keys = list(keys)
        self.timeouts = []

    def getch(self):
        return self._keys.pop(0) if self._keys else -1

    def timeout(self, ms):
        self.timeouts.append(ms)


def test_read_escape_remainder_lone_esc():
    win = _FakeKeyWin([-1])
    assert _read_escape_remainder(win, {}) is None


def test_read_escape_remainder_double_esc():
    win = _FakeKeyWin([27])
    assert _read_escape_remainder(win, {}) is None


def test_read_escape_remainder_alt_f():
    win = _FakeKeyWin([ord("f")])
    assert _read_escape_remainder(win, {"f": "fwd"}) == "alt:fwd"


def test_read_escape_remainder_alt_unknown_noop():
    win = _FakeKeyWin([ord("x")])
    assert _read_escape_remainder(win, {"f": "fwd", "b": "bwd"}) is None


def test_read_escape_remainder_ctrl_left():
    win = _FakeKeyWin([ord("["), ord("1"), ord(";"), ord("5"), ord("D")])
    assert _read_escape_remainder(win, {}) == "seq:ctrl-left"


def test_read_escape_remainder_ctrl_right():
    win = _FakeKeyWin([ord("["), ord("1"), ord(";"), ord("5"), ord("C")])
    assert _read_escape_remainder(win, {}) == "seq:ctrl-right"


def test_read_escape_remainder_restores_timeout():
    win = _FakeKeyWin([-1])
    _read_escape_remainder(win, {}, restore_ms=200)
    assert win.timeouts == [0, 200]


# ── transpose chars (Ctrl+T) ──────────────────────────────────────────────

def test_transpose_chars_mid_word(repl):
    set_buf(repl, "abcd", caret=2)
    repl._transpose_chars()
    assert "".join(repl._input_buf) == "acbd"
    assert repl._input_cursor == 3


def test_transpose_chars_at_end(repl):
    set_buf(repl, "abcd", caret=4)
    repl._transpose_chars()
    assert "".join(repl._input_buf) == "abdc"
    assert repl._input_cursor == 4


def test_transpose_chars_at_start_noop(repl):
    set_buf(repl, "abcd", caret=0)
    repl._transpose_chars()
    assert "".join(repl._input_buf) == "abcd"
    assert repl._input_cursor == 0


def test_transpose_chars_single_char_noop(repl):
    set_buf(repl, "a", caret=1)
    repl._transpose_chars()
    assert "".join(repl._input_buf) == "a"


def test_transpose_chars_empty_noop(repl):
    set_buf(repl, "", caret=0)
    repl._transpose_chars()
    assert "".join(repl._input_buf) == ""


def test_transpose_chars_advances_caret(repl):
    set_buf(repl, "abc", caret=1)
    repl._transpose_chars()
    assert "".join(repl._input_buf) == "bac"
    assert repl._input_cursor == 2


# ── search failed indicator ───────────────────────────────────────────────

def test_apply_search_sets_failed_flag(repl):
    repl._cmd_history = ["models", "ai hello"]
    repl._search_q = "zzz"
    repl._search_idx = 2
    repl._apply_search()
    assert repl._search_failed


def test_apply_search_clears_failed_flag(repl):
    repl._cmd_history = ["models", "ai hello"]
    repl._search_q = "zzz"
    repl._search_idx = 2
    repl._apply_search()
    assert repl._search_failed
    repl._search_q = "ai"
    repl._apply_search()
    assert not repl._search_failed


def test_apply_search_empty_query_not_failed(repl):
    repl._cmd_history = ["models"]
    repl._search_q = ""
    repl._search_save = (list("ab"), 1)
    repl._apply_search()
    assert not repl._search_failed


# ── input view (caret column + horizontal scroll) ────────────────────────

def test_input_view_fits_window(repl):
    line, caret = repl._input_view(20, "hello", 3)
    assert line == "hello"
    assert caret == 2 + 3


def test_input_view_caret_at_end(repl):
    line, caret = repl._input_view(20, "hello", 5)
    assert line == "hello"
    assert caret == 7


def test_input_view_overflows_scrolls_to_caret(repl):
    line, caret = repl._input_view(10, "a" * 20, 20)
    assert line == "a" * 7  # max_w = 10 - 2 - 1 = 7
    assert caret == 2 + 7


def test_input_view_mid_caret_shows_window_around_caret(repl):
    line, caret = repl._input_view(10, "0123456789", 6)
    assert line == "3456789"
    assert caret == 2 + (6 - 3)


def test_input_view_empty(repl):
    line, caret = repl._input_view(10, "", 0)
    assert line == ""
    assert caret == 2


def test_input_view_caret_clamped(repl):
    line, caret = repl._input_view(20, "hi", 99)
    assert line == "hi"
    assert caret == 4


# ── interrupt active command (Ctrl+C) ────────────────────────────────────

def test_interrupt_active_kills_busy_thread(repl):
    import threading
    import time

    interrupted = threading.Event()

    def busy():
        try:
            while True:
                pass
        except KeyboardInterrupt:
            interrupted.set()

    t = threading.Thread(target=busy, daemon=True)
    t.start()
    repl._active_thread = t
    repl._interrupt_active()
    t.join(timeout=5)
    assert not t.is_alive()
    assert interrupted.is_set()
    # the command is gone, so the poll tick clears it without a second call
    time.sleep(0.05)


def test_interrupt_active_no_thread_noop(repl):
    repl._active_thread = None
    repl._interrupt_active()  # must not raise


def test_interrupt_active_dead_thread_noop(repl):
    import threading

    done = threading.Event()

    def fn():
        done.set()

    t = threading.Thread(target=fn, daemon=True)
    t.start()
    done.wait(timeout=5)
    repl._active_thread = t
    repl._interrupt_active()  # thread already finished — must not raise
    t.join(timeout=5)


def test_interrupt_active_main_thread_noop(repl):
    import threading

    repl._active_thread = threading.current_thread()
    repl._interrupt_active()  # must not raise or interrupt the caller


# ── output-pane content search ───────────────────────────────────────────

def seed_output(repl, lines):
    repl._output_surface.clear()
    for ln in lines:
        repl._output_surface.write(ln)


def test_out_find_forward_finds_first(repl):
    seed_output(repl, ["λ echo one", "result: alpha", "result: beta"])
    repl._out_search_q = "result"
    assert repl._out_find(-1, fwd=True) == 1


def test_out_find_forward_wraps_to_head(repl):
    seed_output(repl, ["result: head", "no match", "result: tail"])
    repl._out_search_q = "result"
    assert repl._out_find(3, fwd=True) == 0


def test_out_find_backward_finds_last_before_start(repl):
    seed_output(repl, ["result: one", "x", "result: two", "result: three"])
    repl._out_search_q = "result"
    assert repl._out_find(2, fwd=False) == 2
    assert repl._out_find(1, fwd=False) == 0


def test_out_find_backward_wraps_to_tail(repl):
    seed_output(repl, ["x", "no match", "result: tail"])
    repl._out_search_q = "result"
    assert repl._out_find(0, fwd=False) == 2


def test_out_find_no_match_returns_minus_one(repl):
    seed_output(repl, ["alpha", "beta"])
    repl._out_search_q = "zzz"
    assert repl._out_find(-1, fwd=True) == -1
    assert repl._out_find(1, fwd=False) == -1


def test_out_find_empty_query_returns_minus_one(repl):
    seed_output(repl, ["alpha"])
    repl._out_search_q = ""
    assert repl._out_find(-1, fwd=True) == -1


def test_out_find_case_insensitive(repl):
    seed_output(repl, ["Error: boom", "fine"])
    repl._out_search_q = "error"
    assert repl._out_find(-1, fwd=True) == 0


def test_apply_out_search_scrolls_to_match_top(repl):
    seed_output(repl, [f"line {i}" for i in range(30)])
    repl._out_search_q = "line 25"
    repl._apply_out_search(rows=10)
    assert repl._out_search_sel == 25
    assert repl._out_scroll == max(30 - 10 - 25, 0)


def test_apply_out_search_no_match_sets_failed(repl):
    seed_output(repl, ["alpha", "beta"])
    repl._out_search_q = "zzz"
    repl._apply_out_search(rows=10)
    assert repl._out_search_failed
    assert repl._out_search_sel == -1


def test_apply_out_search_clears_failed_on_match(repl):
    seed_output(repl, ["alpha", "result: beta"])
    repl._out_search_q = "zzz"
    repl._apply_out_search(rows=10)
    assert repl._out_search_failed
    repl._out_search_q = "result"
    repl._apply_out_search(rows=10)
    assert not repl._out_search_failed
    assert repl._out_search_sel == 1


def test_apply_out_search_next_from_current_match(repl):
    seed_output(repl, ["result: one", "x", "result: two", "result: three"])
    repl._out_search_q = "result"
    repl._apply_out_search(rows=10, start=-1, fwd=True)
    assert repl._out_search_sel == 0
    repl._apply_out_search(rows=10, start=repl._out_search_sel + 1, fwd=True)
    assert repl._out_search_sel == 2
    repl._apply_out_search(rows=10, start=repl._out_search_sel + 1, fwd=True)
    assert repl._out_search_sel == 3


def test_apply_out_search_prev_wraps(repl):
    seed_output(repl, ["result: one", "x", "result: two"])
    repl._out_search_q = "result"
    repl._out_search_sel = 0
    repl._apply_out_search(rows=10, start=repl._out_search_sel - 1, fwd=False)
    assert repl._out_search_sel == 2


def test_apply_out_search_empty_query_resets_selection(repl):
    seed_output(repl, ["result: one"])
    repl._out_search_q = "result"
    repl._out_search_sel = 0
    repl._apply_out_search(rows=10)
    repl._out_search_q = ""
    repl._apply_out_search(rows=10)
    assert repl._out_search_sel == -1
    assert not repl._out_search_failed


def test_repeat_out_search_moves_next(repl):
    seed_output(repl, ["result: one", "x", "result: two", "result: three"])
    repl._out_search_last = "result"
    repl._out_search_sel = 0
    repl._repeat_out_search(rows=10, fwd=True)
    assert repl._out_search_sel == 2
    assert repl._out_search_q == ""


def test_repeat_out_search_moves_prev(repl):
    seed_output(repl, ["result: one", "x", "result: two"])
    repl._out_search_last = "result"
    repl._out_search_sel = 0
    repl._repeat_out_search(rows=10, fwd=False)
    assert repl._out_search_sel == 2


def test_repeat_out_search_wraps(repl):
    seed_output(repl, ["result: one", "x", "result: two"])
    repl._out_search_last = "result"
    repl._out_search_sel = 2
    repl._repeat_out_search(rows=10, fwd=True)
    assert repl._out_search_sel == 0


def test_repeat_out_search_no_last_noop(repl):
    seed_output(repl, ["result: one"])
    repl._out_search_last = ""
    repl._out_search_sel = -1
    repl._repeat_out_search(rows=10, fwd=True)
    assert repl._out_search_sel == -1
    assert repl._out_search_q == ""


# ── curses event loop (_main) ────────────────────────────────────────────

class _MainFakeRepl:
    COMMANDS = ["ai", "echo", "exit"]

    def __init__(self):
        self.io = "old-io"
        self.console = types.SimpleNamespace(_io="old-console")
        self._history = ["echo hello", "ai test"]
        self.dispatched = []

    def _dispatch(self, cmd):
        self.dispatched.append(cmd)


class _FakeLogBuffer:
    def __init__(self, entries=None):
        if entries is None:
            entries = [types.SimpleNamespace(timestamp=1700000000.0, level="INFO", source="test", message="boot")]
        self._entries = entries

    def get(self):
        return self._entries


class _FakeWin:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def getmaxyx(self):
        return (self.rows, self.cols)

    def erase(self):
        pass

    def addstr(self, y, x, text, attr=0):
        pass

    def move(self, y, x):
        pass

    def refresh(self):
        pass


class _ScriptStdscr(_FakeWin):
    def __init__(self, keys, rows=24, cols=80):
        super().__init__(rows, cols)
        self._keys = list(keys)

    def keypad(self, flag):
        pass

    def timeout(self, ms):
        pass

    def getch(self):
        return self._keys.pop(0) if self._keys else -1


class _ErrorWin:
    def __init__(self, fail):
        self.fail = fail

    def getmaxyx(self):
        return (2, 20)

    def erase(self):
        if self.fail == "erase":
            raise curses.error("erase")

    def addstr(self, y, x, text, attr=0):
        if self.fail == "addstr":
            raise curses.error("addstr")

    def refresh(self):
        if self.fail == "refresh":
            raise curses.error("refresh")


class _MoveErrorWin:
    def getmaxyx(self):
        return (1, 20)

    def erase(self):
        pass

    def addstr(self, y, x, text, attr=0):
        pass

    def move(self, y, x):
        raise curses.error("move")

    def refresh(self):
        pass


def _drive(tui, scr, term=None, escdelay_raise=False, term_error=False, resize_error=False):
    if term is None:
        term = types.SimpleNamespace(lines=scr.rows, columns=scr.cols)

    def _newwin(r, c, top, left):
        return _FakeWin(r, c)

    esc = (
        patch.object(curses, "set_escdelay", side_effect=RuntimeError("boom"))
        if escdelay_raise
        else patch.object(curses, "set_escdelay")
    )
    getsize = (
        patch("os.get_terminal_size", side_effect=OSError("no tty"))
        if term_error
        else patch("os.get_terminal_size", return_value=term)
    )
    resize = (
        patch.object(curses, "resizeterm", side_effect=curses.error("resize"))
        if resize_error
        else patch.object(curses, "resizeterm")
    )
    with patch.object(curses, "curs_set"), patch.object(curses, "raw"), esc, \
         patch.object(curses, "color_pair", side_effect=lambda n: n), \
         patch.object(curses, "newwin", side_effect=_newwin), \
         resize, getsize, \
         patch.object(tui_mod, "_init_pairs"):
        tui._main(scr)


def _run_main(repl, keys, rows=24, cols=80, term=None, history=None, output=None,
              term_error=False, resize_error=False):
    tui = TuiRepl(repl, _FakeLogBuffer())
    if history is not None:
        tui._cmd_history = list(history)
    if output is not None:
        tui._output_surface.write("\n".join(output))
    tui._running = True
    scr = _ScriptStdscr(keys, rows=rows, cols=cols)
    _drive(tui, scr, term=term, term_error=term_error, resize_error=resize_error)
    return tui


def _wait_until(pred, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def _init_render_state(tui):
    tui._input_buf = []
    tui._input_cursor = 0
    tui._rows = 24
    tui._out_scroll = 0
    tui._log_scroll = 0
    tui._scroll_target = 0
    tui._active_cmd = None
    tui._searching = False
    tui._search_fwd = False
    tui._search_q = ""
    tui._search_failed = False
    tui._out_searching = False
    tui._out_search_q = ""
    tui._out_search_failed = False
    return tui


def test_main_types_command_dispatches_and_prompts():
    repl = _MainFakeRepl()
    tui = _run_main(repl, [ord("h"), ord("i"), ord("\n"), ord("q"), ord("\n")])
    assert _wait_until(lambda: "hi" in repl.dispatched)
    assert "hi" in repl.dispatched
    assert tui._cmd_history == ["hi", "q"]
    assert "\u03bb hi" in "".join(tui._output_surface.capture)
    assert repl.io == "old-io"
    assert repl.console._io == "old-console"


def test_main_ctrl_c_without_command_exits():
    tui = _run_main(_MainFakeRepl(), [3])
    assert tui._running is False
    assert tui._input_buf == []


def test_main_poll_tick_clears_finished_command():
    repl = _MainFakeRepl()
    keys = [ord("h"), ord("i"), ord("\n")] + [-1] * 8 + [ord("q"), ord("\n")]
    tui = _run_main(repl, keys)
    assert _wait_until(lambda: "hi" in repl.dispatched)
    assert tui._active_thread is None
    assert tui._active_cmd is None


def test_main_history_back_fills_line():
    tui = _run_main(_MainFakeRepl(), [curses.KEY_UP, curses.KEY_UP, 3], history=["echo hello", "ai test"])
    assert "".join(tui._input_buf) == "echo hello"
    assert tui._history_pos == 0


def test_main_history_up_down_clears_past_newest():
    tui = _run_main(_MainFakeRepl(), [curses.KEY_UP, curses.KEY_UP, curses.KEY_DOWN, curses.KEY_DOWN, 3], history=["echo hello", "ai test"])
    assert tui._input_buf == []
    assert tui._history_pos == len(tui._cmd_history)


def test_main_ctrl_r_search_enter_keeps_match():
    tui = _run_main(_MainFakeRepl(), [18, ord("h"), ord("e"), ord("\n"), 3], history=["echo hello", "ai test"])
    assert "".join(tui._input_buf) == "echo hello"
    assert not tui._searching


def test_main_esc_cancels_search_restores():
    tui = _run_main(_MainFakeRepl(), [ord("a"), 18, ord("x"), 27, 3])
    assert "".join(tui._input_buf) == "a"
    assert not tui._searching


def test_main_tab_completes_command():
    tui = _run_main(_MainFakeRepl(), [ord("e"), ord("c"), 9, 3])
    assert "".join(tui._input_buf) == "echo "


def test_main_ctrl_l_and_ctrl_o():
    tui = _run_main(_MainFakeRepl(), [12, 15, 3], output=["hello", "world"])
    assert tui._output_surface.capture == []
    assert tui._scroll_target == 1


def test_main_pgup_pgdn_scrolls_panes():
    tui = _run_main(_MainFakeRepl(), [15, curses.KEY_PPAGE, 15, curses.KEY_NPAGE, curses.KEY_NPAGE, 3])
    assert tui._log_scroll == 10
    assert tui._out_scroll == 0
    assert tui._scroll_target == 0


def test_main_kill_yank_and_editing():
    tui = _run_main(_MainFakeRepl(), [ord("h"), ord("i"), 1, 11, 25, 3])
    assert "".join(tui._input_buf) == "hi"
    assert tui._input_cursor == 2


def test_main_ctrl_d_delete_char():
    tui = _run_main(_MainFakeRepl(), [ord("h"), ord("i"), 1, 4, 3])
    assert "".join(tui._input_buf) == "i"


def test_main_ctrl_w_delete_word_back():
    tui = _run_main(_MainFakeRepl(), [ord("a"), ord("b"), ord(" "), ord("c"), ord("d"), 23, 3])
    assert "".join(tui._input_buf) == "ab "


def test_main_ctrl_t_transpose():
    tui = _run_main(_MainFakeRepl(), [ord("h"), ord("i"), 20, 3])
    assert "".join(tui._input_buf) == "ih"


def test_main_backspace():
    tui = _run_main(_MainFakeRepl(), [ord("a"), ord("b"), 8, 3])
    assert "".join(tui._input_buf) == "a"


def test_main_key_left_right():
    tui = _run_main(_MainFakeRepl(), [ord("a"), ord("b"), curses.KEY_LEFT, ord("x"), curses.KEY_RIGHT, 3])
    assert "".join(tui._input_buf) == "axb"
    assert tui._input_cursor == 3


def test_main_key_delete_char():
    tui = _run_main(_MainFakeRepl(), [ord("a"), ord("b"), ord("c"), 1, curses.KEY_DC, 3])
    assert "".join(tui._input_buf) == "bc"


def test_main_alt_word_navigation_and_delete():
    tui = _run_main(_MainFakeRepl(), [ord("a"), ord("a"), ord(" "), ord("b"), ord("b"), 1, 27, ord("f"), 27, ord("b"), 27, ord("d"), 3])
    assert "".join(tui._input_buf) == " bb"
    assert tui._input_cursor == 0


def test_main_seq_ctrl_arrows():
    tui = _run_main(_MainFakeRepl(), [ord("a"), ord("a"), ord(" "), ord("b"), ord("b"), 1, 27, 91, 53, 67, 27, 91, 53, 68, 3])
    assert tui._input_cursor == 0


def test_main_key_resize():
    tui = _run_main(_MainFakeRepl(), [curses.KEY_RESIZE, 3])
    assert tui._rows == 24
    assert tui._cols == 80


def test_main_detect_terminal_resize():
    term = types.SimpleNamespace(lines=30, columns=100)
    tui = _run_main(_MainFakeRepl(), [-1, 3], rows=24, cols=80, term=term)
    assert tui._rows == 30
    assert tui._cols == 100


def test_main_out_search_accept_and_repeat():
    out = ["target: one"] + ["filler " + str(i) for i in range(20)] + ["target: two"]
    tui = _run_main(_MainFakeRepl(), [47, ord("t"), ord("\n"), 110, 3], output=out)
    assert tui._out_search_last == "t"
    assert tui._out_search_sel == 21
    assert not tui._out_searching


def test_main_out_search_cancel_restores():
    tui = _run_main(_MainFakeRepl(), [47, 27, 3], output=["a", "b"])
    assert not tui._out_searching
    assert tui._out_search_q == ""
    assert tui._out_search_sel == -1


def test_main_out_search_backspace():
    tui = _run_main(_MainFakeRepl(), [47, ord("z"), 8, 27, 3], output=["a", "b"])
    assert tui._out_search_sel == -1


def test_main_escdelay_failure_swallowed():
    tui = TuiRepl(_MainFakeRepl(), _FakeLogBuffer())
    tui._running = True
    scr = _ScriptStdscr([3])
    _drive(tui, scr, escdelay_raise=True)
    assert tui._running is False


def test_run_wraps_main_and_restores_io():
    repl = _MainFakeRepl()
    tui = TuiRepl(repl, _FakeLogBuffer())
    scr = _ScriptStdscr([3])
    term = types.SimpleNamespace(lines=24, columns=80)

    def _fake_wrapper(main):
        _drive(tui, scr, term=term)

    with patch.object(curses, "wrapper", side_effect=_fake_wrapper), \
         patch("domains.logging.cli_logger.set_cli_terminal") as sct:
        tui.run()
    assert tui._running is False
    assert repl.io == "old-io"
    assert repl.console._io == "old-console"
    assert sct.call_count == 2


# ── remaining uncovered branches ─────────────────────────────────────────

def test_interrupt_active_async_exc_result_paths():
    tui = TuiRepl(_MainFakeRepl(), None)
    thread = threading.Thread(target=time.sleep, args=(1.0,), daemon=True)
    thread.start()
    tui._active_thread = thread
    with patch.object(tui_mod, "_SET_ASYNC_EXC", return_value=0):
        tui._interrupt_active()
    with patch.object(tui_mod, "_SET_ASYNC_EXC", return_value=2):
        tui._interrupt_active()

    def _boom(*a, **k):
        raise RuntimeError("boom")

    with patch.object(tui_mod, "_SET_ASYNC_EXC", _boom):
        tui._interrupt_active()
    thread.join()


def test_init_pairs_with_colors():
    with patch.object(curses, "has_colors", return_value=True), \
         patch.object(curses, "start_color"), \
         patch.object(curses, "use_default_colors"), \
         patch.object(curses, "init_pair") as ip:
        tui_mod._init_pairs()
    assert ip.call_count == 7


def test_init_pairs_without_colors():
    with patch.object(curses, "has_colors", return_value=False), \
         patch.object(curses, "init_pair") as ip:
        tui_mod._init_pairs()
    assert ip.call_count == 0


def test_read_escape_remainder_bracket_interrupted():
    win = _FakeKeyWin([ord("["), -1])
    assert _read_escape_remainder(win, {}) is None


def test_read_escape_remainder_non_printable():
    win = _FakeKeyWin([200])
    assert _read_escape_remainder(win, {}) is None


def test_blit_handles_curses_errors():
    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    line = tui_mod.RenderLine("hello world", tui_mod.STYLE_INFO)
    with patch.object(curses, "color_pair", side_effect=lambda n: n):
        tui._blit(_ErrorWin("erase"), [line])
        tui._blit(_ErrorWin("addstr"), [line])
        tui._blit(_ErrorWin("refresh"), [line])


def test_render_status_handles_errors():
    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    with patch.object(curses, "color_pair", side_effect=lambda n: n):
        tui._render_status(_ErrorWin("erase"), 80)
        tui._render_status(_ErrorWin("addstr"), 80)


def test_render_input_handles_move_error():
    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    tui._input_buf = ["h", "i"]
    tui._input_cursor = 2
    with patch.object(curses, "color_pair", side_effect=lambda n: n):
        tui._render_input(_MoveErrorWin(), 20)


def test_tui_io_write_routes_to_surface():
    from domains.shell.surface import TextSurface

    surf = TextSurface()
    io = TuiIo(surf)
    io.write("hello", end=" ")
    assert "".join(surf.capture) == "hello "


def test_blit_truncates_lines_past_window():
    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    lines = [tui_mod.RenderLine(f"line {i}", tui_mod.STYLE_INFO) for i in range(5)]
    with patch.object(curses, "color_pair", side_effect=lambda n: n):
        tui._blit(_FakeWin(2, 20), lines)


def test_render_input_handles_addstr_error():
    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    tui._input_buf = ["h", "i"]
    tui._input_cursor = 2
    with patch.object(curses, "color_pair", side_effect=lambda n: n):
        tui._render_input(_ErrorWin("addstr"), 20)


def test_render_status_zero_cols():
    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    with patch.object(curses, "color_pair", side_effect=lambda n: n):
        tui._render_status(_FakeWin(1, 20), 0)


def test_render_status_searching_failed_label():
    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    tui._searching = True
    tui._search_failed = True
    tui._search_q = "zzz"
    with patch.object(curses, "color_pair", side_effect=lambda n: n):
        tui._render_status(_FakeWin(1, 20), 20)


def test_render_status_out_searching_label():
    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    tui._out_searching = True
    tui._out_search_failed = True
    tui._out_search_q = "zzz"
    with patch.object(curses, "color_pair", side_effect=lambda n: n):
        tui._render_status(_FakeWin(1, 20), 20)


def test_main_search_direction_ctrl_s():
    tui = _run_main(_MainFakeRepl(), [19, ord("a"), 19, ord("\n"), 3],
                    history=["echo hello", "ai test"])
    assert "".join(tui._input_buf) == "ai test"
    assert not tui._searching


def test_main_search_backspace():
    tui = _run_main(_MainFakeRepl(), [18, ord("a"), ord("b"), 8, ord("\n"), 3],
                    history=["echo hello", "ai test"])
    assert "".join(tui._input_buf) == "ai test"


def test_main_ctrl_e_end():
    tui = _run_main(_MainFakeRepl(), [ord("h"), ord("i"), 1, 5, 3])
    assert tui._input_cursor == 2


def test_main_ctrl_u_kill_to_start():
    tui = _run_main(_MainFakeRepl(), [ord("h"), ord("i"), ord(" "), ord("w"), 21, 3])
    assert tui._input_buf == []
    assert tui._input_cursor == 0


def test_main_log_pane_pgdn():
    tui = _run_main(_MainFakeRepl(), [15, curses.KEY_PPAGE, curses.KEY_NPAGE, 3])
    assert tui._log_scroll == 0
    assert tui._scroll_target == 1


def test_main_ctrl_arrow_codes():
    tui = _run_main(_MainFakeRepl(),
                    [ord("a"), ord("a"), ord(" "), ord("b"), ord("b"), 1,
                     tui_mod._KEY_CTRL_RIGHT, tui_mod._KEY_CTRL_LEFT, 3])
    assert tui._input_cursor == 0


class _BusyRepl(_MainFakeRepl):
    def __init__(self):
        super().__init__()
        self.interrupts = 0

    def _dispatch(self, cmd):
        end = time.monotonic() + 10
        try:
            while time.monotonic() < end:
                time.sleep(0.001)
        except KeyboardInterrupt:
            self.interrupts += 1
        else:
            self.dispatched.append(cmd)


def test_main_ctrl_c_interrupts_active_command():
    repl = _BusyRepl()
    keys = [ord("h"), ord("i"), ord("\n"), -1, -1, -1, 3] + [-1] * 8 + [ord("q"), ord("\n")]
    tui = _run_main(repl, keys)
    assert "^C" in "".join(tui._output_surface.capture)
    assert _wait_until(
        lambda: repl.interrupts == 1
        and (tui._active_thread is None or not tui._active_thread.is_alive())
    )


def test_main_unknown_key_blits_and_continues():
    tui = _run_main(_MainFakeRepl(), [999, 3])
    assert tui._running is False


def test_main_getch_keyboard_interrupt_breaks():
    class _KIStdscr(_ScriptStdscr):
        def getch(self):
            raise KeyboardInterrupt()

    tui = TuiRepl(_MainFakeRepl(), _FakeLogBuffer())
    tui._running = True
    _drive(tui, _KIStdscr([]))
    assert tui._running is False


def test_main_detect_resize_terminal_oserror():
    tui = _run_main(_MainFakeRepl(), [-1, 3], term_error=True)
    assert tui._rows == 24


def test_main_detect_resize_non_positive():
    term = types.SimpleNamespace(lines=0, columns=80)
    tui = _run_main(_MainFakeRepl(), [-1, 3], term=term)
    assert tui._rows == 24


def test_main_detect_resize_resizeterm_error():
    tui = _run_main(_MainFakeRepl(), [-1, 3], resize_error=True)
    assert tui._rows == 24


def test_main_detect_resize_resizeterm_error_on_change():
    term = types.SimpleNamespace(lines=30, columns=100)
    tui = _run_main(_MainFakeRepl(), [-1, 3], term=term, resize_error=True)
    assert tui._rows == 24


def test_main_search_direction_backward():
    tui = _run_main(_MainFakeRepl(), [18, ord("a"), 18, ord("\n"), 3],
                    history=["echo hello", "ai test"])
    assert not tui._searching


def test_main_pgup_scrolls_output_pane():
    tui = _run_main(_MainFakeRepl(), [curses.KEY_PPAGE, curses.KEY_NPAGE, 3])
    assert tui._scroll_target == 0
    assert tui._out_scroll == 0


def test_run_handles_cli_logger_import_error():
    import sys

    repl = _MainFakeRepl()
    tui = TuiRepl(repl, _FakeLogBuffer())
    scr = _ScriptStdscr([3])
    term = types.SimpleNamespace(lines=24, columns=80)

    def _fake_wrapper(main):
        _drive(tui, scr, term=term)

    with patch.object(curses, "wrapper", side_effect=_fake_wrapper), \
         patch.dict(sys.modules, {"domains.logging.cli_logger": None}):
        tui.run()
    assert tui._running is False
    assert repl.io == "old-io"


def test_module_async_exc_import_fallback():
    import importlib

    with patch.object(ctypes, "pythonapi", None):
        importlib.reload(tui_mod)
    assert tui_mod._SET_ASYNC_EXC is None
    importlib.reload(tui_mod)
    assert tui_mod._SET_ASYNC_EXC is not None
