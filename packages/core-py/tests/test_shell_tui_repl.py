"""
Tests for the curses-free logic of the split-pane TUI shell: path
completion, token completion, and reverse history search.

``TuiRepl.__init__`` builds only surfaces and the layout (no curses), so the
completion/search helpers can be exercised without a terminal.
"""

import os

import pytest

from domains.shell.tui_repl import TuiRepl, _complete_path, _read_escape_remainder


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
