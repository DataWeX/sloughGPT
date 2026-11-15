"""
Tests for the curses-free logic of the split-pane TUI shell: path
completion, token completion, and reverse history search.

``TuiRepl.__init__`` builds only surfaces and the layout (no curses), so the
completion/search helpers can be exercised without a terminal.
"""

import ctypes
import curses
import os
import threading
import time
import types
from contextlib import contextmanager
from unittest.mock import patch

import pytest

import domain.shell._internal.tui_repl as tui_mod

# regions are pane.Rect, NOT window.Rect -- the two are positional transposes
# of each other.
from domain.shell._internal.pane import Rect
from domain.shell._internal.tui_repl import TuiIo, TuiRepl, _complete_path
from domain.shell._internal.tui_session import (
    KEY_ALT,
    KEY_ESC,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_UNKNOWN,
    Key,
    decode_key,
    from_curses,
)


class _FakeRepl:
    COMMANDS = ["ai", "alias", "about", "bg", "cd", "echo", "exit"]

    def __init__(self):
        self.console = types.SimpleNamespace(_io="fake-console")


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
    from domain.shell._internal.surface import TextSurface

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


def _presses(stream: str) -> list[Key]:
    """Every press in ``stream``, decoded over one shared cursor.

    Replaces ``_FakeKeyWin`` + ``_read_escape_remainder``. The "read what
    follows Esc" decision moved out of ``tui_repl`` and into ``decode_key`` in
    ``tui_session`` when the dispatch was ported onto ``Key`` objects, so these
    cases are now asserted against the ported entry point rather than against
    the helper it absorbed.

    The source yields a single byte per call because that is what a terminal
    is: bytes arrive one at a time and a sequence is only known to be over when
    the source stops producing. ``stream`` is therefore the literal bytes the
    keypress would have put on the wire.

    One cursor for the whole stream, rather than a fresh one per press, is what
    makes "the parser consumed its own sequence" assertable: a parse that stops
    short leaves its tail behind, and the loop then decodes that tail as an
    *extra* press instead of quietly dropping it. A per-press cursor would
    re-read the same bytes and never notice the leak — which is how the old
    `_read_escape_remainder` bug survived tests that only checked the first
    result.
    """
    data = stream.encode("latin-1")
    pos = 0

    def read_byte() -> bytes:
        nonlocal pos
        if pos >= len(data):
            return b""
        byte = data[pos]
        pos += 1
        return bytes([byte])

    presses: list[Key] = []
    while pos < len(data):
        presses.append(decode_key(read_byte))
    return presses


def _decode(stream: str) -> tuple[str, str]:
    """The first press as ``(kind, char)``, the vocabulary the loop dispatches on.

    For presses that carry no modifier this is the whole story; the two Ctrl
    arrow cases assert through `_presses` instead, because whether they are
    *chorded* lives on a flag the pair cannot show.
    """
    key = _presses(stream)[0]
    return key.kind, key.char


def test_decode_lone_esc_reports_esc():
    # Was `None` from `_read_escape_remainder`, which the caller read as Esc.
    assert _decode("\x1b") == (KEY_ESC, "")


def test_decode_double_esc_reports_unknown_not_esc():
    """Two Escs inside one read are unmodelled, so they are inert — a delta.

    Was ``None`` from ``_read_escape_remainder``, which the caller read as Esc,
    so this is one of the behaviour changes the port needs a veto on: pressing
    Esc twice within ``tui_session.ESC_DELAY`` (25 ms) used to cancel, and now
    reads ``KEY_UNKNOWN`` and does nothing. The second Esc is *consumed*, so the
    user has to press a third time.

    It is reachable — key repeat easily lands two Escs inside 25 ms — and it is
    a consequence of a deliberate rule, not an oversight: ``_decode_escape``
    reports ``KEY_UNKNOWN`` for any introducer it has no case for rather than
    claiming Esc, on the grounds that "I read something I cannot name" is not
    the same statement as "the user pressed Esc". ``ESC ESC`` is the case where
    that rule looks wrong, because both bytes *are* named. Changing it is a
    decoder decision, not a test decision, so the test states what the code
    does today and this comment carries the question.
    """
    assert _decode("\x1b\x1b") == (KEY_UNKNOWN, "")


def test_decode_alt_f_reports_alt_char():
    # Was the *bound* action string "alt:fwd"; the binding table lives in the
    # dispatch now, so the decoder's contract stops at (kind, char).
    assert _decode("\x1bf") == (KEY_ALT, "f")


def test_decode_alt_unbound_char_is_inert():
    # Was `None` = "no binding, do nothing". `KEY_ALT`/`"x"` has no branch in
    # the dispatch either, so it is inert there for the same reason.
    assert _decode("\x1bx") == (KEY_ALT, "x")


def test_decode_ctrl_left_consumes_the_whole_sequence():
    # The old parse matched only after reading five bytes one at a time and
    # could leave `;5D` in the stream — so the *next* read handed the prompt a
    # literal `5` and then a `D`. Counting the presses in one stream is what
    # makes that leakable at all: checking only the first result passes even
    # when a tail survives it.
    presses = _presses("\x1b[1;5D")
    assert len(presses) == 1, f"sequence left bytes behind: {presses[1:]!r}"
    assert presses[0].matches(KEY_LEFT, ctrl=True)


def test_decode_ctrl_right_consumes_the_whole_sequence():
    presses = _presses("\x1b[1;5C")
    assert len(presses) == 1, f"sequence left bytes behind: {presses[1:]!r}"
    assert presses[0].matches(KEY_RIGHT, ctrl=True)


def test_decode_the_bare_csi_form_is_the_same_chord():
    # Two encodings of one key. Without the fold these read differently, which
    # is why Ctrl+Left moved the caret by one character on terminals sending
    # the bare form.
    bare, named = _presses("\x1b[5D"), _presses("\x1b[1;5D")
    assert len(bare) == len(named) == 1
    assert bare[0].matches(KEY_LEFT, ctrl=True)
    assert named[0].matches(KEY_LEFT, ctrl=True)


def test_decode_interrupted_csi_is_unknown_not_esc():
    # Was `None`, which the caller read as Esc. A half-read sequence is not an
    # Esc press, so reporting Esc here would cancel whatever was open.
    assert _decode("\x1b[") == (KEY_UNKNOWN, "")


def test_decode_non_printable_after_esc_is_unknown():
    # Was `None`. Byte 200 is not printable ASCII, so Esc followed by it is an
    # unmodelled sequence rather than a character or a cancel.
    assert _decode("\x1b\xc8") == (KEY_UNKNOWN, "")


# Note: `test_read_escape_remainder_restores_timeout` has no replacement, and
# deliberately so. It asserted that the helper put the window's 200 ms timeout
# back after reading an escape tail. There is no window-scoped timeout left to
# restore: the poll interval is an *argument* to `read_key(timeout=0.1)`, so
# every read starts its own deadline and nothing survives it to be restored.


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
            entries = [
                types.SimpleNamespace(
                    timestamp=1700000000.0, level="INFO", source="test", message="boot"
                )
            ]
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
    """A window whose script is a list of **key presses**, not of bytes.

    Each element is one thing the user did: ``b"\\x1b[5C"`` is a single
    Ctrl+Right, ``ord("a")`` is one printable key, ``curses.KEY_UP`` is one key
    as ncurses folds it, ``Key(...)`` is one already-normalised key, and ``-1``
    is the timeout sentinel. The decoder — not the script — decides where a
    press *ends*, so the script has to say where one *begins*: written as four
    elements, ``27, 91, 53, 67`` would hand the loop four keys for the one
    keystroke they spell between them.
    """

    def __init__(self, keys, rows=24, cols=80):
        super().__init__(rows, cols)
        self._keys = list(keys)

    def keypad(self, flag):
        pass

    def timeout(self, ms):
        pass

    def next_key(self):
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


class _FakeSession:
    """The loop's two inputs, scripted — keys and geometry, both from ``scr``.

    ``_main`` asks its session for exactly two things, ``read_key`` and
    ``size``, so this stands in for ``TtySession`` with a window the tests
    already script. It reproduces what ``TtySession.read_key`` actually does
    rather than approximating it: one press in, one ``Key`` out, decoded by the
    same ``decode_key`` the terminal path uses, with the byte source scoped to
    that press — which is ``terminal_bytes`` with a zero-length ``ESC_DELAY``,
    i.e. "the tail arrived with the sequence or it is not coming".

    Two inputs bypass the byte decoder, both on purpose. ``curses.KEY_*`` codes
    at or above 256 are ncurses' *folded* form of a key that has no single
    byte, so they go through ``from_curses``, the mirror ``probe_session``
    pins; and an already-built ``Key`` is passed straight through, because some
    vocabulary has no producer on either wire (``KEY_RESIZE``, and the
    Ctrl+arrow chords, since this ncurses build defines no constant for them
    to fold from).

    ``-1`` is the timeout sentinel and becomes ``None``: the poll tick the loop
    expects from a ``read_key`` whose deadline expired. It is not a key.
    Reporting it as one would run the key branches instead of the refresh,
    which is the whole difference the poll interval exists to make.
    """

    owns_stdin = False

    def __init__(self, scr):
        self._scr = scr

    def size(self):
        return self._scr.getmaxyx()

    def read_key(self, timeout=None):
        press = self._scr.next_key()
        if press is None or press == -1:
            return None
        if isinstance(press, Key):
            return press
        if isinstance(press, int):
            if press >= 256:
                return from_curses(press, curses)
            press = bytes([press])
        data = press if isinstance(press, bytes) else press.encode("latin-1")
        pos = 0

        def read_byte() -> bytes:
            nonlocal pos
            if pos >= len(data):
                return b""
            byte = data[pos]
            pos += 1
            return bytes([byte])

        return decode_key(read_byte)

    def raw(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write(self, text):
        pass


@contextmanager
def _loop(tui, scr, term=None, term_error=False, resize_error=False):
    """Give *tui* a scripted session and hold the one patch its loop still reads.

    This replaces the curses scaffolding `_drive` used to install. `curs_set`,
    `raw`, `newwin`, `color_pair`, `has_colors` and `wrapper` are all gone from
    `_main`/`run`, so patching them proved only that nothing called them — a
    claim about ncurses, not about this loop. What the loop still reads from
    outside itself is the kernel's idea of the window size, and that is the one
    thing left to script.

    `set_escdelay` is armed to *raise*, unconditionally. Esc delay is the
    constant `tui_session.ESC_DELAY` now rather than a call into ncurses, so
    the patch can never fire — a stronger claim than surviving it when it does.

    `resize_error` is kept for the same reason: it arms a `resizeterm` that is
    no longer reachable, so a loop driven with it on proves the call is not
    made rather than that a failure there would be tolerated.
    """
    if term is None:
        term = types.SimpleNamespace(lines=scr.rows, columns=scr.cols)
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
    previous = tui._session
    with (
        patch.object(curses, "set_escdelay", side_effect=RuntimeError("boom")),
        resize,
        getsize,
    ):
        tui._session = _FakeSession(scr)
        try:
            yield
        finally:
            tui._session = previous


def _drive(tui, scr, term=None, term_error=False, resize_error=False):
    with _loop(tui, scr, term=term, term_error=term_error, resize_error=resize_error):
        tui._main()


def _run_main(
    repl,
    keys,
    rows=24,
    cols=80,
    term=None,
    history=None,
    output=None,
    term_error=False,
    resize_error=False,
):
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
    tui = _run_main(
        _MainFakeRepl(),
        [curses.KEY_UP, curses.KEY_UP, 3],
        history=["echo hello", "ai test"],
    )
    assert "".join(tui._input_buf) == "echo hello"
    assert tui._history_pos == 0


def test_main_history_up_down_clears_past_newest():
    tui = _run_main(
        _MainFakeRepl(),
        [curses.KEY_UP, curses.KEY_UP, curses.KEY_DOWN, curses.KEY_DOWN, 3],
        history=["echo hello", "ai test"],
    )
    assert tui._input_buf == []
    assert tui._history_pos == len(tui._cmd_history)


def test_main_ctrl_r_search_enter_keeps_match():
    tui = _run_main(
        _MainFakeRepl(),
        [18, ord("h"), ord("e"), ord("\n"), 3],
        history=["echo hello", "ai test"],
    )
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
    tui = _run_main(
        _MainFakeRepl(),
        [15, curses.KEY_PPAGE, 15, curses.KEY_NPAGE, curses.KEY_NPAGE, 3],
    )
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
    tui = _run_main(
        _MainFakeRepl(),
        [ord("a"), ord("b"), curses.KEY_LEFT, ord("x"), curses.KEY_RIGHT, 3],
    )
    assert "".join(tui._input_buf) == "axb"
    assert tui._input_cursor == 3


def test_main_key_delete_char():
    tui = _run_main(_MainFakeRepl(), [ord("a"), ord("b"), ord("c"), 1, curses.KEY_DC, 3])
    assert "".join(tui._input_buf) == "bc"


def test_main_alt_word_navigation_and_delete():
    # Alt+<letter> is ESC plus the letter *in one press*. Scripting them as
    # separate elements would decode the Esc alone and then type the letter,
    # which is precisely the desync the decoder exists to prevent.
    tui = _run_main(
        _MainFakeRepl(),
        [
            ord("a"),
            ord("a"),
            ord(" "),
            ord("b"),
            ord("b"),
            1,
            b"\x1bf",
            b"\x1bb",
            b"\x1bd",
            3,
        ],
    )
    assert "".join(tui._input_buf) == " bb"
    assert tui._input_cursor == 0


def test_main_seq_ctrl_arrows():
    tui = _run_main(
        _MainFakeRepl(),
        [
            ord("a"),
            ord("a"),
            ord(" "),
            ord("b"),
            ord("b"),
            1,
            b"\x1b[5C",
            b"\x1b[5D",
            3,
        ],
    )
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


def test_main_never_calls_set_escdelay():
    """Esc delay is a constant in `tui_session`, not a call into ncurses.

    This used to prove the loop *survived* `curses.set_escdelay` raising, with
    the failing patch armed only when a test asked for it. `set_escdelay` is
    gone from `_main` along with the rest of the ncurses setup — the wait is
    `tui_session.ESC_DELAY`, read by the decoder — so `_loop` now arms the
    raising patch for **every** driven loop. The claim this makes is therefore
    not one test's claim but the whole file's: if any path reached ncurses'
    esc delay, every test driven through `_drive` would fail here, not just
    this one. This test exists to name that guarantee and to check the loop
    still terminates cleanly with the trap set.
    """
    tui = TuiRepl(_MainFakeRepl(), _FakeLogBuffer())
    tui._running = True
    _drive(tui, _ScriptStdscr([3]))
    assert tui._running is False


def test_run_wraps_main_and_restores_io():
    """`run()` is the raw-mode scope now, not `curses.wrapper`.

    The wrapper used to be patched out here because entering it needs a real
    terminal. Its replacement is `self._session.raw()`, and the scripted
    session's `raw()` is a no-op context manager, so the real entry point runs
    unpatched — which is a better test than one that swaps the entry point out.
    What the wrapper guaranteed is still what is asserted: the loop runs to
    completion, and both I/O bindings are put back afterwards.
    """
    repl = _MainFakeRepl()
    tui = TuiRepl(repl, _FakeLogBuffer())
    scr = _ScriptStdscr([3])
    term = types.SimpleNamespace(lines=24, columns=80)

    with (
        _loop(tui, scr, term=term),
        patch("domain.logging._internal.cli_logger.set_cli_terminal") as sct,
    ):
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


def test_curses_never_registers_colour_pairs():
    """ncurses must not be told about colour pairs at all any more.

    ``_init_pairs()`` used to call ``curses.init_pair()`` for each of the seven
    pairs. It was emptied when the engine started emitting raw SGR, then removed
    outright. Reintroducing it would put ncurses' colour state back in play for a
    screen it no longer owns -- and the damage would not show up as a render
    difference anyone could spot, because ncurses would simply be maintaining a
    second colour model that nothing reads.

    Asserting *absence* is what makes this a regression guard. The old test
    asserted ``call_count == 7``, pinning the very thing that had to die; this
    one pins that it stays dead, and both of its cases (with and without colour)
    collapse into it, because the correct count is 0 either way.
    """
    tui = TuiRepl(_MainFakeRepl(), _FakeLogBuffer())
    tui._running = True
    scr = _ScriptStdscr([3])
    with patch.object(curses, "init_pair") as ip:
        _drive(tui, scr)
    assert ip.call_count == 0, (
        f"curses.init_pair was called {ip.call_count} time(s): the renderer is "
        "registering colour pairs it does not read"
    )


# `test_read_escape_remainder_bracket_interrupted` and
# `test_read_escape_remainder_non_printable` used to sit here, apart from the
# other seven only because they were appended later. Both drove the same
# helper, and that helper is `decode_key` now -- so they became
# `test_decode_interrupted_csi_is_unknown_not_esc` and
# `test_decode_non_printable_after_esc_is_unknown` above, under the rest of
# their family. A second copy here would pin the same bytes twice by two names.


def test_blit_writes_to_the_layer_and_clips_past_its_width():
    """A line longer than its layer must be clipped, not written past it.

    Replaces ``test_blit_truncates_lines_past_window``, which passed
    ``_FakeWin(2, 20)`` as the *layer name*. ``_get_layer`` recognised no pane
    and returned ``None``, so ``_blit`` returned before reading a single line:
    the test asserted nothing and still passed. A test that cannot fail is not a
    weak test, it is a missing one that looks like coverage.

    The layer here is real, so there is something to clip against. ``_blit``
    takes ``ln.text[: w - offset_x - 1]``, so with ``offset_x == 0`` a line is
    cut at ``w - 1`` and the final column is never inked. Both halves are
    asserted: content must land *and* the last column must stay clear, because
    a blit that wrote nothing would satisfy the clipping assertion alone.
    """
    from domain.shell._internal.pane import Border, Pane, PaneLayout

    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    layout = PaneLayout()
    layout.panes.append(Pane("console", 1.0, border=Border("none")))
    tui._layout = layout
    regions = layout.compute(24, 80)
    _setup_layers(tui, regions)

    layer = tui._layer_console
    cols = layer.framebuffer.cols
    assert cols > 2, "layer is too narrow for the clipping rule to be observable"
    lines = [
        tui_mod.RenderLine(f"{i:02d}" + "X" * (cols * 3), tui_mod.STYLE_INFO) for i in range(5)
    ]
    with patch.object(curses, "color_pair", side_effect=lambda n: n):
        tui._blit("console", lines, offset_y=0, offset_x=0)

    for row in range(layer.framebuffer.rows):
        first = layer.framebuffer.get(row, 0).char
        last = layer.framebuffer.get(row, cols - 1).char
        if row < len(lines):
            # Positive evidence the line arrived: its own leading digit. The
            # pane is textured (see _PANE_PATTERN), so a non-blank cell is no
            # longer proof of ink -- the backdrop is non-blank on its own.
            assert first in "0123456789", (
                f"row {row}: line text never reached the layer (got {first!r})"
            )
        # The clip must keep text out of the final column; the backdrop's own
        # glyph there is fine -- it is not text.
        assert last != "X", (
            f"row {row}: column {cols - 1} holds {last!r}; a line of "
            f"{cols * 3} chars was not clipped to the layer width"
        )


def test_render_never_writes_to_the_curses_window():
    """The render path is separated from curses, not protected from it.

    Replaces ``test_render_status_handles_errors``,
    ``test_render_input_handles_move_error`` and
    ``test_render_input_handles_addstr_error``, which all asserted the same
    thing in three places: that a curses write failure is swallowed. Removing
    curses from the screen-writer deleted the ``except curses.error: pass`` those
    tests guarded, so they went red -- correctly, as it turned out, because the
    tolerance had been lost rather than moved.

    The guarantee now lives somewhere better. ``stdscr`` is input-only (see the
    single member function below), so no render call can fail on a curses write
    because none is made. Driving a whole loop against a window that raises on
    every write proves that directly, and covers what three overlapping tests
    only danced around.
    """
    tui = _init_render_state(TuiRepl(_MainFakeRepl(), _FakeLogBuffer()))
    tui._input_buf = ["h", "i"]
    tui._input_cursor = 2
    tui._searching = True
    tui._search_failed = True
    tui._search_q = "zzz"
    tui._out_searching = True
    tui._out_search_failed = True
    tui._out_search_q = "zzz"
    scr = _WritePoisonScr()
    with patch.object(curses, "color_pair", side_effect=lambda n: n):
        tui._render_status(80)
        tui._render_status(0)
        tui._render_input(20)
        tui._render_confirm()
        tui._render_ask()
        tui._render_select({"output": Rect(0, 0, 24, 80)})
        tui._render_all(
            {
                "output": Rect(0, 0, 24, 80),
                "status": Rect(22, 0, 1, 80),
                "input": Rect(23, 0, 1, 80),
                "console": Rect(0, 0, 24, 80),
            }
        )
        tui._draw_borders(
            {
                "output": Rect(0, 0, 24, 80),
                "status": Rect(22, 0, 1, 80),
                "input": Rect(23, 0, 1, 80),
                "console": Rect(0, 0, 24, 80),
            }
        )
    assert scr.write_calls == [], (
        f"the render path called curses {scr.write_calls} -- curses must be "
        "input-only now that the engine owns the screen"
    )
    # And confirm the poison was real: a window that never raises cannot prove
    # anything, so assert at least one writer exists to have been caught.
    try:
        scr.addnstr(0, 0, "x", 1)
    except curses.error:
        pass
    else:  # pragma: no cover - guards against the spy rotting silently
        raise AssertionError("_WritePoisonScr.addnstr no longer raises")


def test_tui_io_write_routes_to_surface():
    from domain.shell._internal.surface import TextSurface

    surf = TextSurface()
    io = TuiIo(surf)
    io.write("hello", end=" ")
    assert "".join(surf.capture) == "hello "


def test_render_status_zero_cols():
    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    with patch.object(curses, "color_pair", side_effect=lambda n: n):
        tui._render_status(0)


def test_render_status_searching_failed_label():
    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    tui._searching = True
    tui._search_failed = True
    tui._search_q = "zzz"
    with patch.object(curses, "color_pair", side_effect=lambda n: n):
        tui._render_status(20)


def test_render_status_out_searching_label():
    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    tui._out_searching = True
    tui._out_search_failed = True
    tui._out_search_q = "zzz"
    with patch.object(curses, "color_pair", side_effect=lambda n: n):
        tui._render_status(20)


def test_main_search_direction_ctrl_s():
    tui = _run_main(
        _MainFakeRepl(),
        [19, ord("a"), 19, ord("\n"), 3],
        history=["echo hello", "ai test"],
    )
    assert "".join(tui._input_buf) == "ai test"
    assert not tui._searching


def test_main_search_backspace():
    tui = _run_main(
        _MainFakeRepl(),
        [18, ord("a"), ord("b"), 8, ord("\n"), 3],
        history=["echo hello", "ai test"],
    )
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
    """Ctrl+Left/Right move by word, fed as `Key`s because no code exists here.

    The old version fed `tui_mod._KEY_CTRL_LEFT/RIGHT`, which were
    `getattr(curses, "KEY_CTRL_LEFT", 554)` — a code *this* ncurses build does
    not define (both names are missing, so the fallback was never a real
    `getch()` result). It feeds `Key(KEY_RIGHT, ctrl=True)` now, and the
    reason that is an improvement rather than a rename is that the modifier
    rides on the arrow: "which integer is Ctrl+Left" no longer has to be
    answered at all, which is what the `554` fallback existed to dodge. The
    byte form of these chords is `ESC [ 1 ; 5 C`, which
    `test_main_seq_ctrl_arrows` drives; there is no folded constant to feed
    here either, so what is left to test is the dispatch, and the vocabulary
    is what the dispatch reads.

    Asserted in both directions rather than only on the round trip: a
    round-trip alone passes when *neither* arrow does anything, so the
    forward-only run is what makes the pair able to fail.
    """
    home = [ord("a"), ord("a"), ord(" "), ord("b"), ord("b"), 1]
    forward = _run_main(_MainFakeRepl(), [*home, Key(KEY_RIGHT, ctrl=True), 3])
    assert forward._input_cursor != 0, "Ctrl+Right did not move the caret"
    backward = _run_main(
        _MainFakeRepl(),
        [*home, Key(KEY_RIGHT, ctrl=True), Key(KEY_LEFT, ctrl=True), 3],
    )
    assert backward._input_cursor == 0, "Ctrl+Left did not bring the caret back"


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
        lambda: (
            repl.interrupts == 1
            and (tui._active_thread is None or not tui._active_thread.is_alive())
        )
    )


def test_main_unknown_key_blits_and_continues():
    tui = _run_main(_MainFakeRepl(), [999, 3])
    assert tui._running is False


def test_main_key_read_keyboard_interrupt_breaks():
    """A KeyboardInterrupt out of the *read* ends the loop, as it must.

    The reader is `session.read_key` now, not `stdscr.getch`, so the poison
    moves with it: whatever raises comes up through the session and out of the
    loop's `try`, and `_running` is cleared on the way.
    """

    class _KIStdscr(_ScriptStdscr):
        def next_key(self):
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


def test_main_detect_resize_commits_when_kernel_size_changes():
    """The kernel owns the window size, so a size change always commits.

    This used to assert ``tui._rows == 24``: that a ``curses.resizeterm``
    failure rolled the new size back. ``resizeterm`` is no longer called at all
    -- it cleared the screen as a side effect, and nothing reads ncurses'
    geometry model now that the engine owns the framebuffer. Withdrawing
    curses' veto over the size is the point of the change, so the assertion
    flips: the observed kernel size wins, even when the curses patch behind it
    is still set to raise.

    The ``resize_error`` flag is kept deliberately. It still arms the failing
    ``resizeterm`` patch, so this test now proves the *unreachability* of that
    call rather than its failure mode -- which is a stronger claim.
    """
    term = types.SimpleNamespace(lines=30, columns=100)
    tui = _run_main(_MainFakeRepl(), [-1, 3], term=term, resize_error=True)
    assert tui._rows == 30
    assert tui._cols == 100


def test_main_search_direction_backward():
    tui = _run_main(
        _MainFakeRepl(),
        [18, ord("a"), 18, ord("\n"), 3],
        history=["echo hello", "ai test"],
    )
    assert not tui._searching


def test_main_pgup_scrolls_output_pane():
    tui = _run_main(_MainFakeRepl(), [curses.KEY_PPAGE, curses.KEY_NPAGE, 3])
    assert tui._scroll_target == 0
    assert tui._out_scroll == 0


def test_run_handles_cli_logger_import_error():
    """`run()` must still start the loop when the logger cannot be imported.

    Terminal bookkeeping is a side effect, not a dependency: if
    `set_cli_terminal` cannot be reached the loop runs anyway and simply never
    tells the logger the screen is busy. Two things had to change for this to
    be that test rather than a test that happens to pass. The `sys.modules`
    key named the module's *old* path, so the import succeeded and the
    fallback branch never ran; and nothing asserted the branch was taken. Both
    are fixed, and the entry point is the real `run()` now — `curses.wrapper`
    is no longer what needs replacing, `self._session.raw()` is.
    """
    import sys

    repl = _MainFakeRepl()
    tui = TuiRepl(repl, _FakeLogBuffer())
    scr = _ScriptStdscr([3])
    term = types.SimpleNamespace(lines=24, columns=80)

    with (
        # Entered before the module is masked, so the target resolves while it
        # is still importable; exits last, so the real attribute is restored.
        patch("domain.logging._internal.cli_logger.set_cli_terminal") as sct,
        patch.dict(sys.modules, {"domain.logging._internal.cli_logger": None}),
        _loop(tui, scr, term=term),
    ):
        tui.run()
    assert tui._running is False
    assert repl.io == "old-io"
    assert sct.call_count == 0, (
        "set_cli_terminal was reached, so the ImportError fallback was never "
        "taken and this is not the path its name claims"
    )


def test_module_async_exc_import_fallback():
    import importlib

    with patch.object(ctypes, "pythonapi", None):
        importlib.reload(tui_mod)
    assert tui_mod._SET_ASYNC_EXC is None
    importlib.reload(tui_mod)
    assert tui_mod._SET_ASYNC_EXC is not None


# ── _draw_borders ───────────────────────────────────────────────────────


class _FakeStdscr:
    def __init__(self, rows=24, cols=80):
        self._rows = rows
        self._cols = cols
        self.addnstr_calls = []
        self.addch_calls = []
        self.refreshed = False

    def addnstr(self, y, x, text, n, attr=0):
        self.addnstr_calls.append((y, x, text, n, attr))

    def addch(self, y, x, ch, attr=0):
        self.addch_calls.append((y, x, ch, attr))

    def refresh(self):
        self.refreshed = True

    def getmaxyx(self):
        return (self._rows, self._cols)


class _WritePoisonScr(_FakeStdscr):
    """A curses window that raises on every *write* and stays silent if ignored.

    The input side -- ``getch``, ``getmaxyx``, ``timeout``, ``keypad`` -- keeps
    working, because that is the only part of curses the REPL still uses. Every
    method that draws raises instead of recording, so a render that reaches for
    the window fails twice: once for the exception it raised, and once for having
    called it at all.

    This replaces ``*_handles_errors``, which asserted the renderer *survived* a
    curses write failure. The renderer no longer writes to curses, so there is
    nothing to survive -- and "survives an impossible call" is a weaker claim than
    "never makes the call". Note ``_FakeStdscr`` still gets hit if anything does
    write: ``_WritePoisonScr`` only adds poison, it does not hide the spies.
    """

    def __init__(self, rows=24, cols=80):
        super().__init__(rows, cols)
        self.write_calls: list[str] = []

    def _poison(self, name: str):
        def _raise(*a, **k):
            self.write_calls.append(name)
            raise curses.error(name)

        return _raise

    def erase(self):
        return self._poison("erase")()

    def clear(self):
        return self._poison("clear")()

    def clrtoeol(self):
        return self._poison("clrtoeol")()

    def addstr(self, y, x, text, attr=0):
        return self._poison("addstr")()

    def addnstr(self, y, x, text, n, attr=0):
        return self._poison("addnstr")()

    def addch(self, y, x, ch, attr=0):
        return self._poison("addch")()

    def move(self, y, x):
        return self._poison("move")()

    def refresh(self):
        return self._poison("refresh")()

    def hline(self, y, x, ch, n, attr=0):
        return self._poison("hline")()

    def vline(self, y, x, ch, n, attr=0):
        return self._poison("vline")()


def _setup_layers(tui, regions):
    """Attach an engine and a layer per pane, so ``_draw_borders`` has a target.

    ``_draw_borders`` returns immediately when ``tui._engine is None``. Any test
    that forgets this therefore asserts nothing at all while still passing --
    which is exactly how ``test_draw_borders_skips_empty_border`` and
    ``test_draw_borders_skips_invisible_pane`` came to pass without ever reaching
    the ``continue`` branch they were named for. Centralising it means a test can
    only be vacuous by omitting the call, where at least the omission is visible
    in the test body instead of being buried in twelve lines of setup.

    Regions are looked up rather than indexed: a layout with a single pane really
    does produce a single-entry ``regions``, and the layers for the panes that
    were not laid out simply do not exist.
    """
    from domain.shell._internal.graphics import GraphicsEngine, Layer

    tui._engine = GraphicsEngine()
    tui._engine._rows = 24
    tui._engine._cols = 80
    # (region key, layer attribute, has a background layer, background z, z)
    spec = (
        ("console", "_layer_console", True, 0, 1),
        ("output", "_layer_output", True, 2, 3),
        ("status", "_layer_status", False, None, 4),
        ("input", "_layer_input", False, None, 5),
    )
    for key, attr, has_bg, bg_z, z in spec:
        r = regions.get(key)
        if r is None:
            continue
        if has_bg and bg_z is not None:
            setattr(tui, attr + "_bg", Layer(key + "_bg", r.rows, r.cols, z=bg_z))
        setattr(tui, attr, Layer(key, r.rows, r.cols, z=z))


def test_draw_borders_horizontal():
    from domain.shell._internal.pane import Border, Pane, PaneLayout

    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    layout = PaneLayout()
    layout.panes.append(Pane("console", 0.5, border=Border("all")))
    layout.panes.append(Pane("output", 0.5, border=Border("all")))
    layout.panes.append(Pane("status", fixed=1))
    layout.panes.append(Pane("input", fixed=1))
    tui._layout = layout
    regions = layout.compute(24, 80)

    _setup_layers(tui, regions)

    tui._draw_borders(regions)

    # Check that borders were drawn on engine layers
    console_ch = tui._layer_console.framebuffer.get(0, 0).char
    output_ch = tui._layer_output.framebuffer.get(0, 0).char
    # Vertical borders overwrite horizontal at corners (correct box-drawing)
    assert console_ch in ("─", "│") or output_ch in ("─", "│")


def test_draw_borders_vertical():
    from domain.shell._internal.pane import Border, Pane, PaneLayout

    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    layout = PaneLayout()
    layout.panes.append(Pane("console", 0.5, border=Border("all")))
    layout.panes.append(Pane("output", 0.5, border=Border("all")))
    layout.panes.append(Pane("status", fixed=1))
    layout.panes.append(Pane("input", fixed=1))
    tui._layout = layout
    regions = layout.compute(24, 80)

    _setup_layers(tui, regions)

    tui._draw_borders(regions)

    # Check that vertical borders were drawn on engine layers
    vert_count = 0
    for r in range(regions["console"].rows):
        ch = tui._layer_console.framebuffer.get(r, 0).char
        if ch == "│":
            vert_count += 1
    assert vert_count >= 2


def test_draw_borders_skips_empty_border():
    from domain.shell._internal.pane import Border, Pane, PaneLayout

    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    layout = PaneLayout()
    layout.panes.append(Pane("console", 1.0, border=Border("none")))
    tui._layout = layout
    regions = layout.compute(24, 80)
    # Required, not cosmetic: without this `_draw_borders` returns at the
    # `_engine is None` guard and the test passes without ever reaching the
    # `is_empty` branch it claims to cover.
    _setup_layers(tui, regions)
    scr = _FakeStdscr()
    with (
        patch.object(curses, "color_pair", side_effect=lambda n: n),
        patch.object(curses, "has_colors", return_value=True),
    ):
        tui._draw_borders(regions)
    # Cell (0,0) is where `─` would land if the border were drawn at all, so
    # blank is positive evidence the `continue` fired rather than an absence of
    # evidence that anything ran.
    assert tui._layer_console.framebuffer.get(0, 0).char.strip() == ""
    assert scr.addnstr_calls == []
    assert scr.addch_calls == []


def test_draw_borders_skips_invisible_pane():
    """An invisible pane's borders must not be drawn, even if regions say so.

    ``PaneLayout.compute`` filters on ``visible`` before it ever lays anything
    out (``pane.py:301``), so an invisible pane never appears in ``regions``.
    Through the natural path ``_draw_borders`` therefore skips on
    ``if not r`` -- *before* reaching ``if not pane.visible`` -- and the guard it
    is named for is unreachable while still looking covered.

    ``regions`` is supplied explicitly here so the test gets past ``if not r``
    and lands on the ``visible`` check itself. Note the resulting redundancy is
    real, not incidental: ``compute`` already hides the pane, so this branch is
    defence in depth against a caller that builds regions by hand.
    """
    from domain.shell._internal.pane import Border, Pane, PaneLayout

    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    layout = PaneLayout()
    layout.panes.append(Pane("console", 1.0, border=Border("all"), visible=False))
    tui._layout = layout
    regions = {"console": Rect(0, 0, 24, 80)}
    _setup_layers(tui, regions)  # now possible, because `regions` carries it
    scr = _FakeStdscr()
    with (
        patch.object(curses, "color_pair", side_effect=lambda n: n),
        patch.object(curses, "has_colors", return_value=True),
    ):
        tui._draw_borders(regions)
    # Positive evidence the layer exists to have been drawn *to* -- otherwise a
    # missing layer would satisfy "nothing drawn" trivially.
    assert tui._layer_console is not None, "regions did not reach _setup_layers"
    assert tui._layer_console.framebuffer.get(0, 0).char.strip() == ""
    assert scr.addnstr_calls == []


def test_draw_borders_custom_char():
    from domain.shell._internal.pane import Border, Pane, PaneLayout

    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    layout = PaneLayout()
    layout.panes.append(Pane("console", 0.5, border=Border("horizontal", ch="#")))
    layout.panes.append(Pane("output", 0.5, border=Border("none")))
    layout.panes.append(Pane("status", fixed=1))
    layout.panes.append(Pane("input", fixed=1))
    tui._layout = layout
    regions = layout.compute(24, 80)

    _setup_layers(tui, regions)

    tui._draw_borders(regions)

    # Check that custom char was used on engine layer
    top_ch = tui._layer_console.framebuffer.get(0, 0).char
    assert top_ch == "#"


def test_draw_borders_never_touches_curses():
    """Borders are drawn onto engine layers, and curses is not involved.

    Replaces ``test_draw_borders_handles_curses_error``, which handed
    ``_draw_borders`` a window whose ``addnstr`` raised and asserted the raise
    was swallowed. It also forgot to set up an engine, so ``_draw_borders``
    returned at the ``_engine is None`` guard before either the border or the
    raise could happen -- it passed without exercising anything.

    Both flaws are fixed at once: the engine is attached so the method really
    draws, and the window is one that raises on every write. Positive evidence
    matters here too -- the engine layer must actually carry the border, or a
    silent early return would satisfy the curses assertion trivially.
    """
    from domain.shell._internal.pane import Border, Pane, PaneLayout

    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    layout = PaneLayout()
    layout.panes.append(Pane("console", 1.0, border=Border("all")))
    tui._layout = layout
    regions = layout.compute(24, 80)
    _setup_layers(tui, regions)

    scr = _WritePoisonScr()
    with (
        patch.object(curses, "color_pair", side_effect=lambda n: n),
        patch.object(curses, "has_colors", return_value=True),
    ):
        tui._draw_borders(regions)

    assert tui._layer_console.framebuffer.get(0, 0).char in ("─", "│"), (
        "no border reached the engine layer, so nothing was proved about curses"
    )
    assert scr.write_calls == [], f"_draw_borders called curses {scr.write_calls}"


# ── _render_confirm ────────────────────────────────────────────────────


def test_render_confirm_yes_default():
    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    tui._confirm_message = "Delete file?"
    tui._confirm_default = True
    with patch.object(curses, "color_pair", side_effect=lambda n: n):
        tui._render_confirm()
    content = "".join(tui._input_surface.capture)
    assert "Delete file?" in content
    assert "Y/n" in content


def test_render_confirm_no_default():
    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    tui._confirm_message = "Proceed?"
    tui._confirm_default = False
    with patch.object(curses, "color_pair", side_effect=lambda n: n):
        tui._render_confirm()
    content = "".join(tui._input_surface.capture)
    assert "y/N" in content


# ── _render_ask ─────────────────────────────────────────────────────────


def test_render_ask_with_default():
    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    tui._ask_message = "Enter name"
    tui._ask_default = "anon"
    tui._ask_buf = list("anon")
    tui._ask_cursor = 4
    with patch.object(curses, "color_pair", side_effect=lambda n: n):
        tui._render_ask()
    content = "".join(tui._input_surface.capture)
    assert "Enter name" in content
    assert "[anon]" in content


def test_render_ask_without_default():
    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    tui._ask_message = "Type something"
    tui._ask_default = ""
    tui._ask_buf = list("hi")
    tui._ask_cursor = 2
    with patch.object(curses, "color_pair", side_effect=lambda n: n):
        tui._render_ask()
    content = "".join(tui._input_surface.capture)
    assert "Type something" in content
    assert "hi_" in content


# ── _render_select ──────────────────────────────────────────────────────


def test_render_select_basic():
    from domain.shell._internal.pane import Rect

    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    tui._select_title = "Pick one"
    tui._select_options = ["alpha", "bravo", "charlie"]
    tui._select_idx = 1
    tui._select_scroll = 0
    tui._select_filter = ""
    regions = {"output": Rect(0, 0, 10, 40)}
    with patch.object(curses, "color_pair", side_effect=lambda n: n):
        tui._render_select(regions)
    content = "".join(tui._output_surface.capture)
    assert "Pick one" in content
    assert "> bravo" in content
    assert "alpha" in content


def test_render_select_with_filter():
    from domain.shell._internal.pane import Rect

    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    tui._select_title = "Pick"
    tui._select_options = ["alpha", "bravo", "alpine"]
    tui._select_idx = 0
    tui._select_scroll = 0
    tui._select_filter = "alp"
    regions = {"output": Rect(0, 0, 10, 40)}
    with patch.object(curses, "color_pair", side_effect=lambda n: n):
        tui._render_select(regions)
    content = "".join(tui._output_surface.capture)
    assert "alpha" in content
    assert "alpine" in content
    assert "bravo" not in content


def test_render_select_overflow_shows_count():
    from domain.shell._internal.pane import Rect

    tui = _init_render_state(TuiRepl(_MainFakeRepl(), None))
    tui._select_title = "Big list"
    tui._select_options = [f"item-{i}" for i in range(20)]
    tui._select_idx = 5
    tui._select_scroll = 0
    tui._select_filter = ""
    regions = {"output": Rect(0, 0, 6, 40)}
    with patch.object(curses, "color_pair", side_effect=lambda n: n):
        tui._render_select(regions)
    content = "".join(tui._output_surface.capture)
    assert "20 items" in content
    assert "6/20" in content


# ── prompt_select / prompt_confirm / prompt_ask ────────────────────────
#
# These pin the *blocking* half of a prompt: the command thread parks in
# `prompt_*` until an answer arrives. `_X_event` is the open latch `_main`
# dispatches on; `_X_done` is what releases the waiter, so a helper below that
# stands in for "the user pressed a key" sets `_done` (having already set the
# result) — setting the latch would only say "still open".


def test_prompt_select_empty_returns_empty():
    tui = TuiRepl(_MainFakeRepl(), None)
    assert tui.prompt_select("Pick", []) == ""


def test_prompt_select_returns_result():
    import threading

    tui = TuiRepl(_MainFakeRepl(), None)

    def set_result():
        time.sleep(0.05)
        tui._select_result = "bravo"
        tui._select_done.set()

    t = threading.Thread(target=set_result)
    t.start()
    result = tui.prompt_select("Pick", ["alpha", "bravo", "charlie"])
    t.join(timeout=1)
    assert result == "bravo"
    assert not tui._select_event.is_set(), "the overlay must be closed on return"


def test_prompt_confirm_returns_result():
    import threading

    tui = TuiRepl(_MainFakeRepl(), None)

    def set_result():
        time.sleep(0.05)
        tui._confirm_result = True
        tui._confirm_done.set()

    t = threading.Thread(target=set_result)
    t.start()
    result = tui.prompt_confirm("OK?")
    t.join(timeout=1)
    assert result is True
    assert not tui._confirm_event.is_set(), "the overlay must be closed on return"


def test_prompt_confirm_timeout_returns_default():
    import threading

    tui = TuiRepl(_MainFakeRepl(), None)

    def release():
        time.sleep(0.05)
        tui._confirm_done.set()

    threading.Thread(target=release, daemon=True).start()
    result = tui.prompt_confirm("OK?", default=False)
    assert result is False


def test_prompt_ask_returns_result():
    import threading

    tui = TuiRepl(_MainFakeRepl(), None)

    def set_result():
        time.sleep(0.05)
        tui._ask_result = "hello"
        tui._ask_done.set()

    t = threading.Thread(target=set_result)
    t.start()
    result = tui.prompt_ask("Enter text")
    t.join(timeout=1)
    assert result == "hello"
    assert not tui._ask_event.is_set(), "the overlay must be closed on return"


def test_prompt_ask_timeout_returns_default():
    import threading

    tui = TuiRepl(_MainFakeRepl(), None)

    def release():
        time.sleep(0.05)
        tui._ask_done.set()

    threading.Thread(target=release, daemon=True).start()
    result = tui.prompt_ask("Enter text", default="fallback")
    assert result == "fallback"


# ── _render_all ─────────────────────────────────────────────────────────


def test_render_all_draws_borders_and_content():
    from domain.shell._internal.log_buffer import LogBuffer, LogEntry
    from domain.shell._internal.pane import Border, Pane, PaneLayout

    tui = _init_render_state(TuiRepl(_MainFakeRepl(), LogBuffer()))
    tui._log_surface._buffer.append(LogEntry(time.time(), "INFO", "test", "log line"))
    layout = PaneLayout()
    layout.panes.append(Pane("console", 0.3, border=Border("all")))
    layout.panes.append(Pane("output", 0.3, border=Border("all")))
    layout.panes.append(Pane("status", fixed=1))
    layout.panes.append(Pane("input", fixed=1))
    tui._layout = layout
    tui._output_surface.write("output line")
    regions = layout.compute(24, 80)

    _setup_layers(tui, regions)

    # Mock engine.render to verify it's called
    render_called = [False]

    def mock_render():
        render_called[0] = True

    tui._engine.render = mock_render

    tui._render_all(regions)
    assert render_called[0]
