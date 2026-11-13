"""Tests for the TUI prompt-mode key handlers — no curses screen required.

`TuiRepl._main` used to hold three `if/elif` chains over the raw `getch()`
integer, one per overlay. They are now `_handle_select_key`,
`_handle_confirm_key` and `_handle_ask_key`, each taking the single `Key` the
loop read.

That split is what makes this file possible. Before it, the only way to reach a
keypress decision was to drive a real curses screen through a pty; after it, the
decision functions are ordinary methods over plain state, so the whole code
space can be swept here. The tests below are the ones that matter most because
they are the ones a pty test cannot easily write:

- **An unhandled key must be inert.** The handlers cancel on `key.kind`, so
  anything the decoder reports as KEY_ESC would dismiss the prompt — and for
  confirm, answer the dialog with the opposite of its default. That is why the
  vocabulary separates KEY_UNKNOWN (a key read and not modelled: F1, Shift+Left,
  an unparseable sequence) from KEY_ESC (the user pressed Esc). The codes that
  must stay inert are pinned in `NOT_CANCEL`.
- **A chord must read as a chord.** Ctrl+P and Ctrl+N used to arrive as
  KEY_ESC with the raw code needed alongside them, which is why this file fed
  handlers two arguments. They are `KEY_CTRL` with the letter in `char` now.

The exhaustive equivalence check against the pre-extraction chains lives in
.splitwork/probe_prompt_modes.py; it slices the old code out of git and compares
every code against several prompt states. This file is the standing subset that
runs as part of the suite.
"""

from __future__ import annotations

import curses
import threading
import time

import pytest

from domain.shell._internal.tui_repl import TuiRepl
from domain.shell._internal.tui_session import (
    KEY_ALT,
    KEY_ENTER,
    KEY_LEFT,
    KEY_RIGHT,
    Key,
    from_curses,
)

OPTIONS = [f"opt-{i:02d}" for i in range(20)]


class _Rect:
    """The only thing the handlers read out of the layout."""

    rows = 10
    cols = 40


REGIONS = {"output": _Rect()}

# `_render_all` also reads the status and input regions (it ends by compositing
# and positioning the caret there), so a test that drives it needs those too.
FULL_REGIONS = {"output": _Rect(), "status": _Rect(), "input": _Rect()}


@pytest.fixture
def repl():
    """A TuiRepl carrying only the state the handlers touch.

    `object.__new__` skips `__init__`, which is deliberate: the handlers are
    pure state transitions over a handful of fields, and building a whole
    TuiRepl to test them would drag in a curses surface and a REPL. The render
    methods are stubbed because the chain tails call the renderer once more
    after updating state, and rendering is not what is under test.
    """
    t = object.__new__(TuiRepl)
    t._select_event = threading.Event()
    t._confirm_event = threading.Event()
    t._ask_event = threading.Event()
    # The *answer* signal, deliberately not the latch above. `_main` routes keys
    # on `*_event` (is it open?) while `prompt_*` blocks on `*_done` (has it been
    # answered?) — conflating those two questions into one event is the bug
    # `TestPromptOverlayLatch` pins shut.
    t._select_done = threading.Event()
    t._confirm_done = threading.Event()
    t._ask_done = threading.Event()
    t._select_options = list(OPTIONS)
    t._select_title = "pick one"
    t._select_idx = 0
    t._select_scroll = 0
    t._select_filter = ""
    t._select_result = None
    t._confirm_message = "go?"
    t._confirm_default = False
    t._confirm_result = None
    t._ask_message = "name"
    t._ask_default = "dflt"
    t._ask_result = None
    t._ask_buf = []
    t._ask_cursor = 0
    t._overlay_cursor_hidden = False
    t._engine = None  # `_paint_overlay` composites through it when present
    t._render_select = lambda regions: None
    t._render_confirm = lambda: None
    t._render_ask = lambda: None
    return t


def press(repl: TuiRepl, ch: int) -> bool:
    """Apply one keypress the way `_main` does: normalise, then handle.

    `ch` stays the raw code because that is what this file has always fed in —
    the terminal's input, not its interpretation. The handlers take only the
    `Key` now, so `from_curses` is the whole of the bridge, and `probe_session`
    is what pins it to agree with the `decode_key` path `_main` actually uses.
    """
    return repl._handle_select_key(from_curses(ch, curses), REGIONS)


def press_confirm(repl: TuiRepl, ch: int) -> bool:
    return repl._handle_confirm_key(from_curses(ch, curses))


def press_ask(repl: TuiRepl, ch: int) -> bool:
    return repl._handle_ask_key(from_curses(ch, curses))


def press_ask_key(repl: TuiRepl, key: Key) -> bool:
    """Apply an already-decoded press to the ask overlay.

    `press_ask` takes a raw code because that is what this file has always fed
    in, with `from_curses` as the whole bridge. A chorded arrow has no `curses`
    code on this build — Ctrl+Left is the arrow *with a flag*, not a constant
    named after the chord — so the decoder's own output is the only honest
    input for those cases. Feeding a built `Key` rather than a stand-in is what
    keeps these tests about the handler instead of about the decode.
    """
    return repl._handle_ask_key(key)


def visible(repl: TuiRepl) -> range:
    """The slice of the filtered list the overlay is currently showing."""
    filtered = [o for o in repl._select_options if repl._select_filter.lower() in o.lower()]
    if not filtered:
        filtered = list(repl._select_options)
    max_show = min(len(filtered), _Rect.rows - 2)
    return range(repl._select_scroll, repl._select_scroll + max_show)


def assert_selection_visible(repl: TuiRepl) -> None:
    """The highlighted option must be inside the window.

    Asserting this instead of hand-computed `_select_idx` / `_select_scroll`
    pairs. The numbers depend on `max_show`, which depends on the layout and
    on how many options survive the filter, so pinning them means recomputing
    them every time the layout changes — and getting one wrong produces a test
    failure that looks like a handler bug. The invariant is the actual
    requirement, and it holds for every state the handler can reach.
    """
    assert repl._select_scroll <= repl._select_idx < visible(repl).stop


# Codes that must not act on an overlay. Two different reasons sit in this one
# list, and the split matters when a test here fails.
#
# The control bytes are named now. 0 and 28..31 report KEY_CTRL with @ \ ] ^ _
# rather than falling through, and 2..26 report their letters, so none of them
# can reach the cancel test by accident. They never used to be at risk from
# that: cancel was `ch in (27, 7, 3)`, so a code that matched no branch did
# nothing. The risk arrived with the handlers, which now cancel on `key.kind`.
#
# The curses pseudo-keys are the same question wearing a different hat.
# KEY_F1 (265) has no kind in the vocabulary; KEY_RESIZE (410) has one of its
# own. What a code must *not* become is a key the widget acts on by default —
# for confirm, an answer.
#
# 343 is deliberately absent. It is KEY_ENTER, and Enter answering is the
# correct behaviour, so a test that required `_confirm_result is None` for it
# was pinning the accident that `from_curses` used to fold KEY_ENTER into Esc
# and Enter therefore did nothing. It has a test of its own below.
NOT_CANCEL = [0, 2, 4, 6, 9, 11, 12, 15, 17, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31]
NOT_CANCEL += [265, 410]  # KEY_F1, KEY_RESIZE from real curses

# Byte 23 is Ctrl+W, which the ask overlay now *binds* — it erases the word
# before the caret — so there it is consumed rather than passed over. It stays
# inert in select and confirm, which is why 23 is still in `NOT_CANCEL` for
# those and only the ask case drops it. Being handled is not the same as being
# a cancel: the event must still be set, which is what the assertion below
# checks for everything else and `TestAskWordMotion` checks for 23.
ASK_NOT_CONSUMED = [ch for ch in NOT_CANCEL if ch != 23]


class TestUnhandledCodesAreNotCancel:
    """The regression this refactor could have introduced."""

    @pytest.mark.parametrize("ch", NOT_CANCEL)
    def test_select_stays_open(self, repl, ch):
        repl._select_event.set()
        assert press(repl, ch) is False
        assert repl._select_event.is_set()
        assert repl._select_result is None

    @pytest.mark.parametrize("ch", NOT_CANCEL)
    def test_confirm_does_not_flip_the_default(self, repl, ch):
        repl._confirm_event.set()
        assert press_confirm(repl, ch) is False
        assert repl._confirm_event.is_set()
        assert repl._confirm_result is None

    @pytest.mark.parametrize("ch", ASK_NOT_CONSUMED)
    def test_ask_stays_open(self, repl, ch):
        repl._ask_event.set()
        assert press_ask(repl, ch) is False
        assert repl._ask_event.is_set()


class TestAskWordMotion:
    """The four chords this change binds into the ask overlay.

    None of them may cancel, which is the same invariant
    `TestUnhandledCodesAreNotCancel` holds for everything *un*bound — Ctrl+W
    sits in both stories, so it is asserted here rather than dropped from the
    coverage there.
    """

    def _loaded(self, repl, text="one two", cursor=None):
        repl._ask_event.set()
        repl._ask_buf[:] = list(text)
        repl._ask_cursor = len(repl._ask_buf) if cursor is None else cursor

    def _assert_open(self, repl):
        assert repl._ask_event.is_set(), "a word chord cancelled the overlay"
        assert repl._ask_result is None, "a word chord answered the prompt"

    def test_ctrl_w_erases_the_word_before_the_caret(self, repl):
        self._loaded(repl, cursor=7)
        assert press_ask(repl, 23) is True
        assert "".join(repl._ask_buf) == "one "
        assert repl._ask_cursor == 4
        self._assert_open(repl)

    def test_alt_d_erases_the_word_after_the_caret(self, repl):
        self._loaded(repl, cursor=0)
        assert press_ask_key(repl, Key(KEY_ALT, "d")) is True
        assert "".join(repl._ask_buf) == " two"
        assert repl._ask_cursor == 0  # kill-word leaves the caret put
        self._assert_open(repl)

    def test_ctrl_arrow_jumps_a_word_in_each_direction(self, repl):
        self._loaded(repl, cursor=0)
        assert press_ask_key(repl, Key(KEY_RIGHT, ctrl=True)) is True
        assert repl._ask_cursor == 3
        assert press_ask_key(repl, Key(KEY_LEFT, ctrl=True)) is True
        assert repl._ask_cursor == 0
        self._assert_open(repl)

    def test_a_plain_arrow_moves_one_and_a_chorded_one_moves_a_word(self, repl):
        """The whole reason `matches` names the modifier.

        Both presses report `kind == "left"`; if the plain branch could see the
        chord, this assertion fails on the second line no matter which branch
        sits first in the chain.
        """
        self._loaded(repl, cursor=6)
        press_ask_key(repl, Key(KEY_LEFT))
        assert repl._ask_cursor == 5, "plain Left should step one character"
        self._loaded(repl, cursor=6)
        press_ask_key(repl, Key(KEY_LEFT, ctrl=True))
        assert repl._ask_cursor == 4, "Ctrl+Left should jump to the word start"
        self._assert_open(repl)

    def test_a_plain_w_still_types_instead_of_erasing(self, repl):
        """`matches(KEY_CTRL, char=...)` must not catch the plain letter."""
        self._loaded(repl, cursor=7)
        assert press_ask(repl, ord("w")) is True
        assert "".join(repl._ask_buf) == "one twow"
        assert repl._ask_cursor == 8  # typed at the end, so the caret advanced
        self._assert_open(repl)


class TestEnterIsEnter:
    """KEY_ENTER must answer, not be ignored and not cancel.

    `curses.KEY_ENTER` (343) was in `NOT_CANCEL` because `from_curses` folded
    it into KEY_ESC, so Enter reached a widget that matched neither the cancel
    test (which read the raw code, and 343 was not in it) nor the Enter branch.
    It did nothing, and the list above recorded "nothing" as the desired state.

    It is claimed now, because `decode_key` reads terminfo's `kent` (`ESC O M`)
    as Enter and the two backends have to agree for one physical keystroke.
    Enter then answers, which is the behaviour, so these tests replace the
    accident rather than tolerate it.
    """

    def test_select_accepts_the_highlighted_option(self, repl):
        repl._select_event.set()
        repl._select_idx = 1
        assert press(repl, 343) is True
        assert not repl._select_event.is_set()
        assert repl._select_result == repl._select_options[1]

    def test_confirm_uses_the_default_rather_than_flipping_it(self, repl):
        repl._confirm_event.set()
        repl._confirm_default = True
        assert press_confirm(repl, 343) is True
        assert not repl._confirm_event.is_set()
        assert repl._confirm_result is True

    def test_ask_accepts_its_buffer(self, repl):
        repl._ask_event.set()
        repl._ask_buf[:] = list("hello")
        assert press_ask(repl, 343) is True
        assert not repl._ask_event.is_set()
        assert repl._ask_result == "hello"


class TestReadlineChords:
    """Ctrl+P / Ctrl+N are `KEY_CTRL` with the letter in `char`.

    They used to normalise to KEY_ESC alongside everything the vocabulary had
    no name for, which is why the handlers ordered them before the cancel test
    — two presses sharing one kind, so position was the only thing separating
    them. They have kinds of their own now, so the ordering no longer carries
    that weight and the chords read as chords.
    """

    def test_ctrl_p_moves_up_and_does_not_cancel(self, repl):
        repl._select_event.set()
        repl._select_idx = 5
        assert press(repl, 16) is True
        assert repl._select_idx == 4
        assert repl._select_event.is_set()

    def test_ctrl_n_moves_down_and_does_not_cancel(self, repl):
        repl._select_event.set()
        repl._select_idx = 5
        assert press(repl, 14) is True
        assert repl._select_idx == 6
        assert repl._select_event.is_set()

    def test_up_key_still_moves_up(self, repl):
        repl._select_event.set()
        repl._select_idx = 5
        assert press(repl, curses.KEY_UP) is True
        assert repl._select_idx == 4


class TestCancelChords:
    @pytest.mark.parametrize("ch", [27, 7, 3])
    def test_select_cancel_sets_no_result(self, repl, ch):
        repl._select_event.set()
        assert press(repl, ch) is True
        assert not repl._select_event.is_set()
        assert repl._select_result is None

    @pytest.mark.parametrize("ch", [27, 7, 3])
    def test_confirm_cancel_flips_the_default(self, repl, ch):
        repl._confirm_event.set()
        repl._confirm_default = False
        press_confirm(repl, ch)
        assert not repl._confirm_event.is_set()
        assert repl._confirm_result is True

    def test_confirm_cancel_flips_a_true_default(self, repl):
        repl._confirm_event.set()
        repl._confirm_default = True
        press_confirm(repl, 27)
        assert repl._confirm_result is False


class TestSelect:
    def test_enter_accepts_the_highlighted_option(self, repl):
        repl._select_event.set()
        repl._select_idx = 7
        assert press(repl, 10) is True
        assert repl._select_result == "opt-07"
        assert not repl._select_event.is_set()

    def test_enter_with_nothing_highlighted_accepts_nothing(self, repl):
        repl._select_event.set()
        repl._select_idx = 99
        press(repl, 10)
        assert repl._select_result is None

    def test_page_up(self, repl):
        repl._select_event.set()
        repl._select_idx = 10
        repl._select_scroll = 8
        assert press(repl, curses.KEY_PPAGE) is True
        # A page is `max_show` = rows - 2 = 8 rows, so from 10 that is row 2.
        assert repl._select_idx == 2
        assert_selection_visible(repl)

    @pytest.mark.parametrize("ch", [curses.KEY_NPAGE])
    def test_page_down(self, repl, ch):
        repl._select_event.set()
        repl._select_idx = 0
        assert press(repl, ch) is True
        assert repl._select_idx == 8
        assert_selection_visible(repl)

    def test_page_up_at_the_top_clamps_to_zero(self, repl):
        repl._select_event.set()
        repl._select_idx = 1
        repl._select_scroll = 0
        press(repl, curses.KEY_PPAGE)
        assert repl._select_idx == 0
        assert repl._select_scroll == 0

    @pytest.mark.parametrize("ch", [curses.KEY_BACKSPACE, 127, 8])
    def test_all_three_backspace_codes_truncate_the_filter(self, repl, ch):
        repl._select_event.set()
        repl._select_filter = "abc"
        assert press(repl, ch) is True
        assert repl._select_filter == "ab"

    def test_printable_extends_the_filter_and_resets_the_cursor(self, repl):
        repl._select_event.set()
        repl._select_filter = "ab"
        repl._select_idx = 9
        repl._select_scroll = 4
        assert press(repl, ord("c")) is True
        assert repl._select_filter == "abc"
        assert repl._select_idx == 0
        assert repl._select_scroll == 0

    def test_latin1_does_not_extend_the_filter(self, repl):
        """The vocabulary admits Latin-1 as KEY_CHAR; this widget never did."""
        repl._select_event.set()
        assert press(repl, 233) is False
        assert repl._select_filter == ""


class TestConfirm:
    @pytest.mark.parametrize("ch", [ord("y"), ord("Y")])
    def test_yes(self, repl, ch):
        repl._confirm_event.set()
        assert press_confirm(repl, ch) is True
        assert repl._confirm_result is True
        assert not repl._confirm_event.is_set()

    @pytest.mark.parametrize("ch", [ord("n"), ord("N")])
    def test_no(self, repl, ch):
        repl._confirm_event.set()
        assert press_confirm(repl, ch) is True
        assert repl._confirm_result is False

    def test_enter_takes_the_default(self, repl):
        repl._confirm_event.set()
        repl._confirm_default = True
        press_confirm(repl, 10)
        assert repl._confirm_result is True

    @pytest.mark.parametrize("ch", [ord("z"), ord("q"), 32, ord(" ")])
    def test_other_printables_are_ignored(self, repl, ch):
        repl._confirm_event.set()
        assert press_confirm(repl, ch) is False
        assert repl._confirm_event.is_set()


class TestAsk:
    def test_enter_accepts_the_buffer(self, repl):
        repl._ask_event.set()
        repl._ask_buf = list("abc")
        assert press_ask(repl, 10) is True
        assert repl._ask_result == "abc"

    @pytest.mark.parametrize("ch", [curses.KEY_BACKSPACE, 127, 8])
    def test_all_three_backspace_codes_delete_a_character(self, repl, ch):
        repl._ask_event.set()
        repl._ask_buf = list("abc")
        repl._ask_cursor = 3  # caret at the end, so the cursor move is visible
        assert press_ask(repl, ch) is True
        assert repl._ask_buf == ["a", "b"]
        assert repl._ask_cursor == 2

    def test_backspace_on_an_empty_buffer_is_harmless(self, repl):
        repl._ask_event.set()
        assert press_ask(repl, 8) is True
        assert repl._ask_buf == []
        assert repl._ask_cursor == 0

    def test_left_and_right_move_the_caret_within_bounds(self, repl):
        repl._ask_event.set()
        repl._ask_buf = list("abc")
        repl._ask_cursor = 3
        press_ask(repl, curses.KEY_LEFT)
        assert repl._ask_cursor == 2
        press_ask(repl, curses.KEY_LEFT)
        press_ask(repl, curses.KEY_LEFT)
        press_ask(repl, curses.KEY_LEFT)
        assert repl._ask_cursor == 0, "caret must not run off the left"
        press_ask(repl, curses.KEY_RIGHT)
        press_ask(repl, curses.KEY_RIGHT)
        press_ask(repl, curses.KEY_RIGHT)
        press_ask(repl, curses.KEY_RIGHT)
        assert repl._ask_cursor == 3, "caret must not run off the right"

    def test_printable_inserts_at_the_caret_not_the_end(self, repl):
        repl._ask_event.set()
        repl._ask_buf = list("ac")
        repl._ask_cursor = 1
        assert press_ask(repl, ord("b")) is True
        assert repl._ask_buf == ["a", "b", "c"]
        assert repl._ask_cursor == 2

    def test_latin1_does_not_insert(self, repl):
        repl._ask_event.set()
        assert press_ask(repl, 233) is False
        assert repl._ask_buf == []


class TestScroll:
    """Scrolling is the one handler that reads the layout, so it gets its own.

    Note that `scroll` only ever *catches up* with the cursor, never leads it:
    the cursor can only leave the top of the window by being at the top and
    moving up, so scroll-back happens exactly when `idx == scroll` beforehand.
    """

    def test_moving_down_past_the_window_scrolls(self, repl):
        repl._select_event.set()
        repl._select_scroll = 0
        for _ in range(9):  # more rows than the 8-row window shows
            assert_selection_visible(repl)
            press(repl, curses.KEY_DOWN)
        assert repl._select_idx == 9
        assert_selection_visible(repl)
        assert repl._select_scroll > 0, "the window must have followed the cursor"

    def test_moving_up_from_the_top_of_the_window_scrolls_back(self, repl):
        repl._select_event.set()
        repl._select_idx = repl._select_scroll = 2
        assert press(repl, curses.KEY_UP) is True
        assert repl._select_idx == 1
        assert repl._select_scroll == 1
        assert_selection_visible(repl)

    def test_up_stays_put_when_the_cursor_is_mid_window(self, repl):
        repl._select_event.set()
        repl._select_idx = 5
        repl._select_scroll = 0
        assert press(repl, curses.KEY_UP) is True
        assert repl._select_idx == 4
        assert repl._select_scroll == 0, "no scroll needed while inside the window"

    def test_up_at_the_top_stops_at_zero(self, repl):
        repl._select_event.set()
        repl._select_idx = 0
        repl._select_scroll = 0
        press(repl, curses.KEY_UP)
        assert (repl._select_idx, repl._select_scroll) == (0, 0)

    def test_down_at_the_end_stops_at_the_last_option(self, repl):
        repl._select_event.set()
        repl._select_idx = 19
        press(repl, curses.KEY_DOWN)
        assert repl._select_idx == 19


class TestPromptOverlayLatch:
    """One event cannot mean both "the overlay is open" and "the answer is in".

    `prompt_*` cleared its event and then waited on it, while `_main` routed
    keys to an overlay only when that same event was set — and no production
    code ever set it. A command thread asking a question therefore blocked for
    300 s with nothing painted, then silently returned the default. The real
    casualty is `souls_cmd.out.select()`, which switches soul through exactly
    this path.

    So the two questions get one signal each: `_main` dispatches and paints on
    the latch (`*_event`), `prompt_*` blocks on the answer (`*_done`).
    """

    @staticmethod
    def _wait(pred, timeout: float = 1.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if pred():
                return True
            time.sleep(0.01)
        return bool(pred())

    @pytest.mark.parametrize(
        ("method", "args", "event_attr"),
        [
            ("prompt_select", ("pick", ["alpha", "bravo"]), "_select_event"),
            ("prompt_confirm", ("go?", False), "_confirm_event"),
            ("prompt_ask", ("name", ""), "_ask_event"),
        ],
    )
    def test_a_prompt_opens_its_overlay_while_it_waits(self, repl, method, args, event_attr):
        done_attr = event_attr.replace("_event", "_done")
        thread = threading.Thread(target=lambda: getattr(repl, method)(*args), daemon=True)
        thread.start()

        assert self._wait(lambda: getattr(repl, event_attr).is_set()), (
            f"{method} never opened {event_attr}. `_main` only routes a keypress "
            "to a prompt whose event is set, so the question would go unanswered."
        )
        # The waiter must be released by the *answer*, not by the latch.
        getattr(repl, done_attr).set()
        thread.join(timeout=2)
        assert not thread.is_alive(), f"{done_attr} did not release {method}"
        assert not getattr(repl, event_attr).is_set(), "the overlay must close on answer"

    def test_a_keypress_answers_the_ask_overlay_and_closes_it(self, repl):
        """The full round trip: key -> buffer -> waiter released -> latch down."""
        result: dict = {}
        thread = threading.Thread(
            target=lambda: result.setdefault("value", repl.prompt_ask("name", "dflt")),
            daemon=True,
        )
        thread.start()

        assert self._wait(lambda: repl._ask_event.is_set()), "ask overlay never opened"
        assert repl._handle_ask_key(Key(KEY_ENTER)) is True, "Enter must be consumed"
        thread.join(timeout=2)

        assert not thread.is_alive(), "Enter did not release prompt_ask"
        assert result["value"] == "dflt", "what the key built is what comes back"
        assert not repl._ask_event.is_set(), "_main would keep routing keys to a closed overlay"

    def test_the_poll_tick_paints_an_open_overlay(self, repl):
        """A prompt is opened by a command thread while `_main` is inside
        `read_key()`. The 100 ms tick repaints every pane but never the
        overlay, so without one paint there the box would not appear until the
        user happened to press a key.
        """
        assert hasattr(repl, "_paint_overlay"), (
            "the tick needs one place to paint an open overlay after the panes"
        )
        painted: list[str] = []
        repl._render_select = lambda regions: painted.append("select")
        repl._render_confirm = lambda: painted.append("confirm")
        repl._render_ask = lambda: painted.append("ask")

        repl._paint_overlay(REGIONS)
        assert painted == [], "nothing open: the tick must not paint an overlay"

        repl._ask_event.set()
        repl._paint_overlay(REGIONS)
        assert painted == ["ask"]

        repl._select_event.set()
        repl._paint_overlay(REGIONS)
        assert painted == ["ask", "select"], (
            "one call paints exactly one overlay, and select outranks ask — the "
            "same order `_main` dispatches them in"
        )

    def test_painting_an_overlay_composites_it(self, repl):
        """`_blit` only writes layer buffers; `_render_all` is what flushes.

        So an overlay painted *after* the pane repaint would sit in a buffer
        that the next pane repaint clears before anybody composites it — the
        box would be drawn, handled, and never seen. This is why the ask prompt
        was invisible over a pty while the keys behind it worked.
        """

        class _Engine:
            def __init__(self):
                self.renders = 0

            def render(self):
                self.renders += 1

            def show_cursor(self, visible=True):
                pass

        engine = _Engine()
        repl._engine = engine

        repl._ask_event.set()
        repl._paint_overlay(REGIONS)
        assert engine.renders == 1, "painting without compositing draws nothing"

        repl._ask_event.clear()
        repl._paint_overlay(REGIONS)
        assert engine.renders == 1, "an idle tick has nothing to composite"

    def test_the_hardware_cursor_is_parked_while_an_overlay_is_drawn(self, repl):
        """Each overlay draws its own caret (`ask` renders a literal `_`), so
        the terminal cursor left at the λ caret would blink as a second one
        inside the box — and must come back when the overlay goes.
        """

        class _Engine:
            def __init__(self):
                self.visible = [True]

            def render(self):
                pass

            def show_cursor(self, visible=True):
                self.visible.append(visible)

        engine = _Engine()
        repl._engine = engine

        repl._paint_overlay(REGIONS)
        assert engine.visible == [True], "idle: leave the cursor alone"

        repl._ask_event.set()
        repl._paint_overlay(REGIONS)
        assert engine.visible[-1] is False, "a drawn overlay must park the cursor"

        repl._ask_event.clear()
        repl._paint_overlay(REGIONS)
        assert engine.visible[-1] is True, "the cursor must come back with the λ prompt"

    def test_the_pane_repaint_draws_the_overlay_in_the_same_pass(self, repl):
        """`_render_all` is the only thing that composites, so the overlay has
        to be drawn *inside* that pass — before the flush, not after it.

        Drawn after, every tick emits the λ line and then the box: the screen
        is right at the end of the tick and wrong at every instant between the
        two writes, which is how a snapshot taken over a pty could read a frame
        with no overlay at all while the keys behind it worked perfectly.
        """

        class _Engine:
            def __init__(self):
                self.renders = 0

            def render(self):
                self.renders += 1

            def show_cursor(self, visible=True):
                pass

        drawn: list[str] = []
        engine = _Engine()
        repl._engine = engine
        repl._layout = type("_Layout", (), {"panes": []})()
        repl._render_status = lambda cols: None
        repl._render_input = lambda cols: 0
        repl._render_ask = lambda: drawn.append("ask")

        repl._ask_event.set()
        repl._render_all(FULL_REGIONS)

        assert drawn == ["ask"], "_render_all bypassed the open overlay"
        assert engine.renders == 1, "the overlay belongs in the frame that flushes"


class TestAskOverlayCaret:
    """Where the ask overlay draws its caret.

    It drew `prompt + buffer + "_"` — the caret pinned to the end of the text
    whatever `_ask_cursor` said. The handler moved the cursor correctly and the
    field looked inert, which is worse than no word motion at all: the feature
    is there, it just never shows. Over a pty, Ctrl+Left then typing a character
    read as "nothing happened".
    """

    class _Surface:
        """The two calls `_render_ask` makes on the input surface."""

        def __init__(self):
            self.text = ""

        def clear(self):
            self.text = ""

        def write(self, text):
            self.text += text

        def render(self, rows, col):
            return []

    @pytest.fixture
    def drawn(self, repl):
        surf = self._Surface()
        repl._input_surface = surf
        repl._blit = lambda *args, **kwargs: None
        # The shared fixture stubs the renderers so handler tests never draw;
        # this class is *about* the drawing, so bind the real one back on.
        repl._render_ask = lambda: TuiRepl._render_ask(repl)
        repl._ask_message = "name"
        repl._ask_default = ""
        repl._ask_buf = list("one two")
        return repl, surf

    def test_the_caret_sits_where_the_cursor_is(self, drawn):
        repl, surf = drawn
        repl._ask_cursor = 4  # between "one " and "two"
        repl._render_ask()
        assert surf.text == "  name: one _two"

    def test_at_the_end_of_the_buffer_it_renders_as_it_always_has(self, drawn):
        repl, surf = drawn
        repl._ask_cursor = len(repl._ask_buf)
        repl._render_ask()
        assert surf.text == "  name: one two_"

    def test_at_the_start_it_sits_right_after_the_prompt(self, drawn):
        repl, surf = drawn
        repl._ask_cursor = 0
        repl._render_ask()
        assert surf.text == "  name: _one two"
