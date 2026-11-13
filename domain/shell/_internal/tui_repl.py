"""
TuiRepl — split-panel shell with three fixed regions using GraphicsEngine.

Layout:
  ┌─ Console Panel (infra logs) ─────────────────── top 30% ─┐
  │ 16:56:37 INF [startup] Task queue initialized            │
  │ 16:56:38 WRN [runtime] Orphan process 21415 killed       │
  ├─ Shell Output (command results) ─────────────── remainder ┤
  │ $ models                                                  │
  │ gpt2   124M   loaded                                      │
  ├───────────────────────────────────────────────────────────┤
  │ [OUTPUT] LIVE  ai "2+2"                    120x24 (fixed) │
  │ _                                             (fixed line) │
  └───────────────────────────────────────────────────────────┘

Uses the GraphicsEngine for rendering (compositing framebuffer model) and
`TtySession` for input (raw descriptor, decoded by `decode_key`).  Command
dispatch runs on a background thread so the UI stays responsive.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

try:
    import ctypes

    _SET_ASYNC_EXC = ctypes.pythonapi.PyThreadState_SetAsyncExc
except (ImportError, AttributeError):
    _SET_ASYNC_EXC = None

from .graphics import (
    Attr,
    Color,
    GraphicsEngine,
    Layer,
    Pattern,
    TabManager,
)
from .io import TuiIo
from .pane import Border, Pane, PaneLayout, Rect
from .surface import (
    STYLE_CRITICAL,
    STYLE_DEBUG,
    STYLE_ERROR,
    STYLE_INFO,
    STYLE_WARN,
    LogSurface,
    RenderLine,
    TextSurface,
)
from .textedit import delete_back, delete_forward, word_backward, word_forward
from .tui_session import (
    KEY_ALT,
    KEY_BACKSPACE,
    KEY_CHAR,
    KEY_CTRL,
    KEY_CTRL_C,
    KEY_DELETE,
    KEY_DOWN,
    KEY_END,
    KEY_ENTER,
    KEY_ESC,
    KEY_HOME,
    KEY_LEFT,
    KEY_PAGE_DOWN,
    KEY_PAGE_UP,
    KEY_RESIZE,
    KEY_RIGHT,
    KEY_TAB,
    KEY_UP,
    Key,
    TtySession,
)

if TYPE_CHECKING:
    from .log_buffer import LogBuffer
    from .repl import ShellREPL

# ── Chords and printable text ─────────────────────────────────────────────
#
# This block used to record a gap: Ctrl+P, Ctrl+N and Ctrl+G could not be
# expressed in the shared vocabulary, so the loop read raw `getch()` codes for
# exactly those three and the shared `Key` everywhere else. Two vocabularies for
# one stream — the split `tui_session` exists to remove, and the reason the
# port could not be attempted before the vocabulary was extended.
#
# They are named now, and two of the mappings are worth stating because neither
# is obvious:
#
#   Ctrl+G -> KEY_ESC, not ctrl/"g". Not an accident: the byte decoder reports
#            byte 7 as Esc precisely so it lands in *both* cancel sets — this
#            loop's and all eight `interactive_*` prompts, whose cancel is
#            spelled `(_KEY_CTRL_C, _KEY_ESC)`. That reproduces
#            `_CANCEL_CHORDS = (27, 7, 3)` exactly, with no consumer left to
#            update.
#   Ctrl+P -> ctrl/"p", not KEY_UP, despite doing the same job here. Reporting
#            it as `up` would make a chord indistinguishable from an arrow key
#            for every other consumer, and only this loop treats them as
#            equivalent.
_CANCEL = (KEY_ESC, KEY_CTRL_C)  # Esc, Ctrl+G, Ctrl+C — see above
_CTRL_UP_CHAR = "p"  # Ctrl+P — previous line, same as KEY_UP
_CTRL_DOWN_CHAR = "n"  # Ctrl+N — next line, same as KEY_DOWN


def _chord(key: Key, letter: str) -> bool:
    """True for Ctrl+<letter>, the one place that decides what a chord looks like."""
    return key.kind == KEY_CTRL and key.char == letter


# The prompts filter on ASCII printable only. The decoder also reports the
# Latin-1 band and any assembled UTF-8 as KEY_CHAR, and none of these widgets
# ever accepted those — this is an ASCII input field. The original test was
# `32 <= ch < 127`; it now runs over the character the `Key` carries, which is
# the same test with the raw code no longer available.
_ASCII_PRINTABLE = range(32, 127)


def _printable(key: Key) -> str | None:
    """The character to insert for this press, or None if it is not ASCII printable."""
    if key.kind == KEY_CHAR and len(key.char) == 1 and ord(key.char) in _ASCII_PRINTABLE:
        return key.char
    return None


# ── Colour pairs ──────────────────────────────────────────────────────────
#
# Legacy. Nothing here is read by the renderer any more: the engine emits raw
# SGR and picks its colours from its own `Color` enum, so a style maps straight
# to `Color.GREEN` in `_blit`, never to `color_pair(_P_LOG_INFO)`. The numbers
# are kept because `_STYLE_PAIRS` still records which style used to be which
# pair, and it is pinned by `test_tui_repl.py` and `test_tui_repl_utils.py`.
#
# There is deliberately no `_init_pairs()` here. It existed to call
# `curses.init_pair()`, was emptied when ncurses stopped owning the screen, and
# was then removed outright rather than left as a no-op named for work it no
# longer did -- a function whose name promises colour pairs and whose body
# `return`s is worse than its absence.

_P_LOG_INFO = 1
_P_LOG_WARN = 2
_P_LOG_ERROR = 3
_P_LOG_DEBUG = 4
_P_LOG_CRITICAL = 5
_P_PROMPT = 6
_P_BORDER = 7

_STYLE_PAIRS = {
    STYLE_INFO: _P_LOG_INFO,
    STYLE_WARN: _P_LOG_WARN,
    STYLE_ERROR: _P_LOG_ERROR,
    STYLE_DEBUG: _P_LOG_DEBUG,
    STYLE_CRITICAL: _P_LOG_CRITICAL,
}

# Python's `_curses` never exposed the ncurses extended-keypad modifiers, so
# Ctrl+Left / Ctrl+Right were read here as the folded codes 554 / 569 — a pair
# of integers whose provenance was "whatever this build happens to use". The
# decoder now parses the CSI modifier itself — either encoding, `ESC [ 5 D` or
# `ESC [ 1 ; 5 D`, so the chord arrives as `Key(KEY_LEFT, ctrl=True)` on every
# terminal that sends one, and nothing in this file needs to know a number for
# it or a constant name for it.

_PANE_PATTERN: dict[str, Pattern] = {
    "console": Pattern.DOTS,
    "output": Pattern.SHADE_LIGHT,
}


def _complete_path(token: str) -> list[str]:
    """Filesystem matches for a path token (``~`` expanded)."""
    base = os.path.expanduser(token)
    if "/" in base:
        d, prefix = os.path.split(base)
        d = d or "/"
    else:
        d, prefix = ".", base
    try:
        entries = sorted(os.listdir(d))
    except OSError:
        return []
    return [os.path.join(d, e) for e in entries if e.startswith(prefix)]


# Alt and Ctrl+arrow chords used to be resolved here: `_read_escape_remainder`
# polled the bytes after a bare ESC with a 0 ms timeout, because curses had
# already consumed the 27 and `getch()` was the only reader left. It existed to
# work around a split the vocabulary has since closed — `decode_key` parses the
# whole sequence, including the `ESC [ 1 ; 5 D` form this function half-parsed
# (it tested `"5" in seq` rather than reading the modifier), and reports
# `KEY_ALT` / `Key(KEY_LEFT, ctrl=True)` directly. Two readers of one byte
# stream, one of them 100 lines long, is what "no reader but the decoder"
# removes.


# ── TuiRepl ───────────────────────────────────────────────────────────────

# NOTE: TuiIo lives in .io as a first-class ShellIO (imported at top).


class TuiRepl:
    """Three-pane terminal shell composing a PaneLayout with surfaces."""

    CONSOLE_RATIO = 0.3
    CONSOLE_MIN = 4
    OUTPUT_MIN = 6

    def __init__(self, repl: ShellREPL, log_buffer: LogBuffer) -> None:
        self._repl = repl
        self._log_buffer = log_buffer
        self._running = False
        self._cmd_history: list[str] = []
        self._history_pos = 0
        # Unsent draft saved on first Up from fresh input, restored on
        # Down past newest (readline semantics — browsing never loses typing).
        self._history_draft: list[str] | None = None
        self._history_draft_cursor = 0
        self._search_fwd = False
        self._search_failed = False
        self._out_search_q = ""
        self._out_search_sel = -1
        self._out_search_save = 0
        self._out_search_failed = False

        self._kill_ring: list[str] = []
        self._yank_active = False
        self._yank_idx = -1
        self._yank_start = 0
        self._yank_len = 0

        self._repl_lock = threading.Lock()
        self._output_surface = TextSurface()
        self._input_surface = TextSurface()
        self._tui_io = TuiIo(self._output_surface)
        # The TUI's own reader. `TuiRepl` runs the shell's prompt loop itself in
        # TUI mode — the `interactive_*` prompts never start their read loop here —
        # so exactly one session reads the terminal, and this is it. It exists
        # for two things the loop could not do without a reader of its own:
        # `raw()` for the duration of `_main`, and `read_key(timeout=...)` for
        # the poll interval that `stdscr.timeout(100)` used to supply.
        self._session = TtySession(self._tui_io)
        self._tui_io._tui_ref = self
        self._dirty = True  # Track when redraw is needed

        # Wire interactive prompts into Console
        self._repl.console._tui_repl = self

        # Interactive select state (thread-safe signal between command thread and UI thread)
        #
        # Two signals per prompt, on purpose. `_select_event` is the *open*
        # latch: `_main` routes a keypress to the overlay and paints it only
        # while this is set. `_select_done` is the *answer*: what releases the
        # command thread parked inside `prompt_select`. Asking one event both
        # questions is how they went unanswered — `prompt_*` cleared it and
        # waited, `_main` waited for it to be set, and nothing ever set it.
        self._select_event = threading.Event()
        self._select_done = threading.Event()
        self._select_title = ""
        self._select_options: list[str] = []
        self._select_result: str | None = None
        self._select_idx = 0
        self._select_scroll = 0
        self._select_filter = ""

        # Interactive confirm state
        self._confirm_event = threading.Event()
        self._confirm_done = threading.Event()
        self._confirm_message = ""
        self._confirm_default = False
        self._confirm_result: bool | None = None

        # Interactive ask state
        self._ask_event = threading.Event()
        self._ask_done = threading.Event()
        self._ask_message = ""
        self._ask_default = ""
        self._ask_result: str | None = None
        self._ask_buf: list[str] = []
        self._ask_cursor = 0

        # Whether `_paint_overlay` has parked the hardware cursor for an open
        # overlay. It has to remember, or it would re-emit the escape on every
        # 100 ms tick and never put the cursor back.
        self._overlay_cursor_hidden = False

        self._old_io = None
        self._old_console_io = None
        self._log_surface = LogSurface(log_buffer)

        # Graphics engine (replaces curses rendering)
        self._engine: GraphicsEngine | None = None
        self._layer_console: Layer | None = None
        self._layer_output: Layer | None = None
        self._layer_status: Layer | None = None
        self._layer_input: Layer | None = None

        # Tab/focus manager
        self._tab_manager = TabManager()
        self._focused_pane: str = "output"  # default focus

        # Panes (arranger) — pure geometry, no rendering knowledge.
        self._layout = PaneLayout(
            [
                Pane(
                    "console",
                    ratio=self.CONSOLE_RATIO,
                    min_rows=self.CONSOLE_MIN,
                    border=Border("horizontal"),
                ),
                Pane(
                    "output",
                    ratio=1.0 - self.CONSOLE_RATIO,
                    min_rows=self.OUTPUT_MIN,
                    border=Border("top"),
                ),
                Pane("status", fixed=1),
                Pane("input", fixed=1),
            ]
        )

    def _repeat_out_search(self, rows: int, fwd: bool) -> None:
        """Repeat the last accepted output-pane search from the current match.

        ``n``/``N`` at an empty prompt move to the next/previous match of
        ``_out_search_last`` (wrapping), leaving the search mode closed.
        """
        if not self._out_search_last:
            return
        self._out_search_q = self._out_search_last
        self._apply_out_search(
            rows,
            start=self._out_search_sel + (1 if fwd else -1),
            fwd=fwd,
        )
        self._out_search_q = ""

    # ── Interactive prompts (called from command threads) ──────────────

    def prompt_select(self, title: str, options: list[str]) -> str:
        """Block the calling thread until the user picks an option."""
        if not options:
            return ""
        self._select_title = title
        self._select_options = list(options)
        self._select_idx = 0
        self._select_scroll = 0
        self._select_filter = ""
        self._select_result = None
        # Open the overlay, then block on the *answer*. The stale-answer clear
        # has to come first: a key could only set `_select_done` once the latch
        # is up, so clearing after opening would swallow that answer.
        self._select_done.clear()
        self._select_event.set()
        self._select_done.wait(timeout=300)
        self._select_event.clear()  # idempotent: answering cleared it already
        return self._select_result if self._select_result is not None else options[0]

    def prompt_confirm(self, message: str, default: bool = False) -> bool:
        """Block the calling thread until the user confirms yes/no."""
        self._confirm_message = message
        self._confirm_default = default
        self._confirm_result = None
        self._confirm_done.clear()
        self._confirm_event.set()
        self._confirm_done.wait(timeout=300)
        self._confirm_event.clear()
        return self._confirm_result if self._confirm_result is not None else default

    def prompt_ask(self, message: str, default: str = "") -> str:
        """Block the calling thread until the user provides text input."""
        self._ask_message = message
        self._ask_default = default
        self._ask_result = None
        self._ask_buf = list(default) if default else []
        self._ask_cursor = len(self._ask_buf)
        self._ask_done.clear()
        self._ask_event.set()
        self._ask_done.wait(timeout=300)
        self._ask_event.clear()
        return self._ask_result if self._ask_result is not None else default

    # ── Prompt key handling ──────────────────────────────────────────────
    #
    # These are the whole of what a keypress *means* for each overlay, split out
    # of the event loop so the loop is left owning only rendering and dispatch.
    # Each returns whether it consumed the key, which is what decides whether
    # the loop repaints every pane or only the overlay.
    #
    # Each takes one argument, the `Key` the loop read. There used to be two —
    # a raw code alongside the normalised key — because the vocabulary could not
    # express Ctrl+P, Ctrl+N or Ctrl+G, and three decisions at the top of the
    # module therefore read codes the decoder was folding into KEY_ESC. That
    # is closed: the chords are named, so `_CANCEL` and `_chord` say what they
    # mean and no handler reaches past the key for a byte.

    def _handle_select_key(self, key: Key, regions: dict) -> bool:
        """Apply one key to the open select overlay.

        Args:
            key: the press read by the loop, already normalised.
            regions: current layout, needed for the scroll arithmetic.

        Returns:
            True when a branch consumed the key. False means the press is not
            one this widget acts on, so the loop leaves the panes alone and
            repaints only the overlay.
        """
        filtered = [o for o in self._select_options if self._select_filter.lower() in o.lower()]
        if not filtered:
            filtered = list(self._select_options)
        max_show = min(len(filtered), regions["output"].rows - 2)

        # Up and Down come first, though the ordering is no longer load-bearing:
        # Ctrl+P and Ctrl+N are chords now, so they cannot collide with the
        # cancel test the way two KEY_ESC results once would have.
        if key.matches(KEY_UP) or _chord(key, _CTRL_UP_CHAR):
            self._select_idx = max(0, self._select_idx - 1)
            self._select_scroll = min(self._select_scroll, self._select_idx)
        elif key.matches(KEY_DOWN) or _chord(key, _CTRL_DOWN_CHAR):
            self._select_idx = min(len(filtered) - 1, self._select_idx + 1)
            if self._select_idx >= self._select_scroll + max_show:
                self._select_scroll = self._select_idx - max_show + 1
        elif key.kind in _CANCEL:  # Esc / Ctrl+G / Ctrl+C — cancel
            self._select_result = None
            self._select_event.clear()
            self._select_done.set()
        elif key.kind == KEY_ENTER:
            if 0 <= self._select_idx < len(filtered):
                self._select_result = filtered[self._select_idx]
            else:
                self._select_result = None
            self._select_event.clear()
            self._select_done.set()
        elif key.kind == KEY_PAGE_UP:
            self._select_idx = max(0, self._select_idx - max_show)
            self._select_scroll = max(0, self._select_scroll - max_show)
        elif key.kind == KEY_PAGE_DOWN:
            self._select_idx = min(len(filtered) - 1, self._select_idx + max_show)
            if self._select_idx >= self._select_scroll + max_show:
                self._select_scroll = self._select_idx - max_show + 1
        elif key.kind == KEY_BACKSPACE:  # KEY_BACKSPACE, 127 and 8 all land here
            self._select_filter = self._select_filter[:-1]
            self._select_idx = 0
            self._select_scroll = 0
        elif (text := _printable(key)) is not None:
            self._select_filter += text
            self._select_idx = 0
            self._select_scroll = 0
        else:
            return False
        return True

    def _handle_confirm_key(self, key: Key) -> bool:
        """Apply one key to the open confirm overlay. Returns whether consumed."""
        if key.kind in _CANCEL:  # Esc / Ctrl+G / Ctrl+C — cancel
            self._confirm_result = not self._confirm_default
            self._confirm_event.clear()
        elif key.kind == KEY_ENTER:  # Enter — use the default
            self._confirm_result = self._confirm_default
            self._confirm_event.clear()
        elif key.kind == KEY_CHAR and key.char in ("y", "Y"):
            self._confirm_result = True
            self._confirm_event.clear()
        elif key.kind == KEY_CHAR and key.char in ("n", "N"):
            self._confirm_result = False
            self._confirm_event.clear()
        else:
            return False
        # Unlike select and ask, every branch above is an *answer* — confirm
        # has no motion to consume — so releasing the waiter here is safe.
        # Each of them has already recorded its result and dropped the latch.
        self._confirm_done.set()
        return True

    def _handle_ask_key(self, key: Key) -> bool:
        """Apply one key to the open ask overlay. Returns whether consumed."""
        if key.kind in _CANCEL:  # Esc / Ctrl+G / Ctrl+C — cancel
            self._ask_result = self._ask_default
            self._ask_event.clear()
            self._ask_done.set()
        elif key.kind == KEY_ENTER:  # Enter — accept
            self._ask_result = "".join(self._ask_buf)
            self._ask_event.clear()
            self._ask_done.set()
        elif key.kind == KEY_BACKSPACE:  # KEY_BACKSPACE, 127 and 8 all land here
            if self._ask_buf:
                self._ask_buf.pop()
                self._ask_cursor = max(0, self._ask_cursor - 1)
        elif key.matches(KEY_LEFT):
            self._ask_cursor = max(0, self._ask_cursor - 1)
        elif key.matches(KEY_RIGHT):
            self._ask_cursor = min(len(self._ask_buf), self._ask_cursor + 1)
        elif key.matches(KEY_LEFT, ctrl=True):
            # Ctrl+Left — word back (backward-word). `matches` names the
            # modifier, so the plain-Left branch above cannot swallow it.
            self._ask_cursor = word_backward(self._ask_buf, self._ask_cursor)
        elif key.matches(KEY_RIGHT, ctrl=True):
            # Ctrl+Right — word forward (forward-word)
            self._ask_cursor = word_forward(self._ask_buf, self._ask_cursor)
        elif key.matches(KEY_CTRL, char="w"):
            # Ctrl+W — erase the word before the caret (unix-word-rubout)
            self._ask_cursor, _ = delete_back(self._ask_buf, self._ask_cursor)
        elif key.matches(KEY_ALT, char="d"):
            # Alt+D — erase the word after the caret (kill-word)
            self._ask_cursor, _ = delete_forward(self._ask_buf, self._ask_cursor)
        elif (text := _printable(key)) is not None:
            self._ask_buf.insert(self._ask_cursor, text)
            self._ask_cursor += 1
        else:
            return False
        return True

    def _render_select(self, regions: dict) -> None:
        """Render the interactive select overlay on the output pane."""
        filtered = [o for o in self._select_options if self._select_filter.lower() in o.lower()]
        if not filtered:
            filtered = list(self._select_options)
        rows = regions["output"].rows
        cols = regions["output"].cols
        max_show = min(len(filtered), rows - 2)
        lines: list[str] = []
        lines.append(f"  {self._select_title}")
        if self._select_filter:
            lines.append(f"  Filter: {self._select_filter}_")
        lines.append("")
        for i in range(max_show):
            idx = self._select_scroll + i
            if idx >= len(filtered):
                break
            prefix = " > " if idx == self._select_idx else "   "
            text = filtered[idx]
            if len(prefix) + len(text) > cols - 2:
                text = text[: cols - 2 - len(prefix)]
            lines.append(f"{prefix}{text}")
        if len(filtered) > max_show:
            lines.append(f"   ({len(filtered)} items, {self._select_idx + 1}/{len(filtered)})")
        lines.append("")
        lines.append("  Enter: select  Esc: cancel  Type to filter")
        self._output_surface.clear()
        for line in lines:
            self._output_surface.write(line)
        self._blit("output", self._output_surface.render(rows, 0))

    def _render_confirm(self) -> None:
        """Render the confirm prompt overlay on the input pane."""
        hint = "Y/n" if self._confirm_default else "y/N"
        prompt = f"  {self._confirm_message} [{hint}] "
        self._input_surface.clear()
        self._input_surface.write(prompt)
        self._blit("input", self._input_surface.render(1, 0))

    def _render_ask(self) -> None:
        """Render the ask prompt overlay on the input pane."""
        suffix = f" [{self._ask_default}]" if self._ask_default else ""
        prompt = f"  {self._ask_message}{suffix}: "
        # The caret goes where the cursor *is*. Pinned to the tail of the text
        # it made the field look inert the moment word motion moved the cursor
        # anywhere but the end — the movement worked and nothing showed for it,
        # which is worse than not having the movement at all.
        before = "".join(self._ask_buf[: self._ask_cursor])
        after = "".join(self._ask_buf[self._ask_cursor :])
        display = f"{prompt}{before}_{after}"
        self._input_surface.clear()
        self._input_surface.write(display)
        self._blit("input", self._input_surface.render(1, 0))

    def _overlay_open(self) -> bool:
        """Whether a prompt overlay is currently claiming the keypress."""
        return (
            self._select_event.is_set() or self._confirm_event.is_set() or self._ask_event.is_set()
        )

    def _draw_overlay(self, regions: dict) -> None:
        """Write whichever prompt is open into the engine layers — no flush.

        The order matches the dispatch chain in `_main`: select is the
        outermost widget, ask the innermost, and only one can be open at a
        time, so this checks them in the same sequence.
        """
        if self._select_event.is_set():
            self._render_select(regions)
        elif self._confirm_event.is_set():
            self._render_confirm()
        elif self._ask_event.is_set():
            self._render_ask()

    def _sync_overlay_cursor(self, overlay_open: bool) -> None:
        """Park the hardware cursor while an overlay draws its own.

        Every overlay draws its own caret — ask renders a literal `_` — so the
        terminal cursor left at the λ caret would blink as a second one inside
        the box. It has to come back with the λ prompt, or the cursor stays
        hidden after the overlay closes.
        """
        if self._engine is None:
            return
        if overlay_open and not self._overlay_cursor_hidden:
            self._engine.show_cursor(False)
            self._overlay_cursor_hidden = True
        elif not overlay_open and self._overlay_cursor_hidden:
            self._engine.show_cursor(True)
            self._overlay_cursor_hidden = False

    def _paint_overlay(self, regions: dict) -> None:
        """Draw the open overlay and composite it, for the paths that repaint
        nothing else.

        `_render_all` draws the overlay itself (see the call before its flush),
        so the pane repaint and the tick are covered there. This is for the one
        case they miss: an unconsumed key, where `_main` deliberately leaves the
        panes alone and repaints only the box — without a composite here that
        write would never reach the terminal.
        """
        self._draw_overlay(regions)
        if self._engine is None:
            return
        if self._overlay_open():
            self._engine.render()
        self._sync_overlay_cursor(self._overlay_open())

    def run(self) -> None:
        """Enter raw mode and start the event loop."""
        self._running = True
        store = getattr(self._repl, "_history_store", None)
        history = store.list() if store is not None else getattr(self._repl, "_history", None)
        self._cmd_history = list(history) if history else []
        self._history_pos = len(self._cmd_history)

        try:
            from domain.logging._internal.cli_logger import set_cli_terminal
        except ImportError:
            set_cli_terminal = None
        if set_cli_terminal is not None:
            set_cli_terminal(False)
        try:
            with self._session.raw():
                self._main()
        finally:
            if set_cli_terminal is not None:
                set_cli_terminal(True)

    # ── Rendering (GraphicsEngine) ─────────────────────────────────────

    def _draw_borders(self, regions: dict[str, Rect]) -> None:
        """Draw border characters for all panes using the engine."""
        if self._engine is None:
            return
        for pane in self._layout.panes:
            if not pane.visible or pane.border.is_empty:
                continue
            r = regions.get(pane.name)
            if not r:
                continue
            # Draw borders on the engine's front buffer
            layer = self._get_layer(pane.name)
            if layer is None:
                continue
            # Highlight focused pane border
            is_focused = pane.name == self._focused_pane
            border_fg = Color.CYAN if is_focused else Color.WHITE
            border_attr = Attr.BOLD if is_focused else Attr.NONE
            h_char = pane.border.ch or "─"
            v_char = pane.border.ch or "│"
            if pane.border.top and r.rows > 0:
                layer.write(0, 0, h_char * r.cols, fg=border_fg, attr=border_attr)
            if pane.border.bottom and r.rows > 1:
                layer.write(r.rows - 1, 0, h_char * r.cols, fg=border_fg, attr=border_attr)
            if pane.border.left and r.cols > 0:
                for y in range(r.rows):
                    layer.write(y, 0, v_char, fg=border_fg, attr=border_attr)
            if pane.border.right and r.cols > 1:
                for y in range(r.rows):
                    layer.write(y, r.cols - 1, v_char, fg=border_fg, attr=border_attr)

    def _get_layer(self, pane_name: str) -> Layer | None:
        """Get the engine layer for a pane name."""
        if pane_name == "console":
            return self._layer_console
        elif pane_name == "output":
            return self._layer_output
        elif pane_name == "status":
            return self._layer_status
        elif pane_name == "input":
            return self._layer_input
        return None

    def _draw_pane_borders(self, layer: Layer, pane: Pane) -> None:
        """Draw border characters for *pane* onto *layer*."""
        if self._engine is None:
            return
        if not pane.visible or pane.border.is_empty:
            return
        h, w = layer.framebuffer.rows, layer.framebuffer.cols
        h_char = pane.border.ch or "─"
        v_char = pane.border.ch or "│"
        if pane.border.top and h > 0:
            layer.write(0, 0, h_char * w, fg=Color.WHITE)
        if pane.border.bottom and h > 1:
            layer.write(h - 1, 0, h_char * w, fg=Color.WHITE)
        if pane.border.left and w > 0:
            for y in range(h):
                layer.write(y, 0, v_char, fg=Color.WHITE)
        if pane.border.right and w > 1:
            for y in range(h):
                layer.write(y, w - 1, v_char, fg=Color.WHITE)

    def _blit(
        self,
        layer_name: str,
        lines: list[RenderLine],
        offset_y: int = 0,
        offset_x: int = 0,
        pane: Pane | None = None,
    ) -> None:
        """Write *lines* into the engine layer named *layer_name*.

        If *pane* is given, its borders are drawn on the layer after clear.
        """
        if self._engine is None:
            return
        layer = self._get_layer(layer_name)
        if layer is None:
            return

        # Repaint this pane's backdrop before drawing content.  For a textured
        # pane this *replaces* clear() — one pass over the layer either way —
        # and it has to run first so that the spaces written into a line
        # overwrite the texture rather than letting it bleed through them.
        # See _PANE_PATTERN.
        pattern = _PANE_PATTERN.get(layer.name)
        if pattern is not None:
            layer.fill_pattern(pattern, fg=Color.BRIGHT_BLACK)
        else:
            layer.clear()

        # Draw borders if pane has them
        if pane is not None and not pane.border.is_empty:
            self._draw_pane_borders(layer, pane)

        h, w = layer.framebuffer.rows, layer.framebuffer.cols
        for y, ln in enumerate(lines):
            wy = offset_y + y
            if wy >= h:
                break
            # Map surface style to engine colors and display modes
            fg = Color.DEFAULT
            attr = Attr.NONE
            if ln.style == STYLE_INFO:
                fg = Color.GREEN
            elif ln.style == STYLE_WARN:
                fg = Color.YELLOW
                attr = Attr.BOLD
            elif ln.style == STYLE_ERROR:
                fg = Color.RED
                attr = Attr.BOLD
            elif ln.style == STYLE_DEBUG:
                fg = Color.CYAN
            elif ln.style == STYLE_CRITICAL:
                fg = Color.MAGENTA
                attr = Attr.REVERSE | Attr.BOLD
            text = ln.text[: w - offset_x - 1] if offset_x < w else ""
            if text:
                layer.write(wy, offset_x, text, fg=fg, attr=attr)

    def _render_all(self, regions) -> None:
        if self._engine is None:
            return
        # Draw borders on content layers
        _panes = {p.name: p for p in self._layout.panes}
        # Compute content offsets from borders.
        for pane in self._layout.panes:
            if pane.name == "console":
                oy = pane.border_top + pane.pad_top
                ox = pane.border_left + pane.pad_left
                self._blit(
                    "console",
                    self._log_surface.render(
                        regions["console"].rows - pane.border_top - pane.border_bottom,
                        self._log_scroll,
                    ),
                    oy,
                    ox,
                    pane=pane,
                )
            elif pane.name == "output":
                oy = pane.border_top + pane.pad_top
                ox = pane.border_left + pane.pad_left
                self._blit(
                    "output",
                    self._output_surface.render(
                        regions["output"].rows - pane.border_top - pane.border_bottom,
                        self._out_scroll,
                    ),
                    oy,
                    ox,
                    pane=pane,
                )
        self._render_status(regions["status"].cols)
        caret_col = self._render_input(regions["input"].cols)
        # The open overlay joins the layers *before* the composite, so the frame
        # that reaches the terminal already contains it. Drawn after the flush
        # it would sit in a buffer the next pane repaint clears before anybody
        # composites it — and the pair of writes would flash the λ line
        # underneath the box once per tick, which a snapshot can catch.
        self._draw_overlay(regions)
        # Composite and render to terminal
        self._engine.render()
        # Position the hardware cursor on the input row — without this the
        # cursor stays where the last diff left it and history browsing
        # looks mixed/desynced.
        try:
            reg = regions["input"]
            self._engine.set_cursor(reg.top, reg.left + max(0, min(caret_col, reg.cols - 1)))
        except Exception:
            pass
        # …and then out of the way if an overlay is drawn over that row: it
        # carries its own caret, so a blinking terminal cursor in the middle of
        # the box would read as a second one.
        self._sync_overlay_cursor(self._overlay_open())

    def _input_view(self, cols: int, buf: str, caret: int) -> tuple[str, int]:
        """Compute the visible input line and its caret column.

        ``cols`` is the window width; returns ``(line, caret_col)`` where
        ``line`` fits the window (prompt + ``buf``) and ``caret_col`` is the
        absolute window column the cursor should occupy.  When the buffer is
        wider than the window the view scrolls horizontally so the caret
        stays visible; a buffer at the window edge reveals its tail.
        """
        prompt = "λ "
        max_w = max(cols - len(prompt) - 1, 0)
        caret = min(max(caret, 0), len(buf))
        if len(buf) <= max_w:
            return buf, len(prompt) + caret
        start = min(caret, len(buf) - max_w)
        return buf[start : start + max_w], len(prompt) + (caret - start)

    def _place_layers(self, regions) -> None:
        """Pin each engine layer to its pane region (top/left)."""
        pairs = (
            ("console", "console"),
            ("output", "output"),
            ("status", "status"),
            ("input", "input"),
        )
        for layer_name, region_name in pairs:
            layer = getattr(self, "_layer_" + layer_name, None)
            region = regions.get(region_name)
            if layer is not None and region is not None:
                layer.set_position(region.left, region.top)

    def _render_input(self, cols: int) -> int:
        """Draw the command line using the engine with display modes.

        Returns the caret column (absolute window col) so the caller can
        position the terminal cursor after compositing.
        """
        if self._engine is None or self._layer_input is None:
            return 0
        from .prompt import PromptContext, PromptRenderer

        layer = self._layer_input
        layer.clear()
        # Shared expander with line mode; TUI keeps its minimal PS1
        # (no badge/exit prefix — single-row input pane).
        prompt = PromptRenderer.render(PromptContext(ps1="λ "))
        buf = "".join(self._input_buf)
        line, caret_col = self._input_view(cols, buf, self._input_cursor)
        layer.write(0, 0, prompt, fg=Color.CYAN, attr=Attr.BOLD)
        if line:
            layer.write(0, len(prompt), line)
        return caret_col

    def _render_status(self, cols: int) -> None:
        """Draw the chrome bar using the engine with display modes."""
        if self._engine is None or self._layer_status is None:
            return
        layer = self._layer_status
        layer.clear()
        if cols <= 0:
            return
        if self._searching:
            label = "forward-i-search" if self._search_fwd else "reverse-i-search"
            if self._search_failed:
                label = f"failed {label}"
            prompt = f"({label})\u0060{self._search_q}\u0060:"
            layer.write(0, 0, prompt[: cols - 2], fg=Color.CYAN, attr=Attr.BOLD)
            return
        if self._out_searching:
            label = "output-search"
            if self._out_search_failed:
                label = f"failed {label}"
            prompt = f"({label})\u0060{self._out_search_q}\u0060:"
            layer.write(0, 0, prompt[: cols - 2], fg=Color.CYAN, attr=Attr.BOLD)
            return

        # ── Normal status bar ──
        scroll = self._out_scroll if self._scroll_target == 0 else self._log_scroll

        # [OUTPUT] or [LOG] — whichever pane PgUp/PgDn scrollback drives.
        # The label is the only place that distinction is legible, so it has
        # to actually change with the target: both arms used to spell OUTPUT,
        # which left Ctrl+O with nothing to show. The field is then held at a
        # fixed width so the rest of the bar does not shift when the shorter
        # LOG replaces OUTPUT.
        label = "OUTPUT" if self._scroll_target == 0 else "LOG"
        hot = self._scroll_target == 0
        layer.write(0, 0, "[", fg=Color.WHITE, attr=Attr.DIM)
        layer.write(
            0,
            1,
            label,
            fg=Color.CYAN if hot else Color.WHITE,
            attr=Attr.BOLD if hot else Attr.DIM,
        )
        col = 1 + len(label)
        layer.write(0, col, "]", fg=Color.WHITE, attr=Attr.DIM)
        # Hold the label field at 8 columns so the rest of the bar does not
        # shift when the shorter LOG replaces OUTPUT.
        col = max(col + 1, 8)

        # Separator
        layer.write(0, col, " ", fg=Color.WHITE)
        col += 1

        # Scroll indicator. "SCROLL" is the word a reader scans for, and it is
        # what the status bar is asserted on: an arrow and a bare count read
        # as a glyph accident at a glance, and said nothing about *which*
        # pane was held back from the tail. The count stays, after the word.
        if scroll > 0:
            scroll_txt = f"SCROLL ↑{scroll}"
            layer.write(0, col, scroll_txt, fg=Color.YELLOW, attr=Attr.BOLD)
            col += len(scroll_txt) + 1
        else:
            # LIVE indicator
            layer.write(0, col, "●", fg=Color.GREEN, attr=Attr.BOLD)
            col += 2
            layer.write(0, col, "LIVE", fg=Color.GREEN)
            col += 4 + 1

        # Active command (dimmed)
        if col < cols and self._active_cmd:
            cmd = self._active_cmd[: cols - col - 4]
            layer.write(0, col, "│", fg=Color.WHITE, attr=Attr.DIM)
            col += 2
            layer.write(0, col, cmd, fg=Color.WHITE, attr=Attr.DIM)
            col += len(cmd) + 1

        # Right-aligned size
        suffix = f"{cols}x{self._rows}"
        if cols - len(suffix) - 1 >= col:
            layer.write(0, cols - len(suffix) - 1, suffix, fg=Color.WHITE, attr=Attr.DIM)

    # ── Completion ──────────────────────────────────────────────────────

    def _complete(self) -> None:
        """Tab-complete the token under the caret: commands on the leading
        token, filesystem paths anywhere (incl. a fallback when no command
        matches)."""
        buf = "".join(self._input_buf)
        caret = self._input_cursor
        start = buf.rfind(" ", 0, caret) + 1
        token = buf[start:caret]
        leading = " " not in buf[:start]
        if leading and not token.startswith((".", "/", "~")):
            matches = [k for k in self._repl.COMMANDS if k.startswith(token)]
            if len(matches) == 1:
                self._set_token(start, caret, matches[0] + " ")
                return
            if matches:
                common = os.path.commonprefix(matches)
                if len(common) > len(token):
                    self._set_token(start, caret, common)
                else:
                    self._output_surface.write("  " + "  ".join(matches))
                return
        self._complete_path_token(start, caret, token)

    def _complete_path_token(self, start: int, caret: int, token: str) -> None:
        matches = _complete_path(token)
        if not matches:
            return
        if len(matches) == 1:
            suffix = "/" if os.path.isdir(matches[0]) else " "
            self._set_token(start, caret, matches[0] + suffix)
            return
        common = os.path.commonprefix(matches)
        if len(common) > len(token):
            self._set_token(start, caret, common)
        else:
            self._output_surface.write("  " + "  ".join(matches))

    def _set_token(self, start: int, caret: int, text: str) -> None:
        self._input_buf[start:caret] = list(text)
        self._input_cursor = start + len(text)

    # ── Line editing (readline-style) ──────────────────────────────────

    _KILL_RING_MAX = 10

    def _move_home(self) -> None:
        """Move the caret to the start of the line."""
        self._input_cursor = 0

    def _move_end(self) -> None:
        """Move the caret to the end of the line."""
        self._input_cursor = len(self._input_buf)

    def _history_back(self) -> None:
        """Step to the previous (older) history entry, filling the input row.

        No-op at the oldest entry; mirrors readline ``previous-history``.
        The unsent draft is saved on first step off fresh input.
        """
        if self._cmd_history and self._history_pos > 0:
            if self._history_pos == len(self._cmd_history):
                self._history_draft = list(getattr(self, "_input_buf", []))
                self._history_draft_cursor = getattr(self, "_input_cursor", 0)
            self._history_pos -= 1
            self._input_buf = list(self._cmd_history[self._history_pos])
            self._input_cursor = len(self._input_buf)

    def _history_fwd(self) -> None:
        """Step to the next (newer) history entry; past the newest restores
        the saved draft (or clears when none) — readline ``next-history``."""
        if self._cmd_history and self._history_pos < len(self._cmd_history) - 1:
            self._history_pos += 1
            self._input_buf = list(self._cmd_history[self._history_pos])
            self._input_cursor = len(self._input_buf)
        else:
            self._history_pos = len(self._cmd_history)
            if self._history_draft is not None:
                self._input_buf = list(self._history_draft)
                self._input_cursor = self._history_draft_cursor
                self._history_draft = None
            else:
                self._input_buf.clear()
                self._input_cursor = 0

    def _move_word_forward(self) -> None:
        """Move the caret to the end of the next word (Alt+F / Ctrl+Right)."""
        self._input_cursor = word_forward(self._input_buf, self._input_cursor)

    def _move_word_backward(self) -> None:
        """Move the caret to the start of the current or previous word (Alt+B / Ctrl+Left)."""
        self._input_cursor = word_backward(self._input_buf, self._input_cursor)

    def _transpose_chars(self) -> None:
        """Swap the character before the caret with the one at the caret
        (Ctrl+T).

        Mirrors readline's ``transpose-chars``: with a character on both
        sides the pair swaps and the caret advances past them; at the end
        of the line the last two characters swap and the caret stays at the
        end.  No-op at the start of the line or on a one-character line.
        """
        i = self._input_cursor
        n = len(self._input_buf)
        if i == 0:
            return
        if i == n:
            if n >= 2:
                self._input_buf[n - 2], self._input_buf[n - 1] = (
                    self._input_buf[n - 1],
                    self._input_buf[n - 2],
                )
                self._input_cursor = n
            return
        self._input_buf[i - 1], self._input_buf[i] = (
            self._input_buf[i],
            self._input_buf[i - 1],
        )
        self._input_cursor = i + 1

    def _push_kill(self, text: str) -> None:
        """Add killed text to the kill ring (most recent last, capped).

        Any push cancels an in-progress yank cycle.

        Args:
            text: the killed substring; empty text is ignored.

        Side effects:
            - appends to ``_kill_ring`` (oldest entry dropped past the cap)
            - resets ``_yank_active`` / ``_yank_idx``
        """
        if not text:
            return
        self._kill_ring.append(text)
        if len(self._kill_ring) > self._KILL_RING_MAX:
            del self._kill_ring[: len(self._kill_ring) - self._KILL_RING_MAX]
        self._yank_active = False
        self._yank_idx = -1

    def _kill_to_start(self) -> None:
        """Delete the text before the caret (Ctrl+U); pushed to the ring."""
        killed = "".join(self._input_buf[: self._input_cursor])
        self._input_buf = self._input_buf[self._input_cursor :]
        self._input_cursor = 0
        self._push_kill(killed)

    def _kill_to_end(self) -> None:
        """Delete the text from the caret to the end (Ctrl+K); pushed to ring."""
        killed = "".join(self._input_buf[self._input_cursor :])
        self._input_buf = self._input_buf[: self._input_cursor]
        self._push_kill(killed)

    def _delete_at_cursor(self) -> None:
        """Delete the character under the caret, if any (Ctrl+D / Delete).

        Single-character deletion is not pushed to the kill ring (readline
        behaviour).
        """
        if self._input_cursor < len(self._input_buf):
            self._input_buf.pop(self._input_cursor)

    def _delete_word_back(self) -> None:
        """Delete the word (and any trailing whitespace) before the caret.

        Mirrors readline's ``unix-word-rubout``. The scan lives in `textedit`
        so the prompts erase exactly the run this one jumps over; only the
        kill ring is shell-specific, so the push stays here.
        """
        self._input_cursor, killed = delete_back(self._input_buf, self._input_cursor)
        self._push_kill(killed)

    def _delete_word_forward(self) -> None:
        """Delete the word (and any leading whitespace) after the caret.

        Mirrors readline's ``kill-word``: the caret does not move. Pushed to
        the kill ring for the same reason as above.
        """
        _, killed = delete_forward(self._input_buf, self._input_cursor)
        self._push_kill(killed)

    def _yank(self) -> None:
        """Insert the most recent kill at the caret; repeat to cycle (Ctrl+Y).

        The first press pastes the newest ring entry at the caret.  An
        immediately following press replaces it with the next-older entry.
        If the buffer no longer contains the previously yanked text at the
        yank position (e.g. the line was edited), the next press starts a
        fresh yank from the newest entry.
        """
        if not self._kill_ring:
            return
        if self._yank_active:
            prev = self._kill_ring[self._yank_idx]
            here = "".join(self._input_buf[self._yank_start : self._yank_start + len(prev)])
            if here != prev:
                self._yank_active = False  # line was edited since — start fresh
        if not self._yank_active:
            self._yank_active = True
            self._yank_idx = len(self._kill_ring) - 1
            self._yank_start = self._input_cursor
            text = self._kill_ring[self._yank_idx]
        else:
            prev = self._kill_ring[self._yank_idx]
            del self._input_buf[self._yank_start : self._yank_start + len(prev)]
            self._input_cursor = self._yank_start
            self._yank_idx = max(self._yank_idx - 1, 0)
            text = self._kill_ring[self._yank_idx]
        self._input_buf[self._yank_start : self._yank_start] = list(text)
        self._input_cursor = self._yank_start + len(text)

    # ── Interrupt (Ctrl+C) ─────────────────────────────────────────────

    def _interrupt_active(self) -> None:
        """Raise ``KeyboardInterrupt`` in the running command thread.

        The command executes ``ShellREPL._dispatch`` on a background thread;
        raising ``KeyboardInterrupt`` there lets the REPL's own handler print
        "Aborted" and record the exit code.  Best-effort: a thread blocked in
        a syscall (e.g. ``time.sleep``) is interrupted once it returns to
        Python code.  No-op when no command is running or the async-exc
        facility is unavailable.

        Side effects:
            - interrupts the thread executing the active command
        """
        thread = self._active_thread
        if _SET_ASYNC_EXC is None or thread is None:
            return
        if not thread.is_alive():
            return
        tid = thread.ident
        if tid is None or tid == threading.get_ident():
            return
        try:
            res = _SET_ASYNC_EXC(ctypes.c_long(tid), ctypes.py_object(KeyboardInterrupt))
            if res == 0:
                return  # thread already finished
            if res != 1:
                _SET_ASYNC_EXC(ctypes.c_long(tid), None)
        except (ValueError, SystemError, TypeError, RuntimeError):
            pass

    # ── Reverse history search ─────────────────────────────────────────

    def _search_back(self, start: int, fwd: bool = False) -> int:
        """Nearest history index to ``start`` whose entry contains the query.

        Returns -1 when the query is empty, the history is empty, or no
        entry matches.  ``start`` may equal ``len(history)`` (the position
        past the newest entry) — backward search then begins at the newest
        entry instead of falling off the end.
        """
        q = self._search_q
        n = len(self._cmd_history)
        if not q or n == 0:
            return max(min(start, n - 1), 0)
        if fwd:
            if start >= n:
                return -1
            idx = max(start, 0)
        else:
            if start < 0:
                return -1
            idx = min(start, n - 1)
        step = 1 if fwd else -1
        while 0 <= idx < n:
            if q in self._cmd_history[idx]:
                return idx
            idx += step
        return -1

    def _apply_search(self) -> None:
        """Fill the input row with the match nearest to ``_search_idx``.

        Scans in the active direction (``_search_fwd`` False → backward to
        older entries, True → forward to newer entries) starting at
        ``_search_idx`` inclusive, so a fresh query starts from the far end
        of the history and navigation restarts just past the last match.
        Sets ``_search_failed`` when the query matches no entry.
        """
        self._search_failed = False
        if self._search_q and self._cmd_history:
            found = self._search_back(self._search_idx, fwd=self._search_fwd)
            if found >= 0:
                self._search_idx = found
                self._input_buf = list(self._cmd_history[found])
                self._input_cursor = len(self._input_buf)
            else:
                self._search_failed = True
        elif self._search_save is not None:
            self._input_buf = list(self._search_save[0])
            self._input_cursor = self._search_save[1]

    def _end_search(self, restore: bool) -> None:
        self._searching = False
        self._search_q = ""
        if restore and self._search_save is not None:
            self._input_buf = list(self._search_save[0])
            self._input_cursor = self._search_save[1]
        self._search_save = None

    # ── Output-pane content search ─────────────────────────────────────────

    def _out_find(self, start: int, fwd: bool) -> int:
        """Nearest capture index to ``start`` whose line contains the query.

        Case-insensitive substring match over the output buffer, wrapping
        from the tail back to the head (and vice versa for ``fwd`` False).
        Returns -1 when the query is empty, the buffer is empty, or no line
        matches.
        """
        q = self._out_search_q.lower()
        lines = self._output_surface.capture
        n = len(lines)
        if not q or n == 0:
            return -1
        if start < 0:
            start = 0 if fwd else n - 1
        if fwd:
            for idx in range(start, n):
                if q in lines[idx].lower():
                    return idx
            for idx in range(start):
                if q in lines[idx].lower():
                    return idx
        else:
            for idx in range(start, -1, -1):
                if q in lines[idx].lower():
                    return idx
            for idx in range(n - 1, start, -1):
                if q in lines[idx].lower():
                    return idx
        return -1

    def _apply_out_search(self, rows: int, start: int | None = None, fwd: bool = True) -> None:
        """Jump the output pane to the nearest line matching the query.

        ``_out_scroll`` is set so the matched line lands at the top of the
        pane (relative to the live tail).  ``_out_search_sel`` tracks the
        current match so n/N can cycle; ``_out_search_failed`` is raised when
        nothing matches.
        """
        self._out_search_failed = False
        lines = self._output_surface.capture
        n = len(lines)
        if self._out_search_q and n > 0:
            found = self._out_find(start if start is not None else self._out_search_sel, fwd)
            if found >= 0:
                self._out_search_sel = found
                self._out_scroll = max(n - rows - found, 0)
            else:
                self._out_search_failed = True
        else:
            self._out_search_sel = -1

    # ── Main loop ─────────────────────────────────────────────────────────

    def _main(self) -> None:
        # Terminal setup now lives in three places, none of them here: raw mode
        # in `run()` via `self._session.raw()` (so Ctrl+C arrives as a key
        # rather than SIGINT), the escdelay in `tui_session.ESC_DELAY` (so Esc
        # is distinguishable — this is what `curses.set_escdelay(25)` did), and
        # the alternate screen in `GraphicsEngine.open()` below.
        #
        # `curses.keypad(True)` has no counterpart because there is nothing left
        # to ask. It told ncurses to *fold* sequences into KEY_* codes; the
        # decoder parses them itself, so `ESC O A` arrives as KEY_UP by being
        # read, not by being translated by a third party.
        #
        # Cursor visibility is deliberately *not* set here. `curses.curs_set`
        # wrote to the screen, and the engine hides the cursor in `open()`,
        # so setting it here would just be undone -- see `show_cursor` after
        # the engine is up.
        rows, cols = self._session.size()
        self._rows = rows
        self._cols = cols
        self._input_buf: list[str] = []
        self._input_cursor = 0
        self._history_pos = len(self._cmd_history)
        self._out_scroll = 0
        self._log_scroll = 0
        self._scroll_target = 0  # kept for compat, but _tab_manager is the source of truth
        self._active_cmd: str | None = None
        self._active_thread: threading.Thread | None = None
        self._searching = False
        self._search_q = ""
        self._search_idx = 0
        self._search_save: tuple[list[str], int] | None = None
        self._out_searching = False
        self._out_search_q = ""
        self._out_search_last = ""
        self._out_search_sel = -1
        self._out_search_save = 0
        self._out_search_failed = False

        # Poll key input so background command output streams live. The 100 ms
        # now travels with the read itself — `read_key(timeout=0.1)` — rather
        # than being installed once on a window object as a side effect of
        # entering the loop, which is the difference between a setting and an
        # argument.

        # Bind I/O once.
        self._old_io = self._repl.io
        self._old_console_io = getattr(self._repl.console, "_io", None)
        self._repl.io = self._tui_io
        if self._old_console_io is not None:
            self._repl.console._io = self._tui_io

        # Initialize GraphicsEngine
        self._engine = GraphicsEngine()
        self._engine.open()
        # `open()` hides the cursor because a framebuffer engine draws the caret
        # as part of the image it owns. This shell does want a visible caret on
        # the input row, so the engine -- not curses -- shows it.
        self._engine.show_cursor(True)

        regions = self._layout.compute(rows, cols)
        self._log_surface.set_width(regions["console"].cols)
        self._output_surface.set_width(regions["output"].cols)

        # One layer per pane. There used to be a second, patterned layer
        # beneath each content layer as a "wallpaper", and it was the
        # compositing defect rather than a backdrop: a pattern cell is never
        # blank, so it can never be transparent, and it rendered straight
        # through the spaces *inside* every line of text — `░░py░<expr>░░`
        # instead of `  py <expr>`. A pane's field belongs to the pane: each
        # pane paints its own backdrop on its own layer, first — see
        # _PANE_PATTERN.
        self._layer_console = self._engine.create_layer("console", z=1)
        self._layer_output = self._engine.create_layer("output", z=3)
        self._layer_status = self._engine.create_layer("status", z=4)
        self._layer_input = self._engine.create_layer("input", z=5)

        # Resize layers to match pane regions
        self._layer_console.resize(regions["console"].rows, regions["console"].cols)
        self._layer_output.resize(regions["output"].rows, regions["output"].cols)
        self._layer_status.resize(regions["status"].rows, regions["status"].cols)
        self._layer_input.resize(regions["input"].rows, regions["input"].cols)
        self._place_layers(regions)

        # Register focus regions in TabManager
        self._tab_manager = TabManager()
        self._tab_manager.add_region(
            "output",
            regions["output"].top,
            regions["output"].left,
            regions["output"].cols,
            regions["output"].rows,
            tab_order=0,
            on_focus=lambda: setattr(self, "_focused_pane", "output"),
        )
        self._tab_manager.add_region(
            "console",
            regions["console"].top,
            regions["console"].left,
            regions["console"].cols,
            regions["console"].rows,
            tab_order=1,
            on_focus=lambda: setattr(self, "_focused_pane", "console"),
        )
        # Focus the default pane
        self._tab_manager.focus(self._focused_pane)

        # No curses windows. They used to be created per pane purely so `_blit`
        # could ask them their size to infer which layer to draw on -- nothing
        # ever wrote to them. ncurses is an input source here and no more.

        def _redraw() -> None:
            self._render_all(regions)

        def _resize(nrows: int, ncols: int) -> None:
            """Rebuild the layout and layers at ``nrows`` x ``ncols``."""
            nonlocal regions
            self._rows = nrows
            self._cols = ncols
            regions = self._layout.compute(nrows, ncols)
            self._log_surface.set_width(regions["console"].cols)
            self._output_surface.set_width(regions["output"].cols)
            # Resize engine layers
            if self._engine is not None:
                self._engine.handle_resize()
            if self._layer_console is not None:
                self._layer_console.resize(regions["console"].rows, regions["console"].cols)
            if self._layer_output is not None:
                self._layer_output.resize(regions["output"].rows, regions["output"].cols)
            if self._layer_status is not None:
                self._layer_status.resize(regions["status"].rows, regions["status"].cols)
            if self._layer_input is not None:
                self._layer_input.resize(regions["input"].rows, regions["input"].cols)
            self._place_layers(regions)
            # No clear here. The engine resized its own buffers above, and a
            # `\x1b[2J` at this spot would wipe the frame the redraw below is
            # about to draw. This used to be `stdscr.erase()`, which is where
            # the habit came from.
            _redraw()

        def _detect_resize() -> tuple[int, int] | None:
            """Return the kernel's window size if it changed, else ``None``.

            This is now the *only* resize mechanism, not a fallback. It began
            as one: ``readline`` (imported by the shell for line-mode history)
            claims SIGWINCH before curses starts, so ncurses skipped its own
            handler and ``getch()`` never returned ``KEY_RESIZE`` — polling
            recovered the event that ncurses had lost. With curses gone there
            is no handler left to lose it, and nothing else in this process
            installs one, so if this poll ever stopped running the window would
            simply never resize.

            It reports the size rather than writing it. This used to call
            ``curses.resizeterm()``, which cleared the screen as a side effect
            of telling ncurses the new geometry. The size belongs to whoever
            owns the framebuffer, which is the engine.
            """
            try:
                ts = os.get_terminal_size()
            except OSError:
                return None
            nrows, ncols = ts.lines, ts.columns
            if nrows <= 0 or ncols <= 0:
                return None
            if nrows == self._rows and ncols == self._cols:
                return None
            return nrows, ncols

        # No clear at this spot, and there never was one that worked. The one
        # that used to be here — `stdscr.erase()` — emitted `\x1b[2J` *lazily*:
        # ncurses only marked its window blank, then flushed that pending clear
        # on some later `getch()`. By then the engine had drawn a frame and
        # recorded it in its emitted buffer, so the diff saw "no change" and the screen
        # stayed blank. Clearing is a one-time thing and `GraphicsEngine.open()`
        # does it eagerly, before the first frame exists to be wiped.
        _redraw()

        while self._running:
            # The only thing in the loop that can block, and only for 100 ms.
            # Everything else the engine draws, so nothing repaints over it.
            try:
                key = self._session.read_key(timeout=0.1)
            except KeyboardInterrupt:
                self._running = False
                break
            except EOFError:
                # The input stream is finished (terminal hangup, drained
                # stdin). End the session: the byte source raises rather than
                # fabricating Esc, which would otherwise repaint every poll
                # tick forever at full CPU.
                self._running = False
                break

            if key is None:
                # Poll tick: refresh every pane so background output streams live.
                if self._active_thread is not None and not self._active_thread.is_alive():
                    self._active_cmd = None
                    self._active_thread = None
                size = _detect_resize()
                if size is not None:
                    _resize(*size)
                # A prompt can be opened by a command thread while this loop
                # sits in read_key(), so the tick is what shows it. `_render_all`
                # draws the open overlay in the same pass as the panes, meaning
                # one repaint both puts it up and composites it — no second
                # write for it to race against.
                _redraw()
                continue

            # ── Interactive prompts ──────────────────────────────────────
            # Each overlay owns what a key means; the loop owns drawing them.
            # `handled` decides the repaint: a consumed key rewrites pane
            # content so everything redraws, an unconsumed one leaves the panes
            # alone and only the overlay is repainted.
            if self._select_event.is_set():
                if self._handle_select_key(key, regions):
                    # Consumed: the panes changed, so repaint them. `_render_all`
                    # draws a still-open overlay in that same pass.
                    _redraw()
                else:
                    # Unconsumed: the panes are untouched by design, so only the
                    # box is redrawn — and composited, or it never reaches the
                    # terminal.
                    self._paint_overlay(regions)
                continue

            if self._confirm_event.is_set():
                if self._handle_confirm_key(key):
                    _redraw()
                else:
                    self._paint_overlay(regions)
                continue

            if self._ask_event.is_set():
                if self._handle_ask_key(key):
                    _redraw()
                else:
                    self._paint_overlay(regions)
                continue

            if self._searching:
                # Incremental history search: Ctrl+R scans backward (older),
                # Ctrl+S / Ctrl+F scan forward (newer).  Typing a character
                # restarts from the far end of the current direction.
                if key.kind == KEY_CTRL and key.char in ("r", "s", "f"):  # direction
                    self._search_fwd = key.char in ("s", "f")
                    if self._search_fwd:
                        self._search_idx = self._search_idx + 1 if self._search_idx >= 0 else 0
                    else:
                        self._search_idx -= 1
                elif key.kind in _CANCEL:  # Esc / Ctrl+G / Ctrl+C — cancel
                    self._end_search(restore=True)
                elif key.kind == KEY_ENTER:
                    self._end_search(restore=False)
                elif key.kind == KEY_BACKSPACE:
                    self._search_q = self._search_q[:-1]
                    self._search_idx = -1 if self._search_fwd else self._history_pos
                elif (ch := _printable(key)) is not None:
                    self._search_q += ch
                    self._search_idx = -1 if self._search_fwd else self._history_pos
                self._apply_search()
                self._render_status(regions["status"].cols)
                continue

            if self._out_searching:
                # Output-pane content search (/): typing refines the query
                # (n/N are literal here), Enter accepts, Esc/Ctrl+G/Ctrl+C
                # cancel.  After accepting, n/N at an empty prompt repeat the
                # search from the current match.
                if key.kind in _CANCEL:  # Esc / Ctrl+G / Ctrl+C — cancel
                    self._out_searching = False
                    self._out_search_q = ""
                    self._out_search_sel = -1
                    self._out_scroll = self._out_search_save
                    _redraw()
                elif key.kind == KEY_ENTER:  # Enter — accept, keep scroll at match
                    if self._out_search_q:
                        self._out_search_last = self._out_search_q
                    self._out_searching = False
                    self._out_search_q = ""
                    _redraw()
                elif key.kind == KEY_BACKSPACE:
                    self._out_search_q = self._out_search_q[:-1]
                    self._out_search_sel = -1
                    self._apply_out_search(regions["output"].rows)
                    _redraw()
                elif (ch := _printable(key)) is not None:
                    self._out_search_q += ch
                    self._apply_out_search(regions["output"].rows)
                    _redraw()
                continue

            if key.kind == KEY_ENTER:
                cmd = "".join(self._input_buf).strip()
                self._input_buf.clear()
                self._input_cursor = 0
                if cmd:
                    self._cmd_history.append(cmd)
                    self._history_pos = len(self._cmd_history)
                    if cmd in ("exit", "q", "quit"):
                        self._running = False
                        break
                    self._output_surface.write(f"λ {cmd}")

                    def _run() -> None:
                        with self._repl_lock:
                            self._repl._dispatch(cmd)

                    self._active_cmd = cmd
                    self._active_thread = threading.Thread(target=_run, daemon=True)
                    self._active_thread.start()
                _redraw()

            elif key.matches(KEY_UP) or _chord(key, _CTRL_UP_CHAR):
                # UP / Ctrl+P — previous history
                self._history_back()
                _redraw()

            elif key.matches(KEY_DOWN) or _chord(key, _CTRL_DOWN_CHAR):
                # DOWN / Ctrl+N — next history
                self._history_fwd()
                _redraw()

            elif key.matches(KEY_LEFT):
                if self._input_cursor > 0:
                    self._input_cursor -= 1
                    _redraw()

            elif key.matches(KEY_RIGHT):
                if self._input_cursor < len(self._input_buf):
                    self._input_cursor += 1
                    _redraw()

            elif key.kind == KEY_ALT and key.char in ("f", "b", "d"):
                # Alt+f/b/d — word forward / word back / delete word. The whole
                # sequence was resolved by the time `key` was built, so what
                # used to be a second, differently-bounded read inside the
                # branch is now a comparison. A lone Esc never reaches here:
                # it stays KEY_ESC and falls through to the refresh below, which
                # is what it did before.
                if key.char == "f":
                    self._move_word_forward()
                elif key.char == "b":
                    self._move_word_backward()
                else:
                    self._delete_word_forward()
                _redraw()

            elif key.matches(KEY_LEFT, ctrl=True):
                # Ctrl+Left — word back, readline `backward-word`. Both it and
                # the plain arrow above name `kind="left"`; only `matches`
                # separates them, so this branch is correct wherever it sits.
                self._move_word_backward()
                _redraw()

            elif key.matches(KEY_RIGHT, ctrl=True):
                # Ctrl+Right — word forward, readline `forward-word`
                self._move_word_forward()
                _redraw()

            elif key.kind == KEY_BACKSPACE:
                if self._input_cursor > 0:
                    self._input_buf.pop(self._input_cursor - 1)
                    self._input_cursor -= 1
                    _redraw()

            elif key.kind == KEY_DELETE:
                self._delete_at_cursor()
                _redraw()

            elif key.kind == KEY_TAB:  # Tab completion (commands or paths)
                self._complete()
                _redraw()

            elif _chord(key, "l"):  # Ctrl+L clear
                self._output_surface.clear()
                self._out_scroll = 0
                _redraw()

            elif _chord(key, "o"):  # Ctrl+O — cycle focus between output and log panes
                self._tab_manager.focus_next()
                self._scroll_target = 0 if self._focused_pane == "output" else 1
                _redraw()

            elif key.kind == KEY_CTRL and key.char in ("r", "s"):
                # Ctrl+R / Ctrl+S — enter reverse / forward history search
                fwd = key.char == "s"
                self._searching = True
                self._search_fwd = fwd
                self._search_q = ""
                self._search_idx = -1 if fwd else self._history_pos
                self._search_save = (list(self._input_buf), self._input_cursor)
                _redraw()

            elif key.kind == KEY_CHAR and key.char == "/" and not self._input_buf:
                # / on empty prompt — output-pane search
                self._out_searching = True
                self._out_search_q = ""
                self._out_search_sel = -1
                self._out_search_save = self._out_scroll
                self._out_search_failed = False
                _redraw()

            elif key.kind == KEY_PAGE_UP:
                if self._scroll_target == 1:
                    self._log_scroll += 10
                else:
                    self._out_scroll += 10
                _redraw()

            elif key.kind == KEY_PAGE_DOWN:
                if self._scroll_target == 1:
                    self._log_scroll = max(self._log_scroll - 10, 0)
                else:
                    self._out_scroll = max(self._out_scroll - 10, 0)
                _redraw()

            elif key.kind == KEY_RESIZE:
                # Nothing delivers this without ncurses having handled SIGWINCH
                # itself, and readline may have claimed the signal first anyway.
                # The polled kernel size is the reliable source either way; the
                # session's size is just the fallback `stdscr.getmaxyx()` was.
                size = _detect_resize() or self._session.size()
                _resize(*size)

            elif key.kind == KEY_HOME:  # Ctrl+A / Home — start of line
                self._move_home()
                _redraw()

            elif key.kind == KEY_END:  # Ctrl+E / End — end of line
                self._move_end()
                _redraw()

            elif _chord(key, "u"):  # Ctrl+U — kill to start of line
                self._kill_to_start()
                _redraw()

            elif _chord(key, "k"):  # Ctrl+K — kill to end of line
                self._kill_to_end()
                _redraw()

            elif _chord(key, "w"):  # Ctrl+W — delete word before caret
                self._delete_word_back()
                _redraw()

            elif _chord(key, "d"):  # Ctrl+D — delete char at caret
                self._delete_at_cursor()
                _redraw()

            elif _chord(key, "y"):  # Ctrl+Y — yank the most recent kill (repeat to cycle)
                self._yank()
                _redraw()

            elif _chord(key, "t"):  # Ctrl+T — transpose chars before/at caret
                self._transpose_chars()
                _redraw()

            elif (
                key.kind == KEY_CHAR
                and key.char in ("n", "N")
                and not self._input_buf
                and self._out_search_last
                and self._out_scroll > 0
            ):
                # n / N — repeat the last output-pane search.  Only at an
                # empty prompt while scrolled back (reading mode), so command
                # text keeps its 'n'/'N'.
                self._repeat_out_search(regions["output"].rows, fwd=key.char == "n")
                _redraw()

            elif key.kind == KEY_CTRL_C:  # Ctrl+C — interrupt the running command, else exit
                if self._active_thread is not None and self._active_thread.is_alive():
                    self._interrupt_active()
                    self._output_surface.write("^C")
                    _redraw()
                else:
                    self._running = False
                    break

            elif (text := _printable(key)) is not None:
                self._input_buf.insert(self._input_cursor, text)
                self._input_cursor += 1
                _redraw()

            else:
                # Background command output: refresh output pane each frame.
                self._blit(
                    "output",
                    self._output_surface.render(regions["output"].rows, self._out_scroll),
                )
                self._blit(
                    "console",
                    self._log_surface.render(regions["console"].rows, self._log_scroll),
                )

        # ── Restore ──
        if self._old_io is not None:
            self._repl.io = self._old_io
        if self._old_console_io is not None:
            self._repl.console._io = self._old_console_io
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        # No refresh here. The engine has already left the alternate screen and
        # homed the cursor; anything written after that would land on the real
        # screen and fight the shell's own output. Teardown after this returns
        # is `self._session.__exit__` restoring termios, which is the whole of
        # the teardown `curses.wrapper`'s `endwin()` used to perform beyond it.
