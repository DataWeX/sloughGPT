"""
TuiRepl — split-panel curses shell with three fixed regions.

Layout:
  ┌─ Console Panel (infra logs) ─────────────────── top 30% ─┐
  │ 16:56:37 INF [startup] Task queue initialized            │
  │ 16:56:38 WRN [runtime] Orphan process 21415 killed       │
  ├─ Shell Output (command results) ─────────────── remainder ┤
  │ $ models                                                  │
  │ gpt2   124M   loaded                                      │
  ├───────────────────────────────────────────────────────────┤
  │ [OUTPUT] LIVE  ai "2+2"                    120x24 (fixed) │
  │ λ _                                           (fixed line) │
  └───────────────────────────────────────────────────────────┘

Follows the split-window model: ``pane.PaneLayout`` (pure geometry) is the
arranger, ``surface`` objects draw their own content into the assigned
regions, and this module is only the *display layer* — it blits surfaces
onto the curses screen and forwards keys.  Command dispatch runs on a
background thread so the UI stays responsive.
"""

from __future__ import annotations

import os
import curses
import threading
from typing import TYPE_CHECKING

try:
    import ctypes
    _SET_ASYNC_EXC = ctypes.pythonapi.PyThreadState_SetAsyncExc
except (ImportError, AttributeError):
    _SET_ASYNC_EXC = None

from .pane import Pane, PaneLayout
from .surface import LogSurface, RenderLine, STYLE_INFO, STYLE_WARN, STYLE_ERROR, STYLE_DEBUG, STYLE_CRITICAL, TextSurface

if TYPE_CHECKING:
    from .repl import ShellREPL
    from .log_buffer import LogBuffer

# ── Colour pairs ──────────────────────────────────────────────────────────

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

# Python's ``_curses`` does not expose the ncurses extended-keypad
# modifiers; the folded key values for Ctrl+Left / Ctrl+Right on an
# xterm-256color terminal are stable (554 / 569) but some ncurses builds
# name them, so prefer the named constant when present.
_KEY_CTRL_LEFT = getattr(curses, "KEY_CTRL_LEFT", 554)
_KEY_CTRL_RIGHT = getattr(curses, "KEY_CTRL_RIGHT", 569)


def _init_pairs() -> None:
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(_P_LOG_INFO, curses.COLOR_GREEN, -1)
        curses.init_pair(_P_LOG_WARN, curses.COLOR_YELLOW, -1)
        curses.init_pair(_P_LOG_ERROR, curses.COLOR_RED, -1)
        curses.init_pair(_P_LOG_DEBUG, curses.COLOR_CYAN, -1)
        curses.init_pair(_P_LOG_CRITICAL, curses.COLOR_MAGENTA, -1)
        curses.init_pair(_P_PROMPT, curses.COLOR_CYAN, -1)
        curses.init_pair(_P_BORDER, curses.COLOR_WHITE, -1)


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


_ESC_FINALS = frozenset(
    "ABCDEFHZ~"  # arrow / function-sequence final bytes
)


def _read_escape_remainder(stdscr, alt_map, restore_ms: int = 100):
    """Resolve what follows a bare ESC key press.

    Curses reports Alt+<key> and modified-arrow chords as an ESC prefix
    followed by the rest of the sequence (it only folds them into a single
    key when the terminal's terminfo names them).  This polls the next few
    bytes with a 0 ms timeout so a lone Esc resolves instantly and Alt
    chords are decoded.

    Args:
        stdscr: the curses window; ``getch`` is polled non-blocking.
        alt_map: dict of printable Alt+char → callback name string.
        restore_ms: input timeout (ms) to restore before returning; the
            caller's poll interval.

    Returns:
        the decoded action string (``alt:f``, ``seq:ctrl-left``, ...) or
        ``None`` when the press was a lone Esc.

    Side effects:
        - temporarily switches the window to a 0 ms input timeout, restoring
          ``restore_ms`` before returning.
    """
    stdscr.timeout(0)
    try:
        first = stdscr.getch()
        if first in (-1, 27):
            return None
        if first == ord("["):
            parts = []
            while True:
                b = stdscr.getch()
                if b == -1:
                    break
                parts.append(chr(b))
                if chr(b) in _ESC_FINALS:
                    break
            seq = "".join(parts)
            if seq.endswith("C") and "5" in seq:
                return "seq:ctrl-right"
            if seq.endswith("D") and "5" in seq:
                return "seq:ctrl-left"
            return None
        if 32 <= first < 127:
            name = alt_map.get(chr(first))
            return f"alt:{name}" if name else None
        return None
    finally:
        stdscr.timeout(restore_ms)


# ── TuiIo — routes command output into a TextSurface ──────────────────────

class TuiIo:
    """ShellIO-compatible writer that feeds a TextSurface."""

    def __init__(self, surface: TextSurface) -> None:
        self._surface = surface

    def write(self, text: str, end: str = "\n") -> None:
        self._surface.write(text, end)

    def flush(self) -> None:
        pass

    def read(self, prompt: str = "") -> str:
        raise NotImplementedError("input comes from the curses event loop")


# ── TuiRepl ───────────────────────────────────────────────────────────────

class TuiRepl:
    """Three-pane curses shell composing a PaneLayout with surfaces."""

    CONSOLE_RATIO = 0.3
    CONSOLE_MIN = 4
    OUTPUT_MIN = 6

    def __init__(self, repl: ShellREPL, log_buffer: LogBuffer) -> None:
        self._repl = repl
        self._log_buffer = log_buffer
        self._running = False
        self._cmd_history: list[str] = []
        self._history_pos = 0
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
        self._tui_io = TuiIo(self._output_surface)

        self._old_io = None
        self._old_console_io = None
        self._log_surface = LogSurface(log_buffer)

        # Panes (arranger) — pure geometry, no rendering knowledge.
        self._layout = PaneLayout([
            Pane("console", ratio=self.CONSOLE_RATIO, min_rows=self.CONSOLE_MIN),
            Pane("output", ratio=1.0 - self.CONSOLE_RATIO, min_rows=self.OUTPUT_MIN),
            Pane("status", fixed=1),
            Pane("input", fixed=1),
        ])

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

    def run(self) -> None:
        """Enter curses mode and start the event loop."""
        self._running = True
        history = getattr(self._repl, "_history", None)
        self._cmd_history = list(history) if history else []
        self._history_pos = len(self._cmd_history)

        try:
            from domains.logging.cli_logger import set_cli_terminal
        except ImportError:
            set_cli_terminal = None
        if set_cli_terminal is not None:
            set_cli_terminal(False)
        try:
            curses.wrapper(self._main)
        finally:
            if set_cli_terminal is not None:
                set_cli_terminal(True)

    # ── Rendering ─────────────────────────────────────────────────────────

    def _blit(self, win: curses._CursesWindow, lines: list[RenderLine]) -> None:
        try:
            win.erase()
        except curses.error:
            return
        h, w = win.getmaxyx()
        for y, ln in enumerate(lines):
            if y >= h:
                break
            pair = _STYLE_PAIRS.get(ln.style)
            attr = curses.color_pair(pair) if pair else 0
            text = ln.text[: w - 1]
            try:
                if text:
                    win.addstr(y, 0, text, attr)
            except curses.error:
                pass
        try:
            win.refresh()
        except curses.error:
            pass

    def _render_all(self, regions, win_console, win_output, win_status, win_input) -> None:
        self._blit(win_console, self._log_surface.render(regions["console"].rows, self._log_scroll))
        self._blit(win_output, self._output_surface.render(regions["output"].rows, self._out_scroll))
        self._render_status(win_status, regions["status"].cols)
        self._render_input(win_input, regions["input"].cols)

    def _input_view(self, cols: int, buf: str, caret: int) -> tuple[str, int]:
        """Compute the visible input line and its caret column.

        ``cols`` is the window width; returns ``(line, caret_col)`` where
        ``line`` fits the window (prompt + ``buf``) and ``caret_col`` is the
        absolute window column the cursor should occupy.  When the buffer is
        wider than the window the view scrolls horizontally so the caret
        stays visible; a buffer at the window edge reveals its tail.
        """
        prompt = "\u03bb "
        max_w = max(cols - len(prompt) - 1, 0)
        caret = min(max(caret, 0), len(buf))
        if len(buf) <= max_w:
            return buf, len(prompt) + caret
        start = min(caret, len(buf) - max_w)
        return buf[start:start + max_w], len(prompt) + (caret - start)

    def _render_input(self, win: curses._CursesWindow, cols: int) -> None:
        """Draw the command line: prompt plus the buffered input, with the
        terminal cursor parked on the caret."""
        try:
            win.erase()
            prompt = "\u03bb "
            buf = "".join(self._input_buf)
            line, caret_col = self._input_view(cols, buf, self._input_cursor)
            win.addstr(0, 0, prompt, curses.color_pair(_P_PROMPT))
            if line:
                win.addstr(0, len(prompt), line)
            try:
                win.move(0, min(max(caret_col, 0), max(cols - 1, 0)))
            except curses.error:
                pass
            win.refresh()
        except curses.error:
            pass

    def _render_status(self, win: curses._CursesWindow, cols: int) -> None:
        """Draw the chrome bar: scroll focus, live/scroll state, active
        command, terminal size, or the incremental-search prompt while
        active (``reverse-i-search`` for Ctrl+R, ``forward-i-search`` for
        Ctrl+S)."""
        try:
            win.erase()
        except curses.error:
            return
        if cols <= 0:
            return
        try:
            if self._searching:
                label = "forward-i-search" if self._search_fwd else "reverse-i-search"
                if self._search_failed:
                    label = f"failed {label}"
                prompt = f"({label})\u0060{self._search_q}\u0060:"
                win.addstr(0, 0, prompt[: cols - 2], curses.color_pair(_P_PROMPT))
                win.refresh()
                return
            if self._out_searching:
                label = "output-search"
                if self._out_search_failed:
                    label = f"failed {label}"
                prompt = f"({label})\u0060{self._out_search_q}\u0060:"
                win.addstr(0, 0, prompt[: cols - 2], curses.color_pair(_P_PROMPT))
                win.refresh()
                return
            target = "OUTPUT" if self._scroll_target == 0 else "LOG"
            scroll = self._out_scroll if self._scroll_target == 0 else self._log_scroll
            scroll_txt = f"SCROLL \u2191{scroll}" if scroll > 0 else "LIVE"
            head = f"[{target}]"
            win.addstr(0, 0, head, curses.color_pair(_P_PROMPT))
            col = len(head) + 1
            if col < cols:
                attr = curses.color_pair(_P_LOG_WARN) if scroll > 0 else curses.color_pair(_P_LOG_INFO)
                win.addstr(0, col, scroll_txt, attr)
                col += len(scroll_txt) + 1
            if col < cols and self._active_cmd:
                cmd = self._active_cmd[: cols - col - 4]
                win.addstr(0, col, cmd)
                col += len(cmd) + 1
            suffix = f"{cols}x{self._rows}"
            if cols - len(suffix) - 1 >= col:
                win.addstr(0, cols - len(suffix) - 1, suffix, curses.color_pair(_P_BORDER))
            win.refresh()
        except curses.error:
            pass

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
        """
        if self._cmd_history and self._history_pos > 0:
            self._history_pos -= 1
            self._input_buf = list(self._cmd_history[self._history_pos])
            self._input_cursor = len(self._input_buf)

    def _history_fwd(self) -> None:
        """Step to the next (newer) history entry; past the newest clears
        the input row (readline ``next-history``)."""
        if self._cmd_history and self._history_pos < len(self._cmd_history) - 1:
            self._history_pos += 1
            self._input_buf = list(self._cmd_history[self._history_pos])
            self._input_cursor = len(self._input_buf)
        else:
            self._history_pos = len(self._cmd_history)
            self._input_buf.clear()
            self._input_cursor = 0

    def _move_word_forward(self) -> None:
        """Move the caret to the end of the next word (Alt+F / Ctrl+Right).

        Words are maximal runs of non-whitespace characters.  From the
        caret, skip any whitespace, then advance to just past the end of
        the word; from mid-word this lands at the end of the current word
        (readline ``forward-word`` semantics).
        """
        n = len(self._input_buf)
        i = self._input_cursor
        while i < n and self._input_buf[i] == " ":
            i += 1
        while i < n and self._input_buf[i] != " ":
            i += 1
        self._input_cursor = i

    def _move_word_backward(self) -> None:
        """Move the caret to the start of the current or previous word
        (Alt+B / Ctrl+Left).

        Mirrors readline's ``backward-word``: from the caret, skip back
        over any whitespace, then back over the word, landing at its first
        character.
        """
        i = self._input_cursor
        while i > 0 and self._input_buf[i - 1] == " ":
            i -= 1
        while i > 0 and self._input_buf[i - 1] != " ":
            i -= 1
        self._input_cursor = i

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
        self._input_buf = self._input_buf[self._input_cursor:]
        self._input_cursor = 0
        self._push_kill(killed)

    def _kill_to_end(self) -> None:
        """Delete the text from the caret to the end (Ctrl+K); pushed to ring."""
        killed = "".join(self._input_buf[self._input_cursor:])
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

        Mirrors readline's ``unix-word-rubout``: from the caret, skip back
        over whitespace, then back over the word, and delete that range.
        The deleted word is pushed to the kill ring.
        """
        end = self._input_cursor
        i = end
        while i > 0 and self._input_buf[i - 1] == " ":
            i -= 1
        while i > 0 and self._input_buf[i - 1] != " ":
            i -= 1
        killed = "".join(self._input_buf[i:end])
        del self._input_buf[i:end]
        self._input_cursor = i
        self._push_kill(killed)

    def _delete_word_forward(self) -> None:
        """Delete the word (and any leading whitespace) after the caret.

        Mirrors readline's ``kill-word``: from the caret, skip forward
        over whitespace, then over the word, and delete that range; the
        caret does not move.  The deleted word is pushed to the kill ring.
        """
        start = self._input_cursor
        i = start
        n = len(self._input_buf)
        while i < n and self._input_buf[i] == " ":
            i += 1
        while i < n and self._input_buf[i] != " ":
            i += 1
        killed = "".join(self._input_buf[start:i])
        del self._input_buf[start:i]
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
            here = "".join(self._input_buf[self._yank_start:self._yank_start + len(prev)])
            if here != prev:
                self._yank_active = False  # line was edited since — start fresh
        if not self._yank_active:
            self._yank_active = True
            self._yank_idx = len(self._kill_ring) - 1
            self._yank_start = self._input_cursor
            text = self._kill_ring[self._yank_idx]
        else:
            prev = self._kill_ring[self._yank_idx]
            del self._input_buf[self._yank_start:self._yank_start + len(prev)]
            self._input_cursor = self._yank_start
            self._yank_idx = max(self._yank_idx - 1, 0)
            text = self._kill_ring[self._yank_idx]
        self._input_buf[self._yank_start:self._yank_start] = list(text)
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
            for idx in range(0, start):
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

    def _main(self, stdscr: curses._CursesWindow) -> None:
        _init_pairs()
        curses.curs_set(1)
        curses.raw()
        stdscr.keypad(True)
        try:
            curses.set_escdelay(25)
        except Exception:
            pass

        rows, cols = stdscr.getmaxyx()
        self._rows = rows
        self._cols = cols
        self._input_buf: list[str] = []
        self._input_cursor = 0
        self._history_pos = len(self._cmd_history)
        self._out_scroll = 0
        self._log_scroll = 0
        self._scroll_target = 0  # 0 = output pane, 1 = log pane
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

        # Poll key input so background command output streams live.
        stdscr.timeout(100)

        # Bind I/O once.
        self._old_io = self._repl.io
        self._old_console_io = getattr(self._repl.console, "_io", None)
        self._repl.io = self._tui_io
        if self._old_console_io is not None:
            self._repl.console._io = self._tui_io

        regions = self._layout.compute(rows, cols)
        self._log_surface.set_width(regions["console"].cols)
        self._output_surface.set_width(regions["output"].cols)
        win_console = curses.newwin(regions["console"].rows, regions["console"].cols, regions["console"].top, regions["console"].left)
        win_output = curses.newwin(regions["output"].rows, regions["output"].cols, regions["output"].top, regions["output"].left)
        win_status = curses.newwin(regions["status"].rows, regions["status"].cols, regions["status"].top, regions["status"].left)
        win_input = curses.newwin(regions["input"].rows, regions["input"].cols, regions["input"].top, regions["input"].left)

        def _redraw() -> None:
            self._render_all(regions, win_console, win_output, win_status, win_input)

        def _resize(nrows: int, ncols: int) -> None:
            """Rebuild the layout and windows at ``nrows`` x ``ncols``.

            The terminal size changed (ncurses KEY_RESIZE or a poll-detected
            ``SIGWINCH`` that readline claimed); recompute pane regions, size
            the surfaces, recreate the curses windows and redraw everything.
            """
            nonlocal regions, win_console, win_output, win_status, win_input
            self._rows = nrows
            self._cols = ncols
            regions = self._layout.compute(nrows, ncols)
            self._log_surface.set_width(regions["console"].cols)
            self._output_surface.set_width(regions["output"].cols)
            win_console = curses.newwin(regions["console"].rows, regions["console"].cols, regions["console"].top, regions["console"].left)
            win_output = curses.newwin(regions["output"].rows, regions["output"].cols, regions["output"].top, regions["output"].left)
            win_status = curses.newwin(regions["status"].rows, regions["status"].cols, regions["status"].top, regions["status"].left)
            win_input = curses.newwin(regions["input"].rows, regions["input"].cols, regions["input"].top, regions["input"].left)
            _redraw()

        def _detect_resize(stdscr) -> bool:
            """Return True and resync curses when the kernel window size
            changed but ncurses never reported KEY_RESIZE.

            ``readline`` (imported by the shell for line-mode history) claims
            SIGWINCH before curses starts, so ncurses skips its own handler and
            getch() never returns KEY_RESIZE.  Polling the terminal size each
            poll tick recovers the event.  Without readline this is a no-op
            because KEY_RESIZE is delivered normally.
            """
            try:
                ts = os.get_terminal_size()
            except OSError:
                return False
            nrows, ncols = ts.lines, ts.columns
            if nrows <= 0 or ncols <= 0:
                return False
            if nrows == self._rows and ncols == self._cols:
                return False
            try:
                curses.resizeterm(nrows, ncols)
            except curses.error:
                return False
            self._rows = nrows
            self._cols = ncols
            return True

        _redraw()

        while self._running:
            try:
                ch = stdscr.getch()
            except KeyboardInterrupt:
                self._running = False
                break

            if ch == -1:
                # Poll tick: refresh every pane so background output streams
                # live and the status/input rows survive curses' screen clears.
                if self._active_thread is not None and not self._active_thread.is_alive():
                    self._active_cmd = None
                    self._active_thread = None
                if _detect_resize(stdscr):
                    _resize(self._rows, self._cols)
                self._blit(win_output, self._output_surface.render(regions["output"].rows, self._out_scroll))
                self._blit(win_console, self._log_surface.render(regions["console"].rows, self._log_scroll))
                self._render_status(win_status, regions["status"].cols)
                self._render_input(win_input, regions["input"].cols)
                continue

            if self._searching:
                # Incremental history search: Ctrl+R scans backward (older),
                # Ctrl+S / Ctrl+F scan forward (newer).  Typing a character
                # restarts from the far end of the current direction.
                if ch in (18, 19, 6):  # Ctrl+R / Ctrl+S / Ctrl+F — direction
                    self._search_fwd = ch in (19, 6)
                    if self._search_fwd:
                        self._search_idx = self._search_idx + 1 if self._search_idx >= 0 else 0
                    else:
                        self._search_idx -= 1
                elif ch in (27, 7, 3):  # Esc / Ctrl+G / Ctrl+C — cancel
                    self._end_search(restore=True)
                elif ch == ord("\n"):
                    self._end_search(restore=False)
                elif ch in (curses.KEY_BACKSPACE, 127, 8):
                    self._search_q = self._search_q[:-1]
                    self._search_idx = -1 if self._search_fwd else self._history_pos
                elif 32 <= ch < 127:
                    self._search_q += chr(ch)
                    self._search_idx = -1 if self._search_fwd else self._history_pos
                self._apply_search()
                self._render_status(win_status, regions["status"].cols)
                continue

            if self._out_searching:
                # Output-pane content search (/): typing refines the query
                # (n/N are literal here), Enter accepts, Esc/Ctrl+G/Ctrl+C
                # cancel.  After accepting, n/N at an empty prompt repeat the
                # search from the current match.
                if ch in (27, 7, 3):  # Esc / Ctrl+G / Ctrl+C — cancel
                    self._out_searching = False
                    self._out_search_q = ""
                    self._out_search_sel = -1
                    self._out_scroll = self._out_search_save
                    _redraw()
                elif ch == ord("\n"):  # Enter — accept, keep scroll at match
                    if self._out_search_q:
                        self._out_search_last = self._out_search_q
                    self._out_searching = False
                    self._out_search_q = ""
                    _redraw()
                elif ch in (curses.KEY_BACKSPACE, 127, 8):
                    self._out_search_q = self._out_search_q[:-1]
                    self._out_search_sel = -1
                    self._apply_out_search(regions["output"].rows)
                    _redraw()
                elif 32 <= ch < 127:
                    self._out_search_q += chr(ch)
                    self._apply_out_search(regions["output"].rows)
                    _redraw()
                continue

            if ch == ord("\n"):
                cmd = "".join(self._input_buf).strip()
                self._input_buf.clear()
                self._input_cursor = 0
                if cmd:
                    self._cmd_history.append(cmd)
                    self._history_pos = len(self._cmd_history)
                    if cmd in ("exit", "q", "quit"):
                        self._running = False
                        break
                    self._output_surface.write(f"\u03bb {cmd}")
                    def _run() -> None:
                        with self._repl_lock:
                            self._repl._dispatch(cmd)
                    self._active_cmd = cmd
                    self._active_thread = threading.Thread(target=_run, daemon=True)
                    self._active_thread.start()
                _redraw()

            elif ch in (curses.KEY_UP, 16):  # UP / Ctrl+P — previous history
                self._history_back()
                _redraw()

            elif ch in (curses.KEY_DOWN, 14):  # DOWN / Ctrl+N — next history
                self._history_fwd()
                _redraw()

            elif ch == curses.KEY_LEFT:
                if self._input_cursor > 0:
                    self._input_cursor -= 1
                    _redraw()

            elif ch == curses.KEY_RIGHT:
                if self._input_cursor < len(self._input_buf):
                    self._input_cursor += 1
                    _redraw()

            elif ch == 27:  # Esc / Alt+<key> / Ctrl+<arrow> prefix
                action = _read_escape_remainder(stdscr, {"f": "fwd", "b": "bwd", "d": "delword"})
                if action == "alt:fwd":
                    self._move_word_forward()
                    _redraw()
                elif action == "alt:bwd":
                    self._move_word_backward()
                    _redraw()
                elif action == "alt:delword":
                    self._delete_word_forward()
                    _redraw()
                elif action == "seq:ctrl-right":
                    self._move_word_forward()
                    _redraw()
                elif action == "seq:ctrl-left":
                    self._move_word_backward()
                    _redraw()

            elif ch in (_KEY_CTRL_LEFT, _KEY_CTRL_RIGHT):  # Ctrl+Left / Ctrl+Right
                if ch == _KEY_CTRL_RIGHT:
                    self._move_word_forward()
                else:
                    self._move_word_backward()
                _redraw()

            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                if self._input_cursor > 0:
                    self._input_buf.pop(self._input_cursor - 1)
                    self._input_cursor -= 1
                    _redraw()

            elif ch == curses.KEY_DC:
                self._delete_at_cursor()
                _redraw()

            elif ch == 9:  # Tab completion (commands or paths)
                self._complete()
                _redraw()

            elif ch == 12:  # Ctrl+L clear
                self._output_surface.clear()
                self._out_scroll = 0
                _redraw()

            elif ch == 15:  # Ctrl+O — toggle log/output focus for scrollback
                self._scroll_target = 1 - self._scroll_target
                _redraw()

            elif ch in (18, 19):  # Ctrl+R / Ctrl+S — enter reverse / forward history search
                self._searching = True
                self._search_fwd = ch == 19
                self._search_q = ""
                self._search_idx = -1 if ch == 19 else self._history_pos
                self._search_save = (list(self._input_buf), self._input_cursor)
                _redraw()

            elif ch == 47 and not self._input_buf:  # / on empty prompt — output-pane search
                self._out_searching = True
                self._out_search_q = ""
                self._out_search_sel = -1
                self._out_search_save = self._out_scroll
                self._out_search_failed = False
                _redraw()

            elif ch == curses.KEY_PPAGE:
                if self._scroll_target == 1:
                    self._log_scroll += 10
                else:
                    self._out_scroll += 10
                _redraw()

            elif ch == curses.KEY_NPAGE:
                if self._scroll_target == 1:
                    self._log_scroll = max(self._log_scroll - 10, 0)
                else:
                    self._out_scroll = max(self._out_scroll - 10, 0)
                _redraw()

            elif ch == curses.KEY_RESIZE:
                _resize(*stdscr.getmaxyx())

            elif ch in (1, curses.KEY_HOME):  # Ctrl+A / Home — start of line
                self._move_home()
                _redraw()

            elif ch in (5, curses.KEY_END):  # Ctrl+E / End — end of line
                self._move_end()
                _redraw()

            elif ch == 21:  # Ctrl+U — kill to start of line
                self._kill_to_start()
                _redraw()

            elif ch == 11:  # Ctrl+K — kill to end of line
                self._kill_to_end()
                _redraw()

            elif ch == 23:  # Ctrl+W — delete word before caret
                self._delete_word_back()
                _redraw()

            elif ch == 4:  # Ctrl+D — delete char at caret
                self._delete_at_cursor()
                _redraw()

            elif ch == 25:  # Ctrl+Y — yank the most recent kill (repeat to cycle)
                self._yank()
                _redraw()

            elif ch == 20:  # Ctrl+T — transpose chars before/at caret
                self._transpose_chars()
                _redraw()

            elif ch in (110, 78) and not self._input_buf and self._out_search_last and self._out_scroll > 0:
                # n / N — repeat the last output-pane search.  Only at an
                # empty prompt while scrolled back (reading mode), so command
                # text keeps its 'n'/'N'.
                self._repeat_out_search(regions["output"].rows, fwd=ch == 110)
                _redraw()

            elif ch == 3:  # Ctrl+C — interrupt the running command, else exit
                if self._active_thread is not None and self._active_thread.is_alive():
                    self._interrupt_active()
                    self._output_surface.write("^C")
                    _redraw()
                else:
                    self._running = False
                    break

            elif 32 <= ch < 127:
                self._input_buf.insert(self._input_cursor, chr(ch))
                self._input_cursor += 1
                _redraw()

            else:
                # Background command output: refresh output pane each frame.
                self._blit(win_output, self._output_surface.render(regions["output"].rows, self._out_scroll))
                self._blit(win_console, self._log_surface.render(regions["console"].rows, self._log_scroll))

        # ── Restore ──
        if self._old_io is not None:
            self._repl.io = self._old_io
        if self._old_console_io is not None:
            self._repl.console._io = self._old_console_io
        if rows > 0:
            stdscr.move(rows - 1, 0)
        stdscr.refresh()
