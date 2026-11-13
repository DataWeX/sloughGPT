"""Persistent ANSI terminal emulator backed by a ``Framebuffer``.

``ShellREPL(use_tui=True)`` runs curses, and curses requires a real TTY — which
the uvicorn process does not have — so the TUI runs in a child process behind a
pseudo-terminal.  What returns over the pty master is an ANSI byte stream, while
the browser needs cells.  This module is that seam: feed it bytes and it
maintains a :class:`~domain.shell._internal.graphics.Framebuffer`, which
:mod:`~domain.shell._internal.screen_wire` can diff and serialise straight onto
the wire.

::

    ShellREPL(use_tui=True)   child process, needs a real TTY
        └─ ANSI bytes → pty slave
             └─ pty master → TermScreen.feed()  → Framebuffer   ← this module
                  └─ screen_wire.diff_frame()   → frame dict
                       └─ transport → browser → ScreenBuffer → ScreenGrid

The cursor/erase/scroll subset was proven by
``packages/core-py/tests/test_shell_tui_live.py``, which emulates a terminal in
order to assert on the rendered screen.  This lifts that model into production
and adds the two things a test harness can skip but a live pty cannot:

* **Styling** — SGR is parsed into ``fg``/``bg``/``attr`` so the browser
  receives the TUI's colours rather than a monochrome grid.
* **Fragment tolerance** — ``os.read`` on a pty master splits *both* multi-byte
  UTF-8 characters and escape sequences across reads.  Decoding per chunk and
  discarding the partial tail corrupts the screen exactly as often as the
  kernel splits a buffer: rare enough to pass every test, common enough to
  break in production.  Incomplete input is therefore retained until the bytes
  that finish it arrive.

Two deliberate constraints, because both are cheap to violate and expensive to
debug later:

* **Only ``Framebuffer``'s public API is used** (``get``/``set``/``put``/
  ``clear``/``resize``/``snapshot``/``restore``), matching the boundary
  ``screen_wire`` already draws: no module outside ``graphics`` reaches into
  ``_cells``.
* **No lock — this type is single-threaded by design.**  ``feed()`` mutates the
  buffer in place, so a frame must be *taken* on the same thread that fed it
  (read pty → ``feed()`` → ``diff_frame()`` → hand the immutable dict onward).
  The frame, not the screen, is what crosses the thread boundary.
"""

from __future__ import annotations

import codecs
import unicodedata

from .graphics import Attr, Cell, Color, Framebuffer, rgb_to_ansi256

# SGR parameter → attribute bit set.
_SGR_ATTR: dict[int, int] = {
    1: int(Attr.BOLD),
    2: int(Attr.DIM),
    3: int(Attr.ITALIC),
    4: int(Attr.UNDERLINE),
    5: int(Attr.BLINK),
    7: int(Attr.REVERSE),
    8: int(Attr.HIDDEN),
    9: int(Attr.STRIKETHROUGH),
}

# SGR parameter → attribute bits cleared.  21 is not officially "bold off"
# (xterm reads it as a second underline style), but it is written that way by
# enough software that treating it as bold-off is the compatible reading.
_SGR_OFF: dict[int, int] = {
    21: int(Attr.BOLD) | int(Attr.DIM),
    22: int(Attr.BOLD) | int(Attr.DIM),
    23: int(Attr.ITALIC),
    24: int(Attr.UNDERLINE),
    25: int(Attr.BLINK),
    27: int(Attr.REVERSE),
    28: int(Attr.HIDDEN),
    29: int(Attr.STRIKETHROUGH),
}

_TAB_STOP = 8

# An unterminated OSC/DCS (or a parameter run with no final byte) would
# otherwise pin ``_pending`` forever, letting a truncated or bogus child
# stream grow the buffer without limit *and* force a full rescan of it on
# every read — O(n²) across the stream.  Nothing legitimately this long
# exists, so exceeding it means the sequence is garbage and is dropped.
_MAX_PENDING = 1 << 13


def _char_width(ch: str) -> int:
    """Display columns occupied by ``ch`` — 2 for wide (CJK), else 1.

    Combining marks are not special-cased: ``east_asian_width`` reports them as
    ambiguous, so they occupy one cell, which is what the proven
    ``test_shell_tui_live`` model already does.  curses does not emit them.
    """
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _int_at(parts: list[str], i: int, default: int = 0) -> int:
    """Raw parameter at ``i`` (unlike ``_csi``'s ``num``, 0 stays 0)."""
    try:
        return int(parts[i])
    except (IndexError, ValueError):
        return default


class TermScreen:
    """A persistent ANSI terminal emulator backed by a ``Framebuffer``.

    Consume bytes with :meth:`feed`, read the screen with :meth:`row`/
    :meth:`text`, and ship it with ``screen_wire.snapshot_frame`` /
    ``diff_frame``.  The buffer mutates in place, so snapshot on the thread
    that fed it (see module docstring).
    """

    def __init__(self, rows: int = 24, cols: int = 80) -> None:
        self.rows = rows
        self.cols = cols
        self.framebuffer = Framebuffer(rows, cols)

        # Cursor and scroll region.
        self.r = 0
        self.c = 0
        self.scroll_top = 0
        self.scroll_bottom = rows - 1
        self.save_r = 0
        self.save_c = 0

        # The pen: what the next printed character is drawn with.
        self.fg = int(Color.DEFAULT)
        self.bg = int(Color.DEFAULT)
        self.attr = int(Attr.NONE)

        # Primary-screen contents, saved while the alternate screen is shown.
        self._primary: Framebuffer | None = None
        self._primary_cursor: tuple[int, int] = (0, 0)

        # Fragment tolerance — see module docstring.
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._pending = ""

    # ── queries ───────────────────────────────────────────────────────────

    def row(self, r: int) -> str:
        """Visible text on one row (trailing blanks stripped)."""
        if not 0 <= r < self.rows:
            return ""
        return "".join(self.framebuffer.get(r, c).char for c in range(self.cols)).rstrip()

    def text(self) -> str:
        return "\n".join(self.row(r) for r in range(self.rows))

    def rows_with(self, sub: str) -> list[int]:
        return [r for r in range(self.rows) if sub in self.row(r)]

    def find(self, sub: str) -> int | None:
        hits = self.rows_with(sub)
        return hits[0] if hits else None

    def resize(self, rows: int, cols: int) -> None:
        """Resize, preserving top-left content and clamping the caret."""
        self.framebuffer.resize(rows, cols)
        self.rows, self.cols = rows, cols
        self.r = min(self.r, rows - 1)
        self.c = min(self.c, cols - 1)
        self.save_r = min(self.save_r, rows - 1)
        self.save_c = min(self.save_c, cols - 1)
        self.scroll_top = min(self.scroll_top, rows - 1)
        self.scroll_bottom = rows - 1
        if self.scroll_top > self.scroll_bottom:
            self.scroll_top = 0
        # A snapshot taken at the old size would restore itself and resize the
        # screen back down, so the saved primary screen is dropped instead.
        self._primary = None

    # ── byte stream ───────────────────────────────────────────────────────

    def feed(self, data: bytes) -> None:
        """Consume raw pty output, updating the cell grid."""
        if data:
            self._pending += self._decoder.decode(data)
        self._parse()
        # Checked *after* parsing, because ``_pending`` must hold only the
        # unparsed tail — a stalled sequence with no final byte.  Checking
        # before would measure the whole chunk and throw away a legitimate
        # read: the curses TUI opens with a ~10 KB repaint, which is well over
        # the cap, and discarding it left a permanently blank screen.
        if len(self._pending) > _MAX_PENDING:
            # Discard rather than tail-truncate: keeping the tail would drop
            # the opening ``ESC ]`` and print the body as literal text.
            self._pending = ""

    def _parse(self) -> None:
        text = self._pending
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if ch != "\x1b":
                if ch in "\n\v\f":
                    self._linefeed()
                elif ch == "\r":
                    self.c = 0
                elif ch == "\b":
                    self.c = max(0, self.c - 1)
                elif ch == "\t":
                    self.c = min((self.c // _TAB_STOP + 1) * _TAB_STOP, self.cols - 1)
                elif ord(ch) >= 32:
                    self._put(ch)
                # Remaining C0 controls (BEL, NUL, …) have no effect on cells.
                i += 1
                continue

            # Escape: the tail may be an incomplete sequence, in which case we
            # stop and keep it — never consume a fragment of one.
            j = i + 1
            if j >= n:
                break
            c2 = text[j]

            if c2 == "[":
                k = j + 1
                body = ""
                final = None
                while k < n:
                    cc = text[k]
                    if "\x20" <= cc <= "\x2f" or "\x30" <= cc <= "\x3f":
                        body += cc
                    else:
                        final = cc
                        break
                    k += 1
                if final is None:
                    break
                self._csi(body, final)
                i = k + 1
                continue

            if c2 == "]":
                k = j + 1
                while k < n:
                    if text[k] == "\x07":
                        k += 1
                        break
                    if text[k] == "\x1b" and k + 1 < n and text[k + 1] == "\\":
                        k += 2
                        break
                    k += 1
                else:
                    break  # OSC not yet terminated
                i = k
                continue

            if c2 in "()*+":
                if j + 1 >= n:
                    break  # charset designator missing its byte
                i = j + 2
                continue

            if c2 == "O":
                if j + 1 >= n:
                    break  # SS2/SS3 missing its final byte
                i = j + 2
                continue

            if c2 in "=>":
                i = j + 1
            elif c2 == "7":
                self.save_r, self.save_c = self.r, self.c
                i = j + 1
            elif c2 == "8":
                self.r, self.c = self._clamp(self.save_r, self.save_c)
                i = j + 1
            elif c2 in "DH":  # IND / HTS
                if c2 == "D":
                    self._linefeed()
                i = j + 1
            elif c2 == "E":  # NEL
                self.c = 0
                self._linefeed()
                i = j + 1
            elif c2 == "M":  # RI
                self._reverse_index()
                i = j + 1
            elif c2 == "c":  # RIS — full reset
                self.framebuffer.clear()
                self.r = self.c = 0
                self.scroll_top = 0
                self.scroll_bottom = self.rows - 1
                self.fg = int(Color.DEFAULT)
                self.bg = int(Color.DEFAULT)
                self.attr = int(Attr.NONE)
                i = j + 1
            else:
                # Unknown escape: drop the ESC and reprocess its final byte so
                # a printable one still reaches the screen.
                i = j

        self._pending = text[i:]

    # ── CSI dispatch ──────────────────────────────────────────────────────

    def _csi(self, body: str, final: str) -> None:
        priv = body.startswith("?")
        if priv:
            body = body[1:]
        parts = body.split(";")

        def num(i: int, d: int) -> int:
            try:
                v = int(parts[i])
            except (IndexError, ValueError):
                return d
            return v if v else d

        if final == "A":  # CUU
            self.r, self.c = self._clamp(self.r - num(0, 1), self.c)
        elif final == "B":  # CUD
            self.r, self.c = self._clamp(self.r + num(0, 1), self.c)
        elif final == "C":  # CUF
            self.r, self.c = self._clamp(self.r, self.c + num(0, 1))
        elif final == "D":  # CUB
            self.r, self.c = self._clamp(self.r, self.c - num(0, 1))
        elif final in ("H", "f"):  # CUP / HVP
            self.r, self.c = self._clamp(num(0, 1) - 1, num(1, 1) - 1)
        elif final == "G":  # CHA
            self.r, self.c = self._clamp(self.r, num(0, 1) - 1)
        elif final == "d":  # VPA
            self.r, self.c = self._clamp(num(0, 1) - 1, self.c)
        elif final == "J":  # ED
            self._erase_display(num(0, 0))
        elif final == "K":  # EL
            n = num(0, 0)
            if n == 0:
                self._erase(self.r, self.c, self.cols)
            elif n == 1:
                self._erase(self.r, 0, self.c + 1)
            else:
                self._erase(self.r, 0, self.cols)
        elif final == "X":  # ECH
            self._erase(self.r, self.c, self.c + num(0, 1))
        elif final == "r":  # DECSTBM
            if not body:
                self.scroll_top, self.scroll_bottom = 0, self.rows - 1
            else:
                top = max(0, num(0, 1) - 1)
                bot = min(self.rows - 1, num(1, self.rows) - 1)
                self.scroll_top, self.scroll_bottom = min(top, bot), max(top, bot)
        elif final == "S":  # SU
            self._region_scroll(num(0, 1))
        elif final == "T":  # SD
            self._region_scroll_down(num(0, 1))
        elif final == "P":  # DCH
            self._delete_chars(num(0, 1))
        elif final == "@":  # ICH
            self._insert_chars(num(0, 1))
        elif final == "L":  # IL
            self._insert_lines(num(0, 1))
        elif final == "M":  # DL
            self._delete_lines(num(0, 1))
        elif final == "m":  # SGR
            self._sgr(parts)
        elif final == "s":  # save cursor
            self.save_r, self.save_c = self.r, self.c
        elif final == "u":  # restore cursor
            self.r, self.c = self._clamp(self.save_r, self.save_c)
        elif final in ("h", "l") and priv and "1049" in body:
            self._alternate_screen(final == "h")

    def _sgr(self, parts: list[str]) -> None:
        """Apply ``CSI … m`` — the pen the next printed character uses."""
        if not parts:
            parts = ["0"]
        i = 0
        while i < len(parts):
            p = _int_at(parts, i, 0)
            if p == 0:
                self.fg = int(Color.DEFAULT)
                self.bg = int(Color.DEFAULT)
                self.attr = int(Attr.NONE)
            elif p in _SGR_OFF:
                self.attr &= ~_SGR_OFF[p]
            elif p in _SGR_ATTR:
                self.attr |= _SGR_ATTR[p]
            elif 30 <= p <= 37:
                self.fg = p - 30
            elif p == 39:
                self.fg = int(Color.DEFAULT)
            elif 40 <= p <= 47:
                self.bg = p - 40
            elif p == 49:
                self.bg = int(Color.DEFAULT)
            elif 90 <= p <= 97:
                self.fg = p - 90 + 8
            elif 100 <= p <= 107:
                self.bg = p - 100 + 8
            elif p in (38, 48):
                i += self._extended_color(parts, i, fg=p == 38)
            i += 1

    def _extended_color(self, parts: list[str], i: int, *, fg: bool) -> int:
        """Consume ``38/48;5;N`` or ``38/48;2;R;G;B``; return extra params used."""
        if i + 1 >= len(parts):
            return 0
        mode = parts[i + 1]
        if mode == "5" and i + 2 < len(parts):
            idx = _int_at(parts, i + 2, 0) & 0xFF
        elif mode == "2" and i + 4 < len(parts):
            idx = rgb_to_ansi256(
                _int_at(parts, i + 2, 0),
                _int_at(parts, i + 3, 0),
                _int_at(parts, i + 4, 0),
            )
        else:
            return 0
        if fg:
            self.fg = idx
        else:
            self.bg = idx
        return 4 if mode == "2" else 2

    def _alternate_screen(self, enter: bool) -> None:
        """Switch to (saving the primary screen) or back from alt-screen."""
        if enter:
            if self._primary is None:
                self._primary = self.framebuffer.snapshot()
                self._primary_cursor = (self.r, self.c)
            self.framebuffer.clear()
            self.r = self.c = 0
        elif self._primary is not None:
            self.framebuffer.restore(self._primary)
            self._primary = None
            self.r, self.c = self._clamp(*self._primary_cursor)
        else:
            self.framebuffer.clear()
            self.r = self.c = 0

    # ── cell operations ───────────────────────────────────────────────────

    def _put(self, ch: str) -> None:
        w = _char_width(ch)
        if self.c + w > self.cols:
            # Does not fit on this row — a wide glyph needs both columns, so
            # wrap before writing rather than spilling a half-glyph.
            self.c = 0
            self._linefeed()
        if 0 <= self.r < self.rows:
            self.framebuffer.put(self.r, self.c, ch, self.fg, self.bg, self.attr)
            if w == 2 and self.c + 1 < self.cols:
                self.framebuffer.put(self.r, self.c + 1, "", self.fg, self.bg, self.attr)
        self.c += w

    def _linefeed(self) -> None:
        if self.r == self.scroll_bottom:
            self._region_scroll(1)
        else:
            self.r = min(self.r + 1, self.rows - 1)

    def _reverse_index(self) -> None:
        if self.r == self.scroll_top:
            self._region_scroll_down(1)
        elif self.r > 0:
            self.r -= 1

    def _clamp(self, r: int, c: int) -> tuple[int, int]:
        return max(0, min(r, self.rows - 1)), max(0, min(c, self.cols - 1))

    def _clear_screen(self) -> None:
        self.framebuffer.clear()
        self.r = self.c = 0

    def _erase(self, row: int, c0: int, c1: int) -> None:
        """Blank ``[c0, c1)`` on ``row`` with default attributes."""
        if not 0 <= row < self.rows:
            return
        for c in range(max(0, c0), min(self.cols, c1)):
            self.framebuffer.set(row, c, Cell())

    def _erase_row(self, row: int) -> None:
        self._erase(row, 0, self.cols)

    def _copy_row(self, src: int, dst: int) -> None:
        for c in range(self.cols):
            self.framebuffer.set(dst, c, self.framebuffer.get(src, c))

    def _erase_display(self, n: int) -> None:
        """ED — unlike the earlier test model this does *not* move the cursor;
        a terminal clears the screen under wherever the cursor already is, and
        well-formed output issues ``CSI H`` separately when it wants home."""
        if n == 0:
            self._erase(self.r, self.c, self.cols)
            for r in range(self.r + 1, self.rows):
                self._erase_row(r)
        elif n == 1:
            self._erase(self.r, 0, self.c + 1)
            for r in range(self.r):
                self._erase_row(r)
        elif n != 3:  # 2 = whole screen; 3 = erase scrollback (no model here)
            for r in range(self.rows):
                self._erase_row(r)

    def _region_scroll(self, n: int) -> None:
        top, bot = self.scroll_top, self.scroll_bottom
        for _ in range(n):
            for r in range(top, bot):
                self._copy_row(r + 1, r)
            self._erase_row(bot)

    def _region_scroll_down(self, n: int) -> None:
        top, bot = self.scroll_top, self.scroll_bottom
        for _ in range(n):
            for r in range(bot, top, -1):
                self._copy_row(r - 1, r)
            self._erase_row(top)

    def _delete_chars(self, n: int) -> None:
        if not (0 <= self.r < self.rows and 0 <= self.c < self.cols):
            return
        for _ in range(n):
            for c in range(self.c, self.cols - 1):
                self.framebuffer.set(self.r, c, self.framebuffer.get(self.r, c + 1))
            self._erase(self.r, self.cols - 1, self.cols)

    def _insert_chars(self, n: int) -> None:
        if not (0 <= self.r < self.rows and 0 <= self.c < self.cols):
            return
        for _ in range(n):
            for c in range(self.cols - 1, self.c, -1):
                self.framebuffer.set(self.r, c, self.framebuffer.get(self.r, c - 1))
            self._erase(self.r, self.c, self.c + 1)

    def _insert_lines(self, n: int) -> None:
        if not (self.scroll_top <= self.r <= self.scroll_bottom):
            return
        for _ in range(n):
            for r in range(self.scroll_bottom, self.r, -1):
                self._copy_row(r - 1, r)
            self._erase_row(self.r)

    def _delete_lines(self, n: int) -> None:
        if not (self.scroll_top <= self.r <= self.scroll_bottom):
            return
        for _ in range(n):
            for r in range(self.r, self.scroll_bottom):
                self._copy_row(r + 1, r)
            self._erase_row(self.scroll_bottom)
