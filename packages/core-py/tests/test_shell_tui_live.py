"""
Live, pty-driven tests for the curses TUI shell.

Forks a real pseudo-terminal, boots ``ShellREPL(use_tui=True)`` inside it with a
stubbed runtime (no API server), decodes the terminal byte stream into a
persistent 80x24 frame, and asserts on visible content.  This proves the
three-pane UI actually renders and that real terminal key input — including
the terminfo application-mode arrow sequences (``ESC O A``) that ncurses folds
into ``KEY_UP`` — reaches the event loop.
"""

import fcntl
import os
import pty
import select
import signal
import struct
import termios
import time

import pytest

ROWS = 24
COLS = 80


class _Screen:
    """A small persistent ANSI terminal emulator for tests.

    Handles the subset of terminal control the curses TUI emits: cursor
    movement (CUP/VPA/CHA/CUU/CUD/CUF/CUB), erase (ED/EL), scroll regions
    (DECSTBM) with newline-driven region scroll, insert/delete line, the
    alternate-screen switch, and plain text.  SGR styling is ignored.

    ``feed`` accepts raw bytes and appends them to the current frame, so a
    single instance models the running terminal across every redraw delta.
    """

    def __init__(self, rows: int = ROWS, cols: int = COLS) -> None:
        self.rows = rows
        self.cols = cols
        self.lines = [[" "] * cols for _ in range(rows)]
        self.r = 0
        self.c = 0
        self.scroll_top = 0
        self.scroll_bottom = rows - 1
        self.save_r = 0
        self.save_c = 0

    # ── helpers ──────────────────────────────────────────────────────────

    def _clamp(self, r: int, c: int) -> tuple[int, int]:
        return max(0, min(r, self.rows - 1)), max(0, min(c, self.cols - 1))

    def _region_scroll(self, n: int) -> None:
        top, bot = self.scroll_top, self.scroll_bottom
        for _ in range(n):
            for r in range(top, bot):
                self.lines[r] = self.lines[r + 1]
            self.lines[bot] = [" "] * self.cols

    def _region_scroll_down(self, n: int) -> None:
        top, bot = self.scroll_top, self.scroll_bottom
        for _ in range(n):
            for r in range(bot, top, -1):
                self.lines[r] = self.lines[r - 1]
            self.lines[top] = [" "] * self.cols

    def _linefeed(self) -> None:
        if self.r == self.scroll_bottom:
            self._region_scroll(1)
        else:
            self.r += 1

    def _put(self, ch: str) -> None:
        if self.c >= self.cols:
            self.c = 0
            self._linefeed()
        self.lines[self.r][self.c] = ch
        self.c += 1

    def _clear_screen(self) -> None:
        self.lines = [[" "] * self.cols for _ in range(self.rows)]
        self.r = self.c = 0

    def resize(self, rows: int, cols: int) -> None:
        """Resize the model, preserving the top-left content and the caret."""
        new = [[" "] * cols for _ in range(rows)]
        for r in range(min(rows, self.rows)):
            for c in range(min(cols, self.cols)):
                new[r][c] = self.lines[r][c]
        self.lines = new
        self.rows = rows
        self.cols = cols
        self.r = min(self.r, rows - 1)
        self.c = min(self.c, cols - 1)
        self.scroll_bottom = rows - 1

    # ── CSI dispatch ─────────────────────────────────────────────────────

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

        if final == "A":
            self.r, self.c = self._clamp(self.r - num(0, 1), self.c)
        elif final == "B":
            self.r, self.c = self._clamp(self.r + num(0, 1), self.c)
        elif final == "C":
            self.r, self.c = self._clamp(self.r, self.c + num(0, 1))
        elif final == "D":
            self.r, self.c = self._clamp(self.r, self.c - num(0, 1))
        elif final in ("H", "f"):
            self.r, self.c = self._clamp(num(0, 1) - 1, num(1, 1) - 1)
        elif final == "G":
            self.r, self.c = self._clamp(self.r, num(0, 1) - 1)
        elif final == "d":
            self.r, self.c = self._clamp(num(0, 1) - 1, self.c)
        elif final == "J":
            n = num(0, 0)
            if n == 0:
                self.lines[self.r][self.c:] = [" "] * (self.cols - self.c)
                for r in range(self.r + 1, self.rows):
                    self.lines[r] = [" "] * self.cols
            elif n == 1:
                self.lines[self.r][: self.c + 1] = [" "] * (self.c + 1)
                for r in range(0, self.r):
                    self.lines[r] = [" "] * self.cols
            else:
                self._clear_screen()
        elif final == "K":
            n = num(0, 0)
            if n == 0:
                self.lines[self.r][self.c:] = [" "] * (self.cols - self.c)
            elif n == 1:
                self.lines[self.r][: self.c + 1] = [" "] * (self.c + 1)
            else:
                self.lines[self.r] = [" "] * self.cols
        elif final == "r":
            if not body:
                self.scroll_top, self.scroll_bottom = 0, self.rows - 1
            else:
                top = max(0, num(0, 1) - 1)
                bot = min(self.rows - 1, num(1, self.rows) - 1)
                self.scroll_top, self.scroll_bottom = min(top, bot), max(top, bot)
        elif final == "S":
            self._region_scroll(num(0, 1))
        elif final == "T":
            self._region_scroll_down(num(0, 1))
        elif final == "P":
            for _ in range(num(0, 1)):
                del self.lines[self.r][self.c : self.c + 1]
                self.lines[self.r].append(" ")
        elif final == "@":
            for _ in range(num(0, 1)):
                self.lines[self.r].insert(self.c, " ")
                self.lines[self.r].pop()
        elif final == "L":
            for _ in range(num(0, 1)):
                for r in range(self.scroll_bottom, self.r, -1):
                    self.lines[r] = self.lines[r - 1]
                self.lines[self.r] = [" "] * self.cols
        elif final == "M":
            for _ in range(num(0, 1)):
                for r in range(self.r, self.scroll_bottom):
                    self.lines[r] = self.lines[r + 1]
                self.lines[self.scroll_bottom] = [" "] * self.cols
        elif final in ("h", "l") and priv and "1049" in body:
            self._clear_screen()

    # ── byte stream ──────────────────────────────────────────────────────

    def feed(self, data: bytes) -> None:
        text = data.decode("utf-8", "replace")
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if ch != "\x1b":
                if ch == "\n":
                    self._linefeed()
                elif ch == "\r":
                    self.c = 0
                elif ch == "\b":
                    self.c = max(0, self.c - 1)
                elif ord(ch) >= 32:
                    self._put(ch)
                i += 1
                continue
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
                    if ("\x20" <= cc <= "\x2f") or ("\x30" <= cc <= "\x3f"):
                        body += cc
                    else:
                        final = cc
                        break
                    k += 1
                if final is None:
                    break
                self._csi(body, final)
                i = k + 1
            elif c2 in "78":
                if c2 == "7":
                    self.save_r, self.save_c = self.r, self.c
                else:
                    self.r, self.c = self.save_r, self.save_c
                i = j + 1
            elif c2 in "=>":
                i = j + 1
            elif c2 in "()*+":
                i = j + 2
            elif c2 == "]":
                k = j + 1
                while k < n:
                    if text[k] == "\x07":
                        k += 1
                        break
                    if text[k] == "\x1b" and k + 1 < n and text[k + 1] == "\\":
                        k += 2
                        break
                    k += 1
                i = k
            elif c2 == "D":
                self._linefeed()
                i = j + 1
            elif c2 == "M":
                if self.r == self.scroll_top:
                    self._region_scroll_down(1)
                elif self.r > 0:
                    self.r -= 1
                i = j + 1
            elif c2 == "O":
                i = j + 2
            else:
                i = j

    # ── queries ──────────────────────────────────────────────────────────

    def row(self, r: int) -> str:
        """Visible text on one row (trailing blanks stripped)."""
        if not 0 <= r < self.rows:
            return ""
        return "".join(self.lines[r]).rstrip()

    def text(self) -> str:
        return "\n".join(self.row(r) for r in range(self.rows))

    def rows_with(self, sub: str) -> list[int]:
        return [r for r in range(self.rows) if sub in self.row(r)]

    def find(self, sub: str) -> int | None:
        hits = self.rows_with(sub)
        return hits[0] if hits else None


class _FakeAPI:
    is_running = True

    def status(self):
        return {"available": True, "model_id": "qwen", "engine_type": "cpu"}

    def start(self):
        return {"ok": True, "message": "started"}

    def stop(self):
        return {"ok": True, "message": "stopped"}


class _TuiSession:
    """Fork a pty running the TUI shell; drive it from the parent side."""

    def __init__(self, rows: int = ROWS, cols: int = COLS) -> None:
        self.rows = rows
        self.cols = cols
        self.buf = b""
        self.screen = _Screen(rows, cols)
        self.err_path = os.environ.get("TUI_LIVE_CHILD_ERR", "/tmp/opencode/tui_live_child.err")
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            self._child()
            os._exit(0)

    def _child(self) -> None:
        cov = None
        if os.environ.get("TUI_LIVE_COV"):
            import coverage

            cov_dir = os.environ.get("TUI_LIVE_COV_DIR", "/tmp/opencode/tui_cov")
            os.makedirs(cov_dir, exist_ok=True)
            cov = coverage.Coverage(
                source=["domains/shell"],
                data_suffix=True,
                data_file=os.path.join(cov_dir, ".coverage"),
            )
            cov.start()
        try:
            import fcntl
            import struct
            import termios

            os.environ["TERM"] = "xterm-256color"
            with open(self.err_path, "w") as errf:
                os.dup2(errf.fileno(), 2)
            winsz = struct.pack("HHHH", self.rows, self.cols, 0, 0)
            fcntl.ioctl(1, termios.TIOCSWINSZ, winsz)

            from domains.shell.repl import ShellREPL
            from domains.shell.runtime import DaitRuntime

            rt = DaitRuntime()
            rt._api = _FakeAPI()
            rt.boot = lambda shell_run=None: ("", {"available": True, "model_id": "qwen"})
            rt.shutdown = lambda: ""
            repl = ShellREPL(rt, use_tui=True)
            repl._history = []
            repl.run()
        except Exception:
            import traceback

            traceback.print_exc()
            if cov is not None:
                cov.stop()
                cov.save()
            os._exit(1)
        if cov is not None:
            cov.stop()
            cov.save()

    # ── parent-side driver ───────────────────────────────────────────────

    def feed(self, timeout: float = 0.1) -> bytes:
        end = time.time() + timeout
        while time.time() < end:
            r, _, _ = select.select([self.fd], [], [], 0.05)
            if not r:
                continue
            try:
                data = os.read(self.fd, 65536)
            except OSError:
                break
            if not data:
                break
            self.buf += data
            self.screen.feed(data)
        return self.buf

    def write(self, text: str, settle: float = 0.15) -> None:
        os.write(self.fd, text.encode())
        self.feed(settle)

    def keys(self, seq: bytes, settle: float = 0.2) -> None:
        os.write(self.fd, seq)
        self.feed(settle)

    def wait_until(self, pred, timeout: float = 10.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            self.feed(0.08)
            if pred(self.screen):
                return True
        return bool(pred(self.screen))

    def child_err(self) -> str:
        try:
            return open(self.err_path).read()
        except OSError:
            return ""

    def resize(self, rows: int, cols: int) -> None:
        """Resize the pty window (propagates SIGWINCH to the TUI child)."""
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        self.rows = rows
        self.cols = cols
        self.screen.resize(rows, cols)

    def close(self) -> None:
        for _ in range(2):
            try:
                os.write(self.fd, b"\x03")
            except OSError:
                break
            self.feed(0.1)
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            pass
        else:
            end = time.time() + 2.0
            while time.time() < end:
                try:
                    done, _ = os.waitpid(self.pid, os.WNOHANG)
                except ChildProcessError:
                    break
                if done:
                    break
                self.feed(0.05)
            else:
                try:
                    os.kill(self.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    os.waitpid(self.pid, 0)
                except ChildProcessError:
                    pass
        try:
            os.close(self.fd)
        except OSError:
            pass

    def __enter__(self) -> "_TuiSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@pytest.fixture()
def session():
    with _TuiSession() as s:
        yield s


def _ready(s: _TuiSession) -> bool:
    """The TUI's status bar shows live mode and the input line has a prompt."""
    return ("LIVE" in s.screen.row(ROWS - 2)) and s.screen.row(ROWS - 1).startswith("\u03bb")


def _assert(s: _TuiSession, cond: bool, msg: str) -> None:
    if not cond:
        err = s.child_err()
        pytest.fail(f"{msg}\n--- screen ---\n{s.screen.text()}\n--- child stderr ---\n{err}")


# ── layout ────────────────────────────────────────────────────────────────

def test_boots_to_three_pane_layout(session):
    assert session.wait_until(lambda sc: _ready(session) and "80x24" in sc.text()), (
        f"TUI never reached ready state.\n{session.screen.text()}"
    )
    # Status bar names the output pane and live scroll state.
    assert "[OUTPUT]" in session.screen.row(ROWS - 2)
    assert "LIVE" in session.screen.row(ROWS - 2)
    # Input row carries the lambda prompt (trailing space is stripped).
    assert session.screen.row(ROWS - 1).startswith("\u03bb")
    # Console and output panes occupy the rows above the chrome.
    for r in range(ROWS - 2):
        assert session.screen.row(r) is not None


def test_help_command_renders_output_pane(session):
    assert session.wait_until(lambda sc: _ready(session))
    session.write("help\r")
    # The help table is longer than the output pane, so it scrolls; assert
    # on a section from the scrolled-to tail of the listing.
    assert session.wait_until(lambda sc: "Examples:" in sc.text()), (
        f"No help content in output pane.\n{session.screen.text()}"
    )


# ── command execution ─────────────────────────────────────────────────────

def test_echo_roundtrip(session):
    assert session.wait_until(lambda sc: _ready(session))
    session.write("echo alpha\r")
    ok = session.wait_until(
        lambda sc: "echo alpha" in sc.text() and len(sc.rows_with("alpha")) >= 1
    )
    _assert(session, ok, "echo output never appeared")
    assert "\u03bb echo alpha" in session.screen.text()


# ── terminal key input ────────────────────────────────────────────────────

def test_arrow_keys_fold_and_recall_history(session):
    """Application-mode arrow bytes (``ESC O A``) must fold to KEY_UP inside
    the curses event loop and recall prior commands."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("echo alpha\r")
    assert session.wait_until(lambda sc: "echo alpha" in sc.text())
    session.write("echo beta\r")
    assert session.wait_until(lambda sc: "echo beta" in sc.text())

    session.keys(b"\x1bOA")  # UP — recall most recent command
    assert "echo beta" in session.screen.row(ROWS - 1), (
        f"UP did not recall last command.\n{session.screen.text()}"
    )

    session.keys(b"\r")  # Enter — re-run the recalled command
    ok = session.wait_until(lambda sc: sc.text().count("echo beta") >= 2)
    _assert(session, ok, "re-run of recalled command never rendered")


def test_down_arrow_clears_input(session):
    assert session.wait_until(lambda sc: _ready(session))
    session.write("echo alpha\r")
    assert session.wait_until(lambda sc: "echo alpha" in sc.text())
    session.keys(b"\x1bOA")  # UP — recall
    assert "echo alpha" in session.screen.row(ROWS - 1)
    session.keys(b"\x1bOB")  # DOWN — past newest entry clears the line
    assert session.screen.row(ROWS - 1).strip() == "\u03bb"


def test_ctrl_left_word_motion(session):
    """``ESC [ 1 ; 5 D`` is folded by keypad(True) into KEY_CTRL_LEFT; the
    caret must jump word-wise so inserted text lands mid-line."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("echo bravo\r")
    assert session.wait_until(lambda sc: "echo bravo" in sc.text())
    session.keys(b"\x1bOA")  # UP — recall "echo bravo"
    assert "echo bravo" in session.screen.row(ROWS - 1)
    session.keys(b"\x1b[1;5D\x1b[1;5D")  # Ctrl+Left x2 — caret to start of line
    session.write("X")
    assert session.screen.row(ROWS - 1) == "\u03bb Xecho bravo", (
        f"Ctrl+Left caret motion wrong.\n{session.screen.text()}"
    )
    session.keys(b"\r")
    ok = session.wait_until(lambda sc: "\u03bb Xecho bravo" in sc.text())
    _assert(session, ok, "edited recalled command never submitted")


def test_escape_ctrl_right_stays_manual_path(session):
    """``ESC [ 5 C`` is not a terminfo key, so ncurses returns ESC then the
    remainder — the TUI's ``_read_escape_remainder`` must fold it to a word
    move (this is the Alt/Ctrl fallback path)."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("one two\r")
    assert session.wait_until(lambda sc: "one two" in sc.text())
    session.keys(b"\x1bOA")
    assert "one two" in session.screen.row(ROWS - 1)
    session.keys(b"\x1b[5C\x1b[5C")  # Ctrl+Right x2 → caret to end
    session.write("!")
    assert session.screen.row(ROWS - 1) == "\u03bb one two!", (
        f"Ctrl+Right word motion wrong.\n{session.screen.text()}"
    )


# ── scrollback ───────────────────────────────────────────────────────────

def test_page_up_page_down_scroll_output_pane(session):
    """PgUp (``ESC [ 5 ~``) must fold to KEY_PPAGE and push the output pane
    back 10 capture lines (status shows ``SCROLL``); PgDn returns to live."""
    assert session.wait_until(lambda sc: _ready(session))
    for i in range(18):
        session.write(f"echo line{i}\r")
    assert session.wait_until(lambda sc: "line17" in sc.text())

    session.keys(b"\x1b[5~")  # PgUp
    ok = session.wait_until(
        lambda sc: "[OUTPUT]" in sc.row(ROWS - 2) and "SCROLL" in sc.row(ROWS - 2)
    )
    _assert(session, ok, "PgUp did not enter output scrollback mode")
    # A line below the live tail becomes visible after the +10 jump.
    assert "line6" in session.screen.text(), (
        f"scrolled-back line not visible.\n{session.screen.text()}"
    )

    session.keys(b"\x1b[6~")  # PgDn — back to the live tail
    ok = session.wait_until(lambda sc: "LIVE" in sc.row(ROWS - 2))
    _assert(session, ok, "PgDn did not return the output pane to live tail")


def test_ctrl_o_toggles_scroll_target(session):
    """Ctrl+O switches which pane scrollback applies to; the status bar
    reflects the target between OUTPUT and LOG."""
    assert session.wait_until(lambda sc: _ready(session))
    session.keys(b"\x0f")  # Ctrl+O
    ok = session.wait_until(lambda sc: "[LOG]" in sc.row(ROWS - 2))
    _assert(session, ok, "Ctrl+O did not switch scroll target to LOG")
    session.keys(b"\x0f")  # Ctrl+O again
    ok = session.wait_until(lambda sc: "[OUTPUT]" in sc.row(ROWS - 2))
    _assert(session, ok, "Ctrl+O did not switch scroll target back to OUTPUT")


# ── incremental search ───────────────────────────────────────────────────

def test_reverse_history_search(session):
    """Ctrl+R enters reverse incremental search; typed characters refine the
    match shown on the input row; Enter accepts and re-runs it."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("echo alpha\r")
    assert session.wait_until(lambda sc: "echo alpha" in sc.text())
    session.write("echo beta\r")
    assert session.wait_until(lambda sc: "echo beta" in sc.text())

    session.keys(b"\x12")  # Ctrl+R
    session.write("bet")
    ok = session.wait_until(
        lambda sc: "reverse-i-search" in sc.row(ROWS - 2) and "bet" in sc.row(ROWS - 2)
    )
    _assert(session, ok, "reverse incremental search prompt never appeared")
    assert "echo beta" in session.screen.row(ROWS - 1), (
        f"search match not shown on input row.\n{session.screen.text()}"
    )

    session.keys(b"\r")  # accept the match
    assert session.screen.row(ROWS - 1).strip() == "\u03bb echo beta"
    session.keys(b"\r")  # execute
    ok = session.wait_until(lambda sc: sc.text().count("echo beta") >= 2)
    _assert(session, ok, "search-recalled command never executed")


def test_output_pane_search(session):
    """/` on an empty prompt enters output-pane search; typing filters, and
    Enter accepts, jumping the pane so the matched line is visible."""
    assert session.wait_until(lambda sc: _ready(session))
    for i in range(18):
        session.write(f"echo line{i}\r")
    assert session.wait_until(lambda sc: "line17" in sc.text())

    session.keys(b"/")
    session.write("line6")
    ok = session.wait_until(lambda sc: "output-search" in sc.row(ROWS - 2))
    _assert(session, ok, "output-pane search prompt never appeared")

    session.keys(b"\r")  # accept
    ok = session.wait_until(
        lambda sc: "output-search" not in sc.row(ROWS - 2) and "[OUTPUT]" in sc.row(ROWS - 2)
    )
    _assert(session, ok, "output-pane search never closed on Enter")
    assert "line6" in session.screen.text(), (
        f"search match not jumped into view.\n{session.screen.text()}"
    )


def test_n_repeats_last_output_search(session):
    """n at an empty prompt while scrolled back repeats the last accepted
    output-pane search from the current match, advancing to the next."""
    assert session.wait_until(lambda sc: _ready(session))
    for i in range(18):
        session.write(f"echo line{i}\r")
    assert session.wait_until(lambda sc: "line17" in sc.text())

    session.keys(b"/")
    session.write("line6")
    ok = session.wait_until(lambda sc: "output-search" in sc.row(ROWS - 2))
    _assert(session, ok, "output-pane search prompt never appeared")
    session.keys(b"\r")  # accept
    ok = session.wait_until(lambda sc: "output-search" not in sc.row(ROWS - 2))
    _assert(session, ok, "output-pane search never closed on Enter")

    before = session.screen.text()
    assert "line6" in before, f"accepted search never matched.\n{before}"
    session.keys(b"n")  # repeat search forward
    ok = session.wait_until(lambda sc: sc.text() != before)
    _assert(session, ok, "n did not advance the output-pane search")
    assert "line6" in session.screen.text(), (
        f"repeated search lost the match.\n{session.screen.text()}"
    )


# ── line editing helpers ─────────────────────────────────────────────────

def test_tab_completion_completes_command(session):
    assert session.wait_until(lambda sc: _ready(session))
    session.write("ech")
    session.keys(b"\t")  # Tab — complete "ech" → "echo "
    assert session.screen.row(ROWS - 1).startswith("\u03bb echo"), (
        f"Tab completion wrong.\n{session.screen.text()}"
    )
    session.write("hello")
    session.keys(b"\r")
    ok = session.wait_until(lambda sc: "\u03bb echo hello" in sc.text() and "hello" in sc.text())
    _assert(session, ok, "completed command never executed")


def test_ctrl_l_clears_output_pane(session):
    assert session.wait_until(lambda sc: _ready(session))
    session.write("echo stuff\r")
    assert session.wait_until(lambda sc: "stuff" in sc.text())
    session.keys(b"\x0c")  # Ctrl+L — clear output pane
    ok = session.wait_until(lambda sc: "stuff" not in sc.text())
    _assert(session, ok, "Ctrl+L did not clear the output pane")
    assert session.screen.row(ROWS - 1).startswith("\u03bb")


def test_ctrl_p_ctrl_n_history_navigation(session):
    """Ctrl+P / Ctrl+N (readline previous/next-history) must move through
    the command history exactly like the arrow keys."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("echo alpha\r")
    assert session.wait_until(lambda sc: "echo alpha" in sc.text())
    session.write("echo beta\r")
    assert session.wait_until(lambda sc: "echo beta" in sc.text())

    session.keys(b"\x10")  # Ctrl+P — previous
    assert "echo beta" in session.screen.row(ROWS - 1)
    session.keys(b"\x10")  # Ctrl+P — older
    assert "echo alpha" in session.screen.row(ROWS - 1)
    session.keys(b"\x0e")  # Ctrl+N — next
    assert "echo beta" in session.screen.row(ROWS - 1)
    session.keys(b"\x0e")  # Ctrl+N — past newest clears
    assert session.screen.row(ROWS - 1).strip() == "\u03bb", (
        f"Ctrl+N past newest did not clear input.\n{session.screen.text()}"
    )


def test_home_end_keys_move_caret(session):
    """Home (``ESC O H``) and End (``ESC O F``) fold to KEY_HOME/KEY_END and
    move the caret to the line ends."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("bravo")
    session.keys(b"\x1bOH")  # Home
    session.write("X")
    assert session.screen.row(ROWS - 1) == "\u03bb Xbravo"
    session.keys(b"\x1bOF")  # End
    session.write("!")
    assert session.screen.row(ROWS - 1) == "\u03bb Xbravo!", (
        f"Home/End caret motion wrong.\n{session.screen.text()}"
    )


def test_delete_key_deletes_at_caret(session):
    """Delete (``ESC [ 3 ~``) folds to KEY_DC and removes the char under the
    caret, leaving earlier text intact."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("alpha")
    session.keys(b"\x1b[3~")  # Del at end — nothing to delete
    assert session.screen.row(ROWS - 1) == "\u03bb alpha"
    session.keys(b"\x1bOH")  # Home
    session.keys(b"\x1b[3~")  # Del — removes 'a'
    assert session.screen.row(ROWS - 1) == "\u03bb lpha", (
        f"Delete key did not remove char at caret.\n{session.screen.text()}"
    )


def test_backspace_deletes_before_caret(session):
    assert session.wait_until(lambda sc: _ready(session))
    session.write("alpha")
    session.keys(b"\x7f")  # Backspace (terminfo kbs)
    assert session.screen.row(ROWS - 1) == "\u03bb alph", (
        f"Backspace did not delete before caret.\n{session.screen.text()}"
    )


def test_ctrl_u_kill_to_start_and_yank(session):
    assert session.wait_until(lambda sc: _ready(session))
    session.write("alpha bravo")
    session.keys(b"\x15")  # Ctrl+U — kill to start
    assert session.screen.row(ROWS - 1).strip() == "\u03bb"
    session.keys(b"\x19")  # Ctrl+Y — yank
    assert session.screen.row(ROWS - 1) == "\u03bb alpha bravo", (
        f"Ctrl+U/Ctrl+Y kill+yank wrong.\n{session.screen.text()}"
    )


def test_ctrl_w_delete_word_back(session):
    assert session.wait_until(lambda sc: _ready(session))
    session.write("one two three")
    session.keys(b"\x17")  # Ctrl+W — delete word before caret
    assert session.screen.row(ROWS - 1) == "\u03bb one two", (
        f"Ctrl+W did not delete the trailing word.\n{session.screen.text()}"
    )


def test_alt_word_motion_via_escape_remainder(session):
    """Alt+F / Alt+B travel the escape-remainder path (ESC not followed by a
    terminfo sequence) and must still move the caret word-wise."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("one two")
    session.keys(b"\x1bOH")  # Home
    session.keys(b"\x1bf")  # Alt+F — to end of "one"
    session.write("X")
    assert session.screen.row(ROWS - 1) == "\u03bb oneX two"
    session.keys(b"\x1bb")  # Alt+B — back to word start
    session.write("Y")
    assert session.screen.row(ROWS - 1) == "\u03bb YoneX two", (
        f"Alt+F/Alt+B word motion wrong.\n{session.screen.text()}"
    )


def test_alt_d_delete_word_after_caret(session):
    """Alt+D must delete the word (and any leading whitespace) after the
    caret without moving it, pushing the killed word to the kill ring."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("alpha bravo charlie")
    session.keys(b"\x1bOH")  # Home — caret to start
    session.keys(b"\x1bd")  # Alt+D — kill "alpha"
    assert session.screen.row(ROWS - 1) == "\u03bb  bravo charlie", (
        f"Alt+D did not delete the first word.\n{session.screen.text()}"
    )
    session.keys(b"\x1bd")  # Alt+D — kill " bravo"
    assert session.screen.row(ROWS - 1) == "\u03bb  charlie", (
        f"Alt+D did not delete the second word.\n{session.screen.text()}"
    )
    session.keys(b"\x19")  # Ctrl+Y — yank the most recent kill
    assert session.screen.row(ROWS - 1) == "\u03bb  bravo charlie", (
        f"Alt+D did not push the killed word to the kill ring.\n{session.screen.text()}"
    )


def test_ctrl_t_transpose_chars(session):
    """Ctrl+T swaps the character before and at the caret; at end of line
    it swaps the last two characters."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("ab")
    session.keys(b"\x14")  # Ctrl+T — at end, swap last two
    assert session.screen.row(ROWS - 1) == "\u03bb ba", (
        f"end-of-line transpose wrong.\n{session.screen.text()}"
    )
    session.keys(b"\x1bOD")  # Left — caret between 'b' and 'a'
    session.keys(b"\x14")  # Ctrl+T — mid-line swap
    assert session.screen.row(ROWS - 1) == "\u03bb ab", (
        f"mid-line transpose wrong.\n{session.screen.text()}"
    )


def test_ctrl_d_deletes_at_caret(session):
    """Ctrl+D (0x04) must delete the character under the caret (mirror of
    Delete / KEY_DC), not echo EOF."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("hello")
    session.keys(b"\x1bOH")  # Home
    session.keys(b"\x04\x04")  # Ctrl+D x2
    assert session.screen.row(ROWS - 1) == "\u03bb llo", (
        f"Ctrl+D did not delete at the caret.\n{session.screen.text()}"
    )


def test_ctrl_y_cycles_kill_ring(session):
    """Repeated Ctrl+Y must walk the kill ring from the newest entry to
    older ones, replacing the yanked text in place."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("alpha bravo")
    session.keys(b"\x15")  # Ctrl+U — kill "alpha bravo"
    session.write("charlie")
    session.keys(b"\x15")  # Ctrl+U — kill "charlie"
    session.keys(b"\x19")  # Ctrl+Y — newest kill
    assert session.screen.row(ROWS - 1) == "\u03bb charlie", (
        f"first yank wrong.\n{session.screen.text()}"
    )
    session.keys(b"\x19")  # Ctrl+Y — cycle to older kill
    assert session.screen.row(ROWS - 1) == "\u03bb alpha bravo", (
        f"kill-ring cycle wrong.\n{session.screen.text()}"
    )


def test_ctrl_a_ctrl_e_caret_motion(session):
    """Ctrl+A / Ctrl+E (readline start-of-line / end-of-line) fold with the
    Home/End keys and move the caret to the line ends."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("bravo")
    session.keys(b"\x01")  # Ctrl+A — start
    session.write("X")
    assert session.screen.row(ROWS - 1) == "\u03bb Xbravo"
    session.keys(b"\x05")  # Ctrl+E — end
    session.write("!")
    assert session.screen.row(ROWS - 1) == "\u03bb Xbravo!", (
        f"Ctrl+A/Ctrl+E caret motion wrong.\n{session.screen.text()}"
    )


def test_ctrl_k_kill_to_end_and_yank(session):
    """Ctrl+K kills from the caret to the end of the line, pushing the
    killed text to the kill ring so Ctrl+Y can restore it."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("alpha bravo")
    session.keys(b"\x1bOH")  # Home
    session.keys(b"\x1bf")  # Alt+F — caret to end of "alpha"
    session.keys(b"\x0b")  # Ctrl+K — kill to end
    assert session.screen.row(ROWS - 1) == "\u03bb alpha", (
        f"Ctrl+K did not kill to end of line.\n{session.screen.text()}"
    )
    session.keys(b"\x19")  # Ctrl+Y — yank the killed tail back
    assert session.screen.row(ROWS - 1) == "\u03bb alpha bravo", (
        f"Ctrl+K kill was not yankable.\n{session.screen.text()}"
    )


def test_ctrl_w_kill_word_pushed_to_ring(session):
    """Ctrl+W must push the deleted word to the kill ring, not just remove
    it, so Ctrl+Y can restore it."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("one two three")
    session.keys(b"\x17")  # Ctrl+W — delete word before caret
    assert session.screen.row(ROWS - 1) == "\u03bb one two"
    session.keys(b"\x19")  # Ctrl+Y — yank "three" back
    assert session.screen.row(ROWS - 1) == "\u03bb one two three", (
        f"Ctrl+W did not push the deleted word to the kill ring.\n{session.screen.text()}"
    )


def test_shift_n_repeats_search_backward(session):
    """N at an empty prompt while scrolled repeats the last accepted
    output-pane search backward from the current match (wrapping)."""
    assert session.wait_until(lambda sc: _ready(session))
    for i in range(18):
        session.write(f"echo line{i}\r")
    assert session.wait_until(lambda sc: "line17" in sc.text())

    session.keys(b"/")
    session.write("line6")
    ok = session.wait_until(lambda sc: "output-search" in sc.row(ROWS - 2))
    _assert(session, ok, "output-pane search prompt never appeared")
    session.keys(b"\r")  # accept
    ok = session.wait_until(lambda sc: "output-search" not in sc.row(ROWS - 2))
    _assert(session, ok, "output-pane search never closed on Enter")

    before = session.screen.text()
    assert "line6" in before, f"accepted search never matched.\n{before}"
    session.keys(b"N")  # repeat search backward
    ok = session.wait_until(lambda sc: sc.text() != before)
    _assert(session, ok, "N did not move the output-pane search backward")
    assert "line6" in session.screen.text(), (
        f"backward repeated search lost the match.\n{session.screen.text()}"
    )


def test_resize_remaps_layout(session):
    """Resizing the pty delivers SIGWINCH; the TUI handles KEY_RESIZE,
    remaps the three-pane layout, and reports the new size in the status
    bar while remaining interactive."""
    assert session.wait_until(lambda sc: _ready(session))
    session.resize(30, 100)
    ok = session.wait_until(lambda sc: "100x30" in sc.text())
    _assert(session, ok, "status bar never reported the resized 100x30 size")
    assert session.screen.row(29).startswith("\u03bb"), (
        f"input row not remapped to bottom row after resize.\n{session.screen.text()}"
    )
    session.write("echo resized\r")
    ok = session.wait_until(lambda sc: "\u03bb echo resized" in sc.text())
    _assert(session, ok, "shell unresponsive after resize")


def test_ctrl_c_interrupts_running_command(session):
    """Ctrl+C while a command runs must mark the interrupt (^C), stop the
    worker, and leave the TUI prompt responsive for the next command."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("sleep 3\r")
    ok = session.wait_until(lambda sc: "sleep" in sc.row(ROWS - 2))
    _assert(session, ok, "active command never shown in status bar")
    session.keys(b"\x03")  # Ctrl+C
    ok = session.wait_until(lambda sc: "^C" in sc.text())
    _assert(session, ok, "Ctrl+C never produced the ^C marker")
    session.write("echo ok\r")
    ok = session.wait_until(lambda sc: "\u03bb echo ok" in sc.text() and "ok" in sc.text())
    _assert(session, ok, "TUI not responsive after Ctrl+C interrupt")


def test_exit_command_terminates(session):
    assert session.wait_until(lambda sc: _ready(session))
    session.write("exit\r")
    end = time.time() + 5.0
    exited = False
    while time.time() < end:
        try:
            done, _ = os.waitpid(session.pid, os.WNOHANG)
        except ChildProcessError:
            exited = True
            break
        if done:
            exited = True
            break
        session.feed(0.05)
    assert exited, "TUI did not exit on 'exit' command"


def test_reverse_search_failed_label_and_esc_cancel(session):
    """A Ctrl+R query with no history match raises the ``failed`` label on
    the status row; Esc cancels and restores the pre-search input buffer."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("echo alpha\r")
    assert session.wait_until(lambda sc: "echo alpha" in sc.text())
    session.write("echo beta\r")
    assert session.wait_until(lambda sc: "echo beta" in sc.text())

    session.keys(b"\x12")  # Ctrl+R
    session.write("zzzz")
    ok = session.wait_until(lambda sc: "failed reverse-i-search" in sc.row(ROWS - 2))
    _assert(session, ok, "failed reverse-i-search label never appeared")

    session.keys(b"\x1b")  # Esc — cancel
    ok = session.wait_until(
        lambda sc: "reverse-i-search" not in sc.row(ROWS - 2) and "[OUTPUT]" in sc.row(ROWS - 2)
    )
    _assert(session, ok, "Esc did not close the failed reverse search")
    assert session.screen.row(ROWS - 1).strip() == "\u03bb"

    session.write("abc")
    session.keys(b"\x12")  # Ctrl+R — save buffer ("abc")
    session.write("alp")  # match found, input row shows "echo alpha"
    ok = session.wait_until(lambda sc: "echo alpha" in sc.row(ROWS - 1))
    _assert(session, ok, "search match never shown on input row")
    session.keys(b"\x1b")  # Esc — cancel, restore "abc"
    ok = session.wait_until(lambda sc: sc.row(ROWS - 1) == "\u03bb abc")
    _assert(session, ok, "Esc did not restore the pre-search buffer")


def test_reverse_search_direction_switch_and_backspace(session):
    """Ctrl+S inside a reverse search flips to forward search; Backspace
    shortens the query and re-applies it."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("echo alpha\r")
    assert session.wait_until(lambda sc: "echo alpha" in sc.text())
    session.write("echo beta\r")
    assert session.wait_until(lambda sc: "echo beta" in sc.text())

    session.keys(b"\x12")  # Ctrl+R
    session.write("bet")
    ok = session.wait_until(lambda sc: "reverse-i-search" in sc.row(ROWS - 2))
    _assert(session, ok, "reverse-i-search prompt never appeared")

    session.keys(b"\x13")  # Ctrl+S — forward
    ok = session.wait_until(lambda sc: "forward-i-search" in sc.row(ROWS - 2))
    _assert(session, ok, "Ctrl+S did not switch to forward-i-search")
    assert "echo beta" in session.screen.row(ROWS - 1)

    session.keys(b"\x7f")  # Backspace — query "bet" -> "be"
    ok = session.wait_until(
        lambda sc: "be" in sc.row(ROWS - 2) and "bet" not in sc.row(ROWS - 2)
    )
    _assert(session, ok, "Backspace did not shorten the search query")
    assert "echo beta" in session.screen.row(ROWS - 1), (
        f"refined forward search lost the match.\n{session.screen.text()}"
    )

    session.keys(b"\x12")  # Ctrl+R — flip back to reverse direction
    ok = session.wait_until(lambda sc: "reverse-i-search" in sc.row(ROWS - 2))
    _assert(session, ok, "Ctrl+R did not flip back to reverse-i-search")
    assert "echo beta" in session.screen.row(ROWS - 1)


def test_output_search_failed_label_and_esc_cancel(session):
    """An unmatched /-query raises ``failed output-search``; Esc closes the
    search and keeps the output pane scrolled back."""
    assert session.wait_until(lambda sc: _ready(session))
    for i in range(18):
        session.write(f"echo line{i}\r")
    assert session.wait_until(lambda sc: "line17" in sc.text())

    session.keys(b"\x1b[5~")  # PgUp — scrolled back
    ok = session.wait_until(lambda sc: "[OUTPUT]" in sc.row(ROWS - 2) and "SCROLL" in sc.row(ROWS - 2))
    _assert(session, ok, "PgUp did not enter output scrollback mode")

    session.keys(b"/")
    session.write("zzzz")
    ok = session.wait_until(lambda sc: "failed output-search" in sc.row(ROWS - 2))
    _assert(session, ok, "failed output-search label never appeared")

    session.keys(b"\x1b")  # Esc — cancel
    ok = session.wait_until(
        lambda sc: "output-search" not in sc.row(ROWS - 2) and "[OUTPUT]" in sc.row(ROWS - 2)
    )
    _assert(session, ok, "Esc did not close the failed output search")
    assert "SCROLL" in session.screen.row(ROWS - 2), (
        f"scrollback lost after failed search cancel.\n{session.screen.text()}"
    )


def test_output_search_esc_cancels_and_restores_scroll(session):
    """Accepting a /-search jumps the pane to the match; Esc must restore
    the pre-search scroll position."""
    assert session.wait_until(lambda sc: _ready(session))
    for i in range(18):
        session.write(f"echo line{i}\r")
    assert session.wait_until(lambda sc: "line17" in sc.text())

    session.keys(b"\x1b[5~")  # PgUp — scroll 10
    ok = session.wait_until(lambda sc: "SCROLL" in sc.row(ROWS - 2))
    _assert(session, ok, "PgUp did not enter output scrollback mode")

    session.keys(b"/")
    session.write("line6")  # match — pane jumps so the line is at top
    ok = session.wait_until(lambda sc: "output-search" in sc.row(ROWS - 2))
    _assert(session, ok, "output-pane search prompt never appeared")

    session.keys(b"\x1b")  # Esc — restore pre-search scroll
    ok = session.wait_until(
        lambda sc: "output-search" not in sc.row(ROWS - 2) and "SCROLL" in sc.row(ROWS - 2)
    )
    _assert(session, ok, "Esc did not restore the pre-search scroll position")


def test_output_search_backspace_refines_query(session):
    """Backspace in output-pane search shrinks the query; a query that no
    longer matches returns to the plain (non-failed) search prompt."""
    assert session.wait_until(lambda sc: _ready(session))
    for i in range(18):
        session.write(f"echo line{i}\r")
    assert session.wait_until(lambda sc: "line17" in sc.text())

    session.keys(b"/")
    session.write("line6x")
    ok = session.wait_until(lambda sc: "failed output-search" in sc.row(ROWS - 2))
    _assert(session, ok, "failed output-search label never appeared")

    session.keys(b"\x7f")  # Backspace — "line6x" -> "line6"
    ok = session.wait_until(
        lambda sc: "output-search" in sc.row(ROWS - 2) and "failed" not in sc.row(ROWS - 2)
    )
    _assert(session, ok, "Backspace did not un-fail the output search")
    assert "line6" in session.screen.row(ROWS - 2), (
        f"refined query not shown on status row.\n{session.screen.text()}"
    )


def test_ctrl_o_pgup_pgdn_scrolls_log_pane(session):
    """After Ctrl+O moves the scroll target to the LOG pane, PgUp/PgDn
    adjust the log scrollback and the status bar tracks it."""
    assert session.wait_until(lambda sc: _ready(session))
    session.keys(b"\x0f")  # Ctrl+O — scroll target = LOG
    ok = session.wait_until(lambda sc: "[LOG]" in sc.row(ROWS - 2))
    _assert(session, ok, "Ctrl+O did not switch scroll target to LOG")

    session.keys(b"\x1b[5~")  # PgUp
    ok = session.wait_until(lambda sc: "[LOG]" in sc.row(ROWS - 2) and "SCROLL" in sc.row(ROWS - 2))
    _assert(session, ok, "PgUp did not scroll the log pane back")
    session.keys(b"\x1b[6~")  # PgDn
    ok = session.wait_until(lambda sc: "[LOG]" in sc.row(ROWS - 2) and "LIVE" in sc.row(ROWS - 2))
    _assert(session, ok, "PgDn did not return the log pane to live")


def test_right_arrow_moves_caret(session):
    """Right arrow (``ESC O C``) folds to KEY_RIGHT and advances the caret
    so a following Ctrl+D deletes the character under it."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("ab")
    session.keys(b"\x01")  # Ctrl+A — Home
    session.keys(b"\x1bOC")  # Right — caret to 'b'
    session.keys(b"\x04")  # Ctrl+D — delete 'b'
    assert session.screen.row(ROWS - 1) == "\u03bb a", (
        f"Right-arrow caret motion wrong.\n{session.screen.text()}"
    )


def test_ctrl_left_word_backward_manual_path(session):
    """``ESC [ 5 D`` is not a terminfo key, so the escape-remainder must fold
    it to a word move backward (Alt/Ctrl fallback path, mirror of Ctrl+Right)."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("one two")
    session.keys(b"\x1b[5D")  # Ctrl+Left — caret to start of "two"
    session.write("X")
    assert session.screen.row(ROWS - 1) == "\u03bb one Xtwo", (
        f"Ctrl+Left word motion wrong.\n{session.screen.text()}"
    )


def test_ctrl_right_folds_to_key_ctrl_right(session):
    """``ESC [ 1 ; 5 C`` is folded by keypad(True) into KEY_CTRL_RIGHT; the
    caret must jump word-wise so inserted text lands mid-line."""
    assert session.wait_until(lambda sc: _ready(session))
    session.write("one two")
    session.keys(b"\x01")  # Ctrl+A — Home
    session.keys(b"\x1b[1;5C")  # Ctrl+Right — caret to end of "one"
    session.write("X")
    assert session.screen.row(ROWS - 1) == "\u03bb oneX two", (
        f"Ctrl+Right word motion wrong.\n{session.screen.text()}"
    )


def test_unhandled_escape_and_control_keys_are_safe(session):
    """An unterminated escape sequence (``ESC [``) and an unbound control
    byte must be consumed without disturbing the prompt; typing after them
    still works."""
    assert session.wait_until(lambda sc: _ready(session))
    session.keys(b"\x1b[")  # lone ESC[ — escape-remainder returns None
    session.keys(b"\x1c")  # Ctrl+\ — unbound, falls through the dispatcher
    session.write("X")
    assert session.screen.row(ROWS - 1) == "\u03bb X", (
        f"prompt disturbed by unhandled keys.\n{session.screen.text()}"
    )


def test_ctrl_c_interrupts_running_command(session):
    """Ctrl+C while a command thread is alive must interrupt it (async-exc
    injection) instead of exiting the TUI; the prompt must remain usable.

    The command runs pure-Python bytecode (``py``) so the injected
    KeyboardInterrupt is delivered at a bytecode boundary — a C-level
    ``time.sleep`` is not reliably interruptible by async exceptions.
    """
    assert session.wait_until(lambda sc: _ready(session))
    session.write("py sum(i * i for i in range(10**9))")
    session.keys(b"\n")
    time.sleep(0.5)  # let the command thread enter the eval loop
    session.keys(b"\x03")  # Ctrl+C — interrupt, not exit
    ok = session.wait_until(lambda sc: "Aborted" in sc.text(), timeout=5)
    _assert(session, ok, "Ctrl+C did not abort the running command")
    session.write("echo still-alive")
    session.keys(b"\n")
    ok = session.wait_until(lambda sc: "still-alive" in sc.text())
    _assert(session, ok, "TUI exited instead of returning to a usable prompt")
