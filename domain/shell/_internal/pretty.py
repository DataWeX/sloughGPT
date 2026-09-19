"""pretty — terminal output helpers for the shell REPL.

Syntax highlighting and traceback formatting via Pygments (when installed),
with graceful plain-text fallbacks so the shell never depends on it.

Research notes (prompt_toolkit / ptpython conventions applied here):
- tracebacks render with the ``PythonTracebackLexer`` + terminal formatter,
  mirroring ptpython's colored tracebacks;
- output degrades cleanly when color is unavailable (``NO_COLOR``, pipes);
- heavy imports stay lazy: ``pygments`` is imported once at module load
  behind try/except — failure only disables highlighting.
"""

from __future__ import annotations

import os
import sys
import traceback

try:
    from pygments import highlight as _pygments_highlight
    from pygments.formatters import TerminalFormatter
    from pygments.lexers import PythonTracebackLexer, TextLexer, get_lexer_by_name

    _HAVE_PYGMENTS = True
except ImportError:  # pragma: no cover - fallback path
    _pygments_highlight = None  # type: ignore[assignment]
    TerminalFormatter = None  # type: ignore[assignment,misc]
    PythonTracebackLexer = None  # type: ignore[assignment,misc]
    TextLexer = None  # type: ignore[assignment,misc]
    get_lexer_by_name = None  # type: ignore[assignment]
    _HAVE_PYGMENTS = False


def color_supported() -> bool:
    """Whether the current stdout supports ANSI color."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    stream = sys.stdout
    return hasattr(stream, "isatty") and bool(stream.isatty())


def highlight(text: str, lexer_name: str = "python") -> str:
    """Syntax-highlight ``text`` for terminal display.

    Falls back to the unmodified text when Pygments is missing, the
    lexer is unknown, or color is unsupported.
    """
    if not _HAVE_PYGMENTS or not color_supported():
        return text
    try:
        assert get_lexer_by_name is not None and TerminalFormatter is not None
        try:
            lexer = get_lexer_by_name(lexer_name)
        except Exception:
            lexer = TextLexer()
        assert _pygments_highlight is not None
        return _pygments_highlight(text, lexer, TerminalFormatter())
    except Exception:
        return text


def format_traceback(limit: int | None = None) -> str:
    """Format the current exception with syntax highlighting.

    Must be called from inside an ``except`` block. Falls back to the
    standard library rendering when Pygments or color is unavailable.
    """
    rendered = "".join(traceback.format_exception(*sys.exc_info(), limit=limit))
    if not _HAVE_PYGMENTS or not color_supported():
        return rendered
    try:
        assert _pygments_highlight is not None
        assert PythonTracebackLexer is not None and TerminalFormatter is not None
        return _pygments_highlight(rendered, PythonTracebackLexer(), TerminalFormatter())
    except Exception:
        return rendered


def format_error_brief(exc: BaseException) -> str:
    """One-line ``Type: message`` summary for an exception."""
    message = str(exc).strip().splitlines()
    detail = message[0] if message else ""
    name = type(exc).__name__
    return f"{name}: {detail}" if detail else name


def terminal_page_size(reserved: int = 3, minimum: int = 5) -> int:
    """Usable output lines per screenful (terminal height minus prompt)."""
    try:
        import shutil

        return max(minimum, shutil.get_terminal_size().lines - reserved)
    except Exception:
        return 21


def live_capable() -> bool:
    """Whether this process can drive a live alternate-screen pager."""
    if os.environ.get("NO_COLOR"):
        pass  # NO_COLOR only gates color, not the viewport
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        import termios  # noqa: F401
    except ImportError:
        return False
    stream = sys.stdout
    return hasattr(stream, "isatty") and bool(stream.isatty())


class Pager:
    """--More-- style output pager: breaks endless readout into screenfuls.

    Feed every printed chunk through :meth:`process`; once a page fills up
    it asks ``prompt_fn`` whether to continue (anything but ``q``) or drop
    the rest until :meth:`reset`. With no ``prompt_fn`` it never pauses,
    so piped/captured/TUI output is unaffected.

    Usage in the run loop: ``reset()`` before each prompt, and wire
    ``prompt_fn`` to a terminal read.
    """

    def __init__(self, page_size: int | None = None, prompt_fn=None) -> None:
        self._page_size = page_size
        self.prompt_fn = prompt_fn
        self._used = 0
        self._dropped = False

    def reset(self, page_size: int | None = None) -> None:
        """Start a fresh page (call once per prompt)."""
        self._page_size = page_size
        self._used = 0
        self._dropped = False

    @property
    def page_size(self) -> int:
        return self._page_size if self._page_size is not None else terminal_page_size()

    @property
    def dropped(self) -> bool:
        """Whether output is currently being suppressed (user quit)."""
        return self._dropped

    def process(self, text: str, end: str = "\n") -> tuple[str, str] | None:
        """Filter one printed chunk. Returns ``(text, end)`` or ``None``."""
        if self._dropped:
            return None
        if not text:
            lines = 1 if end == "\n" else 0
        else:
            lines = text.count("\n") + (1 if end == "\n" else 0)
        if self.prompt_fn is not None and self._used + lines > self.page_size:
            try:
                answer = (self.prompt_fn() or "").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "q"
            if answer.startswith("q"):
                self._dropped = True
                return None
            self._used = 0
        self._used += lines
        return (text, end)


def terminal_key_reader():
    """Single-key reader for the live viewport (termios raw mode).

    Returns canonical names: ``"up"``/``"down"``/``"right"``/``"left"``,
    ``"home"``/``"end"``, printable characters as-is, ``"\\n"`` for Enter.
    Raises ImportError where termios is unavailable (caller falls back).
    """
    import termios
    import tty

    fd = sys.stdin.fileno()

    def _read() -> str:
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                import select

                if select.select([sys.stdin], [], [], 0.1)[0]:
                    seq = sys.stdin.read(2)
                    return {
                        "[A": "up",
                        "[B": "down",
                        "[C": "right",
                        "[D": "left",
                        "[H": "home",
                        "[F": "end",
                    }.get(seq, "esc")
                return "esc"
            if ch == "\r":
                return "\n"
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    return _read


class LivePager:
    """Less-like live viewport pager over an alternate screen.

    Unlike :class:`Pager` (paper-style, prints ``--More--`` inline), this
    opens a live window: output fills the screen, ``j/k/space/b`` scroll,
    and ``q`` quits. Scrolling past the end of buffered output exits the
    window and resumes streaming — the next overflow re-opens it. ``q``
    drops the remainder until :meth:`reset`, matching line-mode semantics.

    All terminal access goes through the injected ``emit`` (raw writes),
    ``read_key`` (single keys: ``"up"``/``"down"``/``" "``/``"q"`` etc.)
    and ``height_fn`` callables, so the viewport logic is fully
    unit-testable without a terminal.
    """

    def __init__(self, emit=None, read_key=None, height_fn=None) -> None:
        self.emit = emit
        self.read_key = read_key
        self.height_fn = height_fn
        self._lines: list[str] = []
        self._offset = 0
        self._active = False
        self._dropped = False

    def reset(self) -> None:
        """Leave the viewport and clear the buffer (call once per prompt)."""
        if self._active:
            self._exit_screen()
        self._lines = []
        self._offset = 0
        self._dropped = False

    @property
    def dropped(self) -> bool:
        return self._dropped

    @property
    def height(self) -> int:
        try:
            return max(3, int(self.height_fn())) if self.height_fn else terminal_page_size()
        except Exception:
            return terminal_page_size()

    def _viewport(self) -> list[str]:
        h = self.height
        return self._lines[self._offset : self._offset + h]

    def _status(self) -> str:
        total = len(self._lines)
        shown = min(self._offset + self.height, total)
        return f"-- lines {min(self._offset + 1, total)}-{shown}/{total} (j/k/space/b, q: quit) --"

    def _enter_screen(self) -> None:
        if not self._active:
            self.emit("\x1b[?1049h\x1b[H")
            self._active = True

    def _exit_screen(self) -> None:
        if self._active:
            self.emit("\x1b[?1049l")
            self._active = False

    def _draw(self) -> None:
        self._enter_screen()
        self.emit("\x1b[H\x1b[J")
        for line in self._viewport():
            self.emit(line + "\r\n")
        self.emit(self._status())

    def process(self, text: str, end: str = "\n") -> tuple[str, str] | None:
        """Feed one printed chunk. Returns ``(text, end)`` to print live,
        or ``None`` when the chunk was shown in (or dropped by) the viewer.
        """
        if self._dropped:
            return None
        chunk = text + end if end else text
        self._lines.extend(chunk.split("\n"))
        if self._lines and self._lines[-1] == "" and end == "\n":
            self._lines.pop()
        if len(self._lines) <= self.height:
            return (text, end)
        return self._view()

    def _view(self) -> None:
        """Run the live viewport until quit or scroll-past-end."""
        self._offset = max(0, len(self._lines) - self.height)
        self._draw()
        assert self.read_key is not None
        while True:
            try:
                key = self.read_key()
            except (EOFError, KeyboardInterrupt):
                key = "q"
            if key in ("q", "Q", "\x03"):
                self._dropped = True
                self._exit_screen()
                return None
            h = self.height
            if key in ("j", "down", " ", "\n", "f"):
                if self._offset + h < len(self._lines):
                    self._offset += 1 if key in ("j", "down", "\n") else h
                else:
                    # Past the end: leave the window, resume streaming.
                    self._exit_screen()
                    return None
            elif key in ("k", "up", "b"):
                self._offset = max(0, self._offset - (1 if key in ("k", "up") else h))
            elif key in ("g", "home"):
                self._offset = 0
            elif key in ("G", "end"):
                self._offset = max(0, len(self._lines) - h)
            self._draw()
