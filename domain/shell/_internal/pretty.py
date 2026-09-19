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
