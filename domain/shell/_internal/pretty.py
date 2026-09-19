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
