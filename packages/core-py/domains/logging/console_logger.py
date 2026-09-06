"""
ConsoleLogger — colored terminal output for API servers and long-running processes.

Uses LogFormatter for all formatting. Routes records to stderr with ANSI colors.

Usage::

    from domains.logging import ConsoleLogger, LogLevel

    log = ConsoleLogger("slo.api", level=LogLevel.DEBUG)
    log.info("server started", port=8000)
    log.error("model load failed", exception="RuntimeError: OOM")

    # With type tag
    log.tag("MODEL").info("loaded", model="gpt2")

    # JSON mode
    log = ConsoleLogger("slo.api", level=LogLevel.DEBUG, format="json")
"""

from __future__ import annotations

import os
import sys
from typing import Optional, TextIO

from .base import Logger, LogLevel, LogRecord


_NO_COLOR = os.environ.get("NO_COLOR", "").strip() == "1"
_FORCE_COLOR = os.environ.get("FORCE_COLOR", "").strip() == "1"


def _default_color_enabled(stream: Optional[TextIO] = None) -> bool:
    if _NO_COLOR:
        return False
    if _FORCE_COLOR:
        return True
    try:
        return bool((stream or sys.stderr).isatty())
    except (AttributeError, ValueError):
        return False


class ConsoleLogger(Logger):
    """Colored terminal logger — drop-in replacement for ``coloredlogs``.

    Writes to ``stderr`` by default (matches standard logging convention).
    Pass ``stream=sys.stdout`` to redirect.

    Parameters:
        name:    Logger name (e.g. ``"slo.api"``).
        level:   Minimum severity to emit.
        stream:  Output stream (default ``sys.stderr``).
        colors:  Enable ANSI color output (default: auto-detect TTY).
        context: Default context attached to every record.
        format:  Output format — ``"human"`` (default), ``"json"``, or ``"slo"``.
    """

    def __init__(
        self,
        name: str = "slo",
        level: LogLevel = LogLevel.INFO,
        stream: Optional[TextIO] = None,
        colors: Optional[bool] = None,
        context=None,
        format: str = "human",
    ) -> None:
        super().__init__(name=name, level=level, context=context)
        self._stream = stream or sys.stderr
        self._colors = _default_color_enabled(self._stream) if colors is None else colors
        self._format = format
        from .config import LogFormatter
        self._formatter = LogFormatter(fmt=format, colors=self._colors)

    # ── Cursor methods (for StatusBlock TTY detection) ──────────────────

    def cursor_up(self, n: int = 1) -> None:
        """Move cursor up n lines. No-op if stream is not a TTY."""
        if hasattr(self._stream, "isatty") and self._stream.isatty() and n > 0:
            try:
                self._stream.write(f"\033[{n}A")
                self._stream.flush()
            except (OSError, ValueError):
                pass

    def cursor_down(self, n: int = 1) -> None:
        """Move cursor down n lines. No-op if stream is not a TTY."""
        if hasattr(self._stream, "isatty") and self._stream.isatty() and n > 0:
            try:
                self._stream.write(f"\033[{n}B")
                self._stream.flush()
            except (OSError, ValueError):
                pass

    def clear_line(self) -> None:
        """Clear the current line. No-op if stream is not a TTY."""
        if hasattr(self._stream, "isatty") and self._stream.isatty():
            try:
                self._stream.write("\033[2K")
                self._stream.flush()
            except (OSError, ValueError):
                pass

    def clear_lines(self, n: int = 1) -> None:
        """Clear n lines starting from cursor position up."""
        if hasattr(self._stream, "isatty") and self._stream.isatty():
            for _ in range(n):
                self.cursor_up(1)
                self.clear_line()

    def hide_cursor(self) -> None:
        """Hide the terminal cursor. No-op if stream is not a TTY."""
        if hasattr(self._stream, "isatty") and self._stream.isatty():
            try:
                self._stream.write("\033[?25l")
                self._stream.flush()
            except (OSError, ValueError):
                pass

    def show_cursor(self) -> None:
        """Restore the terminal cursor. No-op if stream is not a TTY."""
        if hasattr(self._stream, "isatty") and self._stream.isatty():
            try:
                self._stream.write("\033[?25h")
                self._stream.flush()
            except (OSError, ValueError):
                pass

    def emit(self, record: LogRecord) -> None:
        """Format and write the record to the output stream (thread-safe)."""
        line = self._formatter.format_oop(record)
        with self._lock:
            try:
                self._stream.write(line + "\n")
                self._stream.flush()
            except (OSError, ValueError):
                pass

    def _format_record(self, record: LogRecord) -> str:
        """Format a LogRecord and return the string (does not emit)."""
        return self._formatter.format_oop(record)

    def _parse_exception(self, exc_text: str):
        """Parse an exception string into (exc_type, exc_msg, file_info).

        Handles:
          - "ValueError: bad input" → ("ValueError", "bad input", None)
          - "just a message" → (None, "just a message", None)
          - Traceback strings → extracts type, message, and file info
        """
        import re

        if not exc_text:
            return (None, "", None)

        # Check for traceback format
        tb_match = re.search(
            r"Traceback \(most recent call last\):\n"
            r"\s+File '([^']+)', line (\d+), in (\w+)\n"
            r".+?\n"
            r"(\w+): (.+)",
            exc_text,
            re.DOTALL,
        )
        if tb_match:
            file_path = tb_match.group(1)
            line_no = tb_match.group(2)
            func_name = tb_match.group(3)
            exc_type = tb_match.group(4)
            exc_msg = tb_match.group(5)
            file_info = f"{file_path}:{line_no} in {func_name}()"
            return (exc_type, exc_msg, file_info)

        # Check for "Type: message" format
        colon_match = re.match(r"(\w+[\w.]*)\s*:\s*(.+)", exc_text, re.DOTALL)
        if colon_match:
            return (colon_match.group(1), colon_match.group(2).strip(), None)

        return (None, exc_text, None)

    def _get_exception_color(self, exc_type: str) -> str:
        """Return ANSI color code for an exception type."""
        from .config import _A

        _PROGRAMMING = {"ValueError", "TypeError", "KeyError", "IndexError",
                        "AttributeError", "NameError", "SyntaxError"}
        _SYSTEM = {"RuntimeError", "OSError", "MemoryError", "PermissionError"}
        _TRANSIENT = {"TimeoutError", "ConnectionError", "BrokenPipeError"}
        _DEPENDENCY = {"ImportError", "ModuleNotFoundError"}

        if exc_type in _PROGRAMMING:
            return _A.RED + _A.BOLD
        elif exc_type in _SYSTEM:
            return _A.MAGENTA + _A.BOLD
        elif exc_type in _TRANSIENT:
            return _A.YELLOW + _A.BOLD
        elif exc_type in _DEPENDENCY:
            return _A.CYAN + _A.BOLD
        else:
            return _A.RED

    def _format_exception(self, exc_text: str):
        """Format an exception string into display parts [type] and message."""
        exc_type, exc_msg, _ = self._parse_exception(exc_text)
        if exc_type:
            return [f"[{exc_type}]", exc_msg]
        return ["", exc_text]
