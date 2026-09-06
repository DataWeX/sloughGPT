"""
ShellLogger — ANSI output for the interactive REPL.

Uses LogFormatter for all formatting. Routes records through ANSI escape codes
for in-terminal display.

Usage::

    from domains.logging import ShellLogger, LogLevel

    log = ShellLogger("slo.shell")
    log.info("model loaded", model="gpt2")
    log.error("command failed", exception="FileNotFoundError")
"""

from __future__ import annotations

import os
import sys
from typing import Optional, TextIO

from .base import Logger, LogLevel, LogRecord


_NO_COLOR = os.environ.get("NO_COLOR", "").strip() == "1"
_COLOR_ENABLED = not _NO_COLOR and sys.stdout.isatty()


class ShellLogger(Logger):
    """ANSI-based logger for the shell REPL.

    Writes to ``stdout`` by default (matches REPL convention).
    Respects ``NO_COLOR`` env var.

    Parameters:
        name:    Logger name (e.g. ``"slo.shell.repl"``).
        level:   Minimum severity to emit.
        stream:  Output stream (default ``sys.stdout``).
        colors:  Enable ANSI color output (default: auto-detect).
        context: Default context attached to every record.
    """

    def __init__(
        self,
        name: str = "slo.shell",
        level: LogLevel = LogLevel.INFO,
        stream: Optional[TextIO] = None,
        colors: Optional[bool] = None,
        context=None,
    ) -> None:
        super().__init__(name=name, level=level, context=context)
        self._stream = stream or sys.stdout
        self._colors = _COLOR_ENABLED if colors is None else colors
        from .config import LogFormatter
        self._formatter = LogFormatter(fmt="shell", colors=self._colors)

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
