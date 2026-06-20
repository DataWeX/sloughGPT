"""
ShellLogger — ANSI output for the interactive REPL.

Inherits Logger and routes records through ANSI escape codes for
in-terminal display.  Designed for the shell REPL where Rich is
not available and raw ANSI is the output model.

Usage::

    from domains.logging import ShellLogger, LogLevel

    log = ShellLogger("man.shell")
    log.info("model loaded", model="gpt2")
    log.error("command failed", exception="FileNotFoundError")
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import Optional, TextIO

from .base import Logger, LogLevel, LogRecord


# ── ANSI detection ─────────────────────────────────────────────────────

_NO_COLOR = os.environ.get("NO_COLOR", "").strip() == "1"
_COLOR_ENABLED = not _NO_COLOR and sys.stdout.isatty()


def _c(code: str) -> str:
    return code if _COLOR_ENABLED else ""


class _Ansi:
    RESET  = _c("\033[0m")
    BOLD   = _c("\033[1m")
    DIM    = _c("\033[2m")
    RED    = _c("\033[31m")
    GREEN  = _c("\033[32m")
    YELLOW = _c("\033[33m")
    CYAN   = _c("\033[36m")
    GREY   = _c("\033[90m")


# ── Level formatting ───────────────────────────────────────────────────

_LEVEL_STYLE = {
    LogLevel.DEBUG:    (_Ansi.DIM + _Ansi.CYAN,    "·"),
    LogLevel.INFO:     (_Ansi.GREEN,                "ℹ"),
    LogLevel.WARNING:  (_Ansi.YELLOW + _Ansi.BOLD,  "!"),
    LogLevel.ERROR:    (_Ansi.RED + _Ansi.BOLD,     "✗"),
    LogLevel.CRITICAL: (_Ansi.RED + _Ansi.BOLD,     "✗"),
}


class ShellLogger(Logger):
    """ANSI-based logger for the shell REPL.

    Writes to ``stdout`` by default (matches REPL convention).
    Respects ``NO_COLOR`` env var.

    Parameters:
        name:    Logger name (e.g. ``"man.shell.repl"``).
        level:   Minimum severity to emit.
        stream:  Output stream (default ``sys.stdout``).
        colors:  Enable ANSI color output (default: auto-detect).
        context: Default context attached to every record.
    """

    def __init__(
        self,
        name: str = "man.shell",
        level: LogLevel = LogLevel.INFO,
        stream: Optional[TextIO] = None,
        colors: Optional[bool] = None,
        context=None,
    ) -> None:
        super().__init__(name=name, level=level, context=context)
        self._stream = stream or sys.stdout
        self._colors = _COLOR_ENABLED if colors is None else colors

    def _format_time(self, ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime("%H:%M:%S")

    def _format_record(self, record: LogRecord) -> str:
        parts = []

        # Timestamp (dim)
        ts = self._format_time(record.timestamp)
        if self._colors:
            parts.append(f"{_Ansi.DIM}{ts}{_Ansi.RESET}")
        else:
            parts.append(ts)

        # Icon + level
        color, icon = _LEVEL_STYLE.get(record.level, ("", "·"))
        level_label = record.level.value.upper()
        if self._colors:
            parts.append(f"{color}{icon} {level_label}{_Ansi.RESET}")
        else:
            parts.append(f"{icon} {level_label}")

        # Logger name (dim cyan)
        if self._colors:
            parts.append(f"{_Ansi.DIM}{_Ansi.CYAN}{record.logger}{_Ansi.RESET}")
        else:
            parts.append(record.logger)

        # Context
        if record.context:
            ctx_str = " ".join(f"{k}={v}" for k, v in record.context.items())
            if self._colors:
                parts.append(f"{_Ansi.DIM}{ctx_str}{_Ansi.RESET}")
            else:
                parts.append(ctx_str)

        # Message
        parts.append(record.message)

        # Exception
        if record.exception:
            if self._colors:
                parts.append(f"{_Ansi.RED}{record.exception}{_Ansi.RESET}")
            else:
                parts.append(record.exception)

        return " ".join(parts)

    def emit(self, record: LogRecord) -> None:
        """Format and write the record to the output stream (thread-safe)."""
        line = self._format_record(record)
        with self._lock:
            try:
                self._stream.write(line + "\n")
                self._stream.flush()
            except (OSError, ValueError):
                pass
