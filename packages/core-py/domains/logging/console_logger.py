"""
ConsoleLogger — colored terminal output for API servers and long-running processes.

Replaces the raw ``coloredlogs.install()`` pattern with a proper Logger
subclass.  Routes records to stderr with ANSI colors.

Usage::

    from domains.logging import ConsoleLogger, LogLevel

    log = ConsoleLogger("man.api", level=LogLevel.DEBUG)
    log.info("server started", port=8000)
    log.error("model load failed", exception="RuntimeError: OOM")
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import TextIO

from .base import Logger, LogLevel, LogRecord


# ── ANSI codes ─────────────────────────────────────────────────────────

_NO_COLOR = os.environ.get("NO_COLOR", "").strip() == "1"
_COLOR_ENABLED = not _NO_COLOR and sys.stderr.isatty()


def _c(code: str) -> str:
    return code if _COLOR_ENABLED else ""


class _Ansi:
    RESET   = _c("\033[0m")
    BOLD    = _c("\033[1m")
    DIM     = _c("\033[2m")
    RED     = _c("\033[31m")
    GREEN   = _c("\033[32m")
    YELLOW  = _c("\033[33m")
    BLUE    = _c("\033[34m")
    CYAN    = _c("\033[36m")
    WHITE   = _c("\033[37m")
    GREY    = _c("\033[90m")
    BG_RED  = _c("\033[41m")


# ── Level formatting ───────────────────────────────────────────────────

_LEVEL_STYLE = {
    LogLevel.DEBUG:    (_Ansi.DIM + _Ansi.CYAN,    "DEBUG   "),
    LogLevel.INFO:     (_Ansi.GREEN,                "INFO    "),
    LogLevel.WARNING:  (_Ansi.YELLOW + _Ansi.BOLD,  "WARNING "),
    LogLevel.ERROR:    (_Ansi.RED + _Ansi.BOLD,     "ERROR   "),
    LogLevel.CRITICAL: (_Ansi.BG_RED + _Ansi.BOLD,  "CRITICAL"),
}


class ConsoleLogger(Logger):
    """Colored terminal logger — drop-in replacement for ``coloredlogs``.

    Writes to ``stderr`` by default (matches standard logging convention).
    Pass ``stream=sys.stdout`` to redirect.

    Parameters:
        name:    Logger name (e.g. ``"man.api"``).
        level:   Minimum severity to emit.
        stream:  Output stream (default ``sys.stderr``).
        colors:  Enable ANSI color output (default: auto-detect TTY).
        context: Default context attached to every record.
    """

    def __init__(
        self,
        name: str = "man",
        level: LogLevel = LogLevel.INFO,
        stream: TextIO | None = None,
        colors: bool | None = None,
        context=None,
    ) -> None:
        super().__init__(name=name, level=level, context=context)
        self._stream = stream or sys.stderr
        self._colors = _COLOR_ENABLED if colors is None else colors

    # ── Formatting ──────────────────────────────────────────────────────

    def _format_time(self, ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime("%H:%M:%S")

    def _format_record(self, record: LogRecord) -> str:
        parts = []

        # Timestamp
        parts.append(f"{_Ansi.GREY}{self._format_time(record.timestamp)}{_Ansi.RESET}" if self._colors
                     else self._format_time(record.timestamp))

        # Level
        color, label = _LEVEL_STYLE.get(record.level, ("", record.level.value.upper().ljust(9)))
        if self._colors:
            parts.append(f"{color}{label}{_Ansi.RESET}")
        else:
            parts.append(label)

        # Logger name
        if self._colors:
            parts.append(f"{_Ansi.CYAN}{_Ansi.DIM}{record.logger}{_Ansi.RESET}")
        else:
            parts.append(record.logger)

        # Context (if any)
        if record.context:
            ctx_str = " ".join(f"{k}={v}" for k, v in record.context.items())
            if self._colors:
                parts.append(f"{_Ansi.GREY}{ctx_str}{_Ansi.RESET}")
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

    # ── Emit ────────────────────────────────────────────────────────────

    def emit(self, record: LogRecord) -> None:
        """Format and write the record to the output stream (thread-safe)."""
        line = self._format_record(record)
        with self._lock:
            try:
                self._stream.write(line + "\n")
                self._stream.flush()
            except (OSError, ValueError):
                # Stream closed (e.g. during shutdown) — silently drop
                pass
