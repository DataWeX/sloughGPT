"""
ConsoleLogger — colored terminal output for API servers and long-running processes.

Replaces the raw ``coloredlogs.install()`` pattern with a proper Logger
subclass.  Routes records to stderr with ANSI colors.

Supports two formats:
- ``human`` (default): colored, readable terminal output with type tags
- ``json``: structured JSON lines for log aggregation

Output format (human):
    HH:MM:SS LVL  [TAG]  message  key=val ...  (exception)

Colors:
    - Timestamp:   grey
    - Level:       green/yellow/red by severity
    - Tag:         bold cyan in brackets
    - Context:     grey key=value pairs
    - Exception:   bold red
    - Error code:  yellow in parens

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

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional, TextIO

from .base import Logger, LogLevel, LogRecord


# ── ANSI codes ─────────────────────────────────────────────────────────

_NO_COLOR = os.environ.get("NO_COLOR", "").strip() == "1"
_FORCE_COLOR = os.environ.get("FORCE_COLOR", "").strip() == "1"
_SLO_LOG_COLOR = os.environ.get("SLO_LOG_COLOR", "").strip().lower()


def _default_color_enabled(stream: Optional[TextIO] = None) -> bool:
    """Auto-detect color support for a stream.

    Precedence: NO_COLOR always disables; SLO_LOG_COLOR/FORCE_COLOR force
    enable/disable explicitly; otherwise fall back to TTY detection on the
    target stream at call time.
    """
    if _NO_COLOR:
        return False
    if _SLO_LOG_COLOR in ("1", "true", "yes", "on"):
        return True
    if _SLO_LOG_COLOR in ("0", "false", "no", "off"):
        return False
    if _FORCE_COLOR:
        return True
    try:
        return bool((stream or sys.stderr).isatty())
    except (AttributeError, ValueError):
        return False


class _Ansi:
    RESET    = "\033[0m"
    BOLD     = "\033[1m"
    DIM      = "\033[2m"
    ITALIC   = "\033[3m"
    UNDER    = "\033[4m"
    RED      = "\033[31m"
    GREEN    = "\033[32m"
    YELLOW   = "\033[33m"
    BLUE     = "\033[34m"
    MAGENTA  = "\033[35m"
    CYAN     = "\033[36m"
    WHITE    = "\033[37m"
    GREY     = "\033[90m"
    BG_RED   = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_BLUE  = "\033[44m"


# ── Level formatting ───────────────────────────────────────────────────

_LEVEL_STYLE = {
    LogLevel.DEBUG:    (_Ansi.DIM,                            "DBG"),
    LogLevel.INFO:     (_Ansi.GREEN,                          "INF"),
    LogLevel.WARNING:  (_Ansi.YELLOW + _Ansi.BOLD,            "WRN"),
    LogLevel.ERROR:    (_Ansi.RED + _Ansi.BOLD,               "ERR"),
    LogLevel.CRITICAL: (_Ansi.BG_RED + _Ansi.BOLD + _Ansi.WHITE, "CRI"),
}

_LEVEL_JSON = {
    LogLevel.DEBUG:    "DEBUG",
    LogLevel.INFO:     "INFO",
    LogLevel.WARNING:  "WARN",
    LogLevel.ERROR:    "ERROR",
    LogLevel.CRITICAL: "CRIT",
}


# ── Tag formatting ─────────────────────────────────────────────────────

_TAG_STYLE = {
    "REQ":    (_Ansi.CYAN + _Ansi.BOLD,     "REQ"),
    "AUTH":   (_Ansi.MAGENTA + _Ansi.BOLD,  "AUTH"),
    "MODEL":  (_Ansi.BLUE + _Ansi.BOLD,     "MODEL"),
    "SOUL":   (_Ansi.CYAN,                  "SOUL"),
    "TRAIN":  (_Ansi.GREEN + _Ansi.BOLD,    "TRAIN"),
    "INFRA":  (_Ansi.GREY + _Ansi.BOLD,     "INFRA"),
    "START":  (_Ansi.GREEN + _Ansi.BOLD,    "START"),
    "SLOW":   (_Ansi.YELLOW + _Ansi.ITALIC, "SLOW"),
    "ERROR":  (_Ansi.RED + _Ansi.BOLD,      "ERROR"),
    "WARN":   (_Ansi.YELLOW + _Ansi.BOLD,   "WARN"),
    "OK":     (_Ansi.GREEN,                 "OK"),
}


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
        format:  Output format — ``"human"`` (default) or ``"json"``.
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

    # ── Formatting ──────────────────────────────────────────────────────

    def _format_time(self, ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime("%H:%M:%S")

    def _format_time_iso(self, ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds")

    def _format_record(self, record: LogRecord) -> str:
        if self._format == "json":
            return self._format_json(record)
        return self._format_human(record)

    def _format_json(self, record: LogRecord) -> str:
        """Structured JSON log line — one JSON object per line."""
        entry = {
            "ts": self._format_time_iso(record.timestamp),
            "level": _LEVEL_JSON.get(record.level, record.level.value.upper()),
            "logger": record.logger,
            "msg": record.message,
        }
        if record.tag:
            entry["tag"] = record.tag
        if record.error_code:
            entry["code"] = record.error_code
        if record.context:
            entry["ctx"] = record.context
        if record.exception:
            entry["err"] = record.exception
        return json.dumps(entry, default=str, ensure_ascii=False)

    def _format_human(self, record: LogRecord) -> str:
        """Human-readable colored output for terminals.

        Format: HH:MM:SS LVL [TAG] logger message key=val ... (exception)
        """
        parts = []

        # Timestamp — dimmed grey
        ts = self._format_time(record.timestamp)
        if self._colors:
            parts.append(f"{_Ansi.GREY}{ts}{_Ansi.RESET}")
        else:
            parts.append(ts)

        # Level — colored badge
        color, abbrev = _LEVEL_STYLE.get(record.level, (_Ansi.WHITE, "???"))
        if self._colors:
            parts.append(f"{color}{_Ansi.BOLD}{abbrev:>3}{_Ansi.RESET}")
        else:
            parts.append(abbrev.rjust(3))

        # Type tag — colored bracket badge
        if record.tag:
            tag_color, tag_text = _TAG_STYLE.get(record.tag, (_Ansi.CYAN, record.tag))
            if self._colors:
                parts.append(f"{tag_color}{_Ansi.BOLD}[{tag_text}]{_Ansi.RESET}")
            else:
                parts.append(f"[{tag_text}]")

        # Logger name — dimmed grey
        logger_name = record.logger.split(".")[-1] if record.logger else ""
        if logger_name:
            if self._colors:
                parts.append(f"{_Ansi.GREY}{_Ansi.DIM}{logger_name}{_Ansi.RESET}")
            else:
                parts.append(logger_name)

        # Message
        parts.append(record.message)

        # Error code — yellow parens
        if record.error_code:
            if self._colors:
                parts.append(f"{_Ansi.YELLOW}({record.error_code}){_Ansi.RESET}")
            else:
                parts.append(f"({record.error_code})")

        # Context — key=value pairs
        if record.context:
            ctx_parts = []
            for k, v in record.context.items():
                if self._colors:
                    ctx_parts.append(f"{_Ansi.DIM}{k}={_Ansi.WHITE}{v}{_Ansi.RESET}")
                else:
                    ctx_parts.append(f"{k}={v}")
            parts.append(" ".join(ctx_parts))

        # Exception — enhanced formatting
        if record.exception:
            exc_parts = self._format_exception(record.exception)
            parts.extend(exc_parts)

        return " ".join(parts)

    def _format_exception(self, exception: str) -> list:
        """Enhanced exception formatting with error type detection and file/line info."""
        parts = []

        # Parse exception to extract type, message, and file/line info
        exc_type, exc_msg, file_info = self._parse_exception(exception)

        # Error type badge — colored by category
        if exc_type:
            exc_color = self._get_exception_color(exc_type)
            if self._colors:
                parts.append(f"{exc_color}{_Ansi.BOLD}[{exc_type}]{_Ansi.RESET}")
            else:
                parts.append(f"[{exc_type}]")

        # Exception message
        if exc_msg:
            if self._colors:
                parts.append(f"{_Ansi.RED}{exc_msg}{_Ansi.RESET}")
            else:
                parts.append(exc_msg)

        # File/line info — dimmed
        if file_info:
            if self._colors:
                parts.append(f"{_Ansi.DIM}at {file_info}{_Ansi.RESET}")
            else:
                parts.append(f"at {file_info}")

        return parts

    def _parse_exception(self, exception: str) -> tuple:
        """Parse exception string to extract type, message, and file/line info."""
        import re

        exc_type = None
        exc_msg = exception
        file_info = None

        # Handle multi-line stack traces - get the last line (actual exception)
        lines = exception.strip().split('\n')
        last_line = lines[-1].strip() if lines else exception

        # Extract file/line info from any line
        for line in lines:
            file_match = re.search(r"File '([^']+)', line (\d+)(?:, in (\w+))?", line)
            if file_match:
                file_path = file_match.group(1)
                line_num = file_match.group(2)
                func_name = file_match.group(3)
                file_info = f"{file_path}:{line_num}"
                if func_name:
                    file_info += f" in {func_name}()"
                break

        # Extract exception type and message from the last line
        if ':' in last_line:
            parts = last_line.split(':', 1)
            potential_type = parts[0].strip()

            # Check if it looks like an exception type (PascalCase)
            if potential_type[0].isupper() and not potential_type.startswith('File'):
                exc_type = potential_type
                exc_msg = parts[1].strip() if len(parts) > 1 else ""

        return exc_type, exc_msg, file_info

    def _get_exception_color(self, exc_type: str) -> str:
        """Get color for exception type based on category."""
        # Categorize common exceptions
        if exc_type in ('TypeError', 'ValueError', 'KeyError', 'IndexError', 'AttributeError'):
            return _Ansi.RED + _Ansi.BOLD  # Programming errors
        elif exc_type in ('RuntimeError', 'MemoryError', 'OSError', 'IOError'):
            return _Ansi.MAGENTA + _Ansi.BOLD  # System errors
        elif exc_type in ('TimeoutError', 'ConnectionError', 'KeyboardInterrupt'):
            return _Ansi.YELLOW + _Ansi.BOLD  # Transient/timeout errors
        elif exc_type in ('ImportError', 'ModuleNotFoundError'):
            return _Ansi.CYAN + _Ansi.BOLD  # Dependency errors
        else:
            return _Ansi.RED  # Default red for unknown types

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
