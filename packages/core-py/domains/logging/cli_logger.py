"""
CLILogger — native ANSI output for ``sloughgpt`` CLI commands.

Inherits Logger and routes records through raw ANSI escape codes for
formatted terminal output. No Rich dependency — pure TTY.

Usage::

    from domains.logging import CLILogger, LogLevel

    log = CLILogger("slo.cli")
    log.info("model loaded", model="gpt2", params="124M")
    log.success("training complete", loss="0.42")
"""

from __future__ import annotations

import json as _json
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, TextIO

from .base import Logger, LogLevel, LogRecord


# ── TTY gate ────────────────────────────────────────────────────────────

_TERMINAL_ENABLED = True


def set_cli_terminal(enabled: bool) -> None:
    """Enable or disable terminal output (used by the shell TUI)."""
    global _TERMINAL_ENABLED
    _TERMINAL_ENABLED = enabled


# ── ANSI codes ──────────────────────────────────────────────────────────


def _color_enabled(stream: Optional[TextIO] = None) -> bool:
    """Auto-detect color support. Respects NO_COLOR / FORCE_COLOR / TTY."""
    no_color = os.environ.get("NO_COLOR", "").strip() == "1"
    force_color = os.environ.get("FORCE_COLOR", "").strip() == "1"
    slo_color = os.environ.get("SLO_LOG_COLOR", "").strip().lower()

    if no_color:
        return False
    if slo_color in ("1", "true", "yes", "on"):
        return True
    if slo_color in ("0", "false", "no", "off"):
        return False
    if force_color:
        return True
    try:
        s = stream or sys.stdout
        return hasattr(s, "isatty") and s.isatty()
    except (AttributeError, ValueError):
        return False


class _A:
    """ANSI escape codes."""
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


# ── Level → style mapping ──────────────────────────────────────────────

_LEVEL_STYLE = {
    LogLevel.DEBUG:    (_A.DIM + _A.CYAN,            "·",  "debug"),
    LogLevel.INFO:     (_A.GREEN,                     "ℹ",  "info"),
    LogLevel.WARNING:  (_A.BOLD + _A.YELLOW,          "!",  "warning"),
    LogLevel.ERROR:    (_A.BOLD + _A.RED,             "✗",  "error"),
    LogLevel.CRITICAL: (_A.BG_RED + _A.BOLD + _A.WHITE, "✗", "critical"),
}


# ── Terminal width ──────────────────────────────────────────────────────

def _term_width() -> int:
    """Get terminal width, fallback to 80."""
    try:
        return os.get_terminal_size().columns
    except (AttributeError, ValueError, OSError):
        return 80


# ── Helpers ─────────────────────────────────────────────────────────────

def _c(text: str, color: str, enabled: bool) -> str:
    """Wrap text in ANSI color if enabled."""
    if enabled:
        return f"{color}{text}{_A.RESET}"
    return text


def _write(stream: TextIO, text: str) -> None:
    """Write to stream, swallow errors."""
    try:
        stream.write(text)
        stream.flush()
    except (OSError, ValueError):
        pass


def _is_tty(stream: TextIO) -> bool:
    """Return True if *stream* is an interactive terminal."""
    return hasattr(stream, "isatty") and stream.isatty()


# ── CLILogger ───────────────────────────────────────────────────────────

class CLILogger(Logger):
    """Native ANSI CLI logger — inherits from Logger, outputs via escape codes.

    Supports all base Logger methods plus CLI-specific helpers:
    ``success()``, ``step()``, ``header()``, ``section()``, ``table()``,
    ``json()``, ``status()``, ``key_value()``, ``command()``.

    Parameters:
        name:    Logger name (e.g. ``"slo.cli"``).
        level:   Minimum severity to emit.
        stream:  Output stream (default ``sys.stdout``).
        colors:  Enable ANSI color output (default: auto-detect TTY).
        context: Default context attached to every record.
    """

    _HIDE_CURSOR = "\033[?25l"
    _SHOW_CURSOR = "\033[?25h"

    def __init__(
        self,
        name: str = "slo.cli",
        level: LogLevel = LogLevel.INFO,
        stream: Optional[TextIO] = None,
        colors: Optional[bool] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, level=level, context=context)
        self._stream = stream or sys.stdout
        self._colors = _color_enabled(self._stream) if colors is None else colors
        self._cursor_hidden = False
        from .config import LogFormatter
        self._formatter = LogFormatter(fmt="cli", colors=self._colors)

    # ── Cursor lifecycle ─────────────────────────────────────────────────

    def hide_cursor(self) -> None:
        """Hide the terminal cursor. Safe to call multiple times."""
        if not self._cursor_hidden and _is_tty(self._stream):
            _write(self._stream, self._HIDE_CURSOR)
            self._cursor_hidden = True

    def show_cursor(self) -> None:
        """Restore the terminal cursor. Safe to call multiple times."""
        if self._cursor_hidden and _is_tty(self._stream):
            _write(self._stream, self._SHOW_CURSOR)
            self._cursor_hidden = False

    def cursor_up(self, n: int = 1) -> None:
        """Move cursor up n lines."""
        if _is_tty(self._stream) and n > 0:
            _write(self._stream, f"\033[{n}A")

    def cursor_down(self, n: int = 1) -> None:
        """Move cursor down n lines."""
        if _is_tty(self._stream) and n > 0:
            _write(self._stream, f"\033[{n}B")

    def clear_line(self) -> None:
        """Clear the current line."""
        if _is_tty(self._stream):
            _write(self._stream, "\033[2K")

    def clear_lines(self, n: int = 1) -> None:
        """Clear n lines starting from cursor position up."""
        if _is_tty(self._stream):
            for _ in range(n):
                self.cursor_up(1)
                self.clear_line()

    def save_position(self) -> None:
        """Save cursor position."""
        if _is_tty(self._stream):
            _write(self._stream, "\033[s")

    def restore_position(self) -> None:
        """Restore cursor position."""
        if _is_tty(self._stream):
            _write(self._stream, "\033[u")

    # ── Core emit ───────────────────────────────────────────────────────

    def emit(self, record: LogRecord) -> None:
        """Format and write the record to the output stream (thread-safe)."""
        if not _TERMINAL_ENABLED:
            return
        line = self._formatter.format_oop(record)
        with self._lock:
            _write(self._stream, line + "\n")

    # ── CLI-specific helpers ────────────────────────────────────────────

    def success(self, msg: str, **ctx: Any) -> None:
        """Log a success (green checkmark)."""
        if not _TERMINAL_ENABLED:
            return
        import time as _time
        from datetime import datetime as _dt
        ts = _dt.fromtimestamp(_time.time()).strftime("%H:%M:%S")
        c = self._colors
        primary = f"  {_c(ts, _A.DIM, c)} {_c('✓', _A.GREEN, c)} {msg}"
        meta = " ".join(f"{k}={v}" for k, v in ctx.items()) if ctx else ""
        with self._lock:
            _write(self._stream, primary + "\n")
            if meta:
                _write(self._stream, f"    {_c(meta, _A.DIM, c)}\n")

    def step(self, msg: str, **ctx: Any) -> None:
        """Log a step/action (cyan arrow)."""
        if not _TERMINAL_ENABLED:
            return
        import time as _time
        from datetime import datetime as _dt
        ts = _dt.fromtimestamp(_time.time()).strftime("%H:%M:%S")
        c = self._colors
        primary = f"  {_c(ts, _A.DIM, c)} {_c('→', _A.CYAN, c)} {msg}"
        meta = " ".join(f"{k}={v}" for k, v in ctx.items()) if ctx else ""
        with self._lock:
            _write(self._stream, primary + "\n")
            if meta:
                _write(self._stream, f"    {_c(meta, _A.DIM, c)}\n")

    def header(self, title: str, char: str = "─") -> None:
        """Print a bold header with a separator line."""
        if not _TERMINAL_ENABLED:
            return
        c = self._colors
        width = _term_width()
        with self._lock:
            _write(self._stream, "\n")
            _write(self._stream, _c(f"  {title}", _A.BOLD, c) + "\n")
            _write(self._stream, _c(f"  {char * (width - 4)}", _A.DIM, c) + "\n")

    def section(self, title: str) -> None:
        """Print a section divider."""
        if not _TERMINAL_ENABLED:
            return
        c = self._colors
        width = _term_width()
        with self._lock:
            _write(self._stream, "\n")
            _write(self._stream, _c(f"  {title}", _A.BOLD, c) + "\n")
            _write(self._stream, _c(f"  {'·' * (width - 4)}", _A.DIM, c) + "\n")

    def table(
        self,
        headers: List[str],
        rows: List[List[str]],
        align: Optional[List[str]] = None,
    ) -> None:
        """Print an ASCII table with column alignment."""
        if not rows:
            return
        if not _TERMINAL_ENABLED:
            return
        c = self._colors

        # Compute column widths
        col_count = len(headers)
        widths = [len(h) for h in headers]
        for row in rows:
            for i in range(min(len(row), col_count)):
                widths[i] = max(widths[i], len(str(row[i])))

        # Parse alignment
        def _align_col(i: int) -> str:
            if align and i < len(align):
                return align[i]
            return "l"

        def _fmt_cell(text: str, width: int, a: str) -> str:
            t = str(text)
            if a == "r":
                return t.rjust(width)
            elif a == "c":
                return t.center(width)
            return t.ljust(width)

        with self._lock:
            # Header
            header_parts = []
            for i, h in enumerate(headers):
                cell = _fmt_cell(h, widths[i], _align_col(i))
                header_parts.append(_c(cell, _A.BOLD, c))
            _write(self._stream, "  ".join(header_parts) + "\n")

            # Separator
            sep_parts = [_c("-" * w, _A.DIM, c) for w in widths]
            _write(self._stream, "  ".join(sep_parts) + "\n")

            # Rows
            for row in rows:
                cells = []
                for i in range(col_count):
                    val = str(row[i]) if i < len(row) else ""
                    cells.append(_fmt_cell(val, widths[i], _align_col(i)))
                _write(self._stream, "  ".join(cells) + "\n")

    def json(self, data: Any, indent: int = 2) -> None:
        """Pretty-print JSON."""
        if not _TERMINAL_ENABLED:
            return
        text = _json.dumps(data, indent=indent, default=str, ensure_ascii=False)
        with self._lock:
            _write(self._stream, text + "\n")

    def status(self, label: str, value: str, status: str = "ok") -> None:
        """Print a key-value status line with a colored indicator."""
        if not _TERMINAL_ENABLED:
            return
        c = self._colors
        colors = {"ok": _A.GREEN, "warn": _A.YELLOW, "error": _A.RED, "info": _A.BLUE}
        icons = {"ok": "✓", "warn": "!", "error": "✗", "info": "ℹ"}
        color = colors.get(status, _A.WHITE)
        icon = icons.get(status, "•")
        with self._lock:
            _write(self._stream, f"  {_c(icon, color, c)} {label}: {value}\n")

    def divider(self, char: str = "-") -> None:
        """Print a separator line."""
        if not _TERMINAL_ENABLED:
            return
        c = self._colors
        with self._lock:
            _write(self._stream, _c(char * _term_width(), _A.DIM, c) + "\n")

    def key_value(self, key: str, value: str, indent: int = 2) -> None:
        """Print a dim key: value pair."""
        if not _TERMINAL_ENABLED:
            return
        c = self._colors
        padding = " " * indent
        with self._lock:
            if key:
                _write(self._stream, f"{padding}{_c(key + ':', _A.DIM, c)} {value}\n")
            else:
                _write(self._stream, f"{padding}{value}\n")

    def blank(self, count: int = 1) -> None:
        """Print blank lines."""
        if not _TERMINAL_ENABLED:
            return
        with self._lock:
            for _ in range(count):
                _write(self._stream, "\n")

    def command(self, cmd: str, description: str = "") -> None:
        """Print a command with optional description."""
        if not _TERMINAL_ENABLED:
            return
        c = self._colors
        with self._lock:
            line = _c(f"  {cmd:<30}", _A.CYAN, c)
            if description:
                line += _c(f" {description}", _A.DIM, c)
            _write(self._stream, line + "\n")

    # ── Timing ──────────────────────────────────────────────────────────

    @contextmanager
    def timer(self, label: str = "elapsed") -> Generator[None, None, None]:
        """Context manager that logs elapsed time on exit.

        Usage::

            with log.timer("model load"):
                load_model()
            # prints: ℹ [info] slo.cli model load (42ms)
        """
        start = time.monotonic()
        try:
            yield
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000
            self.info(label, elapsed_ms=f"{elapsed_ms:.0f}ms")
