"""
LineModeLogDisplay — shows server/infra logs in the shell's line mode.

Instead of dumping logs inline (noisy), this module provides:
  1. A status badge on the prompt showing unread warning/error count
  2. A ``logs`` command to view recent log entries
  3. A ``--last`` flag to peek at the most recent log

The TUI mode already has a dedicated console pane; this module fills the
gap for the default line-mode shell.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .log_buffer import LogBuffer

# ── ANSI color constants (disabled via NO_COLOR env var) ─────────────

_COLOR_ENABLED = not os.environ.get("NO_COLOR")
if _COLOR_ENABLED:
    _C_RED = "\033[31m"
    _C_YELLOW = "\033[33m"
    _C_GREEN = "\033[32m"
    _C_DIM = "\033[2m"
    _C_BOLD = "\033[1m"
    _C_RESET = "\033[0m"
else:
    _C_RED = _C_YELLOW = _C_GREEN = _C_DIM = _C_BOLD = _C_RESET = ""

# Level → colour mapping
_LEVEL_COLORS = {
    "DEBUG": _C_DIM,
    "INFO": _C_GREEN,
    "WARNING": _C_YELLOW,
    "ERROR": _C_RED,
    "CRITICAL": _C_RED + _C_BOLD,
}

# Level → short label
_LEVEL_LABELS = {
    "DEBUG": "DBG",
    "INFO": "INF",
    "WARNING": "WRN",
    "ERROR": "ERR",
    "CRITICAL": "CRT",
}


class LineModeLogDisplay:
    """Manages log visibility for the line-mode shell.

    The display tracks unread warnings/errors and renders a badge on the
    prompt.  Calling ``poll()`` before each prompt render checks for new
    entries.  The ``logs`` command calls ``clear_counts()`` to dismiss
    the badge.

    Attributes:
        unread_warnings: Number of warning-level entries not yet shown.
        unread_errors: Number of error/critical entries not yet shown.
    """

    def __init__(self, buffer: LogBuffer) -> None:
        self._buffer = buffer
        self._last_index: int = len(buffer)
        self.unread_warnings: int = 0
        self.unread_errors: int = 0

    def poll(self) -> None:
        """Check for new log entries and update unread counts.

        Call this once per prompt render (before ``badge()``).
        """
        entries = self._buffer.get()
        new_entries = entries[self._last_index:]
        self._last_index = len(entries)

        for entry in new_entries:
            if entry.level == "WARNING":
                self.unread_warnings += 1
            elif entry.level in ("ERROR", "CRITICAL"):
                self.unread_errors += 1

    def badge(self) -> str:
        """Return a prompt badge string for unread warnings/errors.

        Returns:
            Empty string when nothing is unread, otherwise a coloured
            badge like `` \\033[33m\\u26a02\\033[0m`` (2 warnings).

        Examples::

            ""              # nothing unread
            " \\u26a02"     # 2 warnings (yellow)
            " \\u27151"     # 1 error (red)
        """
        if self.unread_errors > 0:
            return f" {_C_RED}\u2715{self.unread_errors}{_C_RESET}"
        if self.unread_warnings > 0:
            return f" {_C_YELLOW}\u26a0{self.unread_warnings}{_C_RESET}"
        return ""

    def clear_counts(self) -> None:
        """Reset unread counters (user has acknowledged via ``logs``)."""
        self.unread_warnings = 0
        self.unread_errors = 0

    @staticmethod
    def _format_entry(entry: LogEntry) -> str:
        """Format a single log entry as a coloured string.

        Args:
            entry: The log entry to format.

        Returns:
            Formatted string with timestamp, level badge, source, and message.
        """
        ts = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S")
        color = _LEVEL_COLORS.get(entry.level, "")
        label = _LEVEL_LABELS.get(entry.level, entry.level[:3].upper())
        src = entry.source.split(".")[-1] if entry.source else ""
        msg = entry.message[:120]
        return f"  {_C_DIM}{ts}{_C_RESET} {color}{label}{_C_RESET} {_C_DIM}{src}{_C_RESET} {msg}"

    def render_recent(
        self,
        n: int = 20,
        level: str | None = None,
        source: str | None = None,
    ) -> str:
        """Render the last *n* log entries as a formatted string.

        Args:
            n: Maximum entries to show.
            level: Optional filter — ``"WARNING"``, ``"ERROR"``, ``"INFO"``, etc.
            source: Optional source substring filter.

        Returns:
            Multi-line string suitable for printing.
        """
        entries = self._buffer.get(level=level, source=source, limit=n)
        if not entries:
            return f"  {_C_DIM}No log entries{_C_RESET}"
        return "\n".join(self._format_entry(e) for e in entries)

    def render_last(self) -> str:
        """Render just the most recent log entry (for ``--last`` flag)."""
        entries = self._buffer.get(limit=1)
        if not entries:
            return f"  {_C_DIM}No log entries{_C_RESET}"
        return self._format_entry(entries[0])
