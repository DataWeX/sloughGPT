"""
surface — content surfaces for the shell TUI.

Follows the split-window model: the layout engine (``pane``) decides where
each pane sits, and a *surface* draws the pane's content.  Surfaces have no
curses dependency — they render into a plain list of ``RenderLine`` records
(already clipped to the pane width) that the display layer blits to the
screen.  This keeps content code testable and renderer-agnostic.
"""

from __future__ import annotations

import re
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .log_buffer import LogBuffer

# Style hints — mapped to colour pairs by the display layer.
STYLE_INFO = "info"
STYLE_WARN = "warn"
STYLE_ERROR = "error"
STYLE_DEBUG = "debug"
STYLE_CRITICAL = "critical"


@dataclass
class RenderLine:
    """A single clipped line plus an optional style hint."""

    text: str
    style: str | None = None


# CSI / OSC / charset / single-byte ANSI escapes emitted by console helpers
# (e.g. ``_C_CYAN``) or relayed subprocess output.  A text surface stores plain
# text only — curses would otherwise render these as literal ``^[[..`` glyphs.
_ANSI_RE = re.compile(
    r"\x1b(?:\[[0-9;:?]*[ -/]*[@-~]|[\]()].|[PX^_].*?\x1b\\|.)"
)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences, keeping visible characters."""
    return _ANSI_RE.sub("", text)


def clip(text: str, width: int) -> str:
    """Truncate ``text`` to ``width`` columns (no ANSI handling).

    If the text exceeds ``width``, it is truncated with an ellipsis marker
    so the user can see that content was clipped rather than silently losing
    the tail.
    """
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 1] + "\u2026"


class Surface:
    """Base class: draws content into a region."""

    def set_width(self, cols: int) -> None:
        """Record the available width for line clipping."""
        raise NotImplementedError

    def render(self, rows: int) -> list[RenderLine]:
        """Return up to ``rows`` clipped lines for the display layer."""
        raise NotImplementedError


class TextSurface(Surface):
    """Buffers written text and renders the tail (auto-scroll).

    Mirrors ``ConsoleIO`` semantics: ``write(text, end)`` appends
    ``text + end``; an empty ``end`` leaves a partial line open so the
    next write continues it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lines: deque[str] = deque(maxlen=2000)
        self._partial = ""
        self._width = 80

    def set_width(self, cols: int) -> None:
        self._width = max(cols, 1)

    def write(self, text: str, end: str = "\n") -> None:
        """Append text, keeping a trailing partial line open.

        ANSI escape sequences are stripped so the surface holds plain text.
        """
        chunk = strip_ansi(text) + end if text else end
        with self._lock:
            parts = chunk.split("\n")
            self._partial += parts[0]
            for p in parts[1:]:
                self._lines.append(self._partial)
                self._partial = p

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()
            self._partial = ""

    @property
    def capture(self) -> list[str]:
        with self._lock:
            out = list(self._lines)
            if self._partial:
                out.append(self._partial)
            return out

    def render(self, rows: int, offset: int = 0) -> list[RenderLine]:
        """Return up to ``rows`` clipped lines for the display layer.

        ``offset`` scrolls back from the tail (0 = follow latest output).
        """
        if rows <= 0:
            return []
        with self._lock:
            lines = list(self._lines)
            if self._partial:
                lines.append(self._partial)
            if offset > 0:
                start = max(len(lines) - rows - offset, 0)
                lines = lines[start:start + rows]
            else:
                lines = lines[-rows:]
        return [RenderLine(clip(ln, self._width)) for ln in lines]


class LogSurface(Surface):
    """Tails a LogBuffer, formatting entries with level colour hints."""

    _LEVEL_STYLE = {
        "CRITICAL": STYLE_CRITICAL,
        "ERROR": STYLE_ERROR,
        "WARNING": STYLE_WARN,
        "INFO": STYLE_INFO,
        "DEBUG": STYLE_DEBUG,
    }

    def __init__(self, log_buffer: LogBuffer) -> None:
        self._buffer = log_buffer
        self._last_seen = 0
        self._width = 80

    def set_width(self, cols: int) -> None:
        self._width = max(cols, 1)

    def _format(self) -> list[tuple[str, str | None]]:
        entries = self._buffer.get()
        rows: list[tuple[str, str | None]] = []
        for e in entries:
            ts = datetime.fromtimestamp(e.timestamp).strftime("%H:%M:%S")
            style = self._LEVEL_STYLE.get(e.level.upper())
            rows.append((f"{ts} {e.level:<7s} {e.source}  {e.message}", style))
        return rows

    def render(self, rows: int, offset: int = 0) -> list[RenderLine]:
        if rows <= 0:
            return []
        all_rows = self._format()
        if offset > 0:
            start = max(len(all_rows) - rows - offset, 0)
            window = all_rows[start:start + rows]
        else:
            window = all_rows[-rows:]
        return [RenderLine(clip(t, self._width), s) for t, s in window]
