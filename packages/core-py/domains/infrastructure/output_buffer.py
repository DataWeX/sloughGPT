"""
Shared output buffer — ring buffer with subscriber support for SSE streaming.

Moved from shell/stdio.py to shared infrastructure so both the shell REPL
and the API server can use the same buffer. Extended with:
  - Subscriber cursors for real-time SSE streaming
  - Log handler integration (logging.Handler subclass)
  - JSON serialization for SSE events
  - Singletons for server-wide use
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, List, Optional


class OutputLine:
    """A single line of output with style and metadata."""

    __slots__ = ("text", "style", "indent", "level", "source", "timestamp")

    def __init__(
        self,
        text: str = "",
        style: str = "",
        indent: int = 0,
        level: str = "info",
        source: str = "server",
        timestamp: float | None = None,
    ) -> None:
        self.text = text
        self.style = style       # ANSI prefix (shell) or level tag (server)
        self.indent = indent
        self.level = level
        self.source = source
        self.timestamp = timestamp or time.time()

    def render(self, width: int = 0, color: bool = True) -> str:
        """Render for terminal output (shell mode)."""
        prefix = " " * self.indent
        line = prefix + self.text
        if color and self.style:
            return self.style + line + "\033[0m"
        return line

    def to_dict(self) -> dict:
        """Serialize to dict (server mode)."""
        return {
            "text": self.text,
            "level": self.level,
            "source": self.source,
            "ts": self.timestamp,
        }

    def to_sse(self) -> str:
        """Serialize to JSON for SSE event."""
        import json
        return json.dumps(self.to_dict())

    def __repr__(self) -> str:
        return f"OutputLine({self.text[:30]!r})"


class OutputBuffer:
    """Thread-safe ring buffer of OutputLines with subscriber support.

    Used by:
      - Shell REPL (via StdioWriter) for terminal output + pager
      - API server for capturing all log/stdout output → SSE streaming
      - Training pipelines for real-time loss/progress streaming

    Thread safety: all mutations go through ``self._lock``.
    Subscribers get a cursor that wakes on each append.
    """

    def __init__(self, max_lines: int = 5000) -> None:
        self._lines: list[OutputLine] = []
        self._max = max_lines
        self._view_top = 0        # first visible line index (shell pager)
        self._view_height = 0     # number of visible lines (shell pager)
        self._lock = threading.Lock()
        self._subscribers: dict[str, _Subscriber] = {}
        self._seq = 0

    def append(self, line: OutputLine) -> None:
        """Append a line and notify all subscribers."""
        with self._lock:
            self._lines.append(line)
            self._seq += 1
            if len(self._lines) > self._max:
                excess = len(self._lines) - self._max
                self._lines = self._lines[excess:]
                self._view_top = max(0, self._view_top - excess)
            for sub in self._subscribers.values():
                sub._notify(line)

    def append_text(self, text: str, style: str = "", indent: int = 0, **kwargs) -> None:
        """Append raw text as an OutputLine."""
        self.append(OutputLine(text, style, indent, **kwargs))

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()
            self._view_top = 0
            self._seq = 0

    @property
    def lines(self) -> list[OutputLine]:
        return self._lines

    @property
    def count(self) -> int:
        return len(self._lines)

    @property
    def seq(self) -> int:
        return self._seq

    # ── Tail access (server mode) ──

    def tail(self, n: int = 100) -> list[OutputLine]:
        """Get last N lines."""
        with self._lock:
            return list(self._lines)[-n:]

    def tail_dicts(self, n: int = 100) -> list[dict]:
        """Get last N lines as dicts."""
        return [l.to_dict() for l in self.tail(n)]

    # ── Viewport scrolling (shell mode) ──

    def scroll(self, delta: int) -> None:
        with self._lock:
            max_top = max(0, len(self._lines) - self._view_height)
            self._view_top = max(0, min(self._view_top + delta, max_top))

    def scroll_to_bottom(self) -> None:
        with self._lock:
            self._view_top = max(0, len(self._lines) - self._view_height)

    @property
    def visible_lines(self) -> list[OutputLine]:
        with self._lock:
            return self._lines[self._view_top:self._view_top + self._view_height]

    def set_viewport(self, height: int) -> None:
        self._view_height = max(1, height)
        with self._lock:
            max_top = max(0, len(self._lines) - self._view_height)
            self._view_top = max(0, min(self._view_top, max_top))

    # ── Subscribers (server SSE mode) ──

    def subscribe(self, name: str | None = None) -> _Subscriber:
        """Create a subscriber with its own cursor."""
        name = name or f"sub-{id(self)}-{self._seq}"
        sub = _Subscriber(name=name, buffer=self)
        with self._lock:
            self._subscribers[name] = sub
        return sub

    def unsubscribe(self, name: str) -> None:
        with self._lock:
            self._subscribers.pop(name, None)


class _Subscriber:
    """A cursor into OutputBuffer. Non-blocking read via event signal."""

    def __init__(self, name: str, buffer: OutputBuffer):
        self.name = name
        self._buffer = buffer
        self._pending: list[OutputLine] = []
        self._event = threading.Event()

    def _notify(self, line: OutputLine) -> None:
        """Called by buffer on append (under buffer lock)."""
        self._pending.append(line)
        self._event.set()

    def read(self, timeout: float = 0.1) -> list[OutputLine]:
        """Block up to timeout, then return pending lines."""
        self._event.wait(timeout=timeout)
        self._event.clear()
        with self._buffer._lock:
            lines = list(self._pending)
            self._pending.clear()
        return lines

    def read_all(self) -> list[OutputLine]:
        """Non-blocking: return all pending lines."""
        with self._buffer._lock:
            lines = list(self._pending)
            self._pending.clear()
        return lines


# ── Log handler ──────────────────────────────────────────────────────────

class BufferLogHandler(logging.Handler):
    """Routes log records into an OutputBuffer."""

    def __init__(self, buffer: OutputBuffer, level: int = logging.INFO):
        super().__init__(level=level)
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._buffer.append_text(msg, level=record.levelname.lower(), source=record.name)
        except Exception:
            pass


# ── Singletons ──────────────────────────────────────────────────────────

_server_buffer: OutputBuffer | None = None


def get_server_buffer() -> OutputBuffer:
    """Global server output buffer singleton."""
    global _server_buffer
    if _server_buffer is None:
        _server_buffer = OutputBuffer(max_lines=10_000)
    return _server_buffer


def install_log_bridge(buffer: OutputBuffer | None = None, level: int = logging.INFO) -> BufferLogHandler:
    """Install a log handler that routes all logging into the buffer."""
    buf = buffer or get_server_buffer()
    handler = BufferLogHandler(buf, level=level)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    logging.root.addHandler(handler)
    return handler
