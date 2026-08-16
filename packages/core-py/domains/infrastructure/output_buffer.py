"""
Shared output buffer — unified structured log capture.

Captures logs from ALL sources into a single structured schema:
  - Python logging (via BufferLogHandler)
  - stdout/stderr (via TeeWriter)
  - Frontend logs (via /logs/ingest endpoint)
  - Training progress, API requests, lifecycle events

Each log entry carries structured fields (level, source, tag, context)
so the frontend can render without regex parsing.

Used by:
  - Shell REPL (via StdioWriter) for terminal output + pager
  - API server for capturing all log/stdout output → SSE streaming
  - Training pipelines for real-time loss/progress streaming
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from domains.logging.bridge import record_extra_context


@dataclass
class OutputLine:
    """A single structured log entry.

    Attributes:
        text: The log message (clean, without timestamp/level prefix).
        level: Log level — debug, info, warning, error, critical.
        source: Module/source that produced the log (e.g. "slo.startup", "uvicorn").
        tag: Optional category tag (e.g. "START", "INFRA", "MODEL", "REQ").
        context: Extra structured metadata (e.g. {"method": "GET", "status": 200}).
        timestamp: Unix timestamp.
        style: ANSI prefix (shell mode only, ignored in server mode).
        indent: Indentation level (shell mode only).
    """

    text: str = ""
    level: str = "info"
    source: str = ""
    tag: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    style: str = ""
    indent: int = 0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def render(self, width: int = 0, color: bool = True) -> str:
        """Render for terminal output (shell mode)."""
        prefix = " " * self.indent
        line = prefix + self.text
        if color and self.style:
            return self.style + line + "\033[0m"
        return line

    def to_dict(self) -> dict:
        """Serialize to dict for SSE/API transport."""
        d: dict[str, Any] = {
            "text": self.text,
            "level": self.level,
            "source": self.source,
            "ts": self.timestamp,
        }
        if self.tag:
            d["tag"] = self.tag
        if self.context:
            d["context"] = self.context
        return d

    def to_sse(self) -> str:
        """Serialize to JSON for SSE event."""
        return json.dumps(self.to_dict())

    def __repr__(self) -> str:
        tag = f" [{self.tag}]" if self.tag else ""
        return f"OutputLine({self.level}{tag} {self.source}: {self.text[:40]!r})"


class OutputBuffer:
    """Thread-safe ring buffer of OutputLines with subscriber support.

    Thread safety: all mutations go through ``self._lock``.
    Subscribers get a cursor that wakes on each append.
    """

    def __init__(self, max_lines: int = 5000) -> None:
        self._lines: list[OutputLine] = []
        self._max = max_lines
        self._view_top = 0
        self._view_height = 0
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

    def append_text(self, text: str, **kwargs) -> OutputLine:
        """Append raw text as an OutputLine. Kept for backward compat."""
        line = OutputLine(text=text, **kwargs)
        self.append(line)
        return line

    def append_log(
        self,
        text: str,
        level: str = "info",
        source: str = "",
        tag: str = "",
        context: dict[str, Any] | None = None,
    ) -> OutputLine:
        """Append a structured log entry."""
        line = OutputLine(
            text=text,
            level=level,
            source=source,
            tag=tag,
            context=context or {},
        )
        self.append(line)
        return line

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

    def tail(self, n: int = 100) -> list[OutputLine]:
        with self._lock:
            return list(self._lines)[-n:]

    def tail_dicts(self, n: int = 100) -> list[dict]:
        return [l.to_dict() for l in self.tail(n)]

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
            self._view_top = min(self._view_top, max_top)

    def subscribe(self, name: str | None = None) -> _Subscriber:
        name = name or f"sub-{id(self)}-{self._seq}"
        sub = _Subscriber(name=name, buffer=self)
        with self._lock:
            self._subscribers[name] = sub
            if name.startswith("sub-"):
                self._seq += 1
        return sub

    def unsubscribe(self, name: str) -> None:
        with self._lock:
            self._subscribers.pop(name, None)


class _Subscriber:
    """A cursor into OutputBuffer. Non-blocking read via event signal.

    Supports both sync (threading.Event) and async (asyncio.Event) readers.
    The async path avoids blocking the event loop in SSE streaming handlers.

    Thread safety: ``_notify`` is called from background threads (logging,
    stdout).  The threading.Event is inherently thread-safe.  The asyncio.Event
    is woken via ``loop.call_soon_threadsafe`` so the event loop is never
    accessed from a foreign thread.
    """

    def __init__(self, name: str, buffer: OutputBuffer):
        self.name = name
        self._buffer = buffer
        self._pending: list[OutputLine] = []
        self._event = threading.Event()
        self._async_event: asyncio.Event | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None

    def _notify(self, line: OutputLine) -> None:
        self._pending.append(line)
        self._event.set()
        if self._async_event is not None and self._async_loop is not None:
            self._async_loop.call_soon_threadsafe(self._async_event.set)

    def read(self, timeout: float = 0.1) -> list[OutputLine]:
        self._event.wait(timeout=timeout)
        self._event.clear()
        with self._buffer._lock:
            lines = list(self._pending)
            self._pending.clear()
        return lines

    async def async_read(self, timeout: float = 0.2) -> list[OutputLine]:
        """Async read that yields to the event loop instead of blocking.

        Creates the asyncio.Event lazily on first call and captures the
        running loop so ``_notify`` can wake it thread-safely.
        """
        if self._async_event is None:
            self._async_event = asyncio.Event()
            self._async_loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(self._async_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        self._async_event.clear()
        with self._buffer._lock:
            lines = list(self._pending)
            self._pending.clear()
        return lines

    def read_all(self) -> list[OutputLine]:
        with self._buffer._lock:
            lines = list(self._pending)
            self._pending.clear()
        return lines


# ── Capture handlers ────────────────────────────────────────────────────


class BufferLogHandler(logging.Handler):
    """Captures Python logging records into structured OutputLines.

    Extracts structured extras (tag, error_code, context) from the record
    and stores them as fields — not formatted into text.  Any other
    non-standard extra keys are auto-captured into context so structured
    telemetry surfaces in the frontend stream.
    """

    _LEVEL_MAP = {
        logging.DEBUG: "debug",
        logging.INFO: "info",
        logging.WARNING: "warning",
        logging.ERROR: "error",
        logging.CRITICAL: "critical",
    }

    def __init__(self, buffer: OutputBuffer, level: int = logging.INFO):
        super().__init__(level=level)
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            tag = getattr(record, "tag", "")
            error_code = getattr(record, "error_code", None)

            context = record_extra_context(record)
            if error_code:
                context["error_code"] = error_code

            self._buffer.append_log(
                text=msg,
                level=self._LEVEL_MAP.get(record.levelno, "info"),
                source=record.name,
                tag=tag,
                context=context,
            )
        except Exception:
            pass


class _TeeWriter:
    """Tees writes to both the original stream and the OutputBuffer.

    Parses common output formats into structured fields:
      - "HH:MM:SS LVL [TAG] source message"
      - "LEVEL:     message" (uvicorn)
      - Rich CLI output (plain text)
      - System warnings
    """

    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    # "HH:MM:SS LVL [TAG] source message"
    _STRUCTURED_RE = re.compile(
        r"^(\d{2}:\d{2}:\d{2})\s+"
        r"(INF|WRN|ERR|DBG|INFO|WARNING|ERROR|DEBUG)\s+"
        r"(?:\[([^\]]+)\]\s+)?"
        r"(\S+)\s+"
        r"(.*)$"
    )
    # "LEVEL:     message" (uvicorn)
    _UVICORN_RE = re.compile(r"^(INFO|WARNING|ERROR|DEBUG):\s{2,}(.*)$")
    # "(process:PID): program-WARNING **: time: message"
    _SYSTEM_RE = re.compile(
        r"^\(.*?\):\s+\S+[-](?:WARNING|ERROR)\s+\*\*:\s+\d{2}:\d{2}:\d{2}\.\d+:\s+(.*)$"
    )

    def __init__(self, original, buffer: OutputBuffer, source: str = "stdout"):
        self._original = original
        self._buffer = buffer
        self._source = source
        self._buf = ""

    def write(self, data: str):
        self._original.write(data)
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = self._ANSI_RE.sub("", line.rstrip())
            if not line.strip():
                continue
            self._parse_and_append(line)

    def _parse_and_append(self, line: str):
        """Parse a raw stdout line into structured fields."""
        # Structured: "HH:MM:SS LVL [TAG] source message"
        m = self._STRUCTURED_RE.match(line)
        if m:
            self._buffer.append_log(
                text=m.group(5),
                level=_norm_level(m.group(2)),
                source=m.group(4),
                tag=m.group(3) or "",
            )
            return

        # Uvicorn: "LEVEL:     message"
        m = self._UVICORN_RE.match(line)
        if m:
            self._buffer.append_log(
                text=m.group(2),
                level=_norm_level(m.group(1)),
                source="uvicorn",
            )
            return

        # System warning
        m = self._SYSTEM_RE.match(line)
        if m:
            self._buffer.append_log(
                text=m.group(1),
                level="warning",
                source="system",
            )
            return

        # Plain text — store as-is
        self._buffer.append_log(text=line, level="info", source=self._source)

    def flush(self):
        self._original.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


def _norm_level(raw: str) -> str:
    """Normalize level strings to lowercase."""
    low = raw.lower()
    if low in ("inf", "info"):
        return "info"
    if low in ("wrn", "warning"):
        return "warning"
    if low in ("err", "error"):
        return "error"
    if low in ("dbg", "debug"):
        return "debug"
    return low


# ── Singletons + installation ──────────────────────────────────────────

_server_buffer: OutputBuffer | None = None


def get_server_buffer() -> OutputBuffer:
    global _server_buffer
    if _server_buffer is None:
        _server_buffer = OutputBuffer(max_lines=10_000)
    return _server_buffer


def install_log_bridge(buffer: OutputBuffer | None = None, level: int = logging.INFO) -> BufferLogHandler:
    """Install a log handler that routes all Python logging into the buffer."""
    buf = buffer or get_server_buffer()
    handler = BufferLogHandler(buf, level=level)
    logging.root.addHandler(handler)
    return handler


def install_stdio_bridge(buffer: OutputBuffer | None = None):
    """Tee stdout/stderr into the OutputBuffer with structured parsing."""
    buf = buffer or get_server_buffer()
    import sys
    sys.stdout = _TeeWriter(sys.stdout, buf, "stdout")
    sys.stderr = _TeeWriter(sys.stderr, buf, "stderr")
