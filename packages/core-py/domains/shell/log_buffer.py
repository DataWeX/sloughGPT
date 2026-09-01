"""
LogBuffer — thread-safe ring buffer for collecting infra and API server logs.

Feeds from two sources:
1. Standard Python ``logging`` handlers (``LogBufferHandler``) — captures all
   ``slo.*`` logger output (kernel, runtime, init, commands, API server subprocess).
2. ``ShellLogger`` bridge — captures the REPL's own diagnostic log records.

Shell display output (``Console`` → ``ConsoleIO`` → ``/dev/tty``) is NOT captured
here — this buffer is for infrastructure observability only.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field


@dataclass
class LogEntry:
    timestamp: float
    level: str
    source: str
    message: str
    context: dict = field(default_factory=dict)


class LogBuffer:
    """Thread-safe ring buffer of log entries."""

    def __init__(self, max_size: int = 2000) -> None:
        self._max_size = max_size
        self._entries: deque[LogEntry] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def append(self, entry: LogEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    def get(
        self, level: str | None = None,
        source: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[LogEntry]:
        with self._lock:
            entries = list(self._entries)
        if level:
            entries = [e for e in entries if e.level.upper() == level.upper()]
        if source:
            entries = [e for e in entries if source.lower() in e.source.lower()]
        if offset:
            entries = entries[offset:]
        if limit:
            entries = entries[-limit:]
        return entries

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


# ── Singleton ──────────────────────────────────────────────────────────────

_BUFFER: LogBuffer | None = None
_BUFFER_LOCK = threading.Lock()


def get_log_buffer() -> LogBuffer:
    global _BUFFER
    if _BUFFER is None:
        with _BUFFER_LOCK:
            if _BUFFER is None:
                _BUFFER = LogBuffer()
    return _BUFFER


# ── Python logging handler bridge ──────────────────────────────────────────


class LogBufferHandler(logging.Handler):
    """Python logging.Handler that feeds records into the LogBuffer.

    Attach to any standard ``logging.Logger`` to capture its output::

        logger = logging.getLogger("slo.kernel")
        logger.addHandler(LogBufferHandler())
    """

    def __init__(self, buffer: LogBuffer | None = None) -> None:
        super().__init__()
        self._buffer = buffer if buffer is not None else get_log_buffer()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = LogEntry(
                timestamp=record.created,
                level=record.levelname,
                source=record.name,
                message=record.getMessage(),
            )
            self._buffer.append(entry)
        except Exception:
            self.handleError(record)
