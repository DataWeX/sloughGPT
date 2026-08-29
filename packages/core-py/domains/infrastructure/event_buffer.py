"""
Thread-safe ring buffer for dashboard events.

Captures punchy one-liner summaries from tagged log records and stores
them for the /dashboard/stream SSE endpoint and CLI monitor.

Usage::

    from domains.infrastructure.event_buffer import get_event_buffer
    buf = get_event_buffer()
    buf.record("TRAIN", "Train step 310/500 — loss 2.341")
    events = buf.recent(20)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, asdict
from threading import Lock
from typing import Optional


@dataclass(frozen=True)
class DashboardEvent:
    """A single punchy event for the dashboard."""
    ts: float
    category: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


class EventBuffer:
    """Thread-safe ring buffer of dashboard events.

    Stores the last ``maxlen`` events. Each event is a timestamped
    one-liner with a category tag (MODEL, TRAIN, INFERENCE, SYSTEM, ERROR).
    """

    def __init__(self, maxlen: int = 50):
        self._events: deque[DashboardEvent] = deque(maxlen=maxlen)
        self._lock = Lock()

    def record(self, category: str, message: str, ts: Optional[float] = None) -> None:
        """Append an event to the buffer."""
        event = DashboardEvent(
            ts=ts or time.time(),
            category=category,
            message=message,
        )
        with self._lock:
            self._events.append(event)

    def recent(self, n: int = 20) -> list[dict]:
        """Return the last N events as dicts (newest first)."""
        with self._lock:
            items = list(self._events)[-n:]
        return [e.to_dict() for e in reversed(items)]

    def clear(self) -> None:
        """Flush all events."""
        with self._lock:
            self._events.clear()


_buffer: Optional[EventBuffer] = None


def get_event_buffer() -> EventBuffer:
    """Get (or create) the singleton EventBuffer."""
    global _buffer
    if _buffer is None:
        _buffer = EventBuffer()
    return _buffer
