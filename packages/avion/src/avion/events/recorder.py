"""Event recorder — append-only log with filtering, export, listeners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from avion.events.models import Event, EventType


class EventRecorder:
    """Records events for a session.

    Usage::

        rec = EventRecorder(session_name="chat_check")
        rec.record(EventType.NAVIGATE, name="/chat", success=True)
        rec.record(EventType.CLICK, name="text='Start'")
        failures = rec.failed()
        rec.save("arken_output/events.json")
    """

    def __init__(self, session_name: str = "arken_session"):
        self.session_name = session_name
        self._events: list[Event] = []
        self._listeners: list[Callable[[Event], None]] = []
        self._seq = 0

    def __len__(self) -> int:
        return len(self._events)

    def on_event(self, listener: Callable[[Event], None]) -> None:
        """Register a callback fired for every recorded event."""
        self._listeners.append(listener)

    def record(
        self,
        event_type: EventType,
        name: str = "",
        data: dict[str, Any] | None = None,
        success: bool = True,
        error: str = "",
        duration_ms: float = 0.0,
        screenshot: bytes | None = None,
    ) -> Event:
        """Append an event and notify listeners."""
        self._seq += 1
        payload = dict(data or {})
        if screenshot is not None:
            payload["screenshot"] = screenshot
        event = Event(
            type=event_type,
            name=name,
            data=payload,
            success=success,
            error=error,
            duration_ms=duration_ms,
            seq=self._seq,
        )
        self._events.append(event)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass
        return event

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def of_type(self, *types: EventType) -> list[Event]:
        """Filter events by type."""
        return [e for e in self._events if e.type in types]

    def failed(self) -> list[Event]:
        """Events with success=False."""
        return [e for e in self._events if not e.success]

    def stats(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        for e in self._events:
            by_type[e.type.value] = by_type.get(e.type.value, 0) + 1
        return {
            "session": self.session_name,
            "total": len(self._events),
            "failed": len(self.failed()),
            "by_type": by_type,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.session_name,
            "events": [e.to_dict() for e in self._events],
        }

    def save(self, path: str) -> None:
        """Export events as JSON (screenshots stored as presence flags)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def clear(self) -> None:
        self._events.clear()
