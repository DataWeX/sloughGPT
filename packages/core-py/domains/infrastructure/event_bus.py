"""
Event Bus — typed async pub/sub for decoupled component communication.

Components register handlers via on()/once() and fire events via emit().
Error isolation: one bad handler never crashes the bus.
Optional history for late subscribers via replay().
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Union

logger = logging.getLogger("man.event_bus")


class EventPriority(int, Enum):
    MONITOR = 0    # Logging, metrics — never block
    NORMAL = 1     # Standard handlers
    HIGH = 2       # Core state updates (must run before monitors)
    CRITICAL = 3   # Lifecycle — must run before anything else


@dataclass
class Event:
    """Envelope for every event on the bus."""
    name: str
    data: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)
    source: str = ""


EventHandler = Callable[..., Union[Awaitable[Any], Any]]


@dataclass
class Subscription:
    handler: EventHandler
    priority: EventPriority = EventPriority.NORMAL
    once: bool = False


class EventBus:
    """Async pub-sub event bus with priorities, history, error isolation."""

    def __init__(self, max_history: int = 100):
        self._subscriptions: dict[str, list[Subscription]] = defaultdict(list)
        self._wildcards: list[Subscription] = []
        self._history: dict[str, list[Event]] = defaultdict(list)
        self._max_history = max_history

    # ── Subscribe ──

    def on(
        self,
        event: str,
        handler: EventHandler,
        priority: EventPriority = EventPriority.NORMAL,
    ):
        """Subscribe `handler` to `event`. Handler receives `(event_name, data)`."""
        sub = Subscription(handler=handler, priority=priority, once=False)
        if event == "*":
            self._wildcards.append(sub)
        else:
            self._subscriptions[event].append(sub)
        self._subscriptions[event].sort(key=lambda s: s.priority, reverse=True)

    def once(
        self,
        event: str,
        handler: EventHandler,
        priority: EventPriority = EventPriority.NORMAL,
    ):
        """Subscribe for a single emission, then auto-remove."""
        sub = Subscription(handler=handler, priority=priority, once=True)
        if event == "*":
            self._wildcards.append(sub)
        else:
            self._subscriptions[event].append(sub)
        self._subscriptions[event].sort(key=lambda s: s.priority, reverse=True)

    def off(self, event: str, handler: EventHandler) -> bool:
        """Unsubscribe a specific handler from an event. Returns True if removed."""
        if event == "*":
            before = len(self._wildcards)
            self._wildcards = [s for s in self._wildcards if s.handler != handler]
            return len(self._wildcards) < before
        subs = self._subscriptions.get(event, [])
        before = len(subs)
        self._subscriptions[event] = [s for s in subs if s.handler != handler]
        return len(self._subscriptions[event]) < before

    def clear(self, event: str | None = None):
        """Remove all subscriptions. If event is None, clear everything."""
        if event is None:
            self._subscriptions.clear()
            self._wildcards.clear()
        else:
            self._subscriptions.pop(event, None)

    # ── Emit ──

    async def emit(
        self,
        event: str,
        data: dict[str, Any] | None = None,
        source: str = "",
    ) -> int:
        """Fire an event. Returns number of handlers called (not counting errors)."""
        evt = Event(
            name=event,
            data=data or {},
            source=source,
        )
        self._store_history(event, evt)

        subs = list(self._subscriptions.get(event, []))
        subs.extend(self._wildcards)

        called = 0
        for sub in subs:
            called += 1
            try:
                result = sub.handler(event, data or {})
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(
                    "Event handler failed for %s on %s",
                    getattr(sub.handler, "__name__", "?"),
                    event,
                    extra={"tag": "INFRA"},
                )
            if sub.once:
                self.off(event, sub.handler)
        return called

    def emit_sync(
        self,
        event: str,
        data: dict[str, Any] | None = None,
        source: str = "",
    ) -> int:
        """Synchronous emit — for use outside async contexts (e.g. constructors)."""
        evt = Event(
            name=event,
            data=data or {},
            source=source,
        )
        self._store_history(event, evt)

        subs = list(self._subscriptions.get(event, []))
        subs.extend(self._wildcards)

        called = 0
        for sub in subs:
            called += 1
            try:
                result = sub.handler(event, data or {})
                if asyncio.iscoroutine(result):
                    logger.warning(
                        "Ignored async handler %s on %s in sync emit",
                        getattr(sub.handler, "__name__", "?"),
                        event,
                        extra={"tag": "INFRA"},
                    )
            except Exception:
                logger.exception(
                    "Sync handler failed for %s on %s",
                    getattr(sub.handler, "__name__", "?"),
                    event,
                    extra={"tag": "INFRA"},
                )
            if sub.once:
                self.off(event, sub.handler)
        return called

    # ── History / replay ──

    def history(self, event: str | None = None) -> list[Event]:
        """Return event history, optionally filtered by event name."""
        if event is None:
            result: list[Event] = []
            for evts in self._history.values():
                result.extend(evts)
            result.sort(key=lambda e: e.timestamp)
            return result
        return list(self._history.get(event, []))

    def replay(
        self,
        event: str | None = None,
        handler: EventHandler | None = None,
    ) -> list[Event]:
        """Replay past events to a handler. If no handler, return matching events."""
        past = self.history(event)
        if handler:
            for evt in past:
                try:
                    result = handler(evt.name, evt.data)
                    if asyncio.iscoroutine(result):
                        import warnings
                        warnings.warn("async handler passed to sync replay()")
                except Exception:
                    logger.exception("Replay handler failed for %s", evt.name, extra={"tag": "INFRA"})
        return past

    def _store_history(self, event: str, evt: Event):
        h = self._history[event]
        h.append(evt)
        if len(h) > self._max_history:
            h.pop(0)

    @property
    def subscriber_count(self) -> int:
        c = len(self._wildcards)
        for subs in self._subscriptions.values():
            c += len(subs)
        return c


# ── Singleton ──

_default_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus


def set_event_bus(bus: EventBus):
    global _default_bus
    _default_bus = bus
