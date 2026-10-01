"""Event bus — pub/sub over recorded events.

Patterns match EventType values: exact ("click"), wildcard ("*"),
or prefix ("task_*"). Higher priority subscribers run first.
Interceptors can rewrite or drop events before delivery.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

from avion.events.models import Event


@dataclass
class Subscription:
    """One subscriber registration."""

    id: int
    pattern: str
    callback: Callable[[Event], None] = field(repr=False)
    priority: int = 0

    def matches(self, event: Event) -> bool:
        if self.pattern == "*":
            return True
        value = event.type.value
        if self.pattern.endswith("*"):
            return value.startswith(self.pattern[:-1])
        return value == self.pattern


class EventBus:
    """Delivers events to matching subscribers.

    Usage::

        bus = EventBus()
        bus.subscribe("click", lambda e: print(e.name))
        bus.add_interceptor(lambda e: None if e.name == "noise" else e)
        bus.publish(event)
    """

    def __init__(self):
        self._subs: dict[int, Subscription] = {}
        self._ids = itertools.count(1)
        self._interceptors: list[Callable[[Event], Event | None]] = []

    def subscribe(
        self,
        pattern: str,
        callback: Callable[[Event], None],
        priority: int = 0,
    ) -> Subscription:
        sub = Subscription(
            id=next(self._ids), pattern=pattern, callback=callback, priority=priority
        )
        self._subs[sub.id] = sub
        return sub

    def unsubscribe(self, sub_id: int) -> bool:
        return self._subs.pop(sub_id, None) is not None

    def add_interceptor(self, interceptor: Callable[[Event], Event | None]) -> None:
        """Interceptors run first; returning None drops the event."""
        self._interceptors.append(interceptor)

    def publish(self, event: Event) -> int:
        """Deliver to matching subscribers by priority. Returns count."""
        for interceptor in self._interceptors:
            try:
                event = interceptor(event)
            except Exception:
                continue
            if event is None:
                return 0
        matched = [s for s in self._subs.values() if s.matches(event)]
        matched.sort(key=lambda s: s.priority, reverse=True)
        delivered = 0
        for sub in matched:
            try:
                sub.callback(event)
                delivered += 1
            except Exception:
                continue
        return delivered

    @property
    def subscriptions(self) -> list[Subscription]:
        return list(self._subs.values())
