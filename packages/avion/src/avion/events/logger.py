"""EventLogger — journal-first fan-out with per-sink isolation.

Write path, in order:

1. journal append + flush   (durability *before* delivery)
2. every registered sink, each in its own try/except

A failing sink can never block the journal, its siblings, or the caller:
failures are counted per sink, and after ``degrade_after`` consecutive
failures a sink is disabled until ``reset_sink(name)``. No sink is
required — a journal-only logger is valid and useful.

Stdlib-only imports, enforced by ``tests/test_stdlib_only.py``.
"""

from __future__ import annotations

from typing import Any

from avion.events.journal import EventJournal
from avion.events.models import Event
from avion.events.sinks import EventSink

DEFAULT_DEGRADE_AFTER = 5


class EventLogger:
    """Owns the journal (source of truth) and delivers to registered sinks.

    Usage::

        logger = EventLogger(EventJournal("avion_output/events.jsonl"))
        logger.register_sink(StdlibSink("avion"))
        logger.log(Event(EventType.CLICK, name="text='Start'"))
    """

    def __init__(
        self,
        journal: EventJournal | None = None,
        sinks: list[EventSink] | None = None,
        *,
        degrade_after: int = DEFAULT_DEGRADE_AFTER,
    ):
        self._journal = journal
        self._sinks: dict[str, EventSink] = {}
        self._degrade_after = max(1, degrade_after)
        self._failures: dict[str, int] = {}
        self._consecutive: dict[str, int] = {}
        self._disabled: set[str] = set()
        self._seq = 0
        self.journal_failures = 0
        for sink in sinks or []:
            self.register_sink(sink)

    # ── registration ───────────────────────────────────────────────────

    @property
    def journal(self) -> EventJournal | None:
        return self._journal

    @property
    def sinks(self) -> list[str]:
        """Registered sink names, in registration order."""
        return list(self._sinks)

    def register_sink(self, sink: EventSink) -> None:
        name = sink.name
        self._sinks[name] = sink
        self._disabled.discard(name)  # re-registering revives a sink
        self._consecutive[name] = 0

    def unregister_sink(self, name: str) -> bool:
        """Remove a sink. Returns False when it was not registered."""
        self._disabled.discard(name)
        self._consecutive.pop(name, None)
        return self._sinks.pop(name, None) is not None

    def reset_sink(self, name: str) -> bool:
        """Revive a degraded sink and clear its consecutive-failure count."""
        if name not in self._sinks:
            return False
        self._disabled.discard(name)
        self._consecutive[name] = 0
        return True

    # ── write path ─────────────────────────────────────────────────────

    def log(self, event: Event) -> Event:
        """Persist the event, then fan it out to sinks. Never raises for
        sink problems; a journal problem is counted and delivery still
        proceeds so the work is not silently lost everywhere at once."""
        if self._journal is not None:
            try:
                self._journal.append(event)
            except Exception:
                self.journal_failures += 1
        else:
            self._seq += 1
            event.seq = self._seq

        for name, sink in list(self._sinks.items()):
            if name in self._disabled:
                continue
            try:
                sink.emit(event)
                self._consecutive[name] = 0
            except Exception:
                self._failures[name] = self._failures.get(name, 0) + 1
                self._consecutive[name] = self._consecutive.get(name, 0) + 1
                if self._consecutive[name] >= self._degrade_after:
                    self._disabled.add(name)
        return event

    # ── views ──────────────────────────────────────────────────────────

    def events(self) -> list[Event]:
        """Every event in the journal (empty when journal-less)."""
        if self._journal is None:
            return []
        return self._journal.events()

    def sink_failures(self, name: str) -> int:
        """Total failures recorded against one sink."""
        return self._failures.get(name, 0)

    def sink_status(self) -> dict[str, dict[str, Any]]:
        """Per-sink health: cumulative failures + whether it is degraded."""
        return {
            name: {
                "failures": self._failures.get(name, 0),
                "disabled": name in self._disabled,
            }
            for name in self._sinks
        }

    def stats(self) -> dict[str, Any]:
        """Health snapshot of the whole logger."""
        return {
            "journal": str(self._journal.path) if self._journal else None,
            "journal_failures": self.journal_failures,
            "sinks": self.sink_status(),
        }

    def close(self) -> None:
        if self._journal is not None:
            self._journal.close()

    def __enter__(self) -> EventLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
