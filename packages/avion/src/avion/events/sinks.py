"""Sinks — pluggable delivery targets for journaled events.

This is the step-5 bridge: the stdlib/system log is *one sink* registered
through the same ``EventSink`` API as everything else, not a hardcoded
adapter. Add a delivery target by implementing two members::

    class MySink:
        @property
        def name(self) -> str: return "mine"
        def emit(self, event) -> None: ...

Sinks are registered on an ``EventLogger``; they never write to the
journal themselves (the journal is the logger's job, first and alone).

Stdlib-only imports, enforced by ``tests/test_stdlib_only.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from avion.events.models import Event


@runtime_checkable
class EventSink(Protocol):
    """Delivery target for an already-journaled event."""

    @property
    def name(self) -> str: ...

    def emit(self, event: Event) -> None: ...


class CallbackSink:
    """Forward each event to a plain callable."""

    def __init__(self, callback: Callable[[Event], None], name: str = "callback"):
        self._callback = callback
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def emit(self, event: Event) -> None:
        self._callback(event)


class StdlibSink:
    """Bridge journaled events into the stdlib logging system.

    The payload rides on the LogRecord under a single ``avion_event`` key:
    stdlib raises ``KeyError`` when ``extra`` collides with a LogRecord
    attribute, so fields are never splatted in bare.

    Level mapping: no error and success → INFO, failed → WARNING,
    failed with an error message → ERROR.
    """

    def __init__(self, logger_name: str = "avion", level: int = logging.INFO):
        self._logger = logging.getLogger(logger_name)
        self._level = level
        self._name = f"stdlib:{logger_name}"

    @property
    def name(self) -> str:
        return self._name

    def emit(self, event: Event) -> None:
        if not event.success:
            level = logging.ERROR if event.error else logging.WARNING
        else:
            level = self._level
        # to_dict() already drops screenshot bytes (flag only) — never
        # re-attach raw data here or a LogRecord carries megabytes.
        self._logger.log(
            level,
            "%s %s",
            event.type.value,
            event.name,
            extra={"avion_event": event.to_dict()},
        )


class BusSink:
    """Publish events to an EventBus-like object (duck-typed: ``publish``).

    Duck-typed so the sink layer does not depend on the bus module — the
    coupling only runs sink -> protocol, never the other way.
    """

    def __init__(self, bus: object, name: str = "bus"):
        self._bus = bus
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def emit(self, event: Event) -> None:
        publish = getattr(self._bus, "publish", None)
        if publish is None:
            raise AttributeError(f"{type(self._bus).__name__} has no publish()")
        publish(event)
