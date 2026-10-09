"""Events package — record, broadcast, and replay session events.

The journal (``EventJournal``) is the source of truth; ``EventRecorder``
and ``StructuredLogger`` are views over the ``EventLogger`` that owns it.
Delivery is pluggable through the ``EventSink`` API — ``StdlibSink`` is
just the stdlib/system-log bridge, one sink among many.
"""

from avion.events.bus import EventBus, Subscription
from avion.events.journal import EventJournal
from avion.events.logger import EventLogger
from avion.events.models import Event, EventType
from avion.events.recorder import EventRecorder
from avion.events.replay import EventReplay, ReplayConfig, ReplayResult, ReplayStep
from avion.events.sinks import BusSink, CallbackSink, EventSink, StdlibSink

__all__ = [
    "BusSink",
    "CallbackSink",
    "Event",
    "EventBus",
    "EventJournal",
    "EventLogger",
    "EventRecorder",
    "EventReplay",
    "EventSink",
    "EventType",
    "ReplayConfig",
    "ReplayResult",
    "ReplayStep",
    "StdlibSink",
    "Subscription",
]
