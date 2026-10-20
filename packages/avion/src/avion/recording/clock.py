"""Recording clock — ONE monotonic time source for a recording.

WHY ONE CLOCK: the frame sampler and the event stream must be alignable
when the viewer replays them. Both sides stamp ``now_ns()``
(``time.monotonic_ns()`` since clock construction), so "the frame for this
event" is a subtraction on one axis, not a guess across two.

DESYNC RISK: ``Event.timestamp`` stays WALL clock (``time.time()``) for
humans — wall time can step (NTP correction, leap, suspend) while
monotonic time never does, so the two series can drift apart mid-recording.
Alignment code must use ``_t_ns`` (this clock) and treat ``timestamp`` as
display only; never subtract wall timestamps across a recording.

Stdlib-only on purpose — same rule as the events layer
(tests/test_stdlib_only.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class RecordingClock:
    """Shared monotonic source; one instance per recording session."""

    monotonic: Callable[[], int] = time.monotonic_ns
    wall: Callable[[], float] = time.time
    origin_ns: int = 0
    started_wall: float = 0.0
    _origin_set: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self.arm()

    def arm(self) -> None:
        """Fix the origin (idempotent — re-arming would rewind stamps)."""
        if not self._origin_set:
            self.origin_ns = self.monotonic()
            self.started_wall = self.wall()
            self._origin_set = True

    def now_ns(self) -> int:
        """Monotonic nanoseconds since the clock was armed (never decreases)."""
        return self.monotonic() - self.origin_ns

    def elapsed_wall(self) -> float:
        """Wall-clock seconds since arm (display only — see desync risk)."""
        return self.wall() - self.started_wall

    def note(self) -> str:
        """Manifest blurb describing the clock contract."""
        return (
            "frames and events share one time.monotonic_ns() origin; "
            "Event.timestamp stays wall-clock for humans and may step "
            "(NTP/suspend) — align on _t_ns, never on timestamp"
        )
