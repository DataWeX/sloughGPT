"""Time-travel debugging — snapshot page states, step back and forth."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PageSnapshot:
    """One captured page state."""

    seq: int
    url: str = ""
    title: str = ""
    text_excerpt: str = ""
    screenshot_hash: str = ""
    timestamp: float = field(default_factory=time.time)

    def diff(self, other: PageSnapshot) -> dict[str, dict[str, str]]:
        """Fields that changed between this snapshot and another."""
        changes: dict[str, dict[str, str]] = {}
        for key in ("url", "title", "text_excerpt", "screenshot_hash"):
            before, after = getattr(self, key), getattr(other, key)
            if before != after:
                changes[key] = {"before": before, "after": after}
        return changes

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "url": self.url,
            "title": self.title,
            "text_excerpt": self.text_excerpt,
            "screenshot_hash": self.screenshot_hash,
            "timestamp": self.timestamp,
        }


class Timeline:
    """Ordered snapshots with a cursor for stepping."""

    def __init__(self):
        self._snapshots: list[PageSnapshot] = []
        self._cursor = -1

    def __len__(self) -> int:
        return len(self._snapshots)

    def append(self, snapshot: PageSnapshot) -> PageSnapshot:
        self._snapshots.append(snapshot)
        self._cursor = len(self._snapshots) - 1
        return snapshot

    @property
    def current(self) -> PageSnapshot | None:
        if 0 <= self._cursor < len(self._snapshots):
            return self._snapshots[self._cursor]
        return None

    def get(self, index: int) -> PageSnapshot | None:
        if 0 <= index < len(self._snapshots):
            return self._snapshots[index]
        return None

    def back(self) -> PageSnapshot | None:
        """Step one snapshot back. None when already at the start."""
        if self._cursor > 0:
            self._cursor -= 1
            return self.current
        return None

    def forward(self) -> PageSnapshot | None:
        """Step one snapshot forward. None when already at the end."""
        if self._cursor < len(self._snapshots) - 1:
            self._cursor += 1
            return self.current
        return None

    def diff_between(self, a: int, b: int) -> dict[str, dict[str, str]]:
        first, second = self.get(a), self.get(b)
        if first is None or second is None:
            raise IndexError(f"snapshot {a} or {b} out of range")
        return first.diff(second)


class TimeTravel:
    """Captures snapshots and walks the timeline.

    Usage::

        tt = TimeTravel()
        tt.capture(url="/chat", title="Chat", text_excerpt="hi")
        tt.capture(url="/souls", title="Souls", text_excerpt="pick")
        tt.back()       # back at /chat
        tt.diff_between(0, 1)  # what changed
    """

    def __init__(self):
        self.timeline = Timeline()
        self._seq = 0

    @property
    def current(self) -> PageSnapshot | None:
        return self.timeline.current

    def capture(
        self,
        url: str = "",
        title: str = "",
        text_excerpt: str = "",
        screenshot: bytes | None = None,
    ) -> PageSnapshot:
        from avion.vision.detector import ImageAnalyzer

        self._seq += 1
        snap = PageSnapshot(
            seq=self._seq,
            url=url,
            title=title,
            text_excerpt=text_excerpt[:500],
            screenshot_hash=ImageAnalyzer().compute_hash(screenshot) if screenshot else "",
        )
        return self.timeline.append(snap)

    def back(self) -> PageSnapshot | None:
        return self.timeline.back()

    def forward(self) -> PageSnapshot | None:
        return self.timeline.forward()

    def diff_between(self, a: int, b: int) -> dict[str, dict[str, str]]:
        return self.timeline.diff_between(a, b)

    def clear(self) -> None:
        self.timeline = Timeline()
        self._seq = 0
