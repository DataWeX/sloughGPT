"""Performance markers — checkpoint durations, percentiles, slowest ops."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile, no numpy needed."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100) * (len(ordered) - 1)
    low, high = int(rank), min(int(rank) + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


@dataclass
class MarkerCheckpoint:
    """One timed span."""

    name: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": self.timestamp,
        }


@dataclass
class MarkerSummary:
    """Stats for one marker name."""

    name: str
    count: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "avg_ms": round(self.avg_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "max_ms": round(self.max_ms, 2),
        }


@dataclass
class PerformanceReport:
    """Whole-run report."""

    total_ms: float
    summaries: list[MarkerSummary]
    slowest: list[MarkerCheckpoint]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_ms": round(self.total_ms, 2),
            "markers": [s.to_dict() for s in self.summaries],
            "slowest": [c.to_dict() for c in self.slowest],
        }


class PerformanceMarker:
    """Collect timings under named markers.

    Usage::

        perf = PerformanceMarker()
        with perf.measure("goto"):
            await arken.goto("/chat")
        perf.mark("click", 12.5)
        report = perf.report()
    """

    def __init__(self):
        self._start = time.monotonic()
        self._durations: dict[str, list[float]] = {}
        self._checkpoints: list[MarkerCheckpoint] = []

    def mark(self, name: str, duration_ms: float) -> MarkerCheckpoint:
        cp = MarkerCheckpoint(name=name, duration_ms=duration_ms)
        self._checkpoints.append(cp)
        self._durations.setdefault(name, []).append(duration_ms)
        return cp

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start = time.monotonic()
        try:
            yield
        finally:
            self.mark(name, (time.monotonic() - start) * 1000)

    @property
    def checkpoints(self) -> list[MarkerCheckpoint]:
        return list(self._checkpoints)

    def summary_for(self, name: str) -> MarkerSummary | None:
        values = self._durations.get(name, [])
        if not values:
            return None
        return MarkerSummary(
            name=name,
            count=len(values),
            avg_ms=sum(values) / len(values),
            p50_ms=_percentile(values, 50),
            p95_ms=_percentile(values, 95),
            max_ms=max(values),
        )

    def slowest(self, limit: int = 5) -> list[MarkerCheckpoint]:
        return sorted(self._checkpoints, key=lambda c: c.duration_ms, reverse=True)[:limit]

    def report(self) -> PerformanceReport:
        return PerformanceReport(
            total_ms=(time.monotonic() - self._start) * 1000,
            summaries=[s for name in self._durations if (s := self.summary_for(name)) is not None],
            slowest=self.slowest(),
        )

    def clear(self) -> None:
        self._durations.clear()
        self._checkpoints.clear()
        self._start = time.monotonic()
