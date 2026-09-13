"""Startup history tracking — stores recent startup times for performance monitoring.

Usage:
    from infrastructure.startup_history import get_startup_history

    history = get_startup_history()
    history.record_startup(
        total_duration=45.2,
        stage_durations={"critical": 5.1, "ready": 40.1},
        hook_durations={"db_pool": 0.5, "model_load": 40.0},
    )

    stats = history.get_stats()
    print(f"Average startup: {stats['avg_duration']}s")
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StartupRecord:
    """A single startup record."""

    timestamp: float
    total_duration: float
    stage_durations: dict[str, float] = field(default_factory=dict)
    hook_durations: dict[str, float] = field(default_factory=dict)
    model_load_duration: float = 0.0
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_duration": round(self.total_duration, 2),
            "stage_durations": {k: round(v, 2) for k, v in self.stage_durations.items()},
            "hook_durations": {k: round(v, 2) for k, v in self.hook_durations.items()},
            "model_load_duration": round(self.model_load_duration, 2),
            "success": self.success,
            "error": self.error,
        }


class StartupHistory:
    """Stores recent startup records for performance monitoring.

    Keeps the last 50 startup records in memory. Calculates averages,
    percentiles, and comparisons for monitoring dashboards.
    """

    def __init__(self, max_records: int = 50) -> None:
        self._lock = threading.Lock()
        self._records: deque[StartupRecord] = deque(maxlen=max_records)
        self._current_startup: StartupRecord | None = None
        self._current_start_time: float = 0.0

    def start_startup(self) -> None:
        """Mark the beginning of a startup."""
        with self._lock:
            self._current_start_time = time.monotonic()
            self._current_startup = StartupRecord(timestamp=time.time(), total_duration=0.0)

    def record_stage(self, stage: str, duration: float) -> None:
        """Record a stage duration."""
        with self._lock:
            if self._current_startup:
                self._current_startup.stage_durations[stage] = duration

    def record_hook(self, hook: str, duration: float) -> None:
        """Record a hook duration."""
        with self._lock:
            if self._current_startup:
                self._current_startup.hook_durations[hook] = duration

    def record_model_load(self, duration: float) -> None:
        """Record model load duration."""
        with self._lock:
            if self._current_startup:
                self._current_startup.model_load_duration = duration

    def finish_startup(self, success: bool = True, error: str | None = None) -> StartupRecord:
        """Finish the current startup and store the record."""
        with self._lock:
            if self._current_startup is None:
                return StartupRecord(timestamp=time.time(), total_duration=0.0)

            self._current_startup.total_duration = time.monotonic() - self._current_start_time
            self._current_startup.success = success
            self._current_startup.error = error

            record = self._current_startup
            self._records.append(record)
            self._current_startup = None
            return record

    def get_records(self, limit: int = 10) -> list[dict]:
        """Get recent startup records."""
        with self._lock:
            return [r.to_dict() for r in list(self._records)[-limit:]]

    def get_stats(self) -> dict[str, Any]:
        """Calculate startup performance statistics."""
        with self._lock:
            if not self._records:
                return {
                    "count": 0,
                    "avg_duration": 0,
                    "min_duration": 0,
                    "max_duration": 0,
                    "p50_duration": 0,
                    "p95_duration": 0,
                    "success_rate": 0,
                }

            durations = [r.total_duration for r in self._records]
            successes = sum(1 for r in self._records if r.success)

            sorted_durations = sorted(durations)
            n = len(sorted_durations)

            return {
                "count": n,
                "avg_duration": round(sum(durations) / n, 2),
                "min_duration": round(min(durations), 2),
                "max_duration": round(max(durations), 2),
                "p50_duration": round(sorted_durations[n // 2], 2),
                "p95_duration": round(sorted_durations[int(n * 0.95)] if n > 1 else sorted_durations[0], 2),
                "success_rate": round(successes / n, 2),
            }

    def get_stage_stats(self) -> dict[str, dict[str, float]]:
        """Calculate per-stage statistics."""
        with self._lock:
            if not self._records:
                return {}

            stage_totals: dict[str, list[float]] = {}
            for record in self._records:
                for stage, duration in record.stage_durations.items():
                    if stage not in stage_totals:
                        stage_totals[stage] = []
                    stage_totals[stage].append(duration)

            result = {}
            for stage, durations in stage_totals.items():
                sorted_d = sorted(durations)
                n = len(sorted_d)
                result[stage] = {
                    "avg": round(sum(durations) / n, 2),
                    "min": round(min(durations), 2),
                    "max": round(max(durations), 2),
                    "p50": round(sorted_d[n // 2], 2),
                    "count": n,
                }

            return result

    def get_slow_startups(self, threshold_seconds: float = 60.0) -> list[dict]:
        """Get startups that exceeded the threshold."""
        with self._lock:
            return [
                r.to_dict()
                for r in self._records
                if r.total_duration > threshold_seconds
            ]

    def get_alerts(self) -> list[dict[str, Any]]:
        """Check for startup performance alerts.

        Returns alerts for:
        - Slow startups (exceeding p95 by 50%)
        - Failed startups
        - Startup time regression (current > 2x average)
        """
        alerts = []
        with self._lock:
            if len(self._records) < 3:
                return alerts

            durations = [r.total_duration for r in self._records]
            sorted_durations = sorted(durations)
            n = len(sorted_durations)
            avg = sum(durations) / n
            p95 = sorted_durations[int(n * 0.95)] if n > 1 else sorted_durations[0]

            # Check for slow startups
            slow_threshold = p95 * 1.5
            for r in self._records[-5:]:  # Check last 5
                if r.total_duration > slow_threshold:
                    alerts.append({
                        "type": "slow_startup",
                        "severity": "warning",
                        "message": f"Startup took {r.total_duration:.1f}s (threshold: {slow_threshold:.1f}s)",
                        "timestamp": r.timestamp,
                        "duration": r.total_duration,
                    })

            # Check for failed startups
            for r in self._records[-5:]:
                if not r.success:
                    alerts.append({
                        "type": "failed_startup",
                        "severity": "error",
                        "message": f"Startup failed: {r.error or 'unknown error'}",
                        "timestamp": r.timestamp,
                        "error": r.error,
                    })

            # Check for regression (current > 2x average)
            if self._records:
                last = self._records[-1]
                if last.total_duration > avg * 2:
                    alerts.append({
                        "type": "startup_regression",
                        "severity": "warning",
                        "message": f"Startup {last.total_duration:.1f}s is {last.total_duration/avg:.1f}x slower than average ({avg:.1f}s)",
                        "timestamp": last.timestamp,
                        "duration": last.total_duration,
                        "avg_duration": avg,
                    })

        return alerts


# ── Singleton ───────────────────────────────────────────────────────
_history: StartupHistory | None = None
_history_lock = threading.Lock()


def get_startup_history() -> StartupHistory:
    """Return the global startup history instance."""
    global _history
    if _history is None:
        with _history_lock:
            if _history is None:
                _history = StartupHistory()
    return _history
