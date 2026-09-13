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

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default persistence path
DEFAULT_HISTORY_PATH = Path(os.environ.get(
    "SLO_STARTUP_HISTORY_PATH",
    str(Path.home() / ".slogpt" / "startup_history.json")
))


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

    def __init__(self, max_records: int = 50, persist_path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._records: deque[StartupRecord] = deque(maxlen=max_records)
        self._current_startup: StartupRecord | None = None
        self._current_start_time: float = 0.0
        self._persist_path = persist_path or DEFAULT_HISTORY_PATH
        self._load_from_disk()

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

        # Record to Prometheus metrics
        try:
            from domain.infrastructure.metrics import get_metrics
            metrics = get_metrics()
            for stage, dur in record.stage_durations.items():
                metrics.record_startup_stage_duration(stage, dur)
            for hook, dur in record.hook_durations.items():
                metrics.record_startup_hook(hook, dur)
            if not record.success:
                metrics.increment_startup_failure_count()
        except Exception:
            pass

        self._save_to_disk()
        return record

    def _save_to_disk(self) -> None:
        """Persist startup records to disk."""
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            records = [r.to_dict() for r in self._records]
            with open(self._persist_path, "w") as f:
                json.dump(records, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save startup history: %s", e)

    def _load_from_disk(self) -> None:
        """Load startup records from disk."""
        try:
            if self._persist_path.exists():
                with open(self._persist_path) as f:
                    records = json.load(f)
                for record_data in records:
                    record = StartupRecord(
                        timestamp=record_data.get("timestamp", 0),
                        total_duration=record_data.get("total_duration", 0),
                        stage_durations=record_data.get("stage_durations", {}),
                        hook_durations=record_data.get("hook_durations", {}),
                        model_load_duration=record_data.get("model_load_duration", 0),
                        success=record_data.get("success", True),
                        error=record_data.get("error"),
                    )
                    self._records.append(record)
                logger.info("Loaded %d startup records from disk", len(self._records))
        except Exception as e:
            logger.warning("Failed to load startup history: %s", e)

    def clear_history(self) -> None:
        """Clear all startup records."""
        with self._lock:
            self._records.clear()
        self._save_to_disk()

    def export_history(self) -> list[dict]:
        """Export all startup records."""
        with self._lock:
            return [r.to_dict() for r in self._records]

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

    def compare_runs(self, run_a_index: int, run_b_index: int) -> dict[str, Any] | None:
        """Compare two startup runs.

        Args:
            run_a_index: Index of first run (0 = oldest)
            run_b_index: Index of second run

        Returns:
            Comparison dict or None if indices invalid.
        """
        with self._lock:
            if run_a_index < 0 or run_a_index >= len(self._records):
                return None
            if run_b_index < 0 or run_b_index >= len(self._records):
                return None

            a = self._records[run_a_index]
            b = self._records[run_b_index]

            # Compare stages
            stage_comparison = {}
            all_stages = set(a.stage_durations.keys()) | set(b.stage_durations.keys())
            for stage in all_stages:
                a_dur = a.stage_durations.get(stage, 0)
                b_dur = b.stage_durations.get(stage, 0)
                diff = b_dur - a_dur
                pct = (diff / a_dur * 100) if a_dur > 0 else 0
                stage_comparison[stage] = {
                    "run_a": round(a_dur, 2),
                    "run_b": round(b_dur, 2),
                    "diff": round(diff, 2),
                    "diff_percent": round(pct, 1),
                }

            # Compare hooks
            hook_comparison = {}
            all_hooks = set(a.hook_durations.keys()) | set(b.hook_durations.keys())
            for hook in all_hooks:
                a_dur = a.hook_durations.get(hook, 0)
                b_dur = b.hook_durations.get(hook, 0)
                diff = b_dur - a_dur
                pct = (diff / a_dur * 100) if a_dur > 0 else 0
                hook_comparison[hook] = {
                    "run_a": round(a_dur, 2),
                    "run_b": round(b_dur, 2),
                    "diff": round(diff, 2),
                    "diff_percent": round(pct, 1),
                }

            total_diff = b.total_duration - a.total_duration
            total_pct = (total_diff / a.total_duration * 100) if a.total_duration > 0 else 0

            return {
                "run_a": a.to_dict(),
                "run_b": b.to_dict(),
                "total_diff": round(total_diff, 2),
                "total_diff_percent": round(total_pct, 1),
                "stage_comparison": stage_comparison,
                "hook_comparison": hook_comparison,
            }

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

    def get_optimization_suggestions(self) -> list[dict[str, Any]]:
        """Analyze startup history and provide optimization suggestions.

        Returns actionable suggestions to improve startup performance.
        """
        suggestions = []
        with self._lock:
            if len(self._records) < 2:
                return suggestions

            # Analyze stage performance
            stage_stats = self.get_stage_stats()
            for stage, stats in stage_stats.items():
                if stats["avg"] > 30:
                    suggestions.append({
                        "type": "slow_stage",
                        "severity": "warning",
                        "stage": stage,
                        "message": f"Stage '{stage}' averages {stats['avg']}s. Consider optimizing hooks in this stage.",
                        "current_avg": stats["avg"],
                        "threshold": 30,
                    })

                # Check for high variance
                if stats["max"] > stats["avg"] * 2 and stats["count"] > 3:
                    suggestions.append({
                        "type": "high_variance",
                        "severity": "info",
                        "stage": stage,
                        "message": f"Stage '{stage}' has high variance (avg: {stats['avg']}s, max: {stats['max']}s). Check for non-deterministic behavior.",
                        "avg": stats["avg"],
                        "max": stats["max"],
                    })

            # Analyze hook performance
            hook_stats: dict[str, list[float]] = {}
            for record in self._records:
                for hook, duration in record.hook_durations.items():
                    if hook not in hook_stats:
                        hook_stats[hook] = []
                    hook_stats[hook].append(duration)

            for hook, durations in hook_stats.items():
                avg_duration = sum(durations) / len(durations)
                if avg_duration > 10:
                    suggestions.append({
                        "type": "slow_hook",
                        "severity": "warning",
                        "hook": hook,
                        "message": f"Hook '{hook}' averages {avg_duration:.1f}s. Consider parallelizing or optimizing.",
                        "current_avg": round(avg_duration, 1),
                        "threshold": 10,
                    })

            # Check for model load specifically
            model_durations = [
                r.hook_durations.get("model_load", 0)
                for r in self._records
                if "model_load" in r.hook_durations
            ]
            if model_durations:
                avg_model = sum(model_durations) / len(model_durations)
                if avg_model > 60:
                    suggestions.append({
                        "type": "slow_model_load",
                        "severity": "warning",
                        "hook": "model_load",
                        "message": f"Model load averages {avg_model:.0f}s. Consider using a smaller model, quantization, or model caching.",
                        "current_avg": round(avg_model, 1),
                        "threshold": 60,
                    })

            # Check for overall startup time
            durations = [r.total_duration for r in self._records]
            avg_total = sum(durations) / len(durations)
            if avg_total > 120:
                suggestions.append({
                    "type": "slow_overall",
                    "severity": "warning",
                    "message": f"Overall startup averages {avg_total:.0f}s. Consider disabling non-essential hooks.",
                    "current_avg": round(avg_total, 1),
                    "threshold": 120,
                })

            # Check for frequent failures
            failures = sum(1 for r in self._records if not r.success)
            if failures > len(self._records) * 0.2:  # More than 20% failures
                suggestions.append({
                    "type": "frequent_failures",
                    "severity": "error",
                    "message": f"{failures}/{len(self._records)} startups failed ({failures/len(self._records)*100:.0f}%). Check system resources and configuration.",
                    "failure_count": failures,
                    "total_count": len(self._records),
                })

        return suggestions


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
