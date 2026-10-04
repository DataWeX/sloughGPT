"""Startup history tracking — stores recent startup times for performance monitoring.

Records persist to a MogDB collection (``data/startup_history_mogdb``), one
document per boot, following the feedback-controller store pattern. If mogdb
is unavailable the store degrades to in-memory only — observability must
never fail boot. The retired JSON-file store (SLO_STARTUP_HISTORY_PATH) is
migrated once and its original kept as a ``.bak``.

Usage (called by StagedLoader, not by hand):
    from infrastructure.startup_history import get_startup_history

    history = get_startup_history()
    history.start_startup()
    history.record_stage("critical", 5.1)
    history.record_hook("db_pool", 0.5)
    record = history.finish_startup(success=True)

    stats = history.get_stats()
    print(f"Average startup: {stats['avg_duration']}s")
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def legacy_history_path() -> Path:
    """Where the retired JSON-file store wrote records (migration source).

    ``SLO_STARTUP_HISTORY_PATH`` is read at migration time, not import time.
    """
    return Path(
        os.environ.get(
            "SLO_STARTUP_HISTORY_PATH", str(Path.home() / ".slogpt" / "startup_history.json")
        )
    )


def _import_mogdb():
    """Import mogdb, adding packages/mogdb/src to sys.path for this session.

    The server bootstrap does not put packages/mogdb/src on sys.path and the
    shared env does not install mogdb — the CLI db group uses this same
    self-serve path insert (apps/cli/src/groups/db.py).
    """
    from domain.shared import find_repo_root

    src = find_repo_root(Path(__file__).resolve()) / "packages" / "mogdb" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from mogdb import MogDB

    return MogDB


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

    Keeps the last 50 startup records in memory, durable across restarts via
    the MogDB store when available. Calculates averages, percentiles, and
    comparisons for monitoring dashboards.
    """

    def __init__(self, max_records: int = 50, db_path: Path | None = None) -> None:
        # RLock, not Lock: get_optimization_suggestions holds the lock while
        # calling get_stage_stats(), which acquires it again. A plain Lock
        # self-deadlocks here — and because health.startup_history is an async
        # handler, that freeze took down the whole uvicorn event loop.
        self._lock = threading.RLock()
        self._records: deque[StartupRecord] = deque(maxlen=max_records)
        self._max_records = max_records
        self._current_startup: StartupRecord | None = None
        self._current_start_time: float = 0.0
        self._db_path = db_path or self._default_db_path()
        self._store: Any = None  # MogDB collection; None = degraded in-memory mode
        self._open_store()
        self._migrate_legacy_json()
        self._load_from_store()

    @staticmethod
    def _default_db_path() -> Path:
        from domain.shared import find_repo_root

        return find_repo_root(Path(__file__).resolve()) / "data" / "startup_history_mogdb"

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

        # Record to Prometheus metrics. Observability must never fail the
        # caller, but the import is the one that exists — `get_metrics` did
        # not, so this block raised ImportError every single time (silently).
        try:
            from domain.infrastructure.metrics import get_metrics_collector

            metrics = get_metrics_collector()
            for stage, dur in record.stage_durations.items():
                metrics.record_startup_stage_duration(stage, dur)
            for hook, dur in record.hook_durations.items():
                metrics.record_startup_hook(hook, dur)
            if not record.success:
                metrics.increment_startup_failure_count()
        except Exception:
            logger.debug("startup history metrics replay failed", exc_info=True)

        self._persist_record(record)
        return record

    def _open_store(self) -> None:
        """Open the MogDB record store (``data/startup_history_mogdb``).

        Degraded mode: any failure — mogdb import, path, open — leaves
        ``self._store`` as None and history runs in-memory only, with one
        warning. Startup observability must never fail boot (the WebhookStore
        degraded-mode precedent).
        """
        try:
            MogDB = _import_mogdb()
            self._store = MogDB(str(self._db_path)).collection("startup_records")
            self._store.create_sorted_index("timestamp")
        except Exception:
            self._store = None
            logger.warning(
                "startup history degraded to in-memory only (MogDB unavailable at %s)",
                self._db_path,
                exc_info=True,
            )

    def _persist_record(self, record: StartupRecord) -> None:
        """Append a finished record to the store and prune to max_records."""
        if self._store is None:
            return
        try:
            self._store.insert_one(record.to_dict())
            excess = self._store.count() - self._max_records
            if excess > 0:
                oldest = self._store.find(sort=[("timestamp", 1)], limit=excess)
                self._store.delete_many({"_id": {"$in": [d["_id"] for d in oldest]}})
        except Exception:
            logger.warning("failed to persist startup record", exc_info=True)

    def _load_from_store(self) -> None:
        """Load the newest max_records from the store into memory."""
        if self._store is None:
            return
        try:
            docs = self._store.find(sort=[("timestamp", -1)], limit=self._max_records)
            for data in reversed(docs):
                self._records.append(
                    StartupRecord(
                        timestamp=data.get("timestamp", 0),
                        total_duration=data.get("total_duration", 0),
                        stage_durations=data.get("stage_durations", {}),
                        hook_durations=data.get("hook_durations", {}),
                        model_load_duration=data.get("model_load_duration", 0),
                        success=data.get("success", True),
                        error=data.get("error"),
                    )
                )
            if docs:
                logger.info("Loaded %d startup records from MogDB", len(docs))
        except Exception:
            logger.warning("failed to load startup history from MogDB", exc_info=True)

    def _migrate_legacy_json(self) -> None:
        """One-time import of the retired JSON-file history, kept as ``.bak``.

        The old store rewrote SLO_STARTUP_HISTORY_PATH on every finish. On the
        first boot with a healthy store: import its records (skipping
        timestamps the store already has), then rename the original to .bak —
        never delete user data. A failed rename just re-runs next boot; the
        dedupe keeps it idempotent.
        """
        if self._store is None:
            return
        legacy = legacy_history_path()
        if not legacy.exists():
            return
        try:
            with open(legacy) as f:
                records = json.load(f)
            if not isinstance(records, list):
                raise ValueError(f"unexpected legacy format: {type(records).__name__}")
            imported = 0
            for data in records:
                if self._store.find_one({"timestamp": data.get("timestamp", -1)}) is None:
                    self._store.insert_one(data)
                    imported += 1
            backup = legacy.parent / (legacy.name + ".bak")
            legacy.rename(backup)
            logger.info(
                "migrated %d legacy startup records from %s (original kept at %s)",
                imported,
                legacy,
                backup,
            )
        except Exception:
            logger.warning("legacy startup history migration failed", exc_info=True)

    def clear_history(self) -> None:
        """Clear all startup records (memory and store)."""
        with self._lock:
            self._records.clear()
        if self._store is not None:
            try:
                self._store.delete_many({})
            except Exception:
                logger.warning("failed to clear startup history store", exc_info=True)

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
                "p95_duration": round(
                    sorted_durations[int(n * 0.95)] if n > 1 else sorted_durations[0], 2
                ),
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
            return [r.to_dict() for r in self._records if r.total_duration > threshold_seconds]

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
                    alerts.append(
                        {
                            "type": "slow_startup",
                            "severity": "warning",
                            "message": f"Startup took {r.total_duration:.1f}s (threshold: {slow_threshold:.1f}s)",
                            "timestamp": r.timestamp,
                            "duration": r.total_duration,
                        }
                    )

            # Check for failed startups
            for r in self._records[-5:]:
                if not r.success:
                    alerts.append(
                        {
                            "type": "failed_startup",
                            "severity": "error",
                            "message": f"Startup failed: {r.error or 'unknown error'}",
                            "timestamp": r.timestamp,
                            "error": r.error,
                        }
                    )

            # Check for regression (current > 2x average)
            if self._records:
                last = self._records[-1]
                if last.total_duration > avg * 2:
                    alerts.append(
                        {
                            "type": "startup_regression",
                            "severity": "warning",
                            "message": f"Startup {last.total_duration:.1f}s is {last.total_duration / avg:.1f}x slower than average ({avg:.1f}s)",
                            "timestamp": last.timestamp,
                            "duration": last.total_duration,
                            "avg_duration": avg,
                        }
                    )

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
                    suggestions.append(
                        {
                            "type": "slow_stage",
                            "severity": "warning",
                            "stage": stage,
                            "message": f"Stage '{stage}' averages {stats['avg']}s. Consider optimizing hooks in this stage.",
                            "current_avg": stats["avg"],
                            "threshold": 30,
                        }
                    )

                # Check for high variance
                if stats["max"] > stats["avg"] * 2 and stats["count"] > 3:
                    suggestions.append(
                        {
                            "type": "high_variance",
                            "severity": "info",
                            "stage": stage,
                            "message": f"Stage '{stage}' has high variance (avg: {stats['avg']}s, max: {stats['max']}s). Check for non-deterministic behavior.",
                            "avg": stats["avg"],
                            "max": stats["max"],
                        }
                    )

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
                    suggestions.append(
                        {
                            "type": "slow_hook",
                            "severity": "warning",
                            "hook": hook,
                            "message": f"Hook '{hook}' averages {avg_duration:.1f}s. Consider parallelizing or optimizing.",
                            "current_avg": round(avg_duration, 1),
                            "threshold": 10,
                        }
                    )

            # Check for model load specifically
            model_durations = [
                r.hook_durations.get("model_load", 0)
                for r in self._records
                if "model_load" in r.hook_durations
            ]
            if model_durations:
                avg_model = sum(model_durations) / len(model_durations)
                if avg_model > 60:
                    suggestions.append(
                        {
                            "type": "slow_model_load",
                            "severity": "warning",
                            "hook": "model_load",
                            "message": f"Model load averages {avg_model:.0f}s. Consider using a smaller model, quantization, or model caching.",
                            "current_avg": round(avg_model, 1),
                            "threshold": 60,
                        }
                    )

            # Check for overall startup time
            durations = [r.total_duration for r in self._records]
            avg_total = sum(durations) / len(durations)
            if avg_total > 120:
                suggestions.append(
                    {
                        "type": "slow_overall",
                        "severity": "warning",
                        "message": f"Overall startup averages {avg_total:.0f}s. Consider disabling non-essential hooks.",
                        "current_avg": round(avg_total, 1),
                        "threshold": 120,
                    }
                )

            # Check for frequent failures
            failures = sum(1 for r in self._records if not r.success)
            if failures > len(self._records) * 0.2:  # More than 20% failures
                suggestions.append(
                    {
                        "type": "frequent_failures",
                        "severity": "error",
                        "message": f"{failures}/{len(self._records)} startups failed ({failures / len(self._records) * 100:.0f}%). Check system resources and configuration.",
                        "failure_count": failures,
                        "total_count": len(self._records),
                    }
                )

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
