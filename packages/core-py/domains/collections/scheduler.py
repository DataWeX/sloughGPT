from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
from pathlib import Path

from .sources import Record, Source, FileSource
from .stores import Store, MemoryStore, FileStore
from .collector import Collector
from .validators import CollectorRunner


@dataclass
class JobConfig:
    name: str
    interval: float = 60.0
    enabled: bool = True
    max_runs: int | None = None
    timeout: float | None = None
    on_complete: Callable[[str, int], None] | None = None
    on_error: Callable[[str, Exception], None] | None = None


class JobScheduler:
    def __init__(self):
        self._jobs: dict[str, JobConfig] = {}
        self._collectors: dict[str, Collector] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._stats: dict[str, dict] = {}
        self._lock = threading.Lock()

    def add_job(self, config: JobConfig, collector: Collector) -> JobScheduler:
        with self._lock:
            self._jobs[config.name] = config
            self._collectors[config.name] = collector
            self._stop_events[config.name] = threading.Event()
            self._stats[config.name] = {
                "runs": 0, "total_collected": 0, "errors": 0,
                "last_run": None, "last_duration": None, "status": "idle"
            }
        return self

    def remove_job(self, name: str) -> bool:
        with self._lock:
            if name in self._jobs:
                self.stop_job(name)
                del self._jobs[name]
                del self._collectors[name]
                del self._stop_events[name]
                del self._stats[name]
                return True
            return False

    def start_job(self, name: str) -> bool:
        with self._lock:
            if name not in self._jobs or name in self._threads:
                return False
            config = self._jobs[name]
            if not config.enabled:
                return False

        stop_event = self._stop_events[name]

        def run_loop():
            self._stats[name]["status"] = "running"
            run_count = 0
            while not stop_event.is_set():
                start = time.monotonic()
                try:
                    count = self._collectors[name].collect()
                    duration = time.monotonic() - start
                    self._stats[name]["runs"] += 1
                    self._stats[name]["total_collected"] += count
                    self._stats[name]["last_run"] = time.time()
                    self._stats[name]["last_duration"] = duration
                    if config.on_complete:
                        config.on_complete(name, count)
                except Exception as e:
                    self._stats[name]["errors"] += 1
                    if config.on_error:
                        config.on_error(name, e)
                run_count += 1
                if config.max_runs and run_count >= config.max_runs:
                    break
                stop_event.wait(config.interval)
            self._stats[name]["status"] = "stopped"

        thread = threading.Thread(target=run_loop, daemon=True, name=f"job:{name}")
        with self._lock:
            self._threads[name] = thread
        thread.start()
        return True

    def start_all(self) -> int:
        count = 0
        for name in self._jobs:
            if self.start_job(name):
                count += 1
        return count

    def stop_job(self, name: str) -> bool:
        with self._lock:
            if name in self._stop_events:
                self._stop_events[name].set()
                thread = self._threads.get(name)
                if thread:
                    thread.join(timeout=5.0)
                    del self._threads[name]
                self._stats[name]["status"] = "stopped"
                return True
        return False

    def stop_all(self) -> int:
        count = 0
        for name in list(self._jobs.keys()):
            if self.stop_job(name):
                count += 1
        return count

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def job_stats(self, name: str) -> dict | None:
        with self._lock:
            return self._stats.get(name)

    def list_jobs(self) -> list[str]:
        with self._lock:
            return list(self._jobs.keys())

    def get_collector(self, name: str) -> Collector | None:
        return self._collectors.get(name)

    def is_running(self, name: str) -> bool:
        return name in self._threads and self._threads[name].is_alive()


class CollectorMonitor:
    def __init__(self, runner: CollectorRunner | None = None, scheduler: JobScheduler | None = None):
        self._runner = runner
        self._scheduler = scheduler
        self._health_checks: dict[str, Callable[[], bool]] = {}
        self._alerts: list[dict] = []

    def add_health_check(self, name: str, check_fn: Callable[[], bool]) -> CollectorMonitor:
        self._health_checks[name] = check_fn
        return self

    def check_health(self) -> dict[str, bool]:
        results = {}
        for name, check_fn in self._health_checks.items():
            try:
                results[name] = check_fn()
            except Exception:
                results[name] = False
        return results

    def get_overview(self) -> dict:
        overview = {"timestamp": time.time(), "healthy": True, "components": {}}

        if self._runner:
            runner_stats = self._runner.stats()
            total_collected = sum(s.get("total_collected", 0) for s in runner_stats.values())
            total_errors = sum(s.get("errors", 0) for s in runner_stats.values())
            overview["components"]["runner"] = {
                "collectors": len(runner_stats),
                "total_collected": total_collected,
                "total_errors": total_errors,
            }

        if self._scheduler:
            scheduler_stats = self._scheduler.stats()
            running_jobs = sum(1 for s in scheduler_stats.values() if s.get("status") == "running")
            total_runs = sum(s.get("runs", 0) for s in scheduler_stats.values())
            overview["components"]["scheduler"] = {
                "jobs": len(scheduler_stats),
                "running": running_jobs,
                "total_runs": total_runs,
            }

        health = self.check_health()
        overview["health"] = health
        overview["healthy"] = all(health.values()) if health else True

        return overview

    def check_alerts(self) -> list[dict]:
        overview = self.get_overview()
        alerts = []
        if not overview["healthy"]:
            alerts.append({"type": "health", "message": "System unhealthy", "severity": "critical"})
        for name, healthy in overview.get("health", {}).items():
            if not healthy:
                alerts.append({"type": "check", "name": name, "message": f"Check {name} failed", "severity": "warning"})
        self._alerts = alerts
        return alerts

    def format_report(self) -> str:
        overview = self.get_overview()
        lines = [f"Collection Monitor Report", f"{'=' * 40}"]
        lines.append(f"Healthy: {'Yes' if overview['healthy'] else 'NO'}")
        lines.append("")
        for component, stats in overview.get("components", {}).items():
            lines.append(f"[{component.upper()}]")
            for k, v in stats.items():
                lines.append(f"  {k}: {v}")
            lines.append("")
        if overview.get("health"):
            lines.append("[HEALTH CHECKS]")
            for name, ok in overview["health"].items():
                lines.append(f"  {name}: {'OK' if ok else 'FAIL'}")
        return "\n".join(lines)


class CollectorExporter:
    def __init__(self, store: Store | None = None):
        self._store = store

    def set_store(self, store: Store) -> CollectorExporter:
        self._store = store
        return self

    def to_jsonl(self, path: str) -> int:
        if not self._store:
            return 0
        count = 0
        with open(path, "w") as f:
            for record in self._store.read_all():
                f.write(json.dumps(record.to_dict()) + "\n")
                count += 1
        return count

    def to_json(self, path: str) -> int:
        if not self._store:
            return 0
        records = list(self._store.read_all())
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in records], f, indent=2)
        return len(records)

    def to_text(self, path: str) -> int:
        if not self._store:
            return 0
        count = 0
        with open(path, "w") as f:
            for record in self._store.read_all():
                f.write(record.content + "\n")
                count += 1
        return count

    def to_memory(self) -> MemoryStore:
        if not self._store:
            return MemoryStore()
        store = MemoryStore()
        for record in self._store.read_all():
            store.write(record)
        return store

    def to_dicts(self) -> list[dict]:
        if not self._store:
            return []
        return [r.to_dict() for r in self._store.read_all()]

    def summary(self) -> dict:
        if not self._store:
            return {"count": 0, "total_bytes": 0, "sources": {}}
        total_bytes = 0
        sources = {}
        count = 0
        for record in self._store.read_all():
            count += 1
            total_bytes += len(record.content)
            src = record.metadata.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        return {"count": count, "total_bytes": total_bytes, "sources": sources}
