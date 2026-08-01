"""
Agent Run History — file-backed persistence for orchestration run records.

Each orchestration run stores the goal, per-task status, logs, final response,
and timestamps. A run is written as a single JSON file under
``data/agent_runs/`` so history survives restarts and is queryable.

Usage:
    from domains.agents.run_history import get_agent_run_store

    store = get_agent_run_store()
    run_id = store.start(goal="Research AI agents", context="")
    store.append_log(run_id, "Planning...")
    store.complete(run_id, response="...", tasks=[...])

    runs = store.list_runs(limit=20)
    detail = store.get(run_id)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("slo.agents.runs")

RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "agent_runs")


def _now_iso() -> str:
    """Return current UTC time as an ISO 8601 string."""
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


_id_counter = 0


def _new_run_id() -> str:
    """Return a unique, sortable run id."""
    global _id_counter
    _id_counter += 1
    return f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{_id_counter:06d}"


class AgentRunStore:
    """File-backed store for orchestration run history.

    Each run is persisted as ``{RUNS_DIR}/{run_id}.json``. Writes are guarded
    by a process-level lock so concurrent SSE streams do not corrupt files.
    """

    def __init__(self, directory: str = RUNS_DIR, max_runs: int = 200):
        self._dir = directory
        self._max_runs = max_runs
        self._lock = threading.RLock()
        os.makedirs(self._dir, exist_ok=True)

    # ── Path helpers ─────────────────────────────────────────────────────

    def _path(self, run_id: str) -> str:
        return os.path.join(self._dir, f"{run_id}.json")

    def _safe_id(self, run_id: str) -> bool:
        return bool(run_id) and run_id.replace("_", "").replace("-", "").isalnum()

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self, goal: str, context: str = "") -> str:
        """Record a new run in ``running`` state and return its id."""
        run_id = _new_run_id()
        record = {
            "id": run_id,
            "goal": goal,
            "context": context,
            "status": "running",
            "started_at": _now_iso(),
            "finished_at": None,
            "tasks": [],
            "completed_count": 0,
            "failed_count": 0,
            "response": "",
            "error": "",
            "logs": [f"[{_now_iso()}] Started: {goal}"],
        }
        self._save(run_id, record)
        self._prune()
        return run_id

    def append_log(self, run_id: str, message: str) -> None:
        """Append a log line to an existing run (no-op if run is gone)."""
        with self._lock:
            record = self._load(run_id)
            if record is None:
                return
            record.setdefault("logs", []).append(f"[{_now_iso()}] {message}")
            self._save(run_id, record)

    def set_tasks(self, run_id: str, tasks: List[Dict[str, Any]]) -> None:
        """Set the planned task list and refresh task counts."""
        with self._lock:
            record = self._load(run_id)
            if record is None:
                return
            record["tasks"] = list(tasks)
            record["completed_count"] = sum(1 for t in tasks if t.get("status") == "completed")
            record["failed_count"] = sum(1 for t in tasks if t.get("status") == "failed")
            self._save(run_id, record)

    def complete(self, run_id: str, response: str, tasks: Optional[List[Dict[str, Any]]] = None) -> None:
        """Mark a run as completed with its final response."""
        with self._lock:
            record = self._load(run_id)
            if record is None:
                return
            if tasks is not None:
                record["tasks"] = list(tasks)
            record["completed_count"] = sum(1 for t in record["tasks"] if t.get("status") == "completed")
            record["failed_count"] = sum(1 for t in record["tasks"] if t.get("status") == "failed")
            record["response"] = response
            record["status"] = "completed"
            record["finished_at"] = _now_iso()
            record.setdefault("logs", []).append(f"[{_now_iso()}] Completed")
            self._save(run_id, record)

    def fail(self, run_id: str, error: str) -> None:
        """Mark a run as failed with an error message."""
        with self._lock:
            record = self._load(run_id)
            if record is None:
                return
            record["error"] = error
            record["status"] = "failed"
            record["finished_at"] = _now_iso()
            record.setdefault("logs", []).append(f"[{_now_iso()}] Failed: {error}")
            self._save(run_id, record)

    # ── Queries ──────────────────────────────────────────────────────────

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Return a single run record, or None if it does not exist."""
        if not self._safe_id(run_id):
            return None
        return self._load(run_id)

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return run records sorted newest-first, truncated to ``limit``."""
        runs = []
        for fname in sorted(os.listdir(self._dir), reverse=True):
            if not fname.endswith(".json"):
                continue
            run_id = fname[:-5]
            record = self._load(run_id)
            if record is not None:
                runs.append(record)
            if len(runs) >= limit:
                break
        return runs

    def clear(self) -> int:
        """Delete all run records; returns the number removed."""
        removed = 0
        with self._lock:
            for fname in os.listdir(self._dir):
                if fname.endswith(".json"):
                    try:
                        os.remove(os.path.join(self._dir, fname))
                        removed += 1
                    except OSError:
                        pass
        return removed

    # ── Internals ────────────────────────────────────────────────────────

    def _save(self, run_id: str, record: Dict[str, Any]) -> None:
        path = self._path(run_id)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
        os.replace(tmp, path)

    def _load(self, run_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(run_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _prune(self) -> None:
        """Delete oldest run files beyond ``max_runs`` to bound disk usage."""
        with self._lock:
            files = sorted(
                (f for f in os.listdir(self._dir) if f.endswith(".json")),
                reverse=True,
            )
            for stale in files[self._max_runs:]:
                try:
                    os.remove(os.path.join(self._dir, stale))
                except OSError:
                    pass


_default_store: Optional[AgentRunStore] = None


def get_agent_run_store() -> AgentRunStore:
    """Return the process-wide AgentRunStore singleton."""
    global _default_store
    if _default_store is None:
        _default_store = AgentRunStore()
    return _default_store


def reset_agent_run_store() -> None:
    """Reset the singleton (used by tests)."""
    global _default_store
    _default_store = None
