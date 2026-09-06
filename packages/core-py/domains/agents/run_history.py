"""
Agent Run History — MogDB-backed persistence for orchestration run records.

Each orchestration run stores the goal, per-task status, logs, final response,
and timestamps. Runs are stored in a MogDB collection under
``data/mogdb/agent_runs`` so history survives restarts and is queryable.

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

import datetime
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("slo.agents.runs")

RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "agent_runs")

_db = None
_collection = None


def _now_iso() -> str:
    """Return current UTC time as an ISO 8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


_id_counter = 0


def _new_run_id() -> str:
    """Return a unique, sortable run id."""
    global _id_counter
    _id_counter += 1
    return f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{_id_counter:06d}"


def _get_collection(db_path: Optional[str] = None):
    global _db, _collection
    if _collection is not None:
        return _collection
    if db_path is None:
        from domains.shared import find_repo_root
        repo = find_repo_root(os.path.dirname(__file__))
        db_path = os.path.join(repo, "data", "mogdb", "agent_runs")
    from mogdb import MogDB
    _db = MogDB(db_path)
    _collection = _db.collection("runs")
    return _collection


def set_mogdb_path(db_path: str) -> None:
    """Override the default MogDB path (used by tests)."""
    global _db, _collection
    from mogdb import MogDB
    _db = MogDB(db_path)
    _collection = _db.collection("runs")


def reset_mogdb() -> None:
    """Reset the module-level MogDB singletons (used by tests)."""
    global _db, _collection
    _db = None
    _collection = None


class AgentRunStore:
    """MogDB-backed store for orchestration run history.

    Each run is stored as a document with ``_id == run_id``. Writes are guarded
    by a process-level lock so concurrent SSE streams do not corrupt state.
    """

    def __init__(self, db_path: Optional[str] = None, max_runs: int = 200):
        self._max_runs = max_runs
        self._lock = threading.RLock()
        self._col = _get_collection(db_path)

    # ── Path helpers ─────────────────────────────────────────────────────

    def _safe_id(self, run_id: str) -> bool:
        return bool(run_id) and run_id.replace("_", "").replace("-", "").isalnum()

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self, goal: str, context: str = "") -> str:
        """Record a new run in ``running`` state and return its id."""
        run_id = _new_run_id()
        record = {
            "_id": run_id,
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
        self._col.insert_one(record)
        self._prune()
        return run_id

    def append_log(self, run_id: str, message: str) -> None:
        """Append a log line to an existing run (no-op if run is gone)."""
        with self._lock:
            existing = self._col.find_one({"_id": run_id})
            if existing is None:
                return
            logs = existing.get("logs", [])
            logs.append(f"[{_now_iso()}] {message}")
            self._col.update_one({"_id": run_id}, {"$set": {"logs": logs}})

    def set_tasks(self, run_id: str, tasks: List[Dict[str, Any]]) -> None:
        """Set the planned task list and refresh task counts."""
        with self._lock:
            existing = self._col.find_one({"_id": run_id})
            if existing is None:
                return
            completed = sum(1 for t in tasks if t.get("status") == "completed")
            failed = sum(1 for t in tasks if t.get("status") == "failed")
            self._col.update_one(
                {"_id": run_id},
                {"$set": {
                    "tasks": list(tasks),
                    "completed_count": completed,
                    "failed_count": failed,
                }},
            )

    def complete(self, run_id: str, response: str, tasks: Optional[List[Dict[str, Any]]] = None) -> None:
        """Mark a run as completed with its final response."""
        with self._lock:
            existing = self._col.find_one({"_id": run_id})
            if existing is None:
                return
            run_tasks = list(tasks) if tasks is not None else existing.get("tasks", [])
            completed = sum(1 for t in run_tasks if t.get("status") == "completed")
            failed = sum(1 for t in run_tasks if t.get("status") == "failed")
            logs = existing.get("logs", [])
            logs.append(f"[{_now_iso()}] Completed")
            self._col.update_one(
                {"_id": run_id},
                {"$set": {
                    "tasks": run_tasks,
                    "completed_count": completed,
                    "failed_count": failed,
                    "response": response,
                    "status": "completed",
                    "finished_at": _now_iso(),
                    "logs": logs,
                }},
            )

    def fail(self, run_id: str, error: str) -> None:
        """Mark a run as failed with an error message."""
        with self._lock:
            existing = self._col.find_one({"_id": run_id})
            if existing is None:
                return
            logs = existing.get("logs", [])
            logs.append(f"[{_now_iso()}] Failed: {error}")
            self._col.update_one(
                {"_id": run_id},
                {"$set": {
                    "error": error,
                    "status": "failed",
                    "finished_at": _now_iso(),
                    "logs": logs,
                }},
            )

    # ── Queries ──────────────────────────────────────────────────────────

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Return a single run record, or None if it does not exist."""
        if not self._safe_id(run_id):
            return None
        doc = self._col.find_one({"_id": run_id})
        if doc is None:
            return None
        return dict(doc)

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return run records sorted newest-first, truncated to ``limit``."""
        docs = self._col.find({}, sort=[("_id", -1)], limit=limit)
        return [dict(d) for d in docs]

    def clear(self) -> int:
        """Delete all run records; returns the number removed."""
        with self._lock:
            all_docs = self._col.find({})
            removed = len(all_docs) if isinstance(all_docs, list) else sum(1 for _ in all_docs)
            self._col.delete_many({})
        return removed

    # ── Internals ────────────────────────────────────────────────────────

    def _prune(self) -> None:
        """Delete oldest run records beyond ``max_runs`` to bound disk usage."""
        with self._lock:
            all_docs = self._col.find({}, sort=[("_id", -1)])
            ids = [d["_id"] for d in all_docs]
            stale = ids[self._max_runs:]
            if stale:
                self._col.delete_many({"_id": {"$in": stale}})


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
