"""Persistent Training Job Store

Stores training jobs in MogDB (the project's embedded document database)
for crash recovery. Jobs persist across server restarts.
"""

from __future__ import annotations

import builtins
import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Overrides the default job-store location. Tests set this to a temp path so a
# test run can never INSERT rows into the repo's data/training_jobs.db (a real
# regression: a phantom row showed up as a "Recoverable Job" in the UI).
JOB_STORE_ENV_VAR = "SLO_TRAINING_JOBS_DB"

from mogdb import MogDB

from domain.shared import find_repo_root, parse_iso, utc_now_iso

logger = logging.getLogger("slo.job_store")


def _heartbeat_instant(value: object) -> datetime | None:
    """Absolute instant for a stored timestamp, whatever era wrote it.

    Current rows are UTC ``...Z`` strings from :func:`utc_now_iso`; legacy rows
    are naive *local* strings (``datetime.now()`` / ``time.localtime()``).
    Naive input is therefore interpreted as local time so mixed-era rows
    compare on the same axis, and malformed input returns ``None``.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return parse_iso(text)
    return parsed if parsed.tzinfo is not None else parsed.astimezone()


class JobStore:
    """
    MogDB-backed persistent job store.

    Features:
    - Persists across server restarts
    - Tracks job state, progress, checkpoints
    - Detects crashed/interrupted jobs
    - Supports recovery/resume

    The ``db_path`` argument is a directory in which MogDB keeps its
    collection journals (``jobs`` and ``job_events``).
    """

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = os.environ.get(JOB_STORE_ENV_VAR) or str(
                find_repo_root(Path(__file__).resolve()) / "data" / "training_jobs.db"
            )
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = None
        self._jobs = None
        self._events = None
        try:
            self._db = MogDB(str(self.db_path))
            self._jobs = self._db.collection("jobs")
            self._events = self._db.collection("job_events")
        except Exception:
            logger.warning(
                "JobStore: failed to open MogDB at %s, operating in degraded mode", self.db_path
            )

    @property
    def is_available(self) -> bool:
        return self._db is not None and self._jobs is not None

    @staticmethod
    def _new_job_doc(
        job_id: str,
        name: str,
        config: dict[str, Any],
        dataset: str,
        now: str,
        user_id: str = "",
        workspace_id: str = "",
    ) -> dict[str, Any]:
        """Build the full stored document for a new job."""
        return {
            "_id": job_id,
            "id": job_id,
            "name": name,
            "status": "pending",
            "dataset": dataset,
            "data_path": None,
            "config": config,
            "progress": 0.0,
            "current_epoch": 0,
            "total_epochs": 0,
            "global_step": 0,
            "loss": None,
            "train_loss": None,
            "eval_loss": None,
            "checkpoint_path": None,
            "checkpoint_dir": None,
            "error": None,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "created_at": now,
            "started_at": None,
            "updated_at": now,
            "completed_at": None,
            "last_heartbeat": now,
            "crashed": 0,
        }

    @staticmethod
    def _doc_to_job(doc: dict[str, Any]) -> dict[str, Any]:
        """Convert a stored MogDB document to the job dict returned to callers."""
        return {k: v for k, v in doc.items() if k not in ("_id", "_created", "_updated")}

    @staticmethod
    def store_row_to_job(row: dict[str, Any]) -> dict[str, Any]:
        """Map a durable store row to the in-memory job dict shape.

        Store columns are ``checkpoint_path`` / ``total_epochs`` / nested
        ``config.model``; the API and UI read ``checkpoint`` / ``epochs`` /
        top-level ``model``. Without this, a job that left the live dict
        (auto-purge after 1h, or a restart-seeded row never re-hydrated)
        serializes with missing checkpoint/model/epochs — Resume and history
        detail break for exactly the jobs that need them.

        Side effects:
            - None; returns a shallow-copied dict with API field aliases filled.
        """
        out = JobStore._doc_to_job(row)
        config = out.get("config") if isinstance(out.get("config"), dict) else {}
        if not out.get("checkpoint"):
            out["checkpoint"] = out.get("checkpoint_path") or ""
        if out.get("epochs") is None:
            out["epochs"] = out.get("total_epochs")
        if not out.get("model"):
            out["model"] = (config or {}).get("model") or "sloughgpt"
        if not out.get("method") and config:
            out["method"] = config.get("method")
        if out.get("checkpoint_path") is None and out.get("checkpoint"):
            out["checkpoint_path"] = out["checkpoint"]
        return out

    def create(
        self,
        job_id: str,
        name: str,
        config: dict[str, Any],
        dataset: str = "",
        user_id: str = "",
        workspace_id: str = "",
    ) -> dict:
        """Create a new job."""
        if not self.is_available:
            return {"id": job_id, "status": "error", "error": "Job store unavailable"}
        now = utc_now_iso()
        with self._lock:
            self._jobs.insert_one(
                self._new_job_doc(job_id, name, config, dataset, now, user_id, workspace_id)
            )
        return self.get(job_id)

    def get(self, job_id: str) -> dict | None:
        """Get a job by ID."""
        if not self.is_available:
            return None
        doc = self._jobs.find_one({"_id": job_id})
        return self.store_row_to_job(doc) if doc else None

    def list(
        self,
        status: str | None = None,
        include_crashed: bool = True,
        user_id: str = "",
    ) -> list[dict]:
        """List all jobs, optionally filtered by status and user."""
        if not self.is_available:
            return []
        query: dict[str, Any] = {}
        if status:
            query["status"] = status
        if not include_crashed:
            query["crashed"] = 0
        if user_id:
            query["user_id"] = user_id

        docs = self._jobs.find(query, sort=[("created_at", -1)])
        return [self.store_row_to_job(d) for d in docs]

    def list_by_workspace(self, workspace_id: str, status: str | None = None) -> list[dict]:
        """List jobs for a workspace."""
        if not self.is_available:
            return []
        query: dict[str, Any] = {"workspace_id": workspace_id}
        if status:
            query["status"] = status
        docs = self._jobs.find(query, sort=[("created_at", -1)])
        return [self.store_row_to_job(d) for d in docs]

    def update(self, job_id: str, **kwargs) -> dict | None:
        """Update job fields.

        A completed row's status is immutable: any update that tries to move it
        to a non-completed status is rejected whole (journal forensics, Sep 28:
        a stale writer persisted status=running 2.6ms after completed and the
        job resurfaced as a phantom recoverable after restart).
        """
        kwargs["updated_at"] = utc_now_iso()

        # Don't allow updating id
        kwargs.pop("id", None)

        with self._lock:
            if "status" in kwargs:
                current = self._jobs.find_one({"_id": job_id})
                if (
                    current is not None
                    and current.get("status") == "completed"
                    and kwargs["status"] != "completed"
                ):
                    logger.warning(
                        "update(%s): rejected resurrection of completed row to %r",
                        job_id,
                        kwargs["status"],
                    )
                    return current
            self._jobs.update_one({"_id": job_id}, {"$set": kwargs})

        return self.get(job_id)

    def update_progress(
        self,
        job_id: str,
        progress: float,
        epoch: int = 0,
        step: int = 0,
        loss: float | None = None,
    ) -> None:
        """Update job progress."""
        self.update(
            job_id,
            progress=progress,
            current_epoch=epoch,
            global_step=step,
            loss=loss,
            train_loss=loss,
            last_heartbeat=utc_now_iso(),
        )

    def mark_started(self, job_id: str) -> None:
        """Mark job as started."""
        self.update(
            job_id,
            status="running",
            started_at=utc_now_iso(),
            last_heartbeat=utc_now_iso(),
        )

    def mark_completed(self, job_id: str, checkpoint_path: str = "") -> None:
        """Mark job as completed."""
        self.update(
            job_id,
            status="completed",
            progress=100,
            completed_at=utc_now_iso(),
            checkpoint_path=checkpoint_path,
        )

    def mark_failed(self, job_id: str, error: str) -> None:
        """Mark job as failed."""
        self.update(job_id, status="failed", error=error, completed_at=utc_now_iso())

    def mark_crashed(self, job_id: str) -> None:
        """Mark job as crashed/interrupted."""
        self.update(job_id, crashed=1, status="interrupted", updated_at=utc_now_iso())

    def mark_recovering(self, job_id: str) -> None:
        """Mark a job as being actively recovered.

        Sets a fresh heartbeat so the row is not mistaken for a crashed or
        recoverable job while the recovery run is alive (see
        ``detect_crashed_jobs`` / ``get_recoverable_jobs``).
        """
        self.update(
            job_id,
            status="recovering",
            crashed=0,
            last_heartbeat=utc_now_iso(),
        )

    @staticmethod
    def is_stale_heartbeat(job: dict, timeout_seconds: int = 300) -> bool:
        """Return True when a job's heartbeat is absent or older than ``timeout_seconds``.

        ``job`` is a store row dict (as returned by ``get`` / ``list``).
        """
        hb = job.get("last_heartbeat")
        if not hb:
            return True
        last = _heartbeat_instant(hb)
        if last is None:
            return True
        return (datetime.now(UTC) - last).total_seconds() > timeout_seconds

    def heartbeat(self, job_id: str) -> None:
        """Update heartbeat timestamp."""
        self.update(job_id, last_heartbeat=utc_now_iso())

    def delete(self, job_id: str) -> bool:
        """Delete a job."""
        with self._lock:
            deleted = self._jobs.delete_one({"_id": job_id}) > 0
            self._events.delete_many({"job_id": job_id})
            return deleted

    def detect_crashed_jobs(self, timeout_seconds: int = 300) -> builtins.list[dict]:
        """
        Detect jobs that may have crashed.

        Jobs that are 'running' (or 'recovering') but haven't sent a heartbeat
        in timeout_seconds are considered potentially crashed.

        Heartbeats are UTC ``...Z`` strings written by :func:`utc_now_iso`, but
        rows written before that migration are naive local strings — so the
        cutoff is enforced in Python against absolute instants (see
        ``_heartbeat_instant``) rather than as a lexicographic string
        comparison across mixed formats.
        """
        cutoff_epoch = datetime.now(UTC).timestamp() - timeout_seconds

        docs = self._jobs.find(
            {
                "status": {"$in": ["running", "recovering"]},
                "crashed": 0,
            }
        )
        stale = []
        for doc in docs:
            hb = _heartbeat_instant(doc.get("last_heartbeat"))
            if hb is None or hb.timestamp() < cutoff_epoch:
                stale.append(doc)
        return [self.store_row_to_job(d) for d in stale]

    def get_recoverable_jobs(self) -> builtins.list[dict]:
        """Get jobs that can be recovered.

        Returns 'interrupted' and 'failed' jobs (both are accepted by the
        recovery endpoint), plus 'recovering' rows whose heartbeat went stale
        (a recovery run that died without completing). A 'recovering' row with
        a fresh heartbeat is actively being recovered and is NOT listed here.
        """
        cutoff_epoch = datetime.now(UTC).timestamp() - 300

        docs = self._jobs.find(sort=[("created_at", -1)])
        recoverable = []
        for doc in docs:
            status = doc.get("status")
            if status in ("interrupted", "failed"):
                recoverable.append(doc)
            elif status == "recovering":
                hb = _heartbeat_instant(doc.get("last_heartbeat"))
                if hb is None or hb.timestamp() < cutoff_epoch:
                    recoverable.append(doc)

        # Hide jobs the recovery endpoint would reject anyway (no dataset
        # recorded, dataset gone, or a trainer this path cannot restart). A
        # Resume button that is guaranteed to 422 is worse than no button.
        from .recovery_policy import recovery_incompatibility

        eligible = []
        for doc in recoverable:
            job = self.store_row_to_job(doc)
            if recovery_incompatibility(job) is None:
                eligible.append(job)
        return eligible

    def log_event(self, job_id: str, event: str, data: dict | None = None) -> None:
        """Log a job event."""
        with self._lock:
            self._events.insert_one(
                {
                    "job_id": job_id,
                    "event": event,
                    "data": json.dumps(data) if data else None,
                    "timestamp": utc_now_iso(),
                }
            )

    def get_events(self, job_id: str, limit: int = 50) -> builtins.list[dict]:
        """Get events for a job."""
        docs = self._events.find(
            {"job_id": job_id},
            sort=[("timestamp", -1)],
            limit=limit,
        )

        def _safe_json(s: str) -> Any:
            try:
                return json.loads(s)
            except (json.JSONDecodeError, TypeError):
                return None

        return [
            {
                "event": doc.get("event"),
                "data": _safe_json(doc["data"]) if doc.get("data") else None,
                "timestamp": doc.get("timestamp"),
            }
            for doc in docs
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get job statistics."""
        with self._lock:
            docs = self._jobs.find()

            stats: dict[str, Any] = {}
            for doc in docs:
                status = doc.get("status", "unknown")
                stats[status] = stats.get(status, 0) + 1

            stats["total"] = len(docs)
            stats["crashed"] = sum(1 for d in docs if d.get("crashed"))
            return stats


# Global store instance
_job_store: JobStore | None = None


def get_job_store() -> JobStore:
    """Get the global job store instance."""
    global _job_store
    if _job_store is None:
        _job_store = JobStore()
    return _job_store


class PersistentTrainingJobs:
    """Dict-like wrapper around JobStore for backward compatibility.

    Provides the same ``training_jobs[job_id]`` interface while persisting
    all mutations to MogDB. Falls back to an in-memory dict if JobStore
    is unavailable.

    ``__getitem__`` returns the LIVE in-process object, so the established
    ``training_jobs[job_id][field] = value`` pattern works as callers
    expect. Only JSON-serializable, non-private fields are persisted to
    MogDB (private ``_`` keys hold threads/events); the live object keeps
    everything for the life of the process.
    """

    def __init__(self):
        self._fallback: dict[str, dict[str, Any]] = {}
        self._live: dict[str, dict[str, Any]] = {}

    def _store(self) -> JobStore | None:
        try:
            s = get_job_store()
            return s if s.is_available else None
        except Exception:
            return None

    @staticmethod
    def _persistable(value: dict[str, Any]) -> dict[str, Any]:
        """Strip private/non-serializable keys before MogDB persistence."""
        return {k: v for k, v in value.items() if not k.startswith("_")}

    def _persist(self, key: str, value: dict[str, Any]) -> None:
        store = self._store()
        persistable = self._persistable(value)
        if store:
            try:
                existing = store.get(key)
                if existing:
                    # Merge: update existing doc with new values
                    updates = {k: v for k, v in persistable.items() if k not in ("id", "_id")}
                    if updates:
                        store.update(key, **updates)
                else:
                    # Create new doc
                    doc = {**persistable, "_id": key, "id": key}
                    store._jobs.insert_one(doc)
            except Exception as exc:
                logger.debug("JobStore persist failed for %s: %s", key, exc)
        else:
            self._fallback[key] = value

    def save(self, key: str) -> None:
        """Write the live object back to MogDB (after in-place mutation)."""
        live = self._live.get(key)
        if live is not None:
            self._persist(key, live)

    def __getitem__(self, key: str) -> dict[str, Any]:
        if key in self._live:
            return self._live[key]
        store = self._store()
        if store:
            # get() already maps store rows to the API job shape.
            doc = store.get(key)
            if doc is not None:
                self._live[key] = doc
                return doc
        return self._fallback[key]

    def __setitem__(self, key: str, value: dict[str, Any]) -> None:
        # Most job-creation paths never set created_at; stamp it here so every
        # job registered through training_jobs[id] = {...} has a timestamp.
        if not value.get("created_at"):
            value["created_at"] = utc_now_iso()
        self._live[key] = value
        self._persist(key, value)

    def __delitem__(self, key: str) -> None:
        self._live.pop(key, None)
        store = self._store()
        if store:
            store.delete(key)
        else:
            del self._fallback[key]

    def __contains__(self, key: object) -> bool:
        # Live first: ephemeral jobs (recovery runs registered via set_live)
        # have no store row, and a GET on their id must not 404.
        if key in self._live:
            return True
        store = self._store()
        if store:
            return store.get(str(key)) is not None
        return key in self._fallback

    def __len__(self) -> int:
        store = self._store()
        if store:
            return len(store.list())
        return len(self._fallback)

    def __iter__(self):
        store = self._store()
        if store:
            return iter(j["id"] for j in store.list())
        return iter(self._fallback)

    def get(self, key: str, default=None) -> dict[str, Any] | None:
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key: str, *args):
        self._live.pop(key, None)
        store = self._store()
        if store:
            doc = store.get(key)
            if doc:
                store.delete(key)
                return doc
            if args:
                return args[0]
            raise KeyError(key)
        return self._fallback.pop(key, *args)

    def discard_live(self, key: str) -> None:
        """Drop a process-local live entry without touching the durable store.

        Used by list auto-purge so finished jobs leave the in-memory hot path
        while JobStore keeps history for restart / resume UI.
        """
        self._live.pop(key, None)
        self._fallback.pop(key, None)

    def set_live(self, key: str, value: dict[str, Any]) -> None:
        """Register a process-local job WITHOUT creating a store row.

        Recovery runs are ephemeral: the durable record is the original job's
        row (terminal writes target it), so a store row for the recovery id
        would be born ``running`` and never finalized — surfacing as a phantom
        "running" job while the process lives and as a phantom recoverable row
        after the next restart, when ``restore`` marks it interrupted.
        """
        if not value.get("created_at"):
            value["created_at"] = utc_now_iso()
        self._live[key] = value

    def values(self):
        # Live objects win: in-place progress/status mutations between
        # persists must be visible to polling readers in the same process.
        # Store rows are mapped to the API job shape so a job that left the
        # live dict still exposes checkpoint/model/epochs after auto-purge.
        store = self._store()
        if store:
            merged = {j["id"]: j for j in store.list()}
            merged.update(self._live)
            return list(merged.values())
        merged = dict(self._fallback)
        merged.update(self._live)
        return list(merged.values())

    def items(self):
        return [(j["id"], j) for j in self.values()]

    def keys(self):
        store = self._store()
        if store:
            ids = [j["id"] for j in store.list()]
            # Include live-only entries (set_live recovery runs have no row).
            ids += [k for k in self._live if k not in set(ids)]
            return ids
        return list(dict.fromkeys([*self._fallback.keys(), *self._live.keys()]))

    def update(self, other=None, **kwargs):
        if other:
            for k, v in other.items() if hasattr(other, "items") else other:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def setdefault(self, key: str, default=None):
        if key not in self:
            self[key] = default if default is not None else {}
        return self[key]

    def clear(self) -> None:
        """Remove all jobs from both the persistent store and the fallback dict."""
        self._live.clear()
        store = self._store()
        if store:
            for doc in store.list():
                doc_id = doc.get("id") or doc.get("_id")
                if doc_id:
                    store.delete(doc_id)
        self._fallback.clear()
