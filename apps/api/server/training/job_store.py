"""Persistent Training Job Store

Stores training jobs in MogDB (the project's embedded document database)
for crash recovery. Jobs persist across server restarts.
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from mogdb import MogDB
from domains.shared import find_repo_root

logger = logging.getLogger("slo.job_store")


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

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(find_repo_root(Path(__file__).resolve()) / "data" / "training_jobs.db")
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
        config: Dict[str, Any],
        dataset: str,
        now: str,
    ) -> Dict[str, Any]:
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
            "created_at": now,
            "started_at": None,
            "updated_at": now,
            "completed_at": None,
            "last_heartbeat": now,
            "crashed": 0,
        }

    @staticmethod
    def _doc_to_job(doc: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a stored MogDB document to the job dict returned to callers."""
        return {
            k: v
            for k, v in doc.items()
            if k not in ("_id", "_created", "_updated")
        }

    def create(self, job_id: str, name: str, config: Dict[str, Any], dataset: str = "") -> Dict:
        """Create a new job."""
        if not self.is_available:
            return {"id": job_id, "status": "error", "error": "Job store unavailable"}
        now = datetime.now().isoformat()
        with self._lock:
            self._jobs.insert_one(self._new_job_doc(job_id, name, config, dataset, now))
        return self.get(job_id)

    def get(self, job_id: str) -> Optional[Dict]:
        """Get a job by ID."""
        if not self.is_available:
            return None
        doc = self._jobs.find_one({"_id": job_id})
        return self._doc_to_job(doc) if doc else None

    def list(self, status: Optional[str] = None, include_crashed: bool = True) -> List[Dict]:
        """List all jobs, optionally filtered by status."""
        if not self.is_available:
            return []
        query: Dict[str, Any] = {}
        if status:
            query["status"] = status
        if not include_crashed:
            query["crashed"] = 0

        docs = self._jobs.find(query, sort=[("created_at", -1)])
        return [self._doc_to_job(d) for d in docs]

    def update(self, job_id: str, **kwargs) -> Optional[Dict]:
        """Update job fields."""
        kwargs["updated_at"] = datetime.now().isoformat()

        # Don't allow updating id
        kwargs.pop("id", None)

        with self._lock:
            self._jobs.update_one({"_id": job_id}, {"$set": kwargs})

        return self.get(job_id)

    def update_progress(
        self,
        job_id: str,
        progress: float,
        epoch: int = 0,
        step: int = 0,
        loss: Optional[float] = None,
    ) -> None:
        """Update job progress."""
        self.update(
            job_id,
            progress=progress,
            current_epoch=epoch,
            global_step=step,
            loss=loss,
            train_loss=loss,
            last_heartbeat=datetime.now().isoformat(),
        )

    def mark_started(self, job_id: str) -> None:
        """Mark job as started."""
        self.update(
            job_id,
            status="running",
            started_at=datetime.now().isoformat(),
            last_heartbeat=datetime.now().isoformat(),
        )

    def mark_completed(self, job_id: str, checkpoint_path: str = "") -> None:
        """Mark job as completed."""
        self.update(
            job_id,
            status="completed",
            progress=100,
            completed_at=datetime.now().isoformat(),
            checkpoint_path=checkpoint_path,
        )

    def mark_failed(self, job_id: str, error: str) -> None:
        """Mark job as failed."""
        self.update(job_id, status="failed", error=error, completed_at=datetime.now().isoformat())

    def mark_crashed(self, job_id: str) -> None:
        """Mark job as crashed/interrupted."""
        self.update(job_id, crashed=1, status="interrupted", updated_at=datetime.now().isoformat())

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
            last_heartbeat=datetime.now().isoformat(),
        )

    @staticmethod
    def is_stale_heartbeat(job: Dict, timeout_seconds: int = 300) -> bool:
        """Return True when a job's heartbeat is absent or older than ``timeout_seconds``.

        ``job`` is a store row dict (as returned by ``get`` / ``list``).
        """
        hb = job.get("last_heartbeat")
        if not hb:
            return True
        try:
            last = datetime.fromisoformat(hb)
        except ValueError:
            return True
        return (datetime.now() - last).total_seconds() > timeout_seconds

    def heartbeat(self, job_id: str) -> None:
        """Update heartbeat timestamp."""
        self.update(job_id, last_heartbeat=datetime.now().isoformat())

    def delete(self, job_id: str) -> bool:
        """Delete a job."""
        with self._lock:
            deleted = self._jobs.delete_one({"_id": job_id}) > 0
            self._events.delete_many({"job_id": job_id})
            return deleted

    def detect_crashed_jobs(self, timeout_seconds: int = 300) -> List[Dict]:
        """
        Detect jobs that may have crashed.

        Jobs that are 'running' (or 'recovering') but haven't sent a heartbeat
        in timeout_seconds are considered potentially crashed.

        Note: heartbeats are persisted with ``datetime.now().isoformat()``
        ('T' separator), so the cutoff is built in the SAME format.
        """
        cutoff = datetime.fromtimestamp(datetime.now().timestamp() - timeout_seconds).isoformat()

        docs = self._jobs.find(
            {
                "status": {"$in": ["running", "recovering"]},
                "crashed": 0,
                "last_heartbeat": {"$lt": cutoff},
            }
        )
        return [self._doc_to_job(d) for d in docs]

    def get_recoverable_jobs(self) -> List[Dict]:
        """Get jobs that can be recovered.

        Returns 'interrupted' and 'failed' jobs (both are accepted by the
        recovery endpoint), plus 'recovering' rows whose heartbeat went stale
        (a recovery run that died without completing). A 'recovering' row with
        a fresh heartbeat is actively being recovered and is NOT listed here.
        """
        cutoff = datetime.fromtimestamp(datetime.now().timestamp() - 300).isoformat()

        docs = self._jobs.find(sort=[("created_at", -1)])
        recoverable = []
        for doc in docs:
            status = doc.get("status")
            if status in ("interrupted", "failed"):
                recoverable.append(doc)
            elif status == "recovering":
                hb = doc.get("last_heartbeat")
                if hb is None or hb < cutoff:
                    recoverable.append(doc)

        return [self._doc_to_job(d) for d in recoverable]

    def log_event(self, job_id: str, event: str, data: Optional[Dict] = None) -> None:
        """Log a job event."""
        with self._lock:
            self._events.insert_one(
                {
                    "job_id": job_id,
                    "event": event,
                    "data": json.dumps(data) if data else None,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    def get_events(self, job_id: str, limit: int = 50) -> List[Dict]:
        """Get events for a job."""
        docs = self._events.find(
            {"job_id": job_id},
            sort=[("timestamp", -1)],
            limit=limit,
        )

        return [
            {
                "event": doc.get("event"),
                "data": json.loads(doc["data"]) if doc.get("data") else None,
                "timestamp": doc.get("timestamp"),
            }
            for doc in docs
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get job statistics."""
        with self._lock:
            docs = self._jobs.find()

            stats: Dict[str, Any] = {}
            for doc in docs:
                status = doc.get("status", "unknown")
                stats[status] = stats.get(status, 0) + 1

            stats["total"] = len(docs)
            stats["crashed"] = sum(1 for d in docs if d.get("crashed"))
            return stats


# Global store instance
_job_store: Optional[JobStore] = None


def get_job_store() -> JobStore:
    """Get the global job store instance."""
    global _job_store
    if _job_store is None:
        _job_store = JobStore()
    return _job_store
