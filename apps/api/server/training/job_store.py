"""Persistent Training Job Store

Stores training jobs in SQLite for crash recovery.
Jobs persist across server restarts.
"""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger("man.job_store")


class JobStore:
    """
    SQLite-backed persistent job store.

    Features:
    - Persists across server restarts
    - Tracks job state, progress, checkpoints
    - Detects crashed/interrupted jobs
    - Supports recovery/resume
    """

    def __init__(self, db_path: str = "data/training_jobs.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    status TEXT DEFAULT 'pending',
                    dataset TEXT,
                    data_path TEXT,
                    config TEXT,
                    progress REAL DEFAULT 0,
                    current_epoch INTEGER DEFAULT 0,
                    total_epochs INTEGER DEFAULT 0,
                    global_step INTEGER DEFAULT 0,
                    loss REAL,
                    train_loss REAL,
                    eval_loss REAL,
                    checkpoint_path TEXT,
                    checkpoint_dir TEXT,
                    error TEXT,
                    created_at TEXT,
                    started_at TEXT,
                    updated_at TEXT,
                    completed_at TEXT,
                    last_heartbeat TEXT,
                    crashed INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS job_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    event TEXT,
                    data TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (job_id) REFERENCES jobs(id)
                )
            """)
            conn.commit()
            conn.close()

    def create(self, job_id: str, name: str, config: Dict[str, Any], dataset: str = "") -> Dict:
        """Create a new job."""
        now = datetime.now().isoformat()
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                """
                INSERT INTO jobs (id, name, status, dataset, config, created_at, updated_at, last_heartbeat)
                VALUES (?, ?, 'pending', ?, ?, ?, ?, ?)
            """,
                (job_id, name, dataset, json.dumps(config), now, now, now),
            )
            conn.commit()
            conn.close()

        return self.get(job_id)

    def get(self, job_id: str) -> Optional[Dict]:
        """Get a job by ID."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            conn.close()

            if not row:
                return None

            return self._row_to_dict(row)

    def list(self, status: Optional[str] = None, include_crashed: bool = True) -> List[Dict]:
        """List all jobs, optionally filtered by status."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM jobs"
            params = []

            conditions = []
            if status:
                conditions.append("status = ?")
                params.append(status)
            if not include_crashed:
                conditions.append("crashed = 0")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY created_at DESC"

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_dict(row) for row in rows]

    def update(self, job_id: str, **kwargs) -> Optional[Dict]:
        """Update job fields."""
        kwargs["updated_at"] = datetime.now().isoformat()

        # Don't allow updating id
        kwargs.pop("id", None)

        with self._lock:
            conn = sqlite3.connect(str(self.db_path))

            set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
            values = list(kwargs.values()) + [job_id]

            conn.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", values)
            conn.commit()
            conn.close()

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

    def heartbeat(self, job_id: str) -> None:
        """Update heartbeat timestamp."""
        self.update(job_id, last_heartbeat=datetime.now().isoformat())

    def delete(self, job_id: str) -> bool:
        """Delete a job."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            conn.execute("DELETE FROM job_events WHERE job_id = ?", (job_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
            conn.close()
            return deleted

    def detect_crashed_jobs(self, timeout_seconds: int = 300) -> List[Dict]:
        """
        Detect jobs that may have crashed.

        Jobs that are 'running' but haven't sent a heartbeat in timeout_seconds
        are considered crashed.
        """
        cutoff = datetime.now().timestamp() - timeout_seconds

        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'running'
                AND last_heartbeat < datetime(?, 'unixepoch')
                AND crashed = 0
            """,
                (cutoff,),
            )

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_dict(row) for row in rows]

    def get_recoverable_jobs(self) -> List[Dict]:
        """Get jobs that can be recovered (interrupted or crashed)."""
        return self.list(status="interrupted", include_crashed=True)

    def log_event(self, job_id: str, event: str, data: Optional[Dict] = None) -> None:
        """Log a job event."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                """
                INSERT INTO job_events (job_id, event, data, timestamp)
                VALUES (?, ?, ?, ?)
            """,
                (job_id, event, json.dumps(data) if data else None, datetime.now().isoformat()),
            )
            conn.commit()
            conn.close()

    def get_events(self, job_id: str, limit: int = 50) -> List[Dict]:
        """Get events for a job."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM job_events
                WHERE job_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (job_id, limit),
            )
            rows = cursor.fetchall()
            conn.close()

            return [
                {
                    "event": row["event"],
                    "data": json.loads(row["data"]) if row["data"] else None,
                    "timestamp": row["timestamp"],
                }
                for row in rows
            ]

    def get_stats(self) -> Dict[str, Any]:
        """Get job statistics."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row

            stats = {}

            cursor = conn.execute("SELECT status, COUNT(*) as count FROM jobs GROUP BY status")
            for row in cursor:
                stats[row["status"]] = row["count"]

            cursor = conn.execute("SELECT COUNT(*) as total FROM jobs")
            stats["total"] = cursor.fetchone()["total"]

            cursor = conn.execute("SELECT COUNT(*) as count FROM jobs WHERE crashed = 1")
            stats["crashed"] = cursor.fetchone()["count"]

            conn.close()
            return stats

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """Convert a row to a dictionary."""
        result = dict(row)

        # Parse JSON fields
        if result.get("config") and isinstance(result["config"], str):
            try:
                result["config"] = json.loads(result["config"])
            except Exception:
                pass

        return result


# Global store instance
_job_store: Optional[JobStore] = None


def get_job_store() -> JobStore:
    """Get the global job store instance."""
    global _job_store
    if _job_store is None:
        _job_store = JobStore()
    return _job_store
