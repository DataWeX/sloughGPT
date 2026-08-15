"""Persistent training runtime — graceful shutdown + startup restore.

Single registry of live training jobs plus the persistence bridge between the
in-memory ``training_jobs`` dict and the SQLite ``JobStore``.

Why this exists: before this module, only ``/training/hf-start`` and the
auto-config path wrote to the ``JobStore``. The main distill path, visual
training, and the auto-train router (turbo + SSE) lived only in memory, so a
server restart silently destroyed a running job. On shutdown the old code
merely flipped store rows to ``interrupted`` without ever signalling the
trainers to stop, so in-flight work was killed mid-step and no final
checkpoint was saved.

``TrainingRuntime`` provides:

* ``register``/``sync`` — persist a live job's initial row and then flush its
  progress / terminal state to the store on every update.
* ``shutdown`` — signal every tracked cancel event (trainer cooperatively
  saves a final checkpoint and exits), set the auto-train cancel events, wait
  a bounded grace period, then mark anything still running as ``interrupted``.
* ``restore`` — on startup, mark stale ``running`` rows as ``interrupted``
  (their previous process is gone) and seed the in-memory job list from the
  store so the job-history UI and the recovery panel show what survived.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("slo.training.runtime")

_TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "cancelled", "interrupted", "abandoned", "recovered"}
)
_STALE_STATUSES = frozenset({"running", "pending", "queued", "starting", "recovering"})


class TrainingRuntime:
    """Registry + persistence bridge for live training jobs."""

    def __init__(self, store: Any = None, grace_timeout_s: float = 10.0) -> None:
        """Create the runtime.

        Args:
            store: A ``JobStore`` instance. Defaults to the process-global
                ``get_job_store()`` on first use.

        Side effects:
            - None; the store is lazily resolved.
        """
        self._store = store
        self._grace_timeout_s = grace_timeout_s
        self._lock = threading.Lock()
        # job_id -> {"job": dict, "cancel_event": threading.Event | None}
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def _get_store(self):
        """Resolve the JobStore lazily (injectable for tests)."""
        if self._store is None:
            from training.job_store import get_job_store

            self._store = get_job_store()
        return self._store

    # ── Registry ──────────────────────────────────────────────────────────

    def register(
        self,
        job_id: str,
        job: dict[str, Any],
        cancel_event: Optional[threading.Event] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a live job and persist its initial row.

        Args:
            job_id: Stable job id (store primary key).
            job: The in-memory job dict (mutated in place by the worker).
            cancel_event: Cooperative ``threading.Event``; set on shutdown.
            config: Hyperparameter dict stored in the row's ``config`` column
                (used by recovery to restart the job).

        Side effects:
            - Creates a ``pending`` row in the JobStore if absent.
        """
        with self._lock:
            self._jobs[job_id] = {"job": job, "cancel_event": cancel_event}
        self._ensure_row(job_id, job, config)

    def _ensure_row(
        self, job_id: str, job: dict[str, Any], config: Optional[Dict[str, Any]]
    ) -> None:
        """Create a store row for the job if it does not already exist."""
        store = self._get_store()
        if store.get(job_id) is not None:
            return
        cfg = config if config is not None else job.get("config")
        if not isinstance(cfg, dict):
            cfg = {}
        try:
            store.create(
                job_id,
                str(job.get("name") or "training"),
                dict(cfg),
                str(job.get("dataset") or ""),
            )
            if job.get("status") == "running":
                store.mark_started(job_id)
        except Exception as e:
            logger.warning("JobStore create failed for %s: %s", job_id, e, extra={"tag": "TRAIN"})

    def _get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Return the tracked in-memory job dict, if any."""
        with self._lock:
            entry = self._jobs.get(job_id)
        return entry.get("job") if entry else None

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        """Return the tracked job dict (mutable reference) or None.

        Callers may mutate the returned dict in place; ``sync`` reads it back.
        """
        return self._get_job(job_id)

    # ── Persistence ───────────────────────────────────────────────────────

    def sync(self, job_id: str) -> None:
        """Flush a job's in-memory state to the persistent store.

        Maps the in-memory status into the store columns. Call from progress
        callbacks and from every terminal branch (completed/failed/cancelled).

        Side effects:
            - Updates the JobStore row for ``job_id``.
        """
        rec = self._get_job(job_id)
        if rec is None:
            return
        store = self._get_store()
        if store.get(job_id) is None:
            self._ensure_row(job_id, rec, rec.get("config"))
        status = str(rec.get("status") or "running")
        fields: Dict[str, Any] = {
            "name": rec.get("name"),
            "status": status,
            "progress": rec.get("progress", 0),
            "current_epoch": rec.get("current_epoch", 0),
            "total_epochs": rec.get("epochs"),
            "global_step": rec.get("global_step", 0),
            "loss": rec.get("loss"),
            "train_loss": rec.get("train_loss"),
            "eval_loss": rec.get("eval_loss"),
            "data_path": rec.get("data_path"),
            "checkpoint_path": rec.get("checkpoint"),
            "checkpoint_dir": rec.get("checkpoint_dir"),
            "error": rec.get("error"),
            "last_heartbeat": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }
        if status == "completed":
            fields["completed_at"] = fields["last_heartbeat"]
        elif status == "interrupted":
            fields["crashed"] = 1
        try:
            store.update(job_id, **{k: v for k, v in fields.items() if v is not None})
        except Exception as e:
            logger.warning("JobStore update failed for %s: %s", job_id, e, extra={"tag": "TRAIN"})

    # ── Shutdown ──────────────────────────────────────────────────────────

    def _signal_auto_train_cancel(self) -> None:
        """Set the auto-train router's module-level cancel events."""
        try:
            import routers.auto_train as at

            ev = getattr(at, "_auto_train_cancel_event", None)
            if ev is not None:
                try:
                    ev.set()
                except Exception:
                    pass
            tev = getattr(at, "_turbo_cancel_event", None)
            if tev is not None:
                try:
                    tev.set()
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Auto-train cancel signal failed: %s", e, extra={"tag": "TRAIN"})

    def shutdown(self, timeout_s: Optional[float] = None) -> None:
        """Gracefully stop every tracked job and persist the final state.

        Sequence:
        1. Set every tracked cancel event plus the auto-train cancel events.
           Cooperative trainers (``SloughGPTTrainer.train(cancel_event=...)``)
           then save a final checkpoint and exit on their own.
        2. Wait up to ``timeout_s`` for all jobs to reach a terminal status.
        3. Flush each job's final state to the store; mark anything still
           running as ``interrupted`` so ``/recovery/recoverable`` surfaces it.

        Args:
            timeout_s: Drain budget. Defaults to the constructor value.

        Side effects:
            - Mutates the store: terminal jobs flushed, live jobs interrupted.
        """
        timeout = timeout_s if timeout_s is not None else self._grace_timeout_s
        with self._lock:
            job_ids = list(self._jobs.keys())
        self._signal_auto_train_cancel()
        cooperative = []
        for jid in job_ids:
            entry = self._jobs.get(jid)
            ev = entry.get("cancel_event") if entry else None
            if ev is not None:
                try:
                    ev.set()
                except Exception:
                    pass
                cooperative.append(jid)

        # Jobs without a cooperative cancel event (e.g. SSE auto-train) cannot
        # persist progress during shutdown — their stream generator is gone.
        # Mark them interrupted now so the drain below only waits for trainers
        # that can still save a final checkpoint.
        for jid in job_ids:
            if jid in cooperative:
                continue
            rec = self._get_job(jid)
            if rec is not None and str(rec.get("status")) not in _TERMINAL_STATUSES:
                rec["status"] = "interrupted"
                rec["error"] = rec.get("error") or "Server shutdown"
                self.sync(jid)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pending = [
                jid
                for jid in job_ids
                if (rec := self._get_job(jid)) is not None
                and str(rec.get("status")) not in _TERMINAL_STATUSES
            ]
            if not pending:
                break
            time.sleep(0.25)

        for jid in job_ids:
            rec = self._get_job(jid)
            if rec is None:
                continue
            self.sync(jid)
            if str(rec.get("status")) not in _TERMINAL_STATUSES:
                store = self._get_store()
                try:
                    store.mark_crashed(jid)
                    logger.info(
                        "Marked job %s as interrupted on shutdown",
                        jid,
                        extra={"tag": "TRAIN"},
                    )
                except Exception as e:
                    logger.warning("Interrupt mark failed for %s: %s", jid, e, extra={"tag": "TRAIN"})

    # ── Startup restore ───────────────────────────────────────────────────

    def restore(self) -> None:
        """Mark stale rows interrupted and seed the in-memory job list.

        A new server process means any job the previous process left in a
        running/pending state is gone; those rows become ``interrupted`` so
        the user can recover from the last checkpoint. Surviving rows are
        then mirrored into ``training_jobs`` so the job-history UI and the
        recovery panel show what actually persisted.

        Side effects:
            - Updates the store: stale rows -> interrupted.
            - Populates the in-memory ``training.jobs.training_jobs`` dict.
        """
        store = self._get_store()
        try:
            rows = store.list()
        except Exception as e:
            logger.warning("JobStore list failed on restore: %s", e, extra={"tag": "TRAIN"})
            return
        for row in rows:
            if row.get("status") in _STALE_STATUSES:
                try:
                    store.mark_crashed(row["id"])
                    logger.info(
                        "Marked stale job %s as interrupted on startup",
                        row["id"],
                        extra={"tag": "START"},
                    )
                except Exception as e:
                    logger.warning("Stale mark failed for %s: %s", row["id"], e, extra={"tag": "START"})

        try:
            from training.jobs import training_jobs
        except Exception as e:
            logger.warning("Could not import training job registry: %s", e, extra={"tag": "START"})
            return

        for row in rows:
            jid = row["id"]
            if jid in training_jobs:
                continue
            config = row.get("config") or {}
            training_jobs[jid] = {
                "id": jid,
                "name": row.get("name") or "training",
                "model": config.get("model") or "sloughgpt",
                "dataset": row.get("dataset") or "",
                "data_path": row.get("data_path") or "",
                "data_source": config.get("data_source") or config.get("source_kind") or "",
                "status": row.get("status") or "interrupted",
                "progress": row.get("progress") or 0,
                "epochs": row.get("total_epochs"),
                "current_epoch": row.get("current_epoch") or 0,
                "global_step": row.get("global_step") or 0,
                "loss": row.get("loss"),
                "train_loss": row.get("train_loss"),
                "eval_loss": row.get("eval_loss"),
                "checkpoint": row.get("checkpoint_path") or "",
                "checkpoint_dir": row.get("checkpoint_dir") or "",
                "error": row.get("error"),
                "config": config,
                "created_at": row.get("created_at"),
            }


_runtime: Optional[TrainingRuntime] = None
_runtime_lock = threading.Lock()


def get_training_runtime() -> TrainingRuntime:
    """Return the process-global ``TrainingRuntime`` singleton."""
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = TrainingRuntime()
    return _runtime
