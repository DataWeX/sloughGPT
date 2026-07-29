"""
Shared thread pool executor for training jobs.

Replaces per-endpoint raw ``threading.Thread`` spawns with a single
concurrent pool that enforces a configurable concurrency limit and
provides per-job tracking via a future registry.

Usage::

    executor = get_training_executor()
    job_id = executor.submit(train_fn, job_id="my_job", epochs=10)
    status = executor.status(job_id)
    executor.cancel(job_id)
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger("slo.training.executor")


class JobStatus(str, Enum):
    """Lifecycle states for a training job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobInfo:
    """Metadata for a single training job."""

    job_id: str
    tree_id: Optional[str] = None
    status: JobStatus = JobStatus.QUEUED
    future: Optional[Future] = field(default=None, repr=False)
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    cancel_requested: bool = False
    result: Any = None

    def elapsed(self) -> Optional[float]:
        """Wall-clock seconds since submission (or total if completed)."""
        end = self.completed_at or time.time()
        return end - self.submitted_at

    def to_dict(self) -> dict[str, Any]:
        d = {
            "job_id": self.job_id,
            "tree_id": self.tree_id,
            "status": self.status.value,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_s": self.elapsed(),
            "error": self.error,
            "cancel_requested": self.cancel_requested,
        }
        if self.status == JobStatus.COMPLETED and self.result is not None:
            if isinstance(self.result, dict):
                d["result_keys"] = list(self.result.keys())
                d["result_size_bytes"] = sum(
                    v.nbytes if hasattr(v, "nbytes") else sys.getsizeof(v)
                    for v in self.result.values()
                )
            else:
                d["result_type"] = type(self.result).__name__
        return d


class TrainingExecutor:
    """Shared thread pool for CPU-bound training jobs.

    Concurrency is bounded by ``max_workers`` (default 2). Excess jobs
    are queued by the underlying ``ThreadPoolExecutor`` and start when a
    slot opens. Each job is tracked via a :class:`JobInfo` record keyed
    by ``job_id``.

    Cancellation sets a flag; the training function must check
    ``executor.is_cancelled(job_id)`` periodically to honour it.
    """

    def __init__(self, max_workers: int = 2, thread_name_prefix: str = "train"):
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._jobs: dict[str, JobInfo] = {}
        self._lock = threading.Lock()

    # ── Submit ────────────────────────────────────────────────────────

    def submit(
        self,
        fn: Callable[..., Any],
        job_id: str,
        *args: Any,
        tree_id: Optional[str] = None,
        _call_args: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Submit a training function to the pool.

        Args:
            fn: Synchronous training function.  Receives ``job_id`` as the
                first positional arg, followed by *args / **kwargs*.
            job_id: Unique identifier (caller ensures uniqueness).
            *args: Forwarded to *fn* after *job_id*.
            tree_id: Optional ModelTree name for isolation (Point-Graph-Queue).
            _call_args: Extra keyword args merged into *kwargs*.
            **kwargs: Forwarded to *fn*.

        Returns:
            The *job_id* for tracking.
        """
        if _call_args:
            kwargs = {**_call_args, **kwargs}

        info = JobInfo(job_id=job_id, tree_id=tree_id)
        with self._lock:
            self._jobs[job_id] = info

        def _wrapper() -> Any:
            # Switch RM to training mode for optimal pool sizing
            from domains.infrastructure.resource_manager import get_resource_manager
            rm = get_resource_manager()
            prev_mode = rm.mode
            if prev_mode != "training":
                rm.recompute("training")
            info.status = JobStatus.RUNNING
            info.started_at = time.time()
            try:
                result = fn(job_id, *args, **kwargs)
                info.status = JobStatus.COMPLETED
                info.result = result
                return result
            except Exception as exc:
                info.status = JobStatus.FAILED
                info.error = str(exc)
                raise
            finally:
                info.completed_at = time.time()
                if prev_mode != "training":
                    rm.recompute(prev_mode)

        future = self._executor.submit(_wrapper)
        info.future = future
        logger.info(
            "Submitted training job %s tree=%s (pool=%d/%d)",
            job_id, tree_id or "-", self._running(), self._max_workers,
            extra={"tag": "TRAIN"},
        )
        return job_id

    def submit_training(
        self,
        fn: Callable[..., Any],
        job_id: str,
        tree_id: str,
        point_library: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Submit a training job routed to a specific ModelTree.

        This is the primary integration point with the Point-Graph-Queue
        architecture (LLM.md).  The ``tree_id`` isolates this job to a
        specific ModelTree, and the optional ``point_library`` receives
        the trained weights as compressed Points on completion.

        The training function ``fn`` receives:
            - ``job_id`` (str)
            - ``tree_id`` (str)
            - ``point_library`` (PointLibrary | None)
            - ``is_cancelled`` (callable → bool)
            - *args / **kwargs

        After ``fn`` returns, if ``point_library`` is not None and the
        result is a dict of ``{name: numpy_array}``, the arrays are
        compressed into Points and added to the library automatically.
        """
        import functools

        is_cancelled_fn = functools.partial(self.is_cancelled, job_id)

        def _wrapped_fn(jid: str, *a: Any, **kw: Any) -> Any:
            result = fn(jid, tree_id, point_library, is_cancelled_fn, *a, **kw)
            # Auto-store trained weights as Points in the library
            if point_library is not None and isinstance(result, dict):
                try:
                    from domains.infrastructure.pugqeep import PointCompressor
                    compressor = PointCompressor()
                    for name, weights in result.items():
                        if hasattr(weights, "nbytes"):
                            point = compressor.compress_cluster(
                                weights, name, n_clusters=16,
                            )
                            point_library.add(point)
                    logger.info(
                        "Stored %d trained Points in library for tree %s",
                        len(result), tree_id,
                        extra={"tag": "TRAIN"},
                    )
                except Exception as e:
                    logger.warning("Failed to store trained Points: %s", e, extra={"tag": "TRAIN"})
            return result

        return self.submit(
            _wrapped_fn, job_id, *args, tree_id=tree_id, **kwargs,
        )

    # ── Query ─────────────────────────────────────────────────────────

    def status(self, job_id: str) -> Optional[dict[str, Any]]:
        """Return job metadata dict, or None if unknown."""
        info = self._jobs.get(job_id)
        if info is None:
            return None
        return info.to_dict()

    def result_summary(self, job_id: str) -> Optional[dict[str, Any]]:
        """Return shape/dtype summary for a completed job's trained weights.

        Returns None if the job is unknown or not completed.
        The actual weight arrays are NOT returned (too large for HTTP).
        """
        info = self._jobs.get(job_id)
        if info is None or info.status != JobStatus.COMPLETED:
            return None
        if info.result is None or not isinstance(info.result, dict):
            return None
        return {
            "job_id": job_id,
            "tree_id": info.tree_id,
            "weights": {
                name: {
                    "shape": list(w.shape) if hasattr(w, "shape") else None,
                    "dtype": str(w.dtype) if hasattr(w, "dtype") else type(w).__name__,
                    "nbytes": w.nbytes if hasattr(w, "nbytes") else None,
                }
                for name, w in info.result.items()
            },
            "total_bytes": sum(
                w.nbytes if hasattr(w, "nbytes") else 0
                for w in info.result.values()
            ),
        }

    def list_jobs(self) -> list[dict[str, Any]]:
        """Return metadata for all tracked jobs, newest first."""
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(
            [j.to_dict() for j in jobs],
            key=lambda d: d["submitted_at"],
            reverse=True,
        )

    def is_cancelled(self, job_id: str) -> bool:
        """Check if cancellation was requested for *job_id*."""
        info = self._jobs.get(job_id)
        return info is not None and info.cancel_requested

    def active_count(self) -> int:
        """Number of currently running jobs."""
        return self._running()

    # ── Cancel ────────────────────────────────────────────────────────

    def cancel(self, job_id: str) -> bool:
        """Request cancellation for *job_id*.

        Sets a flag that ``is_cancelled(job_id)`` returns True for.
        The training function must check this periodically.
        If the job hasn't started yet, the future is cancelled outright.

        Returns:
            True if cancellation was applied (flag set or future cancelled).
        """
        info = self._jobs.get(job_id)
        if info is None:
            return False

        info.cancel_requested = True

        if info.future is not None and not info.future.done():
            cancelled = info.future.cancel()
            if cancelled:
                info.status = JobStatus.CANCELLED
                info.completed_at = time.time()
                logger.info("Cancelled queued job %s", job_id, extra={"tag": "TRAIN"})
                return True
            # Job is already running — flag is set, function must check is_cancelled()
            logger.info("Cancellation requested for running job %s (must check is_cancelled())", job_id, extra={"tag": "TRAIN"})
            return True
        return False

    def purge_completed(self, max_age_s: float = 3600.0) -> int:
        """Remove completed/failed/cancelled jobs older than *max_age_s*.

        Returns:
            Number of jobs purged.
        """
        cutoff = time.time() - max_age_s
        purged = 0
        with self._lock:
            to_remove = [
                jid
                for jid, info in self._jobs.items()
                if info.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
                and info.completed_at is not None
                and info.completed_at < cutoff
            ]
            for jid in to_remove:
                del self._jobs[jid]
                purged += 1
        if purged:
            logger.info("Purged %d old training jobs", purged, extra={"tag": "TRAIN"})
        return purged

    # ── Shutdown ──────────────────────────────────────────────────────

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the underlying thread pool."""
        self._executor.shutdown(wait=wait, cancel_futures=not wait)
        logger.info("TrainingExecutor shut down (workers=%d)", self._max_workers, extra={"tag": "TRAIN"})

    # ── Internals ─────────────────────────────────────────────────────

    def _running(self) -> int:
        """Count jobs in RUNNING state."""
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == JobStatus.RUNNING)


# ── Singleton ─────────────────────────────────────────────────────────

_instance: Optional[TrainingExecutor] = None
_instance_lock = threading.Lock()


def get_training_executor() -> TrainingExecutor:
    """Return (and lazily create) the global TrainingExecutor singleton.

    Pool size is determined by ``ResourceManager.train_pool_size``,
    overridable via ``SLO_TRAIN_POOL_SIZE`` env var.
    """
    global _instance
    if _instance is not None:
        return _instance
    with _instance_lock:
        if _instance is not None:
            return _instance
        from domains.infrastructure.resource_manager import get_resource_manager
        rm = get_resource_manager()
        import os
        max_workers = int(os.environ.get("SLO_TRAIN_POOL_SIZE", rm.train_pool_size))
        _instance = TrainingExecutor(max_workers=max_workers)
        logger.info("TrainingExecutor created (workers=%d)", max_workers, extra={"tag": "TRAIN"})
        return _instance


# ── Checkpoint Compression ──────────────────────────────────────────────


def compress_checkpoint(
    soul_path: str,
    n_clusters: int = 16,
    output_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Compress a ``.soul`` checkpoint into PointLibrary vectors.

    Args:
        soul_path: Path to the ``.soul`` checkpoint file.
        n_clusters: Number of VQ clusters for compression.
        output_dir: Directory for the ``.points.json`` library file.
            Defaults to the same directory as *soul_path*.

    Returns:
        Dict with ``point_count``, ``compression_ratio``,
        ``total_raw_bytes``, ``total_compressed_bytes``,
        ``library_path``, and ``meta_path`` — or ``None`` on failure.
    """
    import json
    from pathlib import Path

    soul_file = Path(soul_path)
    if not soul_file.exists():
        logger.warning("Checkpoint not found: %s", soul_path, extra={"tag": "TRAIN"})
        return None

    try:
        from domains.infrastructure.pugqeep import PointCompressor
        from domains.infrastructure.pugqeep.library import PointLibrary
        from domains.training.slonet import import_from_sou
    except ImportError as exc:
        logger.warning("Pugqeep/SloNet not available: %s", exc, extra={"tag": "TRAIN"})
        return None

    try:
        model = import_from_sou(str(soul_file))
        if model is None:
            return None
        state_dict = model.state_dict()
    except Exception as exc:
        logger.warning("Failed to load model from %s: %s", soul_path, exc, extra={"tag": "TRAIN"})
        return None

    out_dir = Path(output_dir) if output_dir else soul_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    lib_name = soul_file.stem
    lib_path = out_dir / f"{lib_name}.points.json"

    compressor = PointCompressor()
    library = PointLibrary(name=lib_name, storage_dir=out_dir)

    import numpy as np
    total_raw = 0
    total_compressed = 0
    for name, weights in state_dict.items():
        if not isinstance(weights, np.ndarray):
            weights = np.asarray(weights, dtype=np.float32)
        total_raw += weights.nbytes
        point_id = f"{lib_name}.{name}"
        point = compressor.compress_cluster(
            weights, point_id, n_clusters=n_clusters,
        )
        total_compressed += point.nbytes()
        library.add(point)

    library.save(lib_path)

    meta = {
        "source": str(soul_file),
        "n_clusters": n_clusters,
        "point_count": len(library.list_all()),
        "total_raw_bytes": total_raw,
        "total_compressed_bytes": total_compressed,
        "compression_ratio": total_raw / max(total_compressed, 1),
        "lineage": [str(soul_file)],
        "metadata": {
            "vocab_size": getattr(model, "vocab_size", 0),
            "n_embed": getattr(model, "n_embed", 0),
            "n_layer": getattr(model, "n_layer", 0),
            "n_head": getattr(model, "n_head", 4),
            "block_size": getattr(model, "block_size", 32),
            "use_rope": getattr(model, "use_rope", True),
            "weight_shapes": {name: list(w.shape) for name, w in state_dict.items()},
        },
    }
    meta_path = out_dir / f"{lib_name}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    stats = {
        **meta,
        "library_path": str(lib_path),
        "meta_path": str(meta_path),
    }
    logger.info(
        "Compressed %s → %d points (%.1fx ratio)",
        soul_file.name, stats["point_count"], stats["compression_ratio"],
        extra={"tag": "TRAIN"},
    )
    return stats
