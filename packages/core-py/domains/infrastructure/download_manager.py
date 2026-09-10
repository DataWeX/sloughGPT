"""
Download Manager — generic orchestration for pluggable download backends.

Delegates source-specific logic (cache resolution, file listing, download
execution) to a ``DownloadBackend`` implementation.  This module handles
only scheduling, progress tracking, cancellation, and stale cleanup.

The default backend is ``HFDownloadBackend`` (HuggingFace models), but any
``DownloadBackend`` implementation can be injected via ``set_backend()``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("slo.infrastructure.download_manager")

# ---------------------------------------------------------------------------
# Backend (pluggable)
# ---------------------------------------------------------------------------

_backend = None
_backend_lock = threading.Lock()


def get_backend():
    """Return the current download backend, lazily initializing the HF default."""
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                try:
                    from domains.infrastructure.hf_hub import HFDownloadBackend
                    _backend = HFDownloadBackend()
                except ImportError:
                    from domains.infrastructure.download_backend import DownloadBackend
                    _backend = _NullBackend()
    return _backend


def set_backend(backend) -> None:
    """Override the download backend (for testing or alternative sources)."""
    global _backend
    with _backend_lock:
        _backend = backend


def reset_backend() -> None:
    """Reset to default (HF) backend."""
    global _backend
    with _backend_lock:
        _backend = None


class _NullBackend:
    """Fallback when no backend is available — all operations are no-ops."""

    def is_cached(self, resource_id, deep_check=False):
        return False

    def get_cache_dir(self, resource_id):
        return ""

    def estimate_total(self, resource_id):
        return 0

    def list_files(self, resource_id):
        return []

    def download(self, resource_id, on_progress, on_file_complete):
        return {"status": "failed", "error": "no backend configured"}

    def cleanup(self, resource_id):
        return False

    def list_incomplete(self):
        return []


# ---------------------------------------------------------------------------
# Module-level convenience — delegate to backend
# ---------------------------------------------------------------------------


def is_download_complete(model_id: str, deep_check: bool = False) -> bool:
    """Check if a model is fully cached on disk."""
    return get_backend().is_cached(model_id, deep_check=deep_check)


def cleanup_incomplete(model_id: str) -> bool:
    """Remove an incomplete/partial download."""
    return get_backend().cleanup(model_id)


def list_incomplete_models() -> List[str]:
    """Scan cache and return resource IDs with incomplete downloads."""
    return get_backend().list_incomplete()


# ---------------------------------------------------------------------------
# Status / Progress
# ---------------------------------------------------------------------------


class DownloadStatus(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    PAUSED = "paused"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DownloadProgress:
    model_id: str
    status: DownloadStatus
    bytes_downloaded: int = 0
    total_bytes: int = 0
    speed_bytes_per_sec: float = 0.0
    eta_seconds: float = 0.0
    percentage: float = 0.0
    current_file: str = ""
    error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    files_completed: int = 0
    files_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "status": self.status.value,
            "bytes_downloaded": self.bytes_downloaded,
            "total_bytes": self.total_bytes,
            "speed_mb_per_sec": round(self.speed_bytes_per_sec / (1024 * 1024), 2),
            "eta_seconds": round(self.eta_seconds, 1),
            "percentage": round(self.percentage, 1),
            "current_file": self.current_file,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "files_completed": self.files_completed,
            "files_total": self.files_total,
        }


@dataclass
class DownloadStats:
    """Cumulative download statistics."""
    total_downloads: int = 0
    completed_downloads: int = 0
    failed_downloads: int = 0
    cancelled_downloads: int = 0
    total_bytes_downloaded: int = 0
    total_download_time: float = 0.0
    average_speed: float = 0.0
    peak_speed: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_downloads": self.total_downloads,
            "completed_downloads": self.completed_downloads,
            "failed_downloads": self.failed_downloads,
            "cancelled_downloads": self.cancelled_downloads,
            "total_bytes_downloaded": self.total_bytes_downloaded,
            "total_download_time": round(self.total_download_time, 2),
            "average_speed_mb_per_sec": round(self.average_speed / (1024 * 1024), 2),
            "peak_speed_mb_per_sec": round(self.peak_speed / (1024 * 1024), 2),
        }


# ---------------------------------------------------------------------------
# DownloadManager — generic orchestration
# ---------------------------------------------------------------------------


class DownloadManager:
    """Download manager singleton.

    Orchestrates downloads via a pluggable ``DownloadBackend``.
    Handles scheduling, progress tracking, cancellation, and stale cleanup.
    """

    def __init__(self):
        self._downloads: Dict[str, DownloadProgress] = {}
        self._lock = threading.Lock()
        self._tasks: Dict[str, asyncio.Task] = {}
        self._cleanup_ttl = 300
        self._callbacks: Dict[str, list] = {}
        self._stats = DownloadStats()

    def get_progress(self, model_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._downloads.get(model_id)
            return entry.to_dict() if entry else None

    def list_downloads(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {mid: entry.to_dict() for mid, entry in self._downloads.items()}

    def is_downloading(self, model_id: str) -> bool:
        with self._lock:
            entry = self._downloads.get(model_id)
            return entry is not None and entry.status in (
                DownloadStatus.QUEUED,
                DownloadStatus.DOWNLOADING,
                DownloadStatus.PAUSED,
            )

    def is_cached(self, model_id: str) -> bool:
        """Whether the resource is fully cached on disk (survives restart)."""
        return get_backend().is_cached(model_id)

    def cancel(self, model_id: str) -> bool:
        with self._lock:
            entry = self._downloads.get(model_id)
            if entry and entry.status in (DownloadStatus.QUEUED, DownloadStatus.DOWNLOADING):
                entry.status = DownloadStatus.CANCELLED
                task = self._tasks.pop(model_id, None)
                if task and not task.done():
                    task.cancel()
                get_backend().on_cancel(model_id)
                return True
            return False

    def pause(self, model_id: str) -> bool:
        """Pause an in-progress download.

        The download can later be resumed with ``resume()``.
        Returns True if the download was paused, False if not found or not pausable.
        """
        paused = False
        with self._lock:
            entry = self._downloads.get(model_id)
            if entry and entry.status == DownloadStatus.DOWNLOADING:
                entry.status = DownloadStatus.PAUSED
                paused = True
        if paused:
            self._notify_callbacks(model_id)
        return paused

    def resume(self, model_id: str) -> bool:
        """Resume a paused download.

        Re-queues the download so it can continue from where it left off.
        Returns True if the download was resumed, False if not found or not resumable.
        """
        resumed = False
        with self._lock:
            entry = self._downloads.get(model_id)
            if entry and entry.status == DownloadStatus.PAUSED:
                entry.status = DownloadStatus.QUEUED
                resumed = True
        if resumed:
            self._notify_callbacks(model_id)
        return resumed

    def is_paused(self, model_id: str) -> bool:
        """Check if a download is paused."""
        with self._lock:
            entry = self._downloads.get(model_id)
            return entry is not None and entry.status == DownloadStatus.PAUSED

    def verify(self, model_id: str) -> Dict[str, Any]:
        """Verify integrity of a cached resource.

        Checks that all expected files exist, have correct sizes,
        and match checksums where available.

        Returns dict with:
            - valid: bool
            - files_checked: int
            - files_valid: int
            - errors: list of error strings
        """
        backend = get_backend()
        return backend.verify(model_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get cumulative download statistics."""
        with self._lock:
            return self._stats.to_dict()

    def _record_download_complete(self, model_id: str, bytes_downloaded: int, elapsed: float, speed: float) -> None:
        """Record statistics for a completed download."""
        with self._lock:
            self._stats.total_downloads += 1
            self._stats.completed_downloads += 1
            self._stats.total_bytes_downloaded += bytes_downloaded
            self._stats.total_download_time += elapsed
            if self._stats.total_download_time > 0:
                self._stats.average_speed = (
                    self._stats.total_bytes_downloaded / self._stats.total_download_time
                )
            if speed > self._stats.peak_speed:
                self._stats.peak_speed = speed

    def _record_download_failed(self, model_id: str) -> None:
        """Record statistics for a failed download."""
        with self._lock:
            self._stats.total_downloads += 1
            self._stats.failed_downloads += 1

    def _record_download_cancelled(self, model_id: str) -> None:
        """Record statistics for a cancelled download."""
        with self._lock:
            self._stats.total_downloads += 1
            self._stats.cancelled_downloads += 1

    def _set_progress(self, model_id: str, **kwargs) -> None:
        with self._lock:
            if model_id not in self._downloads:
                self._downloads[model_id] = DownloadProgress(
                    model_id=model_id,
                    status=DownloadStatus.QUEUED,
                )
            entry = self._downloads[model_id]
            for key, value in kwargs.items():
                if hasattr(entry, key):
                    setattr(entry, key, value)

    def _notify_callbacks(self, model_id: str) -> None:
        with self._lock:
            for cb in self._callbacks.get(model_id, []):
                try:
                    cb(self._downloads[model_id].to_dict())
                except Exception as e:
                    logger.warning("download_manager: callback failed", extra={
                        "model_id": model_id, "error": str(e),
                    })

    def on_progress(self, model_id: str, callback: Callable) -> None:
        with self._lock:
            self._callbacks.setdefault(model_id, []).append(callback)

    async def download(
        self,
        model_id: str,
        total_bytes_hint: int = 0,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Download a resource via the configured backend.

        Registers with CancelManager so downloads appear in /operations.
        Retries up to ``max_retries`` times on failure with exponential backoff.
        """
        backend = get_backend()

        if backend.is_cached(model_id):
            return {"status": "already_cached", "model_id": model_id}

        backend.prepare_download(model_id)

        if self.is_downloading(model_id):
            return {"status": "already_downloading", "model_id": model_id}

        total_est = backend.estimate_total(model_id) or total_bytes_hint

        self._set_progress(
            model_id,
            status=DownloadStatus.QUEUED,
            total_bytes=total_est,
            started_at=time.time(),
        )
        self._notify_callbacks(model_id)

        from domains.infrastructure.cancel_manager import get_cancel_manager, OpType
        mgr = get_cancel_manager()
        cancel_event = threading.Event()
        op_id = mgr.register(
            op_type=OpType.DOWNLOAD,
            label=f"download:{model_id}",
            cancel_fn=lambda: cancel_event.set(),
        )
        mgr.start(op_id)

        task = asyncio.create_task(
            self._download_worker(model_id, total_est, cancel_event, max_retries)
        )
        self._tasks[model_id] = task

        try:
            result = await task
            if result.get("status") == "complete":
                mgr.finish(op_id)
            elif result.get("status") == "cancelled":
                mgr.finish(op_id, "cancelled")
            else:
                mgr.finish(op_id, result.get("error", "unknown"))
            return result
        except asyncio.CancelledError:
            self._set_progress(model_id, status=DownloadStatus.CANCELLED)
            mgr.finish(op_id, "cancelled")
            try:
                from domains.infrastructure.event_buffer import get_event_buffer
                get_event_buffer().record("DOWNLOAD", f"{model_id} cancelled")
            except Exception as exc:
                logger.debug("Failed to record download cancel event: %s", exc)
            return {"status": "cancelled", "model_id": model_id}
        except Exception as e:
            self._set_progress(
                model_id,
                status=DownloadStatus.FAILED,
                error=str(e),
            )
            self._notify_callbacks(model_id)
            mgr.finish(op_id, str(e))
            try:
                from domains.infrastructure.event_buffer import get_event_buffer
                get_event_buffer().record("ERROR", f"download {model_id} failed: {str(e)[:40]}")
            except Exception as exc:
                logger.debug("Failed to record download error event: %s", exc)
            return {"status": "failed", "model_id": model_id, "error": str(e)}

    async def download_with(
        self,
        model_id: str,
        backend,
        total_bytes_hint: int = 0,
    ) -> Dict[str, Any]:
        """Download a resource using a specific backend.

        Unlike ``download()`` which uses the global backend, this method
        uses the provided backend directly.  Useful for external downloads
        or when switching backends per-request.
        """
        if backend.is_cached(model_id):
            return {"status": "already_cached", "model_id": model_id}

        backend.prepare_download(model_id)

        if self.is_downloading(model_id):
            return {"status": "already_downloading", "model_id": model_id}

        total_est = backend.estimate_total(model_id) or total_bytes_hint

        self._set_progress(
            model_id,
            status=DownloadStatus.QUEUED,
            total_bytes=total_est,
            started_at=time.time(),
        )
        self._notify_callbacks(model_id)

        from domains.infrastructure.cancel_manager import get_cancel_manager, OpType
        mgr = get_cancel_manager()
        cancel_event = threading.Event()
        op_id = mgr.register(
            op_type=OpType.DOWNLOAD,
            label=f"download:{model_id}",
            cancel_fn=lambda: cancel_event.set(),
        )
        mgr.start(op_id)

        task = asyncio.create_task(
            self._download_worker_with(model_id, backend, total_est, cancel_event)
        )
        self._tasks[model_id] = task

        try:
            result = await task
            if result.get("status") == "complete":
                mgr.finish(op_id)
            elif result.get("status") == "cancelled":
                mgr.finish(op_id, "cancelled")
            else:
                mgr.finish(op_id, result.get("error", "unknown"))
            return result
        except asyncio.CancelledError:
            self._set_progress(model_id, status=DownloadStatus.CANCELLED)
            mgr.finish(op_id, "cancelled")
            return {"status": "cancelled", "model_id": model_id}
        except Exception as e:
            self._set_progress(
                model_id,
                status=DownloadStatus.FAILED,
                error=str(e),
            )
            self._notify_callbacks(model_id)
            mgr.finish(op_id, str(e))
            return {"status": "failed", "model_id": model_id, "error": str(e)}

    async def _download_worker_with(
        self,
        model_id: str,
        backend,
        total_bytes_hint: int,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Run a download using a specific backend."""
        with self._lock:
            entry = self._downloads.get(model_id)
            if entry and entry.status == DownloadStatus.CANCELLED:
                return {"status": "cancelled", "model_id": model_id}
        self._set_progress(model_id, status=DownloadStatus.DOWNLOADING)
        self._notify_callbacks(model_id)
        start_time = time.time()

        def _progress_cb(mid: str, downloaded: int, total: int, speed: float):
            try:
                with self._lock:
                    cur = self._downloads.get(mid)
                    if cur and cur.status == DownloadStatus.CANCELLED:
                        return
                pct = (downloaded / total * 100) if total > 0 else 0
                self._set_progress(
                    mid,
                    bytes_downloaded=downloaded,
                    total_bytes=total,
                    speed_bytes_per_sec=speed,
                    percentage=pct,
                    status=DownloadStatus.DOWNLOADING,
                )
                self._notify_callbacks(mid)
            except Exception as e:
                logger.warning("download_manager: progress callback failed", extra={
                    "model_id": mid, "error": str(e),
                })

        def _file_cb(mid: str, fpath: str):
            try:
                self._set_progress(mid, current_file=fpath)
                self._notify_callbacks(mid)
            except Exception as e:
                logger.warning("download_manager: file callback failed", extra={
                    "model_id": mid, "file": fpath, "error": str(e),
                })

        def _do_download():
            def _cancel_check():
                if cancel_event and cancel_event.is_set():
                    raise InterruptedError("Download cancelled")

            def _progress_cb_inner(mid, downloaded, total, speed):
                _cancel_check()
                _progress_cb(mid, downloaded, total, speed)

            if backend.supports_compression(model_id):
                backend.download_compressed(model_id, _progress_cb_inner, _file_cb)
            else:
                backend.download(model_id, _progress_cb_inner, _file_cb)

        await asyncio.to_thread(_do_download)

        with self._lock:
            entry = self._downloads.get(model_id)
            if entry and entry.status == DownloadStatus.CANCELLED:
                return {"status": "cancelled", "model_id": model_id}

        elapsed = time.time() - start_time
        cache_dir = backend.get_cache_dir(model_id)
        self._set_progress(
            model_id,
            status=DownloadStatus.COMPLETE,
            completed_at=time.time(),
            percentage=100.0,
        )
        self._notify_callbacks(model_id)

        # Record statistics
        with self._lock:
            entry = self._downloads.get(model_id)
            bytes_downloaded = entry.bytes_downloaded if entry else 0
            speed = entry.speed_bytes_per_sec if entry else 0
        self._record_download_complete(model_id, bytes_downloaded, elapsed, speed)

        logger.info("Downloaded %s in %.1fs → %s", model_id, elapsed, cache_dir)
        return {
            "status": "complete",
            "model_id": model_id,
            "cache_dir": cache_dir,
            "elapsed_seconds": round(elapsed, 1),
        }

    async def _download_worker(
        self,
        model_id: str,
        total_bytes_hint: int,
        cancel_event: Optional[threading.Event] = None,
        max_retries: int = 3,
    ):
        """Run the backend download in a thread executor, updating progress.

        Retries up to ``max_retries`` times on failure with exponential backoff.
        """
        with self._lock:
            entry = self._downloads.get(model_id)
            if entry and entry.status == DownloadStatus.CANCELLED:
                return {"status": "cancelled", "model_id": model_id}
        self._set_progress(model_id, status=DownloadStatus.DOWNLOADING)
        self._notify_callbacks(model_id)
        start_time = time.time()

        try:
            from domains.infrastructure.event_buffer import get_event_buffer
            size_str = ""
            if total_bytes_hint > 0:
                if total_bytes_hint > 1e9:
                    size_str = f" ({total_bytes_hint / 1e9:.1f}GB)"
                elif total_bytes_hint > 1e6:
                    size_str = f" ({total_bytes_hint / 1e6:.0f}MB)"
            get_event_buffer().record("DOWNLOAD", f"{model_id} started{size_str}")
        except Exception as exc:
            logger.debug("Failed to record download start event: %s", exc)

        def _progress_cb(mid: str, downloaded: int, total: int, speed: float):
            try:
                with self._lock:
                    cur = self._downloads.get(mid)
                    if cur and cur.status == DownloadStatus.CANCELLED:
                        return
                pct = (downloaded / total * 100) if total > 0 else 0
                self._set_progress(
                    mid,
                    bytes_downloaded=downloaded,
                    total_bytes=total,
                    speed_bytes_per_sec=speed,
                    percentage=pct,
                    status=DownloadStatus.DOWNLOADING,
                )
                self._notify_callbacks(mid)
            except Exception as e:
                logger.warning("download_manager: progress callback failed", extra={
                    "model_id": mid, "error": str(e),
                })

        def _file_cb(mid: str, fpath: str):
            try:
                self._set_progress(mid, current_file=fpath)
                self._notify_callbacks(mid)
            except Exception as e:
                logger.warning("download_manager: file callback failed", extra={
                    "model_id": mid, "file": fpath, "error": str(e),
                })

        last_error = None
        for attempt in range(max_retries + 1):
            # Check cancellation before each attempt
            with self._lock:
                entry = self._downloads.get(model_id)
                if entry and entry.status == DownloadStatus.CANCELLED:
                    return {"status": "cancelled", "model_id": model_id}

            def _do_download():
                backend = get_backend()

                def _cancel_check():
                    if cancel_event and cancel_event.is_set():
                        raise InterruptedError("Download cancelled")

                def _progress_cb_inner(mid, downloaded, total, speed):
                    _cancel_check()
                    _progress_cb(mid, downloaded, total, speed)

                if backend.supports_compression(model_id):
                    backend.download_compressed(model_id, _progress_cb_inner, _file_cb)
                else:
                    backend.download(model_id, _progress_cb_inner, _file_cb)

            try:
                await asyncio.to_thread(_do_download)
                # Success — break out of retry loop
                last_error = None
                break
            except InterruptedError:
                return {"status": "cancelled", "model_id": model_id}
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        "Download %s failed (attempt %d/%d): %s — retrying in %ds",
                        model_id, attempt + 1, max_retries + 1, e, wait,
                    )
                    self._set_progress(
                        model_id,
                        status=DownloadStatus.DOWNLOADING,
                        error=f"Retry {attempt + 1}/{max_retries}: {e}",
                    )
                    self._notify_callbacks(model_id)
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        "Download %s failed after %d attempts: %s",
                        model_id, max_retries + 1, e,
                    )

        if last_error is not None:
            self._record_download_failed(model_id)
            return {"status": "failed", "model_id": model_id, "error": str(last_error)}

        with self._lock:
            entry = self._downloads.get(model_id)
            if entry and entry.status == DownloadStatus.CANCELLED:
                self._record_download_cancelled(model_id)
                return {"status": "cancelled", "model_id": model_id}

        elapsed = time.time() - start_time
        cache_dir = get_backend().get_cache_dir(model_id)
        self._set_progress(
            model_id,
            status=DownloadStatus.COMPLETE,
            completed_at=time.time(),
            percentage=100.0,
        )
        self._notify_callbacks(model_id)

        # Record statistics
        with self._lock:
            entry = self._downloads.get(model_id)
            bytes_downloaded = entry.bytes_downloaded if entry else 0
            speed = entry.speed_bytes_per_sec if entry else 0
        self._record_download_complete(model_id, bytes_downloaded, elapsed, speed)

        try:
            from domains.infrastructure.event_buffer import get_event_buffer
            get_event_buffer().record("DOWNLOAD", f"{model_id} complete ({elapsed:.1f}s)")
        except Exception:
            pass

        logger.info("Downloaded %s in %.1fs → %s", model_id, elapsed, cache_dir,
            extra={
                "op": "download.complete",
                "dur_ms": int(elapsed * 1000),
                "ok": True,
                "download": {"resource": model_id, "elapsed_s": round(elapsed, 1)},
            })
        return {
            "status": "complete",
            "model_id": model_id,
            "cache_dir": cache_dir,
            "elapsed_seconds": round(elapsed, 1),
        }

    def cleanup_stale(self, max_age: float = 300) -> None:
        now = time.time()
        with self._lock:
            stale = []
            for mid, entry in self._downloads.items():
                end_time = entry.completed_at or entry.started_at
                if entry.status in (
                    DownloadStatus.COMPLETE,
                    DownloadStatus.FAILED,
                    DownloadStatus.CANCELLED,
                ) and (now - end_time) > max_age:
                    stale.append(mid)
            for mid in stale:
                del self._downloads[mid]
                self._tasks.pop(mid, None)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_download_manager: Optional[DownloadManager] = None
_download_manager_lock = threading.Lock()


def get_download_manager() -> DownloadManager:
    global _download_manager
    if _download_manager is None:
        with _download_manager_lock:
            if _download_manager is None:
                _download_manager = DownloadManager()
    return _download_manager


def reset_download_manager() -> None:
    """Reset the singleton (for testing)."""
    global _download_manager
    with _download_manager_lock:
        _download_manager = None
