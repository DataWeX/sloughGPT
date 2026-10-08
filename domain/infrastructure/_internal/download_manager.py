"""
Download Manager — simplified orchestration using downcraft as foundation.

Delegates core download logic (resume, compression, integrity) to downcraft.
This module handles only scheduling, progress tracking, cancellation, and stale cleanup.

downcraft provides:
- Cross-session resume via HTTP Range headers
- LZ4 compression support (optional)
- SHA-256 integrity verification
- Atomic file writes (no corrupt files on crash)

This module adds:
- Async scheduling with CancelManager integration
- Progress tracking with callbacks
- Statistics collection
- Stale download cleanup
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .download_backend import DownloadBackend

logger = logging.getLogger("slo.infrastructure.download_manager")

# ---------------------------------------------------------------------------
# Try to import downcraft, fallback to None
# ---------------------------------------------------------------------------

_downcraft = None
try:
    import downcraft

    _downcraft = downcraft
except ImportError:
    logger.debug("downcraft not available, downloads will use fallback")


# ---------------------------------------------------------------------------
# Status / Progress
# ---------------------------------------------------------------------------


class DownloadStatus(StrEnum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    PAUSED = "paused"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DownloadProgress:
    resource_id: str
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
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

    def to_dict(self) -> dict[str, Any]:
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
# Module-level convenience
# ---------------------------------------------------------------------------


def is_download_complete(resource_id: str, deep_check: bool = False) -> bool:
    """Check if a model is fully cached on disk."""
    if _downcraft is None:
        return False
    # Check if the file exists and is complete
    dest = Path(os.environ.get("SLO_CACHE_DIR", Path.home() / ".cache" / "sloughgpt")) / resource_id
    return dest.exists() and dest.stat().st_size > 0


def cleanup_incomplete(resource_id: str) -> bool:
    """Remove an incomplete/partial download."""
    if _downcraft is None:
        return False
    # Remove .sgpart file if it exists
    dest = Path(os.environ.get("SLO_CACHE_DIR", Path.home() / ".cache" / "sloughgpt")) / resource_id
    part = dest.with_suffix(dest.suffix + ".sgpart")
    if part.exists():
        part.unlink()
        return True
    return False


def list_incomplete_models() -> list[str]:
    """Scan cache and return resource IDs with incomplete downloads."""
    if _downcraft is None:
        return []
    # List .sgpart files
    cache_dir = Path(os.environ.get("SLO_CACHE_DIR", Path.home() / ".cache" / "sloughgpt"))
    if not cache_dir.exists():
        return []
    return [f.stem for f in cache_dir.glob("*.sgpart")]


# ---------------------------------------------------------------------------
# DownloadManager — simplified with downcraft
# ---------------------------------------------------------------------------


class DownloadManager:
    """Download manager singleton.

    Uses downcraft for core download logic (resume, compression, integrity).
    Handles scheduling, progress tracking, cancellation, and stale cleanup.
    """

    def __init__(self):
        self._downloads: dict[str, DownloadProgress] = {}
        self._lock = threading.Lock()
        self._tasks: dict[str, asyncio.Task] = {}
        self._cleanup_ttl = 300
        self._callbacks: dict[str, list] = {}
        self._stats = DownloadStats()

    def get_progress(self, resource_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._downloads.get(resource_id)
            return entry.to_dict() if entry else None

    def list_downloads(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {mid: entry.to_dict() for mid, entry in self._downloads.items()}

    def is_downloading(self, resource_id: str) -> bool:
        with self._lock:
            entry = self._downloads.get(resource_id)
            return entry is not None and entry.status in (
                DownloadStatus.QUEUED,
                DownloadStatus.DOWNLOADING,
                DownloadStatus.PAUSED,
            )

    def is_cached(self, resource_id: str) -> bool:
        """Whether the resource is fully cached on disk (survives restart)."""
        return is_download_complete(resource_id)

    def cancel(self, resource_id: str) -> bool:
        with self._lock:
            entry = self._downloads.get(resource_id)
            if entry and entry.status in (DownloadStatus.QUEUED, DownloadStatus.DOWNLOADING):
                entry.status = DownloadStatus.CANCELLED
                task = self._tasks.pop(resource_id, None)
                if task and not task.done():
                    task.cancel()
                return True
            return False

    def pause(self, resource_id: str) -> bool:
        """Pause an in-progress download.

        The download can later be resumed with ``resume()``.
        Returns True if the download was paused, False if not found or not pausable.
        """
        paused = False
        with self._lock:
            entry = self._downloads.get(resource_id)
            if entry and entry.status == DownloadStatus.DOWNLOADING:
                entry.status = DownloadStatus.PAUSED
                paused = True
        if paused:
            self._notify_callbacks(resource_id)
        return paused

    def resume(self, resource_id: str) -> bool:
        """Resume a paused download.

        Re-queues the download so it can continue from where it left off.
        Returns True if the download was resumed, False if not found or not resumable.
        """
        resumed = False
        with self._lock:
            entry = self._downloads.get(resource_id)
            if entry and entry.status == DownloadStatus.PAUSED:
                entry.status = DownloadStatus.QUEUED
                resumed = True
        if resumed:
            self._notify_callbacks(resource_id)
        return resumed

    def is_paused(self, resource_id: str) -> bool:
        """Check if a download is paused."""
        with self._lock:
            entry = self._downloads.get(resource_id)
            return entry is not None and entry.status == DownloadStatus.PAUSED

    def verify(self, resource_id: str) -> dict[str, Any]:
        """Verify integrity of a cached resource."""
        dest = (
            Path(os.environ.get("SLO_CACHE_DIR", Path.home() / ".cache" / "sloughgpt")) / resource_id
        )
        valid = dest.exists() and dest.stat().st_size > 0
        return {
            "valid": valid,
            "files_checked": 1 if valid else 0,
            "files_valid": 1 if valid else 0,
            "errors": [] if valid else ["File not found or empty"],
        }

    def get_stats(self) -> dict[str, Any]:
        """Get cumulative download statistics."""
        with self._lock:
            return self._stats.to_dict()

    def _record_download_complete(
        self, resource_id: str, bytes_downloaded: int, elapsed: float, speed: float
    ) -> None:
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

    def _record_download_failed(self, resource_id: str) -> None:
        """Record statistics for a failed download."""
        with self._lock:
            self._stats.total_downloads += 1
            self._stats.failed_downloads += 1

    def _record_download_cancelled(self, resource_id: str) -> None:
        """Record statistics for a cancelled download."""
        with self._lock:
            self._stats.total_downloads += 1
            self._stats.cancelled_downloads += 1

    def _set_progress(self, resource_id: str, **kwargs) -> None:
        with self._lock:
            if resource_id not in self._downloads:
                self._downloads[resource_id] = DownloadProgress(
                    resource_id=resource_id,
                    status=DownloadStatus.QUEUED,
                )
            entry = self._downloads[resource_id]
            for key, value in kwargs.items():
                if hasattr(entry, key):
                    setattr(entry, key, value)

    def _notify_callbacks(self, resource_id: str) -> None:
        with self._lock:
            for cb in self._callbacks.get(resource_id, []):
                try:
                    cb(self._downloads[resource_id].to_dict())
                except Exception as e:
                    logger.warning(
                        "download_manager: callback failed",
                        extra={
                            "resource_id": resource_id,
                            "error": str(e),
                        },
                    )

    def on_progress(self, resource_id: str, callback: Callable) -> None:
        with self._lock:
            self._callbacks.setdefault(resource_id, []).append(callback)

    async def download(
        self,
        resource_id: str,
        url: str,
        dest: str | Path = "",
        total_bytes_hint: int = 0,
        checksum: str = "",
        compressed: bool = False,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Download a resource using downcraft.

        Args:
            resource_id: Unique identifier for the download.
            url: HTTP/HTTPS URL to download from.
            dest: Local destination path (default: ~/.cache/sloughgpt/{resource_id}).
            total_bytes_hint: Expected total bytes (0 = auto-detect).
            checksum: SHA-256 hex string to verify after download.
            compressed: If True, expect LZ4-compressed response.
            max_retries: Number of retry attempts on failure.

        Returns:
            Dict with status, resource_id, elapsed_seconds, etc.
        """
        if _downcraft is None:
            return {"status": "failed", "resource_id": resource_id, "error": "downcraft not installed"}

        if is_download_complete(resource_id):
            return {"status": "already_cached", "resource_id": resource_id}

        if self.is_downloading(resource_id):
            return {"status": "already_downloading", "resource_id": resource_id}

        # Resolve destination
        if not dest:
            cache_dir = Path(os.environ.get("SLO_CACHE_DIR", Path.home() / ".cache" / "sloughgpt"))
            dest = cache_dir / resource_id
        else:
            dest = Path(dest)

        total_est = total_bytes_hint

        self._set_progress(
            resource_id,
            status=DownloadStatus.QUEUED,
            total_bytes=total_est,
            started_at=time.time(),
        )
        self._notify_callbacks(resource_id)

        from .cancel_manager import OpType, get_cancel_manager

        mgr = get_cancel_manager()
        cancel_event = threading.Event()
        op_id = mgr.register(
            op_type=OpType.DOWNLOAD,
            label=f"download:{resource_id}",
            cancel_fn=lambda: cancel_event.set(),
        )
        mgr.start(op_id)

        task = asyncio.create_task(
            self._download_worker(
                resource_id, url, dest, total_est, checksum, compressed, cancel_event, max_retries
            )
        )
        self._tasks[resource_id] = task

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
            self._set_progress(resource_id, status=DownloadStatus.CANCELLED)
            mgr.finish(op_id, "cancelled")
            try:
                from .event_buffer import get_event_buffer

                get_event_buffer().record("DOWNLOAD", f"{resource_id} cancelled")
            except Exception as exc:
                logger.debug("Failed to record download cancel event: %s", exc)
            return {"status": "cancelled", "resource_id": resource_id}
        except Exception as e:
            self._set_progress(
                resource_id,
                status=DownloadStatus.FAILED,
                error=str(e),
            )
            self._notify_callbacks(resource_id)
            mgr.finish(op_id, str(e))
            try:
                from .event_buffer import get_event_buffer

                get_event_buffer().record("ERROR", f"download {resource_id} failed: {str(e)[:40]}")
            except Exception as exc:
                logger.debug("Failed to record download error event: %s", exc)
            return {"status": "failed", "resource_id": resource_id, "error": str(e)}

    async def download_model(
        self,
        model_id: str,
        url: str = "",
        dest: str | Path = "",
        total_bytes_hint: int = 0,
    ) -> dict[str, Any]:
        """Download a model by id — direct URL, or the Hub when omitted.

        This is the seam the API router and the CLI call.  Two paths:

        * ``url`` given → the generic single-resource engine (``download()``),
          which brings resume, integrity verification, pause/resume and
          cancellation.  The URL travels as a keyword argument: passing it
          positionally is what turned ``total_bytes_hint`` into
          ``url=0`` (→ ``Invalid URL '0'``) after ``download()`` grew its
          ``url`` parameter on 2026-09-16.
        * no ``url`` → the model is identified by its Hub id, so
          ``HFDownloadBackend`` (``download_hf_model``) resolves the file
          list and cache layout — one call into ``download_with()``, the
          backend-generic seam.  Progress is mirrored into the registry so
          ``GET /models/downloads`` surfaces it like any other download.

        Args:
            model_id: Model id or download id (``"gpt2"``, ``"org/name"``).
                The facade takes model ids — its callers are model-domain
                (router, CLI).  Internally it hands the id to the generic
                surface as ``resource_id``; one vocabulary below this line.
            url: Optional direct HTTP(S) URL — wins over the model-id path.
            dest: Optional destination path for the generic path.
            total_bytes_hint: Expected total bytes (0 = auto-detect).

        Returns:
            The generic result envelope: ``status`` (``complete`` /
            ``already_cached`` / ``failed`` / ``cancelled`` /
            ``already_downloading``) and ``resource_id``.

        Note:
            Cancelling flips the entry to cancelled, but an in-flight Hub
            transfer cannot be interrupted (the Hub path has no cancel
            hook): the entry stays cancelled and the finished files remain
            cached for the next attempt.
        """
        if url:
            return await self.download(
                model_id, url=url, dest=dest, total_bytes_hint=total_bytes_hint
            )

        from .hf_hub import HFDownloadBackend

        return await self.download_with(
            model_id, HFDownloadBackend(), total_bytes_hint=total_bytes_hint
        )

    async def download_with(
        self,
        resource_id: str,
        backend: DownloadBackend,
        total_bytes_hint: int = 0,
    ) -> dict[str, Any]:
        """Download a resource through an arbitrary ``DownloadBackend``.

        The backend-generic twin of ``download()``: the backend owns the
        source-specific logic (manifests, file lists, compression), this
        method owns scheduling, progress mirroring, cancellation and the
        terminal registry states.  Two callers share it:

        * ``download_model()`` with no URL passes ``HFDownloadBackend()`` —
          this is where the Hub path has always lived, just extracted;
        * the models router's ``_run_external_download`` passes an
          ``ExternalDownloadBackend`` for a registered peer server.  That
          call site already existed but the method did not: the
          AttributeError died in a bare ``except`` while the route answered
          ``download_started``, so ``POST /models/external/download`` was a
          silent no-op.

        Backend result vocabularies differ — HF says ``complete`` /
        ``already_cached``, the external backend says ``completed`` /
        ``error`` — so statuses are normalized here: ``completed`` maps to
        ``complete``, and any *returned* error status flips the registry
        entry to ``failed`` (an error result must never be recorded as a
        completed download).

        Args:
            resource_id: Model id or download id — the registry key.
            backend: Any ``DownloadBackend``; only ``download()`` is called.
            total_bytes_hint: Expected total bytes before the first
                progress callback (0 = auto-detect).

        Returns:
            Dict with ``status`` (``complete`` / ``failed`` / ``cancelled`` /
            ``already_cached`` / ``already_downloading``) and ``resource_id``.

        Note:
            Cancelling flips the entry to cancelled once the backend
            returns, but the transfer itself is only interruptible if the
            backend's ``download()`` cooperates — the Hub and external
            paths have no cancel hook (same caveat as ``download_model``).
        """
        if self.is_downloading(resource_id):
            return {"status": "already_downloading", "resource_id": resource_id}

        started = time.time()

        def _on_progress(mid: str, done: int, total: int, speed: float) -> None:
            pct = (done / total * 100) if total > 0 else 0.0
            self._set_progress(
                mid,
                status=DownloadStatus.DOWNLOADING,
                bytes_downloaded=done,
                total_bytes=total,
                percentage=pct,
                speed_bytes_per_sec=speed,
            )
            self._notify_callbacks(mid)

        def _on_file_complete(mid: str, file_path: str) -> None:
            # ExternalDownloadBackend invokes this unconditionally per file
            # (None would TypeError), and HFDownloadBackend calls it when
            # given — so every path through this seam gets a callable.
            logger.debug("File complete for %s: %s", mid, file_path)

        self._set_progress(
            resource_id,
            status=DownloadStatus.QUEUED,
            started_at=started,
            total_bytes=total_bytes_hint,
        )
        self._notify_callbacks(resource_id)

        try:
            result = await asyncio.to_thread(
                backend.download,
                resource_id,
                on_progress=_on_progress,
                on_file_complete=_on_file_complete,
            )
        except Exception as exc:
            self._set_progress(resource_id, status=DownloadStatus.FAILED, error=str(exc))
            self._notify_callbacks(resource_id)
            self._record_download_failed(resource_id)
            return {"status": "failed", "resource_id": resource_id, "error": str(exc)}

        with self._lock:
            entry = self._downloads.get(resource_id)
            cancelled = entry is not None and entry.status == DownloadStatus.CANCELLED
        if cancelled:
            self._record_download_cancelled(resource_id)
            return {"status": "cancelled", "resource_id": resource_id}

        raw_status = str(result.get("status", "complete"))
        if raw_status in ("error", "failed"):
            error = str(result.get("error") or raw_status)
            self._set_progress(resource_id, status=DownloadStatus.FAILED, error=error)
            self._notify_callbacks(resource_id)
            self._record_download_failed(resource_id)
            return {"status": "failed", "resource_id": resource_id, "error": error}

        elapsed = time.time() - started
        total_bytes = int(result.get("total_bytes") or 0)
        progress: dict[str, Any] = {
            "status": DownloadStatus.COMPLETE,
            "percentage": 100.0,
            "completed_at": time.time(),
        }
        if total_bytes:
            progress["total_bytes"] = total_bytes
        self._set_progress(resource_id, **progress)
        self._notify_callbacks(resource_id)

        with self._lock:
            entry = self._downloads.get(resource_id)
            bytes_done = entry.bytes_downloaded if entry else 0
            speed = entry.speed_bytes_per_sec if entry else 0
        self._record_download_complete(resource_id, bytes_done, elapsed, speed)

        return {
            "status": "complete" if raw_status == "completed" else raw_status,
            "resource_id": resource_id,
            "cache_dir": result.get("cache_dir", ""),
            "elapsed_seconds": elapsed,
        }

    async def _download_worker(
        self,
        resource_id: str,
        url: str,
        dest: Path,
        total_bytes_hint: int,
        checksum: str,
        compressed: bool,
        cancel_event: threading.Event | None = None,
        max_retries: int = 3,
    ):
        """Run the downcraft download in a thread executor, updating progress."""
        with self._lock:
            entry = self._downloads.get(resource_id)
            if entry and entry.status == DownloadStatus.CANCELLED:
                return {"status": "cancelled", "resource_id": resource_id}
        self._set_progress(resource_id, status=DownloadStatus.DOWNLOADING)
        self._notify_callbacks(resource_id)
        start_time = time.time()

        try:
            from .event_buffer import get_event_buffer

            size_str = ""
            if total_bytes_hint > 0:
                if total_bytes_hint > 1e9:
                    size_str = f" ({total_bytes_hint / 1e9:.1f}GB)"
                elif total_bytes_hint > 1e6:
                    size_str = f" ({total_bytes_hint / 1e6:.0f}MB)"
            get_event_buffer().record("DOWNLOAD", f"{resource_id} started{size_str}")
        except Exception as exc:
            logger.debug("Failed to record download start event: %s", exc)

        def _progress_cb(bytes_done: int, total: int, speed: float):
            try:
                with self._lock:
                    cur = self._downloads.get(resource_id)
                    if cur and cur.status == DownloadStatus.CANCELLED:
                        return
                pct = (bytes_done / total * 100) if total > 0 else 0
                self._set_progress(
                    resource_id,
                    bytes_downloaded=bytes_done,
                    total_bytes=total,
                    speed_bytes_per_sec=speed,
                    percentage=pct,
                    status=DownloadStatus.DOWNLOADING,
                )
                self._notify_callbacks(resource_id)
            except Exception as e:
                logger.warning(
                    "download_manager: progress callback failed",
                    extra={
                        "resource_id": resource_id,
                        "error": str(e),
                    },
                )

        last_error = None
        for attempt in range(max_retries + 1):
            # Check cancellation before each attempt
            with self._lock:
                entry = self._downloads.get(resource_id)
                if entry and entry.status == DownloadStatus.CANCELLED:
                    return {"status": "cancelled", "resource_id": resource_id}

            def _do_download():
                def _cancel_check():
                    if cancel_event and cancel_event.is_set():
                        raise InterruptedError("Download cancelled")

                def _progress_inner(bytes_done, total, speed):
                    _cancel_check()
                    _progress_cb(bytes_done, total, speed)

                # Use downcraft for the actual download
                result = downcraft.download(
                    url=url,
                    dest=dest,
                    expected_size=total_bytes_hint,
                    checksum=checksum,
                    on_progress=_progress_inner,
                    compressed=compressed,
                )
                return result

            try:
                await asyncio.to_thread(_do_download)
                # Success — break out of retry loop
                last_error = None
                break
            except InterruptedError:
                return {"status": "cancelled", "resource_id": resource_id}
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2**attempt  # exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        "Download %s failed (attempt %d/%d): %s — retrying in %ds",
                        resource_id,
                        attempt + 1,
                        max_retries + 1,
                        e,
                        wait,
                    )
                    self._set_progress(
                        resource_id,
                        status=DownloadStatus.DOWNLOADING,
                        error=f"Retry {attempt + 1}/{max_retries}: {e}",
                    )
                    self._notify_callbacks(resource_id)
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        "Download %s failed after %d attempts: %s",
                        resource_id,
                        max_retries + 1,
                        e,
                    )

        if last_error is not None:
            # Terminal state: without this the entry stays "downloading"
            # forever — the UI card spins at 0% and the model stays locked
            # behind is_downloading(), so no retry can ever start.
            self._set_progress(
                resource_id,
                status=DownloadStatus.FAILED,
                error=str(last_error),
                completed_at=time.time(),
            )
            self._notify_callbacks(resource_id)
            self._record_download_failed(resource_id)
            return {"status": "failed", "resource_id": resource_id, "error": str(last_error)}

        with self._lock:
            entry = self._downloads.get(resource_id)
            if entry and entry.status == DownloadStatus.CANCELLED:
                self._record_download_cancelled(resource_id)
                return {"status": "cancelled", "resource_id": resource_id}

        elapsed = time.time() - start_time
        self._set_progress(
            resource_id,
            status=DownloadStatus.COMPLETE,
            completed_at=time.time(),
            percentage=100.0,
        )
        self._notify_callbacks(resource_id)

        # Record statistics
        with self._lock:
            entry = self._downloads.get(resource_id)
            bytes_downloaded = entry.bytes_downloaded if entry else 0
            speed = entry.speed_bytes_per_sec if entry else 0
        self._record_download_complete(resource_id, bytes_downloaded, elapsed, speed)

        try:
            from .event_buffer import get_event_buffer

            get_event_buffer().record("DOWNLOAD", f"{resource_id} complete ({elapsed:.1f}s)")
        except Exception:
            pass

        logger.info(
            "Downloaded %s in %.1fs → %s",
            resource_id,
            elapsed,
            dest,
            extra={
                "op": "download.complete",
                "dur_ms": int(elapsed * 1000),
                "ok": True,
                "download": {"resource": resource_id, "elapsed_s": round(elapsed, 1)},
            },
        )
        return {
            "status": "complete",
            "resource_id": resource_id,
            "dest": str(dest),
            "elapsed_seconds": round(elapsed, 1),
        }

    def cleanup_stale(self, max_age: float = 300) -> None:
        now = time.time()
        with self._lock:
            stale = []
            for mid, entry in self._downloads.items():
                end_time = entry.completed_at or entry.started_at
                if (
                    entry.status
                    in (
                        DownloadStatus.COMPLETE,
                        DownloadStatus.FAILED,
                        DownloadStatus.CANCELLED,
                    )
                    and (now - end_time) > max_age
                ):
                    stale.append(mid)
            for mid in stale:
                del self._downloads[mid]
                self._tasks.pop(mid, None)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_download_manager: DownloadManager | None = None
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
