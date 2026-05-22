"""
Model Download Manager — tracks HuggingFace download progress.

Provides a singleton `DownloadManager` that:
  - Starts async downloads with HF Hub progress callbacks
  - Exposes per-model status (queued, downloading, complete, failed)
  - Reports bytes_downloaded, total_bytes, speed, eta, percentage
  - Cleans up completed/failed entries after a TTL
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


class DownloadStatus(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
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


class DownloadManager:
    """
    Singleton that manages concurrent model downloads with progress tracking.

    Usage:
        mgr = get_download_manager()
        await mgr.download("gpt2")
        progress = mgr.get_progress("gpt2")
    """

    def __init__(self):
        self._downloads: Dict[str, DownloadProgress] = {}
        self._lock = threading.Lock()
        self._tasks: Dict[str, asyncio.Task] = {}
        self._cleanup_ttl = 300  # 5 minutes
        self._callbacks: Dict[str, list] = {}

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
            )

    def is_cached(self, model_id: str) -> bool:
        cache_dir = HF_CACHE / f"models--{model_id.replace('/', '--')}"
        if not cache_dir.exists():
            return False
        safetensors_files = list(cache_dir.rglob("*.safetensors"))
        return len(safetensors_files) > 0

    def cancel(self, model_id: str) -> bool:
        with self._lock:
            entry = self._downloads.get(model_id)
            if entry and entry.status in (DownloadStatus.QUEUED, DownloadStatus.DOWNLOADING):
                entry.status = DownloadStatus.CANCELLED
                task = self._tasks.pop(model_id, None)
                if task and not task.done():
                    task.cancel()
                return True
            return False

    def _set_progress(self, model_id: str, **kwargs):
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

    def _notify_callbacks(self, model_id: str):
        with self._lock:
            for cb in self._callbacks.get(model_id, []):
                try:
                    cb(self._downloads[model_id].to_dict())
                except Exception:
                    pass

    def on_progress(self, model_id: str, callback: Callable):
        with self._lock:
            self._callbacks.setdefault(model_id, []).append(callback)

    async def download(
        self,
        model_id: str,
        total_bytes_hint: int = 0,
    ) -> Dict[str, Any]:
        """
        Download a model from HuggingFace Hub with progress tracking.

        Args:
            model_id: HuggingFace model ID (e.g. "gpt2", "Qwen/Qwen2.5-0.5B-Instruct")
            total_bytes_hint: Optional known total size for progress calculation.

        Returns:
            Dict with status, cache_dir, and elapsed time.
        """
        if self.is_cached(model_id):
            return {"status": "already_cached", "model_id": model_id}

        if self.is_downloading(model_id):
            return {"status": "already_downloading", "model_id": model_id}

        self._set_progress(
            model_id,
            status=DownloadStatus.QUEUED,
            total_bytes=total_bytes_hint,
            started_at=time.time(),
        )
        self._notify_callbacks(model_id)

        loop = asyncio.get_event_loop()
        task = asyncio.create_task(self._download_worker(model_id, total_bytes_hint))
        self._tasks[model_id] = task

        try:
            result = await task
            return result
        except asyncio.CancelledError:
            self._set_progress(model_id, status=DownloadStatus.CANCELLED)
            return {"status": "cancelled", "model_id": model_id}
        except Exception as e:
            self._set_progress(
                model_id,
                status=DownloadStatus.FAILED,
                error=str(e),
            )
            self._notify_callbacks(model_id)
            return {"status": "failed", "model_id": model_id, "error": str(e)}

    async def _download_worker(self, model_id: str, total_bytes_hint: int):
        """Run the actual download in a thread executor with progress callbacks."""
        self._set_progress(model_id, status=DownloadStatus.DOWNLOADING)
        self._notify_callbacks(model_id)

        start_time = time.time()
        bytes_so_far = 0

        def _hf_progress_callback(progress: dict):
            nonlocal bytes_so_far, start_time
            # progress has keys: status, completed, total, name, context
            status = progress.get("status", "")
            completed = progress.get("completed", 0)
            total = progress.get("total", 0)
            filename = progress.get("name", "")

            now = time.time()
            elapsed = now - start_time

            if total > 0 and total_bytes_hint == 0:
                self._set_progress(model_id, total_bytes=total)

            if status == "complete":
                bytes_so_far += completed
            else:
                bytes_so_far = completed

            speed = bytes_so_far / elapsed if elapsed > 0 else 0
            remaining = total - bytes_so_far if total > 0 else 0
            eta = remaining / speed if speed > 0 else 0
            pct = (bytes_so_far / total * 100) if total > 0 else 0

            self._set_progress(
                model_id,
                bytes_downloaded=bytes_so_far,
                current_file=filename,
                speed_bytes_per_sec=speed,
                eta_seconds=eta,
                percentage=pct,
                status=DownloadStatus.DOWNLOADING,
            )
            self._notify_callbacks(model_id)

        def _do_download():
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model_id,
                cache_dir=str(HF_CACHE.parent),
                progress_callback=_hf_progress_callback,
            )

        await asyncio.to_thread(_do_download)

        elapsed = time.time() - start_time
        cache_dir = HF_CACHE / f"models--{model_id.replace('/', '--')}"
        self._set_progress(
            model_id,
            status=DownloadStatus.COMPLETE,
            completed_at=time.time(),
            percentage=100.0,
        )
        self._notify_callbacks(model_id)

        logger.info("Downloaded %s in %.1fs → %s", model_id, elapsed, cache_dir)
        return {
            "status": "complete",
            "model_id": model_id,
            "cache_dir": str(cache_dir),
            "elapsed_seconds": round(elapsed, 1),
        }

    def cleanup_stale(self, max_age: float = 300):
        """Remove completed/failed/cancelled entries older than max_age seconds."""
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


_download_manager: Optional[DownloadManager] = None


def get_download_manager() -> DownloadManager:
    global _download_manager
    if _download_manager is None:
        _download_manager = DownloadManager()
    return _download_manager
