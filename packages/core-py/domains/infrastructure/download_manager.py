"""
Model Download Manager — wraps ``downcraft`` for HuggingFace model downloads
with cross-session resume, persistent state, and progress tracking.

Delegates all actual HTTP work to ``downcraft`` (generic HTTP downloader
with Range-header resume).  This module exists only to integrate with the
existing server API (``DownloadManager`` singleton, progress callbacks, etc.).
"""

import asyncio
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("slo.infrastructure.download_manager")

try:
    from downcraft import downloader as sg_downloader
    from downcraft import state as sg_state
    from domains.infrastructure.hf_hub import (
        get_cache_dir,
        is_download_complete as hf_is_download_complete,
        list_model_files,
    )
except ImportError:
    logger.warning("downcraft not available — download management disabled",
        extra={"tag": "INFRA"})
    sg_downloader = None
    sg_state = None
    def get_cache_dir(model_id: str) -> str:
        return str(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}")
    def hf_is_download_complete(model_id: str, deep_check: bool = False) -> bool:
        return False
    def list_model_files(model_id: str) -> List[str]:
        return []

# Re-export for backward compat
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


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


# ---------------------------------------------------------------------------
# Re-export cache health helpers (using downcraft under the hood)
# ---------------------------------------------------------------------------

def _cache_dir(model_id: str) -> Path:
    """Get the HF cache directory path for a model."""
    return Path(get_cache_dir(model_id))


def _has_weight_files(cache_dir: Path) -> bool:
    """Check if a cache dir has any model weight files (safetensors or bin > 1KB)."""
    for ext in ("*.safetensors", "*.bin"):
        for f in cache_dir.rglob(ext):
            try:
                if f.stat().st_size > 1_000:
                    return True
            except OSError:
                continue
    return False


def _has_incomplete_downloads(cache_dir: Path) -> bool:
    """Check for in-progress or interrupted download markers."""
    incomplete = list(cache_dir.rglob("*.incomplete"))
    if incomplete:
        return True
    locks = list(cache_dir.rglob("*.lock"))
    return len(locks) > 0


def _get_snapshot_ref(cache_dir: Path) -> Optional[str]:
    refs_main = cache_dir / "refs" / "main"
    if not refs_main.exists():
        return None
    try:
        return refs_main.read_text().strip()
    except Exception:
        return None


def _has_complete_snapshot(cache_dir: Path) -> bool:
    commit = _get_snapshot_ref(cache_dir)
    if not commit:
        return False
    snapshot_dir = cache_dir / "snapshots" / commit
    if not snapshot_dir.exists():
        return False
    return _has_weight_files(snapshot_dir)


def is_download_complete(model_id: str, deep_check: bool = False) -> bool:
    """Check if a model is fully downloaded.

    Delegates to ``domains.infrastructure.hf_hub.is_download_complete`` for the
    canonical check (respects ``HF_HOME`` env var and uses proper
    cache directory resolution).

    Args:
        model_id: HuggingFace model ID
        deep_check: If True, verifies every expected weight file exists
            via Hub API (network call). Skip for batch listing.
    """
    return hf_is_download_complete(model_id, deep_check=deep_check)


def cleanup_incomplete(model_id: str) -> bool:
    """Remove an incomplete/partial download from HF cache."""
    cache_dir = _cache_dir(model_id)
    if not cache_dir.exists():
        return False
    logger.warning("Removing incomplete cache for %s: %s", model_id, cache_dir,
        extra={"tag": "INFRA"})
    shutil.rmtree(str(cache_dir), ignore_errors=True)
    # Also clean persistent state
    if sg_state is not None:
        sg_state.get_state().remove(model_id)
    return True


def list_incomplete_models() -> List[str]:
    """Scan HF cache and return model IDs with incomplete downloads."""
    base = Path.home() / ".cache" / "huggingface" / "hub"
    if not base.exists():
        return []
    result = []
    for entry in sorted(base.iterdir()):
        if not entry.name.startswith("models--") or not entry.is_dir():
            continue
        model_id = entry.name[len("models--"):].replace("--", "/")
        if _has_incomplete_downloads(entry):
            result.append(model_id)
        elif not _has_complete_snapshot(entry) and _has_weight_files(entry):
            result.append(model_id)
    return result


# ---------------------------------------------------------------------------
# DownloadManager — wraps downcraft for backward-compatible API
# ---------------------------------------------------------------------------

class DownloadManager:
    """
    Download manager singleton.

    Wraps ``downcraft`` to provide the existing server API
    (``download()``, ``is_cached()``, ``get_progress()``, etc.)
    with the addition of cross-session resume via persistent state.
    """

    def __init__(self):
        self._downloads: Dict[str, DownloadProgress] = {}
        self._lock = threading.Lock()
        self._tasks: Dict[str, asyncio.Task] = {}
        self._cleanup_ttl = 300
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
        """Whether the model is fully cached on disk (survives restart)."""
        return is_download_complete(model_id)

    def cancel(self, model_id: str) -> bool:
        with self._lock:
            entry = self._downloads.get(model_id)
            if entry and entry.status in (DownloadStatus.QUEUED, DownloadStatus.DOWNLOADING):
                entry.status = DownloadStatus.CANCELLED
                task = self._tasks.pop(model_id, None)
                if task and not task.done():
                    task.cancel()
                if sg_state is not None:
                    sg_state.get_state().set_status(model_id, "cancelled")
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
        """Download a HuggingFace model using downcraft (with cross-session resume).

        Unlike the old implementation (which cleaned up and restarted on every
        resume), this delegates to ``downcraft`` which preserves partial
        downloads across restarts via ``~/.downcraft/state.json``.
        """
        if is_download_complete(model_id):
            return {"status": "already_cached", "model_id": model_id}

        # Clean up incomplete HF cache markers, then let downcraft resume
        cache_dir = _cache_dir(model_id)
        if cache_dir.exists():
            incomplete = list(cache_dir.rglob("*.incomplete")) + list(cache_dir.rglob("*.lock"))
            for f in incomplete:
                try:
                    f.unlink()
                except OSError:
                    pass

        if self.is_downloading(model_id):
            return {"status": "already_downloading", "model_id": model_id}

        # Estimate total from Hub API
        try:
            files = list_model_files(model_id)
            total_est = sum(f.size for f in files if not f.is_ignored) or total_bytes_hint
        except Exception:
            total_est = total_bytes_hint

        self._set_progress(
            model_id,
            status=DownloadStatus.QUEUED,
            total_bytes=total_est,
            started_at=time.time(),
        )
        self._notify_callbacks(model_id)

        loop = asyncio.get_event_loop()
        task = asyncio.create_task(self._download_worker(model_id, total_est))
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
        """Run the downcraft in a thread executor, updating progress."""
        self._set_progress(model_id, status=DownloadStatus.DOWNLOADING)
        self._notify_callbacks(model_id)
        start_time = time.time()

        def _progress_cb(mid: str, downloaded: int, total: int, speed: float):
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

        def _file_cb(mid: str, fpath: str):
            self._set_progress(mid, current_file=fpath)
            self._notify_callbacks(mid)

        def _do_download():
            from domains.infrastructure.hf_hub import download_hf_model
            download_hf_model(
                model_id,
                on_progress=_progress_cb,
                on_file_complete=_file_cb,
            )

        await asyncio.to_thread(_do_download)

        elapsed = time.time() - start_time
        cache_dir = _cache_dir(model_id)
        self._set_progress(
            model_id,
            status=DownloadStatus.COMPLETE,
            completed_at=time.time(),
            percentage=100.0,
        )
        self._notify_callbacks(model_id)

        logger.info("Downloaded %s in %.1fs → %s", model_id, elapsed, cache_dir,
            extra={"tag": "INFRA"})
        return {
            "status": "complete",
            "model_id": model_id,
            "cache_dir": str(cache_dir),
            "elapsed_seconds": round(elapsed, 1),
        }

    def cleanup_stale(self, max_age: float = 300):
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
