"""
Persistent download state — survives process restarts and power loss.

Tracks per-model download progress in a JSON file at
``~/.downcraft/state.json``, flushed to disk after every chunk.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

STATE_DIR = Path.home() / ".downcraft"
STATE_FILE = STATE_DIR / "state.json"
LOCK_FILE = STATE_DIR / "state.lock"

FLUSH_INTERVAL = 2.0  # seconds between disk flushes


@dataclass
class FileProgress:
    path: str
    url: str
    bytes_downloaded: int = 0
    total_bytes: int = 0
    checksum: str = ""
    complete: bool = False


@dataclass
class ModelState:
    model_id: str
    status: str = "queued"  # queued | downloading | complete | failed
    files: Dict[str, FileProgress] = field(default_factory=dict)
    started_at: float = 0.0
    completed_at: Optional[float] = None
    error: str = ""
    cache_dir: str = ""

    @property
    def total_bytes(self) -> int:
        return sum(f.total_bytes for f in self.files.values())

    @property
    def bytes_downloaded(self) -> int:
        return sum(f.bytes_downloaded for f in self.files.values())

    @property
    def percentage(self) -> float:
        t = self.total_bytes
        if t == 0:
            return 0.0
        return round(self.bytes_downloaded / t * 100, 1)

    @property
    def files_completed(self) -> int:
        return sum(1 for f in self.files.values() if f.complete)

    @property
    def files_total(self) -> int:
        return len(self.files)


class PersistentState:
    """Thread-safe, crash-safe download state persisted to JSON.

    On every update, the state is flushed to disk (throttled to
    FLUSH_INTERVAL seconds) so a process restart sees the latest
    partial progress.
    """

    def __init__(self, state_dir: Union[str, Path] = STATE_DIR):
        self._state_dir = Path(state_dir)
        self._state_file = self._state_dir / "state.json"
        self._lock_file = self._state_dir / "state.lock"
        self._mutex = threading.Lock()
        self._models: Dict[str, ModelState] = {}
        self._dirty = False
        self._last_flush = 0.0
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, model_id: str) -> Optional[ModelState]:
        with self._mutex:
            return self._models.get(model_id)

    def list(self) -> List[ModelState]:
        with self._mutex:
            return list(self._models.values())

    def create(self, model_id: str, cache_dir: str) -> ModelState:
        with self._mutex:
            st = ModelState(
                model_id=model_id,
                status="queued",
                started_at=time.time(),
                cache_dir=cache_dir,
            )
            self._models[model_id] = st
            self._mark_dirty()
            return st

    def set_status(self, model_id: str, status: str, error: str = ""):
        with self._mutex:
            st = self._models.get(model_id)
            if st is None:
                return
            st.status = status
            if error:
                st.error = error
            if status in ("complete", "failed"):
                st.completed_at = time.time()
            self._mark_dirty()

    def update_file_progress(
        self,
        model_id: str,
        file_path: str,
        url: str,
        bytes_downloaded: int,
        total_bytes: int,
        checksum: str = "",
        complete: bool = False,
    ):
        with self._mutex:
            st = self._models.get(model_id)
            if st is None:
                return
            fp = st.files.get(file_path)
            if fp is None:
                fp = FileProgress(
                    path=file_path,
                    url=url,
                    bytes_downloaded=bytes_downloaded,
                    total_bytes=total_bytes,
                    checksum=checksum,
                    complete=complete,
                )
                st.files[file_path] = fp
            else:
                fp.url = url
                fp.bytes_downloaded = bytes_downloaded
                fp.total_bytes = total_bytes
                if checksum:
                    fp.checksum = checksum
                if complete:
                    fp.complete = True
            if complete:
                # Check if all files done → mark model complete
                if all(f.complete for f in st.files.values()):
                    st.status = "complete"
                    st.completed_at = time.time()
            if st.status != "complete":
                st.status = "downloading"
            self._mark_dirty()

    def remove(self, model_id: str):
        with self._mutex:
            self._models.pop(model_id, None)
            self._mark_dirty()

    def flush(self):
        """Force-flush state to disk immediately."""
        with self._mutex:
            if self._dirty:
                self._write()
                self._dirty = False
                self._last_flush = time.time()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _mark_dirty(self):
        self._dirty = True
        now = time.time()
        if now - self._last_flush >= FLUSH_INTERVAL:
            self._write()
            self._dirty = False
            self._last_flush = now

    def _load(self):
        if not self._state_file.exists():
            return
        try:
            data = json.loads(self._state_file.read_text())
            for mid, d in data.get("models", {}).items():
                files = {}
                for fp_d in d.get("files", []):
                    fp = FileProgress(**fp_d)
                    files[fp.path] = fp
                st = ModelState(
                    model_id=mid,
                    status=d.get("status", "queued"),
                    files=files,
                    started_at=d.get("started_at", 0.0),
                    completed_at=d.get("completed_at"),
                    error=d.get("error", ""),
                    cache_dir=d.get("cache_dir", ""),
                )
                self._models[mid] = st
        except Exception as e:
            logger.warning("Failed to load state from %s: %s", self._state_file, e)

    def _write(self):
        self._state_dir.mkdir(parents=True, exist_ok=True)
        models_out = {}
        for mid, st in self._models.items():
            models_out[mid] = {
                "status": st.status,
                "files": [asdict(f) for f in st.files.values()],
                "started_at": st.started_at,
                "completed_at": st.completed_at,
                "error": st.error,
                "cache_dir": st.cache_dir,
            }
        tmp = str(self._state_file) + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"models": models_out, "updated_at": time.time()}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(self._state_file))


# Module-level singleton
_state: Optional[PersistentState] = None


def get_state() -> PersistentState:
    global _state
    if _state is None:
        _state = PersistentState()
    return _state
