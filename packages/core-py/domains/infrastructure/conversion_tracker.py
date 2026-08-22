"""
Model Conversion Tracker — provides real-time status for model download + SLNC conversion.

Tracks per-model conversion stages so the frontend can show progress instead of a
blind spinner during the 10-30s conversion process.

Stages:
  idle → downloading → converting → protecting → loading → ready
                                      or → error

Usage:
    from domains.infrastructure.conversion_tracker import get_tracker

    tracker = get_tracker()
    tracker.start("gpt2", stage="downloading")
    tracker.update("gpt2", stage="converting", progress=0.5)
    tracker.finish("gpt2")
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("slo.infrastructure.conversion_tracker")


class ConversionStage(str, Enum):
    IDLE = "idle"
    DOWNLOADING = "downloading"
    CONVERTING = "converting"
    PROTECTING = "protecting"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


@dataclass
class ConversionStatus:
    model_id: str
    stage: ConversionStage = ConversionStage.IDLE
    progress: float = 0.0  # 0.0 → 1.0
    message: str = ""
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "stage": self.stage.value,
            "progress": round(self.progress, 2),
            "message": self.message,
            "error": self.error,
            "elapsed_s": round(time.time() - self.started_at, 1),
        }


class ConversionTracker:
    """Tracks conversion status for all models."""

    def __init__(self):
        self._statuses: dict[str, ConversionStatus] = {}

    def start(self, model_id: str, stage: ConversionStage = ConversionStage.IDLE, message: str = "") -> ConversionStatus:
        """Start tracking a model conversion."""
        status = ConversionStatus(
            model_id=model_id,
            stage=stage,
            message=message or self._default_message(stage),
            started_at=time.time(),
        )
        self._statuses[model_id] = status
        logger.info("Conversion started: %s → %s", model_id, stage.value)
        return status

    def update(self, model_id: str, stage: ConversionStage = None, progress: float = None, message: str = None) -> ConversionStatus:
        """Update conversion status."""
        status = self._statuses.get(model_id)
        if status is None:
            status = self.start(model_id, stage or ConversionStage.IDLE)

        if stage is not None:
            status.stage = stage
        if progress is not None:
            status.progress = min(1.0, max(0.0, progress))
        if message is not None:
            status.message = message
        elif stage is not None:
            status.message = self._default_message(stage)

        status.updated_at = time.time()
        status.elapsed_s = time.time() - status.started_at
        return status

    def finish(self, model_id: str) -> ConversionStatus:
        """Mark conversion as complete."""
        status = self._statuses.get(model_id)
        if status:
            status.stage = ConversionStage.READY
            status.progress = 1.0
            status.message = "Ready"
            status.updated_at = time.time()
            status.elapsed_s = time.time() - status.started_at
            logger.info("Conversion complete: %s (%.1fs)", model_id, status.elapsed_s)
        return status

    def fail(self, model_id: str, error: str) -> ConversionStatus:
        """Mark conversion as failed."""
        status = self._statuses.get(model_id)
        if status:
            status.stage = ConversionStage.ERROR
            status.error = error
            status.message = f"Error: {error}"
            status.updated_at = time.time()
        return status

    def get(self, model_id: str) -> Optional[dict]:
        """Get status for a model."""
        status = self._statuses.get(model_id)
        return status.to_dict() if status else None

    def get_all(self) -> list[dict]:
        """Get all active conversions."""
        return [s.to_dict() for s in self._statuses.values()]

    def get_active(self) -> list[dict]:
        """Get only in-progress conversions (not ready/error)."""
        active_stages = {ConversionStage.IDLE, ConversionStage.DOWNLOADING,
                         ConversionStage.CONVERTING, ConversionStage.PROTECTING,
                         ConversionStage.LOADING}
        return [s.to_dict() for s in self._statuses.values() if s.stage in active_stages]

    def clear(self, model_id: str = None) -> None:
        """Clear tracked status."""
        if model_id:
            self._statuses.pop(model_id, None)
        else:
            self._statuses.clear()

    @staticmethod
    def _default_message(stage: ConversionStage) -> str:
        messages = {
            ConversionStage.IDLE: "Preparing...",
            ConversionStage.DOWNLOADING: "Downloading model weights...",
            ConversionStage.CONVERTING: "Converting to optimized format (.slnc)...",
            ConversionStage.PROTECTING: "Protecting files from accidental deletion...",
            ConversionStage.LOADING: "Loading into memory...",
            ConversionStage.READY: "Ready",
            ConversionStage.ERROR: "Error",
        }
        return messages.get(stage, "")


# Module-level singleton
_tracker: ConversionTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker() -> ConversionTracker:
    """Get or create the global conversion tracker."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = ConversionTracker()
    return _tracker
