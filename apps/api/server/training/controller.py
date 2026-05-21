"""Training State Controller

Manages global training state: running, paused, idle.
Provides unified control over all training operations.
"""

import threading
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional, Dict
from datetime import datetime
import logging

logger = logging.getLogger("sloughgpt.training")


class TrainingState(str, Enum):
    """Training state machine states."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingController:
    """
    Global training state controller.

    Tracks overall training status and provides control methods.
    """

    _state: TrainingState = field(default=TrainingState.IDLE, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    # Current job info
    current_job_id: Optional[str] = field(default=None, init=False)
    current_job_name: Optional[str] = field(default=None, init=False)
    started_at: Optional[datetime] = field(default=None, init=False)
    paused_at: Optional[datetime] = field(default=None, init=False)

    # Statistics
    total_jobs: int = field(default=0, init=False)
    completed_jobs: int = field(default=0, init=False)
    failed_jobs: int = field(default=0, init=False)

    @property
    def state(self) -> TrainingState:
        """Get current state (thread-safe)."""
        with self._lock:
            return self._state

    def _set_state(self, new_state: TrainingState) -> None:
        """Set state (thread-safe)."""
        with self._lock:
            self._state = new_state

    def is_idle(self) -> bool:
        """Check if training is idle."""
        return self.state == TrainingState.IDLE

    def is_running(self) -> bool:
        """Check if training is running."""
        return self.state == TrainingState.RUNNING

    def is_paused(self) -> bool:
        """Check if training is paused."""
        return self.state == TrainingState.PAUSED

    def can_start(self) -> bool:
        """Check if training can be started."""
        return self.state in (TrainingState.IDLE, TrainingState.COMPLETED, TrainingState.FAILED)

    def can_pause(self) -> bool:
        """Check if training can be paused."""
        return self.state == TrainingState.RUNNING

    def can_resume(self) -> bool:
        """Check if training can be resumed."""
        return self.state == TrainingState.PAUSED

    def can_stop(self) -> bool:
        """Check if training can be stopped."""
        return self.state in (TrainingState.RUNNING, TrainingState.PAUSED)

    def start(self, job_id: str, job_name: str = "training") -> Dict[str, Any]:
        """
        Start a new training session.

        Returns:
            Dict with success status and message
        """
        with self._lock:
            if not self.can_start():
                return {
                    "success": False,
                    "state": self._state.value,
                    "message": f"Cannot start: training is {self._state.value}",
                    "current_job": self.current_job_id,
                }

            self._state = TrainingState.RUNNING
            self.current_job_id = job_id
            self.current_job_name = job_name
            self.started_at = datetime.now()
            self.paused_at = None
            self.total_jobs += 1

            return {
                "success": True,
                "state": self._state.value,
                "job_id": job_id,
                "job_name": job_name,
                "message": "Training started",
            }

    def pause(self) -> Dict[str, Any]:
        """Pause current training."""
        with self._lock:
            if not self.can_pause():
                return {
                    "success": False,
                    "state": self._state.value,
                    "message": f"Cannot pause: training is {self._state.value}",
                }

            self._state = TrainingState.PAUSED
            self.paused_at = datetime.now()

            return {
                "success": True,
                "state": self._state.value,
                "message": "Training paused",
                "paused_at": self.paused_at.isoformat(),
            }

    def resume(self) -> Dict[str, Any]:
        """Resume paused training."""
        with self._lock:
            if not self.can_resume():
                return {
                    "success": False,
                    "state": self._state.value,
                    "message": f"Cannot resume: training is {self._state.value}",
                }

            self._state = TrainingState.RUNNING
            resume_duration = (
                (datetime.now() - self.paused_at).total_seconds() if self.paused_at else 0
            )
            self.paused_at = None

            return {
                "success": True,
                "state": self._state.value,
                "message": "Training resumed",
                "resume_duration_seconds": resume_duration,
            }

    def stop(self) -> Dict[str, Any]:
        """Stop current training."""
        with self._lock:
            if not self.can_stop():
                return {
                    "success": False,
                    "state": self._state.value,
                    "message": f"Cannot stop: training is {self._state.value}",
                }

            self._state = TrainingState.STOPPING

            return {
                "success": True,
                "state": self._state.value,
                "message": "Training stop requested",
            }

    def complete(self) -> Dict[str, Any]:
        """Mark training as completed."""
        with self._lock:
            self._state = TrainingState.COMPLETED
            self.completed_jobs += 1

            return {
                "success": True,
                "state": self._state.value,
                "job_id": self.current_job_id,
                "message": "Training completed",
                "total_jobs": self.total_jobs,
                "completed_jobs": self.completed_jobs,
            }

    def fail(self, error: str = "") -> Dict[str, Any]:
        """Mark training as failed."""
        with self._lock:
            self._state = TrainingState.FAILED
            self.failed_jobs += 1

            return {
                "success": True,
                "state": self._state.value,
                "job_id": self.current_job_id,
                "message": "Training failed",
                "error": error,
                "total_jobs": self.total_jobs,
                "failed_jobs": self.failed_jobs,
            }

    def reset(self) -> Dict[str, Any]:
        """Reset to idle state."""
        with self._lock:
            self._state = TrainingState.IDLE
            self.current_job_id = None
            self.current_job_name = None
            self.started_at = None
            self.paused_at = None

            return {
                "success": True,
                "state": self._state.value,
                "message": "Training controller reset",
            }

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive training status."""
        with self._lock:
            return {
                "state": self._state.value,
                "is_idle": self._state == TrainingState.IDLE,
                "is_running": self._state == TrainingState.RUNNING,
                "is_paused": self._state == TrainingState.PAUSED,
                "is_stopping": self._state == TrainingState.STOPPING,
                "is_completed": self._state == TrainingState.COMPLETED,
                "is_failed": self._state == TrainingState.FAILED,
                "can_start": self.can_start(),
                "can_pause": self.can_pause(),
                "can_resume": self.can_resume(),
                "can_stop": self.can_stop(),
                "current_job_id": self.current_job_id,
                "current_job_name": self.current_job_name,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "paused_at": self.paused_at.isoformat() if self.paused_at else None,
                "runtime_seconds": (
                    (datetime.now() - self.started_at).total_seconds()
                    if self.started_at and self._state == TrainingState.RUNNING
                    else None
                ),
                "total_jobs": self.total_jobs,
                "completed_jobs": self.completed_jobs,
                "failed_jobs": self.failed_jobs,
            }


# Global controller instance
training_controller = TrainingController()


def get_training_controller() -> TrainingController:
    """Get the global training controller instance."""
    return training_controller
