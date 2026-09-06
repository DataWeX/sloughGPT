"""Training state — single source of truth for training state management."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "models" / "auto-training"
LORA_DIR = REPO_ROOT / "data" / "user_adapters"
TURBO_DIR = REPO_ROOT / "models" / "turbo-trained"
MAX_CHECKPOINT_DISK_MB = 500
VALID_CKPT_NAME = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
SOU_MAGIC = b"SOUL"

for d in (CHECKPOINTS_DIR, LORA_DIR, TURBO_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingState:
    running: bool = False
    config: dict = None
    student_net: object | None = None
    student_tokenizer: object | None = None
    complete_enqueued: bool = False

    def __post_init__(self):
        if self.config is None:
            self.config = {}


_state = TrainingState()

_turbo_lock = threading.Lock()
_turbo_cancel_event = threading.Event()
_turbo_pause_event = threading.Event()
_turbo_state: dict[str, Any] = {
    "status": "idle",
    "job_id": None,
    "global_step": 0,
    "total_steps": 0,
    "progress": 0.0,
    "loss": None,
    "learning_rate": None,
    "steps_per_sec": None,
    "eta_s": None,
    "elapsed_s": None,
    "avg_quality": None,
    "result": None,
    "error": None,
    "paused": False,
    "last_heartbeat": 0.0,
}

_auto_train_cancel_event: threading.Event | None = None
_auto_train_pause_event: threading.Event | None = None

try:
    from domains.infrastructure.pugqeep import PGQ
    _auto_train_pgq = PGQ(
        name="auto-train",
        storage_dir=REPO_ROOT / "models" / "auto-training" / ".pgq",
    )
except Exception:
    _auto_train_pgq = None


def get_state() -> TrainingState:
    return _state


def get_turbo_state() -> dict:
    return _turbo_state


def get_turbo_lock() -> threading.Lock:
    return _turbo_lock


def get_turbo_pause_event() -> threading.Event:
    return _turbo_pause_event


def get_turbo_cancel_event() -> threading.Event:
    return _turbo_cancel_event


def get_cancel_event() -> threading.Event | None:
    return _auto_train_cancel_event


def set_cancel_event(event: threading.Event):
    global _auto_train_cancel_event
    _auto_train_cancel_event = event


def get_pause_event() -> threading.Event | None:
    return _auto_train_pause_event


def set_pause_event(event: threading.Event):
    global _auto_train_pause_event
    _auto_train_pause_event = event


def get_pgq():
    return _auto_train_pgq
