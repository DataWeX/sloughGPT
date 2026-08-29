"""Bridge between x86 VM syscalls and the existing training HTTP API.

VM assembly programs call ``int 0x80`` with ``SYS_TRAIN_START`` /
``SYS_TRAIN_STATUS`` / ``SYS_TRAIN_GET_RESULT`` to launch and monitor
training jobs.

The bridge is a thin HTTP proxy — it translates VM syscalls into requests
to the existing ``POST /training/start`` and ``GET /training/jobs/{id}``
endpoints.  No training logic lives here.

Usage from x86 assembly::

    mov ebx, config_addr
    mov eax, 28          ; SYS_TRAIN_START
    int 0x80
    ; job_id in eax

    mov ebx, job_id
    mov eax, 29          ; SYS_TRAIN_STATUS
    int 0x80
    ; EAX = 0=running, 1=done, 2=error, -1=not_found

    mov ebx, job_id
    mov ecx, buffer_addr
    mov edx, buffer_size
    mov eax, 30          ; SYS_TRAIN_GET_RESULT
    int 0x80
    ; EAX = bytes written
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Optional

import requests

logger = logging.getLogger("slo.vm.training")

_API_BASE = "http://localhost:8000"
_TRAINING_TIMEOUT_S = 5
_POLL_INTERVAL_S = 2


class VMTrainingBridge:
    """Thin proxy: x86 VM syscalls -> existing training REST API.

    Thread-safe job tracking with a ``ThreadPoolExecutor`` for the initial
    ``POST /training/start`` call (which spawns the real training in the
    server's own thread pool).  The bridge itself contains zero training code.
    """

    def __init__(self) -> None:
        self._jobs: dict[int, dict[str, Any]] = {}
        self._next_job_id = 1
        self._lock = threading.Lock()
        self._session = requests.Session()

    def start(self, config_json: str) -> int:
        """Parse JSON config and call ``POST /training/start``.

        Returns a local job_id (>=1) on success, -1 on error.
        """
        try:
            body = json.loads(config_json)
        except json.JSONDecodeError as e:
            logger.error("TRAIN_START: invalid JSON: %s", e)
            return -1

        if not isinstance(body, dict):
            logger.error("TRAIN_START: config must be a JSON object")
            return -1

        # Map VM config keys to the API TrainingRequest schema
        payload: dict[str, Any] = {
            "dataset": body.get("dataset", ""),
            "epochs": body.get("epochs", 3),
            "learning_rate": float(body.get("lr", body.get("learning_rate", 1e-3))),
            "batch_size": int(body.get("batch_size", 32)),
            "n_embed": int(body.get("embed_dim", body.get("n_embed", 128))),
            "n_layer": int(body.get("n_layer", 4)),
            "n_head": int(body.get("n_head", 4)),
            "block_size": int(body.get("block_size", 128)),
            "name": body.get("name", "vm-training"),
            "model": body.get("model", "sloughgpt"),
        }
        if body.get("data_path"):
            payload["data_path"] = body["data_path"]

        try:
            resp = self._session.post(
                f"{_API_BASE}/training/start",
                json=payload,
                timeout=_TRAINING_TIMEOUT_S,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error("TRAIN_START: API call failed: %s", e)
            return -1

        api_job_id = data.get("job_id", "")
        job_id = self._next_job_id
        self._next_job_id += 1

        with self._lock:
            self._jobs[job_id] = {
                "api_job_id": api_job_id,
                "status": "running",
            }

        logger.debug(
            "TRAIN_START job=%d api_job_id=%s config=%s",
            job_id, api_job_id, body,
        )
        return job_id

    def status(self, job_id: int) -> dict:
        """Poll ``GET /training/jobs/{api_job_id}`` and return local status.

        Keys: ``status`` (``"running"``|``"completed"``|``"failed"``|``"not_found"``),
              ``progress`` (0-1 float), ``error`` (str or None).
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return {"status": "not_found", "progress": 0.0, "error": None}

        if job["status"] in ("completed", "failed"):
            return {
                "status": job["status"],
                "progress": job.get("progress", 1.0),
                "error": job.get("error"),
            }

        api_job_id = job.get("api_job_id", "")
        if not api_job_id:
            return {"status": "running", "progress": 0.0, "error": None}

        try:
            resp = self._session.get(
                f"{_API_BASE}/training/jobs/{api_job_id}",
                timeout=_TRAINING_TIMEOUT_S,
            )
            if resp.status_code == 404:
                return {"status": "not_found", "progress": 0.0, "error": None}
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.warning("TRAIN_STATUS job=%d: poll failed: %s", job_id, e)
            return {"status": "running", "progress": 0.0, "error": None}

        api_status = data.get("status", "running")
        api_progress = data.get("progress", 0)

        if api_status == "completed":
            with self._lock:
                self._jobs[job_id]["status"] = "completed"
                self._jobs[job_id]["progress"] = 1.0
                self._jobs[job_id]["_result_data"] = data
        elif api_status in ("failed", "cancelled"):
            err = data.get("error", api_status)
            with self._lock:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["error"] = err

        return {
            "status": "running" if api_status == "running" else api_status,
            "progress": api_progress / 100.0,
            "error": data.get("error"),
        }

    def get_result_json(self, job_id: int) -> Optional[str]:
        """Return completed job result as JSON string, or None."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.get("status") != "completed":
                return None
            data = job.get("_result_data", {})
        return json.dumps({
            "success": data.get("status") == "completed",
            "final_loss": data.get("loss"),
            "eval_loss": data.get("eval_loss"),
            "model_path": data.get("checkpoint"),
            "checkpoint_name": data.get("checkpoint", "").split("/")[-1] if data.get("checkpoint") else None,
            "epochs_completed": data.get("current_epoch", 0),
        })

    def stop(self, job_id: int) -> bool:
        """Ask the API to stop a running training job.

        POSTs to ``/training/jobs/{api_job_id}/stop`` and marks the local
        record as ``stopping``.  Returns True when the API accepted the stop.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            api_job_id = job.get("api_job_id", "")

        if not api_job_id:
            return False

        try:
            resp = self._session.post(
                f"{_API_BASE}/training/jobs/{api_job_id}/stop",
                timeout=_TRAINING_TIMEOUT_S,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("TRAIN_STOP job=%d: API call failed: %s", job_id, e)
            return False

        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "stopping"
        logger.debug("TRAIN_STOP job=%d api_job_id=%s", job_id, api_job_id)
        return True

    def remove(self, job_id: int) -> bool:
        """Remove a completed/failed job from tracking. Returns True if found."""
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def job_info(self, job_id: int) -> Optional[dict[str, Any]]:
        """Return the tracked record for a job, or None if unknown.

        Keys: ``api_job_id``, ``status``, ``progress`` (0-1), ``error``.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {
                "api_job_id": job.get("api_job_id", ""),
                "status": job.get("status", "running"),
                "progress": job.get("progress", 0.0),
                "error": job.get("error"),
            }

    def alive_count(self) -> int:
        """Return number of jobs still running."""
        with self._lock:
            return sum(
                1 for j in self._jobs.values() if j.get("status") == "running"
            )


# Module-level singleton
_bridge: Optional[VMTrainingBridge] = None


def get_bridge() -> VMTrainingBridge:
    global _bridge
    if _bridge is None:
        _bridge = VMTrainingBridge()
    return _bridge
