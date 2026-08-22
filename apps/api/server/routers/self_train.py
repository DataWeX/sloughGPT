"""
Self-Train Router - Start/stop/status for self-training subprocess.
"""
import asyncio
import logging
import re
from fastapi import APIRouter
from pydantic import BaseModel, Field
import state as server_state
import subprocess
import sys
from pathlib import Path
from typing import Optional

from schemas.common import success_response, raise_error, classify_and_raise, safe_audit_log
from domains.infrastructure.errors import AppError

logger = logging.getLogger("slo.api.self_train")


class SelfTrainRequest(BaseModel):
    """Validated request body for starting self-training.

    Attributes:
        model: Model name — alphanumeric, dots, hyphens, slashes, underscores only.
        temperature: Sampling temperature between 0.0 and 2.0.
        forever: Whether to train indefinitely.
    """
    model: Optional[str] = Field(default=None, max_length=128)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    forever: bool = False


class SelfTrainRouter:
    def __init__(self):
        self._repo_root = Path(__file__).resolve().parents[4]
        self._model_name_re = re.compile(r'^[a-zA-Z0-9_./-]+$')
        self.router = APIRouter(prefix="/self-train", tags=["self-train"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/start", self.start_self_train, methods=["POST"])
        self.router.add_api_route("/stop", self.stop_self_train, methods=["POST"])
        self.router.add_api_route("/status", self.get_self_train_status, methods=["GET"])

    async def start_self_train(self, req: Optional[SelfTrainRequest] = None) -> dict:
        """Start self-training in a subprocess.

        Args:
            req: Validated config with model, temperature, forever keys.

        Returns:
            Status dict with pid or error.
        """
        proc = server_state._self_train_proc
        if proc is not None and proc.poll() is None:
            return success_response(data={"status": "already_running", "pid": proc.pid})
        try:
            script = self._repo_root / "scripts" / "self_train.py"
            cmd = [sys.executable, str(script)]
            if req and req.model:
                if not self._model_name_re.match(req.model):
                    raise_error("Invalid model name — only alphanumeric, dots, hyphens, slashes, underscores allowed", "E_VAL_REQUEST", status_code=422)
                cmd.extend(["--model", req.model])
            if req and req.temperature is not None:
                cmd.extend(["--temperature", str(req.temperature)])
            if req and req.forever:
                cmd.append("--forever")
            proc = await asyncio.to_thread(subprocess.Popen, cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            server_state._self_train_proc = proc
            logger.info("Self-training started (pid=%d, model=%s)", proc.pid, req.model if req and req.model else "default")
            safe_audit_log("self_train.start", resource=req.model if req and req.model else "default", detail=f"pid={proc.pid}", temperature=req.temperature if req and req.temperature is not None else None, forever=bool(req and req.forever))
            return success_response(data={"status": "started", "pid": proc.pid})
        except AppError:
            raise
        except Exception as e:
            logger.warning("Self-training start failed: %s", e)
            classify_and_raise(e, source="self_train_start")

    async def stop_self_train(self) -> dict:
        """Stop the running self-training subprocess.

        Attempts a graceful terminate with a 5-second timeout, then falls
        back to kill if the process does not exit. Clears the process
        reference in server state.

        Returns:
            Success envelope with status "stopped" or "not_running".

        Side effects:
            - Terminates or kills the self-training subprocess.
            - Clears server_state._self_train_proc.
            - Writes an audit log entry for the stop action.
        """
        proc = server_state._self_train_proc
        if proc is None or proc.poll() is not None:
            return success_response(data={"status": "not_running"})
        try:
            proc.terminate()
            await asyncio.to_thread(proc.wait, 5)
            server_state._self_train_proc = None
            logger.info("Self-training stopped gracefully (pid=%d)", proc.pid)
            safe_audit_log("self_train.stop", resource=str(proc.pid), detail="stopped")
            return success_response(data={"status": "stopped"})
        except Exception as e:
            proc.kill()
            server_state._self_train_proc = None
            logger.warning("Self-training killed after terminate timeout (pid=%d): %s", proc.pid, e)
            safe_audit_log("self_train.stop", resource=str(proc.pid), detail="killed")
            raise_error(str(e), "E_INFRA_STARTUP", details={"status": "killed"})

    async def get_self_train_status(self) -> dict:
        """Check the current status of the self-training subprocess.

        Returns whether training is running, has exited, or has not started.
        Includes the last 50 lines of training history from the history file.

        Returns:
            Success envelope with status (not_started/running/exited),
            optional pid/returncode, and history lines.
        """
        proc = server_state._self_train_proc
        history_path = self._repo_root / "data" / "self_train_history.txt"
        history = []
        if history_path.exists():
            _text = await asyncio.to_thread(history_path.read_text)
            history = _text.strip().split("\n")[-50:]
        if proc is None:
            return success_response(data={"status": "not_started", "history": history})
        ret = proc.poll()
        if ret is None:
            return success_response(data={"status": "running", "pid": proc.pid, "history": history})
        return success_response(data={"status": "exited", "returncode": ret, "history": history})


router = SelfTrainRouter().router
