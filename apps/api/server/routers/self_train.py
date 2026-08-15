"""
Self-Train Router - Start/stop/status for self-training subprocess.
"""
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import state as server_state
import subprocess
import sys
from pathlib import Path
from typing import Optional

from schemas.common import success_response, error_response


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

    async def start_self_train(self, req: Optional[SelfTrainRequest] = None):
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
                    raise HTTPException(status_code=422, detail="Invalid model name — only alphanumeric, dots, hyphens, slashes, underscores allowed")
                cmd.extend(["--model", req.model])
            if req and req.temperature is not None:
                cmd.extend(["--temperature", str(req.temperature)])
            if req and req.forever:
                cmd.append("--forever")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            server_state._self_train_proc = proc
            try:
                from infrastructure.auth import get_audit_logger
                get_audit_logger().log(
                    "self_train.start",
                    resource=(req.model if req and req.model else "default"),
                    detail=f"pid={proc.pid}",
                    extra={"temperature": req.temperature if req and req.temperature is not None else None, "forever": bool(req and req.forever)},
                )
            except Exception:
                pass
            return success_response(data={"status": "started", "pid": proc.pid})
        except HTTPException:
            raise
        except Exception as e:
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="self_train_start")
            return error_response(str(e), "E_INFRA_STARTUP")

    async def stop_self_train(self):
        """Stop self-training subprocess."""
        proc = server_state._self_train_proc
        if proc is None or proc.poll() is not None:
            return success_response(data={"status": "not_running"})
        try:
            proc.terminate()
            proc.wait(timeout=5)
            server_state._self_train_proc = None
            try:
                from infrastructure.auth import get_audit_logger
                get_audit_logger().log("self_train.stop", resource=str(proc.pid), detail="stopped")
            except Exception:
                pass
            return success_response(data={"status": "stopped"})
        except Exception as e:
            proc.kill()
            server_state._self_train_proc = None
            try:
                from infrastructure.auth import get_audit_logger
                get_audit_logger().log("self_train.stop", resource=str(proc.pid), detail="killed")
            except Exception:
                pass
            return success_response(data={"status": "killed", "error": str(e)})

    async def get_self_train_status(self):
        """Get self-training status."""
        proc = server_state._self_train_proc
        history_path = self._repo_root / "data" / "self_train_history.txt"
        history = []
        if history_path.exists():
            history = history_path.read_text().strip().split("\n")[-50:]
        if proc is None:
            return success_response(data={"status": "not_started", "history": history})
        ret = proc.poll()
        if ret is None:
            return success_response(data={"status": "running", "pid": proc.pid, "history": history})
        return success_response(data={"status": "exited", "returncode": ret, "history": history})


router = SelfTrainRouter().router
