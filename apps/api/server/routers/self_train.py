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

from schemas.common import success_response

_REPO_ROOT = Path(__file__).resolve().parents[4]

router = APIRouter(prefix="/self-train", tags=["self-train"])

_MODEL_NAME_RE = re.compile(r'^[a-zA-Z0-9_./-]+$')


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


@router.post("/start")
async def start_self_train(req: Optional[SelfTrainRequest] = None):
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
        script = _REPO_ROOT / "scripts" / "self_train.py"
        cmd = [sys.executable, str(script)]
        if req and req.model:
            if not _MODEL_NAME_RE.match(req.model):
                raise HTTPException(status_code=422, detail="Invalid model name — only alphanumeric, dots, hyphens, slashes, underscores allowed")
            cmd.extend(["--model", req.model])
        if req and req.temperature is not None:
            cmd.extend(["--temperature", str(req.temperature)])
        if req and req.forever:
            cmd.append("--forever")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        server_state._self_train_proc = proc
        return success_response(data={"status": "started", "pid": proc.pid})
    except HTTPException:
        raise
    except Exception as e:
        return success_response(data={"status": "error", "error": str(e)})


@router.post("/stop")
async def stop_self_train():
    """Stop self-training subprocess."""
    proc = server_state._self_train_proc
    if proc is None or proc.poll() is not None:
        return success_response(data={"status": "not_running"})
    try:
        proc.terminate()
        proc.wait(timeout=5)
        server_state._self_train_proc = None
        return success_response(data={"status": "stopped"})
    except Exception as e:
        proc.kill()
        server_state._self_train_proc = None
        return success_response(data={"status": "killed", "error": str(e)})


@router.get("/status")
async def get_self_train_status():
    """Get self-training status."""
    proc = server_state._self_train_proc
    history_path = _REPO_ROOT / "data" / "self_train_history.txt"
    history = []
    if history_path.exists():
        history = history_path.read_text().strip().split("\n")[-50:]
    if proc is None:
        return success_response(data={"status": "not_started", "history": history})
    ret = proc.poll()
    if ret is None:
        return success_response(data={"status": "running", "pid": proc.pid, "history": history})
    return success_response(data={"status": "exited", "returncode": ret, "history": history})
