"""
Self-Train Router - Start/stop/status for self-training subprocess.
"""
from typing import Optional
from fastapi import APIRouter
import state as server_state
import subprocess
import sys
from pathlib import Path

from schemas.common import success_response

_REPO_ROOT = Path(__file__).resolve().parents[4]

router = APIRouter(prefix="/self-train", tags=["self-train"])


@router.post("/start")
async def start_self_train(params: Optional[dict] = None):
    """Start self-training in a subprocess.

    Args:
        params: optional config with model, temperature, forever keys

    Returns:
        status dict with pid or error
    """
    params = params or {}
    proc = server_state._self_train_proc
    if proc is not None and proc.poll() is None:
        return success_response(data={"status": "already_running", "pid": proc.pid})
    try:
        script = _REPO_ROOT / "scripts" / "self_train.py"
        cmd = [sys.executable, str(script)]
        if params.get("model"):
            cmd.extend(["--model", params["model"]])
        if params.get("temperature"):
            cmd.extend(["--temperature", str(params["temperature"])])
        if params.get("forever"):
            cmd.append("--forever")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        server_state._self_train_proc = proc
        return success_response(data={"status": "started", "pid": proc.pid})
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
