"""
Self-Train Router - Start/stop/status for self-training subprocess.
"""

import asyncio
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import state as server_state
from fastapi import APIRouter, Depends
from infrastructure.auth import require_auth_if_enabled
from infrastructure.contract import RouteSpec, create_router, params_from_model
from pydantic import BaseModel, Field
from schemas.common import (
    classify_and_raise,
    endpoint,
    raise_error,
    safe_audit_log,
    success_response,
)

from domain.agents import ToolSpec
from domain.infrastructure import AppError

logger = logging.getLogger("slo.api.self_train")


class SelfTrainRequest(BaseModel):
    """Validated request body for starting self-training.

    Attributes:
        model: Model name — alphanumeric, dots, hyphens, slashes, underscores only.
        temperature: Sampling temperature between 0.0 and 2.0.
        forever: Whether to train indefinitely.
    """

    model: str | None = Field(default=None, max_length=128)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    forever: bool = False


class SelfTrainRouter:
    # One projection per process (contract registry: same identity
    # re-registering = double projection). Later instances — test fixtures —
    # share the router the first instance registered.
    _shared_router: ClassVar[APIRouter | None] = None

    def __init__(self):
        self._repo_root = Path(__file__).resolve().parents[3]
        self._model_name_re = re.compile(r"^[a-zA-Z0-9_./-]+$")
        if SelfTrainRouter._shared_router is not None:
            self.router = SelfTrainRouter._shared_router
            return
        # Descriptor projection (card 20261101_154058): full paths live in the
        # RouteSpecs below — no prefix, so test apps that mount this router
        # directly serve the same byte-identical paths as the aggregate.
        self.router = APIRouter()
        self._register_routes()
        SelfTrainRouter._shared_router = self.router

    def _register_routes(self):
        # envelope=False: the handlers keep their success_response wraps, so the
        # wire bytes are unchanged by the projection.
        self.router.include_router(
            create_router(
                ToolSpec(
                    name="training.self_train.start",
                    description=(
                        "Start self-training in a subprocess.\n\n"
                        "Args:\n"
                        "    req: Validated config with model, temperature, forever keys.\n\n"
                        "Returns:\n"
                        "    Status dict with pid or error."
                    ),
                    parameters=params_from_model(SelfTrainRequest),
                    execute=self.start_self_train,
                    auth_scope="authenticated",
                    idempotent=False,
                    version="1",
                ),
                RouteSpec(
                    path="/self-train/start",
                    envelope=False,
                    summary="Start Self-Training",
                    tags=["self-train"],
                ),
                self.start_self_train,
            )
        )
        self.router.include_router(
            create_router(
                ToolSpec(
                    name="training.self_train.stop",
                    description="Stop the running self-training subprocess.",
                    parameters=[],
                    execute=self.stop_self_train,
                    auth_scope="authenticated",
                    idempotent=False,
                    version="1",
                ),
                RouteSpec(
                    path="/self-train/stop",
                    envelope=False,
                    summary="Stop Self-Training",
                    tags=["self-train"],
                ),
                self.stop_self_train,
            )
        )
        self.router.include_router(
            create_router(
                ToolSpec(
                    name="training.self_train.status",
                    description="Check the current status of the self-training subprocess.",
                    parameters=[],
                    execute=self.get_self_train_status,
                    auth_scope="public",
                    idempotent=True,
                    version="1",
                ),
                RouteSpec(
                    path="/self-train/status",
                    envelope=False,
                    summary="Get Self-Training Status",
                    tags=["self-train"],
                ),
                self.get_self_train_status,
            )
        )

    @endpoint("self_train.start")
    async def start_self_train(
        self,
        req: SelfTrainRequest = None,  # model-style body — required by the projection; default is a signature-level fallback
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
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
                    raise_error(
                        "Invalid model name — only alphanumeric, dots, hyphens, slashes, underscores allowed",
                        "E_VAL_REQUEST",
                        status_code=422,
                    )
                if ".." in req.model:
                    raise_error(
                        "Model name cannot contain path traversal", "E_VAL_REQUEST", status_code=422
                    )
                cmd.extend(["--model", req.model])
            if req and req.temperature is not None:
                cmd.extend(["--temperature", str(req.temperature)])
            if req and req.forever:
                cmd.append("--forever")
            proc = await asyncio.to_thread(
                subprocess.Popen, cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
            )
            server_state._self_train_proc = proc
            logger.info(
                "Self-training started (pid=%d, model=%s)",
                proc.pid,
                req.model if req and req.model else "default",
            )
            safe_audit_log(
                "self_train.start",
                resource=req.model if req and req.model else "default",
                detail=f"pid={proc.pid}",
                temperature=req.temperature if req and req.temperature is not None else None,
                forever=bool(req and req.forever),
            )
            return success_response(data={"status": "started", "pid": proc.pid})
        except AppError as e:
            classify_and_raise(e, source="self_train.start_self_train")
        except Exception as e:
            logger.warning("Self-training start failed: %s", e)
            classify_and_raise(e, source="self_train_start")

    @endpoint("self_train.stop")
    async def stop_self_train(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Stop the running self-training subprocess."""
        try:
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
                logger.warning(
                    "Self-training killed after terminate timeout (pid=%d): %s", proc.pid, e
                )
                safe_audit_log("self_train.stop", resource=str(proc.pid), detail="killed")
                raise_error(str(e), "E_INFRA_STARTUP", details={"status": "killed"})
        except Exception as e:
            classify_and_raise(e, source="self_train.stop")

    @endpoint("self_train.status")
    async def get_self_train_status(self) -> dict:
        """Check the current status of the self-training subprocess."""
        try:
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
                return success_response(
                    data={"status": "running", "pid": proc.pid, "history": history}
                )
            return success_response(
                data={"status": "exited", "returncode": ret, "history": history}
            )
        except Exception as e:
            classify_and_raise(e, source="self_train.status")


router = SelfTrainRouter().router
