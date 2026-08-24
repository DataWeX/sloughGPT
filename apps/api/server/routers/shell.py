"""
Shell Router — Dait shell command execution endpoints.

Thin HTTP wrapper over ``domains.shell.repl.ShellREPL``.  Runs commands
in an isolated ``MemoryIO`` instance and returns captured output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time as _time
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from domains.shell.io import MemoryIO
from domains.shell.repl import ShellREPL
from domains.shell.runtime import DaitRuntime
from schemas.common import safe_audit_log

logger = logging.getLogger("slo.api.shell")

router = APIRouter(prefix="/shell", tags=["shell"])

_repl: Optional[ShellREPL] = None
_repl_lock = threading.Lock()


def _get_repl() -> ShellREPL:
    """Return a singleton ShellREPL, creating it on first call."""
    global _repl
    if _repl is not None:
        return _repl
    with _repl_lock:
        if _repl is not None:
            return _repl
        io = MemoryIO()
        runtime = DaitRuntime()
        _repl = ShellREPL(runtime, io=io)
        return _repl


# ── Request / Response schemas ────────────────────────────────────────────────


class ShellExecRequest(BaseModel):
    """Request to execute a shell command."""

    command: str = Field(..., min_length=1, max_length=10000, description="Shell command to execute")
    timeout_ms: int = Field(30000, ge=100, le=120000, description="Execution timeout in ms")


class ShellExecResponse(BaseModel):
    """Result of a shell command execution."""

    output: str = Field(..., description="Captured stdout/stderr output")
    exit_code: int = Field(..., description="Command exit code (0 = success)")
    elapsed_ms: float = Field(..., description="Execution time in milliseconds")


def _sse_line(stream: str, phase: str, status: str, data: dict, meta: dict = None, message: str = "") -> str:
    """Build a standard SSE data line."""
    envelope = {
        "stream": stream,
        "phase": phase,
        "status": status,
        "message": message,
        "data": data,
        "meta": meta or {},
    }
    return "data: " + json.dumps(envelope, default=str) + "\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/exec", response_model=ShellExecResponse)
async def exec_command(req: ShellExecRequest):
    """Execute a shell command and return captured output.

    Runs the command in a Dait ``ShellREPL`` instance with captured I/O.
    State persists across calls (same singleton REPL).
    """
    repl = _get_repl()
    t0 = _time.monotonic()

    try:
        output, exit_code = await asyncio.to_thread(repl.execute, req.command)
    except Exception as e:
        logger.warning("Shell exec error: %s", e, extra={"tag": "SHELL"})
        output = f"Shell error: {type(e).__name__}: {e}"
        exit_code = 1

    elapsed = (_time.monotonic() - t0) * 1000

    safe_audit_log("shell.exec", resource=req.command[:80], detail=f"exit={exit_code} elapsed={elapsed:.0f}ms")

    return ShellExecResponse(
        output=output,
        exit_code=exit_code,
        elapsed_ms=round(elapsed, 2),
    )


@router.post("/exec/stream")
async def exec_command_stream(req: ShellExecRequest, request: Request):
    """Execute a shell command with SSE streaming output.

    Yields lines as they are produced, then a completion event.
    Supports cancellation via AbortController disconnect.
    """
    repl = _get_repl()
    t0 = _time.monotonic()

    def generate():
        output_lines: list[str] = []
        exit_code = 1

        try:
            output, exit_code = repl.execute(req.command)
            output_lines = output.split("\n") if output else []
        except Exception as e:
            logger.warning("Shell stream exec error: %s", e, extra={"tag": "SHELL"})
            yield _sse_line("shell", "STREAMING", "error", {"error": str(e)})
            return

        # Yield output lines
        for i, line in enumerate(output_lines):
            if line or i < len(output_lines) - 1:  # skip trailing empty
                yield _sse_line("shell", "STREAMING", "working", {"line": line, "index": i})

        elapsed = (_time.monotonic() - t0) * 1000
        safe_audit_log("shell.exec_stream", resource=req.command[:80], detail=f"exit={exit_code} elapsed={elapsed:.0f}ms lines={len(output_lines)}")
        yield _sse_line("shell", "STREAMING", "complete", {
            "exit_code": exit_code,
            "lines": len(output_lines),
        }, meta={"elapsed_ms": round(elapsed, 2)})

    return StreamingResponse(generate(), media_type="text/event-stream")
