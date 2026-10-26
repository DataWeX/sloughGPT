"""
Shell Router — Dait shell command execution endpoints.

Thin HTTP wrapper over ``domain.shell._internal.repl.ShellREPL``.  Runs commands
in an isolated ``MemoryIO`` instance and returns captured output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import re
import threading
import time as _time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from infrastructure.auth import require_auth_if_enabled
from infrastructure.shell_sandbox import ShellSecurityError, validate_command
from pydantic import BaseModel, Field
from schemas.common import classify_and_raise, endpoint, raise_error, safe_audit_log

from domain.shell import DaitRuntime, MemoryIO, ShellREPL

logger = logging.getLogger("slo.api.shell")

router = APIRouter(prefix="/shell", tags=["shell"])

_repl: ShellREPL | None = None
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

    command: str = Field(
        ..., min_length=1, max_length=10000, description="Shell command to execute"
    )
    timeout_ms: int = Field(30000, ge=100, le=120000, description="Execution timeout in ms")


class ShellExecResponse(BaseModel):
    """Result of a shell command execution."""

    output: str = Field(..., description="Captured stdout/stderr output")
    exit_code: int = Field(..., description="Command exit code (0 = success)")
    elapsed_ms: float = Field(..., description="Execution time in milliseconds")


def _sse_line(
    stream: str, phase: str, status: str, data: dict, meta: dict = None, message: str = ""
) -> str:
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


# ── Wire hygiene + singleton serialization ───────────────────────────────────

_ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"  # CSI … final byte
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC … BEL / ST
    r"|\x1b[@-Z\\-_]"  # other 2-byte escapes
)


def _sanitize_line(line: str) -> str:
    """Terminal-hygienic single line for HTTP/SSE bodies.

    Strips ANSI escape sequences, applies carriage-return overwrite
    (text before the last ``\\r`` is replaced — column-0 semantics, so a
    spinner frame's residue collapses away and the content written after
    the final ``\\r`` survives), and drops remaining C0 control chars
    except tab.  Raw bytes stay server-side; the wire carries
    display-safe text.
    """
    line = _ANSI_RE.sub("", line)
    if "\r" in line:
        line = line.split("\r")[-1]
    return "".join(ch for ch in line if ch >= " " or ch == "\t")


def _sanitize_output(output: str) -> str:
    """Sanitize every line of a completed command output."""
    return "\n".join(_sanitize_line(line) for line in output.split("\n"))


# The singleton REPL is shared process-wide; serialize executions so two
# concurrent requests cannot interleave writes into one MemoryIO.
_exec_lock = threading.Lock()


def _execute_locked(repl, command: str) -> tuple[str, int]:
    with _exec_lock:
        return repl.execute(command)


def _stream_events(repl, command: str, t0: float):
    """Yield SSE frames for *command* while it runs (POST /exec/stream).

    A worker thread executes against a queue-fed MemoryIO; this generator
    reassembles complete lines from the raw chunks, sanitizes each, and
    yields — so slow commands stream their output instead of dumping it
    post-hoc.  Serialized on ``_exec_lock`` like the sync endpoint.
    """
    q: queue.Queue = queue.Queue()
    result: dict = {}

    def worker() -> None:
        try:
            with _exec_lock:
                out, code = repl.execute(command, on_write=q.put)
            result["output"] = out
            result["exit_code"] = code
        except Exception as e:
            logger.warning("Shell stream exec error: %s", e, extra={"tag": "SHELL"})
            result["error"] = e
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    buf = ""
    index = 0
    received = 0
    while True:
        item = q.get()
        if item is None:
            break
        received += 1
        buf += item
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            yield _sse_line(
                "shell", "STREAMING", "working", {"line": _sanitize_line(line), "index": index}
            )
            index += 1

    if received == 0 and result.get("output"):
        # repl produced output without streaming through on_write
        # (mock/legacy path) — fall back to post-hoc line emission
        lines = result["output"].split("\n")
        for i, line in enumerate(lines):
            if line or i < len(lines) - 1:  # skip trailing empty
                yield _sse_line(
                    "shell", "STREAMING", "working", {"line": _sanitize_line(line), "index": index}
                )
                index += 1
    else:
        tail = _sanitize_line(buf)
        if tail:  # final line without trailing newline
            yield _sse_line("shell", "STREAMING", "working", {"line": tail, "index": index})
            index += 1

    err = result.get("error")
    if err is not None:
        error_type = type(err).__name__
        is_conn = isinstance(err, (ConnectionError, OSError)) and "connect" in str(err).lower()
        is_timeout = isinstance(err, TimeoutError)
        hint = ""
        if is_conn:
            hint = " Is the API server running? Use 'api start'."
        elif is_timeout:
            hint = " Request timed out."
        yield _sse_line(
            "shell",
            "STREAMING",
            "error",
            {
                "error": f"{error_type}: {err}{hint}",
                "error_type": error_type,
            },
        )
        return

    exit_code = result.get("exit_code", 1)
    elapsed = (_time.monotonic() - t0) * 1000
    safe_audit_log(
        "shell.exec_stream",
        resource=command[:80],
        detail=f"exit={exit_code} elapsed={elapsed:.0f}ms lines={index}",
    )
    yield _sse_line(
        "shell",
        "STREAMING",
        "complete",
        {
            "exit_code": exit_code,
            "lines": index,
        },
        meta={"elapsed_ms": round(elapsed, 2)},
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/exec", response_model=ShellExecResponse)
@endpoint("shell.exec")
async def exec_command(req: ShellExecRequest, auth_user: dict = Depends(require_auth_if_enabled)):
    """Execute a shell command and return captured output.

    Runs the command in a Dait ``ShellREPL`` instance with captured I/O.
    State persists across calls (same singleton REPL).
    """
    try:
        validate_command(req.command)
    except ShellSecurityError as e:
        raise_error(str(e), "E_SHELL_SECURITY", status_code=403)

    repl = _get_repl()
    t0 = _time.monotonic()

    try:
        output, exit_code = await asyncio.to_thread(_execute_locked, repl, req.command)
        output = _sanitize_output(output)
    except Exception as e:
        logger.warning("Shell exec error: %s", e, extra={"tag": "SHELL"})
        classify_and_raise(e, source="shell.exec")

    elapsed = (_time.monotonic() - t0) * 1000

    safe_audit_log(
        "shell.exec", resource=req.command[:80], detail=f"exit={exit_code} elapsed={elapsed:.0f}ms"
    )

    return ShellExecResponse(
        output=output,
        exit_code=exit_code,
        elapsed_ms=round(elapsed, 2),
    )


@router.post("/exec/stream")
@endpoint("shell.exec_stream")
async def exec_command_stream(
    req: ShellExecRequest, request: Request, auth_user: dict = Depends(require_auth_if_enabled)
):
    """Execute a shell command with SSE streaming output.

    Yields lines as they are produced, then a completion event.
    Supports cancellation via AbortController disconnect.
    """
    try:
        validate_command(req.command)
    except ShellSecurityError as e:
        raise_error(str(e), "E_SHELL_SECURITY", status_code=403)

    repl = _get_repl()
    t0 = _time.monotonic()

    def generate():
        """Emit line events WHILE the command runs (see _stream_events)."""
        yield from _stream_events(repl, req.command, t0)

    return StreamingResponse(generate(), media_type="text/event-stream")
