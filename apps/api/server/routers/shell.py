"""
Shell Router — Dait shell command execution + VM console endpoints.

Thin HTTP wrapper over ``domain.shell._internal.repl.ShellREPL``.  Runs commands
in an isolated ``MemoryIO`` instance and returns captured output.

Route-map card 6/7 (20261101_154057): ``routers/vm.py`` merged in here as a
second prefixed sub-router — ``_shell_router`` (/shell) + ``_vm_router`` (/vm)
behind one aggregate ``router``, paths byte-identical so the /developer page's
VM tab and the §12 training-launch journey keep their API surface.
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
from pydantic import BaseModel, Field, model_validator
from schemas.common import (
    classify_and_raise,
    endpoint,
    raise_error,
    safe_audit_log,
    success_response,
)

from domain.shell import DaitRuntime, MemoryIO, ShellREPL

logger = logging.getLogger("slo.api.shell")
# Moved verbatim from routers/vm.py (route-map card 6/7): the VM console keeps
# its own observability category, so log-based dashboards see no change.
_vm_logger = logging.getLogger("slo.api.vm")

_shell_router = APIRouter(prefix="/shell", tags=["shell"])
# Moved verbatim from routers/vm.py (route-map card 6/7, card 20261101_154057).
_vm_router = APIRouter(prefix="/vm", tags=["vm"])
# One facade for both halves (souls/settings aggregate pattern).
router = APIRouter()
router.include_router(_shell_router)
router.include_router(_vm_router)

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
    raw: bool = Field(
        False,
        description=(
            "Pass output through un-sanitized: ANSI/OSC escape sequences and "
            "control bytes are preserved for consumers that interpret them "
            "(the terminal engine). Default false = terminal-hygienic wire."
        ),
    )


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


def _stream_events(repl, command: str, t0: float, raw: bool = False):
    """Yield SSE frames for *command* while it runs (POST /exec/stream).

    A worker thread executes against a queue-fed MemoryIO; this generator
    reassembles complete lines from the raw chunks, sanitizes each, and
    yields — so slow commands stream their output instead of dumping it
    post-hoc.  Serialized on ``_exec_lock`` like the sync endpoint.
    ``raw=True`` skips the per-line sanitize pass (escape sequences reach
    the terminal engine intact); line reassembly itself is unchanged.
    """
    clean = (lambda s: s) if raw else _sanitize_line
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
            yield _sse_line("shell", "STREAMING", "working", {"line": clean(line), "index": index})
            index += 1

    if received == 0 and result.get("output"):
        # repl produced output without streaming through on_write
        # (mock/legacy path) — fall back to post-hoc line emission
        lines = result["output"].split("\n")
        for i, line in enumerate(lines):
            if line or i < len(lines) - 1:  # skip trailing empty
                yield _sse_line(
                    "shell", "STREAMING", "working", {"line": clean(line), "index": index}
                )
                index += 1
    else:
        tail = clean(buf)
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


@_shell_router.post("/exec", response_model=ShellExecResponse)
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
        if not req.raw:
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


@_shell_router.post("/exec/stream")
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
        yield from _stream_events(repl, req.command, t0, raw=req.raw)

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Moved verbatim from routers/vm.py (route-map card 6/7, card 20261101_154057) ──
# x86 assembly sandbox + training-launch endpoints. Deduped onto the module
# header above: time → _time, logger → _vm_logger; handlers byte-identical.
# ── Request / Response schemas ────────────────────────────────────────────────


class VMRunRequest(BaseModel):
    """Request to run x86 assembly in the VM.

    Either ``source`` (raw x86 assembly) or ``program`` (a builtin program
    name, resolved via the builtin registry) must be supplied.
    """

    program: str | None = Field(
        None, max_length=32, description="Builtin program name (e.g. 'hello')"
    )
    source: str = Field(None, max_length=50000, description="x86 assembly source code")
    max_steps: int = Field(5000, ge=1, le=1000000, description="Max CPU steps")
    memory_size: int = Field(
        0x100000, ge=0x10000, le=0x1000000, description="VM memory size in bytes"
    )
    role: str = Field("user", max_length=20, description="Permission role: user, admin, kernel")
    debug: bool = Field(False, description="Include register dump and trace in response")
    keyboard_input: str | None = Field(
        None, max_length=10000, description="Simulated keyboard input for INT 16h"
    )

    @model_validator(mode="after")
    def _require_program_or_source(self):
        if self.program is None and self.source is None:
            raise ValueError("Either 'program' or 'source' must be provided")
        return self


class VMRegister(BaseModel):
    """Single register state."""

    name: str
    value: int
    hex: str


class VMRunResponse(BaseModel):
    """Execution result from the VM."""

    success: bool
    exit_code: int
    steps_executed: int
    elapsed_ms: float
    output: str
    registers: list[VMRegister]
    eip: int
    eip_hex: str
    status: str
    error: str | None = None
    trace: list[dict] | None = None
    vga_text: str | None = None
    vga_cells: list[dict] | None = None
    keyboard_buffer: str | None = None
    memory_dump: str | None = None
    training_job_id: int | None = None
    training_result: str | None = None


class VMTrainingJobResponse(BaseModel):
    """Status of a training job launched from a VM syscall."""

    job_id: int
    api_job_id: str
    status: str
    progress: float
    error: str | None = None
    result: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────


@_vm_router.post("/run", response_model=VMRunResponse)
@endpoint("vm.run_assembly")
async def run_assembly(
    req: VMRunRequest, auth_user: dict = Depends(require_auth_if_enabled)
) -> dict:
    """Run x86 assembly code in the sandboxed VM.

    Assembles the source, spawns a user process, and executes it.
    Returns registers, output, and optional trace.
    """
    t0 = _time.monotonic()

    try:
        from domain.shell import InsFault, MemFault, Role, X86VirtualSystem
    except ImportError as e:
        raise_error(f"VM module not available: {e}", "E_BAD_REQUEST", status_code=503)

    if req.program:
        try:
            from vm_builtins import get_builtin

            req_source = get_builtin(req.program)
        except (KeyError, ImportError):
            raise_error(f"Unknown builtin program: {req.program}", "E_NOT_FOUND", status_code=404)
    else:
        req_source = req.source

    vs = X86VirtualSystem(memory_size=req.memory_size)
    pid = vs.spawn("web_user", req_source)
    if pid is None:
        return VMRunResponse(
            success=False,
            exit_code=-1,
            steps_executed=0,
            elapsed_ms=(_time.monotonic() - t0) * 1000,
            output="",
            registers=[],
            eip=0,
            eip_hex="0x0",
            status="spawn_failed",
            error="Failed to spawn process — assembly may have errors",
        )

    role_map = {"user": Role.USER, "admin": Role.ADMIN, "kernel": Role.KERNEL}
    vs._syscall._rbac.assign(pid, role_map.get(req.role, Role.USER))

    vs.scheduler.start(vs.cpu)
    current = vs.scheduler.current
    if current is None:
        return VMRunResponse(
            success=False,
            exit_code=-1,
            steps_executed=0,
            elapsed_ms=(_time.monotonic() - t0) * 1000,
            output="",
            registers=[],
            eip=0,
            eip_hex="0x0",
            status="no_process",
            error="No process available to run",
        )

    current.restore_to_cpu(vs.cpu)

    if req.keyboard_input:
        for ch in req.keyboard_input:
            vs.cpu.push_key(ch)

    output_buffer: list[str] = []
    original_write = vs._syscall._sys_write

    def _captured_write(fd, buf_addr, count):
        if fd in (1, 2):
            data = bytes(vs.cpu._read8(buf_addr + i) for i in range(count))
            output_buffer.append(data.decode("ascii", errors="replace"))
            return count
        return original_write(fd, buf_addr, count)

    launched_job_id: int | None = None
    original_train_start = vs._syscall._sys_train_start

    def _captured_train_start(config_addr):
        job_id = original_train_start(config_addr)
        nonlocal launched_job_id
        if job_id is not None and job_id >= 1:
            launched_job_id = job_id
        return job_id

    training_result: str | None = None
    original_train_get_result = vs._syscall._sys_train_get_result

    def _captured_train_get_result(job_id, buf_addr, buf_size):
        nonlocal training_result
        written = original_train_get_result(job_id, buf_addr, buf_size)
        if written and written > 0:
            try:
                data = bytes(vs.cpu._read8(buf_addr + i) for i in range(written))
                training_result = data.decode("utf-8", errors="replace")
            except Exception as exc:
                _vm_logger.debug("Training result decoding failed: %s", exc)
                training_result = None
        return written

    vs._syscall._sys_write = _captured_write
    vs._syscall._sys_train_start = _captured_train_start
    vs._syscall._sys_train_get_result = _captured_train_get_result
    vs.cpu._trace_enabled = req.debug
    vs.cpu._trace.clear()
    safe_audit_log(
        "vm.run",
        resource="vm",
        detail=f"max_steps={req.max_steps} debug={req.debug} role={req.role}",
    )
    try:
        try:
            vs.cpu.run(max_steps=req.max_steps)
        except (InsFault, MemFault) as exc:
            # A runaway program (fetch beyond loaded code, stack/memory
            # fault) halts cleanly, matching X86VirtualSystem.run().
            _vm_logger.debug("VM execution fault: %s", exc)
    finally:
        vs._syscall._sys_write = original_write
        vs._syscall._sys_train_start = original_train_start
        vs._syscall._sys_train_get_result = original_train_get_result

    exit_code = vs.cpu._regs[0] & 0xFFFFFFFF
    elapsed = (_time.monotonic() - t0) * 1000

    reg_names = ["EAX", "ECX", "EDX", "EBX", "ESP", "EBP", "ESI", "EDI"]
    registers = [
        VMRegister(name=n, value=vs.cpu._regs[i], hex=f"0x{vs.cpu._regs[i]:08X}")
        for i, n in enumerate(reg_names)
    ]

    trace_data = None
    if req.debug and vs.cpu._trace:
        trace_data = [
            {
                "step": t.get("step", idx),
                "eip": f"0x{t.get('eip', 0):08X}",
                "opcode": t.get("opcode", "?"),
                "operands": t.get("operands", ""),
            }
            for idx, t in enumerate(vs.cpu._trace[:200])
        ]

    vga_text = None
    vga_cells = None
    try:
        VGA_COLORS = [
            "#000000",
            "#0000AA",
            "#00AA00",
            "#00AAAA",
            "#AA0000",
            "#AA00AA",
            "#AA5500",
            "#AAAAAA",
            "#555555",
            "#5555FF",
            "#55FF55",
            "#55FFFF",
            "#FF5555",
            "#FF55FF",
            "#FFFF55",
            "#FFFFFF",
        ]
        cells = []
        for i in range(80 * 25):
            ch = vs.cpu._read8(0xB8000 + i * 2)
            attr = vs.cpu._read8(0xB8000 + i * 2 + 1)
            fg = attr & 0x0F
            bg = (attr >> 4) & 0x07
            char = chr(ch) if 32 <= ch < 127 else " " if ch == 0 else "?"
            cells.append(
                {
                    "ch": char,
                    "fg": VGA_COLORS[fg],
                    "bg": VGA_COLORS[bg],
                }
            )
        vga_cells = cells
        # Also build plain text for backward compat
        lines = []
        for row in range(25):
            line = "".join(cells[row * 80 + col]["ch"] for col in range(80)).rstrip()
            if line:
                lines.append(line)
        vga_text = "\n".join(lines) if lines else None
    except Exception as exc:
        _vm_logger.debug("VGA text render failed: %s", exc)

    mem_dump = None
    if req.debug:
        try:
            esp = vs.cpu._regs[4] & 0xFFFFFFFF
            base = max(0, esp - 64)
            data = [vs.cpu._read8(base + i) for i in range(128)]
            rows = []
            for row in range(0, len(data), 16):
                chunk = data[row : row + 16]
                hex_part = " ".join(f"{b:02X}" for b in chunk)
                ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
                rows.append(f"{base + row:08X}  {hex_part:<48s}  {ascii_part}")
            mem_dump = "\n".join(rows)
        except Exception as exc:
            _vm_logger.debug("Memory dump failed: %s", exc)

    kbd_state = None
    try:
        kbd_buf = getattr(vs.cpu, "_kbd_buffer", None)
        if kbd_buf:
            kbd_state = "".join(chr(b) for b in kbd_buf if 32 <= b < 127)
    except Exception as exc:
        _vm_logger.debug("Keyboard state capture failed: %s", exc)

    return VMRunResponse(
        success=True,
        exit_code=exit_code,
        steps_executed=vs.cpu._step_count,
        elapsed_ms=round(elapsed, 2),
        output="".join(output_buffer),
        registers=registers,
        eip=vs.cpu._eip,
        eip_hex=f"0x{vs.cpu._eip:08X}",
        status="halted" if vs.cpu._step_count > 0 else "empty",
        trace=trace_data,
        vga_text=vga_text,
        vga_cells=vga_cells,
        keyboard_buffer=kbd_state,
        memory_dump=mem_dump,
        training_job_id=launched_job_id,
        training_result=training_result,
    )


@_vm_router.get("/training/jobs/{job_id}", response_model=VMTrainingJobResponse)
@endpoint("vm.training_job_status")
async def training_job_status(job_id: str) -> dict:
    """Return the status of a training job launched via VM syscall."""
    try:
        job_num = int(job_id)
    except ValueError:
        raise_error("Training job not found", "E_NOT_FOUND", status_code=404)
    try:
        from domain.shell import get_bridge
    except ImportError as e:
        raise_error(f"VM training bridge unavailable: {e}", "E_BAD_REQUEST", status_code=503)

    bridge = get_bridge()
    status = bridge.status(job_num)
    if status["status"] == "not_found":
        raise_error("Training job not found", "E_NOT_FOUND", status_code=404)

    info = bridge.job_info(job_num) or {}
    result = None
    if status["status"] == "completed":
        result = bridge.get_result_json(job_num)
    return VMTrainingJobResponse(
        job_id=job_num,
        api_job_id=str(info.get("api_job_id", "")),
        status=status["status"],
        progress=status["progress"],
        error=status["error"],
        result=result,
    )


@_vm_router.post("/training/jobs/{job_id}/stop")
@endpoint("vm.training_job_stop")
async def training_job_stop(
    job_id: str, auth_user: dict = Depends(require_auth_if_enabled)
) -> dict:
    """Request a stop for a running training job launched via VM syscall."""
    try:
        job_num = int(job_id)
    except ValueError:
        raise_error("Training job not found", "E_NOT_FOUND", status_code=404)
    try:
        from domain.shell import get_bridge
    except ImportError as e:
        raise_error(f"VM training bridge unavailable: {e}", "E_BAD_REQUEST", status_code=503)

    bridge = get_bridge()
    ok = bridge.stop(job_num)
    if not ok:
        raise_error("Training job not found or not stoppable", "E_NOT_FOUND", status_code=404)
    return success_response(data={"status": "stopping", "job_id": job_num})


@_vm_router.get("/builtins")
@endpoint("vm.list_builtins")
async def list_builtins() -> dict:
    """List built-in x86 assembly programs with their source code."""
    try:
        from vm_builtins import BUILTIN_PROGRAMS

        programs = [
            {"name": name, "description": entry["description"], "code": entry["program"]()}
            for name, entry in BUILTIN_PROGRAMS.items()
        ]
    except ImportError:
        return success_response(data={"programs": []})
    return success_response(data={"programs": programs})


@_vm_router.get("/info")
@endpoint("vm.vm_info")
async def vm_info() -> dict:
    """Return VM capabilities and limits."""
    reg_names = ["EAX", "ECX", "EDX", "EBX", "ESP", "EBP", "ESI", "EDI"]
    registers = {name: {"size_bits": 32, "name": name} for name in reg_names}
    return success_response(
        data={
            "isa": "x86-32",
            "max_steps": 1000000,
            "default_memory": 0x100000,
            "max_memory": 0x1000000,
            "registers": registers,
            "features": [
                "protected mode (32-bit)",
                "flat memory model",
                "ring 0 only",
                "INT 0x80 syscalls",
                "PIT timer",
                "keyboard/screen I/O",
                "process scheduling",
                "RBAC permissions",
            ],
        }
    )
