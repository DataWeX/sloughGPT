"""
VM Router — x86 assembly execution endpoints.

Provides a sandboxed x86 virtual machine that runs assembly programs
and returns execution results (registers, memory, output, trace).
Also exposes interactive console sessions (shell REPL) with SSE output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel, Field, model_validator
from schemas.common import endpoint, raise_error, safe_audit_log, success_response

logger = logging.getLogger("slo.api.vm")
router = APIRouter(prefix="/vm", tags=["vm"])


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


@router.post("/run", response_model=VMRunResponse)
@endpoint("vm.run_assembly")
async def run_assembly(
    req: VMRunRequest, auth_user: dict = Depends(require_auth_if_enabled)
) -> dict:
    """Run x86 assembly code in the sandboxed VM.

    Assembles the source, spawns a user process, and executes it.
    Returns registers, output, and optional trace.
    """
    t0 = time.monotonic()

    try:
        from domain.shell._internal.vm import InsFault, MemFault, X86VirtualSystem
        from domain.shell._internal.vm_permissions import Role
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
            elapsed_ms=(time.monotonic() - t0) * 1000,
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
            elapsed_ms=(time.monotonic() - t0) * 1000,
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
                logger.debug("Training result decoding failed: %s", exc)
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
            logger.debug("VM execution fault: %s", exc)
    finally:
        vs._syscall._sys_write = original_write
        vs._syscall._sys_train_start = original_train_start
        vs._syscall._sys_train_get_result = original_train_get_result

    exit_code = vs.cpu._regs[0] & 0xFFFFFFFF
    elapsed = (time.monotonic() - t0) * 1000

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
        logger.debug("VGA text render failed: %s", exc)

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
            logger.debug("Memory dump failed: %s", exc)

    kbd_state = None
    try:
        kbd_buf = getattr(vs.cpu, "_kbd_buffer", None)
        if kbd_buf:
            kbd_state = "".join(chr(b) for b in kbd_buf if 32 <= b < 127)
    except Exception as exc:
        logger.debug("Keyboard state capture failed: %s", exc)

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


@router.get("/training/jobs/{job_id}", response_model=VMTrainingJobResponse)
@endpoint("vm.training_job_status")
async def training_job_status(job_id: str) -> dict:
    """Return the status of a training job launched via VM syscall."""
    try:
        job_num = int(job_id)
    except ValueError:
        raise_error("Training job not found", "E_NOT_FOUND", status_code=404)
    try:
        from domain.shell._internal.vm_training_bridge import get_bridge
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


@router.post("/training/jobs/{job_id}/stop")
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
        from domain.shell._internal.vm_training_bridge import get_bridge
    except ImportError as e:
        raise_error(f"VM training bridge unavailable: {e}", "E_BAD_REQUEST", status_code=503)

    bridge = get_bridge()
    ok = bridge.stop(job_num)
    if not ok:
        raise_error("Training job not found or not stoppable", "E_NOT_FOUND", status_code=404)
    return success_response(data={"status": "stopping", "job_id": job_num})


@router.get("/builtins")
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


@router.get("/info")
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


# ── Interactive console sessions ──────────────────────────────────────────────

_SESSION_TTL_SECONDS = 1800.0
_MAX_SESSIONS = 8


class VmConsoleSession:
    """A live ``X86VirtualSystem`` running the ``shell`` REPL builtin.

    Output is captured from the SYS_WRITE syscall into a history buffer
    and mirrored to an asyncio queue consumed by the SSE stream endpoint.
    A pump task runs the CPU in small chunks, calling ``transfer_key()``
    between steps so buffered keyboard input is delivered.
    """

    def __init__(self, session_id: str, role: str) -> None:
        from vm_builtins import get_builtin

        from domain.shell._internal.vm import X86VirtualSystem
        from domain.shell._internal.vm_permissions import Role

        self.id = session_id
        self.role = role
        self.created_at = time.time()
        self.last_active = time.monotonic()
        self.closed = False
        self.completed = False
        self.history: list[str] = []
        self.queue: asyncio.Queue[dict] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None

        vs = X86VirtualSystem(memory_size=0x100000)
        pid = vs.spawn("web_user", get_builtin("shell"))
        if pid is None:
            raise RuntimeError("shell process spawn failed")
        role_map = {"user": Role.USER, "admin": Role.ADMIN, "kernel": Role.KERNEL}
        vs._syscall._rbac.assign(pid, role_map.get(role, Role.USER))
        vs.scheduler.start(vs.cpu)
        current = vs.scheduler.current
        if current is None:
            raise RuntimeError("no scheduled process for shell session")
        current.restore_to_cpu(vs.cpu)

        original_write = vs._syscall._sys_write

        def _captured_write(fd: int, buf_addr: int, count: int) -> int:
            if fd in (1, 2):
                data = bytes(vs.cpu._read8(buf_addr + i) for i in range(count))
                text = data.decode("ascii", errors="replace")
                # Append to history BEFORE emitting: the stream endpoint
                # uses history length as a high-water mark to skip queue
                # items it already included in the backlog (no duplicates).
                self.history.append(text)
                self._emit({"type": "output", "text": text, "seq": len(self.history) - 1})
                return count
            return original_write(fd, buf_addr, count)

        vs._syscall._sys_write = _captured_write
        self.vs = vs

    def _emit(self, item: dict) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self.queue.put_nowait, item)
        except RuntimeError:
            pass  # loop shut down between check and dispatch

    def _run_chunk(self, max_steps: int = 2000) -> bool:
        """Run up to ``max_steps`` instructions. True once the CPU halted."""
        cpu = self.vs.cpu
        for _ in range(max_steps):
            if self.closed:
                return True
            cpu.transfer_key()
            if not cpu.step():
                return True
        return False

    async def _pump(self) -> None:
        try:
            while not self.closed:
                halted = await asyncio.to_thread(self._run_chunk)
                if halted:
                    self.completed = True
                    exit_code = self.vs.cpu._regs[0] & 0xFFFFFFFF
                    self._emit({"type": "status", "status": "complete", "exit_code": exit_code})
                    return
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("VM session %s stopped: %s", self.id, exc)
            self.completed = True
            self._emit({"type": "status", "status": "error", "error": str(exc)})

    def send_keys(self, text: str) -> int:
        """Queue user input. Returns characters actually delivered."""
        self.last_active = time.monotonic()
        delivered = 0
        for ch in text:
            if ch in ("\b", "\x7f"):
                self.vs.cpu.push_scancode(0x0E)
                delivered += 1
                continue
            before = len(self.vs.cpu._kbd_buffer)
            self.vs.cpu.push_key(ch)
            if len(self.vs.cpu._kbd_buffer) > before:
                delivered += 1
        return delivered

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            self.queue.put_nowait({"type": "status", "status": "closed"})
        except Exception:
            pass


_SESSIONS: dict[str, VmConsoleSession] = {}


def _evict_stale_sessions() -> None:
    now = time.monotonic()
    stale = [sid for sid, s in _SESSIONS.items() if now - s.last_active > _SESSION_TTL_SECONDS]
    for sid in stale:
        session = _SESSIONS.pop(sid, None)
        if session is not None:
            session.closed = True
            if session._task is not None:
                session._task.cancel()


def _get_session(session_id: str) -> VmConsoleSession:
    _evict_stale_sessions()
    session = _SESSIONS.get(session_id)
    if session is None:
        raise_error("VM session not found", "E_NOT_FOUND", status_code=404)
    return session


def _console_sse(phase: str, status: str, data: dict | None = None, message: str = "") -> str:
    payload = {
        "stream": "vm_console",
        "phase": phase,
        "status": status,
        "data": data or {},
        "meta": {},
        "message": message,
    }
    return "data: " + json.dumps(payload) + "\n\n"


class VMConsoleCreateRequest(BaseModel):
    role: str = Field("user", max_length=20, description="Permission role for the shell")


class VMConsoleInputRequest(BaseModel):
    text: str = Field(..., max_length=2000, description="Characters to deliver to stdin")


@router.post("/session")
@endpoint("vm.session_create")
async def create_console_session(
    req: VMConsoleCreateRequest, auth_user: dict = Depends(require_auth_if_enabled)
) -> dict:
    """Start an interactive console session running the shell REPL."""
    _evict_stale_sessions()
    if len(_SESSIONS) >= _MAX_SESSIONS:
        raise_error("Too many VM sessions — close one first", "E_RATE_LIMITED", status_code=429)
    session_id = uuid.uuid4().hex[:16]
    try:
        session = VmConsoleSession(session_id, req.role)
    except Exception as exc:
        raise_error(f"Failed to start VM session: {exc}", "E_BAD_REQUEST", status_code=500)
    session._loop = asyncio.get_running_loop()
    session._task = asyncio.create_task(session._pump())
    _SESSIONS[session_id] = session
    safe_audit_log("vm.session_create", resource=session_id, detail=f"role={req.role}")
    return success_response(data={"session_id": session_id, "role": req.role, "status": "running"})


@router.post("/session/{session_id}/input")
@endpoint("vm.session_input")
async def console_session_input(
    session_id: str,
    req: VMConsoleInputRequest,
    auth_user: dict = Depends(require_auth_if_enabled),
) -> dict:
    """Deliver characters to the session keyboard buffer."""
    session = _get_session(session_id)
    delivered = session.send_keys(req.text)
    return success_response(data={"accepted": delivered})


@router.get("/session/{session_id}/stream")
@endpoint("vm.session_stream")
async def console_session_stream(
    session_id: str, auth_user: dict = Depends(require_auth_if_enabled)
) -> StreamingResponse:
    """SSE stream: backlog, then live output; ends on halt/error/close."""
    session = _get_session(session_id)
    session.last_active = time.monotonic()
    # Copy first, then derive the high-water mark from the copy: output
    # appended before the copy is in the backlog; anything arriving after
    # carries a seq greater than the copy's last index.
    history_snapshot = list(session.history)
    backlog = "".join(history_snapshot)
    backlog_upto = len(history_snapshot) - 1

    async def _events():
        yield _console_sse("start", "continue", {"session_id": session_id})
        if backlog:
            yield _console_sse("output", "continue", {"text": backlog, "backlog": True})
        while True:
            if session.queue.empty() and (session.completed or session.closed):
                yield _console_sse("status", "complete", {"status": "closed"})
                return
            try:
                item = await asyncio.wait_for(session.queue.get(), timeout=15.0)
            except TimeoutError:
                if session.closed or session.completed:
                    yield _console_sse("status", "complete", {"status": "closed"})
                    return
                yield _console_sse("heartbeat", "continue")
                continue
            if item.get("type") == "output":
                if int(item.get("seq", -1)) <= backlog_upto:
                    continue  # already delivered via backlog
                yield _console_sse("output", "continue", {"text": item.get("text", "")})
            elif item.get("type") == "status":
                data = {k: v for k, v in item.items() if k != "type"}
                yield _console_sse("status", str(item.get("status", "complete")), data)
                return

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/session/{session_id}")
@endpoint("vm.session_delete")
async def delete_console_session(
    session_id: str, auth_user: dict = Depends(require_auth_if_enabled)
) -> dict:
    """Terminate a console session and release its VM."""
    session = _get_session(session_id)
    await session.close()
    _SESSIONS.pop(session_id, None)
    safe_audit_log("vm.session_close", resource=session_id, detail="delete")
    return success_response(data={"closed": True})
