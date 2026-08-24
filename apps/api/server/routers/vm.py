"""
VM Router — x86 assembly execution endpoints.

Provides a sandboxed x86 virtual machine that runs assembly programs
and returns execution results (registers, memory, output, trace).
"""

from __future__ import annotations

import time
import logging
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field
from pydantic import model_validator
from schemas.common import raise_error, success_response, classify_and_raise, safe_audit_log

logger = logging.getLogger("slo.api.vm")
router = APIRouter(prefix="/vm", tags=["vm"])


# ── Request / Response schemas ────────────────────────────────────────────────


class VMRunRequest(BaseModel):
    """Request to run x86 assembly in the VM.

    Either ``source`` (raw x86 assembly) or ``program`` (a builtin program
    name, resolved via the builtin registry) must be supplied.
    """

    program: Optional[str] = Field(
        None, max_length=32, description="Builtin program name (e.g. 'hello')"
    )
    source: str = Field(None, max_length=50000, description="x86 assembly source code")
    max_steps: int = Field(5000, ge=1, le=1000000, description="Max CPU steps")
    memory_size: int = Field(0x100000, ge=0x10000, le=0x1000000, description="VM memory size in bytes")
    role: str = Field("user", max_length=20, description="Permission role: user, admin, kernel")
    debug: bool = Field(False, description="Include register dump and trace in response")
    keyboard_input: Optional[str] = Field(None, max_length=10000, description="Simulated keyboard input for INT 16h")

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
    error: Optional[str] = None
    trace: Optional[list[dict]] = None
    vga_text: Optional[str] = None
    vga_cells: Optional[list[dict]] = None
    keyboard_buffer: Optional[str] = None
    memory_dump: Optional[str] = None
    training_job_id: Optional[int] = None
    training_result: Optional[str] = None


class VMTrainingJobResponse(BaseModel):
    """Status of a training job launched from a VM syscall."""

    job_id: int
    api_job_id: str
    status: str
    progress: float
    error: Optional[str] = None
    result: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/run", response_model=VMRunResponse)
async def run_assembly(req: VMRunRequest) -> dict:
    """Run x86 assembly code in the sandboxed VM.

    Assembles the source, spawns a user process, and executes it.
    Returns registers, output, and optional trace.
    """
    t0 = time.monotonic()

    try:
        from domains.shell.vm import X86VirtualSystem, InsFault, MemFault
        from domains.shell.vm_permissions import Role
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

    try:
        vs = X86VirtualSystem(memory_size=req.memory_size)
        pid = vs.spawn("web_user", req_source)
        if pid is None:
            return VMRunResponse(
                success=False, exit_code=-1, steps_executed=0,
                elapsed_ms=(time.monotonic() - t0) * 1000, output="",
                registers=[], eip=0, eip_hex="0x0", status="spawn_failed",
                error="Failed to spawn process — assembly may have errors",
            )

        role_map = {"user": Role.USER, "admin": Role.ADMIN, "kernel": Role.KERNEL}
        vs._syscall._rbac.assign(pid, role_map.get(req.role, Role.USER))

        vs.scheduler.start(vs.cpu)
        current = vs.scheduler.current
        if current is None:
            return VMRunResponse(
                success=False, exit_code=-1, steps_executed=0,
                elapsed_ms=(time.monotonic() - t0) * 1000, output="",
                registers=[], eip=0, eip_hex="0x0", status="no_process",
                error="No process available to run",
            )

        current.restore_to_cpu(vs.cpu)

        if req.keyboard_input:
            kbd = vs.devices.get("keyboard")
            if kbd:
                for ch in req.keyboard_input:
                    kbd._scancode_buffer.append(ord(ch) & 0xFF)

        output_buffer: list[str] = []
        original_write = vs._syscall._sys_write

        def _captured_write(fd, buf_addr, count):
            if fd in (1, 2):
                data = bytes(vs.cpu._read8(buf_addr + i) for i in range(count))
                output_buffer.append(data.decode("ascii", errors="replace"))
                return count
            return original_write(fd, buf_addr, count)

        launched_job_id: Optional[int] = None
        original_train_start = vs._syscall._sys_train_start

        def _captured_train_start(config_addr):
            job_id = original_train_start(config_addr)
            nonlocal launched_job_id
            if job_id is not None and job_id >= 1:
                launched_job_id = job_id
            return job_id

        training_result: Optional[str] = None
        original_train_get_result = vs._syscall._sys_train_get_result

        def _captured_train_get_result(job_id, buf_addr, buf_size):
            nonlocal training_result
            written = original_train_get_result(job_id, buf_addr, buf_size)
            if written and written > 0:
                try:
                    data = bytes(vs.cpu._read8(buf_addr + i) for i in range(written))
                    training_result = data.decode("utf-8", errors="replace")
                except Exception:
                    training_result = None
            return written

        vs._syscall._sys_write = _captured_write
        vs._syscall._sys_train_start = _captured_train_start
        vs._syscall._sys_train_get_result = _captured_train_get_result
        vs.cpu._trace_enabled = req.debug
        vs.cpu._trace.clear()
        safe_audit_log("vm.run", resource="vm", detail=f"max_steps={req.max_steps} debug={req.debug} role={req.role}")
        try:
            try:
                vs.cpu.run(max_steps=req.max_steps)
            except (InsFault, MemFault):
                # A runaway program (fetch beyond loaded code, stack/memory
                # fault) halts cleanly, matching X86VirtualSystem.run().
                pass
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
                "#000000", "#0000AA", "#00AA00", "#00AAAA",
                "#AA0000", "#AA00AA", "#AA5500", "#AAAAAA",
                "#555555", "#5555FF", "#55FF55", "#55FFFF",
                "#FF5555", "#FF55FF", "#FFFF55", "#FFFFFF",
            ]
            cells = []
            for i in range(80 * 25):
                ch = vs.cpu._read8(0xB8000 + i * 2)
                attr = vs.cpu._read8(0xB8000 + i * 2 + 1)
                fg = attr & 0x0F
                bg = (attr >> 4) & 0x07
                char = chr(ch) if 32 <= ch < 127 else " " if ch == 0 else "?"
                cells.append({
                    "ch": char,
                    "fg": VGA_COLORS[fg],
                    "bg": VGA_COLORS[bg],
                })
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
            kbd = vs.devices.get("keyboard")
            if kbd and hasattr(kbd, "_scancode_buffer"):
                kbd_state = "".join(chr(b) for b in kbd._scancode_buffer if 32 <= b < 127)
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

    except Exception as e:
        logger.exception("VM execution error")
        classify_and_raise(e, source="vm.run")


@router.get("/training/jobs/{job_id}", response_model=VMTrainingJobResponse)
async def training_job_status(job_id: str) -> dict:
    """Return the status of a training job launched via VM syscall."""
    try:
        try:
            job_num = int(job_id)
        except ValueError:
            raise_error("Training job not found", "E_NOT_FOUND", status_code=404)
        try:
            from domains.shell.vm_training_bridge import get_bridge
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
    except Exception as e:
        classify_and_raise(e, source="vm.training_job_status")


@router.post("/training/jobs/{job_id}/stop")
async def training_job_stop(job_id: str) -> dict:
    """Request a stop for a running training job launched via VM syscall."""
    try:
        try:
            job_num = int(job_id)
        except ValueError:
            raise_error("Training job not found", "E_NOT_FOUND", status_code=404)
        try:
            from domains.shell.vm_training_bridge import get_bridge
        except ImportError as e:
            raise_error(f"VM training bridge unavailable: {e}", "E_BAD_REQUEST", status_code=503)

        bridge = get_bridge()
        ok = bridge.stop(job_num)
        if not ok:
            raise_error("Training job not found or not stoppable", "E_NOT_FOUND", status_code=404)
        return success_response(data={"status": "stopping", "job_id": job_num})
    except Exception as e:
        classify_and_raise(e, source="vm.training_job_stop")


@router.get("/builtins")
async def list_builtins() -> dict:
    """List built-in x86 assembly programs with their source code."""
    try:
        try:
            from vm_builtins import BUILTIN_PROGRAMS
            programs = [
                {"name": name, "description": entry["description"], "code": entry["program"]()}
                for name, entry in BUILTIN_PROGRAMS.items()
            ]
        except ImportError:
            return success_response(data={"programs": []})
        return success_response(data={"programs": programs})
    except Exception as e:
        classify_and_raise(e, source="vm.builtins")


@router.get("/info")
async def vm_info() -> dict:
    """Return VM capabilities and limits."""
    try:
        reg_names = ["EAX", "ECX", "EDX", "EBX", "ESP", "EBP", "ESI", "EDI"]
        registers = {name: {"size_bits": 32, "name": name} for name in reg_names}
        return success_response(data={
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
        })
    except Exception as e:
        classify_and_raise(e, source="vm.info")