"""
x86 VM Execution Engine — high-level API for the X86CPU.

Provides program loading, breakpoint debugging, execution tracing,
event hooks, and I/O device management on top of the raw X86CPU.

Usage:
    engine = VMEngine()
    engine.load_source("mov eax, 1; mov ebx, 2; add eax, ebx; hlt")
    engine.run()
    print(engine.registers())
"""

from __future__ import annotations

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Any

from .vm import (
    X86CPU, X86Assembler, InsFault, Halt, MemFault,
    ProcessTable, ProcessControlBlock, ProcessState, Scheduler,
)

logger = logging.getLogger("slo.vm.engine")


# ── Event Types ──────────────────────────────────────────────────────────────

@dataclass
class Breakpoint:
    """A breakpoint set at a specific address or condition."""
    address: int
    enabled: bool = True
    hit_count: int = 0
    condition: Callable[[], bool] | None = None
    label: str = ""

    def should_trigger(self) -> bool:
        if not self.enabled:
            return False
        if self.condition is not None:
            try:
                if not self.condition():
                    return False
            except Exception:
                return False
        return True


@dataclass
class StepEvent:
    """Emitted after every instruction execution."""
    eip: int
    opcode: int
    registers: dict[str, int]
    flags: dict[str, bool]
    instruction_bytes: bytes


@dataclass
class BreakpointEvent:
    """Emitted when a breakpoint is hit."""
    breakpoint: Breakpoint
    eip: int
    registers: dict[str, int]


@dataclass
class FaultEvent:
    """Emitted on a CPU fault."""
    fault_type: type
    message: str
    eip: int
    registers: dict[str, int]


@dataclass
class SyscallEvent:
    """Emitted on a syscall instruction (INT 0x80)."""
    number: int
    args: dict[str, int]
    eip: int


@dataclass
class ExecutionTrace:
    """Complete trace of an execution run."""
    steps: list[StepEvent] = field(default_factory=list)
    breakpoints_hit: list[BreakpointEvent] = field(default_factory=list)
    faults: list[FaultEvent] = field(default_factory=list)
    syscalls: list[SyscallEvent] = field(default_factory=list)
    total_instructions: int = 0
    total_time_ms: float = 0.0
    exit_reason: str = ""


# ── I/O Devices ──────────────────────────────────────────────────────────────

class DeviceBus:
    """Simple I/O device bus — maps port numbers to device handlers."""

    def __init__(self):
        self._devices: dict[int, Callable[[int, int, int], None]] = {}
        self._readers: dict[int, Callable[[int], int]] = {}
        self._log: list[tuple[int, int, int]] = []

    def register_out(self, port: int, handler: Callable[[int, int, int], None]):
        self._devices[port] = handler

    def register_in(self, port: int, handler: Callable[[int], int]):
        self._readers[port] = handler

    def outb(self, port: int, value: int):
        self._log.append((port, value, 0))
        if port in self._devices:
            self._devices[port](port, value, 8)

    def outw(self, port: int, value: int):
        self._log.append((port, value, 1))
        if port in self._devices:
            self._devices[port](port, value, 16)

    def outd(self, port: int, value: int):
        self._log.append((port, value, 2))
        if port in self._devices:
            self._devices[port](port, value, 32)

    def inb(self, port: int) -> int:
        if port in self._readers:
            return self._readers[port](port) & 0xFF
        return 0

    def inw(self, port: int) -> int:
        if port in self._readers:
            return self._readers[port](port) & 0xFFFF
        return 0

    def ind(self, port: int) -> int:
        if port in self._readers:
            return self._readers[port](port) & 0xFFFFFFFF
        return 0

    @property
    def log(self) -> list[tuple[int, int, int]]:
        return list(self._log)


class ConsoleDevice:
    """Console I/O device — captures output, provides input."""

    def __init__(self):
        self.output: list[int] = []
        self.input_buffer: list[int] = []
        self.on_output: Callable[[int], None] | None = None

    def write_byte(self, port: int, value: int, width: int):
        self.output.append(value & 0xFF)
        if self.on_output:
            self.on_output(value & 0xFF)

    def read_byte(self, port: int) -> int:
        if self.input_buffer:
            return self.input_buffer.pop(0)
        return 0

    def feed_input(self, data: bytes):
        self.input_buffer.extend(data)


# ── VMEngine ─────────────────────────────────────────────────────────────────

class VMEngine:
    """
    High-level x86 VM execution engine.

    Wraps X86CPU with:
    - Source assembly loading and binary loading
    - Breakpoint debugging (address-based, conditional)
    - Execution tracing and profiling
    - Event hooks (on_step, on_breakpoint, on_fault)
    - I/O device bus
    - Process management (optional)
    """

    def __init__(self, memory_size: int = 0x400000):
        self._assembler = X86Assembler()
        self._cpu = X86CPU(memory_size=memory_size)
        self._device_bus = DeviceBus()
        self._console = ConsoleDevice()

        # Breakpoints
        self._breakpoints: dict[int, Breakpoint] = {}
        self._bp_counter = 0

        # Event hooks
        self._on_step: Callable[[StepEvent], None] | None = None
        self._on_breakpoint: Callable[[BreakpointEvent], None] | None = None
        self._on_fault: Callable[[FaultEvent], None] | None = None
        self._on_syscall: Callable[[SyscallEvent], None] | None = None
        self._on_halt: Callable[[int], None] | None = None

        # Execution state
        self._tracing = False
        self._trace = ExecutionTrace()
        self._stepping = False
        self._running = False
        self._halted = False
        self._break_requested = False
        self._skip_breakpoint_check = False

        # Process management (optional)
        self._process_table = ProcessTable()
        self._scheduler: Scheduler | None = None

        # Setup console I/O
        self._device_bus.register_out(0x3F8, self._console.write_byte)
        self._device_bus.register_in(0x3F8, self._console.read_byte)

        # Connect DeviceBus to CPU I/O handlers
        # CPU io_out expects fn(val), DeviceBus expects fn(port, val, width)
        self._cpu.register_io_out(0x3F8, lambda val: self._console.write_byte(0x3F8, val, 8))
        self._cpu.register_io_in(0x3F8, lambda: self._console.read_byte(0x3F8))

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def cpu(self) -> X86CPU:
        return self._cpu

    @property
    def assembler(self) -> X86Assembler:
        return self._assembler

    @property
    def devices(self) -> DeviceBus:
        return self._device_bus

    @property
    def console(self) -> ConsoleDevice:
        return self._console

    @property
    def process_table(self) -> ProcessTable:
        return self._process_table

    @property
    def trace(self) -> ExecutionTrace:
        return self._trace

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_halted(self) -> bool:
        return self._halted

    # ── Program Loading ──────────────────────────────────────────────────

    def load_source(self, source: str, org: int = 0x1000) -> int:
        """Assemble source and load into memory. Returns entry point."""
        code = self._assembler.assemble(source, org=org)
        self._cpu.load(bytes(code), org)
        self._cpu.eip = org
        return org

    def load_bytes(self, code: bytes, org: int = 0x1000) -> int:
        """Load raw bytes into memory. Returns entry point."""
        self._cpu.load(code, org)
        self._cpu.eip = org
        return org

    def load_file(self, path: str, org: int = 0x1000) -> int:
        """Load a binary file into memory."""
        with open(path, "rb") as f:
            code = f.read()
        return self.load_bytes(code, org)

    def set_entry(self, address: int):
        """Set the instruction pointer."""
        self._cpu.eip = address

    # ── Register Access ──────────────────────────────────────────────────

    def registers(self) -> dict[str, int]:
        """Return all general-purpose registers as a dict."""
        return {
            "eax": self._cpu.eax,
            "ecx": self._cpu.ecx,
            "edx": self._cpu.edx,
            "ebx": self._cpu.ebx,
            "esp": self._cpu.esp,
            "ebp": self._cpu.ebp,
            "esi": self._cpu.esi,
            "edi": self._cpu.edi,
            "eip": self._cpu.eip,
            "ax": self._cpu._get16(0),
            "cx": self._cpu._get16(1),
            "dx": self._cpu._get16(2),
            "bx": self._cpu._get16(3),
            "al": self._cpu._get8l(0),
            "cl": self._cpu._get8l(1),
            "dl": self._cpu._get8l(2),
            "bl": self._cpu._get8l(3),
            "ah": self._cpu._get8h(0),
            "ch": self._cpu._get8h(1),
            "dh": self._cpu._get8h(2),
            "bh": self._cpu._get8h(3),
        }

    def flags(self) -> dict[str, bool]:
        """Return all flags as a dict."""
        return {
            "cf": self._cpu.cf,
            "zf": self._cpu.zf,
            "sf": self._cpu.sf,
            "of": self._cpu.of,
        }

    def get_reg(self, name: str) -> int:
        """Get a single register by name."""
        name = name.lower()
        reg_map = {
            "eax": lambda: self._cpu.eax,
            "ecx": lambda: self._cpu.ecx,
            "edx": lambda: self._cpu.edx,
            "ebx": lambda: self._cpu.ebx,
            "esp": lambda: self._cpu.esp,
            "ebp": lambda: self._cpu.ebp,
            "esi": lambda: self._cpu.esi,
            "edi": lambda: self._cpu.edi,
            "eip": lambda: self._cpu.eip,
            "ax": lambda: self._cpu._get16(0),
            "cx": lambda: self._cpu._get16(1),
            "dx": lambda: self._cpu._get16(2),
            "bx": lambda: self._cpu._get16(3),
            "sp": lambda: self._cpu._get16(4),
            "bp": lambda: self._cpu._get16(5),
            "si": lambda: self._cpu._get16(6),
            "di": lambda: self._cpu._get16(7),
            "al": lambda: self._cpu._get8l(0),
            "cl": lambda: self._cpu._get8l(1),
            "dl": lambda: self._cpu._get8l(2),
            "bl": lambda: self._cpu._get8l(3),
            "ah": lambda: self._cpu._get8h(0),
            "ch": lambda: self._cpu._get8h(1),
            "dh": lambda: self._cpu._get8h(2),
            "bh": lambda: self._cpu._get8h(3),
        }
        if name in reg_map:
            return reg_map[name]()
        raise ValueError(f"Unknown register: {name}")

    def set_reg(self, name: str, value: int):
        """Set a single register by name."""
        name = name.lower()
        setter_map = {
            "eax": lambda v: setattr(self._cpu, "eax", v),
            "ecx": lambda v: setattr(self._cpu, "ecx", v),
            "edx": lambda v: setattr(self._cpu, "edx", v),
            "ebx": lambda v: setattr(self._cpu, "ebx", v),
            "esp": lambda v: setattr(self._cpu, "esp", v),
            "ebp": lambda v: setattr(self._cpu, "ebp", v),
            "esi": lambda v: setattr(self._cpu, "esi", v),
            "edi": lambda v: setattr(self._cpu, "edi", v),
            "eip": lambda v: setattr(self._cpu, "eip", v),
            "ax": lambda v: self._cpu._set16(0, v),
            "cx": lambda v: self._cpu._set16(1, v),
            "dx": lambda v: self._cpu._set16(2, v),
            "bx": lambda v: self._cpu._set16(3, v),
            "sp": lambda v: self._cpu._set16(4, v),
            "bp": lambda v: self._cpu._set16(5, v),
            "si": lambda v: self._cpu._set16(6, v),
            "di": lambda v: self._cpu._set16(7, v),
            "al": lambda v: self._cpu._set8l(0, v),
            "cl": lambda v: self._cpu._set8l(1, v),
            "dl": lambda v: self._cpu._set8l(2, v),
            "bl": lambda v: self._cpu._set8l(3, v),
            "ah": lambda v: self._cpu._set8h(0, v),
            "ch": lambda v: self._cpu._set8h(1, v),
            "dh": lambda v: self._cpu._set8h(2, v),
            "bh": lambda v: self._cpu._set8h(3, v),
        }
        if name in setter_map:
            setter_map[name](value & 0xFFFFFFFF)
        else:
            raise ValueError(f"Unknown register: {name}")

    # ── Memory Access ────────────────────────────────────────────────────

    def read_memory(self, address: int, size: int = 4) -> bytes:
        """Read bytes from memory."""
        result = bytearray()
        for i in range(size):
            addr = (address + i) & 0xFFFFFFFF
            if addr < self._cpu._mem_size:
                result.append(self._cpu._mem[addr])
            else:
                result.append(0)
        return bytes(result)

    def write_memory(self, address: int, data: bytes):
        """Write bytes to memory."""
        for i, b in enumerate(data):
            addr = (address + i) & 0xFFFFFFFF
            if addr < self._cpu._mem_size:
                self._cpu._mem[addr] = b

    def read_dword(self, address: int) -> int:
        """Read a 32-bit dword from memory."""
        return self._cpu._read32(address)

    def write_dword(self, address: int, value: int):
        """Write a 32-bit dword to memory."""
        self._cpu._write32(address, value)

    def read_word(self, address: int) -> int:
        """Read a 16-bit word from memory."""
        return self._cpu._read16(address)

    def write_word(self, address: int, value: int):
        """Write a 16-bit word to memory."""
        self._cpu._write16(address, value)

    def read_byte(self, address: int) -> int:
        """Read a single byte from memory."""
        addr = address & 0xFFFFFFFF
        if addr >= self._cpu._mem_size:
            return 0
        return self._cpu._mem[addr]

    def write_byte(self, address: int, value: int):
        """Write a single byte to memory."""
        addr = address & 0xFFFFFFFF
        if addr >= self._cpu._mem_size:
            return
        self._cpu._mem[addr] = value & 0xFF

    def dump_memory(self, address: int, length: int = 64) -> str:
        """Hex dump of memory region."""
        lines = []
        for offset in range(0, length, 16):
            addr = address + offset
            chunk = self.read_memory(addr, min(16, length - offset))
            hex_str = " ".join(f"{b:02x}" for b in chunk)
            ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            lines.append(f"  {addr:08x}  {hex_str:<48s}  {ascii_str}")
        return "\n".join(lines)

    # ── Breakpoints ──────────────────────────────────────────────────────

    def set_breakpoint(self, address: int, label: str = "",
                       condition: Callable[[], bool] | None = None) -> int:
        """Set a breakpoint. Returns breakpoint ID."""
        self._bp_counter += 1
        bp = Breakpoint(address=address, label=label, condition=condition)
        self._breakpoints[self._bp_counter] = bp
        return self._bp_counter

    def set_breakpoint_once(self, address: int, label: str = "") -> int:
        """Set a one-shot breakpoint (auto-disables after first hit)."""
        bp_id = self.set_breakpoint(address, label)
        original = self._breakpoints[bp_id]
        original.condition = lambda: True
        # Wrap to auto-disable
        orig_trigger = original.should_trigger
        def auto_disable():
            if orig_trigger():
                original.enabled = False
                return True
            return False
        original.should_trigger = auto_disable
        return bp_id

    def remove_breakpoint(self, bp_id: int):
        """Remove a breakpoint by ID."""
        self._breakpoints.pop(bp_id, None)

    def enable_breakpoint(self, bp_id: int):
        """Enable a breakpoint."""
        if bp_id in self._breakpoints:
            self._breakpoints[bp_id].enabled = True

    def disable_breakpoint(self, bp_id: int):
        """Disable a breakpoint."""
        if bp_id in self._breakpoints:
            self._breakpoints[bp_id].enabled = False

    def clear_breakpoints(self):
        """Remove all breakpoints."""
        self._breakpoints.clear()

    def list_breakpoints(self) -> list[dict]:
        """List all breakpoints."""
        return [
            {"id": bp_id, "address": bp.address, "label": bp.label,
             "enabled": bp.enabled, "hit_count": bp.hit_count}
            for bp_id, bp in self._breakpoints.items()
        ]

    # ── Event Hooks ──────────────────────────────────────────────────────

    def on_step(self, callback: Callable[[StepEvent], None]):
        """Register a callback for every instruction step."""
        self._on_step = callback

    def on_breakpoint(self, callback: Callable[[BreakpointEvent], None]):
        """Register a callback when a breakpoint is hit."""
        self._on_breakpoint = callback

    def on_fault(self, callback: Callable[[FaultEvent], None]):
        """Register a callback on CPU fault."""
        self._on_fault = callback

    def on_syscall(self, callback: Callable[[SyscallEvent], None]):
        """Register a callback on syscall instruction."""
        self._on_syscall = callback

    def on_halt(self, callback: Callable[[int], None]):
        """Register a callback on HLT instruction."""
        self._on_halt = callback

    # ── Execution ────────────────────────────────────────────────────────

    def step(self) -> bool:
        """Execute a single instruction. Returns False on fault/halt."""
        if self._trace is None:
            self._trace = ExecutionTrace()
        eip = self._cpu.eip

        # Check breakpoints before execution
        if not self._skip_breakpoint_check:
            for bp_id, bp in self._breakpoints.items():
                if bp.address == eip and bp.should_trigger():
                    bp.hit_count += 1
                    bp_event = BreakpointEvent(
                        breakpoint=bp,
                        eip=eip,
                        registers=self.registers(),
                    )
                    if self._on_breakpoint:
                        self._on_breakpoint(bp_event)
                    self._trace.breakpoints_hit.append(bp_event)
                    return False

        # Emit step event if tracing
        if self._tracing or self._on_step:
            opcode = self._cpu._mem[eip & 0xFFFFFFFF]
            event = StepEvent(
                eip=eip,
                opcode=opcode,
                registers=self.registers(),
                flags=self.flags(),
                instruction_bytes=self.read_memory(eip, 4),
            )
            if self._on_step:
                self._on_step(event)
            if self._tracing:
                self._trace.steps.append(event)
                self._trace.total_instructions += 1

        # Execute
        try:
            result = self._cpu.step()
        except (InsFault, MemFault) as e:
            fault_event = FaultEvent(
                fault_type=type(e),
                message=f"{type(e).__name__}: {e}",
                eip=eip,
                registers=self.registers(),
            )
            if self._on_fault:
                self._on_fault(fault_event)
            self._trace.faults.append(fault_event)
            return False

        # X86CPU.step() returns False on HLT or unexpected error
        if not result:
            # Check if it was a HLT instruction (opcode 0xF4)
            opcode = self._cpu._mem[eip & 0xFFFFFFFF]
            if opcode == 0xF4 or opcode == 0x66:
                # 0x66 prefix: check if the next byte is 0xF4 (HLT in 16-bit mode)
                if opcode == 0x66:
                    next_op = self._cpu._mem[(eip + 1) & 0xFFFFFFFF]
                    if next_op != 0xF4:
                        # Not a 16-bit HLT — some other 0x66-prefixed fault
                        fault_event = FaultEvent(
                            fault_type=InsFault,
                            message="unexpected CPU error (check server logs)",
                            eip=eip,
                            registers=self.registers(),
                        )
                        if self._on_fault:
                            self._on_fault(fault_event)
                        self._trace.faults.append(fault_event)
                        return False
                self._halted = True
                if self._on_halt:
                    self._on_halt(eip)
            else:
                # Unexpected Python exception caught by X86CPU.step()
                fault_event = FaultEvent(
                    fault_type=InsFault,
                    message="unexpected CPU error (check server logs)",
                    eip=eip,
                    registers=self.registers(),
                )
                if self._on_fault:
                    self._on_fault(fault_event)
                self._trace.faults.append(fault_event)
            return False

        return True

    def run(self, max_steps: int = 0) -> ExecutionTrace:
        """
        Run until HLT, fault, breakpoint, or max_steps.

        Returns the execution trace.
        """
        self._running = True
        self._halted = False
        self._break_requested = False
        self._trace = ExecutionTrace()
        start_time = time.monotonic()

        try:
            steps = 0
            while not self._halted and not self._break_requested:
                # Check breakpoints
                eip = self._cpu.eip
                for bp_id, bp in self._breakpoints.items():
                    if bp.address == eip and bp.should_trigger():
                        bp.hit_count += 1
                        bp_event = BreakpointEvent(
                            breakpoint=bp,
                            eip=eip,
                            registers=self.registers(),
                        )
                        if self._on_breakpoint:
                            self._on_breakpoint(bp_event)
                        if self._tracing:
                            self._trace.breakpoints_hit.append(bp_event)
                        self._trace.exit_reason = "breakpoint"
                        self._running = False
                        return self._trace

                # Execute one step
                if not self.step():
                    break

                steps += 1
                if max_steps > 0 and steps >= max_steps:
                    self._trace.exit_reason = "max_steps"
                    break

        finally:
            elapsed = (time.monotonic() - start_time) * 1000
            self._trace.total_time_ms = elapsed
            if not self._trace.exit_reason:
                if self._halted:
                    self._trace.exit_reason = "halt"
                elif self._break_requested:
                    self._trace.exit_reason = "break_request"
                else:
                    self._trace.exit_reason = "fault"
            self._running = False

        return self._trace

    def step_over(self) -> bool:
        """Step over a CALL instruction (execute CALL target as a unit)."""
        eip = self._cpu.eip
        opcode = self._cpu._mem[eip & 0xFFFFFFFF]

        # CALL rel32 (E8 xx xx xx xx) — step over by running until EIP past the call
        if opcode == 0xE8:
            # CALL is 5 bytes
            call_end = eip + 5
            while self._cpu.eip != call_end and not self._halted:
                if not self.step():
                    return False
            return True

        # Otherwise just single step
        return self.step()

    def step_out(self) -> bool:
        """Run until the current function returns (RET instruction)."""
        call_depth = 0
        while not self._halted:
            eip = self._cpu.eip
            opcode = self._cpu._mem[eip & 0xFFFFFFFF]

            if opcode == 0xC3 or opcode == 0xC2:  # RET
                if call_depth == 0:
                    return self.step()
                call_depth -= 1
            elif opcode == 0xE8:  # CALL
                call_depth += 1

            if not self.step():
                return False
        return False

    def continue_execution(self) -> ExecutionTrace:
        """Continue execution after a breakpoint hit."""
        if self._halted:
            return self._trace
        # Temporarily disable breakpoint checking to step past the current one
        self._skip_breakpoint_check = True
        try:
            self.step()
        finally:
            self._skip_breakpoint_check = False
        return self.run()

    def request_break(self):
        """Request a break from the current execution loop."""
        self._break_requested = True

    # ── Tracing ──────────────────────────────────────────────────────────

    def enable_tracing(self):
        """Enable instruction-level tracing."""
        self._tracing = True

    def disable_tracing(self):
        """Disable instruction-level tracing."""
        self._tracing = False

    def get_trace_summary(self) -> dict:
        """Get a summary of the execution trace."""
        t = self._trace
        return {
            "total_instructions": t.total_instructions,
            "total_time_ms": round(t.total_time_ms, 2),
            "breakpoints_hit": len(t.breakpoints_hit),
            "faults": len(t.faults),
            "syscalls": len(t.syscalls),
            "exit_reason": t.exit_reason,
            "instructions_per_ms": (
                round(t.total_instructions / t.total_time_ms, 1)
                if t.total_time_ms > 0 else 0
            ),
        }

    # ── Process Management ───────────────────────────────────────────────

    def create_process(self, name: str = "unnamed",
                       priority: int = 0) -> ProcessControlBlock:
        """Create a new process."""
        return self._process_table.create(name=name, priority=priority)

    def switch_to_process(self, pid: int):
        """Switch CPU state to a different process."""
        pcb = self._process_table.get(pid)
        if pcb is None:
            raise ValueError(f"Process {pid} not found")
        # Save current state if running
        if self._scheduler and self._scheduler.current:
            self._scheduler.current.save_from_cpu(self._cpu)
        # Load new process
        pcb.restore_to_cpu(self._cpu)
        pcb.state = ProcessState.RUNNING

    # ── Disassembly ──────────────────────────────────────────────────────

    def disassemble(self, address: int, count: int = 10) -> list[str]:
        """Disassemble instructions at the given address (non-destructive)."""
        lines = []
        ip = address & 0xFFFFFFFF
        for _ in range(count):
            if ip >= self._cpu._mem_size:
                break
            opcode = self._cpu._mem[ip]
            # Map 0x50-0x5F to PUSH/POP r32
            name = self._opcode_name(opcode)
            # Compute instruction length (simplified)
            length = self._instruction_length(opcode, ip)
            lines.append(f"  0x{ip:08x}: {name}")
            ip = (ip + length) & 0xFFFFFFFF
        return lines

    def _instruction_length(self, opcode: int, ip: int) -> int:
        """Estimate instruction length without executing."""
        if opcode in (0x90, 0xF4, 0xC3, 0xCC, 0xFA, 0xFB, 0xFC, 0xFD):
            return 1
        if opcode in (0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57,
                       0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F):
            return 1
        if opcode in (0x66, 0xF0, 0xF2, 0xF3):
            return 2
        if opcode == 0xE8:
            return 5
        if opcode == 0xEB or opcode == 0x74 or opcode == 0x75:
            return 2
        if 0xB0 <= opcode <= 0xBF:
            return 2
        if opcode == 0xC2 or opcode == 0xCA:
            return 3
        if opcode == 0x68:
            return 5
        if opcode == 0x6A:
            return 2
        if opcode == 0xCD:
            return 2
        if opcode == 0xC6 or opcode == 0x80 or opcode == 0x83:
            return 6
        if opcode == 0x81:
            return 6
        if 0x88 <= opcode <= 0x8B:
            return 2
        if opcode == 0xFF:
            return 2
        return 1

    def _opcode_name(self, opcode: int) -> str:
        """Get a human-readable name for common opcodes."""
        names = {
            0x90: "NOP",
            0xF4: "HLT",
            0xC3: "RET",
            0xC2: "RET imm16",
            0xCC: "INT3",
            0xCD: "INT imm8",
            0xFA: "CLI",
            0xFB: "STI",
            0xFC: "CLD",
            0xFD: "STD",
            0x50: "PUSH EAX", 0x51: "PUSH ECX", 0x52: "PUSH EDX",
            0x53: "PUSH EBX", 0x54: "PUSH ESP", 0x55: "PUSH EBP",
            0x56: "PUSH ESI", 0x57: "PUSH EDI",
            0x58: "POP EAX", 0x59: "POP ECX", 0x5A: "POP EDX",
            0x5B: "POP EBX", 0x5C: "POP ESP", 0x5D: "POP EBP",
            0x5E: "POP ESI", 0x5F: "POP EDI",
        }
        return names.get(opcode, f"OP 0x{opcode:02X}")

    # ── Utility ──────────────────────────────────────────────────────────

    def reset(self):
        """Reset the engine to a clean state."""
        self._cpu = X86CPU(memory_size=self._cpu._mem_size)
        self._breakpoints.clear()
        self._bp_counter = 0
        self._trace = ExecutionTrace()
        self._halted = False
        self._running = False
        self._break_requested = False
        self._tracing = False
        self._console.output.clear()

    def state_snapshot(self) -> dict:
        """Capture a complete state snapshot for debugging."""
        esp = self._cpu.esp
        stack_valid = 0 <= esp < self._cpu._mem_size
        return {
            "registers": self.registers(),
            "flags": self.flags(),
            "stack": self.dump_memory(esp, 32) if stack_valid else "(invalid ESP)",
            "breakpoints": self.list_breakpoints(),
            "trace_summary": self.get_trace_summary() if self._trace.total_instructions > 0 else None,
        }

    def __repr__(self):
        return (
            f"VMEngine(eip=0x{self._cpu.eip:08X}, "
            f"esp=0x{self._cpu.esp:08X}, "
            f"bps={len(self._breakpoints)}, "
            f"running={self._running})"
        )
