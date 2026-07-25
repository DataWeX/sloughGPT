"""
Virtual Hardware Layer — TinkerCAD-style hardware simulation.

Simulates a complete computer system:
  VirtualCPU      — configurable registers, ALU, control flow
  VirtualRAM      — byte-addressable memory with allocation tracking
  VirtualBus      — memory-mapped I/O connecting CPU to devices
  Device          — base class for all virtual hardware
  TensorAccel     — GPU-like tensor accelerator (memory-mapped command interface)
  BlockStorage    — disk-like block storage
  MemoryManager   — tiered storage with auto-swap (registers → RAM → disk)
  VirtualSystem   — wires all components together, runs the simulation

Architecture:
  CPU executes ISA instructions. When it hits IN/OUT, the bus routes to devices.
  Devices are memory-mapped: each has an address range on the bus.
  The CPU never knows what a device IS — it just reads/writes addresses.

  ┌──────┐    ┌─────┐    ┌────────────────────┐
  │ CPU  │◄──►│ Bus │◄──►│ RAM | GPU | Storage │
  └──────┘    └─────┘    └────────────────────┘

This layer has ZERO knowledge of AI, models, or domain logic.
AI operations are executed by devices (e.g. TensorAccel),
not by the CPU itself.
"""

from __future__ import annotations

import time
import struct
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("slo.shell.hardware")

# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_RAM_SIZE = 1024 * 1024  # 1 MB
DEFAULT_REG_COUNT = 16
DEFAULT_CLOCK_MHZ = 1000
MAX_INSTRUCTIONS = 100_000
MAX_CALL_DEPTH = 256


# ── Device Base ────────────────────────────────────────────────────────────


class Device:
    """Base class for all virtual hardware devices.

    Subclass this to create new device types (GPU, storage, sensors, etc.).
    The bus routes I/O operations to devices based on address ranges.
    """

    name: str = "device"
    size: int = 256  # address space size (bytes) mapped on the bus

    def __init__(self, base_addr: int = 0):
        self.base_addr = base_addr
        self._registers = bytearray(self.size)

    def read_byte(self, offset: int) -> int:
        """Read a single byte from device register at offset."""
        if 0 <= offset < self.size:
            return self._registers[offset]
        return 0

    def write_byte(self, offset: int, value: int) -> None:
        """Write a single byte to device register at offset."""
        if 0 <= offset < self.size:
            self._registers[offset] = value & 0xFF

    def read_word(self, offset: int) -> int:
        """Read a 32-bit word from device register at offset."""
        lo = self.read_byte(offset)
        hi = self.read_byte(offset + 1) if offset + 1 < self.size else 0
        return lo | (hi << 8)

    def write_word(self, offset: int, value: int) -> None:
        """Write a 32-bit word to device register at offset."""
        self.write_byte(offset, value & 0xFF)
        self.write_byte(offset + 1, (value >> 8) & 0xFF)

    def control(self, command: int, *args: Any) -> Any:
        """Send a control command to the device.

        Override in subclasses for device-specific commands.
        """
        return None

    def tick(self) -> None:
        """Called every CPU cycle. Override for time-dependent behavior."""

    def reset(self) -> None:
        """Reset device to initial state."""
        self._registers = bytearray(self.size)


# ── Virtual RAM ────────────────────────────────────────────────────────────


class VirtualRAM(Device):
    """Byte-addressable random access memory.

    Simulates physical RAM with allocation tracking and tiered storage.
    """

    name = "ram"

    def __init__(self, size_bytes: int = DEFAULT_RAM_SIZE):
        super().__init__(base_addr=0)
        self.size = size_bytes
        self._data = bytearray(size_bytes)
        self._allocations: dict[str, tuple[int, int]] = {}  # name → (addr, size)
        self._free_list: list[tuple[int, int]] = [(0, size_bytes)]  # (addr, size)
        self._total_allocated = 0

    def read_byte(self, offset: int) -> int:
        if 0 <= offset < self.size:
            return self._data[offset]
        return 0

    def write_byte(self, offset: int, value: int) -> None:
        if 0 <= offset < self.size:
            self._data[offset] = value & 0xFF

    def read_block(self, addr: int, size: int) -> bytes:
        """Read a block of bytes from RAM."""
        return bytes(self._data[addr:addr + size])

    def write_block(self, addr: int, data: bytes) -> None:
        """Write a block of bytes to RAM."""
        end = min(addr + len(data), self.size)
        self._data[addr:end] = data[:end - addr]

    def alloc(self, name: str, size: int) -> int:
        """Allocate a named block in RAM. Returns the address."""
        # Find first fit in free list
        for i, (faddr, fsize) in enumerate(self._free_list):
            if fsize >= size:
                # Allocate from this block
                self._free_list.pop(i)
                if fsize > size:
                    # Split: remaining goes back to free list
                    self._free_list.append((faddr + size, fsize - size))
                self._allocations[name] = (faddr, size)
                self._total_allocated += size
                return faddr
        raise MemoryError(f"RAM: no contiguous block of {size} bytes available")

    def free(self, name: str) -> None:
        """Free a named allocation."""
        if name not in self._allocations:
            return
        addr, size = self._allocations.pop(name)
        self._total_allocated -= size
        # Add to free list and merge adjacent blocks
        self._free_list.append((addr, size))
        self._free_list.sort()
        merged = []
        for faddr, fsize in self._free_list:
            if merged and merged[-1][0] + merged[-1][1] == faddr:
                merged[-1] = (merged[-1][0], merged[-1][1] + fsize)
            else:
                merged.append((faddr, fsize))
        self._free_list = merged

    def get_block(self, name: str) -> tuple[int, int] | None:
        """Get (addr, size) for a named allocation."""
        return self._allocations.get(name)

    @property
    def usage(self) -> dict[str, Any]:
        """Memory usage statistics."""
        return {
            "total": self.size,
            "allocated": self._total_allocated,
            "free": self.size - self._total_allocated,
            "fragments": len(self._free_list),
            "allocations": len(self._allocations),
        }


# ── Virtual Bus ────────────────────────────────────────────────────────────


class VirtualBus:
    """Memory-mapped I/O bus connecting CPU to devices.

    Each device occupies an address range on the bus.
    The CPU reads/writes addresses; the bus routes to the correct device.
    """

    def __init__(self):
        self._devices: list[tuple[int, int, Device]] = []  # (base, end, device)
        self._io_space: dict[int, Device] = {}  # port → device (for IN/OUT)

    def attach(self, device: Device, base_addr: int | None = None) -> None:
        """Attach a device to the bus at a base address."""
        if base_addr is None:
            # Auto-assign: after the last attached device
            if self._devices:
                last_base, last_size, _ = self._devices[-1]
                base_addr = last_base + last_size
            else:
                base_addr = 0
        device.base_addr = base_addr
        end = base_addr + device.size
        self._devices.append((base_addr, end, device))
        self._devices.sort(key=lambda x: x[0])
        logger.debug("bus: attached %s at 0x%04X-0x%04X", device.name, base_addr, end - 1)

    def attach_io(self, port: int, device: Device) -> None:
        """Attach a device to an I/O port for IN/OUT instructions."""
        self._io_space[port] = device

    def read(self, addr: int) -> int:
        """Read a byte from the bus at address."""
        for base, end, dev in self._devices:
            if base <= addr < end:
                return dev.read_byte(addr - base)
        return 0

    def write(self, addr: int, value: int) -> None:
        """Write a byte to the bus at address."""
        for base, end, dev in self._devices:
            if base <= addr < end:
                dev.write_byte(addr - base, value)
                return

    def read_io(self, port: int) -> int:
        """Read from an I/O port (IN instruction)."""
        dev = self._io_space.get(port)
        if dev:
            return dev.read_byte(0)
        return 0

    def write_io(self, port: int, value: int) -> None:
        """Write to an I/O port (OUT instruction)."""
        dev = self._io_space.get(port)
        if dev:
            dev.write_byte(0, value)

    def read_word(self, addr: int) -> int:
        """Read a 16-bit word from the bus."""
        lo = self.read(addr)
        hi = self.read(addr + 1)
        return lo | (hi << 8)

    def write_word(self, addr: int, value: int) -> None:
        """Write a 16-bit word to the bus."""
        self.write(addr, value & 0xFF)
        self.write(addr + 1, (value >> 8) & 0xFF)

    def tick_all(self) -> None:
        """Tick all attached devices (called every CPU cycle)."""
        for _, _, dev in self._devices:
            dev.tick()

    def reset_all(self) -> None:
        """Reset all attached devices."""
        for _, _, dev in self._devices:
            dev.reset()


# ── Virtual CPU ────────────────────────────────────────────────────────────


class VirtualCPU:
    """Simulates a CPU with registers, ALU, and control flow.

    Executes ISA instructions. No AI knowledge — just registers, integers,
    floats, and control flow. Device I/O goes through the bus.

    ISA:
      Load/Move:   LOAD_CONST, MOV, STORE, LOAD, FREE, PRINT
      Integer ALU: IADD, ISUB, IMUL, IDIV, IAND, IOR, IXOR, ISHL, ISHR, INEG, INC, DEC
      Float ALU:   FADD, FSUB, FMUL, FDIV
      Comparison:  CMP, TEST, ICMP, FCMP
      Control:     JMP, JZ, JNZ, JGT, JGE, JLT, JLE, CALL, RET, LOOP, HALT
      I/O:         IN, OUT (to bus/devices)
      Memory:      ALLOC, MEMINFO
    """

    def __init__(self, num_regs: int = DEFAULT_REG_COUNT, bus: VirtualBus | None = None):
        self.regs = [0] * num_regs
        self.pc = 0
        self.sp = 0  # stack pointer (for CALL/RET)
        self.flags = {"zero": False, "negative": False, "carry": False}
        self.bus = bus or VirtualBus()
        self.program: list[Any] = []
        self._labels: dict[str, int] = {}
        self._running = False
        self._step_count = 0
        self._call_depth = 0
        self._stack: list[int] = []  # return address stack for CALL/RET
        self._output: list[str] = []  # captured output from PRINT
        self._heap: dict[str, int] = {}  # named allocations → address
        self._max_instructions = MAX_INSTRUCTIONS

    def load_program(self, instructions: list, labels: dict[str, int] | None = None) -> None:
        """Load instructions and optional label map."""
        self.program = instructions
        self._labels = labels or {}
        self.pc = 0

    def run(self) -> list[str]:
        """Execute the program. Returns list of PRINT output strings."""
        self._running = True
        self._step_count = 0
        self._call_depth = 0
        self._stack = []
        self._output = []

        while self._running and self.pc < len(self.program):
            if self._step_count >= self._max_instructions:
                self._output.append(f"instruction limit reached ({self._max_instructions})")
                break

            inst = self.program[self.pc]
            self._execute(inst)
            self.bus.tick_all()
            self._step_count += 1

        return self._output

    def step(self) -> bool:
        """Execute a single instruction. Returns False if halted."""
        if not self._running or self.pc >= len(self.program):
            return False
        if self._step_count >= self._max_instructions:
            return False
        inst = self.program[self.pc]
        self._execute(inst)
        self.bus.tick_all()
        self._step_count += 1
        return self._running

    def _execute(self, inst: Any) -> None:
        """Execute a single instruction."""
        opcode = inst.opcode
        ops = inst.operands

        handler = _INSTRUCTION_TABLE.get(opcode)
        if handler:
            handler(self, ops)
        else:
            self._output.append(f"unknown opcode: {opcode}")

    def _check_arity(self, ops: list, expected: int) -> None:
        if len(ops) < expected:
            self._output.append(f"operand count mismatch for opcode (expected {expected})")

    def _reg(self, operand: Any) -> int:
        """Resolve a register operand to register index."""
        if isinstance(operand, str) and operand.startswith("R"):
            return int(operand[1:])
        return int(operand)

    def _val(self, operand: Any) -> int | float:
        """Resolve an operand to a value."""
        if isinstance(operand, str) and operand.startswith("R"):
            idx = int(operand[1:])
            return self.regs[idx]
        if isinstance(operand, (int, float)):
            return operand
        return 0

    def _resolve_label(self, operand: Any) -> int:
        """Resolve a label reference to an instruction index."""
        if isinstance(operand, int):
            return operand
        if isinstance(operand, str):
            if operand in self._labels:
                return self._labels[operand]
            try:
                return int(operand)
            except ValueError:
                self._output.append(f"undefined label: {operand}")
                return self.pc + 1
        return int(operand)

    def _set_flags(self, value: int) -> None:
        """Set CPU flags based on a result value."""
        self.flags["zero"] = value == 0
        self.flags["negative"] = value < 0

    def _truthy(self, val: Any) -> bool:
        """Check if a value is truthy."""
        if isinstance(val, (int, float)):
            return val != 0
        if isinstance(val, str):
            return len(val) > 0
        if isinstance(val, np.ndarray):
            return val.size > 0 and np.any(val != 0)
        return bool(val)


# ── Instruction Handlers ───────────────────────────────────────────────────


def _op_load_const(cpu: VirtualCPU, ops: list) -> None:
    """LOAD_CONST Rd, value"""
    cpu._check_arity(ops, 2)
    rd = cpu._reg(ops[0])
    val = ops[1]
    cpu.regs[rd] = val


def _op_mov(cpu: VirtualCPU, ops: list) -> None:
    """MOV Rd, Rs"""
    cpu._check_arity(ops, 2)
    rd = cpu._reg(ops[0])
    rs = cpu._reg(ops[1])
    cpu.regs[rd] = cpu.regs[rs]


def _op_store(cpu: VirtualCPU, ops: list) -> None:
    """STORE Rs, name — heap[name] = Rs"""
    cpu._check_arity(ops, 2)
    rs = cpu._reg(ops[0])
    name = str(ops[1])
    cpu._heap[name] = cpu.regs[rs]


def _op_load(cpu: VirtualCPU, ops: list) -> None:
    """LOAD Rd, name — Rd = heap[name]"""
    cpu._check_arity(ops, 2)
    rd = cpu._reg(ops[0])
    name = str(ops[1])
    if name not in cpu._heap:
        cpu._output.append(f"heap key not found: {name}")
        return
    cpu.regs[rd] = cpu._heap[name]


def _op_free(cpu: VirtualCPU, ops: list) -> None:
    """FREE name"""
    cpu._check_arity(ops, 1)
    name = str(ops[0])
    cpu._heap.pop(name, None)


def _op_print(cpu: VirtualCPU, ops: list) -> None:
    """PRINT Rs"""
    cpu._check_arity(ops, 1)
    val = cpu._val(ops[0])
    if isinstance(val, np.ndarray):
        cpu._output.append(np.array2string(val, precision=4, suppress_small=True))
    else:
        cpu._output.append(str(val))


# ── Integer ALU ────────────────────────────────────────────────────────────


def _op_iadd(cpu: VirtualCPU, ops: list) -> None:
    """IADD Rd, Ra, Rb"""
    cpu._check_arity(ops, 3)
    rd, a, b = cpu._reg(ops[0]), int(cpu._val(ops[1])), int(cpu._val(ops[2]))
    cpu.regs[rd] = a + b


def _op_isub(cpu: VirtualCPU, ops: list) -> None:
    """ISUB Rd, Ra, Rb"""
    cpu._check_arity(ops, 3)
    rd, a, b = cpu._reg(ops[0]), int(cpu._val(ops[1])), int(cpu._val(ops[2]))
    cpu.regs[rd] = a - b


def _op_imul(cpu: VirtualCPU, ops: list) -> None:
    """IMUL Rd, Ra, Rb"""
    cpu._check_arity(ops, 3)
    rd, a, b = cpu._reg(ops[0]), int(cpu._val(ops[1])), int(cpu._val(ops[2]))
    cpu.regs[rd] = a * b


def _op_idiv(cpu: VirtualCPU, ops: list) -> None:
    """IDIV Rd, Ra, Rb"""
    cpu._check_arity(ops, 3)
    rd, a, b = cpu._reg(ops[0]), int(cpu._val(ops[1])), int(cpu._val(ops[2]))
    cpu.regs[rd] = a // b if b != 0 else 0


def _op_iand(cpu: VirtualCPU, ops: list) -> None:
    """IAND Rd, Ra, Rb"""
    cpu._check_arity(ops, 3)
    rd, a, b = cpu._reg(ops[0]), int(cpu._val(ops[1])), int(cpu._val(ops[2]))
    cpu.regs[rd] = a & b


def _op_ior(cpu: VirtualCPU, ops: list) -> None:
    """IOR Rd, Ra, Rb"""
    cpu._check_arity(ops, 3)
    rd, a, b = cpu._reg(ops[0]), int(cpu._val(ops[1])), int(cpu._val(ops[2]))
    cpu.regs[rd] = a | b


def _op_ixor(cpu: VirtualCPU, ops: list) -> None:
    """IXOR Rd, Ra, Rb"""
    cpu._check_arity(ops, 3)
    rd, a, b = cpu._reg(ops[0]), int(cpu._val(ops[1])), int(cpu._val(ops[2]))
    cpu.regs[rd] = a ^ b


def _op_ishl(cpu: VirtualCPU, ops: list) -> None:
    """ISHL Rd, Ra, Rb"""
    cpu._check_arity(ops, 3)
    rd, a, b = cpu._reg(ops[0]), int(cpu._val(ops[1])), int(cpu._val(ops[2]))
    cpu.regs[rd] = a << b


def _op_ishr(cpu: VirtualCPU, ops: list) -> None:
    """ISHR Rd, Ra, Rb"""
    cpu._check_arity(ops, 3)
    rd, a, b = cpu._reg(ops[0]), int(cpu._val(ops[1])), int(cpu._val(ops[2]))
    cpu.regs[rd] = a >> b


def _op_ineg(cpu: VirtualCPU, ops: list) -> None:
    """INEG Rd, Ra"""
    cpu._check_arity(ops, 2)
    rd = cpu._reg(ops[0])
    cpu.regs[rd] = -int(cpu._val(ops[1]))


def _op_inc(cpu: VirtualCPU, ops: list) -> None:
    """INC Rd"""
    cpu._check_arity(ops, 1)
    rd = cpu._reg(ops[0])
    cpu.regs[rd] = int(cpu.regs[rd]) + 1


def _op_dec(cpu: VirtualCPU, ops: list) -> None:
    """DEC Rd"""
    cpu._check_arity(ops, 1)
    rd = cpu._reg(ops[0])
    cpu.regs[rd] = int(cpu.regs[rd]) - 1


# ── Float ALU ──────────────────────────────────────────────────────────────


def _op_fadd(cpu: VirtualCPU, ops: list) -> None:
    """FADD Rd, Ra, Rb"""
    cpu._check_arity(ops, 3)
    rd = cpu._reg(ops[0])
    a, b = float(cpu._val(ops[1])), float(cpu._val(ops[2]))
    cpu.regs[rd] = a + b


def _op_fsub(cpu: VirtualCPU, ops: list) -> None:
    """FSUB Rd, Ra, Rb"""
    cpu._check_arity(ops, 3)
    rd = cpu._reg(ops[0])
    a, b = float(cpu._val(ops[1])), float(cpu._val(ops[2]))
    cpu.regs[rd] = a - b


def _op_fmul(cpu: VirtualCPU, ops: list) -> None:
    """FMUL Rd, Ra, Rb"""
    cpu._check_arity(ops, 3)
    rd = cpu._reg(ops[0])
    a, b = float(cpu._val(ops[1])), float(cpu._val(ops[2]))
    cpu.regs[rd] = a * b


def _op_fdiv(cpu: VirtualCPU, ops: list) -> None:
    """FDIV Rd, Ra, Rb"""
    cpu._check_arity(ops, 3)
    rd = cpu._reg(ops[0])
    a, b = float(cpu._val(ops[1])), float(cpu._val(ops[2]))
    cpu.regs[rd] = a / b if b != 0 else 0.0


# ── Comparison ─────────────────────────────────────────────────────────────


def _op_cmp(cpu: VirtualCPU, ops: list) -> None:
    """CMP Ra, Rb — set flags from comparison"""
    cpu._check_arity(ops, 2)
    a, b = cpu._val(ops[0]), cpu._val(ops[1])
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        diff = float(a) - float(b)
        cpu.flags["zero"] = diff == 0
        cpu.flags["negative"] = diff < 0
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        cpu.flags["zero"] = np.array_equal(a, b)
        cpu.flags["negative"] = False
    else:
        s_a, s_b = str(a), str(b)
        cpu.flags["zero"] = s_a == s_b
        cpu.flags["negative"] = s_a < s_b


def _op_icmp(cpu: VirtualCPU, ops: list) -> None:
    """ICMP Ra, Rb — integer comparison, sets flags"""
    cpu._check_arity(ops, 2)
    a, b = int(cpu._val(ops[0])), int(cpu._val(ops[1]))
    diff = a - b
    cpu.flags["zero"] = diff == 0
    cpu.flags["negative"] = diff < 0


def _op_fcmp(cpu: VirtualCPU, ops: list) -> None:
    """FCMP Ra, Rb — float comparison, sets flags"""
    cpu._check_arity(ops, 2)
    a, b = float(cpu._val(ops[0])), float(cpu._val(ops[1]))
    diff = a - b
    cpu.flags["zero"] = abs(diff) < 1e-10
    cpu.flags["negative"] = diff < 0


def _op_test(cpu: VirtualCPU, ops: list) -> None:
    """TEST Ra — set flags based on truthiness"""
    cpu._check_arity(ops, 1)
    val = cpu._val(ops[0])
    cpu.flags["zero"] = not cpu._truthy(val)
    cpu.flags["negative"] = False


# ── Control Flow ───────────────────────────────────────────────────────────


def _op_jmp(cpu: VirtualCPU, ops: list) -> None:
    """JMP label"""
    cpu._check_arity(ops, 1)
    cpu.pc = cpu._resolve_label(ops[0]) - 1  # -1 because pc auto-increments


def _op_jz(cpu: VirtualCPU, ops: list) -> None:
    """JZ label — jump if zero flag set"""
    cpu._check_arity(ops, 1)
    if cpu.flags["zero"]:
        cpu.pc = cpu._resolve_label(ops[0]) - 1


def _op_jnz(cpu: VirtualCPU, ops: list) -> None:
    """JNZ label — jump if zero flag clear"""
    cpu._check_arity(ops, 1)
    if not cpu.flags["zero"]:
        cpu.pc = cpu._resolve_label(ops[0]) - 1


def _op_jgt(cpu: VirtualCPU, ops: list) -> None:
    """JGT label — jump if not zero and not negative (a > b)"""
    cpu._check_arity(ops, 1)
    if not cpu.flags["zero"] and not cpu.flags["negative"]:
        cpu.pc = cpu._resolve_label(ops[0]) - 1


def _op_jge(cpu: VirtualCPU, ops: list) -> None:
    """JGE label — jump if not negative (a >= b)"""
    cpu._check_arity(ops, 1)
    if not cpu.flags["negative"]:
        cpu.pc = cpu._resolve_label(ops[0]) - 1


def _op_jlt(cpu: VirtualCPU, ops: list) -> None:
    """JLT label — jump if negative (a < b)"""
    cpu._check_arity(ops, 1)
    if cpu.flags["negative"]:
        cpu.pc = cpu._resolve_label(ops[0]) - 1


def _op_jle(cpu: VirtualCPU, ops: list) -> None:
    """JLE label — jump if zero or negative (a <= b)"""
    cpu._check_arity(ops, 1)
    if cpu.flags["zero"] or cpu.flags["negative"]:
        cpu.pc = cpu._resolve_label(ops[0]) - 1


def _op_call(cpu: VirtualCPU, ops: list) -> None:
    """CALL label — push return address, jump"""
    cpu._check_arity(ops, 1)
    if cpu._call_depth >= MAX_CALL_DEPTH:
        cpu._output.append("call stack overflow")
        cpu._running = False
        return
    cpu._stack.append(cpu.pc + 1)
    cpu._call_depth += 1
    cpu.pc = cpu._resolve_label(ops[0]) - 1


def _op_ret(cpu: VirtualCPU, ops: list) -> None:
    """RET — pop return address, jump back"""
    if not cpu._stack:
        cpu._output.append("ret with empty call stack")
        cpu._running = False
        return
    cpu.pc = cpu._stack.pop() - 1  # -1 because pc auto-increments
    cpu._call_depth -= 1


def _op_loop(cpu: VirtualCPU, ops: list) -> None:
    """LOOP Rd, label — decrement Rd, jump if non-zero"""
    cpu._check_arity(ops, 2)
    rd = cpu._reg(ops[0])
    cpu.regs[rd] = int(cpu.regs[rd]) - 1
    if cpu.regs[rd] > 0:
        cpu.pc = cpu._resolve_label(ops[1]) - 1


def _op_halt(cpu: VirtualCPU, ops: list) -> None:
    """HALT"""
    cpu._running = False


# ── I/O Instructions ──────────────────────────────────────────────────────


def _op_in(cpu: VirtualCPU, ops: list) -> None:
    """IN Rd, port — read from I/O port into register"""
    cpu._check_arity(ops, 2)
    rd = cpu._reg(ops[0])
    port = int(cpu._val(ops[1]))
    cpu.regs[rd] = cpu.bus.read_io(port)


def _op_out(cpu: VirtualCPU, ops: list) -> None:
    """OUT Rs, port — write register to I/O port"""
    cpu._check_arity(ops, 2)
    rs = cpu._reg(ops[0])
    port = int(cpu._val(ops[1]))
    cpu.bus.write_io(port, cpu.regs[rs])


# ── Memory Management ─────────────────────────────────────────────────────


def _op_alloc(cpu: VirtualCPU, ops: list) -> None:
    """ALLOC Rd, size — allocate block in RAM, store address in Rd"""
    cpu._check_arity(ops, 2)
    rd = cpu._reg(ops[0])
    size = int(cpu._val(ops[1]))
    ram = None
    for _, _, dev in cpu.bus._devices:
        if isinstance(dev, VirtualRAM):
            ram = dev
            break
    if ram is None:
        cpu._output.append("no RAM device on bus")
        return
    name = f"_alloc_{cpu._step_count}"
    try:
        addr = ram.alloc(name, size)
        cpu._heap[name] = addr
        cpu.regs[rd] = addr
    except MemoryError as e:
        cpu._output.append(str(e))
        cpu.regs[rd] = -1


def _op_meminfo(cpu: VirtualCPU, ops: list) -> None:
    """MEMINFO Rd — put RAM usage stats into Rd as a dict"""
    cpu._check_arity(ops, 1)
    rd = cpu._reg(ops[0])
    for _, _, dev in cpu.bus._devices:
        if isinstance(dev, VirtualRAM):
            cpu.regs[rd] = dev.usage
            return
    cpu.regs[rd] = {}


# ── Instruction Table ─────────────────────────────────────────────────────

_INSTRUCTION_TABLE: dict[str, Any] = {
    # Load/Move
    "LOAD_CONST": _op_load_const,
    "MOV": _op_mov,
    "STORE": _op_store,
    "LOAD": _op_load,
    "FREE": _op_free,
    "PRINT": _op_print,
    # Integer ALU
    "IADD": _op_iadd,
    "ISUB": _op_isub,
    "IMUL": _op_imul,
    "IDIV": _op_idiv,
    "IAND": _op_iand,
    "IOR": _op_ior,
    "IXOR": _op_ixor,
    "ISHL": _op_ishl,
    "ISHR": _op_ishr,
    "INEG": _op_ineg,
    "INC": _op_inc,
    "DEC": _op_dec,
    # Float ALU
    "FADD": _op_fadd,
    "FSUB": _op_fsub,
    "FMUL": _op_fmul,
    "FDIV": _op_fdiv,
    # Comparison
    "CMP": _op_cmp,
    "TEST": _op_test,
    "ICMP": _op_icmp,
    "FCMP": _op_fcmp,
    # Control Flow
    "JMP": _op_jmp,
    "JZ": _op_jz,
    "JNZ": _op_jnz,
    "JGT": _op_jgt,
    "JGE": _op_jge,
    "JLT": _op_jlt,
    "JLE": _op_jle,
    "CALL": _op_call,
    "RET": _op_ret,
    "LOOP": _op_loop,
    "HALT": _op_halt,
    # I/O
    "IN": _op_in,
    "OUT": _op_out,
    # Memory
    "ALLOC": _op_alloc,
    "MEMINFO": _op_meminfo,
}

# Instruction set documentation
ISA: dict[str, str] = {
    # Load/Move
    "LOAD_CONST": "Rd, value        — Rd = constant (int/float/string)",
    "MOV":        "Rd, Rs           — Rd = Rs",
    "STORE":      "Rs, name         — heap[name] = Rs",
    "LOAD":       "Rd, name         — Rd = heap[name]",
    "FREE":       "name             — delete heap[name]",
    "PRINT":      "Rs               — output Rs as string",
    # Integer ALU
    "IADD":       "Rd, Ra, Rb       — Rd = int(Ra) + int(Rb)",
    "ISUB":       "Rd, Ra, Rb       — Rd = int(Ra) - int(Rb)",
    "IMUL":       "Rd, Ra, Rb       — Rd = int(Ra) * int(Rb)",
    "IDIV":       "Rd, Ra, Rb       — Rd = int(Ra) // int(Rb)",
    "IAND":       "Rd, Ra, Rb       — Rd = int(Ra) & int(Rb)",
    "IOR":        "Rd, Ra, Rb       — Rd = int(Ra) | int(Rb)",
    "IXOR":       "Rd, Ra, Rb       — Rd = int(Ra) ^ int(Rb)",
    "ISHL":       "Rd, Ra, Rb       — Rd = int(Ra) << int(Rb)",
    "ISHR":       "Rd, Ra, Rb       — Rd = int(Ra) >> int(Rb)",
    "INEG":       "Rd, Ra           — Rd = -int(Ra)",
    "INC":        "Rd               — Rd = int(Rd) + 1",
    "DEC":        "Rd               — Rd = int(Rd) - 1",
    # Float ALU
    "FADD":       "Rd, Ra, Rb       — Rd = float(Ra) + float(Rb)",
    "FSUB":       "Rd, Ra, Rb       — Rd = float(Ra) - float(Rb)",
    "FMUL":       "Rd, Ra, Rb       — Rd = float(Ra) * float(Rb)",
    "FDIV":       "Rd, Ra, Rb       — Rd = float(Ra) / float(Rb)",
    # Comparison
    "CMP":        "Ra, Rb           — set flags from comparison",
    "TEST":       "Ra               — set flags based on truthiness",
    "ICMP":       "Ra, Rb           — integer comparison, sets flags",
    "FCMP":       "Ra, Rb           — float comparison, sets flags",
    # Control Flow
    "JMP":        "label            — unconditional jump",
    "JZ":         "label            — jump if zero flag set",
    "JNZ":        "label            — jump if zero flag clear",
    "JGT":        "label            — jump if a > b",
    "JGE":        "label            — jump if a >= b",
    "JLT":        "label            — jump if a < b",
    "JLE":        "label            — jump if a <= b",
    "CALL":       "label            — push return addr, jump",
    "RET":        "                 — pop return addr, jump back",
    "LOOP":       "Rd, label        — Rd--, jump if Rd > 0",
    "HALT":       "                 — stop execution",
    # I/O
    "IN":         "Rd, port         — Rd = read from I/O port",
    "OUT":        "Rs, port         — write Rs to I/O port",
    # Memory
    "ALLOC":      "Rd, size         — Rd = allocate block in RAM",
    "MEMINFO":    "Rd               — Rd = RAM usage stats",
}

# Example programs (hardware-level, no AI)
COUNTER_ASM = """
    LOAD_CONST R0, 10
loop:
    PRINT R0
    DEC R0
    TEST R0
    JNZ loop
    HALT
"""

FIB_ASM = """
    LOAD_CONST R0, 0
    LOAD_CONST R1, 1
    LOAD_CONST R2, 10
loop:
    PRINT R0
    IADD R3, R0, R1
    MOV R0, R1
    MOV R1, R3
    DEC R2
    TEST R2
    JNZ loop
    HALT
"""

COLLATZ_ASM = """
    LOAD_CONST R0, 27
    LOAD_CONST R1, 0
loop:
    PRINT R0
    INC R1
    ICMP R0, 1
    JZ done
    IAND R2, R0, 1
    TEST R2
    JNZ odd
    IDIV R0, R0, 2
    JMP loop
odd:
    IMUL R0, R0, 3
    INC R0
    JMP loop
done:
    PRINT R1
    HALT
"""


# ── Virtual Tensor Accelerator (GPU Device) ───────────────────────────────


class TensorAccel(Device):
    """GPU-like tensor accelerator — executes tensor operations via memory-mapped I/O.

    The CPU sends commands to this device via OUT instructions.
    The device reads source tensors from RAM, computes, and writes results back.

    I/O Ports (memory-mapped registers):
      0x00 CMD      — command register (write to trigger execution)
      0x01 SRC1     — source1 RAM address (high byte)
      0x02 SRC1_LO  — source1 RAM address (low byte)
      0x03 SRC2     — source2 RAM address
      0x04 SRC2_LO
      0x05 DST      — destination RAM address
      0x06 DST_LO
      0x07 STATUS   — 0=busy, 1=done, 2=error
      0x10-0x1F     — shape registers (for reshape ops)
    """

    # Command constants
    CMD_NOP = 0
    CMD_ADD = 1
    CMD_SUB = 2
    CMD_MUL = 3
    CMD_DIV = 4
    CMD_MATMUL = 5
    CMD_RELU = 6
    CMD_SIGMOID = 7
    CMD_TANH = 8
    CMD_SOFTMAX = 9
    CMD_TRANSPOSE = 10
    CMD_DOT = 11
    CMD_NORM = 12
    CMD_SUM = 13
    CMD_MEAN = 14
    CMD_MAX = 15
    CMD_ARGMAX = 16

    name = "tensor_accel"
    size = 32  # 32 registers

    def __init__(self, ram: VirtualRAM | None = None):
        super().__init__(base_addr=0xF000)
        self._ram = ram
        self._status = 0  # 0=busy, 1=done, 2=error
        self._last_result: Any = None

    def control(self, command: int, *args: Any) -> Any:
        """Execute a tensor command. Called by I/O handler."""
        self._status = 0  # busy
        try:
            if command == self.CMD_ADD:
                return self._binary_op(args[0], args[1], args[2], lambda a, b: a + b)
            elif command == self.CMD_SUB:
                return self._binary_op(args[0], args[1], args[2], lambda a, b: a - b)
            elif command == self.CMD_MUL:
                return self._binary_op(args[0], args[1], args[2], lambda a, b: a * b)
            elif command == self.CMD_DIV:
                return self._binary_op(args[0], args[1], args[2], lambda a, b: a / b)
            elif command == self.CMD_MATMUL:
                return self._binary_op(args[0], args[1], args[2], lambda a, b: a @ b)
            elif command == self.CMD_RELU:
                return self._unary_op(args[0], args[1], lambda a: np.maximum(0, a))
            elif command == self.CMD_SIGMOID:
                return self._unary_op(args[0], args[1], lambda a: 1.0 / (1.0 + np.exp(-np.clip(a, -500, 500))))
            elif command == self.CMD_TANH:
                return self._unary_op(args[0], args[1], lambda a: np.tanh(a))
            elif command == self.CMD_SOFTMAX:
                return self._unary_op(args[0], args[1], lambda a: np.exp(a - np.max(a)) / np.sum(np.exp(a - np.max(a))))
            elif command == self.CMD_TRANSPOSE:
                return self._unary_op(args[0], args[1], lambda a: a.T)
            elif command == self.CMD_DOT:
                return self._binary_op(args[0], args[1], args[2], lambda a, b: float(np.dot(a.ravel(), b.ravel())))
            elif command == self.CMD_NORM:
                return self._unary_op(args[0], args[1], lambda a: float(np.linalg.norm(a)))
            elif command == self.CMD_SUM:
                return self._unary_op(args[0], args[1], lambda a: float(np.sum(a)))
            elif command == self.CMD_MEAN:
                return self._unary_op(args[0], args[1], lambda a: float(np.mean(a)))
            elif command == self.CMD_MAX:
                return self._unary_op(args[0], args[1], lambda a: float(np.max(a)))
            elif command == self.CMD_ARGMAX:
                return self._unary_op(args[0], args[1], lambda a: int(np.argmax(a)))
            else:
                self._status = 2  # error
                return None
        except Exception:
            self._status = 2
            return None

    def _binary_op(self, src1_addr: int, src2_addr: int, dst_addr: int, op) -> Any:
        """Execute a binary tensor op: result = op(src1, src2)"""
        if self._ram is None:
            self._status = 2
            return None
        # Read tensors from RAM (stored as JSON-serializable numpy arrays)
        a = self._read_tensor(src1_addr)
        b = self._read_tensor(src2_addr)
        result = op(a, b)
        self._write_tensor(dst_addr, result)
        self._status = 1  # done
        return result

    def _unary_op(self, src_addr: int, dst_addr: int, op) -> Any:
        """Execute a unary tensor op: result = op(src)"""
        if self._ram is None:
            self._status = 2
            return None
        a = self._read_tensor(src_addr)
        result = op(a)
        self._write_tensor(dst_addr, result)
        self._status = 1
        return result

    def _read_tensor(self, addr: int) -> np.ndarray:
        """Read a tensor from RAM at the given address."""
        # Read length prefix (4 bytes) then data
        raw_len = self._ram.read_block(addr, 4)
        length = struct.unpack("<I", raw_len)[0]
        raw_data = self._ram.read_block(addr + 4, length)
        return np.frombuffer(raw_data, dtype=np.float64).copy()

    def _write_tensor(self, addr: int, tensor: np.ndarray) -> None:
        """Write a tensor to RAM at the given address."""
        data = tensor.astype(np.float64).tobytes()
        raw_len = struct.pack("<I", len(data))
        self._ram.write_block(addr, raw_len)
        self._ram.write_block(addr + 4, data)

    @property
    def status(self) -> int:
        return self._status


# ── Block Storage Device ───────────────────────────────────────────────────


class BlockStorage(Device):
    """Disk-like block storage device.

    Simulates a block device with sectors. Used for swap and persistent storage.
    """

    name = "storage"
    SECTOR_SIZE = 512

    def __init__(self, sectors: int = 2048):
        super().__init__(base_addr=0xF100)
        self.sector_count = sectors
        self._data = bytearray(sectors * self.SECTOR_SIZE)
        self._read_count = 0
        self._write_count = 0

    def read_sector(self, sector: int) -> bytes:
        """Read a sector from the device."""
        if 0 <= sector < self.sector_count:
            self._read_count += 1
            offset = sector * self.SECTOR_SIZE
            return bytes(self._data[offset:offset + self.SECTOR_SIZE])
        return b"\x00" * self.SECTOR_SIZE

    def write_sector(self, sector: int, data: bytes) -> None:
        """Write a sector to the device."""
        if 0 <= sector < self.sector_count:
            self._write_count += 1
            offset = sector * self.SECTOR_SIZE
            end = min(offset + self.SECTOR_SIZE, len(self._data))
            self._data[offset:end] = data[:end - offset]

    def read_block(self, offset: int, size: int) -> bytes:
        """Read raw bytes from the device."""
        return bytes(self._data[offset:offset + size])

    def write_block(self, offset: int, data: bytes) -> None:
        """Write raw bytes to the device."""
        end = min(offset + len(data), len(self._data))
        self._data[offset:end] = data[:end - offset]

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "sectors": self.sector_count,
            "size_bytes": self.sector_count * self.SECTOR_SIZE,
            "reads": self._read_count,
            "writes": self._write_count,
        }


# ── Memory Manager (Tiered Storage) ───────────────────────────────────────


class MemoryManager:
    """Tiered memory management with automatic swapping.

    Handles tensor allocation across memory tiers:
      L0: CPU registers (fastest, 16 slots)
      L1: RAM heap (fast, managed by VirtualRAM)
      L2: Disk swap (slow, uses BlockStorage)

    Automatically swaps LRU tensors to disk when RAM is full.
    Programs allocate freely — the MM handles the hardware reality.
    """

    def __init__(self, ram: VirtualRAM, disk: BlockStorage | None = None):
        self._ram = ram
        self._disk = disk
        self._allocations: dict[str, dict[str, Any]] = {}
        self._access_order: list[str] = []
        self._swap_sectors: dict[str, int] = {}  # name → sector offset
        self._next_swap_sector = 0

    def alloc(self, name: str, data: bytes) -> int:
        """Allocate data in the fastest available tier. Returns address."""
        size = len(data)
        try:
            addr = self._ram.alloc(name, size)
            self._ram.write_block(addr, data)
            self._allocations[name] = {"tier": "ram", "addr": addr, "size": size}
            self._touch(name)
            return addr
        except MemoryError:
            # RAM full — swap LRU to disk and try again
            if self._disk:
                self._swap_lru()
                try:
                    addr = self._ram.alloc(name, size)
                    self._ram.write_block(addr, data)
                    self._allocations[name] = {"tier": "ram", "addr": addr, "size": size}
                    self._touch(name)
                    return addr
                except MemoryError:
                    pass
            # Store directly on disk
            return self._alloc_disk(name, data)

    def read(self, name: str) -> bytes | None:
        """Read data, loading from disk if needed."""
        info = self._allocations.get(name)
        if info is None:
            return None
        self._touch(name)

        if info["tier"] == "disk":
            # Load from disk to RAM
            data = self._read_disk(name)
            if data is None:
                return None
            try:
                addr = self._ram.alloc(name, len(data))
                self._ram.write_block(addr, data)
                self._free_disk(name)
                self._allocations[name] = {"tier": "ram", "addr": addr, "size": len(data)}
                return data
            except MemoryError:
                self._swap_lru()
                try:
                    addr = self._ram.alloc(name, len(data))
                    self._ram.write_block(addr, data)
                    self._free_disk(name)
                    self._allocations[name] = {"tier": "ram", "addr": addr, "size": len(data)}
                    return data
                except MemoryError:
                    return data  # return from disk read

        return self._ram.read_block(info["addr"], info["size"])

    def free(self, name: str) -> None:
        """Free an allocation."""
        info = self._allocations.pop(name, None)
        if info is None:
            return
        if info["tier"] == "ram":
            self._ram.free(name)
        else:
            self._free_disk(name)
        self._access_order = [n for n in self._access_order if n != name]

    def _touch(self, name: str) -> None:
        """Update access order."""
        self._access_order = [n for n in self._access_order if n != name]
        self._access_order.append(name)

    def _swap_lru(self) -> None:
        """Swap least-recently-used allocation to disk."""
        if not self._access_order or not self._disk:
            return
        lru_name = self._access_order[0]
        info = self._allocations.get(lru_name)
        if info is None or info["tier"] != "ram":
            return
        # Read data from RAM
        data = self._ram.read_block(info["addr"], info["size"])
        # Write to disk
        sector = self._next_swap_sector
        sectors_needed = (len(data) + self._disk.SECTOR_SIZE - 1) // self._disk.SECTOR_SIZE
        for i in range(sectors_needed):
            chunk = data[i * self._disk.SECTOR_SIZE:(i + 1) * self._disk.SECTOR_SIZE]
            self._disk.write_sector(sector + i, chunk)
        self._swap_sectors[lru_name] = sector
        # Free from RAM
        self._ram.free(lru_name)
        self._allocations[lru_name] = {"tier": "disk", "sector": sector, "size": len(data)}
        self._next_swap_sector += sectors_needed

    def _alloc_disk(self, name: str, data: bytes) -> int:
        """Allocate directly on disk."""
        if not self._disk:
            raise MemoryError("no disk device and RAM is full")
        sector = self._next_swap_sector
        sectors_needed = (len(data) + self._disk.SECTOR_SIZE - 1) // self._disk.SECTOR_SIZE
        for i in range(sectors_needed):
            chunk = data[i * self._disk.SECTOR_SIZE:(i + 1) * self._disk.SECTOR_SIZE]
            self._disk.write_sector(sector + i, chunk)
        self._swap_sectors[name] = sector
        self._allocations[name] = {"tier": "disk", "sector": sector, "size": len(data)}
        self._next_swap_sector += sectors_needed
        self._touch(name)
        return sector

    def _read_disk(self, name: str) -> bytes | None:
        """Read data from disk swap."""
        info = self._allocations.get(name)
        if info is None or info["tier"] != "disk" or self._disk is None:
            return None
        sector = info["sector"]
        sectors_needed = (info["size"] + self._disk.SECTOR_SIZE - 1) // self._disk.SECTOR_SIZE
        data = b""
        for i in range(sectors_needed):
            data += self._disk.read_sector(sector + i)
        return data[:info["size"]]

    def _free_disk(self, name: str) -> None:
        """Free disk swap space."""
        self._swap_sectors.pop(name, None)

    @property
    def stats(self) -> dict[str, Any]:
        """Memory manager statistics."""
        ram_count = sum(1 for v in self._allocations.values() if v["tier"] == "ram")
        disk_count = sum(1 for v in self._allocations.values() if v["tier"] == "disk")
        return {
            "total_allocations": len(self._allocations),
            "ram_allocations": ram_count,
            "disk_allocations": disk_count,
            "ram_usage": self._ram.usage,
        }


# ── Virtual System ─────────────────────────────────────────────────────────


class VirtualSystem:
    """A complete simulated computer system.

    Wires CPU, RAM, bus, and devices together. Configure hardware specs,
    attach devices, load programs, and run.

    Usage:
        system = VirtualSystem(ram_mb=256, gpu_vram_mb=128)
        system.load_program(assembly_source)
        output = system.run()
    """

    def __init__(
        self,
        ram_size: int = DEFAULT_RAM_SIZE,
        cpu_regs: int = DEFAULT_REG_COUNT,
        enable_gpu: bool = True,
        enable_storage: bool = True,
    ):
        # Create components
        self.bus = VirtualBus()
        self.ram = VirtualRAM(size_bytes=ram_size)
        self.cpu = VirtualCPU(num_regs=cpu_regs, bus=self.bus)

        # Attach RAM at address 0
        self.bus.attach(self.ram, base_addr=0)

        # Optional GPU
        self.gpu: TensorAccel | None = None
        if enable_gpu:
            self.gpu = TensorAccel(ram=self.ram)
            self.bus.attach(self.gpu, base_addr=0xF000)

        # Optional storage
        self.storage: BlockStorage | None = None
        if enable_storage:
            self.storage = BlockStorage(sectors=2048)
            self.bus.attach(self.storage, base_addr=0xF100)

        # Memory manager
        self.mm = MemoryManager(ram=self.ram, disk=self.storage)

    def load_program(self, source: str) -> None:
        """Load an assembly program (parsed by the assembler in vm.py)."""
        from domains.shell.vm import ProgramLoader
        loader = ProgramLoader()
        instructions, labels = loader.load_with_labels(source)
        self.cpu.load_program(instructions, labels)

    def run(self) -> list[str]:
        """Run the loaded program. Returns PRINT output."""
        return self.cpu.run()

    def reset(self) -> None:
        """Reset the entire system."""
        self.cpu = VirtualCPU(num_regs=len(self.cpu.regs), bus=self.bus)
        self.bus.reset_all()

    def status(self) -> dict[str, Any]:
        """Get system status."""
        return {
            "ram": self.ram.usage,
            "gpu": {"status": self.gpu.status} if self.gpu else None,
            "storage": self.storage.stats if self.storage else None,
            "memory_manager": self.mm.stats,
            "cpu_steps": self.cpu._step_count,
        }
