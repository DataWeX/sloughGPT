"""
AI Networking Processor — Core VM module.

Constants, exceptions, data structures, Memory, DeviceBus, CPU,
Assembler, VMRunner, opcode handlers, and dispatch table.

This is the single core VM file. Device drivers live in vm_devices.py,
assembly programs and self-test live in vm_programs.py.

Layers 4-6 (drivers, libraries, applications) sit on top of this machine.
The VM never imports domain classes. It only knows registers, tensors,
and the generic DEV_OPEN/DEV_CALL/DEV_CLOSE protocol.
"""

from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Callable

import numpy as np

logger = logging.getLogger("slo.vm")


# ── Constants ────────────────────────────────────────────────────────────────

NUM_REGS = 16
MAX_INSTRUCTIONS = 100_000
MAX_CALL_DEPTH = 256
MEM_SIZE = 65536
STACK_BASE = 0xFFFF
F_ZERO = 1 << 0
F_NEG = 1 << 2


# ── Exceptions ───────────────────────────────────────────────────────────────

class VMFault(Exception):
    """Base VM fault."""


class InsFault(VMFault):
    """Invalid instruction or operand."""


class Halt(VMFault):
    """Program halted (normal termination)."""


class MemFault(VMFault):
    """Memory access violation."""


class SysFault(VMFault):
    """Syscall error."""


class DeviceFault(VMFault):
    """Device error."""


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Instruction:
    """Decoded instruction ready for execution."""
    opcode: str
    operands: list
    line_num: int = 0
    raw: str = ""


@dataclass
class TraceEntry:
    """Snapshot of machine state at one execution cycle."""
    cycle: int
    pc: int
    instruction: str
    registers: dict
    heap_keys: list


# ── ISA Definition ───────────────────────────────────────────────────────────

OPCODES = {
    "LOAD_CONST": "Rd, value          Rd = constant",
    "LOAD_SHAPE": "Rd, rows, cols     Rd = zeros(rows,cols)",
    "MOV":        "Rd, Rs             Rd = Rs",
    "STORE":      "Rs, key            heap[key] = Rs",
    "LOAD":       "Rd, key            Rd = heap[key]",
    "FREE":       "key                delete heap[key]",
    "PRINT":      "Rs                 output Rs",
    "NOP":        "                   no operation",
    "IADD":       "Rd, Ra, Rb         Rd = int(Ra) + int(Rb)",
    "ISUB":       "Rd, Ra, Rb         Rd = int(Ra) - int(Rb)",
    "IMUL":       "Rd, Ra, Rb         Rd = int(Ra) * int(Rb)",
    "IDIV":       "Rd, Ra, Rb         Rd = int(Ra) // int(Rb)",
    "IAND":       "Rd, Ra, Rb         Rd = int(Ra) & int(Rb)",
    "IOR":        "Rd, Ra, Rb         Rd = int(Ra) | int(Rb)",
    "IXOR":       "Rd, Ra, Rb         Rd = int(Ra) ^ int(Rb)",
    "ISHL":       "Rd, Ra, Rb         Rd = int(Ra) << int(Rb)",
    "ISHR":       "Rd, Ra, Rb         Rd = int(Ra) >> int(Rb)",
    "INEG":       "Rd, Ra             Rd = -int(Ra)",
    "INC":        "Rd                 Rd += 1",
    "DEC":        "Rd                 Rd -= 1",
    "ICMP":       "Ra, Rb             set CMP_FLAG",
    "ADD":        "Rd, Ra, Rb         Rd = Ra + Rb",
    "SUB":        "Rd, Ra, Rb         Rd = Ra - Rb",
    "MUL":        "Rd, Ra, Rb         Rd = Ra * Rb",
    "DIV":        "Rd, Ra, Rb         Rd = Ra / Rb",
    "NEG":        "Rd, Ra             Rd = -Ra",
    "ABS":        "Rd, Ra             Rd = |Ra|",
    "MATMUL":     "Rd, Ra, Rb         Rd = Ra @ Rb",
    "TRANSPOSE":  "Rd, Ra             Rd = Ra.T",
    "DOT":        "Rd, Ra, Rb         Rd = Ra . Rb",
    "NORM":       "Rd, Ra             Rd = ||Ra||",
    "SUM":        "Rd, Ra             Rd = sum(Ra)",
    "MEAN":       "Rd, Ra             Rd = mean(Ra)",
    "MAX":        "Rd, Ra             Rd = max(Ra)",
    "ARGMAX":     "Rd, Ra             Rd = argmax(Ra)",
    "RESHAPE":    "Rd, Ra, R, C       Rd = reshape(Ra, (R,C))",
    "SHAPE":      "Rd, Ra             Rd = list(Ra.shape)",
    "SIZE":       "Rd, Ra             Rd = element count",
    "RELU":       "Rd, Ra             Rd = max(0, Ra)",
    "GELU":       "Rd, Ra             Rd = gelu(Ra)",
    "SIGMOID":    "Rd, Ra             Rd = sigmoid(Ra)",
    "TANH":       "Rd, Ra             Rd = tanh(Ra)",
    "SOFTMAX":    "Rd, Ra             Rd = softmax(Ra)",
    "LAYERNORM":  "Rd, Ra             Rd = layer_norm(Ra)",
    "RMSNORM":    "Rd, Ra             Rd = rms_norm(Ra)",
    "RANDN":      "Rd, rows, cols     Rd = randn(rows,cols)",
    "RANDUNIF":   "Rd, rows, cols, lo, hi  Rd = uniform(lo,hi)",
    "CMP":        "Ra, Rb             set CMP_FLAG (-1/0/+1)",
    "TEST":       "Ra                 set CMP_FLAG (0 or 1)",
    "JMP":        "label              PC = label",
    "JZ":         "label              if CMP_FLAG == 0: PC = label",
    "JNZ":        "label              if CMP_FLAG != 0: PC = label",
    "JGT":        "label              if CMP_FLAG > 0: PC = label",
    "JGE":        "label              if CMP_FLAG >= 0: PC = label",
    "JLT":        "label              if CMP_FLAG < 0: PC = label",
    "JLE":        "label              if CMP_FLAG <= 0: PC = label",
    "CALL":       "label              push PC+1, PC = label",
    "RET":        "                   pop call stack",
    "LOOP":       "Rd, label          Rd -= 1; if Rd != 0: PC = label",
    "HALT":       "                   stop execution",
    "SYSCALL":    "                   R0=handler(R7, [R0..R5])",
    "DEV_OPEN":   "Rd, name           Rd = device handle",
    "DEV_CALL":   "Rd, H, method, args...  Rd = device.method(*args)",
    "DEV_CLOSE":  "H                  release device handle",
    "DEV_INFO":   "Rd, H              Rd = device.info()",
    "PUSH":       "Rs                 stack.push(Rs); sp -= 1",
    "POP":        "Rd                 sp += 1; Rd = stack[sp]",
    "FADD":       "Rd, Ra, Rb         Rd = float(Ra) + float(Rb)",
    "FSUB":       "Rd, Ra, Rb         Rd = float(Ra) - float(Rb)",
    "FMUL":       "Rd, Ra, Rb         Rd = float(Ra) * float(Rb)",
    "FDIV":       "Rd, Ra, Rb         Rd = float(Ra) / float(Rb)",
    "FCMP":       "Ra, Rb             set CMP_FLAG for floats",
    "ALLOC":      "Rd, size           Rd = heap.alloc(size)",
    "MEMINFO":    "Rd                 Rd = heap.usage()",
    "IN":         "Rd, port           Rd = bus.read_io(port)",
    "OUT":        "port, Rs           bus.write_io(port, Rs)",
}


# ── Memory Subsystem ─────────────────────────────────────────────────────────

class Memory:
    """Named tensor heap with LRU access tracking."""

    def __init__(self):
        self._heap = {}
        self._lru = []
        self._alloc_sizes = {}

    def store(self, key, value):
        self._heap[key] = value
        self._touch(key)
        self._alloc_sizes[key] = value.nbytes if isinstance(value, np.ndarray) else 0

    def load(self, key):
        if key not in self._heap:
            raise InsFault(f"heap key not found: {key}")
        self._touch(key)
        return self._heap[key]

    def free(self, key):
        self._heap.pop(key, None)
        self._alloc_sizes.pop(key, None)
        if key in self._lru:
            self._lru.remove(key)

    def contains(self, key):
        return key in self._heap

    def lru_evict(self):
        if not self._lru:
            return None
        key = self._lru.pop(0)
        self._heap.pop(key, None)
        self._alloc_sizes.pop(key, None)
        return key

    def usage(self):
        return {
            "entries": len(self._heap),
            "keys": list(self._heap.keys()),
            "bytes_tracked": sum(self._alloc_sizes.values()),
            "lru_order": list(self._lru),
        }

    def _touch(self, key):
        if key in self._lru:
            self._lru.remove(key)
        self._lru.append(key)


# ── Device Bus ───────────────────────────────────────────────────────────────

class Device:
    """Generic device interface. Subclass to wrap any library."""

    def call(self, method, *args):
        raise DeviceFault(f"device does not support: {method}")

    def info(self):
        return {"type": "base", "methods": []}


class ConsoleDevice(Device):
    """Console I/O device — port 0 for stdin, port 1 for stdout.

    Registered as "0" (stdin) and "1" (stdout) on the device bus so that
    IN R0, 0 reads a line and OUT 1, R0 prints it.
    """

    def __init__(self, port: int, stdin_fn=None, stdout_fn=None):
        self._port = port
        self._stdin_fn = stdin_fn or (lambda: "")
        self._stdout_fn = stdout_fn or (lambda v: None)
        self._buffer: list = []

    def info(self):
        return {"type": "console", "port": self._port, "status": 1}

    def write(self, value):
        self._stdout_fn(value)

    def read(self):
        return self._stdin_fn()

    def call(self, method, *args):
        if method == "read":
            return self.read()
        if method == "write":
            return self.write(*args)
        return super().call(method, *args)


class FileDevice(Device):
    """File I/O device — provides read/write access to the host filesystem.

    Commands (via DEV_CALL):
      open(path, mode) -> fd
      read(fd, size) -> bytes
      write(fd, data) -> bytes_written
      close(fd)
      listdir(path) -> list[str]
      exists(path) -> bool
    """

    def __init__(self):
        self._files: dict[int, any] = {}
        self._next_fd: int = 1

    def info(self):
        return {"type": "file", "open_files": len(self._files)}

    def call(self, method, *args):
        if method == "open":
            path, mode = args[0], args[1] if len(args) > 1 else "r"
            fh = open(path, mode)
            fd = self._next_fd
            self._next_fd += 1
            self._files[fd] = fh
            return fd
        if method == "read":
            fd, size = args[0], args[1] if len(args) > 1 else 4096
            fh = self._files.get(fd)
            if fh is None:
                raise DeviceFault(f"bad fd: {fd}")
            return fh.read(size)
        if method == "write":
            fd, data = args[0], args[1]
            fh = self._files.get(fd)
            if fh is None:
                raise DeviceFault(f"bad fd: {fd}")
            return fh.write(data)
        if method == "close":
            fd = args[0]
            fh = self._files.pop(fd, None)
            if fh:
                fh.close()
            return True
        if method == "listdir":
            import os
            return os.listdir(args[0])
        if method == "exists":
            import os
            return os.path.exists(args[0])
        return super().call(method, *args)


class IRQDevice(Device):
    """Interrupt request device — fires timer and keyboard interrupts.

    Registers IRQ handlers on a CPU. When tick() is called, fires timer IRQ.
    Keyboard input is queued and fires keyboard IRQ.
    """

    TIMER_IRQ = 0
    KEYBOARD_IRQ = 1

    def __init__(self):
        self._tick_count = 0
        self._key_queue: list = []

    def info(self):
        return {"type": "irq", "ticks": self._tick_count, "keys_pending": len(self._key_queue)}

    def tick(self, cpu):
        """Fire timer interrupt every tick."""
        self._tick_count += 1
        if self._tick_count % 10 == 0:
            cpu.fire_irq(self.TIMER_IRQ)

    def push_key(self, key):
        """Queue a keypress and fire keyboard interrupt."""
        self._key_queue.append(key)

    def read_key(self):
        """Read next key from queue (returns 0 if empty)."""
        if self._key_queue:
            return self._key_queue.pop(0)
        return 0

    def call(self, method, *args):
        if method == "tick":
            return self._tick_count
        if method == "read_key":
            return self.read_key()
        return super().call(method, *args)


class BlockDevice(Device):
    """Sector-based block storage — 512-byte sectors.

    Provides read_sector/write_sector for raw I/O and read_block/write_block
    for higher-level access. Tracks I/O statistics.
    """

    SECTOR_SIZE = 512

    def __init__(self, num_sectors: int = 256):
        self._sectors = [bytearray(self.SECTOR_SIZE) for _ in range(num_sectors)]
        self._num_sectors = num_sectors
        self._reads = 0
        self._writes = 0

    def info(self):
        return {
            "type": "block",
            "sectors": self._num_sectors,
            "sector_size": self.SECTOR_SIZE,
            "reads": self._reads,
            "writes": self._writes,
        }

    def read_sector(self, sector_idx: int) -> bytearray:
        if not (0 <= sector_idx < self._num_sectors):
            raise DeviceFault(f"sector out of range: {sector_idx}")
        self._reads += 1
        return self._sectors[sector_idx]

    def write_sector(self, sector_idx: int, data: bytes) -> None:
        if not (0 <= sector_idx < self._num_sectors):
            raise DeviceFault(f"sector out of range: {sector_idx}")
        self._writes += 1
        if isinstance(data, str):
            data = data.encode('utf-8')
        self._sectors[sector_idx][:len(data)] = data[:self.SECTOR_SIZE]

    def read_block(self, sector_idx: int, size: int) -> bytes:
        data = self.read_sector(sector_idx)
        return bytes(data[:size])

    def write_block(self, sector_idx: int, data: bytes) -> None:
        self.write_sector(sector_idx, data)

    def call(self, method, *args):
        if method == "read_sector":
            return self.read_sector(*args)
        if method == "write_sector":
            return self.write_sector(*args)
        if method == "read_block":
            return self.read_block(*args)
        if method == "write_block":
            return self.write_block(*args)
        return super().call(method, *args)


class FlatFS:
    """Simple flat filesystem on top of a BlockDevice.

    File table stored in sector 0 (4 bytes: num_files, then per-file entry):
      [2 bytes: name_len] [name bytes] [2 bytes: start_sector] [2 bytes: num_sectors]

    File data starts at sector 1. Max 32 files, max 32 chars per name.
    """

    MAX_FILES = 32
    MAX_NAME = 32
    TABLE_SECTOR = 0
    DATA_START = 1

    def __init__(self, block_device: BlockDevice):
        self._block = block_device
        self._files: dict[str, tuple[int, int]] = {}  # name -> (start_sector, num_sectors)
        self._load_table()

    def _load_table(self):
        raw = bytes(self._block.read_sector(self.TABLE_SECTOR))
        if raw[:2] == b'\x00\x00':
            return
        n = int.from_bytes(raw[:2], 'big')
        pos = 2
        for _ in range(n):
            name_len = int.from_bytes(raw[pos:pos+2], 'big')
            pos += 2
            name = raw[pos:pos+name_len].decode('utf-8', errors='replace')
            pos += name_len
            start = int.from_bytes(raw[pos:pos+2], 'big')
            pos += 2
            count = int.from_bytes(raw[pos:pos+2], 'big')
            pos += 2
            self._files[name] = (start, count)

    def _save_table(self):
        data = len(self._files).to_bytes(2, 'big')
        for name, (start, count) in self._files.items():
            name_bytes = name.encode('utf-8')[:self.MAX_NAME]
            data += len(name_bytes).to_bytes(2, 'big')
            data += name_bytes
            data += start.to_bytes(2, 'big')
            data += count.to_bytes(2, 'big')
        # Pad to sector size
        data = data.ljust(self._block.SECTOR_SIZE, b'\x00')
        self._block.write_sector(self.TABLE_SECTOR, data)

    def list_files(self) -> list[str]:
        return list(self._files.keys())

    def exists(self, name: str) -> bool:
        return name in self._files

    def write(self, name: str, data: bytes) -> None:
        """Write data to a file, allocating sectors as needed."""
        sectors_needed = (len(data) + self._block.SECTOR_SIZE - 1) // self._block.SECTOR_SIZE

        # Find free sectors (simple: use sectors after all existing files)
        used = set()
        for _, (_, count) in self._files.items():
            for s in range(self.DATA_START, self.DATA_START + count):
                used.add(s)

        free_sectors = []
        s = self.DATA_START
        while len(free_sectors) < sectors_needed:
            if s not in used:
                free_sectors.append(s)
            s += 1
            if s >= self._block._num_sectors:
                raise DeviceFault("no space on disk")

        # Write data sectors
        for i, sector_idx in enumerate(free_sectors):
            chunk = data[i * self._block.SECTOR_SIZE:(i + 1) * self._block.SECTOR_SIZE]
            self._block.write_sector(sector_idx, chunk)

        self._files[name] = (free_sectors[0], sectors_needed)
        self._save_table()

    def read(self, name: str) -> bytes:
        """Read entire file contents."""
        if name not in self._files:
            raise DeviceFault(f"file not found: {name}")
        start, count = self._files[name]
        data = b''
        for i in range(count):
            data += bytes(self._block.read_sector(start + i))
        return data

    def delete(self, name: str) -> bool:
        """Delete a file and free its sectors."""
        if name not in self._files:
            return False
        del self._files[name]
        self._save_table()
        return True

    def size(self, name: str) -> int:
        if name not in self._files:
            return 0
        _, count = self._files[name]
        return count * self._block.SECTOR_SIZE


class DeviceBus:
    """Device registry and generic dispatch."""

    def __init__(self):
        self._devices = {}

    def register(self, name, device):
        self._devices[name] = device

    def register_console(self, stdin_fn=None, stdout_fn=None):
        """Register console I/O on ports 0 (stdin) and 1 (stdout)."""
        self._devices["0"] = ConsoleDevice(0, stdin_fn=stdin_fn, stdout_fn=stdout_fn)
        self._devices["1"] = ConsoleDevice(1, stdin_fn=stdin_fn, stdout_fn=stdout_fn)

    def open(self, name):
        if name not in self._devices:
            raise DeviceFault(f"no such device: {name}")
        return self._devices[name]

    def call(self, device, method, *args):
        return device.call(method, *args)

    def info(self, device):
        return device.info()

    def list_devices(self):
        return list(self._devices.keys())


# ── CPU ──────────────────────────────────────────────────────────────────────

class CPU:
    """Central processing unit with integer ALU, tensor ALU, and control flow."""

    def __init__(self, memory=None, devices=None):
        self.regs = [0] * NUM_REGS
        self.pc = 0
        self.sp = STACK_BASE
        self._cmp_flag = 0
        self._carry_flag = False
        self._call_stack = []
        self._stack = {}
        self._memory = memory or Memory()
        self._devices = devices or DeviceBus()
        self._instructions = []
        self._running = False
        self._step_count = 0
        self._max_instructions = MAX_INSTRUCTIONS
        self._output = []
        self._tracing = False
        self._trace = []
        self._irq_handlers: dict[int, callable] = {}
        self._irq_pending: list[int] = []

    def load_program(self, instructions):
        self._instructions = list(instructions)
        self.pc = 0
        self._cmp_flag = 0
        self._call_stack.clear()
        self._running = False
        self._step_count = 0
        self._trace.clear()

    def register_irq(self, irq_num: int, handler: callable) -> None:
        """Register an interrupt handler for IRQ number."""
        self._irq_handlers[irq_num] = handler

    def fire_irq(self, irq_num: int) -> None:
        """Queue an interrupt to be processed."""
        self._irq_pending.append(irq_num)

    def _process_irqs(self):
        """Process pending interrupts."""
        while self._irq_pending:
            irq = self._irq_pending.pop(0)
            handler = self._irq_handlers.get(irq)
            if handler:
                handler(self)

    def step(self):
        if self.pc >= len(self._instructions):
            if self._instructions:
                self._output.append(f"[VM] PC out of bounds: {self.pc}")
                self._running = False
            return False
        inst = self._instructions[self.pc]
        self._step_count += 1
        if self._tracing:
            self._record_trace(inst)
        old_pc = self.pc
        self._pc_changed = False
        try:
            self._dispatch(inst)
        except Halt:
            self._running = False
            return False
        if not self._pc_changed and self.pc == old_pc:
            self.pc += 1
        return True

    def run(self, max_steps=None):
        if max_steps is not None:
            self._max_instructions = max_steps
        self._running = True
        while self._running and self._step_count < self._max_instructions:
            self._process_irqs()
            if not self.step():
                break
        if self._step_count >= self._max_instructions:
            self._output.append(f"[VM] instruction limit ({self._max_instructions})")
        return self._output

    def _dispatch(self, inst):
        handler = _OPCODE_TABLE.get(inst.opcode)
        if handler is None:
            self._output.append(f"[VM] unknown opcode: {inst.opcode}")
            return
        handler(self, inst.operands)

    def _reg(self, operand):
        if isinstance(operand, str) and operand.startswith("R") and operand[1:].isdigit():
            idx = int(operand[1:])
            if 0 <= idx < NUM_REGS:
                return idx
        raise InsFault(f"invalid register: {operand}")

    def _val(self, operand):
        if isinstance(operand, str) and operand.startswith("R") and operand[1:].isdigit():
            return self.regs[int(operand[1:])]
        return operand

    def _check_arity(self, ops, expected):
        if len(ops) < expected:
            raise InsFault(f"expected {expected} operands, got {len(ops)}")

    def _truthy(self, val):
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return val != 0
        if isinstance(val, np.ndarray):
            return val.size > 0 and bool(np.any(val))
        return bool(val)

    def _parse_tensor(self, val):
        if isinstance(val, np.ndarray):
            return val
        if isinstance(val, list):
            return np.array(val, dtype=np.float64)
        if isinstance(val, (int, float)):
            return np.float64(val)
        raise InsFault(f"cannot parse as tensor: {type(val).__name__}")

    def _record_trace(self, inst):
        regs_snapshot = {}
        for i, v in enumerate(self.regs):
            if v != 0:
                if isinstance(v, np.ndarray):
                    regs_snapshot[f"R{i}"] = np.array2string(v, precision=3, suppress_small=True)
                else:
                    regs_snapshot[f"R{i}"] = v
        self._trace.append(TraceEntry(
            cycle=self._step_count,
            pc=self.pc,
            instruction=f"{inst.opcode} {', '.join(str(o) for o in inst.operands)}",
            registers=regs_snapshot,
            heap_keys=list(self._memory._heap.keys()),
        ))

    def get_trace(self):
        return list(self._trace)

    def format_trace(self):
        lines = []
        for e in self._trace:
            regs = " ".join(f"{k}={v}" for k, v in e.registers.items())
            lines.append(f"[{e.cycle:05d}] PC={e.pc:04d}  {e.instruction:<36s}  {regs}")
        return lines


# ── Assembler ────────────────────────────────────────────────────────────────

class Assembler:
    """Assembly text to Instruction list."""

    _RE_REGISTER = re.compile(r"^R(\d+)$")
    _RE_INT = re.compile(r"^-?\d+$")
    _RE_FLOAT = re.compile(r"^-?\d+\.\d*$")
    _RE_STRING = re.compile(r'^"(.*)"$')
    _RE_TENSOR = re.compile(r"^\[(.+)\]$")

    @staticmethod
    def _is_int(s):
        try:
            int(s, 0)
            return True
        except ValueError:
            return False

    def assemble(self, source):
        lines = source.strip().split("\n")

        # Pass 1: collect labels
        labels = {}
        raw_lines = []
        for line in lines:
            line = line.split(";")[0].split("#")[0].strip()
            if not line:
                continue
            if ":" in line and not line.startswith("["):
                prefix, _, rest = line.partition(":")
                prefix = prefix.strip()
                if prefix and " " not in prefix and "," not in prefix:
                    labels[prefix] = len(raw_lines)
                    line = rest.strip()
                    if not line:
                        continue
            elif line.endswith(":") and " " not in line and "," not in line:
                labels[line[:-1]] = len(raw_lines)
                continue
            raw_lines.append(line)

        # Pass 2: parse instructions with resolved labels
        instructions = []
        for line_num, line in enumerate(raw_lines):
            parts = line.split(None, 1)
            opcode = parts[0].upper()
            operand_str = parts[1].strip() if len(parts) > 1 else ""

            operands = self._parse_operands(operand_str, labels)
            instructions.append(Instruction(
                opcode=opcode, operands=operands,
                line_num=line_num, raw=line,
            ))

        return instructions

    def _parse_operands(self, text, labels):
        if not text:
            return []
        operands = []
        for part in self._split_operands(text):
            part = part.strip()
            if not part:
                continue
            if part in labels:
                operands.append(labels[part])
            elif self._RE_REGISTER.match(part):
                operands.append(part)
            elif self._is_int(part):
                operands.append(int(part, 0))
            elif self._RE_FLOAT.match(part):
                operands.append(float(part))
            elif self._RE_STRING.match(part):
                operands.append(self._RE_STRING.match(part).group(1))
            elif self._RE_TENSOR.match(part):
                operands.append(part)
            else:
                operands.append(part)
        return operands

    def _split_operands(self, text):
        result = []
        current = []
        in_string = False
        in_tensor = 0
        for ch in text:
            if ch == '"':
                in_string = not in_string
                current.append(ch)
            elif ch == '[' and not in_string:
                in_tensor += 1
                current.append(ch)
            elif ch == ']' and not in_string:
                in_tensor -= 1
                current.append(ch)
            elif ch == ',' and not in_string and in_tensor == 0:
                result.append("".join(current))
                current = []
            else:
                current.append(ch)
        if current:
            result.append("".join(current))
        return result


ProgramLoader = Assembler  # backward-compatible alias


# ── VM Runner ────────────────────────────────────────────────────────────────

class VMRunner:
    """Convenience: assemble + run + trace."""

    def __init__(self, devices=None):
        self._assembler = Assembler()
        self._devices = devices or DeviceBus()
        self.cpu = None

    def assemble_and_run(self, source, trace=False, max_steps=None):
        instructions = self._assembler.assemble(source)
        self.cpu = CPU(devices=self._devices)
        self.cpu._tracing = trace
        self.cpu.load_program(instructions)
        return self.cpu.run(max_steps=max_steps)

    def disassemble(self, source):
        instructions = self._assembler.assemble(source)
        lines = []
        for i, inst in enumerate(instructions):
            ops = ", ".join(str(o) for o in inst.operands) if inst.operands else ""
            lines.append(f"  {i:04d}: {inst.opcode:<12s} {ops}")
        return lines


# ── Backward Compat Aliases ──────────────────────────────────────────────────

ProgramLoader = Assembler
VirtualCPU = CPU
Assembler.load = Assembler.assemble


# ═══════════════════════════════════════════════════════════════════════════════
# Opcode Handlers (formerly vm_alu.py)
# ═══════════════════════════════════════════════════════════════════════════════


# ── Integer ALU ──────────────────────────────────────────────────────────────

def _op_iadd(cpu, ops):
    cpu._check_arity(ops, 3)
    a, b = int(cpu._val(ops[1])), int(cpu._val(ops[2]))
    result = a + b
    cpu._carry_flag = result > 0xFFFFFFFF
    cpu.regs[cpu._reg(ops[0])] = result & 0xFFFFFFFF

def _op_isub(cpu, ops):
    cpu._check_arity(ops, 3)
    a, b = int(cpu._val(ops[1])), int(cpu._val(ops[2]))
    result = a - b
    cpu._carry_flag = result < 0
    cpu.regs[cpu._reg(ops[0])] = result

def _op_imul(cpu, ops):
    cpu._check_arity(ops, 3)
    cpu.regs[cpu._reg(ops[0])] = int(cpu._val(ops[1])) * int(cpu._val(ops[2]))

def _op_idiv(cpu, ops):
    cpu._check_arity(ops, 3)
    a, b = int(cpu._val(ops[1])), int(cpu._val(ops[2]))
    cpu.regs[cpu._reg(ops[0])] = a // b if b != 0 else 0

def _op_iand(cpu, ops):
    cpu._check_arity(ops, 3)
    cpu.regs[cpu._reg(ops[0])] = int(cpu._val(ops[1])) & int(cpu._val(ops[2]))

def _op_ior(cpu, ops):
    cpu._check_arity(ops, 3)
    cpu.regs[cpu._reg(ops[0])] = int(cpu._val(ops[1])) | int(cpu._val(ops[2]))

def _op_ixor(cpu, ops):
    cpu._check_arity(ops, 3)
    cpu.regs[cpu._reg(ops[0])] = int(cpu._val(ops[1])) ^ int(cpu._val(ops[2]))

def _op_ishl(cpu, ops):
    cpu._check_arity(ops, 3)
    cpu.regs[cpu._reg(ops[0])] = int(cpu._val(ops[1])) << int(cpu._val(ops[2]))

def _op_ishr(cpu, ops):
    cpu._check_arity(ops, 3)
    cpu.regs[cpu._reg(ops[0])] = int(cpu._val(ops[1])) >> int(cpu._val(ops[2]))

def _op_ineg(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = -int(cpu._val(ops[1]))

def _op_inc(cpu, ops):
    cpu._check_arity(ops, 1)
    idx = cpu._reg(ops[0])
    cpu.regs[idx] = int(cpu.regs[idx]) + 1

def _op_dec(cpu, ops):
    cpu._check_arity(ops, 1)
    idx = cpu._reg(ops[0])
    cpu.regs[idx] = int(cpu.regs[idx]) - 1

def _op_icmp(cpu, ops):
    cpu._check_arity(ops, 2)
    a, b = int(cpu._val(ops[0])), int(cpu._val(ops[1]))
    cpu._cmp_flag = -1 if a < b else (1 if a > b else 0)


# ── Float ALU ──────────────────────────────────────────────────────────────

def _op_fadd(cpu, ops):
    cpu._check_arity(ops, 3)
    cpu.regs[cpu._reg(ops[0])] = float(cpu._val(ops[1])) + float(cpu._val(ops[2]))

def _op_fsub(cpu, ops):
    cpu._check_arity(ops, 3)
    cpu.regs[cpu._reg(ops[0])] = float(cpu._val(ops[1])) - float(cpu._val(ops[2]))

def _op_fmul(cpu, ops):
    cpu._check_arity(ops, 3)
    cpu.regs[cpu._reg(ops[0])] = float(cpu._val(ops[1])) * float(cpu._val(ops[2]))

def _op_fdiv(cpu, ops):
    cpu._check_arity(ops, 3)
    b = float(cpu._val(ops[2]))
    if b == 0:
        raise InsFault("division by zero")
    cpu.regs[cpu._reg(ops[0])] = float(cpu._val(ops[1])) / b

def _op_fcmp(cpu, ops):
    cpu._check_arity(ops, 2)
    a, b = float(cpu._val(ops[0])), float(cpu._val(ops[1]))
    cpu._cmp_flag = -1 if a < b else (1 if a > b else 0)


# ── Stack Operations ───────────────────────────────────────────────────────

def _op_push(cpu, ops):
    cpu._check_arity(ops, 1)
    if cpu.sp <= 0:
        raise InsFault("stack overflow")
    cpu.sp -= 1
    cpu._stack[cpu.sp] = cpu._val(ops[0])

def _op_pop(cpu, ops):
    cpu._check_arity(ops, 1)
    if cpu.sp >= STACK_BASE:
        raise InsFault("stack underflow")
    cpu.regs[cpu._reg(ops[0])] = cpu._stack[cpu.sp]
    cpu.sp += 1


# ── Memory Operations ──────────────────────────────────────────────────────

def _op_alloc(cpu, ops):
    cpu._check_arity(ops, 2)
    size = int(cpu._val(ops[1]))
    name = f"_alloc_{cpu._step_count}"
    cpu._memory.store(name, np.zeros(size, dtype=np.float64))
    cpu.regs[cpu._reg(ops[0])] = size

def _op_meminfo(cpu, ops):
    cpu._check_arity(ops, 1)
    usage = cpu._memory.usage()
    cpu.regs[cpu._reg(ops[0])] = usage.get("entries", 0)


# ── I/O Operations ─────────────────────────────────────────────────────────

def _op_in(cpu, ops):
    cpu._check_arity(ops, 2)
    port = int(cpu._val(ops[1]))
    try:
        device = cpu._devices._devices.get(str(port))
        if device:
            if hasattr(device, 'read'):
                val = device.read()
                try:
                    val = int(val)
                except (ValueError, TypeError):
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        pass
                cpu.regs[cpu._reg(ops[0])] = val
            else:
                cpu.regs[cpu._reg(ops[0])] = device.info().get("status", 0)
        else:
            cpu.regs[cpu._reg(ops[0])] = 0
    except Exception:
        cpu.regs[cpu._reg(ops[0])] = 0

def _op_out(cpu, ops):
    cpu._check_arity(ops, 2)
    port = int(cpu._val(ops[0]))
    val = cpu._val(ops[1])
    try:
        device = cpu._devices._devices.get(str(port))
        if device and hasattr(device, 'write'):
            device.write(val)
    except Exception:
        pass


# ── Tensor ALU ───────────────────────────────────────────────────────────────

def _op_add(cpu, ops):
    cpu._check_arity(ops, 3)
    rd = cpu._reg(ops[0])
    a, b = cpu._val(ops[1]), cpu._val(ops[2])
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        cpu.regs[rd] = cpu._parse_tensor(a) + cpu._parse_tensor(b)
    else:
        cpu.regs[rd] = (a or 0) + (b or 0)

def _op_sub(cpu, ops):
    cpu._check_arity(ops, 3)
    rd = cpu._reg(ops[0])
    a, b = cpu._val(ops[1]), cpu._val(ops[2])
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        cpu.regs[rd] = cpu._parse_tensor(a) - cpu._parse_tensor(b)
    else:
        cpu.regs[rd] = (a or 0) - (b or 0)

def _op_mul(cpu, ops):
    cpu._check_arity(ops, 3)
    rd = cpu._reg(ops[0])
    a, b = cpu._val(ops[1]), cpu._val(ops[2])
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        cpu.regs[rd] = cpu._parse_tensor(a) * cpu._parse_tensor(b)
    else:
        cpu.regs[rd] = (a or 0) * (b or 0)

def _op_div(cpu, ops):
    cpu._check_arity(ops, 3)
    rd = cpu._reg(ops[0])
    a = cpu._parse_tensor(cpu._val(ops[1]))
    b = cpu._parse_tensor(cpu._val(ops[2]))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = a / b
        result = np.where(np.isinf(result), 0.0, result)
        result = np.where(np.isnan(result), 0.0, result)
    cpu.regs[rd] = result

def _op_neg(cpu, ops):
    cpu._check_arity(ops, 2)
    rd = cpu._reg(ops[0])
    a = cpu._val(ops[1])
    cpu.regs[rd] = -cpu._parse_tensor(a) if isinstance(a, np.ndarray) else -(a or 0)

def _op_abs(cpu, ops):
    cpu._check_arity(ops, 2)
    rd = cpu._reg(ops[0])
    a = cpu._val(ops[1])
    cpu.regs[rd] = np.abs(a) if isinstance(a, np.ndarray) else abs(a or 0)

def _op_matmul(cpu, ops):
    cpu._check_arity(ops, 3)
    rd = cpu._reg(ops[0])
    a = cpu._parse_tensor(cpu._val(ops[1]))
    b = cpu._parse_tensor(cpu._val(ops[2]))
    if a.ndim == 0:
        a = a.reshape(1, 1)
    if b.ndim == 0:
        b = b.reshape(1, 1)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    cpu.regs[rd] = a @ b

def _op_transpose(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = cpu._parse_tensor(cpu._val(ops[1])).T

def _op_dot(cpu, ops):
    cpu._check_arity(ops, 3)
    a = cpu._parse_tensor(cpu._val(ops[1])).ravel()
    b = cpu._parse_tensor(cpu._val(ops[2])).ravel()
    cpu.regs[cpu._reg(ops[0])] = float(np.dot(a, b))

def _op_norm(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = float(np.linalg.norm(cpu._parse_tensor(cpu._val(ops[1]))))

def _op_sum(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = float(np.sum(cpu._parse_tensor(cpu._val(ops[1]))))

def _op_mean(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = float(np.mean(cpu._parse_tensor(cpu._val(ops[1]))))

def _op_max(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = float(np.max(cpu._parse_tensor(cpu._val(ops[1]))))

def _op_argmax(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = int(np.argmax(cpu._parse_tensor(cpu._val(ops[1]))))

def _op_reshape(cpu, ops):
    cpu._check_arity(ops, 4)
    a = cpu._parse_tensor(cpu._val(ops[1]))
    rows = int(ops[2]) if isinstance(ops[2], (int, float)) else -1
    cols = int(ops[3]) if isinstance(ops[3], (int, float)) else -1
    cpu.regs[cpu._reg(ops[0])] = a.reshape(rows, cols)

def _op_shape(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = list(cpu._parse_tensor(cpu._val(ops[1])).shape)

def _op_size(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = int(cpu._parse_tensor(cpu._val(ops[1])).size)

def _op_relu(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = np.maximum(0, cpu._parse_tensor(cpu._val(ops[1])))

def _op_gelu(cpu, ops):
    cpu._check_arity(ops, 2)
    a = cpu._parse_tensor(cpu._val(ops[1]))
    cpu.regs[cpu._reg(ops[0])] = 0.5 * a * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (a + 0.044715 * a ** 3)))

def _op_sigmoid(cpu, ops):
    cpu._check_arity(ops, 2)
    a = cpu._parse_tensor(cpu._val(ops[1]))
    cpu.regs[cpu._reg(ops[0])] = 1.0 / (1.0 + np.exp(-np.clip(a, -500, 500)))

def _op_tanh(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = np.tanh(cpu._parse_tensor(cpu._val(ops[1])))

def _op_softmax(cpu, ops):
    cpu._check_arity(ops, 2)
    a = cpu._parse_tensor(cpu._val(ops[1]))
    shifted = a - np.max(a)
    exp_a = np.exp(shifted)
    cpu.regs[cpu._reg(ops[0])] = exp_a / np.sum(exp_a)

def _op_layernorm(cpu, ops):
    cpu._check_arity(ops, 2)
    a = cpu._parse_tensor(cpu._val(ops[1]))
    cpu.regs[cpu._reg(ops[0])] = (a - np.mean(a)) / np.sqrt(np.var(a) + 1e-5)

def _op_rmsnorm(cpu, ops):
    cpu._check_arity(ops, 2)
    a = cpu._parse_tensor(cpu._val(ops[1]))
    cpu.regs[cpu._reg(ops[0])] = a / np.sqrt(np.mean(a ** 2) + 1e-5)

def _op_randn(cpu, ops):
    cpu._check_arity(ops, 3)
    r = int(ops[1]) if isinstance(ops[1], (int, float)) else 1
    c = int(ops[2]) if isinstance(ops[2], (int, float)) else 1
    cpu.regs[cpu._reg(ops[0])] = np.random.randn(r, c)

def _op_randunif(cpu, ops):
    cpu._check_arity(ops, 5)
    r = int(ops[1]) if isinstance(ops[1], (int, float)) else 1
    c = int(ops[2]) if isinstance(ops[2], (int, float)) else 1
    lo = float(ops[3]) if isinstance(ops[3], (int, float)) else 0.0
    hi = float(ops[4]) if isinstance(ops[4], (int, float)) else 1.0
    cpu.regs[cpu._reg(ops[0])] = np.random.uniform(lo, hi, (r, c))


# ── Comparison ───────────────────────────────────────────────────────────────

def _op_cmp(cpu, ops):
    cpu._check_arity(ops, 2)
    a, b = cpu._val(ops[0]), cpu._val(ops[1])
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        diff = a - b
        cpu._cmp_flag = -1 if diff < 0 else (1 if diff > 0 else 0)
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if np.array_equal(a, b):
            cpu._cmp_flag = 0
        elif np.all(a < b):
            cpu._cmp_flag = -1
        elif np.all(a > b):
            cpu._cmp_flag = 1
        else:
            cpu._cmp_flag = 0
    else:
        cpu._cmp_flag = -1 if str(a) < str(b) else (1 if str(a) > str(b) else 0)

def _op_test(cpu, ops):
    cpu._check_arity(ops, 1)
    cpu._cmp_flag = 1 if cpu._truthy(cpu._val(ops[0])) else 0


# ── Control Flow ─────────────────────────────────────────────────────────────

def _resolve_label(cpu, operand):
    if isinstance(operand, int):
        return operand
    if isinstance(operand, str) and operand.isdigit():
        return int(operand)
    raise InsFault(f"invalid jump target: {operand}")

def _op_jmp(cpu, ops):
    cpu._check_arity(ops, 1)
    cpu.pc = _resolve_label(cpu, ops[0])
    cpu._pc_changed = True

def _op_jz(cpu, ops):
    cpu._check_arity(ops, 1)
    if cpu._cmp_flag == 0:
        cpu.pc = _resolve_label(cpu, ops[0])
        cpu._pc_changed = True

def _op_jnz(cpu, ops):
    cpu._check_arity(ops, 1)
    if cpu._cmp_flag != 0:
        cpu.pc = _resolve_label(cpu, ops[0])
        cpu._pc_changed = True

def _op_jgt(cpu, ops):
    cpu._check_arity(ops, 1)
    if cpu._cmp_flag > 0:
        cpu.pc = _resolve_label(cpu, ops[0])
        cpu._pc_changed = True

def _op_jge(cpu, ops):
    cpu._check_arity(ops, 1)
    if cpu._cmp_flag >= 0:
        cpu.pc = _resolve_label(cpu, ops[0])
        cpu._pc_changed = True

def _op_jlt(cpu, ops):
    cpu._check_arity(ops, 1)
    if cpu._cmp_flag < 0:
        cpu.pc = _resolve_label(cpu, ops[0])
        cpu._pc_changed = True

def _op_jle(cpu, ops):
    cpu._check_arity(ops, 1)
    if cpu._cmp_flag <= 0:
        cpu.pc = _resolve_label(cpu, ops[0])
        cpu._pc_changed = True

def _op_call(cpu, ops):
    cpu._check_arity(ops, 1)
    if len(cpu._call_stack) >= MAX_CALL_DEPTH:
        cpu._output.append("[VM] call stack overflow")
        cpu._running = False
        raise Halt("call stack overflow")
    cpu._call_stack.append(cpu.pc + 1)
    cpu.pc = _resolve_label(cpu, ops[0])
    cpu._pc_changed = True

def _op_ret(cpu, ops):
    if not cpu._call_stack:
        raise InsFault("ret with empty call stack")
    cpu.pc = cpu._call_stack.pop()
    cpu._pc_changed = True

def _op_loop(cpu, ops):
    cpu._check_arity(ops, 2)
    idx = cpu._reg(ops[0])
    cpu.regs[idx] = int(cpu.regs[idx]) - 1
    if cpu.regs[idx] != 0:
        cpu.pc = _resolve_label(cpu, ops[1])
        cpu._pc_changed = True

def _op_halt(cpu, ops):
    raise Halt()


# ── System Calls ─────────────────────────────────────────────────────────────

# Syscall numbers (match kernel_syscall.SyscallNumber)
SYS_PRINT = 111
SYS_EXIT = 2
SYS_ALLOC = 20
SYS_FREE = 21
SYS_OPEN = 120
SYS_READ = 121
SYS_WRITE = 122
SYS_CLOSE = 123
SYS_UPTIME = 200
SYS_STATS = 201

# Kernel-provided syscall handler (set by Kernel or VirtualSystem)
_syscall_handler = None


def set_syscall_handler(handler):
    """Set the global syscall handler function.

    The handler receives (syscall_num, args) and returns a value.
    """
    global _syscall_handler
    _syscall_handler = handler


def _op_syscall(cpu, ops):
    """SYSCALL — software interrupt for kernel services.

    Convention:
      R7 = syscall number
      R0-R5 = arguments
      R0 = return value
    """
    num = int(cpu.regs[7])
    args = [cpu.regs[i] for i in range(6)]

    if _syscall_handler is not None:
        result = _syscall_handler(num, args)
        cpu.regs[0] = result if result is not None else 0
    else:
        cpu.regs[0] = 0


# ── Data Movement ────────────────────────────────────────────────────────────

def _op_load_const(cpu, ops):
    cpu._check_arity(ops, 2)
    rd = cpu._reg(ops[0])
    val = ops[1]
    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
        import json as _json
        try:
            parsed = _json.loads(val)
            if isinstance(parsed, list):
                cpu.regs[rd] = np.array(parsed, dtype=np.float64)
                return
        except (ValueError, TypeError):
            pass
        inner = val[1:-1].strip()
        if inner:
            parts = []
            for item in inner.split(","):
                item = item.strip()
                try:
                    parts.append(int(item))
                except ValueError:
                    try:
                        parts.append(float(item))
                    except ValueError:
                        parts.append(item)
            cpu.regs[rd] = np.array(parts, dtype=np.float64)
        else:
            cpu.regs[rd] = np.array([], dtype=np.float64)
    else:
        cpu.regs[rd] = val

def _op_load_shape(cpu, ops):
    cpu._check_arity(ops, 3)
    r = int(ops[1]) if isinstance(ops[1], (int, float)) else 1
    c = int(ops[2]) if isinstance(ops[2], (int, float)) else 1
    cpu.regs[cpu._reg(ops[0])] = np.zeros((r, c))

def _op_mov(cpu, ops):
    cpu._check_arity(ops, 2)
    cpu.regs[cpu._reg(ops[0])] = cpu._val(ops[1])

def _op_store(cpu, ops):
    cpu._check_arity(ops, 2)
    key = str(ops[1])
    cpu._memory.store(key, cpu._val(ops[0]))

def _op_load(cpu, ops):
    cpu._check_arity(ops, 2)
    key = str(ops[1])
    cpu.regs[cpu._reg(ops[0])] = cpu._memory.load(key)

def _op_free(cpu, ops):
    cpu._check_arity(ops, 1)
    cpu._memory.free(str(ops[0]))

def _op_print(cpu, ops):
    cpu._check_arity(ops, 1)
    val = cpu._val(ops[0])
    if isinstance(val, np.ndarray):
        cpu._output.append(np.array2string(val, precision=4, suppress_small=True))
    else:
        cpu._output.append(str(val))

def _op_nop(cpu, ops):
    pass


# ── Device Bus Ops ───────────────────────────────────────────────────────────

def _op_dev_open(cpu, ops):
    cpu._check_arity(ops, 2)
    rd = cpu._reg(ops[0])
    name = str(cpu._val(ops[1]))
    dev = cpu._devices.open(name)
    cpu.regs[rd] = name

def _op_dev_call(cpu, ops):
    cpu._check_arity(ops, 3)
    rd = cpu._reg(ops[0])
    handle = str(cpu._val(ops[1]))
    method = str(cpu._val(ops[2]))
    extra_args = [cpu._val(o) for o in ops[3:]]
    dev = cpu._devices.open(handle)
    cpu.regs[rd] = cpu._devices.call(dev, method, *extra_args)

def _op_dev_close(cpu, ops):
    cpu._check_arity(ops, 1)
    pass

def _op_dev_info(cpu, ops):
    cpu._check_arity(ops, 2)
    rd = cpu._reg(ops[0])
    handle = str(cpu._val(ops[1]))
    dev = cpu._devices.open(handle)
    cpu.regs[rd] = cpu._devices.info(dev)


# ── Opcode Table ─────────────────────────────────────────────────────────────

_OPCODE_TABLE = {
    "LOAD_CONST": _op_load_const, "LOAD_SHAPE": _op_load_shape,
    "MOV": _op_mov, "STORE": _op_store, "LOAD": _op_load,
    "FREE": _op_free, "PRINT": _op_print, "NOP": _op_nop,
    "IADD": _op_iadd, "ISUB": _op_isub, "IMUL": _op_imul, "IDIV": _op_idiv,
    "IAND": _op_iand, "IOR": _op_ior, "IXOR": _op_ixor,
    "ISHL": _op_ishl, "ISHR": _op_ishr, "INEG": _op_ineg,
    "INC": _op_inc, "DEC": _op_dec, "ICMP": _op_icmp,
    "ADD": _op_add, "SUB": _op_sub, "MUL": _op_mul, "DIV": _op_div,
    "NEG": _op_neg, "ABS": _op_abs,
    "MATMUL": _op_matmul, "TRANSPOSE": _op_transpose,
    "DOT": _op_dot, "NORM": _op_norm,
    "SUM": _op_sum, "MEAN": _op_mean, "MAX": _op_max, "ARGMAX": _op_argmax,
    "RESHAPE": _op_reshape, "SHAPE": _op_shape, "SIZE": _op_size,
    "RELU": _op_relu, "GELU": _op_gelu, "SIGMOID": _op_sigmoid,
    "TANH": _op_tanh, "SOFTMAX": _op_softmax,
    "LAYERNORM": _op_layernorm, "RMSNORM": _op_rmsnorm,
    "RANDN": _op_randn, "RANDUNIF": _op_randunif,
    "CMP": _op_cmp, "TEST": _op_test,
    "JMP": _op_jmp, "JZ": _op_jz, "JNZ": _op_jnz,
    "JGT": _op_jgt, "JGE": _op_jge, "JLT": _op_jlt, "JLE": _op_jle,
    "CALL": _op_call, "RET": _op_ret, "LOOP": _op_loop, "HALT": _op_halt,
    "DEV_OPEN": _op_dev_open, "DEV_CALL": _op_dev_call,
    "DEV_CLOSE": _op_dev_close, "DEV_INFO": _op_dev_info,
    "PUSH": _op_push, "POP": _op_pop,
    "FADD": _op_fadd, "FSUB": _op_fsub, "FMUL": _op_fmul, "FDIV": _op_fdiv,
    "FCMP": _op_fcmp,
    "ALLOC": _op_alloc, "MEMINFO": _op_meminfo,
    "IN": _op_in, "OUT": _op_out,
    "SYSCALL": _op_syscall,
}


# ── Program Loader ───────────────────────────────────────────────────────────

class DiskProgramLoader:
    """Load and execute programs from a FlatFS filesystem.

    Programs are stored as assembly source (.asm) files. The loader reads
    the source, assembles it, and can execute it directly or return the
    compiled instructions.
    """

    def __init__(self, filesystem: FlatFS):
        self._fs = filesystem
        self._assembler = Assembler()

    def list_programs(self) -> list[str]:
        """List all .asm files on the filesystem."""
        return [f for f in self._fs.list_files() if f.endswith('.asm')]

    def load_source(self, name: str) -> str:
        """Read assembly source from filesystem."""
        if not name.endswith('.asm'):
            name = name + '.asm'
        data = self._fs.read(name)
        return data.decode('utf-8', errors='replace').rstrip('\x00')

    def assemble(self, source: str) -> list:
        """Assemble source into instructions."""
        return self._assembler.assemble(source)

    def run(self, name: str, max_steps: int = 10000,
            stdin_fn=None, stdout_fn=None) -> dict:
        """Load, assemble, and run a program. Returns output and stats."""
        source = self.load_source(name)
        instructions = self.assemble(source)

        bus = DeviceBus()
        if stdout_fn or stdin_fn:
            bus.register_console(stdin_fn=stdin_fn, stdout_fn=stdout_fn)
        cpu = CPU(devices=bus)
        cpu.load_program(instructions)
        output = cpu.run(max_steps=max_steps)

        return {
            "name": name,
            "output": output,
            "steps": cpu._step_count,
            "source": source,
        }

    def save_program(self, name: str, source: str) -> None:
        """Save assembly source to filesystem."""
        if not name.endswith('.asm'):
            name = name + '.asm'
        self._fs.write(name, source.encode('utf-8'))


# ── Integrated Virtual System ────────────────────────────────────────────────

class VirtualSystem:
    """Integrated virtual computer — CPU + Memory + DeviceBus + optional devices.

    Wires together the components into a single runnable system.
    """

    def __init__(self, enable_block: bool = False, enable_console: bool = True,
                 stdin_fn=None, stdout_fn=None, syscall_handler=None):
        self.memory = Memory()
        self.bus = DeviceBus()

        if enable_console:
            self.bus.register_console(stdin_fn=stdin_fn, stdout_fn=stdout_fn)
        if enable_block:
            self.block = BlockDevice()
            self.bus.register("block", self.block)

        self.cpu = CPU(memory=self.memory, devices=self.bus)

        if syscall_handler is not None:
            set_syscall_handler(syscall_handler)

    def load_program(self, source: str) -> int:
        """Assemble and load program. Returns instruction count."""
        assembler = Assembler()
        instructions = assembler.assemble(source)
        self.cpu.load_program(instructions)
        return len(instructions)

    def run(self, max_steps: int = 10000) -> list[str]:
        """Run the loaded program. Returns printed output."""
        return self.cpu.run(max_steps=max_steps)

    def status(self) -> dict:
        return {
            "pc": self.cpu.pc,
            "sp": self.cpu.sp,
            "regs": list(self.cpu.regs),
            "cmp_flag": self.cpu._cmp_flag,
            "carry_flag": self.cpu._carry_flag,
            "steps": self.cpu._step_count,
            "devices": self.bus.list_devices(),
            "heap_entries": len(self.memory._heap),
        }

    def reset(self) -> None:
        self.cpu = CPU(memory=self.memory, devices=self.bus)


# ── Re-exports from submodules ──────────────────────────────────────────────

from .vm_programs import (  # noqa: E402, F401
    HELLO_ASM, CLASSICAL_ASM, TENSOR_MATH_ASM, MATRIX_MUL_ASM,
    NEURAL_NET_ASM, LOOP_ASM, FUNCTION_ASM, MIXED_ASM,
    NPU_PROGRAM_ASM, COUNTER_ASM, FIB_ASM, COLLATZ_ASM,
    self_test,
)

from .vm_devices import (  # noqa: E402, F401
    TensorDevice, PythonExecDevice, SlonetDevice,
    MultimodalDevice, EngineDevice, SlonetTrainingDevice, NPUVMDevice,
)
