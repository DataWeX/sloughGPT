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


class DeviceBus:
    """Device registry and generic dispatch."""

    def __init__(self):
        self._devices = {}

    def register(self, name, device):
        self._devices[name] = device

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

    def load_program(self, instructions):
        self._instructions = list(instructions)
        self.pc = 0
        self._cmp_flag = 0
        self._call_stack.clear()
        self._running = False
        self._step_count = 0
        self._trace.clear()

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
    cpu.regs[cpu._reg(ops[0])] = int(cpu._val(ops[1])) + int(cpu._val(ops[2]))

def _op_isub(cpu, ops):
    cpu._check_arity(ops, 3)
    cpu.regs[cpu._reg(ops[0])] = int(cpu._val(ops[1])) - int(cpu._val(ops[2]))

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
}


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
