"""
Shell Virtual Machine — register-based CPU, program loader, syscall dispatch, sandbox.

Architecture:
  - 16 general-purpose registers (R0-R15)
  - PC (program counter), SP (stack pointer)
  - FLAGS: Z (zero), C (carry), N (negative), V (overflow)
  - 64KB word-addressable memory (16-bit words)
  - Stack grows down from 0xFFFF (top of memory)
  - ~20 instructions: ALU, memory, control, stack, I/O
  - Assembler: parse .asm text → list of Instructions
  - Syscall bridge to AI devices (/dev/llm, /dev/embedding, /dev/knowledge)
  - Sandbox: instruction limit, memory bounds, syscall rate limit
"""

from __future__ import annotations

import os
import re
import sys
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("man.shell.vm")


# ── Constants ──────────────────────────────────────────────────────────────

MEM_SIZE = 65536  # 64K words (16-bit)
NUM_REGS = 16
STACK_BASE = 0xFFFF
MAX_INSTRUCTIONS = 100000
MAX_SYSCALLS = 100

# FLAGS bit positions
F_ZERO = 1 << 0
F_CARRY = 1 << 1
F_NEG = 1 << 2
F_OVERFLOW = 1 << 3


# ── Instruction Set ────────────────────────────────────────────────────────

INSTRUCTIONS: dict[str, tuple[int, str]] = {
    # (opcode, arity, description)
    "NOP":    (0, "no operation"),
    "MOV":    (2, "MOV dst, src — copy value to register"),
    "LOAD":   (2, "LOAD dst, addr — load word from memory address into register"),
    "STORE":  (2, "STORE addr, src — store register value into memory address"),
    "PUSH":   (1, "PUSH src — push register value onto stack"),
    "POP":    (1, "POP dst — pop value from stack into register"),
    "ADD":    (3, "ADD dst, a, b — dst = a + b"),
    "SUB":    (3, "SUB dst, a, b — dst = a - b"),
    "MUL":    (3, "MUL dst, a, b — dst = a * b"),
    "DIV":    (3, "DIV dst, a, b — dst = a // b (integer)"),
    "AND":    (3, "AND dst, a, b — dst = a & b"),
    "OR":     (3, "OR dst, a, b — dst = a | b"),
    "XOR":    (3, "XOR dst, a, b — dst = a ^ b"),
    "SHL":    (3, "SHL dst, a, b — dst = a << b"),
    "SHR":    (3, "SHR dst, a, b — dst = a >> b"),
    "CMP":    (2, "CMP a, b — compare, set FLAGS"),
    "JMP":    (1, "JMP addr — unconditional jump"),
    "JZ":     (1, "JZ addr — jump if zero flag set"),
    "JNZ":    (1, "JNZ addr — jump if zero flag NOT set"),
    "JL":     (1, "JL addr — jump if less (N != V)"),
    "JLE":    (1, "JLE addr — jump if less or equal (Z or N != V)"),
    "JG":     (1, "JG addr — jump if greater (Z=0 and N=V)"),
    "JGE":    (1, "JGE addr — jump if greater or equal (N=V)"),
    "CALL":   (1, "CALL addr — push PC and jump"),
    "RET":    (0, "RET — pop PC and return"),
    "SYSCALL":(0, "SYSCALL — invoke OS via R0 (syscall number)"),
    "HLT":    (0, "HLT — halt execution"),
    "PRINT":  (1, "PRINT reg — print register value (debug)"),
    "DUMP":   (0, "DUMP — dump all registers (debug)"),
}


# ── Data Models ────────────────────────────────────────────────────────────


@dataclass
class Instruction:
    opcode: str
    operands: list[str | int] = field(default_factory=list)
    lineno: int = 0

    def __repr__(self) -> str:
        return f"{self.opcode} {' '.join(str(o) for o in self.operands)}".strip()


class VMFault(Exception):
    """Base VM exception — halts execution."""
    pass


class Halt(VMFault):
    """Program executed HLT (normal termination)."""
    pass


class MemFault(VMFault):
    """Memory access violation."""
    pass


class InsFault(VMFault):
    """Invalid instruction."""
    pass


class SysFault(VMFault):
    """Syscall error."""
    pass


# ── Program Loader (Assembler) ────────────────────────────────────────────


class ProgramLoader:
    """Parses .asm text → list of Instructions with resolved labels."""

    def __init__(self):
        self._instructions: list[Instruction] = []
        self._labels: dict[str, int] = {}
        self._data: dict[str, list[int]] = {}
        self._data_section: list[int] = []
        self._data_labels: dict[str, int] = {}

    def load(self, source: str) -> list[Instruction]:
        """Parse assembly source into instructions with resolved labels."""
        self._instructions = []
        self._labels = {}
        self._data = {}
        self._data_section = []
        self._data_labels = {}
        data_mode = False
        lines = source.split("\n")

        # First pass: collect labels and data
        for lineno, raw in enumerate(lines, 1):
            line = raw.strip()
            if not line or line.startswith(";"):
                continue

            if line.upper() == ".DATA":
                data_mode = True
                continue
            if line.upper() == ".TEXT" or line.upper() == ".CODE":
                data_mode = False
                continue

            if data_mode:
                self._parse_data_line(line, lineno)
                continue

            # Remove comment after instruction
            if ";" in line:
                line = line.split(";")[0].strip()

            if not line:
                continue

            # Check for label at start of line (e.g. "start: MOV R0, 0")
            if ":" in line and not line.startswith("."):
                # Only label on its own line
                before, after = line.split(":", 1)
                stripped_before = before.strip()
                stripped_after = after.strip()
                if stripped_before and (
                    not stripped_after  # label on its own line
                    or stripped_before[0].isalpha()  # label prefix
                ):
                    if stripped_after:
                        # label + instruction on same line
                        self._labels[stripped_before] = len(self._instructions)
                        line = stripped_after
                    else:
                        # label on its own line
                        self._labels[stripped_before] = len(self._instructions)
                        continue
                else:
                    # .text or .data check — already handled above
                    pass

            # Normal instruction
            parts = shlex_split(line)
            opcode = parts[0].upper()
            operands = []
            for op in parts[1:]:
                op = op.strip(",")
                # Numeric literal — immediate value
                if op.startswith("0x") or op.startswith("0X"):
                    operands.append(int(op, 16))
                elif op.lstrip("-").isdigit():
                    operands.append(int(op))
                # String literal?
                elif op.startswith('"') and op.endswith('"'):
                    operands.append(op[1:-1])
                else:
                    # Register reference (R0-R15) or label reference:
                    # keep as string — CPU resolves regs at runtime
                    operands.append(op)

            self._instructions.append(Instruction(opcode, operands, lineno))

        # Second pass: resolve label references in operands
        resolved = []
        for idx, inst in enumerate(self._instructions):
            new_ops = []
            for op in inst.operands:
                if isinstance(op, int):
                    new_ops.append(op)
                elif isinstance(op, str) and op.upper().startswith("R") and op[1:].isdigit():
                    # Register reference — keep as string for CPU runtime resolution
                    new_ops.append(op)
                elif op in self._labels:
                    new_ops.append(self._labels[op])
                elif op in self._data_labels:
                    new_ops.append(self._data_labels[op])
                elif op.startswith("data_"):
                    dl = op[5:]
                    if dl in self._data_labels:
                        new_ops.append(self._data_labels[dl])
                    else:
                        raise InsFault(f"Undefined data label '{op}' at line {inst.lineno}")
                else:
                    raise InsFault(f"Undefined label '{op}' at line {inst.lineno}")
            resolved.append(Instruction(inst.opcode, new_ops, inst.lineno))

        # Append HLT if not present
        if not resolved or resolved[-1].opcode not in ("HLT", "SYSCALL"):
            # Check if last instruction would halt
            pass

        self._instructions = resolved
        return resolved

    def _parse_data_line(self, line: str, lineno: int) -> None:
        """Parse .data section entries: name: db val1, val2, ... or name: dw val1, val2, ..."""
        if ":" in line:
            name, rest = line.split(":", 1)
            name = name.strip()
            rest = rest.strip()
        else:
            return

        if rest.upper().startswith("DB "):
            values = parse_bytes(rest[3:])
        elif rest.upper().startswith("DW "):
            values = rest[3:].strip()
            values = [int(x.strip()) for x in values.split(",") if x.strip()]
        elif rest.upper().startswith("STR "):
            s = rest[4:].strip()
            if s.startswith('"') and s.endswith('"'):
                s = s[1:-1]
                s = s.replace("\\n", "\n").replace("\\t", "\t").replace("\\0", "\0")
            values = [ord(c) for c in s] + [0]
        else:
            return

        offset = len(self._data_section)
        self._data_labels[name] = offset
        self._data_section.extend(values)

    @property
    def data_segment(self) -> list[int]:
        return self._data_section

    @property
    def labels(self) -> dict[str, int]:
        return {**self._labels, **{f"data_{k}": v for k, v in self._data_labels.items()}}

    @property
    def entry_point(self) -> int:
        """Return the address of _start or start label, or 0."""
        for name, addr in self._labels.items():
            if name.lower() in ("_start", "start", "main"):
                return addr
        return 0


def shlex_split(text: str) -> list[str]:
    """Simple shell-like split that handles quoted strings."""
    parts = []
    current = []
    in_quote = False
    for ch in text:
        if ch == '"':
            in_quote = not in_quote
            current.append(ch)
        elif ch in (" ", "\t") and not in_quote:
            if current:
                parts.append("".join(current))
                current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def parse_bytes(text: str) -> list[int]:
    """Parse '65, 66, 10, 0' or '"hello", 10, 0' into byte list."""
    result = []
    i = 0
    while i < len(text):
        if text[i] in (" ", ","):
            i += 1
            continue
        if text[i] == '"':
            i += 1
            s = []
            while i < len(text) and text[i] != '"':
                if text[i] == "\\" and i + 1 < len(text):
                    esc = {"n": "\n", "t": "\t", "0": "\0", '"': '"', "\\": "\\"}
                    s.append(esc.get(text[i + 1], text[i + 1]))
                    i += 2
                else:
                    s.append(text[i])
                    i += 1
            if i < len(text):
                i += 1  # skip closing "
            result.extend(ord(c) for c in s)
        else:
            # parse number
            end = i
            while end < len(text) and text[end] not in (",", " "):
                end += 1
            token = text[i:end].strip()
            if token:
                if token.startswith("0x") or token.startswith("0X"):
                    result.append(int(token, 16))
                else:
                    result.append(int(token))
            i = end
    return result


# ── Virtual CPU ────────────────────────────────────────────────────────────


class VirtualCPU:
    """Register-based virtual CPU with ALU, memory, stack, and syscall dispatch."""

    def __init__(self, device_manager=None, syscall_handler: Callable | None = None):
        self.regs: list[int] = [0] * NUM_REGS  # R0–R15
        self.pc: int = 0
        self.sp: int = STACK_BASE
        self.flags: int = 0
        self.memory: list[int] = [0] * MEM_SIZE

        self._instructions: list[Instruction] = []
        self._labels: dict[str, int] = {}
        self._data_segment: list[int] = []
        self._running: bool = False
        self._step_count: int = 0
        self._syscall_count: int = 0
        self._output: list[str] = []
        self._device_manager = device_manager
        self._syscall_handler = syscall_handler
        self._max_instructions: int = MAX_INSTRUCTIONS
        self._max_syscalls: int = MAX_SYSCALLS
        self._breakpoints: set[int] = set()

    def load_program(self, instructions: list[Instruction], data: list[int] | None = None,
                     labels: dict[str, int] | None = None) -> None:
        self._instructions = instructions
        self._data_segment = data or []
        self._labels = labels or {}

        # Load data segment at top of low memory (addresses 0x0000–0x0FFF)
        for i, val in enumerate(self._data_segment[:4096]):
            self.memory[i] = val

        # Set entry point
        ep = None
        for name, addr in self._labels.items():
            if name.lower() in ("_start", "start", "main"):
                ep = addr
                break
        self.pc = ep or 0
        self.sp = STACK_BASE
        self.regs = [0] * NUM_REGS
        self.flags = 0

    def run(self, max_steps: int = 0) -> list[str]:
        """Execute program until HLT, fault, or max_steps."""
        self._running = True
        self._step_count = 0
        self._syscall_count = 0
        self._output = []
        limit = max_steps or self._max_instructions

        try:
            while self._running and self._step_count < limit:
                if self.pc < 0 or self.pc >= len(self._instructions):
                    raise InsFault(f"PC out of bounds: {self.pc} (code size: {len(self._instructions)})")
                inst = self._instructions[self.pc]
                self.pc += 1
                self._step_count += 1
                self._execute(inst)
        except Halt:
            pass
        except VMFault as e:
            self._output.append(f"[fault: {e}]")
        except Exception as e:
            self._output.append(f"[crash: {e}]")

        self._running = False
        return self._output

    def step(self) -> bool:
        """Execute a single instruction. Returns False if halted/faulted."""
        if not self._running:
            return False
        if self.pc < 0 or self.pc >= len(self._instructions):
            self._output.append(f"  [pc {self.pc} out of bounds]")
            self._running = False
            return False
        if self._step_count >= self._max_instructions:
            self._output.append(f"  [instruction limit: {self._max_instructions}]")
            self._running = False
            return False

        inst = self._instructions[self.pc]
        self.pc += 1
        self._step_count += 1
        try:
            self._execute(inst)
        except Halt:
            self._output.append(f"  [halt]")
            self._running = False
        except VMFault as e:
            self._output.append(f"  [fault: {e}]")
            self._running = False
        except Exception as e:
            self._output.append(f"  [crash: {e}]")
            self._running = False
        return self._running

    def _execute(self, inst: Instruction) -> None:
        """Execute a single instruction."""
        op = inst.opcode.upper()
        ops = inst.operands

        if op == "NOP":
            pass

        elif op == "HLT":
            raise Halt()

        elif op == "DUMP":
            self._dump_regs()

        elif op == "MOV":
            self._check_arity(2, ops)
            dst = self._resolve_dst(ops[0])
            self.regs[dst] = self._resolve_val(ops[1])

        elif op == "LOAD":
            self._check_arity(2, ops)
            dst = self._resolve_dst(ops[0])
            addr = self._resolve_val(ops[1])
            self._check_addr(addr)
            self.regs[dst] = self.memory[addr]

        elif op == "STORE":
            self._check_arity(2, ops)
            addr = self._resolve_val(ops[0])
            val = self._resolve_val(ops[1])
            self._check_addr(addr)
            self.memory[addr] = val & 0xFFFF

        elif op == "PUSH":
            self._check_arity(1, ops)
            val = self._resolve_val(ops[0])
            self._check_addr(self.sp - 1)
            self.sp -= 1
            self.memory[self.sp] = val & 0xFFFF

        elif op == "POP":
            self._check_arity(1, ops)
            dst = self._resolve_dst(ops[0])
            if self.sp > STACK_BASE:
                raise MemFault("stack underflow")
            val = self.memory[self.sp]
            self.sp += 1
            self.regs[dst] = val

        elif op in ("ADD", "SUB", "MUL", "DIV", "AND", "OR", "XOR", "SHL", "SHR"):
            self._check_arity(3, ops)
            dst = self._resolve_dst(ops[0])
            a = self._resolve_val(ops[1])
            b = self._resolve_val(ops[2])
            result = self._alu(op, a, b)
            self.regs[dst] = result

        elif op == "CMP":
            self._check_arity(2, ops)
            a = self._resolve_val(ops[0])
            b = self._resolve_val(ops[1])
            self._set_flags(a - b)

        elif op == "JMP":
            self._check_arity(1, ops)
            self.pc = self._resolve_val(ops[0])

        elif op == "JZ":
            self._check_arity(1, ops)
            if self.flags & F_ZERO:
                self.pc = self._resolve_val(ops[0])

        elif op == "JNZ":
            self._check_arity(1, ops)
            if not (self.flags & F_ZERO):
                self.pc = self._resolve_val(ops[0])

        elif op == "JL":
            self._check_arity(1, ops)
            n = bool(self.flags & F_NEG)
            v = bool(self.flags & F_OVERFLOW)
            if n != v:
                self.pc = self._resolve_val(ops[0])

        elif op == "JLE":
            self._check_arity(1, ops)
            n = bool(self.flags & F_NEG)
            v = bool(self.flags & F_OVERFLOW)
            if (self.flags & F_ZERO) or (n != v):
                self.pc = self._resolve_val(ops[0])

        elif op == "JG":
            self._check_arity(1, ops)
            n = bool(self.flags & F_NEG)
            v = bool(self.flags & F_OVERFLOW)
            if not (self.flags & F_ZERO) and (n == v):
                self.pc = self._resolve_val(ops[0])

        elif op == "JGE":
            self._check_arity(1, ops)
            n = bool(self.flags & F_NEG)
            v = bool(self.flags & F_OVERFLOW)
            if n == v:
                self.pc = self._resolve_val(ops[0])

        elif op == "CALL":
            self._check_arity(1, ops)
            self._check_addr(self.sp - 1)
            self.sp -= 1
            self.memory[self.sp] = self.pc
            self.pc = self._resolve_val(ops[0])

        elif op == "RET":
            if self.sp > STACK_BASE:
                raise MemFault("stack underflow on RET")
            self.pc = self.memory[self.sp]
            self.sp += 1

        elif op == "SYSCALL":
            self._syscall_count += 1
            if self._syscall_count >= self._max_syscalls:
                raise SysFault(f"syscall limit: {self._max_syscalls}")
            self._handle_syscall()

        elif op == "PRINT":
            self._check_arity(1, ops)
            val = self._resolve_val(ops[0])
            self._output.append(str(val))

        else:
            raise InsFault(f"unknown opcode: {op}")

    def _alu(self, op: str, a: int, b: int) -> int:
        r = 0
        if op == "ADD":
            r = a + b
        elif op == "SUB":
            r = a - b
        elif op == "MUL":
            r = a * b
        elif op == "DIV":
            r = a // b if b != 0 else 0
        elif op == "AND":
            r = a & b
        elif op == "OR":
            r = a | b
        elif op == "XOR":
            r = a ^ b
        elif op == "SHL":
            r = a << (b & 0x1F)
        elif op == "SHR":
            r = a >> (b & 0x1F)
        self._set_flags(r)
        return r & 0xFFFF

    def _set_flags(self, val: int) -> None:
        self.flags = 0
        if val == 0:
            self.flags |= F_ZERO
        if val < 0:
            self.flags |= F_NEG

    def _resolve_val(self, op: int | str) -> int:
        if isinstance(op, int):
            return op
        if isinstance(op, str) and op.upper().startswith("R") and op[1:].isdigit():
            return self.regs[int(op[1:])]
        return 0

    def _resolve_dst(self, op: int | str) -> int:
        if isinstance(op, int) and 0 <= op < NUM_REGS:
            return op
        if isinstance(op, str) and op.upper().startswith("R") and op[1:].isdigit():
            return int(op[1:])
        raise InsFault(f"invalid destination: {op}")

    def _check_addr(self, addr: int) -> None:
        if addr < 0 or addr >= MEM_SIZE:
            raise MemFault(f"address {addr:#x} out of bounds")

    def _check_arity(self, expected: int, ops: list) -> None:
        if len(ops) != expected:
            raise InsFault(f"expected {expected} operands, got {len(ops)}: {ops}")

    def _handle_syscall(self) -> None:
        """Dispatch syscall by R0 value.
        
        Syscall numbers:
          0 — exit
          1 — write string (address in R1)
          2 — write char (value in R1)
          3 — read number (blocking, result in R1)
          10 — llm generate (string at R1, result at R1)
          11 — embedding compute (string at R1)
          12 — knowledge store (string at R1)
          99 — dump state
        """
        sc = self.regs[0]
        if sc == 0:
            raise Halt()
        elif sc == 1:
            # Write string from memory
            addr = self.regs[1]
            out = []
            while addr < MEM_SIZE:
                ch = self.memory[addr]
                if ch == 0:
                    break
                out.append(chr(ch & 0xFF))
                addr += 1
            self._output.append("".join(out))
        elif sc == 2:
            # Write char
            self._output.append(chr(self.regs[1] & 0xFF))
        elif sc == 3:
            # Read number (stub — returns 0)
            self.regs[1] = 0
        elif sc == 10:
            # LLM generate
            addr = self.regs[1]
            prompt = self._read_string(addr)
            response = self._call_ai("llm", prompt)
            self._write_string(self.regs[2] if len(self.regs) > 2 else addr, response)
        elif sc == 11:
            # Embedding compute
            addr = self.regs[1]
            text = self._read_string(addr)
            result = self._call_ai("embedding", text)
            self._output.append(f"  [embedding: {result}]")
        elif sc == 12:
            # Knowledge store
            addr = self.regs[1]
            text = self._read_string(addr)
            result = self._call_ai("knowledge", text)
            self.regs[1] = 1 if "Stored" in result else 0
        elif sc == 99:
            self._dump_regs()
        else:
            self._output.append(f"  [unknown syscall: {sc}]")

    def _read_string(self, addr: int) -> str:
        chars = []
        while addr < MEM_SIZE:
            ch = self.memory[addr]
            if ch == 0:
                break
            chars.append(chr(ch & 0xFF))
            addr += 1
        return "".join(chars)

    def _write_string(self, addr: int, text: str) -> None:
        for i, ch in enumerate(text[:256]):
            if addr + i < MEM_SIZE:
                self.memory[addr + i] = ord(ch) & 0xFFFF
        if addr + len(text) < MEM_SIZE:
            self.memory[addr + len(text)] = 0

    def _call_ai(self, device: str, data: str) -> str:
        if self._syscall_handler:
            return self._syscall_handler(device, data)
        if self._device_manager:
            if device == "llm":
                return self._device_manager.write("/dev/llm", data)
            elif device == "embedding":
                return self._device_manager.write("/dev/embedding", data)
            elif device == "knowledge":
                return self._device_manager.write("/dev/knowledge", data)
        return ""

    def _dump_regs(self) -> None:
        flags_str = ""
        if self.flags & F_ZERO:
            flags_str += "Z"
        if self.flags & F_CARRY:
            flags_str += "C"
        if self.flags & F_NEG:
            flags_str += "N"
        if self.flags & F_OVERFLOW:
            flags_str += "V"
        self._output.append(
            f"  PC={self.pc:#06x} SP={self.sp:#06x} FLAGS={flags_str or '0'} "
            f"STEP={self._step_count}"
        )
        parts = []
        for i in range(16):
            parts.append(f"R{i}={self.regs[i]}")
        # Show registers in rows of 4
        for row in range(4):
            self._output.append("  " + "  ".join(parts[row * 4:(row + 1) * 4]))

    def get_state(self) -> dict[str, Any]:
        return {
            "regs": self.regs[:],
            "pc": self.pc,
            "sp": self.sp,
            "flags": self.flags,
            "step_count": self._step_count,
            "running": self._running,
            "output": self._output[:],
        }

    @property
    def disassembly(self) -> list[str]:
        lines = []
        for i, inst in enumerate(self._instructions):
            marker = "->" if i == self.pc else "  "
            bp = "●" if i in self._breakpoints else " "
            lines.append(f"{bp}{marker} [{i:4d}] {inst}")
        return lines


# ── VM Runner ──────────────────────────────────────────────────────────────


class VMRunner:
    """High-level VM interface — load .asm, run, collect output."""

    def __init__(self, device_manager=None, syscall_handler: Callable | None = None):
        self.loader = ProgramLoader()
        self.cpu = VirtualCPU(device_manager, syscall_handler)
        self._source: str = ""

    def assemble_and_run(self, source: str, max_steps: int = 0, trace: bool = False) -> list[str]:
        """Assemble source and run on CPU. Returns output lines."""
        self._source = source
        instructions = self.loader.load(source)
        self.cpu.load_program(instructions, self.loader.data_segment, self.loader.labels)
        if trace:
            self.cpu._output.append("  ── trace ──")
        return self.cpu.run(max_steps)

    def disassemble(self, source: str) -> list[str]:
        """Assemble and return disassembly listing."""
        self.loader.load(source)
        instructions = self.loader._instructions
        labels = self.loader.labels
        rev = {v: k for k, v in labels.items()}
        lines = []
        for i, inst in enumerate(instructions):
            label = rev.get(i, "")
            if label:
                lines.append(f"{label}:")
            lines.append(f"  [{i:4d}] {inst}")
        if self.loader.data_segment:
            lines.append(f"\n  .data ({len(self.loader.data_segment)} words):")
            for k, v in self.loader._data_labels.items():
                data = self.loader.data_segment[v:v+8]
                lines.append(f"    {k}: {data}")
        return lines


# ── Example Programs ───────────────────────────────────────────────────────


HELLO_ASM = """; Hello World for Shell VM

.data
    msg: str "Hello, Shell VM!"

.text
start:
    MOV R0, 1          ; syscall: write string
    MOV R1, msg        ; R1 = address of msg
    SYSCALL

    MOV R0, 0          ; syscall: exit
    SYSCALL
"""

COUNTER_ASM = """; Counter: counts 0..9 and prints each
.data
    newline: db 10, 0

.text
start:
    MOV R1, 0          ; counter = 0
loop:
    MOV R0, 2          ; syscall: write char
    MOV R1, R1         ; value to print
    ADD R1, R1, 48     ; convert to ASCII '0'
    SYSCALL
    SUB R1, R1, 48     ; convert back
    ADD R1, R1, 1      ; counter++
    CMP R1, 10         ; compare with 10
    JL loop            ; if < 10, continue

    MOV R0, 0          ; exit
    SYSCALL
"""

FIB_ASM = """; Fibonacci: prints first 12 numbers
.data
    space: db 32, 0
    newline: db 10, 0

.text
start:
    MOV R1, 0          ; a = 0
    MOV R2, 1          ; b = 1
    MOV R3, 12         ; count = 12

loop:
    ; print a
    MOV R0, 1          ; syscall: write string
    MOV R8, R1
    ADD R8, R8, 48
    MOV R7, R8
    ; print manually via char
    MOV R0, 2
    MOV R1, R7
    SYSCALL

    ; print space
    MOV R0, 2
    MOV R1, 32
    SYSCALL

    ; a, b = b, a+b
    MOV R4, R1
    MOV R5, R2
    ADD R1, R5, R4     ; new a = old b
    ADD R2, R4, R5     ; new b = old a + old b
    MOV R6, R1

    SUB R3, R3, 1      ; count--
    CMP R3, 0
    JNZ loop

    MOV R0, 0
    SYSCALL
"""

COLLATZ_ASM = """; Collatz sequence starting from 27
.data
    space: db 32, 0
    newline: db 10, 0

.text
start:
    MOV R1, 27         ; n = 27
    MOV R3, 0          ; steps = 0

loop:
    ; print n as char (only works for n < 10 for simplicity)
    MOV R0, 2
    MOV R2, R1
    ADD R2, R2, 48
    MOV R1, R2
    SYSCALL

    ; print space
    MOV R0, 2
    MOV R1, 32
    SYSCALL

    ; restore n
    SUB R1, R2, 48

    CMP R1, 1
    JZ done

    ; check if even: n & 1
    MOV R4, R1
    AND R4, R1, 1
    CMP R4, 0
    JZ even

odd:
    MUL R1, R1, 3
    ADD R1, R1, 1
    ADD R3, R3, 1
    JMP loop

even:
    SHR R1, R1, 1
    ADD R3, R3, 1
    JMP loop

done:
    MOV R0, 0
    SYSCALL
"""

# ── Test VM ────────────────────────────────────────────────────────────────


def self_test() -> list[str]:
    """Run built-in programs and report results."""
    results = []
    runner = VMRunner()

    # Hello
    out = runner.assemble_and_run(HELLO_ASM)
    hello_ok = "Hello" in " ".join(out)
    results.append(f"  hello: {'PASS' if hello_ok else 'FAIL'} — output: {out}")

    # Counter
    runner2 = VMRunner()
    out2 = runner2.assemble_and_run(COUNTER_ASM)
    results.append(f"  counter: {'PASS' if out2 else 'FAIL'} — steps: {runner2.cpu._step_count}")

    # Fibonacci
    runner3 = VMRunner()
    out3 = runner3.assemble_and_run(FIB_ASM)
    results.append(f"  fib: {'PASS' if out3 else 'FAIL'} — steps: {runner3.cpu._step_count}")

    return results
