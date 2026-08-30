"""
VM Debugger — high-level debugging interface for the x86 VM.

Provides an interactive debugger with:
  - Breakpoint management (set, remove, list, conditional)
  - Step-through execution (stepi, step over, step out)
  - Memory inspection (hex dump, watchpoints)
  - Register inspection and modification
  - Symbol table and source-level debugging
  - Execution trace analysis
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Callable, Any

from .vm_engine import (
    VMEngine, Breakpoint, StepEvent, BreakpointEvent,
    FaultEvent, SyscallEvent, ExecutionTrace,
)

logger = logging.getLogger("slo.vm.debugger")


# ── Symbol Table ──────────────────────────────────────────────────────────────

@dataclass
class Symbol:
    """A named address in the VM."""
    name: str
    address: int
    size: int = 0
    kind: str = "label"  # label, function, data, syscall


class SymbolTable:
    """Maps names to addresses for debugging."""

    def __init__(self):
        self._symbols: dict[str, Symbol] = {}
        self._addr_to_name: dict[int, str] = {}

    def add(self, name: str, address: int, size: int = 0, kind: str = "label"):
        sym = Symbol(name=name, address=address, size=size, kind=kind)
        self._symbols[name] = sym
        self._addr_to_name[address] = name

    def resolve(self, name_or_addr: str | int) -> int | None:
        if isinstance(name_or_addr, int):
            return name_or_addr
        # Try direct lookup
        if name_or_addr in self._symbols:
            return self._symbols[name_or_addr].address
        # Try decimal literal
        try:
            return int(name_or_addr)
        except ValueError:
            pass
        # Try hex literal
        try:
            return int(name_or_addr, 16)
        except ValueError:
            pass
        return None

    def name_for(self, address: int) -> str | None:
        return self._addr_to_name.get(address)

    def all(self) -> list[Symbol]:
        return list(self._symbols.values())


# ── Watchpoint ────────────────────────────────────────────────────────────────

@dataclass
class Watchpoint:
    """Watches a memory address for changes."""
    address: int
    size: int = 4
    label: str = ""
    last_value: int = -1
    enabled: bool = True
    hit_count: int = 0


# ── Debugger ──────────────────────────────────────────────────────────────────

class Debugger:
    """
    Interactive debugger for the x86 VM.

    Wraps VMEngine with higher-level debugging operations:
    - Symbol-aware breakpoints
    - Step over / step out
    - Memory watchpoints
    - Register modification
    - Trace analysis
    """

    def __init__(self, engine: VMEngine | None = None):
        self._engine = engine or VMEngine()
        self._symbols = SymbolTable()
        self._watchpoints: dict[int, Watchpoint] = {}
        self._wp_counter = 0
        self._call_stack_depth = 0
        self._step_over_addr: int | None = None
        self._step_out_addr: int | None = None
        self._output_callback: Callable[[str], None] | None = None

    @property
    def engine(self) -> VMEngine:
        return self._engine

    @property
    def symbols(self) -> SymbolTable:
        return self._symbols

    def set_output(self, callback: Callable[[str], None]):
        """Set callback for debugger output."""
        self._output_callback = callback

    def _out(self, text: str):
        if self._output_callback:
            self._output_callback(text)
        else:
            print(text)

    # ── Symbol Loading ────────────────────────────────────────────────────

    def load_symbols(self, source: str, org: int = 0x1000):
        """Parse assembly source to extract labels as symbols."""
        lines = source.strip().split("\n")
        addr = org
        for line in lines:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            # Detect label (ends with ':')
            label_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:', line)
            if label_match:
                name = label_match.group(1)
                self._symbols.add(name, addr, kind="label")
            # Estimate instruction size (rough: 1-3 bytes per instruction)
            if not line.endswith(":") and not line.startswith(";"):
                addr += 2  # rough estimate

    # ── Breakpoints ───────────────────────────────────────────────────────

    def bp_set(self, target: str | int, label: str = "") -> int:
        """Set breakpoint by address or symbol name."""
        addr = self._symbols.resolve(target)
        if addr is None:
            raise ValueError(f"Cannot resolve: {target}")
        bp_id = self._engine.set_breakpoint(addr, label)
        name = self._symbols.name_for(addr) or f"0x{addr:08x}"
        self._out(f"Breakpoint {bp_id}: {name} {label}")
        return bp_id

    def bp_remove(self, bp_id: int):
        """Remove a breakpoint."""
        self._engine.remove_breakpoint(bp_id)
        self._out(f"Breakpoint {bp_id} removed")

    def bp_list(self) -> list[dict]:
        """List all breakpoints."""
        bps = self._engine.list_breakpoints()
        for bp in bps:
            name = self._symbols.name_for(bp["address"])
            bp["symbol"] = name or ""
        return bps

    def bp_clear(self):
        """Remove all breakpoints."""
        self._engine.clear_breakpoints()
        self._out("All breakpoints removed")

    # ── Watchpoints ───────────────────────────────────────────────────────

    def wp_set(self, target: str | int, size: int = 4, label: str = "") -> int:
        """Set a memory watchpoint."""
        addr = self._symbols.resolve(target)
        if addr is None:
            raise ValueError(f"Cannot resolve: {target}")
        self._wp_counter += 1
        wp = Watchpoint(address=addr, size=size, label=label)
        self._watchpoints[self._wp_counter] = wp
        self._out(f"Watchpoint {self._wp_counter}: [{addr:08x}+{size}] {label}")
        return self._wp_counter

    def wp_remove(self, wp_id: int):
        """Remove a watchpoint."""
        self._watchpoints.pop(wp_id, None)
        self._out(f"Watchpoint {wp_id} removed")

    def wp_list(self) -> list[dict]:
        """List all watchpoints."""
        return [
            {"id": wp_id, "address": wp.address, "size": wp.size,
             "label": wp.label, "enabled": wp.enabled, "hit_count": wp.hit_count}
            for wp_id, wp in self._watchpoints.items()
        ]

    def _check_watchpoints(self):
        """Check if any watchpoint value has changed."""
        for wp_id, wp in self._watchpoints.items():
            if not wp.enabled:
                continue
            current = self._engine.read_dword(wp.address)
            if wp.last_value == -1:
                wp.last_value = current
            elif current != wp.last_value:
                wp.hit_count += 1
                name = self._symbols.name_for(wp.address) or f"0x{wp.address:08x}"
                self._out(f"Watchpoint {wp_id} hit: {name} "
                          f"0x{wp.last_value:08x} -> 0x{current:08x}")
                wp.last_value = current

    # ── Execution ─────────────────────────────────────────────────────────

    def stepi(self, count: int = 1) -> bool:
        """Execute single instruction(s). Returns False on fault/halt."""
        for _ in range(count):
            if not self._engine.step():
                return False
            self._check_watchpoints()
        return True

    def step_over(self) -> bool:
        """Step over a CALL instruction (run until return address)."""
        eip = self._engine.get_reg("eip")
        opcode = self._engine.read_byte(eip)
        # CALL opcode: 0xE8 (near), 0x9A (far), 0xFF /2
        if opcode == 0xE8:
            # Near CALL: read offset, compute return address
            offset = self._engine.read_dword(eip + 1)
            call_target = eip + 5 + offset
            return_addr = eip + 5
            self._step_over_addr = return_addr
            self._out(f"Step over: call 0x{call_target:08x}, "
                      f"return at 0x{return_addr:08x}")
            return self.run_until(return_addr)
        elif opcode == 0xFF:
            # Indirect CALL — just step into
            return self.stepi()
        else:
            # Not a CALL, just step
            return self.stepi()

    def step_out(self) -> bool:
        """Run until current function returns."""
        esp = self._engine.get_reg("esp")
        # Read return address from stack
        ret_addr = self._engine.read_dword(esp)
        if ret_addr == 0:
            self._out("No return address on stack")
            return False
        self._step_out_addr = ret_addr
        self._out(f"Step out: running until return to 0x{ret_addr:08x}")
        return self.run_until(ret_addr)

    def run_until(self, target_addr: int, max_steps: int = 100000) -> bool:
        """Run until EIP equals target_addr."""
        steps = 0
        while steps < max_steps:
            eip = self._engine.get_reg("eip")
            if eip == target_addr:
                return True
            if not self._engine.step():
                return False
            self._check_watchpoints()
            steps += 1
        self._out(f"Timeout: did not reach 0x{target_addr:08x} in {max_steps} steps")
        return False

    def continue_exec(self, max_steps: int = 1000000) -> ExecutionTrace:
        """Continue execution until breakpoint, fault, or halt."""
        self._engine._break_requested = False
        return self._engine.run(max_steps=max_steps)

    # ── Inspection ────────────────────────────────────────────────────────

    def dump_regs(self) -> dict[str, int]:
        """Print all registers."""
        regs = self._engine.registers()
        for name, value in regs.items():
            sym = self._symbols.name_for(value)
            sym_str = f" <{sym}>" if sym else ""
            self._out(f"  {name:4s} = 0x{value:08x}{sym_str}")
        return regs

    def dump_flags(self):
        """Print all flags."""
        flags = self._engine.flags()
        for name, value in flags.items():
            self._out(f"  {name:2s} = {int(value)}")

    def dump_memory(self, target: str | int, length: int = 64):
        """Hex dump memory at address or symbol."""
        addr = self._symbols.resolve(target)
        if addr is None:
            raise ValueError(f"Cannot resolve: {target}")
        hex_dump = self._engine.dump_memory(addr, length)
        name = self._symbols.name_for(addr)
        header = f"  {name} ({addr:08x}):" if name else f"  {addr:08x}:"
        self._out(header)
        self._out(hex_dump)

    def dump_stack(self, depth: int = 8):
        """Dump stack contents."""
        esp = self._engine.get_reg("esp")
        self._out(f"  Stack (ESP=0x{esp:08x}):")
        for i in range(depth):
            addr = esp + i * 4
            value = self._engine.read_dword(addr)
            marker = " <- ESP" if i == 0 else ""
            self._out(f"    [0x{addr:08x}] 0x{value:08x}{marker}")

    def disassemble(self, count: int = 10):
        """Disassemble from current EIP."""
        eip = self._engine.get_reg("eip")
        self._out(f"  Disassembly from 0x{eip:08x}:")
        # Simple hex dump of instructions (full disassembly needs x86 decoder)
        for i in range(count):
            addr = eip + i
            byte = self._engine.read_byte(addr)
            name = self._symbols.name_for(addr)
            prefix = f"  {name}: " if name else f"  0x{addr:08x}: "
            self._out(f"{prefix}{byte:02x}")

    # ── Analysis ──────────────────────────────────────────────────────────

    def analyze_trace(self, trace: ExecutionTrace) -> dict:
        """Analyze an execution trace."""
        analysis = {
            "total_instructions": trace.total_instructions,
            "total_time_ms": trace.total_time_ms,
            "exit_reason": trace.exit_reason,
            "breakpoints_hit": len(trace.breakpoints_hit),
            "faults": len(trace.faults),
            "syscalls": len(trace.syscalls),
        }

        # Most executed addresses
        if trace.steps:
            addr_counts: dict[int, int] = {}
            for step in trace.steps:
                addr_counts[step.eip] = addr_counts.get(step.eip, 0) + 1
            top_addrs = sorted(addr_counts.items(), key=lambda x: -x[1])[:10]
            analysis["hot_addresses"] = [
                {"address": addr, "count": count,
                 "symbol": self._symbols.name_for(addr) or ""}
                for addr, count in top_addrs
            ]

        # Syscall summary
        if trace.syscalls:
            syscall_counts: dict[int, int] = {}
            for sc in trace.syscalls:
                syscall_counts[sc.number] = syscall_counts.get(sc.number, 0) + 1
            analysis["syscall_summary"] = [
                {"number": num, "count": count}
                for num, count in sorted(syscall_counts.items())
            ]

        return analysis

    def list_symbols(self) -> list[dict]:
        """List all symbols."""
        return [
            {"name": sym.name, "address": sym.address,
             "size": sym.size, "kind": sym.kind}
            for sym in self._symbols.all()
        ]
