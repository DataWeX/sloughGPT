"""
Comprehensive tests for domains.shell.vm_engine.

Pure-logic tests for Breakpoint, StepEvent, BreakpointEvent, FaultEvent,
SyscallEvent, ExecutionTrace, DeviceBus, ConsoleDevice, and VMEngine.
No mocks — uses real X86CPU and X86Assembler.
"""

from __future__ import annotations

import struct
import pytest

from domains.shell.vm import (
    X86CPU,
    X86Assembler,
    InsFault,
    Halt,
    MemFault,
    ProcessTable,
    ProcessControlBlock,
    ProcessState,
    Scheduler,
)
from domains.shell.vm_engine import (
    Breakpoint,
    StepEvent,
    BreakpointEvent,
    FaultEvent,
    SyscallEvent,
    ExecutionTrace,
    DeviceBus,
    ConsoleDevice,
    VMEngine,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

NOP = bytes([0x90])
HLT = bytes([0xF4])


def _make_engine(memory_size=0x10000) -> VMEngine:
    return VMEngine(memory_size=memory_size)


def _load_nop_hlt(engine: VMEngine, org=0x1000):
    engine.load_bytes(NOP + HLT, org=org)


# =============================================================================
# Breakpoint
# =============================================================================

class TestBreakpoint:
    def test_creation(self):
        bp = Breakpoint(address=0x1000)
        assert bp.address == 0x1000
        assert bp.enabled is True
        assert bp.hit_count == 0
        assert bp.condition is None

    def test_should_trigger_enabled(self):
        bp = Breakpoint(address=0x1000, enabled=True)
        assert bp.should_trigger() is True

    def test_should_trigger_disabled(self):
        bp = Breakpoint(address=0x1000, enabled=False)
        assert bp.should_trigger() is False

    def test_should_trigger_with_condition_true(self):
        bp = Breakpoint(address=0x1000, condition=lambda: True)
        assert bp.should_trigger() is True

    def test_should_trigger_with_condition_false(self):
        bp = Breakpoint(address=0x1000, condition=lambda: False)
        assert bp.should_trigger() is False

    def test_should_trigger_condition_exception(self):
        bp = Breakpoint(address=0x1000, condition=lambda: 1 / 0)
        assert bp.should_trigger() is False

    def test_label(self):
        bp = Breakpoint(address=0x1000, label="main_loop")
        assert bp.label == "main_loop"


# =============================================================================
# StepEvent
# =============================================================================

class TestStepEvent:
    def test_creation(self):
        e = StepEvent(
            eip=0x1000,
            opcode=0x90,
            registers={"eax": 0},
            flags={"zf": False},
            instruction_bytes=b"\x90",
        )
        assert e.eip == 0x1000
        assert e.opcode == 0x90
        assert e.registers["eax"] == 0


# =============================================================================
# BreakpointEvent
# =============================================================================

class TestBreakpointEvent:
    def test_creation(self):
        bp = Breakpoint(address=0x1000)
        e = BreakpointEvent(breakpoint=bp, eip=0x1000, registers={"eax": 1})
        assert e.breakpoint is bp
        assert e.eip == 0x1000


# =============================================================================
# FaultEvent
# =============================================================================

class TestFaultEvent:
    def test_creation(self):
        e = FaultEvent(
            fault_type=InsFault,
            message="bad instruction",
            eip=0x1000,
            registers={"eax": 0},
        )
        assert e.fault_type is InsFault
        assert "bad instruction" in e.message


# =============================================================================
# SyscallEvent
# =============================================================================

class TestSyscallEvent:
    def test_creation(self):
        e = SyscallEvent(number=1, args={"eax": 4}, eip=0x1000)
        assert e.number == 1
        assert e.args["eax"] == 4


# =============================================================================
# ExecutionTrace
# =============================================================================

class TestExecutionTrace:
    def test_default(self):
        t = ExecutionTrace()
        assert t.steps == []
        assert t.breakpoints_hit == []
        assert t.faults == []
        assert t.syscalls == []
        assert t.total_instructions == 0
        assert t.total_time_ms == 0.0
        assert t.exit_reason == ""


# =============================================================================
# DeviceBus
# =============================================================================

class TestDeviceBus:
    def test_outb_log(self):
        bus = DeviceBus()
        bus.outb(0x3F8, 0x41)
        assert len(bus.log) == 1
        assert bus.log[0] == (0x3F8, 0x41, 0)

    def test_outw_log(self):
        bus = DeviceBus()
        bus.outw(0x3F8, 0x1234)
        assert bus.log[0] == (0x3F8, 0x1234, 1)

    def test_outd_log(self):
        bus = DeviceBus()
        bus.outd(0x3F8, 0x12345678)
        assert bus.log[0] == (0x3F8, 0x12345678, 2)

    def test_outb_calls_handler(self):
        bus = DeviceBus()
        received = []
        bus.register_out(0x3F8, lambda p, v, w: received.append((p, v, w)))
        bus.outb(0x3F8, 0x41)
        assert received == [(0x3F8, 0x41, 8)]

    def test_outw_calls_handler_with_16(self):
        bus = DeviceBus()
        received = []
        bus.register_out(0x3F8, lambda p, v, w: received.append(w))
        bus.outw(0x3F8, 1)
        assert received == [16]

    def test_outd_calls_handler_with_32(self):
        bus = DeviceBus()
        received = []
        bus.register_out(0x3F8, lambda p, v, w: received.append(w))
        bus.outd(0x3F8, 1)
        assert received == [32]

    def test_inb_no_reader(self):
        bus = DeviceBus()
        assert bus.inb(0x3F8) == 0

    def test_inb_with_reader(self):
        bus = DeviceBus()
        bus.register_in(0x3F8, lambda p: 0xAB)
        assert bus.inb(0x3F8) == 0xAB

    def test_inb_mask_8bit(self):
        bus = DeviceBus()
        bus.register_in(0x3F8, lambda p: 0x1FF)
        assert bus.inb(0x3F8) == 0xFF

    def test_inw_no_reader(self):
        bus = DeviceBus()
        assert bus.inw(0x3F8) == 0

    def test_inw_with_reader(self):
        bus = DeviceBus()
        bus.register_in(0x3F8, lambda p: 0x12345)
        assert bus.inw(0x3F8) == 0x2345

    def test_ind_no_reader(self):
        bus = DeviceBus()
        assert bus.ind(0x3F8) == 0

    def test_ind_with_reader(self):
        bus = DeviceBus()
        bus.register_in(0x3F8, lambda p: 0x1FFFFFFFF)
        assert bus.ind(0x3F8) == 0xFFFFFFFF

    def test_log_returns_copy(self):
        bus = DeviceBus()
        bus.outb(0x3F8, 1)
        log = bus.log
        bus.outb(0x3F8, 2)
        assert len(log) == 1
        assert len(bus.log) == 2


# =============================================================================
# ConsoleDevice
# =============================================================================

class TestConsoleDevice:
    def test_write_byte(self):
        dev = ConsoleDevice()
        dev.write_byte(0x3F8, 0x41, 8)
        assert dev.output == [0x41]

    def test_write_byte_masks_to_8bit(self):
        dev = ConsoleDevice()
        dev.write_byte(0x3F8, 0x141, 8)
        assert dev.output == [0x41]

    def test_read_byte_empty(self):
        dev = ConsoleDevice()
        assert dev.read_byte(0x3F8) == 0

    def test_read_byte_from_buffer(self):
        dev = ConsoleDevice()
        dev.input_buffer = [0x41, 0x42]
        assert dev.read_byte(0x3F8) == 0x41
        assert dev.read_byte(0x3F8) == 0x42
        assert dev.read_byte(0x3F8) == 0

    def test_feed_input(self):
        dev = ConsoleDevice()
        dev.feed_input(b"hello")
        assert list(dev.input_buffer) == list(b"hello")

    def test_on_output_callback(self):
        dev = ConsoleDevice()
        received = []
        dev.on_output = lambda v: received.append(v)
        dev.write_byte(0x3F8, 0x42, 8)
        assert received == [0x42]


# =============================================================================
# VMEngine — properties
# =============================================================================

class TestVMEngineProperties:
    def test_cpu_property(self):
        e = _make_engine()
        assert isinstance(e.cpu, X86CPU)

    def test_assembler_property(self):
        e = _make_engine()
        assert isinstance(e.assembler, X86Assembler)

    def test_devices_property(self):
        e = _make_engine()
        assert isinstance(e.devices, DeviceBus)

    def test_console_property(self):
        e = _make_engine()
        assert isinstance(e.console, ConsoleDevice)

    def test_process_table_property(self):
        e = _make_engine()
        assert isinstance(e.process_table, ProcessTable)

    def test_trace_property(self):
        e = _make_engine()
        assert isinstance(e.trace, ExecutionTrace)

    def test_is_running_initial(self):
        e = _make_engine()
        assert e.is_running is False

    def test_is_halted_initial(self):
        e = _make_engine()
        assert e.is_halted is False


# =============================================================================
# VMEngine — program loading
# =============================================================================

class TestVMEngineLoading:
    def test_load_bytes(self):
        e = _make_engine()
        org = e.load_bytes(b"\x90\x90", org=0x1000)
        assert org == 0x1000
        assert e.cpu.eip == 0x1000

    def test_load_source(self):
        e = _make_engine()
        org = e.load_source("nop", org=0x1000)
        assert org == 0x1000

    def test_load_source_hlt(self):
        e = _make_engine()
        e.load_source("hlt", org=0x1000)
        result = e.step()
        assert result is False
        assert e.is_halted is True

    def test_set_entry(self):
        e = _make_engine()
        e.set_entry(0x2000)
        assert e.cpu.eip == 0x2000


# =============================================================================
# VMEngine — register access
# =============================================================================

class TestVMEngineRegisters:
    def test_registers_returns_all(self):
        e = _make_engine()
        regs = e.registers()
        expected = ["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi",
                     "eip", "ax", "cx", "dx", "bx", "al", "cl", "dl", "bl",
                     "ah", "ch", "dh", "bh"]
        for r in expected:
            assert r in regs

    def test_get_reg_32bit(self):
        e = _make_engine()
        e.cpu.eax = 0xDEADBEEF
        assert e.get_reg("eax") == 0xDEADBEEF

    def test_get_reg_16bit(self):
        e = _make_engine()
        e.cpu.eax = 0x12345678
        assert e.get_reg("ax") == 0x5678

    def test_get_reg_8bit_low(self):
        e = _make_engine()
        e.cpu.eax = 0x12345678
        assert e.get_reg("al") == 0x78

    def test_get_reg_8bit_high(self):
        e = _make_engine()
        e.cpu.eax = 0x12345678
        assert e.get_reg("ah") == 0x56

    def test_get_reg_unknown(self):
        e = _make_engine()
        with pytest.raises(ValueError, match="Unknown register"):
            e.get_reg("xyz")

    def test_set_reg_32bit(self):
        e = _make_engine()
        e.set_reg("eax", 0xCAFEBABE)
        assert e.cpu.eax == 0xCAFEBABE

    def test_set_reg_masks_to_32bit(self):
        e = _make_engine()
        e.set_reg("eax", 0x1FFFFFFFF)
        assert e.cpu.eax == 0xFFFFFFFF

    def test_set_reg_16bit(self):
        e = _make_engine()
        e.set_reg("ax", 0x1234)
        assert e.cpu._get16(0) == 0x1234

    def test_set_reg_8bit_low(self):
        e = _make_engine()
        e.set_reg("al", 0xAB)
        assert e.cpu._get8l(0) == 0xAB

    def test_set_reg_8bit_high(self):
        e = _make_engine()
        e.set_reg("ah", 0xCD)
        assert e.cpu._get8h(0) == 0xCD

    def test_set_reg_unknown(self):
        e = _make_engine()
        with pytest.raises(ValueError, match="Unknown register"):
            e.set_reg("xyz", 0)

    def test_flags(self):
        e = _make_engine()
        f = e.flags()
        assert "cf" in f
        assert "zf" in f
        assert "sf" in f
        assert "of" in f


# =============================================================================
# VMEngine — memory access
# =============================================================================

class TestVMEngineMemory:
    def test_read_write_byte(self):
        e = _make_engine()
        e.write_byte(0x1000, 0x42)
        assert e.read_byte(0x1000) == 0x42

    def test_read_byte_out_of_bounds(self):
        e = _make_engine()
        assert e.read_byte(0xFFFFFFFF) == 0

    def test_write_byte_out_of_bounds(self):
        e = _make_engine()
        e.write_byte(0xFFFFFFFF, 0x42)
        assert e.read_byte(0xFFFFFFFF) == 0

    def test_read_write_dword(self):
        e = _make_engine()
        e.write_dword(0x1000, 0xDEADBEEF)
        assert e.read_dword(0x1000) == 0xDEADBEEF

    def test_read_write_word(self):
        e = _make_engine()
        e.write_word(0x1000, 0x1234)
        assert e.read_word(0x1000) == 0x1234

    def test_read_write_memory_bytes(self):
        e = _make_engine()
        e.write_memory(0x1000, b"\x01\x02\x03")
        assert e.read_memory(0x1000, 3) == b"\x01\x02\x03"

    def test_read_memory_out_of_bounds(self):
        e = _make_engine()
        result = e.read_memory(0xFFFFFFFF, 4)
        assert result == b"\x00\x00\x00\x00"

    def test_dump_memory(self):
        e = _make_engine()
        e.write_memory(0x1000, b"Hello")
        dump = e.dump_memory(0x1000, 5)
        assert "Hello" in dump
        assert "00001000" in dump


# =============================================================================
# VMEngine — breakpoints
# =============================================================================

class TestVMEngineBreakpoints:
    def test_set_breakpoint(self):
        e = _make_engine()
        bp_id = e.set_breakpoint(0x1000, label="test")
        assert bp_id == 1
        assert len(e.list_breakpoints()) == 1

    def test_set_breakpoint_increments_id(self):
        e = _make_engine()
        id1 = e.set_breakpoint(0x1000)
        id2 = e.set_breakpoint(0x2000)
        assert id2 == id1 + 1

    def test_remove_breakpoint(self):
        e = _make_engine()
        bp_id = e.set_breakpoint(0x1000)
        e.remove_breakpoint(bp_id)
        assert len(e.list_breakpoints()) == 0

    def test_remove_nonexistent(self):
        e = _make_engine()
        e.remove_breakpoint(999)

    def test_enable_disable(self):
        e = _make_engine()
        bp_id = e.set_breakpoint(0x1000)
        e.disable_breakpoint(bp_id)
        bps = e.list_breakpoints()
        assert bps[0]["enabled"] is False
        e.enable_breakpoint(bp_id)
        bps = e.list_breakpoints()
        assert bps[0]["enabled"] is True

    def test_clear_breakpoints(self):
        e = _make_engine()
        e.set_breakpoint(0x1000)
        e.set_breakpoint(0x2000)
        e.clear_breakpoints()
        assert len(e.list_breakpoints()) == 0

    def test_list_breakpoints(self):
        e = _make_engine()
        e.set_breakpoint(0x1000, label="main")
        bps = e.list_breakpoints()
        assert len(bps) == 1
        assert bps[0]["address"] == 0x1000
        assert bps[0]["label"] == "main"
        assert bps[0]["enabled"] is True
        assert bps[0]["hit_count"] == 0

    def test_set_breakpoint_once(self):
        e = _make_engine()
        bp_id = e.set_breakpoint_once(0x1000)
        bp = e._breakpoints[bp_id]
        assert bp.should_trigger() is True
        assert bp.enabled is False


# =============================================================================
# VMEngine — event hooks
# =============================================================================

class TestVMEngineHooks:
    def test_on_step(self):
        e = _make_engine()
        events = []
        e.on_step(lambda ev: events.append(ev))
        e.load_bytes(NOP + HLT, org=0x1000)
        e.step()
        assert len(events) == 1
        assert isinstance(events[0], StepEvent)

    def test_on_halt(self):
        e = _make_engine()
        halted_at = []
        e.on_halt(lambda addr: halted_at.append(addr))
        e.load_bytes(HLT, org=0x1000)
        e.step()
        assert halted_at == [0x1000]

    def test_on_fault(self):
        e = _make_engine()
        faults = []
        e.on_fault(lambda ev: faults.append(ev))
        e.cpu._mem[0x1000] = 0x06
        e.cpu.eip = 0x1000
        e.step()
        assert len(faults) == 1
        assert isinstance(faults[0], FaultEvent)


# =============================================================================
# VMEngine — execution
# =============================================================================

class TestVMEngineExecution:
    def test_step_nop(self):
        e = _make_engine()
        e.load_bytes(NOP + HLT, org=0x1000)
        result = e.step()
        assert result is True
        assert e.cpu.eip == 0x1001

    def test_step_hlt(self):
        e = _make_engine()
        e.load_bytes(HLT, org=0x1000)
        result = e.step()
        assert result is False
        assert e.is_halted is True

    def test_run_halt(self):
        e = _make_engine()
        e.load_bytes(NOP + NOP + HLT, org=0x1000)
        trace = e.run()
        assert trace.exit_reason == "halt"

    def test_run_max_steps(self):
        e = _make_engine()
        e.load_bytes(NOP * 100, org=0x1000)
        trace = e.run(max_steps=5)
        assert trace.exit_reason == "max_steps"

    def test_run_with_tracing(self):
        e = _make_engine()
        e.enable_tracing()
        e.load_bytes(NOP + HLT, org=0x1000)
        trace = e.run()
        assert len(trace.steps) >= 1

    def test_run_with_breakpoint(self):
        e = _make_engine()
        e.load_bytes(NOP + NOP + NOP + HLT, org=0x1000)
        e.set_breakpoint(0x1001, label="bp1")
        trace = e.run()
        assert trace.exit_reason == "breakpoint"

    def test_run_with_breakpoint_callback(self):
        e = _make_engine()
        e.load_bytes(NOP + NOP + HLT, org=0x1000)
        bp_hits = []
        e.on_breakpoint(lambda ev: bp_hits.append(ev))
        e.set_breakpoint(0x1001)
        e.run()
        assert len(bp_hits) == 1

    def test_step_breakpoint(self):
        e = _make_engine()
        e.load_bytes(NOP + NOP + HLT, org=0x1000)
        e.set_breakpoint(0x1000)
        result = e.step()
        assert result is False

    def test_step_over_non_call(self):
        e = _make_engine()
        e.load_bytes(NOP + HLT, org=0x1000)
        result = e.step_over()
        assert result is True
        assert e.cpu.eip == 0x1001

    def test_continue_execution(self):
        e = _make_engine()
        e.load_bytes(NOP + NOP + HLT, org=0x1000)
        e.set_breakpoint(0x1000)
        e.step()
        trace = e.continue_execution()
        assert trace.exit_reason == "halt"

    def test_continue_when_halted(self):
        e = _make_engine()
        e.load_bytes(HLT, org=0x1000)
        e.step()
        trace = e.continue_execution()
        assert e.is_halted is True

    def test_request_break(self):
        e = _make_engine()
        e.load_bytes(NOP * 1000, org=0x1000)
        def maybe_break(ev):
            if ev.eip == 0x1005:
                e.request_break()
        e.on_step(maybe_break)
        trace = e.run()
        assert trace.exit_reason == "break_request"


# =============================================================================
# VMEngine — tracing
# =============================================================================

class TestVMEngineTracing:
    def test_enable_disable_tracing(self):
        e = _make_engine()
        e.enable_tracing()
        assert e._tracing is True
        e.disable_tracing()
        assert e._tracing is False

    def test_trace_summary(self):
        e = _make_engine()
        e.enable_tracing()
        e.load_bytes(NOP + HLT, org=0x1000)
        e.run()
        summary = e.get_trace_summary()
        assert summary["total_instructions"] >= 1
        assert summary["exit_reason"] == "halt"
        assert summary["total_time_ms"] >= 0


# =============================================================================
# VMEngine — reset
# =============================================================================

class TestVMEngineReset:
    def test_reset_clears_state(self):
        e = _make_engine()
        e.load_bytes(NOP + HLT, org=0x1000)
        e.step()
        e.set_breakpoint(0x1000)
        e.enable_tracing()
        e.reset()
        assert e.is_halted is False
        assert e.is_running is False
        assert e.list_breakpoints() == []
        assert e._tracing is False
        assert e.console.output == []


# =============================================================================
# VMEngine — state snapshot
# =============================================================================

class TestVMEngineState:
    def test_state_snapshot(self):
        e = _make_engine()
        snap = e.state_snapshot()
        assert "registers" in snap
        assert "flags" in snap
        assert "breakpoints" in snap
        assert "stack" in snap
        assert "trace_summary" in snap

    def test_state_snapshot_with_trace(self):
        e = _make_engine()
        e.enable_tracing()
        e.load_bytes(NOP + HLT, org=0x1000)
        e.run()
        snap = e.state_snapshot()
        assert snap["trace_summary"] is not None


# =============================================================================
# VMEngine — disassembly
# =============================================================================

class TestVMEngineDisassembly:
    def test_disassemble_nop(self):
        e = _make_engine()
        e.load_bytes(NOP, org=0x1000)
        lines = e.disassemble(0x1000, count=1)
        assert len(lines) == 1
        assert "NOP" in lines[0]

    def test_disassemble_hlt(self):
        e = _make_engine()
        e.load_bytes(HLT, org=0x1000)
        lines = e.disassemble(0x1000, count=1)
        assert "HLT" in lines[0]

    def test_disassemble_out_of_bounds(self):
        e = _make_engine()
        lines = e.disassemble(0xFFFFFFFF, count=5)
        assert lines == []


# =============================================================================
# VMEngine — instruction length
# =============================================================================

class TestVMEngineInstructionLength:
    @pytest.mark.parametrize("opcode,expected", [
        (0x90, 1),  # NOP
        (0xF4, 1),  # HLT
        (0xC3, 1),  # RET
        (0x50, 1),  # PUSH EAX
        (0x58, 1),  # POP EAX
        (0x66, 2),  # prefix
        (0xE8, 5),  # CALL rel32
        (0xEB, 2),  # JMP short
        (0x74, 2),  # JZ short
        (0xB0, 2),  # MOV AL, imm8
        (0x68, 5),  # PUSH imm32
        (0x6A, 2),  # PUSH imm8
        (0xCD, 2),  # INT imm8
    ])
    def test_instruction_lengths(self, opcode, expected):
        e = _make_engine()
        assert e._instruction_length(opcode, 0) == expected


# =============================================================================
# VMEngine — opcode name
# =============================================================================

class TestVMEngineOpcodeName:
    def test_known_opcodes(self):
        e = _make_engine()
        assert e._opcode_name(0x90) == "NOP"
        assert e._opcode_name(0xF4) == "HLT"
        assert e._opcode_name(0xC3) == "RET"
        assert e._opcode_name(0x50) == "PUSH EAX"
        assert e._opcode_name(0x58) == "POP EAX"

    def test_unknown_opcode(self):
        e = _make_engine()
        name = e._opcode_name(0xFF)
        assert "FF" in name


# =============================================================================
# VMEngine — repr
# =============================================================================

class TestVMEngineRepr:
    def test_repr(self):
        e = _make_engine()
        r = repr(e)
        assert "VMEngine" in r
        assert "eip=" in r
        assert "esp=" in r


# =============================================================================
# VMEngine — process management
# =============================================================================

class TestVMEngineProcessManagement:
    def test_create_process(self):
        e = _make_engine()
        pcb = e.create_process("test", priority=5)
        assert pcb.name == "test"
        assert pcb.priority == 5
        assert pcb.state == ProcessState.CREATED

    def test_switch_to_process(self):
        e = _make_engine()
        pcb = e.create_process("p1")
        e.set_reg("eax", 0x1234)
        e.switch_to_process(pcb.pid)
        assert pcb.state == ProcessState.RUNNING


# =============================================================================
# VMEngine — console I/O bus integration
# =============================================================================

class TestVMEngineConsoleBus:
    def test_console_registered_on_port(self):
        e = _make_engine()
        e.devices.outb(0x3F8, 0x41)
        assert e.console.output == [0x41]

    def test_console_input_on_port(self):
        e = _make_engine()
        e.console.feed_input(b"\x42")
        val = e.devices.inb(0x3F8)
        assert val == 0x42
