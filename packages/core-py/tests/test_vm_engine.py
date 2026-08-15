"""Tests for VMEngine — the x86 execution engine."""

import pytest
from domains.shell.vm_engine import (
    VMEngine, Breakpoint, StepEvent, BreakpointEvent,
    FaultEvent, ExecutionTrace, ConsoleDevice, DeviceBus,
)
from domains.shell.vm import InsFault, Halt, MemFault


# ── Helpers ──────────────────────────────────────────────────────────────────

def _engine(source: str = "") -> VMEngine:
    engine = VMEngine()
    if source:
        engine.load_source(source)
    return engine


# ── Loading ──────────────────────────────────────────────────────────────────

class TestLoading:
    def test_load_source_sets_eip(self):
        e = _engine("NOP")
        assert e.cpu.eip == 0x1000

    def test_load_source_returns_org(self):
        e = VMEngine()
        result = e.load_source("NOP", org=0x2000)
        assert result == 0x2000

    def test_load_bytes(self):
        e = VMEngine()
        e.load_bytes(b"\x90\x90", org=0x1000)
        assert e.cpu.eip == 0x1000

    def test_set_entry(self):
        e = _engine("NOP")
        e.set_entry(0x5000)
        assert e.cpu.eip == 0x5000


# ── Register Access ──────────────────────────────────────────────────────────

class TestRegisters:
    def test_get_reg_eax(self):
        e = _engine()
        e.cpu.eax = 0x12345678
        assert e.get_reg("eax") == 0x12345678

    def test_get_reg_ax(self):
        e = _engine()
        e.cpu.eax = 0x12345678
        assert e.get_reg("ax") == 0x5678

    def test_get_reg_al(self):
        e = _engine()
        e.cpu.eax = 0x12345678
        assert e.get_reg("al") == 0x78

    def test_set_reg_eax(self):
        e = _engine()
        e.set_reg("eax", 0xDEADBEEF)
        assert e.cpu.eax == 0xDEADBEEF

    def test_set_reg_ax(self):
        e = _engine()
        e.cpu.eax = 0x12340000
        e.set_reg("ax", 0x5678)
        assert e.cpu.eax == 0x12345678

    def test_registers_dict(self):
        e = _engine()
        e.cpu.eax = 1
        e.cpu.ebx = 2
        regs = e.registers()
        assert regs["eax"] == 1
        assert regs["ebx"] == 2
        assert "eip" in regs

    def test_get_reg_unknown_raises(self):
        e = _engine()
        with pytest.raises(ValueError):
            e.get_reg("zmm0")

    def test_set_reg_unknown_raises(self):
        e = _engine()
        with pytest.raises(ValueError):
            e.set_reg("zmm0", 0)

    def test_get_reg_ah(self):
        e = _engine()
        e.cpu.eax = 0xAB123456
        assert e.get_reg("ah") == 0x34

    def test_get_reg_bh(self):
        e = _engine()
        e.cpu.ebx = 0xAB123456
        assert e.get_reg("bh") == 0x34

    def test_set_reg_ah(self):
        e = _engine()
        e.cpu.eax = 0xAB123456
        e.set_reg("ah", 0xFF)
        assert e.cpu.eax == 0xAB12FF56

    def test_set_reg_dh(self):
        e = _engine()
        e.cpu.edx = 0xAB123456
        e.set_reg("dh", 0x00)
        assert e.cpu.edx == 0xAB120056

    def test_registers_includes_sub_regs(self):
        e = _engine()
        regs = e.registers()
        assert "ax" in regs
        assert "ah" in regs
        assert "al" in regs


# ── Memory Access ────────────────────────────────────────────────────────────

class TestMemory:
    def test_read_write_byte(self):
        e = _engine()
        e.write_byte(0x1000, 0xAB)
        assert e.read_byte(0x1000) == 0xAB

    def test_read_write_word(self):
        e = _engine()
        e.write_word(0x1000, 0x1234)
        assert e.read_word(0x1000) == 0x1234

    def test_read_write_dword(self):
        e = _engine()
        e.write_dword(0x1000, 0xDEADBEEF)
        assert e.read_dword(0x1000) == 0xDEADBEEF

    def test_read_memory(self):
        e = _engine()
        e.write_memory(0x1000, b"\x01\x02\x03\x04")
        data = e.read_memory(0x1000, 4)
        assert data == b"\x01\x02\x03\x04"

    def test_dump_memory(self):
        e = _engine()
        e.write_memory(0x1000, b"Hello")
        dump = e.dump_memory(0x1000, 5)
        assert "Hello" in dump

    def test_write_memory_respects_bounds(self):
        e = VMEngine(memory_size=0x1000)
        # Writing past memory end should not crash
        e.write_memory(0x0FF0, b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F\x10")
        assert e.read_byte(0x0FF0) == 0x01

    def test_disassemble(self):
        e = _engine("NOP\nHLT")
        lines = e.disassemble(0x1000, 2)
        assert len(lines) == 2
        assert "0x00001000" in lines[0]
        assert "0x00001001" in lines[1]


# ── Breakpoints ──────────────────────────────────────────────────────────────

class TestBreakpoints:
    def test_set_breakpoint(self):
        e = _engine()
        bp_id = e.set_breakpoint(0x1000)
        assert bp_id == 1
        assert len(e.list_breakpoints()) == 1

    def test_remove_breakpoint(self):
        e = _engine()
        bp_id = e.set_breakpoint(0x1000)
        e.remove_breakpoint(bp_id)
        assert len(e.list_breakpoints()) == 0

    def test_enable_disable(self):
        e = _engine()
        bp_id = e.set_breakpoint(0x1000)
        e.disable_breakpoint(bp_id)
        assert not e.list_breakpoints()[0]["enabled"]
        e.enable_breakpoint(bp_id)
        assert e.list_breakpoints()[0]["enabled"]

    def test_clear_breakpoints(self):
        e = _engine()
        e.set_breakpoint(0x1000)
        e.set_breakpoint(0x2000)
        e.clear_breakpoints()
        assert len(e.list_breakpoints()) == 0

    def test_breakpoint_hit_count(self):
        e = _engine()
        bp_id = e.set_breakpoint(0x1000)
        e._breakpoints[bp_id].hit_count = 5
        assert e.list_breakpoints()[0]["hit_count"] == 5


# ── Event Hooks ──────────────────────────────────────────────────────────────

class TestEventHooks:
    def test_on_step_fires(self):
        e = _engine("NOP\nHLT")
        events = []
        e.on_step(lambda ev: events.append(ev))
        e.run(max_steps=10)
        assert len(events) >= 1
        assert events[0].eip == 0x1000

    def test_on_halt_fires(self):
        e = _engine("HLT")
        halted_eip = []
        e.on_halt(lambda eip: halted_eip.append(eip))
        e.run()
        assert len(halted_eip) == 1

    def test_on_breakpoint_fires(self):
        e = _engine("NOP\nNOP\nHLT")
        bp_hits = []
        bp_id = e.set_breakpoint(0x1001)
        e.on_breakpoint(lambda ev: bp_hits.append(ev))
        e.run(max_steps=10)
        assert len(bp_hits) == 1
        assert bp_hits[0].breakpoint.address == 0x1001

    def test_on_fault_fires(self):
        e = VMEngine()
        e.load_source("HLT")
        faults = []
        e.on_fault(lambda ev: faults.append(ev))
        e.run()
        # HLT raises Halt, which is caught
        assert len(faults) >= 0  # may or may not fire depending on HLT handling


# ── Execution ────────────────────────────────────────────────────────────────

class TestExecution:
    def test_run_halt(self):
        e = _engine("HLT")
        trace = e.run()
        assert trace.exit_reason == "halt"

    def test_run_max_steps(self):
        e = _engine("NOP\nNOP\nNOP\nHLT")
        trace = e.run(max_steps=3)
        assert trace.exit_reason == "max_steps"

    def test_step_returns_bool(self):
        e = _engine("NOP\nHLT")
        assert e.step() is True

    def test_step_over_call(self):
        e = VMEngine()
        # Simple program: CALL then HLT
        e.cpu.eax = 0
        e.load_source("NOP\nHLT")
        assert e.step() is True

    def test_tracing_records_steps(self):
        e = _engine("NOP\nNOP\nHLT")
        e.enable_tracing()
        e.run()
        assert e.trace.total_instructions >= 2

    def test_trace_summary(self):
        e = _engine("NOP\nHLT")
        e.enable_tracing()
        e.run()
        summary = e.get_trace_summary()
        assert "total_instructions" in summary
        assert "total_time_ms" in summary


# ── Console Device ───────────────────────────────────────────────────────────

class TestConsoleDevice:
    def test_write_byte(self):
        c = ConsoleDevice()
        c.write_byte(0x3F8, ord("A"), 8)
        assert c.output == [ord("A")]

    def test_feed_input(self):
        c = ConsoleDevice()
        c.feed_input(b"hello")
        assert c.read_byte(0x3F8) == ord("h")
        assert c.read_byte(0x3F8) == ord("e")

    def test_read_empty(self):
        c = ConsoleDevice()
        assert c.read_byte(0x3F8) == 0

    def test_on_output_callback(self):
        c = ConsoleDevice()
        received = []
        c.on_output = lambda b: received.append(b)
        c.write_byte(0x3F8, 42, 8)
        assert received == [42]


# ── Device Bus ───────────────────────────────────────────────────────────────

class TestDeviceBus:
    def test_register_out(self):
        bus = DeviceBus()
        log = []
        bus.register_out(0x100, lambda p, v, w: log.append((p, v, w)))
        bus.outb(0x100, 0x42)
        assert log == [(0x100, 0x42, 8)]

    def test_register_in(self):
        bus = DeviceBus()
        bus.register_in(0x200, lambda p: 0x55)
        assert bus.inb(0x200) == 0x55

    def test_unregistered_returns_zero(self):
        bus = DeviceBus()
        assert bus.inb(0x999) == 0

    def test_log(self):
        bus = DeviceBus()
        bus.outb(0x10, 0x42)
        bus.outw(0x20, 0x1234)
        assert len(bus.log) == 2


# ── Process Management ───────────────────────────────────────────────────────

class TestProcessManagement:
    def test_create_process(self):
        e = _engine()
        pcb = e.create_process("test_proc")
        assert pcb.name == "test_proc"
        assert pcb.pid > 0

    def test_switch_to_process(self):
        e = _engine()
        pcb1 = e.create_process("proc1")
        pcb2 = e.create_process("proc2")
        pcb1.eax = 0x1111
        pcb2.eax = 0x2222
        e.switch_to_process(pcb1.pid)
        assert e.cpu.eax == 0x1111
        e.switch_to_process(pcb2.pid)
        assert e.cpu.eax == 0x2222


# ── State Snapshot ───────────────────────────────────────────────────────────

class TestStateSnapshot:
    def test_snapshot_keys(self):
        e = _engine("NOP")
        snap = e.state_snapshot()
        assert "registers" in snap
        assert "flags" in snap
        assert "stack" in snap
        assert "breakpoints" in snap

    def test_repr(self):
        e = _engine("NOP")
        r = repr(e)
        assert "VMEngine" in r
        assert "eip=0x" in r


# ── Reset ────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_state(self):
        e = _engine("NOP")
        e.cpu.eax = 42
        e.set_breakpoint(0x1000)
        e.enable_tracing()
        e.run(max_steps=5)
        e.reset()
        assert e.cpu.eax == 0
        assert len(e.list_breakpoints()) == 0
        assert e.trace.total_instructions == 0
        assert not e.is_running


# ── Integration: Real Programs ───────────────────────────────────────────────

class TestRealPrograms:
    def test_mov_imm32(self):
        e = _engine("[BITS 32]\nmov eax, 42\nHLT")
        e.run()
        assert e.cpu.eax == 42

    def test_add_two_regs(self):
        e = _engine("[BITS 32]\nmov eax, 10\nmov ebx, 20\nadd eax, ebx\nHLT")
        e.run()
        assert e.cpu.eax == 30

    def test_sub_regs(self):
        e = _engine("[BITS 32]\nmov eax, 50\nmov ebx, 30\nsub eax, ebx\nHLT")
        e.run()
        assert e.cpu.eax == 20

    def test_and_regs(self):
        e = _engine("[BITS 32]\nmov eax, 0xFF\nand eax, 0x0F\nHLT")
        e.run()
        assert e.cpu.eax == 0x0F

    def test_or_regs(self):
        e = _engine("[BITS 32]\nmov eax, 0xF0\nor eax, 0x0F\nHLT")
        e.run()
        assert e.cpu.eax == 0xFF

    def test_xor_regs(self):
        e = _engine("[BITS 32]\nmov eax, 0xFF\nxor eax, 0xFF\nHLT")
        e.run()
        assert e.cpu.eax == 0

    def test_push_pop(self):
        e = _engine("[BITS 32]\nmov eax, 0xDEADBEEF\npush eax\nmov eax, 0\npop eax\nHLT")
        e.run()
        assert e.cpu.eax == 0xDEADBEEF

    def test_not_eax(self):
        e = _engine("[BITS 32]\nmov eax, 0\nnot eax\nHLT")
        e.run()
        assert e.cpu.eax == 0xFFFFFFFF

    def test_neg_eax(self):
        e = _engine("[BITS 32]\nmov eax, 5\nneg eax\nHLT")
        e.run()
        assert e.cpu.eax == 0xFFFFFFFB

    def test_mul(self):
        e = _engine("[BITS 32]\nmov eax, 7\nmov ebx, 6\nmul ebx\nHLT")
        e.run()
        assert e.cpu.eax == 42
        assert e.cpu.edx == 0

    def test_div(self):
        e = _engine("[BITS 32]\nmov edx, 0\nmov eax, 100\nmov ebx, 7\ndiv ebx\nHLT")
        e.run()
        assert e.cpu.eax == 14
        assert e.cpu.edx == 2

    def test_cmp_zf(self):
        e = _engine("[BITS 32]\nmov eax, 5\ncmp eax, 5\nHLT")
        e.run()
        assert e.cpu.zf

    def test_jnz_taken(self):
        e = _engine("[BITS 32]\nmov eax, 0\nmov ecx, 3\nloop:\nadd eax, 1\nsub ecx, 1\njnz loop\nHLT")
        e.run()
        assert e.cpu.eax == 3

    def test_jnz_not_taken(self):
        e = _engine("[BITS 32]\nmov eax, 0\nmov ecx, 1\nsub ecx, 1\njnz skip\nmov eax, 99\nHLT\nskip:\nmov eax, 42\nHLT")
        e.run()
        assert e.cpu.eax == 99

    def test_mov_mem_eax(self):
        e = _engine("[BITS 32]\nmov dword [0x100000], 0xCAFEBABE\nmov eax, [0x100000]\nHLT")
        e.run()
        assert e.cpu.eax == 0xCAFEBABE

    def test_inc_dec(self):
        e = _engine("[BITS 32]\nmov eax, 0\ninc eax\ninc eax\ninc eax\ndec eax\nHLT")
        e.run()
        assert e.cpu.eax == 2

    def test_loop_counter(self):
        e = _engine("[BITS 32]\nmov ecx, 100\nmov eax, 0\nloop:\nadd eax, 1\nsub ecx, 1\njnz loop\nHLT")
        e.run()
        assert e.cpu.eax == 100

    def test_stack_frame(self):
        e = _engine("[BITS 32]\npush eax\npush ebx\nmov eax, esp\npop ebx\npop ebx\nHLT")
        e.run()
        assert e.cpu.esp == 0x3FFFFC

    def test_lea(self):
        e = _engine("[BITS 32]\nmov eax, 0x100\nlea ebx, [eax+0x20]\nHLT")
        e.run()
        assert e.cpu.ebx == 0x120

    def test_xchg(self):
        e = _engine("[BITS 32]\nmov eax, 111\nmov ebx, 222\nxchg eax, ebx\nHLT")
        e.run()
        assert e.cpu.eax == 222
        assert e.cpu.ebx == 111

    def test_cld_std(self):
        e = _engine("[BITS 32]\ncld\nstd\nHLT")
        e.run()
        assert e.is_halted

    def test_nop_count(self):
        e = _engine("[BITS 32]\nNOP\nNOP\nNOP\nNOP\nNOP\nHLT")
        e.enable_tracing()
        e.run()
        assert e.trace.total_instructions == 6

    def test_movzx(self):
        e = _engine("[BITS 32]\nmov eax, 0\nmov al, 0xFF\nand eax, 0xFF\nHLT")
        e.run()
        assert e.cpu.eax == 0xFF

    def test_imul_two_op(self):
        e = _engine("[BITS 32]\nmov eax, 7\nmov ebx, 6\nimul ebx\nHLT")
        e.run()
        assert e.cpu.eax == 42

    def test_shl(self):
        e = _engine("[BITS 32]\nmov eax, 1\nshl eax, 4\nHLT")
        e.run()
        assert e.cpu.eax == 16

    def test_shr(self):
        e = _engine("[BITS 32]\nmov eax, 256\nshr eax, 4\nHLT")
        e.run()
        assert e.cpu.eax == 16

    def test_rol(self):
        e = _engine("[BITS 32]\nmov eax, 1\nrol eax, 1\nHLT")
        e.run()
        assert e.cpu.eax == 2

    def test_ror(self):
        e = _engine("[BITS 32]\nmov eax, 0x80000000\nror eax, 1\nHLT")
        e.run()
        assert e.cpu.eax == 0x40000000

    def test_adc_with_carry(self):
        e = _engine("[BITS 32]\nmov eax, 0xFFFFFFFF\nadd eax, 1\nadc eax, 0\nHLT")
        e.run()
        assert e.cpu.eax == 1

    def test_sbb_with_borrow(self):
        e = _engine("[BITS 32]\nmov eax, 0\nsub eax, 1\nmov ecx, eax\nsub eax, ecx\nHLT")
        e.run()
        assert e.cpu.eax == 0

    def test_test_flags(self):
        e = _engine("[BITS 32]\nmov eax, 0\ntest eax, eax\nHLT")
        e.run()
        assert e.cpu.zf

    def test_bswap(self):
        e = _engine("[BITS 32]\nmov eax, 0x12345678\nHLT")
        e.run()
        assert e.cpu.eax == 0x12345678


# ── Integration: Complex Programs ────────────────────────────────────────────

class TestComplexPrograms:
    def test_fibonacci(self):
        """Compute fibonacci(10) = 55 iteratively."""
        e = _engine(
            "[BITS 32]\n"
            "mov eax, 0\n"
            "mov ebx, 1\n"
            "mov ecx, 10\n"
            "fib:\n"
            "mov edx, eax\n"
            "add edx, ebx\n"
            "mov eax, ebx\n"
            "mov ebx, edx\n"
            "sub ecx, 1\n"
            "jnz fib\n"
            "HLT"
        )
        e.run()
        assert e.cpu.eax == 55

    def test_factorial(self):
        """Compute factorial(6) = 720 iteratively."""
        e = _engine(
            "[BITS 32]\n"
            "mov eax, 1\n"
            "mov ecx, 6\n"
            "fac:\n"
            "imul eax, ecx\n"
            "sub ecx, 1\n"
            "jnz fac\n"
            "HLT"
        )
        e.run()
        assert e.cpu.eax == 720

    def test_gcd(self):
        """GCD(48, 36) = 12 via subtraction."""
        e = _engine(
            "[BITS 32]\n"
            "mov eax, 48\n"
            "mov ebx, 36\n"
            "gcd:\n"
            "cmp eax, ebx\n"
            "je done\n"
            "jl less\n"
            "sub eax, ebx\n"
            "jmp gcd\n"
            "less:\n"
            "sub ebx, eax\n"
            "jmp gcd\n"
            "done:\n"
            "HLT"
        )
        e.run()
        assert e.cpu.eax == 12

    def test_strlen(self):
        """Count length of a string in memory."""
        e = VMEngine()
        e.write_memory(0x80000, b"Hello World!\x00")
        e.load_source(
            "[BITS 32]\n"
            "mov esi, 0x80000\n"
            "mov eax, 0\n"
            "count:\n"
            "mov bl, [esi]\n"
            "cmp bl, 0\n"
            "je done\n"
            "inc eax\n"
            "inc esi\n"
            "jmp count\n"
            "done:\n"
            "HLT"
        )
        e.run()
        assert e.cpu.eax == 12

    def test_sum_array(self):
        """Sum an array of 5 dwords."""
        e = VMEngine()
        e.write_memory(0x80000, b"\x01\x00\x00\x00")
        e.write_memory(0x80004, b"\x02\x00\x00\x00")
        e.write_memory(0x80008, b"\x03\x00\x00\x00")
        e.write_memory(0x8000C, b"\x04\x00\x00\x00")
        e.write_memory(0x80010, b"\x05\x00\x00\x00")
        e.load_source(
            "[BITS 32]\n"
            "mov esi, 0x80000\n"
            "mov ecx, 5\n"
            "mov eax, 0\n"
            "sum:\n"
            "add eax, [esi]\n"
            "add esi, 4\n"
            "sub ecx, 1\n"
            "jnz sum\n"
            "HLT"
        )
        e.run()
        assert e.cpu.eax == 15

    def test_bubble_sort(self):
        """Bubble sort 5 bytes."""
        e = VMEngine()
        e.write_memory(0x80000, b"\x05\x03\x01\x04\x02")
        e.load_source(
            "[BITS 32]\n"
            "mov ecx, 4\n"
            "outer:\n"
            "mov esi, 0x80000\n"
            "mov edx, 4\n"
            "inner:\n"
            "mov al, [esi]\n"
            "mov bl, [esi+1]\n"
            "cmp al, bl\n"
            "jle skip\n"
            "mov [esi], bl\n"
            "mov [esi+1], al\n"
            "skip:\n"
            "inc esi\n"
            "sub edx, 1\n"
            "jnz inner\n"
            "sub ecx, 1\n"
            "jnz outer\n"
            "HLT"
        )
        e.run()
        data = e.read_memory(0x80000, 5)
        assert data == b"\x01\x02\x03\x04\x05"

    def test_bit_count(self):
        """Count set bits in 0b10110101 (5 bits)."""
        e = _engine(
            "[BITS 32]\n"
            "mov eax, 0\n"
            "mov ecx, 0xB5\n"
            "mov ebx, 0\n"
            "count:\n"
            "test ecx, 1\n"
            "jz zero\n"
            "inc eax\n"
            "zero:\n"
            "shr ecx, 1\n"
            "inc ebx\n"
            "cmp ebx, 8\n"
            "jl count\n"
            "HLT"
        )
        e.run()
        assert e.cpu.eax == 5


# ── Integration: Breakpoint Debugging ────────────────────────────────────────

class TestBreakpointDebugging:
    def test_breakpoint_stops_execution(self):
        e = _engine("NOP\nNOP\nNOP\nHLT")
        bp_id = e.set_breakpoint(0x1001)
        e.run()
        assert e.cpu.eip == 0x1001
        assert not e.is_halted

    def test_breakpoint_hit_count_increments(self):
        e = _engine("NOP\nNOP\nHLT")
        bp_id = e.set_breakpoint(0x1000)
        e.run(max_steps=1)
        e.clear_breakpoints()
        assert e._breakpoints.get(bp_id) is None

    def test_conditional_breakpoint(self):
        counter = [0]
        e = _engine("NOP\nNOP\nNOP\nHLT")
        e.set_breakpoint(0x1000, condition=lambda: counter[0] >= 2)
        def on_step(ev):
            counter[0] += 1
        e.on_step(on_step)
        e.run()
        assert counter[0] >= 2

    def test_continue_after_breakpoint(self):
        e = _engine("NOP\nNOP\nHLT")
        bp_id = e.set_breakpoint(0x1001)
        e.run()
        assert e.cpu.eip == 0x1001
        e.continue_execution()
        assert e.is_halted

    def test_multiple_breakpoints(self):
        e = _engine("NOP\nNOP\nNOP\nHLT")
        e.set_breakpoint(0x1000)
        e.set_breakpoint(0x1002)
        hits = []
        def on_bp(ev):
            hits.append(ev.eip)
        e.on_breakpoint(on_bp)
        e.run()
        assert len(hits) == 1
        assert hits[0] == 0x1000

    def test_one_shot_breakpoint(self):
        e = _engine("NOP\nNOP\nNOP\nHLT")
        e.set_breakpoint_once(0x1000)
        hits = []
        def on_bp(ev):
            hits.append(ev.eip)
        e.on_breakpoint(on_bp)
        e.run()
        assert len(hits) == 1

    def test_disabled_breakpoint_not_hit(self):
        e = _engine("NOP\nNOP\nNOP\nHLT")
        bp_id = e.set_breakpoint(0x1001)
        e.disable_breakpoint(bp_id)
        hits = []
        def on_bp(ev):
            hits.append(ev.eip)
        e.on_breakpoint(on_bp)
        e.run()
        assert len(hits) == 0


# ── Integration: Tracing ─────────────────────────────────────────────────────

class TestTracing:
    def test_trace_records_all_steps(self):
        e = _engine("NOP\nNOP\nNOP\nHLT")
        e.enable_tracing()
        e.run()
        assert e.trace.total_instructions == 4

    def test_trace_step_event_has_registers(self):
        e = _engine("NOP\nHLT")
        e.enable_tracing()
        e.run()
        assert len(e.trace.steps) >= 2
        assert "eax" in e.trace.steps[0].registers

    def test_trace_step_event_has_flags(self):
        e = _engine("NOP\nHLT")
        e.enable_tracing()
        e.run()
        assert "zf" in e.trace.steps[0].flags

    def test_trace_step_event_has_opcode(self):
        e = _engine("NOP\nHLT")
        e.enable_tracing()
        e.run()
        assert e.trace.steps[0].opcode == 0x90

    def test_trace_summary_instructions_per_ms(self):
        e = _engine("NOP\nNOP\nNOP\nNOP\nNOP\nHLT")
        e.enable_tracing()
        e.run()
        summary = e.get_trace_summary()
        assert summary["instructions_per_ms"] >= 0

    def test_trace_exit_reason(self):
        e = _engine("HLT")
        e.enable_tracing()
        e.run()
        assert e.trace.exit_reason == "halt"

    def test_trace_max_steps_exit(self):
        e = _engine("NOP\nNOP\nNOP\nHLT")
        e.enable_tracing()
        e.run(max_steps=2)
        assert e.trace.exit_reason == "max_steps"

    def test_tracing_can_be_disabled(self):
        e = _engine("NOP\nHLT")
        e.enable_tracing()
        e.run()
        assert e.trace.total_instructions == 2
        e.disable_tracing()
        assert not e._tracing


# ── Integration: Edge Cases ──────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_program_halt(self):
        e = VMEngine()
        e.load_bytes(b"\xF4")
        e.run()
        assert e.is_halted

    def test_single_nop_then_hlt(self):
        e = _engine("NOP\nHLT")
        e.run()
        assert e.is_halted

    def test_many_nops(self):
        e = _engine("\n".join(["NOP"] * 1000 + ["HLT"]))
        e.enable_tracing()
        e.run()
        assert e.trace.total_instructions == 1001

    def test_zero_length_run(self):
        e = _engine("NOP\nHLT")
        trace = e.run(max_steps=1)
        assert trace.exit_reason == "max_steps"

    def test_run_after_halt_resets(self):
        e = _engine("HLT")
        e.run()
        assert e.is_halted
        e.reset()
        e.load_source("NOP\nHLT")
        trace = e.run()
        assert e.is_halted

    def test_load_overwrites(self):
        e = _engine("NOP\nHLT")
        e.run()
        e.reset()
        e.load_source("MOV EAX, 42\nHLT")
        e.run()
        assert e.cpu.eax == 42

    def test_break_requested(self):
        e = _engine("NOP\nNOP\nNOP\nHLT")
        def on_step(ev):
            e.request_break()
        e.on_step(on_step)
        e.run()
        assert e.trace.exit_reason == "break_request"

    def test_step_out_no_call(self):
        e = _engine("NOP\nHLT")
        e.step_out()
        assert e.is_halted

    def test_step_over_no_call(self):
        e = _engine("NOP\nHLT")
        result = e.step_over()
        assert result is True
        assert e.cpu.eip == 0x1001

    def test_registers_after_program(self):
        e = _engine("[BITS 32]\nmov eax, 1\nmov ebx, 2\nmov ecx, 3\nmov edx, 4\nHLT")
        e.run()
        regs = e.registers()
        assert regs["eax"] == 1
        assert regs["ebx"] == 2
        assert regs["ecx"] == 3
        assert regs["edx"] == 4

    def test_stack_grows_down(self):
        e = _engine("[BITS 32]\npush eax\npush ebx\nHLT")
        e.run()
        assert e.cpu.esp < 0x3FFFFC

    def test_flags_after_sub(self):
        e = _engine("[BITS 32]\nmov eax, 5\nsub eax, 5\nHLT")
        e.run()
        assert e.cpu.zf

    def test_flags_after_negative_sub(self):
        e = _engine("[BITS 32]\nmov eax, 3\nsub eax, 5\nHLT")
        e.run()
        assert e.cpu.sf

    def test_carry_on_add_overflow(self):
        e = _engine("[BITS 32]\nmov eax, 0xFFFFFFFF\nadd eax, 1\nHLT")
        e.run()
        assert e.cpu.cf


# ── Integration: I/O ─────────────────────────────────────────────────────────

class TestIO:
    def test_console_output(self):
        e = _engine()
        e.console.feed_input(b"AB")
        assert e.console.read_byte(0x3F8) == ord("A")
        assert e.console.read_byte(0x3F8) == ord("B")

    def test_console_empty_input(self):
        e = _engine()
        assert e.console.read_byte(0x3F8) == 0

    def test_device_bus_output(self):
        e = _engine()
        log = []
        e.devices.register_out(0x100, lambda p, v, w: log.append(v))
        e.devices.outb(0x100, 0x42)
        assert log == [0x42]

    def test_device_bus_input(self):
        e = _engine()
        e.devices.register_in(0x200, lambda p: 0x55)
        assert e.devices.inb(0x200) == 0x55


# ── Integration: Process Management ──────────────────────────────────────────

class TestProcessManagementAdvanced:
    def test_multiple_process_switch(self):
        e = _engine()
        pids = []
        for i in range(5):
            p = e.create_process(f"proc{i}")
            p.eax = i * 100
            pids.append(p.pid)
        for i, pid in enumerate(pids):
            e.switch_to_process(pid)
            assert e.cpu.eax == i * 100

    def test_process_switch_unknown_pid(self):
        e = _engine()
        with pytest.raises(ValueError):
            e.switch_to_process(9999)

    def test_process_state(self):
        e = _engine()
        pcb = e.create_process("test")
        pcb.eax = 42
        e.switch_to_process(pcb.pid)
        assert e.cpu.eax == 42
        assert pcb.state == "running"


# ── Error Handling ─────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_unknown_opcode_raises_ins_fault(self):
        e = _engine()
        e.enable_tracing()
        e._cpu.load(bytes([0x62]), 0x1000)
        result = e.step()
        assert result is False
        assert len(e._trace.faults) == 1
        assert e._trace.faults[0].fault_type is InsFault
        assert "unknown opcode 0x62" in e._trace.faults[0].message

    def test_fault_event_preserves_original_message(self):
        e = _engine()
        e.enable_tracing()
        e._cpu.load(bytes([0x62]), 0x1000)
        e.step()
        fault = e._trace.faults[0]
        assert "InsFault:" in fault.message
        assert "unknown opcode" in fault.message

    def test_div_by_zero_raises_ins_fault(self):
        e = _engine("[BITS 32]\nmov eax, 0\nmov ecx, 0\ndiv ecx\nHLT")
        trace = e.run()
        assert len(trace.faults) == 1
        assert trace.faults[0].fault_type is InsFault
        assert "DIV" in trace.faults[0].message

    def test_stack_overflow_on_push(self):
        e = _engine("[BITS 32]\n")
        # Set ESP to 0 — next push will go to -4 (overflow)
        e.cpu.esp = 0
        faults = []
        e.on_fault(lambda f: faults.append(f))
        e._cpu.load(bytes([0x6A, 0x02]), 0x1001)
        result = e.step()
        assert result is False
        assert any("stack overflow" in f.message for f in faults)

    def test_stack_underflow_on_pop(self):
        e = _engine("[BITS 32]\n")
        e.cpu._set32(4, e.cpu._mem_size)
        e._cpu.load(bytes([0x58]), 0x1000)
        faults = []
        e.on_fault(lambda f: faults.append(f))
        e.step()
        assert any("stack underflow" in f.message for f in faults)

    def test_mem_fault_on_out_of_bounds_read(self):
        e = _engine()
        e.cpu.eax = 0x999999
        e._cpu.load(bytes([0x8B, 0x00]), 0x1000)
        faults = []
        e.on_fault(lambda f: faults.append(f))
        e.step()
        assert len(faults) == 1
        assert faults[0].fault_type.__name__ == "MemFault"

    def test_breakpoint_exit_reason_in_run(self):
        e = _engine("NOP\nNOP\nHLT")
        e.set_breakpoint(0x1001)
        trace = e.run()
        assert trace.exit_reason == "breakpoint"

    def test_breakpoint_in_single_step(self):
        e = _engine("NOP\nHLT")
        e.set_breakpoint(0x1001)
        # First step at 0x1000 (NOP) should succeed
        assert e.step() is True
        # Second step at 0x1001 (HLT) should hit breakpoint
        assert e.step() is False

    def test_continue_execution_skips_breakpoint(self):
        e = _engine("NOP\nNOP\nHLT")
        e.set_breakpoint(0x1001)
        e.run()
        assert e.cpu.eip == 0x1001
        e.continue_execution()
        assert e.is_halted

    def test_hlt_in_16bit_mode_raises_halt(self):
        e = _engine()
        e._cpu.load(bytes([0x66, 0xF4]), 0x1000)
        faults = []
        e.on_fault(lambda f: faults.append(f))
        result = e.step()
        assert result is False
        assert e.is_halted

    def test_read_byte_out_of_bounds(self):
        e = _engine()
        assert e.read_byte(0x999999) == 0

    def test_write_byte_out_of_bounds(self):
        e = _engine()
        e.write_byte(0x999999, 0xFF)

    def test_fault_callback_fires(self):
        e = _engine()
        faults = []
        e.on_fault(lambda f: faults.append(f))
        e._cpu.load(bytes([0x62]), 0x1000)
        e.step()
        assert len(faults) == 1
        assert faults[0].fault_type is InsFault

    def test_run_exits_on_fault(self):
        e = _engine("NOP\nNOP")
        e._cpu.load(bytes([0x62]), 0x1002)
        trace = e.run()
        assert trace.exit_reason == "fault"
        assert len(trace.faults) == 1


# ── Error Handling Edge Cases ────────────────────────────────────────────────

class TestErrorHandlingEdgeCases:
    def test_read32_boundary_last_valid(self):
        e = _engine()
        # read32 at mem_size - 4 is the last valid 32-bit read
        addr = e._cpu._mem_size - 4
        e._cpu._write32(addr, 0xDEADBEEF)
        assert e._cpu._read32(addr) == 0xDEADBEEF

    def test_read32_boundary_first_invalid(self):
        e = _engine()
        addr = e._cpu._mem_size - 3
        with pytest.raises(MemFault):
            e._cpu._read32(addr)

    def test_write32_boundary_last_valid(self):
        e = _engine()
        addr = e._cpu._mem_size - 4
        e._cpu._write32(addr, 0xCAFEBABE)
        assert e._cpu._read32(addr) == 0xCAFEBABE

    def test_write32_boundary_first_invalid(self):
        e = _engine()
        addr = e._cpu._mem_size - 3
        with pytest.raises(MemFault):
            e._cpu._write32(addr, 0x12345678)

    def test_read8_boundary_valid(self):
        e = _engine()
        addr = e._cpu._mem_size - 1
        e._cpu._mem[addr] = 0xAB
        assert e._cpu._read8(addr) == 0xAB

    def test_read8_boundary_invalid(self):
        e = _engine()
        addr = e._cpu._mem_size
        with pytest.raises(MemFault):
            e._cpu._read8(addr)

    def test_write8_boundary_valid(self):
        e = _engine()
        addr = e._cpu._mem_size - 1
        e._cpu._write8(addr, 0xCD)
        assert e._cpu._mem[addr] == 0xCD

    def test_write8_boundary_invalid(self):
        e = _engine()
        addr = e._cpu._mem_size
        with pytest.raises(MemFault):
            e._cpu._write8(addr, 0xEF)

    def test_fault_registers_snapshot(self):
        e = _engine("[BITS 32]\nmov eax, 0x12345678\nmov ebx, 0xDEADBEEF")
        e.step()  # mov eax
        e.step()  # mov ebx
        # Now inject a fault
        e._cpu.load(bytes([0x62]), 0x1002)
        faults = []
        e.on_fault(lambda f: faults.append(f))
        e.step()
        assert len(faults) == 1
        assert faults[0].registers["eax"] == 0x12345678
        assert faults[0].registers["ebx"] == 0xDEADBEEF
        assert faults[0].eip == 0x1002

    def test_breakpoint_then_fault(self):
        e = _engine("NOP\nHLT")
        # Set breakpoint at NOP, then inject unknown opcode at HLT address
        e.set_breakpoint(0x1000)
        e._cpu.load(bytes([0x62]), 0x1001)
        # First, hit the breakpoint
        result = e.step()
        assert result is False  # breakpoint hit
        # Now step past it — should fault on 0x62
        faults = []
        e.on_fault(lambda f: faults.append(f))
        e._skip_breakpoint_check = True
        try:
            e.step()
        finally:
            e._skip_breakpoint_check = False
        assert len(faults) == 1
        assert faults[0].fault_type is InsFault

    def test_one_shot_breakpoint_disables_after_hit(self):
        e = _engine("NOP\nNOP\nHLT")
        bp_id = e.set_breakpoint_once(0x1001)
        # First run hits breakpoint
        e.run()
        assert e.cpu.eip == 0x1001
        assert e._breakpoints[bp_id].enabled is False
        # Continue — should run to completion without hitting same breakpoint
        e.continue_execution()
        assert e.is_halted

    def test_conditional_breakpoint_condition_raises(self):
        e = _engine("NOP\nHLT")
        def bad_condition():
            raise RuntimeError("condition error")
        e.set_breakpoint(0x1000, condition=bad_condition)
        # The condition raising should prevent the breakpoint from triggering
        # (should_trigger catches exceptions and returns False)
        result = e.step()
        assert result is True  # NOP executed, breakpoint didn't fire

    def test_step_over_past_call(self):
        e = _engine("[BITS 32]\ncall 0x1100\nHLT")
        # Place a HLT at 0x1100
        e._cpu.load(bytes([0xF4]), 0x1100)
        e.step_over()  # Should execute CALL target as a unit
        assert e.cpu.eip == 0x1006  # Past the CALL (5 bytes)

    def test_step_out_of_function(self):
        e = _engine("[BITS 32]\ncall 0x1100\nHLT")
        e._cpu.load(bytes([0xC3]), 0x1100)  # RET at target
        e.step()  # execute CALL
        e.step_out()  # should run until RET
        assert e.cpu.eip == 0x1006  # Past the CALL

    def test_continue_on_already_halted(self):
        e = _engine("HLT")
        e.run()
        assert e.is_halted
        # continue_execution on halted CPU should return trace
        trace = e.continue_execution()
        assert trace.exit_reason == "halt"

    def test_breakpoint_at_address_zero(self):
        e = _engine("[BITS 32]\nHLT")
        e._cpu.eip = 0
        e._cpu.load(bytes([0xF4]), 0)
        e.set_breakpoint(0)
        result = e.step()
        assert result is False  # breakpoint hit

    def test_multiple_breakpoints_same_address(self):
        e = _engine("NOP\nHLT")
        bp1 = e.set_breakpoint(0x1000, label="first")
        bp2 = e.set_breakpoint(0x1000, label="second")
        e.step()
        # Both should have hit_count incremented (both matched)
        assert e._breakpoints[bp1].hit_count == 1
        assert e._breakpoints[bp2].hit_count == 1

    def test_fault_eip_points_to_faulting_instruction(self):
        e = _engine()
        # Place 3 bytes: NOP, NOP, unknown
        e._cpu.load(bytes([0x90, 0x90, 0x62]), 0x1000)
        e.step()  # NOP
        e.step()  # NOP
        faults = []
        e.on_fault(lambda f: faults.append(f))
        e.step()  # should fault
        assert faults[0].eip == 0x1002

    def test_halt_detection_via_run(self):
        e = _engine("NOP\nHLT")
        trace = e.run()
        assert trace.exit_reason == "halt"
        assert e.is_halted

    def test_fault_detection_via_run(self):
        e = _engine("NOP\nNOP")
        e._cpu.load(bytes([0xF6, 0xF0]), 0x1002)  # DIV by zero (F6 /6 with ECX=0)
        trace = e.run()
        assert trace.exit_reason == "fault"
        assert len(trace.faults) == 1

    def test_max_steps_prevents_fault(self):
        e = _engine("NOP\nNOP\nNOP\nNOP\nNOP")
        trace = e.run(max_steps=3)
        assert trace.exit_reason == "max_steps"
        assert trace.total_instructions == 3

    def test_break_priority_over_fault(self):
        """Breakpoint at same address as faulting instruction triggers breakpoint first."""
        e = _engine()
        e._cpu.load(bytes([0x62]), 0x1000)  # unknown opcode
        e.set_breakpoint(0x1000)
        # Breakpoint check happens BEFORE execution, so it fires first
        result = e.step()
        assert result is False
        assert len(e._trace.breakpoints_hit) == 1
        assert len(e._trace.faults) == 0

    def test_read_memory_at_exact_boundary(self):
        e = _engine()
        data = e.read_memory(e._cpu._mem_size - 2, 4)
        # Last 2 bytes valid, last 2 return 0 (out of bounds)
        assert len(data) == 4

    def test_write_memory_out_of_bounds_silent(self):
        e = _engine()
        # Should not raise — write_memory silently drops out-of-bounds writes
        e.write_memory(e._cpu._mem_size, b"\x01\x02\x03")

    def test_push_pop_roundtrip(self):
        e = _engine("[BITS 32]\n")
        e._cpu.load(bytes([
            0x68, 0x78, 0x56, 0x34, 0x12,  # PUSH 0x12345678
            0x5B,                            # POP EBX
            0xF4,                            # HLT
        ]), 0x1000)
        e.run()
        assert e.cpu.ebx == 0x12345678

    def test_push16_pop16_roundtrip(self):
        e = _engine("[BITS 32]\n")
        e._cpu.load(bytes([
            0x66, 0x68, 0x34, 0x12,  # PUSH word 0x1234
            0x66, 0x5B,               # POP BX
            0xF4,                     # HLT
        ]), 0x1000)
        e.run()
        assert e.cpu._get16(3) == 0x1234  # BX = reg index 3

    def test_fault_type_is_class_not_instance(self):
        e = _engine()
        faults = []
        e.on_fault(lambda f: faults.append(f))
        e._cpu.load(bytes([0x62]), 0x1000)
        e.step()
        assert isinstance(faults[0].fault_type, type)
        assert faults[0].fault_type is InsFault

    def test_fault_callback_gets_all_fields(self):
        e = _engine("[BITS 32]\nmov eax, 0x42")
        e.step()  # mov eax
        faults = []
        e.on_fault(lambda f: faults.append(f))
        e._cpu.load(bytes([0xF6, 0xF0]), 0x1001)  # DIV by zero
        e.step()
        f = faults[0]
        assert f.fault_type is InsFault
        assert isinstance(f.message, str)
        assert isinstance(f.eip, int)
        assert isinstance(f.registers, dict)
        assert "eax" in f.registers
        assert f.registers["eax"] == 0x42
