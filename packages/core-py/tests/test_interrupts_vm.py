"""Tests for domains.shell.kernel_interrupts and domains.shell.vm_engine."""

from domains.shell.kernel_interrupts import InterruptType, Interrupt, InterruptVector
from domains.shell.vm_engine import (
    Breakpoint, StepEvent, ExecutionTrace, SyscallEvent, FaultEvent, BreakpointEvent,
)


class TestInterruptType:
    def test_all_members(self):
        assert len(InterruptType) == 11
    def test_values(self):
        assert InterruptType.TIMER.value == 0
        assert InterruptType.INFERENCE_DONE.value == 1


class TestInterrupt:
    def test_fields(self):
        i = Interrupt(vector=InterruptType.TIMER, source_pid=1, data="x", priority=0)
        assert i.vector == InterruptType.TIMER
    def test_defaults(self):
        i = Interrupt(vector=InterruptType.CUSTOM)
        assert i.source_pid is None


class TestInterruptVector:
    def test_register_and_fire(self):
        iv = InterruptVector()
        called = []
        iv.register(InterruptType.TIMER, lambda i: called.append(i.vector))
        iv.fire(Interrupt(vector=InterruptType.TIMER))
        assert len(called) == 1
    def test_fire_no_handler(self):
        InterruptVector().fire(Interrupt(vector=InterruptType.CUSTOM))


class TestBreakpoint:
    def test_trigger_enabled(self):
        assert Breakpoint(address=0x100, enabled=True).should_trigger() is True
    def test_trigger_disabled(self):
        assert Breakpoint(address=0x100, enabled=False).should_trigger() is False
    def test_trigger_condition_false(self):
        assert Breakpoint(address=0x100, enabled=True, condition=lambda: False).should_trigger() is False
    def test_trigger_condition_true(self):
        assert Breakpoint(address=0x100, enabled=True, condition=lambda: True).should_trigger() is True


class TestStepEvent:
    def test_fields(self):
        se = StepEvent(eip=0, opcode=1, registers={}, flags={}, instruction_bytes=b"")
        assert se.eip == 0


class TestSyscallEvent:
    def test_fields(self):
        se = SyscallEvent(number=1, args={"fd": 1}, eip=4)
        assert se.number == 1


class TestFaultEvent:
    def test_fields(self):
        fe = FaultEvent(fault_type=ValueError, message="err", eip=8, registers={})
        assert fe.fault_type == ValueError


class TestBreakpointEvent:
    def test_fields(self):
        bp = Breakpoint(address=0x100)
        bpe = BreakpointEvent(breakpoint=bp, eip=0x100, registers={})
        assert bpe.breakpoint is bp


class TestExecutionTrace:
    def test_defaults(self):
        et = ExecutionTrace()
        assert et.steps == []
        assert et.breakpoints_hit == []
        assert et.faults == []
