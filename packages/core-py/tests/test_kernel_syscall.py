"""Coverage tests for domains/shell/kernel_syscall.py."""

import pytest

import domains.shell.kernel_syscall as ksc
from domains.shell.kernel_process import Process, ProcessState, Priority
from domains.shell.kernel_syscall import (
    SyscallNumber,
    SyscallResult,
    SyscallTable,
    build_default_syscall_table,
    get_syscall_table,
    syscall,
)


def _proc(pid=1, state=ProcessState.READY):
    return Process(pid=pid, name=f"p{pid}", state=state)


def test_syscall_number_enum():
    assert SyscallNumber.NONE == 0
    assert SyscallNumber.FORK == 1
    assert SyscallNumber.MALLOC == 20
    assert SyscallNumber.TENSOR_ALLOC == SyscallNumber.MALLOC
    assert SyscallNumber.OPEN_DEVICE == 40
    assert SyscallNumber.INFERENCE_START == 60
    assert SyscallNumber.TRAIN_START == 70
    assert SyscallNumber.INT_ENABLE == 80
    assert SyscallNumber.SCHED_YIELD == 90
    assert SyscallNumber.NEURAL_FORWARD == 100
    assert SyscallNumber.CONSOLE_WRITE == 111
    assert SyscallNumber.UPTIME == 200
    assert SyscallNumber.NOP == 255
    assert ksc.SYSCALLS is SyscallNumber


def test_syscall_result_helpers():
    ok = SyscallResult.ok("v")
    assert ok.success is True
    assert ok.value == "v"
    assert bool(ok) is True
    fail = SyscallResult.fail("bad", errno=9)
    assert fail.success is False
    assert fail.error == "bad"
    assert fail.errno == 9
    assert bool(fail) is False


def test_syscall_result_getitem():
    r = SyscallResult(success=True, value=42, error=None)
    assert r["value"] == 42
    assert r["missing"] is None


def test_register_two_arg_form():
    t = SyscallTable()
    t.register(SyscallNumber.NOP, lambda caller, *a: True)
    assert t.has(SyscallNumber.NOP)
    entry = t.get_entry(SyscallNumber.NOP)
    assert entry.name == "NOP"
    assert entry.handler is not None


def test_register_three_arg_form():
    t = SyscallTable()
    t.register(SyscallNumber.NOP, "custom_nop", lambda caller, *a: True,
               min_args=2, description="custom")
    entry = t.get_entry(SyscallNumber.NOP)
    assert entry.name == "custom_nop"
    assert entry.min_args == 2
    assert entry.description == "custom"


def test_unregister():
    t = SyscallTable()
    t.register(SyscallNumber.NOP, lambda c, *a: True)
    t.unregister(SyscallNumber.NOP)
    assert not t.has(SyscallNumber.NOP)
    assert t.get_entry(SyscallNumber.NOP) is None


def test_dispatch_caller_number_convention():
    t = SyscallTable()
    t.register(SyscallNumber.GETPID, "getpid", ksc._syscall_getpid)
    proc = _proc(7)
    result = t.dispatch(proc, SyscallNumber.GETPID)
    assert result.success is True
    assert result.value == 7


def test_dispatch_number_caller_convention():
    t = SyscallTable()
    t.register(SyscallNumber.GETPID, "getpid", ksc._syscall_getpid)
    proc = _proc(7)
    result = t.dispatch(SyscallNumber.GETPID, proc)
    assert result.success is True
    assert result.value == 7


def test_dispatch_stopped_process_guard():
    t = SyscallTable()
    t.register(SyscallNumber.NOP, "nop", ksc._syscall_nop)
    proc = _proc(1, state=ProcessState.STOPPED)
    result = t.dispatch(proc, SyscallNumber.NOP)
    assert result.success is False
    assert result.error == "Process is stopped"
    assert result.errno == 1


def test_dispatch_unknown_syscall():
    t = SyscallTable()
    proc = _proc()
    result = t.dispatch(proc, SyscallNumber.FILE_OPEN)
    assert result.success is False
    assert result.error == f"Unknown syscall: {int(SyscallNumber.FILE_OPEN)}"
    assert result.errno == 2


def test_dispatch_min_args_error():
    t = SyscallTable()
    t.register(SyscallNumber.SET_PRIORITY, "set_priority", ksc._syscall_set_priority,
               min_args=1)
    proc = _proc()
    result = t.dispatch(proc, SyscallNumber.SET_PRIORITY)
    assert result.success is False
    assert "requires 1 args" in result.error
    assert result.errno == 3


def test_dispatch_handler_returns_syscall_result():
    t = SyscallTable()
    t.register(SyscallNumber.NOP, "r", lambda c, *a: SyscallResult.fail("handled", 7))
    proc = _proc()
    result = t.dispatch(proc, SyscallNumber.NOP)
    assert result.success is False
    assert result.error == "handled"
    assert result.errno == 7


def test_dispatch_handler_returns_two_tuple():
    t = SyscallTable()
    t.register(SyscallNumber.NOP, "r", lambda c, *a: (False, "two"))
    proc = _proc()
    result = t.dispatch(proc, SyscallNumber.NOP)
    assert result.success is False
    assert result.value == "two"


def test_dispatch_handler_returns_three_tuple():
    t = SyscallTable()
    t.register(SyscallNumber.NOP, "r", lambda c, *a: (False, "v", "err"))
    proc = _proc()
    result = t.dispatch(proc, SyscallNumber.NOP)
    assert result.success is False
    assert result.value == "v"
    assert result.error == "err"


def test_dispatch_handler_returns_plain_value():
    t = SyscallTable()
    t.register(SyscallNumber.NOP, "r", lambda c, *a: 123)
    proc = _proc()
    result = t.dispatch(proc, SyscallNumber.NOP)
    assert result.success is True
    assert result.value == 123


def test_dispatch_handler_raises():
    t = SyscallTable()
    t.register(SyscallNumber.NOP, "boom", lambda c, *a: 1 / 0)
    proc = _proc()
    result = t.dispatch(proc, SyscallNumber.NOP)
    assert result.success is False
    assert "division by zero" in result.error
    assert result.errno == 4


def test_dispatch_tracks_calls():
    t = SyscallTable()
    t.register(SyscallNumber.NOP, "nop", ksc._syscall_nop)
    proc = _proc()
    for _ in range(3):
        t.dispatch(proc, SyscallNumber.NOP)
    stats = t.stats()
    assert stats["registered"] == 1
    assert stats["total_calls"] == 3
    assert stats["calls_by_number"]["NOP"] == 3
    assert SyscallNumber.NOP in t._last_call


def test_list_syscalls():
    t = SyscallTable()
    t.register(SyscallNumber.NOP, "nop", ksc._syscall_nop, min_args=0, description="noop")
    t.dispatch(_proc(), SyscallNumber.NOP)
    listing = t.list_syscalls()
    assert listing[0]["number"] == 255
    assert listing[0]["name"] == "nop"
    assert listing[0]["description"] == "noop"
    assert listing[0]["min_args"] == 0
    assert listing[0]["call_count"] == 1


def test_default_table_process_syscalls():
    t = build_default_syscall_table()
    proc = _proc(5)
    assert t.dispatch(proc, SyscallNumber.FORK).value == 6
    assert t.dispatch(proc, SyscallNumber.WAIT).success is True
    assert t.dispatch(proc, SyscallNumber.GETPID).value == 5
    r = t.dispatch(proc, SyscallNumber.SET_PRIORITY, int(Priority.HIGH))
    assert r.success is True
    assert proc.priority == Priority.HIGH
    t.dispatch(proc, SyscallNumber.SET_PRIORITY, 99)
    assert proc.priority == Priority.HIGH
    r = t.dispatch(proc, SyscallNumber.EXIT)
    assert r.success is True
    assert proc.state == ProcessState.STOPPED


def test_default_table_memory_syscalls():
    t = build_default_syscall_table()
    proc = _proc()
    assert t.dispatch(proc, SyscallNumber.MALLOC, (2, 3)).success is True
    assert t.dispatch(proc, SyscallNumber.FREE, 1).value is True


def test_default_table_device_syscalls():
    t = build_default_syscall_table()
    proc = _proc()
    assert t.dispatch(proc, SyscallNumber.OPEN_DEVICE, "null").value == 1
    assert t.dispatch(proc, SyscallNumber.CLOSE_DEVICE, 1).value is True
    assert t.dispatch(proc, SyscallNumber.READ_DEVICE, 1).value == b""
    assert t.dispatch(proc, SyscallNumber.WRITE_DEVICE, 1, b"x").value == 0


def test_default_table_scheduler_syscalls():
    t = build_default_syscall_table()
    proc = _proc(3)
    r = t.dispatch(proc, SyscallNumber.SCHED_YIELD)
    assert r.success is True
    assert proc.state == ProcessState.READY
    r = t.dispatch(proc, SyscallNumber.SCHED_SLEEP, 1.0)
    assert r.success is True
    assert proc.state == ProcessState.WAITING


def test_default_table_console_syscalls():
    t = build_default_syscall_table()
    proc = _proc()
    assert t.dispatch(proc, SyscallNumber.CONSOLE_WRITE, "hello").value == 5
    assert t.dispatch(proc, SyscallNumber.CONSOLE_READ).value == ""


def test_default_table_inference_syscalls():
    t = build_default_syscall_table()
    proc = _proc()
    assert t.dispatch(proc, SyscallNumber.INFERENCE_START, "prompt").value == 0
    assert t.dispatch(proc, SyscallNumber.INFERENCE_CANCEL, 1).value is True


def test_default_table_training_syscalls():
    t = build_default_syscall_table()
    proc = _proc()
    assert t.dispatch(proc, SyscallNumber.TRAIN_START).value == 0
    assert t.dispatch(proc, SyscallNumber.TRAIN_STOP, 1).value is True
    r = t.dispatch(proc, SyscallNumber.TRAIN_STATUS, 1)
    assert r.success is True
    assert r.value == {"status": "idle"}


def test_default_table_uptime_stats_nop():
    t = build_default_syscall_table()
    proc = _proc(9)
    proc.started_at = 0.0
    assert t.dispatch(proc, SyscallNumber.UPTIME).value >= 0.0
    stats = t.dispatch(proc, SyscallNumber.STATS).value
    assert stats["pid"] == 9
    assert stats["name"] == "p9"
    assert stats["state"] == proc.state.name
    assert stats["memory_bytes"] == 0
    assert t.dispatch(proc, SyscallNumber.NOP).success is True


def test_get_syscall_table_caches(monkeypatch):
    monkeypatch.setattr(ksc, "_SYSCALL_TABLE", None)
    t1 = get_syscall_table()
    t2 = get_syscall_table()
    assert t1 is t2
    assert t1.has(SyscallNumber.GETPID)


def test_syscall_convenience():
    proc = _proc(11)
    result = syscall(SyscallNumber.GETPID, proc)
    assert result.success is True
    assert result.value == 11
