"""
Coverage tests for previously-uncovered branches in the x86 VM layer:
Scheduler edge cases, X86SyscallHandler syscall error paths, the
training-bridge syscalls, and the X86Shell wrapper lifecycle.
"""

import importlib
import sys
import time
import types

import pytest

import domains.shell.vm as vm
from domains.shell.vm import (
    BlockDevice,
    FlatFS,
    PageFrameAllocator,
    ProcessState,
    ProcessTable,
    Scheduler,
    X86CPU,
    X86Shell,
    X86SyscallHandler,
    X86VirtualSystem,
)

HALT_SRC = "[BITS 32]\nhlt\n"
LOOP_SRC = "[BITS 32]\nlabel:\njmp label\n"


def _standalone_handler(memory_size=0x100000, filesystem=None):
    cpu = X86CPU(memory_size=memory_size)
    ptable = ProcessTable()
    scheduler = Scheduler(ptable, quantum=10)
    allocator = PageFrameAllocator(total_memory=memory_size)
    handler = X86SyscallHandler(cpu, ptable, scheduler, allocator,
                                filesystem=filesystem)
    return cpu, ptable, scheduler, allocator, handler


def _install_fake_bridge(monkeypatch, bridge):
    if "requests" not in sys.modules:
        monkeypatch.setitem(sys.modules, "requests", types.ModuleType("requests"))
    mod = importlib.import_module("domains.shell.vm_training_bridge")
    monkeypatch.setattr(mod, "get_bridge", lambda: bridge)
    return mod


# ══════════════════════════════════════════════════════════════════════════════
# Scheduler edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestSchedulerEdgeCases:
    def setup_method(self):
        self.ptable = ProcessTable()
        self.scheduler = Scheduler(self.ptable, quantum=10)
        self.cpu = X86CPU(memory_size=0x100000)

    def test_preempt_no_current_is_noop(self):
        self.scheduler._preempt(self.cpu)

    def test_exit_current_no_current_is_noop(self):
        self.scheduler.exit_current(self.cpu)

    def test_block_current_no_current_is_noop(self):
        self.scheduler.block_current(self.cpu)

    def test_switch_to_missing_pid_returns_false(self):
        assert self.scheduler.switch_to(self.cpu, 999) is False

    def test_switch_to_terminated_pid_returns_false(self):
        pcb = self.ptable.create(name="dead")
        pcb.state = ProcessState.TERMINATED
        assert self.scheduler.switch_to(self.cpu, pcb.pid) is False

    def test_switch_to_ready_process_succeeds(self):
        pcb = self.ptable.create(name="alive")
        assert self.scheduler.switch_to(self.cpu, pcb.pid) is True
        assert self.scheduler.current.pid == pcb.pid
        assert pcb.state == ProcessState.RUNNING

    def test_switch_to_preempts_current(self):
        first = self.ptable.create(name="first")
        second = self.ptable.create(name="second")
        assert self.scheduler.switch_to(self.cpu, first.pid) is True
        assert self.scheduler.switch_to(self.cpu, second.pid) is True
        assert self.scheduler.current.pid == second.pid
        assert first.pid in self.scheduler._ready_queue


# ══════════════════════════════════════════════════════════════════════════════
# X86SyscallHandler edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestSyscallHandlerTick:
    def test_tick_increments_counter(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        before = vs._syscall._ticks
        vs._syscall.tick()
        assert vs._syscall._ticks == before + 1


class TestSyscallHandlerReadWrite:
    def test_read_rejects_bad_args(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        assert vs._syscall._sys_read(-1, 0x30000, 5) == -1
        assert vs._syscall._sys_read(0, 0x30000, 0) == -1

    def test_read_stdin_with_and_without_key(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        vs.cpu.push_key('a')
        vs.cpu.transfer_key()
        n = vs._syscall._sys_read(0, 0x30000, 10)
        assert n == 1
        assert vs.cpu._mem[0x30000] == ord('a')
        assert vs._syscall._sys_read(0, 0x30000, 10) == 0

    def test_read_file_full_and_partial(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        vs.filesystem.write("e.txt", b"data")
        vs._syscall._write_string(0x20000, "e.txt")
        fd = vs._syscall._sys_open(0x20000, 0)
        assert vs._syscall._sys_read(fd, 0x30000, 10) == 10
        assert bytes(vs.cpu._mem[0x30000:0x30004]) == b"data"
        assert vs._syscall._sys_read(fd, 0x31000, 2) == 2
        assert bytes(vs.cpu._mem[0x31000:0x31002]) == b"da"

    def test_read_unknown_fd_returns_minus_one(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        assert vs._syscall._sys_read(99, 0x30000, 5) == -1

    def test_write_rejects_bad_args(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        assert vs._syscall._sys_write(-1, 0x30000, 5) == -1
        assert vs._syscall._sys_write(1, 0x30000, 0) == -1

    def test_write_to_stdout(self, capsys):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        vs.cpu._write8(0x40000, ord('h'))
        vs.cpu._write8(0x40001, ord('i'))
        assert vs._syscall._sys_write(1, 0x40000, 2) == 2
        assert capsys.readouterr().out == "hi"

    def test_write_to_file(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        vs._syscall._write_string(0x20000, "out.txt")
        fd = vs._syscall._sys_open(0x20000, 2)
        vs.cpu._write8(0x40000, ord('z'))
        assert vs._syscall._sys_write(fd, 0x40000, 1) == 1

    def test_write_unknown_fd_returns_minus_one(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        vs.cpu._write8(0x40000, ord('z'))
        assert vs._syscall._sys_write(77, 0x40000, 1) == -1


class TestSyscallHandlerOpenClose:
    def test_open_without_filesystem_returns_minus_one(self):
        _, _, _, _, h = _standalone_handler()
        assert h._sys_open(0x10000, 0) == -1

    def test_open_empty_name_returns_minus_one(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        vs.cpu._write8(0x20000, 0)
        assert vs._syscall._sys_open(0x20000, 0) == -1

    def test_open_mode0_missing_file_returns_minus_one(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        vs._syscall._write_string(0x20000, "missing.txt")
        assert vs._syscall._sys_open(0x20000, 0) == -1

    def test_open_mode0_existing_file(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        vs.filesystem.write("e.txt", b"data")
        vs._syscall._write_string(0x20000, "e.txt")
        assert vs._syscall._sys_open(0x20000, 0) == 3

    def test_open_mode1_does_not_require_file(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        vs._syscall._write_string(0x20000, "w.txt")
        assert vs._syscall._sys_open(0x20000, 1) == 3

    def test_open_mode2_creates_file(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        vs._syscall._write_string(0x20000, "new.txt")
        fd = vs._syscall._sys_open(0x20000, 2)
        assert fd == 3
        assert vs.filesystem.exists("new.txt")

    def test_close_valid_and_invalid_fd(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        vs.filesystem.write("e.txt", b"data")
        vs._syscall._write_string(0x20000, "e.txt")
        fd = vs._syscall._sys_open(0x20000, 0)
        assert vs._syscall._sys_close(fd) == 0
        assert vs._syscall._sys_close(fd) == -1


class TestSyscallHandlerFork:
    def test_fork_without_current_returns_minus_one(self):
        _, _, _, _, h = _standalone_handler()
        assert h._sys_fork() == -1

    def test_fork_allocation_failure_returns_minus_one(self):
        vs = X86VirtualSystem(memory_size=0x100000)
        assert vs._syscall._sys_fork() == -1

    def test_fork_success(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        parent = vs.scheduler.current
        child_pid = vs._syscall._sys_fork()
        child = vs._ptable.get(child_pid)
        assert child_pid > 0
        assert child.eax == 0
        assert child.stack_base is not None
        assert child.esp == child.stack_base + 0x4000
        assert child_pid in parent.children


class TestSyscallHandlerExec:
    def test_exec_without_filesystem_returns_minus_one(self):
        _, _, _, _, h = _standalone_handler()
        assert h._sys_exec(0x10000) == -1

    def test_exec_missing_file_returns_minus_one(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        vs._syscall._write_string(0x10000, "nope.asm")
        assert vs._syscall._sys_exec(0x10000) == -1

    def test_exec_without_current_returns_minus_one(self):
        _, _, _, _, h = _standalone_handler()
        h._fs = FlatFS(BlockDevice())
        h._fs.write("x.asm", b"[BITS 32]\nhlt\n")
        h._write_string(0x10000, "x.asm")
        assert h._sys_exec(0x10000) == -1

    def test_exec_assemble_failure_returns_minus_one(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        vs.filesystem.write("bad.asm", b"[BITS 32]\ndw 99999999999999999999999\n")
        vs._syscall._write_string(0x10000, "bad.asm")
        assert vs._syscall._sys_exec(0x10000) == -1

    def test_exec_empty_code_returns_minus_one(self, monkeypatch):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        vs.filesystem.write("empty.asm", b"[BITS 32]\nmov\n")
        vs._syscall._write_string(0x10000, "empty.asm")
        monkeypatch.setattr(vm.X86Assembler, "assemble",
                            lambda self, source, org=0: b"")
        assert vs._syscall._sys_exec(0x10000) == -1

    def test_exec_allocation_failure_returns_minus_one(self):
        fs = FlatFS(BlockDevice(num_sectors=1024))
        vs = X86VirtualSystem(memory_size=0x100000, filesystem=fs)
        vs.filesystem.write("big.asm", ("nop\n" * 70000).encode())
        vs._syscall._write_string(0x10000, "big.asm")
        assert vs._syscall._sys_exec(0x10000) == -1

    def test_exec_second_assemble_failure_returns_minus_one(self, monkeypatch):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        vs.filesystem.write("ok.asm", b"[BITS 32]\nhlt\n")
        vs._syscall._write_string(0x10000, "ok.asm")
        orig = vm.X86Assembler.assemble
        calls = {"n": 0}

        def raisy(self, source, org=0):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OverflowError("boom")
            return orig(self, source, org)

        monkeypatch.setattr(vm.X86Assembler, "assemble", raisy)
        assert vs._syscall._sys_exec(0x10000) == -1

    def test_exec_success_resets_process(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        vs.filesystem.write("prog.asm", b"[BITS 32]\nhlt\n")
        vs._syscall._write_string(0x10000, "prog.asm")
        current = vs.scheduler.current
        assert vs._syscall._sys_exec(0x10000) == 0
        assert current.eip != 0
        assert current.eax == 0

    def test_exec_prepends_bits_directive(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        vs.filesystem.write("plain.asm", b"hlt\n")
        vs._syscall._write_string(0x10000, "plain.asm")
        assert vs._syscall._sys_exec(0x10000) == 0


class TestSyscallHandlerWait:
    def test_wait_without_current_returns_minus_one(self):
        _, _, _, _, h = _standalone_handler()
        assert h._sys_wait() == -1

    def test_wait_returns_terminated_child_pid(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        child = vs._ptable.create(name="child")
        child.state = ProcessState.TERMINATED
        vs.scheduler.current.children.append(child.pid)
        assert vs._syscall._sys_wait() == child.pid
        assert vs._ptable.get(child.pid) is None

    def test_wait_blocks_when_no_terminated_child(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        child = vs._ptable.create(name="child")
        vs.scheduler.current.children.append(child.pid)
        assert vs._syscall._sys_wait() == 0
        assert vs.scheduler.current is None


class TestSyscallHandlerBreak:
    def test_brk_without_current_returns_minus_one(self):
        _, _, _, _, h = _standalone_handler()
        assert h._sys_brk(0x100) == -1

    def test_brk_sets_heap_break(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        assert vs._syscall._sys_brk(0x123000) == 0
        assert vs.scheduler.current.heap_break == 0x123000

    def test_sbrk_without_current_returns_minus_one(self):
        _, _, _, _, h = _standalone_handler()
        assert h._sys_sbrk(0x10) == -1

    def test_sbrk_returns_old_break(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        old = vs._syscall._heap_break
        assert vs._syscall._sys_sbrk(0x1000) == old
        assert vs._syscall._heap_break == old + 0x1000


class TestSyscallHandlerKill:
    def test_kill_unknown_pid_returns_minus_one(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        assert vs._syscall._sys_kill(999, 9) == -1

    def test_kill_signal_not_nine_returns_zero(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        pid = vs.spawn("sig", "[BITS 32]\n[ORG 0x100000]\nhlt\n")
        assert vs._syscall._sys_kill(pid, 1) == 0

    def test_kill_self_exits_current(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        current = vs.scheduler.current
        assert vs._syscall._sys_kill(current.pid, 9) == 0

    def test_kill_other_terminates_and_removes(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        pid = vs.spawn("victim", "[BITS 32]\n[ORG 0x100000]\nhlt\n")
        assert vs._syscall._sys_kill(pid, 9) == 0
        assert vs._ptable.get(pid) is None

    def test_kill_role_escalation_blocked(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        kernel_pid = vs.scheduler.current.pid
        user_pid = vs.spawn("user", "[BITS 32]\n[ORG 0x100000]\nhlt\n")
        vs.scheduler.switch_to(vs.cpu, user_pid)
        assert vs._syscall._sys_kill(kernel_pid, 9) == -1

    def test_kill_without_current_terminates(self):
        _, ptable, _, _, h = _standalone_handler()
        victim = ptable.create(name="victim")
        victim.state = ProcessState.RUNNING
        assert h._sys_kill(victim.pid, 9) == 0
        assert ptable.get(victim.pid) is None


class TestSyscallHandlerHeap:
    def test_malloc_rejects_non_positive_size(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        assert vs._syscall._sys_malloc(0) == 0
        assert vs._syscall._sys_malloc(-5) == 0

    def test_malloc_allocates_from_heap(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        addr = vs._syscall._sys_malloc(16)
        assert addr == 0x400000

    def test_malloc_out_of_memory_returns_zero(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        vs._syscall._heap[0x400000] = 0x100000
        assert vs._syscall._sys_malloc(16) == 0

    def test_free_valid_and_invalid(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        addr = vs._syscall._sys_malloc(32)
        assert vs._syscall._sys_free(addr) == 0
        assert vs._syscall._sys_free(addr + 1) == -1


class TestSyscallHandlerReadDir:
    def test_readdir_without_filesystem_returns_zero(self):
        _, _, _, _, h = _standalone_handler()
        assert h._sys_readdir(0x10000, 5) == 0

    def test_readdir_lists_files(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        vs.filesystem.write("a.txt", b"1")
        vs.filesystem.write("b.txt", b"2")
        assert vs._syscall._sys_readdir(0x50000, 5) == 2
        assert vs._syscall._read_string(0x50000) == "a.txt"
        assert vs._syscall._read_string(0x50000 + 32) == "b.txt"


class TestSyscallHandlerUname:
    def test_uname_writes_system_info(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        assert vs._syscall._sys_uname(0x60000) == 0
        assert vs._syscall._read_string(0x60000) == "SloughOS"
        assert vs._syscall._read_string(0x60000 + 65) == "sloughvm"


class TestSyscallHandlerTraining:
    class _FakeBridge:
        def __init__(self, status_payload=None):
            self.status_payload = status_payload or {"status": "completed"}

        def start(self, config_json):
            return 7

        def status(self, job_id):
            return self.status_payload

        def get_result_json(self, job_id):
            return None

    @pytest.mark.parametrize(("status", "expected"), [
        ("running", 0),
        ("completed", 1),
        ("failed", 2),
        ("not_found", -1),
        ("bogus", -1),
    ])
    def test_train_status_mapping(self, monkeypatch, status, expected):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        _install_fake_bridge(monkeypatch, self._FakeBridge({"status": status}))
        assert vs._syscall._sys_train_status(1) == expected

    def test_train_start_returns_job_id(self, monkeypatch):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        _install_fake_bridge(monkeypatch, self._FakeBridge())
        vs._syscall._write_string(0x20000, '{"dataset":"d","epochs":1}')
        assert vs._syscall._sys_train_start(0x20000) == 7

    def test_train_get_result_none_returns_zero(self, monkeypatch):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        _install_fake_bridge(monkeypatch, self._FakeBridge())
        assert vs._syscall._sys_train_get_result(1, 0x30000, 100) == 0

    def test_train_get_result_writes_json(self, monkeypatch):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        bridge = self._FakeBridge()
        bridge.get_result_json = lambda job_id: '{"final_loss": 0.5}'
        _install_fake_bridge(monkeypatch, bridge)
        n = vs._syscall._sys_train_get_result(1, 0x30000, 100)
        assert n == len('{"final_loss": 0.5}')
        assert vs._syscall._read_string(0x30000) == '{"final_loss": 0.5}'

    def test_train_get_result_truncates(self, monkeypatch):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        bridge = self._FakeBridge()
        bridge.get_result_json = lambda job_id: "x" * 100
        _install_fake_bridge(monkeypatch, bridge)
        assert vs._syscall._sys_train_get_result(1, 0x31000, 10) == 9
        assert vs._syscall._read_string(0x31000) == "x" * 9


# ══════════════════════════════════════════════════════════════════════════════
# X86Shell wrapper
# ══════════════════════════════════════════════════════════════════════════════

class TestX86Shell:
    def test_init_defaults(self):
        sh = X86Shell()
        assert sh._source is not None
        assert not sh.running

    def test_start_runs_until_halt(self):
        sh = X86Shell(source=HALT_SRC, memory_size=1024 * 1024)
        sh.start(max_steps=1000)
        sh._thread.join(timeout=2.0)
        assert not sh.running

    def test_run_loop_hits_step_limit(self):
        sh = X86Shell(source=LOOP_SRC, memory_size=1024 * 1024)
        sh.start(max_steps=5)
        sh._thread.join(timeout=2.0)
        assert not sh.running

    def test_stop_flips_running_flag(self):
        sh = X86Shell(source=LOOP_SRC, memory_size=1024 * 1024)
        sh.start(max_steps=1_000_000)
        assert sh.running
        sh.stop()
        assert not sh.running

    def test_type_keys_pushes_to_keyboard_buffer(self):
        sh = X86Shell(memory_size=1024 * 1024)
        sh.type_keys("hi")
        assert len(sh._cpu._kbd_buffer) == 2

    def test_read_screen_reads_vga_text(self):
        sh = X86Shell(memory_size=1024 * 1024)
        sh._cpu._mem[0xB8000] = ord('A')
        assert sh.read_screen(width=10, height=1) == "A"
        sh._cpu._mem[0xB8000] = 8
        assert sh.read_screen(width=10, height=1) == ""
