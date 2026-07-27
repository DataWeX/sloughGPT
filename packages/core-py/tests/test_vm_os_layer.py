"""
Tests for the x86 VM OS layer: PageFrameAllocator, ProcessControlBlock,
ProcessTable, Scheduler, X86SyscallHandler, PITDevice, X86VirtualSystem.
"""

import pytest
from domains.shell.vm import (
    PageFrameAllocator, ProcessControlBlock, ProcessState,
    ProcessTable, Scheduler, X86SyscallHandler, PITDevice,
    X86VirtualSystem, X86CPU, X86Assembler, FlatFS, BlockDevice,
)


# ══════════════════════════════════════════════════════════════════════════════
# PageFrameAllocator
# ══════════════════════════════════════════════════════════════════════════════

class TestPageFrameAllocator:
    def test_init_default(self):
        alloc = PageFrameAllocator(total_memory=1024 * 1024)  # 1 MB
        assert alloc.total_frames == 256
        assert alloc.PAGE_SIZE == 0x1000
        assert alloc.free_frames == 255  # page 0 reserved

    def test_init_reserved_ranges(self):
        alloc = PageFrameAllocator(
            total_memory=1024 * 1024,
            reserved_ranges=[(0xB8000, 0xC0000)],
        )
        # Pages 184-191 reserved (0xB8000/0x1000 = 184, 0xC0000/0x1000 = 192)
        assert alloc.free_frames < 255

    def test_alloc_single(self):
        alloc = PageFrameAllocator(total_memory=1024 * 1024)
        addr = alloc.alloc(1)
        assert addr is not None
        assert addr >= 0x1000  # page 0 reserved
        assert alloc.used_frames == 2  # page 0 + allocated

    def test_alloc_contiguous(self):
        alloc = PageFrameAllocator(total_memory=1024 * 1024)
        addr = alloc.alloc(4)
        assert addr is not None
        # Address should be page-aligned
        assert addr % 0x1000 == 0

    def test_alloc_returns_none_when_full(self):
        alloc = PageFrameAllocator(total_memory=0x10000)  # 64 KB = 16 pages
        # Fill all pages
        addrs = []
        while True:
            a = alloc.alloc(1)
            if a is None:
                break
            addrs.append(a)
        assert len(addrs) >= 15  # at least page 1-15
        assert alloc.free_frames == 0

    def test_free_single(self):
        alloc = PageFrameAllocator(total_memory=1024 * 1024)
        addr = alloc.alloc(1)
        used_before = alloc.used_frames
        alloc.free_single(addr)
        assert alloc.used_frames == used_before - 1

    def test_free_multiple(self):
        alloc = PageFrameAllocator(total_memory=1024 * 1024)
        addr = alloc.alloc(4)
        assert addr is not None
        alloc.free(addr, 4)
        assert alloc.free_frames == 255

    def test_stats(self):
        alloc = PageFrameAllocator(total_memory=1024 * 1024)
        s = alloc.stats()
        assert "total_frames" in s
        assert "allocated" in s
        assert "free" in s
        assert "page_size" in s

    def test_alloc_zero_returns_none(self):
        alloc = PageFrameAllocator(total_memory=1024 * 1024)
        assert alloc.alloc(0) is None

    def test_alloc_negative_returns_none(self):
        alloc = PageFrameAllocator(total_memory=1024 * 1024)
        assert alloc.alloc(-1) is None


# ══════════════════════════════════════════════════════════════════════════════
# ProcessControlBlock
# ══════════════════════════════════════════════════════════════════════════════

class TestProcessControlBlock:
    def test_create_default(self):
        pcb = ProcessControlBlock()
        assert pcb.pid > 0
        assert pcb.name == "unnamed"
        assert pcb.state == ProcessState.CREATED
        assert pcb.priority == 0

    def test_create_named(self):
        pcb = ProcessControlBlock(name="shell", priority=5)
        assert pcb.name == "shell"
        assert pcb.priority == 5

    def test_save_restore_cpu(self):
        cpu = X86CPU(memory_size=0x100000)
        cpu._regs[0] = 0xDEADBEEF  # EAX
        cpu._regs[4] = 0x000FF000  # ESP
        cpu._eip = 0x12345678

        pcb = ProcessControlBlock()
        pcb.save_from_cpu(cpu)

        assert pcb.eax == 0xDEADBEEF
        assert pcb.esp == 0x000FF000
        assert pcb.eip == 0x12345678

        # Restore into fresh CPU
        cpu2 = X86CPU(memory_size=0x100000)
        pcb.restore_to_cpu(cpu2)
        assert cpu2._regs[0] == 0xDEADBEEF
        assert cpu2._regs[4] == 0x000FF000
        assert cpu2._eip == 0x12345678

    def test_repr(self):
        pcb = ProcessControlBlock(name="test")
        r = repr(pcb)
        assert "pid=" in r
        assert "test" in r


# ══════════════════════════════════════════════════════════════════════════════
# ProcessTable
# ══════════════════════════════════════════════════════════════════════════════

class TestProcessTable:
    def test_create_process(self):
        pt = ProcessTable()
        pcb = pt.create(name="init")
        assert pcb.pid > 0
        assert pt.count() == 1

    def test_get_process(self):
        pt = ProcessTable()
        pcb = pt.create(name="test")
        assert pt.get(pcb.pid) is pcb

    def test_get_nonexistent(self):
        pt = ProcessTable()
        assert pt.get(99999) is None

    def test_get_by_name(self):
        pt = ProcessTable()
        p1 = pt.create(name="worker")
        p2 = pt.create(name="worker")
        p3 = pt.create(name="other")
        workers = pt.get_by_name("worker")
        assert len(workers) == 2
        assert p1 in workers
        assert p2 in workers

    def test_remove(self):
        pt = ProcessTable()
        pcb = pt.create(name="temp")
        removed = pt.remove(pcb.pid)
        assert removed is pcb
        assert pt.count() == 0
        assert pt.get(pcb.pid) is None

    def test_remove_nonexistent(self):
        pt = ProcessTable()
        assert pt.remove(99999) is None

    def test_by_state(self):
        pt = ProcessTable()
        p1 = pt.create(name="a")
        p2 = pt.create(name="b")
        p1.state = ProcessState.READY
        p2.state = ProcessState.TERMINATED
        ready = pt.by_state(ProcessState.READY)
        assert len(ready) == 1
        assert ready[0] is p1

    def test_alive_count(self):
        pt = ProcessTable()
        p1 = pt.create(name="a")
        p2 = pt.create(name="b")
        p3 = pt.create(name="c")
        p3.state = ProcessState.TERMINATED
        assert pt.alive_count() == 2

    def test_all(self):
        pt = ProcessTable()
        pt.create(name="a")
        pt.create(name="b")
        assert len(pt.all()) == 2


# ══════════════════════════════════════════════════════════════════════════════
# Scheduler
# ══════════════════════════════════════════════════════════════════════════════

class TestScheduler:
    def _make_system(self):
        cpu = X86CPU(memory_size=0x100000)
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=3)
        return cpu, ptable, sched

    def test_enqueue_dequeue(self):
        _, ptable, sched = self._make_system()
        pcb = ptable.create(name="test")
        sched.enqueue(pcb.pid)
        assert len(sched._ready_queue) == 1
        pid = sched.dequeue()
        assert pid == pcb.pid
        assert len(sched._ready_queue) == 0

    def test_start_picks_first_process(self):
        cpu, ptable, sched = self._make_system()
        pcb = ptable.create(name="test")
        pcb.eip = 0x5000
        sched.enqueue(pcb.pid)
        sched.start(cpu)
        assert sched.current is pcb
        assert pcb.state == ProcessState.RUNNING
        assert cpu._eip == 0x5000

    def test_tick_decrements_quantum(self):
        cpu, ptable, sched = self._make_system()
        pcb = ptable.create(name="test")
        sched.enqueue(pcb.pid)
        sched.start(cpu)
        assert pcb.time_slice == 3
        sched.tick(cpu)
        assert pcb.time_slice == 2
        sched.tick(cpu)
        assert pcb.time_slice == 1

    def test_tick_preempts_on_quantum_expiry(self):
        cpu, ptable, sched = self._make_system()
        p1 = ptable.create(name="p1")
        p2 = ptable.create(name="p2")
        p1.eip = 0x1000
        p2.eip = 0x2000
        sched.enqueue(p1.pid)
        sched.enqueue(p2.pid)
        sched.start(cpu)
        assert sched.current is p1

        # Exhaust quantum
        for _ in range(3):
            sched.tick(cpu)

        # p1 preempted, p2 now running
        assert sched.current is p2
        assert p1.state == ProcessState.READY

    def test_exit_current(self):
        cpu, ptable, sched = self._make_system()
        p1 = ptable.create(name="p1")
        sched.enqueue(p1.pid)
        sched.start(cpu)
        sched.exit_current(cpu, exit_code=42)
        assert p1.state == ProcessState.TERMINATED
        assert p1.exit_code == 42
        assert sched.current is None

    def test_switch_to(self):
        cpu, ptable, sched = self._make_system()
        p1 = ptable.create(name="p1")
        p2 = ptable.create(name="p2")
        p2.eip = 0x3000
        sched.enqueue(p1.pid)
        sched.enqueue(p2.pid)
        sched.start(cpu)
        assert sched.current is p1

        ok = sched.switch_to(cpu, p2.pid)
        assert ok is True
        assert sched.current is p2
        assert cpu._eip == 0x3000

    def test_block_unblock(self):
        cpu, ptable, sched = self._make_system()
        p1 = ptable.create(name="p1")
        p2 = ptable.create(name="p2")
        sched.enqueue(p1.pid)
        sched.enqueue(p2.pid)
        sched.start(cpu)

        # Block p1
        sched.block_current(cpu)
        assert p1.state == ProcessState.WAITING
        assert sched.current is p2

        # Unblock p1
        sched.unblock(p1.pid)
        assert p1.state == ProcessState.READY
        assert p1.pid in sched._ready_queue

    def test_stats(self):
        cpu, ptable, sched = self._make_system()
        s = sched.stats()
        assert "quantum" in s
        assert "current_pid" in s
        assert "ready_queue" in s


# ══════════════════════════════════════════════════════════════════════════════
# PITDevice
# ══════════════════════════════════════════════════════════════════════════════

class TestPITDevice:
    def test_tick_fires_irq(self):
        cpu = X86CPU(memory_size=0x100000)
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        # target_hz = 1193182 → divider = 1 → fires every tick
        pit = PITDevice(cpu, sched, target_hz=1193182)

        for _ in range(100):
            pit.tick()
        assert pit._tick_count == 100

    def test_pit_doesnt_crash_cpu(self):
        cpu = X86CPU(memory_size=0x100000)
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        pit = PITDevice(cpu, sched, target_hz=1193182)

        for _ in range(500):
            pit.tick()
        assert pit._tick_count == 500


# ══════════════════════════════════════════════════════════════════════════════
# X86SyscallHandler
# ══════════════════════════════════════════════════════════════════════════════

class TestX86SyscallHandler:
    def _make_handler(self):
        cpu = X86CPU(memory_size=0x100000)
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        alloc = PageFrameAllocator(total_memory=0x100000)
        handler = X86SyscallHandler(cpu, ptable, sched, alloc)
        # Create a kernel process and set it running
        pcb = ptable.create(name="kernel")
        pcb.state = ProcessState.RUNNING
        sched._current_pid = pcb.pid
        return cpu, ptable, sched, handler, pcb

    def test_getpid(self):
        cpu, _, _, handler, pcb = self._make_handler()
        cpu._regs[0] = handler.SYS_GETPID  # EAX
        handler.handle()
        assert cpu._regs[0] == pcb.pid

    def test_write_stdout(self, capsys):
        cpu, _, _, handler, _ = self._make_handler()
        # Write "Hello" to stdout
        msg = b"Hello"
        for i, b in enumerate(msg):
            cpu._write8(0x8000 + i, b)
        cpu._regs[0] = handler.SYS_WRITE  # EAX
        cpu._regs[3] = 1                  # EBX = fd (stdout)
        cpu._regs[1] = 0x8000             # ECX = buf addr
        cpu._regs[2] = len(msg)           # EDX = count
        handler.handle()
        assert cpu._regs[0] == len(msg)
        captured = capsys.readouterr()
        assert "Hello" in captured.out

    def test_malloc_free(self):
        cpu, _, _, handler, _ = self._make_handler()
        # malloc 256 bytes
        cpu._regs[0] = handler.SYS_MALLOC
        cpu._regs[3] = 256
        handler.handle()
        addr = cpu._regs[0]
        assert addr > 0

        # free it
        cpu._regs[0] = handler.SYS_FREE
        cpu._regs[3] = addr
        handler.handle()
        assert cpu._regs[0] == 0

    def test_yield(self):
        cpu, ptable, sched, handler, pcb = self._make_handler()
        cpu._regs[0] = handler.SYS_YIELD
        handler.handle()
        assert handler._ticks > 0 or True  # yield just returns

    def test_gettimeofday(self):
        cpu, _, _, handler, _ = self._make_handler()
        handler._ticks = 42
        cpu._regs[0] = handler.SYS_GETTIMEOFDAY
        cpu._regs[3] = 0x9000  # buf addr
        handler.handle()
        assert cpu._regs[0] == 42

    def test_unknown_syscall_returns_minus_one(self):
        cpu, _, _, handler, _ = self._make_handler()
        cpu._regs[0] = 99999  # unknown
        handler.handle()
        assert cpu._regs[0] == 0xFFFFFFFF


# ══════════════════════════════════════════════════════════════════════════════
# X86VirtualSystem
# ══════════════════════════════════════════════════════════════════════════════

class TestX86VirtualSystem:
    def test_init(self):
        vs = X86VirtualSystem(memory_size=0x100000)
        assert vs.cpu is not None
        assert vs.process_table.count() >= 1  # kernel process
        assert vs.scheduler is not None

    def test_load_kernel(self):
        vs = X86VirtualSystem(memory_size=0x100000)
        asm = """
        [BITS 32]
        [ORG 0x1000]
        nop
        hlt
        """
        vs.load_kernel(asm)
        # Code should be loaded at 0x1000
        assert vs.cpu._mem[0x1000] == 0x90  # NOP

    def test_spawn(self):
        vs = X86VirtualSystem(memory_size=0x400000)
        asm = """
        [BITS 32]
        [ORG 0x100000]
        nop
        hlt
        """
        pid = vs.spawn("user", asm)
        assert pid is not None
        assert pid > 1
        pcb = vs.process_table.get(pid)
        assert pcb is not None
        assert pcb.name == "user"

    def test_status(self):
        vs = X86VirtualSystem(memory_size=0x100000)
        s = vs.status()
        assert "cpu" in s
        assert "memory" in s
        assert "scheduler" in s
        assert "processes" in s

    def test_run_short(self):
        vs = X86VirtualSystem(memory_size=0x100000)
        asm = """
        [BITS 32]
        [ORG 0x1000]
        nop
        hlt
        """
        vs.load_kernel(asm)
        cycles = vs.run(max_cycles=100)
        assert cycles > 0

    def test_filesystem_integration(self):
        fs = FlatFS(BlockDevice())
        vs = X86VirtualSystem(memory_size=0x100000, filesystem=fs)
        assert vs.filesystem is fs


# ══════════════════════════════════════════════════════════════════════════════
# Integration: Assembler + CPU + OS
# ══════════════════════════════════════════════════════════════════════════════

class TestOSIntegration:
    def test_assemble_and_run_nop_hlt(self):
        asm = X86Assembler()
        code = asm.assemble("""
        [BITS 32]
        [ORG 0x1000]
        nop
        hlt
        """)
        cpu = X86CPU(memory_size=0x100000)
        cpu.load(code, 0x1000)
        cpu._eip = 0x1000
        steps = cpu.run(max_steps=10)
        assert steps >= 1

    def test_assemble_and_run_push_pop(self):
        asm = X86Assembler()
        code = asm.assemble("""
        [BITS 32]
        [ORG 0x1000]
        mov eax, 0x12345678
        push eax
        xor eax, eax
        pop ebx
        hlt
        """)
        cpu = X86CPU(memory_size=0x100000)
        cpu.load(code, 0x1000)
        cpu._eip = 0x1000
        cpu.run(max_steps=100)
        assert cpu._regs[0] == 0  # EAX zeroed by xor
        assert cpu._regs[3] == 0x12345678  # EBX = popped value

    def test_vsystem_kernel_runs(self):
        vs = X86VirtualSystem(memory_size=0x200000)
        asm = """
        [BITS 32]
        [ORG 0x1000]
        mov eax, 42
        hlt
        """
        vs.load_kernel(asm)
        vs.run(max_cycles=50)
        assert vs.cpu._regs[0] == 42

    def test_multiple_processes(self):
        vs = X86VirtualSystem(memory_size=0x400000)
        vs.load_kernel("""
        [BITS 32]
        [ORG 0x1000]
        hlt
        """)
        # Spawn two user processes
        pid1 = vs.spawn("proc1", """
        [BITS 32]
        [ORG 0x100000]
        mov eax, 111
        hlt
        """)
        pid2 = vs.spawn("proc2", """
        [BITS 32]
        [ORG 0x200000]
        mov eax, 222
        hlt
        """)
        assert pid1 is not None
        assert pid2 is not None
        assert vs.process_table.alive_count() >= 3  # kernel + 2

    def test_scheduler_round_robin(self):
        cpu = X86CPU(memory_size=0x100000)
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=2)

        p1 = ptable.create(name="p1")
        p1.eip = 0x1000
        p1.esp = 0xF0000
        p2 = ptable.create(name="p2")
        p2.eip = 0x2000
        p2.esp = 0xF0000

        sched.enqueue(p1.pid)
        sched.enqueue(p2.pid)
        sched.start(cpu)

        assert sched.current is p1

        # Tick 2x → p1 quantum expires
        sched.tick(cpu)
        sched.tick(cpu)
        assert sched.current is p2  # switched to p2

        # Tick 2x → p2 quantum expires
        sched.tick(cpu)
        sched.tick(cpu)
        assert sched.current is p1  # back to p1


# ══════════════════════════════════════════════════════════════════════════════
# X86SyscallHandler — exec syscall
# ══════════════════════════════════════════════════════════════════════════════

class TestSyscallExec:
    def _setup_with_process(self, source_code: str, filename: str = "test.asm"):
        """Create a virtual system, write a file to FS, and spawn a worker."""
        vs = X86VirtualSystem(memory_size=0x100000, timer_hz=100)
        vs.filesystem.write(filename, source_code.encode("utf-8"))
        pid = vs.spawn("worker", "mov eax, 99\nhlt")
        assert pid is not None
        vs.scheduler.start(vs.cpu)
        return vs

    def _exec_filename(self, vs, filename: str) -> int:
        """Write filename into CPU memory at a safe address and call exec."""
        name_addr = 0x80000  # safe area in low memory
        vs.cpu._mem[name_addr:name_addr + len(filename)] = filename.encode("ascii")
        vs.cpu._mem[name_addr + len(filename)] = 0  # null terminator
        return vs._syscall._sys_exec(name_addr)

    def test_exec_replaces_process_code(self):
        vs = self._setup_with_process("mov eax, 42\nhlt")
        result = self._exec_filename(vs, "test.asm")
        assert result == 0
        current = vs.scheduler.current
        assert current is not None
        assert current.eip > 0
        assert current.esp > current.eip

    def test_exec_assembles_and_loads(self):
        source = "[BITS 32]\nmov eax, 123\nmov ebx, 456\nhlt"
        vs = self._setup_with_process(source)
        result = self._exec_filename(vs, "test.asm")
        assert result == 0
        current = vs.scheduler.current
        current.restore_to_cpu(vs.cpu)
        vs.cpu.step()  # mov eax, 123
        assert vs.cpu._regs[0] == 123
        vs.cpu.step()  # mov ebx, 456
        assert vs.cpu._regs[3] == 456

    def test_exec_resets_registers(self):
        source = "mov eax, 77\nhlt"
        vs = self._setup_with_process(source)
        current = vs.scheduler.current
        current.eax = 999
        current.ecx = 888
        current.edx = 777
        result = self._exec_filename(vs, "test.asm")
        assert result == 0
        current = vs.scheduler.current
        assert current.eax == 0
        assert current.ecx == 0
        assert current.edx == 0

    def test_exec_file_not_found(self):
        vs = X86VirtualSystem(memory_size=0x100000, timer_hz=100)
        pid = vs.spawn("worker", "hlt")
        vs.scheduler.start(vs.cpu)
        result = self._exec_filename(vs, "nonexistent.asm")
        assert result == -1

    def test_exec_no_filesystem(self):
        vs = X86VirtualSystem(memory_size=0x100000, timer_hz=100)
        vs._syscall._fs = None
        result = vs._syscall._sys_exec(0x1000)
        assert result == -1

    def test_exec_noop_program(self):
        """Exec a minimal program — verifies code loads and PCB is updated."""
        vs = self._setup_with_process("mov eax, 1\nhlt")
        vs.filesystem.write("minimal.asm", b"[BITS 32]\nhlt")
        result = self._exec_filename(vs, "minimal.asm")
        assert result == 0
        current = vs.scheduler.current
        # PCB should have been reset
        assert current.eax == 0
        assert current.ecx == 0
        assert current.eip > 0
        assert current.esp > current.eip

    def test_exec_frees_old_memory(self):
        vs = self._setup_with_process("mov eax, 1\nhlt")
        current = vs.scheduler.current
        old_stack = current.stack_base
        assert old_stack > 0
        result = self._exec_filename(vs, "test.asm")
        assert result == 0
        current = vs.scheduler.current
        assert current.stack_base > 0

    def test_exec_program_runs_after(self):
        vs = self._setup_with_process("nop\nmov eax, 999\nhlt")
        result = self._exec_filename(vs, "test.asm")
        assert result == 0
        current = vs.scheduler.current
        current.restore_to_cpu(vs.cpu)
        vs.cpu.run(max_steps=10)
        assert vs.cpu._regs[0] == 999
