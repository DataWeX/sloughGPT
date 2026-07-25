"""Tests for AI-Native Kernel primitives."""

import pytest
import time
import numpy as np
from domains.shell.kernel_process import Process, ProcessState, Priority, TensorRef
from domains.shell.kernel_memory import TensorMemory, MemoryBlock
from domains.shell.kernel_scheduler import Scheduler
from domains.shell.kernel_interrupts import InterruptManager, InterruptVector, InterruptType, Interrupt
from domains.shell.kernel_devices import DeviceDriver, DeviceManager, DeviceType, DeviceState
from domains.shell.kernel_syscall import SyscallTable, SyscallNumber, SyscallResult, SYSCALLS
from domains.shell.kernel_core import Kernel, get_kernel, reset_kernel


# ── Process tests ────────────────────────────────────────────────────────────

class TestProcess:
    def test_create(self):
        p = Process(pid=1, name="test")
        assert p.pid == 1
        assert p.name == "test"
        assert p.state == ProcessState.CREATED
        assert p.priority == Priority.NORMAL

    def test_transition(self):
        p = Process(pid=1, name="test")
        p.transition(ProcessState.READY)
        assert p.state == ProcessState.READY
        p.transition(ProcessState.RUNNING)
        assert p.state == ProcessState.RUNNING
        assert p.started_at is not None

    def test_transition_to_done(self):
        p = Process(pid=1, name="test")
        p.transition(ProcessState.RUNNING)
        p.transition(ProcessState.STOPPED)
        assert p.state == ProcessState.STOPPED
        assert p.finished_at is not None

    def test_uptime(self):
        p = Process(pid=1, name="test")
        p.transition(ProcessState.RUNNING)
        time.sleep(0.01)
        assert p.uptime > 0

    def test_is_active(self):
        p = Process(pid=1, name="test")
        assert p.is_active  # CREATED is active
        p.transition(ProcessState.STOPPED)
        assert not p.is_active

    def test_is_done(self):
        p = Process(pid=1, name="test")
        assert not p.is_done
        p.transition(ProcessState.STOPPED)
        assert p.is_done

    def test_tensor_acquire_release(self):
        p = Process(pid=1, name="test")
        ref = TensorRef(block_id=1, shape=(10,), dtype="float32", size_bytes=40, owner_pid=1)
        p.acquire_tensor(ref)
        assert len(p.tensors) == 1
        assert p.memory_bytes == 40
        released = p.release_tensor(1)
        assert released is ref
        assert len(p.tensors) == 0
        assert p.memory_bytes == 0

    def test_status_line(self):
        p = Process(pid=1, name="test")
        line = p.status_line()
        assert "1" in line
        assert "test" in line


# ── Memory tests ─────────────────────────────────────────────────────────────

class TestTensorMemory:
    def test_allocate(self):
        mem = TensorMemory(capacity_bytes=1024 * 1024)
        block = mem.allocate((10, 10), "float32")
        assert block.shape == (10, 10)
        assert block.dtype == "float32"
        assert block.size_bytes == 400
        assert block.data is not None
        assert block.data.shape == (10, 10)

    def test_allocate_different_dtypes(self):
        mem = TensorMemory(capacity_bytes=1024 * 1024)
        f16 = mem.allocate((5,), "float16")
        assert f16.size_bytes == 10
        i64 = mem.allocate((5,), "int64")
        assert i64.size_bytes == 40

    def test_read_write(self):
        mem = TensorMemory(capacity_bytes=1024 * 1024)
        block = mem.allocate((3, 3), "float32")
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        assert mem.write(block.block_id, data)
        read_back = mem.read(block.block_id)
        np.testing.assert_array_equal(read_back, data)

    def test_write_shape_mismatch(self):
        mem = TensorMemory(capacity_bytes=1024 * 1024)
        block = mem.allocate((3, 3), "float32")
        data = np.array([1, 2, 3], dtype=np.float32)
        assert not mem.write(block.block_id, data)

    def test_free_block(self):
        mem = TensorMemory(capacity_bytes=1024 * 1024)
        block = mem.allocate((10,), "float32")
        used_before = mem.used
        assert mem.free_block(block.block_id)
        assert mem.used < used_before
        assert mem.get(block.block_id) is None

    def test_free_pid(self):
        mem = TensorMemory(capacity_bytes=1024 * 1024)
        mem.allocate((10,), "float32", owner_pid=1)
        mem.allocate((20,), "float32", owner_pid=1)
        mem.allocate((30,), "float32", owner_pid=2)
        freed = mem.free_pid(1)
        assert freed == 2

    def test_capacity_exceeded(self):
        mem = TensorMemory(capacity_bytes=100)
        mem.allocate((10,), "float32")  # 40 bytes
        with pytest.raises(MemoryError):
            mem.allocate((100,), "float32")  # 400 bytes

    def test_stats(self):
        mem = TensorMemory(capacity_bytes=1024)
        mem.allocate((10,), "float32")
        stats = mem.stats()
        assert stats["active_blocks"] == 1
        assert stats["used_bytes"] == 40

    def test_defragment(self):
        mem = TensorMemory(capacity_bytes=1024 * 1024)
        b1 = mem.allocate((10,), "float32")
        b2 = mem.allocate((10,), "float32")
        mem.free_block(b1.block_id)
        mem.free_block(b2.block_id)
        reclaimed = mem.defragment()
        assert reclaimed == 2


# ── Scheduler tests ──────────────────────────────────────────────────────────

class TestScheduler:
    def test_add_process(self):
        sched = Scheduler()
        p = Process(pid=1, name="test")
        sched.add(p)
        assert sched.process_count == 1
        assert p.state == ProcessState.READY

    def test_tick_picks_ready(self):
        sched = Scheduler()
        p = Process(pid=1, name="test")
        sched.add(p)
        result = sched.tick()
        assert result is not None
        assert result.pid == 1
        assert result.state == ProcessState.RUNNING

    def test_priority_ordering(self):
        sched = Scheduler()
        low = Process(pid=1, name="low", priority=Priority.LOW)
        high = Process(pid=2, name="high", priority=Priority.HIGH)
        sched.add(low)
        sched.add(high)
        result = sched.tick()
        assert result.pid == 2  # high priority first

    def test_complete_process(self):
        sched = Scheduler()
        p = Process(pid=1, name="test")
        sched.add(p)
        sched.tick()
        sched.complete(1, result="done")
        assert p.state == ProcessState.ZOMBIE
        assert p.result == "done"

    def test_reap_zombie(self):
        sched = Scheduler()
        p = Process(pid=1, name="test")
        sched.add(p)
        sched.tick()
        sched.complete(1)
        reapplied = sched.reap(1)
        assert reapplied is not None
        assert sched.process_count == 0

    def test_wait_wake(self):
        sched = Scheduler()
        blocker = Process(pid=1, name="blocker")
        waiter = Process(pid=2, name="waiter")
        sched.add(blocker)
        sched.add(waiter)
        sched.tick()  # runs blocker
        sched.wait_for(2, 1)
        assert waiter.state == ProcessState.WAITING
        sched.wake(2)
        assert waiter.state == ProcessState.READY

    def test_stats(self):
        sched = Scheduler()
        sched.add(Process(pid=1, name="a"))
        sched.add(Process(pid=2, name="b"))
        stats = sched.stats()
        assert stats["total_processes"] == 2


# ── Interrupt tests ──────────────────────────────────────────────────────────

class TestInterruptManager:
    def test_fire_and_handle(self):
        mgr = InterruptManager()
        handled = []
        mgr.on_process_done(lambda i: handled.append(i.source_pid))
        mgr.signal_process_done(42, result="ok")
        assert 42 in handled

    def test_mask_unmask(self):
        mgr = InterruptManager()
        handled = []
        mgr.vector.register(InterruptType.INFERENCE_DONE, lambda i: handled.append(True))
        mgr.vector.mask(InterruptType.INFERENCE_DONE)
        mgr.vector.fire(Interrupt(vector=InterruptType.INFERENCE_DONE))
        assert len(handled) == 0
        mgr.vector.unmask(InterruptType.INFERENCE_DONE)
        mgr.vector.fire(Interrupt(vector=InterruptType.INFERENCE_DONE))
        assert len(handled) == 1

    def test_enqueue_dequeue(self):
        iv = InterruptVector()
        low = Interrupt(vector=InterruptType.CUSTOM, priority=10)
        high = Interrupt(vector=InterruptType.CUSTOM, priority=1)
        iv.enqueue(low)
        iv.enqueue(high)
        first = iv.dequeue()
        assert first.priority == 1

    def test_process_pending(self):
        iv = InterruptVector()
        count = [0]
        iv.register(InterruptType.CUSTOM, lambda i: count.__setitem__(0, count[0] + 1))
        iv.enqueue(Interrupt(vector=InterruptType.CUSTOM))
        iv.enqueue(Interrupt(vector=InterruptType.CUSTOM))
        handled = iv.process_pending()
        assert handled == 2

    def test_stats(self):
        mgr = InterruptManager()
        mgr.signal_process_done(1)
        stats = mgr.vector.stats()
        assert stats["total_fired"] == 1


# ── Device tests ─────────────────────────────────────────────────────────────

class TestDeviceDriver:
    def test_register_open_close(self):
        class TestDev(DeviceDriver):
            def __init__(self):
                super().__init__("test_dev", DeviceType.CUSTOM)
            def read(self, offset=0, size=-1):
                return "hello"
            def write(self, data):
                return True

        mgr = DeviceManager()
        dev = TestDev()
        assert mgr.register(dev)
        assert dev.state == DeviceState.CLOSED

        handle = mgr.open("test_dev")
        assert handle is not None
        assert handle.fd > 0
        assert dev.state == DeviceState.OPEN

        data = mgr.read(handle.fd)
        assert data == "hello"

        assert mgr.write(handle.fd, "world")
        assert mgr.close(handle.fd)
        assert dev.state == DeviceState.CLOSED

    def test_unregister(self):
        class TestDev(DeviceDriver):
            def __init__(self):
                super().__init__("test2", DeviceType.CUSTOM)

        mgr = DeviceManager()
        dev = TestDev()
        mgr.register(dev)
        assert mgr.unregister("test2")
        assert mgr.get("test2") is None

    def test_list_devices(self):
        class Dev1(DeviceDriver):
            def __init__(self):
                super().__init__("d1", DeviceType.INFERENCE)
        class Dev2(DeviceDriver):
            def __init__(self):
                super().__init__("d2", DeviceType.TRAINING)

        mgr = DeviceManager()
        mgr.register(Dev1())
        mgr.register(Dev2())
        devs = mgr.list_devices()
        assert len(devs) == 2


# ── Syscall tests ────────────────────────────────────────────────────────────

class TestSyscallTable:
    def test_dispatch(self):
        table = SyscallTable()
        table.register(SyscallNumber.UPTIME, lambda k: SyscallResult.ok(k.uptime))
        kernel = Kernel()
        kernel.boot()
        result = table.dispatch(kernel, SyscallNumber.UPTIME)
        assert result.success
        assert isinstance(result.value, float)
        assert result.value >= 0
        kernel.shutdown()

    def test_unknown_syscall(self):
        table = SyscallTable()
        kernel = Kernel()
        kernel.boot()
        result = table.dispatch(kernel, 0xFF)
        assert not result.success
        kernel.shutdown()

    def test_exception_handling(self):
        table = SyscallTable()
        def bad_handler(k):
            raise ValueError("boom")
        table.register(SyscallNumber.UPTIME, bad_handler)
        kernel = Kernel()
        kernel.boot()
        result = table.dispatch(kernel, SyscallNumber.UPTIME)
        assert not result.success
        assert "boom" in result.error
        kernel.shutdown()


# ── Full Kernel tests ────────────────────────────────────────────────────────

class TestKernel:
    def setup_method(self):
        reset_kernel()

    def teardown_method(self):
        reset_kernel()

    def test_boot_shutdown(self):
        k = Kernel()
        log = k.boot()
        assert "booted" in log.lower()
        assert k.running
        log = k.shutdown()
        assert "shut down" in log.lower()
        assert not k.running

    def test_singleton(self):
        k1 = get_kernel()
        k2 = get_kernel()
        assert k1 is k2

    def test_create_process(self):
        k = Kernel()
        k.boot()
        pid = k.create_process("worker")
        assert pid > 0
        proc = k.scheduler.get(pid)
        assert proc.name == "worker"
        k.shutdown()

    def test_alloc_tensor(self):
        k = Kernel()
        k.boot()
        info = k.alloc_tensor((100, 100), "float32")
        assert info is not None
        assert info["block_id"] > 0
        assert info["size_bytes"] == 40000
        k.shutdown()

    def test_free_tensor(self):
        k = Kernel()
        k.boot()
        info = k.alloc_tensor((10, 10), "float32")
        assert k.free_tensor(info["block_id"])
        assert k.memory.get(info["block_id"]) is None
        k.shutdown()

    def test_open_close_device(self):
        k = Kernel()
        k.boot()
        fd = k.open_device("null")
        assert fd is not None
        assert k.close_device(fd)
        k.shutdown()

    def test_tick(self):
        k = Kernel()
        k.boot()
        k.create_process("a")
        k.create_process("b")
        result = k.tick()
        assert result is not None
        assert result["current_pid"] is not None
        k.shutdown()

    def test_run(self):
        k = Kernel()
        k.boot()
        k.create_process("a")
        results = k.run(max_ticks=10)
        assert len(results) > 0
        k.shutdown()

    def test_syscall_tensor_alloc(self):
        k = Kernel()
        k.boot()
        from domains.shell.kernel_syscall import SyscallNumber
        result = k.syscall(SyscallNumber.TENSOR_ALLOC, (5, 5), "float64")
        assert result.success
        assert result.value["size_bytes"] == 200
        k.shutdown()

    def test_info(self):
        k = Kernel()
        k.boot()
        info = k.info()
        assert "uptime_s" in info
        assert "memory" in info
        assert "devices" in info
        assert "syscalls" in info
        k.shutdown()

    def test_process_dependency(self):
        k = Kernel()
        k.boot()
        pid1 = k.create_process("dep1")
        pid2 = k.create_process("dep2", depends_on=[pid1])
        # Tick — dep2 should not run because dep1 isn't done
        k.tick()
        proc2 = k.scheduler.get(pid2)
        assert proc2.state != ProcessState.RUNNING
        # Complete dep1
        k.scheduler.complete(pid1)
        # Now dep2 should be able to run
        k.tick()
        k.shutdown()

    def test_run_program(self):
        k = Kernel()
        k.boot()
        result = k.run_program("LOAD_CONST R0, 42\nPRINT R0\nHALT")
        assert result["output"] == ["42"]
        assert result["steps"] == 3
        assert result["regs"]["R0"] == 42
        k.shutdown()

    def test_run_program_with_trace(self):
        k = Kernel()
        k.boot()
        result = k.run_program("MOV R0, 10\nMOV R1, 20\nIADD R2, R0, R1\nPRINT R2\nHALT", trace=True)
        assert len(result["trace"]) > 0
        assert result["output"] == ["30"]
        k.shutdown()

    def test_spawn_shell(self):
        k = Kernel()
        k.boot()
        proc = k.spawn_shell()
        assert proc.name == "shell"
        assert proc.state.name == "READY"
        assert proc.pid == 2
        procs = k.list_processes()
        assert len(procs) == 2
        k.shutdown()
