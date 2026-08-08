"""Tests for kernel interrupts, memory, and scheduler primitives."""

from __future__ import annotations

import time

import numpy as np

from domains.shell.kernel_interrupts import (
    Interrupt,
    InterruptManager,
    InterruptType,
    InterruptVector,
)
from domains.shell.kernel_memory import MemoryBlock, TensorMemory
from domains.shell.kernel_process import Process, ProcessState, Priority
from domains.shell import kernel_scheduler
from domains.shell.kernel_scheduler import Scheduler


class TestInterruptType:
    def test_values(self):
        assert InterruptType.TIMER == 0
        assert InterruptType.INFERENCE_DONE == 1
        assert InterruptType.TRAINING_STEP == 2
        assert InterruptType.DATA_READY == 3
        assert InterruptType.GRADIENT_UPDATE == 4
        assert InterruptType.DEVICE_ERROR == 5
        assert InterruptType.MEMORY_FULL == 6
        assert InterruptType.PROCESS_DONE == 7
        assert InterruptType.USER_INPUT == 8
        assert InterruptType.NETWORK_IO == 9
        assert InterruptType.CUSTOM == 10


class TestInterrupt:
    def test_defaults(self):
        i = Interrupt(vector=InterruptType.TIMER)
        assert i.source_pid is None
        assert i.data is None
        assert i.priority == 0

    def test_fields(self):
        i = Interrupt(InterruptType.CUSTOM, source_pid=5, data={"x": 1}, priority=3)
        assert i.source_pid == 5
        assert i.data == {"x": 1}
        assert i.priority == 3


class TestInterruptVector:
    def test_register_and_fire(self):
        v = InterruptVector()
        calls = []
        v.register(InterruptType.TIMER, lambda i: calls.append(i.data))
        result = v.fire(Interrupt(InterruptType.TIMER, data=42))
        assert result is True
        assert calls == [42]

    def test_unregister(self):
        v = InterruptVector()
        v.register(InterruptType.TIMER, lambda i: None)
        v.unregister(InterruptType.TIMER)
        assert v.fire(Interrupt(InterruptType.TIMER)) is False

    def test_mask_unmask(self):
        v = InterruptVector()
        v.register(InterruptType.TIMER, lambda i: None)
        v.mask(InterruptType.TIMER)
        assert v.is_masked(InterruptType.TIMER)
        assert v.fire(Interrupt(InterruptType.TIMER)) is False
        v.unmask(InterruptType.TIMER)
        assert not v.is_masked(InterruptType.TIMER)
        assert v.fire(Interrupt(InterruptType.TIMER)) is True

    def test_fire_no_handler_records_history(self):
        v = InterruptVector()
        assert v.fire(Interrupt(InterruptType.TIMER)) is False
        assert len(v.history) == 1

    def test_fire_handler_raises(self):
        v = InterruptVector()
        v.register(InterruptType.TIMER, lambda i: (_ for _ in ()).throw(RuntimeError("boom")))
        assert v.fire(Interrupt(InterruptType.TIMER)) is False

    def test_history_truncated(self):
        v = InterruptVector()
        v._max_history = 2
        for _ in range(5):
            v.fire(Interrupt(InterruptType.TIMER))
        assert len(v.history) == 2

    def test_enqueue_dequeue_priority_order(self):
        v = InterruptVector()
        v.enqueue(Interrupt(InterruptType.TIMER, priority=2))
        v.enqueue(Interrupt(InterruptType.TIMER, priority=0))
        v.enqueue(Interrupt(InterruptType.TIMER, priority=1))
        assert v.pending_count == 3
        assert [v.dequeue().priority for _ in range(3)] == [0, 1, 2]
        assert v.dequeue() is None

    def test_process_pending(self):
        v = InterruptVector()
        seen = []
        v.register(InterruptType.TIMER, lambda i: seen.append(i.data))
        v.enqueue(Interrupt(InterruptType.TIMER, data=1))
        v.enqueue(Interrupt(InterruptType.MEMORY_FULL, data=2))
        v.register(InterruptType.MEMORY_FULL, lambda i: seen.append(i.data))
        v.enqueue(Interrupt(InterruptType.DATA_READY, data=3))  # no handler
        assert v.process_pending() == 2
        assert seen == [1, 2]

    def test_stats(self):
        v = InterruptVector()
        v.register(InterruptType.TIMER, lambda i: None)
        v.mask(InterruptType.DATA_READY)
        v.enqueue(Interrupt(InterruptType.TIMER))
        v.fire(Interrupt(InterruptType.TIMER))
        s = v.stats()
        assert s["registered_handlers"] == 1
        assert s["masked_vectors"] == 1
        assert s["pending_interrupts"] == 1
        assert s["total_fired"] == 1
        assert "TIMER" in s["handlers"]


class TestInterruptManager:
    def test_register_helpers(self):
        m = InterruptManager()
        seen = []
        m.on_inference_done(lambda i: seen.append("inf"))
        m.on_training_step(lambda i: seen.append("train"))
        m.on_process_done(lambda i: seen.append("proc"))
        m.on_device_error(lambda i: seen.append("err"))
        m.on_memory_full(lambda i: seen.append("mem"))
        m.signal_inference_done(1, "r1")
        m.signal_process_done(2, "r2")
        m.signal_device_error(3, "bad gpu")
        m.vector.fire(Interrupt(InterruptType.TRAINING_STEP))
        m.vector.fire(Interrupt(InterruptType.MEMORY_FULL))
        assert seen == ["inf", "proc", "err", "train", "mem"]

    def test_stats_delegates(self):
        m = InterruptManager()
        s = m.stats()
        assert "registered_handlers" in s


class TestMemoryBlock:
    def test_num_elements(self):
        assert MemoryBlock(1, (2, 3), "f32", 24).num_elements == 6
        assert MemoryBlock(1, (), "f32", 0).num_elements == 1

    def test_defaults(self):
        b = MemoryBlock(1, (2,), "f32", 8)
        assert b.owner_pid is None
        assert b.data is None
        assert b.freed is False


class TestTensorMemory:
    def test_initial_stats(self):
        m = TensorMemory(capacity_bytes=1024)
        assert m.capacity == 1024
        assert m.used == 0
        assert m.free == 1024
        assert m.utilization == 0.0

    def test_allocate_tracks_used(self):
        m = TensorMemory(capacity_bytes=1024)
        b = m.allocate((2, 2), "float32", owner_pid=1)
        assert b.block_id == 1
        assert b.size_bytes == 16
        assert m.used == 16
        assert m.free == 1008
        assert b.data.shape == (2, 2)

    def test_allocate_out_of_memory(self):
        m = TensorMemory(capacity_bytes=100)
        with np.testing.assert_raises(MemoryError):
            m.allocate((1000, 1000), "float64")

    def test_free_block(self):
        m = TensorMemory(capacity_bytes=1024)
        b = m.allocate((2, 2), "float32", owner_pid=1)
        assert m.free_block(b.block_id) is True
        assert m.used == 0
        assert b.freed is True
        assert m.free_block(b.block_id) is False
        assert m.free_block(999) is False

    def test_get(self):
        m = TensorMemory(capacity_bytes=1024)
        b = m.allocate((2, 2))
        assert m.get(b.block_id) is b
        m.free_block(b.block_id)
        assert m.get(b.block_id) is None
        assert m.get(999) is None

    def test_read(self):
        m = TensorMemory(capacity_bytes=1024)
        b = m.allocate((2, 2))
        out = m.read(b.block_id)
        assert out is not None
        assert out is not b.data
        m.free_block(b.block_id)
        assert m.read(b.block_id) is None

    def test_write_success(self):
        m = TensorMemory(capacity_bytes=1024)
        b = m.allocate((2, 2), "float32")
        assert m.write(b.block_id, np.array([[1, 2], [3, 4]], dtype=np.float32)) is True
        np.testing.assert_array_equal(b.data, [[1, 2], [3, 4]])

    def test_write_dtype_cast(self):
        m = TensorMemory(capacity_bytes=1024)
        b = m.allocate((2,), "float32")
        assert m.write(b.block_id, np.array([1, 2], dtype=np.int32)) is True
        assert b.data.dtype == np.dtype("float32")

    def test_write_rejections(self):
        m = TensorMemory(capacity_bytes=1024)
        b = m.allocate((2, 2), "float32")
        assert m.write(999, np.zeros((2, 2))) is False
        m.free_block(b.block_id)
        assert m.write(b.block_id, np.zeros((2, 2))) is False
        b2 = m.allocate((2, 2), "float32")
        assert m.write(b2.block_id, np.zeros((3, 3))) is False

    def test_write_when_data_none(self):
        m = TensorMemory(capacity_bytes=1024)
        b = m.allocate((2, 2), "float32")
        m.free_block(b.block_id)
        b.freed = False
        b.data = None
        assert m.write(b.block_id, np.zeros((2, 2))) is False

    def test_free_pid(self):
        m = TensorMemory(capacity_bytes=1024)
        m.allocate((2, 2), "float32", owner_pid=1)
        m.allocate((2, 2), "float32", owner_pid=1)
        m.allocate((2, 2), "float32", owner_pid=2)
        assert m.free_pid(1) == 2
        assert m.used == 16
        assert m.free_pid(1) == 0

    def test_stats(self):
        m = TensorMemory(capacity_bytes=1024)
        b = m.allocate((2, 2), "float32", owner_pid=1)
        m.free_block(b.block_id)
        s = m.stats()
        assert s["capacity_bytes"] == 1024
        assert s["used_bytes"] == 0
        assert s["free_bytes"] == 1024
        assert s["utilization"] == 0.0
        assert s["total_blocks"] == 1
        assert s["active_blocks"] == 0
        assert s["freed_blocks"] == 1

    def test_defragment(self):
        m = TensorMemory(capacity_bytes=1024)
        b1 = m.allocate((2, 2), "float32")
        b2 = m.allocate((2, 2), "float32")
        m.free_block(b1.block_id)
        m.free_block(b2.block_id)
        assert m.defragment() == 2
        assert m.stats()["total_blocks"] == 0
        assert m.defragment() == 0


def _proc(pid, priority=Priority.NORMAL, state=ProcessState.CREATED):
    return Process(pid=pid, name=f"p{pid}", priority=priority, state=state)


class TestScheduler:
    def test_initial_state(self):
        s = Scheduler()
        assert s.current_pid is None
        assert s.current_process is None
        assert s.process_count == 0
        assert s.active_count == 0

    def test_add_transitions_to_ready(self):
        s = Scheduler()
        p = _proc(1)
        s.add(p)
        assert p.state == ProcessState.READY
        assert s.get(1) is p
        assert s.process_count == 1

    def test_add_preserves_non_created_state(self):
        s = Scheduler()
        p = _proc(1, state=ProcessState.RUNNING)
        s.add(p)
        assert p.state == ProcessState.RUNNING

    def test_active_count(self):
        s = Scheduler()
        s.add(_proc(1))
        s.add(_proc(2))
        s._processes[1].state = ProcessState.RUNNING
        assert s.active_count == 1

    def test_list_all(self):
        s = Scheduler()
        s.add(_proc(1))
        s.add(_proc(2))
        assert [p.pid for p in s.list_all()] == [1, 2]

    def test_remove(self):
        s = Scheduler()
        p = _proc(1)
        s.add(p)
        s._current_pid = 1
        assert s.remove(1) is p
        assert s.current_pid is None
        assert s.remove(1) is None

    def test_set_callbacks(self):
        s = Scheduler()
        s.set_callbacks(on_complete=lambda *a: None, on_interrupt=lambda *a: None)
        assert "complete" in s._callbacks
        assert "interrupt" in s._callbacks
        s.set_callbacks()
        s.set_callbacks(on_complete=None)

    def test_pick_next_skips_sleeping(self, monkeypatch):
        fixed = time.time()
        monkeypatch.setattr(kernel_scheduler.time, "time", lambda: fixed)
        s = Scheduler()
        s.add(_proc(1))
        s.add(_proc(2))
        s.wait_for(1, 10)
        assert s._pick_next() == 2

    def test_pick_next_expired_sleep(self, monkeypatch):
        fixed = time.time()
        monkeypatch.setattr(kernel_scheduler.time, "time", lambda: fixed)
        s = Scheduler()
        s.add(_proc(1))
        s.wait_for(1, 10)
        s._sleeping[1] = fixed - 1
        assert s._pick_next() == 1

    def test_pick_next_skips_missing(self):
        s = Scheduler()
        s.add(_proc(1))
        s.remove(1)
        s._queues["normal"].append(999)
        assert s._pick_next() is None

    def test_pick_next_skips_stopped(self):
        s = Scheduler()
        s.add(_proc(1))
        s.add(_proc(2))
        s._processes[1].state = ProcessState.STOPPED
        assert s._pick_next() == 2

    def test_deps_satisfied(self):
        s = Scheduler()
        dep = _proc(2)
        s.add(dep)
        p = _proc(1)
        p.metadata["deps"] = [2]
        assert s._deps_satisfied(p) is False
        dep.state = ProcessState.STOPPED
        assert s._deps_satisfied(p) is True
        p.metadata["deps"] = [99]
        assert s._deps_satisfied(p) is True
        p.metadata["deps"] = []
        assert s._deps_satisfied(p) is True

    def test_tick_runs_current(self):
        s = Scheduler()
        s.add(_proc(1))
        s.tick()
        assert s.current_pid == 1
        assert s.current_process is s._processes[1]
        assert s.tick() is s._processes[1]

    def test_tick_picks_next(self):
        s = Scheduler()
        s.add(_proc(1))
        s.add(_proc(2))
        proc = s.tick()
        assert proc.pid == 1
        assert s.current_pid == 1

    def test_tick_empty(self):
        s = Scheduler()
        assert s.tick() is None

    def test_wait_and_wake(self):
        s = Scheduler()
        s.add(_proc(1))
        s.wait_for(1, 0)
        assert s._processes[1].state == ProcessState.WAITING
        assert 1 in s._sleeping
        s.wake(1)
        assert s._processes[1].state == ProcessState.READY
        assert 1 not in s._sleeping

    def test_wake_missing_and_nonwaiting(self):
        s = Scheduler()
        s.add(_proc(1))
        s.wake(999)
        s._processes[1].state = ProcessState.RUNNING
        s.wake(1)
        assert s._processes[1].state == ProcessState.RUNNING

    def test_complete_sets_zombie_and_callback(self):
        s = Scheduler()
        s.add(_proc(1))
        calls = []
        s.set_callbacks(on_complete=lambda proc, result: calls.append((proc.pid, result)))
        s.complete(1, {"loss": 0.1})
        assert s._processes[1].state == ProcessState.ZOMBIE
        assert s._processes[1].result == {"loss": 0.1}
        assert calls == [(1, {"loss": 0.1})]

    def test_complete_missing_and_callback_error(self):
        s = Scheduler()
        s.complete(999)
        s.add(_proc(1))
        s._current_pid = 1
        s.set_callbacks(on_complete=lambda *a: (_ for _ in ()).throw(RuntimeError("cb")))
        s.complete(1)
        assert s.current_pid is None

    def test_reap_specific(self):
        s = Scheduler()
        s.add(_proc(1))
        assert s.reap(1) is None
        s.complete(1)
        out = s.reap(1)
        assert out is not None
        assert out.state == ProcessState.ZOMBIE
        assert s.reap(1) is None

    def test_reap_all(self):
        s = Scheduler()
        s.add(_proc(1))
        s.add(_proc(2))
        s.add(_proc(3))
        s.complete(1)
        s._finish(2)
        out = s.reap()
        assert len(out) == 2
        assert s.process_count == 1

    def test_finish(self):
        s = Scheduler()
        s.add(_proc(1))
        s._current_pid = 1
        s._finish(1)
        assert s._processes[1].state == ProcessState.STOPPED
        assert s.current_pid is None

    def test_queue_sizes_and_stats(self):
        s = Scheduler()
        s.add(_proc(1))
        assert s.queue_sizes() == {"realtime": 0, "high": 1, "normal": 0, "low": 0, "idle": 0}
        st = s.stats()
        assert st["total"] == 1
        assert st["active"] == 0
        assert st["current_pid"] is None
        assert st["queues"]["high"] == 1
        assert st["sleeping"] == 0

    def test_priority_to_queue(self):
        s = Scheduler()
        assert s._priority_to_queue(Priority.CRITICAL) == "realtime"
        assert s._priority_to_queue(Priority.HIGH) == "realtime"
        assert s._priority_to_queue(Priority.NORMAL) == "high"
        assert s._priority_to_queue(Priority.LOW) == "normal"
        assert s._priority_to_queue(Priority.IDLE) == "low"
        assert s._priority_to_queue(0) == "realtime"
        assert s._priority_to_queue(5) == "idle"
        assert s._priority_to_queue(Priority.CRITICAL.value) == "realtime"
