"""Tests for domains/shell/kernel_process.py — Process primitives."""

from __future__ import annotations

import time

from domains.shell.kernel_process import (
    Priority,
    Process,
    ProcessState,
    TensorRef,
)


class TestProcessState:
    def test_enum_values(self):
        assert ProcessState.CREATED == 0
        assert ProcessState.READY == 1
        assert ProcessState.RUNNING == 2
        assert ProcessState.WAITING == 3
        assert ProcessState.STOPPED == 4
        assert ProcessState.ZOMBIE == 5


class TestPriority:
    def test_enum_values(self):
        assert Priority.CRITICAL == 0
        assert Priority.HIGH == 1
        assert Priority.NORMAL == 2
        assert Priority.LOW == 3
        assert Priority.IDLE == 4


class TestTensorRef:
    def test_defaults(self):
        ref = TensorRef(block_id=1, shape=(4, 4), dtype="float32", size_bytes=128, owner_pid=7)
        assert ref.block_id == 1
        assert ref.shape == (4, 4)
        assert ref.dtype == "float32"
        assert ref.size_bytes == 128
        assert ref.owner_pid == 7


class TestProcess:
    def test_defaults(self):
        p = Process(pid=1, name="test")
        assert p.state == ProcessState.CREATED
        assert p.priority == Priority.NORMAL
        assert p.uptime == 0.0
        assert p.tensors == []
        assert p.memory_bytes == 0

    def test_uptime_zero_before_start(self):
        p = Process(pid=1, name="test")
        p.started_at = None
        assert p.uptime == 0.0

    def test_uptime_while_running(self):
        p = Process(pid=1, name="test")
        p.started_at = time.time() - 5
        assert 4.9 <= p.uptime <= 5.1

    def test_uptime_after_finish(self):
        p = Process(pid=1, name="test")
        p.started_at = 100.0
        p.finished_at = 107.0
        assert p.uptime == 7.0

    def test_is_active_states(self):
        for state in (ProcessState.CREATED, ProcessState.READY,
                      ProcessState.RUNNING, ProcessState.WAITING):
            p = Process(pid=1, name="x", state=state)
            assert p.is_active

    def test_is_active_false_when_done(self):
        for state in (ProcessState.STOPPED, ProcessState.ZOMBIE):
            p = Process(pid=1, name="x", state=state)
            assert not p.is_active

    def test_is_done(self):
        for state in (ProcessState.STOPPED, ProcessState.ZOMBIE):
            p = Process(pid=1, name="x", state=state)
            assert p.is_done
        p = Process(pid=1, name="x", state=ProcessState.RUNNING)
        assert not p.is_done

    def test_transition_to_running_records_started_once(self):
        p = Process(pid=1, name="x")
        p.transition(ProcessState.RUNNING)
        started = p.started_at
        assert started is not None
        p.transition(ProcessState.WAITING)
        p.transition(ProcessState.RUNNING)
        assert p.started_at == started

    def test_transition_to_stopped_records_finished_once(self):
        p = Process(pid=1, name="x")
        p.transition(ProcessState.RUNNING)
        p.transition(ProcessState.STOPPED)
        finished = p.finished_at
        assert finished is not None
        p.transition(ProcessState.RUNNING)
        p.transition(ProcessState.ZOMBIE)
        assert p.finished_at == finished

    def test_transition_no_timestamp_when_running_again(self):
        p = Process(pid=1, name="x")
        p.transition(ProcessState.RUNNING)
        p.transition(ProcessState.STOPPED)
        p.finished_at = None
        p.transition(ProcessState.STOPPED)
        assert p.finished_at is not None

    def test_acquire_tensor(self):
        p = Process(pid=1, name="x")
        ref = TensorRef(block_id=9, shape=(2, 2), dtype="float32", size_bytes=64, owner_pid=1)
        p.acquire_tensor(ref)
        assert p.tensors == [ref]
        assert p.memory_bytes == 64

    def test_release_tensor_found(self):
        p = Process(pid=1, name="x")
        a = TensorRef(block_id=1, shape=(), dtype="f32", size_bytes=10, owner_pid=1)
        b = TensorRef(block_id=2, shape=(), dtype="f32", size_bytes=20, owner_pid=1)
        p.acquire_tensor(a)
        p.acquire_tensor(b)
        out = p.release_tensor(1)
        assert out is a
        assert p.tensors == [b]
        assert p.memory_bytes == 20

    def test_release_tensor_not_found(self):
        p = Process(pid=1, name="x")
        assert p.release_tensor(42) is None

    def test_status_line_characters(self):
        p = Process(pid=7, name="worker", state=ProcessState.RUNNING, priority=Priority.CRITICAL)
        line = p.status_line()
        assert "7" in line and "*" in line and "!" in line
        assert "worker" in line

    def test_status_line_all_states(self):
        for state, char in [
            (ProcessState.CREATED, "C"),
            (ProcessState.READY, "R"),
            (ProcessState.RUNNING, "*"),
            (ProcessState.WAITING, "W"),
            (ProcessState.STOPPED, "S"),
            (ProcessState.ZOMBIE, "Z"),
        ]:
            p = Process(pid=1, name="x", state=state)
            assert char in p.status_line()

    def test_status_line_all_priorities(self):
        for priority, char in [
            (Priority.CRITICAL, "!"),
            (Priority.HIGH, "H"),
            (Priority.NORMAL, " "),
            (Priority.LOW, "L"),
            (Priority.IDLE, "."),
        ]:
            p = Process(pid=1, name="x", priority=priority)
            assert char in p.status_line()
