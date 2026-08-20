"""Tests for domains.shell.kernel_process — ProcessState, Priority, TensorRef, Process."""

from domains.shell.kernel_process import ProcessState, Priority, TensorRef, Process


class TestProcessState:
    def test_all_members(self):
        assert len(ProcessState) == 6
    def test_values(self):
        assert ProcessState.CREATED.value == 0
        assert ProcessState.RUNNING.value == 2
        assert ProcessState.ZOMBIE.value == 5


class TestPriority:
    def test_all_members(self):
        assert len(Priority) == 5
    def test_values(self):
        assert Priority.CRITICAL.value == 0
        assert Priority.HIGH.value == 1
        assert Priority.IDLE.value == 4


class TestTensorRef:
    def test_fields(self):
        tr = TensorRef(block_id=0, shape=(2, 3), dtype="float32", size_bytes=24, owner_pid=1)
        assert tr.block_id == 0
        assert tr.shape == (2, 3)
        assert tr.size_bytes == 24


class TestProcess:
    def test_defaults(self):
        p = Process(pid=1, name="test")
        assert p.pid == 1
        assert p.state == ProcessState.CREATED
        assert p.priority == Priority.NORMAL
        assert p.result is None
        assert p.error is None

    def test_uptime_not_started(self):
        p = Process(pid=1, name="test")
        assert p.uptime == 0.0

    def test_uptime_running(self):
        import time
        p = Process(pid=1, name="test", started_at=time.time() - 1.0)
        assert p.uptime >= 0.9

    def test_custom_priority(self):
        p = Process(pid=1, name="test", priority=Priority.HIGH)
        assert p.priority == Priority.HIGH
