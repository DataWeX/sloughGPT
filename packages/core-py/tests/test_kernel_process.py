"""Tests for domains.shell.kernel_process — ProcessState, Priority, TensorRef, Process."""

from domains.shell.kernel_process import ProcessState, Priority, TensorRef, Process


class TestProcessState:
    def test_all_members(self):
        assert len(ProcessState) == 6
    def test_values(self):
        assert ProcessState.CREATED.value == 0
        assert ProcessState.RUNNING.value == 2
        assert ProcessState.ZOMBIE.value == 5

    def test_member_names(self):
        names = [s.name for s in ProcessState]
        assert "CREATED" in names
        assert "READY" in names
        assert "RUNNING" in names
        assert "WAITING" in names
        assert "STOPPED" in names
        assert "ZOMBIE" in names

    def test_created_is_first(self):
        assert ProcessState(0) == ProcessState.CREATED

    def test_zombie_is_last(self):
        assert ProcessState(5) == ProcessState.ZOMBIE

    def test_ready_value(self):
        assert ProcessState.READY.value == 1

    def test_waiting_value(self):
        assert ProcessState.WAITING.value == 3

    def test_stopped_value(self):
        assert ProcessState.STOPPED.value == 4

    def test_intenum_behavior(self):
        assert ProcessState.CREATED == 0
        assert ProcessState.RUNNING == 2
        assert int(ProcessState.ZOMBIE) == 5

    def test_iteration(self):
        states = list(ProcessState)
        assert len(states) == 6

    def test_is_intenum(self):
        from enum import IntEnum
        assert issubclass(ProcessState, IntEnum)

    def test_comparison(self):
        assert ProcessState.CREATED < ProcessState.RUNNING
        assert ProcessState.ZOMBIE > ProcessState.STOPPED

    def test_identity(self):
        assert ProcessState.CREATED is ProcessState.CREATED

    def test_create_from_value(self):
        assert ProcessState(0) is ProcessState.CREATED
        assert ProcessState(5) is ProcessState.ZOMBIE


class TestPriority:
    def test_all_members(self):
        assert len(Priority) == 5
    def test_values(self):
        assert Priority.CRITICAL.value == 0
        assert Priority.HIGH.value == 1
        assert Priority.IDLE.value == 4

    def test_member_names(self):
        names = [p.name for p in Priority]
        assert "CRITICAL" in names
        assert "HIGH" in names
        assert "NORMAL" in names
        assert "LOW" in names
        assert "IDLE" in names

    def test_normal_value(self):
        assert Priority.NORMAL.value == 2

    def test_low_value(self):
        assert Priority.LOW.value == 3

    def test_is_intenum(self):
        from enum import IntEnum
        assert issubclass(Priority, IntEnum)

    def test_ordering(self):
        assert Priority.CRITICAL < Priority.HIGH < Priority.NORMAL < Priority.LOW < Priority.IDLE

    def test_critical_is_zero(self):
        assert Priority.CRITICAL == 0

    def test_iteration(self):
        priorities = list(Priority)
        assert len(priorities) == 5

    def test_create_from_value(self):
        assert Priority(0) is Priority.CRITICAL
        assert Priority(4) is Priority.IDLE


class TestTensorRef:
    def test_fields(self):
        tr = TensorRef(block_id=0, shape=(2, 3), dtype="float32", size_bytes=24, owner_pid=1)
        assert tr.block_id == 0
        assert tr.shape == (2, 3)
        assert tr.size_bytes == 24

    def test_owner_pid(self):
        tr = TensorRef(block_id=0, shape=(2, 3), dtype="float32", size_bytes=24, owner_pid=42)
        assert tr.owner_pid == 42

    def test_dtype(self):
        tr = TensorRef(block_id=0, shape=(4, 4), dtype="float64", size_bytes=128, owner_pid=1)
        assert tr.dtype == "float64"

    def test_shape_is_tuple(self):
        tr = TensorRef(block_id=0, shape=(2, 3), dtype="float32", size_bytes=24, owner_pid=1)
        assert isinstance(tr.shape, tuple)

    def test_large_shape(self):
        tr = TensorRef(block_id=0, shape=(1024, 1024, 3), dtype="float32", size_bytes=12582912, owner_pid=1)
        assert tr.shape == (1024, 1024, 3)

    def test_equality(self):
        tr1 = TensorRef(block_id=0, shape=(2, 3), dtype="float32", size_bytes=24, owner_pid=1)
        tr2 = TensorRef(block_id=0, shape=(2, 3), dtype="float32", size_bytes=24, owner_pid=1)
        assert tr1 == tr2

    def test_inequality(self):
        tr1 = TensorRef(block_id=0, shape=(2, 3), dtype="float32", size_bytes=24, owner_pid=1)
        tr2 = TensorRef(block_id=1, shape=(2, 3), dtype="float32", size_bytes=24, owner_pid=1)
        assert tr1 != tr2

    def test_dataclass_fields(self):
        tr = TensorRef(block_id=0, shape=(2, 3), dtype="float32", size_bytes=24, owner_pid=1)
        fields = {f.name for f in tr.__dataclass_fields__.values()}
        assert "block_id" in fields
        assert "shape" in fields
        assert "dtype" in fields
        assert "size_bytes" in fields
        assert "owner_pid" in fields

    def test_int_types(self):
        tr = TensorRef(block_id=0, shape=(2, 3), dtype="float32", size_bytes=24, owner_pid=1)
        assert isinstance(tr.block_id, int)
        assert isinstance(tr.size_bytes, int)
        assert isinstance(tr.owner_pid, int)


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

    def test_is_active_created(self):
        p = Process(pid=1, name="test")
        assert p.is_active is True

    def test_is_active_running(self):
        import time
        p = Process(pid=1, name="test", state=ProcessState.RUNNING, started_at=time.time())
        assert p.is_active is True

    def test_is_active_waiting(self):
        p = Process(pid=1, name="test", state=ProcessState.WAITING)
        assert p.is_active is True

    def test_is_active_ready(self):
        p = Process(pid=1, name="test", state=ProcessState.READY)
        assert p.is_active is True

    def test_is_done_stopped(self):
        p = Process(pid=1, name="test", state=ProcessState.STOPPED)
        assert p.is_done is True

    def test_is_done_zombie(self):
        p = Process(pid=1, name="test", state=ProcessState.ZOMBIE)
        assert p.is_done is True

    def test_is_done_running(self):
        p = Process(pid=1, name="test", state=ProcessState.RUNNING)
        assert p.is_done is False

    def test_transition_to_ready(self):
        p = Process(pid=1, name="test")
        p.transition(ProcessState.READY)
        assert p.state == ProcessState.READY

    def test_transition_to_running_sets_started_at(self):
        import time
        p = Process(pid=1, name="test")
        before = time.time()
        p.transition(ProcessState.RUNNING)
        assert p.started_at is not None
        assert p.started_at >= before

    def test_transition_to_stopped_sets_finished_at(self):
        import time
        p = Process(pid=1, name="test")
        p.transition(ProcessState.RUNNING)
        p.transition(ProcessState.STOPPED)
        assert p.finished_at is not None

    def test_transition_to_zombie_sets_finished_at(self):
        import time
        p = Process(pid=1, name="test")
        p.transition(ProcessState.RUNNING)
        p.transition(ProcessState.ZOMBIE)
        assert p.finished_at is not None

    def test_transition_does_not_overwrite_started_at(self):
        import time
        p = Process(pid=1, name="test", started_at=time.time() - 5.0)
        original = p.started_at
        p.transition(ProcessState.RUNNING)
        assert p.started_at == original

    def test_transition_does_not_overwrite_finished_at(self):
        import time
        p = Process(pid=1, name="test")
        p.transition(ProcessState.STOPPED)
        finished = p.finished_at
        p.transition(ProcessState.ZOMBIE)
        assert p.finished_at == finished

    def test_acquire_tensor(self):
        import time
        p = Process(pid=1, name="test")
        ref = TensorRef(block_id=0, shape=(2, 3), dtype="float32", size_bytes=24, owner_pid=1)
        p.acquire_tensor(ref)
        assert len(p.tensors) == 1
        assert p.memory_bytes == 24

    def test_acquire_multiple_tensors(self):
        import time
        p = Process(pid=1, name="test")
        ref1 = TensorRef(block_id=0, shape=(2, 3), dtype="float32", size_bytes=24, owner_pid=1)
        ref2 = TensorRef(block_id=1, shape=(4, 4), dtype="float32", size_bytes=64, owner_pid=1)
        p.acquire_tensor(ref1)
        p.acquire_tensor(ref2)
        assert len(p.tensors) == 2
        assert p.memory_bytes == 88

    def test_release_tensor(self):
        import time
        p = Process(pid=1, name="test")
        ref = TensorRef(block_id=0, shape=(2, 3), dtype="float32", size_bytes=24, owner_pid=1)
        p.acquire_tensor(ref)
        released = p.release_tensor(0)
        assert released is not None
        assert released.block_id == 0
        assert len(p.tensors) == 0
        assert p.memory_bytes == 0

    def test_release_nonexistent_tensor(self):
        import time
        p = Process(pid=1, name="test")
        released = p.release_tensor(999)
        assert released is None

    def test_status_line(self):
        p = Process(pid=42, name="test_process")
        status = p.status_line()
        assert "[  42]" in status
        assert "test_process" in status

    def test_status_line_created(self):
        p = Process(pid=1, name="test")
        status = p.status_line()
        assert "C" in status

    def test_status_line_running(self):
        import time
        p = Process(pid=1, name="test", started_at=time.time())
        p.transition(ProcessState.RUNNING)
        status = p.status_line()
        assert "*" in status

    def test_custom_entry(self):
        def my_func():
            return 42
        p = Process(pid=1, name="test", entry=my_func)
        assert p.entry is not None
        assert p.entry() == 42

    def test_args_and_kwargs(self):
        p = Process(pid=1, name="test", args=(1, 2), kwargs={"x": 3})
        assert p.args == (1, 2)
        assert p.kwargs == {"x": 3}

    def test_depends_on(self):
        p = Process(pid=1, name="test", depends_on=[2, 3])
        assert p.depends_on == [2, 3]

    def test_metadata(self):
        p = Process(pid=1, name="test", metadata={"key": "value"})
        assert p.metadata["key"] == "value"

    def test_cpu_time_default(self):
        p = Process(pid=1, name="test")
        assert p.cpu_time_ms == 0.0

    def test_inference_count_default(self):
        p = Process(pid=1, name="test")
        assert p.inference_count == 0

    def test_tokens_generated_default(self):
        p = Process(pid=1, name="test")
        assert p.tokens_generated == 0

    def test_custom_cpu_time(self):
        p = Process(pid=1, name="test", cpu_time_ms=123.45)
        assert p.cpu_time_ms == 123.45

    def test_uptime_after_stop(self):
        import time
        t0 = time.time() - 2.0
        p = Process(pid=1, name="test", started_at=t0, finished_at=time.time())
        assert p.uptime >= 1.9

    def test_status_line_priority_critical(self):
        p = Process(pid=1, name="test", priority=Priority.CRITICAL)
        status = p.status_line()
        assert "!" in status

    def test_status_line_priority_high(self):
        p = Process(pid=1, name="test", priority=Priority.HIGH)
        status = p.status_line()
        assert "H" in status

    def test_status_line_priority_idle(self):
        p = Process(pid=1, name="test", priority=Priority.IDLE)
        status = p.status_line()
        assert "." in status

    def test_dataclass_fields(self):
        p = Process(pid=1, name="test")
        fields = {f.name for f in p.__dataclass_fields__.values()}
        assert "pid" in fields
        assert "name" in fields
        assert "state" in fields
        assert "priority" in fields
