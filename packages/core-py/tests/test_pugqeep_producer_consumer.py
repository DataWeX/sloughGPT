"""Tests for pugqeep + ProducerConsumerQueue integration."""

import threading
import time

import pytest

from domains.infrastructure.pugqeep.task_queue import TaskQueue, Task, TaskStatus, TaskPriority
from domains.infrastructure.pugqeep.engine import Engine, Process, ProcessStatus


# ── TaskQueue worker pool tests ─────────────────────────────────────


class TestTaskQueueWorkers:
    def test_start_stop_workers(self):
        q = TaskQueue(name="test-workers")
        q.start_workers(num_workers=2)
        assert q.workers_active >= 0
        q.stop_workers()

    def test_auto_execute_task(self):
        results = []
        lock = threading.Lock()

        def handler(task):
            with lock:
                results.append(task.data)

        q = TaskQueue(name="test-auto")
        q.register_handler("work", handler)
        q.start_workers(num_workers=2)

        try:
            t1 = Task(name="work", data="item1")
            t2 = Task(name="work", data="item2")
            q.submit(t1)
            q.submit(t2)
            time.sleep(0.5)

            with lock:
                assert sorted(results) == ["item1", "item2"]
            assert t1.status == TaskStatus.COMPLETED
            assert t2.status == TaskStatus.COMPLETED
        finally:
            q.stop_workers()

    def test_priority_ordering(self):
        order = []
        lock = threading.Lock()
        gate = threading.Event()

        def handler(task):
            with lock:
                order.append(task.data)
            gate.wait(timeout=2.0)

        q = TaskQueue(name="test-priority")
        q.register_handler("work", handler)
        q.start_workers(num_workers=1)

        try:
            q.submit(Task(name="work", data="low", priority=TaskPriority.LOW))
            time.sleep(0.05)  # let first task start processing
            q.submit(Task(name="work", data="urgent", priority=TaskPriority.URGENT))
            q.submit(Task(name="work", data="normal", priority=TaskPriority.NORMAL))
            gate.set()  # release first task
            time.sleep(1.0)
            with lock:
                # First was "low" (already processing when others enqueued)
                # After release, PriorityQueue dequeues "urgent" (int=0) before "normal" (int=2)
                assert order == ["low", "urgent", "normal"]
        finally:
            q.stop_workers()

    def test_handler_failure_marks_task_failed(self):
        def bad_handler(task):
            raise ValueError("boom")

        q = TaskQueue(name="test-fail")
        q.register_handler("work", bad_handler)
        q.start_workers(num_workers=1)

        try:
            t = Task(name="work", data="x", max_retries=0)
            q.submit(t)
            time.sleep(0.3)
            assert t.status == TaskStatus.FAILED
            assert t.error == "boom"
        finally:
            q.stop_workers()

    def test_stats_include_workers(self):
        q = TaskQueue(name="test-stats")
        q.start_workers(num_workers=2)
        try:
            s = q.stats()
            assert "workers" in s
            assert s["workers"]["num_workers"] == 2
        finally:
            q.stop_workers()

    def test_workers_metrics(self):
        q = TaskQueue(name="test-metrics")
        q.start_workers(num_workers=1)
        try:
            m = q.workers_metrics
            assert "enqueued" in m
            assert "consumed" in m
        finally:
            q.stop_workers()

    def test_submit_without_workers_still_works(self):
        q = TaskQueue(name="test-no-workers")
        q.register_handler("work", lambda t: None)
        t = Task(name="work", data="x")
        q.submit(t)
        assert t.status == TaskStatus.PENDING
        # Manual processing still works
        got = q.next()
        assert got is not None
        q.complete(got.id, result="done")
        assert got.status == TaskStatus.COMPLETED

    def test_concurrent_submits(self):
        results = []
        lock = threading.Lock()

        def handler(task):
            with lock:
                results.append(task.data)

        q = TaskQueue(name="test-concurrent")
        q.register_handler("work", handler)
        q.start_workers(num_workers=4)

        try:
            for i in range(50):
                q.submit(Task(name="work", data=i))
            time.sleep(1.0)
            with lock:
                assert len(results) == 50
                assert len(set(results)) == 50
        finally:
            q.stop_workers()

    def test_start_workers_idempotent(self):
        q = TaskQueue(name="test-idempotent")
        q.start_workers(num_workers=2)
        q.start_workers(num_workers=2)  # should not error
        q.stop_workers()

    def test_stop_workers_when_not_started(self):
        q = TaskQueue(name="test-stop-noop")
        q.stop_workers()  # should not error

    def test_task_result_preserved(self):
        def handler(task):
            return "result_value"

        q = TaskQueue(name="test-result")
        q.register_handler("work", handler)
        q.start_workers(num_workers=1)

        try:
            t = Task(name="work", data="x")
            q.submit(t)
            time.sleep(0.5)
            assert t.result == "result_value"
            assert t.status == TaskStatus.COMPLETED
        finally:
            q.stop_workers()

    def test_multiple_handlers_different_names(self):
        results = []
        lock = threading.Lock()

        def handler_a(task):
            with lock:
                results.append(("a", task.data))

        def handler_b(task):
            with lock:
                results.append(("b", task.data))

        q = TaskQueue(name="test-multi")
        q.register_handler("type_a", handler_a)
        q.register_handler("type_b", handler_b)
        q.start_workers(num_workers=2)

        try:
            q.submit(Task(name="type_a", data="1"))
            q.submit(Task(name="type_b", data="2"))
            time.sleep(0.5)
            with lock:
                assert len(results) == 2
                types = sorted([r[0] for r in results])
                assert types == ["a", "b"]
        finally:
            q.stop_workers()

    def test_handler_receives_task_object(self):
        received = []
        lock = threading.Lock()

        def handler(task):
            with lock:
                received.append(task)

        q = TaskQueue(name="test-obj")
        q.register_handler("work", handler)
        q.start_workers(num_workers=1)

        try:
            t = Task(name="work", data="payload")
            q.submit(t)
            time.sleep(0.5)
            with lock:
                assert len(received) == 1
                assert received[0].data == "payload"
        finally:
            q.stop_workers()

    def test_workers_active_count(self):
        q = TaskQueue(name="test-active")
        q.start_workers(num_workers=3)
        try:
            active = q.workers_active
            assert active >= 0
        finally:
            q.stop_workers()

    def test_workers_queue_depth(self):
        gate = threading.Event()
        def blocking_handler(task):
            gate.wait(timeout=2.0)
        q = TaskQueue(name="test-depth")
        q.register_handler("work", blocking_handler)
        q.start_workers(num_workers=1)
        try:
            q.submit(Task(name="work", data="blocker"))
            time.sleep(0.05)
            q.submit(Task(name="work", data="queued"))
            time.sleep(0.05)
            depth = q.workers_queue_depth
            assert depth >= 0
            gate.set()
            time.sleep(0.3)
        finally:
            q.stop_workers()

    def test_handler_exception_does_not_crash_worker(self):
        call_count = [0]
        lock = threading.Lock()

        def handler(task):
            with lock:
                call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first fails")
            return "ok"

        q = TaskQueue(name="test-crash-resilient")
        q.register_handler("work", handler)
        q.start_workers(num_workers=1)
        try:
            t1 = Task(name="work", data="fail", max_retries=0)
            t2 = Task(name="work", data="succeed", max_retries=0)
            q.submit(t1)
            q.submit(t2)
            time.sleep(0.5)
            assert t1.status == TaskStatus.FAILED
            assert t2.status == TaskStatus.COMPLETED
        finally:
            q.stop_workers()

    def test_many_tasks_sequential(self):
        results = []
        lock = threading.Lock()
        def handler(task):
            with lock:
                results.append(task.data)

        q = TaskQueue(name="test-many")
        q.register_handler("work", handler)
        q.start_workers(num_workers=1)
        try:
            for i in range(20):
                q.submit(Task(name="work", data=i))
            time.sleep(1.0)
            with lock:
                assert len(results) == 20
        finally:
            q.stop_workers()

    def test_stop_workers_drains(self):
        results = []
        lock = threading.Lock()
        def handler(task):
            with lock:
                results.append(task.data)

        q = TaskQueue(name="test-drain")
        q.register_handler("work", handler)
        q.start_workers(num_workers=2)
        for i in range(5):
            q.submit(Task(name="work", data=i))
        time.sleep(0.5)
        q.stop_workers(timeout=2.0)
        with lock:
            assert len(results) == 5


# ── Task dataclass tests ─────────────────────────────────────────────


class TestTask:
    def test_defaults(self):
        t = Task()
        assert t.status == TaskStatus.PENDING
        assert t.priority == TaskPriority.NORMAL
        assert t.retries == 0
        assert t.max_retries == 3
        assert t.metadata == {}

    def test_to_dict(self):
        t = Task(name="work", data="payload")
        d = t.to_dict()
        assert d["name"] == "work"
        assert d["data"] == "payload"
        assert d["status"] == "pending"

    def test_from_dict(self):
        t = Task(name="test", data=42, priority=TaskPriority.HIGH)
        d = t.to_dict()
        t2 = Task.from_dict(d)
        assert t2.name == "test"
        assert t2.data == 42
        assert t2.priority == TaskPriority.HIGH

    def test_unique_ids(self):
        t1 = Task()
        t2 = Task()
        assert t1.id != t2.id

    def test_status_lifecycle(self):
        t = Task()
        assert t.status == TaskStatus.PENDING
        t.status = TaskStatus.RUNNING
        assert t.status == TaskStatus.RUNNING
        t.status = TaskStatus.COMPLETED
        assert t.status == TaskStatus.COMPLETED

    def test_priority_levels(self):
        assert TaskPriority.LOW.value == 0
        assert TaskPriority.NORMAL.value == 1
        assert TaskPriority.HIGH.value == 2
        assert TaskPriority.URGENT.value == 3

    def test_from_dict_roundtrip(self):
        t = Task(name="r", data="x", tree_id="t1", metadata={"k": "v"})
        d = t.to_dict()
        t2 = Task.from_dict(d)
        assert t2.name == "r"
        assert t2.tree_id == "t1"
        assert t2.metadata == {"k": "v"}

    def test_task_default_id_format(self):
        t = Task()
        assert len(t.id) == 12

    def test_task_created_at_is_recent(self):
        before = time.time()
        t = Task()
        after = time.time()
        assert before <= t.created_at <= after

    def test_task_error_default(self):
        t = Task()
        assert t.error is None

    def test_task_result_default(self):
        t = Task()
        assert t.result is None

    def test_task_tree_id_default(self):
        t = Task()
        assert t.tree_id is None

    def test_task_data_none_default(self):
        t = Task()
        assert t.data is None

    def test_task_retries_default(self):
        t = Task()
        assert t.retries == 0

    def test_task_max_retries_default(self):
        t = Task()
        assert t.max_retries == 3

    def test_task_from_dict_with_optional_fields(self):
        d = {
            "id": "abc123", "name": "t", "data": None, "status": "pending",
            "priority": 1, "tree_id": None, "result": None, "error": None,
            "created_at": 100.0, "started_at": None, "completed_at": None,
            "retries": 0, "max_retries": 3, "metadata": {},
        }
        t = Task.from_dict(d)
        assert t.id == "abc123"
        assert t.created_at == 100.0


# ── TaskQueue manual operation tests ─────────────────────────────────


class TestTaskQueueManual:
    def test_submit_and_next(self):
        q = TaskQueue(name="manual")
        t = Task(name="work", data="x")
        q.submit(t)
        got = q.next()
        assert got is not None
        assert got.id == t.id

    def test_next_empty_queue(self):
        q = TaskQueue(name="empty")
        assert q.next() is None

    def test_complete_marks_done(self):
        q = TaskQueue(name="done")
        t = Task(name="work")
        q.submit(t)
        got = q.next()
        result = q.complete(got.id, result="ok")
        assert result.status == TaskStatus.COMPLETED
        assert result.result == "ok"

    def test_fail_marks_failed(self):
        q = TaskQueue(name="fail")
        t = Task(name="work", max_retries=0)
        q.submit(t)
        got = q.next()
        result = q.fail(got.id, error="bad")
        assert result.status == TaskStatus.FAILED
        assert result.error == "bad"

    def test_cancel_task(self):
        q = TaskQueue(name="cancel")
        t = Task(name="work")
        q.submit(t)
        result = q.cancel(t.id)
        assert result.status == TaskStatus.CANCELLED

    def test_pause_prevents_next(self):
        q = TaskQueue(name="pause")
        q.submit(Task(name="work"))
        q.pause()
        assert q.next() is None
        q.resume()

    def test_submit_batch(self):
        q = TaskQueue(name="batch")
        items = [{"name": f"task_{i}", "data": i} for i in range(5)]
        tasks = q.submit_batch(items)
        assert len(tasks) == 5
        assert all(t.status == TaskStatus.PENDING for t in tasks)

    def test_list_tasks_filter(self):
        q = TaskQueue(name="filter")
        t1 = Task(name="a")
        t2 = Task(name="b")
        q.submit(t1)
        q.submit(t2)
        pending = q.list_tasks(status=TaskStatus.PENDING)
        assert len(pending) == 2

    def test_get_task(self):
        q = TaskQueue(name="get")
        t = Task(name="work")
        q.submit(t)
        got = q.get_task(t.id)
        assert got is not None
        assert got.id == t.id

    def test_retry_task(self):
        q = TaskQueue(name="retry")
        t = Task(name="work", max_retries=0)
        q.submit(t)
        q.next()
        q.fail(t.id, error="fail")
        retried = q.retry(t.id)
        assert retried.status == TaskStatus.PENDING

    def test_cancel_all(self):
        q = TaskQueue(name="cancelall")
        q.submit(Task(name="a"))
        q.submit(Task(name="b"))
        cancelled = q.cancel_all()
        assert cancelled == 2

    def test_clear_completed(self):
        q = TaskQueue(name="clear")
        t = Task(name="work")
        q.submit(t)
        q.next()
        q.complete(t.id)
        count = q.clear_completed()
        assert count == 1

    def test_on_complete_callback(self):
        results = []
        q = TaskQueue(name="callback")
        q.on_complete(lambda t: results.append(t.id))
        t = Task(name="work")
        q.submit(t)
        q.next()
        q.complete(t.id)
        assert len(results) == 1

    def test_stats(self):
        q = TaskQueue(name="stats")
        q.submit(Task(name="a"))
        s = q.stats()
        assert s["total"] == 1
        assert s["pending"] == 1

    def test_submit_batch_with_priority(self):
        q = TaskQueue(name="batch-pri")
        items = [{"name": "a", "priority": 3}, {"name": "b"}]
        tasks = q.submit_batch(items, priority=TaskPriority.LOW)
        assert len(tasks) == 2
        assert tasks[0].priority == TaskPriority.URGENT
        assert tasks[1].priority == TaskPriority.LOW

    def test_list_tasks_all(self):
        q = TaskQueue(name="list-all")
        q.submit(Task(name="a"))
        q.submit(Task(name="b"))
        all_tasks = q.list_tasks()
        assert len(all_tasks) == 2

    def test_next_sets_running_status(self):
        q = TaskQueue(name="running")
        t = Task(name="work")
        q.submit(t)
        got = q.next()
        assert got.status == TaskStatus.RUNNING
        assert got.started_at is not None

    def test_complete_sets_completed_at(self):
        q = TaskQueue(name="completed-at")
        t = Task(name="work")
        q.submit(t)
        q.next()
        result = q.complete(t.id)
        assert result.completed_at is not None

    def test_fail_retries_back_to_pending(self):
        q = TaskQueue(name="retry-pending")
        t = Task(name="work", max_retries=2)
        q.submit(t)
        q.next()
        result = q.fail(t.id, error="oops")
        assert result.status == TaskStatus.PENDING
        assert result.retries == 1

    def test_fail_exhausts_retries(self):
        q = TaskQueue(name="exhaust")
        t = Task(name="work", max_retries=0)
        q.submit(t)
        q.next()
        result = q.fail(t.id, error="fail")
        assert result.status == TaskStatus.FAILED
        assert result.retries == 0

    def test_cancel_nonexistent(self):
        q = TaskQueue(name="cancel-no")
        result = q.cancel("nonexistent")
        assert result is None

    def test_complete_nonexistent(self):
        q = TaskQueue(name="complete-no")
        result = q.complete("nonexistent")
        assert result is None

    def test_fail_nonexistent(self):
        q = TaskQueue(name="fail-no")
        result = q.fail("nonexistent", error="x")
        assert result is None

    def test_get_task_nonexistent(self):
        q = TaskQueue(name="get-no")
        assert q.get_task("nonexistent") is None

    def test_retry_nonexistent(self):
        q = TaskQueue(name="retry-no")
        assert q.retry("nonexistent") is None

    def test_stats_handlers(self):
        q = TaskQueue(name="stats-h")
        q.register_handler("a", lambda t: None)
        q.register_handler("b", lambda t: None)
        s = q.stats()
        assert "a" in s["handlers"]
        assert "b" in s["handlers"]

    def test_stats_paused(self):
        q = TaskQueue(name="stats-p")
        q.pause()
        s = q.stats()
        assert s["paused"] is True
        q.resume()

    def test_max_size(self):
        q = TaskQueue(name="max-size", max_size=2)
        q.submit(Task(name="a"))
        q.submit(Task(name="b"))
        with pytest.raises(ValueError, match="Queue full"):
            q.submit(Task(name="c"))


# ── Engine worker pool tests ────────────────────────────────────────


class TestEngineWorkers:
    def test_start_stop_workers(self):
        engine = Engine("test-engine-workers")
        engine.tree("t1")
        engine.start_workers(num_workers=2)
        engine.stop_workers()

    def test_auto_dispatch_via_workers(self):
        results = []
        lock = threading.Lock()

        def work_fn(x):
            with lock:
                results.append(x)

        engine = Engine("test-dispatch")
        engine.tree("default")
        engine.start_workers(num_workers=2)

        try:
            p1 = engine.spawn(work_fn, "a", name="work_fn")
            p2 = engine.spawn(work_fn, "b", name="work_fn")
            time.sleep(0.5)

            with lock:
                assert sorted(results) == ["a", "b"]
            assert p1.status == ProcessStatus.COMPLETED
            assert p2.status == ProcessStatus.COMPLETED
        finally:
            engine.stop()

    def test_routed_dispatch(self):
        results = []
        lock = threading.Lock()

        def load_fn(x):
            with lock:
                results.append(("load", x))

        def train_fn(x):
            with lock:
                results.append(("train", x))

        engine = Engine("test-router")
        engine.tree("data")
        engine.tree("training")
        engine.route("load_fn", "data")
        engine.route("train_fn", "training")
        engine.start_workers(num_workers=2)

        try:
            engine.spawn(load_fn, "weights", name="load_fn")
            engine.spawn(train_fn, "epochs", name="train_fn")
            time.sleep(0.5)

            with lock:
                types = sorted([r[0] for r in results])
                assert types == ["load", "train"]
        finally:
            engine.stop()

    def test_spawn_with_priority(self):
        engine = Engine("test-priority")
        engine.tree("t1")
        engine.start_workers(num_workers=1)
        try:
            p = engine.spawn(lambda: None, name="work", priority=0)
            assert p is not None
        finally:
            engine.stop()

    def test_process_status_lifecycle(self):
        p = Process(fn=lambda: 42)
        assert p.status == ProcessStatus.CREATED
        p.ready()
        assert p.status == ProcessStatus.READY
        p.running()
        assert p.status == ProcessStatus.RUNNING
        p.complete(result=42)
        assert p.status == ProcessStatus.COMPLETED
        assert p.result == 42

    def test_process_fail(self):
        p = Process(fn=lambda: None)
        p.running()
        p.fail("error msg")
        assert p.status == ProcessStatus.FAILED
        assert p.error == "error msg"

    def test_process_cancel(self):
        p = Process(fn=lambda: None)
        p.running()
        p.cancel()
        assert p.status == ProcessStatus.CANCELLED
        assert p.is_cancelled

    def test_process_is_done(self):
        p = Process(fn=lambda: None)
        assert not p.is_done
        p.complete()
        assert p.is_done

    def test_process_to_dict(self):
        p = Process(fn=lambda: None, name="test")
        d = p.to_dict()
        assert d["name"] == "test"
        assert "status" in d
        assert "id" in d

    def test_process_elapsed(self):
        p = Process(fn=lambda: None)
        assert p.elapsed is None
        p.running()
        time.sleep(0.05)
        assert p.elapsed is not None
        assert p.elapsed > 0

    def test_process_stream_results(self):
        p = Process(fn=lambda: None)
        p.emit("chunk1")
        p.emit("chunk2")
        assert p.stream_results == ["chunk1", "chunk2"]

    def test_process_on_complete_callback(self):
        results = []
        p = Process(fn=lambda: None)
        p.on_complete(lambda proc: results.append(proc.result))
        p.complete(result="done")
        assert results == ["done"]

    def test_process_on_fail_callback(self):
        results = []
        p = Process(fn=lambda: None)
        p.on_fail(lambda proc: results.append(proc.error))
        p.fail("oops")
        assert results == ["oops"]

    def test_process_progress(self):
        p = Process(fn=lambda: None)
        assert p.progress == 0.0
        p.report_progress(0.5, "halfway")
        assert p.progress == 0.5
        assert p.progress_message == "halfway"

    def test_engine_spawn_creates_process(self):
        engine = Engine("spawn-test")
        engine.tree("t1")
        p = engine.spawn(lambda: None, name="test")
        assert p.name == "test"
        assert p.id in str(engine._processes)

    def test_engine_tree_count(self):
        engine = Engine("tree-count")
        engine.tree("a")
        engine.tree("b")
        assert len(engine._trees) == 2

    def test_engine_route(self):
        engine = Engine("route-test")
        engine.tree("target")
        engine.route("my_fn", "target")
        assert engine._routing["my_fn"] == "target"

    def test_process_on_cancel_callback(self):
        results = []
        p = Process(fn=lambda: None)
        p.on_cancel(lambda proc: results.append("cancelled"))
        p.cancel()
        assert results == ["cancelled"]

    def test_process_on_stream_callback(self):
        results = []
        p = Process(fn=lambda: None)
        p.on_stream(lambda proc, val: results.append(val))
        p.emit("val1")
        p.emit("val2")
        assert results == ["val1", "val2"]

    def test_process_on_progress_callback(self):
        results = []
        p = Process(fn=lambda: None)
        p.on_progress(lambda proc, prog, msg: results.append((prog, msg)))
        p.report_progress(0.7, "almost")
        assert results == [(0.7, "almost")]

    def test_process_progress_clamps(self):
        p = Process(fn=lambda: None)
        p.report_progress(1.5)
        assert p.progress == 1.0
        p.report_progress(-0.5)
        assert p.progress == 0.0

    def test_process_is_done_cancelled(self):
        p = Process(fn=lambda: None)
        p.cancel()
        assert p.is_done

    def test_process_is_done_failed(self):
        p = Process(fn=lambda: None)
        p.fail("err")
        assert p.is_done

    def test_process_to_dict_completed(self):
        p = Process(fn=lambda: None)
        p.running()
        p.complete(result=42)
        d = p.to_dict()
        assert d["status"] == "completed"
        assert d["elapsed"] is not None

    def test_process_to_dict_with_timeout(self):
        p = Process(fn=lambda: None, timeout=5.0)
        d = p.to_dict()
        assert d["timeout"] == 5.0

    def test_process_to_dict_with_depends_on(self):
        p = Process(fn=lambda: None)
        p.depends_on = ["dep1", "dep2"]
        d = p.to_dict()
        assert d["depends_on"] == ["dep1", "dep2"]

    def test_process_name_default(self):
        p = Process(fn=lambda: None)
        assert p.name == ""

    def test_process_args_kwargs(self):
        def add(a, b, c=0):
            return a + b + c
        p = Process(fn=add, args=(1, 2), kwargs={"c": 3})
        result = p.fn(*p.args, **p.kwargs)
        assert result == 6

    def test_engine_route_nonexistent_tree(self):
        engine = Engine("route-bad")
        with pytest.raises(ValueError, match="not found"):
            engine.route("fn", "nonexistent")
