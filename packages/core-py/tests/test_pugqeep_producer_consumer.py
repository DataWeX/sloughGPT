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
