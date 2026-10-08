"""Tests for domain.infrastructure._internal.fire_and_forget — shared
bounded fire-and-forget worker pool."""

import threading
import time

from domain.infrastructure._internal.fire_and_forget import (
    FireAndForgetPool,
    get_pool,
)


class TestFireAndForgetPool:
    def test_runs_submitted_task_in_background(self):
        pool = FireAndForgetPool(size=2, queue_size=4)
        done = threading.Event()
        result = {}

        def task():
            result["tid"] = threading.get_ident()
            done.set()

        assert pool.submit(task) is True
        assert done.wait(timeout=2), "task never ran"
        assert result["tid"] != threading.get_ident()
        pool.shutdown()

    def test_multiple_tasks_all_complete(self):
        pool = FireAndForgetPool(size=2, queue_size=8)
        done = threading.Event()
        count = {"n": 0}
        lock = threading.Lock()

        def task():
            with lock:
                count["n"] += 1
            if count["n"] == 5:
                done.set()

        for _ in range(5):
            assert pool.submit(task) is True
        assert done.wait(timeout=2), "tasks never completed"
        assert count["n"] == 5
        pool.shutdown()

    def test_drops_when_queue_full(self):
        pool = FireAndForgetPool(size=1, queue_size=1)
        blocked = threading.Event()
        release = threading.Event()

        def blocker():
            blocked.set()
            release.wait()

        assert pool.submit(blocker) is True
        assert blocked.wait(timeout=2)
        assert pool.submit(blocker) is True
        assert pool.submit(blocker) is False
        release.set()
        pool.shutdown()

    def test_dispatch_never_blocks(self):
        pool = FireAndForgetPool(size=1, queue_size=0)
        blocked = threading.Event()
        big_wait = threading.Event()

        def blocker():
            blocked.set()
            big_wait.wait()

        pool.submit(blocker)
        blocked.wait(timeout=2)
        start = time.perf_counter()
        pool.submit(blocker)
        assert time.perf_counter() - start < 0.05
        pool.submit(blocker)
        big_wait.set()
        pool.shutdown()

    def test_stats(self):
        pool = FireAndForgetPool(size=1, queue_size=2)
        blocked = threading.Event()
        wait = threading.Event()

        def blocker():
            blocked.set()
            wait.wait()

        pool.submit(blocker)
        blocked.wait(timeout=2)
        assert pool.submit(blocker) is True
        assert pool.submit(blocker) is True
        assert pool.submit(blocker) is False  # cap 2 full -> dropped
        stats = pool.stats
        assert stats["submitted"] == 4
        assert stats["dropped"] == 1
        assert stats["workers"] == 1
        wait.set()
        pool.shutdown()

    def test_shutdown_closes_and_rejects(self):
        pool = FireAndForgetPool(size=1, queue_size=2)
        pool.shutdown()
        assert pool.submit(lambda: None) is False
        assert pool.stats["workers"] == 0

    def test_shutdown_idempotent(self):
        pool = FireAndForgetPool(size=1, queue_size=2)
        pool.shutdown()
        pool.shutdown()
        assert pool.stats["workers"] == 0

    def test_task_failure_swallowed(self):
        pool = FireAndForgetPool(size=1, queue_size=2)
        done = threading.Event()

        def boom():
            raise RuntimeError("expected")

        def finisher():
            done.set()

        pool.submit(boom)
        pool.submit(finisher)
        assert done.wait(timeout=2), "worker died on task failure"
        pool.shutdown()


class TestGetPool:
    def test_singleton(self):
        assert get_pool() is get_pool()
        assert get_pool().stats["workers"] >= 1

    def test_default_not_closed(self):
        assert get_pool().submit(lambda: None) is True
