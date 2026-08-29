"""Tests for ProducerConsumerQueue — thread-safe work distribution."""

import threading
import time

import pytest

from domains.infrastructure.producer_consumer import (
    ProducerConsumerQueue,
    ShutdownMode,
    _PriorityItem,
)


# ── Basic sync tests (no handler = manual get) ──────────────────────


class TestSyncBasic:
    def test_put_get_roundtrip(self):
        q = ProducerConsumerQueue[int](num_consumers=0, name="test-roundtrip")
        assert q.put(42)
        ok, item = q.get(timeout=1.0)
        assert ok
        assert item == 42

    def test_put_nowait(self):
        q = ProducerConsumerQueue[str](num_consumers=0, name="test-nowait")
        assert q.put_nowait("hello")
        ok, item = q.get(timeout=1.0)
        assert ok and item == "hello"

    def test_get_timeout_returns_false(self):
        q = ProducerConsumerQueue[int](num_consumers=0, name="test-timeout")
        ok, item = q.get(timeout=0.05)
        assert not ok
        assert item is None

    def test_qsize(self):
        q = ProducerConsumerQueue[int](num_consumers=0, name="test-qsize")
        assert q.qsize == 0
        q.put(1)
        q.put(2)
        assert q.qsize == 2

    def test_empty_property(self):
        q = ProducerConsumerQueue[int](num_consumers=0, name="test-empty")
        assert q.empty
        q.put(1)
        assert not q.empty


# ── Backpressure tests ──────────────────────────────────────────────


class TestBackpressure:
    def test_full_queue_drops_item(self):
        q = ProducerConsumerQueue[int](maxsize=2, num_consumers=0, name="test-backpressure")
        q.put(1)
        q.put(2)
        assert not q.put_nowait(3)  # full
        assert q.qsize == 2
        assert q.metrics["dropped"] == 1

    def test_unbounded_queue(self):
        q = ProducerConsumerQueue[int](maxsize=0, num_consumers=0, name="test-unbounded")
        for i in range(1000):
            q.put(i)
        assert q.qsize == 1000


# ── Priority tests ──────────────────────────────────────────────────


class TestPriority:
    def test_priority_ordering(self):
        q = ProducerConsumerQueue[str](priority=True, num_consumers=0, name="test-priority")
        q.put("low", priority=10)
        q.put("high", priority=1)
        q.put("medium", priority=5)

        _, item1 = q.get(timeout=1.0)
        _, item2 = q.get(timeout=1.0)
        _, item3 = q.get(timeout=1.0)

        assert item1 == "high"
        assert item2 == "medium"
        assert item3 == "low"

    def test_same_priority_fifo(self):
        q = ProducerConsumerQueue[str](priority=True, num_consumers=0, name="test-fifo")
        q.put("first", priority=1)
        q.put("second", priority=1)
        q.put("third", priority=1)

        _, a = q.get(timeout=1.0)
        _, b = q.get(timeout=1.0)
        _, c = q.get(timeout=1.0)

        assert a == "first"
        assert b == "second"
        assert c == "third"


# ── Consumer handler tests ──────────────────────────────────────────


class TestConsumerHandler:
    def test_handler_called(self):
        results = []
        lock = threading.Lock()

        def handler(item):
            with lock:
                results.append(item)

        q = ProducerConsumerQueue[int](num_consumers=2, handler=handler, name="test-handler")
        q.start()
        try:
            for i in range(10):
                q.put(i)
            time.sleep(0.5)
            with lock:
                assert sorted(results) == list(range(10))
        finally:
            q.stop()

    def test_handler_exception_counted(self):
        def bad_handler(item):
            raise ValueError("boom")

        q = ProducerConsumerQueue[int](num_consumers=1, handler=bad_handler, name="test-error")
        q.start()
        try:
            q.put(1)
            time.sleep(0.3)
            assert q.metrics["errors"] >= 1
        finally:
            q.stop()

    def test_multiple_consumers_process_all(self):
        results = []
        lock = threading.Lock()

        def handler(item):
            with lock:
                results.append(item)

        q = ProducerConsumerQueue[int](num_consumers=4, handler=handler, name="test-dist")
        q.start()
        try:
            for i in range(40):
                q.put(i)
            time.sleep(0.5)
            with lock:
                assert len(results) == 40
                assert sorted(results) == list(range(40))
        finally:
            q.stop()


# ── Shutdown tests ──────────────────────────────────────────────────


class TestShutdown:
    def test_drain_mode_finishes_work(self):
        results = []
        lock = threading.Lock()

        def handler(item):
            with lock:
                results.append(item)

        q = ProducerConsumerQueue[int](
            num_consumers=2, handler=handler,
            shutdown_mode=ShutdownMode.DRAIN, name="test-drain"
        )
        q.start()
        for i in range(10):
            q.put(i)
        q.stop(timeout=3.0)
        with lock:
            assert sorted(results) == list(range(10))

    def test_drop_mode_discards(self):
        started = threading.Event()

        def slow_handler(item):
            started.set()
            time.sleep(0.5)

        q = ProducerConsumerQueue[int](
            maxsize=100, num_consumers=1,
            handler=slow_handler,
            shutdown_mode=ShutdownMode.DROP, name="test-drop"
        )
        q.start()
        # Fill queue with items that consumer can't keep up with
        for i in range(20):
            q.put_nowait(i)
        time.sleep(0.05)
        q.stop(timeout=1.0)
        # Consumer was blocked on first item, rest were dropped
        assert q.metrics["dropped"] > 0

    def test_stop_is_idempotent(self):
        q = ProducerConsumerQueue[int](num_consumers=0, name="test-idempotent")
        q.start()
        q.stop()
        q.stop()  # should not raise


# ── Metrics tests ───────────────────────────────────────────────────


class TestMetrics:
    def test_metrics_tracking(self):
        results = []
        lock = threading.Lock()

        def handler(item):
            with lock:
                results.append(item)

        q = ProducerConsumerQueue[int](num_consumers=2, handler=handler, name="test-metrics")
        q.start()
        try:
            q.put(1)
            q.put(2)
            time.sleep(0.3)
            m = q.metrics
            assert m["enqueued"] == 2
            assert m["consumed"] == 2
            assert m["errors"] == 0
        finally:
            q.stop()

    def test_is_running(self):
        q = ProducerConsumerQueue[int](num_consumers=1, name="test-running")
        assert not q.is_running
        q.start()
        assert q.is_running
        q.stop()
        assert not q.is_running


# ── Thread safety stress test ───────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_producers(self):
        results = []
        lock = threading.Lock()

        def handler(item):
            with lock:
                results.append(item)

        q = ProducerConsumerQueue[int](num_consumers=4, handler=handler, name="test-stress")
        q.start()
        try:
            def producer(start):
                for i in range(50):
                    q.put(start + i)

            threads = [threading.Thread(target=producer, args=(i * 100,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            time.sleep(1.0)
            with lock:
                assert len(results) == 200
                assert len(set(results)) == 200  # all unique
        finally:
            q.stop()
