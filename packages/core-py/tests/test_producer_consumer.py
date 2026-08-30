"""Tests for ProducerConsumerQueue — thread-safe work distribution."""

import asyncio
import threading
import time

import pytest

from domains.infrastructure.producer_consumer import (
    ProducerConsumerQueue,
    ShutdownMode,
    _PriorityItem,
    get_producer_consumer_queue,
    set_producer_consumer_queue,
)


# ── ShutdownMode ──────────────────────────────────────────────────────


class TestShutdownMode:
    def test_drain_value(self):
        assert ShutdownMode.DRAIN.value == "drain"

    def test_drop_value(self):
        assert ShutdownMode.DROP.value == "drop"

    def test_drain_is_str_subclass(self):
        assert isinstance(ShutdownMode.DRAIN, str)

    def test_mode_comparison(self):
        assert ShutdownMode.DRAIN == "drain"
        assert ShutdownMode.DROP == "drop"


# ── PriorityItem ──────────────────────────────────────────────────────


class TestPriorityItem:
    def test_ordering_by_priority(self):
        a = _PriorityItem(priority=1, sequence=1, item="a")
        b = _PriorityItem(priority=2, sequence=2, item="b")
        assert a < b

    def test_same_priority_ordering_by_sequence(self):
        a = _PriorityItem(priority=1, sequence=1, item="a")
        b = _PriorityItem(priority=1, sequence=2, item="b")
        assert a < b

    def test_item_not_compared(self):
        a = _PriorityItem(priority=1, sequence=1, item="z")
        b = _PriorityItem(priority=1, sequence=1, item="a")
        assert not (a < b)
        assert not (a > b)

    def test_eq_same_priority_and_seq(self):
        a = _PriorityItem(priority=1, sequence=1, item="x")
        b = _PriorityItem(priority=1, sequence=1, item="y")
        assert a == b


# ── Sync Basic ────────────────────────────────────────────────────────


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

    def test_put_get_multiple_items(self):
        q = ProducerConsumerQueue[str](num_consumers=0, name="test-multi")
        for i in range(5):
            q.put(f"item-{i}")
        for i in range(5):
            ok, item = q.get(timeout=1.0)
            assert ok
            assert item == f"item-{i}"

    def test_put_returns_false_when_shutdown_with_consumers(self):
        q = ProducerConsumerQueue[int](num_consumers=1, name="test-put-shutdown")
        q.start()
        q.stop()
        assert not q.put(1)

    def test_get_returns_false_when_shutdown_and_empty_with_consumers(self):
        q = ProducerConsumerQueue[int](num_consumers=1, name="test-get-shutdown")
        q.start()
        q.stop()
        ok, item = q.get(timeout=0.05)
        assert not ok
        assert item is None

    def test_get_drains_remaining_items_after_stop(self):
        q = ProducerConsumerQueue[int](num_consumers=0, name="test-drain-manual")
        q.put(1)
        q.put(2)
        q.start()
        q.stop(timeout=1.0)
        ok1, item1 = q.get(timeout=0.1)
        ok2, item2 = q.get(timeout=0.1)
        assert ok1 and item1 == 1
        assert ok2 and item2 == 2

    def test_task_done_no_error(self):
        q = ProducerConsumerQueue[int](num_consumers=0, name="test-taskdone")
        q.task_done()

    def test_task_done_after_put(self):
        q = ProducerConsumerQueue[int](num_consumers=0, name="test-taskdone2")
        q.put(1)
        q.get(timeout=1.0)
        q.task_done()


# ── Backpressure ──────────────────────────────────────────────────────


class TestBackpressure:
    def test_full_queue_drops_item(self):
        q = ProducerConsumerQueue[int](maxsize=2, num_consumers=0, name="test-backpressure")
        q.put(1)
        q.put(2)
        assert not q.put_nowait(3)
        assert q.qsize == 2
        assert q.metrics["dropped"] == 1

    def test_unbounded_queue(self):
        q = ProducerConsumerQueue[int](maxsize=0, num_consumers=0, name="test-unbounded")
        for i in range(1000):
            q.put(i)
        assert q.qsize == 1000

    def test_full_property(self):
        q = ProducerConsumerQueue[int](maxsize=1, num_consumers=0, name="test-full")
        assert not q.full
        q.put(1)
        assert q.full

    def test_put_timeout_returns_false(self):
        q = ProducerConsumerQueue[int](maxsize=1, num_consumers=0, name="test-put-timeout")
        q.put(1)
        result = q.put(2, timeout=0.01)
        assert not result
        assert q.metrics["dropped"] == 1

    def test_unbounded_full_always_false(self):
        q = ProducerConsumerQueue[int](maxsize=0, num_consumers=0, name="test-unbounded-full")
        assert not q.full


# ── Priority ──────────────────────────────────────────────────────────


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

    def test_negative_priority(self):
        q = ProducerConsumerQueue[str](priority=True, num_consumers=0, name="test-neg")
        q.put("normal", priority=0)
        q.put("negative", priority=-5)
        _, item = q.get(timeout=1.0)
        assert item == "negative"

    def test_priority_put_nowait(self):
        q = ProducerConsumerQueue[str](priority=True, num_consumers=0, name="test-pnowait")
        q.put_nowait("a", priority=10)
        q.put_nowait("b", priority=1)
        _, item = q.get(timeout=1.0)
        assert item == "b"

    def test_priority_with_consumers(self):
        results = []
        lock = threading.Lock()

        def handler(item):
            with lock:
                results.append(item)

        q = ProducerConsumerQueue[str](
            priority=True, num_consumers=1, handler=handler, name="test-pcons"
        )
        q.start()
        try:
            q.put("low", priority=10)
            q.put("high", priority=1)
            time.sleep(0.3)
            with lock:
                assert results[0] == "high"
        finally:
            q.stop()


# ── Consumer Handler ──────────────────────────────────────────────────


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

    def test_handler_receives_all_items_exactly_once(self):
        received = []
        lock = threading.Lock()

        def handler(item):
            with lock:
                received.append(item)

        q = ProducerConsumerQueue[int](num_consumers=2, handler=handler, name="test-once")
        q.start()
        try:
            items = list(range(20))
            for i in items:
                q.put(i)
            time.sleep(0.5)
            with lock:
                assert sorted(received) == items
        finally:
            q.stop()

    def test_handler_does_not_block_other_consumers(self):
        slow_done = threading.Event()
        fast_results = []
        lock = threading.Lock()

        def slow_handler(item):
            if item == 0:
                time.sleep(0.3)
                slow_done.set()

        def fast_handler(item):
            if item != 0:
                with lock:
                    fast_results.append(item)

        q = ProducerConsumerQueue[int](
            num_consumers=2, handler=lambda i: None, name="test-noblock"
        )
        results2 = []
        lock2 = threading.Lock()

        def combined(item):
            if item == 0:
                time.sleep(0.2)
            else:
                with lock2:
                    results2.append(item)

        q2 = ProducerConsumerQueue[int](num_consumers=2, handler=combined, name="test-noblock2")
        q2.start()
        try:
            q2.put(0)
            for i in range(1, 10):
                q2.put(i)
            time.sleep(0.5)
            with lock2:
                assert len(results2) == 9
        finally:
            q2.stop()


# ── Shutdown ──────────────────────────────────────────────────────────


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
        for i in range(20):
            q.put_nowait(i)
        time.sleep(0.05)
        q.stop(timeout=1.0)
        assert q.metrics["dropped"] > 0

    def test_stop_is_idempotent(self):
        q = ProducerConsumerQueue[int](num_consumers=0, name="test-idempotent")
        q.start()
        q.stop()
        q.stop()

    def test_start_is_idempotent(self):
        q = ProducerConsumerQueue[int](num_consumers=2, name="test-start-idem")
        q.start()
        q.start()
        assert len(q._consumers) == 2
        q.stop()

    def test_stop_without_start(self):
        q = ProducerConsumerQueue[int](num_consumers=2, name="test-nostart")
        q.stop()

    def test_consumer_threads_are_daemon(self):
        q = ProducerConsumerQueue[int](num_consumers=2, name="test-daemon")
        q.start()
        for t in q._consumers:
            assert t.daemon is True
        q.stop()

    def test_consumer_threads_named(self):
        q = ProducerConsumerQueue[int](num_consumers=3, name="test-named")
        q.start()
        names = [t.name for t in q._consumers]
        assert names == ["test-named-consumer-0", "test-named-consumer-1", "test-named-consumer-2"]
        q.stop()

    def test_drain_timeout_stops_anyway(self):
        def slow(item):
            time.sleep(2.0)

        q = ProducerConsumerQueue[int](
            num_consumers=1, handler=slow,
            shutdown_mode=ShutdownMode.DRAIN, name="test-drain-timeout"
        )
        q.start()
        q.put(1)
        time.sleep(0.05)
        q.stop(timeout=0.1)
        assert not q.is_running


# ── Metrics ───────────────────────────────────────────────────────────


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

    def test_metrics_initial_state(self):
        q = ProducerConsumerQueue[int](num_consumers=0, name="test-init-metrics")
        m = q.metrics
        assert m["enqueued"] == 0
        assert m["consumed"] == 0
        assert m["dropped"] == 0
        assert m["errors"] == 0
        assert m["queued"] == 0
        assert m["active_consumers"] == 0

    def test_active_consumers_count(self):
        def noop(item):
            time.sleep(0.1)

        q = ProducerConsumerQueue[int](num_consumers=2, handler=noop, name="test-active")
        q.start()
        time.sleep(0.1)
        assert q.active_consumers == 2
        q.stop()

    def test_active_consumers_zero_when_stopped(self):
        q = ProducerConsumerQueue[int](num_consumers=2, name="test-active0")
        assert q.active_consumers == 0

    def test_repr(self):
        q = ProducerConsumerQueue[int](num_consumers=2, name="test-repr")
        r = repr(q)
        assert "ProducerConsumerQueue" in r
        assert "test-repr" in r
        assert "consumers=2" in r

    def test_repr_priority(self):
        q = ProducerConsumerQueue[int](num_consumers=1, priority=True, name="test-repr-p")
        r = repr(q)
        assert "priority=True" in r

    def test_repr_no_priority(self):
        q = ProducerConsumerQueue[int](num_consumers=1, priority=False, name="test-repr-np")
        r = repr(q)
        assert "priority=False" in r

    def test_metrics_dropped(self):
        q = ProducerConsumerQueue[int](maxsize=1, num_consumers=0, name="test-metrics-drop")
        q.put_nowait(1)
        q.put_nowait(2)
        m = q.metrics
        assert m["dropped"] == 1

    def test_metrics_error_count(self):
        def bad(item):
            time.sleep(0.01)
            raise RuntimeError("fail")

        q = ProducerConsumerQueue[int](num_consumers=1, handler=bad, name="test-metrics-err")
        q.start()
        q.put(1)
        q.put(2)
        time.sleep(0.5)
        m = q.metrics
        assert m["errors"] >= 2
        q.stop()


# ── Async API ─────────────────────────────────────────────────────────


class TestAsyncAPI:
    def test_async_put_and_get(self):
        q = ProducerConsumerQueue[str](num_consumers=0, name="test-async")

        async def run():
            await q.async_put("hello")
            ok, item = await q.async_get(timeout=1.0)
            return ok, item

        ok, item = asyncio.run(run())
        assert ok
        assert item == "hello"

    def test_async_get_timeout(self):
        q = ProducerConsumerQueue[int](num_consumers=0, name="test-async-timeout")

        async def run():
            return await q.async_get(timeout=0.05)

        ok, item = asyncio.run(run())
        assert not ok
        assert item is None


# ── Global Queue ──────────────────────────────────────────────────────


class TestGlobalQueue:
    def test_get_creates_singleton(self):
        set_producer_consumer_queue(None)
        q = get_producer_consumer_queue()
        assert q is not None
        assert q.name == "global"

    def test_get_returns_same_instance(self):
        set_producer_consumer_queue(None)
        q1 = get_producer_consumer_queue()
        q2 = get_producer_consumer_queue()
        assert q1 is q2

    def test_set_replaces_global(self):
        custom = ProducerConsumerQueue[int](num_consumers=0, name="custom")
        set_producer_consumer_queue(custom)
        q = get_producer_consumer_queue()
        assert q is custom
        assert q.name == "custom"


# ── Thread Safety ─────────────────────────────────────────────────────


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
                assert len(set(results)) == 200
        finally:
            q.stop()

    def test_concurrent_gets(self):
        q = ProducerConsumerQueue[int](num_consumers=0, name="test-conc-get")
        for i in range(10):
            q.put(i)

        results = []
        lock = threading.Lock()

        def getter():
            ok, item = q.get(timeout=1.0)
            if ok:
                with lock:
                    results.append(item)

        threads = [threading.Thread(target=getter) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        with lock:
            assert sorted(results) == list(range(10))

    def test_put_get_interleaved(self):
        q = ProducerConsumerQueue[int](num_consumers=0, name="test-interleave")
        results = []

        for i in range(5):
            q.put(i)
            ok, item = q.get(timeout=1.0)
            results.append(item)

        assert results == list(range(5))
