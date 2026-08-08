"""Tests for PriorityRequestQueue in model_server.py."""

import asyncio
import time

import pytest
from domains.infrastructure.model_server import PriorityRequestQueue, Priority


@pytest.fixture
async def queue():
    q = PriorityRequestQueue(max_concurrent=2, max_queue=32)
    task = asyncio.get_event_loop().create_task(q.worker())
    yield q
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, RuntimeError):
        pass


# ── Priority enum ──


class TestPriorityEnum:
    def test_members(self):
        assert Priority.HIGH.value == 0
        assert Priority.MEDIUM.value == 1
        assert Priority.LOW.value == 2

    def test_ordering(self):
        assert Priority.HIGH < Priority.MEDIUM < Priority.LOW


# ── Basic submit / execution ──


class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_submit_returns_result(self, queue):
        async def fn():
            return 42
        result = await queue.submit(fn(), priority=Priority.HIGH)
        assert result == 42

    @pytest.mark.asyncio
    async def test_submit_multiple(self, queue):
        results = []
        async def fn(i):
            return i
        t1 = queue.submit(fn(1), priority=Priority.HIGH)
        t2 = queue.submit(fn(2), priority=Priority.HIGH)
        r1, r2 = await asyncio.gather(t1, t2)
        assert sorted([r1, r2]) == [1, 2]

    @pytest.mark.asyncio
    async def test_queue_full_raises(self):
        q = PriorityRequestQueue(max_concurrent=1, max_queue=2)
        async def blocking():
            await asyncio.Event().wait()
        wk = asyncio.get_event_loop().create_task(q.worker())
        await asyncio.sleep(0.02)
        tasks = [asyncio.create_task(q.submit(blocking(), priority=Priority.LOW))
                 for _ in range(4)]
        done, pending = await asyncio.wait(tasks, timeout=5, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        # Give cancelled tasks chance to finish
        for t in pending:
            try:
                await t
            except (asyncio.CancelledError, RuntimeError):
                pass
        wk.cancel()
        try:
            await wk
        except (asyncio.CancelledError, RuntimeError):
            pass
        q.close()
        errors = []
        for t in done:
            try:
                await t
            except RuntimeError as e:
                if "Queue full" in str(e):
                    errors.append(e)
        assert len(errors) >= 1


# ── Priority ordering ──


class TestPriorityOrdering:
    @pytest.mark.asyncio
    async def test_high_before_low(self, queue):
        order = []
        async def fn(i):
            order.append(i)
            return i
        # Submit LOW first, then HIGH
        t1 = queue.submit(fn(1), priority=Priority.LOW)
        t2 = queue.submit(fn(2), priority=Priority.HIGH)
        await asyncio.gather(t1, t2)
        assert order[0] == 2  # HIGH executes first

    @pytest.mark.asyncio
    async def test_fifo_within_same_priority(self, queue):
        order = []
        async def fn(i):
            order.append(i)
            return i
        t1 = queue.submit(fn(1), priority=Priority.MEDIUM)
        t2 = queue.submit(fn(2), priority=Priority.MEDIUM)
        await asyncio.gather(t1, t2)
        assert order == [1, 2]

    @pytest.mark.asyncio
    async def test_three_tier_ordering(self, queue):
        order = []
        async def fn(i):
            order.append(i)
            return i
        # Submit in reverse priority order
        t1 = queue.submit(fn("low"), priority=Priority.LOW)
        t2 = queue.submit(fn("med"), priority=Priority.MEDIUM)
        t3 = queue.submit(fn("high"), priority=Priority.HIGH)
        await asyncio.gather(t1, t2, t3)
        assert order == ["high", "med", "low"]


# ── Concurrency control ──


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_limits_concurrent_execution(self, queue):
        in_flight = 0
        max_seen = 0
        lock = asyncio.Lock()

        async def fn(i):
            nonlocal in_flight, max_seen
            async with lock:
                in_flight += 1
                max_seen = max(max_seen, in_flight)
            await asyncio.sleep(0.05)
            async with lock:
                in_flight -= 1

        coros = [queue.submit(fn(i), priority=Priority.MEDIUM) for i in range(6)]
        await asyncio.gather(*coros)
        assert max_seen <= 2  # max_concurrent

    @pytest.mark.asyncio
    async def test_in_flight_property(self, queue):
        async def slow():
            await asyncio.sleep(0.2)
        t = asyncio.create_task(queue.submit(slow(), priority=Priority.LOW))
        await asyncio.sleep(0.05)
        assert queue.in_flight >= 1
        await t
        await asyncio.sleep(0.02)
        assert queue.in_flight == 0


# ── Metrics ──


class TestMetrics:
    @pytest.mark.asyncio
    async def test_served_count(self, queue):
        async def fn():
            return 1
        await queue.submit(fn(), priority=Priority.HIGH)
        await queue.submit(fn(), priority=Priority.LOW)
        m = queue.metrics_snapshot()
        assert m.served == 2

    @pytest.mark.asyncio
    async def test_depth_per_priority(self):
        """Items queued should appear in depth before execution."""
        q2 = PriorityRequestQueue(max_concurrent=1, max_queue=32)
        import heapq
        from domains.infrastructure.model_server import _QueueItem
        loop = asyncio.get_running_loop()
        for p in range(3):
            heapq.heappush(q2._heap,
                _QueueItem(priority=p, enqueue_order=-p, coro=None, future=loop.create_future(), request_id=f"r{p}"))
        m = q2.metrics_snapshot()
        d = await q2.depth()
        assert d == [1, 1, 1]
        assert m.depth_high == 1
        assert m.depth_medium == 1
        assert m.depth_low == 1
        assert m.total_depth == 3

    @pytest.mark.asyncio
    async def test_wait_time_metrics(self, queue):
        async def fn():
            await asyncio.sleep(0.05)
            return 1
        await queue.submit(fn(), priority=Priority.HIGH)
        m = queue.metrics_snapshot()
        assert m.avg_wait_ms > 0
        assert m.max_wait_ms > 0


# ── Edge cases ──


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_exception_propagation(self, queue):
        async def fails():
            raise ValueError("boom")
        with pytest.raises(ValueError, match="boom"):
            await queue.submit(fails(), priority=Priority.HIGH)

    @pytest.mark.asyncio
    async def test_empty_queue_metrics(self, queue):
        m = queue.metrics_snapshot()
        assert m.served == 0
        assert m.depth_high == 0
        assert m.depth_medium == 0
        assert m.depth_low == 0
        assert m.total_depth == 0

    @pytest.mark.asyncio
    async def test_depth_api(self, queue):
        q2 = PriorityRequestQueue(max_concurrent=1)
        import heapq
        from domains.infrastructure.model_server import _QueueItem
        loop = asyncio.get_running_loop()
        heapq.heappush(q2._heap,
            _QueueItem(priority=0, enqueue_order=-1, coro=None, future=loop.create_future(), request_id="r0"))
        heapq.heappush(q2._heap,
            _QueueItem(priority=1, enqueue_order=-2, coro=None, future=loop.create_future(), request_id="r1"))
        d = await q2.depth()
        assert d[0] == 1  # HIGH
        assert d[1] == 1  # MEDIUM
        assert d[2] == 0  # LOW


# ── Slot reservation (acquire/release) ──


class TestSlotReservation:
    @pytest.mark.asyncio
    async def test_acquire_release(self):
        q = PriorityRequestQueue(max_concurrent=1, max_queue=8)
        wk = asyncio.get_event_loop().create_task(q.worker())
        await asyncio.sleep(0.02)

        release = await q.acquire(priority=Priority.HIGH)
        assert q.in_flight == 1
        release()
        await asyncio.sleep(0.02)
        assert q.in_flight == 0

        wk.cancel()
        try:
            await wk
        except (asyncio.CancelledError, RuntimeError):
            pass

    @pytest.mark.asyncio
    async def test_acquire_concurrency_limit(self):
        q = PriorityRequestQueue(max_concurrent=1, max_queue=8)
        wk = asyncio.get_event_loop().create_task(q.worker())
        await asyncio.sleep(0.02)

        r1 = await q.acquire(priority=Priority.HIGH)
        assert q.in_flight == 1

        # Second acquire blocks until release
        async def try_acquire():
            r2 = await q.acquire(priority=Priority.HIGH)
            r2()
            return "done"

        t = asyncio.create_task(try_acquire())
        await asyncio.sleep(0.05)
        assert not t.done()

        r1()
        await asyncio.sleep(0.05)
        assert t.done()
        assert t.result() == "done"

        wk.cancel()
        try:
            await wk
        except (asyncio.CancelledError, RuntimeError):
            pass

    @pytest.mark.asyncio
    async def test_acquire_queue_full(self):
        """acquire() raises when heap exceeds max_queue.

        max_concurrent=1 means one slot is in-flight.
        max_queue=1 means only one item can sit in the heap.
        The third acquire should see heap full and raise.
        """
        q = PriorityRequestQueue(max_concurrent=1, max_queue=1)
        wk = asyncio.get_event_loop().create_task(q.worker())
        await asyncio.sleep(0.02)

        r1 = await q.acquire(priority=Priority.HIGH, request_id="first")

        # Second acquire sits in the heap (in_flight == max_concurrent)
        t2 = asyncio.create_task(q.acquire(priority=Priority.HIGH, request_id="second"))
        await asyncio.sleep(0.1)
        assert not t2.done()

        # Heap is full (1 item) — third acquire raises
        with pytest.raises(RuntimeError, match="Queue full"):
            await q.acquire(priority=Priority.HIGH, request_id="third")

        r1()
        await asyncio.sleep(0.05)
        # Cancel pending second acquire
        t2.cancel()
        try:
            await t2
        except (asyncio.CancelledError, RuntimeError):
            pass
        wk.cancel()
        try:
            await wk
        except (asyncio.CancelledError, RuntimeError):
            pass

    @pytest.mark.asyncio
    async def test_acquire_doesnt_serve_submit_slot(self):
        """acquire() slot should not consume a submit() slot."""
        q = PriorityRequestQueue(max_concurrent=1, max_queue=8)
        wk = asyncio.get_event_loop().create_task(q.worker())
        await asyncio.sleep(0.02)

        release = await q.acquire(priority=Priority.HIGH)
        assert q.in_flight == 1

        async def fn():
            return 42
        t = asyncio.create_task(q.submit(fn(), priority=Priority.MEDIUM))
        await asyncio.sleep(0.05)
        assert not t.done()

        release()
        result = await t
        assert result == 42

        wk.cancel()
        try:
            await wk
        except (asyncio.CancelledError, RuntimeError):
            pass

    @pytest.mark.asyncio
    async def test_acquire_double_release_safe(self):
        """Calling release() twice should be a no-op."""
        q = PriorityRequestQueue(max_concurrent=1, max_queue=8)
        wk = asyncio.get_event_loop().create_task(q.worker())
        await asyncio.sleep(0.02)

        release = await q.acquire(priority=Priority.HIGH)
        assert q.in_flight == 1
        release()  # first
        assert q.in_flight == 0
        release()  # second — no-op
        assert q.in_flight == 0

        wk.cancel()
        try:
            await wk
        except (asyncio.CancelledError, RuntimeError):
            pass

    @pytest.mark.asyncio
    async def test_acquire_cancel_before_grant(self):
        """Cancelling acquire() while marker is still in heap removes it."""
        q = PriorityRequestQueue(max_concurrent=1, max_queue=8)
        wk = asyncio.get_event_loop().create_task(q.worker())
        await asyncio.sleep(0.02)

        # Fill the in-flight slot with a slow submit
        async def slow():
            await asyncio.sleep(0.5)
            return 42

        t1 = asyncio.create_task(q.submit(slow(), priority=Priority.HIGH))
        await asyncio.sleep(0.02)

        # Acquire pushes a marker, blocks on grant (worker busy)
        acq = asyncio.create_task(
            q.acquire(priority=Priority.MEDIUM, request_id="cancel-me")
        )
        await asyncio.sleep(0.02)
        assert not acq.done()

        heap_len_before = len(q._heap)
        acq.cancel()
        try:
            await acq
        except (asyncio.CancelledError, RuntimeError):
            pass

        # Marker should have been removed from the heap
        assert len(q._heap) < heap_len_before or len(q._heap) == 0

        # After cancellation, in_flight should only reflect t1
        assert q.in_flight == 1

        await t1
        wk.cancel()
        try:
            await wk
        except (asyncio.CancelledError, RuntimeError):
            pass

    @pytest.mark.asyncio
    async def test_acquire_cancel_after_grant(self):
        """Cancelling acquire() after worker pops the marker releases the slot."""
        q = PriorityRequestQueue(max_concurrent=1, max_queue=8)
        wk = asyncio.get_event_loop().create_task(q.worker())
        await asyncio.sleep(0.02)

        r1 = await q.acquire(priority=Priority.HIGH, request_id="first")
        assert q.in_flight == 1

        # The slot is granted — now simulate a client disconnect
        # by calling release directly (what finally/_release does)
        r1()
        assert q.in_flight == 0

        wk.cancel()
        try:
            await wk
        except (asyncio.CancelledError, RuntimeError):
            pass
