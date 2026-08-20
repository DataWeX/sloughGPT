"""Meaningful tests for PriorityRequestQueue and SessionKVCache — async queue ordering, metrics, cache prefix matching, TTL, LRU."""

import asyncio
import time
import pytest
from domains.infrastructure.model_server import (
    PriorityRequestQueue, Priority, QueueMetrics, _QueueItem,
    SessionKVCache,
)


# ── PriorityRequestQueue ──────────────────────────────────────────────


class TestPriorityOrdering:
    @pytest.mark.asyncio
    async def test_submit_returns_result(self):
        q = PriorityRequestQueue(max_concurrent=2)

        async def worker_task():
            await q.worker()

        wt = asyncio.create_task(worker_task())

        async def compute():
            return 42

        result = await q.submit(compute(), Priority.MEDIUM, "r1")
        assert result == 42

        wt.cancel()
        try:
            await wt
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_multiple_sequential(self):
        q = PriorityRequestQueue(max_concurrent=2)

        async def wt():
            await q.worker()

        worker = asyncio.create_task(wt())

        results = []
        for i in range(5):
            async def val(v=i):
                return v
            r = await q.submit(val(), Priority.MEDIUM, f"r{i}")
            results.append(r)

        assert results == [0, 1, 2, 3, 4]

        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        q = PriorityRequestQueue(max_concurrent=3)

        async def wt():
            await q.worker()

        worker = asyncio.create_task(wt())

        async def slow_task(val):
            await asyncio.sleep(0.01)
            return val

        tasks = [q.submit(slow_task(i), Priority.MEDIUM, f"c{i}") for i in range(3)]
        results = await asyncio.gather(*tasks)
        assert sorted(results) == [0, 1, 2]

        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass


class TestQueueMetrics:
    def test_metrics_initial(self):
        q = PriorityRequestQueue()
        m = q.metrics_snapshot()
        assert m.total_depth == 0
        assert m.served == 0

    @pytest.mark.asyncio
    async def test_metrics_after_serve(self):
        q = PriorityRequestQueue(max_concurrent=2)

        async def wt():
            await q.worker()

        worker = asyncio.create_task(wt())

        async def compute():
            return "done"

        await q.submit(compute(), Priority.HIGH, "m1")
        await asyncio.sleep(0.05)

        m = q.metrics_snapshot()
        assert m.served >= 1

        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass


class TestQueueClose:
    @pytest.mark.asyncio
    async def test_close_clears_heap(self):
        q = PriorityRequestQueue(max_concurrent=1, max_queue=10)

        async def slow():
            await asyncio.sleep(10)

        future = asyncio.get_running_loop().create_future()
        item = _QueueItem(
            priority=Priority.LOW.value,
            enqueue_order=0,
            coro=slow(),
            future=future,
        )
        q._heap.append(item)

        q.close()
        assert len(q._heap) == 0
        assert future.cancelled()


class TestQueueFull:
    @pytest.mark.asyncio
    async def test_submit_raises_when_full(self):
        q = PriorityRequestQueue(max_concurrent=1, max_queue=1)
        # Fill the queue
        async def slow():
            await asyncio.sleep(10)

        future = asyncio.get_running_loop().create_future()
        item = _QueueItem(
            priority=Priority.LOW.value,
            enqueue_order=0,
            coro=slow(),
            future=future,
        )
        q._heap.append(item)

        with pytest.raises(RuntimeError, match="Queue full"):
            await q.submit(asyncio.sleep(0), Priority.LOW, "overflow")


# ── SessionKVCache ────────────────────────────────────────────────────

class TestSessionKVCacheGet:
    def test_get_empty(self):
        c = SessionKVCache()
        pkv, plen = c.get("s1", [1, 2, 3])
        assert pkv is None
        assert plen == 0

    def test_get_matching_prefix(self):
        c = SessionKVCache()
        c.store("s1", [1, 2, 3, 4], "pkv_data")
        pkv, plen = c.get("s1", [1, 2, 3, 4, 5, 6])
        assert pkv == "pkv_data"
        assert plen == 4

    def test_get_no_prefix_match(self):
        c = SessionKVCache()
        c.store("s1", [1, 2, 3], "pkv_data")
        pkv, plen = c.get("s1", [9, 8, 7])
        assert pkv is None
        assert plen == 0

    def test_get_partial_prefix(self):
        c = SessionKVCache()
        c.store("s1", [1, 2, 3], "pkv_data")
        pkv, plen = c.get("s1", [1, 2, 99])
        assert pkv == "pkv_data"
        assert plen == 2

    def test_get_different_sessions(self):
        c = SessionKVCache()
        c.store("s1", [1, 2, 3], "pkv1")
        c.store("s2", [1, 2, 3], "pkv2")
        pkv, plen = c.get("s2", [1, 2, 3, 4])
        assert pkv == "pkv2"

    def test_get_shorter_than_cached(self):
        c = SessionKVCache()
        c.store("s1", [1, 2, 3, 4, 5], "pkv_data")
        pkv, plen = c.get("s1", [1, 2])
        assert pkv == "pkv_data"
        assert plen == 2


class TestSessionKVCacheStore:
    def test_store_and_size(self):
        c = SessionKVCache()
        c.store("s1", [1], "d1")
        assert c.size == 1

    def test_store_lru_eviction(self):
        c = SessionKVCache(max_sessions=2)
        c.store("s1", [1], "d1")
        c.store("s2", [2], "d2")
        c.store("s3", [3], "d3")
        assert c.size == 2
        # s1 should be evicted (oldest)
        assert c.get("s1", [1]) == (None, 0)

    def test_store_overwrite(self):
        c = SessionKVCache()
        c.store("s1", [1], "old")
        c.store("s1", [1], "new")
        pkv, _ = c.get("s1", [1])
        assert pkv == "new"


class TestSessionKVCacheTTL:
    def test_expired_entry_gone_after_evict(self):
        c = SessionKVCache(ttl=0.01)
        c.store("s1", [1], "d1")
        time.sleep(0.02)
        c.evict_expired()
        pkv, plen = c.get("s1", [1])
        assert pkv is None

    def test_evict_expired(self):
        c = SessionKVCache(ttl=0.01)
        c.store("s1", [1], "d1")
        c.store("s2", [2], "d2")
        time.sleep(0.02)
        c.evict_expired()
        assert c.size == 0

    def test_non_expired_stays(self):
        c = SessionKVCache(ttl=10.0)
        c.store("s1", [1], "d1")
        c.evict_expired()
        assert c.size == 1


class TestSessionKVCacheClear:
    def test_clear(self):
        c = SessionKVCache()
        c.store("s1", [1], "d1")
        c.clear("s1")
        assert c.size == 0

    def test_clear_nonexistent(self):
        c = SessionKVCache()
        c.clear("nope")  # Should not raise


class TestSessionKVCacheStats:
    def test_stats(self):
        c = SessionKVCache(max_sessions=10, ttl=120.0)
        c.store("s1", [1], "d1")
        s = c.stats()
        assert s["entries"] == 1
        assert s["max_sessions"] == 10
        assert s["ttl_seconds"] == 120.0
