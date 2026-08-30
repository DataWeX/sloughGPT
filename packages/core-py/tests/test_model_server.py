"""Comprehensive tests for model_server.py — PriorityRequestQueue, SessionKVCache,
ModelMetrics, ModelStatus, Priority, QueueMetrics.

Covers: priority queue ordering, submit/acquire, cache get/store/clear/evict,
metrics recording, snapshot, reset, idle management.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from domains.infrastructure.model_server import (
    Priority,
    PriorityRequestQueue,
    QueueMetrics,
    SessionKVCache,
    ModelMetrics,
    ModelStatus,
    _is_intel_mac,
)


# ---------------------------------------------------------------------------
# Priority enum
# ---------------------------------------------------------------------------

class TestPriority:
    def test_ordering(self):
        assert Priority.HIGH < Priority.MEDIUM < Priority.LOW

    def test_values(self):
        assert Priority.HIGH.value == 0
        assert Priority.MEDIUM.value == 1
        assert Priority.LOW.value == 2


# ---------------------------------------------------------------------------
# ModelStatus enum
# ---------------------------------------------------------------------------

class TestModelStatus:
    def test_all_states(self):
        states = [s.value for s in ModelStatus]
        assert "uninitialized" in states
        assert "loading" in states
        assert "ready" in states
        assert "degraded" in states
        assert "error" in states
        assert "unloaded" in states


# ---------------------------------------------------------------------------
# ModelMetrics
# ---------------------------------------------------------------------------

class TestModelMetrics:
    def test_record_success(self):
        m = ModelMetrics()
        m.record_success(100.0, 50)
        assert m.requests_completed == 1
        assert m.tokens_generated_total == 50
        assert m.max_generation_time_ms == 100.0
        assert m.min_generation_time_ms == 100.0
        assert m.consecutive_failures == 0

    def test_record_failure(self):
        m = ModelMetrics()
        m.record_failure("OOM")
        assert m.requests_failed == 1
        assert m.last_error == "OOM"
        assert m.consecutive_failures == 1

    def test_record_timeout(self):
        m = ModelMetrics()
        m.record_timeout()
        assert m.requests_timed_out == 1
        assert m.consecutive_failures == 1

    def test_avg_generation_time_ms(self):
        m = ModelMetrics()
        m.record_success(100.0, 10)
        m.record_success(200.0, 20)
        assert m.avg_generation_time_ms == 150.0

    def test_avg_generation_time_ms_zero(self):
        m = ModelMetrics()
        assert m.avg_generation_time_ms == 0.0

    def test_error_rate(self):
        m = ModelMetrics()
        m.requests_total = 10
        m.requests_failed = 3
        assert m.error_rate == 0.3

    def test_error_rate_zero(self):
        m = ModelMetrics()
        assert m.error_rate == 0.0

    def test_snapshot(self):
        m = ModelMetrics()
        m.record_success(50.0, 10)
        snap = m.snapshot()
        assert snap["requests_completed"] == 1
        assert snap["tokens_generated_total"] == 10
        assert "error_rate" in snap
        assert "avg_generation_time_ms" in snap

    def test_snapshot_inf_min(self):
        m = ModelMetrics()
        snap = m.snapshot()
        assert snap["min_generation_time_ms"] == 0.0

    def test_reset(self):
        m = ModelMetrics()
        m.record_success(100.0, 50)
        m.record_failure("err")
        m.reset()
        assert m.requests_completed == 0
        assert m.requests_failed == 0
        assert m.tokens_generated_total == 0

    def test_multiple_successes_min_max(self):
        m = ModelMetrics()
        m.record_success(50.0, 10)
        m.record_success(200.0, 20)
        m.record_success(100.0, 15)
        assert m.min_generation_time_ms == 50.0
        assert m.max_generation_time_ms == 200.0

    def test_consecutive_failures_reset_on_success(self):
        m = ModelMetrics()
        m.record_failure("e1")
        m.record_failure("e2")
        assert m.consecutive_failures == 2
        m.record_success(10.0, 1)
        assert m.consecutive_failures == 0

    def test_last_request_time_set(self):
        m = ModelMetrics()
        before = time.time()
        m.record_success(10.0, 1)
        after = time.time()
        assert before <= m.last_request_time <= after


# ---------------------------------------------------------------------------
# QueueMetrics
# ---------------------------------------------------------------------------

class TestQueueMetrics:
    def test_defaults(self):
        qm = QueueMetrics()
        assert qm.depth_high == 0
        assert qm.total_depth == 0
        assert qm.avg_wait_ms == 0.0


# ---------------------------------------------------------------------------
# SessionKVCache
# ---------------------------------------------------------------------------

class TestSessionKVCache:
    def test_store_and_get(self):
        cache = SessionKVCache(max_sessions=10, ttl=600.0)
        cache.store("s1", [1, 2, 3], "pkv_data")
        pkv, prefix_len = cache.get("s1", [1, 2, 3, 4])
        assert pkv == "pkv_data"
        assert prefix_len == 3

    def test_get_no_match(self):
        cache = SessionKVCache()
        cache.store("s1", [1, 2], "data")
        pkv, prefix_len = cache.get("s1", [9, 8])
        assert pkv is None
        assert prefix_len == 0

    def test_get_unknown_session(self):
        cache = SessionKVCache()
        pkv, prefix_len = cache.get("unknown", [1])
        assert pkv is None
        assert prefix_len == 0

    def test_get_partial_prefix(self):
        cache = SessionKVCache()
        cache.store("s1", [1, 2, 3], "data")
        pkv, prefix_len = cache.get("s1", [1, 2, 9])
        assert pkv == "data"
        assert prefix_len == 2

    def test_clear(self):
        cache = SessionKVCache()
        cache.store("s1", [1], "data")
        cache.clear("s1")
        pkv, _ = cache.get("s1", [1])
        assert pkv is None

    def test_clear_unknown(self):
        cache = SessionKVCache()
        cache.clear("nonexistent")  # should not raise

    def test_lru_eviction(self):
        cache = SessionKVCache(max_sessions=2, ttl=600.0)
        cache.store("s1", [1], "d1")
        time.sleep(0.01)
        cache.store("s2", [2], "d2")
        time.sleep(0.01)
        cache.store("s3", [3], "d3")
        assert cache.size == 2
        pkv, _ = cache.get("s1", [1])
        assert pkv is None  # s1 was evicted

    def test_ttl_eviction(self):
        cache = SessionKVCache(max_sessions=100, ttl=0.01)
        cache.store("s1", [1], "data")
        time.sleep(0.02)
        cache.evict_expired()
        assert cache.size == 0

    def test_stats(self):
        cache = SessionKVCache(max_sessions=5, ttl=10.0)
        stats = cache.stats()
        assert stats["entries"] == 0
        assert stats["max_sessions"] == 5
        assert stats["ttl_seconds"] == 10.0

    def test_size(self):
        cache = SessionKVCache()
        assert cache.size == 0
        cache.store("s1", [1], "d1")
        assert cache.size == 1
        cache.store("s2", [2], "d2")
        assert cache.size == 2

    def test_overwrite_same_session(self):
        cache = SessionKVCache()
        cache.store("s1", [1], "old")
        cache.store("s1", [2], "new")
        pkv, _ = cache.get("s1", [2])
        assert pkv == "new"
        assert cache.size == 1


# ---------------------------------------------------------------------------
# PriorityRequestQueue — async tests
# ---------------------------------------------------------------------------

class TestPriorityRequestQueue:
    @pytest.mark.asyncio
    async def test_submit_simple(self):
        q = PriorityRequestQueue(max_concurrent=2)

        async def worker():
            return 42

        async def run():
            workers = [asyncio.create_task(q.worker()) for _ in range(2)]
            result = await q.submit(worker(), Priority.MEDIUM)
            q.close()
            for w in workers:
                w.cancel()
                try:
                    await w
                except asyncio.CancelledError:
                    pass
            return result

        result = await asyncio.wait_for(run(), timeout=5.0)
        assert result == 42

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        q = PriorityRequestQueue(max_concurrent=1)
        results = []

        async def make_task(val, priority):
            await asyncio.sleep(0.05)
            results.append(val)
            return val

        async def run():
            workers = [asyncio.create_task(q.worker()) for _ in range(1)]
            # Submit low priority first, then high
            f1 = asyncio.create_task(q.submit(make_task(1, Priority.LOW), Priority.LOW))
            await asyncio.sleep(0.02)
            f2 = asyncio.create_task(q.submit(make_task(0, Priority.HIGH), Priority.HIGH))
            await asyncio.sleep(0.2)
            q.close()
            for w in workers:
                w.cancel()
                try:
                    await w
                except asyncio.CancelledError:
                    pass
            await asyncio.gather(f1, f2, return_exceptions=True)

        await asyncio.wait_for(run(), timeout=5.0)
        # High priority should be served first
        assert results[0] == 0

    @pytest.mark.asyncio
    async def test_metrics_snapshot(self):
        q = PriorityRequestQueue(max_concurrent=2)
        snap = q.metrics_snapshot()
        assert isinstance(snap, QueueMetrics)
        assert snap.served == 0

    @pytest.mark.asyncio
    async def test_depth(self):
        q = PriorityRequestQueue(max_concurrent=1)

        async def slow():
            await asyncio.sleep(1.0)

        async def run():
            workers = [asyncio.create_task(q.worker()) for _ in range(1)]
            # Fill the concurrency slot
            f = asyncio.create_task(q.submit(slow(), Priority.MEDIUM))
            await asyncio.sleep(0.05)
            d = await q.depth()
            q.close()
            f.cancel()
            for w in workers:
                w.cancel()
                try:
                    await w
                except asyncio.CancelledError:
                    pass
            return d

        d = await asyncio.wait_for(run(), timeout=5.0)
        assert d[1] == 0  # the one in-flight is not counted in heap

    @pytest.mark.asyncio
    async def test_in_flight(self):
        q = PriorityRequestQueue(max_concurrent=1)
        assert q.in_flight == 0

    @pytest.mark.asyncio
    async def test_close_cancels_pending(self):
        q = PriorityRequestQueue(max_concurrent=1)

        async def slow():
            await asyncio.sleep(10.0)

        async def run():
            workers = [asyncio.create_task(q.worker()) for _ in range(1)]
            f1 = asyncio.create_task(q.submit(slow(), Priority.MEDIUM))
            await asyncio.sleep(0.05)
            # Submit another — will be pending
            f2 = asyncio.create_task(q.submit(slow(), Priority.LOW))
            await asyncio.sleep(0.05)
            q.close()
            for w in workers:
                w.cancel()
                try:
                    await w
                except asyncio.CancelledError:
                    pass
            return f1, f2

        f1, f2 = await asyncio.wait_for(run(), timeout=5.0)
        # f2 should have been cancelled
        assert f2.cancelled() or f2.done()


# ---------------------------------------------------------------------------
# _is_intel_mac
# ---------------------------------------------------------------------------

class TestIntelMac:
    def test_returns_bool(self):
        result = _is_intel_mac()
        assert isinstance(result, bool)
