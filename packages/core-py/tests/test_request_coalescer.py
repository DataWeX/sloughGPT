"""Tests for RequestCoalescer."""

import asyncio
import pytest
import time
from domains.infrastructure.request_coalescer import (
    RequestCoalescer,
    _hash_key,
    get_coalescer,
    reset_coalescer,
)


class TestHashKey:
    def test_deterministic(self):
        assert _hash_key("a", "b") == _hash_key("a", "b")

    def test_order_matters_for_positional_args(self):
        assert _hash_key("a", "b") != _hash_key("b", "a")

    def test_different_inputs_different_hash(self):
        assert _hash_key("a", "b") != _hash_key("a", "c")

    def test_empty_input(self):
        h = _hash_key()
        assert isinstance(h, str)
        assert len(h) == 32

    def test_complex_structures(self):
        h1 = _hash_key([{"role": "user", "content": "hi"}], {"temperature": 0.7})
        h2 = _hash_key([{"role": "user", "content": "hi"}], {"temperature": 0.7})
        assert h1 == h2


class TestCoalescerBasic:
    @pytest.mark.asyncio
    async def test_first_request_returns_none(self):
        c = RequestCoalescer(ttl_seconds=60)
        result = await c.start("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_second_request_returns_existing(self):
        c = RequestCoalescer(ttl_seconds=60)
        await c.start("key1")
        existing = await c.start("key1")
        assert existing is not None

    @pytest.mark.asyncio
    async def test_complete_wakes_waiters(self):
        c = RequestCoalescer(ttl_seconds=60)
        await c.start("key1")

        # Second requester
        existing = await c.start("key1")
        assert existing is not None

        # Complete the first request
        await c.complete("key1", "hello world")

        # Waiter should wake up
        await asyncio.wait_for(existing.event.wait(), timeout=1.0)
        assert existing.result == "hello world"
        assert existing.error is None

    @pytest.mark.asyncio
    async def test_complete_error_wakes_waiters(self):
        c = RequestCoalescer(ttl_seconds=60)
        await c.start("key1")
        existing = await c.start("key1")

        await c.complete_error("key1", ValueError("boom"))
        await asyncio.wait_for(existing.event.wait(), timeout=1.0)
        assert existing.result is None
        assert isinstance(existing.error, ValueError)
        assert str(existing.error) == "boom"

    @pytest.mark.asyncio
    async def test_remove_clears_entry(self):
        c = RequestCoalescer(ttl_seconds=60)
        await c.start("key1")
        await c.remove("key1")
        # After removal, a new start should return None (no existing)
        result = await c.start("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_in_flight_count(self):
        c = RequestCoalescer(ttl_seconds=60)
        assert c.in_flight_count == 0
        await c.start("key1")
        assert c.in_flight_count == 1
        await c.start("key2")
        assert c.in_flight_count == 2
        await c.complete("key1", "a")
        await c.remove("key1")
        assert c.in_flight_count == 1


class TestCoalescerConcurrent:
    @pytest.mark.asyncio
    async def test_concurrent_waiters_all_get_result(self):
        c = RequestCoalescer(ttl_seconds=60)
        await c.start("key1")

        # Spawn 5 concurrent waiters
        waiters = []
        for _ in range(5):
            existing = await c.start("key1")
            assert existing is not None
            waiters.append(existing)

        # Complete
        await c.complete("key1", "result")

        # All waiters should wake up
        for w in waiters:
            await asyncio.wait_for(w.event.wait(), timeout=1.0)
            assert w.result == "result"

    @pytest.mark.asyncio
    async def test_different_keys_independent(self):
        c = RequestCoalescer(ttl_seconds=60)
        await c.start("key1")
        await c.start("key2")

        await c.complete("key1", "result1")
        await c.complete("key2", "result2")

        # Verify entries are separate
        assert c.in_flight_count == 2

    @pytest.mark.asyncio
    async def test_first_requester_actually_runs(self):
        """Simulate the coalescing pattern: first runs, waiters skip."""
        c = RequestCoalescer(ttl_seconds=60)
        results = []

        async def simulated_request(request_id):
            key = c.hash("prompt", "params")
            existing = await c.start(key)
            if existing is not None:
                # Waiter — skip work
                await asyncio.wait_for(existing.event.wait(), timeout=1.0)
                results.append((request_id, existing.result, "cached"))
            else:
                # First request — do work
                await asyncio.sleep(0.05)
                await c.complete(key, "generated")
                results.append((request_id, "generated", "fresh"))

        await asyncio.gather(
            simulated_request("r1"),
            simulated_request("r2"),
            simulated_request("r3"),
        )

        # Exactly 1 fresh, 2 cached
        fresh = [r for r in results if r[2] == "fresh"]
        cached = [r for r in results if r[2] == "cached"]
        assert len(fresh) == 1
        assert len(cached) == 2

    @pytest.mark.asyncio
    async def test_complete_then_new_request(self):
        """After complete, new start returns None (new request)."""
        c = RequestCoalescer(ttl_seconds=60)
        await c.start("key1")
        await c.complete("key1", "result")
        await c.remove("key1")

        result = await c.start("key1")
        assert result is None


class TestCoalescerTTL:
    @pytest.mark.asyncio
    async def test_stale_entry_evicted(self):
        c = RequestCoalescer(ttl_seconds=0.01)
        await c.start("key1")
        # Manually run cleanup (background task sleeps 5s, too long for test)
        await c._cleanup_stale()
        # Entry should be gone
        result = await c.start("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_max_entries_eviction(self):
        c = RequestCoalescer(ttl_seconds=60, max_entries=3)
        await c.start("k1")
        await c.start("k2")
        await c.start("k3")
        await c.start("k4")
        # Manually trigger cleanup to enforce max_entries
        await c._cleanup_stale()
        assert c.in_flight_count <= 3
        # k1 (oldest) should have been evicted
        result = await c.start("k1")
        assert result is None


class TestCoalescerSingleton:
    def test_get_coalescer_returns_singleton(self):
        reset_coalescer()
        c1 = get_coalescer()
        c2 = get_coalescer()
        assert c1 is c2

    def test_reset_coalescer(self):
        c1 = get_coalescer()
        reset_coalescer()
        c2 = get_coalescer()
        assert c1 is not c2


class TestCoalescerHash:
    def test_hash_matches_provider_messages_format(self):
        """Hash should work with the message format used in inference.py."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "You are helpful."},
        ]
        params = {"temperature": 0.7, "top_p": 0.9}
        h1 = RequestCoalescer().hash(messages, params, 512, "qwen2.5-0.5b")
        h2 = RequestCoalescer().hash(messages, params, 512, "qwen2.5-0.5b")
        assert h1 == h2

    def test_hash_differs_with_different_model(self):
        messages = [{"role": "user", "content": "hello"}]
        params = {"temperature": 0.7}
        h1 = RequestCoalescer().hash(messages, params, 512, "model-a")
        h2 = RequestCoalescer().hash(messages, params, 512, "model-b")
        assert h1 != h2

    def test_hash_differs_with_different_temperature(self):
        messages = [{"role": "user", "content": "hello"}]
        h1 = RequestCoalescer().hash(messages, {"temperature": 0.0}, 512, "model")
        h2 = RequestCoalescer().hash(messages, {"temperature": 1.0}, 512, "model")
        assert h1 != h2


class TestCoalescerEdgeCases:
    @pytest.mark.asyncio
    async def test_complete_nonexistent_key_no_error(self):
        """Completing a key that was never started should not raise."""
        c = RequestCoalescer(ttl_seconds=60)
        await c.complete("ghost", "result")  # should not raise

    @pytest.mark.asyncio
    async def test_complete_error_nonexistent_key_no_error(self):
        """Error-completing a key that was never started should not raise."""
        c = RequestCoalescer(ttl_seconds=60)
        await c.complete_error("ghost", ValueError("x"))  # should not raise

    @pytest.mark.asyncio
    async def test_remove_nonexistent_key_no_error(self):
        """Removing a key that was never started should not raise."""
        c = RequestCoalescer(ttl_seconds=60)
        await c.remove("ghost")  # should not raise

    @pytest.mark.asyncio
    async def test_concurrent_start_same_key_only_one_none(self):
        """Multiple concurrent starts on the same key — only one gets None."""
        c = RequestCoalescer(ttl_seconds=60)
        results = []

        async def starter():
            r = await c.start("race_key")
            results.append(r)

        await asyncio.gather(starter(), starter(), starter())
        none_count = sum(1 for r in results if r is None)
        assert none_count == 1  # exactly one was the "first" requester
