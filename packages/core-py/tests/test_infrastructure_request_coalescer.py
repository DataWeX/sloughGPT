"""Tests for RequestCoalescer — deduplicates concurrent requests."""
from __future__ import annotations

import asyncio

import pytest

from domains.infrastructure.request_coalescer import (
    RequestCoalescer,
    _hash_key,
    get_coalescer,
    reset_coalescer,
)


class TestHashKey:
    def test_deterministic(self):
        assert _hash_key("a", "b") == _hash_key("a", "b")

    def test_different_inputs(self):
        assert _hash_key("a", "b") != _hash_key("a", "c")

    def test_fixed_length(self):
        h = _hash_key("test")
        assert len(h) == 32


class TestRequestCoalescer:
    def test_start_returns_none_first_call(self):
        co = RequestCoalescer()
        result = asyncio.get_event_loop().run_until_complete(co.start("key1"))
        assert result is None

    def test_start_returns_existing_second_call(self):
        co = RequestCoalescer()
        asyncio.get_event_loop().run_until_complete(co.start("key1"))
        result = asyncio.get_event_loop().run_until_complete(co.start("key1"))
        assert result is not None

    def test_complete_wakes_waiters(self):
        co = RequestCoalescer()

        async def run():
            existing = await co.start("k1")
            assert existing is None
            await co.complete("k1", "result_data")
            # Second caller gets the result
            entry = await co.start("k1")
            assert entry is not None
            await entry.event.wait()
            return entry.result

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result == "result_data"

    def test_in_flight_count(self):
        co = RequestCoalescer()
        asyncio.get_event_loop().run_until_complete(co.start("k1"))
        asyncio.get_event_loop().run_until_complete(co.start("k2"))
        assert co.in_flight_count == 2

    def test_remove(self):
        co = RequestCoalescer()
        asyncio.get_event_loop().run_until_complete(co.start("k1"))
        asyncio.get_event_loop().run_until_complete(co.remove("k1"))
        assert co.in_flight_count == 0

    def test_hash_method(self):
        co = RequestCoalescer()
        h = co.hash("a", "b")
        assert len(h) == 32


class TestSingleton:
    def test_get_coalescer(self):
        reset_coalescer()
        a = get_coalescer()
        b = get_coalescer()
        assert a is b
