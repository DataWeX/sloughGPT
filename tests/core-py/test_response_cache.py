"""Tests for domain.inference._internal.response_cache.py — unified LLM response cache."""

from __future__ import annotations

import threading

import pytest

from domain.inference._internal.response_cache import (
    DETERMINISTIC_TEMPERATURE,
    ResponseCache,
    bump_cache_generation,
    current_generation,
)


@pytest.fixture(autouse=True)
def _fresh_generation():
    """Pin the generation token so keys are stable within a test."""
    start = current_generation()
    yield
    # advance past any bumps a test made so other tests start clean
    while current_generation() < start:
        bump_cache_generation()


class TestKeyDiscipline:
    def test_identical_request_same_key(self):
        msgs = [{"role": "user", "content": "hi"}]
        a = ResponseCache.build_key(msgs, "m", 0.1, 128)
        b = ResponseCache.build_key(list(msgs), "m", 0.1, 128)
        assert a == b

    def test_message_content_change_changes_key(self):
        a = ResponseCache.build_key([{"role": "user", "content": "hi"}], "m", 0.1, 128)
        b = ResponseCache.build_key([{"role": "user", "content": "bye"}], "m", 0.1, 128)
        assert a != b

    def test_model_change_changes_key(self):
        msgs = [{"role": "user", "content": "hi"}]
        a = ResponseCache.build_key(msgs, "model-a", 0.1, 128)
        b = ResponseCache.build_key(msgs, "model-b", 0.1, 128)
        assert a != b

    def test_temperature_change_changes_key(self):
        msgs = [{"role": "user", "content": "hi"}]
        a = ResponseCache.build_key(msgs, "m", 0.1, 128)
        b = ResponseCache.build_key(msgs, "m", 0.7, 128)
        assert a != b

    def test_max_tokens_change_changes_key(self):
        msgs = [{"role": "user", "content": "hi"}]
        a = ResponseCache.build_key(msgs, "m", 0.1, 128)
        b = ResponseCache.build_key(msgs, "m", 0.1, 256)
        assert a != b

    def test_generation_change_changes_key(self):
        msgs = [{"role": "user", "content": "hi"}]
        a = ResponseCache.build_key(msgs, "m", 0.1, 128, generation=1)
        b = ResponseCache.build_key(msgs, "m", 0.1, 128, generation=2)
        assert a != b

    def test_role_order_change_changes_key(self):
        a = ResponseCache.build_key(
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
            "m",
            0.1,
            128,
        )
        b = ResponseCache.build_key(
            [{"role": "assistant", "content": "y"}, {"role": "user", "content": "x"}],
            "m",
            0.1,
            128,
        )
        assert a != b


class TestDeterministicGate:
    def test_low_temp_is_deterministic(self):
        assert ResponseCache.is_deterministic(0.0) is True
        assert ResponseCache.is_deterministic(DETERMINISTIC_TEMPERATURE) is True

    def test_high_temp_not_deterministic(self):
        assert ResponseCache.is_deterministic(0.8) is False


class TestStoreLookup:
    def test_put_then_get_hit(self):
        c = ResponseCache()
        k = ResponseCache.build_key([{"role": "user", "content": "hi"}], "m", 0.0, 32)
        assert c.put(k, "hello there", "m") is True
        assert c.get(k) == "hello there"

    def test_miss_on_unknown_key(self):
        c = ResponseCache()
        k = ResponseCache.build_key([{"role": "user", "content": "hi"}], "m", 0.0, 32)
        assert c.get(k) is None

    def test_empty_response_not_stored(self):
        c = ResponseCache()
        assert c.put("k", "", "m") is False
        assert len(c) == 0


class TestInvalidationByGeneration:
    def test_generation_bump_invalidates(self):
        c = ResponseCache()
        msgs = [{"role": "user", "content": "hi"}]
        k = ResponseCache.build_key(msgs, "m", 0.0, 32)
        c.put(k, "old-model-reply", "m")

        bump_cache_generation()

        k2 = ResponseCache.build_key(msgs, "m", 0.0, 32)
        assert k2 != k
        assert c.get(k2) is None  # stale reply is not replayable


class TestLRUAndTTL:
    def test_lru_eviction_at_capacity(self):
        c = ResponseCache(max_entries=2)
        c.put("a", "ra", "m")
        c.put("b", "rb", "m")
        c.get("a")  # refresh a -> b is now LRU
        c.put("c", "rc", "m")  # evicts b
        assert c.get("a") == "ra"
        assert c.get("b") is None
        assert c.get("c") == "rc"

    def test_ttl_expiry(self):
        c = ResponseCache(ttl_seconds=0.0)
        k = ResponseCache.build_key([{"role": "user", "content": "hi"}], "m", 0.0, 32)
        c.put(k, "reply", "m")
        import time

        time.sleep(0.01)  # ttl 0 -> already expired
        assert c.get(k) is None


class TestStats:
    def test_hit_miss_counts(self):
        c = ResponseCache()
        k = ResponseCache.build_key([{"role": "user", "content": "hi"}], "m", 0.0, 32)
        c.get(k)  # miss
        c.put(k, "reply", "m")
        c.get(k)  # hit
        c.get(k)  # hit
        s = c.get_stats()
        assert s["hits"] == 2
        assert s["misses"] == 1
        assert s["stores"] == 1

    def test_bypass_counters(self):
        c = ResponseCache()
        c.record_bypass_stochastic()
        c.record_bypass_stream()
        s = c.get_stats()
        assert s["bypass_stochastic"] == 1
        assert s["bypass_stream"] == 1


class TestSemanticTier:
    def test_disabled_by_default(self):
        c = ResponseCache()
        assert c.get_semantic("hello", "m", 0.0) is None

    def test_disabled_when_stochastic(self):
        c = ResponseCache(semantic_enabled=True)
        # even with flag on, stochastic requests never use the semantic tier
        assert c.get_semantic("hello", "m", 0.9) is None


class TestConcurrency:
    def test_parallel_puts_and_gets_are_consistent(self):
        c = ResponseCache(max_entries=256)
        errors = []

        def worker(i):
            try:
                for j in range(50):
                    key = f"k{i}-{j}"
                    c.put(key, f"r{i}-{j}", "m")
                    c.get(key)
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
