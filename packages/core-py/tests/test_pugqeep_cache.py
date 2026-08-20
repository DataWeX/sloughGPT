"""Tests for domains.infrastructure.pugqeep.cache — CacheEntry, CacheStats, MemoryStore, DiskStore, TieredCache."""

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest


class TestCacheEntry:
    def test_touch(self):
        from domains.infrastructure.pugqeep.cache import CacheEntry, Tier
        entry = CacheEntry(key="k", tier=Tier.MEMORY)
        assert entry.access_count == 0
        entry.touch()
        assert entry.access_count == 1
        entry.touch()
        assert entry.access_count == 2

    def test_is_expired_no_ttl(self):
        from domains.infrastructure.pugqeep.cache import CacheEntry, Tier
        entry = CacheEntry(key="k", tier=Tier.MEMORY)
        assert entry.is_expired() is False

    def test_is_expired_with_ttl(self):
        from domains.infrastructure.pugqeep.cache import CacheEntry, Tier
        entry = CacheEntry(key="k", tier=Tier.MEMORY, ttl=0.01)
        time.sleep(0.02)
        assert entry.is_expired() is True

    def test_is_not_expired_within_ttl(self):
        from domains.infrastructure.pugqeep.cache import CacheEntry, Tier
        entry = CacheEntry(key="k", tier=Tier.MEMORY, ttl=10.0)
        assert entry.is_expired() is False


class TestCacheStats:
    def test_hit_rate_zero(self):
        from domains.infrastructure.pugqeep.cache import CacheStats
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate(self):
        from domains.infrastructure.pugqeep.cache import CacheStats
        stats = CacheStats(hits=3, misses=1)
        assert stats.hit_rate == pytest.approx(0.75)

    def test_hit_rate_all_hits(self):
        from domains.infrastructure.pugqeep.cache import CacheStats
        stats = CacheStats(hits=10, misses=0)
        assert stats.hit_rate == 1.0


class TestMemoryStore:
    def test_put_get(self):
        from domains.infrastructure.pugqeep.cache import MemoryStore
        store = MemoryStore()
        store.put("a", "value_a")
        assert store.get("a") == "value_a"

    def test_get_miss(self):
        from domains.infrastructure.pugqeep.cache import MemoryStore
        store = MemoryStore()
        assert store.get("missing") is None

    def test_remove(self):
        from domains.infrastructure.pugqeep.cache import MemoryStore
        store = MemoryStore()
        store.put("a", 1)
        assert store.remove("a") is True
        assert store.get("a") is None

    def test_remove_nonexistent(self):
        from domains.infrastructure.pugqeep.cache import MemoryStore
        store = MemoryStore()
        assert store.remove("nope") is False

    def test_exists(self):
        from domains.infrastructure.pugqeep.cache import MemoryStore
        store = MemoryStore()
        store.put("x", 42)
        assert store.exists("x") is True
        assert store.exists("y") is False

    def test_list_keys(self):
        from domains.infrastructure.pugqeep.cache import MemoryStore
        store = MemoryStore()
        store.put("b", 2)
        store.put("a", 1)
        keys = store.list_keys()
        assert set(keys) == {"a", "b"}

    def test_size_bytes(self):
        from domains.infrastructure.pugqeep.cache import MemoryStore
        store = MemoryStore()
        store.put("a", 1, size_bytes=100)
        store.put("b", 2, size_bytes=200)
        assert store.size_bytes() == 300

    def test_lru_eviction(self):
        from domains.infrastructure.pugqeep.cache import MemoryStore
        store = MemoryStore(max_size_bytes=300)
        store.put("a", 1, size_bytes=100)
        store.put("b", 2, size_bytes=100)
        store.put("c", 3, size_bytes=100)
        # Explicitly evict — target 150 bytes means free until <= 150
        evicted = store.evict_lru(150)
        assert "a" in evicted
        assert store.get("a") is None

    def test_lfu_eviction(self):
        from domains.infrastructure.pugqeep.cache import MemoryStore
        store = MemoryStore(max_size_bytes=300)
        store.put("a", 1, size_bytes=100)
        store.put("b", 2, size_bytes=100)
        store.put("c", 3, size_bytes=100)
        # Access "a" and "b" more than "c"
        store.get("a")
        store.get("a")
        store.get("b")
        # Evict to free 150 bytes — "c" has 0 accesses
        evicted = store.evict_lfu(150, {"a": 3, "b": 2, "c": 0})
        assert "c" in evicted
        assert store.exists("a") is True

    def test_lru_eviction_order(self):
        from domains.infrastructure.pugqeep.cache import MemoryStore
        store = MemoryStore(max_size_bytes=300)
        store.put("a", 1, size_bytes=100)
        store.put("b", 2, size_bytes=100)
        store.put("c", 3, size_bytes=100)
        # Access "c" to make it most recent
        store.get("c")
        # Evict 100 bytes — "a" (LRU) should be evicted first
        evicted = store.evict_lru(100)
        assert evicted[0] == "a"

    def test_put_existing_moves_to_end(self):
        from domains.infrastructure.pugqeep.cache import MemoryStore
        store = MemoryStore(max_size_bytes=200)
        store.put("a", 1, size_bytes=100)
        store.put("b", 2, size_bytes=100)
        # Re-put "a" — should move it to end (most recent)
        store.put("a", 10, size_bytes=100)
        keys = store.list_keys()
        assert keys[-1] == "a"  # most recent at end


class TestDiskStore:
    def test_put_get_scalar(self):
        from domains.infrastructure.pugqeep.cache import DiskStore
        with tempfile.TemporaryDirectory() as tmp:
            store = DiskStore(Path(tmp))
            store.put("key1", "hello")
            assert store.get("key1") == "hello"

    def test_put_get_ndarray(self):
        from domains.infrastructure.pugqeep.cache import DiskStore
        with tempfile.TemporaryDirectory() as tmp:
            store = DiskStore(Path(tmp))
            arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            store.put("arr", arr)
            result = store.get("arr")
            np.testing.assert_array_equal(result, arr)

    def test_get_miss(self):
        from domains.infrastructure.pugqeep.cache import DiskStore
        with tempfile.TemporaryDirectory() as tmp:
            store = DiskStore(Path(tmp))
            assert store.get("missing") is None

    def test_remove(self):
        from domains.infrastructure.pugqeep.cache import DiskStore
        with tempfile.TemporaryDirectory() as tmp:
            store = DiskStore(Path(tmp))
            store.put("k", "v")
            assert store.remove("k") is True
            assert store.get("k") is None

    def test_remove_nonexistent(self):
        from domains.infrastructure.pugqeep.cache import DiskStore
        with tempfile.TemporaryDirectory() as tmp:
            store = DiskStore(Path(tmp))
            assert store.remove("nope") is False

    def test_exists(self):
        from domains.infrastructure.pugqeep.cache import DiskStore
        with tempfile.TemporaryDirectory() as tmp:
            store = DiskStore(Path(tmp))
            store.put("x", 42)
            assert store.exists("x") is True
            assert store.exists("y") is False

    def test_list_keys(self):
        from domains.infrastructure.pugqeep.cache import DiskStore
        with tempfile.TemporaryDirectory() as tmp:
            store = DiskStore(Path(tmp))
            store.put("a", 1)
            store.put("b", 2)
            keys = store.list_keys()
            assert set(keys) == {"a", "b"}

    def test_size_bytes(self):
        from domains.infrastructure.pugqeep.cache import DiskStore
        with tempfile.TemporaryDirectory() as tmp:
            store = DiskStore(Path(tmp))
            store.put("a", "hello")
            assert store.size_bytes() > 0


class TestTieredCache:
    def test_put_get_memory(self):
        from domains.infrastructure.pugqeep.cache import TieredCache, Tier
        cache = TieredCache(memory_max_mb=1, hot_max_mb=1)
        cache.put("k", "value", tier=Tier.MEMORY)
        assert cache.get("k") == "value"

    def test_get_miss(self):
        from domains.infrastructure.pugqeep.cache import TieredCache
        cache = TieredCache(memory_max_mb=1, hot_max_mb=1)
        assert cache.get("missing") is None

    def test_exists(self):
        from domains.infrastructure.pugqeep.cache import TieredCache, Tier
        cache = TieredCache(memory_max_mb=1, hot_max_mb=1)
        cache.put("k", "v", tier=Tier.MEMORY)
        assert cache.exists("k") is True
        assert cache.exists("nope") is False

    def test_remove(self):
        from domains.infrastructure.pugqeep.cache import TieredCache, Tier
        cache = TieredCache(memory_max_mb=1, hot_max_mb=1)
        cache.put("k", "v", tier=Tier.MEMORY)
        assert cache.remove("k") is True
        assert cache.get("k") is None
        assert cache.exists("k") is False

    def test_remove_nonexistent(self):
        from domains.infrastructure.pugqeep.cache import TieredCache
        cache = TieredCache(memory_max_mb=1, hot_max_mb=1)
        assert cache.remove("nope") is False

    def test_ttl_expiration(self):
        from domains.infrastructure.pugqeep.cache import TieredCache, Tier
        cache = TieredCache(memory_max_mb=1, hot_max_mb=1)
        cache.put("k", "v", tier=Tier.MEMORY, ttl=0.01)
        time.sleep(0.02)
        assert cache.get("k") is None

    def test_list_keys_all(self):
        from domains.infrastructure.pugqeep.cache import TieredCache, Tier
        cache = TieredCache(memory_max_mb=1, hot_max_mb=1)
        cache.put("a", 1, tier=Tier.MEMORY)
        cache.put("b", 2, tier=Tier.HOT)
        keys = cache.list_keys()
        assert set(keys) == {"a", "b"}

    def test_list_keys_by_tier(self):
        from domains.infrastructure.pugqeep.cache import TieredCache, Tier
        cache = TieredCache(memory_max_mb=1, hot_max_mb=1)
        cache.put("a", 1, tier=Tier.MEMORY)
        cache.put("b", 2, tier=Tier.HOT)
        mem_keys = cache.list_keys(tier=Tier.MEMORY)
        assert mem_keys == ["a"]
        hot_keys = cache.list_keys(tier=Tier.HOT)
        assert hot_keys == ["b"]

    def test_stats(self):
        from domains.infrastructure.pugqeep.cache import TieredCache, Tier
        cache = TieredCache(memory_max_mb=1, hot_max_mb=1)
        cache.put("a", 1, tier=Tier.MEMORY, size_bytes=100)
        cache.get("a")  # hit
        cache.get("miss")
        stats = cache.stats()
        assert stats["total_entries"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.5)
        assert stats["eviction_policy"] == "lru"

    def test_ndarray_size_auto(self):
        from domains.infrastructure.pugqeep.cache import TieredCache, Tier
        cache = TieredCache(memory_max_mb=1, hot_max_mb=1)
        arr = np.zeros((10,), dtype=np.float32)
        cache.put("arr", arr, tier=Tier.MEMORY)
        stats = cache.stats()
        assert stats["tier_sizes"]["memory"] == arr.nbytes

    def test_pinned_not_evicted(self):
        from domains.infrastructure.pugqeep.cache import TieredCache, Tier
        cache = TieredCache(memory_max_mb=1, hot_max_mb=1, disk_dir=Path(tempfile.mkdtemp()))
        # Put many small items to trigger eviction, one pinned
        cache.put("pinned", "x", tier=Tier.MEMORY, pinned=True, size_bytes=500 * 1024 * 1024)
        cache.put("not_pinned", "y", tier=Tier.MEMORY, size_bytes=500 * 1024 * 1024)
        # Both exist if no eviction triggers yet
        assert cache.exists("pinned") is True

    def test_cleanup_expired(self):
        from domains.infrastructure.pugqeep.cache import TieredCache, Tier
        cache = TieredCache(memory_max_mb=1, hot_max_mb=1)
        cache.put("a", 1, tier=Tier.MEMORY, ttl=0.01)
        cache.put("b", 2, tier=Tier.MEMORY)  # no TTL
        time.sleep(0.02)
        removed = cache.cleanup_expired()
        assert removed == 1
        assert cache.exists("a") is False
        assert cache.exists("b") is True

    def test_hot_store_put_get(self):
        from domains.infrastructure.pugqeep.cache import HotStore
        store = HotStore()
        store.put("k", "v")
        assert store.get("k") == "v"

    def test_hot_store_remove(self):
        from domains.infrastructure.pugqeep.cache import HotStore
        store = HotStore()
        store.put("k", "v")
        assert store.remove("k") is True
        assert store.get("k") is None

    def test_hot_store_exists(self):
        from domains.infrastructure.pugqeep.cache import HotStore
        store = HotStore()
        store.put("x", 1)
        assert store.exists("x") is True
        assert store.exists("y") is False

    def test_hot_store_list_keys(self):
        from domains.infrastructure.pugqeep.cache import HotStore
        store = HotStore()
        store.put("a", 1)
        store.put("b", 2)
        assert set(store.list_keys()) == {"a", "b"}

    def test_hot_store_size_bytes(self):
        from domains.infrastructure.pugqeep.cache import HotStore
        store = HotStore()
        store.put("a", 1, size_bytes=50)
        store.put("b", 2, size_bytes=50)
        assert store.size_bytes() == 100

    def test_disk_tier(self):
        from domains.infrastructure.pugqeep.cache import TieredCache, Tier
        with tempfile.TemporaryDirectory() as tmp:
            cache = TieredCache(memory_max_mb=1, hot_max_mb=1, disk_dir=Path(tmp))
            cache.put("k", "disk_value", tier=Tier.DISK)
            result = cache.get("k")
            assert result == "disk_value"

    def test_evict_memory_demotes_to_hot(self):
        from domains.infrastructure.pugqeep.cache import TieredCache, Tier, MemoryStore
        with tempfile.TemporaryDirectory() as tmp:
            cache = TieredCache(memory_max_mb=1, hot_max_mb=1, disk_dir=Path(tmp))
            # Override memory store to have tiny capacity so eviction triggers
            cache._memory = MemoryStore(max_size_bytes=200)
            cache.put("k", "v", tier=Tier.MEMORY, size_bytes=100)
            # Add another item to push over capacity
            cache.put("extra", "x", tier=Tier.MEMORY, size_bytes=100)
            # Now evict 100 bytes
            freed = cache.evict(Tier.MEMORY, 100)
            # "k" should have been evicted (LRU) and demoted to hot
            entry_k = cache._entries.get("k")
            assert entry_k is not None
            assert entry_k.tier == Tier.HOT

    def test_auto_promote_hot_to_memory(self):
        from domains.infrastructure.pugqeep.cache import TieredCache, Tier
        cache = TieredCache(memory_max_mb=1, hot_max_mb=1, promote_threshold=2)
        cache.put("k", "v", tier=Tier.HOT)
        cache.get("k")  # access 1
        cache.get("k")  # access 2 — should trigger auto-promote to memory
        entry = cache._entries.get("k")
        assert entry.tier == Tier.MEMORY
