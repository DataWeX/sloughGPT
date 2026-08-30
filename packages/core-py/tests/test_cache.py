"""Tests for domains.infrastructure.cache — CacheEntry and CacheManager."""

import asyncio
import time
import pytest
from domains.infrastructure.cache import CacheEntry, CacheManager


class TestCacheEntry:
    def test_fields(self):
        now = time.time()
        entry = CacheEntry(
            key="k", value="v", ttl=60,
            created_at=now, accessed_count=1, last_accessed=now,
        )
        assert entry.key == "k"
        assert entry.value == "v"
        assert entry.ttl == 60
        assert entry.accessed_count == 1

    def test_none_ttl(self):
        entry = CacheEntry(
            key="k", value=42, ttl=None,
            created_at=0.0, accessed_count=0, last_accessed=0.0,
        )
        assert entry.ttl is None

    def test_value_types(self):
        entry = CacheEntry(
            key="k", value={"a": [1, 2]}, ttl=10,
            created_at=0.0, accessed_count=0, last_accessed=0.0,
        )
        assert entry.value["a"] == [1, 2]

    def test_string_value(self):
        entry = CacheEntry(
            key="k", value="hello world", ttl=60,
            created_at=1.0, accessed_count=1, last_accessed=1.0,
        )
        assert entry.value == "hello world"

    def test_integer_value(self):
        entry = CacheEntry(
            key="k", value=12345, ttl=60,
            created_at=1.0, accessed_count=1, last_accessed=1.0,
        )
        assert entry.value == 12345

    def test_list_value(self):
        entry = CacheEntry(
            key="k", value=[1, 2, 3], ttl=60,
            created_at=1.0, accessed_count=1, last_accessed=1.0,
        )
        assert entry.value == [1, 2, 3]

    def test_nested_dict_value(self):
        entry = CacheEntry(
            key="k", value={"outer": {"inner": [1, 2]}}, ttl=60,
            created_at=1.0, accessed_count=1, last_accessed=1.0,
        )
        assert entry.value["outer"]["inner"] == [1, 2]

    def test_zero_ttl(self):
        entry = CacheEntry(
            key="k", value="v", ttl=0,
            created_at=1.0, accessed_count=1, last_accessed=1.0,
        )
        assert entry.ttl == 0

    def test_large_accessed_count(self):
        entry = CacheEntry(
            key="k", value="v", ttl=60,
            created_at=1.0, accessed_count=1000000, last_accessed=1.0,
        )
        assert entry.accessed_count == 1000000

    def test_negative_ttl(self):
        entry = CacheEntry(
            key="k", value="v", ttl=-5,
            created_at=1.0, accessed_count=0, last_accessed=1.0,
        )
        assert entry.ttl == -5

    def test_float_value(self):
        entry = CacheEntry(
            key="k", value=3.14159, ttl=60,
            created_at=1.0, accessed_count=1, last_accessed=1.0,
        )
        assert entry.value == pytest.approx(3.14159)

    def test_bool_value(self):
        entry = CacheEntry(
            key="k", value=True, ttl=60,
            created_at=1.0, accessed_count=1, last_accessed=1.0,
        )
        assert entry.value is True

    def test_set_value(self):
        entry = CacheEntry(
            key="k", value="old", ttl=60,
            created_at=1.0, accessed_count=1, last_accessed=1.0,
        )
        entry.value = "new"
        assert entry.value == "new"

    def test_set_key(self):
        entry = CacheEntry(
            key="old", value="v", ttl=60,
            created_at=1.0, accessed_count=1, last_accessed=1.0,
        )
        entry.key = "new"
        assert entry.key == "new"

    def test_large_key(self):
        entry = CacheEntry(
            key="k" * 10000, value="v", ttl=60,
            created_at=1.0, accessed_count=0, last_accessed=1.0,
        )
        assert len(entry.key) == 10000

    def test_tuple_value(self):
        entry = CacheEntry(
            key="k", value=(1, 2, 3), ttl=60,
            created_at=1.0, accessed_count=1, last_accessed=1.0,
        )
        assert entry.value == (1, 2, 3)

    def test_set_value_none(self):
        entry = CacheEntry(
            key="k", value="v", ttl=60,
            created_at=1.0, accessed_count=1, last_accessed=1.0,
        )
        entry.value = None
        assert entry.value is None

    def test_zero_accessed_count(self):
        entry = CacheEntry(
            key="k", value="v", ttl=60,
            created_at=1.0, accessed_count=0, last_accessed=1.0,
        )
        assert entry.accessed_count == 0

    def test_zero_last_accessed(self):
        entry = CacheEntry(
            key="k", value="v", ttl=60,
            created_at=1.0, accessed_count=0, last_accessed=0.0,
        )
        assert entry.last_accessed == 0.0

    def test_dataclass_fields(self):
        from dataclasses import fields
        field_names = [f.name for f in fields(CacheEntry)]
        assert "key" in field_names
        assert "value" in field_names
        assert "ttl" in field_names
        assert "created_at" in field_names
        assert "accessed_count" in field_names
        assert "last_accessed" in field_names

    def test_none_value(self):
        entry = CacheEntry(
            key="k", value=None, ttl=60,
            created_at=1.0, accessed_count=1, last_accessed=1.0,
        )
        assert entry.value is None

    def test_set_ttl(self):
        entry = CacheEntry(
            key="k", value="v", ttl=60,
            created_at=1.0, accessed_count=1, last_accessed=1.0,
        )
        entry.ttl = 120
        assert entry.ttl == 120

    def test_set_created_at(self):
        entry = CacheEntry(
            key="k", value="v", ttl=60,
            created_at=1.0, accessed_count=1, last_accessed=1.0,
        )
        entry.created_at = 99.0
        assert entry.created_at == 99.0


class TestCacheManager:
    def setup_method(self):
        self.mgr = CacheManager()

    def test_defaults(self):
        assert self.mgr.max_size == 10000
        assert self.mgr.default_ttl == 3600
        assert self.mgr.cache == {}

    def test_stats_init(self):
        assert self.mgr.stats["hits"] == 0
        assert self.mgr.stats["misses"] == 0
        assert self.mgr.stats["sets"] == 0
        assert self.mgr.stats["evictions"] == 0

    def test_is_not_initialized(self):
        assert self.mgr.is_initialized is False

    def test_component_name(self):
        assert self.mgr.component_name == "cache_manager"

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        await self.mgr.set("key1", "value1")
        result = await self.mgr.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_miss(self):
        result = await self.mgr.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_increments_stats(self):
        await self.mgr.set("key1", "value1")
        assert self.mgr.stats["sets"] == 1

    @pytest.mark.asyncio
    async def test_get_hit_increments_stats(self):
        await self.mgr.set("key1", "value1")
        await self.mgr.get("key1")
        assert self.mgr.stats["hits"] == 1

    @pytest.mark.asyncio
    async def test_get_miss_increments_stats(self):
        await self.mgr.get("nonexistent")
        assert self.mgr.stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_delete(self):
        await self.mgr.set("key1", "value1")
        result = await self.mgr.delete("key1")
        assert result is True
        assert await self.mgr.get("key1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        result = await self.mgr.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_increments_stats(self):
        await self.mgr.set("key1", "value1")
        await self.mgr.delete("key1")
        assert self.mgr.stats["deletes"] == 1

    @pytest.mark.asyncio
    async def test_clear(self):
        await self.mgr.set("key1", "value1")
        await self.mgr.set("key2", "value2")
        result = await self.mgr.clear()
        assert result is True
        assert await self.mgr.get("key1") is None
        assert await self.mgr.get("key2") is None

    @pytest.mark.asyncio
    async def test_ttl_expiry(self):
        await self.mgr.set("key1", "value1", ttl=1)
        time.sleep(1.1)
        result = await self.mgr.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_overwrite(self):
        await self.mgr.set("key1", "value1")
        await self.mgr.set("key1", "value2")
        result = await self.mgr.get("key1")
        assert result == "value2"

    @pytest.mark.asyncio
    async def test_multiple_keys(self):
        await self.mgr.set("a", 1)
        await self.mgr.set("b", 2)
        await self.mgr.set("c", 3)
        assert await self.mgr.get("a") == 1
        assert await self.mgr.get("b") == 2
        assert await self.mgr.get("c") == 3

    @pytest.mark.asyncio
    async def test_custom_ttl(self):
        await self.mgr.set("key1", "value1", ttl=10)
        assert self.mgr.cache["key1"].ttl == 10

    @pytest.mark.asyncio
    async def test_statistics(self):
        await self.mgr.set("key1", "value1")
        await self.mgr.get("key1")
        await self.mgr.get("nonexistent")
        stats = await self.mgr.get_cache_statistics()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["sets"] == 1
        assert stats["cache_size"] == 1

    @pytest.mark.asyncio
    async def test_hit_rate(self):
        await self.mgr.set("key1", "value1")
        await self.mgr.get("key1")
        await self.mgr.get("key1")
        stats = await self.mgr.get_cache_statistics()
        assert stats["hits"] == 2
        assert stats["hit_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_eviction(self):
        self.mgr.max_size = 2
        await self.mgr.set("key1", "value1")
        await self.mgr.set("key2", "value2")
        await self.mgr.set("key3", "value3")  # Should evict key1
        assert self.mgr.stats["evictions"] == 1

    @pytest.mark.asyncio
    async def test_lru_eviction_order(self):
        self.mgr.max_size = 2
        await self.mgr.set("key1", "value1")
        await self.mgr.set("key2", "value2")
        await self.mgr.get("key1")  # Access key1, so key2 is LRU
        await self.mgr.set("key3", "value3")  # Should evict key2
        assert await self.mgr.get("key1") == "value1"
        assert await self.mgr.get("key3") == "value3"

    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        self.mgr.cache["expired"] = CacheEntry(
            key="expired", value="v", ttl=1,
            created_at=time.time() - 10, accessed_count=0, last_accessed=0.0,
        )
        await self.mgr._cleanup_expired_entries()
        assert "expired" not in self.mgr.cache

    @pytest.mark.asyncio
    async def test_cleanup_does_not_remove_valid(self):
        await self.mgr.set("valid", "value", ttl=3600)
        await self.mgr._cleanup_expired_entries()
        assert "valid" in self.mgr.cache

    def test_max_size_property(self):
        self.mgr.max_size = 500
        assert self.mgr.max_size == 500

    def test_default_ttl_property(self):
        self.mgr.default_ttl = 1800
        assert self.mgr.default_ttl == 1800

    @pytest.mark.asyncio
    async def test_access_count_increments(self):
        await self.mgr.set("key1", "value1")
        await self.mgr.get("key1")
        await self.mgr.get("key1")
        entry = self.mgr.cache["key1"]
        assert entry.accessed_count == 3

    @pytest.mark.asyncio
    async def test_last_accessed_updates(self):
        await self.mgr.set("key1", "value1")
        initial = self.mgr.cache["key1"].last_accessed
        time.sleep(0.01)
        await self.mgr.get("key1")
        assert self.mgr.cache["key1"].last_accessed >= initial

    @pytest.mark.asyncio
    async def test_utilization(self):
        self.mgr.max_size = 10
        await self.mgr.set("k", "v")
        stats = await self.mgr.get_cache_statistics()
        assert stats["utilization"] == 0.1

    @pytest.mark.asyncio
    async def test_complex_value_types(self):
        await self.mgr.set("list", [1, 2, 3])
        await self.mgr.set("dict", {"key": "value"})
        await self.mgr.set("nested", [{"a": 1}, {"b": 2}])
        assert await self.mgr.get("list") == [1, 2, 3]
        assert await self.mgr.get("dict") == {"key": "value"}

    @pytest.mark.asyncio
    async def test_set_returns_true(self):
        result = await self.mgr.set("key", "value")
        assert result is True

    @pytest.mark.asyncio
    async def test_empty_string_value(self):
        await self.mgr.set("key", "")
        assert await self.mgr.get("key") == ""

    @pytest.mark.asyncio
    async def test_none_value(self):
        await self.mgr.set("key", None)
        result = await self.mgr.get("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_stats_deletes(self):
        self.mgr.stats["deletes"] = 0
        await self.mgr.set("k", "v")
        await self.mgr.delete("k")
        assert self.mgr.stats["deletes"] == 1

    @pytest.mark.asyncio
    async def test_concurrent_sets(self):
        async def set_val(key, val):
            await self.mgr.set(key, val)

        await asyncio.gather(*[set_val(f"k{i}", i) for i in range(10)])
        assert len(self.mgr.cache) == 10

    @pytest.mark.asyncio
    async def test_cleanup_multiple_expired(self):
        now = time.time()
        for i in range(5):
            self.mgr.cache[f"exp{i}"] = CacheEntry(
                key=f"exp{i}", value=i, ttl=1,
                created_at=now - 10, accessed_count=0, last_accessed=0.0,
            )
        await self.mgr._cleanup_expired_entries()
        assert len(self.mgr.cache) == 0
        assert self.mgr.stats["evictions"] == 5

    @pytest.mark.asyncio
    async def test_cleanup_mixed_expired_and_valid(self):
        now = time.time()
        self.mgr.cache["expired"] = CacheEntry(
            key="expired", value="v", ttl=1,
            created_at=now - 10, accessed_count=0, last_accessed=0.0,
        )
        self.mgr.cache["valid"] = CacheEntry(
            key="valid", value="v", ttl=3600,
            created_at=now, accessed_count=0, last_accessed=now,
        )
        await self.mgr._cleanup_expired_entries()
        assert "expired" not in self.mgr.cache
        assert "valid" in self.mgr.cache

    @pytest.mark.asyncio
    async def test_eviction_frees_space(self):
        self.mgr.max_size = 3
        await self.mgr.set("a", 1)
        await self.mgr.set("b", 2)
        await self.mgr.set("c", 3)
        await self.mgr.set("d", 4)  # evicts a
        assert len(self.mgr.cache) == 3
        assert await self.mgr.get("a") is None

    @pytest.mark.asyncio
    async def test_lru_with_multiple_accesses(self):
        self.mgr.max_size = 2
        await self.mgr.set("a", 1)
        await self.mgr.set("b", 2)
        await self.mgr.get("a")
        await self.mgr.get("a")
        await self.mgr.get("b")  # b is now most recent
        await self.mgr.set("c", 3)  # evicts a (LRU)
        assert await self.mgr.get("b") == 2
        assert await self.mgr.get("c") == 3

    @pytest.mark.asyncio
    async def test_empty_cache_get(self):
        result = await self.mgr.get("anything")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_cache_delete(self):
        result = await self.mgr.delete("anything")
        assert result is False

    @pytest.mark.asyncio
    async def test_empty_cache_clear(self):
        result = await self.mgr.clear()
        assert result is True

    @pytest.mark.asyncio
    async def test_overwrite_increments_sets(self):
        await self.mgr.set("k", "v1")
        await self.mgr.set("k", "v2")
        assert self.mgr.stats["sets"] == 2

    @pytest.mark.asyncio
    async def test_delete_then_reinsert(self):
        await self.mgr.set("k", "v1")
        await self.mgr.delete("k")
        await self.mgr.set("k", "v2")
        assert await self.mgr.get("k") == "v2"

    @pytest.mark.asyncio
    async def test_statistics_after_operations(self):
        await self.mgr.set("a", 1)
        await self.mgr.set("b", 2)
        await self.mgr.get("a")
        await self.mgr.get("a")
        await self.mgr.get("miss")
        await self.mgr.delete("b")
        stats = await self.mgr.get_cache_statistics()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["sets"] == 2
        assert stats["deletes"] == 1
        assert stats["cache_size"] == 1

    @pytest.mark.asyncio
    async def test_hit_rate_zero_requests(self):
        stats = await self.mgr.get_cache_statistics()
        assert stats["hit_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_hit_rate_mixed(self):
        await self.mgr.set("k", "v")
        await self.mgr.get("k")
        await self.mgr.get("miss")
        stats = await self.mgr.get_cache_statistics()
        assert stats["hit_rate"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_cleanup_loop(self):
        assert self.mgr.cleanup_task is None
        assert self.mgr.cleanup_interval == 300

    @pytest.mark.asyncio
    async def test_default_ttl_used(self):
        await self.mgr.set("k", "v")
        assert self.mgr.cache["k"].ttl == 3600

    @pytest.mark.asyncio
    async def test_zero_ttl_uses_default(self):
        await self.mgr.set("k", "v", ttl=0)
        assert self.mgr.cache["k"].ttl == self.mgr.default_ttl

    @pytest.mark.asyncio
    async def test_large_ttl(self):
        await self.mgr.set("k", "v", ttl=999999)
        assert self.mgr.cache["k"].ttl == 999999

    @pytest.mark.asyncio
    async def test_get_after_set_returns_same_value(self):
        values = [1, "hello", [1, 2], {"a": 1}, None, True, 3.14]
        for v in values:
            await self.mgr.set(f"k_{v}", v)
            assert await self.mgr.get(f"k_{v}") == v

    @pytest.mark.asyncio
    async def test_max_size_one(self):
        self.mgr.max_size = 1
        await self.mgr.set("a", 1)
        await self.mgr.set("b", 2)  # evicts a
        assert await self.mgr.get("a") is None
        assert await self.mgr.get("b") == 2

    @pytest.mark.asyncio
    async def test_many_gets_increase_access_count(self):
        await self.mgr.set("k", "v")
        for _ in range(100):
            await self.mgr.get("k")
        assert self.mgr.cache["k"].accessed_count == 101

    @pytest.mark.asyncio
    async def test_clear_resets_cache(self):
        for i in range(10):
            await self.mgr.set(f"k{i}", i)
        await self.mgr.clear()
        assert len(self.mgr.cache) == 0

    @pytest.mark.asyncio
    async def test_max_size_one_evicts_immediately(self):
        self.mgr.max_size = 1
        await self.mgr.set("a", 1)
        assert len(self.mgr.cache) == 1
        await self.mgr.set("b", 2)
        assert len(self.mgr.cache) == 1
        assert await self.mgr.get("a") is None
        assert await self.mgr.get("b") == 2

    @pytest.mark.asyncio
    async def test_default_ttl_override(self):
        self.mgr.default_ttl = 10
        await self.mgr.set("k", "v")
        assert self.mgr.cache["k"].ttl == 10

    @pytest.mark.asyncio
    async def test_concurrent_gets(self):
        await self.mgr.set("k", "v")

        async def get_val():
            return await self.mgr.get("k")

        results = await asyncio.gather(*[get_val() for _ in range(10)])
        assert all(r == "v" for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_deletes(self):
        await self.mgr.set("k", "v")

        async def del_val():
            return await self.mgr.delete("k")

        results = await asyncio.gather(*[del_val() for _ in range(5)])
        assert sum(results) == 1

    @pytest.mark.asyncio
    async def test_eviction_stats_increment(self):
        self.mgr.max_size = 1
        await self.mgr.set("a", 1)
        await self.mgr.set("b", 2)
        await self.mgr.set("c", 3)
        assert self.mgr.stats["evictions"] == 2

    @pytest.mark.asyncio
    async def test_utilization_full(self):
        self.mgr.max_size = 5
        for i in range(5):
            await self.mgr.set(f"k{i}", i)
        stats = await self.mgr.get_cache_statistics()
        assert stats["utilization"] == 1.0

    @pytest.mark.asyncio
    async def test_utilization_empty(self):
        stats = await self.mgr.get_cache_statistics()
        assert stats["utilization"] == 0.0

    @pytest.mark.asyncio
    async def test_max_size_in_stats(self):
        self.mgr.max_size = 500
        stats = await self.mgr.get_cache_statistics()
        assert stats["max_size"] == 500

    @pytest.mark.asyncio
    async def test_cache_size_in_stats(self):
        await self.mgr.set("a", 1)
        await self.mgr.set("b", 2)
        stats = await self.mgr.get_cache_statistics()
        assert stats["cache_size"] == 2

    @pytest.mark.asyncio
    async def test_none_ttl_uses_default(self):
        await self.mgr.set("k", "v", ttl=None)
        assert self.mgr.cache["k"].ttl == self.mgr.default_ttl
