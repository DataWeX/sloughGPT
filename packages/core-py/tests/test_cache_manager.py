"""Tests for domains/infrastructure/cache/__init__.py (CacheManager)."""

import asyncio

import pytest

from domains.infrastructure.cache import CacheEntry, CacheManager, ComponentException


class _BadDict(dict):
    def __contains__(self, key):
        raise RuntimeError("contains boom")

    def clear(self):
        raise RuntimeError("clear boom")


@pytest.fixture
def manager():
    m = CacheManager()
    yield m
    if m.cleanup_task is not None and not m.cleanup_task.done():
        m.cleanup_task.cancel()


class TestCacheEntry:
    def test_fields(self):
        e = CacheEntry("k", "v", 10, 1.0, 2, 3.0)
        assert e.key == "k"
        assert e.value == "v"
        assert e.ttl == 10
        assert e.created_at == 1.0
        assert e.accessed_count == 2
        assert e.last_accessed == 3.0


class TestCacheManager:
    async def test_initialize_starts_cleanup(self, manager):
        manager.cleanup_interval = 0.05
        await manager.initialize()
        assert manager.is_initialized is True
        assert manager.cleanup_task is not None

    async def test_initialize_failure_raises(self, manager, monkeypatch):
        def boom(msg, **kwargs):
            raise RuntimeError("init failed")

        monkeypatch.setattr(manager.logger, "info", boom)
        with pytest.raises(ComponentException):
            await manager.initialize()

    async def test_shutdown_cancels_cleanup(self, manager):
        await manager.initialize()
        assert manager.is_initialized is True
        await manager.shutdown()
        assert manager.is_initialized is False
        assert manager.cleanup_task is None or manager.cleanup_task.done()

    async def test_shutdown_no_task(self, manager):
        await manager.shutdown()
        assert manager.is_initialized is False

    async def test_shutdown_failure_raises(self, manager, monkeypatch):
        await manager.initialize()

        def boom(msg, **kwargs):
            raise RuntimeError("shutdown failed")

        monkeypatch.setattr(manager.logger, "info", boom)
        with pytest.raises(ComponentException):
            await manager.shutdown()

    async def test_get_miss(self, manager):
        assert await manager.get("nope") is None
        assert manager.stats["misses"] == 1

    async def test_get_hit(self, manager):
        await manager.set("k", "v")
        assert await manager.get("k") == "v"
        assert manager.stats["hits"] == 1
        assert manager.cache["k"].accessed_count == 2

    async def test_get_expired_deletes_and_misses(self, manager):
        await manager.set("k", "v", ttl=10)
        manager.cache["k"].created_at -= 100
        assert await manager.get("k") is None
        assert "k" not in manager.cache
        assert manager.stats["misses"] == 1

    async def test_get_exception_returns_none(self, manager):
        manager.cache = _BadDict()
        assert await manager.get("k") is None
        assert manager.stats["misses"] == 1

    async def test_set_and_retrieve(self, manager):
        assert await manager.set("k", "v") is True
        assert manager.stats["sets"] == 1
        assert await manager.get("k") == "v"

    async def test_set_default_ttl(self, manager):
        await manager.set("k", "v")
        assert manager.cache["k"].ttl == manager.default_ttl

    async def test_set_evicts_lru_when_full(self, manager):
        manager.max_size = 2
        await manager.set("a", 1)
        await manager.set("b", 2)
        await manager.set("c", 3)
        assert "a" not in manager.cache
        assert manager.stats["evictions"] == 1
        assert manager.stats["sets"] == 3

    async def test_set_exception_returns_false(self, manager, monkeypatch):
        manager.max_size = 0

        async def boom():
            raise RuntimeError("evict boom")

        monkeypatch.setattr(manager, "_evict_lru", boom)
        assert await manager.set("k", "v") is False

    async def test_delete_hit(self, manager):
        await manager.set("k", "v")
        assert await manager.delete("k") is True
        assert manager.stats["deletes"] == 1

    async def test_delete_miss(self, manager):
        assert await manager.delete("k") is False

    async def test_delete_exception_returns_false(self, manager):
        manager.cache = _BadDict()
        assert await manager.delete("k") is False

    async def test_clear(self, manager):
        await manager.set("a", 1)
        await manager.set("b", 2)
        assert await manager.clear() is True
        assert manager.cache == {}

    async def test_clear_exception_returns_false(self, manager):
        manager.cache = _BadDict()
        assert await manager.clear() is False

    async def test_get_cache_statistics(self, manager):
        await manager.set("a", 1)
        await manager.set("b", 2)
        await manager.get("a")
        stats = await manager.get_cache_statistics()
        assert stats["hits"] == 1
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 1.0
        assert stats["cache_size"] == 2
        assert stats["max_size"] == manager.max_size
        assert stats["utilization"] == 2 / manager.max_size

    async def test_statistics_empty(self, manager):
        stats = await manager.get_cache_statistics()
        assert stats["hit_rate"] == 0.0
        assert stats["utilization"] == 0.0

    async def test_evict_lru_empty_noop(self, manager):
        await manager._evict_lru()
        assert manager.stats["evictions"] == 0

    async def test_evict_lru_removes_oldest(self, manager):
        await manager.set("a", 1)
        await manager.set("b", 2)
        manager.cache["a"].last_accessed = 0.0
        await manager._evict_lru()
        assert "a" not in manager.cache
        assert manager.stats["evictions"] == 1

    async def test_cleanup_expired_entries(self, manager):
        await manager.set("a", 1, ttl=1)
        await manager.set("b", 2, ttl=1)
        await manager.set("c", 3, ttl=9999)
        manager.cache["a"].created_at -= 100
        manager.cache["b"].created_at -= 100
        await manager._cleanup_expired_entries()
        assert "a" not in manager.cache
        assert "b" not in manager.cache
        assert "c" in manager.cache
        assert manager.stats["evictions"] == 2

    async def test_cleanup_expired_entries_none(self, manager):
        await manager.set("a", 1, ttl=9999)
        await manager._cleanup_expired_entries()
        assert "a" in manager.cache
        assert manager.stats["evictions"] == 0

    async def test_cleanup_loop_cancel_breaks(self, manager):
        manager.cleanup_interval = 0.01
        manager.is_initialized = True
        task = asyncio.create_task(manager._cleanup_loop())
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert task.done()

    async def test_cleanup_loop_handles_errors(self, manager, monkeypatch):
        manager.cleanup_interval = 0.01
        manager.is_initialized = True
        state = {"calls": 0}
        real_sleep = asyncio.sleep

        async def fake_sleep(delay):
            await real_sleep(0)

        async def flaky():
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("cleanup boom")

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(manager, "_cleanup_expired_entries", flaky)
        task = asyncio.create_task(manager._cleanup_loop())
        for _ in range(1000):
            if state["calls"] >= 2:
                break
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert state["calls"] >= 2
