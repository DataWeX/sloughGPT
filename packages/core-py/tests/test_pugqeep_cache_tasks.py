"""Tests for pugqeep cache, point_weight, task_queue, store, and facade modules."""

import base64
import json
import time
from pathlib import Path

import numpy as np
import pytest

from domains.infrastructure.pugqeep.cache import (
    CacheEntry,
    CacheStats,
    DiskStore,
    EvictionPolicy,
    HotStore,
    MemoryStore as CacheMemoryStore,
    Tier,
    TieredCache,
)
from domains.infrastructure.pugqeep.compressor import PointCompressor
from domains.infrastructure.pugqeep.config import (
    CompressorConfig,
    LibraryConfig,
    PointConfig,
)
from domains.infrastructure.pugqeep.facade import PGQ
from domains.infrastructure.pugqeep.library import PointLibrary
from domains.infrastructure.pugqeep.point import Point
from domains.infrastructure.pugqeep.point_weight import PointWeight
from domains.infrastructure.pugqeep.store import (
    DirectoryStore,
    JSONStore,
    MemoryStore as FunctionMemoryStore,
)
from domains.infrastructure.pugqeep.task_queue import (
    Task,
    TaskPriority,
    TaskQueue,
    TaskStatus,
)


# ---- PointWeight ---------------------------------------------------------


class TestPointWeight:
    def test_init_stores_fields(self):
        p = Point(identity="w", function_type="linear", params={"a": 1.0, "b": 0.0})
        pw = PointWeight(p, shape=(4, 2), dtype="float32")
        assert pw.point is p
        assert pw.shape == (4, 2)
        assert pw.dtype == "float32"
        assert pw._cached is None

    def test_generate_reshapes_and_caches(self):
        p = Point(identity="w", function_type="linear", params={"a": 2.0, "b": 1.0})
        pw = PointWeight(p, shape=(2, 4))
        out = pw.generate()
        assert out.shape == (2, 4)
        assert out.dtype == np.float32
        assert pw._cached is not None
        assert pw.generate() is out

    def test_data_property_lazy(self):
        p = Point(identity="w", function_type="cluster",
                  params={"centroids": np.array([1.0, 5.0], dtype=np.float32),
                          "assignments": np.array([0, 1, 0, 1], dtype=np.uint8)})
        pw = PointWeight(p, shape=(2, 2))
        np.testing.assert_array_equal(pw.data, np.array([[1, 5], [1, 5]]))

    def test_invalidate_cache(self):
        p = Point(identity="w", function_type="linear", params={"a": 1.0, "b": 0.0})
        pw = PointWeight(p, shape=(3,))
        first = pw.generate()
        assert first is pw._cached
        pw.invalidate_cache()
        assert pw._cached is None
        assert pw.generate() is not first

    def test_from_array_cluster(self):
        rng = np.random.RandomState(1)
        weights = rng.randn(64).astype(np.float32)
        pw = PointWeight.from_array(weights, identity="w", method="cluster", n_clusters=8)
        assert pw.point.function_type == "cluster"
        assert pw.shape == weights.shape
        assert pw.accuracy() > 0.5

    def test_from_array_function(self):
        i = np.arange(100, dtype=np.float32)
        weights = (2.0 * i + 1.0).reshape(10, 10)
        pw = PointWeight.from_array(weights, identity="w", method="function")
        assert pw.point.function_type == "linear"
        np.testing.assert_allclose(pw.generate(), weights, atol=1e-2)

    def test_from_array_auto_picks_cluster_for_random(self):
        rng = np.random.RandomState(2)
        weights = rng.randn(128).astype(np.float32)
        pw = PointWeight.from_array(weights, identity="w", method="auto", n_clusters=16)
        assert pw.accuracy() > 0.5

    def test_from_point(self):
        p = Point(identity="w", function_type="raw",
                  params={"data_b64": base64.b64encode(np.arange(4, dtype=np.float32).tobytes()).decode(),
                          "dtype": "float32"},
                  dtype="float32", shape=(2, 2))
        pw = PointWeight.from_point(p, shape=(2, 2))
        assert pw.shape == (2, 2)
        np.testing.assert_array_equal(pw.generate(), np.arange(4).reshape(2, 2))

    def test_nbytes_and_accuracy(self):
        p = Point(identity="w", function_type="cluster",
                  params={"centroids": np.zeros(4, dtype=np.float32),
                          "assignments": np.zeros(16, dtype=np.uint8)},
                  accuracy=0.9)
        pw = PointWeight(p, shape=(4, 4))
        assert pw.nbytes() == 4 * 4 + 16
        assert pw.accuracy() == 0.9

    def test_repr(self):
        p = Point(identity="w", function_type="cluster",
                  params={"centroids": np.zeros(4, dtype=np.float32),
                          "assignments": np.zeros(4, dtype=np.uint8)},
                  accuracy=0.95)
        pw = PointWeight(p, shape=(2, 2))
        assert "cluster" in repr(pw)
        assert "0.950" in repr(pw)


class _FakeParam:
    def __init__(self, name, data):
        self.name = name
        self.data = data


class _FakeModule:
    def __init__(self, weight, bias):
        self.weight = _FakeParam("weight", weight)
        self.bias = _FakeParam("bias", bias)


class _FakeModel:
    def __init__(self):
        self.modules = {"": self, "blocks.0": _FakeModule(
            np.zeros((4, 4), dtype=np.float32), np.zeros(4, dtype=np.float32))}
        self._params = [
            _FakeParam("weight", np.zeros((4, 4), dtype=np.float32)),
            _FakeParam("bias", np.zeros(4, dtype=np.float32)),
            _FakeParam("tiny", np.zeros(8, dtype=np.float32)),
        ]

    def parameters(self):
        return self._params

    def named_modules(self):
        return list(self.modules.items())


class TestCompressSlonetToPoints:
    def test_compresses_all_weights(self):
        from domains.infrastructure.pugqeep.point_weight import compress_slonet_to_points
        points = compress_slonet_to_points(_FakeModel(), method="cluster", n_clusters=4)
        assert len(points) == 5
        for pw in points.values():
            assert isinstance(pw, PointWeight)

    def test_tiny_arrays_stored_raw(self):
        from domains.infrastructure.pugqeep.point_weight import compress_slonet_to_points
        points = compress_slonet_to_points(_FakeModel(), method="cluster", n_clusters=4)
        assert points["tiny"].point.function_type == "raw"


# ---- CacheEntry / CacheStats --------------------------------------------


class TestCacheEntry:
    def test_touch_updates_metadata(self):
        entry = CacheEntry(key="k", tier=Tier.MEMORY)
        before = entry.last_accessed
        entry.touch()
        assert entry.access_count == 1
        assert entry.last_accessed >= before

    def test_no_ttl_never_expires(self):
        entry = CacheEntry(key="k", tier=Tier.MEMORY)
        assert entry.is_expired() is False

    def test_expired_with_negative_ttl(self):
        entry = CacheEntry(key="k", tier=Tier.MEMORY, created_at=time.time() - 100, ttl=10)
        assert entry.is_expired() is True

    def test_not_expired_within_ttl(self):
        entry = CacheEntry(key="k", tier=Tier.MEMORY, created_at=time.time() - 1, ttl=100)
        assert entry.is_expired() is False


class TestCacheStats:
    def test_hit_rate_empty(self):
        assert CacheStats().hit_rate == 0.0

    def test_hit_rate(self):
        stats = CacheStats(hits=3, misses=1)
        assert stats.hit_rate == pytest.approx(0.75)


# ---- Cache MemoryStore ---------------------------------------------------


class TestCacheMemoryStore:
    def test_get_put_roundtrip(self):
        store = CacheMemoryStore(max_size_bytes=1024)
        store.put("a", np.array([1, 2, 3]), 12)
        assert store.exists("a")
        assert store.size_bytes() == 12
        np.testing.assert_array_equal(store.get("a"), np.array([1, 2, 3]))
        assert store.list_keys() == ["a"]

    def test_get_moves_to_end(self):
        store = CacheMemoryStore(max_size_bytes=1024)
        store.put("a", 1, 1)
        store.put("b", 2, 1)
        store.put("c", 3, 1)
        store.get("a")
        assert store.list_keys() == ["b", "c", "a"]

    def test_get_missing_returns_none(self):
        store = CacheMemoryStore()
        assert store.get("missing") is None

    def test_put_existing_moves_to_end(self):
        store = CacheMemoryStore()
        store.put("a", 1, 1)
        store.put("b", 2, 1)
        store.put("a", 10, 2)
        assert store.list_keys() == ["b", "a"]
        assert store.size_bytes() == 3

    def test_remove(self):
        store = CacheMemoryStore()
        store.put("a", 1, 1)
        assert store.remove("a") is True
        assert store.remove("a") is False
        assert not store.exists("a")
        assert store.size_bytes() == 0

    def test_evict_lru_frees_space(self):
        store = CacheMemoryStore(max_size_bytes=100)
        for i in range(5):
            store.put(f"k{i}", i, 30)
        evicted = store.evict_lru(target_bytes=10)
        assert len(evicted) >= 1
        assert store.size_bytes() <= 100 - 10
        assert all(k in evicted for k in evicted)

    def test_evict_lru_returns_empty_when_under_limit(self):
        store = CacheMemoryStore(max_size_bytes=1000)
        store.put("a", 1, 10)
        assert store.evict_lru(10) == []

    def test_evict_lfu_sorts_by_access_count(self):
        store = CacheMemoryStore(max_size_bytes=100)
        for i in range(4):
            store.put(f"k{i}", i, 40)
        store.get("k3")
        store.get("k3")
        evicted = store.evict_lfu(target_bytes=10, access_counts={"k0": 0, "k1": 1, "k2": 2, "k3": 5})
        assert "k0" in evicted
        assert "k3" not in evicted

    def test_evict_lfu_returns_empty_when_enough_space(self):
        store = CacheMemoryStore(max_size_bytes=1000)
        store.put("a", 1, 10)
        assert store.evict_lfu(target_bytes=900, access_counts={"a": 0}) == []


# ---- DiskStore -----------------------------------------------------------


class TestDiskStore:
    def test_ndarray_roundtrip(self, tmp_path):
        store = DiskStore(tmp_path)
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        store.put("weights/layer", arr)
        loaded = store.get("weights/layer")
        np.testing.assert_array_equal(loaded, arr)
        assert store.exists("weights/layer")
        assert "weights_layer" in store.list_keys()

    def test_json_value_roundtrip(self, tmp_path):
        store = DiskStore(tmp_path)
        store.put("meta", {"x": 1, "y": [1, 2, 3]})
        assert store.get("meta") == {"x": 1, "y": [1, 2, 3]}

    def test_get_missing_returns_none(self, tmp_path):
        store = DiskStore(tmp_path)
        assert store.get("nope") is None

    def test_remove(self, tmp_path):
        store = DiskStore(tmp_path)
        store.put("a", np.zeros(4))
        assert store.remove("a") is True
        assert store.remove("a") is False
        assert store.get("a") is None

    def test_size_bytes(self, tmp_path):
        store = DiskStore(tmp_path)
        store.put("a", np.zeros(16, dtype=np.float32))
        assert store.size_bytes() > 0

    def test_put_with_extra_meta(self, tmp_path):
        store = DiskStore(tmp_path)
        store.put("a", np.zeros(4), meta={"demoted_from": "memory"})
        meta = json.loads((tmp_path / "a.meta.json").read_text())
        assert meta["demoted_from"] == "memory"


# ---- HotStore ------------------------------------------------------------


class TestHotStore:
    def test_delegates_to_memory(self):
        hot = HotStore(max_size_bytes=1024)
        hot.put("a", np.array([1, 2]), 8)
        assert hot.exists("a")
        np.testing.assert_array_equal(hot.get("a"), np.array([1, 2]))
        assert hot.size_bytes() == 8
        assert hot.list_keys() == ["a"]
        assert hot.remove("a") is True
        assert not hot.exists("a")


# ---- TieredCache ---------------------------------------------------------


class TestTieredCache:
    def _cache(self, **kwargs):
        kwargs.setdefault("memory_max_mb", 512)
        kwargs.setdefault("hot_max_mb", 128)
        return TieredCache(**kwargs)

    def test_put_get_memory(self):
        cache = self._cache()
        cache.put("a", np.array([1, 2, 3]))
        np.testing.assert_array_equal(cache.get("a"), np.array([1, 2, 3]))
        assert cache._entries["a"].tier == Tier.MEMORY
        assert cache.stats()["hits"] == 1
        assert cache.stats()["misses"] == 0

    def test_put_get_hot(self):
        cache = self._cache()
        cache.put("a", np.array([1, 2]), tier=Tier.HOT)
        np.testing.assert_array_equal(cache.get("a"), np.array([1, 2]))
        assert cache._entries["a"].tier == Tier.HOT

    def test_put_get_disk(self, tmp_path):
        cache = self._cache(disk_dir=tmp_path)
        cache.put("a", np.array([1, 2, 3]), tier=Tier.DISK)
        np.testing.assert_array_equal(cache.get("a"), np.array([1, 2, 3]))
        assert cache._entries["a"].tier == Tier.DISK

    def test_get_unknown_key_counts_miss(self):
        cache = self._cache()
        assert cache.get("nope") is None
        assert cache.stats()["misses"] == 1

    def test_ttl_expired_removes_entry(self):
        cache = self._cache()
        cache.put("a", 1, ttl=0.001)
        time.sleep(0.01)
        assert cache.get("a") is None
        assert cache.stats()["misses"] == 1
        assert not cache.exists("a")

    def test_auto_promote_hot_to_memory(self):
        cache = self._cache(promote_threshold=3)
        cache.put("a", np.array([1, 2]), tier=Tier.HOT)
        for _ in range(3):
            cache.get("a")
        assert cache._entries["a"].tier == Tier.MEMORY
        assert cache.stats()["promotions"] == 1

    def test_no_promote_below_threshold(self):
        cache = self._cache(promote_threshold=5)
        cache.put("a", np.array([1, 2]), tier=Tier.HOT)
        for _ in range(3):
            cache.get("a")
        assert cache._entries["a"].tier == Tier.HOT

    def test_auto_promote_disabled(self):
        cache = self._cache(promote_threshold=1, auto_promote=False)
        cache.put("a", np.array([1, 2]), tier=Tier.HOT)
        cache.get("a")
        assert cache._entries["a"].tier == Tier.HOT

    def test_auto_promote_disk_to_hot(self, tmp_path):
        cache = self._cache(promote_threshold=2, disk_dir=tmp_path)
        cache.put("a", np.array([1, 2, 3]), tier=Tier.DISK)
        for _ in range(2):
            cache.get("a")
        assert cache._entries["a"].tier == Tier.HOT

    def test_pinned_entry_not_promoted(self):
        cache = self._cache(promote_threshold=1)
        cache.put("a", np.array([1, 2]), tier=Tier.HOT, pinned=True)
        for _ in range(3):
            cache.get("a")
        assert cache._entries["a"].tier == Tier.HOT
        assert cache.stats()["promotions"] == 0

    def test_remove_all_tiers(self, tmp_path):
        cache = self._cache(disk_dir=tmp_path)
        cache.put("a", np.array([1]), tier=Tier.MEMORY)
        cache.put("b", np.array([1]), tier=Tier.DISK)
        assert cache.remove("a") is True
        assert cache.remove("b") is True
        assert cache.remove("c") is False
        assert cache.list_keys() == []

    def test_exists_and_list_keys(self):
        cache = self._cache()
        cache.put("a", 1)
        cache.put("b", 2, tier=Tier.HOT)
        assert cache.exists("a")
        assert set(cache.list_keys()) == {"a", "b"}
        assert cache.list_keys(Tier.MEMORY) == ["a"]
        assert cache.list_keys(Tier.HOT) == ["b"]

    def test_stats_fields(self):
        cache = self._cache()
        cache.put("a", np.zeros(8, dtype=np.float32))
        cache.get("a")
        stats = cache.stats()
        assert stats["total_entries"] == 1
        assert stats["hits"] == 1
        assert stats["memory_size"] == 32
        assert stats["tier_counts"]["memory"] == 1

    def test_cleanup_expired(self):
        cache = self._cache()
        cache.put("a", 1, ttl=0.001)
        cache.put("b", 2)
        time.sleep(0.01)
        removed = cache.cleanup_expired()
        assert removed == 1
        assert not cache.exists("a")
        assert cache.exists("b")

    def test_evict_demotes_to_hot(self):
        cache = self._cache(memory_max_mb=1)
        cache.put("a", np.zeros(400_000, dtype=np.float32), size_bytes=400_000)
        cache.put("b", np.zeros(400_000, dtype=np.float32), size_bytes=400_000)
        cache.put("c", np.zeros(400_000, dtype=np.float32), size_bytes=400_000)
        assert cache._entries["a"].tier == Tier.HOT
        assert cache.stats()["evictions"] >= 1
        assert cache.stats()["demotions"] >= 1

    def test_memory_evict_writes_to_disk(self, tmp_path):
        cache = self._cache(memory_max_mb=1, disk_dir=tmp_path)
        cache.put("a", np.zeros(400_000, dtype=np.float32), size_bytes=400_000)
        cache.put("b", np.zeros(400_000, dtype=np.float32), size_bytes=400_000)
        cache.put("c", np.zeros(400_000, dtype=np.float32), size_bytes=400_000)
        assert cache._entries["a"].tier == Tier.DISK
        assert cache._entries["b"].tier == Tier.MEMORY

    def test_hot_evict(self, tmp_path):
        cache = self._cache(hot_max_mb=1, disk_dir=tmp_path)
        cache.put("a", np.zeros(400_000, dtype=np.float32), size_bytes=400_000, tier=Tier.HOT)
        cache.put("b", np.zeros(400_000, dtype=np.float32), size_bytes=400_000, tier=Tier.HOT)
        cache.put("c", np.zeros(400_000, dtype=np.float32), size_bytes=400_000, tier=Tier.HOT)
        assert cache._entries["a"].tier == Tier.DISK

    def test_lfu_eviction_policy(self, tmp_path):
        cache = self._cache(memory_max_mb=1, eviction_policy=EvictionPolicy.LFU, disk_dir=tmp_path)
        cache.put("a", np.zeros(400_000, dtype=np.float32), size_bytes=400_000)
        cache.put("b", np.zeros(400_000, dtype=np.float32), size_bytes=400_000)
        cache.get("a")
        cache.put("c", np.zeros(400_000, dtype=np.float32), size_bytes=400_000)
        assert cache._entries["a"].tier == Tier.MEMORY
        assert cache._entries["b"].tier == Tier.DISK

    def test_get_miss_when_data_absent_from_all_tiers(self):
        cache = self._cache()
        cache.put("a", np.array([1, 2, 3]))
        cache._memory.remove("a")
        assert cache.get("a") is None
        assert cache.stats()["misses"] == 1
        assert cache.exists("a")

    def test_lfu_hot_eviction_demotes_to_disk(self, tmp_path):
        cache = self._cache(hot_max_mb=1, eviction_policy=EvictionPolicy.LFU, disk_dir=tmp_path)
        cache.put("a", np.zeros(400_000, dtype=np.float32), size_bytes=400_000, tier=Tier.HOT)
        cache.get("a")
        cache.put("b", np.zeros(400_000, dtype=np.float32), size_bytes=400_000, tier=Tier.HOT)
        cache.put("c", np.zeros(400_000, dtype=np.float32), size_bytes=400_000, tier=Tier.HOT)
        assert cache._entries["b"].tier == Tier.DISK
        assert cache.stats()["evictions"] >= 1
        assert cache.stats()["demotions"] >= 1

    def test_public_evict_method(self):
        cache = self._cache(memory_max_mb=1)
        cache.put("a", np.zeros(400_000, dtype=np.float32), size_bytes=400_000)
        cache.put("b", np.zeros(400_000, dtype=np.float32), size_bytes=400_000)
        freed = cache.evict(Tier.MEMORY, 400_000)
        assert freed >= 400_000
        assert cache.stats()["demotions"] >= 1


# ---- Task / TaskQueue ----------------------------------------------------


class TestTask:
    def test_defaults(self):
        task = Task()
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.NORMAL
        assert task.max_retries == 3
        assert task.retries == 0
        assert task.id

    def test_to_dict_roundtrip(self):
        task = Task(name="t", data={"x": 1}, priority=TaskPriority.HIGH,
                    tree_id="tree", metadata={"a": 1})
        d = task.to_dict()
        assert d["status"] == "pending"
        assert d["priority"] == 2
        restored = Task.from_dict(d)
        assert restored.name == "t"
        assert restored.priority == TaskPriority.HIGH
        assert restored.tree_id == "tree"
        assert restored.metadata == {"a": 1}


class TestTaskQueue:
    def test_submit_and_get(self):
        q = TaskQueue("q")
        task = Task(name="t")
        q.submit(task)
        assert q.get_task(task.id) is task

    def test_queue_full_raises(self):
        q = TaskQueue("q", max_size=2)
        q.submit(Task())
        q.submit(Task())
        with pytest.raises(ValueError):
            q.submit(Task())

    def test_submit_many(self):
        q = TaskQueue("q")
        tasks = [Task(name=f"t{i}") for i in range(3)]
        q.submit_many(tasks)
        assert len(q.list_tasks()) == 3

    def test_next_priority_order(self):
        q = TaskQueue("q")
        low = Task(name="low", priority=TaskPriority.LOW)
        urgent = Task(name="urgent", priority=TaskPriority.URGENT)
        normal = Task(name="normal", priority=TaskPriority.NORMAL)
        q.submit(low)
        q.submit(urgent)
        q.submit(normal)
        assert q.next().name == "urgent"
        assert q.next().name == "normal"
        assert q.next().name == "low"
        assert q.next() is None

    def test_next_same_priority_fifo(self):
        q = TaskQueue("q")
        t1 = Task(name="t1")
        t2 = Task(name="t2")
        t1.created_at = 1.0
        t2.created_at = 2.0
        q.submit(t1)
        q.submit(t2)
        assert q.next().name == "t1"

    def test_complete(self):
        q = TaskQueue("q")
        task = Task(name="t")
        q.submit(task)
        q.next()
        assert q.complete(task.id, "result").result == "result"
        assert task.status == TaskStatus.COMPLETED
        assert q.complete("missing") is None

    def test_fail_retries_then_gives_up(self):
        q = TaskQueue("q")
        task = Task(name="t", max_retries=1)
        q.submit(task)
        q.next()
        q.fail(task.id, "err1")
        assert task.status == TaskStatus.PENDING
        assert task.retries == 1
        q.next()
        q.fail(task.id, "err2")
        assert task.status == TaskStatus.FAILED
        assert task.error == "err2"
        assert q.fail("missing", "x") is None

    def test_cancel_pending(self):
        q = TaskQueue("q")
        task = Task(name="t")
        q.submit(task)
        q.cancel(task.id)
        assert task.status == TaskStatus.CANCELLED
        assert q.next() is None
        assert q.cancel("missing") is None

    def test_cancel_running(self):
        q = TaskQueue("q")
        task = Task(name="t")
        q.submit(task)
        q.next()
        q.cancel(task.id)
        assert task.status == TaskStatus.CANCELLED

    def test_pause_resume(self):
        q = TaskQueue("q")
        task = Task(name="t")
        q.submit(task)
        q.pause()
        assert q.next() is None
        assert q.stats()["paused"] is True
        q.resume()
        assert q.next() is task

    def test_on_complete_callback(self):
        q = TaskQueue("q")
        calls = []
        q.on_complete(lambda t: calls.append(t.name))
        task = Task(name="t")
        q.submit(task)
        q.next()
        q.complete(task.id)
        assert calls == ["t"]

    def test_register_handler(self):
        q = TaskQueue("q")
        q.register_handler("build", lambda t: None)
        assert q.stats()["handlers"] == ["build"]

    def test_list_tasks_filter(self):
        q = TaskQueue("q")
        t1 = Task(name="t1")
        t2 = Task(name="t2", priority=TaskPriority.HIGH)
        q.submit(t1)
        q.submit(t2)
        assert len(q.list_tasks(TaskStatus.PENDING)) == 2
        q.next()
        assert len(q.list_tasks(TaskStatus.RUNNING)) == 1

    def test_stats(self):
        q = TaskQueue("q")
        t = Task(name="t")
        q.submit(t)
        stats = q.stats()
        assert stats["total"] == 1
        assert stats["pending"] == 1
        assert stats["paused"] is False

    def test_clear_completed(self):
        q = TaskQueue("q")
        t = Task(name="t")
        q.submit(t)
        q.next()
        q.complete(t.id)
        assert q.clear_completed() == 1
        assert q.get_task(t.id) is None

    def test_save_load_roundtrip(self, tmp_path):
        q = TaskQueue("q", storage_dir=tmp_path)
        t1 = Task(name="a", priority=TaskPriority.HIGH)
        t2 = Task(name="b")
        q.submit(t1)
        q.submit(t2)
        q.next()
        q.complete(t1.id, "done")
        path = q.save()
        assert path.exists()
        loaded = TaskQueue.load(path)
        assert loaded.name == "q"
        assert loaded.get_task(t2.id).status == TaskStatus.PENDING
        assert loaded.get_task(t1.id).status == TaskStatus.COMPLETED
        assert loaded.get_task(t1.id).result == "done"

    def test_save_no_storage_dir_raises(self):
        q = TaskQueue("q")
        with pytest.raises(ValueError):
            q.save()

    def test_persist_on_mutations(self, tmp_path):
        q = TaskQueue("q", storage_dir=tmp_path)
        t = Task(name="t")
        q.submit(t)
        assert (tmp_path / "q.tasks.json").exists()

    def test_fail_with_storage_persists(self, tmp_path):
        q = TaskQueue("q", storage_dir=tmp_path)
        t = Task(name="t", max_retries=0)
        q.submit(t)
        q.next()
        q.fail(t.id, "err")
        assert (tmp_path / "q.tasks.json").exists()

    def test_cancel_with_storage_persists(self, tmp_path):
        q = TaskQueue("q", storage_dir=tmp_path)
        t = Task(name="t")
        q.submit(t)
        q.cancel(t.id)
        assert (tmp_path / "q.tasks.json").exists()

    def test_clear_completed_with_storage_persists(self, tmp_path):
        q = TaskQueue("q", storage_dir=tmp_path)
        t = Task(name="t")
        q.submit(t)
        q.next()
        q.complete(t.id)
        q.clear_completed()
        assert (tmp_path / "q.tasks.json").exists()

    def test_persist_failure_is_swallowed(self, tmp_path, monkeypatch):
        q = TaskQueue("q", storage_dir=tmp_path)

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(TaskQueue, "save", boom)
        t = Task(name="t")
        q.submit(t)
        assert (tmp_path / "q.tasks.json").exists() is False

    def test_callback_error_is_swallowed(self):
        q = TaskQueue("q")

        def boom(task):
            raise RuntimeError("cb failed")

        q.on_complete(boom)
        t = Task(name="t")
        q.submit(t)
        q.next()
        assert q.complete(t.id).status == TaskStatus.COMPLETED


# ---- Function stores -----------------------------------------------------


class TestFunctionMemoryStore:
    def test_crud(self):
        store = FunctionMemoryStore()
        p = Point(identity="w", function_type="linear", params={"a": 1.0, "b": 0.0})
        store.save(p)
        assert store.load("w") is p
        assert store.count() == 1
        assert store.list_all() == [p]
        assert store.remove("w") is True
        assert store.remove("w") is False
        assert store.load("w") is None
        store.clear()
        assert store.count() == 0


class TestJSONStore:
    def test_roundtrip(self, tmp_path):
        store = JSONStore(tmp_path / "points.json")
        p = Point(identity="w", function_type="cluster",
                  params={"centroids": np.array([1.0, 2.0], dtype=np.float32),
                          "assignments": np.array([0, 1, 0, 1], dtype=np.uint8)},
                  accuracy=0.9)
        store.save(p)
        loaded = store.load("w")
        np.testing.assert_array_equal(loaded.params["centroids"], p.params["centroids"])
        np.testing.assert_array_equal(loaded.params["assignments"], p.params["assignments"])

    def test_load_missing_returns_none(self, tmp_path):
        store = JSONStore(tmp_path / "points.json")
        assert store.load("nope") is None

    def test_remove_and_clear(self, tmp_path):
        store = JSONStore(tmp_path / "points.json")
        store.save(Point(identity="a", function_type="linear", params={"a": 1, "b": 0}))
        store.save(Point(identity="b", function_type="linear", params={"a": 2, "b": 1}))
        assert store.count() == 2
        assert store.remove("a") is True
        assert store.count() == 1
        store.clear()
        assert store.count() == 0

    def test_reload_persists(self, tmp_path):
        path = tmp_path / "points.json"
        store = JSONStore(path)
        store.save(Point(identity="a", function_type="linear", params={"a": 1, "b": 0}))
        store2 = JSONStore(path)
        assert store2.count() == 1
        assert store2.load("a").params["a"] == 1


class TestDirectoryStore:
    def test_roundtrip(self, tmp_path):
        store = DirectoryStore(tmp_path)
        p = Point(identity="a/b", function_type="periodic",
                  params={"a": 1.0, "b": 2.0, "w": 0.5})
        store.save(p)
        loaded = store.load("a/b")
        assert loaded.function_type == "periodic"
        assert loaded.params["a"] == 1.0

    def test_load_missing_returns_none(self, tmp_path):
        store = DirectoryStore(tmp_path)
        assert store.load("nope") is None

    def test_crud_and_path_safety(self, tmp_path):
        store = DirectoryStore(tmp_path)
        store.save(Point(identity="x/y", function_type="raw",
                         params={"data_b64": base64.b64encode(np.ones(4, dtype=np.float32).tobytes()).decode(),
                                 "dtype": "float32"}))
        store.save(Point(identity="z", function_type="linear", params={"a": 1, "b": 0}))
        assert store.count() == 2
        assert len(store.list_all()) == 2
        assert store.remove("x/y") is True
        assert store.count() == 1
        store.clear()
        assert store.count() == 0


# ---- Point serialization -------------------------------------------------


class TestPointSerialization:
    def test_cluster_bytes_roundtrip(self):
        p = Point(identity="w", function_type="cluster",
                  params={"centroids": np.array([0.1, 0.5, 0.9], dtype=np.float32),
                          "assignments": np.array([0, 1, 2, 0], dtype=np.uint8)})
        restored = Point.from_bytes(p.to_bytes())
        assert restored.function_type == "cluster"
        np.testing.assert_array_equal(restored.params["centroids"], p.params["centroids"])
        np.testing.assert_array_equal(restored.params["assignments"], p.params["assignments"])

    def test_cluster_bytes_with_residual_roundtrip(self):
        p = Point(identity="w", function_type="cluster",
                  params={"centroids": np.array([0.1, 0.5], dtype=np.float32),
                          "assignments": np.array([0, 1, 0, 1], dtype=np.uint8)},
                  residual=np.array([0.01, -0.02, 0.03, -0.04], dtype=np.float32))
        restored = Point.from_bytes(p.to_bytes())
        np.testing.assert_allclose(restored.residual, p.residual, atol=1e-6)

    def test_periodic_bytes_roundtrip(self):
        p = Point(identity="w", function_type="periodic",
                  params={"a": 1.5, "b": -2.0, "w": 0.25})
        restored = Point.from_bytes(p.to_bytes())
        assert restored.function_type == "periodic"
        assert restored.params["a"] == pytest.approx(1.5)

    def test_linear_bytes_roundtrip(self):
        p = Point(identity="w", function_type="linear",
                  params={"a": 2.0, "b": 1.0})
        restored = Point.from_bytes(p.to_bytes())
        assert restored.function_type == "linear"
        assert restored.params["a"] == pytest.approx(2.0)

    def test_polynomial_bytes_roundtrip(self):
        p = Point(identity="w", function_type="polynomial",
                  params={"a": 1.0, "b": 2.0, "c": 3.0})
        restored = Point.from_bytes(p.to_bytes())
        assert restored.function_type == "polynomial"
        assert restored.params["c"] == pytest.approx(3.0)

    def test_raw_bytes_roundtrip(self):
        arr = np.arange(8, dtype=np.float32)
        p = Point(identity="w", function_type="raw",
                  params={"data_b64": base64.b64encode(arr.tobytes()).decode(),
                          "dtype": "float32"})
        restored = Point.from_bytes(p.to_bytes())
        np.testing.assert_array_equal(restored.generate(8), arr)

    def test_unknown_type_code_raises(self):
        with pytest.raises(ValueError):
            Point.from_bytes(b"\x00\x00\x00\x00")

    def test_unknown_generate_type_raises(self):
        p = Point(identity="w", function_type="bogus", params={})
        with pytest.raises(ValueError):
            p.generate(4)

    def test_nbytes_raw(self):
        arr = np.arange(8, dtype=np.float32)
        p = Point(identity="w", function_type="raw",
                  params={"data_b64": base64.b64encode(arr.tobytes()).decode(),
                          "dtype": "float32"})
        assert p.nbytes() == 32

    def test_nbytes_function(self):
        p = Point(identity="w", function_type="linear", params={"a": 1.0, "b": 0.0})
        assert p.nbytes() == 4 + 2 * 4

    def test_to_dict_from_dict_cluster(self):
        p = Point(identity="w", function_type="cluster",
                  params={"centroids": np.array([1.0, 2.0], dtype=np.float32),
                          "assignments": np.array([0, 1], dtype=np.uint8)},
                  accuracy=0.95, shape=(2, 2))
        d = p.to_dict()
        restored = Point.from_dict(d)
        np.testing.assert_array_equal(restored.params["centroids"], p.params["centroids"])
        assert restored.accuracy == 0.95
        assert restored.shape == (2, 2)

    def test_to_dict_from_dict_with_residual(self):
        p = Point(identity="w", function_type="linear",
                  params={"a": 1.0, "b": 0.0},
                  residual=np.array([0.1, 0.2], dtype=np.float32))
        restored = Point.from_dict(p.to_dict())
        np.testing.assert_allclose(restored.residual, p.residual, atol=1e-6)

    def test_to_dict_raw(self):
        p = Point(identity="w", function_type="raw",
                  params={"data_b64": "x", "dtype": "float32"})
        assert Point.from_dict(p.to_dict()).params["dtype"] == "float32"


# ---- PointCompressor -----------------------------------------------------


class TestCompressorMeasure:
    def _compressor(self):
        return PointCompressor()

    def test_compress_with_config(self):
        config = CompressorConfig(n_clusters=8, lloyd_iterations=2,
                                   gap_fill_iterations=0, method="cluster")
        c = PointCompressor(config)
        assert c.n_clusters == 8
        assert c.lloyd_iterations == 2
        assert c.method == "cluster"

    def test_compress_unknown_method_raises(self):
        with pytest.raises(ValueError):
            self._compressor().compress(np.zeros(16), "w", method="bogus")

    def test_decompress_delegates(self):
        p = Point(identity="w", function_type="cluster",
                  params={"centroids": np.array([1.0, 2.0], dtype=np.float32),
                          "assignments": np.array([0, 1, 0], dtype=np.uint8)})
        out = self._compressor().decompress(p, 3)
        np.testing.assert_array_equal(out, np.array([1, 2, 1]))

    def test_measure_cluster(self):
        rng = np.random.RandomState(3)
        weights = rng.randn(64).astype(np.float32)
        point = self._compressor().compress_cluster(weights, "w", n_clusters=8)
        m = self._compressor().measure_compression(weights, point)
        assert m["raw_bytes"] == weights.nbytes
        assert m["compressed_bytes"] > 0
        assert m["ratio"] > 1
        assert m["function_type"] == "cluster"

    def test_measure_function_with_residual(self):
        rng = np.random.RandomState(4)
        weights = rng.randn(128).astype(np.float32)
        point = self._compressor().compress_function(weights, "w")
        m = self._compressor().measure_compression(weights, point)
        assert m["compressed_bytes"] > 4
        assert m["accuracy"] < 0.99

    def test_measure_raw(self):
        weights = np.zeros(8, dtype=np.float32)
        point = Point(identity="w", function_type="raw",
                      params={"data_b64": base64.b64encode(weights.tobytes()).decode(),
                              "dtype": "float32"})
        m = self._compressor().measure_compression(weights, point)
        assert m["compressed_bytes"] == weights.nbytes

    def test_compress_function_linear_exact_no_residual(self):
        i = np.arange(100, dtype=np.float32)
        weights = 3.0 * i + 5.0
        point = self._compressor().compress_function(weights, "w")
        assert point.function_type == "linear"
        assert point.residual is None
        assert point.accuracy > 0.99

    def test_compress_function_random_stores_residual(self):
        rng = np.random.RandomState(5)
        weights = rng.randn(256).astype(np.float32)
        point = self._compressor().compress_function(weights, "w")
        assert point.residual is not None

    def test_compress_cluster_residual_none(self):
        rng = np.random.RandomState(6)
        weights = rng.randn(64).astype(np.float32)
        point = self._compressor().compress_cluster(weights, "w", n_clusters=8)
        assert point.residual is None


# ---- PointLibrary --------------------------------------------------------


class TestPointLibrary:
    def _library(self):
        return PointLibrary("test_lib")

    def test_crud(self):
        lib = self._library()
        p = Point(identity="w1", function_type="linear", params={"a": 1.0, "b": 0.0})
        lib.add(p)
        assert lib.has("w1")
        assert lib.get("w1") is p
        assert lib.remove("w1") is True
        assert lib.remove("w1") is False
        assert not lib.has("w1")

    def test_list_by_type(self):
        lib = self._library()
        lib.add(Point(identity="a", function_type="cluster",
                      params={"centroids": np.zeros(2, dtype=np.float32),
                              "assignments": np.zeros(2, dtype=np.uint8)}))
        lib.add(Point(identity="b", function_type="cluster",
                      params={"centroids": np.zeros(2, dtype=np.float32),
                              "assignments": np.zeros(2, dtype=np.uint8)}))
        lib.add(Point(identity="c", function_type="linear", params={"a": 1.0, "b": 0.0}))
        assert len(lib.list_by_type("cluster")) == 2
        assert len(lib.list_by_type("linear")) == 1
        assert lib.list_by_type("raw") == []

    def test_clear(self):
        lib = self._library()
        lib.add(Point(identity="a", function_type="linear", params={"a": 1, "b": 0}))
        lib.clear()
        assert lib.list_all() == []
        assert lib.stats()["total_points"] == 0

    def test_compress_and_store_cluster(self):
        lib = self._library()
        rng = np.random.RandomState(7)
        weights = rng.randn(64).astype(np.float32)
        point = lib.compress_and_store(weights, "w", method="cluster", n_clusters=8)
        assert point.function_type == "cluster"
        assert lib.has("w")

    def test_compress_and_store_function(self):
        lib = self._library()
        i = np.arange(50, dtype=np.float32)
        point = lib.compress_and_store(i * 2.0, "w", method="function")
        assert point.function_type == "linear"

    def test_decompress_to_cluster(self):
        lib = self._library()
        point = lib.compress_and_store(
            np.arange(16, dtype=np.float32), "w", method="cluster", n_clusters=8)
        out = lib.decompress_to("w", shape=(4, 4))
        assert out.shape == (4, 4)

    def test_decompress_to_missing(self):
        lib = self._library()
        assert lib.decompress_to("nope") is None

    def test_search(self):
        lib = self._library()
        lib.add(Point(identity="attn.weight", function_type="linear", params={"a": 1, "b": 0}))
        lib.add(Point(identity="mlp.bias", function_type="linear", params={"a": 1, "b": 0}))
        assert len(lib.search("attn")) == 1
        assert len(lib.search("weight")) == 1

    def test_best_points(self):
        lib = self._library()
        lib.add(Point(identity="low", function_type="linear", params={"a": 1, "b": 0}, accuracy=0.5))
        lib.add(Point(identity="high", function_type="linear", params={"a": 1, "b": 0}, accuracy=0.99))
        lib.add(Point(identity="mid", function_type="linear", params={"a": 1, "b": 0}, accuracy=0.8))
        assert [p.identity for p in lib.best_points(2)] == ["high", "mid"]

    def test_stats_with_points(self):
        lib = self._library()
        lib.add(Point(identity="w", function_type="cluster",
                      params={"centroids": np.zeros(4, dtype=np.float32),
                              "assignments": np.zeros(16, dtype=np.uint8)},
                      accuracy=0.9))
        stats = lib.stats()
        assert stats["total_points"] == 1
        assert stats["types"] == {"cluster": 1}
        assert stats["avg_accuracy"] == pytest.approx(0.9)

    def test_save_without_storage_dir_raises(self):
        lib = self._library()
        with pytest.raises(ValueError):
            lib.save()

    def test_save_load_roundtrip(self, tmp_path):
        lib = self._library()
        lib.add(Point(identity="w", function_type="cluster",
                      params={"centroids": np.array([1.0, 2.0], dtype=np.float32),
                              "assignments": np.array([0, 1, 0, 1], dtype=np.uint8)},
                      accuracy=0.9))
        path = lib.save(tmp_path / "lib.json")
        loaded = PointLibrary.load(path)
        assert loaded.name == "test_lib"
        np.testing.assert_array_equal(loaded.get("w").params["centroids"], np.array([1.0, 2.0]))
        assert loaded.stats()["avg_accuracy"] == pytest.approx(0.9)

    def test_auto_save_config(self, tmp_path):
        config = LibraryConfig(name="auto", storage_dir=tmp_path, auto_save=True)
        lib = PointLibrary(config=config)
        lib.add(Point(identity="w", function_type="linear", params={"a": 1, "b": 0}))
        assert (tmp_path / "auto.points.json").exists()
        lib.remove("w")
        loaded = PointLibrary.load(tmp_path / "auto.points.json")
        assert loaded.has("w") is False


# ---- PGQ facade ----------------------------------------------------------


class TestPGQ:
    def _pgq(self, tmp_path):
        return PGQ(name="sys", storage_dir=tmp_path, cache_dir=tmp_path / "cache")

    def test_init_and_properties(self):
        pgq = PGQ(name="sys")
        assert pgq.name == "sys"
        assert pgq.is_loaded is False
        assert pgq.library is not None
        assert pgq.tree is not None
        assert pgq.cache is not None
        assert pgq.task_queue is not None

    def test_put_compressed_get(self):
        pgq = PGQ(name="sys")
        rng = np.random.RandomState(8)
        data = rng.randn(128).astype(np.float32)
        point = pgq.put("weights", data)
        assert isinstance(point, Point)
        assert pgq.has("weights")
        out = pgq.get("weights")
        assert out.shape == data.shape
        assert out.dtype == np.float32

    def test_put_compress_false_uses_cache(self):
        pgq = PGQ(name="sys")
        data = np.arange(8, dtype=np.float32)
        returned = pgq.put("raw", data, compress=False)
        np.testing.assert_array_equal(returned, data)
        np.testing.assert_array_equal(pgq.get("raw"), data)

    def test_put_method_function(self):
        pgq = PGQ(name="sys")
        i = np.arange(100, dtype=np.float32)
        point = pgq.put("lin", 2.0 * i + 1.0, method="function")
        assert point.function_type == "linear"

    def test_put_raw_and_get_any(self):
        pgq = PGQ(name="sys")
        pgq.put_raw("meta", {"a": 1})
        assert pgq.get_any("meta") == {"a": 1}
        assert pgq.has("meta")

    def test_get_missing_returns_none(self):
        pgq = PGQ(name="sys")
        assert pgq.get("nope") is None

    def test_remove(self):
        pgq = PGQ(name="sys")
        pgq.put("weights", np.zeros(64, dtype=np.float32))
        assert pgq.remove("weights") is True
        assert pgq.remove("weights") is False
        assert not pgq.has("weights")

    def test_remove_raw_data(self):
        pgq = PGQ(name="sys")
        pgq.put_raw("meta", 42)
        assert pgq.remove("meta") is True

    def test_task_ops(self):
        pgq = PGQ(name="sys")
        task = Task(name="build")
        pgq.submit_task(task)
        assert pgq.get_task(task.id) is task
        next_task = pgq.next_task()
        assert next_task.id == task.id
        pgq.complete_task(task.id, "ok")
        assert task.status == TaskStatus.COMPLETED
        pgq.pause_queue()
        pgq.resume_queue()

    def test_task_fail_cancel(self):
        pgq = PGQ(name="sys")
        t1 = Task(name="t1", max_retries=0)
        t2 = Task(name="t2")
        pgq.submit_task(t1)
        pgq.submit_task(t2)
        pgq.next_task()
        pgq.fail_task(t1.id, "boom")
        assert t1.status == TaskStatus.FAILED
        pgq.cancel_task(t2.id)
        assert t2.status == TaskStatus.CANCELLED
        assert pgq.list_tasks(TaskStatus.CANCELLED) == [t2]

    def test_search_and_best(self):
        pgq = PGQ(name="sys")
        pgq.put("attn.weight", np.zeros(64, dtype=np.float32))
        pgq.put("mlp.bias", np.zeros(64, dtype=np.float32))
        assert len(pgq.search("attn")) == 1
        assert len(pgq.best(10)) == 2

    def test_save_load_roundtrip(self, tmp_path):
        pgq = self._pgq(tmp_path)
        pgq.put("weights", np.arange(64, dtype=np.float32))
        task = Task(name="t")
        pgq.submit_task(task)
        path = pgq.save(tmp_path / "sys.json")
        loaded = PGQ.load(path)
        assert loaded.name == "sys"
        assert loaded.has("weights")
        assert loaded.get_task(task.id) is not None

    def test_from_file(self, tmp_path):
        pgq = self._pgq(tmp_path)
        pgq.put("weights", np.arange(32, dtype=np.float32))
        path = pgq.save(tmp_path / "s.json")
        loaded = PGQ.from_file(path)
        assert loaded.name == "sys"
        assert loaded.has("weights")

    def test_stats_and_export(self):
        pgq = PGQ(name="sys")
        pgq.put("weights", np.zeros(64, dtype=np.float32))
        stats = pgq.stats()
        assert set(stats) == {"name", "tree", "cache", "queue"}
        assert pgq.cache_stats()["total_entries"] == 0
        assert "pending" in pgq.queue_stats()
        exported = pgq.export_stats()
        assert exported["version"] == "0.1.0"

    def test_batch_ops(self):
        pgq = PGQ(name="sys")
        data = {
            "a": np.zeros(16, dtype=np.float32),
            "b": np.zeros(32, dtype=np.float32),
            "c": np.zeros(8, dtype=np.float32),
        }
        stats = pgq.put_many(data)
        assert stats["count"] == 3
        assert stats["total_bytes"] > 0
        out = pgq.get_many(["a", "b", "missing"])
        assert set(out) == {"a", "b", "missing"}
        assert out["missing"] is None
        exists = pgq.exists_many(["a", "nope"])
        assert exists == {"a": True, "nope": False}
        assert pgq.remove_many(["a", "b", "c"]) == 3

    def test_cleanup_cache(self):
        pgq = PGQ(name="sys")
        pgq.put_raw("expired", 1, ttl=0.001)
        time.sleep(0.01)
        assert pgq.cleanup_cache() == 1

    def test_submit_training(self):
        pgq = PGQ(name="sys")

        def _train(job_id, tree_id, point_library, is_cancelled):
            assert not is_cancelled()
            point_library.add(Point(identity=f"{tree_id}.w",
                                    function_type="linear",
                                    params={"a": 1.0, "b": 0.0}))
            return {"weights": np.zeros(32, dtype=np.float32)}

        job_id = pgq.submit_training(_train, "job1")
        status = pgq.training_status(job_id)
        assert status is not None
        deadline = time.time() + 5.0
        while not pgq.library.has("sys.w") and time.time() < deadline:
            time.sleep(0.01)
        assert pgq.library.has("sys.w")

    def test_cancel_training_unknown(self):
        pgq = PGQ(name="sys")
        assert pgq.cancel_training("missing-job") is False
