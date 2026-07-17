"""Tests for pugqeep — Point-Graph-Queue system."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import time


class TestPoint:
    def test_cluster_generate(self):
        from domains.infrastructure.pugqeep import Point
        centroids = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        assignments = np.array([0, 1, 2, 0, 1, 2], dtype=np.uint8)
        p = Point(
            identity="test",
            function_type="cluster",
            params={"centroids": centroids, "assignments": assignments},
            accuracy=0.95,
        )
        result = p.generate(6)
        np.testing.assert_array_equal(result, centroids[assignments])

    def test_linear_generate(self):
        from domains.infrastructure.pugqeep import Point
        p = Point(
            identity="test",
            function_type="linear",
            params={"a": 2.0, "b": 1.0},
        )
        result = p.generate(5)
        expected = np.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_roundtrip_dict(self):
        from domains.infrastructure.pugqeep import Point
        p = Point(
            identity="test",
            function_type="linear",
            params={"a": 1.5, "b": 0.5},
            accuracy=0.9,
        )
        d = p.to_dict()
        p2 = Point.from_dict(d)
        assert p2.identity == p.identity
        assert p2.function_type == p.function_type
        assert p2.accuracy == p.accuracy

    def test_nbytes(self):
        from domains.infrastructure.pugqeep import Point
        centroids = np.zeros(16, dtype=np.float32)
        assignments = np.zeros(1000, dtype=np.uint8)
        p = Point(
            identity="test",
            function_type="cluster",
            params={"centroids": centroids, "assignments": assignments},
        )
        assert p.nbytes() == centroids.nbytes + assignments.nbytes

    def test_shape_preserved(self):
        from domains.infrastructure.pugqeep import Point
        p = Point(
            identity="test",
            function_type="linear",
            params={"a": 1.0, "b": 0.0},
            shape=(10, 10),
            dtype="float32",
        )
        assert p.shape == (10, 10)


class TestPointCompressor:
    def test_compress_cluster(self):
        from domains.infrastructure.pugqeep import PointCompressor
        comp = PointCompressor()
        weights = np.random.randn(1000).astype(np.float32)
        point = comp.compress_cluster(weights, "test", n_clusters=8)
        assert point.function_type == "cluster"
        assert point.accuracy > 0.5

    def test_compress_function(self):
        from domains.infrastructure.pugqeep import PointCompressor
        comp = PointCompressor()
        weights = np.arange(100, dtype=np.float32) * 0.01
        point = comp.compress_function(weights, "test")
        assert point.function_type in ("periodic", "linear", "polynomial")
        assert point.accuracy > 0.5

    def test_measure_compression(self):
        from domains.infrastructure.pugqeep import PointCompressor
        comp = PointCompressor()
        weights = np.random.randn(1000).astype(np.float32)
        point = comp.compress_cluster(weights, "test", n_clusters=8)
        stats = comp.measure_compression(weights, point)
        assert "ratio" in stats
        assert "accuracy" in stats
        assert stats["ratio"] > 1.0


class TestPointLibrary:
    def test_add_and_get(self):
        from domains.infrastructure.pugqeep import PointLibrary, Point
        lib = PointLibrary(name="test")
        p = Point(identity="w1", function_type="linear", params={"a": 1.0, "b": 0.0})
        lib.add(p)
        assert lib.has("w1")
        assert lib.get("w1") is p

    def test_remove(self):
        from domains.infrastructure.pugqeep import PointLibrary, Point
        lib = PointLibrary(name="test")
        p = Point(identity="w1", function_type="linear", params={"a": 1.0, "b": 0.0})
        lib.add(p)
        assert lib.remove("w1")
        assert not lib.has("w1")

    def test_search(self):
        from domains.infrastructure.pugqeep import PointLibrary, Point
        lib = PointLibrary(name="test")
        lib.add(Point(identity="attn.qkv", function_type="linear", params={"a": 1.0, "b": 0.0}))
        lib.add(Point(identity="attn.out", function_type="linear", params={"a": 2.0, "b": 0.0}))
        lib.add(Point(identity="ffn.w1", function_type="linear", params={"a": 3.0, "b": 0.0}))
        results = lib.search("attn")
        assert len(results) == 2

    def test_stats(self):
        from domains.infrastructure.pugqeep import PointLibrary, Point
        lib = PointLibrary(name="test")
        lib.add(Point(identity="w1", function_type="linear", params={"a": 1.0, "b": 0.0}))
        s = lib.stats()
        assert s["total_points"] == 1

    def test_save_and_load(self, tmp_path):
        from domains.infrastructure.pugqeep import PointLibrary, Point
        lib = PointLibrary(name="test", storage_dir=tmp_path)
        lib.add(Point(identity="w1", function_type="linear", params={"a": 1.0, "b": 0.0}))
        lib.save()

        loaded = PointLibrary.load(tmp_path / "test.points.json")
        assert loaded.has("w1")


class TestTieredCache:
    def test_memory_put_get(self):
        from domains.infrastructure.pugqeep import TieredCache, Tier
        cache = TieredCache(memory_max_mb=1, hot_max_mb=1)
        cache.put("key1", np.array([1.0, 2.0, 3.0]), Tier.MEMORY)
        result = cache.get("key1")
        assert result is not None
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_cache_hit_rate(self):
        from domains.infrastructure.pugqeep import TieredCache, Tier
        cache = TieredCache(memory_max_mb=1, hot_max_mb=1)
        cache.put("key1", np.array([1.0]), Tier.MEMORY)
        cache.get("key1")
        cache.get("nonexistent")
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_remove(self):
        from domains.infrastructure.pugqeep import TieredCache, Tier
        cache = TieredCache(memory_max_mb=1, hot_max_mb=1)
        cache.put("key1", np.array([1.0]), Tier.MEMORY)
        assert cache.remove("key1")
        assert cache.get("key1") is None


class TestTaskQueue:
    def test_submit_and_next(self):
        from domains.infrastructure.pugqeep import TaskQueue, Task, TaskPriority
        q = TaskQueue(name="test")
        task = Task(name="process", data="input", priority=TaskPriority.HIGH)
        q.submit(task)

        next_task = q.next()
        assert next_task is not None
        assert next_task.id == task.id
        assert next_task.status.value == "running"

    def test_complete(self):
        from domains.infrastructure.pugqeep import TaskQueue, Task
        q = TaskQueue(name="test")
        task = Task(name="process", data="input")
        q.submit(task)
        q.next()

        completed = q.complete(task.id, result="output")
        assert completed.status.value == "completed"
        assert completed.result == "output"

    def test_fail_and_retry(self):
        from domains.infrastructure.pugqeep import TaskQueue, Task
        q = TaskQueue(name="test")
        task = Task(name="process", data="input", max_retries=2)
        q.submit(task)
        q.next()

        failed = q.fail(task.id, "error")
        assert failed.status.value == "pending"  # retry
        assert failed.retries == 1

    def test_priority_ordering(self):
        from domains.infrastructure.pugqeep import TaskQueue, Task, TaskPriority
        q = TaskQueue(name="test")
        q.submit(Task(name="low", priority=TaskPriority.LOW))
        q.submit(Task(name="high", priority=TaskPriority.HIGH))
        q.submit(Task(name="normal", priority=TaskPriority.NORMAL))

        t1 = q.next()
        t2 = q.next()
        t3 = q.next()
        assert t1.name == "high"
        assert t2.name == "normal"
        assert t3.name == "low"

    def test_pause_resume(self):
        from domains.infrastructure.pugqeep import TaskQueue, Task
        q = TaskQueue(name="test")
        q.submit(Task(name="task1"))

        q.pause()
        assert q.next() is None

        q.resume()
        assert q.next() is not None

    def test_stats(self):
        from domains.infrastructure.pugqeep import TaskQueue, Task
        q = TaskQueue(name="test")
        q.submit(Task(name="task1"))
        s = q.stats()
        assert s["total"] == 1
        assert s["pending"] == 1


class TestPGQ:
    def test_put_and_get(self):
        from domains.infrastructure.pugqeep import PGQ
        sys = PGQ(name="test", n_clusters=8)
        w = np.random.randn(100).astype(np.float32) * 0.01
        sys.put("weight1", w, compress=True)

        recovered = sys.get("weight1")
        assert recovered is not None
        assert recovered.shape == (100,)

    def test_put_raw(self):
        from domains.infrastructure.pugqeep import PGQ
        sys = PGQ(name="test")
        sys.put_raw("config", {"lr": 0.01})
        assert sys.get_any("config") == {"lr": 0.01}

    def test_task_ops(self):
        from domains.infrastructure.pugqeep import PGQ, Task
        sys = PGQ(name="test")
        task = Task(name="process", data="input")
        sys.submit_task(task)

        t = sys.next_task()
        assert t is not None

        sys.complete_task(t.id, result="output")
        assert sys.get_task(t.id).status.value == "completed"

    def test_stats(self):
        from domains.infrastructure.pugqeep import PGQ
        sys = PGQ(name="test")
        s = sys.stats()
        assert "tree" in s
        assert "cache" in s
        assert "queue" in s

    def test_put_get_many(self):
        from domains.infrastructure.pugqeep import PGQ
        import numpy as np
        sys = PGQ(name="test", n_clusters=8)
        data = {
            "w1": np.random.randn(100).astype(np.float32),
            "w2": np.random.randn(200).astype(np.float32),
        }
        result = sys.put_many(data, compress=True)
        assert result["count"] == 2

        got = sys.get_many(["w1", "w2", "missing"])
        assert got["w1"] is not None
        assert got["w2"] is not None
        assert got["missing"] is None

    def test_ttl(self):
        from domains.infrastructure.pugqeep import PGQ
        import time
        sys = PGQ(name="test")
        sys.put_raw("short_lived", "data", ttl=0.1)
        assert sys.get_any("short_lived") == "data"
        time.sleep(0.2)
        assert sys.get_any("short_lived") is None

    def test_cleanup_cache(self):
        from domains.infrastructure.pugqeep import PGQ
        import time
        sys = PGQ(name="test")
        sys.put_raw("expire1", "a", ttl=0.05)
        sys.put_raw("expire2", "b", ttl=0.05)
        sys.put_raw("permanent", "c")
        time.sleep(0.1)
        removed = sys.cleanup_cache()
        assert removed == 2
        assert sys.get_any("permanent") == "c"

    def test_save_load_with_tasks(self, tmp_path):
        from domains.infrastructure.pugqeep import PGQ, Task
        import numpy as np
        path = tmp_path / "state.json"
        sys = PGQ(name="test", storage_dir=tmp_path, n_clusters=8)
        sys.put("w", np.random.randn(50).astype(np.float32))
        task = Task(name="job", data="x")
        sys.submit_task(task)
        sys.save(path)

        sys2 = PGQ.load(path)
        assert sys2.name == "test"
        assert sys2.get("w") is not None
        assert len(sys2.list_tasks()) == 1

    def test_eviction_lru(self):
        """LRU eviction when memory tier overflows."""
        from domains.infrastructure.pugqeep.cache import TieredCache, Tier, EvictionPolicy
        import numpy as np
        # 1KB max = very small, forces eviction
        cache = TieredCache(memory_max_mb=0, hot_max_mb=0, eviction_policy=EvictionPolicy.LRU)
        # Override to 1KB
        cache._memory._max_size = 1024
        cache._hot._inner._max_size = 1024

        for i in range(20):
            data = np.ones(64, dtype=np.float32)  # 256 bytes each
            cache.put(f"k{i}", data, tier=Tier.MEMORY, size_bytes=256)

        s = cache.stats()
        assert s["evictions"] > 0, "Should have evicted some entries"
        assert s["memory_size"] <= 1024

    def test_eviction_lfu(self):
        """LFU eviction prefers least-frequent entries."""
        from domains.infrastructure.pugqeep.cache import TieredCache, Tier, EvictionPolicy
        import numpy as np
        cache = TieredCache(memory_max_mb=0, hot_max_mb=0, eviction_policy=EvictionPolicy.LFU)
        cache._memory._max_size = 1024
        cache._hot._inner._max_size = 1024

        # Fill cache
        for i in range(4):
            data = np.ones(64, dtype=np.float32)
            cache.put(f"k{i}", data, tier=Tier.MEMORY, size_bytes=256)

        # Access k0 multiple times to make it "hot"
        for _ in range(5):
            cache.get("k0")
        cache.get("k1")

        # Add more items to trigger eviction — k2/k3 (least freq) should go first
        for i in range(4, 8):
            data = np.ones(64, dtype=np.float32)
            cache.put(f"k{i}", data, tier=Tier.MEMORY, size_bytes=256)

        assert cache.get("k0") is not None, "Most-accessed should survive"
        s = cache.stats()
        assert s["evictions"] > 0

    def test_stats_includes_policy(self):
        from domains.infrastructure.pugqeep.cache import TieredCache, EvictionPolicy
        cache = TieredCache(eviction_policy=EvictionPolicy.LFU)
        s = cache.stats()
        assert s["eviction_policy"] == "lfu"
        assert "memory_max" in s
        assert "hot_max" in s


# ══════════════════════════════════════════════════════════════════════════════
# Cluster serialization roundtrip
# ══════════════════════════════════════════════════════════════════════════════

class TestClusterSerialization:
    def test_cluster_to_bytes_roundtrip(self):
        from domains.infrastructure.pugqeep.point import Point
        centroids = np.array([0.1, 0.5, 0.9, -0.3], dtype=np.float32)
        assignments = np.array([0, 1, 2, 3, 1, 0, 2, 3], dtype=np.uint8)
        p = Point(identity="test.cluster", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments},
                  accuracy=0.95, dtype="float32", shape=(8,))
        data = p.to_bytes()
        p2 = Point.from_bytes(data, identity="test.cluster")
        assert p2.function_type == "cluster"
        np.testing.assert_array_equal(p2.params["centroids"], centroids)
        np.testing.assert_array_equal(p2.params["assignments"], assignments)

    def test_cluster_with_residual_roundtrip(self):
        from domains.infrastructure.pugqeep.point import Point
        centroids = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assignments = np.array([0, 1, 2, 0], dtype=np.uint8)
        residual = np.array([0.01, -0.02, 0.03, -0.04], dtype=np.float32)
        p = Point(identity="test.res", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments},
                  residual=residual, accuracy=0.98)
        data = p.to_bytes()
        p2 = Point.from_bytes(data, identity="test.res")
        np.testing.assert_array_equal(p2.params["centroids"], centroids)
        np.testing.assert_array_equal(p2.params["assignments"], assignments)
        assert p2.residual is not None
        np.testing.assert_array_almost_equal(p2.residual, residual)

    def test_raw_to_bytes_roundtrip(self):
        from domains.infrastructure.pugqeep.point import Point
        raw = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        import base64
        p = Point(identity="test.raw", function_type="raw",
                  params={"data_b64": base64.b64encode(raw.tobytes()).decode(),
                          "shape": [3], "dtype": "float32"},
                  accuracy=1.0)
        data = p.to_bytes()
        p2 = Point.from_bytes(data, identity="test.raw")
        assert p2.function_type == "raw"
        decoded = np.frombuffer(base64.b64decode(p2.params["data_b64"]), dtype="float32")
        np.testing.assert_array_equal(decoded, raw)

    def test_generate_after_bytes_roundtrip(self):
        """Cluster generate should work after bytes roundtrip."""
        from domains.infrastructure.pugqeep.point import Point
        centroids = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        assignments = np.array([0, 1, 2, 1, 0], dtype=np.uint8)
        p = Point(identity="t", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments})
        data = p.to_bytes()
        p2 = Point.from_bytes(data, identity="t")
        result = p2.generate(5)
        expected = centroids[assignments]
        np.testing.assert_array_almost_equal(result, expected)


# ══════════════════════════════════════════════════════════════════════════════
# Config wiring
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigWiring:
    def test_compressor_config_override(self):
        from domains.infrastructure.pugqeep.compressor import PointCompressor
        from domains.infrastructure.pugqeep.config import CompressorConfig
        cfg = CompressorConfig(n_clusters=32, lloyd_iterations=10,
                               gap_fill_iterations=8, gap_fill_max_elements=50_000,
                               method="function")
        c = PointCompressor(config=cfg)
        assert c.n_clusters == 32
        assert c.lloyd_iterations == 10
        assert c.gap_fill_iterations == 8
        assert c.gap_fill_max_elements == 50_000
        assert c.method == "function"

    def test_compressor_residual_threshold(self):
        from domains.infrastructure.pugqeep.compressor import PointCompressor
        c = PointCompressor(residual_threshold=0.80)
        # With low threshold, even decent fits won't store residual
        weights = np.sin(np.linspace(0, 4 * np.pi, 200)).astype(np.float32)
        p = c.compress_function(weights, "test")
        # If accuracy > 0.80, no residual
        if p.accuracy >= 0.80:
            assert p.residual is None

    def test_compressor_defaults_without_config(self):
        from domains.infrastructure.pugqeep.compressor import PointCompressor
        c = PointCompressor()
        assert c.n_clusters == 16
        assert c.lloyd_iterations == 5
        assert c.residual_threshold == 0.99

    def test_tree_config_skip_embeddings(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        from domains.infrastructure.pugqeep.config import TreeConfig
        cfg = TreeConfig(skip_embeddings=False, skip_biases=False, n_clusters=8)
        tree = ModelTree("test", config=cfg)
        weights = {
            "embed_tokens.weight": np.random.randn(100, 64).astype(np.float32),
            "layer.bias": np.random.randn(64).astype(np.float32),
            "layer.weight": np.random.randn(64, 64).astype(np.float32),
        }
        tree.load_weights(weights)
        # With skip=False, all should be compressed as cluster (not raw)
        for name in weights:
            point = tree.library.get(f"test.{name}")
            assert point is not None
            # embed/bias stored as cluster, not raw
            if "embed" in name or name.endswith("bias"):
                assert point.function_type == "cluster", f"{name} should be cluster, got {point.function_type}"

    def test_tree_config_skip_embeddings_on(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        from domains.infrastructure.pugqeep.config import TreeConfig
        cfg = TreeConfig(skip_embeddings=True, skip_biases=True)
        tree = ModelTree("test", config=cfg)
        weights = {
            "embed_tokens.weight": np.random.randn(100, 64).astype(np.float32),
            "layer.bias": np.random.randn(64).astype(np.float32),
            "layer.weight": np.random.randn(64, 64).astype(np.float32),
        }
        tree.load_weights(weights)
        # embed/bias → raw, weight → cluster
        embed_point = tree.library.get("test.embed_tokens.weight")
        bias_point = tree.library.get("test.layer.bias")
        weight_point = tree.library.get("test.layer.weight")
        assert embed_point.function_type == "raw"
        assert bias_point.function_type == "raw"
        assert weight_point.function_type == "cluster"

    def test_library_auto_save(self, tmp_path):
        from domains.infrastructure.pugqeep.library import PointLibrary
        from domains.infrastructure.pugqeep.point import Point
        from domains.infrastructure.pugqeep.config import LibraryConfig
        cfg = LibraryConfig(name="autosave_test", auto_save=True, storage_dir=tmp_path)
        lib = PointLibrary(config=cfg)
        p = Point(identity="x", function_type="linear", params={"a": 1.0, "b": 0.0})
        lib.add(p)
        # File should exist after add
        assert (tmp_path / "autosave_test.points.json").exists()

    def test_compressor_uses_config_lloyd_iterations(self):
        from domains.infrastructure.pugqeep.compressor import PointCompressor
        from domains.infrastructure.pugqeep.config import CompressorConfig
        cfg = CompressorConfig(n_clusters=4, lloyd_iterations=1, gap_fill_iterations=0)
        c = PointCompressor(config=cfg)
        weights = np.random.randn(100).astype(np.float32)
        p = c.compress_cluster(weights, "test", n_clusters=4)
        assert p.function_type == "cluster"
        assert len(p.params["centroids"]) == 4
