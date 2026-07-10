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
