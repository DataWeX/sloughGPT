"""Tests for pugqeep generic pluggable architecture."""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from domains.infrastructure.pugqeep.generic import (
    PGQGeneric,
    CompressionStrategy,
    StorageBackend,
    FunctionType,
    registry,
    ClusterStrategy,
    FunctionStrategy,
    RawStrategy,
    AutoStrategy,
    MemoryStorage,
    JSONStorage,
    DirectoryStorage,
)
from domains.infrastructure.pugqeep.point import Point


# ══════════════════════════════════════════════════════════════════════════════
# Built-in strategies
# ══════════════════════════════════════════════════════════════════════════════

class TestClusterStrategy:
    def test_compress_decompress(self):
        s = ClusterStrategy(n_clusters=8)
        data = np.random.randn(1000).astype(np.float32)
        point = s.compress(data, "test.cluster")
        assert point.function_type == "cluster"
        assert point.accuracy > 0.8
        result = s.decompress(point, 1000)
        assert result.shape == (1000,)
        mse = np.mean((data - result) ** 2)
        var = np.var(data)
        assert mse / (var + 1e-8) < 0.3

    def test_custom_clusters(self):
        s = ClusterStrategy(n_clusters=32)
        data = np.random.randn(500).astype(np.float32)
        point = s.compress(data, "test", n_clusters=32)
        assert len(point.params["centroids"]) >= 32

    def test_small_array(self):
        s = ClusterStrategy(n_clusters=16)
        data = np.random.randn(5).astype(np.float32)
        point = s.compress(data, "test")
        assert point.function_type == "cluster"


class TestFunctionStrategy:
    def test_compress_linear(self):
        s = FunctionStrategy()
        data = np.arange(100, dtype=np.float32) * 2.0 + 1.0
        point = s.compress(data, "test.linear")
        assert point.function_type == "linear"
        assert point.accuracy > 0.99

    def test_compress_periodic(self):
        s = FunctionStrategy()
        i = np.arange(200, dtype=np.float32)
        data = 3.0 * np.cos(i) + 2.0 * np.sin(i) + 5.0
        point = s.compress(data, "test.periodic")
        assert point.function_type == "periodic"
        assert point.accuracy > 0.99

    def test_compress_polynomial(self):
        s = FunctionStrategy()
        i = np.arange(100, dtype=np.float32)
        data = 0.01 * i**2 + 0.5 * i + 10.0
        point = s.compress(data, "test.poly")
        assert point.function_type == "polynomial"
        assert point.accuracy > 0.99

    def test_residual_below_threshold(self):
        s = FunctionStrategy(residual_threshold=0.999)
        data = np.random.randn(100).astype(np.float32)
        point = s.compress(data, "test")
        # Random data won't fit any function well → residual stored
        assert point.residual is not None

    def test_no_residual_above_threshold(self):
        s = FunctionStrategy(residual_threshold=0.5)
        data = np.arange(100, dtype=np.float32) * 2.0 + 1.0
        point = s.compress(data, "test")
        assert point.residual is None


class TestRawStrategy:
    def test_compress_decompress(self):
        s = RawStrategy()
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        point = s.compress(data, "test.raw")
        assert point.function_type == "raw"
        assert point.accuracy == 1.0
        result = s.decompress(point, 4)
        np.testing.assert_array_almost_equal(result, data)

    def test_preserves_shape(self):
        s = RawStrategy()
        data = np.random.randn(3, 4).astype(np.float32)
        point = s.compress(data, "test")
        result = s.decompress(point, 12)
        np.testing.assert_array_almost_equal(result, data.flatten())


class TestAutoStrategy:
    def test_selects_best(self):
        s = AutoStrategy()
        # Linear data → should pick linear or polynomial
        data = np.arange(200, dtype=np.float32) * 3.0 + 7.0
        point = s.compress(data, "test", n_clusters=16)
        assert point.accuracy > 0.95

    def test_random_data(self):
        s = AutoStrategy()
        data = np.random.randn(200).astype(np.float32)
        point = s.compress(data, "test", n_clusters=16)
        assert point.accuracy > 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Storage backends
# ══════════════════════════════════════════════════════════════════════════════

class TestMemoryStorage:
    def test_save_load(self):
        s = MemoryStorage()
        p = Point(identity="a", function_type="linear", params={"a": 1.0, "b": 0.0}, accuracy=0.9)
        s.save(p)
        loaded = s.load("a")
        assert loaded is not None
        assert loaded.identity == "a"

    def test_remove(self):
        s = MemoryStorage()
        p = Point(identity="a", function_type="raw", params={"data_b64": "", "shape": [], "dtype": "float32"}, accuracy=1.0)
        s.save(p)
        assert s.remove("a") is True
        assert s.load("a") is None
        assert s.remove("a") is False

    def test_list_all(self):
        s = MemoryStorage()
        for i in range(5):
            s.save(Point(identity=f"p{i}", function_type="raw", params={"data_b64": "", "shape": [], "dtype": "float32"}, accuracy=1.0))
        assert s.count() == 5
        assert len(s.list_all()) == 5

    def test_clear(self):
        s = MemoryStorage()
        s.save(Point(identity="a", function_type="raw", params={"data_b64": "", "shape": [], "dtype": "float32"}, accuracy=1.0))
        s.clear()
        assert s.count() == 0


class TestJSONStorage:
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = JSONStorage(Path(tmp) / "test.json")
            data = np.random.randn(100).astype(np.float32)
            point = ClusterStrategy().compress(data, "test.json")
            s.save(point)
            loaded = s.load("test.json")
            assert loaded is not None
            assert loaded.accuracy > 0.8

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            s1 = JSONStorage(path)
            s1.save(Point(identity="x", function_type="linear", params={"a": 1.0, "b": 0.0}, accuracy=0.9))
            # New instance loads from same file
            s2 = JSONStorage(path)
            assert s2.load("x") is not None


class TestDirectoryStorage:
    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = DirectoryStorage(Path(tmp))
            data = np.random.randn(50).astype(np.float32)
            point = FunctionStrategy().compress(data, "test.dir")
            s.save(point)
            loaded = s.load("test.dir")
            assert loaded is not None

    def test_remove(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = DirectoryStorage(Path(tmp))
            s.save(Point(identity="a", function_type="raw", params={"data_b64": "", "shape": [], "dtype": "float32"}, accuracy=1.0))
            assert s.remove("a") is True
            assert s.load("a") is None


# ══════════════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════════════

class TestRegistry:
    def test_builtin_compressors(self):
        assert "cluster" in registry.compressors
        assert "function" in registry.compressors
        assert "raw" in registry.compressors
        assert "auto" in registry.compressors

    def test_builtin_storages(self):
        assert "memory" in registry.storages

    def test_register_custom_compressor(self):
        class Q8(CompressionStrategy):
            name = "q8"
            def compress(self, data, identity="unknown", **kwargs):
                return Point(identity=identity, function_type="raw",
                             params={"data_b64": "", "shape": list(data.shape), "dtype": str(data.dtype)}, accuracy=1.0)
            def decompress(self, point, n):
                return np.zeros(n)

        registry.compressors.register(Q8())
        assert "q8" in registry.compressors
        assert registry.compressors.get("q8") is not None
        # cleanup
        del registry.compressors._strategies["q8"]

    def test_register_custom_storage(self):
        class NullStorage(StorageBackend):
            name = "null"
            def save(self, point): pass
            def load(self, identity): return None
            def remove(self, identity): return False
            def list_all(self): return []
            def clear(self): pass
            def count(self): return 0

        registry.storages.register(NullStorage())
        assert "null" in registry.storages
        del registry.storages._backends["null"]


# ══════════════════════════════════════════════════════════════════════════════
# PGQGeneric facade
# ══════════════════════════════════════════════════════════════════════════════

class TestPGQGeneric:
    def test_put_get_cluster(self):
        sys = PGQGeneric(name="test", compressor="cluster", storage=MemoryStorage())
        data = np.random.randn(500).astype(np.float32)
        sys.put("w1", data)
        result = sys.get("w1")
        assert result is not None
        assert result.shape == (500,)
        mse = np.mean((data - result) ** 2)
        assert mse / (np.var(data) + 1e-8) < 0.3

    def test_put_get_function(self):
        sys = PGQGeneric(name="test", compressor="function", storage=MemoryStorage())
        data = np.arange(100, dtype=np.float32) * 2.0 + 1.0
        sys.put("w1", data)
        result = sys.get("w1")
        np.testing.assert_array_almost_equal(result, data, decimal=4)

    def test_put_get_raw(self):
        sys = PGQGeneric(name="test", compressor="raw", storage=MemoryStorage())
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sys.put("w1", data)
        result = sys.get("w1")
        np.testing.assert_array_almost_equal(result, data)

    def test_put_get_auto(self):
        sys = PGQGeneric(name="test", compressor="auto", storage=MemoryStorage())
        data = np.random.randn(300).astype(np.float32)
        sys.put("w1", data)
        result = sys.get("w1")
        assert result is not None
        assert result.shape == (300,)

    def test_has_remove(self):
        sys = PGQGeneric(name="test", storage=MemoryStorage())
        data = np.ones(10, dtype=np.float32)
        sys.put("w1", data)
        assert sys.has("w1") is True
        sys.remove("w1")
        assert sys.has("w1") is False

    def test_put_get_many(self):
        sys = PGQGeneric(name="test", compressor="cluster", storage=MemoryStorage())
        arrays = {f"w{i}": np.random.randn(100).astype(np.float32) for i in range(5)}
        sys.put_many(arrays)
        results = sys.get_many(list(arrays.keys()))
        assert len(results) == 5
        for name in arrays:
            assert results[name] is not None
            assert results[name].shape == (100,)

    def test_search(self):
        sys = PGQGeneric(name="test", storage=MemoryStorage())
        sys.put("model.layer1.weight", np.ones(10, dtype=np.float32))
        sys.put("model.layer2.weight", np.ones(10, dtype=np.float32))
        sys.put("model.bias", np.ones(5, dtype=np.float32))
        results = sys.search("layer")
        assert len(results) == 2

    def test_best(self):
        sys = PGQGeneric(name="test", storage=MemoryStorage())
        sys.put("good", np.arange(100, dtype=np.float32))
        sys.put("bad", np.random.randn(100).astype(np.float32))
        best = sys.best(1)
        assert len(best) == 1

    def test_stats(self):
        sys = PGQGeneric(name="test", compressor="cluster", storage=MemoryStorage())
        sys.put("w1", np.random.randn(100).astype(np.float32))
        stats = sys.stats()
        assert stats["name"] == "test"
        assert stats["compressor"] == "cluster"
        assert stats["num_points"] == 1
        assert stats["ratio"] > 1.0

    def test_custom_compressor(self):
        class HalveStrategy(CompressionStrategy):
            name = "halve"
            def compress(self, data, identity="unknown", **kwargs):
                flat = data.flatten().astype(np.float32)
                centroids = np.array([flat.mean()], dtype=np.float32)
                assignments = np.zeros(len(flat), dtype=np.uint8)
                return Point(identity=identity, function_type="cluster",
                             params={"centroids": centroids, "assignments": assignments},
                             accuracy=0.5)
            def decompress(self, point, n):
                return np.full(n, point.params["centroids"][0])

        registry.compressors.register(HalveStrategy())
        sys = PGQGeneric(name="test", compressor="halve", storage=MemoryStorage())
        data = np.ones(100, dtype=np.float32) * 5.0
        sys.put("w1", data)
        result = sys.get("w1")
        np.testing.assert_array_almost_equal(result, np.full(100, 5.0))
        del registry.compressors._strategies["halve"]

    def test_compose_strategy_instance(self):
        s = ClusterStrategy(n_clusters=32)
        sys = PGQGeneric(name="test", compressor=s, storage=MemoryStorage())
        data = np.random.randn(200).astype(np.float32)
        sys.put("w1", data)
        result = sys.get("w1")
        assert result is not None

    def test_json_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            sys = PGQGeneric(name="test", compressor="cluster",
                             storage=JSONStorage(Path(tmp) / "lib.json"))
            data = np.random.randn(100).astype(np.float32)
            sys.put("w1", data)
            # Reload from disk
            sys2 = PGQGeneric(name="test", compressor="cluster",
                              storage=JSONStorage(Path(tmp) / "lib.json"))
            result = sys2.get("w1")
            assert result is not None

    def test_clear(self):
        sys = PGQGeneric(name="test", storage=MemoryStorage())
        sys.put("w1", np.ones(10, dtype=np.float32))
        sys.put("w2", np.ones(20, dtype=np.float32))
        assert sys.count() == 2
        sys.clear()
        assert sys.count() == 0

    def test_unknown_compressor_raises(self):
        with pytest.raises(ValueError, match="Unknown compressor"):
            PGQGeneric(name="test", compressor="nonexistent", storage=MemoryStorage())

    def test_unknown_storage_raises(self):
        with pytest.raises(ValueError, match="Unknown storage"):
            PGQGeneric(name="test", storage="nonexistent")
