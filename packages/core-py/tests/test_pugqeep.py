"""Tests for pugqeep/compressor.py, library.py, dedup.py — compression, library CRUD, deduplication."""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from domains.infrastructure.pugqeep.compressor import PointCompressor
from domains.infrastructure.pugqeep.point import Point
from domains.infrastructure.pugqeep.library import PointLibrary
from domains.infrastructure.pugqeep.dedup import PointDeduplicator, PointLibrarySync


class TestPointCompressorCluster:
    def test_compress_cluster_basic(self):
        comp = PointCompressor(n_clusters=4, lloyd_iterations=3)
        weights = np.random.randn(100).astype(np.float32)
        p = comp.compress_cluster(weights, identity="test")
        assert p.function_type == "cluster"
        assert p.accuracy > 0.5
        assert "centroids" in p.params
        assert "assignments" in p.params
        assert len(p.params["assignments"]) == 100

    def test_compress_cluster_generates_output(self):
        comp = PointCompressor(n_clusters=4)
        weights = np.random.randn(50).astype(np.float32)
        p = comp.compress_cluster(weights, identity="t")
        reconstructed = p.generate(50)
        assert reconstructed.shape == (50,)

    def test_compress_cluster_small_weights_gets_gap_fill(self):
        comp = PointCompressor(n_clusters=4, lloyd_iterations=2)
        weights = np.random.randn(50).astype(np.float32)
        p = comp.compress_cluster(weights, identity="small")
        assert p.params["centroids"].shape[0] >= 4


class TestPointCompressorFunction:
    def test_compress_function_linear(self):
        comp = PointCompressor()
        weights = np.arange(100, dtype=np.float32) * 2.0 + 1.0
        p = comp.compress_function(weights, identity="lin")
        assert p.function_type == "linear"
        assert p.accuracy > 0.9

    def test_compress_function_periodic(self):
        comp = PointCompressor()
        i = np.arange(100, dtype=np.float32)
        weights = 3.0 * np.cos(i) + 0.5
        p = comp.compress_function(weights, identity="per")
        assert p.function_type == "periodic"

    def test_compress_function_with_residual(self):
        comp = PointCompressor(residual_threshold=0.9999)
        weights = np.arange(50, dtype=np.float32) + np.random.randn(50).astype(np.float32) * 2.0
        p = comp.compress_function(weights, identity="res")
        assert p.residual is not None


class TestPointCompressorCompress:
    def test_compress_cluster_method(self):
        from domains.infrastructure.pugqeep.config import CompressorConfig
        config = CompressorConfig(method="cluster")
        comp = PointCompressor(config=config)
        p = comp.compress(np.random.randn(50).astype(np.float32), identity="c")
        assert p.function_type == "cluster"

    def test_compress_function_method(self):
        from domains.infrastructure.pugqeep.config import CompressorConfig
        config = CompressorConfig(method="function")
        comp = PointCompressor(config=config)
        weights = np.arange(50, dtype=np.float32) * 3.0
        p = comp.compress(weights, identity="f")
        assert p.function_type in ("linear", "periodic", "polynomial")

    def test_unknown_method_raises(self):
        comp = PointCompressor()
        comp.method = "unknown"
        with pytest.raises(ValueError, match="Unknown method"):
            comp.compress(np.zeros(10), identity="x", method="unknown")


class TestPointCompressorDecompress:
    def test_decompress_cluster(self):
        comp = PointCompressor(n_clusters=4)
        weights = np.random.randn(80).astype(np.float32)
        p = comp.compress_cluster(weights, identity="d")
        result = comp.decompress(p, 80)
        assert result.shape == (80,)


class TestPointCompressorMeasure:
    def test_measure_compression(self):
        comp = PointCompressor(n_clusters=4)
        weights = np.random.randn(100).astype(np.float32)
        p = comp.compress_cluster(weights, identity="m")
        m = comp.measure_compression(weights, p)
        assert "raw_bytes" in m
        assert "compressed_bytes" in m
        assert "ratio" in m
        assert m["ratio"] > 0


class TestPointLibrary:
    def test_add_and_get(self):
        lib = PointLibrary(name="test")
        p = Point("p1", "linear", {"a": 1.0, "b": 0.0})
        lib.add(p)
        assert lib.get("p1") is p
        assert lib.get("nonexistent") is None

    def test_has(self):
        lib = PointLibrary()
        lib.add(Point("x", "linear", {"a": 1.0, "b": 0.0}))
        assert lib.has("x")
        assert not lib.has("y")

    def test_remove(self):
        lib = PointLibrary()
        lib.add(Point("x", "linear", {"a": 1.0, "b": 0.0}))
        assert lib.remove("x")
        assert not lib.has("x")
        assert not lib.remove("x")  # already removed

    def test_list_all(self):
        lib = PointLibrary()
        lib.add(Point("a", "linear", {"a": 1.0, "b": 0.0}))
        lib.add(Point("b", "periodic", {"a": 1.0, "b": 0.0, "w": 0.0}))
        assert len(lib.list_all()) == 2

    def test_list_by_type(self):
        lib = PointLibrary()
        lib.add(Point("lin", "linear", {"a": 1.0, "b": 0.0}))
        lib.add(Point("per", "periodic", {"a": 1.0, "b": 0.0, "w": 0.0}))
        lib.add(Point("lin2", "linear", {"a": 2.0, "b": 0.0}))
        assert len(lib.list_by_type("linear")) == 2
        assert len(lib.list_by_type("periodic")) == 1

    def test_clear(self):
        lib = PointLibrary()
        lib.add(Point("a", "linear", {"a": 1.0, "b": 0.0}))
        lib.clear()
        assert len(lib.list_all()) == 0

    def test_compress_and_store(self):
        lib = PointLibrary()
        weights = np.random.randn(50).astype(np.float32)
        p = lib.compress_and_store(weights, identity="c1", method="cluster")
        assert lib.has("c1")
        assert p.function_type == "cluster"

    def test_search(self):
        lib = PointLibrary()
        lib.add(Point("slo.layer1", "linear", {"a": 1.0, "b": 0.0}))
        lib.add(Point("slo.layer2", "linear", {"a": 2.0, "b": 0.0}))
        lib.add(Point("hf.layer1", "linear", {"a": 3.0, "b": 0.0}))
        results = lib.search("slo")
        assert len(results) == 2

    def test_best_points(self):
        lib = PointLibrary()
        lib.add(Point("low", "linear", {"a": 1.0, "b": 0.0}, accuracy=0.5))
        lib.add(Point("high", "linear", {"a": 1.0, "b": 0.0}, accuracy=0.99))
        best = lib.best_points(1)
        assert best[0].identity == "high"

    def test_stats_empty(self):
        lib = PointLibrary(name="empty")
        s = lib.stats()
        assert s["total_points"] == 0
        assert s["avg_accuracy"] == 0.0

    def test_stats_with_points(self):
        lib = PointLibrary(name="test")
        c = np.array([1.0, 2.0], dtype=np.float32)
        a = np.array([0, 1], dtype=np.uint8)
        lib.add(Point("p1", "cluster", {"centroids": c, "assignments": a}, accuracy=0.9))
        s = lib.stats()
        assert s["total_points"] == 1
        assert s["avg_accuracy"] == pytest.approx(0.9)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = PointLibrary(name="test", storage_dir=Path(tmp))
            lib.add(Point("p1", "linear", {"a": 1.0, "b": 0.0}))
            lib.add(Point("p2", "periodic", {"a": 1.0, "b": 0.0, "w": 0.0}))
            path = lib.save()

            loaded = PointLibrary.load(path)
            assert loaded.name == "test"
            assert len(loaded.list_all()) == 2
            assert loaded.get("p1") is not None


class TestPointDeduplicator:
    def test_no_duplicates(self):
        lib = PointLibrary()
        lib.add(Point("a", "linear", {"a": 1.0, "b": 0.0}))
        lib.add(Point("b", "linear", {"a": 2.0, "b": 0.0}))
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        assert dedup.find_duplicates() == []

    def test_finds_duplicates(self):
        lib = PointLibrary()
        lib.add(Point("a", "linear", {"a": 1.0, "b": 0.0}))
        lib.add(Point("b", "linear", {"a": 1.0, "b": 0.0}))  # same params
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        groups = dedup.find_duplicates()
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_deduplicate_removes_duplicates(self):
        lib = PointLibrary()
        lib.add(Point("a", "linear", {"a": 1.0, "b": 0.0}))
        lib.add(Point("b", "linear", {"a": 1.0, "b": 0.0}))
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        result = dedup.deduplicate()
        assert result["merged"] == 1
        assert lib.has("a")
        assert not lib.has("b")

    def test_cross_library_duplicates(self):
        lib1 = PointLibrary()
        lib2 = PointLibrary()
        lib1.add(Point("shared", "linear", {"a": 5.0, "b": 0.0}))
        lib2.add(Point("shared2", "linear", {"a": 5.0, "b": 0.0}))
        dedup = PointDeduplicator()
        dedup.add_library(lib1)
        dedup.add_library(lib2)
        groups = dedup.find_duplicates()
        assert len(groups) == 1


class TestPointLibrarySync:
    def test_export_import_bytes(self):
        lib = PointLibrary(name="sync_test")
        lib.add(Point("p1", "linear", {"a": 1.0, "b": 2.0}))
        sync = PointLibrarySync()
        data = sync.export_bytes(lib)
        lib2 = sync.import_bytes(data)
        assert lib2.name == "sync_test"
        assert len(lib2.list_all()) == 1

    def test_sync_to_and_from_directory(self):
        lib = PointLibrary(name="dir_test")
        lib.add(Point("p1", "linear", {"a": 1.0, "b": 0.0}))
        sync = PointLibrarySync()
        with tempfile.TemporaryDirectory() as tmp:
            sync.sync_to_directory(lib, Path(tmp))
            loaded = sync.sync_from_directory(Path(tmp), name="dir_test")
            assert loaded is not None
            assert loaded.name == "dir_test"

    def test_sync_from_directory_not_found(self):
        sync = PointLibrarySync()
        with tempfile.TemporaryDirectory() as tmp:
            result = sync.sync_from_directory(Path(tmp), name="nonexistent")
            assert result is None

    def test_merge(self):
        lib1 = PointLibrary(name="l1")
        lib2 = PointLibrary(name="l2")
        lib1.add(Point("a", "linear", {"a": 1.0, "b": 0.0}))
        lib2.add(Point("b", "periodic", {"a": 1.0, "b": 0.0, "w": 0.0}))
        sync = PointLibrarySync()
        merged = sync.merge([lib1, lib2])
        assert merged.has("a")
        assert merged.has("b")
