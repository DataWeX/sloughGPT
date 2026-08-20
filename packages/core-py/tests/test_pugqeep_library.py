"""Tests for domains.infrastructure.pugqeep.library — PointLibrary."""

import numpy as np
import tempfile
from pathlib import Path

import pytest
from domains.infrastructure.pugqeep.library import PointLibrary
from domains.infrastructure.pugqeep.point import Point


class TestPointLibraryCRUD:
    def test_add_and_get(self):
        lib = PointLibrary(name="test")
        p = Point(identity="p1", function_type="periodic", params={"a": 1.0})
        lib.add(p)
        assert lib.get("p1") is p

    def test_get_missing(self):
        lib = PointLibrary()
        assert lib.get("nonexistent") is None

    def test_remove(self):
        lib = PointLibrary()
        p = Point(identity="p1", function_type="periodic", params={})
        lib.add(p)
        assert lib.remove("p1") is True
        assert lib.get("p1") is None

    def test_remove_missing(self):
        lib = PointLibrary()
        assert lib.remove("nope") is False

    def test_has(self):
        lib = PointLibrary()
        lib.add(Point(identity="a", function_type="periodic", params={}))
        assert lib.has("a") is True
        assert lib.has("b") is False

    def test_list_all(self):
        lib = PointLibrary()
        lib.add(Point(identity="a", function_type="periodic", params={}))
        lib.add(Point(identity="b", function_type="cluster", params={}))
        assert len(lib.list_all()) == 2

    def test_clear(self):
        lib = PointLibrary()
        lib.add(Point(identity="a", function_type="periodic", params={}))
        lib.add(Point(identity="b", function_type="cluster", params={}))
        lib.clear()
        assert len(lib.list_all()) == 0

    def test_list_by_type(self):
        lib = PointLibrary()
        lib.add(Point(identity="p1", function_type="periodic", params={}))
        lib.add(Point(identity="p2", function_type="periodic", params={}))
        lib.add(Point(identity="c1", function_type="cluster", params={}))
        periodic = lib.list_by_type("periodic")
        assert len(periodic) == 2
        cluster = lib.list_by_type("cluster")
        assert len(cluster) == 1


class TestPointLibraryCompress:
    def test_compress_cluster_and_store(self):
        lib = PointLibrary()
        weights = np.random.randn(128)
        p = lib.compress_and_store(weights, identity="layer1")
        assert lib.has("layer1")
        assert p.function_type == "cluster"

    def test_compress_function_and_store(self):
        lib = PointLibrary()
        weights = np.random.randn(128)
        p = lib.compress_and_store(weights, identity="func1", method="function")
        assert lib.has("func1")
        assert p.function_type in ("periodic", "linear", "polynomial")

    def test_decompress_cluster(self):
        lib = PointLibrary()
        weights = np.random.randn(128)
        lib.compress_and_store(weights, identity="layer1")
        recovered = lib.decompress_to("layer1", shape=(128,))
        assert recovered is not None
        assert recovered.shape == (128,)


class TestPointLibrarySearch:
    def test_search(self):
        lib = PointLibrary()
        lib.add(Point(identity="layer.weight", function_type="periodic", params={}))
        lib.add(Point(identity="layer.bias", function_type="periodic", params={}))
        lib.add(Point(identity="other", function_type="periodic", params={}))
        results = lib.search("layer")
        assert len(results) == 2

    def test_best_points(self):
        lib = PointLibrary()
        lib.add(Point(identity="a", function_type="periodic", params={}, accuracy=0.9))
        lib.add(Point(identity="b", function_type="periodic", params={}, accuracy=0.5))
        lib.add(Point(identity="c", function_type="periodic", params={}, accuracy=0.7))
        best = lib.best_points(n=2)
        assert len(best) == 2
        assert best[0].identity == "a"


class TestPointLibraryStats:
    def test_empty_stats(self):
        lib = PointLibrary(name="empty")
        s = lib.stats()
        assert s["total_points"] == 0
        assert s["avg_accuracy"] == 0.0

    def test_stats_with_points(self):
        lib = PointLibrary(name="test")
        lib.add(Point(identity="a", function_type="periodic", params={}, accuracy=0.8))
        lib.add(Point(identity="b", function_type="periodic", params={}, accuracy=0.6))
        s = lib.stats()
        assert s["total_points"] == 2
        assert s["avg_accuracy"] == pytest.approx(0.7)

    def test_cluster_compressed_bytes(self):
        lib = PointLibrary()
        weights = np.random.randn(100)
        lib.compress_and_store(weights, identity="c1")
        s = lib.stats()
        assert s["total_compressed_bytes"] > 0


class TestPointLibraryPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = PointLibrary(name="persist", storage_dir=Path(tmpdir))
            lib.add(Point(identity="p1", function_type="periodic", params={"a": 1.0}))
            lib.add(Point(identity="p2", function_type="periodic", params={"b": 2.0}))
            lib.save()

            loaded = PointLibrary.load(Path(tmpdir) / "persist.points.json")
            assert loaded.name == "persist"
            assert loaded.has("p1")
            assert loaded.has("p2")

    def test_auto_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = PointLibrary(name="auto", storage_dir=Path(tmpdir), config=None)
            lib._auto_save = True
            lib.add(Point(identity="x", function_type="periodic", params={}))
            assert Path(tmpdir, "auto.points.json").exists()


class TestPointLibraryDuplicateIdentity:
    def test_re_add_overwrites(self):
        lib = PointLibrary()
        p1 = Point(identity="dup", function_type="periodic", params={"a": 1.0})
        p2 = Point(identity="dup", function_type="periodic", params={"a": 2.0})
        lib.add(p1)
        lib.add(p2)
        assert lib.get("dup").params["a"] == 2.0
        assert len(lib.list_all()) == 1
