"""Comprehensive tests for domains.infrastructure.pugqeep.library — PointLibrary.

Covers: CRUD, batch ops, search, validation, compression, persistence,
PointView, stats, thread safety, and edge cases.
"""

import numpy as np
import tempfile
import threading
from pathlib import Path

import pytest
from domains.infrastructure.pugqeep.library import PointLibrary
from domains.infrastructure.pugqeep.point import Point
from domains.infrastructure.pugqeep.point_interface import FunctionType, PointView
from domains.infrastructure.pugqeep.config import LibraryConfig


# ── helpers ──────────────────────────────────────────────────────────────────

def _periodic_params():
    return {"a": 1.0, "b": 0.5, "w": 0.1}

def _linear_params():
    return {"a": 0.5, "b": 0.1}

def _polynomial_params():
    return {"a": 0.1, "b": 0.5, "c": 0.01}

def _cluster_point(identity="c1"):
    centroids = np.random.randn(16).astype(np.float32)
    assignments = np.random.randint(0, 16, size=128).astype(np.uint8)
    return Point(identity=identity, function_type="cluster",
                 params={"centroids": centroids, "assignments": assignments},
                 accuracy=0.85)

def _periodic_point(identity="p1", accuracy=0.9):
    return Point(identity=identity, function_type="periodic",
                 params=_periodic_params(), accuracy=accuracy)

def _linear_point(identity="l1"):
    return Point(identity=identity, function_type="linear",
                 params=_linear_params())

def _polynomial_point(identity="po1"):
    return Point(identity=identity, function_type="polynomial",
                 params=_polynomial_params())


# ═══════════════════════════════════════════════════════════════════════════════
# CRUD
# ═══════════════════════════════════════════════════════════════════════════════

class TestPointLibraryCRUD:
    def test_add_and_get(self):
        lib = PointLibrary(name="test")
        p = _periodic_point("p1")
        lib.add(p)
        assert lib.get("p1") is p

    def test_get_missing(self):
        lib = PointLibrary()
        assert lib.get("nonexistent") is None

    def test_remove(self):
        lib = PointLibrary()
        p = _periodic_point("p1")
        lib.add(p)
        assert lib.remove("p1") is True
        assert lib.get("p1") is None

    def test_remove_missing(self):
        lib = PointLibrary()
        assert lib.remove("nope") is False

    def test_has(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        assert lib.has("a") is True
        assert lib.has("b") is False

    def test_list_all(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        lib.add(_linear_point("b"))
        assert len(lib.list_all()) == 2

    def test_clear(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        lib.add(_linear_point("b"))
        lib.clear()
        assert len(lib.list_all()) == 0

    def test_list_by_type(self):
        lib = PointLibrary()
        lib.add(_periodic_point("p1"))
        lib.add(_periodic_point("p2"))
        lib.add(_cluster_point("c1"))
        periodic = lib.list_by_type("periodic")
        assert len(periodic) == 2
        cluster = lib.list_by_type("cluster")
        assert len(cluster) == 1

    def test_list_identities(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        lib.add(_linear_point("b"))
        ids = lib.list_identities()
        assert set(ids) == {"a", "b"}

    def test_list_types(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        lib.add(_periodic_point("b"))
        lib.add(_cluster_point("c"))
        types = lib.list_types()
        assert types["periodic"] == 2
        assert types["cluster"] == 1

    def test_contains(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        assert "a" in lib
        assert "b" not in lib

    def test_len(self):
        lib = PointLibrary()
        assert len(lib) == 0
        lib.add(_periodic_point("a"))
        assert len(lib) == 1
        lib.add(_linear_point("b"))
        assert len(lib) == 2

    def test_add_returns_true_for_new(self):
        lib = PointLibrary()
        assert lib.add(_periodic_point("a")) is True

    def test_add_returns_false_for_existing(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        assert lib.add(_periodic_point("a")) is False

    def test_re_add_overwrites(self):
        lib = PointLibrary()
        p1 = _periodic_point("dup")
        p1.params = {"a": 1.0, "b": 0.0, "w": 0.0}
        p2 = _periodic_point("dup")
        p2.params = {"a": 2.0, "b": 0.0, "w": 0.0}
        lib.add(p1)
        lib.add(p2)
        assert lib.get("dup").params["a"] == 2.0
        assert len(lib.list_all()) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Operations
# ═══════════════════════════════════════════════════════════════════════════════

class TestPointLibraryBatchOps:
    def test_add_many(self):
        lib = PointLibrary()
        points = [_periodic_point(f"p{i}") for i in range(5)]
        count = lib.add_many(points)
        assert count == 5
        assert len(lib) == 5

    def test_add_many_empty(self):
        lib = PointLibrary()
        count = lib.add_many([])
        assert count == 0

    def test_get_many(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        lib.add(_linear_point("b"))
        result = lib.get_many(["a", "b", "missing"])
        assert result["a"] is not None
        assert result["b"] is not None
        assert result["missing"] is None

    def test_remove_many(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        lib.add(_linear_point("b"))
        lib.add(_cluster_point("c"))
        count = lib.remove_many(["a", "b"])
        assert count == 2
        assert lib.get("a") is None
        assert lib.get("c") is not None

    def test_remove_many_missing(self):
        lib = PointLibrary()
        count = lib.remove_many(["missing1", "missing2"])
        assert count == 0

    def test_exists_many(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        result = lib.exists_many(["a", "b", "c"])
        assert result["a"] is True
        assert result["b"] is False
        assert result["c"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# Search
# ═══════════════════════════════════════════════════════════════════════════════

class TestPointLibrarySearch:
    def test_search(self):
        lib = PointLibrary()
        lib.add(_periodic_point("layer.weight"))
        lib.add(_periodic_point("layer.bias"))
        lib.add(_periodic_point("other"))
        results = lib.search("layer")
        assert len(results) == 2

    def test_search_case_insensitive(self):
        lib = PointLibrary()
        lib.add(_periodic_point("MyPoint"))
        results = lib.search("mypoint")
        assert len(results) == 1

    def test_search_no_match(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        results = lib.search("zzz")
        assert len(results) == 0

    def test_best_points(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a", accuracy=0.9))
        lib.add(_periodic_point("b", accuracy=0.5))
        lib.add(_periodic_point("c", accuracy=0.7))
        best = lib.best_points(n=2)
        assert len(best) == 2
        assert best[0].identity == "a"

    def test_worst_points(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a", accuracy=0.9))
        lib.add(_periodic_point("b", accuracy=0.3))
        lib.add(_periodic_point("c", accuracy=0.7))
        worst = lib.worst_points(n=2)
        assert len(worst) == 2
        assert worst[0].identity == "b"

    def test_best_points_more_than_available(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a", accuracy=0.5))
        best = lib.best_points(n=10)
        assert len(best) == 1

    def test_search_by_type(self):
        lib = PointLibrary()
        lib.add(_periodic_point("p1"))
        lib.add(_periodic_point("p2"))
        lib.add(_cluster_point("c1"))
        results = lib.search_by_type("periodic")
        assert len(results) == 2

    def test_search_by_type_with_query(self):
        lib = PointLibrary()
        lib.add(_periodic_point("layer.weight"))
        lib.add(_periodic_point("layer.bias"))
        lib.add(_periodic_point("other"))
        results = lib.search_by_type("periodic", query="layer")
        assert len(results) == 2

    def test_search_by_type_no_match(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        results = lib.search_by_type("cluster")
        assert len(results) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Compression & Decompression
# ═══════════════════════════════════════════════════════════════════════════════

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

    def test_decompress_nonexistent(self):
        lib = PointLibrary()
        assert lib.decompress_to("missing") is None


# ═══════════════════════════════════════════════════════════════════════════════
# PointView
# ═══════════════════════════════════════════════════════════════════════════════

class TestPointLibraryView:
    def test_view_returns_pointview(self):
        lib = PointLibrary()
        lib.add(_periodic_point("p1"))
        view = lib.view("p1", shape=(10,), dtype="float32")
        assert view is not None
        assert isinstance(view, PointView)
        assert view.identity == "p1"

    def test_view_missing_returns_none(self):
        lib = PointLibrary()
        assert lib.view("missing") is None

    def test_view_cached(self):
        lib = PointLibrary()
        lib.add(_periodic_point("p1"))
        v1 = lib.view("p1")
        v2 = lib.view("p1")
        assert v1 is v2

    def test_views_batch(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        lib.add(_linear_point("b"))
        views = lib.views(["a", "b", "missing"])
        assert views["a"] is not None
        assert views["b"] is not None
        assert views["missing"] is None

    def test_clear_views(self):
        lib = PointLibrary()
        lib.add(_periodic_point("p1"))
        lib.view("p1")
        count = lib.clear_views()
        assert count == 1
        assert len(lib._views) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════════

class TestPointLibraryStats:
    def test_empty_stats(self):
        lib = PointLibrary(name="empty")
        s = lib.stats()
        assert s["total_points"] == 0
        assert s["avg_accuracy"] == 0.0

    def test_stats_with_points(self):
        lib = PointLibrary(name="test")
        lib.add(_periodic_point("a", accuracy=0.8))
        lib.add(_periodic_point("b", accuracy=0.6))
        s = lib.stats()
        assert s["total_points"] == 2
        assert s["avg_accuracy"] == pytest.approx(0.7)

    def test_cluster_compressed_bytes(self):
        lib = PointLibrary()
        weights = np.random.randn(100)
        lib.compress_and_store(weights, identity="c1")
        s = lib.stats()
        assert s["total_compressed_bytes"] > 0

    def test_stats_has_ops(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        lib.get("a")
        lib.get("missing")
        lib.remove("a")
        s = lib.stats()
        assert s["ops"]["adds"] == 1
        assert s["ops"]["hits"] == 1
        assert s["ops"]["misses"] == 1
        assert s["ops"]["removes"] == 1

    def test_hit_rate(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        lib.get("a")
        lib.get("missing")
        assert lib.hit_rate == pytest.approx(0.5)

    def test_hit_rate_empty(self):
        lib = PointLibrary()
        assert lib.hit_rate == 0.0

    def test_stats_views_cached(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        lib.view("a")
        s = lib.stats()
        assert s["views_cached"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════════════════

class TestPointLibraryPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = PointLibrary(name="persist", storage_dir=Path(tmpdir))
            lib.add(_periodic_point("p1"))
            lib.add(_periodic_point("p2"))
            lib.save()

            loaded = PointLibrary.load(Path(tmpdir) / "persist.points.json")
            assert loaded.name == "persist"
            assert loaded.has("p1")
            assert loaded.has("p2")

    def test_auto_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = PointLibrary(name="auto", storage_dir=Path(tmpdir), config=None)
            lib._auto_save = True
            lib.add(_periodic_point("x"))
            assert Path(tmpdir, "auto.points.json").exists()

    def test_save_no_storage_dir_no_path(self):
        lib = PointLibrary(name="nosave")
        with pytest.raises(ValueError):
            lib.save()

    def test_save_with_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = PointLibrary(name="explicit")
            lib.add(_periodic_point("a"))
            path = Path(tmpdir) / "custom.json"
            lib.save(path=path)
            assert path.exists()

    def test_save_cluster_point(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = PointLibrary(name="cluster_save", storage_dir=Path(tmpdir))
            lib.add(_cluster_point("c1"))
            lib.save()
            loaded = PointLibrary.load(Path(tmpdir) / "cluster_save.points.json")
            assert loaded.has("c1")

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with PointLibrary(name="ctx", storage_dir=Path(tmpdir)) as lib:
                lib._auto_save = True
                lib.add(_periodic_point("a"))
            assert Path(tmpdir, "ctx.points.json").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestPointLibraryValidation:
    def test_empty_identity_raises(self):
        lib = PointLibrary()
        p = Point(identity="", function_type="periodic", params=_periodic_params())
        with pytest.raises(ValueError, match="identity cannot be empty"):
            lib.add(p)

    def test_invalid_function_type_raises(self):
        lib = PointLibrary()
        p = Point(identity="bad", function_type="invalid_type", params={})
        with pytest.raises(ValueError, match="Invalid function_type"):
            lib.add(p)

    def test_invalid_accuracy_raises(self):
        lib = PointLibrary()
        p = Point(identity="bad", function_type="periodic", params=_periodic_params(),
                  accuracy=1.5)
        with pytest.raises(ValueError, match="Accuracy must be 0-1"):
            lib.add(p)

    def test_cluster_missing_params_raises(self):
        lib = PointLibrary()
        p = Point(identity="bad_cluster", function_type="cluster", params={})
        with pytest.raises(ValueError, match="centroids"):
            lib.add(p)

    def test_cluster_missing_centroids_raises(self):
        lib = PointLibrary()
        assignments = np.random.randint(0, 16, size=100).astype(np.uint8)
        p = Point(identity="bad", function_type="cluster",
                  params={"assignments": assignments})
        with pytest.raises(ValueError, match="centroids"):
            lib.add(p)

    def test_cluster_missing_assignments_raises(self):
        lib = PointLibrary()
        centroids = np.random.randn(16).astype(np.float32)
        p = Point(identity="bad", function_type="cluster",
                  params={"centroids": centroids})
        with pytest.raises(ValueError, match="assignments"):
            lib.add(p)

    def test_cluster_non_numpy_centroids_raises(self):
        lib = PointLibrary()
        p = Point(identity="bad", function_type="cluster",
                  params={"centroids": [1, 2, 3],
                          "assignments": np.array([0, 1, 2], dtype=np.uint8)})
        with pytest.raises(ValueError, match="centroids must be numpy"):
            lib.add(p)

    def test_cluster_non_numpy_assignments_raises(self):
        lib = PointLibrary()
        p = Point(identity="bad", function_type="cluster",
                  params={"centroids": np.array([1.0, 2.0], dtype=np.float32),
                          "assignments": [0, 1]})
        with pytest.raises(ValueError, match="assignments must be numpy"):
            lib.add(p)

    def test_disable_validation(self):
        lib = PointLibrary(validate=False)
        p = Point(identity="ok", function_type="periodic", params=_periodic_params())
        lib.add(p)
        assert lib.has("ok")

    def test_accuracy_at_boundary(self):
        lib = PointLibrary()
        p = Point(identity="zero", function_type="periodic", params=_periodic_params(),
                  accuracy=0.0)
        lib.add(p)
        p2 = Point(identity="one", function_type="periodic", params=_periodic_params(),
                   accuracy=1.0)
        lib.add(p2)

    def test_add_many_validation(self):
        lib = PointLibrary()
        bad = [Point(identity="", function_type="periodic", params={})]
        with pytest.raises(ValueError):
            lib.add_many(bad)


# ═══════════════════════════════════════════════════════════════════════════════
# Iteration
# ═══════════════════════════════════════════════════════════════════════════════

class TestPointLibraryIteration:
    def test_iter(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        lib.add(_linear_point("b"))
        items = list(lib)
        assert len(items) == 2

    def test_iter_by_type(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        lib.add(_periodic_point("b"))
        lib.add(_cluster_point("c"))
        items = list(lib.iter_by_type("periodic"))
        assert len(items) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Repr
# ═══════════════════════════════════════════════════════════════════════════════

class TestPointLibraryRepr:
    def test_repr(self):
        lib = PointLibrary(name="test")
        lib.add(_periodic_point("a"))
        r = repr(lib)
        assert "PointLibrary" in r
        assert "test" in r
        assert "1" in r  # 1 point


# ═══════════════════════════════════════════════════════════════════════════════
# Duplicate Identity
# ═══════════════════════════════════════════════════════════════════════════════

class TestPointLibraryDuplicateIdentity:
    def test_re_add_overwrites(self):
        lib = PointLibrary()
        p1 = _periodic_point("dup")
        p1.params = {"a": 1.0, "b": 0.0, "w": 0.0}
        p2 = _periodic_point("dup")
        p2.params = {"a": 2.0, "b": 0.0, "w": 0.0}
        lib.add(p1)
        lib.add(p2)
        assert lib.get("dup").params["a"] == 2.0
        assert len(lib.list_all()) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Thread Safety
# ═══════════════════════════════════════════════════════════════════════════════

class TestPointLibraryThreadSafety:
    def test_concurrent_add(self):
        lib = PointLibrary()
        errors = []

        def add_points(prefix):
            try:
                for i in range(20):
                    lib.add(_periodic_point(f"{prefix}_{i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_points, args=(f"t{t}",))
                   for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(lib) == 80

    def test_concurrent_read_write(self):
        lib = PointLibrary()
        for i in range(10):
            lib.add(_periodic_point(f"p{i}"))
        errors = []

        def reads():
            try:
                for _ in range(50):
                    lib.get("p5")
                    lib.list_all()
            except Exception as e:
                errors.append(e)

        def writes():
            try:
                for i in range(20):
                    lib.add(_periodic_point(f"new_{i}"))
            except Exception as e:
                errors.append(e)

        threads = ([threading.Thread(target=reads) for _ in range(3)] +
                   [threading.Thread(target=writes) for _ in range(2)])
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ═══════════════════════════════════════════════════════════════════════════════
# Config Integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestPointLibraryConfig:
    def test_config_overrides(self):
        cfg = LibraryConfig(name="from_config", auto_save=True)
        lib = PointLibrary(config=cfg)
        assert lib.name == "from_config"
        assert lib._auto_save is True

    def test_config_with_storage_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = LibraryConfig(name="with_dir", storage_dir=Path(tmpdir))
            lib = PointLibrary(config=cfg)
            assert lib._storage_dir == Path(tmpdir)

    def test_validate_flag(self):
        lib = PointLibrary(validate=False)
        assert lib._validate is False


# ═══════════════════════════════════════════════════════════════════════════════
# Point Protocol Properties
# ═══════════════════════════════════════════════════════════════════════════════

class TestPointProtocolProperties:
    def test_is_lossless(self):
        p = _periodic_point("a", accuracy=1.0)
        assert p.is_lossless is True

    def test_is_not_lossless(self):
        p = _periodic_point("a", accuracy=0.5)
        assert p.is_lossless is False

    def test_compression_ratio(self):
        p = _periodic_point("a", accuracy=0.9)
        ratio = p.compression_ratio
        assert ratio > 0

    def test_repr(self):
        p = _periodic_point("a")
        r = repr(p)
        assert "a" in r
        assert "periodic" in r

    def test_eq(self):
        p1 = _periodic_point("a")
        p2 = _periodic_point("a")
        assert p1 == p2

    def test_neq(self):
        p1 = _periodic_point("a")
        p2 = _periodic_point("b")
        assert p1 != p2

    def test_hash(self):
        p1 = _periodic_point("a")
        p2 = _periodic_point("a")
        assert hash(p1) == hash(p2)


# ═══════════════════════════════════════════════════════════════════════════════
# PointView Properties
# ═══════════════════════════════════════════════════════════════════════════════

class TestPointViewProperties:
    def test_point_property(self):
        p = _periodic_point("a")
        view = PointView(p, shape=(10,))
        assert view.point is p

    def test_identity_property(self):
        p = _periodic_point("a")
        view = PointView(p)
        assert view.identity == "a"

    def test_function_type_property(self):
        p = _periodic_point("a")
        view = PointView(p)
        assert view.function_type == "periodic"

    def test_shape_property(self):
        p = _periodic_point("a")
        view = PointView(p, shape=(10, 20))
        assert view.shape == (10, 20)

    def test_dtype_property(self):
        p = _periodic_point("a")
        view = PointView(p, dtype="float64")
        assert view.dtype == np.dtype("float64")

    def test_accuracy_property(self):
        p = _periodic_point("a", accuracy=0.85)
        view = PointView(p)
        assert view.accuracy == 0.85

    def test_nbytes_property(self):
        p = _periodic_point("a")
        view = PointView(p)
        assert view.nbytes > 0

    def test_generate(self):
        p = _periodic_point("a")
        view = PointView(p, shape=(10,))
        arr = view.generate()
        assert arr.shape == (10,)

    def test_generate_cached(self):
        p = _periodic_point("a")
        view = PointView(p, shape=(10,))
        arr1 = view.generate()
        arr2 = view.generate()
        assert arr1 is arr2

    def test_clear_cache(self):
        p = _periodic_point("a")
        view = PointView(p, shape=(10,))
        view.generate()
        view.clear_cache()
        assert view._cache is None

    def test_len(self):
        p = _periodic_point("a")
        view = PointView(p, shape=(10,))
        assert len(view) == 10

    def test_len_no_shape(self):
        p = _periodic_point("a")
        view = PointView(p)
        assert len(view) == 0

    def test_repr(self):
        p = _periodic_point("a")
        view = PointView(p, shape=(10,))
        r = repr(view)
        assert "a" in r
        assert "lazy" in r

    def test_repr_cached(self):
        p = _periodic_point("a")
        view = PointView(p, shape=(10,))
        view.generate()
        r = repr(view)
        assert "cached" in r

    def test_array_protocol(self):
        p = _periodic_point("a")
        view = PointView(p, shape=(10,))
        arr = np.array(view)
        assert arr.shape == (10,)

    def test_array_protocol_dtype(self):
        p = _periodic_point("a")
        view = PointView(p, shape=(10,))
        arr = np.array(view, dtype=np.float64)
        assert arr.dtype == np.float64

    def test_getitem_slice(self):
        p = _periodic_point("a")
        view = PointView(p, shape=(10,))
        result = view[2:5]
        assert result.shape[0] == 3

    def test_from_point_and_meta(self):
        p = _periodic_point("a")
        view = PointView.from_point_and_meta(p, shape=(10,), dtype="float32")
        assert view.identity == "a"
        assert view.shape == (10,)


# ═══════════════════════════════════════════════════════════════════════════════
# Additional Coverage
# ═══════════════════════════════════════════════════════════════════════════════

class TestPointLibraryExtraCRUD:
    def test_remove_decrements_count(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        lib.add(_linear_point("b"))
        lib.remove("a")
        assert len(lib) == 1

    def test_clear_resets_by_type(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        lib.add(_cluster_point("c"))
        lib.clear()
        assert lib.list_types() == {}

    def test_list_by_type_empty(self):
        lib = PointLibrary()
        assert lib.list_by_type("periodic") == []

    def test_get_many_all_missing(self):
        lib = PointLibrary()
        result = lib.get_many(["x", "y", "z"])
        assert all(v is None for v in result.values())

    def test_remove_many_partial(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        count = lib.remove_many(["a", "missing"])
        assert count == 1

    def test_exists_many_all_true(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        lib.add(_linear_point("b"))
        result = lib.exists_many(["a", "b"])
        assert all(result.values())


class TestPointLibraryExtraSearch:
    def test_search_empty_library(self):
        lib = PointLibrary()
        assert lib.search("anything") == []

    def test_worst_points_empty(self):
        lib = PointLibrary()
        assert lib.worst_points() == []

    def test_best_points_empty(self):
        lib = PointLibrary()
        assert lib.best_points() == []

    def test_search_by_type_empty(self):
        lib = PointLibrary()
        assert lib.search_by_type("periodic") == []

    def test_search_by_type_all_types(self):
        lib = PointLibrary()
        lib.add(_periodic_point("p1"))
        lib.add(_linear_point("l1"))
        assert len(lib.list_by_type("periodic")) == 1
        assert len(lib.list_by_type("linear")) == 1


class TestPointLibraryExtraStats:
    def test_stats_after_remove(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        lib.remove("a")
        s = lib.stats()
        assert s["total_points"] == 0
        assert s["ops"]["removes"] == 1

    def test_hit_rate_all_misses(self):
        lib = PointLibrary()
        lib.get("missing")
        assert lib.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        lib = PointLibrary()
        lib.add(_periodic_point("a"))
        lib.get("a")
        assert lib.hit_rate == 1.0

    def test_stats_name(self):
        lib = PointLibrary(name="mylib")
        s = lib.stats()
        assert s["name"] == "mylib"


class TestPointLibraryExtraPersistence:
    def test_save_and_load_cluster(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = PointLibrary(name="cluster_lib", storage_dir=Path(tmpdir))
            lib.add(_cluster_point("c1"))
            lib.save()
            loaded = PointLibrary.load(Path(tmpdir) / "cluster_lib.points.json")
            assert loaded.has("c1")

    def test_load_preserves_created_at(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = PointLibrary(name="ts", storage_dir=Path(tmpdir))
            lib.add(_periodic_point("a"))
            lib.save()
            loaded = PointLibrary.load(Path(tmpdir) / "ts.points.json")
            assert loaded._created_at == lib._created_at

    def test_context_manager_no_storage(self):
        with PointLibrary(name="no_dir") as lib:
            lib.add(_periodic_point("a"))
        assert lib.has("a")


class TestPointLibraryExtraIteration:
    def test_iter_empty(self):
        lib = PointLibrary()
        assert list(lib) == []

    def test_iter_by_type_empty(self):
        lib = PointLibrary()
        assert list(lib.iter_by_type("periodic")) == []


class TestPointLibraryExtraValidation:
    def test_negative_accuracy_raises(self):
        lib = PointLibrary()
        p = Point(identity="bad", function_type="periodic",
                  params=_periodic_params(), accuracy=-0.1)
        with pytest.raises(ValueError, match="Accuracy must be 0-1"):
            lib.add(p)


class TestPointLibraryExtraDecompress:
    def test_decompress_function_type(self):
        lib = PointLibrary()
        weights = np.random.randn(128)
        lib.compress_and_store(weights, identity="f1", method="function")
        result = lib.decompress_to("f1")
        assert result is not None

    def test_decompress_cluster_with_shape(self):
        lib = PointLibrary()
        weights = np.random.randn(128)
        lib.compress_and_store(weights, identity="c1")
        result = lib.decompress_to("c1", shape=(128,))
        assert result.shape == (128,)


class TestPointLibraryExtraCompress:
    def test_compress_and_store_returns_point(self):
        lib = PointLibrary()
        weights = np.random.randn(64)
        p = lib.compress_and_store(weights, identity="test")
        assert p.identity == "test"
        assert lib.has("test")


class TestPointLibraryExtraConfig:
    def test_config_name_overrides(self):
        cfg = LibraryConfig(name="custom")
        lib = PointLibrary(config=cfg)
        assert lib.name == "custom"

    def test_default_name(self):
        lib = PointLibrary()
        assert lib.name == "default"
