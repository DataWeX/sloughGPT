"""Tests for PointProtocol, PointView, and improved PointLibrary."""

import threading

import numpy as np
import pytest

from domains.infrastructure.pugqeep.point import Point
from domains.infrastructure.pugqeep.point_interface import (
    PointProtocol, PointView, FunctionType,
)
from domains.infrastructure.pugqeep.library import PointLibrary


# ── PointProtocol compliance ────────────────────────────────────────


class TestPointProtocol:
    def test_point_implements_protocol(self):
        p = Point(identity="t", function_type="raw",
                  params={"data_b64": "", "shape": [], "dtype": "float32"})
        assert isinstance(p, PointProtocol)

    def test_point_has_required_attributes(self):
        p = Point(identity="x", function_type="linear",
                  params={"a": 1.0, "b": 0.0}, accuracy=0.95)
        assert p.identity == "x"
        assert p.function_type == "linear"
        assert p.params == {"a": 1.0, "b": 0.0}
        assert p.accuracy == 0.95
        assert p.residual is None
        assert p.dtype == "float32"
        assert p.shape == ()

    def test_point_generate(self):
        p = Point(identity="lin", function_type="linear",
                  params={"a": 2.0, "b": 1.0})
        arr = p.generate(5)
        assert arr.shape == (5,)
        assert arr[0] == pytest.approx(1.0)  # 2*0 + 1
        assert arr[4] == pytest.approx(9.0)  # 2*4 + 1

    def test_point_nbytes(self):
        p = Point(identity="lin", function_type="linear",
                  params={"a": 1.0, "b": 0.0})
        assert p.nbytes() > 0

    def test_point_is_lossless(self):
        p = Point(identity="a", function_type="raw", params={}, accuracy=1.0)
        assert p.is_lossless
        p2 = Point(identity="b", function_type="cluster", params={}, accuracy=0.95)
        assert not p2.is_lossless

    def test_point_compression_ratio(self):
        p = Point(identity="c", function_type="linear",
                  params={"a": 1.0, "b": 0.0})
        ratio = p.compression_ratio
        assert ratio > 0

    def test_point_eq_and_hash(self):
        p1 = Point(identity="a", function_type="linear", params={})
        p2 = Point(identity="a", function_type="linear", params={})
        p3 = Point(identity="b", function_type="linear", params={})
        assert p1 == p2
        assert p1 != p3
        assert hash(p1) == hash(p2)
        assert hash(p1) != hash(p3)

    def test_point_repr(self):
        p = Point(identity="test", function_type="cluster",
                  params={}, accuracy=0.99)
        r = repr(p)
        assert "test" in r
        assert "cluster" in r
        assert "0.99" in r

    def test_function_type_enum(self):
        assert FunctionType.from_str("cluster") == FunctionType.CLUSTER
        assert FunctionType.from_str("linear") == FunctionType.LINEAR
        with pytest.raises(ValueError):
            FunctionType.from_str("invalid")


# ── PointView ───────────────────────────────────────────────────────


class TestPointView:
    def _make_cluster_point(self, n=100, k=8):
        centroids = np.random.randn(k).astype(np.float32)
        assignments = np.random.randint(0, k, size=n).astype(np.uint8)
        return Point(
            identity="view-test",
            function_type="cluster",
            params={"centroids": centroids, "assignments": assignments},
            accuracy=0.9,
        )

    def test_view_lazy(self):
        p = self._make_cluster_point()
        view = PointView(p, shape=(100,), dtype="float32")
        assert view._cache is None  # not generated yet

    def test_view_generate(self):
        p = self._make_cluster_point()
        view = PointView(p, shape=(100,), dtype="float32")
        arr = view.generate()
        assert arr.shape == (100,)
        assert arr.dtype == np.float32
        assert view._cache is not None  # now cached

    def test_view_cache_hit(self):
        p = self._make_cluster_point()
        view = PointView(p, shape=(100,), dtype="float32")
        arr1 = view.generate()
        arr2 = view.generate()
        assert arr1 is arr2  # same object (cached)

    def test_view_clear_cache(self):
        p = self._make_cluster_point()
        view = PointView(p, shape=(100,), dtype="float32")
        view.generate()
        view.clear_cache()
        assert view._cache is None

    def test_view_numpy_protocol(self):
        p = self._make_cluster_point()
        view = PointView(p, shape=(100,), dtype="float32")
        arr = np.array(view)
        assert arr.shape == (100,)

    def test_view_slicing(self):
        p = self._make_cluster_point()
        view = PointView(p, shape=(100,), dtype="float32")
        slice_arr = view[10:20]
        assert slice_arr.shape == (10,)

    def test_view_partial_decompress_cluster(self):
        """Cluster points should decompress only the sliced portion."""
        centroids = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        assignments = np.tile([0, 1, 2, 3], 25).astype(np.uint8)  # 100 elements
        p = Point(identity="cluster-partial", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments},
                  accuracy=1.0, dtype="float32", shape=(100,))
        view = PointView(p, shape=(100,), dtype="float32")

        # Slice should not trigger full decompression
        slice_arr = view[10:15]
        assert slice_arr.shape == (5,)
        # Verify values match full decompression
        full = view.generate()
        np.testing.assert_array_equal(slice_arr, full[10:15])

    def test_view_len(self):
        p = self._make_cluster_point()
        view = PointView(p, shape=(100,), dtype="float32")
        assert len(view) == 100

    def test_view_properties(self):
        p = self._make_cluster_point()
        view = PointView(p, shape=(10, 10), dtype="float32")
        assert view.identity == "view-test"
        assert view.shape == (10, 10)
        assert view.dtype == np.float32
        assert view.accuracy == 0.9
        assert view.point is p

    def test_view_repr(self):
        p = self._make_cluster_point()
        view = PointView(p, shape=(100,), dtype="float32")
        r = repr(view)
        assert "lazy" in r
        view.generate()
        r = repr(view)
        assert "cached" in r


# ── Improved PointLibrary ───────────────────────────────────────────


class TestPointLibraryInterface:
    def test_add_and_get(self):
        lib = PointLibrary("test-add")
        p = Point(identity="w1", function_type="linear",
                  params={"a": 1.0, "b": 0.0}, accuracy=0.9)
        lib.add(p)
        got = lib.get("w1")
        assert got is p

    def test_add_many(self):
        lib = PointLibrary("test-add-many")
        points = [Point(identity=f"p{i}", function_type="linear",
                        params={"a": float(i), "b": 0.0})
                  for i in range(10)]
        count = lib.add_many(points)
        assert count == 10
        assert len(lib) == 10

    def test_get_many(self):
        lib = PointLibrary("test-get-many")
        for i in range(5):
            lib.add(Point(identity=f"p{i}", function_type="linear",
                          params={"a": float(i), "b": 0.0}))
        results = lib.get_many(["p0", "p2", "p4", "missing"])
        assert results["p0"] is not None
        assert results["p2"] is not None
        assert results["p4"] is not None
        assert results["missing"] is None

    def test_remove_many(self):
        lib = PointLibrary("test-remove-many")
        for i in range(5):
            lib.add(Point(identity=f"p{i}", function_type="linear",
                          params={"a": float(i), "b": 0.0}))
        removed = lib.remove_many(["p0", "p2", "missing"])
        assert removed == 2
        assert len(lib) == 3

    def test_exists_many(self):
        lib = PointLibrary("test-exists-many")
        lib.add(Point(identity="a", function_type="linear", params={}))
        result = lib.exists_many(["a", "b"])
        assert result["a"] is True
        assert result["b"] is False

    def test_list_identities(self):
        lib = PointLibrary("test-list-id")
        for name in ["x", "y", "z"]:
            lib.add(Point(identity=name, function_type="linear", params={}))
        ids = lib.list_identities()
        assert sorted(ids) == ["x", "y", "z"]

    def test_list_types(self):
        lib = PointLibrary("test-list-types")
        cluster_params = {"centroids": np.zeros(4, dtype=np.float32), "assignments": np.zeros(8, dtype=np.uint8)}
        lib.add(Point(identity="a", function_type="linear", params={}))
        lib.add(Point(identity="b", function_type="cluster", params=cluster_params))
        lib.add(Point(identity="c", function_type="linear", params={}))
        types = lib.list_types()
        assert types["linear"] == 2
        assert types["cluster"] == 1

    def test_iterator(self):
        lib = PointLibrary("test-iter")
        for i in range(3):
            lib.add(Point(identity=f"p{i}", function_type="linear", params={}))
        items = list(lib)
        assert len(items) == 3

    def test_iter_by_type(self):
        lib = PointLibrary("test-iter-type")
        cluster_params = {"centroids": np.zeros(4, dtype=np.float32), "assignments": np.zeros(8, dtype=np.uint8)}
        lib.add(Point(identity="a", function_type="linear", params={}))
        lib.add(Point(identity="b", function_type="cluster", params=cluster_params))
        lib.add(Point(identity="c", function_type="linear", params={}))
        linear = list(lib.iter_by_type("linear"))
        assert len(linear) == 2

    def test_contains(self):
        lib = PointLibrary("test-contains")
        lib.add(Point(identity="x", function_type="linear", params={}))
        assert "x" in lib
        assert "y" not in lib

    def test_thread_safety(self):
        lib = PointLibrary("test-thread")
        errors = []

        def writer(start):
            try:
                for i in range(50):
                    lib.add(Point(identity=f"t{start}_{i}", function_type="linear",
                                  params={"a": float(i), "b": 0.0}))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    lib.list_all()
                    len(lib)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(4):
            threads.append(threading.Thread(target=writer, args=(i,)))
        for _ in range(2):
            threads.append(threading.Thread(target=reader))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(lib) == 200

    def test_stats(self):
        lib = PointLibrary("test-stats")
        cluster_params = {"centroids": np.zeros(4, dtype=np.float32), "assignments": np.zeros(8, dtype=np.uint8)}
        lib.add(Point(identity="a", function_type="linear",
                      params={"a": 1.0, "b": 0.0}, accuracy=0.9))
        lib.add(Point(identity="b", function_type="cluster", params=cluster_params))
        s = lib.stats()
        assert s["total_points"] == 2
        assert "ops" in s
        assert s["ops"]["adds"] == 2

    def test_hit_rate(self):
        lib = PointLibrary("test-hit-rate")
        lib.add(Point(identity="a", function_type="linear", params={}))
        lib.get("a")  # hit
        lib.get("missing")  # miss
        assert lib.hit_rate == pytest.approx(0.5)

    def test_context_manager(self):
        with PointLibrary("test-ctx", validate=False) as lib:
            lib.add(Point(identity="a", function_type="linear", params={}))
            assert len(lib) == 1

    def test_search_by_type(self):
        lib = PointLibrary("test-search-type")
        cluster_params = {"centroids": np.zeros(4, dtype=np.float32), "assignments": np.zeros(8, dtype=np.uint8)}
        lib.add(Point(identity="layer_0.weight", function_type="cluster", params=cluster_params))
        lib.add(Point(identity="layer_1.bias", function_type="linear", params={}))
        lib.add(Point(identity="embed.weight", function_type="cluster", params=cluster_params))
        results = lib.search_by_type("cluster", "layer")
        assert len(results) == 1
        assert results[0].identity == "layer_0.weight"

    def test_worst_points(self):
        lib = PointLibrary("test-worst")
        lib.add(Point(identity="a", function_type="linear", params={}, accuracy=0.5))
        lib.add(Point(identity="b", function_type="linear", params={}, accuracy=0.9))
        lib.add(Point(identity="c", function_type="linear", params={}, accuracy=0.3))
        worst = lib.worst_points(2)
        assert worst[0].identity == "c"
        assert worst[1].identity == "a"

    def test_point_view_integration(self):
        lib = PointLibrary("test-view")
        centroids = np.random.randn(8).astype(np.float32)
        assignments = np.random.randint(0, 8, size=100).astype(np.uint8)
        p = Point(identity="w1", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments})
        lib.add(p)
        view = lib.view("w1", shape=(100,), dtype="float32")
        assert view is not None
        arr = view.generate()
        assert arr.shape == (100,)
        # Same view on second call
        view2 = lib.view("w1")
        assert view2 is view

    def test_clear_views(self):
        lib = PointLibrary("test-clear-views")
        lib.add(Point(identity="a", function_type="linear", params={}))
        lib.view("a")
        assert len(lib._views) == 1
        lib.clear_views()
        assert len(lib._views) == 0

    def test_validation_rejects_empty_identity(self):
        lib = PointLibrary("test-validate", validate=True)
        with pytest.raises(ValueError, match="empty"):
            lib.add(Point(identity="", function_type="linear", params={}))

    def test_validation_rejects_bad_accuracy(self):
        lib = PointLibrary("test-validate", validate=True)
        with pytest.raises(ValueError, match="0-1"):
            lib.add(Point(identity="x", function_type="linear", params={}, accuracy=1.5))

    def test_validation_skipped_when_disabled(self):
        lib = PointLibrary("test-no-validate", validate=False)
        lib.add(Point(identity="", function_type="linear", params={}))
        assert len(lib) == 1
