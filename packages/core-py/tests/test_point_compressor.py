"""Tests for point_compressor — backward-compatible shim over pugqeep.

Comprehensive coverage of Point, PointCompressor, PointLibrary,
PointDeduplicator, PointLibrarySync, and ModelTree.
"""

import base64
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import domains.infrastructure.pugqeep as pugqeep
from domains.infrastructure import point_compressor as pc
from domains.infrastructure.pugqeep.point import Point
from domains.infrastructure.pugqeep.compressor import PointCompressor
from domains.infrastructure.pugqeep.library import PointLibrary
from domains.infrastructure.pugqeep.dedup import PointDeduplicator, PointLibrarySync
from domains.infrastructure.pugqeep.model_tree import ModelTree
from domains.infrastructure.pugqeep.config import CompressorConfig, LibraryConfig


# ── Re-exports ───────────────────────────────────────────────────────────────

class TestReexports:
    def test_point_identity(self):
        assert pc.Point is pugqeep.Point

    def test_point_compressor_identity(self):
        assert pc.PointCompressor is pugqeep.PointCompressor

    def test_point_library_identity(self):
        assert pc.PointLibrary is pugqeep.PointLibrary

    def test_model_tree_identity(self):
        assert pc.ModelTree is pugqeep.ModelTree

    def test_point_deduplicator_identity(self):
        assert pc.PointDeduplicator is pugqeep.PointDeduplicator

    def test_point_library_sync_identity(self):
        assert pc.PointLibrarySync is pugqeep.PointLibrarySync

    def test_load_model_to_points_identity(self):
        assert pc.load_model_to_points is pugqeep.load_model_to_points

    def test_pgq_identity(self):
        assert pc.PGQ is pugqeep.PGQ

    def test_point_lib_alias(self):
        assert pc.PointLib is pugqeep.PGQ

    def test_load_library_alias(self):
        assert pc.load_library == pugqeep.PGQ.load

    def test_all_exports_importable(self):
        for name in pc.__all__:
            assert hasattr(pc, name)


class TestSaveLibrary:
    def test_delegates_to_library_save(self):
        calls = []

        class FakeLib:
            def save(self, path):
                calls.append(path)

        pc.save_library(FakeLib(), "/tmp/lib.json")
        assert calls == ["/tmp/lib.json"]


# ── Point — generate ─────────────────────────────────────────────────────────

class TestPointGenerate:
    def test_cluster_generate(self):
        centroids = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        assignments = np.array([0, 1, 2, 0, 1], dtype=np.uint8)
        p = Point(identity="w", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments})
        result = p.generate(5)
        np.testing.assert_array_equal(result, centroids[assignments[:5]])

    def test_raw_generate(self):
        raw = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        data_b64 = base64.b64encode(raw.tobytes()).decode()
        p = Point(identity="w", function_type="raw",
                  params={"data_b64": data_b64, "dtype": "float32", "shape": []})
        result = p.generate(3)
        np.testing.assert_allclose(result, raw)

    def test_periodic_generate(self):
        p = Point(identity="w", function_type="periodic",
                  params={"a": 1.0, "b": 0.0, "w": 0.0})
        result = p.generate(4)
        i = np.arange(4, dtype=np.float32)
        expected = 1.0 * np.cos(i)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_linear_generate(self):
        p = Point(identity="w", function_type="linear",
                  params={"a": 2.0, "b": 1.0})
        result = p.generate(5)
        i = np.arange(5, dtype=np.float32)
        expected = 2.0 * i + 1.0
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_polynomial_generate(self):
        p = Point(identity="w", function_type="polynomial",
                  params={"a": 1.0, "b": 0.0, "c": 0.0})
        result = p.generate(4)
        i = np.arange(4, dtype=np.float32)
        expected = i ** 2
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_unknown_function_type_raises(self):
        p = Point(identity="w", function_type="bogus", params={})
        with pytest.raises(ValueError, match="Unknown function type"):
            p.generate(5)

    def test_cluster_with_residual(self):
        centroids = np.array([0.0, 1.0], dtype=np.float32)
        assignments = np.array([0, 1, 0, 1], dtype=np.uint8)
        residual = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
        p = Point(identity="w", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments},
                  residual=residual)
        result = p.generate(4)
        expected = centroids[assignments] + residual
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_periodic_with_residual(self):
        p = Point(identity="w", function_type="periodic",
                  params={"a": 1.0, "b": 0.0, "w": 0.0},
                  residual=np.array([0.5, 0.5], dtype=np.float32))
        result = p.generate(2)
        assert result[0] == pytest.approx(1.5, abs=1e-5)


# ── Point — nbytes ───────────────────────────────────────────────────────────

class TestPointNbytes:
    def test_cluster_nbytes(self):
        centroids = np.zeros(4, dtype=np.float32)
        assignments = np.zeros(8, dtype=np.uint8)
        p = Point(identity="w", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments})
        assert p.nbytes() == 4 * 4 + 8

    def test_raw_nbytes(self):
        raw = np.zeros(10, dtype=np.float32)
        data_b64 = base64.b64encode(raw.tobytes()).decode()
        p = Point(identity="w", function_type="raw",
                  params={"data_b64": data_b64})
        assert p.nbytes() == 40

    def test_periodic_nbytes(self):
        p = Point(identity="w", function_type="periodic",
                  params={"a": 1.0, "b": 0.0, "w": 0.0})
        assert p.nbytes() == 4 + 3 * 4

    def test_linear_nbytes(self):
        p = Point(identity="w", function_type="linear",
                  params={"a": 1.0, "b": 0.0})
        assert p.nbytes() == 4 + 2 * 4

    def test_polynomial_nbytes(self):
        p = Point(identity="w", function_type="polynomial",
                  params={"a": 1.0, "b": 0.0, "c": 0.0})
        assert p.nbytes() == 4 + 3 * 4

    def test_cluster_with_residual_nbytes(self):
        centroids = np.zeros(2, dtype=np.float32)
        assignments = np.zeros(4, dtype=np.uint8)
        residual = np.zeros(4, dtype=np.float32)
        p = Point(identity="w", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments},
                  residual=residual)
        assert p.nbytes() == 2 * 4 + 4 + 4 * 4

    def test_estimate_raw_bytes_cluster(self):
        assignments = np.zeros(10, dtype=np.uint8)
        centroids = np.zeros(3, dtype=np.float32)
        p = Point(identity="w", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments})
        assert p._estimate_raw_bytes() == 10 * 4

    def test_estimate_raw_bytes_raw(self):
        raw = np.zeros(5, dtype=np.float32)
        data_b64 = base64.b64encode(raw.tobytes()).decode()
        p = Point(identity="w", function_type="raw",
                  params={"data_b64": data_b64})
        assert p._estimate_raw_bytes() == 20

    def test_estimate_raw_bytes_with_shape(self):
        p = Point(identity="w", function_type="periodic",
                  params={"a": 1.0, "b": 0.0, "w": 0.0}, shape=(10,))
        assert p._estimate_raw_bytes() == 10 * 4

    def test_estimate_raw_bytes_no_shape(self):
        p = Point(identity="w", function_type="periodic",
                  params={"a": 1.0, "b": 0.0, "w": 0.0})
        assert p._estimate_raw_bytes() == 0


# ── Point — serialization round-trips ────────────────────────────────────────

class TestPointSerialization:
    def test_periodic_round_trip(self):
        p = Point(identity="w", function_type="periodic",
                  params={"a": 1.5, "b": 2.5, "w": 0.5})
        data = p.to_bytes()
        p2 = Point.from_bytes(data, identity="w")
        assert p2.function_type == "periodic"
        assert p2.params["a"] == pytest.approx(1.5)
        assert p2.params["b"] == pytest.approx(2.5)
        assert p2.params["w"] == pytest.approx(0.5)

    def test_linear_round_trip(self):
        p = Point(identity="w", function_type="linear",
                  params={"a": 3.0, "b": 4.0})
        data = p.to_bytes()
        p2 = Point.from_bytes(data, identity="w")
        assert p2.function_type == "linear"
        assert p2.params["a"] == pytest.approx(3.0)
        assert p2.params["b"] == pytest.approx(4.0)

    def test_polynomial_round_trip(self):
        p = Point(identity="w", function_type="polynomial",
                  params={"a": 1.0, "b": 2.0, "c": 3.0})
        data = p.to_bytes()
        p2 = Point.from_bytes(data, identity="w")
        assert p2.function_type == "polynomial"
        assert p2.params["a"] == pytest.approx(1.0)
        assert p2.params["b"] == pytest.approx(2.0)
        assert p2.params["c"] == pytest.approx(3.0)

    def test_cluster_round_trip(self):
        centroids = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assignments = np.array([0, 1, 2, 0], dtype=np.uint8)
        p = Point(identity="w", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments})
        data = p.to_bytes()
        p2 = Point.from_bytes(data, identity="w")
        assert p2.function_type == "cluster"
        np.testing.assert_allclose(p2.params["centroids"], centroids)
        np.testing.assert_array_equal(p2.params["assignments"], assignments)

    def test_raw_round_trip(self):
        raw = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        data_b64 = base64.b64encode(raw.tobytes()).decode()
        p = Point(identity="w", function_type="raw",
                  params={"data_b64": data_b64, "dtype": "float32", "shape": []})
        data = p.to_bytes()
        p2 = Point.from_bytes(data, identity="w")
        assert p2.function_type == "raw"
        decoded = base64.b64decode(p2.params["data_b64"])
        np.testing.assert_allclose(np.frombuffer(decoded, dtype=np.float32), raw)

    def test_cluster_with_residual_round_trip(self):
        centroids = np.array([1.0, 2.0], dtype=np.float32)
        assignments = np.array([0, 1], dtype=np.uint8)
        residual = np.array([0.1, -0.1], dtype=np.float32)
        p = Point(identity="w", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments},
                  residual=residual)
        data = p.to_bytes()
        p2 = Point.from_bytes(data, identity="w")
        assert p2.residual is not None
        np.testing.assert_allclose(p2.residual, residual)

    def test_from_bytes_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown type code"):
            Point.from_bytes(b"XXXXrest")

    def test_dict_round_trip_periodic(self):
        p = Point(identity="w", function_type="periodic",
                  params={"a": 1.0, "b": 2.0, "w": 3.0}, accuracy=0.95)
        d = p.to_dict()
        p2 = Point.from_dict(d)
        assert p2.identity == "w"
        assert p2.accuracy == 0.95
        assert p2.params["a"] == pytest.approx(1.0)

    def test_dict_round_trip_cluster(self):
        centroids = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assignments = np.array([0, 1, 2], dtype=np.uint8)
        p = Point(identity="w", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments})
        d = p.to_dict()
        p2 = Point.from_dict(d)
        np.testing.assert_allclose(p2.params["centroids"], centroids)
        np.testing.assert_array_equal(p2.params["assignments"], assignments)

    def test_dict_round_trip_raw(self):
        raw = np.array([1.0, 2.0], dtype=np.float32)
        data_b64 = base64.b64encode(raw.tobytes()).decode()
        p = Point(identity="w", function_type="raw",
                  params={"data_b64": data_b64, "dtype": "float32", "shape": []})
        d = p.to_dict()
        p2 = Point.from_dict(d)
        assert p2.function_type == "raw"

    def test_dict_round_trip_linear(self):
        p = Point(identity="w", function_type="linear",
                  params={"a": 5.0, "b": 6.0})
        d = p.to_dict()
        p2 = Point.from_dict(d)
        assert p2.params["a"] == pytest.approx(5.0)
        assert p2.params["b"] == pytest.approx(6.0)

    def test_dict_round_trip_polynomial(self):
        p = Point(identity="w", function_type="polynomial",
                  params={"a": 1.0, "b": 2.0, "c": 3.0})
        d = p.to_dict()
        p2 = Point.from_dict(d)
        assert p2.params["a"] == pytest.approx(1.0)
        assert p2.params["c"] == pytest.approx(3.0)


# ── PointCompressor — compress ───────────────────────────────────────────────

class TestCompressorCoverage:
    def test_compress_default_method(self):
        comp = PointCompressor()
        point = comp.compress(np.random.randn(100).astype(np.float32), "w")
        assert point.function_type == "cluster"

    def test_compress_unknown_method_raises(self):
        comp = PointCompressor()
        with pytest.raises(ValueError):
            comp.compress(np.zeros(10, dtype=np.float32), "w", method="bogus")

    def test_compress_function_method(self):
        comp = PointCompressor()
        x = np.arange(50, dtype=np.float32)
        point = comp.compress(x, "w", method="function")
        assert point.function_type in ("periodic", "linear", "polynomial")

    def test_compress_cluster_method(self):
        comp = PointCompressor()
        x = np.random.randn(200).astype(np.float32)
        point = comp.compress(x, "w", method="cluster")
        assert point.function_type == "cluster"

    def test_compress_cluster_empty_raises(self):
        comp = PointCompressor()
        with pytest.raises(ValueError, match="empty"):
            comp.compress_cluster(np.array([]), "w")

    def test_compress_cluster_nan_raises(self):
        comp = PointCompressor()
        with pytest.raises(ValueError, match="NaN"):
            comp.compress_cluster(np.array([1.0, float("nan")]), "w")

    def test_compress_cluster_inf_raises(self):
        comp = PointCompressor()
        with pytest.raises(ValueError, match="NaN"):
            comp.compress_cluster(np.array([1.0, float("inf")]), "w")

    def test_compress_cluster_zero_clusters_raises(self):
        comp = PointCompressor()
        with pytest.raises(ValueError, match="n_clusters"):
            comp.compress_cluster(np.ones(10, dtype=np.float32), "w", n_clusters=0)

    def test_compress_function_empty_raises(self):
        comp = PointCompressor()
        with pytest.raises(ValueError, match="empty"):
            comp.compress_function(np.array([]), "w")

    def test_compress_function_nan_raises(self):
        comp = PointCompressor()
        with pytest.raises(ValueError, match="NaN"):
            comp.compress_function(np.array([1.0, float("nan")]), "w")

    def test_compress_cluster_overflows_to_all_clusters(self):
        comp = PointCompressor()
        x = np.array([1.0], dtype=np.float32)
        point = comp.compress_cluster(x, "w", n_clusters=100)
        assert point.function_type == "cluster"
        assert len(point.params["assignments"]) == 1

    def test_compress_function_linear_fit(self):
        comp = PointCompressor()
        x = np.arange(100, dtype=np.float32) * 2.0 + 1.0
        point = comp.compress_function(x, "w")
        assert point.function_type == "linear"
        assert point.accuracy > 0.99


# ── PointCompressor — batch compress ─────────────────────────────────────────

class TestCompressorBatch:
    def test_compress_batch_returns_dict(self):
        comp = PointCompressor()
        weights = {
            "w1": np.random.randn(50).astype(np.float32),
            "w2": np.random.randn(50).astype(np.float32),
        }
        result = comp.compress_batch(weights)
        assert "w1" in result
        assert "w2" in result
        assert result["w1"].function_type == "cluster"

    def test_compress_batch_with_prefix(self):
        comp = PointCompressor()
        weights = {"layer1": np.random.randn(50).astype(np.float32)}
        result = comp.compress_batch(weights, prefix="model.")
        assert result["layer1"].identity == "model.layer1"

    def test_compress_batch_function_method(self):
        comp = PointCompressor()
        weights = {"w1": np.arange(50, dtype=np.float32)}
        result = comp.compress_batch(weights, method="function")
        assert result["w1"].function_type in ("periodic", "linear", "polynomial")


# ── PointCompressor — measure_compression ────────────────────────────────────

class TestCompressorMeasure:
    def test_measure_compression_cluster_with_residual(self):
        comp = PointCompressor()
        centroids = np.zeros(4, dtype=np.float32)
        assignments = np.zeros(8, dtype=np.uint8)
        residual = np.zeros(8, dtype=np.float32)
        point = Point(identity="w", function_type="cluster",
                      params={"centroids": centroids, "assignments": assignments},
                      residual=residual, accuracy=0.99)
        m = comp.measure_compression(np.zeros(8, dtype=np.float32), point)
        assert m["compressed_bytes"] == centroids.nbytes + assignments.nbytes + residual.nbytes

    def test_measure_compression_raw(self):
        comp = PointCompressor()
        raw = np.zeros(8, dtype=np.float32)
        point = Point(identity="w", function_type="raw",
                      params={"data_b64": base64.b64encode(raw.tobytes()).decode()})
        m = comp.measure_compression(raw, point)
        assert m["compressed_bytes"] == raw.nbytes

    def test_measure_compression_function_with_residual(self):
        comp = PointCompressor()
        residual = np.zeros(3, dtype=np.float32)
        point = Point(identity="w", function_type="periodic",
                      params={"a": 1.0, "b": 0.0, "w": 0.0},
                      residual=residual, accuracy=0.9)
        m = comp.measure_compression(np.zeros(10, dtype=np.float32), point)
        assert m["compressed_bytes"] == 4 + 3 * 4 + residual.nbytes

    def test_measure_compression_ratio(self):
        comp = PointCompressor()
        x = np.random.randn(1000).astype(np.float32)
        point = comp.compress_cluster(x, "w")
        m = comp.measure_compression(x, point)
        assert m["ratio"] > 0
        assert m["accuracy"] >= 0

    def test_measure_compression_keys(self):
        comp = PointCompressor()
        x = np.ones(50, dtype=np.float32)
        point = comp.compress_cluster(x, "w")
        m = comp.measure_compression(x, point)
        assert "raw_bytes" in m
        assert "compressed_bytes" in m
        assert "ratio" in m
        assert "accuracy" in m
        assert "function_type" in m


# ── PointCompressor — decompress ─────────────────────────────────────────────

class TestCompressorDecompress:
    def test_decompress_cluster(self):
        comp = PointCompressor()
        x = np.random.randn(200).astype(np.float32)
        point = comp.compress_cluster(x, "w")
        result = comp.decompress(point, len(x))
        assert result.shape == (len(x),)

    def test_decompress_function(self):
        comp = PointCompressor()
        x = np.arange(50, dtype=np.float32)
        point = comp.compress_function(x, "w")
        result = comp.decompress(point, len(x))
        assert result.shape == (len(x),)


# ── PointCompressor — config ─────────────────────────────────────────────────

class TestCompressorConfig:
    def test_config_overrides(self):
        cfg = CompressorConfig(n_clusters=32, lloyd_iterations=10)
        comp = PointCompressor(config=cfg)
        assert comp.n_clusters == 32
        assert comp.lloyd_iterations == 10

    def test_defaults_without_config(self):
        comp = PointCompressor()
        assert comp.n_clusters == 16
        assert comp.lloyd_iterations == 5
        assert comp.method == "cluster"

    def test_keyword_overrides(self):
        comp = PointCompressor(n_clusters=64, lloyd_iterations=20)
        assert comp.n_clusters == 64
        assert comp.lloyd_iterations == 20

    def test_residual_threshold(self):
        comp = PointCompressor(residual_threshold=0.9)
        assert comp.residual_threshold == 0.9


# ── PointLibrary — CRUD ─────────────────────────────────────────────────────

class TestLibraryCRUD:
    def test_add_and_get(self):
        lib = PointLibrary("test")
        p = Point(identity="w1", function_type="linear",
                  params={"a": 1.0, "b": 0.0})
        lib.add(p)
        assert lib.get("w1") is p

    def test_add_returns_true_for_new(self):
        lib = PointLibrary("test")
        p = Point(identity="w1", function_type="linear",
                  params={"a": 1.0, "b": 0.0})
        assert lib.add(p) is True

    def test_add_replaces_existing(self):
        lib = PointLibrary("test")
        p1 = Point(identity="w1", function_type="linear",
                   params={"a": 1.0, "b": 0.0})
        p2 = Point(identity="w1", function_type="linear",
                   params={"a": 2.0, "b": 0.0})
        lib.add(p1)
        assert lib.add(p2) is False
        assert lib.get("w1") is p2

    def test_get_missing_returns_none(self):
        lib = PointLibrary("test")
        assert lib.get("nonexistent") is None

    def test_remove_existing(self):
        lib = PointLibrary("test")
        p = Point(identity="w1", function_type="linear",
                  params={"a": 1.0, "b": 0.0})
        lib.add(p)
        assert lib.remove("w1") is True
        assert lib.get("w1") is None

    def test_remove_missing_returns_false(self):
        lib = PointLibrary("test")
        assert lib.remove("nonexistent") is False

    def test_has(self):
        lib = PointLibrary("test")
        p = Point(identity="w1", function_type="linear",
                  params={"a": 1.0, "b": 0.0})
        lib.add(p)
        assert lib.has("w1") is True
        assert lib.has("w2") is False

    def test_clear(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="a", function_type="linear", params={"a": 1.0, "b": 0.0}))
        lib.add(Point(identity="b", function_type="linear", params={"a": 1.0, "b": 0.0}))
        lib.clear()
        assert len(lib) == 0

    def test_len(self):
        lib = PointLibrary("test")
        assert len(lib) == 0
        lib.add(Point(identity="a", function_type="linear", params={"a": 1.0, "b": 0.0}))
        assert len(lib) == 1

    def test_contains(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="a", function_type="linear", params={"a": 1.0, "b": 0.0}))
        assert "a" in lib
        assert "b" not in lib


# ── PointLibrary — batch ops ─────────────────────────────────────────────────

class TestLibraryBatch:
    def test_add_many(self):
        lib = PointLibrary("test")
        points = [
            Point(identity=f"w{i}", function_type="linear",
                  params={"a": float(i), "b": 0.0})
            for i in range(5)
        ]
        count = lib.add_many(points)
        assert count == 5
        assert len(lib) == 5

    def test_get_many(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="a", function_type="linear", params={"a": 1.0, "b": 0.0}))
        lib.add(Point(identity="b", function_type="linear", params={"a": 2.0, "b": 0.0}))
        result = lib.get_many(["a", "b", "c"])
        assert result["a"] is not None
        assert result["b"] is not None
        assert result["c"] is None

    def test_remove_many(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="a", function_type="linear", params={"a": 1.0, "b": 0.0}))
        lib.add(Point(identity="b", function_type="linear", params={"a": 2.0, "b": 0.0}))
        count = lib.remove_many(["a", "b", "c"])
        assert count == 2
        assert len(lib) == 0

    def test_exists_many(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="a", function_type="linear", params={"a": 1.0, "b": 0.0}))
        result = lib.exists_many(["a", "b"])
        assert result["a"] is True
        assert result["b"] is False


# ── PointLibrary — listing ───────────────────────────────────────────────────

class TestLibraryListing:
    def test_list_all(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="a", function_type="linear", params={"a": 1.0, "b": 0.0}))
        lib.add(Point(identity="b", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 0.0}))
        all_points = lib.list_all()
        assert len(all_points) == 2

    def test_list_identities(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="a", function_type="linear", params={"a": 1.0, "b": 0.0}))
        lib.add(Point(identity="b", function_type="linear", params={"a": 1.0, "b": 0.0}))
        ids = lib.list_identities()
        assert set(ids) == {"a", "b"}

    def test_list_by_type(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="a", function_type="linear", params={"a": 1.0, "b": 0.0}))
        lib.add(Point(identity="b", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 0.0}))
        linear = lib.list_by_type("linear")
        assert len(linear) == 1
        assert linear[0].identity == "a"

    def test_list_types(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="a", function_type="linear", params={"a": 1.0, "b": 0.0}))
        lib.add(Point(identity="b", function_type="linear", params={"a": 1.0, "b": 0.0}))
        lib.add(Point(identity="c", function_type="periodic", params={"a": 1.0, "b": 0.0, "w": 0.0}))
        types = lib.list_types()
        assert types["linear"] == 2
        assert types["periodic"] == 1

    def test_iter(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="a", function_type="linear", params={"a": 1.0, "b": 0.0}))
        points = list(lib)
        assert len(points) == 1


# ── PointLibrary — search ────────────────────────────────────────────────────

class TestLibrarySearch:
    def test_search_by_identity(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="model.layer1", function_type="linear", params={"a": 1.0, "b": 0.0}))
        lib.add(Point(identity="model.layer2", function_type="linear", params={"a": 1.0, "b": 0.0}))
        lib.add(Point(identity="other.weight", function_type="linear", params={"a": 1.0, "b": 0.0}))
        results = lib.search("layer")
        assert len(results) == 2

    def test_search_case_insensitive(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="Model.Layer1", function_type="linear", params={"a": 1.0, "b": 0.0}))
        results = lib.search("model")
        assert len(results) == 1

    def test_best_points(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="a", function_type="linear", params={"a": 1.0, "b": 0.0}, accuracy=0.5))
        lib.add(Point(identity="b", function_type="linear", params={"a": 1.0, "b": 0.0}, accuracy=0.9))
        lib.add(Point(identity="c", function_type="linear", params={"a": 1.0, "b": 0.0}, accuracy=0.7))
        best = lib.best_points(2)
        assert best[0].accuracy >= best[1].accuracy
        assert best[0].identity == "b"

    def test_worst_points(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="a", function_type="linear", params={"a": 1.0, "b": 0.0}, accuracy=0.5))
        lib.add(Point(identity="b", function_type="linear", params={"a": 1.0, "b": 0.0}, accuracy=0.9))
        worst = lib.worst_points(1)
        assert worst[0].identity == "a"


# ── PointLibrary — compress_and_store ────────────────────────────────────────

class TestLibraryCompressStore:
    def test_compress_and_store_cluster(self):
        lib = PointLibrary("test")
        x = np.random.randn(200).astype(np.float32)
        point = lib.compress_and_store(x, "w", method="cluster")
        assert point.function_type == "cluster"
        assert lib.has("w")

    def test_compress_and_store_function(self):
        lib = PointLibrary("test")
        x = np.arange(50, dtype=np.float32)
        point = lib.compress_and_store(x, "w", method="function")
        assert point.function_type in ("periodic", "linear", "polynomial")


# ── PointLibrary — views ─────────────────────────────────────────────────────

class TestLibraryViews:
    def test_view_returns_point_view(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="w", function_type="linear", params={"a": 1.0, "b": 0.0}))
        view = lib.view("w")
        assert view is not None

    def test_view_missing_returns_none(self):
        lib = PointLibrary("test")
        assert lib.view("missing") is None

    def test_view_caches(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="w", function_type="linear", params={"a": 1.0, "b": 0.0}))
        v1 = lib.view("w")
        v2 = lib.view("w")
        assert v1 is v2

    def test_clear_views(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="w", function_type="linear", params={"a": 1.0, "b": 0.0}))
        lib.view("w")
        count = lib.clear_views()
        assert count == 1
        assert lib.view("w") is not None


# ── PointLibrary — persistence ───────────────────────────────────────────────

class TestLibraryPersistence:
    def test_save_and_load(self, tmp_path):
        lib = PointLibrary("test", storage_dir=tmp_path)
        lib.add(Point(identity="w", function_type="linear",
                      params={"a": 1.0, "b": 2.0}, accuracy=0.95))
        lib.save()
        loaded = PointLibrary.load(tmp_path / "test.points.json")
        assert loaded.get("w") is not None
        assert loaded.get("w").params["a"] == pytest.approx(1.0)

    def test_save_to_explicit_path(self, tmp_path):
        lib = PointLibrary("test")
        lib.add(Point(identity="w", function_type="linear", params={"a": 1.0, "b": 0.0}))
        path = tmp_path / "custom.json"
        lib.save(path)
        assert path.exists()

    def test_load_preserves_name(self, tmp_path):
        lib = PointLibrary("myname", storage_dir=tmp_path)
        lib.add(Point(identity="w", function_type="linear", params={"a": 1.0, "b": 0.0}))
        lib.save()
        loaded = PointLibrary.load(tmp_path / "myname.points.json")
        assert loaded.name == "myname"


# ── PointLibrary — stats ─────────────────────────────────────────────────────

class TestLibraryStats:
    def test_stats_empty(self):
        lib = PointLibrary("test")
        s = lib.stats()
        assert s["total_points"] == 0
        assert s["ratio"] == 0.0

    def test_stats_with_points(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="a", function_type="linear",
                      params={"a": 1.0, "b": 0.0}, accuracy=0.8))
        s = lib.stats()
        assert s["total_points"] == 1
        assert s["avg_accuracy"] == 0.8

    def test_hit_rate_empty(self):
        lib = PointLibrary("test")
        assert lib.hit_rate == 0.0

    def test_hit_rate(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="a", function_type="linear", params={"a": 1.0, "b": 0.0}))
        lib.get("a")
        lib.get("missing")
        assert lib.hit_rate == 0.5


# ── PointLibrary — validation ────────────────────────────────────────────────

class TestLibraryValidation:
    def test_validate_empty_identity_raises(self):
        lib = PointLibrary("test", validate=True)
        with pytest.raises(ValueError, match="identity"):
            lib.add(Point(identity="", function_type="linear", params={"a": 1.0, "b": 0.0}))

    def test_validate_invalid_type_raises(self):
        lib = PointLibrary("test", validate=True)
        with pytest.raises(ValueError, match="function_type"):
            lib.add(Point(identity="w", function_type="bogus", params={}))

    def test_validate_invalid_accuracy_raises(self):
        lib = PointLibrary("test", validate=True)
        with pytest.raises(ValueError, match="Accuracy"):
            lib.add(Point(identity="w", function_type="linear",
                          params={"a": 1.0, "b": 0.0}, accuracy=2.0))

    def test_validate_cluster_missing_params_raises(self):
        lib = PointLibrary("test", validate=True)
        with pytest.raises(ValueError, match="centroids"):
            lib.add(Point(identity="w", function_type="cluster", params={}))

    def test_no_validate(self):
        lib = PointLibrary("test", validate=False)
        p = Point(identity="", function_type="bogus", params={})
        lib.add(p)
        assert lib.has("")


# ── PointDeduplicator ────────────────────────────────────────────────────────

class TestPointDeduplicator:
    def test_no_duplicates(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="a", function_type="linear", params={"a": 1.0, "b": 0.0}))
        lib.add(Point(identity="b", function_type="linear", params={"a": 2.0, "b": 0.0}))
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        groups = dedup.find_duplicates()
        assert len(groups) == 0

    def test_finds_duplicates(self):
        lib = PointLibrary("test")
        centroids = np.array([1.0, 2.0], dtype=np.float32)
        assignments = np.array([0, 1], dtype=np.uint8)
        p1 = Point(identity="a", function_type="cluster",
                   params={"centroids": centroids.copy(), "assignments": assignments.copy()})
        p2 = Point(identity="b", function_type="cluster",
                   params={"centroids": centroids.copy(), "assignments": assignments.copy()})
        lib.add(p1)
        lib.add(p2)
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        groups = dedup.find_duplicates()
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_deduplicate(self):
        lib = PointLibrary("test")
        centroids = np.array([1.0, 2.0], dtype=np.float32)
        assignments = np.array([0, 1], dtype=np.uint8)
        lib.add(Point(identity="a", function_type="cluster",
                      params={"centroids": centroids.copy(), "assignments": assignments.copy()}))
        lib.add(Point(identity="b", function_type="cluster",
                      params={"centroids": centroids.copy(), "assignments": assignments.copy()}))
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        result = dedup.deduplicate()
        assert result["merged"] == 1
        assert len(lib) == 1


# ── PointLibrarySync ─────────────────────────────────────────────────────────

class TestLibrarySync:
    def test_export_and_import(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="w", function_type="linear",
                      params={"a": 1.0, "b": 0.0}))
        sync = PointLibrarySync()
        data = sync.export_bytes(lib)
        loaded = sync.import_bytes(data)
        assert loaded.get("w") is not None

    def test_sync_to_directory(self, tmp_path):
        lib = PointLibrary("test")
        lib.add(Point(identity="w", function_type="linear",
                      params={"a": 1.0, "b": 0.0}))
        sync = PointLibrarySync()
        sync.sync_to_directory(lib, tmp_path)
        assert (tmp_path / "test.points.json").exists()

    def test_sync_from_directory(self, tmp_path):
        lib = PointLibrary("test")
        lib.add(Point(identity="w", function_type="linear",
                      params={"a": 1.0, "b": 0.0}))
        lib.save(tmp_path / "test.points.json")
        sync = PointLibrarySync()
        loaded = sync.sync_from_directory(tmp_path)
        assert loaded is not None
        assert loaded.get("w") is not None

    def test_sync_from_directory_missing_name(self, tmp_path):
        sync = PointLibrarySync()
        loaded = sync.sync_from_directory(tmp_path, name="nonexistent")
        assert loaded is None

    def test_merge(self):
        lib1 = PointLibrary("l1")
        lib1.add(Point(identity="a", function_type="linear",
                       params={"a": 1.0, "b": 0.0}))
        lib2 = PointLibrary("l2")
        lib2.add(Point(identity="b", function_type="linear",
                       params={"a": 2.0, "b": 0.0}))
        sync = PointLibrarySync()
        merged = sync.merge([lib1, lib2])
        assert len(merged) == 2


# ── ModelTree ────────────────────────────────────────────────────────────────

class TestModelTree:
    def test_init_default(self):
        tree = ModelTree("test")
        assert tree.name == "test"
        assert tree.is_loaded is False
        assert len(tree.library) == 0

    def test_init_with_library(self):
        lib = PointLibrary("custom")
        tree = ModelTree("test", library=lib)
        assert tree.library is lib

    def test_load_weights_sequential(self):
        tree = ModelTree("test", n_clusters=8)
        weights = {
            "layer1": np.random.randn(100).astype(np.float32),
            "layer2": np.random.randn(50).astype(np.float32),
        }
        stats = tree.load_weights(weights)
        assert stats["num_weights"] == 2
        assert tree.is_loaded is True

    def test_get_weight(self):
        tree = ModelTree("test", n_clusters=8)
        weights = {"layer1": np.random.randn(100).astype(np.float32)}
        tree.load_weights(weights)
        result = tree.get_weight("layer1")
        assert result is not None
        assert result.shape == (100,)

    def test_get_weight_missing(self):
        tree = ModelTree("test")
        assert tree.get_weight("nonexistent") is None

    def test_get_weights(self):
        tree = ModelTree("test", n_clusters=8)
        weights = {
            "a": np.random.randn(50).astype(np.float32),
            "b": np.random.randn(50).astype(np.float32),
        }
        tree.load_weights(weights)
        results = tree.get_weights()
        assert "a" in results
        assert "b" in results

    def test_stats(self):
        tree = ModelTree("test", n_clusters=8)
        tree.load_weights({"w": np.ones(100, dtype=np.float32)})
        s = tree.stats()
        assert s["model"] == "test"
        assert s["loaded"] is True
        assert s["num_weights"] == 1

    def test_load_weights_function_method(self):
        tree = ModelTree("test", n_clusters=8)
        tree._method = "function"
        weights = {"w": np.arange(100, dtype=np.float32)}
        stats = tree.load_weights(weights, method="function")
        assert stats["method"] == "function"

    def test_skip_embeddings(self):
        tree = ModelTree("test", n_clusters=8, config=None)
        weights = {"embed_tokens": np.random.randn(100).astype(np.float32)}
        tree.load_weights(weights)
        point = tree.library.get("test.embed_tokens")
        assert point.function_type == "raw"

    def test_skip_biases(self):
        tree = ModelTree("test", n_clusters=8, config=None)
        weights = {"layer.bias": np.random.randn(10).astype(np.float32)}
        tree.load_weights(weights)
        point = tree.library.get("test.layer.bias")
        assert point.function_type == "raw"


# ── ModelTree extended ──────────────────────────────────────────────────────

class TestModelTreeExtended:
    def test_init_with_config(self):
        from domains.infrastructure.pugqeep.config import TreeConfig
        cfg = TreeConfig(name="custom", n_clusters=32, method="function",
                         skip_embeddings=False, skip_biases=False)
        tree = ModelTree("test", config=cfg)
        assert tree.n_clusters == 32
        assert tree._method == "function"
        assert tree._skip_embeddings is False
        assert tree._skip_biases is False

    def test_init_with_compressor(self):
        comp = PointCompressor(n_clusters=32)
        tree = ModelTree("test", compressor=comp)
        assert tree._compressor is comp

    def test_load_weights_with_progress(self):
        tree = ModelTree("test", n_clusters=8)
        progress_calls = []
        weights = {"a": np.random.randn(50).astype(np.float32)}
        tree.load_weights(weights, on_progress=lambda c, t, n: progress_calls.append((c, t, n)))
        assert len(progress_calls) == 1
        assert progress_calls[0] == (1, 1, "a")

    def test_load_weights_function_method(self):
        tree = ModelTree("test", n_clusters=8)
        weights = {"w": np.arange(100, dtype=np.float32)}
        stats = tree.load_weights(weights, method="function")
        assert stats["method"] == "function"
        point = tree.library.get("test.w")
        assert point.function_type in ("periodic", "linear", "polynomial")

    def test_load_weights_small_cluster_fallback(self):
        tree = ModelTree("test", n_clusters=100)
        weights = {"w": np.array([1.0], dtype=np.float32)}
        stats = tree.load_weights(weights, method="cluster")
        point = tree.library.get("test.w")
        assert point.function_type == "raw"

    def test_get_weight_with_shape(self):
        tree = ModelTree("test", n_clusters=8)
        weights = {"w": np.random.randn(20).astype(np.float32)}
        tree.load_weights(weights)
        result = tree.get_weight("w")
        assert result is not None
        assert result.shape == (20,)

    def test_get_weight_raw(self):
        tree = ModelTree("test", n_clusters=8, config=None)
        weights = {"w": np.random.randn(5).astype(np.float32)}
        tree.load_weights(weights)
        result = tree.get_weight("w")
        assert result is not None

    def test_get_weights_subset(self):
        tree = ModelTree("test", n_clusters=8)
        weights = {"a": np.ones(20, dtype=np.float32), "b": np.ones(20, dtype=np.float32)}
        tree.load_weights(weights)
        result = tree.get_weights(["a"])
        assert "a" in result
        assert "b" not in result

    def test_estimate_size_from_shape(self):
        tree = ModelTree("test", n_clusters=8)
        weights = {"w": np.ones(50, dtype=np.float32)}
        tree.load_weights(weights)
        size = tree._estimate_size("w")
        assert size == 50

    def test_estimate_size_from_point(self):
        tree = ModelTree("test", n_clusters=8)
        weights = {"w": np.ones(100, dtype=np.float32)}
        tree.load_weights(weights)
        size = tree._estimate_size("w")
        assert size == 100

    def test_is_loaded_property(self):
        tree = ModelTree("test")
        assert tree.is_loaded is False
        tree.is_loaded = True
        assert tree.is_loaded is True

    def test_stats_before_load(self):
        tree = ModelTree("test")
        s = tree.stats()
        assert s["model"] == "test"
        assert s["loaded"] is False
        assert s["num_weights"] == 0

    def test_stats_after_load(self):
        tree = ModelTree("test")
        tree.load_weights({"w": np.ones(10, dtype=np.float32)})
        s = tree.stats()
        assert s["loaded"] is True
        assert s["num_weights"] == 1


# ── load_model_to_points ────────────────────────────────────────────────────

class TestLoadModelToPoints:
    def test_raises_when_model_not_cached(self):
        from domains.infrastructure.pugqeep.model_tree import load_model_to_points
        with pytest.raises(FileNotFoundError, match="not cached"):
            load_model_to_points("some_nonexistent_model_xyz")


# ── load_from_points ────────────────────────────────────────────────────────

class TestLoadFromPoints:
    def test_load_from_points_basic(self, tmp_path):
        from domains.infrastructure.pugqeep.model_tree import load_from_points
        lib = PointLibrary("test_model")
        lib.add(Point(identity="test_model.w", function_type="linear",
                      params={"a": 1.0, "b": 0.0}))
        lib.save(tmp_path / "test_model.points.json")
        tree, meta = load_from_points(str(tmp_path / "test_model"))
        assert tree.name == "test_model"
        assert tree.is_loaded is True

    def test_load_from_points_not_found(self, tmp_path):
        from domains.infrastructure.pugqeep.model_tree import load_from_points
        with pytest.raises(FileNotFoundError):
            load_from_points(str(tmp_path / "nonexistent"))

    def test_load_from_points_with_meta(self, tmp_path):
        import json
        from domains.infrastructure.pugqeep.model_tree import load_from_points
        lib = PointLibrary("test_model")
        lib.add(Point(identity="test_model.w", function_type="linear",
                      params={"a": 1.0, "b": 0.0}))
        lib.save(tmp_path / "test_model.points.json")
        meta = {"metadata": {"weight_shapes": {"w": [10]}}}
        (tmp_path / "test_model.meta.json").write_text(json.dumps(meta))
        tree, meta_dict = load_from_points(str(tmp_path / "test_model"))
        assert tree._weight_shapes.get("w") == (10,)

    def test_load_from_points_no_prefix_match(self, tmp_path):
        from domains.infrastructure.pugqeep.model_tree import load_from_points
        lib = PointLibrary("test_model")
        lib.add(Point(identity="other.w", function_type="linear",
                      params={"a": 1.0, "b": 0.0}))
        lib.save(tmp_path / "test_model.points.json")
        tree, meta = load_from_points(str(tmp_path / "test_model"))
        assert tree.is_loaded is True


# ── decompress_tree ─────────────────────────────────────────────────────────

class TestDecompressTree:
    def test_decompress_sequential(self):
        from domains.infrastructure.pugqeep.model_tree import decompress_tree
        tree = ModelTree("test", n_clusters=8)
        tree.load_weights({"a": np.ones(50, dtype=np.float32),
                           "b": np.ones(50, dtype=np.float32)})
        result = decompress_tree(tree, num_workers=0)
        assert "a" in result
        assert "b" in result
        assert result["a"].shape == (50,)

    def test_decompress_tree_empty(self):
        from domains.infrastructure.pugqeep.model_tree import decompress_tree
        tree = ModelTree("test")
        result = decompress_tree(tree)
        assert len(result) == 0

    def test_decompress_with_raw_point(self):
        from domains.infrastructure.pugqeep.model_tree import decompress_tree
        tree = ModelTree("test", n_clusters=8, config=None)
        tree.load_weights({"w": np.array([1.0, 2.0, 3.0], dtype=np.float32)})
        result = decompress_tree(tree)
        assert "w" in result


# ── Point extended ──────────────────────────────────────────────────────────

class TestPointExtended:
    def test_periodic_with_different_params(self):
        p = Point(identity="w", function_type="periodic",
                  params={"a": 2.0, "b": 3.0, "w": 1.0})
        result = p.generate(5)
        i = np.arange(5, dtype=np.float32)
        expected = 2.0 * np.cos(i) + 3.0 * np.sin(i) + 1.0
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_polynomial_with_all_params(self):
        p = Point(identity="w", function_type="polynomial",
                  params={"a": 1.0, "b": -2.0, "c": 3.0})
        result = p.generate(5)
        i = np.arange(5, dtype=np.float32)
        expected = i**2 - 2.0 * i + 3.0
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_linear_with_zero_slope(self):
        p = Point(identity="w", function_type="linear",
                  params={"a": 0.0, "b": 5.0})
        result = p.generate(5)
        np.testing.assert_allclose(result, [5.0, 5.0, 5.0, 5.0, 5.0])

    def test_nbytes_raw_empty(self):
        p = Point(identity="w", function_type="raw",
                  params={"data_b64": ""})
        # Empty base64 decodes to empty bytes
        assert p.nbytes() == 0

    def test_nbytes_periodic_with_residual(self):
        residual = np.ones(10, dtype=np.float32)
        p = Point(identity="w", function_type="periodic",
                  params={"a": 1.0, "b": 0.0, "w": 0.0},
                  residual=residual)
        assert p.nbytes() == 4 + 3 * 4 + residual.nbytes

    def test_estimate_raw_bytes_periodic_with_shape(self):
        p = Point(identity="w", function_type="periodic",
                  params={"a": 1.0, "b": 0.0, "w": 0.0},
                  shape=(20,))
        assert p._estimate_raw_bytes() == 20 * 4

    def test_to_dict_and_back_periodic(self):
        p = Point(identity="w", function_type="periodic",
                  params={"a": 1.5, "b": 2.5, "w": 3.5}, accuracy=0.9)
        d = p.to_dict()
        p2 = Point.from_dict(d)
        assert p2.accuracy == pytest.approx(0.9)

    def test_to_dict_raw(self):
        raw = np.array([1.0, 2.0], dtype=np.float32)
        data_b64 = base64.b64encode(raw.tobytes()).decode()
        p = Point(identity="w", function_type="raw",
                  params={"data_b64": data_b64, "dtype": "float32", "shape": []})
        d = p.to_dict()
        assert d["function_type"] == "raw"

    def test_to_dict_linear(self):
        p = Point(identity="w", function_type="linear",
                  params={"a": 1.0, "b": 2.0})
        d = p.to_dict()
        assert d["params"]["a"] == 1.0

    def test_to_dict_polynomial(self):
        p = Point(identity="w", function_type="polynomial",
                  params={"a": 1.0, "b": 2.0, "c": 3.0})
        d = p.to_dict()
        assert d["params"]["c"] == 3.0

    def test_from_dict_raw(self):
        raw = np.array([1.0, 2.0], dtype=np.float32)
        d = {
            "identity": "w",
            "function_type": "raw",
            "params": {
                "data_b64": base64.b64encode(raw.tobytes()).decode(),
                "dtype": "float32",
                "shape": [],
            },
            "accuracy": 1.0,
            "dtype": "float32",
            "shape": [],
        }
        p = Point.from_dict(d)
        assert p.function_type == "raw"

    def test_from_dict_residual(self):
        centroids = np.array([1.0, 2.0], dtype=np.float32)
        assignments = np.array([0, 1], dtype=np.uint8)
        residual = np.array([0.1, -0.1], dtype=np.float32)
        p = Point(identity="w", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments},
                  residual=residual)
        d = p.to_dict()
        p2 = Point.from_dict(d)
        assert p2.residual is not None
        np.testing.assert_allclose(p2.residual, residual)


# ── PointCompressor extended ────────────────────────────────────────────────

class TestCompressorExtended:
    def test_compress_with_method_override(self):
        comp = PointCompressor()
        x = np.arange(50, dtype=np.float32)
        point = comp.compress(x, "w", method="function")
        assert point.function_type in ("periodic", "linear", "polynomial")

    def test_compress_function_periodic_fit(self):
        comp = PointCompressor()
        t = np.arange(100, dtype=np.float32)
        x = 2.0 * np.cos(t) + 1.0 * np.sin(t) + 0.5
        point = comp.compress_function(x, "w")
        assert point.function_type == "periodic"
        assert point.accuracy > 0.99

    def test_compress_function_polynomial_fit(self):
        comp = PointCompressor()
        t = np.arange(100, dtype=np.float32)
        x = 0.01 * t**2 + 0.1 * t + 1.0
        point = comp.compress_function(x, "w")
        assert point.function_type == "polynomial"
        assert point.accuracy > 0.99

    def test_compress_single_element(self):
        comp = PointCompressor()
        x = np.array([42.0], dtype=np.float32)
        point = comp.compress_cluster(x, "w")
        assert point.function_type == "cluster"

    def test_compress_very_large_array(self):
        comp = PointCompressor(n_clusters=16)
        x = np.random.randn(10000).astype(np.float32)
        point = comp.compress_cluster(x, "w")
        assert point.function_type == "cluster"
        assert point.accuracy > 0.8

    def test_measure_compression_function_no_residual(self):
        comp = PointCompressor(residual_threshold=0.01)
        x = np.arange(50, dtype=np.float32)
        point = comp.compress_function(x, "w")
        m = comp.measure_compression(x, point)
        assert m["raw_bytes"] > 0
        assert m["compressed_bytes"] > 0
        assert m["ratio"] > 0

    def test_decompress_roundtrip_cluster(self):
        comp = PointCompressor()
        x = np.random.randn(200).astype(np.float32)
        point = comp.compress_cluster(x, "w")
        result = comp.decompress(point, len(x))
        assert result.shape == x.shape

    def test_compress_batch_with_method(self):
        comp = PointCompressor()
        weights = {"w1": np.arange(50, dtype=np.float32),
                   "w2": np.arange(50, dtype=np.float32)}
        result = comp.compress_batch(weights, method="function")
        assert result["w1"].function_type in ("periodic", "linear", "polynomial")
        assert result["w2"].function_type in ("periodic", "linear", "polynomial")


# ── PointLibrary extended ───────────────────────────────────────────────────

class TestLibraryExtended:
    def test_context_manager(self, tmp_path):
        with PointLibrary("test", storage_dir=tmp_path) as lib:
            lib.add(Point(identity="w", function_type="linear",
                          params={"a": 1.0, "b": 0.0}))
            assert len(lib) == 1

    def test_repr(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="w", function_type="linear",
                      params={"a": 1.0, "b": 0.0}))
        r = repr(lib)
        assert "test" in r
        assert "1" in r

    def test_search_no_match(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="model.w", function_type="linear",
                      params={"a": 1.0, "b": 0.0}))
        results = lib.search("xyz")
        assert len(results) == 0

    def test_list_by_type_empty(self):
        lib = PointLibrary("test")
        results = lib.list_by_type("cluster")
        assert len(results) == 0

    def test_list_identities_empty(self):
        lib = PointLibrary("test")
        assert lib.list_identities() == []

    def test_list_types_empty(self):
        lib = PointLibrary("test")
        assert lib.list_types() == {}

    def test_iter_empty(self):
        lib = PointLibrary("test")
        assert list(lib) == []

    def test_best_points_empty(self):
        lib = PointLibrary("test")
        assert lib.best_points(5) == []

    def test_worst_points_empty(self):
        lib = PointLibrary("test")
        assert lib.worst_points(5) == []

    def test_stats_detailed(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="a", function_type="linear",
                      params={"a": 1.0, "b": 0.0}, accuracy=0.8))
        lib.add(Point(identity="b", function_type="periodic",
                      params={"a": 1.0, "b": 0.0, "w": 0.0}, accuracy=0.9))
        s = lib.stats()
        assert s["total_points"] == 2
        assert s["avg_accuracy"] == pytest.approx(0.85)
        assert "linear" in s["types"]
        assert "periodic" in s["types"]

    def test_auto_save_on_add(self, tmp_path):
        lib = PointLibrary("test", storage_dir=tmp_path,
                           config=LibraryConfig(name="test", storage_dir=tmp_path, auto_save=True))
        lib.add(Point(identity="w", function_type="linear",
                      params={"a": 1.0, "b": 0.0}))
        assert (tmp_path / "test.points.json").exists()

    def test_add_many_with_validation(self):
        lib = PointLibrary("test", validate=True)
        points = [
            Point(identity=f"w{i}", function_type="linear",
                  params={"a": float(i), "b": 0.0})
            for i in range(3)
        ]
        count = lib.add_many(points)
        assert count == 3

    def test_get_many_empty(self):
        lib = PointLibrary("test")
        result = lib.get_many([])
        assert result == {}

    def test_exists_many_empty(self):
        lib = PointLibrary("test")
        result = lib.exists_many([])
        assert result == {}

    def test_views_batch(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="a", function_type="linear",
                      params={"a": 1.0, "b": 0.0}))
        lib.add(Point(identity="b", function_type="linear",
                      params={"a": 2.0, "b": 0.0}))
        views = lib.views(["a", "b", "c"])
        assert views["a"] is not None
        assert views["b"] is not None
        assert views["c"] is None


# ── PointLibrarySync extended ───────────────────────────────────────────────

class TestLibrarySyncExtended:
    def test_export_empty_library(self):
        lib = PointLibrary("empty")
        sync = PointLibrarySync()
        data = sync.export_bytes(lib)
        loaded = sync.import_bytes(data)
        assert len(loaded) == 0

    def test_sync_from_directory_all(self, tmp_path):
        lib = PointLibrary("test")
        lib.add(Point(identity="w", function_type="linear",
                      params={"a": 1.0, "b": 0.0}))
        sync = PointLibrarySync()
        sync.sync_to_directory(lib, tmp_path)
        loaded = sync.sync_from_directory(tmp_path)
        assert loaded is not None
        assert loaded.get("w") is not None

    def test_merge_with_overlapping(self):
        lib1 = PointLibrary("l1")
        lib1.add(Point(identity="a", function_type="linear",
                       params={"a": 1.0, "b": 0.0}))
        lib2 = PointLibrary("l2")
        lib2.add(Point(identity="a", function_type="linear",
                       params={"a": 2.0, "b": 0.0}))
        sync = PointLibrarySync()
        merged = sync.merge([lib1, lib2])
        # Both have same identity, so merged has 1 after dedup
        assert len(merged) == 1


# ── PointDeduplicator extended ──────────────────────────────────────────────

class TestDeduplicatorExtended:
    def test_fingerprint_raw(self):
        raw = np.array([1.0, 2.0], dtype=np.float32)
        p1 = Point(identity="a", function_type="raw",
                   params={"data_b64": base64.b64encode(raw.tobytes()).decode()})
        p2 = Point(identity="b", function_type="raw",
                   params={"data_b64": base64.b64encode(raw.tobytes()).decode()})
        lib = PointLibrary("test")
        lib.add(p1)
        lib.add(p2)
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        groups = dedup.find_duplicates()
        assert len(groups) == 1

    def test_fingerprint_function(self):
        p1 = Point(identity="a", function_type="linear",
                   params={"a": 1.0, "b": 2.0})
        p2 = Point(identity="b", function_type="linear",
                   params={"a": 1.0, "b": 2.0})
        lib = PointLibrary("test")
        lib.add(p1)
        lib.add(p2)
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        groups = dedup.find_duplicates()
        assert len(groups) == 1

    def test_no_duplicates_function(self):
        lib = PointLibrary("test")
        lib.add(Point(identity="a", function_type="linear",
                      params={"a": 1.0, "b": 0.0}))
        lib.add(Point(identity="b", function_type="linear",
                      params={"a": 2.0, "b": 0.0}))
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        groups = dedup.find_duplicates()
        assert len(groups) == 0

    def test_deduplicate_saves_bytes(self):
        lib = PointLibrary("test")
        centroids = np.array([1.0, 2.0], dtype=np.float32)
        assignments = np.array([0, 1], dtype=np.uint8)
        lib.add(Point(identity="a", function_type="cluster",
                      params={"centroids": centroids.copy(), "assignments": assignments.copy()}))
        lib.add(Point(identity="b", function_type="cluster",
                      params={"centroids": centroids.copy(), "assignments": assignments.copy()}))
        dedup = PointDeduplicator()
        dedup.add_library(lib)
        result = dedup.deduplicate()
        assert result["bytes_saved"] > 0


# ── Config extended ─────────────────────────────────────────────────────────

class TestConfigExtended:
    def test_point_config_defaults(self):
        from domains.infrastructure.pugqeep.config import PointConfig
        cfg = PointConfig()
        assert cfg.function_type == "cluster"
        assert cfg.n_clusters == 16
        assert cfg.residual_threshold == 0.99

    def test_compressor_config_custom(self):
        from domains.infrastructure.pugqeep.config import CompressorConfig
        cfg = CompressorConfig(n_clusters=64, lloyd_iterations=20,
                               gap_fill_iterations=8, gap_fill_max_elements=50_000)
        assert cfg.n_clusters == 64
        assert cfg.lloyd_iterations == 20

    def test_library_config_defaults(self):
        from domains.infrastructure.pugqeep.config import LibraryConfig
        cfg = LibraryConfig()
        assert cfg.name == "default"
        assert cfg.auto_save is False

    def test_tree_config_custom(self):
        from domains.infrastructure.pugqeep.config import TreeConfig
        cfg = TreeConfig(n_clusters=32, method="function",
                         skip_embeddings=False, skip_biases=False)
        assert cfg.n_clusters == 32
        assert cfg.method == "function"

    def test_queue_config_defaults(self):
        from domains.infrastructure.pugqeep.config import QueueConfig
        cfg = QueueConfig()
        assert cfg.max_trees == 10
        assert cfg.dedup is True

    def test_engine_config_defaults(self):
        from domains.infrastructure.pugqeep.config import EngineConfig
        cfg = EngineConfig()
        assert cfg.name == "main"
        assert cfg.max_trees == 16
