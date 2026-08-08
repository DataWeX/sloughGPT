"""Tests for point_compressor — backward-compatible shim over pugqeep."""

import numpy as np
import pytest

import domains.infrastructure.pugqeep as pugqeep
from domains.infrastructure import point_compressor as pc


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


class TestCompressorCoverage:
    def test_compress_default_method(self):
        comp = pugqeep.PointCompressor()
        point = comp.compress(np.random.randn(100).astype(np.float32), "w")
        assert point.function_type == "cluster"

    def test_compress_unknown_method_raises(self):
        comp = pugqeep.PointCompressor()
        with pytest.raises(ValueError):
            comp.compress(np.zeros(10, dtype=np.float32), "w", method="bogus")

    def test_measure_compression_cluster_with_residual(self):
        comp = pugqeep.PointCompressor()
        centroids = np.zeros(4, dtype=np.float32)
        assignments = np.zeros(8, dtype=np.uint8)
        residual = np.zeros(8, dtype=np.float32)
        point = pugqeep.Point(
            identity="w", function_type="cluster",
            params={"centroids": centroids, "assignments": assignments},
            residual=residual, accuracy=0.99,
        )
        m = comp.measure_compression(np.zeros(8, dtype=np.float32), point)
        assert m["compressed_bytes"] == centroids.nbytes + assignments.nbytes + residual.nbytes

    def test_measure_compression_raw(self):
        import base64
        comp = pugqeep.PointCompressor()
        raw = np.zeros(8, dtype=np.float32)
        point = pugqeep.Point(
            identity="w", function_type="raw",
            params={"data_b64": base64.b64encode(raw.tobytes()).decode()},
        )
        m = comp.measure_compression(raw, point)
        assert m["compressed_bytes"] == raw.nbytes

    def test_measure_compression_function_with_residual(self):
        comp = pugqeep.PointCompressor()
        residual = np.zeros(3, dtype=np.float32)
        point = pugqeep.Point(
            identity="w", function_type="periodic",
            params={"a": 1.0, "b": 0.0, "w": 0.0},
            residual=residual, accuracy=0.9,
        )
        m = comp.measure_compression(np.zeros(10, dtype=np.float32), point)
        assert m["compressed_bytes"] == 4 + 3 * 4 + residual.nbytes
