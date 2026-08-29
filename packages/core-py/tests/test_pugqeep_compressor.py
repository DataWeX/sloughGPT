"""Tests for domains.infrastructure.pugqeep.compressor — PointCompressor."""

import numpy as np
import pytest
from domains.infrastructure.pugqeep.compressor import PointCompressor
from domains.infrastructure.pugqeep.point import Point


class TestPointCompressorCluster:
    def test_compress_returns_point(self):
        c = PointCompressor()
        weights = np.random.randn(128)
        p = c.compress_cluster(weights, identity="layer1")
        assert isinstance(p, Point)
        assert p.identity == "layer1"
        assert p.function_type == "cluster"

    def test_accuracy_between_0_and_1(self):
        c = PointCompressor()
        weights = np.random.randn(256)
        p = c.compress_cluster(weights)
        assert 0.0 <= p.accuracy <= 1.0

    def test_centroids_match_n_clusters(self):
        n_clusters = 8
        c = PointCompressor(n_clusters=n_clusters)
        weights = np.random.randn(128)
        p = c.compress_cluster(weights)
        assert len(p.params["centroids"]) >= n_clusters

    def test_assignments_match_weights_length(self):
        c = PointCompressor()
        weights = np.random.randn(100)
        p = c.compress_cluster(weights)
        assert len(p.params["assignments"]) == 100

    def test_custom_n_clusters_override(self):
        c = PointCompressor(n_clusters=16)
        weights = np.random.randn(200)
        p = c.compress_cluster(weights, n_clusters=32)
        assert len(p.params["centroids"]) >= 32


class TestPointCompressorFunction:
    def test_compress_periodic(self):
        c = PointCompressor()
        i = np.arange(50, dtype=np.float32)
        weights = 2.0 * np.cos(i) + 0.5 * np.sin(i) + 1.0
        p = c.compress_function(weights, identity="periodic_w")
        assert isinstance(p, Point)
        assert p.identity == "periodic_w"

    def test_compress_linear(self):
        c = PointCompressor()
        weights = np.arange(50, dtype=np.float32) * 0.1
        p = c.compress_function(weights, identity="linear_w")
        assert isinstance(p, Point)

    def test_compress_random(self):
        c = PointCompressor(residual_threshold=0.99)
        weights = np.random.randn(200)
        p = c.compress_function(weights, identity="random")
        assert isinstance(p, Point)
        assert p.accuracy >= 0.0


class TestPointCompressorGeneral:
    def test_compress_cluster_via_compress(self):
        c = PointCompressor(n_clusters=8)
        weights = np.random.randn(128)
        p = c.compress(weights, method="cluster")
        assert p.function_type == "cluster"

    def test_compress_function_via_compress(self):
        c = PointCompressor(n_clusters=8)
        weights = np.random.randn(128)
        p = c.compress(weights, method="function")
        assert p.function_type in ("periodic", "linear", "polynomial")

    def test_compress_unknown_method_raises(self):
        c = PointCompressor()
        with pytest.raises(ValueError, match="Unknown method"):
            c.compress(np.random.randn(32), method="invalid")

    def test_decompress_cluster_roundtrip(self):
        c = PointCompressor()
        weights = np.random.randn(128)
        p = c.compress_cluster(weights)
        recovered = c.decompress(p, n=128)
        assert len(recovered) == 128

    def test_measure_compression(self):
        c = PointCompressor()
        weights = np.random.randn(128)
        p = c.compress_cluster(weights)
        m = c.measure_compression(weights, p)
        assert m["raw_bytes"] == weights.nbytes
        assert m["compressed_bytes"] > 0
        assert m["ratio"] > 0
        assert m["accuracy"] > 0
        assert m["function_type"] == "cluster"
