"""Tests for compression — CompressedWeight VQ storage + LRUCache."""

import numpy as np
import pytest

from domains.infrastructure.compression import CompressedWeight, LRUCache


def _make_weight(shape=(2, 3), seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


class TestCompressedWeight:
    def test_decompress_with_residual_exact(self):
        weight = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        centroids = np.arange(6, dtype=np.float32)
        assignments = np.array([0, 1, 2, 3, 4, 5])
        residual = (weight - centroids[assignments].reshape(2, 3)).ravel().astype(np.float16)
        comp = CompressedWeight(centroids, assignments, residual, (2, 3), np.float32)
        np.testing.assert_allclose(comp.decompress(), weight, atol=1e-3)

    def test_decompress_without_residual_approx(self):
        weight = _make_weight()
        centroids = np.array([0.0, 1.0], dtype=np.float32)
        assignments = np.zeros(weight.size, dtype=np.int32)
        comp = CompressedWeight(centroids, assignments, None, weight.shape, np.float32)
        reconstructed = comp.decompress()
        assert reconstructed.shape == weight.shape
        np.testing.assert_array_equal(reconstructed, np.zeros_like(weight))

    def test_decompress_linear_centroids(self):
        centroids = np.zeros(4, dtype=np.float32)
        assignments = np.array([0, 1, 2, 3])
        residual = np.zeros(4, dtype=np.float16)
        comp = CompressedWeight(
            centroids, assignments, residual, (4,), np.float32,
            centroid_fn="linear", centroid_fn_params={"a": 2.0, "b": 1.0},
        )
        np.testing.assert_array_equal(comp.decompress(), np.array([1.0, 3.0, 5.0, 7.0]))

    def test_decompress_preserves_dtype(self):
        weight = _make_weight()
        centroids = np.array([0.0, 1.0], dtype=np.float32)
        assignments = np.zeros(weight.size, dtype=np.int32)
        comp = CompressedWeight(centroids, assignments, None, weight.shape, np.float32)
        assert comp.decompress().dtype == np.float32

    def test_compressed_bytes_raw(self):
        centroids = np.zeros(8, dtype=np.float32)
        assignments = np.zeros(16, dtype=np.int32)
        residual = np.zeros(16, dtype=np.float16)
        comp = CompressedWeight(centroids, assignments, residual, (4, 4), np.float32)
        assert comp.compressed_bytes == 8 * 4 + 16 * 4 + 16 * 2

    def test_compressed_bytes_linear(self):
        centroids = np.zeros(8, dtype=np.float32)
        assignments = np.zeros(16, dtype=np.int32)
        comp = CompressedWeight(
            centroids, assignments, None, (4, 4), np.float32,
            centroid_fn="linear", centroid_fn_params={"a": 1.0, "b": 0.0},
        )
        assert comp.compressed_bytes == 8 + 16 * 4

    def test_compressed_bytes_no_residual(self):
        centroids = np.zeros(8, dtype=np.float32)
        assignments = np.zeros(16, dtype=np.int32)
        comp = CompressedWeight(centroids, assignments, None, (4, 4), np.float32)
        assert comp.compressed_bytes == 8 * 4 + 16 * 4


class TestLRUCache:
    def test_get_missing_returns_none(self):
        cache = LRUCache(max_size=2)
        assert cache.get("nope") is None

    def test_put_get_roundtrip(self):
        cache = LRUCache(max_size=2)
        arr = np.array([1.0, 2.0])
        cache.put("a", arr)
        assert cache.get("a") is arr

    def test_evicts_oldest(self):
        cache = LRUCache(max_size=2)
        cache.put("a", np.array([1]))
        cache.put("b", np.array([2]))
        cache.put("c", np.array([3]))
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_get_refreshes_recency(self):
        cache = LRUCache(max_size=2)
        cache.put("a", np.array([1]))
        cache.put("b", np.array([2]))
        cache.get("a")
        cache.put("c", np.array([3]))
        assert cache.get("a") is not None
        assert cache.get("b") is None

    def test_put_existing_moves_to_end(self):
        cache = LRUCache(max_size=2)
        cache.put("a", np.array([1]))
        cache.put("b", np.array([2]))
        cache.put("a", np.array([9]))
        cache.put("c", np.array([3]))
        assert cache.get("a") is not None
        assert cache.get("b") is None
        np.testing.assert_array_equal(cache.get("a"), np.array([9]))
