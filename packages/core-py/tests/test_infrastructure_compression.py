"""Tests for CompressedWeight and LRUCache — weight compression utilities."""
from __future__ import annotations

import numpy as np
import pytest

from domains.infrastructure.compression import CompressedWeight, LRUCache


class TestCompressedWeight:
    def test_decompress_basic(self):
        centroids = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        assignments = np.array([0, 1, 2, 0, 1], dtype=np.int32)
        residual = np.zeros(5, dtype=np.float16)
        cw = CompressedWeight(centroids, assignments, residual, (5,), np.float32)
        result = cw.decompress()
        assert result.shape == (5,)
        np.testing.assert_allclose(result, [0.0, 1.0, 2.0, 0.0, 1.0], atol=1e-6)

    def test_decompress_with_residual(self):
        centroids = np.array([0.0, 1.0], dtype=np.float32)
        assignments = np.array([0, 1], dtype=np.int32)
        residual = np.array([0.1, 0.2], dtype=np.float16)
        cw = CompressedWeight(centroids, assignments, residual, (2,), np.float32)
        result = cw.decompress()
        np.testing.assert_allclose(result, [0.1, 1.2], atol=1e-4)

    def test_decompress_no_residual(self):
        centroids = np.array([5.0], dtype=np.float32)
        assignments = np.array([0, 0, 0], dtype=np.int32)
        cw = CompressedWeight(centroids, assignments, None, (3,), np.float32)
        result = cw.decompress()
        np.testing.assert_allclose(result, [5.0, 5.0, 5.0], atol=1e-6)

    def test_decompress_linear_centroids(self):
        centroids = np.zeros(3, dtype=np.float32)  # placeholder
        assignments = np.array([0, 1, 2], dtype=np.int32)
        cw = CompressedWeight(
            centroids, assignments, None, (3,), np.float32,
            centroid_fn="linear", centroid_fn_params={"a": 2.0, "b": 1.0},
        )
        result = cw.decompress()
        # linear: a*i + b = [1, 3, 5]
        np.testing.assert_allclose(result, [1.0, 3.0, 5.0], atol=1e-6)

    def test_compressed_bytes_raw(self):
        centroids = np.zeros(4, dtype=np.float32)
        assignments = np.zeros(4, dtype=np.int32)
        cw = CompressedWeight(centroids, assignments, None, (4,), np.float32)
        assert cw.compressed_bytes == centroids.nbytes + assignments.nbytes

    def test_compressed_bytes_linear(self):
        centroids = np.zeros(100, dtype=np.float32)
        assignments = np.zeros(4, dtype=np.int32)
        cw = CompressedWeight(
            centroids, assignments, None, (4,), np.float32,
            centroid_fn="linear", centroid_fn_params={"a": 1.0, "b": 0.0},
        )
        # linear = 8 bytes (2 float32) + assignments
        assert cw.compressed_bytes == 8 + assignments.nbytes

    def test_compressed_bytes_with_residual(self):
        centroids = np.zeros(2, dtype=np.float32)
        assignments = np.zeros(2, dtype=np.int32)
        residual = np.zeros(2, dtype=np.float16)
        cw = CompressedWeight(centroids, assignments, residual, (2,), np.float32)
        assert cw.compressed_bytes == centroids.nbytes + assignments.nbytes + residual.nbytes


class TestLRUCache:
    def test_put_and_get(self):
        cache = LRUCache(max_size=3)
        arr = np.array([1.0, 2.0])
        cache.put("k1", arr)
        result = cache.get("k1")
        assert result is not None
        np.testing.assert_array_equal(result, arr)

    def test_get_miss(self):
        cache = LRUCache()
        assert cache.get("missing") is None

    def test_evicts_oldest(self):
        cache = LRUCache(max_size=2)
        cache.put("a", np.array([1]))
        cache.put("b", np.array([2]))
        cache.put("c", np.array([3]))  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_update_moves_to_end(self):
        cache = LRUCache(max_size=2)
        cache.put("a", np.array([1]))
        cache.put("b", np.array([2]))
        cache.get("a")  # access "a" → moves to end
        cache.put("c", np.array([3]))  # should evict "b"
        assert cache.get("a") is not None
        assert cache.get("b") is None

    def test_put_update_existing(self):
        cache = LRUCache(max_size=2)
        cache.put("k", np.array([1]))
        cache.put("k", np.array([2]))
        result = cache.get("k")
        assert result is not None
        np.testing.assert_array_equal(result, np.array([2]))
