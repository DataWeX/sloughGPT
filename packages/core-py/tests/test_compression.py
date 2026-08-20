"""Tests for domains.infrastructure.compression — CompressedWeight, LRUCache."""

import numpy as np
from domains.infrastructure.compression import CompressedWeight, LRUCache


class TestCompressedWeight:
    def test_decompress(self):
        centroids = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assignments = np.array([0, 1, 2, 0, 1], dtype=np.int32)
        residual = np.zeros(5, dtype=np.float16)
        shape = (5,)
        cw = CompressedWeight(centroids, assignments, residual, shape, np.dtype("float32"))
        result = cw.decompress()
        assert result.shape == shape
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0, 1.0, 2.0])

    def test_decompress_linear(self):
        centroids = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        assignments = np.array([0, 1, 2], dtype=np.int32)
        shape = (3,)
        cw = CompressedWeight(
            centroids, assignments, None, shape, np.dtype("float32"),
            centroid_fn="linear", centroid_fn_params={"a": 1.0, "b": 0.0},
        )
        result = cw.decompress()
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 2.0])


class TestLRUCache:
    def test_get_miss(self):
        lru = LRUCache(max_size=3)
        assert lru.get("x") is None

    def test_put_get(self):
        lru = LRUCache(max_size=3)
        lru.put("a", np.array([1, 2, 3]))
        result = lru.get("a")
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_eviction(self):
        lru = LRUCache(max_size=2)
        lru.put("a", np.array([1]))
        lru.put("b", np.array([2]))
        lru.put("c", np.array([3]))
        assert lru.get("a") is None
        assert lru.get("b") is not None
        assert lru.get("c") is not None
