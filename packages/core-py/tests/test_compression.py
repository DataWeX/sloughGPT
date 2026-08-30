"""Tests for domains.infrastructure.compression — CompressedWeight, LRUCache."""

import numpy as np
import threading
import pytest
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

    def test_decompress_no_residual(self):
        centroids = np.array([1.0, 2.0], dtype=np.float32)
        assignments = np.array([0, 1, 0, 1], dtype=np.int32)
        shape = (4,)
        cw = CompressedWeight(centroids, assignments, None, shape, np.dtype("float32"))
        result = cw.decompress()
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 1.0, 2.0])

    def test_decompress_with_residual(self):
        centroids = np.array([1.0, 2.0], dtype=np.float32)
        assignments = np.array([0, 1], dtype=np.int32)
        residual = np.array([0.1, -0.1], dtype=np.float16)
        shape = (2,)
        cw = CompressedWeight(centroids, assignments, residual, shape, np.dtype("float32"))
        result = cw.decompress()
        expected = np.array([1.1, 1.9], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_decompress_2d_shape(self):
        centroids = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        assignments = np.array([0, 1, 2, 3], dtype=np.int32)
        shape = (2, 2)
        cw = CompressedWeight(centroids, assignments, None, shape, np.dtype("float32"))
        result = cw.decompress()
        assert result.shape == (2, 2)
        np.testing.assert_array_almost_equal(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_compressed_bytes_raw(self):
        centroids = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assignments = np.array([0, 1, 2], dtype=np.int32)
        shape = (3,)
        cw = CompressedWeight(centroids, assignments, None, shape, np.dtype("float32"))
        assert cw.compressed_bytes == centroids.nbytes + assignments.nbytes

    def test_compressed_bytes_linear(self):
        centroids = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assignments = np.array([0, 1, 2], dtype=np.int32)
        shape = (3,)
        cw = CompressedWeight(
            centroids, assignments, None, shape, np.dtype("float32"),
            centroid_fn="linear", centroid_fn_params={"a": 1.0, "b": 0.0},
        )
        assert cw.compressed_bytes == 8 + assignments.nbytes

    def test_compressed_bytes_with_residual(self):
        centroids = np.array([1.0], dtype=np.float32)
        assignments = np.array([0], dtype=np.int32)
        residual = np.array([0.1], dtype=np.float16)
        shape = (1,)
        cw = CompressedWeight(centroids, assignments, residual, shape, np.dtype("float32"))
        assert cw.compressed_bytes == centroids.nbytes + assignments.nbytes + residual.nbytes

    def test_linear_decompress_different_params(self):
        centroids = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        assignments = np.array([0, 1, 2], dtype=np.int32)
        shape = (3,)
        cw = CompressedWeight(
            centroids, assignments, None, shape, np.dtype("float32"),
            centroid_fn="linear", centroid_fn_params={"a": 2.0, "b": 5.0},
        )
        result = cw.decompress()
        expected = np.array([5.0, 7.0, 9.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_assignments_out_of_range(self):
        centroids = np.array([1.0, 2.0], dtype=np.float32)
        assignments = np.array([0, 1, 0, 1], dtype=np.int32)
        shape = (4,)
        cw = CompressedWeight(centroids, assignments, None, shape, np.dtype("float32"))
        result = cw.decompress()
        assert result.shape == (4,)

    def test_large_compression(self):
        n = 1000
        centroids = np.arange(n, dtype=np.float32)
        assignments = np.random.randint(0, n, size=n * 10).astype(np.int32)
        shape = (n * 10,)
        cw = CompressedWeight(centroids, assignments, None, shape, np.dtype("float32"))
        result = cw.decompress()
        assert result.shape == shape

    def test_slots(self):
        centroids = np.array([1.0], dtype=np.float32)
        assignments = np.array([0], dtype=np.int32)
        shape = (1,)
        cw = CompressedWeight(centroids, assignments, None, shape, np.dtype("float32"))
        assert hasattr(cw, '__slots__')
        assert not hasattr(cw, '__dict__')

    def test_decompress_3d_shape(self):
        centroids = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        assignments = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
        shape = (2, 2, 2)
        cw = CompressedWeight(centroids, assignments, None, shape, np.dtype("float32"))
        result = cw.decompress()
        assert result.shape == (2, 2, 2)
        np.testing.assert_array_almost_equal(result.flatten(), np.arange(1, 9, dtype=np.float32))

    def test_decompress_single_element(self):
        centroids = np.array([42.0], dtype=np.float32)
        assignments = np.array([0], dtype=np.int32)
        shape = (1,)
        cw = CompressedWeight(centroids, assignments, None, shape, np.dtype("float32"))
        result = cw.decompress()
        np.testing.assert_array_almost_equal(result, [42.0])

    def test_decompress_repeated_assignments(self):
        centroids = np.array([5.0, 10.0], dtype=np.float32)
        assignments = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
        shape = (8,)
        cw = CompressedWeight(centroids, assignments, None, shape, np.dtype("float32"))
        result = cw.decompress()
        expected = np.array([5.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_linear_centroid_fn_params_stored(self):
        params = {"a": 3.0, "b": 7.0}
        cw = CompressedWeight(
            np.zeros(3, dtype=np.float32),
            np.array([0, 1, 2], dtype=np.int32),
            None, (3,), np.dtype("float32"),
            centroid_fn="linear", centroid_fn_params=params,
        )
        assert cw.centroid_fn == "linear"
        assert cw.centroid_fn_params == params

    def test_linear_a_zero(self):
        cw = CompressedWeight(
            np.zeros(3, dtype=np.float32),
            np.array([0, 1, 2], dtype=np.int32),
            None, (3,), np.dtype("float32"),
            centroid_fn="linear", centroid_fn_params={"a": 0.0, "b": 99.0},
        )
        result = cw.decompress()
        np.testing.assert_array_almost_equal(result, [99.0, 99.0, 99.0])

    def test_linear_negative_a(self):
        cw = CompressedWeight(
            np.zeros(3, dtype=np.float32),
            np.array([0, 1, 2], dtype=np.int32),
            None, (3,), np.dtype("float32"),
            centroid_fn="linear", centroid_fn_params={"a": -1.0, "b": 10.0},
        )
        result = cw.decompress()
        expected = np.array([10.0, 9.0, 8.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_residual_with_2d_shape(self):
        centroids = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        assignments = np.array([0, 1, 2, 3], dtype=np.int32)
        residual = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float16)
        shape = (2, 2)
        cw = CompressedWeight(centroids, assignments, residual, shape, np.dtype("float32"))
        result = cw.decompress()
        expected = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_compressed_bytes_large_residual(self):
        c = np.array([1.0], dtype=np.float32)
        a = np.array([0], dtype=np.int32)
        r = np.arange(100, dtype=np.float16)
        cw = CompressedWeight(c, a, r, (100,), np.dtype("float32"))
        assert cw.compressed_bytes == 4 + 4 + 200

    def test_assignments_all_same_cluster(self):
        c = np.array([7.0, 8.0], dtype=np.float32)
        a = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        cw = CompressedWeight(c, a, None, (5,), np.dtype("float32"))
        result = cw.decompress()
        np.testing.assert_array_almost_equal(result, [7.0] * 5)

    def test_dtype_preserved(self):
        c = np.array([1.0], dtype=np.float32)
        a = np.array([0], dtype=np.int32)
        cw = CompressedWeight(c, a, None, (1,), np.dtype("float32"))
        assert cw.dtype == np.dtype("float32")

    def test_centroid_fn_none_default(self):
        c = np.array([1.0], dtype=np.float32)
        a = np.array([0], dtype=np.int32)
        cw = CompressedWeight(c, a, None, (1,), np.dtype("float32"))
        assert cw.centroid_fn is None
        assert cw.centroid_fn_params is None

    def test_residual_none_default(self):
        c = np.array([1.0], dtype=np.float32)
        a = np.array([0], dtype=np.int32)
        cw = CompressedWeight(c, a, None, (1,), np.dtype("float32"))
        assert cw.residual is None

    def test_decompress_linear_large(self):
        n = 100
        c = np.zeros(n, dtype=np.float32)
        a = np.arange(n, dtype=np.int32)
        cw = CompressedWeight(
            c, a, None, (n,), np.dtype("float32"),
            centroid_fn="linear", centroid_fn_params={"a": 1.0, "b": 0.0},
        )
        result = cw.decompress()
        np.testing.assert_array_almost_equal(result, np.arange(n, dtype=np.float32))

    def test_compressed_bytes_no_residual_linear(self):
        c = np.zeros(5, dtype=np.float32)
        a = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        cw = CompressedWeight(
            c, a, None, (5,), np.dtype("float32"),
            centroid_fn="linear", centroid_fn_params={"a": 1.0, "b": 0.0},
        )
        assert cw.compressed_bytes == 8 + a.nbytes

    def test_residual_float16_precision(self):
        c = np.array([1.0, 2.0], dtype=np.float32)
        a = np.array([0, 1], dtype=np.int32)
        r = np.array([0.001, 0.002], dtype=np.float16)
        cw = CompressedWeight(c, a, r, (2,), np.dtype("float32"))
        result = cw.decompress()
        assert result.dtype == np.float32

    def test_assignments_negative_indices(self):
        c = np.array([1.0, 2.0], dtype=np.float32)
        a = np.array([0, 1, -1, -2], dtype=np.int32)
        cw = CompressedWeight(c, a, None, (4,), np.dtype("float32"))
        result = cw.decompress()
        assert result.shape == (4,)

    def test_decompress_float64_dtype(self):
        c = np.array([1.0, 2.0], dtype=np.float64)
        a = np.array([0, 1], dtype=np.int32)
        cw = CompressedWeight(c, a, None, (2,), np.dtype("float64"))
        result = cw.decompress()
        assert result.dtype == np.float64

    def test_decompress_int32_centroids(self):
        c = np.array([1, 2, 3], dtype=np.float32)
        a = np.array([0, 1, 2], dtype=np.int32)
        cw = CompressedWeight(c, a, None, (3,), np.dtype("float32"))
        result = cw.decompress()
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_decompress_empty_assignments(self):
        c = np.array([], dtype=np.float32)
        a = np.array([], dtype=np.int32)
        cw = CompressedWeight(c, a, None, (0,), np.dtype("float32"))
        result = cw.decompress()
        assert result.shape == (0,)

    def test_compressed_bytes_zero_elements(self):
        c = np.array([], dtype=np.float32)
        a = np.array([], dtype=np.int32)
        cw = CompressedWeight(c, a, None, (0,), np.dtype("float32"))
        assert cw.compressed_bytes == 0

    def test_residual_matches_shape(self):
        c = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a = np.array([0, 1, 2], dtype=np.int32)
        r = np.array([0.1, 0.2, 0.3], dtype=np.float16)
        cw = CompressedWeight(c, a, r, (3,), np.dtype("float32"))
        result = cw.decompress()
        expected = np.array([1.1, 2.2, 3.3], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_assignments_wrap_around(self):
        c = np.array([10.0, 20.0], dtype=np.float32)
        a = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        cw = CompressedWeight(c, a, None, (6,), np.dtype("float32"))
        result = cw.decompress()
        np.testing.assert_array_almost_equal(result, [10, 20, 10, 20, 10, 20])

    def test_large_residual(self):
        n = 1000
        c = np.ones(n, dtype=np.float32)
        a = np.arange(n, dtype=np.int32)
        r = np.ones(n, dtype=np.float16) * 0.5
        cw = CompressedWeight(c, a, r, (n,), np.dtype("float32"))
        result = cw.decompress()
        np.testing.assert_array_almost_equal(result, np.ones(n) * 1.5, decimal=2)

    def test_negative_centroids(self):
        c = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        a = np.array([0, 1, 2], dtype=np.int32)
        cw = CompressedWeight(c, a, None, (3,), np.dtype("float32"))
        result = cw.decompress()
        np.testing.assert_array_almost_equal(result, [-1.0, -2.0, -3.0])

    def test_mixed_positive_negative_residual(self):
        c = np.array([1.0, 2.0], dtype=np.float32)
        a = np.array([0, 1], dtype=np.int32)
        r = np.array([-0.5, 0.5], dtype=np.float16)
        cw = CompressedWeight(c, a, r, (2,), np.dtype("float32"))
        result = cw.decompress()
        np.testing.assert_array_almost_equal(result, [0.5, 2.5], decimal=3)

    def test_compressed_weight_shape_preserved(self):
        c = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        a = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
        cw = CompressedWeight(c, a, None, (2, 3), np.dtype("float32"))
        result = cw.decompress()
        assert result.shape == (2, 3)
        np.testing.assert_array_almost_equal(result, [[1, 2, 3], [4, 5, 6]])

    def test_1d_shape(self):
        c = np.array([1.0, 2.0], dtype=np.float32)
        a = np.array([0, 1], dtype=np.int32)
        cw = CompressedWeight(c, a, None, (2,), np.dtype("float32"))
        result = cw.decompress()
        assert result.ndim == 1

    def test_residual_converted_to_float32(self):
        c = np.array([1.0, 2.0], dtype=np.float32)
        a = np.array([0, 1], dtype=np.int32)
        r = np.array([0.1, 0.2], dtype=np.float16)
        cw = CompressedWeight(c, a, r, (2,), np.dtype("float32"))
        result = cw.decompress()
        assert result.dtype == np.float32

    def test_linear_very_large_a(self):
        c = np.zeros(3, dtype=np.float32)
        a = np.array([0, 1, 2], dtype=np.int32)
        cw = CompressedWeight(
            c, a, None, (3,), np.dtype("float32"),
            centroid_fn="linear", centroid_fn_params={"a": 1e6, "b": 0.0},
        )
        result = cw.decompress()
        np.testing.assert_array_almost_equal(result, [0, 1e6, 2e6])


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

    def test_get_moves_to_end(self):
        lru = LRUCache(max_size=3)
        lru.put("a", np.array([1]))
        lru.put("b", np.array([2]))
        lru.put("c", np.array([3]))
        lru.get("a")
        lru.put("d", np.array([4]))
        assert lru.get("a") is not None
        assert lru.get("b") is None

    def test_put_existing_key(self):
        lru = LRUCache(max_size=2)
        lru.put("a", np.array([1]))
        lru.put("a", np.array([2]))
        result = lru.get("a")
        np.testing.assert_array_equal(result, [2])

    def test_max_size_one(self):
        lru = LRUCache(max_size=1)
        lru.put("a", np.array([1]))
        lru.put("b", np.array([2]))
        assert lru.get("a") is None
        assert lru.get("b") is not None

    def test_empty_cache(self):
        lru = LRUCache(max_size=5)
        assert lru.get("anything") is None

    def test_get_does_not_count_as_put(self):
        lru = LRUCache(max_size=2)
        lru.put("a", np.array([1]))
        lru.put("b", np.array([2]))
        lru.get("a")
        lru.get("a")
        lru.put("c", np.array([3]))
        assert lru.get("a") is not None
        assert lru.get("b") is None

    def test_thread_safety(self):
        lru = LRUCache(max_size=10)
        errors = []

        def writer(start):
            try:
                for i in range(100):
                    lru.put(f"key{start}_{i}", np.array([i]))
            except Exception as e:
                errors.append(e)

        def reader(start):
            try:
                for i in range(100):
                    lru.get(f"key{start}_{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(4):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader, args=(i,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_many_items(self):
        lru = LRUCache(max_size=50)
        for i in range(100):
            lru.put(f"k{i}", np.array([i]))
        for i in range(50, 100):
            assert lru.get(f"k{i}") is not None
        for i in range(50):
            assert lru.get(f"k{i}") is None

    def test_put_overwrite_preserves_size(self):
        lru = LRUCache(max_size=3)
        lru.put("a", np.array([1]))
        lru.put("b", np.array([2]))
        lru.put("a", np.array([99]))
        lru.put("c", np.array([3]))
        lru.put("d", np.array([4]))
        assert lru.get("b") is None
        np.testing.assert_array_equal(lru.get("a"), [99])

    def test_get_many_times(self):
        lru = LRUCache(max_size=3)
        lru.put("a", np.array([1]))
        for _ in range(20):
            lru.get("a")
        lru.put("b", np.array([2]))
        lru.put("c", np.array([3]))
        lru.put("d", np.array([4]))
        assert lru.get("a") is None
        assert lru.get("b") is not None
        assert lru.get("c") is not None
        assert lru.get("d") is not None

    def test_sequential_fill_and_drain(self):
        lru = LRUCache(max_size=5)
        for i in range(5):
            lru.put(f"k{i}", np.array([i]))
        for i in range(5):
            lru.get(f"k{i}")
        for i in range(5):
            assert lru.get(f"k{i}") is not None

    def test_stress_eviction(self):
        lru = LRUCache(max_size=2)
        lru.put("a", np.array([1]))
        lru.get("a")
        lru.put("b", np.array([2]))
        lru.put("c", np.array([3]))
        assert lru.get("a") is None
        assert lru.get("b") is not None
        assert lru.get("c") is not None

    def test_replace_same_key_repeatedly(self):
        lru = LRUCache(max_size=2)
        for i in range(50):
            lru.put("x", np.array([i]))
        assert lru.get("x") is not None
        np.testing.assert_array_equal(lru.get("x"), [49])

    def test_interleaved_put_get(self):
        lru = LRUCache(max_size=3)
        lru.put("a", np.array([1]))
        lru.get("a")
        lru.put("b", np.array([2]))
        lru.get("b")
        lru.put("c", np.array([3]))
        lru.get("c")
        lru.put("d", np.array([4]))
        assert lru.get("a") is None

    def test_max_size_zero(self):
        lru = LRUCache(max_size=0)
        lru.put("a", np.array([1]))
        assert lru.get("a") is None

    def test_get_returns_same_array(self):
        lru = LRUCache(max_size=3)
        arr = np.array([1, 2, 3])
        lru.put("a", arr)
        result = lru.get("a")
        np.testing.assert_array_equal(result, arr)

    def test_put_large_array(self):
        lru = LRUCache(max_size=2)
        arr = np.ones((1000, 1000))
        lru.put("big", arr)
        result = lru.get("big")
        assert result.shape == (1000, 1000)

    def test_eviction_order_fifo(self):
        lru = LRUCache(max_size=3)
        lru.put("a", np.array([1]))
        lru.put("b", np.array([2]))
        lru.put("c", np.array([3]))
        lru.put("d", np.array([4]))
        assert lru.get("a") is None
        lru.put("e", np.array([5]))
        assert lru.get("b") is None

    def test_concurrent_writes(self):
        lru = LRUCache(max_size=50)
        def write_range(start):
            for i in range(20):
                lru.put(f"t{start}_{i}", np.array([i]))
        threads = [threading.Thread(target=write_range, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(lru._cache) <= 50

    def test_lru_cache_fifo_eviction_order(self):
        lru = LRUCache(max_size=3)
        lru.put("a", np.array([1]))
        lru.put("b", np.array([2]))
        lru.put("c", np.array([3]))
        lru.put("d", np.array([4]))
        assert lru.get("a") is None
        lru.put("e", np.array([5]))
        assert lru.get("b") is None

    def test_lru_cache_put_same_key_preserves_value(self):
        lru = LRUCache(max_size=2)
        lru.put("x", np.array([10]))
        lru.put("x", np.array([20]))
        np.testing.assert_array_equal(lru.get("x"), [20])

    def test_lru_cache_get_nonexistent(self):
        lru = LRUCache(max_size=5)
        assert lru.get("missing") is None

    def test_lru_cache_large_values(self):
        lru = LRUCache(max_size=2)
        arr1 = np.ones((100, 100))
        arr2 = np.zeros((100, 100))
        lru.put("a", arr1)
        lru.put("b", arr2)
        np.testing.assert_array_equal(lru.get("a"), arr1)
        np.testing.assert_array_equal(lru.get("b"), arr2)

    def test_lru_cache_interleaved_access(self):
        lru = LRUCache(max_size=3)
        lru.put("a", np.array([1]))
        lru.put("b", np.array([2]))
        lru.put("c", np.array([3]))
        lru.get("a")
        lru.get("b")
        lru.put("d", np.array([4]))
        assert lru.get("c") is None
        assert lru.get("a") is not None
        assert lru.get("b") is not None
        assert lru.get("d") is not None

    def test_lru_cache_stress_read_write(self):
        lru = LRUCache(max_size=10)
        errors = []

        def read_write(n):
            try:
                for i in range(50):
                    lru.put(f"key{n}_{i}", np.array([i]))
                    lru.get(f"key{n}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_write, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_cache_internal_structure(self):
        lru = LRUCache(max_size=3)
        lru.put("a", np.array([1]))
        lru.put("b", np.array([2]))
        assert len(lru._cache) == 2

    def test_put_none_value(self):
        lru = LRUCache(max_size=3)
        lru.put("a", None)
        assert lru.get("a") is None

    def test_get_after_eviction_returns_none(self):
        lru = LRUCache(max_size=1)
        lru.put("a", np.array([1]))
        lru.put("b", np.array([2]))
        assert lru.get("a") is None
        assert lru.get("b") is not None

    def test_many_gets_same_key(self):
        lru = LRUCache(max_size=2)
        lru.put("a", np.array([1]))
        for _ in range(100):
            lru.get("a")
        lru.put("b", np.array([2]))
        lru.put("c", np.array([3]))
        assert lru.get("a") is None

    def test_put_scalar(self):
        lru = LRUCache(max_size=2)
        lru.put("a", 42)
        assert lru.get("a") == 42

    def test_put_string(self):
        lru = LRUCache(max_size=2)
        lru.put("a", "hello")
        assert lru.get("a") == "hello"
