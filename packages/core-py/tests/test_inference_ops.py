"""Tests for domains.inference.ops — rmsnorm, layernorm, matmul (numpy fallback path).

Covers: normalization correctness, matmul with various dtypes, shape preservation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.inference.ops.rmsnorm import rmsnorm
from domains.inference.ops.layernorm import layernorm
from domains.inference.ops.matmul import matmul


class TestOpsRmsnorm:
    def test_basic(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.ones(3)
        result = rmsnorm(x, w)
        assert result.shape == x.shape
        # Output should have unit RMS
        rms = np.sqrt(np.mean(result ** 2))
        assert rms == pytest.approx(1.0, abs=0.01)

    def test_weighted(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.array([2.0, 2.0, 2.0])
        result = rmsnorm(x, w)
        rms = np.sqrt(np.mean(result ** 2))
        assert rms == pytest.approx(2.0, abs=0.01)

    def test_2d_batch(self):
        x = np.ones((4, 8))
        w = np.ones(8)
        result = rmsnorm(x, w)
        assert result.shape == (4, 8)

    def test_custom_eps(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.ones(3)
        result = rmsnorm(x, w, eps=1e-3)
        assert result.shape == x.shape

    def test_zero_input(self):
        x = np.zeros((2, 4))
        w = np.ones(4)
        result = rmsnorm(x, w)
        np.testing.assert_array_almost_equal(result, np.zeros((2, 4)))

    def test_large_eps_stabilizes(self):
        x = np.full((1, 3), 1e-8)
        w = np.ones(3)
        result = rmsnorm(x, w, eps=1.0)
        assert np.all(np.isfinite(result))

    def test_negative_values(self):
        x = np.array([[-1.0, -2.0, -3.0]])
        w = np.ones(3)
        result = rmsnorm(x, w)
        rms = np.sqrt(np.mean(result ** 2))
        assert rms == pytest.approx(1.0, abs=0.01)

    def test_preserves_relative_order(self):
        x = np.array([[1.0, 5.0, 10.0]])
        w = np.ones(3)
        result = rmsnorm(x, w)
        assert result[0, 0] < result[0, 1] < result[0, 2]

    def test_weight_zero_scales_to_zero(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.zeros(3)
        result = rmsnorm(x, w)
        np.testing.assert_array_almost_equal(result, np.zeros((1, 3)))

    def test_single_element(self):
        x = np.array([[5.0]])
        w = np.array([1.0])
        result = rmsnorm(x, w)
        rms = np.sqrt(np.mean(result ** 2))
        assert rms == pytest.approx(1.0, abs=0.01)

    def test_high_dimensional(self):
        x = np.ones((2, 3, 4, 5))
        w = np.ones(5)
        result = rmsnorm(x, w)
        assert result.shape == (2, 3, 4, 5)

    def test_different_eps_values(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.ones(3)
        for eps in [1e-7, 1e-5, 1e-3, 1e-1]:
            result = rmsnorm(x, w, eps=eps)
            assert result.shape == x.shape
            assert np.all(np.isfinite(result))

    def test_uneven_weight_distribution(self):
        x = np.array([[1.0, 1.0, 1.0]])
        w = np.array([0.1, 1.0, 10.0])
        result = rmsnorm(x, w)
        assert result[0, 0] < result[0, 1] < result[0, 2]

    def test_1d_input(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        result = rmsnorm(x, w)
        assert result.shape == x.shape

    def test_reproducibility(self):
        x = np.random.randn(3, 5)
        w = np.ones(5)
        r1 = rmsnorm(x, w)
        r2 = rmsnorm(x, w)
        np.testing.assert_array_equal(r1, r2)

    def test_per_element_weight(self):
        x = np.array([[2.0, 4.0, 6.0]])
        w = np.array([0.5, 0.5, 0.5])
        result = rmsnorm(x, w)
        rms = np.sqrt(np.mean(result ** 2))
        assert rms == pytest.approx(0.5, abs=0.01)


class TestOpsLayernorm:
    def test_basic(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.ones(3)
        b = np.zeros(3)
        result = layernorm(x, w, b)
        assert result.shape == x.shape
        # Output should be zero-mean
        assert result.mean() == pytest.approx(0.0, abs=1e-5)

    def test_weighted(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.array([2.0, 2.0, 2.0])
        b = np.array([1.0, 1.0, 1.0])
        result = layernorm(x, w, b)
        assert result.mean() == pytest.approx(1.0, abs=1e-5)

    def test_2d_batch(self):
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        result = layernorm(x, w, b)
        assert result.shape == (4, 8)

    def test_zero_mean_per_row(self):
        x = np.random.randn(10, 16).astype(np.float64)
        w = np.ones(16)
        b = np.zeros(16)
        result = layernorm(x, w, b)
        for row in range(10):
            assert result[row].mean() == pytest.approx(0.0, abs=1e-5)

    def test_unit_variance(self):
        x = np.random.randn(5, 8).astype(np.float64) * 100
        w = np.ones(8)
        b = np.zeros(8)
        result = layernorm(x, w, b)
        for row in range(5):
            assert result[row].std() == pytest.approx(1.0, abs=0.1)

    def test_bias_shifts_mean(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.ones(3)
        b = np.array([5.0, 5.0, 5.0])
        result = layernorm(x, w, b)
        assert result.mean() == pytest.approx(5.0, abs=1e-5)

    def test_weight_scales_variance(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.array([3.0, 3.0, 3.0])
        b = np.zeros(3)
        result = layernorm(x, w, b)
        assert result.std() == pytest.approx(3.0, abs=0.1)

    def test_single_element(self):
        x = np.array([[7.0]])
        w = np.array([1.0])
        b = np.array([0.0])
        result = layernorm(x, w, b)
        assert result.shape == (1, 1)

    def test_high_dimensional(self):
        x = np.random.randn(2, 3, 4).astype(np.float32)
        w = np.ones(4)
        b = np.zeros(4)
        result = layernorm(x, w, b)
        assert result.shape == (2, 3, 4)

    def test_custom_eps(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.ones(3)
        b = np.zeros(3)
        for eps in [1e-7, 1e-5, 1e-3]:
            result = layernorm(x, w, b, eps=eps)
            assert np.all(np.isfinite(result))

    def test_identical_values(self):
        x = np.full((2, 4), 3.14)
        w = np.ones(4)
        b = np.zeros(4)
        result = layernorm(x, w, b)
        np.testing.assert_array_almost_equal(result, np.zeros((2, 4)))

    def test_large_eps(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.ones(3)
        b = np.zeros(3)
        result = layernorm(x, w, b, eps=1e6)
        assert np.all(np.isfinite(result))

    def test_zero_weight_and_bias(self):
        x = np.array([[10.0, 20.0, 30.0]])
        w = np.zeros(3)
        b = np.zeros(3)
        result = layernorm(x, w, b)
        np.testing.assert_array_almost_equal(result, np.zeros((1, 3)))

    def test_batch_row_independence(self):
        x = np.array([[1.0, 2.0], [10.0, 20.0]])
        w = np.ones(2)
        b = np.zeros(2)
        result = layernorm(x, w, b)
        # Each row should independently normalize to zero-mean
        assert result[0].mean() == pytest.approx(0.0, abs=1e-5)
        assert result[1].mean() == pytest.approx(0.0, abs=1e-5)

    def test_negative_values(self):
        x = np.array([[-1.0, -2.0, -3.0]])
        w = np.ones(3)
        b = np.zeros(3)
        result = layernorm(x, w, b)
        assert result.mean() == pytest.approx(0.0, abs=1e-5)

    def test_reproducibility(self):
        x = np.random.randn(3, 5)
        w = np.ones(5)
        b = np.zeros(5)
        r1 = layernorm(x, w, b)
        r2 = layernorm(x, w, b)
        np.testing.assert_array_equal(r1, r2)

    def test_1d_input(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones(4)
        b = np.zeros(4)
        result = layernorm(x, w, b)
        assert result.shape == x.shape

    def test_mixed_bias_and_weight(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.array([2.0, 1.0, 0.5])
        b = np.array([0.0, 1.0, -1.0])
        result = layernorm(x, w, b)
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))


class TestOpsMatmul:
    def test_basic(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = matmul(a, b)
        expected = a @ b
        np.testing.assert_array_almost_equal(result, expected)

    def test_shapes(self):
        a = np.ones((2, 3))
        b = np.ones((3, 4))
        result = matmul(a, b)
        assert result.shape == (2, 4)

    def test_float32(self):
        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((3, 2), dtype=np.float32)
        result = matmul(a, b)
        assert result.dtype in (np.float32, np.float64)

    def test_integer(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = matmul(a, b)
        np.testing.assert_array_equal(result, [[19, 22], [43, 50]])

    def test_identity_matrix(self):
        a = np.array([[1.0, 2.0, 3.0]])
        b = np.eye(3)
        result = matmul(a, b)
        np.testing.assert_array_almost_equal(result, a)

    def test_transpose(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = a.T
        result = matmul(a, b)
        expected = a @ b
        np.testing.assert_array_almost_equal(result, expected)

    def test_batch_like_3d(self):
        a = np.ones((2, 3, 4))
        b = np.ones((4, 5))
        result = matmul(a, b)
        assert result.shape == (2, 3, 5)

    def test_zeros(self):
        a = np.zeros((3, 3))
        b = np.ones((3, 3))
        result = matmul(a, b)
        np.testing.assert_array_equal(result, np.zeros((3, 3)))

    def test_diagonal(self):
        d = np.diag([1.0, 2.0, 3.0])
        result = matmul(d, d)
        expected = np.diag([1.0, 4.0, 9.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_large_matrix(self):
        a = np.random.randn(50, 50)
        b = np.random.randn(50, 50)
        result = matmul(a, b)
        expected = a @ b
        np.testing.assert_array_almost_equal(result, expected)

    def test_column_times_row(self):
        a = np.array([[1.0], [2.0], [3.0]])  # (3,1)
        b = np.array([[4.0, 5.0, 6.0]])      # (1,3)
        result = matmul(a, b)
        expected = np.array([[4.0, 5.0, 6.0], [8.0, 10.0, 12.0], [12.0, 15.0, 18.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_float64(self):
        a = np.ones((2, 3), dtype=np.float64)
        b = np.ones((3, 2), dtype=np.float64)
        result = matmul(a, b)
        expected = a @ b
        np.testing.assert_array_almost_equal(result, expected)

    def test_asymmetric(self):
        a = np.random.randn(2, 7)
        b = np.random.randn(7, 3)
        result = matmul(a, b)
        assert result.shape == (2, 3)
        np.testing.assert_array_almost_equal(result, a @ b)

    def test_negative_values(self):
        a = np.array([[-1.0, -2.0], [-3.0, -4.0]])
        b = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = matmul(a, b)
        np.testing.assert_array_almost_equal(result, a)

    def test_scalar_like_1x1(self):
        a = np.array([[3.0]])
        b = np.array([[7.0]])
        result = matmul(a, b)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(21.0)

    def test_reproducibility(self):
        a = np.random.randn(4, 5)
        b = np.random.randn(5, 6)
        r1 = matmul(a, b)
        r2 = matmul(a, b)
        np.testing.assert_array_equal(r1, r2)

    def test_mixed_dtypes_promote(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int32)
        b = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
        result = matmul(a, b)
        assert np.all(np.isfinite(result))
