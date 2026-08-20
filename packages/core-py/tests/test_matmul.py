"""Tests for domains.inference.ops.matmul — matmul dispatch."""

import numpy as np
import pytest
from domains.inference.ops.matmul import matmul


class TestMatmul:
    def test_basic(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        result = matmul(a, b)
        assert np.allclose(result, a @ b)

    def test_float64(self):
        a = np.array([[1, 2]], dtype=np.float64)
        b = np.array([[3], [4]], dtype=np.float64)
        result = matmul(a, b)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(11.0)

    def test_rectangular(self):
        a = np.random.randn(2, 5).astype(np.float32)
        b = np.random.randn(5, 3).astype(np.float32)
        result = matmul(a, b)
        assert result.shape == (2, 3)

    def test_1d_vector(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)
        result = matmul(a, b)
        assert result.shape == ()
        assert result == pytest.approx(32.0)

    def test_large_matrix(self):
        a = np.random.randn(100, 64).astype(np.float32)
        b = np.random.randn(64, 100).astype(np.float32)
        result = matmul(a, b)
        assert result.shape == (100, 100)
        expected = a @ b
        assert np.allclose(result, expected, atol=1e-5)
