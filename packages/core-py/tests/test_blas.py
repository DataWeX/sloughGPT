"""Tests for domains.inference.ops.blas — sgemm numpy fallback + is_available."""

import numpy as np
import pytest
from domains.inference.ops.blas import sgemm, is_available


class TestSgemm:
    def test_basic_matmul(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        result = sgemm(a, b)
        expected = a @ b
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_3x3(self):
        a = np.random.randn(3, 3).astype(np.float32)
        b = np.random.randn(3, 3).astype(np.float32)
        result = sgemm(a, b)
        np.testing.assert_allclose(result, a @ b, atol=1e-6)

    def test_non_square(self):
        a = np.random.randn(2, 4).astype(np.float32)
        b = np.random.randn(4, 3).astype(np.float32)
        result = sgemm(a, b)
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result, a @ b, atol=1e-6)

    def test_alpha_beta(self):
        a = np.ones((2, 2), dtype=np.float32)
        b = np.ones((2, 2), dtype=np.float32)
        # On non-Accelerate platforms, sgemm falls back to np.matmul
        # which ignores alpha/beta. Only test on Accelerate.
        if is_available():
            result = sgemm(a, b, alpha=2.0, beta=0.0)
            expected = np.full((2, 2), 4.0, dtype=np.float32)
            np.testing.assert_allclose(result, expected, atol=1e-6)
        else:
            # Numpy fallback: alpha/beta ignored
            result = sgemm(a, b, alpha=2.0, beta=0.0)
            expected = np.full((2, 2), 2.0, dtype=np.float32)
            np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_shape_mismatch_raises(self):
        a = np.random.randn(2, 3).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        with pytest.raises(ValueError):
            sgemm(a, b)

    def test_empty_matrices(self):
        a = np.empty((0, 3), dtype=np.float32)
        b = np.empty((3, 0), dtype=np.float32)
        result = sgemm(a, b)
        assert result.shape == (0, 0)


class TestIsAvailable:
    def test_returns_bool(self):
        result = is_available()
        assert isinstance(result, bool)

    def test_on_linux_returns_false(self):
        import sys
        if sys.platform != "linux":
            pytest.skip("not Linux")
        assert is_available() is False
