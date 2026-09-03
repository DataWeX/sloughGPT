"""Tests for inference ops — rmsnorm, layernorm, matmul, blas."""
from __future__ import annotations

import numpy as np
import pytest

from domains.inference.ops.blas import is_available, sgemm
from domains.inference.ops.layernorm import layernorm
from domains.inference.ops.matmul import matmul
from domains.inference.ops.rmsnorm import rmsnorm


class TestRmsnorm:
    def test_shape_preserved(self):
        x = np.random.randn(2, 4).astype(np.float32)
        w = np.ones(4, dtype=np.float32)
        out = rmsnorm(x, w)
        assert out.shape == x.shape

    def test_normalizes(self):
        x = np.array([[2.0, 4.0]], dtype=np.float32)
        w = np.ones(2, dtype=np.float32)
        out = rmsnorm(x, w)
        rms = np.sqrt(np.mean(out[0] ** 2))
        assert abs(rms - 1.0) < 1e-4

    def test_weight_scales(self):
        x = np.ones((1, 4), dtype=np.float32)
        w = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        out = rmsnorm(x, w)
        np.testing.assert_allclose(out[0], [2.0, 2.0, 2.0, 2.0], rtol=1e-5)


class TestLayernorm:
    def test_shape_preserved(self):
        x = np.random.randn(2, 4).astype(np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        out = layernorm(x, w, b)
        assert out.shape == x.shape

    def test_zero_mean(self):
        x = np.random.randn(1, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        out = layernorm(x, w, b)
        assert abs(out.mean()) < 1e-5

    def test_bias_shifts(self):
        x = np.zeros((1, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = layernorm(x, w, b)
        np.testing.assert_allclose(out[0], [1.0, 2.0, 3.0, 4.0], atol=1e-5)


class TestMatmul:
    def test_basic(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        out = matmul(a, b)
        expected = a @ b
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_shape(self):
        a = np.random.randn(3, 5).astype(np.float32)
        b = np.random.randn(5, 7).astype(np.float32)
        out = matmul(a, b)
        assert out.shape == (3, 7)


class TestBlas:
    def test_is_available(self):
        # On Linux, Accelerate is not available
        result = is_available()
        assert isinstance(result, bool)

    def test_sgemm_fallback(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        out = sgemm(a, b)
        expected = a @ b
        np.testing.assert_allclose(out, expected, rtol=1e-5)
