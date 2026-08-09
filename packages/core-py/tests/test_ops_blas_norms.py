"""Tests for inference/ops — blas, layernorm, matmul, rmsnorm.

Covers:
  - BLAS sgemm with Accelerate and numpy fallback
  - Layer normalization output, shape, and numerical properties
  - RMS normalization output, shape, and numerical properties
  - matmul dispatch between C path and numpy fallback
"""

import numpy as np
import pytest
from domains.inference.ops import blas
from domains.inference.ops.layernorm import layernorm
from domains.inference.ops.matmul import matmul
from domains.inference.ops.rmsnorm import rmsnorm


# ── BLAS ──────────────────────────────────────────────────────────

class TestBLASSgemm:
    def test_basic(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        c = blas.sgemm(a, b)
        expected = a @ b
        np.testing.assert_allclose(c, expected, rtol=1e-5)

    def test_single_row(self):
        a = np.array([[1, 2, 3]], dtype=np.float32)
        b = np.array([[4], [5], [6]], dtype=np.float32)
        c = blas.sgemm(a, b)
        assert c.shape == (1, 1)
        assert abs(c[0, 0] - 32.0) < 1e-5

    def test_shape_mismatch_raises(self):
        a = np.zeros((2, 3), dtype=np.float32)
        b = np.zeros((4, 5), dtype=np.float32)
        with pytest.raises((AssertionError, ValueError)):
            blas.sgemm(a, b)

    def test_large_random(self):
        rng = np.random.RandomState(42)
        a = rng.randn(64, 128).astype(np.float32)
        b = rng.randn(128, 32).astype(np.float32)
        c = blas.sgemm(a, b)
        expected = a @ b
        np.testing.assert_allclose(c, expected, rtol=1e-4)

    def test_is_available_returns_bool(self):
        result = blas.is_available()
        assert isinstance(result, bool)


# ── LayerNorm ─────────────────────────────────────────────────────

class TestLayerNorm:
    def test_output_shape(self):
        x = np.random.randn(4, 16).astype(np.float32)
        w = np.ones(16, dtype=np.float32)
        b = np.zeros(16, dtype=np.float32)
        out = layernorm(x, w, b)
        assert out.shape == x.shape

    def test_normalizes_per_row(self):
        x = np.random.randn(8, 32).astype(np.float32) * 10
        w = np.ones(32, dtype=np.float32)
        b = np.zeros(32, dtype=np.float32)
        out = layernorm(x, w, b)
        means = out.mean(axis=-1)
        np.testing.assert_allclose(means, 0, atol=1e-5)

    def test_unit_variance(self):
        x = np.random.randn(8, 32).astype(np.float32) * 10
        w = np.ones(32, dtype=np.float32)
        b = np.zeros(32, dtype=np.float32)
        out = layernorm(x, w, b)
        variances = out.var(axis=-1)
        np.testing.assert_allclose(variances, 1, atol=1e-4)

    def test_affine_transform(self):
        x = np.ones((2, 8), dtype=np.float32)
        w = np.full(8, 2.0, dtype=np.float32)
        b = np.full(8, 3.0, dtype=np.float32)
        out = layernorm(x, w, b)
        # mean=1, var=0, (x-1)/sqrt(0+eps)*2+3 ≈ 3
        np.testing.assert_allclose(out, 3.0, atol=1e-4)

    def test_3d_input(self):
        x = np.random.randn(2, 4, 16).astype(np.float32)
        w = np.ones(16, dtype=np.float32)
        b = np.zeros(16, dtype=np.float32)
        out = layernorm(x, w, b)
        assert out.shape == (2, 4, 16)
        means = out.mean(axis=-1)
        np.testing.assert_allclose(means, 0, atol=1e-5)

    def test_custom_eps(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.ones(3, dtype=np.float32)
        b = np.zeros(3, dtype=np.float32)
        out = layernorm(x, w, b, eps=1e-3)
        assert out.shape == x.shape


# ── RMSNorm ───────────────────────────────────────────────────────

class TestRMSNorm:
    def test_output_shape(self):
        x = np.random.randn(4, 16).astype(np.float32)
        w = np.ones(16, dtype=np.float32)
        out = rmsnorm(x, w)
        assert out.shape == x.shape

    def test_rms_is_one_after_norm(self):
        x = np.random.randn(8, 32).astype(np.float32) * 5
        w = np.ones(32, dtype=np.float32)
        out = rmsnorm(x, w)
        rms = np.sqrt(np.mean(out ** 2, axis=-1))
        np.testing.assert_allclose(rms, 1.0, atol=1e-4)

    def test_preserves_direction(self):
        x = np.random.randn(4, 16).astype(np.float32)
        w = np.ones(16, dtype=np.float32)
        out = rmsnorm(x, w)
        # Direction should be preserved (cosine similarity ≈ 1)
        for i in range(4):
            cos = np.dot(out[i], x[i]) / (np.linalg.norm(out[i]) * np.linalg.norm(x[i]))
            assert abs(cos - 1.0) < 1e-4

    def test_scale_weight(self):
        x = np.ones((2, 8), dtype=np.float32)
        w = np.full(8, 2.0, dtype=np.float32)
        out = rmsnorm(x, w)
        # rms(1,1,...,1) = 1, so out = 1/1 * 2 = 2
        np.testing.assert_allclose(out, 2.0, atol=1e-5)

    def test_3d_input(self):
        x = np.random.randn(2, 4, 16).astype(np.float32)
        w = np.ones(16, dtype=np.float32)
        out = rmsnorm(x, w)
        assert out.shape == (2, 4, 16)
        rms = np.sqrt(np.mean(out ** 2, axis=-1))
        np.testing.assert_allclose(rms, 1.0, atol=1e-4)

    def test_zero_input(self):
        x = np.zeros((2, 8), dtype=np.float32)
        w = np.ones(8, dtype=np.float32)
        out = rmsnorm(x, w)
        np.testing.assert_allclose(out, 0, atol=1e-6)


# ── ops.matmul ────────────────────────────────────────────────────

class TestOpsMatmul:
    def test_basic(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        c = matmul(a, b)
        expected = a @ b
        np.testing.assert_allclose(c, expected, rtol=1e-5)

    def test_fallback_path_float64(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.float64)
        b = np.array([[5, 6], [7, 8]], dtype=np.float64)
        c = matmul(a, b)
        expected = a @ b
        np.testing.assert_allclose(c, expected, rtol=1e-10)

    def test_large_random(self):
        rng = np.random.RandomState(99)
        a = rng.randn(32, 64).astype(np.float32)
        b = rng.randn(64, 48).astype(np.float32)
        c = matmul(a, b)
        expected = a @ b
        np.testing.assert_allclose(c, expected, rtol=1e-4)

    def test_1d_shapes(self):
        a = np.array([[1, 2, 3]], dtype=np.float32)
        b = np.array([[4], [5], [6]], dtype=np.float32)
        c = matmul(a, b)
        assert c.shape == (1, 1)
        assert abs(c[0, 0] - 32) < 1e-5
