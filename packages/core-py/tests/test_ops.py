"""Tests for inference ops — layernorm, rmsnorm, blas (numpy fallback)."""

import numpy as np
import pytest
from domains.inference.ops.layernorm import layernorm
from domains.inference.ops.rmsnorm import rmsnorm
from domains.inference.ops.blas import sgemm, is_available


class TestLayernorm:
    def test_output_shape(self):
        x = np.random.randn(2, 4, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        y = layernorm(x, w, b)
        assert y.shape == x.shape

    def test_normalized_mean_near_zero(self):
        x = np.random.randn(4, 16).astype(np.float32) * 10
        w = np.ones(16, dtype=np.float32)
        b = np.zeros(16, dtype=np.float32)
        y = layernorm(x, w, b)
        means = y.mean(axis=-1)
        assert np.allclose(means, 0.0, atol=1e-5)

    def test_normalized_var_near_one(self):
        x = np.random.randn(4, 16).astype(np.float32) * 10
        w = np.ones(16, dtype=np.float32)
        b = np.zeros(16, dtype=np.float32)
        y = layernorm(x, w, b)
        vars = y.var(axis=-1)
        assert np.allclose(vars, 1.0, atol=1e-2)

    def test_weight_scales(self):
        x = np.ones((2, 8), dtype=np.float32)
        w = np.full(8, 2.0, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        y = layernorm(x, w, b)
        # Constant input → zero mean, zero var → 0/eps * w + b ≈ 0
        assert np.allclose(y, 0.0, atol=1e-5)

    def test_bias_shifts(self):
        x = np.ones((2, 8), dtype=np.float32) * 5.0
        w = np.ones(8, dtype=np.float32)
        b = np.full(8, 3.0, dtype=np.float32)
        y = layernorm(x, w, b)
        # Constant input normalized to 0, then + bias
        assert np.allclose(y, 3.0, atol=1e-5)

    def test_3d_input(self):
        x = np.random.randn(2, 3, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        y = layernorm(x, w, b)
        assert y.shape == (2, 3, 8)

    def test_epsilon_prevents_nan(self):
        x = np.zeros((2, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        y = layernorm(x, w, b)
        assert np.all(np.isfinite(y))


class TestRmsnorm:
    def test_output_shape(self):
        x = np.random.randn(2, 4, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        y = rmsnorm(x, w)
        assert y.shape == x.shape

    def test_normalized_rms_near_one(self):
        x = np.random.randn(4, 16).astype(np.float32) * 10
        w = np.ones(16, dtype=np.float32)
        y = rmsnorm(x, w)
        rms = np.sqrt(np.mean(y ** 2, axis=-1))
        assert np.allclose(rms, 1.0, atol=0.1)

    def test_weight_scales(self):
        x = np.ones((2, 8), dtype=np.float32) * 3.0
        w = np.full(8, 2.0, dtype=np.float32)
        y = rmsnorm(x, w)
        # Constant input: rms = 3.0, y = 3.0 / 3.0 * 2.0 = 2.0
        assert np.allclose(y, 2.0, atol=1e-5)

    def test_epsilon_prevents_nan(self):
        x = np.zeros((2, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        y = rmsnorm(x, w)
        assert np.all(np.isfinite(y))

    def test_3d_input(self):
        x = np.random.randn(2, 3, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        y = rmsnorm(x, w)
        assert y.shape == (2, 3, 8)


class TestBlas:
    def test_numpy_fallback(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        c = sgemm(a, b)
        expected = np.matmul(a, b)
        assert np.allclose(c, expected, atol=1e-5)

    def test_square_matrices(self):
        a = np.eye(3, dtype=np.float32)
        b = np.random.randn(3, 3).astype(np.float32)
        c = sgemm(a, b)
        assert np.allclose(c, b, atol=1e-5)

    def test_shape_mismatch_raises(self):
        a = np.random.randn(2, 3).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        with pytest.raises(ValueError):
            sgemm(a, b)

    def test_is_available_returns_bool(self):
        result = is_available()
        assert isinstance(result, bool)
