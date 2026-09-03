"""Tests for numpy_ops — softmax, activations, rope, type conversion."""
from __future__ import annotations

import numpy as np

from domains.infrastructure.numpy_ops import (
    gelu,
    layer_norm,
    rmsnorm,
    rope,
    silu,
    softmax,
    to_float32,
)


class TestSoftmax:
    def test_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        out = softmax(x)
        assert abs(out.sum() - 1.0) < 1e-6

    def test_preserves_shape(self):
        x = np.random.randn(3, 5).astype(np.float32)
        out = softmax(x, axis=-1)
        assert out.shape == x.shape

    def test_large_values_stable(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        out = softmax(x)
        assert abs(out.sum() - 1.0) < 1e-6


class TestRmsnorm:
    def test_output_rms_is_one(self):
        x = np.random.randn(2, 4).astype(np.float32)
        w = np.ones(4, dtype=np.float32)
        out = rmsnorm(x, w)
        rms = np.sqrt(np.mean(out ** 2, axis=-1, keepdims=True))
        np.testing.assert_allclose(rms, 1.0, atol=1e-5)


class TestLayerNorm:
    def test_zero_mean_unit_variance(self):
        x = np.random.randn(1, 16).astype(np.float32)
        w = np.ones(16, dtype=np.float32)
        b = np.zeros(16, dtype=np.float32)
        out = layer_norm(x, w, b)
        assert abs(out.mean()) < 1e-5
        assert abs(out.var() - 1.0) < 0.1

    def test_with_bias(self):
        x = np.zeros((1, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = layer_norm(x, w, b)
        np.testing.assert_allclose(out[0], [1.0, 2.0, 3.0, 4.0], atol=1e-5)


class TestGelu:
    def test_approximation(self):
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        out = gelu(x)
        assert out[1] == 0.0  # gelu(0) = 0
        assert out[2] > 0.5   # gelu(1) > 0.5

    def test_preserves_shape(self):
        x = np.random.randn(3, 4).astype(np.float32)
        assert gelu(x).shape == x.shape


class TestSilu:
    def test_basic(self):
        x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        out = silu(x)
        assert abs(out[0]) < 1e-6  # silu(0) = 0
        assert out[1] > 0.5

    def test_preserves_shape(self):
        x = np.random.randn(2, 3).astype(np.float32)
        assert silu(x).shape == x.shape


class TestRope:
    def test_output_shape(self):
        x = np.random.randn(4, 2, 8).astype(np.float32)
        out = rope(x, pos=0, dim=8)
        assert out.shape == x.shape

    def test_different_pos(self):
        x = np.ones((2, 1, 4), dtype=np.float32)
        out0 = rope(x, pos=0, dim=4)
        out1 = rope(x, pos=10, dim=4)
        assert not np.allclose(out0, out1)


class TestToFloat32:
    def test_float16(self):
        x = np.array([1.0, 2.0], dtype=np.float16)
        out = to_float32(x)
        assert out.dtype == np.float32

    def test_float32_passthrough(self):
        x = np.array([1.0, 2.0], dtype=np.float32)
        out = to_float32(x)
        assert out.dtype == np.float32
