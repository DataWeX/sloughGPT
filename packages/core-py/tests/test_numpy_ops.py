"""
Tests for domains/infrastructure/numpy_ops.py — transformer inference utilities.

Covers:
    - softmax correctness and numerical stability
    - rmsnorm correctness
    - layer_norm correctness and bias handling
    - gelu approximation accuracy
    - silu correctness
    - rope positional encoding correctness
    - to_float32 dtype conversion
"""

import numpy as np
import sys
from pathlib import Path
import pytest

_CORE_PY = Path(__file__).resolve().parents[1]
if str(_CORE_PY) not in sys.path:
    sys.path.insert(0, str(_CORE_PY))

from domains.infrastructure.numpy_ops import (
    to_float32,
    softmax,
    rmsnorm,
    layer_norm,
    gelu,
    silu,
    rope,
)


# ── softmax ───────────────────────────────────────────────────────────


class TestSoftmax:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_all_equal(self):
        x = np.array([5.0, 5.0, 5.0])
        result = softmax(x)
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)

    def test_large_values_stable(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_negative_values(self):
        x = np.array([-10.0, -5.0, -1.0])
        result = softmax(x)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_2d_axis(self):
        x = np.random.randn(4, 8)
        result = softmax(x, axis=-1)
        sums = result.sum(axis=-1)
        np.testing.assert_allclose(sums, np.ones(4), atol=1e-6)

    def test_preserves_shape(self):
        x = np.random.randn(3, 5, 7)
        result = softmax(x, axis=-1)
        assert result.shape == x.shape


# ── rmsnorm ───────────────────────────────────────────────────────────


class TestRmsNorm:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones(4)
        result = rmsnorm(x, w)
        rms = np.sqrt(np.mean(result ** 2))
        np.testing.assert_allclose(rms, 1.0, atol=1e-4)

    def test_with_weight(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.array([2.0, 2.0, 2.0, 2.0])
        result = rmsnorm(x, w)
        expected = rmsnorm(x, np.ones(4)) * 2.0
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_zero_input(self):
        x = np.zeros(4)
        w = np.ones(4)
        result = rmsnorm(x, w)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_2d(self):
        x = np.random.randn(8, 32).astype(np.float32)
        w = np.ones(32, dtype=np.float32)
        result = rmsnorm(x, w)
        assert result.shape == x.shape


# ── layer_norm ────────────────────────────────────────────────────────


class TestLayerNorm:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones(4)
        b = np.zeros(4)
        result = layer_norm(x, w, b)
        np.testing.assert_allclose(result.mean(), 0.0, atol=1e-6)
        np.testing.assert_allclose(result.std(), 1.0, atol=1e-2)

    def test_with_bias(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones(4)
        b = np.array([10.0, 10.0, 10.0, 10.0])
        result = layer_norm(x, w, b)
        np.testing.assert_allclose(result.mean(), 10.0, atol=1e-5)

    def test_none_bias(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones(4)
        result = layer_norm(x, w, None)
        np.testing.assert_allclose(result.mean(), 0.0, atol=1e-6)

    def test_2d(self):
        x = np.random.randn(8, 32).astype(np.float32)
        w = np.ones(32, dtype=np.float32)
        b = np.zeros(32, dtype=np.float32)
        result = layer_norm(x, w, b)
        assert result.shape == x.shape
        means = result.mean(axis=-1)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)


# ── gelu ──────────────────────────────────────────────────────────────


class TestGelu:
    def test_zero(self):
        x = np.array([0.0])
        result = gelu(x)
        np.testing.assert_allclose(result, 0.0, atol=1e-5)

    def test_positive_large(self):
        x = np.array([10.0])
        result = gelu(x)
        np.testing.assert_allclose(result, 10.0, atol=1e-3)

    def test_negative_large(self):
        x = np.array([-10.0])
        result = gelu(x)
        np.testing.assert_allclose(result, 0.0, atol=1e-3)

    def test_approximates_relu(self):
        x = np.linspace(-5, 5, 100)
        result = gelu(x)
        # GELU ≈ ReLU for large |x|
        assert result[80] > 0  # large positive
        assert result[20] < 0  # large negative → near 0

    def test_smooth(self):
        x = np.linspace(-2, 2, 50)
        result = gelu(x)
        assert np.all(np.isfinite(result))


# ── silu ──────────────────────────────────────────────────────────────


class TestSiLU:
    def test_zero(self):
        x = np.array([0.0])
        result = silu(x)
        np.testing.assert_allclose(result, 0.0, atol=1e-5)

    def test_positive(self):
        x = np.array([1.0])
        result = silu(x)
        expected = 1.0 / (1.0 + np.exp(-1.0))
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_negative(self):
        x = np.array([-1.0])
        result = silu(x)
        expected = -1.0 / (1.0 + np.exp(1.0))
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_large_positive(self):
        x = np.array([100.0])
        result = silu(x)
        np.testing.assert_allclose(result, 100.0, atol=1e-3)

    def test_2d(self):
        x = np.random.randn(4, 8).astype(np.float32)
        result = silu(x)
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))


# ── rope ──────────────────────────────────────────────────────────────


class TestRope:
    def test_output_shape(self):
        x = np.random.randn(4, 2, 8).astype(np.float32)
        result = rope(x, pos=0, dim=8)
        assert result.shape == x.shape

    def test_position_shift(self):
        x = np.ones((2, 1, 4), dtype=np.float32)
        r0 = rope(x, pos=0, dim=4)
        r1 = rope(x, pos=1, dim=4)
        # Different positions should produce different outputs
        assert not np.allclose(r0, r1)

    def test_deterministic(self):
        x = np.random.randn(3, 2, 6).astype(np.float32)
        r1 = rope(x, pos=0, dim=6)
        r2 = rope(x, pos=0, dim=6)
        np.testing.assert_array_equal(r1, r2)

    def test_preserves_norm(self):
        x = np.random.randn(4, 2, 8).astype(np.float32)
        result = rope(x, pos=0, dim=8)
        # RoPE is a rotation — preserves vector norms
        norms_in = np.linalg.norm(x, axis=-1)
        norms_out = np.linalg.norm(result, axis=-1)
        np.testing.assert_allclose(norms_in, norms_out, atol=1e-5)


# ── to_float32 ────────────────────────────────────────────────────────


class TestToFloat32:
    def test_float32_passthrough(self):
        x = np.array([1.0, 2.0], dtype=np.float32)
        result = to_float32(x)
        assert result.dtype == np.float32

    def test_float16_convert(self):
        x = np.array([1.0, 2.0], dtype=np.float16)
        result = to_float32(x)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [1.0, 2.0], atol=0.01)

    def test_int32_converts(self):
        x = np.array([1, 2, 3], dtype=np.int32)
        result = to_float32(x)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])
