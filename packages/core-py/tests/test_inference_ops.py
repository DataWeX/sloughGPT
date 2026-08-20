"""
Tests for inference ops: layernorm and rmsnorm.

Covers:
    - Basic correctness against reference implementations
    - Batch dimensions
    - Edge cases (constant input, zero input, large values)
    - Numerical stability (eps parameter)
    - Shape preservation
"""

import numpy as np
import pytest
import sys
from pathlib import Path

_CORE_PY = Path(__file__).resolve().parents[1]
if str(_CORE_PY) not in sys.path:
    sys.path.insert(0, str(_CORE_PY))

from domains.inference.ops.layernorm import layernorm
from domains.inference.ops.rmsnorm import rmsnorm


# ── Reference implementations ─────────────────────────────────────────


def _ref_layernorm(x, weight, bias, eps=1e-5):
    """Reference layer normalization."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def _ref_rmsnorm(x, weight, eps=1e-6):
    """Reference RMS normalization."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms * weight


# ── LayerNorm tests ───────────────────────────────────────────────────


class TestLayerNorm:
    def test_basic_1d(self):
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        result = layernorm(x, w, b)
        expected = _ref_layernorm(x, w, b)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_basic_2d(self):
        x = np.random.randn(8, 64).astype(np.float32)
        w = np.ones(64, dtype=np.float32)
        b = np.zeros(64, dtype=np.float32)
        result = layernorm(x, w, b)
        expected = _ref_layernorm(x, w, b)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_with_weight_and_bias(self):
        x = np.random.randn(4, 32).astype(np.float32)
        w = np.random.randn(32).astype(np.float32)
        b = np.random.randn(32).astype(np.float32)
        result = layernorm(x, w, b)
        expected = _ref_layernorm(x, w, b)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_batch_3d(self):
        x = np.random.randn(2, 4, 128).astype(np.float32)
        w = np.ones(128, dtype=np.float32)
        b = np.zeros(128, dtype=np.float32)
        result = layernorm(x, w, b)
        assert result.shape == x.shape
        expected = _ref_layernorm(x, w, b)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_output_zero_mean(self):
        x = np.random.randn(16, 64).astype(np.float32)
        w = np.ones(64, dtype=np.float32)
        b = np.zeros(64, dtype=np.float32)
        result = layernorm(x, w, b)
        means = result.mean(axis=-1)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)

    def test_output_unit_variance(self):
        x = np.random.randn(16, 64).astype(np.float32)
        w = np.ones(64, dtype=np.float32)
        b = np.zeros(64, dtype=np.float32)
        result = layernorm(x, w, b)
        variances = result.var(axis=-1)
        np.testing.assert_allclose(variances, 1.0, atol=1e-2)

    def test_constant_input(self):
        x = np.full((4, 32), 5.0, dtype=np.float32)
        w = np.ones(32, dtype=np.float32)
        b = np.zeros(32, dtype=np.float32)
        result = layernorm(x, w, b)
        # Constant input -> zero after mean subtraction -> zeros after div by sqrt(0+eps)
        np.testing.assert_allclose(result, 0.0, atol=1e-3)

    def test_shape_preserved(self):
        for shape in [(64,), (8, 64), (2, 4, 128), (1, 1, 1, 256)]:
            x = np.random.randn(*shape).astype(np.float32)
            w = np.ones(shape[-1], dtype=np.float32)
            b = np.zeros(shape[-1], dtype=np.float32)
            result = layernorm(x, w, b)
            assert result.shape == x.shape

    def test_eps_affects_output(self):
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        r1 = layernorm(x, w, b, eps=1e-5)
        r2 = layernorm(x, w, b, eps=1.0)
        # With eps=1.0, the normalization is much weaker
        assert not np.allclose(r1, r2)


# ── RMSNorm tests ─────────────────────────────────────────────────────


class TestRMSNorm:
    def test_basic_1d(self):
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        result = rmsnorm(x, w)
        expected = _ref_rmsnorm(x, w)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_basic_2d(self):
        x = np.random.randn(8, 64).astype(np.float32)
        w = np.ones(64, dtype=np.float32)
        result = rmsnorm(x, w)
        expected = _ref_rmsnorm(x, w)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_with_weight(self):
        x = np.random.randn(4, 32).astype(np.float32)
        w = np.random.randn(32).astype(np.float32)
        result = rmsnorm(x, w)
        expected = _ref_rmsnorm(x, w)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_batch_3d(self):
        x = np.random.randn(2, 4, 128).astype(np.float32)
        w = np.ones(128, dtype=np.float32)
        result = rmsnorm(x, w)
        assert result.shape == x.shape
        expected = _ref_rmsnorm(x, w)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_rms_of_output(self):
        x = np.random.randn(16, 64).astype(np.float32)
        w = np.ones(64, dtype=np.float32)
        result = rmsnorm(x, w)
        # RMS of output should be close to RMS of weight
        rms_out = np.sqrt(np.mean(result ** 2, axis=-1))
        rms_w = np.sqrt(np.mean(w ** 2))
        np.testing.assert_allclose(rms_out, rms_w, rtol=1e-4)

    def test_zero_input(self):
        x = np.zeros((4, 32), dtype=np.float32)
        w = np.ones(32, dtype=np.float32)
        result = rmsnorm(x, w)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_shape_preserved(self):
        for shape in [(64,), (8, 64), (2, 4, 128), (1, 1, 1, 256)]:
            x = np.random.randn(*shape).astype(np.float32)
            w = np.ones(shape[-1], dtype=np.float32)
            result = rmsnorm(x, w)
            assert result.shape == x.shape

    def test_eps_prevents_division_by_zero(self):
        x = np.zeros((4, 32), dtype=np.float32)
        w = np.ones(32, dtype=np.float32)
        # Should not raise
        result = rmsnorm(x, w, eps=1e-6)
        assert np.all(np.isfinite(result))

    def test_large_values_stable(self):
        x = np.random.randn(4, 32).astype(np.float32) * 1000
        w = np.ones(32, dtype=np.float32)
        result = rmsnorm(x, w)
        assert np.all(np.isfinite(result))
