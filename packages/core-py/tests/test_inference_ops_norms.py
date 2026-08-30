"""Tests for inference/ops/layernorm.py and inference/ops/rmsnorm.py."""

import numpy as np
import pytest

from domains.inference.ops.layernorm import layernorm
from domains.inference.ops.rmsnorm import rmsnorm


def _expected_layernorm(x, weight, bias, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def _expected_rmsnorm(x, weight, eps):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms * weight


# ── LayerNorm ────────────────────────────────────────────────────────────────

class TestLayerNormNormalization:
    def test_normalizes_mean_to_zero(self):
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = layernorm(x, np.ones(3), np.zeros(3))
        assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-6)

    def test_normalizes_variance_to_one(self):
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = layernorm(x, np.ones(3), np.zeros(3))
        assert np.allclose(out.var(axis=-1), 1.0, atol=1e-4)

    def test_single_row(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        out = layernorm(x, np.ones(4), np.zeros(4))
        assert np.allclose(out.mean(), 0.0, atol=1e-6)
        assert np.allclose(out.var(), 1.0, atol=1e-4)


class TestLayerNormManualFormula:
    def test_matches_expected(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 8))
        weight = rng.normal(size=(8,))
        bias = rng.normal(size=(8,))
        out = layernorm(x, weight, bias, eps=1e-5)
        assert np.allclose(out, _expected_layernorm(x, weight, bias, 1e-5))

    def test_large_values(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(3, 16)) * 1000
        weight = np.ones(16)
        bias = np.zeros(16)
        out = layernorm(x, weight, bias)
        assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-5)

    def test_negative_values(self):
        x = np.array([[-3.0, -1.0, 1.0, 3.0]])
        out = layernorm(x, np.ones(4), np.zeros(4))
        assert np.allclose(out.mean(), 0.0, atol=1e-6)


class TestLayerNormWeightBias:
    def test_applies_weight(self):
        x = np.array([[1.0, 2.0]])
        weight = np.array([3.0, 3.0])
        bias = np.zeros(2)
        out = layernorm(x, weight, bias)
        expected = _expected_layernorm(x, weight, bias, 1e-5)
        assert np.allclose(out, expected)

    def test_applies_bias(self):
        x = np.array([[1.0, 2.0]])
        weight = np.ones(2)
        bias = np.array([10.0, -10.0])
        out = layernorm(x, weight, bias)
        expected = _expected_layernorm(x, weight, bias, 1e-5)
        assert np.allclose(out, expected)

    def test_weight_and_bias_combined(self):
        x = np.array([[0.0, 1.0]])
        weight = np.array([2.0, 2.0])
        bias = np.array([10.0, 10.0])
        out = layernorm(x, weight, bias)
        assert np.allclose(out, _expected_layernorm(x, weight, bias, 1e-5))

    def test_zero_weight(self):
        x = np.array([[1.0, 2.0, 3.0]])
        weight = np.zeros(3)
        bias = np.zeros(3)
        out = layernorm(x, weight, bias)
        assert np.allclose(out, 0.0)


class TestLayerNormEdgeCases:
    def test_epsilon_prevents_div_by_zero(self):
        x = np.full((2, 3), 5.0)
        out = layernorm(x, np.ones(3), np.zeros(3), eps=1e-5)
        assert np.all(np.isfinite(out))
        assert np.allclose(out, 0.0)

    def test_single_element_last_dim(self):
        x = np.array([[3.0], [7.0]])
        out = layernorm(x, np.ones(1), np.zeros(1))
        assert out.shape == (2, 1)
        assert np.allclose(out, 0.0)

    def test_single_row_input(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        out = layernorm(x, np.ones(4), np.zeros(4))
        assert out.shape == (4,)
        assert np.allclose(out.mean(), 0.0, atol=1e-6)

    def test_large_epsilon(self):
        x = np.array([[1.0, 2.0, 3.0]])
        out = layernorm(x, np.ones(3), np.zeros(3), eps=100.0)
        assert np.all(np.isfinite(out))

    def test_small_epsilon(self):
        x = np.array([[1.0, 2.0, 3.0]])
        out = layernorm(x, np.ones(3), np.zeros(3), eps=1e-10)
        assert np.all(np.isfinite(out))


class TestLayerNormShapes:
    def test_2d_batch(self):
        x = np.random.randn(5, 8)
        out = layernorm(x, np.ones(8), np.zeros(8))
        assert out.shape == (5, 8)

    def test_1d_input(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        out = layernorm(x, np.ones(4), np.zeros(4))
        assert out.shape == (4,)

    def test_preserves_shape(self):
        x = np.random.randn(3, 7)
        w = np.ones(7)
        b = np.zeros(7)
        out = layernorm(x, w, b)
        assert out.shape == x.shape

    def test_large_last_dim(self):
        x = np.random.randn(2, 512)
        out = layernorm(x, np.ones(512), np.zeros(512))
        assert out.shape == (2, 512)


class TestLayerNormOutputType:
    def test_returns_float64(self):
        x = np.array([[1.0, 2.0]], dtype=np.float64)
        out = layernorm(x, np.ones(2), np.zeros(2))
        assert out.dtype == np.float64

    def test_returns_float32(self):
        x = np.array([[1.0, 2.0]], dtype=np.float32)
        out = layernorm(x, np.ones(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
        assert out.dtype == np.float32


# ── RMSNorm ──────────────────────────────────────────────────────────────────

class TestRMSNormManualFormula:
    def test_matches_expected(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(3, 6))
        weight = rng.normal(size=(6,))
        out = rmsnorm(x, weight, eps=1e-6)
        assert np.allclose(out, _expected_rmsnorm(x, weight, 1e-6))

    def test_large_values(self):
        rng = np.random.default_rng(99)
        x = rng.normal(size=(2, 8)) * 1000
        weight = np.ones(8)
        out = rmsnorm(x, weight)
        expected = _expected_rmsnorm(x, weight, 1e-6)
        assert np.allclose(out, expected)

    def test_negative_values(self):
        x = np.array([[-2.0, -4.0]])
        weight = np.ones(2)
        out = rmsnorm(x, weight)
        expected = _expected_rmsnorm(x, weight, 1e-6)
        assert np.allclose(out, expected)


class TestRMSNormDirection:
    def test_keeps_direction_of_input(self):
        x = np.array([[2.0, 4.0]])
        weight = np.ones(2)
        out = rmsnorm(x, weight)
        assert out[0, 0] > 0 and out[0, 1] > 0
        ratio = out[0, 1] / out[0, 0]
        assert np.isclose(ratio, 2.0)

    def test_preserves_ratios(self):
        x = np.array([[1.0, 3.0, 5.0]])
        weight = np.ones(3)
        out = rmsnorm(x, weight)
        assert np.isclose(out[0, 1] / out[0, 0], 3.0)
        assert np.isclose(out[0, 2] / out[0, 0], 5.0)


class TestRMSNormWeight:
    def test_applies_weight(self):
        x = np.array([[1.0, 1.0]])
        weight = np.array([5.0, 1.0])
        out = rmsnorm(x, weight)
        assert np.allclose(out[0, 0], 5.0 * out[0, 1])

    def test_weight_identity(self):
        x = np.array([[3.0, 6.0]])
        weight = np.ones(2)
        out = rmsnorm(x, weight)
        expected = _expected_rmsnorm(x, weight, 1e-6)
        assert np.allclose(out, expected)

    def test_weight_scales_output(self):
        x = np.array([[1.0, 2.0]])
        w1 = np.ones(2)
        w2 = np.ones(2) * 2.0
        out1 = rmsnorm(x, w1)
        out2 = rmsnorm(x, w2)
        assert np.allclose(out2, out1 * 2.0)


class TestRMSNormEdgeCases:
    def test_zero_input_produces_zero_output(self):
        x = np.zeros((2, 4))
        out = rmsnorm(x, np.ones(4), eps=1e-6)
        assert np.allclose(out, 0.0)

    def test_epsilon_stability(self):
        x = np.full((2, 3), 1e-8)
        out = rmsnorm(x, np.ones(3), eps=1e-6)
        assert np.all(np.isfinite(out))

    def test_large_epsilon(self):
        x = np.array([[1.0, 2.0, 3.0]])
        out = rmsnorm(x, np.ones(3), eps=100.0)
        assert np.all(np.isfinite(out))

    def test_small_epsilon(self):
        x = np.array([[1.0, 2.0, 3.0]])
        out = rmsnorm(x, np.ones(3), eps=1e-10)
        assert np.all(np.isfinite(out))


class TestRMSNormShapes:
    def test_single_row_input(self):
        x = np.array([1.0, 2.0, 3.0])
        out = rmsnorm(x, np.ones(3))
        assert out.shape == (3,)

    def test_2d_batch(self):
        x = np.random.randn(5, 8)
        out = rmsnorm(x, np.ones(8))
        assert out.shape == (5, 8)

    def test_preserves_shape(self):
        x = np.random.randn(4, 12)
        w = np.ones(12)
        out = rmsnorm(x, w)
        assert out.shape == x.shape

    def test_large_last_dim(self):
        x = np.random.randn(2, 512)
        out = rmsnorm(x, np.ones(512))
        assert out.shape == (2, 512)


class TestRMSNormOutputType:
    def test_returns_float64(self):
        x = np.array([[1.0, 2.0]], dtype=np.float64)
        out = rmsnorm(x, np.ones(2))
        assert out.dtype == np.float64

    def test_returns_float32(self):
        x = np.array([[1.0, 2.0]], dtype=np.float32)
        out = rmsnorm(x, np.ones(2, dtype=np.float32))
        assert out.dtype == np.float32


class TestRMSNormDeterministic:
    def test_same_input_same_output(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.ones(3)
        out1 = rmsnorm(x, w)
        out2 = rmsnorm(x, w)
        assert np.array_equal(out1, out2)


# ── Additional cross-cutting tests ──────────────────────────────────────────

class TestLayerNormInputDtype:
    def test_int_input_cast_to_float(self):
        x = np.array([[1, 2, 3]], dtype=np.int32)
        out = layernorm(x.astype(np.float64), np.ones(3), np.zeros(3))
        assert out.dtype == np.float64


class TestRMSNormInputDtype:
    def test_int_input_cast_to_float(self):
        x = np.array([[1, 2, 3]], dtype=np.int32)
        out = rmsnorm(x.astype(np.float64), np.ones(3))
        assert out.dtype == np.float64


class TestLayerNormNumericalStability:
    def test_all_same_value(self):
        x = np.full((3, 4), 7.0)
        out = layernorm(x, np.ones(4), np.zeros(4))
        assert np.allclose(out, 0.0)

    def test_very_small_values(self):
        x = np.full((2, 3), 1e-12)
        out = layernorm(x, np.ones(3), np.zeros(3), eps=1e-5)
        assert np.all(np.isfinite(out))

    def test_very_large_values(self):
        x = np.full((2, 3), 1e12)
        out = layernorm(x, np.ones(3), np.zeros(3))
        assert np.allclose(out, 0.0, atol=1e-4)


class TestRMSNormNumericalStability:
    def test_all_same_small_value(self):
        x = np.full((2, 3), 1e-12)
        out = rmsnorm(x, np.ones(3), eps=1e-6)
        assert np.all(np.isfinite(out))

    def test_all_same_large_value(self):
        x = np.full((2, 3), 1e12)
        out = rmsnorm(x, np.ones(3))
        assert np.all(np.isfinite(out))


class TestLayerNormGradientBehavior:
    """Check layernorm has correct gradient-like properties."""

    def test_output_bounded_with_weight(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(4, 8))
        weight = np.ones(8) * 0.1
        bias = np.zeros(8)
        out = layernorm(x, weight, bias)
        assert out.max() < 1.0


class TestRMSNormGradientBehavior:
    """Check rmsnorm has correct gradient-like properties."""

    def test_output_direction_invariant_to_scale(self):
        x = np.array([[1.0, 2.0]])
        out1 = rmsnorm(x, np.ones(2))
        out2 = rmsnorm(x * 5.0, np.ones(2))
        ratio1 = out1[0, 1] / out1[0, 0]
        ratio2 = out2[0, 1] / out2[0, 0]
        assert np.isclose(ratio1, ratio2)


class TestLayerNormConsistency:
    def test_batch_independence(self):
        """Each row in batch is normalized independently."""
        x = np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 0.0]])
        out = layernorm(x, np.ones(3), np.zeros(3))
        assert np.allclose(out[0].mean(), 0.0, atol=1e-6)
        assert np.allclose(out[1].mean(), 0.0, atol=1e-6)
        assert not np.allclose(out[0], out[1])


class TestRMSNormConsistency:
    def test_batch_independence(self):
        """Each row in batch is normalized independently."""
        x = np.array([[1.0, 0.0], [0.0, 1.0]])
        out = rmsnorm(x, np.ones(2))
        assert not np.allclose(out[0], out[1])
