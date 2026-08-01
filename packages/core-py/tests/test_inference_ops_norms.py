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


class TestLayerNorm:
    def test_normalizes_mean_and_variance(self):
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        weight = np.ones(3)
        bias = np.zeros(3)
        out = layernorm(x, weight, bias)
        assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-6)
        assert np.allclose(out.var(axis=-1), 1.0, atol=1e-4)

    def test_matches_manual_formula(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 8))
        weight = rng.normal(size=(8,))
        bias = rng.normal(size=(8,))
        out = layernorm(x, weight, bias, eps=1e-5)
        assert np.allclose(out, _expected_layernorm(x, weight, bias, 1e-5))

    def test_applies_weight_and_bias(self):
        x = np.array([[0.0, 1.0]])
        weight = np.array([2.0, 2.0])
        bias = np.array([10.0, 10.0])
        out = layernorm(x, weight, bias)
        assert np.allclose(out, _expected_layernorm(x, weight, bias, 1e-5))

    def test_epsilon_prevents_div_by_zero(self):
        x = np.full((2, 3), 5.0)
        weight = np.ones(3)
        bias = np.zeros(3)
        out = layernorm(x, weight, bias, eps=1e-5)
        assert np.all(np.isfinite(out))
        assert np.allclose(out, 0.0)

    def test_single_element_last_dim(self):
        x = np.array([[3.0], [7.0]])
        out = layernorm(x, np.ones(1), np.zeros(1))
        assert out.shape == (2, 1)
        assert np.allclose(out, 0.0)

    def test_single_row_input(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        weight = np.ones(4)
        bias = np.zeros(4)
        out = layernorm(x, weight, bias)
        assert out.shape == (4,)
        assert np.allclose(out.mean(), 0.0, atol=1e-6)


class TestRMSNorm:
    def test_matches_manual_formula(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(3, 6))
        weight = rng.normal(size=(6,))
        out = rmsnorm(x, weight, eps=1e-6)
        assert np.allclose(out, _expected_rmsnorm(x, weight, 1e-6))

    def test_keeps_direction_of_input(self):
        x = np.array([[2.0, 4.0]])
        weight = np.ones(2)
        out = rmsnorm(x, weight)
        assert out[0, 0] > 0 and out[0, 1] > 0
        ratio = out[0, 1] / out[0, 0]
        assert np.isclose(ratio, 2.0)

    def test_applies_weight(self):
        x = np.array([[1.0, 1.0]])
        weight = np.array([5.0, 1.0])
        out = rmsnorm(x, weight)
        assert np.allclose(out[0, 0], 5.0 * out[0, 1])

    def test_zero_input_produces_zero_output(self):
        x = np.zeros((2, 4))
        out = rmsnorm(x, np.ones(4), eps=1e-6)
        assert np.allclose(out, 0.0)

    def test_epsilon_stability(self):
        x = np.full((2, 3), 1e-8)
        out = rmsnorm(x, np.ones(3), eps=1e-6)
        assert np.all(np.isfinite(out))

    def test_single_row_input(self):
        x = np.array([1.0, 2.0, 3.0])
        out = rmsnorm(x, np.ones(3))
        assert out.shape == (3,)
