"""Tests for domains.inference.ops.layernorm and rmsnorm."""

from __future__ import annotations

import numpy as np
import pytest

from domains.inference.ops.layernorm import layernorm
from domains.inference.ops.rmsnorm import rmsnorm


class TestLayerNorm:
    def test_basic(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.ones(3)
        b = np.zeros(3)
        result = layernorm(x, w, b)
        mean = result.mean(axis=-1)
        assert np.allclose(mean, 0.0, atol=1e-5)

    def test_shape_preserved(self):
        x = np.random.randn(4, 8)
        w = np.ones(8)
        b = np.zeros(8)
        result = layernorm(x, w, b)
        assert result.shape == x.shape

    def test_with_weight(self):
        x = np.ones((2, 4))
        w = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.zeros(4)
        result = layernorm(x, w, b)
        assert result.shape == x.shape

    def test_with_bias(self):
        x = np.ones((2, 4))
        w = np.ones(4)
        b = np.array([1.0, 2.0, 3.0, 4.0])
        result = layernorm(x, w, b)
        assert result.shape == x.shape

    def test_eps_stability(self):
        x = np.zeros((2, 4))
        w = np.ones(4)
        b = np.zeros(4)
        result = layernorm(x, w, b, eps=1e-10)
        assert np.allclose(result, 0.0, atol=1e-5)


class TestRMSNorm:
    def test_basic(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.ones(3)
        result = rmsnorm(x, w)
        assert result.shape == x.shape

    def test_shape_preserved(self):
        x = np.random.randn(4, 8)
        w = np.ones(8)
        result = rmsnorm(x, w)
        assert result.shape == x.shape

    def test_with_weight(self):
        x = np.ones((2, 4))
        w = np.array([1.0, 2.0, 3.0, 4.0])
        result = rmsnorm(x, w)
        assert result.shape == x.shape

    def test_rms_normalization(self):
        x = np.array([[2.0, 4.0, 6.0]])
        w = np.ones(3)
        result = rmsnorm(x, w)
        rms = np.sqrt(np.mean(result ** 2, axis=-1))
        assert np.allclose(rms, 1.0, atol=1e-5)

    def test_eps_stability(self):
        x = np.zeros((2, 4))
        w = np.ones(4)
        result = rmsnorm(x, w, eps=1e-10)
        assert np.allclose(result, 0.0, atol=1e-5)
