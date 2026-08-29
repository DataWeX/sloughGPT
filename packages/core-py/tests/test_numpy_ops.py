"""Tests for domains.infrastructure.numpy_ops — pure NumPy transformer operations.

Covers: softmax, rmsnorm, layer_norm, gelu, silu, rope, to_float32.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.infrastructure.numpy_ops import (
    to_float32,
    softmax,
    rmsnorm,
    layer_norm,
    gelu,
    silu,
    rope,
)


class TestToFloat32:
    def test_already_float32(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = to_float32(x)
        assert result.dtype == np.float32

    def test_float16(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        result = to_float32(x)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0], rtol=1e-3)

    def test_int_to_float32(self):
        x = np.array([1, 2, 3], dtype=np.int32)
        result = to_float32(x)
        assert result.dtype == np.float32


class TestSoftmax:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert result.shape == x.shape
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)

    def test_stable(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        assert result.sum() == pytest.approx(1.0, abs=1e-5)

    def test_2d(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = softmax(x, axis=1)
        np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0], rtol=1e-5)


class TestRmsnorm:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        result = rmsnorm(x, w)
        assert result.shape == x.shape
        # RMS norm of unit vector should be ~unit
        assert np.abs(np.sqrt(np.mean(result ** 2)) - 1.0) < 0.1

    def test_weighted(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.array([2.0, 2.0, 2.0])
        result = rmsnorm(x, w)
        # With weight=2, output should be ~2x
        assert np.sqrt(np.mean(result ** 2)) > 1.0


class TestLayerNorm:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        b = np.zeros(3)
        result = layer_norm(x, w, b)
        assert result.shape == x.shape
        np.testing.assert_allclose(result.mean(), 0.0, atol=1e-5)

    def test_weighted(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.array([2.0, 2.0, 2.0])
        b = np.array([1.0, 1.0, 1.0])
        result = layer_norm(x, w, b)
        np.testing.assert_allclose(result.mean(), 1.0, atol=1e-5)


class TestGelu:
    def test_basic(self):
        x = np.array([-1.0, 0.0, 1.0])
        result = gelu(x)
        assert result.shape == x.shape
        # GELU(0) = 0
        assert result[1] == pytest.approx(0.0, abs=1e-5)
        # GELU(1) > 0
        assert result[2] > 0

    def test_negative(self):
        x = np.array([-2.0])
        result = gelu(x)
        assert result[0] < 0


class TestSilu:
    def test_basic(self):
        x = np.array([-1.0, 0.0, 1.0])
        result = silu(x)
        assert result.shape == x.shape
        # SiLU(0) = 0
        assert result[1] == pytest.approx(0.0, abs=1e-5)
        # SiLU(1) > 0
        assert result[2] > 0

    def test_large_positive(self):
        x = np.array([100.0])
        result = silu(x)
        assert result[0] == pytest.approx(100.0, abs=0.1)


class TestRope:
    def test_basic(self):
        x = np.ones((4, 2, 8))
        result = rope(x, pos=0, dim=8)
        assert result.shape == x.shape

    def test_different_positions(self):
        x = np.ones((2, 1, 4))
        r1 = rope(x, pos=0, dim=4)
        r2 = rope(x, pos=10, dim=4)
        # Different positions should give different outputs
        assert not np.allclose(r1, r2)

    def test_preserves_shape(self):
        x = np.random.randn(3, 4, 6)
        result = rope(x, pos=5, dim=6)
        assert result.shape == x.shape
