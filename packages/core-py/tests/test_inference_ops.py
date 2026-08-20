"""Tests for domains.inference.ops — rmsnorm, layernorm, matmul (numpy fallback path).

Covers: normalization correctness, matmul with various dtypes, shape preservation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.inference.ops.rmsnorm import rmsnorm
from domains.inference.ops.layernorm import layernorm
from domains.inference.ops.matmul import matmul


class TestOpsRmsnorm:
    def test_basic(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.ones(3)
        result = rmsnorm(x, w)
        assert result.shape == x.shape
        # Output should have unit RMS
        rms = np.sqrt(np.mean(result ** 2))
        assert rms == pytest.approx(1.0, abs=0.01)

    def test_weighted(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.array([2.0, 2.0, 2.0])
        result = rmsnorm(x, w)
        rms = np.sqrt(np.mean(result ** 2))
        assert rms == pytest.approx(2.0, abs=0.01)

    def test_2d_batch(self):
        x = np.ones((4, 8))
        w = np.ones(8)
        result = rmsnorm(x, w)
        assert result.shape == (4, 8)

    def test_custom_eps(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.ones(3)
        result = rmsnorm(x, w, eps=1e-3)
        assert result.shape == x.shape


class TestOpsLayernorm:
    def test_basic(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.ones(3)
        b = np.zeros(3)
        result = layernorm(x, w, b)
        assert result.shape == x.shape
        # Output should be zero-mean
        assert result.mean() == pytest.approx(0.0, abs=1e-5)

    def test_weighted(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.array([2.0, 2.0, 2.0])
        b = np.array([1.0, 1.0, 1.0])
        result = layernorm(x, w, b)
        assert result.mean() == pytest.approx(1.0, abs=1e-5)

    def test_2d_batch(self):
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        result = layernorm(x, w, b)
        assert result.shape == (4, 8)


class TestOpsMatmul:
    def test_basic(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = matmul(a, b)
        expected = a @ b
        np.testing.assert_array_almost_equal(result, expected)

    def test_shapes(self):
        a = np.ones((2, 3))
        b = np.ones((3, 4))
        result = matmul(a, b)
        assert result.shape == (2, 4)

    def test_float32(self):
        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((3, 2), dtype=np.float32)
        result = matmul(a, b)
        assert result.dtype in (np.float32, np.float64)

    def test_integer(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = matmul(a, b)
        np.testing.assert_array_equal(result, [[19, 22], [43, 50]])
