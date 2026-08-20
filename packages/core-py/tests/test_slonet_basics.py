"""Tests for domains.training.slonet — Tensor creation helpers, activations,
broadcast functions, basic math operations.

Covers: zeros/ones/randn/tensor, sigmoid/tanh/relu/gelu/silu, softmax,
broadcast_back/broadcast_forward, _ensure, topk, cross_entropy, mse_loss.
Excludes: full forward/backward pass through model layers (covered elsewhere).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.training.slonet import (
    Tensor,
    zeros, ones, randn, tensor,
    sigmoid, tanh, relu, gelu_np, gelu, silu_np, silu,
    softmax, cross_entropy, mse_loss,
    _broadcast_back, _broadcast_forward, _ensure,
    topk,
)


# ── Tensor creation ──────────────────────────────────────────────────

class TestTensorCreation:
    def test_zeros(self):
        t = zeros((3, 4))
        assert t.shape == (3, 4)
        assert np.allclose(t.data, 0.0)

    def test_ones(self):
        t = ones((2, 3))
        assert t.shape == (2, 3)
        assert np.allclose(t.data, 1.0)

    def test_randn(self):
        t = randn((5,))
        assert t.shape == (5,)
        assert t.data.std() > 0.1

    def test_tensor(self):
        t = tensor(np.array([1.0, 2.0, 3.0]))
        assert t.shape == (3,)
        np.testing.assert_array_almost_equal(t.data, [1.0, 2.0, 3.0])


# ── Activations ──────────────────────────────────────────────────────

class TestSigmoid:
    def test_zero(self):
        t = sigmoid(tensor(np.array([0.0])))
        assert abs(t.data[0] - 0.5) < 1e-5

    def test_range(self):
        t = sigmoid(tensor(np.array([-10.0, 0.0, 10.0])))
        assert 0.0 < t.data[0] < 0.5
        assert abs(t.data[1] - 0.5) < 1e-5
        assert 0.5 < t.data[2] < 1.0


class TestTanh:
    def test_zero(self):
        t = tanh(tensor(np.array([0.0])))
        assert abs(t.data[0]) < 1e-5

    def test_range(self):
        t = tanh(tensor(np.array([-10.0, 10.0])))
        assert t.data[0] < 0.0
        assert t.data[1] > 0.0


class TestRelu:
    def test_positive(self):
        t = relu(tensor(np.array([1.0, 2.0])))
        np.testing.assert_array_almost_equal(t.data, [1.0, 2.0])

    def test_negative(self):
        t = relu(tensor(np.array([-1.0, -2.0])))
        np.testing.assert_array_almost_equal(t.data, [0.0, 0.0])

    def test_mixed(self):
        t = relu(tensor(np.array([-1.0, 0.0, 1.0])))
        np.testing.assert_array_almost_equal(t.data, [0.0, 0.0, 1.0])


class TestGeluNp:
    def test_zero(self):
        assert abs(gelu_np(np.array([0.0]))[0]) < 1e-5

    def test_positive(self):
        v = gelu_np(np.array([1.0]))[0]
        assert 0.0 < v < 1.0


class TestSiluNp:
    def test_zero(self):
        assert abs(silu_np(np.array([0.0]))[0]) < 1e-5

    def test_large_positive(self):
        v = silu_np(np.array([10.0]))[0]
        assert v == pytest.approx(10.0, abs=0.1)


class TestSoftmax:
    def test_sums_to_one(self):
        t = softmax(tensor(np.array([1.0, 2.0, 3.0])))
        assert abs(t.data.sum() - 1.0) < 1e-5

    def test_2d(self):
        t = softmax(tensor(np.array([[1.0, 2.0], [3.0, 4.0]])), dim=1)
        np.testing.assert_allclose(t.data.sum(axis=1), [1.0, 1.0], rtol=1e-5)


# ── Broadcast helpers ────────────────────────────────────────────────

class TestBroadcastBack:
    def test_same_shape(self):
        g = np.array([1.0, 2.0, 3.0])
        result = _broadcast_back(g, (3,))
        np.testing.assert_array_equal(result, g)

    def test_sum_axis(self):
        g = np.ones((2, 3))
        result = _broadcast_back(g, (3,))
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, [2.0, 2.0, 2.0])

    def test_returns_copy(self):
        g = np.array([1.0, 2.0])
        result = _broadcast_back(g, (2,))
        g[0] = 999.0
        assert result[0] != 999.0


class TestBroadcastForward:
    def test_same_shape(self):
        t = np.array([1.0, 2.0])
        result = _broadcast_forward(t, (2,))
        np.testing.assert_array_equal(result, t)

    def test_expand(self):
        t = np.array([1.0, 2.0])
        result = _broadcast_forward(t, (3, 2))
        assert result.shape == (3, 2)


# ── Losses ───────────────────────────────────────────────────────────

class TestCrossEntropy:
    def test_basic(self):
        logits = tensor(np.array([[1.0, 2.0, 3.0]]))
        targets = tensor(np.array([2]))
        loss = cross_entropy(logits, targets)
        assert isinstance(loss, Tensor)
        assert loss.data > 0

    def test_perfect(self):
        logits = tensor(np.array([[0.0, 100.0, 0.0]]))
        targets = tensor(np.array([1]))
        loss = cross_entropy(logits, targets)
        assert loss.data < 0.01


class TestMseLoss:
    def test_perfect(self):
        pred = tensor(np.array([1.0, 2.0]))
        target = tensor(np.array([1.0, 2.0]))
        loss = mse_loss(pred, target)
        assert loss.data < 1e-5

    def test_imperfect(self):
        pred = tensor(np.array([1.0, 2.0]))
        target = tensor(np.array([2.0, 3.0]))
        loss = mse_loss(pred, target)
        assert loss.data > 0


# ── _ensure ──────────────────────────────────────────────────────────

class TestEnsure:
    def test_tensor_passthrough(self):
        t = tensor(np.array([1.0]))
        assert _ensure(t) is t

    def test_array_conversion(self):
        arr = np.array([1.0, 2.0])
        result = _ensure(arr)
        assert isinstance(result, Tensor)


class TestTopk:
    def test_basic(self):
        t = tensor(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))
        vals, idxs = topk(t, 2)
        assert vals.data.shape == (1, 2)
        assert vals.data[0, 0] >= vals.data[0, 1]
