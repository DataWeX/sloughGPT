"""Tests for training.slonet — additional utility functions."""

from __future__ import annotations

import math
import pytest
import numpy as np
from domains.training.slonet import (
    Tensor,
    topk,
    multinomial,
    stack,
    concatenate,
    randint,
    exp,
    isfinite,
    where,
    no_grad,
    is_cuda,
    is_mps,
    cpu,
    _apply_temperature,
    _apply_top_k,
    _apply_top_p,
    _apply_repetition_penalty,
    _to_np,
    _broadcast_back,
    _broadcast_forward,
)


# ── _to_np ──────────────────────────────────────────────────────────────────


class TestToNp:

    def test_tensor(self):
        t = Tensor(np.array([1.0, 2.0]))
        result = _to_np(t)
        assert isinstance(result, np.ndarray)

    def test_array(self):
        arr = np.array([1.0, 2.0])
        result = _to_np(arr)
        assert isinstance(result, np.ndarray)


# ── _broadcast_back ─────────────────────────────────────────────────────────


class TestBroadcastBack:

    def test_same_shape(self):
        g = np.ones((2, 3))
        result = _broadcast_back(g, (2, 3))
        assert result.shape == (2, 3)

    def test_sum_axes(self):
        g = np.ones((2, 3, 4))
        result = _broadcast_back(g, (3, 4))
        assert result.shape == (3, 4)

    def test_expand_dim(self):
        g = np.ones((3,))
        result = _broadcast_back(g, (1, 3))
        assert result.shape == (1, 3)

    def test_broadcast_to(self):
        g = np.ones((1, 3))
        result = _broadcast_back(g, (2, 3))
        assert result.shape == (2, 3)


# ── _broadcast_forward ─────────────────────────────────────────────────────


class TestBroadcastForward:

    def test_same_shape(self):
        t = np.ones((2, 3))
        result = _broadcast_forward(t, (2, 3))
        assert result.shape == (2, 3)

    def test_expand(self):
        t = np.ones((3,))
        result = _broadcast_forward(t, (2, 3))
        assert result.shape == (2, 3)


# ── topk ────────────────────────────────────────────────────────────────────


class TestTopk:

    def test_basic(self):
        t = Tensor(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))
        values, indices = topk(t, k=2)
        assert values.data.tolist() == [[5.0, 4.0]]
        assert indices.data.tolist() == [[4, 2]]

    def test_k1(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        values, indices = topk(t, k=1)
        assert values.data.tolist() == [[3.0]]
        assert indices.data.tolist() == [[2]]


# ── multinomial ─────────────────────────────────────────────────────────────


class TestMultinomial:

    def test_basic(self):
        t = Tensor(np.array([0.1, 0.5, 0.4]))
        out = multinomial(t, num_samples=1)
        assert out.shape == (1, 1)

    def test_uniform(self):
        t = Tensor(np.array([1.0, 1.0, 1.0]))
        out = multinomial(t, num_samples=3)
        assert out.shape == (1, 3)


# ── stack / concatenate ────────────────────────────────────────────────────


class TestStackConcat:

    def test_stack(self):
        t1 = Tensor(np.array([1.0, 2.0]))
        t2 = Tensor(np.array([3.0, 4.0]))
        out = stack([t1, t2], dim=0)
        assert out.shape == (2, 2)

    def test_concatenate(self):
        t1 = Tensor(np.array([[1.0, 2.0]]))
        t2 = Tensor(np.array([[3.0, 4.0]]))
        out = concatenate([t1, t2], dim=-1)
        assert out.shape == (1, 4)


# ── randint / exp / isfinite / where ───────────────────────────────────────


class TestUtilities:

    def test_randint(self):
        out = randint(0, 10, (2, 3))
        assert out.shape == (2, 3)
        assert out.data.dtype == np.float32

    def test_exp(self):
        t = Tensor(np.array([0.0, 1.0]))
        out = exp(t)
        assert out.data[0] == pytest.approx(1.0)
        assert out.data[1] == pytest.approx(np.e, rel=1e-4)

    def test_isfinite(self):
        t = Tensor(np.array([1.0, np.inf, np.nan]))
        result = isfinite(t)
        assert result.tolist() == [True, False, False]

    def test_where(self):
        cond = Tensor(np.array([1.0, 0.0, 1.0]))
        a = Tensor(np.array([10.0, 20.0, 30.0]))
        b = Tensor(np.array([1.0, 2.0, 3.0]))
        out = where(cond, a, b)
        assert out.data.tolist() == [10.0, 2.0, 30.0]


# ── is_cuda / is_mps / cpu ─────────────────────────────────────────────────


class TestDeviceUtils:

    def test_is_cuda(self):
        assert is_cuda(Tensor([1.0])) is False

    def test_is_mps(self):
        assert is_mps(Tensor([1.0])) is False

    def test_cpu(self):
        t = Tensor([1.0])
        assert cpu(t) is t


# ── logit processors ───────────────────────────────────────────────────────


class TestLogitProcessors:

    def test_temperature(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        out = _apply_temperature(logits, 2.0)
        assert out[0, 0] == pytest.approx(0.5)

    def test_temperature_zero(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        out = _apply_temperature(logits, 0.0)
        assert out[0, 0] == pytest.approx(1.0)

    def test_top_k(self):
        logits = np.array([[1.0, 5.0, 3.0, 2.0]])
        out = _apply_top_k(logits, k=2)
        assert out[0, 1] == 5.0
        assert out[0, 0] < 0

    def test_top_k_full(self):
        logits = np.array([[1.0, 2.0]])
        out = _apply_top_k(logits, k=10)
        assert out[0, 0] == 1.0

    def test_top_p(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        out = _apply_top_p(logits, p=0.5)
        assert out[0, 2] == 3.0

    def test_top_p_full(self):
        logits = np.array([[1.0, 2.0]])
        out = _apply_top_p(logits, p=1.0)
        assert out[0, 0] == 1.0

    def test_repetition_penalty(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        out = _apply_repetition_penalty(logits, np.array([0]), penalty=2.0)
        assert out[0, 0] == 0.5  # penalized

    def test_repetition_penalty_noop(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        out = _apply_repetition_penalty(logits, np.array([]), penalty=2.0)
        assert out[0, 0] == 1.0


# ── no_grad ─────────────────────────────────────────────────────────────────


class TestNoGrad:

    def test_context_manager(self):
        import domains.training.slonet as mod
        old = mod._NO_GRAD
        with no_grad():
            assert mod._NO_GRAD is True
        assert mod._NO_GRAD is old

    def test_decorator(self):
        import domains.training.slonet as mod
        old = mod._NO_GRAD

        @no_grad()
        def fn():
            return mod._NO_GRAD

        assert fn() is True
        assert mod._NO_GRAD is old
