"""Tests for training.slonet — activation ops, loss functions, utilities."""

from __future__ import annotations

import pytest
import numpy as np
from domains.training.slonet import (
    Tensor,
    relu,
    gelu,
    gelu_np,
    silu,
    silu_np,
    softmax,
    cross_entropy,
    mse_loss,
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
)


# ── relu ────────────────────────────────────────────────────────────────────


class TestRelu:

    def test_positive(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        out = relu(t)
        assert out.data.tolist() == [1.0, 2.0, 3.0]

    def test_negative(self):
        t = Tensor(np.array([-1.0, -2.0, 0.0]))
        out = relu(t)
        assert out.data.tolist() == [0.0, 0.0, 0.0]

    def test_backward(self):
        t = Tensor(np.array([-1.0, 2.0, -3.0]), requires_grad=True)
        out = relu(t)
        out.grad = Tensor(np.array([1.0, 1.0, 1.0]))
        out._backward_fn(out.grad.data)
        assert t.grad.data.tolist() == [0.0, 1.0, 0.0]


# ── gelu ────────────────────────────────────────────────────────────────────


class TestGelu:

    def test_numpy(self):
        d = np.array([-1.0, 0.0, 1.0])
        out = gelu_np(d)
        assert out.shape == d.shape
        assert out[1] == pytest.approx(0.0, abs=1e-5)

    def test_tensor(self):
        t = Tensor(np.array([-1.0, 0.0, 1.0]))
        out = gelu(t)
        assert isinstance(out, Tensor)
        assert out.shape == (3,)

    def test_backward(self):
        t = Tensor(np.array([0.5]), requires_grad=True)
        out = gelu(t)
        out.grad = Tensor(np.array([1.0]))
        out._backward_fn(out.grad.data)
        assert t.grad is not None


# ── silu ────────────────────────────────────────────────────────────────────


class TestSilu:

    def test_numpy(self):
        d = np.array([0.0, 1.0, -1.0])
        out = silu_np(d)
        assert out.shape == d.shape

    def test_tensor(self):
        t = Tensor(np.array([0.0, 1.0, -1.0]))
        out = silu(t)
        assert isinstance(out, Tensor)
        assert out.shape == (3,)

    def test_backward(self):
        t = Tensor(np.array([1.0]), requires_grad=True)
        out = silu(t)
        out.grad = Tensor(np.array([1.0]))
        out._backward_fn(out.grad.data)
        assert t.grad is not None


# ── softmax ─────────────────────────────────────────────────────────────────


class TestSoftmax:

    def test_sums_to_one(self):
        t = Tensor(np.array([[1.0, 2.0, 3.0]]))
        out = softmax(t, dim=-1)
        assert out.data.sum() == pytest.approx(1.0, rel=1e-5)

    def test_numpy(self):
        d = np.array([[1.0, 2.0, 3.0]])
        out = softmax(d, dim=-1)
        assert out.sum() == pytest.approx(1.0, rel=1e-5)

    def test_large_values(self):
        t = Tensor(np.array([[100.0, 200.0, 300.0]]))
        out = softmax(t, dim=-1)
        assert out.data.sum() == pytest.approx(1.0, rel=1e-5)


# ── cross_entropy ───────────────────────────────────────────────────────────


class TestCrossEntropy:

    def test_perfect_prediction(self):
        logits = Tensor(np.array([[10.0, 0.0, 0.0]]))
        targets = Tensor(np.array([0]))
        out = cross_entropy(logits, targets)
        assert out.item() < 0.01

    def test_backward(self):
        logits = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)
        targets = Tensor(np.array([2]))
        out = cross_entropy(logits, targets)
        out._backward_fn(np.array(1.0))
        assert logits.grad is not None

    def test_3d_logits(self):
        logits = Tensor(np.ones((2, 3, 4)))
        targets = Tensor(np.array([0, 1, 2, 0, 1, 2]))
        out = cross_entropy(logits, targets)
        assert out.item() > 0
        assert out.shape == ()


# ── mse_loss ────────────────────────────────────────────────────────────────


class TestMseLoss:

    def test_perfect(self):
        pred = Tensor(np.array([1.0, 2.0, 3.0]))
        target = Tensor(np.array([1.0, 2.0, 3.0]))
        out = mse_loss(pred, target)
        assert out.item() == pytest.approx(0.0, abs=1e-5)

    def test_imperfect(self):
        pred = Tensor(np.array([1.0, 2.0]))
        target = Tensor(np.array([2.0, 3.0]))
        out = mse_loss(pred, target)
        assert out.item() > 0


# ── topk ────────────────────────────────────────────────────────────────────


class TestTopk:

    def test_basic(self):
        t = Tensor(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))
        values, indices = topk(t, k=2)
        assert values.data.tolist() == [[5.0, 4.0]]
        assert indices.data.tolist() == [[4, 2]]


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


# ── _NoGrad ─────────────────────────────────────────────────────────────────


class TestNoGradClass:

    def test_context(self):
        from domains.training.slonet import _NoGrad
        with _NoGrad():
            pass

    def test_decorator(self):
        from domains.training.slonet import _NoGrad
        @_NoGrad()
        def fn():
            return 42
        assert fn() == 42
