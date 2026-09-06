"""Comprehensive tests for slonet.py — Tensor, autograd ops, logit processors,
GenerationMetrics, GenerateResult, no_grad, broadcast helpers.

Covers: Tensor creation, arithmetic, comparison, backward, forward_grad,
jvp, logit processors, GenerationMetrics.finalize, GenerateResult slicing.
"""
from __future__ import annotations

import numpy as np
import pytest

from domains.training.slonet import (
    Tensor,
    GenerationMetrics,
    GenerateResult,
    no_grad,
    _broadcast_back,
    _broadcast_forward,
    _ensure,
    zeros,
    ones,
    randn,
    tensor,
    sigmoid,
    tanh,
    relu,
    gelu,
    silu,
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
    gelu_np,
    silu_np,
    _apply_temperature,
    _apply_top_k,
    _apply_top_p,
    _apply_repetition_penalty,
    _apply_frequency_penalty,
    _apply_presence_penalty,
    _sample_from_logits,
)


# ---------------------------------------------------------------------------
# Tensor creation
# ---------------------------------------------------------------------------

class TestTensorCreation:
    def test_from_numpy(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        assert t.shape == (3,)
        assert t.data.dtype == np.float32

    def test_from_list(self):
        t = Tensor([1, 2, 3])
        assert t.shape == (3,)
        assert t.data.dtype == np.float32

    def test_from_scalar(self):
        t = Tensor(5.0)
        assert t.shape == ()
        assert t.item() == 5.0

    def test_requires_grad(self):
        t = Tensor([1.0], requires_grad=True)
        assert t.requires_grad is True

    def test_no_grad_context(self):
        with no_grad():
            t = Tensor([1.0], requires_grad=True)
            assert t.requires_grad is False

    def test_no_grad_decorator(self):
        @no_grad()
        def fn():
            return Tensor([1.0], requires_grad=True)
        assert fn().requires_grad is False

    def test_repr(self):
        t = Tensor(np.zeros((2, 3)))
        assert "shape=(2, 3)" in repr(t)

    def test_detach(self):
        t = Tensor([1.0, 2.0], requires_grad=True)
        d = t.detach()
        assert d.requires_grad is False
        np.testing.assert_array_equal(d.data, t.data)
        d.data[0] = 999
        assert t.data[0] == 1.0  # original unchanged

    def test_clone(self):
        t = Tensor([1.0, 2.0])
        c = t.clone()
        np.testing.assert_array_equal(c.data, t.data)
        c.data[0] = 999
        assert t.data[0] == 1.0

    def test_float_conversion(self):
        t = Tensor(np.array([1, 2], dtype=np.int32))
        t.float()
        assert t.data.dtype == np.float32

    def test_long_conversion(self):
        t = Tensor([1.5, 2.5])
        result = t.long()
        assert result.data.dtype == np.int64

    def test_int_conversion(self):
        t = Tensor([1.5, 2.5])
        result = t.int()
        assert result.data.dtype == np.int32

    def test_half_conversion(self):
        t = Tensor([1.0, 2.0])
        result = t.half()
        assert result.data.dtype == np.float16

    def test_double_conversion(self):
        t = Tensor([1.0, 2.0])
        result = t.double()
        assert result.data.dtype == np.float64

    def test_zero_(self):
        t = Tensor([1.0, 2.0, 3.0])
        t.zero_()
        np.testing.assert_array_equal(t.data, [0.0, 0.0, 0.0])

    def test_fill_(self):
        t = Tensor([0.0, 0.0])
        t.fill_(7.0)
        np.testing.assert_array_equal(t.data, [7.0, 7.0])

    def test_copy_(self):
        t = Tensor([0.0, 0.0])
        t.copy_(Tensor([1.0, 2.0]))
        np.testing.assert_array_equal(t.data, [1.0, 2.0])

    def test_contiguous(self):
        t = Tensor([1.0, 2.0])
        assert t.contiguous() is t

    def test_flatten(self):
        t = Tensor(np.ones((2, 3)))
        f = t.flatten()
        assert f.shape == (6,)


# ---------------------------------------------------------------------------
# Arithmetic ops
# ---------------------------------------------------------------------------

class TestArithmetic:
    def test_add(self):
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])
        c = a + b
        np.testing.assert_array_almost_equal(c.data, [4.0, 6.0])

    def test_radd(self):
        a = Tensor([1.0, 2.0])
        c = 3.0 + a
        np.testing.assert_array_almost_equal(c.data, [4.0, 5.0])

    def test_sub(self):
        a = Tensor([5.0, 6.0])
        b = Tensor([1.0, 2.0])
        c = a - b
        np.testing.assert_array_almost_equal(c.data, [4.0, 4.0])

    def test_rsub(self):
        a = Tensor([1.0, 2.0])
        c = 5.0 - a
        np.testing.assert_array_almost_equal(c.data, [4.0, 3.0])

    def test_mul(self):
        a = Tensor([2.0, 3.0])
        b = Tensor([4.0, 5.0])
        c = a * b
        np.testing.assert_array_almost_equal(c.data, [8.0, 15.0])

    def test_rmul(self):
        a = Tensor([2.0, 3.0])
        c = 3.0 * a
        np.testing.assert_array_almost_equal(c.data, [6.0, 9.0])

    def test_neg(self):
        a = Tensor([1.0, -2.0])
        c = -a
        np.testing.assert_array_almost_equal(c.data, [-1.0, 2.0])

    def test_pow(self):
        a = Tensor([2.0, 3.0])
        c = a ** 2
        np.testing.assert_array_almost_equal(c.data, [4.0, 9.0])

    def test_truediv(self):
        a = Tensor([6.0, 8.0])
        b = Tensor([2.0, 4.0])
        c = a / b
        np.testing.assert_array_almost_equal(c.data, [3.0, 2.0])

    def test_matmul(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
        c = a @ b
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_almost_equal(c.data, expected)

    def test_matmul_1d(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        c = a @ b
        assert c.item() == 32.0


# ---------------------------------------------------------------------------
# Comparison ops
# ---------------------------------------------------------------------------

class TestComparison:
    def test_ge(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([2.0, 2.0, 2.0])
        c = a >= b
        np.testing.assert_array_equal(c.data, [0.0, 1.0, 1.0])

    def test_le(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([2.0, 2.0, 2.0])
        c = a <= b
        np.testing.assert_array_equal(c.data, [1.0, 1.0, 0.0])

    def test_gt(self):
        a = Tensor([1.0, 2.0, 3.0])
        c = a > 1.5
        np.testing.assert_array_equal(c.data, [0.0, 1.0, 1.0])

    def test_lt(self):
        a = Tensor([1.0, 2.0, 3.0])
        c = a < 2.5
        np.testing.assert_array_equal(c.data, [1.0, 1.0, 0.0])

    def test_eq_tensor(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([1.0, 3.0, 3.0])
        c = a == b
        np.testing.assert_array_equal(c.data, [1.0, 0.0, 1.0])

    def test_ne_tensor(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([1.0, 3.0, 3.0])
        c = a != b
        np.testing.assert_array_equal(c.data, [0.0, 1.0, 0.0])

    def test_bool_scalar(self):
        assert bool(Tensor(1.0)) is True
        assert bool(Tensor(0.0)) is False

    def test_bool_vector_raises(self):
        with pytest.raises(RuntimeError):
            bool(Tensor([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Tensor methods
# ---------------------------------------------------------------------------

class TestTensorMethods:
    def test_len(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert len(t) == 3

    def test_len_0d_raises(self):
        with pytest.raises(TypeError):
            len(Tensor(5.0))

    def test_item(self):
        t = Tensor(42.0)
        assert t.item() == 42.0

    def test_t(self):
        t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        t2 = t.t()
        assert t2.shape == (2, 2)
        np.testing.assert_array_almost_equal(t2.data, [[1.0, 3.0], [2.0, 4.0]])

    def test_t_non_2d_raises(self):
        with pytest.raises(RuntimeError):
            Tensor(np.ones((2, 3, 4))).t()

    def test_squeeze(self):
        t = Tensor(np.ones((1, 3, 1)))
        s = t.squeeze()
        assert s.shape == (3,)

    def test_unsqueeze(self):
        t = Tensor(np.ones((3,)))
        s = t.unsqueeze(0)
        assert s.shape == (1, 3)

    def test_repeat(self):
        t = Tensor([1.0, 2.0])
        r = t.repeat(2, 3)
        assert r.shape == (2, 6)

    def test_gather(self):
        t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        idx = Tensor(np.array([[0, 1], [1, 0]]))
        g = t.gather(1, idx)
        np.testing.assert_array_almost_equal(g.data, [[1.0, 2.0], [4.0, 3.0]])

    def test_argmax(self):
        t = Tensor([1.0, 5.0, 3.0])
        idx = t.argmax()
        assert idx.item() == 1

    def test_argmin(self):
        t = Tensor([5.0, 1.0, 3.0])
        idx = t.argmin()
        assert idx.item() == 1

    def test_topk(self):
        t = Tensor([1.0, 5.0, 3.0, 2.0])
        vals, idx = t.topk(2)
        assert vals.shape == (1, 2)
        np.testing.assert_array_almost_equal(vals.data, [[5.0, 3.0]])

    def test_to_list(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert t.tolist() == [1.0, 2.0, 3.0]

    def test_dim(self):
        t = Tensor(np.ones((2, 3, 4)))
        assert t.dim() == 3

    def test_numel(self):
        t = Tensor(np.ones((2, 3)))
        assert t.numel() == 6

    def test_size(self):
        t = Tensor(np.ones((2, 3)))
        assert t.size() == (2, 3)
        assert t.size(0) == 2

    def test_expand(self):
        t = Tensor(np.ones((1, 3)))
        e = t.expand(2, 3)
        assert e.shape == (2, 3)

    def test_transpose(self):
        t = Tensor(np.ones((2, 3)))
        tp = t.transpose(0, 1)
        assert tp.shape == (3, 2)

    def test_permute(self):
        t = Tensor(np.ones((2, 3, 4)))
        p = t.permute(2, 0, 1)
        assert p.shape == (4, 2, 3)

    def test_abs(self):
        t = Tensor([-3.0, 2.0])
        a = t.abs()
        np.testing.assert_array_almost_equal(a.data, [3.0, 2.0])

    def test_sqrt(self):
        t = Tensor([4.0, 9.0])
        s = t.sqrt()
        np.testing.assert_array_almost_equal(s.data, [2.0, 3.0])

    def test_sqrt_negative_clamped(self):
        t = Tensor([-1.0])
        s = t.sqrt()
        assert s.item() == 0.0

    def test_clamp(self):
        t = Tensor([1.0, 5.0, 10.0])
        c = t.clamp(min_val=2.0, max_val=8.0)
        np.testing.assert_array_almost_equal(c.data, [2.0, 5.0, 8.0])

    def test_argsort(self):
        t = Tensor([3.0, 1.0, 2.0])
        idx = t.argsort()
        np.testing.assert_array_equal(idx.data, [1, 2, 0])

    def test_argsort_descending(self):
        t = Tensor([3.0, 1.0, 2.0])
        idx = t.argsort(descending=True)
        np.testing.assert_array_equal(idx.data, [0, 2, 1])

    def test_eq_method(self):
        t = Tensor([1.0, 2.0, 3.0])
        e = t.eq(2.0)
        np.testing.assert_array_equal(e.data, [0.0, 1.0, 0.0])

    def test_ne_method(self):
        t = Tensor([1.0, 2.0, 3.0])
        e = t.ne(2.0)
        np.testing.assert_array_equal(e.data, [1.0, 0.0, 1.0])

    def test_gt_method(self):
        t = Tensor([1.0, 2.0, 3.0])
        e = t.gt(1.5)
        np.testing.assert_array_equal(e.data, [0.0, 1.0, 1.0])

    def test_lt_method(self):
        t = Tensor([1.0, 2.0, 3.0])
        e = t.lt(2.5)
        np.testing.assert_array_equal(e.data, [1.0, 1.0, 0.0])

    def test_ge_method(self):
        t = Tensor([1.0, 2.0, 3.0])
        e = t.ge(2.0)
        np.testing.assert_array_equal(e.data, [0.0, 1.0, 1.0])

    def test_le_method(self):
        t = Tensor([1.0, 2.0, 3.0])
        e = t.le(2.0)
        np.testing.assert_array_equal(e.data, [1.0, 1.0, 0.0])

    def test_all(self):
        t = Tensor([1.0, 1.0])
        assert t.all().item() == 1.0

    def test_any(self):
        t = Tensor([0.0, 1.0])
        assert t.any().item() == 1.0

    def test_requires_grad_(self):
        t = Tensor([1.0])
        t.requires_grad_(False)
        assert t.requires_grad is False
        t.requires_grad_(True)
        assert t.requires_grad is True

    def test_view(self):
        t = Tensor(np.ones((2, 3)))
        v = t.view(3, 2)
        assert v.shape == (3, 2)

    def test_to_dtype(self):
        t = Tensor([1.0])
        t2 = t.to(dtype=np.float64)
        # SloNet convention: float dtypes always pin to float32
        assert t2.data.dtype == np.float32

    def test_slice(self):
        t = Tensor([1.0, 2.0, 3.0, 4.0])
        s = t[1:3]
        np.testing.assert_array_almost_equal(s.data, [2.0, 3.0])

    def test_setitem(self):
        t = Tensor([1.0, 2.0, 3.0])
        t[1] = 99.0
        assert t.data[1] == 99.0

    def test_scatter_(self):
        t = Tensor(np.zeros(5))
        idx = Tensor(np.array([1, 3]))
        src = Tensor(np.array([10.0, 20.0]))
        t.scatter_(0, idx, src)
        np.testing.assert_array_almost_equal(t.data, [0.0, 10.0, 0.0, 20.0, 0.0])


# ---------------------------------------------------------------------------
# Autograd — backward
# ---------------------------------------------------------------------------

class TestAutograd:
    def test_add_backward(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        c = a + b
        c.backward()
        np.testing.assert_array_almost_equal(a.grad.data, [1.0, 1.0])
        np.testing.assert_array_almost_equal(b.grad.data, [1.0, 1.0])

    def test_mul_backward(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        c = a * b
        c.backward()
        np.testing.assert_array_almost_equal(a.grad.data, [4.0, 5.0])
        np.testing.assert_array_almost_equal(b.grad.data, [2.0, 3.0])

    def test_sub_backward(self):
        a = Tensor([5.0, 6.0], requires_grad=True)
        b = Tensor([1.0, 2.0], requires_grad=True)
        c = a - b
        c.backward()
        np.testing.assert_array_almost_equal(a.grad.data, [1.0, 1.0])
        np.testing.assert_array_almost_equal(b.grad.data, [-1.0, -1.0])

    def test_neg_backward(self):
        a = Tensor([1.0, -2.0], requires_grad=True)
        c = -a
        c.backward()
        np.testing.assert_array_almost_equal(a.grad.data, [-1.0, -1.0])

    def test_pow_backward(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        c = a ** 3
        c.backward()
        np.testing.assert_array_almost_equal(a.grad.data, [12.0, 27.0])

    def test_matmul_backward(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)
        c = (a @ b).sum()
        c.backward()
        assert a.grad is not None
        assert b.grad is not None
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape

    def test_sigmoid_backward(self):
        x = Tensor([0.0], requires_grad=True)
        y = sigmoid(x)
        y.backward()
        expected = 0.25
        assert abs(x.grad.item() - expected) < 1e-5

    def test_tanh_backward(self):
        x = Tensor([0.0], requires_grad=True)
        y = tanh(x)
        y.backward()
        assert abs(x.grad.item() - 1.0) < 1e-5

    def test_relu_backward(self):
        x = Tensor([-1.0, 2.0], requires_grad=True)
        y = relu(x)
        y.backward()
        np.testing.assert_array_almost_equal(y.data, [0.0, 2.0])
        np.testing.assert_array_almost_equal(x.grad.data, [0.0, 1.0])

    def test_gelu_backward(self):
        x = Tensor([0.0], requires_grad=True)
        y = gelu(x)
        y.backward()
        assert x.grad is not None
        assert abs(x.grad.item() - 0.5) < 0.1

    def test_silu_backward(self):
        x = Tensor([0.0], requires_grad=True)
        y = silu(x)
        y.backward()
        assert x.grad is not None
        assert abs(x.grad.item() - 0.5) < 0.01

    def test_softmax_backward(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = softmax(x)
        y.sum().backward()
        assert x.grad is not None

    def test_cross_entropy_backward(self):
        logits = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)
        targets = Tensor(np.array([2]))
        loss = cross_entropy(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape

    def test_mse_loss_backward(self):
        pred = Tensor([1.0, 2.0], requires_grad=True)
        target = Tensor([2.0, 3.0])
        loss = mse_loss(pred, target)
        loss.backward()
        np.testing.assert_array_almost_equal(pred.grad.data, [-1.0, -1.0])

    def test_broadcast_back_squeeze(self):
        g = np.ones((2, 3))
        result = _broadcast_back(g, (3,))
        assert result.shape == (3,)

    def test_broadcast_back_expand(self):
        g = np.ones((3,))
        result = _broadcast_back(g, (2, 3))
        assert result.shape == (2, 3)

    def test_broadcast_forward(self):
        t = np.ones((3,))
        result = _broadcast_forward(t, (2, 3))
        assert result.shape == (2, 3)

    def test_chained_ops(self):
        a = Tensor([2.0], requires_grad=True)
        b = Tensor([3.0], requires_grad=True)
        c = (a * b + a).sum()
        c.backward()
        assert a.grad.item() == 4.0  # b + 1 = 3 + 1
        assert b.grad.item() == 2.0  # a = 2

    def test_no_grad_no_backward(self):
        with no_grad():
            a = Tensor([1.0], requires_grad=True)
            b = Tensor([2.0], requires_grad=True)
            c = a + b
        assert c.requires_grad is False
        assert len(c._children) == 0


# ---------------------------------------------------------------------------
# Forward-mode AD
# ---------------------------------------------------------------------------

class TestForwardGrad:
    def test_simple_forward(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = a * a  # x^2
        tangents = b.forward_grad({a.id: np.array([1.0, 1.0])})
        # d/dx(x^2) * v = 2x * v
        expected = 2 * a.data
        np.testing.assert_array_almost_equal(tangents[b.id], expected)

    def test_jvp(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = a * a
        v = Tensor(np.ones_like(a.data))
        result = b.jvp(v)
        expected = 2 * a.data
        np.testing.assert_array_almost_equal(result.data, expected)


# ---------------------------------------------------------------------------
# GenerationMetrics
# ---------------------------------------------------------------------------

class TestGenerationMetrics:
    def test_finalize(self):
        m = GenerationMetrics(n_tokens=10, t_start=1.0, t_end=2.0, t_first_token=1.1)
        m.finalize()
        assert m.total_ms == pytest.approx(1000.0)
        assert m.ttft_ms == pytest.approx(100.0)
        assert m.tokens_per_sec == pytest.approx(10.0)
        assert m.prefill_ms == pytest.approx(100.0)

    def test_finalize_no_tokens(self):
        m = GenerationMetrics(n_tokens=0, t_start=1.0, t_end=2.0)
        m.finalize()
        assert m.tokens_per_sec == 0.0

    def test_ttft_zero(self):
        m = GenerationMetrics(t_first_token=0.0, t_start=1.0)
        assert m.ttft_ms == 0.0


# ---------------------------------------------------------------------------
# GenerateResult
# ---------------------------------------------------------------------------

class TestGenerateResult:
    def test_generated_ids(self):
        ids = np.array([[1, 2, 3, 4, 5]])
        m = GenerationMetrics(prompt_tokens=2)
        r = GenerateResult(token_ids=ids, metrics=m)
        np.testing.assert_array_equal(r.generated_ids, [[3, 4, 5]])

    def test_generated_ids_no_prompt(self):
        ids = np.array([[1, 2, 3]])
        r = GenerateResult(token_ids=ids)
        np.testing.assert_array_equal(r.generated_ids, [[1, 2, 3]])

    def test_shape(self):
        r = GenerateResult(token_ids=np.array([[1, 2, 3]]))
        assert r.shape == (1, 3)

    def test_dtype(self):
        r = GenerateResult(token_ids=np.array([[1, 2, 3]]))
        assert r.dtype == np.int64

    def test_getitem(self):
        r = GenerateResult(token_ids=np.array([[1, 2, 3]]))
        np.testing.assert_array_equal(r[0], [1, 2, 3])

    def test_array(self):
        r = GenerateResult(token_ids=np.array([[1, 2, 3]]))
        np.testing.assert_array_equal(np.asarray(r), [[1, 2, 3]])

    def test_eq_result(self):
        r1 = GenerateResult(token_ids=np.array([[1, 2]]))
        r2 = GenerateResult(token_ids=np.array([[1, 2]]))
        assert r1 == r2

    def test_eq_ndarray(self):
        r = GenerateResult(token_ids=np.array([[1, 2]]))
        assert r == np.array([[1, 2]])

    def test_eq_not_implemented(self):
        r = GenerateResult(token_ids=np.array([[1, 2]]))
        assert r.__eq__("other") is NotImplemented


# ---------------------------------------------------------------------------
# Logit processors
# ---------------------------------------------------------------------------

class TestLogitProcessors:
    def test_apply_temperature(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        result = _apply_temperature(logits, 0.5)
        np.testing.assert_array_almost_equal(result, [[2.0, 4.0, 6.0]])

    def test_apply_temperature_zero(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        result = _apply_temperature(logits, 0.0)
        np.testing.assert_array_almost_equal(result, logits)

    def test_apply_top_k(self):
        logits = np.array([[1.0, 5.0, 3.0, 2.0]])
        result = _apply_top_k(logits, 2)
        assert result[0, 0] < -1e8
        assert result[0, 1] == 5.0
        assert result[0, 2] == 3.0
        assert result[0, 3] < -1e8

    def test_apply_top_k_noop(self):
        logits = np.array([[1.0, 2.0]])
        result = _apply_top_k(logits, 5)
        np.testing.assert_array_almost_equal(result, logits)

    def test_apply_top_p(self):
        logits = np.array([[1.0, 5.0, 3.0]])
        result = _apply_top_p(logits, 0.9)
        kept = np.sum(result[0] > -1e8)
        assert kept >= 1

    def test_apply_top_p_noop(self):
        logits = np.array([[1.0, 2.0]])
        result = _apply_top_p(logits, 1.0)
        np.testing.assert_array_almost_equal(result, logits)

    def test_apply_repetition_penalty(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        generated = np.array([1])
        result = _apply_repetition_penalty(logits.copy(), generated, 2.0)
        assert result[0, 1] < logits[0, 1]

    def test_apply_repetition_penalty_noop(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        result = _apply_repetition_penalty(logits.copy(), np.array([]), 2.0)
        np.testing.assert_array_almost_equal(result, logits)

    def test_apply_frequency_penalty(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        generated = np.array([1, 1, 2])
        result = _apply_frequency_penalty(logits.copy(), generated, 0.5)
        assert result[0, 1] < logits[0, 1]

    def test_apply_presence_penalty(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        generated = np.array([1])
        result = _apply_presence_penalty(logits.copy(), generated, 1.0)
        assert result[0, 1] < logits[0, 1]

    def test_sample_from_logits(self):
        logits = np.array([[1.0, 2.0, 3.0, 4.0]])
        tok = _sample_from_logits(logits, temperature=0.1)
        assert 0 <= tok < 4

    def test_sample_from_logits_topk(self):
        logits = np.array([[1.0, 2.0, 3.0, 4.0]])
        tok = _sample_from_logits(logits, temperature=1.0, top_k=2)
        assert tok in [2, 3]

    def test_sample_from_logits_greedy(self):
        logits = np.array([[1.0, 2.0, 3.0, 4.0]])
        tok = _sample_from_logits(logits, temperature=0.001)
        assert tok == 3


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilityFunctions:
    def test_ensure_tensor(self):
        t = _ensure(5.0)
        assert isinstance(t, Tensor)
        assert t.item() == 5.0

    def test_ensure_passthrough(self):
        t = Tensor([1.0])
        assert _ensure(t) is t

    def test_zeros(self):
        t = zeros((2, 3))
        assert t.shape == (2, 3)
        np.testing.assert_array_equal(t.data, 0.0)

    def test_ones(self):
        t = ones((2, 3))
        np.testing.assert_array_equal(t.data, 1.0)

    def test_randn(self):
        t = randn((100,))
        assert t.shape == (100,)
        assert abs(t.data.mean()) < 0.5

    def test_tensor_func(self):
        t = tensor([1.0, 2.0])
        assert t.shape == (2,)

    def test_topk(self):
        vals, idx = topk(Tensor([1.0, 5.0, 3.0]), 2)
        np.testing.assert_array_almost_equal(vals.data, [[5.0, 3.0]])

    def test_multinomial(self):
        probs = Tensor([0.1, 0.2, 0.7])
        idx = multinomial(probs, 1)
        assert idx.shape == (1, 1)

    def test_stack(self):
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])
        s = stack([a, b], dim=0)
        assert s.shape == (2, 2)

    def test_concatenate(self):
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])
        c = concatenate([a, b], dim=0)
        np.testing.assert_array_almost_equal(c.data, [1.0, 2.0, 3.0, 4.0])

    def test_randint(self):
        t = randint(0, 10, (5,))
        assert t.shape == (5,)
        assert all(0 <= v < 10 for v in t.data)

    def test_exp(self):
        t = Tensor([0.0, 1.0])
        e = exp(t)
        np.testing.assert_array_almost_equal(e.data, [1.0, np.e])

    def test_isfinite(self):
        t = Tensor([1.0, np.inf, np.nan])
        result = isfinite(t)
        np.testing.assert_array_equal(result, [True, False, False])

    def test_where(self):
        cond = Tensor([True, False, True])
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        r = where(cond, a, b)
        np.testing.assert_array_almost_equal(r.data, [1.0, 5.0, 3.0])

    def test_gelu_np(self):
        result = gelu_np(np.array([0.0]))
        assert abs(result[0] - 0.0) < 0.01

    def test_silu_np(self):
        result = silu_np(np.array([0.0]))
        assert abs(result[0] - 0.0) < 0.01

    def test_is_cuda(self):
        from domains.training.slonet import is_cuda
        assert is_cuda(Tensor([1.0])) is False

    def test_is_mps(self):
        from domains.training.slonet import is_mps
        assert is_mps(Tensor([1.0])) is False

    def test_cpu(self):
        from domains.training.slonet import cpu
        t = Tensor([1.0])
        assert cpu(t) is t
