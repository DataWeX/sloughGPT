"""Tests for broadcast gradient correctness in backward pass.

Verifies _mul, _add, _sub, _matmul, _neg, _pow, _sum, _mean, _reshape,
_transpose backward handle mismatched ndim broadcasts without shape errors
or incorrect gradients.
"""

import numpy as np
import pytest
from domains.training.slonet import (
    Tensor, cross_entropy, _broadcast_back, _broadcast_forward,
    _add, _sub, _mul, _neg, _pow, _sum, _mean, _reshape, _transpose,
    _ensure, tensor,
)


# ── _broadcast_back ───────────────────────────────────────────────────────────

class TestBroadcastBackFunction:
    def test_no_op_same_shape(self):
        g = np.ones((3, 4))
        result = _broadcast_back(g, (3, 4))
        assert result.shape == (3, 4)

    def test_squeeze_leading_dim(self):
        g = np.ones((2, 3, 4))
        result = _broadcast_back(g, (3, 4))
        assert result.shape == (3, 4)

    def test_squeeze_broadcast_dim(self):
        g = np.ones((3, 4))
        result = _broadcast_back(g, (1, 4))
        assert result.shape == (1, 4)

    def test_returns_copy(self):
        g = np.ones((3, 4))
        result = _broadcast_back(g, (3, 4))
        g[0, 0] = 999
        assert result[0, 0] == 1.0

    def test_scalar_to_array(self):
        g = np.ones((3, 4))
        result = _broadcast_back(g, ())
        assert result.ndim == 0


# ── _broadcast_forward ────────────────────────────────────────────────────────

class TestBroadcastForwardFunction:
    def test_same_shape(self):
        t = np.ones((3, 4))
        result = _broadcast_forward(t, (3, 4))
        assert result.shape == (3, 4)

    def test_expand_dims(self):
        t = np.ones((4,))
        result = _broadcast_forward(t, (3, 4))
        assert result.shape == (3, 4)

    def test_scalar_to_nd(self):
        t = np.array(5.0)
        result = _broadcast_forward(t, (2, 3))
        assert result.shape == (2, 3)


# ── Mul Broadcast Backward ───────────────────────────────────────────────────

class TestMulBroadcastBackward:
    def test_mul_1d_times_2d(self):
        a = Tensor(np.ones(4), requires_grad=True)
        b = Tensor(np.ones((3, 4)), requires_grad=True)
        out = a * b
        out.backward()
        assert a.grad is not None
        assert b.grad is not None
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape
        assert np.allclose(a.grad.data, 3.0)
        assert np.allclose(b.grad.data, 1.0)

    def test_mul_2d_times_1d(self):
        a = Tensor(np.ones((3, 4)), requires_grad=True)
        b = Tensor(np.ones(4), requires_grad=True)
        out = a * b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape
        assert np.allclose(a.grad.data, 1.0)
        assert np.allclose(b.grad.data, 3.0)

    def test_mul_scalar_times_nd(self):
        a = Tensor(np.array(2.0), requires_grad=True)
        b = Tensor(np.ones((2, 3)), requires_grad=True)
        out = a * b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape
        assert np.allclose(a.grad.data, 6.0)
        assert np.allclose(b.grad.data, 2.0)

    def test_mul_3d_times_1d(self):
        a = Tensor(np.ones((2, 3, 4)), requires_grad=True)
        b = Tensor(np.ones(4), requires_grad=True)
        out = a * b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape
        assert np.allclose(b.grad.data, 6.0)

    def test_mul_leading_extra_dims(self):
        a = Tensor(np.ones((2, 1, 4)), requires_grad=True)
        b = Tensor(np.ones((3, 4)), requires_grad=True)
        out = a * b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape

    def test_mul_leading_one_dim_broadcast(self):
        a = Tensor(np.ones((1, 4)), requires_grad=True)
        b = Tensor(np.ones((3, 4)), requires_grad=True)
        out = a * b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape
        assert np.allclose(a.grad.data, 3.0)

    def test_add_leading_one_dim_broadcast(self):
        a = Tensor(np.ones((1, 4)), requires_grad=True)
        b = Tensor(np.ones((3, 4)), requires_grad=True)
        out = a + b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape
        assert np.allclose(a.grad.data, 3.0)

    def test_mul_numeric_correctness(self):
        a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        b = Tensor(np.array([4.0, 5.0]), requires_grad=True)
        out = a * b
        out.backward()
        assert np.allclose(a.grad.data, [4.0, 5.0])
        assert np.allclose(b.grad.data, [2.0, 3.0])

    def test_mul_chain_through_cross_entropy(self):
        a = Tensor(np.ones(3), requires_grad=True)
        b = Tensor(np.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0]]), requires_grad=True)
        logits = a * b
        target = Tensor(np.array([0, 1]))
        loss = cross_entropy(logits, target)
        loss.backward()
        assert a.grad is not None
        assert b.grad is not None
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape

    def test_mul_same_shape(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        b = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)
        out = a * b
        out.backward()
        assert np.allclose(a.grad.data, [4.0, 5.0, 6.0])
        assert np.allclose(b.grad.data, [1.0, 2.0, 3.0])

    def test_mul_rmul(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        out = 3.0 * a
        out.backward()
        assert np.allclose(a.grad.data, 3.0)


# ── Add Broadcast Backward ───────────────────────────────────────────────────

class TestAddBroadcastBackward:
    def test_add_1d_plus_2d(self):
        a = Tensor(np.ones(4), requires_grad=True)
        b = Tensor(np.ones((3, 4)), requires_grad=True)
        out = a + b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape
        assert np.allclose(a.grad.data, 3.0)

    def test_add_scalar_plus_nd(self):
        a = Tensor(np.array(5.0), requires_grad=True)
        b = Tensor(np.ones((2, 3)), requires_grad=True)
        out = a + b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape

    def test_add_same_shape(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        out = a + b
        out.backward()
        assert np.allclose(a.grad.data, 1.0)
        assert np.allclose(b.grad.data, 1.0)

    def test_add_radd(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        out = 5.0 + a
        out.backward()
        assert np.allclose(a.grad.data, 1.0)

    def test_add_numeric(self):
        a = Tensor(np.array([10.0, 20.0]), requires_grad=True)
        b = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        out = a + b
        assert np.allclose(out.data, [11.0, 22.0])


# ── Sub Broadcast Backward ───────────────────────────────────────────────────

class TestSubBroadcastBackward:
    def test_sub_basic(self):
        a = Tensor(np.array([5.0, 6.0]), requires_grad=True)
        b = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        out = a - b
        out.backward()
        assert np.allclose(out.data, [4.0, 4.0])
        assert np.allclose(a.grad.data, 1.0)
        assert np.allclose(b.grad.data, -1.0)

    def test_sub_rsub(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        out = 10.0 - a
        out.backward()
        assert np.allclose(out.data, [9.0, 8.0])
        assert np.allclose(a.grad.data, -1.0)

    def test_sub_1d_2d(self):
        a = Tensor(np.ones(4), requires_grad=True)
        b = Tensor(np.ones((3, 4)) * 2, requires_grad=True)
        out = a - b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape
        assert np.allclose(a.grad.data, 3.0)
        assert np.allclose(b.grad.data, -1.0)


# ── MatMul Broadcast Backward ────────────────────────────────────────────────

class TestMatMulBroadcastBackward:
    def test_matmul_1d_2d(self):
        a = Tensor(np.ones(4), requires_grad=True)
        b = Tensor(np.ones((4, 3)), requires_grad=True)
        out = a @ b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape

    def test_matmul_2d_1d(self):
        a = Tensor(np.ones((3, 4)), requires_grad=True)
        b = Tensor(np.ones(4), requires_grad=True)
        out = a @ b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape

    def test_matmul_2d_2d(self):
        a = Tensor(np.ones((2, 3)), requires_grad=True)
        b = Tensor(np.ones((3, 4)), requires_grad=True)
        out = a @ b
        out.backward()
        assert a.grad.shape == (2, 3)
        assert b.grad.shape == (3, 4)

    def test_matmul_numeric(self):
        a = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
        b = Tensor(np.array([[3.0], [4.0]]), requires_grad=True)
        out = a @ b
        assert np.allclose(out.data, [[11.0]])
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape


# ── Neg Backward ──────────────────────────────────────────────────────────────

class TestNegBackward:
    def test_neg_basic(self):
        a = Tensor(np.array([1.0, -2.0, 3.0]), requires_grad=True)
        out = -a
        out.backward()
        assert np.allclose(out.data, [-1.0, 2.0, -3.0])
        assert np.allclose(a.grad.data, -1.0)

    def test_neg_double_neg(self):
        a = Tensor(np.array([5.0]), requires_grad=True)
        out = -(-a)
        assert np.allclose(out.data, [5.0])


# ── Pow Backward ─────────────────────────────────────────────────────────────

class TestPowBackward:
    def test_pow_2(self):
        a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        out = a ** 2
        out.backward()
        assert np.allclose(out.data, [4.0, 9.0])
        assert np.allclose(a.grad.data, [4.0, 6.0])

    def test_pow_1(self):
        a = Tensor(np.array([5.0, 10.0]), requires_grad=True)
        out = a ** 1
        out.backward()
        assert np.allclose(a.grad.data, 1.0)


# ── Sum/Mean/Max Backward ────────────────────────────────────────────────────

class TestReductionBackward:
    def test_sum_backward(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        out = a.sum()
        out.backward()
        assert np.allclose(out.data, 6.0)
        assert np.allclose(a.grad.data, 1.0)

    def test_mean_backward(self):
        a = Tensor(np.array([2.0, 4.0]), requires_grad=True)
        out = a.mean()
        out.backward()
        assert np.allclose(out.data, 3.0)
        assert np.allclose(a.grad.data, 0.5)

    def test_max_backward(self):
        a = Tensor(np.array([1.0, 5.0, 3.0]), requires_grad=True)
        out = a.max()
        out.backward()
        assert np.allclose(out.data, 5.0)
        assert a.grad.data[1] == 1.0
        assert a.grad.data[0] == 0.0


# ── Reshape / Transpose ──────────────────────────────────────────────────────

class TestReshapeTranspose:
    def test_reshape(self):
        a = Tensor(np.arange(6.0), requires_grad=True)
        out = a.reshape(2, 3)
        assert out.shape == (2, 3)
        out.backward()
        assert a.grad.shape == a.shape

    def test_transpose(self):
        a = Tensor(np.ones((3, 4)), requires_grad=True)
        out = a.T()
        assert out.shape == (4, 3)
        out.backward()
        assert a.grad.shape == a.shape

    def test_transpose_3d_no_crash(self):
        a = Tensor(np.ones((2, 3, 4)), requires_grad=True)
        out = a.T()
        assert out.shape == (4, 3, 2)


# ── _ensure ───────────────────────────────────────────────────────────────────

class TestEnsure:
    def test_ensure_tensor(self):
        t = Tensor([1.0, 2.0])
        assert _ensure(t) is t

    def test_ensure_numpy(self):
        result = _ensure(np.array([1.0, 2.0]))
        assert isinstance(result, Tensor)

    def test_ensure_scalar(self):
        result = _ensure(5.0)
        assert isinstance(result, Tensor)


# ── tensor helper ─────────────────────────────────────────────────────────────

class TestTensorHelper:
    def test_tensor_no_copy(self):
        a = np.array([1.0, 2.0])
        t = tensor(a)
        assert t.shape == (2,)

    def test_tensor_requires_grad(self):
        t = tensor([1.0, 2.0], requires_grad=True)
        assert t.requires_grad is True


# ── _slice ────────────────────────────────────────────────────────────────────

class TestSliceBackward:
    def test_slice_basic_index_helper(self):
        from domains.training.slonet import _basic_index
        assert _basic_index((slice(None), 0, slice(None)))
        assert _basic_index((slice(1, 3),))
        assert _basic_index((Ellipsis, slice(None)))
        assert _basic_index((slice(None), np.int64(2)))
        assert not _basic_index((slice(None), [0, 1]))
        assert not _basic_index((slice(None), np.array([0, 1])))
        assert not _basic_index((slice(None), slice(None), True))


# ── Chain through backward ───────────────────────────────────────────────────

class TestChainBackward:
    def test_chain_mul_add(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        out = (a * b) + (a + b)
        out.backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_chain_through_cross_entropy(self):
        logits = Tensor(np.array([[1.0, 2.0, 3.0],
                                   [4.0, 5.0, 6.0]]), requires_grad=True)
        target = Tensor(np.array([0, 2]))
        loss = cross_entropy(logits, target)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape

    def test_deep_chain(self):
        a = Tensor(np.array([2.0]), requires_grad=True)
        out = a * a * a * a
        out.backward()
        # d/dx x^4 = 4x^3, at x=2 => 32
        assert np.allclose(a.grad.data, 32.0)


# ── Eq / Ne / Comparison ────────────────────────────────────────────────────

class TestComparisonOps:
    def test_ge(self):
        a = Tensor(np.array([1.0, 3.0, 5.0]))
        b = Tensor(np.array([2.0, 3.0, 4.0]))
        out = a >= b
        assert np.allclose(out.data, [0.0, 1.0, 1.0])

    def test_le(self):
        a = Tensor(np.array([1.0, 3.0, 5.0]))
        b = Tensor(np.array([2.0, 3.0, 4.0]))
        out = a <= b
        assert np.allclose(out.data, [1.0, 1.0, 0.0])

    def test_gt(self):
        a = Tensor(np.array([1.0, 3.0, 5.0]))
        b = Tensor(np.array([2.0, 3.0, 4.0]))
        out = a > b
        assert np.allclose(out.data, [0.0, 0.0, 1.0])

    def test_lt(self):
        a = Tensor(np.array([1.0, 3.0, 5.0]))
        b = Tensor(np.array([2.0, 3.0, 4.0]))
        out = a < b
        assert np.allclose(out.data, [1.0, 0.0, 0.0])

    def test_eq_tensor(self):
        a = Tensor(np.array([1.0, 2.0]))
        b = Tensor(np.array([1.0, 3.0]))
        out = a == b
        assert np.allclose(out.data, [1.0, 0.0])

    def test_ne_tensor(self):
        a = Tensor(np.array([1.0, 2.0]))
        b = Tensor(np.array([1.0, 3.0]))
        out = a != b
        assert np.allclose(out.data, [0.0, 1.0])


# ── Tensor utility methods ──────────────────────────────────────────────────

class TestTensorUtils:
    def test_bool_scalar(self):
        assert bool(Tensor(np.array(1.0))) is True
        assert bool(Tensor(np.array(0.0))) is False

    def test_bool_nd_raises(self):
        with pytest.raises(RuntimeError):
            bool(Tensor(np.array([1.0, 2.0])))

    def test_len(self):
        assert len(Tensor(np.array([1.0, 2.0, 3.0]))) == 3

    def test_len_0d_raises(self):
        with pytest.raises(TypeError):
            len(Tensor(np.array(5.0)))

    def test_tolist(self):
        assert Tensor(np.array([1.0, 2.0])).tolist() == [1.0, 2.0]

    def test_item(self):
        assert Tensor(np.array([42.0])).item() == 42.0

    def test_dim(self):
        assert Tensor(np.ones((2, 3))).dim() == 2
        assert Tensor(np.ones(4)).dim() == 1

    def test_numel(self):
        assert Tensor(np.ones((2, 3))).numel() == 6

    def test_size(self):
        t = Tensor(np.ones((2, 3, 4)))
        assert t.size() == (2, 3, 4)
        assert t.size(0) == 2
        assert t.size(1) == 3

    def test_squeeze(self):
        t = Tensor(np.ones((1, 3, 1)))
        s = t.squeeze()
        assert s.shape == (3,)

    def test_unsqueeze(self):
        t = Tensor(np.ones((3,)))
        s = t.unsqueeze(0)
        assert s.shape == (1, 3)

    def test_repeat(self):
        t = Tensor(np.array([1.0, 2.0]))
        r = t.repeat(3)
        assert r.shape == (6,)

    def test_detach(self):
        a = Tensor(np.array([1.0]), requires_grad=True)
        d = a.detach()
        assert d.requires_grad is False

    def test_cpu(self):
        t = Tensor(np.array([1.0]))
        assert t.cpu() is t

    def test_numpy(self):
        a = np.array([1.0, 2.0])
        t = Tensor(a)
        assert np.array_equal(t.numpy(), a)

    def test_float(self):
        t = Tensor(np.array([1, 2], dtype=np.int32))
        f = t.float()
        assert f.data.dtype == np.float32
