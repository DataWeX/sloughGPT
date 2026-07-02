"""Tests for bidirectional DAG: forward-mode AD alongside backward-mode AD."""

import numpy as np
from domains.training.slonet import Tensor, cross_entropy, _layernorm, _rmsnorm, sigmoid, relu, tanh, gelu, silu, _softmax, _maxpool2d, flatten, no_grad, _transpose


class TestForwardGradBasic:
    """Forward-mode AD on primitive ops."""

    def test_add(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        y = a + b
        t = y.forward_grad({a.id: np.array([1.0, 1.0])})
        assert t.get(y.id) is not None
        assert np.allclose(t[y.id], [1.0, 1.0])

    def test_add_both_tangents(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        y = a + b
        t = y.forward_grad({a.id: np.array([1.0, 0.0]), b.id: np.array([0.0, 1.0])})
        assert np.allclose(t[y.id], [1.0, 1.0])

    def test_mul(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        y = a * b
        t = y.forward_grad({a.id: np.array([1.0, 0.0])})
        assert np.allclose(t[y.id], [4.0, 0.0])
        t = y.forward_grad({b.id: np.array([0.0, 1.0])})
        assert np.allclose(t[y.id], [0.0, 3.0])

    def test_neg(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        y = -a
        t = y.forward_grad({a.id: np.array([1.0, 1.0])})
        assert np.allclose(t[y.id], [-1.0, -1.0])

    def test_pow(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        y = a ** 2
        t = y.forward_grad({a.id: np.array([1.0, 1.0])})
        # dy/da = 2*a, JVP = 2*a*t_a
        assert np.allclose(t[y.id], [4.0, 6.0])

    def test_matmul_2d(self):
        A = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        x = Tensor(np.array([1.0, 1.0]), requires_grad=True)
        y = A @ x
        t = y.forward_grad({x.id: np.array([1.0, 0.0])})
        assert np.allclose(t[y.id], [1.0, 3.0])

    def test_matmul_1d_grad(self):
        A = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        x = Tensor(np.random.randn(4).astype(np.float32), requires_grad=True)
        y = A @ x
        t = y.forward_grad({A.id: np.ones((3, 4), dtype=np.float32)})
        assert t[y.id] is not None
        assert t[y.id].shape == (3,)

    def test_transpose(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        y = _transpose(a)
        t = y.forward_grad({a.id: np.array([[1.0, 0.0], [0.0, 0.0]])})
        assert np.allclose(t[y.id], [[1.0, 0.0], [0.0, 0.0]])

    def test_reshape(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        y = a.reshape(4)
        t = y.forward_grad({a.id: np.ones((2, 2))})
        assert np.allclose(t[y.id], [1.0, 1.0, 1.0, 1.0])

    def test_sum(self):
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = a.sum()
        t = y.forward_grad({a.id: np.array([1.0, 1.0, 1.0])})
        assert np.allclose(t[y.id], 3.0)

    def test_mean(self):
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = a.mean()
        t = y.forward_grad({a.id: np.array([1.0, 1.0, 1.0])})
        assert np.allclose(t[y.id], 1.0)

    def test_max(self):
        a = Tensor([1.0, 5.0, 3.0], requires_grad=True)
        y = a.max()
        t = y.forward_grad({a.id: np.array([0.0, 1.0, 0.0])})
        assert np.allclose(t[y.id], 1.0)


class TestForwardGradActivations:
    """Forward-mode AD on activation functions."""

    def test_sigmoid(self):
        x = Tensor(np.array([0.0, 1.0]), requires_grad=True)
        y = sigmoid(x)
        t = y.forward_grad({x.id: np.array([1.0, 1.0])})
        s = 1.0 / (1.0 + np.exp(-np.array([0.0, 1.0])))
        assert np.allclose(t[y.id], s * (1 - s) * 1.0)

    def test_tanh(self):
        x = Tensor(np.array([0.0, 1.0]), requires_grad=True)
        y = tanh(x)
        t = y.forward_grad({x.id: np.array([1.0, 1.0])})
        assert np.allclose(t[y.id], (1 - np.tanh(x.data) ** 2) * 1.0)

    def test_relu(self):
        x = Tensor(np.array([-1.0, 0.0, 1.0]), requires_grad=True)
        y = relu(x)
        t = y.forward_grad({x.id: np.array([1.0, 1.0, 1.0])})
        expected = np.where(x.data > 0, 1.0, 0.0)
        assert np.allclose(t[y.id], expected)


class TestForwardGradNormalization:
    """Forward-mode AD on normalization layers."""

    def test_layernorm(self):
        x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
        w = Tensor(np.ones(4, dtype=np.float32), requires_grad=True)
        b = Tensor(np.zeros(4, dtype=np.float32), requires_grad=True)
        y = _layernorm(x, w, b)
        v = np.random.randn(2, 4).astype(np.float32)
        t = y.forward_grad({x.id: v})
        assert t[y.id] is not None
        assert t[y.id].shape == y.shape

    def test_rmsnorm(self):
        x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
        w = Tensor(np.ones(4, dtype=np.float32), requires_grad=True)
        y = _rmsnorm(x, w)
        v = np.random.randn(2, 4).astype(np.float32)
        t = y.forward_grad({x.id: v})
        assert t[y.id] is not None
        assert t[y.id].shape == y.shape


class TestForwardGradSoftmax:
    """Forward-mode AD on softmax."""

    def test_softmax(self):
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y = _softmax(x)
        s = y.data
        v = np.array([1.0, 0.0, 0.0])
        t = y.forward_grad({x.id: v})
        analytic = s * (v - (s * v).sum())
        assert np.allclose(t[y.id], analytic)


class TestForwardGradCrossEntropy:
    """Forward-mode AD on cross-entropy loss."""

    def test_cross_entropy(self):
        logits = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)
        targets = Tensor(np.array([2]))
        probs = _softmax(logits)
        loss = cross_entropy(probs, targets)
        v = np.random.randn(1, 3).astype(np.float32)
        t = loss.forward_grad({logits.id: v})
        assert t[loss.id] is not None
        assert t[loss.id].shape == ()


class TestDotProductConsistency:
    """Fundamental check: forward_grad(v)·w == v·backward(w) for all ops."""

    def _check(self, y, inputs, tangents):
        """Verify dot-product identity for given computation."""
        # Forward-mode
        fwd = y.forward_grad(tangents)
        jvp = fwd.get(y.id)

        # Backward-mode (all inputs share same seed w=1)
        y.grad = None
        y.backward()
        vjp = sum((inp.grad.data * tangents[inp.id]).sum()
                  for inp in inputs if inp.id in tangents)
        assert np.allclose(jvp.sum(), vjp, atol=1e-5), \
            f"JVP={jvp.sum():.6f} VJP={vjp:.6f}"

    def test_add_dot(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        y = a + b
        self._check(y, [a, b], {a.id: np.array([1.0, 0.0])})

    def test_mul_dot(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        y = a * b
        self._check(y, [a, b], {a.id: np.array([1.0, 0.0])})

    def test_matmul_dot(self):
        A = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        x = Tensor(np.random.randn(4).astype(np.float32), requires_grad=True)
        y = A @ x
        self._check(y, [A, x], {A.id: np.random.randn(3, 4).astype(np.float32)})

    def test_layernorm_dot(self):
        x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
        w = Tensor(np.ones(4, dtype=np.float32), requires_grad=True)
        b = Tensor(np.zeros(4, dtype=np.float32), requires_grad=True)
        y = _layernorm(x, w, b)
        self._check(y, [x, w, b], {x.id: np.random.randn(2, 4).astype(np.float32)})

    def test_rmsnorm_dot(self):
        x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
        w = Tensor(np.ones(4, dtype=np.float32), requires_grad=True)
        y = _rmsnorm(x, w)
        self._check(y, [x, w], {x.id: np.random.randn(2, 4).astype(np.float32)})

    def test_softmax_ce_dot(self):
        logits = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)
        targets = Tensor(np.array([2]))
        probs = _softmax(logits)
        loss = cross_entropy(probs, targets)
        self._check(y=loss, inputs=[logits],
                    tangents={logits.id: np.random.randn(1, 3).astype(np.float32)})

    def test_chain_dot(self):
        A = Tensor(np.random.randn(4, 4).astype(np.float32), requires_grad=True)
        x = Tensor(np.random.randn(4).astype(np.float32), requires_grad=True)
        y = sigmoid(relu(A @ x)).sum()
        self._check(y, [A, x], {x.id: np.random.randn(4).astype(np.float32)})


class TestConsumersGraph:
    """Test that _consumers edges are correctly tracked."""

    def test_add_consumers(self):
        a = Tensor([1.0], requires_grad=True)
        b = Tensor([2.0], requires_grad=True)
        y = a + b
        assert y in a._consumers
        assert y in b._consumers

    def test_chain_consumers(self):
        a = Tensor([1.0], requires_grad=True)
        b = Tensor([2.0], requires_grad=True)
        c = Tensor([3.0], requires_grad=True)
        s = a + b
        y = s * c
        assert any(t.id == s.id for t in a._consumers)
        assert any(t.id == s.id for t in b._consumers)
        assert any(t.id == y.id for t in s._consumers)
        assert any(t.id == y.id for t in c._consumers)

    def test_no_grad_no_consumers(self):
        with no_grad():
            a = Tensor([1.0], requires_grad=True)
            b = Tensor([2.0], requires_grad=True)
            y = a + b
        assert y._consumers == []





class TestBackwardRegression:
    """Ensure backward-mode still works correctly."""

    def test_backward_still_works(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        y = (a * b).sum()
        y.backward()
        assert np.allclose(a.grad.data, [3.0, 4.0])
        assert np.allclose(b.grad.data, [1.0, 2.0])

    def test_backward_chain(self):
        a = Tensor([1.0], requires_grad=True)
        b = Tensor([2.0], requires_grad=True)
        c = Tensor([3.0], requires_grad=True)
        y = (a + b) * c
        y.backward()
        assert np.allclose(a.grad.data, [3.0])
        assert np.allclose(b.grad.data, [3.0])
        assert np.allclose(c.grad.data, [3.0])
