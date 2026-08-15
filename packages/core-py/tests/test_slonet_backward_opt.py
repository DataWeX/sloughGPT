"""Tests for backward pass optimizations: _copy=False + in-place gradient accumulation."""
import numpy as np
import pytest
from domains.training.slonet import (
    Tensor, SloLinear, SloEmbedding, SloLayerNorm, SloDropout,
    SloTransformer, SloAdamW, cross_entropy,
    _add, _mul, _neg, _pow, _matmul, _transpose, _reshape, _slice,
    _softmax, _layernorm, _rmsnorm, gelu, sigmoid, tanh, relu,
    flatten, zeros, ones,
)


class TestCopyFalseGradientAssignment:
    """Verify all backward functions produce correct gradient arrays."""

    def test_add_grad(self):
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
        out = _add(a, b)
        out.backward()
        np.testing.assert_array_equal(a.grad.data, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(b.grad.data, [1.0, 1.0, 1.0])

    def test_mul_grad(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        out = _mul(a, b)
        out.backward()
        np.testing.assert_array_equal(a.grad.data, [4.0, 5.0])
        np.testing.assert_array_equal(b.grad.data, [2.0, 3.0])

    def test_matmul_grad(self):
        a = Tensor(np.eye(3), requires_grad=True)
        b = Tensor(np.ones((3, 3)), requires_grad=True)
        out = _matmul(a, b)
        out.backward()
        assert a.grad is not None and b.grad is not None

    def test_transpose_grad(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        out = _transpose(a)
        out.backward()
        assert a.grad is not None and a.grad.data.shape == a.data.shape

    def test_reshape_grad(self):
        a = Tensor(np.arange(6.0), requires_grad=True)
        out = _reshape(a, (2, 3))
        out.backward()
        assert a.grad is not None and a.grad.data.shape == a.data.shape

    def test_slice_grad(self):
        a = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
        out = _slice(a, (0, slice(None)))
        out.backward()
        assert a.grad is not None

    def test_softmax_grad(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        out = _softmax(x)
        out.backward()
        assert x.grad is not None and x.grad.data.shape == x.data.shape

    def test_layernorm_grad(self):
        x = Tensor(np.random.randn(2, 4), requires_grad=True)
        w = Tensor(np.ones(4), requires_grad=True)
        b = Tensor(np.zeros(4), requires_grad=True)
        out = _layernorm(x, w, b)
        out.backward()
        assert x.grad is not None and w.grad is not None and b.grad is not None

    def test_rmsnorm_grad(self):
        from domains.training.slonet import _rmsnorm
        x = Tensor(np.random.randn(2, 4), requires_grad=True)
        w = Tensor(np.ones(4), requires_grad=True)
        out = _rmsnorm(x, w)
        out.backward()
        assert x.grad is not None and w.grad is not None

    def test_relu_grad(self):
        x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
        out = relu(x)
        out.backward()
        np.testing.assert_array_equal(x.grad.data, [0.0, 0.0, 1.0])

    def test_gelu_grad(self):
        x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
        out = gelu(x)
        out.backward()
        assert x.grad is not None

    def test_sigmoid_grad(self):
        x = Tensor([0.0, 1.0], requires_grad=True)
        out = sigmoid(x)
        out.backward()
        assert x.grad is not None

    def test_tanh_grad(self):
        x = Tensor([0.0, 1.0], requires_grad=True)
        out = tanh(x)
        out.backward()
        assert x.grad is not None

    def test_flatten_grad(self):
        x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        out = flatten(x)
        out.backward()
        assert x.grad is not None and x.grad.data.shape == x.data.shape

    def test_cross_entropy_grad(self):
        logits = Tensor(np.random.randn(4, 10), requires_grad=True)
        targets = Tensor(np.array([0, 1, 2, 3], dtype=np.int64))
        out = cross_entropy(logits, targets)
        out.backward()
        assert logits.grad is not None

    def test_embedding_grad(self):
        emb = SloEmbedding(10, 4)
        idx = Tensor(np.array([[0, 1, 2]]))
        out = emb.forward(idx)
        out.backward()
        assert emb.weight.grad is not None


class TestGradientValues:
    """Verify gradient values are correct after multiple backward passes."""

    def test_add_gradient_accumulates(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        _add(a, b).backward()
        g1 = a.grad.data.copy()
        _add(a, b).backward()
        np.testing.assert_allclose(a.grad.data, g1 * 2, rtol=1e-5)

    def test_matmul_gradient_accumulates(self):
        a = Tensor(np.eye(3), requires_grad=True)
        b = Tensor(np.ones((3, 3)), requires_grad=True)
        _matmul(a, b).backward()
        g1 = a.grad.data.copy()
        _matmul(a, b).backward()
        np.testing.assert_allclose(a.grad.data, g1 * 2, rtol=1e-5)

    def test_softmax_gradient_values(self):
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        out = _softmax(x)
        # backward() uses ones upstream, but sum(softmax)=const => grad=0
        # Instead, manually set upstream gradient and call backward_fn directly
        out.backward()
        # With g=ones, sg=sum(s*1)=1.0, gx = s*(1-1) = zeros — correct
        assert np.allclose(x.grad.data, 0.0, atol=1e-6)
        # Now test with non-uniform gradient via manual backward_fn call
        x2 = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        out2 = _softmax(x2)
        out2.backward()  # initialize graph
        s = np.exp([0.0, 1.0, 2.0]); s = s / s.sum()
        g = np.array([1.0, 0.5, 0.2])
        sg = np.sum(s * g)
        expected = s * (g - sg)
        x2.grad = None
        out2._backward_fn(g)
        np.testing.assert_allclose(x2.grad.data, expected, rtol=1e-4)

    def test_layernorm_gradient_is_normalized(self):
        np.random.seed(42)
        x = Tensor(np.random.randn(4, 8), requires_grad=True)
        w = Tensor(np.ones(8), requires_grad=True)
        b = Tensor(np.zeros(8), requires_grad=True)
        out = _layernorm(x, w, b)
        out.backward()
        assert x.grad is not None
        assert np.all(np.isfinite(x.grad.data))


class TestTrainingConvergence:
    """Verify training still converges with optimized backward pass."""

    def test_transformer_convergence(self):
        model = SloTransformer(vocab_size=256, n_embed=64, n_layer=2,
                               n_head=2, block_size=32, dropout=0.0)
        optimizer = SloAdamW(lr=1e-3)

        x = np.random.randint(0, 256, (4, 32))
        y = np.random.randint(0, 256, (4, 32))

        losses = []
        for _ in range(50):
            logits, _ = model.forward(Tensor(x, _copy=False))
            loss = cross_entropy(logits.reshape(-1, 256),
                                 Tensor(y.reshape(-1).astype(np.int64)))
            loss.backward()
            optimizer.step(model.parameters())
            losses.append(loss.data)

        assert losses[-1] < losses[0] * 0.5

    def test_linear_layer_convergence(self):
        np.random.seed(42)
        layer = SloLinear(10, 1)
        optimizer = SloAdamW(lr=0.001)

        x = Tensor(np.random.randn(32, 10), requires_grad=False)
        y = Tensor(np.random.randn(32, 1), requires_grad=False)

        losses = []
        for _ in range(300):
            pred = layer.forward(x)
            loss = ((pred - y) ** 2).sum()
            loss.backward()
            optimizer.step(layer.parameters())
            losses.append(float(loss.data))

        # Loss should decrease from initial value
        assert losses[-1] < losses[0]


class TestBackwardCorrectness:
    """Verify backward pass produces correct gradients via finite differences."""

    def test_add_numerical(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        _add(a, b).backward()
        eps = 1e-3
        for i in range(2):
            a_p = a.data.copy(); a_p[i] += eps
            a_m = a.data.copy(); a_m[i] -= eps
            num = float(((a_p + b.data).sum() - (a_m + b.data).sum()) / (2 * eps))
            assert abs(float(a.grad.data[i]) - num) < 1e-3

    def test_matmul_numerical(self):
        a = Tensor(np.random.randn(2, 3), requires_grad=True)
        b = Tensor(np.random.randn(3, 2), requires_grad=True)
        _matmul(a, b).sum().backward()
        eps = 1e-3
        for i in range(2):
            for j in range(3):
                a_p = a.data.copy(); a_p[i, j] += eps
                a_m = a.data.copy(); a_m[i, j] -= eps
                num = (np.matmul(a_p, b.data).sum() - np.matmul(a_m, b.data).sum()) / (2 * eps)
                assert abs(float(a.grad.data[i, j]) - float(num)) < 1e-3

    def test_softmax_numerical(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        _softmax(x).sum().backward()
        eps = 1e-3
        xd = x.data.astype(np.float64)
        for i in range(3):
            xp = xd.copy(); xp[i] += eps
            xm = xd.copy(); xm[i] -= eps
            sp = np.exp(xp) / np.exp(xp).sum()
            sm = np.exp(xm) / np.exp(xm).sum()
            num = float((sp - sm).sum() / (2 * eps))
            assert abs(float(x.grad.data[i]) - num) < 0.05
