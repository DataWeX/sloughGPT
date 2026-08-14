"""Gradient checks for all backward functions in slonet.

Uses numerical differentiation (finite differences) to verify analytical gradients.
This catches formula bugs that pattern-matching audits miss.
"""
import numpy as np
import pytest
from domains.training.slonet import (
    Tensor, _add, _mul, _pow, _matmul, _softmax, _sum, _mean, _max,
    sigmoid, tanh, relu, gelu, silu, log_softmax,
    _layernorm, _rmsnorm, _conv2d, _maxpool2d,
    _reshape, _slice, flatten,
    cross_entropy, normalize, pairwise_distance,
    SloLinear, SloTransformer,
)


@pytest.fixture(autouse=True)
def _seed():
    """Fix random seed before every test for reproducibility."""
    np.random.seed(42)


def _numerical_grad(fn, inputs, eps=1e-4, n_check=30):
    """Compute numerical gradient of fn w.r.t. each input using finite differences."""
    numerical = []
    for xi, x in enumerate(inputs):
        flat = x.data.ravel()
        g = np.zeros_like(flat, dtype=np.float64)
        check_indices = np.random.choice(len(flat), min(n_check, len(flat)), replace=False)
        for i in check_indices:
            orig = float(flat[i])
            flat[i] = orig + eps
            for xx in inputs: xx.grad = None
            r = fn()
            if isinstance(r, tuple): r = r[0]
            y_plus = r.data.ravel().astype(np.float64).copy()
            flat[i] = orig - eps
            for xx in inputs: xx.grad = None
            r = fn()
            if isinstance(r, tuple): r = r[0]
            y_minus = r.data.ravel().astype(np.float64).copy()
            flat[i] = orig
            g[i] = (y_plus.sum() - y_minus.sum()) / (2 * eps)
        numerical.append(g.reshape(x.data.shape))
    return numerical


def _analytical_grad(fn, inputs):
    """Compute analytical gradient by calling backward()."""
    for x in inputs: x.grad = None
    r = fn()
    if isinstance(r, tuple): r = r[0]
    r.backward()
    return [x.grad.data.copy().astype(np.float64) for x in inputs]


def _check_grad(name, fn_factory, eps=1e-4, atol=1e-3, n_check=30):
    """Check analytical vs numerical gradient. Returns (pass_bool, details)."""
    fn, inputs = fn_factory()
    analytical = _analytical_grad(fn, inputs)
    numerical = _numerical_grad(fn, inputs, eps=eps, n_check=n_check)
    failures = []
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        mask = np.abs(n.ravel()) > 1e-6
        if mask.sum() == 0:
            continue
        diff = np.abs(a.ravel()[mask] - n.ravel()[mask])
        rel = diff / (np.abs(n.ravel()[mask]) + 1e-8)
        max_abs = diff.max()
        max_rel = rel.max()
        if max_abs > atol and max_rel > 0.10:
            failures.append(f"  input{i}: max_abs={max_abs:.6f} max_rel={max_rel:.6f}")
    return len(failures) == 0, failures


R = dict(requires_grad=True)


class TestGradientChecks:
    """Numerical gradient checks for all backward functions."""

    def test_add(self):
        def f():
            a = Tensor([1.0, 2.0], **R)
            b = Tensor([3.0, 4.0], **R)
            return lambda: _add(a, b), [a, b]
        ok, details = _check_grad("add", f)
        assert ok, "\n".join(details)

    def test_mul(self):
        def f():
            a = Tensor([1.0, 2.0], **R)
            b = Tensor([3.0, 4.0], **R)
            return lambda: _mul(a, b), [a, b]
        ok, details = _check_grad("mul", f)
        assert ok, "\n".join(details)

    def test_pow(self):
        def f():
            p = Tensor([1.0, 2.0, 3.0], **R)
            return lambda: _pow(p, 3.0), [p]
        ok, details = _check_grad("pow", f)
        assert ok, "\n".join(details)

    def test_matmul(self):
        def f():
            x = Tensor(np.random.randn(3, 4), **R)
            W = Tensor(np.random.randn(4, 2), **R)
            return lambda: _matmul(x, W), [x, W]
        ok, details = _check_grad("matmul", f)
        assert ok, "\n".join(details)

    def test_softmax(self):
        def f():
            x = Tensor(np.random.randn(2, 3), **R)
            return lambda: _softmax(x), [x]
        ok, details = _check_grad("softmax", f)
        assert ok, "\n".join(details)

    def test_sigmoid(self):
        def f():
            x = Tensor(np.random.randn(5), **R)
            return lambda: sigmoid(x), [x]
        ok, details = _check_grad("sigmoid", f)
        assert ok, "\n".join(details)

    def test_tanh(self):
        def f():
            x = Tensor(np.random.randn(5), **R)
            return lambda: tanh(x), [x]
        ok, details = _check_grad("tanh", f)
        assert ok, "\n".join(details)

    def test_relu(self):
        def f():
            x = Tensor(np.random.randn(5), **R)
            return lambda: relu(x), [x]
        ok, details = _check_grad("relu", f)
        assert ok, "\n".join(details)

    def test_gelu(self):
        def f():
            x = Tensor(np.random.randn(5), **R)
            return lambda: gelu(x), [x]
        ok, details = _check_grad("gelu", f)
        assert ok, "\n".join(details)

    def test_silu(self):
        def f():
            x = Tensor(np.random.randn(5), **R)
            return lambda: silu(x), [x]
        ok, details = _check_grad("silu", f)
        assert ok, "\n".join(details)

    def test_log_softmax(self):
        def f():
            x = Tensor(np.random.randn(5), **R)
            return lambda: log_softmax(x), [x]
        ok, details = _check_grad("log_softmax", f)
        assert ok, "\n".join(details)

    def test_cross_entropy(self):
        def f():
            x = Tensor(np.random.randn(4, 5), **R)
            t = Tensor(np.array([0, 2, 1, 3]))
            return lambda: cross_entropy(x, t), [x]
        ok, details = _check_grad("cross_entropy", f, eps=1e-3)
        assert ok, "\n".join(details)

    def test_layernorm(self):
        def f():
            x = Tensor(np.random.randn(2, 4), **R)
            w = Tensor(np.random.randn(4), **R)
            b = Tensor(np.random.randn(4), **R)
            return lambda: _layernorm(x, w, b), [x, w, b]
        ok, details = _check_grad("layernorm", f)
        assert ok, "\n".join(details)

    def test_rmsnorm(self):
        def f():
            x = Tensor(np.random.randn(2, 4), **R)
            w = Tensor(np.random.randn(4), **R)
            return lambda: _rmsnorm(x, w), [x, w]
        ok, details = _check_grad("rmsnorm", f)
        assert ok, "\n".join(details)

    def test_normalize(self):
        def f():
            x = Tensor(np.random.randn(3, 4), **R)
            return lambda: normalize(x), [x]
        ok, details = _check_grad("normalize", f)
        assert ok, "\n".join(details)

    def test_pairwise_distance(self):
        def f():
            x1 = Tensor(np.random.randn(2, 4), **R)
            x2 = Tensor(np.random.randn(2, 4), **R)
            return lambda: pairwise_distance(x1, x2), [x1, x2]
        ok, details = _check_grad("pairwise_distance", f)
        assert ok, "\n".join(details)

    def test_conv2d(self):
        def f():
            x = Tensor(np.random.randn(1, 1, 5, 5), **R)
            w = Tensor(np.random.randn(2, 1, 3, 3), **R)
            b = Tensor(np.random.randn(2), **R)
            return lambda: _conv2d(x, w, b), [x, w, b]
        ok, details = _check_grad("conv2d", f, eps=1e-4, atol=5e-3)
        assert ok, "\n".join(details)

    def test_maxpool2d(self):
        def f():
            x = Tensor(np.random.randn(1, 1, 4, 4), **R)
            return lambda: _maxpool2d(x, 2, 2), [x]
        ok, details = _check_grad("maxpool2d", f)
        assert ok, "\n".join(details)

    def test_flatten(self):
        def f():
            x = Tensor(np.random.randn(2, 3), **R)
            return lambda: flatten(x), [x]
        ok, details = _check_grad("flatten", f)
        assert ok, "\n".join(details)

    def test_transpose(self):
        def f():
            x = Tensor(np.random.randn(3, 4), **R)
            return lambda: x.T(), [x]
        ok, details = _check_grad("transpose", f)
        assert ok, "\n".join(details)

    def test_reshape(self):
        def f():
            x = Tensor(np.random.randn(3, 4), **R)
            return lambda: _reshape(x, (4, 3)), [x]
        ok, details = _check_grad("reshape", f)
        assert ok, "\n".join(details)

    def test_slice(self):
        def f():
            x = Tensor(np.random.randn(3, 4), **R)
            return lambda: x[:, 1:3], [x]
        ok, details = _check_grad("slice", f)
        assert ok, "\n".join(details)

    def test_sum(self):
        def f():
            x = Tensor(np.random.randn(3, 4), **R)
            return lambda: _sum(x), [x]
        ok, details = _check_grad("sum", f)
        assert ok, "\n".join(details)

    def test_mean(self):
        def f():
            x = Tensor(np.random.randn(3, 4), **R)
            return lambda: _mean(x), [x]
        ok, details = _check_grad("mean", f)
        assert ok, "\n".join(details)

    def test_max(self):
        def f():
            x = Tensor(np.random.randn(3, 4), **R)
            return lambda: _max(x), [x]
        ok, details = _check_grad("max", f)
        assert ok, "\n".join(details)

    def test_SloLinear(self):
        def f():
            lin = SloLinear(8, 4)
            x = Tensor(np.random.randn(2, 8), **R)
            return lambda: lin(x), [x, lin.weight, lin.bias]
        ok, details = _check_grad("SloLinear", f)
        assert ok, "\n".join(details)


class TestForwardFnConsistency:
    """Verify forward_fn (JVP) matches backward gradient.

    For each op, forward mode (JVP) and backward mode (VJP) must agree:
    t_out . grad = (d_out/d_in . t_in).sum()
    """

    def test_add_forward_backward(self):
        a = Tensor([1.0, 2.0], **R)
        b = Tensor([3.0, 4.0], **R)
        out = _add(a, b)
        t_a = np.array([1.0, 0.0])
        t_b = np.array([0.0, 1.0])
        fwd_out = out._forward_fn(t_a, t_b)
        out.backward()
        vjp_a = (a.grad.data * t_a).sum() + (b.grad.data * t_b).sum()
        # forward: t_out = t_a + t_b; backward gives a.grad + b.grad (broadcast)
        # for this simple add, a.grad=[1,1], b.grad=[1,1]
        # vjp = 1*1 + 0*1 + 0*1 + 1*1 = 2
        # fwd_out = [1,1]; sum = 2
        assert np.allclose(fwd_out.sum(), vjp_a, atol=1e-5)

    def test_mul_forward_backward(self):
        a = Tensor([2.0, 3.0], **R)
        b = Tensor([4.0, 5.0], **R)
        out = _mul(a, b)
        t_a = np.array([1.0, 0.0])
        t_b = np.array([0.0, 1.0])
        fwd_out = out._forward_fn(t_a, t_b)
        out.backward()
        vjp = (a.grad.data * t_a).sum() + (b.grad.data * t_b).sum()
        assert np.allclose(fwd_out.sum(), vjp, atol=1e-5)

    def test_matmul_forward_backward(self):
        x = Tensor(np.random.randn(2, 3), **R)
        W = Tensor(np.random.randn(3, 4), **R)
        out = _matmul(x, W)
        t_x = np.random.randn(2, 3)
        t_W = np.random.randn(3, 4)
        fwd_out = out._forward_fn(t_x, t_W)
        out.backward()
        vjp = (x.grad.data * t_x).sum() + (W.grad.data * t_W).sum()
        assert np.allclose(fwd_out.sum(), vjp, atol=1e-5)

    def test_softmax_forward_backward(self):
        x = Tensor(np.random.randn(2, 3), **R)
        out = _softmax(x)
        t_x = np.random.randn(2, 3)
        fwd_out = out._forward_fn(t_x)
        out.backward()
        vjp = (x.grad.data * t_x).sum()
        assert np.allclose(fwd_out.sum(), vjp, atol=1e-5)

    def test_sigmoid_forward_backward(self):
        x = Tensor(np.random.randn(4), **R)
        out = sigmoid(x)
        t_x = np.random.randn(4)
        fwd_out = out._forward_fn(t_x)
        out.backward()
        vjp = (x.grad.data * t_x).sum()
        assert np.allclose(fwd_out.sum(), vjp, atol=1e-5)

    def test_tanh_forward_backward(self):
        x = Tensor(np.random.randn(4), **R)
        out = tanh(x)
        t_x = np.random.randn(4)
        fwd_out = out._forward_fn(t_x)
        out.backward()
        vjp = (x.grad.data * t_x).sum()
        assert np.allclose(fwd_out.sum(), vjp, atol=1e-5)

    def test_relu_forward_backward(self):
        x = Tensor(np.random.randn(4), **R)
        out = relu(x)
        t_x = np.random.randn(4)
        fwd_out = out._forward_fn(t_x)
        out.backward()
        vjp = (x.grad.data * t_x).sum()
        assert np.allclose(fwd_out.sum(), vjp, atol=1e-5)

    def test_gelu_forward_backward(self):
        x = Tensor(np.random.randn(4), **R)
        out = gelu(x)
        t_x = np.random.randn(4)
        fwd_out = out._forward_fn(t_x)
        out.backward()
        vjp = (x.grad.data * t_x).sum()
        assert np.allclose(fwd_out.sum(), vjp, atol=1e-5)

    def test_silu_forward_backward(self):
        x = Tensor(np.random.randn(4), **R)
        out = silu(x)
        t_x = np.random.randn(4)
        fwd_out = out._forward_fn(t_x)
        out.backward()
        vjp = (x.grad.data * t_x).sum()
        assert np.allclose(fwd_out.sum(), vjp, atol=1e-5)

    def test_layernorm_forward_backward(self):
        x = Tensor(np.random.randn(2, 4), **R)
        w = Tensor(np.random.randn(4), **R)
        b = Tensor(np.random.randn(4), **R)
        out = _layernorm(x, w, b)
        t_x = np.random.randn(2, 4)
        t_w = np.random.randn(4)
        t_b = np.random.randn(4)
        fwd_out = out._forward_fn(t_x, t_w, t_b)
        out.backward()
        vjp = (x.grad.data * t_x).sum() + (w.grad.data * t_w).sum() + (b.grad.data * t_b).sum()
        assert np.allclose(fwd_out.sum(), vjp, atol=1e-4)

    def test_rmsnorm_forward_backward(self):
        x = Tensor(np.random.randn(2, 4), **R)
        w = Tensor(np.random.randn(4), **R)
        out = _rmsnorm(x, w)
        t_x = np.random.randn(2, 4)
        t_w = np.random.randn(4)
        fwd_out = out._forward_fn(t_x, t_w)
        out.backward()
        vjp = (x.grad.data * t_x).sum() + (w.grad.data * t_w).sum()
        assert np.allclose(fwd_out.sum(), vjp, atol=1e-4)

    def test_cross_entropy_forward_backward(self):
        x = Tensor(np.random.randn(4, 5), **R)
        t = Tensor(np.array([0, 2, 1, 3]))
        out = cross_entropy(x, t)
        t_x = np.random.randn(4, 5)
        fwd_out = out._forward_fn(t_x)
        out.backward()
        vjp = (x.grad.data * t_x).sum()
        assert np.allclose(fwd_out, vjp, atol=1e-4)

    def test_log_softmax_forward_backward(self):
        x = Tensor(np.random.randn(4, 5), **R)
        out = log_softmax(x)
        t_x = np.random.randn(4, 5)
        fwd_out = out._forward_fn(t_x)
        out.backward()
        vjp = (x.grad.data * t_x).sum()
        assert np.allclose(fwd_out.sum(), vjp, atol=1e-5)

    def test_sum_forward_backward(self):
        x = Tensor(np.random.randn(3, 4), **R)
        out = _sum(x)
        t_x = np.random.randn(3, 4)
        fwd_out = out._forward_fn(t_x)
        out.backward()
        vjp = (x.grad.data * t_x).sum()
        assert np.allclose(fwd_out, vjp, atol=1e-5)

    def test_mean_forward_backward(self):
        x = Tensor(np.random.randn(3, 4), **R)
        out = _mean(x)
        t_x = np.random.randn(3, 4)
        fwd_out = out._forward_fn(t_x)
        out.backward()
        vjp = (x.grad.data * t_x).sum()
        assert np.allclose(fwd_out, vjp, atol=1e-5)

    def test_max_forward_backward(self):
        x = Tensor(np.random.randn(3, 4), **R)
        out = _max(x)
        t_x = np.random.randn(3, 4)
        fwd_out = out._forward_fn(t_x)
        out.backward()
        vjp = (x.grad.data * t_x).sum()
        assert np.allclose(fwd_out, vjp, atol=1e-5)

    def test_pow_forward_backward(self):
        x = Tensor(np.random.randn(3) + 1.0, **R)
        out = _pow(x, 3.0)
        t_x = np.random.randn(3)
        fwd_out = out._forward_fn(t_x)
        out.backward()
        vjp = (x.grad.data * t_x).sum()
        assert np.allclose(fwd_out.sum(), vjp, atol=1e-5)


class TestTrainingConvergence:
    """Verify the full model trains end-to-end."""

    def test_convergence(self):
        np.random.seed(42)
        model = SloTransformer(
            vocab_size=256, n_embed=64, n_layer=2, n_head=2,
            block_size=32, dropout=0.0,
        )
        from domains.training.slonet import SloAdamW
        opt = SloAdamW(lr=1e-3)
        x = np.random.randint(0, 256, (4, 32))
        y = np.random.randint(0, 256, (4, 32))
        losses = []
        for step in range(100):
            logits, _ = model.forward(Tensor(x, _copy=False))
            loss = cross_entropy(logits.reshape(-1, 256), Tensor(y.reshape(-1)))
            loss.backward()
            opt.step(model.parameters())
            losses.append(float(loss.data))
        assert losses[-1] < losses[0], f"No convergence: {losses[0]:.4f} -> {losses[-1]:.4f}"
        assert losses[-1] < 0.1, f"Loss too high at end: {losses[-1]:.4f}"
