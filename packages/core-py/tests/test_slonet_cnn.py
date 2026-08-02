"""Tests for SloNet CNN layers: conv2d, batchnorm2d, maxpool2d, flatten, cross-attention."""

import numpy as np

from domains.training.slonet import (
    Tensor, _conv2d, _batchnorm2d, _maxpool2d, _im2col, flatten,
    SloConv2D, SloBatchNorm2D, SloMaxPool2D, SloCrossAttention,
)


def _naive_conv(x: np.ndarray, wt: np.ndarray, bt: np.ndarray, stride=1, padding=0):
    """Reference conv2d via nested loops."""
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    n, c, h, w = x.shape
    oc, ic, kh, kw = wt.shape
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1
    out = np.zeros((n, oc, oh, ow), dtype=np.result_type(x.dtype, wt.dtype, bt.dtype))
    for ni in range(n):
        for oci in range(oc):
            for ohi in range(oh):
                for owi in range(ow):
                    ih = ohi * stride
                    iw = owi * stride
                    out[ni, oci, ohi, owi] = (x[ni, :, ih:ih + kh, iw:iw + kw] * wt[oci]).sum() + bt[oci]
    return out


def _fd_grad_x(x_np, w_np, b_np, stride=1, padding=0, eps=1e-6):
    """Numerical gradient of sum-of-squares loss w.r.t. x (float64 for precision)."""
    xf = x_np.astype(np.float64)
    wf = w_np.astype(np.float64)
    bf = b_np.astype(np.float64)
    loss = lambda xx: (_naive_conv(xx, wf, bf, stride, padding) ** 2).sum()
    g = np.zeros_like(xf)
    it = np.nditer(xf, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        xp = xf.copy(); xm = xf.copy()
        xp[i] += eps; xm[i] -= eps
        g[i] = (loss(xp) - loss(xm)) / (2 * eps)
        it.iternext()
    return g


def _fd_grad_w(x_np, w_np, b_np, stride=1, padding=0, eps=1e-6):
    """Numerical gradient of sum-of-squares loss w.r.t. weight."""
    xf = x_np.astype(np.float64)
    wf = w_np.astype(np.float64)
    bf = b_np.astype(np.float64)
    loss = lambda ww: (_naive_conv(xf, ww, bf, stride, padding) ** 2).sum()
    g = np.zeros_like(wf)
    it = np.nditer(wf, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        wp = wf.copy(); wm = wf.copy()
        wp[i] += eps; wm[i] -= eps
        g[i] = (loss(wp) - loss(wm)) / (2 * eps)
        it.iternext()
    return g


def _fd_grad_b(x_np, w_np, b_np, stride=1, padding=0, eps=1e-6):
    """Numerical gradient of sum-of-squares loss w.r.t. bias."""
    xf = x_np.astype(np.float64)
    wf = w_np.astype(np.float64)
    bf = b_np.astype(np.float64)
    loss = lambda bb: (_naive_conv(xf, wf, bb, stride, padding) ** 2).sum()
    g = np.zeros_like(bf)
    it = np.nditer(bf, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        bp = bf.copy(); bm = bf.copy()
        bp[i] += eps; bm[i] -= eps
        g[i] = (loss(bp) - loss(bm)) / (2 * eps)
        it.iternext()
    return g


class TestIm2Col:
    def test_shape(self):
        x = np.random.randn(2, 3, 5, 5).astype(np.float32)
        cols = _im2col(x, 3, 3, 1)
        assert cols.shape == (2 * 3 * 3, 3 * 3 * 3)

    def test_matches_naive_patches(self):
        x = np.random.randn(1, 1, 4, 4).astype(np.float32)
        cols = _im2col(x, 2, 2, 2)
        assert cols.shape == (4, 4)
        expected = x[0, 0, 0, 0], x[0, 0, 0, 1], x[0, 0, 1, 0], x[0, 0, 1, 1]
        assert np.allclose(cols[0], expected)


class TestConv2D:
    def test_forward_matches_naive(self):
        x = Tensor(np.random.randn(2, 2, 5, 5).astype(np.float32), requires_grad=True)
        w = Tensor(np.random.randn(3, 2, 3, 3).astype(np.float32) * 0.5, requires_grad=True)
        b = Tensor(np.random.randn(3).astype(np.float32) * 0.1, requires_grad=True)
        out = _conv2d(x, w, b, stride=1, padding=0)
        expected = _naive_conv(x.data, w.data, b.data, 1, 0)
        assert out.data.shape == (2, 3, 3, 3)
        assert np.allclose(out.data, expected, atol=1e-5)

    def test_forward_stride_and_tuple_padding(self):
        x = Tensor(np.random.randn(1, 1, 7, 7).astype(np.float32), requires_grad=True)
        w = Tensor(np.random.randn(2, 1, 3, 3).astype(np.float32) * 0.5, requires_grad=True)
        b = Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)
        out = _conv2d(x, w, b, stride=2, padding=(1, 2))
        # pad_h=1, pad_w=2 → (1,1,9,11) → oh=(9-3)//2+1=4, ow=(11-3)//2+1=5
        assert out.data.shape == (1, 2, 4, 5)

    def test_error_not_4d(self):
        x = Tensor(np.random.randn(2, 5, 5).astype(np.float32))
        w = Tensor(np.random.randn(3, 2, 3, 3).astype(np.float32))
        b = Tensor(np.zeros(3, dtype=np.float32))
        try:
            _conv2d(x, w, b)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_error_channel_mismatch(self):
        x = Tensor(np.random.randn(1, 4, 5, 5).astype(np.float32))
        w = Tensor(np.random.randn(3, 2, 3, 3).astype(np.float32))
        b = Tensor(np.zeros(3, dtype=np.float32))
        try:
            _conv2d(x, w, b)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_backward_matches_finite_difference(self):
        np.random.seed(7)
        x_np = np.random.randn(2, 2, 4, 4).astype(np.float32)
        w_np = (np.random.randn(2, 2, 2, 2).astype(np.float32) * 0.5)
        b_np = np.random.randn(2).astype(np.float32) * 0.1
        x = Tensor(x_np.copy(), requires_grad=True)
        w = Tensor(w_np.copy(), requires_grad=True)
        b = Tensor(b_np.copy(), requires_grad=True)
        out = _conv2d(x, w, b)
        loss = (out * out).sum()
        loss.backward()
        fd = _fd_grad_x(x_np, w_np, b_np, stride=1, padding=0)
        assert np.allclose(x.grad.data, fd, atol=1e-3)
        assert np.allclose(w.grad.data, _fd_grad_w(x_np, w_np, b_np, stride=1, padding=0), atol=1e-3)
        assert np.allclose(b.grad.data, _fd_grad_b(x_np, w_np, b_np, stride=1, padding=0), atol=1e-3)

    def test_backward_zero_grad_edges(self):
        x = Tensor(np.ones((1, 1, 4, 4), dtype=np.float32), requires_grad=True)
        w = Tensor(np.ones((1, 1, 2, 2), dtype=np.float32), requires_grad=True)
        b = Tensor(np.ones(1, dtype=np.float32), requires_grad=True)
        out = _conv2d(x, w, b, stride=2, padding=1)
        out.backward()
        assert x.grad is not None
        assert w.grad is not None
        assert b.grad is not None
        assert np.all(np.isfinite(x.grad.data))
        assert np.all(np.isfinite(w.grad.data))

    def test_jvp_dot_product(self):
        np.random.seed(3)
        x = Tensor(np.random.randn(2, 1, 4, 4).astype(np.float32), requires_grad=True)
        w = Tensor(np.random.randn(1, 1, 2, 2).astype(np.float32) * 0.5, requires_grad=True)
        b = Tensor(np.zeros(1, dtype=np.float32), requires_grad=True)
        out = _conv2d(x, w, b, stride=1, padding=0)
        tx = np.random.randn(*x.data.shape).astype(np.float32)
        tw = np.random.randn(*w.data.shape).astype(np.float32)
        fwd = out.forward_grad({x.id: tx, w.id: tw})
        jvp = fwd.get(out.id)
        out.grad = None
        out.backward()
        vjp = (x.grad.data * tx).sum() + (w.grad.data * tw).sum()
        assert np.allclose(jvp.sum(), vjp, atol=1e-4)

    def test_slo_conv2d_layer(self):
        conv = SloConv2D(2, 3, kernel_size=3, stride=1, padding=1, name="c1")
        assert conv.name == "c1"
        assert len(conv.parameters()) == 2
        assert conv.weight.data.shape == (3, 2, 3, 3)
        x = Tensor(np.random.randn(1, 2, 6, 6).astype(np.float32))
        out = conv.forward(x)
        assert out.data.shape == (1, 3, 6, 6)

    def test_slo_conv2d_tuple_kernel(self):
        conv = SloConv2D(1, 2, kernel_size=(3, 5), stride=1, padding=0)
        assert conv.kernel_size == (5, 3)
        x = Tensor(np.random.randn(1, 1, 8, 10).astype(np.float32))
        out = conv.forward(x)
        assert out.data.shape == (1, 2, 6, 6)


class TestBatchNorm2D:
    def test_training_forward_manual(self):
        np.random.seed(1)
        x = Tensor(np.random.randn(2, 3, 4, 4).astype(np.float32) * 2 + 1, requires_grad=True)
        g = Tensor(np.ones(3, dtype=np.float32), requires_grad=True)
        b = Tensor(np.zeros(3, dtype=np.float32), requires_grad=True)
        out = _batchnorm2d(x, g, b, np.zeros(3), np.ones(3), 1e-5, training=True)
        mean = x.data.mean(axis=(0, 2, 3), keepdims=True)
        var = x.data.var(axis=(0, 2, 3), keepdims=True)
        expected = (x.data - mean) / np.sqrt(var + 1e-5)
        assert np.allclose(out.data, expected, atol=1e-5)

    def test_eval_uses_running_stats(self):
        x = Tensor(np.random.randn(2, 2, 3, 3).astype(np.float32) * 5, requires_grad=True)
        g = Tensor(np.ones(2, dtype=np.float32), requires_grad=True)
        b = Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)
        running_mean = np.array([0.0, 1.0], dtype=np.float32)
        running_var = np.array([1.0, 4.0], dtype=np.float32)
        out = _batchnorm2d(x, g, b, running_mean, running_var, 1e-5, training=False)
        expected = (x.data - running_mean.reshape(1, 2, 1, 1)) / np.sqrt(running_var.reshape(1, 2, 1, 1) + 1e-5)
        assert np.allclose(out.data, expected, atol=1e-5)

    def test_backward_flow(self):
        np.random.seed(2)
        x = Tensor(np.random.randn(1, 2, 3, 3).astype(np.float32), requires_grad=True)
        g = Tensor(np.ones(2, dtype=np.float32), requires_grad=True)
        b = Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)
        out = _batchnorm2d(x, g, b, np.zeros(2), np.ones(2), 1e-5, training=True)
        (out * out).sum().backward()
        assert x.grad is not None
        assert g.grad is not None
        assert b.grad is not None
        assert np.all(np.isfinite(x.grad.data))

    def test_jvp_dot_product(self):
        np.random.seed(4)
        x = Tensor(np.random.randn(1, 2, 3, 3).astype(np.float32), requires_grad=True)
        g = Tensor(np.ones(2, dtype=np.float32), requires_grad=True)
        b = Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)
        out = _batchnorm2d(x, g, b, np.zeros(2), np.ones(2), 1e-5, training=True)
        tx = np.random.randn(*x.data.shape).astype(np.float32)
        fwd = out.forward_grad({x.id: tx})
        jvp = fwd.get(out.id)
        out.grad = None
        out.backward()
        vjp = (x.grad.data * tx).sum()
        assert np.allclose(jvp.sum(), vjp, atol=1e-4)

    def test_slo_batchnorm2d_layer(self):
        bn = SloBatchNorm2D(4, momentum=0.9, eps=1e-5, name="bn1")
        assert bn.name == "bn1"
        assert len(bn.parameters()) == 2
        assert np.allclose(bn.running_mean, np.zeros(4))
        assert np.allclose(bn.running_var, np.ones(4))
        x = Tensor(np.random.randn(1, 4, 3, 3).astype(np.float32))
        out = bn.forward(x)
        assert out.data.shape == (1, 4, 3, 3)


class TestMaxPool2D:
    def test_forward_values(self):
        x = Tensor(np.array([[[[1, 2], [3, 4]]]], dtype=np.float32), requires_grad=True)
        out = _maxpool2d(x, 2, 2)
        assert out.data[0, 0, 0, 0] == 4.0

    def test_forward_stride(self):
        np.random.seed(5)
        x = Tensor(np.random.randn(1, 1, 5, 5).astype(np.float32), requires_grad=True)
        out = _maxpool2d(x, 2, 2)
        assert out.data.shape == (1, 1, 2, 2)
        for oi in range(2):
            for oj in range(2):
                patch = x.data[0, 0, oi * 2:oi * 2 + 2, oj * 2:oj * 2 + 2]
                assert out.data[0, 0, oi, oj] == patch.max()

    def test_backward_scatters_to_argmax(self):
        x = Tensor(np.array([[[[1, 2], [3, 4]]]], dtype=np.float32), requires_grad=True)
        out = _maxpool2d(x, 2, 2)
        out.backward()
        assert x.grad.data[0, 0, 1, 1] == 1.0
        assert x.grad.data[0, 0, 0, 0] == 0.0

    def test_jvp_dot_product(self):
        np.random.seed(6)
        x = Tensor(np.random.randn(1, 1, 4, 4).astype(np.float32), requires_grad=True)
        out = _maxpool2d(x, 2, 2)
        tx = np.random.randn(*x.data.shape).astype(np.float32)
        fwd = out.forward_grad({x.id: tx})
        jvp = fwd.get(out.id)
        out.grad = None
        out.backward()
        vjp = (x.grad.data * tx).sum()
        assert np.allclose(jvp.sum(), vjp, atol=1e-4)

    def test_slo_maxpool2d_layer(self):
        pool = SloMaxPool2D(kernel_size=2, stride=2, name="p1")
        assert pool.name == "p1"
        assert pool.parameters() == []
        x = Tensor(np.random.randn(1, 3, 6, 6).astype(np.float32))
        out = pool.forward(x)
        assert out.data.shape == (1, 3, 3, 3)


class TestFlatten:
    def test_forward(self):
        x = Tensor(np.random.randn(2, 3, 4, 4).astype(np.float32), requires_grad=True)
        out = flatten(x)
        assert out.data.shape == (2, 48)

    def test_backward_restores_shape(self):
        x = Tensor(np.random.randn(2, 3, 4, 4).astype(np.float32), requires_grad=True)
        out = flatten(x)
        out.backward()
        assert x.grad.data.shape == (2, 3, 4, 4)

    def test_jvp_dot_product(self):
        np.random.seed(8)
        x = Tensor(np.random.randn(2, 2, 3, 3).astype(np.float32), requires_grad=True)
        out = flatten(x)
        tx = np.random.randn(*x.data.shape).astype(np.float32)
        fwd = out.forward_grad({x.id: tx})
        jvp = fwd.get(out.id)
        out.grad = None
        out.backward()
        vjp = (x.grad.data * tx).sum()
        assert np.allclose(jvp.sum(), vjp, atol=1e-4)


class TestCrossAttention:
    def _make(self, d_model=16, n_heads=4):
        return SloCrossAttention(d_model, n_heads, name="ca")

    def test_forward_shape(self):
        ca = self._make()
        x = Tensor(np.random.randn(2, 5, 16).astype(np.float32), requires_grad=True)
        context = Tensor(np.random.randn(2, 7, 16).astype(np.float32), requires_grad=True)
        out = ca.forward(x, context)
        assert out.data.shape == (2, 5, 16)
        assert np.all(np.isfinite(out.data))

    def test_forward_with_mask(self):
        ca = self._make()
        x = Tensor(np.random.randn(1, 3, 16).astype(np.float32), requires_grad=True)
        context = Tensor(np.random.randn(1, 4, 16).astype(np.float32), requires_grad=True)
        mask = Tensor(np.zeros((1, 4, 3, 4), dtype=np.float32))
        out = ca.forward(x, context, mask=mask)
        assert out.data.shape == (1, 3, 16)

    def test_backward_flows_to_projections(self):
        ca = self._make()
        x = Tensor(np.random.randn(2, 4, 16).astype(np.float32), requires_grad=True)
        context = Tensor(np.random.randn(2, 6, 16).astype(np.float32), requires_grad=True)
        out = ca.forward(x, context)
        (out * out).sum().backward()
        assert x.grad is not None
        assert context.grad is not None
        assert np.all(np.isfinite(x.grad.data))
        for p in ca.parameters():
            assert p.grad is not None
            assert np.all(np.isfinite(p.grad.data))

    def test_soul_traits(self):
        ca = self._make()
        assert set(ca.soul_traits) == {"curiosity", "creativity"}
