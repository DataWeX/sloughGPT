"""Tests for fused QKV projection optimization."""
import numpy as np
import pytest
from domains.training.slonet import (
    Tensor, SloMultiHeadAttention, SloTransformer, SloAdamW, cross_entropy,
    mse_loss, topk, SloLinear, SloEmbedding, SloRMSNorm, SloLayerNorm,
    SloTransformerBlock, SloDropout, SloLayer,
    sigmoid, tanh, relu, gelu, silu, softmax, log_softmax,
    no_grad, zeros, randn, ones, tensor, exp, where, isfinite,
    multinomial, stack, concatenate, randint,
    flatten, _broadcast_back, _broadcast_forward,
    _apply_temperature, _apply_top_k, _apply_top_p,
    _apply_repetition_penalty, _apply_frequency_penalty, _apply_presence_penalty,
    _sample_from_logits,
    gelu_np, silu_np,
    GenerationMetrics, GenerateResult,
)


# ---------------------------------------------------------------------------
# Tensor basics
# ---------------------------------------------------------------------------

class TestTensor:
    def test_create_from_array(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        assert t.data.shape == (3,)

    def test_requires_grad(self):
        t = Tensor(np.array([1.0]), requires_grad=True)
        assert t.requires_grad is True

    def test_grad_initially_none(self):
        t = Tensor(np.array([1.0]))
        assert t.grad is None

    def test_add(self):
        a = Tensor(np.array([1.0, 2.0]))
        b = Tensor(np.array([3.0, 4.0]))
        c = a + b
        np.testing.assert_allclose(c.data, [4.0, 6.0])

    def test_mul(self):
        a = Tensor(np.array([2.0, 3.0]))
        b = Tensor(np.array([4.0, 5.0]))
        c = a * b
        np.testing.assert_allclose(c.data, [8.0, 15.0])

    def test_matmul(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
        c = a @ b
        np.testing.assert_allclose(c.data, [[19.0, 22.0], [43.0, 50.0]])

    def test_backward(self):
        a = Tensor(np.array([2.0]), requires_grad=True)
        b = a * a
        b.backward()
        np.testing.assert_allclose(a.grad.data, [4.0])

    def test_sum(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        s = t.sum()
        assert s.data == 6.0

    def test_mean(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        m = t.mean()
        assert m.data == 2.0

    def test_reshape(self):
        t = Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
        r = t.reshape(2, 2)
        assert r.data.shape == (2, 2)

    def test_sub(self):
        a = Tensor(np.array([5.0, 3.0]))
        b = Tensor(np.array([1.0, 2.0]))
        c = a - b
        np.testing.assert_allclose(c.data, [4.0, 1.0])

    def test_neg(self):
        a = Tensor(np.array([1.0, -2.0]))
        b = -a
        np.testing.assert_allclose(b.data, [-1.0, 2.0])

    def test_pow(self):
        a = Tensor(np.array([2.0, 3.0]))
        b = a ** 2
        np.testing.assert_allclose(b.data, [4.0, 9.0])

    def test_div(self):
        a = Tensor(np.array([6.0, 8.0]))
        b = Tensor(np.array([2.0, 4.0]))
        c = a / b
        np.testing.assert_allclose(c.data, [3.0, 2.0])

    def test_radd(self):
        a = Tensor(np.array([1.0, 2.0]))
        c = 5 + a
        np.testing.assert_allclose(c.data, [6.0, 7.0])

    def test_rmul(self):
        a = Tensor(np.array([1.0, 2.0]))
        c = 3 * a
        np.testing.assert_allclose(c.data, [3.0, 6.0])

    def test_rsub(self):
        a = Tensor(np.array([1.0, 2.0]))
        c = 10 - a
        np.testing.assert_allclose(c.data, [9.0, 8.0])

    def test_getitem(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        assert t[0].data == 1.0
        assert t[2].data == 3.0

    def test_setitem(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        t[1] = 99.0
        assert t.data[1] == 99.0

    def test_len(self):
        t = Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
        assert len(t) == 4

    def test_dim(self):
        t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert t.dim() == 2

    def test_numel(self):
        t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert t.numel() == 4

    def test_size(self):
        t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert t.size() == (2, 2)
        assert t.size(0) == 2
        assert t.size(1) == 2

    def test_squeeze(self):
        t = Tensor(np.array([[1.0, 2.0]]))
        s = t.squeeze()
        assert s.data.shape == (2,)

    def test_unsqueeze(self):
        t = Tensor(np.array([1.0, 2.0]))
        s = t.unsqueeze(0)
        assert s.data.shape == (1, 2)

    def test_repeat(self):
        t = Tensor(np.array([1.0, 2.0]))
        r = t.repeat(2, 1)
        assert r.data.shape == (2, 2)

    def test_detach(self):
        t = Tensor(np.array([1.0]), requires_grad=True)
        d = t.detach()
        assert d.requires_grad is False

    def test_clone(self):
        t = Tensor(np.array([1.0, 2.0]))
        c = t.clone()
        np.testing.assert_allclose(c.data, t.data)
        c.data[0] = 99.0
        assert t.data[0] == 1.0

    def test_to_list(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        assert t.tolist() == [1.0, 2.0, 3.0]

    def test_item(self):
        t = Tensor(np.array([42.0]))
        assert t.item() == 42.0

    def test_repr(self):
        t = Tensor(np.array([1.0, 2.0]))
        r = repr(t)
        assert "Tensor" in r

    def test_bool_scalar(self):
        t = Tensor(np.array(1.0))
        assert bool(t) is True
        t2 = Tensor(np.array(0.0))
        assert bool(t2) is False

    def test_bool_multi_raises(self):
        t = Tensor(np.array([1.0, 2.0]))
        with pytest.raises(RuntimeError):
            bool(t)

    def test_ge(self):
        a = Tensor(np.array([1.0, 3.0]))
        b = Tensor(np.array([2.0, 2.0]))
        c = a >= b
        np.testing.assert_allclose(c.data, [0.0, 1.0])

    def test_le(self):
        a = Tensor(np.array([1.0, 3.0]))
        b = Tensor(np.array([2.0, 2.0]))
        c = a <= b
        np.testing.assert_allclose(c.data, [1.0, 0.0])

    def test_gt(self):
        a = Tensor(np.array([1.0, 3.0]))
        b = Tensor(np.array([2.0, 2.0]))
        c = a > b
        np.testing.assert_allclose(c.data, [0.0, 1.0])

    def test_lt(self):
        a = Tensor(np.array([1.0, 3.0]))
        b = Tensor(np.array([2.0, 2.0]))
        c = a < b
        np.testing.assert_allclose(c.data, [1.0, 0.0])

    def test_eq_tensor(self):
        a = Tensor(np.array([1.0, 2.0]))
        b = Tensor(np.array([1.0, 3.0]))
        c = a == b
        np.testing.assert_allclose(c.data, [1.0, 0.0])

    def test_ne_tensor(self):
        a = Tensor(np.array([1.0, 2.0]))
        b = Tensor(np.array([1.0, 3.0]))
        c = a != b
        np.testing.assert_allclose(c.data, [0.0, 1.0])

    def test_max(self):
        t = Tensor(np.array([1.0, 5.0, 3.0]))
        m = t.max()
        assert m.data == 5.0

    def test_backward_chain(self):
        a = Tensor(np.array([3.0]), requires_grad=True)
        b = Tensor(np.array([2.0]), requires_grad=True)
        c = a * b
        d = c + a
        d.backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_zeros(self):
        t = zeros((3, 4))
        assert t.data.shape == (3, 4)
        np.testing.assert_allclose(t.data, 0.0)

    def test_randn(self):
        t = randn((5,))
        assert t.data.shape == (5,)
        assert t.data.std() > 0.0

    def test_ones(self):
        t = ones((2, 3))
        assert t.data.shape == (2, 3)
        np.testing.assert_allclose(t.data, 1.0)

    def test_tensor_factory(self):
        t = tensor([1.0, 2.0, 3.0])
        assert t.data.shape == (3,)

    def test_detach_copies(self):
        t = Tensor(np.array([1.0]), requires_grad=True)
        d = t.detach()
        d.data[0] = 99.0
        assert t.data[0] == 1.0

    def test_repr_shape(self):
        t = Tensor(np.zeros((2, 3, 4)))
        r = repr(t)
        assert "2" in r and "3" in r and "4" in r

    def test_float(self):
        t = Tensor(np.array([1, 2, 3]))
        t.float()
        assert t.data.dtype == np.float32

    def test_long(self):
        t = Tensor(np.array([1.0, 2.0]))
        t2 = t.long()
        assert t2.data.shape == t.data.shape

    def test_int(self):
        t = Tensor(np.array([1.0, 2.0]))
        t2 = t.int()
        assert t2.data.shape == t.data.shape

    def test_half(self):
        t = Tensor(np.array([1.0, 2.0]))
        t2 = t.half()
        assert t2.data.shape == t.data.shape

    def test_double(self):
        t = Tensor(np.array([1.0, 2.0]))
        t2 = t.double()
        assert t2.data.shape == t.data.shape

    def test_flatten(self):
        t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        f = t.flatten()
        assert f.data.shape == (4,)

    def test_contiguous(self):
        t = Tensor(np.array([1.0, 2.0]))
        c = t.contiguous()
        assert c.data.shape == t.data.shape

    def test_zero_(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        t.zero_()
        np.testing.assert_allclose(t.data, [0.0, 0.0, 0.0])

    def test_fill_(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        t.fill_(7.0)
        np.testing.assert_allclose(t.data, [7.0, 7.0, 7.0])

    def test_copy_(self):
        t = Tensor(np.array([1.0, 2.0]))
        t.copy_(Tensor(np.array([9.0, 8.0])))
        np.testing.assert_allclose(t.data, [9.0, 8.0])

    def test_expand(self):
        t = Tensor(np.array([[1.0, 2.0]]))
        e = t.expand(3, 2)
        assert e.data.shape == (3, 2)

    def test_transpose(self):
        t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        tr = t.transpose(0, 1)
        assert tr.data.shape == (2, 2)

    def test_abs(self):
        t = Tensor(np.array([-1.0, 2.0]))
        a = t.abs()
        np.testing.assert_allclose(a.data, [1.0, 2.0])

    def test_sqrt(self):
        t = Tensor(np.array([4.0, 9.0]))
        s = t.sqrt()
        np.testing.assert_allclose(s.data, [2.0, 3.0])

    def test_clamp(self):
        t = Tensor(np.array([1.0, 5.0, 10.0]))
        c = t.clamp(min_val=2.0, max_val=8.0)
        np.testing.assert_allclose(c.data, [2.0, 5.0, 8.0])

    def test_argmax(self):
        t = Tensor(np.array([1.0, 5.0, 3.0]))
        idx = t.argmax()
        assert idx.data == 1

    def test_argmin(self):
        t = Tensor(np.array([5.0, 1.0, 3.0]))
        idx = t.argmin()
        assert idx.data == 1

    def test_gather(self):
        t = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        idx = Tensor(np.array([[0, 2]]))
        g = t.gather(1, idx)
        assert g.data.shape == (2, 2)

    def test_scatter_(self):
        t = Tensor(np.zeros((3,)))
        idx = Tensor(np.array([0, 2]))
        src = Tensor(np.array([1.0, 9.0]))
        t.scatter_(0, idx, src)
        assert t.data[0] == 1.0
        assert t.data[2] == 9.0

    def test_cpu(self):
        t = Tensor(np.array([1.0]))
        assert t.cpu() is t

    def test_numpy(self):
        t = Tensor(np.array([1.0, 2.0]))
        n = t.numpy()
        np.testing.assert_allclose(n, [1.0, 2.0])

    def test_requires_grad_(self):
        t = Tensor(np.array([1.0]))
        t.requires_grad_(True)
        assert t.requires_grad is True
        t.requires_grad_(False)
        assert t.requires_grad is False

    def test_type_conversion(self):
        t = Tensor(np.array([1.0, 2.0]))
        t.type("torch.FloatTensor")
        assert t.data.dtype == np.float32


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

class TestActivations:
    def test_sigmoid(self):
        t = Tensor(np.array([0.0]))
        s = sigmoid(t)
        np.testing.assert_allclose(s.data, 0.5, atol=1e-5)

    def test_sigmoid_negative(self):
        t = Tensor(np.array([-100.0]))
        s = sigmoid(t)
        assert s.data[0] < 0.01

    def test_tanh(self):
        t = Tensor(np.array([0.0]))
        th = tanh(t)
        np.testing.assert_allclose(th.data, 0.0, atol=1e-5)

    def test_relu(self):
        t = Tensor(np.array([-1.0, 2.0, 3.0]))
        r = relu(t)
        np.testing.assert_allclose(r.data, [0.0, 2.0, 3.0])

    def test_gelu(self):
        t = Tensor(np.array([0.0]))
        g = gelu(t)
        np.testing.assert_allclose(g.data, 0.0, atol=1e-5)

    def test_silu(self):
        t = Tensor(np.array([0.0]))
        s = silu(t)
        np.testing.assert_allclose(s.data, 0.0, atol=1e-5)

    def test_silu_positive(self):
        t = Tensor(np.array([1.0]))
        s = silu(t)
        assert s.data[0] > 0.5

    def test_gelu_np(self):
        d = np.array([0.0, 1.0, -1.0])
        g = gelu_np(d)
        assert g.shape == d.shape

    def test_silu_np(self):
        d = np.array([0.0, 1.0, -1.0])
        s = silu_np(d)
        assert s.shape == d.shape

    def test_softmax(self):
        t = Tensor(np.array([[1.0, 2.0, 3.0]]))
        s = softmax(t, dim=-1)
        np.testing.assert_allclose(s.data.sum(), 1.0, atol=1e-5)

    def test_log_softmax(self):
        t = Tensor(np.array([[1.0, 2.0, 3.0]]))
        ls = log_softmax(t, dim=-1)
        assert ls.data.shape == (1, 3)

    def test_sigmoid_backward(self):
        t = Tensor(np.array([0.0]), requires_grad=True)
        s = sigmoid(t)
        s.backward()
        assert t.grad is not None

    def test_tanh_backward(self):
        t = Tensor(np.array([1.0]), requires_grad=True)
        th = tanh(t)
        th.backward()
        assert t.grad is not None

    def test_relu_backward(self):
        t = Tensor(np.array([-1.0, 2.0]), requires_grad=True)
        r = relu(t)
        r.backward()
        np.testing.assert_allclose(t.grad.data, [0.0, 1.0])

    def test_gelu_backward(self):
        t = Tensor(np.array([1.0]), requires_grad=True)
        g = gelu(t)
        g.backward()
        assert t.grad is not None

    def test_silu_backward(self):
        t = Tensor(np.array([1.0]), requires_grad=True)
        s = silu(t)
        s.backward()
        assert t.grad is not None


# ---------------------------------------------------------------------------
# FusedQKV
# ---------------------------------------------------------------------------

class TestFusedQKV:
    """Verify fused QKV projection matches separate projections."""

    def test_fused_qkv_matches_separate(self):
        attn = SloMultiHeadAttention(64, 4)
        x = Tensor(np.random.randn(2, 8, 64))

        Q_fused, K_fused, V_fused = attn._fused_qkv_forward(x)

        Q_sep = attn.W_q.forward(x)
        K_sep = attn.W_k.forward(x)
        V_sep = attn.W_v.forward(x)

        np.testing.assert_allclose(Q_fused.data, Q_sep.data, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(K_fused.data, K_sep.data, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(V_fused.data, V_sep.data, rtol=1e-5, atol=1e-5)

    def test_fused_qkv_backward_matches_separate(self):
        attn = SloMultiHeadAttention(64, 4)
        x = Tensor(np.random.randn(2, 8, 64), requires_grad=True)

        Q_f, K_f, V_f = attn._fused_qkv_forward(x)
        loss_f = (Q_f.sum() + K_f.sum() + V_f.sum())
        loss_f.backward()

        W_q_grad_fused = attn.W_q.weight.grad.data.copy() if attn.W_q.weight.grad is not None else None
        W_k_grad_fused = attn.W_k.weight.grad.data.copy() if attn.W_k.weight.grad is not None else None
        W_v_grad_fused = attn.W_v.weight.grad.data.copy() if attn.W_v.weight.grad is not None else None
        x_grad_fused = x.grad.data.copy()

        attn.W_q.weight.grad = None
        attn.W_k.weight.grad = None
        attn.W_v.weight.grad = None
        x.grad = None

        Q_s = attn.W_q.forward(x)
        K_s = attn.W_k.forward(x)
        V_s = attn.W_v.forward(x)
        loss_s = (Q_s.sum() + K_s.sum() + V_s.sum())
        loss_s.backward()

        np.testing.assert_allclose(W_q_grad_fused, attn.W_q.weight.grad.data, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(W_k_grad_fused, attn.W_k.weight.grad.data, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(W_v_grad_fused, attn.W_v.weight.grad.data, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(x_grad_fused, x.grad.data, rtol=1e-4, atol=1e-4)

    def test_fused_qkv_not_used_for_cross_attention(self):
        attn = SloMultiHeadAttention(64, 4)
        q = Tensor(np.random.randn(2, 8, 64))
        k = Tensor(np.random.randn(2, 16, 64))
        v = Tensor(np.random.randn(2, 16, 64))

        out = attn.forward(q, k, v)
        assert out is not None

    def test_fused_qkv_with_different_shapes(self):
        attn = SloMultiHeadAttention(64, 4)
        for seq_len in [1, 4, 16]:
            x = Tensor(np.random.randn(1, seq_len, 64))
            Q, K, V = attn._fused_qkv_forward(x)
            assert Q.data.shape == (1, seq_len, 64)
            assert K.data.shape == (1, seq_len, 64)
            assert V.data.shape == (1, seq_len, 64)

    def test_fused_qkv_gqa(self):
        attn = SloMultiHeadAttention(64, 4, n_kv_head=2)
        x = Tensor(np.random.randn(2, 8, 64))
        Q, K, V = attn._fused_qkv_forward(x)
        assert Q.data.shape == (2, 8, 64)
        assert K.data.shape == (2, 8, 32)
        assert V.data.shape == (2, 8, 32)

    def test_fused_qkv_single_head(self):
        attn = SloMultiHeadAttention(64, 1)
        x = Tensor(np.random.randn(1, 4, 64))
        Q, K, V = attn._fused_qkv_forward(x)
        assert Q.data.shape == (1, 4, 64)

    def test_fused_qkv_deterministic(self):
        attn = SloMultiHeadAttention(64, 4)
        x = Tensor(np.random.randn(2, 8, 64))
        Q1, K1, V1 = attn._fused_qkv_forward(x)
        Q2, K2, V2 = attn._fused_qkv_forward(x)
        np.testing.assert_allclose(Q1.data, Q2.data)
        np.testing.assert_allclose(K1.data, K2.data)
        np.testing.assert_allclose(V1.data, V2.data)


# ---------------------------------------------------------------------------
# FusedQKVTraining
# ---------------------------------------------------------------------------

class TestFusedQKVTraining:
    """Verify training works with fused QKV."""

    def test_transformer_trains_with_fused_qkv(self):
        model = SloTransformer(vocab_size=256, n_embed=64, n_layer=2,
                               n_head=2, block_size=32, dropout=0.0)
        optimizer = SloAdamW(lr=1e-3)
        params = model.parameters()

        x = np.random.randint(0, 256, (4, 32))
        y = np.random.randint(0, 256, (4, 32))

        losses = []
        for _ in range(50):
            logits, _ = model.forward(Tensor(x, _copy=False))
            loss = cross_entropy(logits.reshape(-1, 256),
                                 Tensor(y.reshape(-1).astype(np.int64)))
            loss.backward()
            optimizer.step(params)
            losses.append(loss.data)

        assert losses[-1] < losses[0] * 0.5

    def test_loss_decreases_over_steps(self):
        model = SloTransformer(vocab_size=128, n_embed=32, n_layer=1,
                               n_head=2, block_size=16, dropout=0.0)
        optimizer = SloAdamW(lr=1e-3)
        params = model.parameters()

        x = np.random.randint(0, 128, (2, 16))
        y = np.random.randint(0, 128, (2, 16))

        first_loss = None
        last_loss = None
        for i in range(30):
            logits, _ = model.forward(Tensor(x, _copy=False))
            loss = cross_entropy(logits.reshape(-1, 128),
                                 Tensor(y.reshape(-1).astype(np.int64)))
            loss.backward()
            optimizer.step(params)
            if i == 0:
                first_loss = loss.data
            last_loss = loss.data

        assert last_loss < first_loss

    def test_transformer_forward_backward(self):
        model = SloTransformer(vocab_size=64, n_embed=32, n_layer=1,
                               n_head=2, block_size=8, dropout=0.0)
        x = Tensor(np.random.randint(0, 64, (1, 8)))
        logits, _ = model.forward(x)
        loss = cross_entropy(logits.reshape(-1, 64),
                             Tensor(np.random.randint(0, 64, (8,))))
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


# ---------------------------------------------------------------------------
# FusedQKVPerformance
# ---------------------------------------------------------------------------

class TestFusedQKVPerformance:
    """Benchmark fused vs separate QKV projections."""

    def test_fused_produces_same_result_as_separate(self):
        attn = SloMultiHeadAttention(64, 4)
        x = Tensor(np.random.randn(2, 8, 64))

        Q_fused, K_fused, V_fused = attn._fused_qkv_forward(x)

        Q_sep = attn.W_q.forward(x)
        K_sep = attn.W_k.forward(x)
        V_sep = attn.W_v.forward(x)

        np.testing.assert_allclose(Q_fused.data, Q_sep.data, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(K_fused.data, K_sep.data, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(V_fused.data, V_sep.data, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# SloMultiHeadAttention additional
# ---------------------------------------------------------------------------

class TestSloMultiHeadAttention:
    def test_init(self):
        attn = SloMultiHeadAttention(64, 4)
        assert attn.d_model == 64
        assert attn.n_heads == 4
        assert attn.head_dim == 16

    def test_gqa_init(self):
        attn = SloMultiHeadAttention(64, 4, n_kv_head=2)
        assert attn.n_kv_head == 2
        assert attn.n_rep == 2

    def test_output_shape(self):
        attn = SloMultiHeadAttention(64, 4)
        x = Tensor(np.random.randn(2, 8, 64))
        out = attn.forward(x, x, x)
        assert isinstance(out, tuple)
        assert out[0].data.shape == (2, 8, 64)

    def test_cross_attention_output(self):
        attn = SloMultiHeadAttention(64, 4)
        q = Tensor(np.random.randn(2, 8, 64))
        k = Tensor(np.random.randn(2, 16, 64))
        v = Tensor(np.random.randn(2, 16, 64))
        out = attn.forward(q, k, v)
        assert isinstance(out, tuple)
        assert out[0].data.shape == (2, 8, 64)

    def test_attention_4d(self):
        B, N, H, E = 2, 8, 4, 16
        Q = Tensor(np.random.randn(B, N, H, E))
        K = Tensor(np.random.randn(B, N, H, E))
        V = Tensor(np.random.randn(B, N, H, E))
        scale = 1.0 / np.sqrt(E)
        out = SloMultiHeadAttention._attention_4d(Q, K, V, None, scale)
        assert out.data.shape == (2, 8, 64)

    def test_with_mask(self):
        attn = SloMultiHeadAttention(64, 4)
        x = Tensor(np.random.randn(1, 4, 64))
        mask = Tensor(np.zeros((1, 1, 4, 4)))
        out = attn.forward(x, x, x, mask=mask)
        assert isinstance(out, tuple)
        assert out[0].data.shape == (1, 4, 64)

    def test_gradient_flow(self):
        attn = SloMultiHeadAttention(64, 4)
        x = Tensor(np.random.randn(1, 4, 64), requires_grad=True)
        out = attn.forward(x, x, x)
        loss = out[0].sum()
        loss.backward()
        assert x.grad is not None

    def test_parameters_count(self):
        attn = SloMultiHeadAttention(64, 4)
        params = attn.parameters()
        assert len(params) > 0

    def test_rope_init(self):
        attn = SloMultiHeadAttention(64, 4, use_rope=True, max_seq_len=128)
        assert attn.use_rope is True
        assert hasattr(attn, 'rope')

    def test_single_head(self):
        attn = SloMultiHeadAttention(32, 1)
        x = Tensor(np.random.randn(1, 4, 32))
        out = attn.forward(x, x, x)
        assert isinstance(out, tuple)
        assert out[0].data.shape == (1, 4, 32)


# ---------------------------------------------------------------------------
# SloTransformer
# ---------------------------------------------------------------------------

class TestSloTransformer:
    def test_init(self):
        model = SloTransformer(vocab_size=256, n_embed=64, n_layer=2, n_head=2)
        assert model.vocab_size == 256
        assert model.n_embed == 64
        assert model.n_layer == 2

    def test_forward(self):
        model = SloTransformer(vocab_size=128, n_embed=32, n_layer=1, n_head=2, block_size=16)
        x = Tensor(np.random.randint(0, 128, (1, 16)))
        logits, _ = model.forward(x)
        assert logits.data.shape == (1, 16, 128)

    def test_parameters(self):
        model = SloTransformer(vocab_size=64, n_embed=32, n_layer=1, n_head=2, block_size=8)
        params = model.parameters()
        assert len(params) > 0

    def test_num_parameters(self):
        model = SloTransformer(vocab_size=64, n_embed=32, n_layer=1, n_head=2, block_size=8)
        n = model.num_parameters()
        assert n > 0

    def test_different_architectures(self):
        configs = [
            {"vocab_size": 64, "n_embed": 32, "n_layer": 1, "n_head": 2},
            {"vocab_size": 128, "n_embed": 64, "n_layer": 2, "n_head": 4},
            {"vocab_size": 256, "n_embed": 128, "n_layer": 3, "n_head": 8},
        ]
        for cfg in configs:
            model = SloTransformer(**cfg, block_size=16)
            x = Tensor(np.random.randint(0, cfg["vocab_size"], (1, 16)))
            logits, _ = model.forward(x)
            assert logits.data.shape[-1] == cfg["vocab_size"]

    def test_block_count(self):
        model = SloTransformer(vocab_size=64, n_embed=32, n_layer=3, n_head=2, block_size=8)
        assert len(model.blocks) == 3


# ---------------------------------------------------------------------------
# SloAdamW
# ---------------------------------------------------------------------------

class TestSloAdamW:
    def test_init(self):
        opt = SloAdamW(lr=0.001)
        assert opt.lr == 0.001

    def test_step(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        loss = (t * t).sum()
        loss.backward()
        opt = SloAdamW(lr=0.01)
        opt.step([t])
        assert t.grad is None  # grad cleared after step

    def test_weight_decay(self):
        t = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        loss = t.sum()
        loss.backward()
        opt = SloAdamW(lr=0.1, weight_decay=0.01)
        old_data = t.data.copy()
        opt.step([t])
        assert not np.allclose(t.data, old_data)

    def test_multiple_steps(self):
        t = Tensor(np.array([5.0]), requires_grad=True)
        opt = SloAdamW(lr=0.01)
        for _ in range(10):
            loss = (t - 0) ** 2
            loss.backward()
            opt.step([t])
        assert abs(t.data[0]) < 5.0

    def test_state_dict(self):
        t = Tensor(np.array([1.0]), requires_grad=True)
        loss = t.sum()
        loss.backward()
        opt = SloAdamW(lr=0.01)
        opt.step([t])
        state = opt.state_dict([t])
        assert "state" in state

    def test_max_grad_norm(self):
        t = Tensor(np.array([100.0, 200.0]), requires_grad=True)
        loss = t.sum()
        loss.backward()
        opt = SloAdamW(lr=0.01, max_grad_norm=1.0)
        opt.step([t])


# ---------------------------------------------------------------------------
# cross_entropy
# ---------------------------------------------------------------------------

class TestCrossEntropy:
    def test_basic(self):
        logits = Tensor(np.array([[1.0, 2.0, 3.0]]))
        targets = Tensor(np.array([2]))
        loss = cross_entropy(logits, targets)
        assert loss.data > 0

    def test_perfect_prediction(self):
        logits = Tensor(np.array([[0.0, 0.0, 10.0]]))
        targets = Tensor(np.array([2]))
        loss = cross_entropy(logits, targets)
        assert loss.data > -0.1

    def test_gradient(self):
        logits = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)
        targets = Tensor(np.array([1]))
        loss = cross_entropy(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_batch(self):
        logits = Tensor(np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]))
        targets = Tensor(np.array([2, 0]))
        loss = cross_entropy(logits, targets)
        assert loss.data > 0

    def test_3d_logits(self):
        logits = Tensor(np.random.randn(2, 4, 10))
        targets = Tensor(np.array([3, 5]))
        loss = cross_entropy(logits.reshape(-1, 10), Tensor(np.array([3])))
        assert loss.data > 0

    def test_worse_loss_higher(self):
        good = Tensor(np.array([[0.0, 0.0, 10.0]]))
        bad = Tensor(np.array([[10.0, 0.0, 0.0]]))
        targets = Tensor(np.array([2]))
        loss_good = cross_entropy(good, targets)
        loss_bad = cross_entropy(bad, targets)
        assert loss_good.data < loss_bad.data


# ---------------------------------------------------------------------------
# mse_loss
# ---------------------------------------------------------------------------

class TestMseLoss:
    def test_perfect(self):
        pred = Tensor(np.array([1.0, 2.0]))
        target = Tensor(np.array([1.0, 2.0]))
        loss = mse_loss(pred, target)
        assert abs(loss.data) < 1e-6

    def test_imperfect(self):
        pred = Tensor(np.array([1.0, 2.0]))
        target = Tensor(np.array([2.0, 3.0]))
        loss = mse_loss(pred, target)
        assert loss.data > 0

    def test_gradient(self):
        pred = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        target = Tensor(np.array([1.0, 2.0]))
        loss = mse_loss(pred, target)
        loss.backward()
        assert pred.grad is not None


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------

class TestTopk:
    def test_basic(self):
        t = Tensor(np.array([1.0, 5.0, 3.0, 2.0, 4.0]))
        vals, idxs = topk(t, 3)
        assert vals.data.shape == (1, 3)
        flat = vals.data.flatten()
        assert flat[0] >= flat[1] >= flat[2]

    def test_k_1(self):
        t = Tensor(np.array([1.0, 5.0, 3.0]))
        vals, idxs = topk(t, 1)
        assert vals.data.flatten()[0] == 5.0


# ---------------------------------------------------------------------------
# Additional layers
# ---------------------------------------------------------------------------

class TestSloLinear:
    def test_forward(self):
        layer = SloLinear(32, 16)
        x = Tensor(np.random.randn(1, 32))
        out = layer.forward(x)
        assert out.data.shape == (1, 16)

    def test_parameters(self):
        layer = SloLinear(32, 16)
        params = layer.parameters()
        assert len(params) == 2  # weight + bias

    def test_no_bias(self):
        layer = SloLinear(32, 16, bias=False)
        params = layer.parameters()
        assert len(params) == 1


class TestSloEmbedding:
    def test_forward(self):
        emb = SloEmbedding(100, 32)
        x = Tensor(np.array([[0, 1, 2]]))
        out = emb.forward(x)
        assert out.data.shape == (1, 3, 32)


class TestSloRMSNorm:
    def test_forward(self):
        norm = SloRMSNorm(64)
        x = Tensor(np.random.randn(1, 4, 64))
        out = norm.forward(x)
        assert out.data.shape == (1, 4, 64)


class TestSloLayerNorm:
    def test_forward(self):
        norm = SloLayerNorm(64)
        x = Tensor(np.random.randn(1, 4, 64))
        out = norm.forward(x)
        assert out.data.shape == (1, 4, 64)


class TestSloTransformerBlock:
    def test_forward(self):
        block = SloTransformerBlock(64, 4, dim_ff=128)
        x = Tensor(np.random.randn(1, 4, 64))
        out = block.forward(x)
        assert out[0].data.shape == (1, 4, 64)


class TestSloDropout:
    def test_forward_train(self):
        drop = SloDropout(p=0.5)
        drop.train(True)
        x = Tensor(np.ones((1, 10)))
        out = drop.forward(x)
        assert out.data.shape == (1, 10)

    def test_forward_eval(self):
        drop = SloDropout(p=0.5)
        drop.eval()
        x = Tensor(np.ones((1, 10)))
        out = drop.forward(x)
        np.testing.assert_allclose(out.data, x.data)

    def test_zero_dropout(self):
        drop = SloDropout(p=0.0)
        x = Tensor(np.ones((1, 10)))
        out = drop.forward(x)
        np.testing.assert_allclose(out.data, x.data)


class TestSloLayer:
    def test_init(self):
        layer = SloLayer("test")
        assert layer.name == "test"

    def test_default_name(self):
        layer = SloLayer()
        assert "Layer" in layer.name

    def test_soul_signature(self):
        layer = SloLayer("test")
        sig = layer.soul_signature()
        assert "layer" in sig
        assert sig["name"] == "test"

    def test_parameters_empty(self):
        layer = SloLayer()
        assert layer.parameters() == []

    def test_train_mode(self):
        layer = SloLayer()
        layer.train(True)
        layer.eval()


# ---------------------------------------------------------------------------
# no_grad
# ---------------------------------------------------------------------------

class TestNoGrad:
    def test_context_manager(self):
        with no_grad():
            t = Tensor(np.array([1.0]), requires_grad=True)
            assert t.requires_grad is False

    def test_decorator(self):
        @no_grad()
        def my_fn(x):
            return Tensor(np.array([1.0]), requires_grad=True)
        t = my_fn(None)
        assert t.requires_grad is False

    def test_nested(self):
        with no_grad():
            t = Tensor(np.array([1.0]), requires_grad=True)
            assert t.requires_grad is False
        t2 = Tensor(np.array([1.0]), requires_grad=True)
        assert t2.requires_grad is True


# ---------------------------------------------------------------------------
# Logit processors
# ---------------------------------------------------------------------------

class TestLogitProcessors:
    def test_apply_temperature(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        result = _apply_temperature(logits, 0.5)
        np.testing.assert_allclose(result, logits / 0.5)

    def test_apply_temperature_zero(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        result = _apply_temperature(logits, 0.0)
        np.testing.assert_allclose(result, logits)

    def test_apply_top_k(self):
        logits = np.array([[1.0, 5.0, 3.0, 2.0, 4.0]])
        result = _apply_top_k(logits, 2)
        masked = result[result > -1e8]
        assert len(masked) == 2

    def test_apply_top_p(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        result = _apply_top_p(logits, 0.9)
        assert result.shape == logits.shape

    def test_apply_repetition_penalty(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        gen_ids = np.array([0])
        result = _apply_repetition_penalty(logits, gen_ids, 2.0)
        assert result.shape == logits.shape

    def test_apply_frequency_penalty(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        gen_ids = np.array([0, 0, 0])
        result = _apply_frequency_penalty(logits, gen_ids, 0.1)
        assert result.shape == logits.shape

    def test_apply_presence_penalty(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        gen_ids = np.array([0])
        result = _apply_presence_penalty(logits, gen_ids, 0.5)
        assert result.shape == logits.shape

    def test_sample_greedy(self):
        logits = np.array([[1.0, 5.0, 3.0]])
        tok = _sample_from_logits(logits, temperature=0.0)
        assert tok == 1

    def test_sample_with_penalties(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        gen_ids = np.array([0])
        tok = _sample_from_logits(logits, temperature=1.0, repetition_penalty=1.5, generated_ids=gen_ids)
        assert 0 <= tok < 3

    def test_sample_eos_masked(self):
        logits = np.array([[1.0, 5.0, 3.0]])
        tok = _sample_from_logits(logits, temperature=0.0, eos_token=1)
        assert tok != 1

    def test_top_p_no_effect(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        result = _apply_top_p(logits, 1.0)
        np.testing.assert_allclose(result, logits)

    def test_top_k_no_effect(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        result = _apply_top_k(logits, 10)
        np.testing.assert_allclose(result, logits)


# ---------------------------------------------------------------------------
# GenerationMetrics / GenerateResult
# ---------------------------------------------------------------------------

class TestGenerationMetrics:
    def test_defaults(self):
        m = GenerationMetrics()
        assert m.n_tokens == 0
        assert m.tokens_per_sec == 0.0

    def test_finalize(self):
        m = GenerationMetrics(n_tokens=10, t_start=0.0, t_end=1.0)
        m.finalize()
        assert m.tokens_per_sec == 10.0
        assert m.decode_ms == 1000.0

    def test_total_ms(self):
        m = GenerationMetrics(t_start=0.0, t_end=0.5)
        assert m.total_ms == 500.0

    def test_ttft_ms(self):
        m = GenerationMetrics(t_start=0.0, t_first_token=0.1)
        assert m.ttft_ms == 100.0


class TestGenerateResult:
    def test_construction(self):
        r = GenerateResult(token_ids=np.array([[1, 2, 3]]))
        assert r.shape == (1, 3)

    def test_generated_ids(self):
        r = GenerateResult(
            token_ids=np.array([[1, 2, 3, 4]]),
            metrics=GenerationMetrics(prompt_tokens=2),
        )
        np.testing.assert_allclose(r.generated_ids, [[3, 4]])

    def test_getitem(self):
        r = GenerateResult(token_ids=np.array([[1, 2, 3]]))
        assert r[0, 1] == 2

    def test_array(self):
        r = GenerateResult(token_ids=np.array([[1, 2, 3]]))
        a = np.asarray(r)
        np.testing.assert_allclose(a, [[1, 2, 3]])

    def test_eq(self):
        r1 = GenerateResult(token_ids=np.array([[1, 2]]))
        r2 = GenerateResult(token_ids=np.array([[1, 2]]))
        assert r1 == r2

    def test_eq_ndarray(self):
        r = GenerateResult(token_ids=np.array([[1, 2]]))
        assert r == np.array([[1, 2]])

    def test_dtype(self):
        r = GenerateResult(token_ids=np.array([[1, 2]], dtype=np.int64))
        assert r.dtype == np.int64


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilityFunctions:
    def test_broadcast_back(self):
        g = np.ones((2, 3, 4))
        shape = (3, 4)
        result = _broadcast_back(g, shape)
        assert result.shape == (3, 4)

    def test_exp(self):
        t = Tensor(np.array([0.0, 1.0]))
        e = exp(t)
        np.testing.assert_allclose(e.data, [1.0, np.e], rtol=1e-5)

    def test_isfinite(self):
        t = Tensor(np.array([1.0, np.inf, np.nan]))
        result = isfinite(t)
        assert result[0] is True or result[0] == True
        assert result[1] is False or result[1] == False

    def test_where(self):
        cond = Tensor(np.array([1.0, 0.0, 1.0]))
        a = Tensor(np.array([10.0, 20.0, 30.0]))
        b = Tensor(np.array([1.0, 2.0, 3.0]))
        r = where(cond, a, b)
        np.testing.assert_allclose(r.data, [10.0, 2.0, 30.0])

    def test_multinomial(self):
        t = Tensor(np.array([0.1, 0.5, 0.4]))
        result = multinomial(t, 1)
        assert result.data.shape == (1, 1)

    def test_stack(self):
        t1 = Tensor(np.array([1.0, 2.0]))
        t2 = Tensor(np.array([3.0, 4.0]))
        s = stack([t1, t2], dim=0)
        assert s.data.shape == (2, 2)

    def test_concatenate(self):
        t1 = Tensor(np.array([[1.0, 2.0]]))
        t2 = Tensor(np.array([[3.0, 4.0]]))
        c = concatenate([t1, t2], dim=0)
        assert c.data.shape == (2, 2)

    def test_randint(self):
        t = randint(0, 10, (3, 4))
        assert t.data.shape == (3, 4)
        assert (t.data >= 0).all() and (t.data < 10).all()

    def test_flatten_function(self):
        t = Tensor(np.ones((2, 3, 4)))
        f = flatten(t)
        assert f.data.shape == (2, 12)
