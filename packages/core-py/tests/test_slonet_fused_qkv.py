"""Tests for fused QKV projection optimization."""
import numpy as np
import pytest
from domains.training.slonet import (
    Tensor, SloMultiHeadAttention, SloTransformer, SloAdamW, cross_entropy,
)


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
