"""Tests for training.slonet — attention classes (SloMultiHeadAttention, SloRotaryEmbedding)."""

from __future__ import annotations

import math
import pytest
import numpy as np
from domains.training.slonet import (
    Tensor,
    SloMultiHeadAttention,
    SloRotaryEmbedding,
    no_grad,
)


# ── SloRotaryEmbedding ──────────────────────────────────────────────────────


class TestSloRotaryEmbedding:

    def test_init(self):
        rope = SloRotaryEmbedding(dim=64, max_seq_len=2048)
        assert rope.dim == 64
        assert rope.max_seq_len == 2048
        assert rope.inv_freq.shape == (32,)

    def test_forward(self):
        rope = SloRotaryEmbedding(dim=64, max_seq_len=2048)
        cos, sin = rope.forward(seq_len=10)
        assert cos.shape == (10, 64)
        assert sin.shape == (10, 64)

    def test_caching(self):
        rope = SloRotaryEmbedding(dim=64, max_seq_len=2048)
        cos1, sin1 = rope.forward(seq_len=10)
        cos2, sin2 = rope.forward(seq_len=10)
        # After caching, values should be the same
        assert np.allclose(cos1, cos2)
        assert np.allclose(sin1, sin2)

    def test_different_lengths(self):
        rope = SloRotaryEmbedding(dim=64, max_seq_len=2048)
        cos1, sin1 = rope.forward(seq_len=5)
        cos2, sin2 = rope.forward(seq_len=10)
        assert cos1.shape[0] == 5
        assert cos2.shape[0] == 10


# ── SloMultiHeadAttention ───────────────────────────────────────────────────


class TestSloMultiHeadAttention:

    def test_init(self):
        attn = SloMultiHeadAttention(d_model=64, n_heads=4)
        assert attn.d_model == 64
        assert attn.n_heads == 4
        assert attn.head_dim == 16
        assert attn.n_kv_head == 4

    def test_init_gqa(self):
        attn = SloMultiHeadAttention(d_model=64, n_heads=4, n_kv_head=2)
        assert attn.n_kv_head == 2
        assert attn.n_rep == 2

    def test_forward(self):
        attn = SloMultiHeadAttention(d_model=64, n_heads=4)
        x = Tensor(np.ones((1, 10, 64)))
        out, kv = attn.forward(x, x, x)
        assert out.shape == (1, 10, 64)

    def test_forward_numpy(self):
        attn = SloMultiHeadAttention(d_model=64, n_heads=4)
        x = np.ones((1, 10, 64))
        out, kv = attn.forward_numpy(x, x, x)
        assert out.shape == (1, 10, 64)

    def test_forward_with_mask(self):
        attn = SloMultiHeadAttention(d_model=64, n_heads=4)
        x = Tensor(np.ones((1, 10, 64)))
        mask = Tensor(np.zeros((1, 1, 10, 10)))
        out, kv = attn.forward(x, x, x, mask=mask)
        assert out.shape == (1, 10, 64)

    def test_forward_with_rope(self):
        attn = SloMultiHeadAttention(d_model=64, n_heads=4, use_rope=True, max_seq_len=100)
        x = Tensor(np.ones((1, 10, 64)))
        out, kv = attn.forward(x, x, x)
        assert out.shape == (1, 10, 64)

    def test_forward_numpy_rope(self):
        attn = SloMultiHeadAttention(d_model=64, n_heads=4, use_rope=True, max_seq_len=100)
        x = np.ones((1, 10, 64))
        out, kv = attn.forward_numpy(x, x, x)
        assert out.shape == (1, 10, 64)

    def test_forward_gqa(self):
        attn = SloMultiHeadAttention(d_model=64, n_heads=4, n_kv_head=2)
        x = Tensor(np.ones((1, 10, 64)))
        out, kv = attn.forward(x, x, x)
        assert out.shape == (1, 10, 64)

    def test_parameters(self):
        attn = SloMultiHeadAttention(d_model=64, n_heads=4)
        params = attn.parameters()
        # q, k, v, o (each with weight + bias)
        assert len(params) == 8

    def test_kv_cache(self):
        attn = SloMultiHeadAttention(d_model=64, n_heads=4)
        x = np.ones((1, 5, 64))
        out1, kv_cache = attn.forward_numpy(x, x, x)
        assert kv_cache[0].shape[1] == 5
        # Second call with new token
        x2 = np.ones((1, 1, 64))
        out2, kv_cache2 = attn.forward_numpy(x2, x2, x2, kv_cache=kv_cache, start_pos=5)
        assert kv_cache2[0].shape[1] == 6

    def test_soul_traits(self):
        attn = SloMultiHeadAttention(d_model=64, n_heads=4)
        assert "curiosity" in attn.soul_traits


# ── Integration ─────────────────────────────────────────────────────────────


class TestAttentionIntegration:

    def test_attention_no_grad(self):
        attn = SloMultiHeadAttention(d_model=64, n_heads=4)
        x = Tensor(np.ones((1, 10, 64)))
        with no_grad():
            out, kv = attn.forward(x, x, x)
            assert out.requires_grad is False

    def test_cross_attention(self):
        attn = SloMultiHeadAttention(d_model=64, n_heads=4)
        q = Tensor(np.ones((1, 10, 64)))
        kv = Tensor(np.ones((1, 20, 64)))
        out, kv_cache = attn.forward(q, kv, kv)
        assert out.shape == (1, 10, 64)
