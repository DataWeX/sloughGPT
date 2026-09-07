"""Tests for training.slonet — SloCrossAttention and NumpyKVState."""

from __future__ import annotations

import math
import pytest
import numpy as np
from domains.training.slonet import (
    Tensor,
    SloCrossAttention,
    NumpyKVState,
    no_grad,
)


# ── SloCrossAttention ───────────────────────────────────────────────────────


class TestSloCrossAttention:

    def test_init(self):
        attn = SloCrossAttention(d_model=64, n_heads=4)
        assert attn.d_model == 64
        assert attn.n_heads == 4
        assert attn.head_dim == 16

    def test_forward(self):
        attn = SloCrossAttention(d_model=64, n_heads=4)
        x = Tensor(np.ones((1, 10, 64)))
        context = Tensor(np.ones((1, 20, 64)))
        out = attn.forward(x, context)
        assert out.shape == (1, 10, 64)

    def test_forward_with_mask(self):
        attn = SloCrossAttention(d_model=64, n_heads=4)
        x = Tensor(np.ones((1, 10, 64)))
        context = Tensor(np.ones((1, 20, 64)))
        mask = Tensor(np.zeros((1, 1, 10, 20)))
        out = attn.forward(x, context, mask=mask)
        assert out.shape == (1, 10, 64)

    def test_parameters(self):
        attn = SloCrossAttention(d_model=64, n_heads=4)
        params = attn.parameters()
        assert len(params) == 4 * 2  # q, k, v, o (each with weight + bias)

    def test_soul_traits(self):
        attn = SloCrossAttention(d_model=64, n_heads=4)
        assert "curiosity" in attn.soul_traits
        assert "creativity" in attn.soul_traits

    def test_no_grad(self):
        attn = SloCrossAttention(d_model=64, n_heads=4)
        x = Tensor(np.ones((1, 10, 64)))
        context = Tensor(np.ones((1, 20, 64)))
        with no_grad():
            out = attn.forward(x, context)
            assert out.requires_grad is False

    def test_cross_attention_asymmetry(self):
        attn = SloCrossAttention(d_model=64, n_heads=4)
        q = Tensor(np.ones((1, 5, 64)))
        kv = Tensor(np.ones((1, 15, 64)))
        out = attn.forward(q, kv)
        assert out.shape == (1, 5, 64)


# ── NumpyKVState ────────────────────────────────────────────────────────────


class TestNumpyKVState:

    def test_init(self):
        state = NumpyKVState()
        assert state.kv_buf_k == []
        assert state.kv_buf_v == []
        assert state.prev_ids is None
        assert state.quantize_kv is False
        assert state.capacity == 0

    def test_reset(self):
        state = NumpyKVState()
        state.kv_buf_k = [np.ones((1, 10, 4, 8))]
        state.kv_buf_v = [np.ones((1, 10, 4, 8))]
        state.kv_len = [10]
        state.prev_ids = np.array([[1, 2, 3]])
        state.capacity = 10
        state.reset()
        assert state.kv_buf_k == []
        assert state.kv_buf_v == []
        assert state.kv_len == []
        assert state.prev_ids is None
        assert state.capacity == 0

    def test_repr(self):
        state = NumpyKVState()
        r = repr(state)
        assert "NumpyKVState" in r
        assert "capacity=0" in r

    def test_repr_with_data(self):
        state = NumpyKVState()
        state.kv_len = [10]
        state.capacity = 100
        state.quantize_kv = True
        state.prev_ids = np.array([[1, 2, 3]])
        r = repr(state)
        assert "filled=10" in r
        assert "quantize_kv=True" in r
        assert "valid=True" in r

    def test_repr_empty(self):
        state = NumpyKVState()
        r = repr(state)
        assert "valid=False" in r


# ── Integration ─────────────────────────────────────────────────────────────


class TestCrossAttentionIntegration:

    def test_different_seq_lengths(self):
        attn = SloCrossAttention(d_model=64, n_heads=4)
        q = Tensor(np.ones((1, 3, 64)))
        kv = Tensor(np.ones((1, 7, 64)))
        out = attn.forward(q, kv)
        assert out.shape == (1, 3, 64)

    def test_batch_size(self):
        attn = SloCrossAttention(d_model=64, n_heads=4)
        q = Tensor(np.ones((4, 5, 64)))
        kv = Tensor(np.ones((4, 10, 64)))
        out = attn.forward(q, kv)
        assert out.shape == (4, 5, 64)
