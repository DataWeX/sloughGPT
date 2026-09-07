"""Tests for training.slonet — SloFeedForward and SloTransformerBlock."""

from __future__ import annotations

import math
import pytest
import numpy as np
from domains.training.slonet import (
    Tensor,
    SloFeedForward,
    SloTransformerBlock,
    SloMultiHeadAttention,
    no_grad,
)


# ── SloFeedForward ──────────────────────────────────────────────────────────


class TestSloFeedForward:

    def test_init(self):
        ff = SloFeedForward(d_model=64, dim_ff=256)
        assert ff.act_name == "gelu"

    def test_init_silu(self):
        ff = SloFeedForward(d_model=64, dim_ff=256, activation="silu")
        assert ff.act_name == "silu"

    def test_forward(self):
        ff = SloFeedForward(d_model=64, dim_ff=256)
        x = Tensor(np.ones((1, 10, 64)))
        out = ff(x)
        assert out.shape == (1, 10, 64)

    def test_forward_silu(self):
        ff = SloFeedForward(d_model=64, dim_ff=256, activation="silu")
        x = Tensor(np.ones((1, 10, 64)))
        out = ff(x)
        assert out.shape == (1, 10, 64)

    def test_forward_numpy(self):
        ff = SloFeedForward(d_model=64, dim_ff=256)
        x = np.ones((1, 10, 64))
        out = ff.forward_numpy(x)
        assert out.shape == (1, 10, 64)

    def test_parameters(self):
        ff = SloFeedForward(d_model=64, dim_ff=256)
        params = ff.parameters()
        assert len(params) == 3 * 2  # w1, w2, w3 (each with weight + bias)

    def test_soul_traits(self):
        ff = SloFeedForward(d_model=64, dim_ff=256)
        assert "creativity" in ff.soul_traits
        assert "confidence" in ff.soul_traits


# ── SloTransformerBlock ─────────────────────────────────────────────────────


class TestSloTransformerBlock:

    def test_init(self):
        block = SloTransformerBlock(d_model=64, n_heads=4)
        assert block.d_model == 64
        assert block.n_heads == 4
        assert block.head_dim == 16

    def test_forward(self):
        block = SloTransformerBlock(d_model=64, n_heads=4)
        x = Tensor(np.ones((1, 10, 64)))
        out, kv = block.forward(x)
        assert out.shape == (1, 10, 64)

    def test_forward_numpy(self):
        block = SloTransformerBlock(d_model=64, n_heads=4)
        x = np.ones((1, 10, 64))
        out, kv = block.forward_numpy(x)
        assert out.shape == (1, 10, 64)

    def test_forward_with_mask(self):
        block = SloTransformerBlock(d_model=64, n_heads=4)
        x = Tensor(np.ones((1, 10, 64)))
        mask = Tensor(np.zeros((1, 1, 10, 10)))
        out, kv = block.forward(x, mask=mask)
        assert out.shape == (1, 10, 64)

    def test_forward_with_rope(self):
        block = SloTransformerBlock(d_model=64, n_heads=4, use_rope=True, max_seq_len=100)
        x = Tensor(np.ones((1, 10, 64)))
        out, kv = block.forward(x)
        assert out.shape == (1, 10, 64)

    def test_forward_layer_norm(self):
        block = SloTransformerBlock(d_model=64, n_heads=4, norm_type="layer_norm")
        x = Tensor(np.ones((1, 10, 64)))
        out, kv = block.forward(x)
        assert out.shape == (1, 10, 64)

    def test_train_mode(self):
        block = SloTransformerBlock(d_model=64, n_heads=4, dropout=0.1)
        block.train(True)
        block.train(False)
        block.eval()

    def test_parameters(self):
        block = SloTransformerBlock(d_model=64, n_heads=4)
        params = block.parameters()
        assert len(params) > 0

    def test_soul_traits(self):
        block = SloTransformerBlock(d_model=64, n_heads=4)
        assert "curiosity" in block.soul_traits
        assert "creativity" in block.soul_traits


# ── Integration ─────────────────────────────────────────────────────────────


class TestFeedForwardIntegration:

    def test_feedforward_no_grad(self):
        ff = SloFeedForward(d_model=64, dim_ff=256)
        x = Tensor(np.ones((1, 10, 64)))
        with no_grad():
            out = ff(x)
            assert out.requires_grad is False

    def test_transformer_block_no_grad(self):
        block = SloTransformerBlock(d_model=64, n_heads=4)
        x = Tensor(np.ones((1, 10, 64)))
        with no_grad():
            out, kv = block.forward(x)
            assert out.requires_grad is False
