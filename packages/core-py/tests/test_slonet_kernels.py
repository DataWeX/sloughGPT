"""Tests for training.slonet_kernels — numpy fallback functions."""

from __future__ import annotations

import pytest
import numpy as np
from domains.training.slonet_kernels import (
    nb_rmsnorm,
    nb_layernorm,
    nb_swiglu,
    nb_softmax,
    nb_embed,
    nb_add_pos,
    nb_swi_glu_mul,
    fused_layer_norm,
    lm_head_argmax,
)


# ── nb_rmsnorm ──────────────────────────────────────────────────────────────


class TestNbRmsnorm:

    def test_basic(self):
        x = np.ones((1, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        out = nb_rmsnorm(x, w, eps=np.float32(1e-5))
        assert out.shape == x.shape
        assert np.allclose(out, 1.0, atol=1e-5)

    def test_weighted(self):
        x = np.ones((1, 4), dtype=np.float32)
        w = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = nb_rmsnorm(x, w, eps=np.float32(1e-5))
        assert out.shape == x.shape

    def test_3d(self):
        x = np.ones((2, 3, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        out = nb_rmsnorm(x, w, eps=np.float32(1e-5))
        assert out.shape == x.shape


# ── nb_layernorm ────────────────────────────────────────────────────────────


class TestNbLayernorm:

    def test_basic(self):
        x = np.ones((1, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        out = nb_layernorm(x, w, b, eps=np.float32(1e-5))
        assert out.shape == x.shape
        assert np.allclose(out, 0.0, atol=1e-5)

    def test_no_bias(self):
        x = np.ones((1, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        out = nb_layernorm(x, w, None, eps=np.float32(1e-5))
        assert out.shape == x.shape

    def test_3d(self):
        x = np.ones((2, 3, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        out = nb_layernorm(x, w, b, eps=np.float32(1e-5))
        assert out.shape == x.shape


# ── nb_swiglu ───────────────────────────────────────────────────────────────


class TestNbSwiglu:

    def test_basic(self):
        h1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = nb_swiglu(h1)
        assert out.shape == h1.shape
        assert out[0, 0] > 0

    def test_negative(self):
        h1 = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        out = nb_swiglu(h1)
        assert out.shape == h1.shape


# ── nb_softmax ──────────────────────────────────────────────────────────────


class TestNbSoftmax:

    def test_basic(self):
        e = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = nb_softmax(e)
        assert out.shape == e.shape
        assert out.sum() == pytest.approx(1.0, rel=1e-5)

    def test_large_values(self):
        e = np.array([[100.0, 200.0, 300.0]], dtype=np.float32)
        out = nb_softmax(e)
        assert out.sum() == pytest.approx(1.0, rel=1e-5)

    def test_3d(self):
        e = np.ones((2, 3, 4), dtype=np.float32)
        out = nb_softmax(e)
        assert out.shape == e.shape
        for i in range(2):
            for j in range(3):
                assert out[i, j].sum() == pytest.approx(1.0, rel=1e-5)


# ── nb_embed ────────────────────────────────────────────────────────────────


class TestNbEmbed:

    def test_basic(self):
        emb = np.ones((10, 4), dtype=np.float32)
        ids = np.array([[0, 1, 2]])
        out = np.zeros((1, 3, 4), dtype=np.float32)
        result = nb_embed(emb, ids, out)
        assert result.shape == (1, 3, 4)
        assert np.allclose(result, 1.0)

    def test_clipping(self):
        emb = np.ones((10, 4), dtype=np.float32)
        ids = np.array([[0, 100, -5]])
        out = np.zeros((1, 3, 4), dtype=np.float32)
        result = nb_embed(emb, ids, out)
        assert result.shape == (1, 3, 4)


# ── nb_add_pos ──────────────────────────────────────────────────────────────


class TestNbAddPos:

    def test_basic(self):
        x = np.zeros((1, 4, 8), dtype=np.float32)
        pos_emb = np.ones((100, 8), dtype=np.float32)
        out = nb_add_pos(x, pos_emb, 0, 4)
        assert out.shape == x.shape
        assert np.allclose(out, 1.0)


# ── nb_swi_glu_mul ──────────────────────────────────────────────────────────


class TestNbSwiGluMul:

    def test_basic(self):
        h1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        h3 = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        out = nb_swi_glu_mul(h1, h3)
        assert out.shape == h1.shape
        assert out[0, 0] > 0


# ── fused_layer_norm ────────────────────────────────────────────────────────


class TestFusedLayerNorm:

    def test_basic(self):
        x = np.ones((1, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        out = fused_layer_norm(x, w, b, np.float32(1e-5))
        assert out.shape == x.shape


# ── lm_head_argmax ──────────────────────────────────────────────────────────


class TestLmHeadArgmax:

    def test_basic(self):
        # x is (embed_dim,), W is (vocab_size, embed_dim)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        W = np.array([[3.0, 2.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        # logits = x @ W.T = [1*3+2*2+3*1, 1*1+2*0+3*0, 1*0+2*0+3*1] = [10, 1, 3]
        # argmax = 0
        result = lm_head_argmax(x, W)
        assert result == 0
