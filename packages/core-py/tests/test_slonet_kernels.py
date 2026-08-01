"""Tests for domains/training/slonet_kernels.py — numba-JIT kernels.

numba is optional: every kernel falls back to a numpy implementation that
must be numerically equivalent to the hand-written loop. This suite validates
the fallbacks against independently-computed numpy references and, when numba
is present, cross-checks both implementations agree.
"""

import numpy as np
import pytest

import domains.training.slonet_kernels as sk

np.random.seed(0)


def _rand(*shape, dtype=np.float32):
    return np.random.randn(*shape).astype(dtype)


def _swiglu(h):
    return 0.5 * h * (1.0 + np.tanh(0.7978845608 * (h + 0.044715 * h**3)))


class TestNumbaDetection:
    def test_check_numba_caches_result(self):
        assert sk._check_numba() is (True if sk._NUMBA_AVAILABLE else False)
        assert sk._check_numba() == sk._NUMBA_AVAILABLE

    def test_ensure_kernels_without_numba_leaves_fallbacks(self, monkeypatch):
        monkeypatch.setattr(sk, "_check_numba", lambda: False)
        monkeypatch.setattr(sk, "_kernels_built", False)
        sk._ensure_kernels()
        assert sk._nb_rmsnorm is None


class TestNorms:
    def test_rmsnorm_2d(self):
        x = _rand(4, 8)
        w = _rand(8)
        eps = np.float32(1e-5)
        ref = (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * w
        np.testing.assert_allclose(sk.nb_rmsnorm(x, w, eps), ref, rtol=1e-5, atol=1e-6)

    def test_rmsnorm_3d(self):
        x = _rand(2, 3, 6)
        w = _rand(6)
        ref = (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)) * w
        np.testing.assert_allclose(sk.nb_rmsnorm(x, w), ref, rtol=1e-5, atol=1e-6)

    def test_layernorm_with_bias(self):
        x = _rand(3, 10)
        w = _rand(10)
        b = _rand(10)
        mu = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
        ref = (x - mu) / np.sqrt(var + 1e-5) * w + b
        np.testing.assert_allclose(sk.nb_layernorm(x, w, b), ref, rtol=1e-5, atol=1e-6)

    def test_layernorm_without_bias(self):
        x = _rand(2, 5)
        w = _rand(5)
        mu = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
        ref = (x - mu) / np.sqrt(var + 1e-5) * w
        np.testing.assert_allclose(sk.nb_layernorm(x, w, None), ref, rtol=1e-5, atol=1e-6)

    def test_fused_layer_norm_matches_nb(self):
        x = _rand(7, 12)
        w = _rand(12)
        b = _rand(12)
        np.testing.assert_allclose(sk.fused_layer_norm(x, w, b), sk.nb_layernorm(x, w, b),
                                   rtol=1e-5, atol=1e-6)


class TestActivations:
    def test_swiglu(self):
        h = _rand(3, 16)
        np.testing.assert_allclose(sk.nb_swiglu(h), _swiglu(h), rtol=1e-5, atol=1e-6)

    def test_swi_glu_mul(self):
        h1 = _rand(3, 16)
        h3 = _rand(3, 16)
        np.testing.assert_allclose(sk.nb_swi_glu_mul(h1, h3), _swiglu(h1) * h3,
                                   rtol=1e-5, atol=1e-6)

    def test_softmax_rows_normalized(self):
        e = _rand(2, 4, 5)
        out = sk.nb_softmax(e.copy())
        assert out is not None
        np.testing.assert_allclose(out.sum(axis=-1), np.ones((2, 4)), rtol=1e-5, atol=1e-6)
        assert np.all(out > 0)
        assert np.all(out <= 1)

    def test_softmax_matches_reference(self):
        e = _rand(3, 7)
        src = e.copy()
        ref = np.exp(e - e.max(axis=-1, keepdims=True))
        ref = ref / ref.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(sk.nb_softmax(src), ref, rtol=1e-5, atol=1e-6)

    def test_softmax_modifies_in_place(self):
        e = np.ones((2, 3), dtype=np.float32)
        sk.nb_softmax(e)
        np.testing.assert_allclose(e, np.full((2, 3), 1 / 3, dtype=np.float32), rtol=1e-6)


class TestEmbedding:
    def test_embed_lookup(self):
        emb = _rand(8, 4)
        ids = np.array([1, 3, 0, 5], dtype=np.int64)
        out = np.empty((4, 4), dtype=np.float32)
        sk.nb_embed(emb, ids, out)
        np.testing.assert_allclose(out, emb[ids], rtol=1e-6)

    def test_embed_clips_out_of_range(self):
        emb = _rand(5, 3)
        ids = np.array([-3, 2, 99], dtype=np.int64)
        out = np.empty((3, 3), dtype=np.float32)
        sk.nb_embed(emb, ids, out)
        np.testing.assert_allclose(out, emb[[0, 2, 4]], rtol=1e-6)


class TestAddPos:
    def test_add_pos(self):
        x = _rand(1, 4, 6)
        pe = _rand(10, 6)
        pos, seq_len = 3, 4
        ref = x + pe[pos:pos + seq_len][None, :, :]
        sk.nb_add_pos(x, pe, pos, seq_len)
        np.testing.assert_allclose(x, ref, rtol=1e-6)

    def test_add_pos_clips_past_end(self):
        x = _rand(1, 3, 5)
        pe = _rand(4, 5)
        ref = x + pe[[3, 3, 3]][None, :, :]
        sk.nb_add_pos(x, pe, 3, 3)
        np.testing.assert_allclose(x, ref, rtol=1e-6)


class TestFusedAttention:
    def test_attention_single(self):
        H, E_head, new_len = 2, 6, 5
        q = _rand(H, E_head)
        k = _rand(H, new_len, E_head)
        v = _rand(H, new_len, E_head)
        scale = np.float32(1 / np.sqrt(E_head))
        scores = np.einsum("hd,hnd->hn", q, k) * scale
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        ref = np.einsum("hn,hnd->hd", attn, v)
        out = sk.fused_attention_single(q, k, v, scale, H, E_head)
        assert out.shape == (H, E_head)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_attention_multi_causal(self):
        seq, H, E_head, new_len = 4, 2, 5, 4
        q = _rand(seq, H, E_head)
        k = _rand(H, new_len, E_head)
        v = _rand(H, new_len, E_head)
        scale = np.float32(1 / np.sqrt(E_head))
        scores = np.einsum("she,hne->hsn", q, k) * scale
        scores = scores + np.triu(np.full((seq, new_len), -1e9, dtype=np.float32), k=1)
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        ref = np.einsum("hsn,hne->she", attn, v)
        out = sk.fused_attention_multi(q, k, v, scale, H, E_head)
        assert out.shape == (seq, H, E_head)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_gqa_expand(self):
        K_H, new_len, E, reps = 3, 5, 4, 2
        k = _rand(K_H, new_len, E)
        out = sk.gqa_expand(k, reps)
        assert out.shape == (K_H * reps, new_len, E)
        np.testing.assert_allclose(out, np.repeat(k, reps, axis=0), rtol=1e-6)


class TestLmHead:
    def test_lm_head_argmax(self):
        x = _rand(1, 8)
        W = _rand(20, 8)
        assert sk.lm_head_argmax(x, W) == int(np.argmax((x @ W.T)[0]))

    def test_lm_head_argmax_identifies_known_best(self):
        W = np.zeros((5, 4), dtype=np.float32)
        W[3] = np.array([1.0, 1.0, 1.0, 1.0])
        x = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        assert sk.lm_head_argmax(x, W) == 3

    def test_lm_head_argmax_int8(self):
        x = _rand(1, 8)
        W8 = np.random.default_rng(1).integers(-20, 20, size=(16, 8)).astype(np.int8)
        scale = np.float32(0.5)
        ref = int(np.argmax((x @ (W8.astype(np.float32) * scale).T)[0]))
        assert sk.lm_head_argmax_int8(x, W8, scale) == ref

    def test_lm_head_argmax_int8_identifies_known_best(self):
        W8 = np.zeros((4, 4), dtype=np.int8)
        W8[1] = np.array([10, 10, 10, 10], dtype=np.int8)
        x = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        assert sk.lm_head_argmax_int8(x, W8, np.float32(1.0)) == 1
