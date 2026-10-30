"""Tests for training.slonet_kernels — numpy fallback functions."""

from __future__ import annotations

import numpy as np
import pytest

from domain.training._internal.slonet_kernels import (
    fused_attention_multi,
    fused_attention_single,
    fused_layer_norm,
    lm_head_argmax,
    nb_add_pos,
    nb_embed,
    nb_layernorm,
    nb_rmsnorm,
    nb_softmax,
    nb_swi_glu_mul,
    nb_swiglu,
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

    def test_matches_reference(self):
        """Public path (C when available, numba otherwise) matches numpy."""
        rng = np.random.default_rng(7)
        h1 = rng.standard_normal((1, 4864), dtype=np.float32) * 3
        h3 = rng.standard_normal((1, 4864), dtype=np.float32)
        out = nb_swi_glu_mul(h1, h3)
        ref = (h1 * (1.0 / (1.0 + np.exp(-h1.astype(np.float64))))) * h3
        np.testing.assert_allclose(out, ref, atol=1e-5)

    def test_edge_saturation_and_nan(self):
        """Huge |v| saturates without Inf/NaN; NaN propagates through."""
        h1 = np.array([-100.0, -20.0, 0.0, 20.0, 100.0, np.nan], dtype=np.float32)
        h3 = np.ones(6, dtype=np.float32)
        out = nb_swi_glu_mul(h1, h3)
        ref = (h1 * (1.0 / (1.0 + np.exp(-h1.astype(np.float64))))) * h3
        np.testing.assert_allclose(out, ref, atol=1e-6)
        assert np.isnan(out[-1])

    def test_c_kernel_guards(self):
        """swiglu_f32_c runs on contiguous f32 and rejects strided/f64/mismatch."""
        from domain.infrastructure._internal.quant_core.wrapper import swiglu_f32_c

        ok = np.zeros(8, dtype=np.float32)
        res = swiglu_f32_c(ok, ok)
        if res is None:
            pytest.skip("C swiglu kernel unavailable")
        np.testing.assert_allclose(res, 0.0)  # silu(0)*0 = 0
        strided = np.zeros((2, 8), dtype=np.float32)[::-1, :]
        assert swiglu_f32_c(strided, strided) is None
        assert swiglu_f32_c(ok.astype(np.float64), ok.astype(np.float64)) is None
        assert swiglu_f32_c(ok, ok[:4]) is None

    def test_c_matches_numba(self):
        """C kernel agrees with the numba kernel to float32 tolerance."""
        import domain.training._internal.slonet_kernels as sk
        from domain.infrastructure._internal.quant_core.wrapper import swiglu_f32_c

        sk._ensure_kernels()
        if sk._nb_swi_glu_mul is None:
            pytest.skip("numba unavailable")
        rng = np.random.default_rng(11)
        h1 = (rng.standard_normal((1, 4864)) * 4).astype(np.float32)
        h3 = rng.standard_normal((1, 4864)).astype(np.float32)
        out_c = swiglu_f32_c(h1, h3)
        if out_c is None:
            pytest.skip("C swiglu kernel unavailable")
        out_nb = sk._nb_swi_glu_mul(h1, h3)
        np.testing.assert_allclose(out_c, out_nb, atol=1e-5)


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


# ── fused_attention_single / fused_attention_multi ──────────────────────────


def _ref_single(q, k, v, scale):
    """NumPy reference: (H, E) q vs (H, L, E) k/v → (H, E)."""
    scores = np.einsum("hd,hnd->hn", q, k) * scale
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)
    return np.einsum("hn,hnd->hd", attn, v)


def _ref_multi(q, k, v, scale):
    """NumPy reference with causal mask: (S, H, E) q → (S, H, E)."""
    scores = np.einsum("she,hne->hsn", q, k) * scale
    causal = np.triu(np.full(scores.shape, -1e9, dtype=np.float64), k=1)
    scores = scores + causal
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)
    return np.einsum("hsn,hne->she", attn, v)


class TestFusedAttentionSingle:
    def test_matches_reference(self):
        rng = np.random.default_rng(7)
        H, E, L = 12, 64, 37
        q = rng.standard_normal((H, E)).astype(np.float32)
        k = rng.standard_normal((H, L, E)).astype(np.float32)
        v = rng.standard_normal((H, L, E)).astype(np.float32)
        scale = np.float32(E**-0.5)
        out = fused_attention_single(q, k, v, scale, H, E)
        assert out.shape == (H, E)
        ref = _ref_single(
            q.astype(np.float64), k.astype(np.float64), v.astype(np.float64), float(scale)
        )
        np.testing.assert_allclose(out, ref, atol=1e-5)

    def test_single_token_context(self):
        rng = np.random.default_rng(11)
        H, E = 4, 16
        q = rng.standard_normal((H, E)).astype(np.float32)
        k = rng.standard_normal((H, 1, E)).astype(np.float32)
        v = rng.standard_normal((H, 1, E)).astype(np.float32)
        out = fused_attention_single(q, k, v, np.float32(0.25), H, E)
        # single context position → softmax weight is 1 → out == v
        np.testing.assert_allclose(out, v[:, 0, :], atol=1e-6)

    def test_large_context_scales_linearly(self):
        """Regression guard: cost must stay O(L), not O(L²) — a 4× L only
        ~4×'s the runtime (allowing generous slack for noise)."""
        import time

        rng = np.random.default_rng(3)
        H, E = 8, 64
        q = rng.standard_normal((H, E)).astype(np.float32)
        k1 = rng.standard_normal((H, 64, E)).astype(np.float32)
        k2 = rng.standard_normal((H, 256, E)).astype(np.float32)
        scale = np.float32(E**-0.5)
        for _ in range(5):
            fused_attention_single(q, k1, k1, scale, H, E)
            fused_attention_single(q, k2, k2, scale, H, E)
        n = 300
        t0 = time.perf_counter()
        for _ in range(n):
            fused_attention_single(q, k1, k1, scale, H, E)
        t64 = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(n):
            fused_attention_single(q, k2, k2, scale, H, E)
        t256 = time.perf_counter() - t0
        # O(L) predicts ~4×; O(L²) predicts ~16×. Fail below 8× to leave
        # headroom for scheduler noise while still catching quadratic blowup.
        assert t256 / t64 < 8.0, f"superlinear scaling: {t256 / t64:.1f}x for 4x L"


class TestFusedAttentionMulti:
    def test_matches_reference(self):
        rng = np.random.default_rng(7)
        S, H, E, L = 9, 6, 32, 20
        q = rng.standard_normal((S, H, E)).astype(np.float32)
        k = rng.standard_normal((H, L, E)).astype(np.float32)
        v = rng.standard_normal((H, L, E)).astype(np.float32)
        out = fused_attention_multi(q, k, v, np.float32(0.125), H, E)
        assert out.shape == (S, H, E)
        ref = _ref_multi(q.astype(np.float64), k.astype(np.float64), v.astype(np.float64), 0.125)
        np.testing.assert_allclose(out, ref, atol=1e-5)

    def test_causal_mask(self):
        """Position s must not see keys n > s: changing a future key must not
        change output at s."""
        rng = np.random.default_rng(5)
        S, H, E, L = 8, 4, 16, 16
        q = rng.standard_normal((S, H, E)).astype(np.float32)
        k = rng.standard_normal((H, L, E)).astype(np.float32)
        v = rng.standard_normal((H, L, E)).astype(np.float32)
        out1 = fused_attention_multi(q, k, v, np.float32(0.25), H, E)
        k2 = k.copy()
        k2[:, S + 3 :, :] += 100.0  # perturb only keys strictly after all queries
        out2 = fused_attention_multi(q, k2, v, np.float32(0.25), H, E)
        np.testing.assert_allclose(out1[: S - 1], out2[: S - 1], atol=1e-5)

    def test_equals_single_on_last_token(self):
        """The final prompt token's multi-token output must match single-token
        attention with the same KV — decode/prefill consistency."""
        rng = np.random.default_rng(13)
        H, E, L = 6, 32, 24
        q = rng.standard_normal((L, H, E)).astype(np.float32)
        k = rng.standard_normal((H, L, E)).astype(np.float32)
        v = rng.standard_normal((H, L, E)).astype(np.float32)
        scale = np.float32(E**-0.5)
        out_multi = fused_attention_multi(q, k, v, scale, H, E)[-1]
        out_single = fused_attention_single(q[-1], k, v, scale, H, E)
        np.testing.assert_allclose(out_multi, out_single, atol=1e-6)


# ── GQA-unexpanded attention (head mapping inside the kernel) ──────────────


def _expand(k, reps):
    """Independent reference expansion: kv head kh → heads [kh*reps, +reps)."""
    return np.repeat(k, reps, axis=0)


class TestGqaUnexpandedAttention:
    """The fused kernels take the KV cache unexpanded as (K_H, L, E) and map
    query head h → kv head h // (H // K_H). Outputs must be identical to the
    old expanded (H, L, E) input — the expansion is pure redundancy."""

    def test_single_matches_expanded(self):
        rng = np.random.default_rng(21)
        H, K_H, L, E = 14, 2, 41, 64
        reps = H // K_H
        q = rng.standard_normal((H, E)).astype(np.float32)
        k = rng.standard_normal((K_H, L, E)).astype(np.float32)
        v = rng.standard_normal((K_H, L, E)).astype(np.float32)
        scale = np.float32(E**-0.5)
        out_unexp = fused_attention_single(q, k, v, scale, H, E)
        out_exp = fused_attention_single(q, _expand(k, reps), _expand(v, reps), scale, H, E)
        np.testing.assert_allclose(out_unexp, out_exp, atol=1e-6)
        # and against the float64 einsum reference
        ref = _ref_single(
            q.astype(np.float64),
            _expand(k, reps).astype(np.float64),
            _expand(v, reps).astype(np.float64),
            float(scale),
        )
        np.testing.assert_allclose(out_unexp, ref, atol=1e-5)

    def test_multi_matches_expanded(self):
        rng = np.random.default_rng(23)
        S, H, K_H, L, E = 7, 8, 2, 33, 32
        reps = H // K_H
        q = rng.standard_normal((S, H, E)).astype(np.float32)
        k = rng.standard_normal((K_H, L, E)).astype(np.float32)
        v = rng.standard_normal((K_H, L, E)).astype(np.float32)
        scale = np.float32(E**-0.5)
        out_unexp = fused_attention_multi(q, k, v, scale, H, E)
        out_exp = fused_attention_multi(q, _expand(k, reps), _expand(v, reps), scale, H, E)
        np.testing.assert_allclose(out_unexp, out_exp, atol=1e-6)

    def test_prefill_decode_consistency_unexpanded(self):
        """Last multi-token output == single-token decode with the same
        unexpanded KV (GQA path must stay decode/prefill consistent)."""
        rng = np.random.default_rng(27)
        H, K_H, L, E = 14, 2, 24, 32
        scale = np.float32(E**-0.5)
        q = rng.standard_normal((L, H, E)).astype(np.float32)
        k = rng.standard_normal((K_H, L, E)).astype(np.float32)
        v = rng.standard_normal((K_H, L, E)).astype(np.float32)
        out_multi = fused_attention_multi(q, k, v, scale, H, E)[-1]
        out_single = fused_attention_single(q[-1], k, v, scale, H, E)
        np.testing.assert_allclose(out_multi, out_single, atol=1e-6)

    def test_numpy_fallback_matches(self):
        """Without numba the wrapper falls back to einsum — it must expand
        the unexpanded input itself and agree with the reference."""
        import domain.training._internal.slonet_kernels as sk

        rng = np.random.default_rng(31)
        H, K_H, L, E = 6, 2, 19, 16
        reps = H // K_H
        q = rng.standard_normal((H, E)).astype(np.float32)
        k = rng.standard_normal((K_H, L, E)).astype(np.float32)
        v = rng.standard_normal((K_H, L, E)).astype(np.float32)
        scale = np.float32(E**-0.5)
        saved_s = sk._nb_fused_attention_single
        saved_m = sk._nb_fused_attention_multi
        # Build kernels BEFORE patching — _ensure_fused() inside the wrapper
        # would otherwise overwrite the None patch and skip the fallback.
        fused_attention_single(q, _expand(k, reps), _expand(v, reps), scale, H, E)
        sk._nb_fused_attention_single = None
        sk._nb_fused_attention_multi = None
        try:
            out_s = fused_attention_single(q, k, v, scale, H, E)
            qq = rng.standard_normal((5, H, E)).astype(np.float32)
            out_m = fused_attention_multi(qq, k, v, scale, H, E)
        finally:
            sk._nb_fused_attention_single = saved_s
            sk._nb_fused_attention_multi = saved_m
        ref_s = _ref_single(
            q.astype(np.float64),
            _expand(k, reps).astype(np.float64),
            _expand(v, reps).astype(np.float64),
            float(scale),
        )
        ref_m = _ref_multi(
            qq.astype(np.float64),
            _expand(k, reps).astype(np.float64),
            _expand(v, reps).astype(np.float64),
            float(scale),
        )
        np.testing.assert_allclose(out_s, ref_s, atol=1e-5)
        np.testing.assert_allclose(out_m, ref_m, atol=1e-5)
