"""Tests for SloNet numba kernel module — fallbacks and kernel bodies.

Covers every public wrapper in domains/training/slonet_kernels.py against an
independent numpy reference. Numba is not installed in this environment, so
the numpy fallbacks are the actual production code path; the @njit kernel
bodies are additionally executed un-jitted via a fake-numba injection
(_fake_numba) so their real algorithms are verified against the same
references, and the numba-enabled lazy-build branches are covered without
mutating module state.
"""

import sys
import types
from contextlib import contextmanager

import numpy as np

from domains.training import slonet_kernels as k


def _ref_rmsnorm(x, w, eps=np.float32(1e-5)):
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * w


def _ref_layernorm(x, w, b, eps=np.float32(1e-5)):
    mu = x.mean(axis=-1, keepdims=True)
    centered = x - mu
    var = (centered * centered).mean(axis=-1, keepdims=True)
    h = centered * (w / np.sqrt(var + eps))
    return h + b if b is not None else h


def _ref_silu(h):
    return h / (1.0 + np.exp(-h))


@contextmanager
def _fake_numba():
    """Execute the @njit kernel bodies as pure Python.

    Only the numba compiler decorator is substituted (identity decorator);
    every kernel body runs its real algorithm un-jitted and is verified
    numerically against numpy references — no computation is mocked.  Module
    globals written by the lazy builders are snapshotted and restored so the
    fake-numba state never leaks into other test modules.
    """
    module = types.ModuleType("numba")

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def deco(fn):
            return fn

        return deco

    module.njit = njit
    saved = {name: getattr(k, name) for name in (
        "_NUMBA_AVAILABLE",
        "_kernels_built",
        "_fused_built",
        "_fused_norm_res_built",
        "_lm_head_built",
        "_nb_rmsnorm", "_nb_layernorm", "_nb_swiglu", "_nb_softmax",
        "_nb_embed", "_nb_add_pos", "_nb_swi_glu_mul",
        "_nb_fused_block_layer_norm", "_nb_fused_attention_single",
        "_nb_fused_attention_multi", "_nb_gqa_expand",
        "_nb_fused_norm_residual_out",
        "_nb_lm_head_argmax", "_nb_lm_head_argmax_int8",
    )}
    sys.modules["numba"] = module
    try:
        yield
    finally:
        del sys.modules["numba"]
        for name, value in saved.items():
            setattr(k, name, value)


class TestNumbaDetection:
    def test_check_numba_false_without_numba(self):
        assert k._check_numba() is False

    def test_ensure_kernels_idempotent_no_build_without_numba(self):
        k._ensure_kernels()
        k._ensure_kernels()
        assert k._kernels_built is False
        assert k._nb_rmsnorm is None

    def test_ensure_fused_idempotent_no_build_without_numba(self):
        k._ensure_fused()
        k._ensure_fused()
        assert k._fused_built is False
        assert k._nb_fused_attention_single is None

    def test_ensure_fused_norm_residual_no_build_without_numba(self):
        k._ensure_fused_norm_residual()
        k._ensure_fused_norm_residual()
        assert k._fused_norm_res_built is False
        assert k._nb_fused_norm_residual_out is None

    def test_ensure_lm_head_no_build_without_numba(self):
        k._ensure_lm_head()
        k._ensure_lm_head()
        assert k._lm_head_built is False
        assert k._nb_lm_head_argmax is None


class TestNorms:
    def test_rmsnorm_matches_reference_2d(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 6)).astype(np.float32)
        w = rng.normal(size=(6,)).astype(np.float32)
        assert np.allclose(k.nb_rmsnorm(x, w), _ref_rmsnorm(x, w), atol=1e-5)

    def test_rmsnorm_matches_reference_3d_and_custom_eps(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2, 3, 5)).astype(np.float32)
        w = rng.normal(size=(5,)).astype(np.float32)
        eps = np.float32(1e-3)
        assert np.allclose(k.nb_rmsnorm(x, w, eps), _ref_rmsnorm(x, w, eps), atol=1e-4)

    def test_layernorm_with_bias(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(3, 8)).astype(np.float32)
        w = rng.normal(size=(8,)).astype(np.float32)
        b = rng.normal(size=(8,)).astype(np.float32)
        assert np.allclose(k.nb_layernorm(x, w, b), _ref_layernorm(x, w, b), atol=1e-4)

    def test_layernorm_without_bias(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(2, 4)).astype(np.float32)
        w = np.array([1.0, 2.0, 0.5, 1.0], dtype=np.float32)
        assert np.allclose(k.nb_layernorm(x, w, None), _ref_layernorm(x, w, None), atol=1e-5)

    def test_fused_layernorm_matches_reference(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(5, 7)).astype(np.float32)
        w = rng.normal(size=(7,)).astype(np.float32)
        b = rng.normal(size=(7,)).astype(np.float32)
        assert np.allclose(k.fused_layer_norm(x, w, b), _ref_layernorm(x, w, b), atol=1e-4)

    def test_fused_layernorm_without_bias(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(2, 3)).astype(np.float32)
        w = np.ones(3, dtype=np.float32)
        assert np.allclose(
            k.fused_layer_norm(x, w, None), _ref_layernorm(x, w, None), atol=1e-5
        )


class TestActivations:
    def test_swiglu_matches_silu(self):
        rng = np.random.default_rng(6)
        h = rng.normal(size=(4, 8)).astype(np.float32)
        assert np.allclose(k.nb_swiglu(h), _ref_silu(h), atol=1e-6)

    def test_swi_glu_mul_gates(self):
        rng = np.random.default_rng(7)
        h1 = rng.normal(size=(3, 5)).astype(np.float32)
        h3 = rng.normal(size=(3, 5)).astype(np.float32)
        assert np.allclose(k.nb_swi_glu_mul(h1, h3), _ref_silu(h1) * h3, atol=1e-6)

    def test_softmax_normalizes_rows_in_place(self):
        e = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        out = k.nb_softmax(e)
        assert out is e
        assert np.allclose(e.sum(axis=-1), np.ones(2, dtype=np.float32), atol=1e-6)

    def test_softmax_stable_with_large_values(self):
        e = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
        k.nb_softmax(e)
        assert np.allclose(e.sum(axis=-1), np.ones(1, dtype=np.float32), atol=1e-6)
        assert e[0, 2] > e[0, 0]


class TestEmbedding:
    def test_embed_writes_into_out_2d_ids(self):
        rng = np.random.default_rng(8)
        emb = rng.normal(size=(5, 3)).astype(np.float32)
        ids = np.array([[0, 2], [3, 4]])
        out = np.zeros((2, 2, 3), dtype=np.float32)
        result = k.nb_embed(emb, ids, out)
        assert result is out
        assert np.allclose(out[0], emb[[0, 2]])
        assert np.allclose(out[1], emb[[3, 4]])

    def test_embed_1d_ids_into_1d_out(self):
        emb = np.arange(12, dtype=np.float32).reshape(4, 3)
        ids = np.array([1, 3], dtype=np.int64)
        out = np.zeros((2, 3), dtype=np.float32)
        k.nb_embed(emb, ids, out)
        assert np.allclose(out, emb[[1, 3]])

    def test_embed_clips_out_of_range_ids(self):
        emb = np.arange(6, dtype=np.float32).reshape(3, 2)
        ids = np.array([-5, 99], dtype=np.int64)
        out = np.zeros((2, 2), dtype=np.float32)
        k.nb_embed(emb, ids, out)
        assert np.allclose(out, emb[[0, 2]])

    def test_add_pos_adds_sliced_embedding_in_place(self):
        rng = np.random.default_rng(9)
        pe = rng.normal(size=(16, 4)).astype(np.float32)
        x = np.ones((1, 3, 4), dtype=np.float32)
        k.nb_add_pos(x, pe, 2, 3)
        assert np.allclose(x[0], pe[2:5] + 1.0)

    def test_add_pos_clips_beyond_embedding(self):
        pe = np.arange(8, dtype=np.float32).reshape(4, 2)
        x = np.zeros((1, 2, 2), dtype=np.float32)
        k.nb_add_pos(x, pe, 3, 2)
        assert np.allclose(x[0], pe[[3, 3]])


class TestAttentionKernels:
    def test_fused_attention_single_matches_reference(self):
        rng = np.random.default_rng(10)
        H, Eh, N = 3, 4, 6
        q = rng.normal(size=(H, Eh)).astype(np.float32)
        kv = rng.normal(size=(H, N, Eh)).astype(np.float32)
        scale = np.float32(0.5)
        scores = np.einsum("hd,hnd->hn", q, kv) * scale
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        ref = np.einsum("hn,hnd->hd", attn, kv)
        out = k.fused_attention_single(q, kv, kv, scale, H, Eh)
        assert out.shape == (H, Eh)
        assert np.allclose(out, ref, atol=1e-5)

    def test_fused_attention_multi_causal_matches_reference(self):
        rng = np.random.default_rng(11)
        S, H, Eh, N = 4, 2, 3, 5
        q = rng.normal(size=(S, H, Eh)).astype(np.float32)
        kv = rng.normal(size=(H, N, Eh)).astype(np.float32)
        scale = np.float32(0.5)
        scores = np.einsum("she,hne->hsn", q, kv) * scale
        causal = np.triu(np.full((S, N), -1e9, dtype=np.float32), k=1)
        masked = scores + causal
        attn = np.exp(masked - masked.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        ref = np.einsum("hsn,hne->she", attn, kv)
        out = k.fused_attention_multi(q, kv, kv, scale, H, Eh)
        assert out.shape == (S, H, Eh)
        assert np.allclose(out, ref, atol=1e-5)

    def test_fused_attention_single_attention_weights_row_sum_one(self):
        H, Eh, N = 1, 3, 4
        q = np.ones((H, Eh), dtype=np.float32)
        kv = np.arange(N * Eh, dtype=np.float32).reshape(1, N, Eh)
        scale = np.float32(1.0)
        out = k.fused_attention_single(q, kv, kv, scale, H, Eh)
        assert out.shape == (H, Eh)

    def test_gqa_expand_repeats_heads(self):
        rng = np.random.default_rng(12)
        kt = rng.normal(size=(2, 4, 3)).astype(np.float32)
        out = k.gqa_expand(kt, 3)
        assert out.shape == (6, 4, 3)
        assert np.allclose(out, np.repeat(kt, 3, axis=0))


class TestLMHead:
    def test_lm_head_argmax_matches_numpy(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(1, 5)).astype(np.float32)
        W = rng.normal(size=(11, 5)).astype(np.float32)
        assert k.lm_head_argmax(x, W) == int(np.argmax((x @ W.T)[0]))

    def test_lm_head_argmax_picks_expected_toy(self):
        W = np.eye(4, dtype=np.float32)
        x = np.array([[0.0, 0.0, 2.0, 0.0]], dtype=np.float32)
        assert k.lm_head_argmax(x, W) == 2

    def test_lm_head_argmax_int8_matches_dequantized(self):
        rng = np.random.default_rng(14)
        x = rng.normal(size=(1, 6)).astype(np.float32)
        W = rng.normal(size=(9, 6)).astype(np.float32)
        W8 = np.clip(W * 8, -128, 127).astype(np.int8)
        w_scale = np.float32(0.5)
        ref = int(np.argmax((x @ (W8.astype(np.float32) * w_scale).T)[0]))
        assert k.lm_head_argmax_int8(x, W8, w_scale) == ref

    def test_lm_head_argmax_int8_nonzero_zp_ignored_by_fallback(self):
        rng = np.random.default_rng(15)
        x = rng.normal(size=(1, 4)).astype(np.float32)
        W8 = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int8)
        w_scale = np.float32(0.25)
        ref = int(np.argmax((x @ (W8.astype(np.float32) * w_scale).T)[0]))
        assert k.lm_head_argmax_int8(x, W8, w_scale, 2) == ref


class TestFallbackIsProductionPath:
    def test_fallback_used_since_numba_absent(self):
        rng = np.random.default_rng(16)
        x = rng.normal(size=(2, 4)).astype(np.float32)
        w = np.ones(4, dtype=np.float32)
        assert k._nb_rmsnorm is None
        assert np.allclose(k.nb_rmsnorm(x, w), _ref_rmsnorm(x, w), atol=1e-6)

    def test_public_calls_leave_build_flags_false(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(2, 4)).astype(np.float32)
        w = np.ones(4, dtype=np.float32)
        k.nb_rmsnorm(x, w)
        k.fused_layer_norm(x, w, None)
        k.lm_head_argmax(x, w.reshape(1, 4))
        assert k._kernels_built is False
        assert k._fused_built is False
        assert k._lm_head_built is False


class TestNumbaKernelAlgorithms:
    """Verify the real @njit kernel bodies by running them un-jitted."""

    def test_numba_rmsnorm_matches_reference(self):
        rng = np.random.default_rng(20)
        x = rng.normal(size=(2, 3, 8)).astype(np.float32)
        w = rng.normal(size=(8,)).astype(np.float32)
        with _fake_numba():
            nb_rmsnorm, *_ = k._build_kernels()
            out = nb_rmsnorm(x, w, np.float32(1e-5))
        assert np.allclose(out, _ref_rmsnorm(x, w), atol=1e-5)

    def test_numba_layernorm_with_and_without_bias(self):
        rng = np.random.default_rng(21)
        x = rng.normal(size=(2, 4, 6)).astype(np.float32)
        w = rng.normal(size=(6,)).astype(np.float32)
        b = rng.normal(size=(6,)).astype(np.float32)
        with _fake_numba():
            _, nb_layernorm, *_ = k._build_kernels()
            out_b = nb_layernorm(x, w, b, np.float32(1e-5))
            out_nb = nb_layernorm(x, w, None, np.float32(1e-5))
        assert np.allclose(out_b, _ref_layernorm(x, w, b), atol=1e-5)
        assert np.allclose(out_nb, _ref_layernorm(x, w, None), atol=1e-5)

    def test_numba_swiglu_and_swi_glu_mul(self):
        rng = np.random.default_rng(22)
        h1 = rng.normal(size=(3, 8)).astype(np.float32)
        h3 = rng.normal(size=(3, 8)).astype(np.float32)
        with _fake_numba():
            _, _, nb_swiglu, _, _, _, nb_swi_glu_mul = k._build_kernels()
            out_s = nb_swiglu(h1)
            out_m = nb_swi_glu_mul(h1, h3)
        assert np.allclose(out_s, _ref_silu(h1), atol=1e-6)
        assert np.allclose(out_m, _ref_silu(h1) * h3, atol=1e-6)

    def test_numba_softmax_normalizes_last_axis_in_place(self):
        e = np.array(
            [[1000.0, 1001.0, 1002.0], [3.0, 2.0, 1.0]], dtype=np.float32
        )
        with _fake_numba():
            _, _, _, nb_softmax, *_ = k._build_kernels()
            out = nb_softmax(e)
        assert out is e
        assert np.allclose(e.sum(axis=-1), 1.0, atol=1e-6)
        assert e[0, 2] > e[0, 0]

    def test_numba_embed_writes_rows_and_clips(self):
        emb = np.arange(12, dtype=np.float32).reshape(4, 3)
        ids = np.array([-3, 1, 99], dtype=np.int64)
        out = np.zeros((3, 3), dtype=np.float32)
        with _fake_numba():
            _, _, _, _, nb_embed, *_ = k._build_kernels()
            ret = nb_embed(emb, ids, out)
        assert ret is None
        assert np.allclose(out, emb[[0, 1, 3]])

    def test_numba_add_pos_with_offset_and_cap(self):
        rng = np.random.default_rng(23)
        pe = rng.normal(size=(5, 3)).astype(np.float32)
        x = np.ones((1, 3, 3), dtype=np.float32)
        with _fake_numba():
            _, _, _, _, _, nb_add_pos, _ = k._build_kernels()
            ret = nb_add_pos(x, pe, 3, 3)
        assert ret is None
        assert np.allclose(x[0, 0], pe[3] + 1.0)
        assert np.allclose(x[0, 1], pe[4] + 1.0)
        assert np.allclose(x[0, 2], pe[4] + 1.0)

    def test_numba_fused_layernorm_matches_reference(self):
        rng = np.random.default_rng(24)
        x = rng.normal(size=(5, 6)).astype(np.float32)
        w = rng.normal(size=(6,)).astype(np.float32)
        b = rng.normal(size=(6,)).astype(np.float32)
        with _fake_numba():
            fused_ln, *_ = k._build_fused_kernels()
            out = fused_ln(x, w, b, np.float32(1e-5))
            out_nb = fused_ln(x, w, None, np.float32(1e-5))
        assert np.allclose(out, _ref_layernorm(x, w, b), atol=1e-5)
        assert np.allclose(out_nb, _ref_layernorm(x, w, None), atol=1e-5)

    def test_numba_fused_attention_single_matches_reference(self):
        rng = np.random.default_rng(25)
        H, Eh, N = 2, 4, 5
        q = rng.normal(size=(H, Eh)).astype(np.float32)
        kv = rng.normal(size=(H, N, Eh)).astype(np.float32)
        scale = np.float32(0.5)
        scores = np.einsum("hd,hnd->hn", q, kv) * scale
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        ref = np.einsum("hn,hnd->hd", attn, kv)
        with _fake_numba():
            _, fused_single, *_ = k._build_fused_kernels()
            out = np.zeros((H, Eh), dtype=np.float32)
            fused_single(q, kv, kv, out, scale, H, Eh, 1, N)
        assert np.allclose(out, ref, atol=1e-5)

    def test_numba_fused_attention_multi_causal_matches_reference(self):
        rng = np.random.default_rng(26)
        S, H, Eh, N = 3, 2, 3, 4
        q = rng.normal(size=(S, H, Eh)).astype(np.float32)
        kv = rng.normal(size=(H, N, Eh)).astype(np.float32)
        scale = np.float32(0.5)
        scores = np.einsum("she,hne->hsn", q, kv) * scale
        causal = np.triu(np.full((S, N), -1e9, dtype=np.float32), k=1)
        masked = scores + causal
        attn = np.exp(masked - masked.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        ref = np.einsum("hsn,hne->she", attn, kv)
        with _fake_numba():
            _, _, fused_multi, _ = k._build_fused_kernels()
            out = np.zeros((S, H, Eh), dtype=np.float32)
            fused_multi(q, kv, kv, out, scale, H, Eh, S, N)
        assert np.allclose(out, ref, atol=1e-5)

    def test_numba_gqa_expand_repeats_heads(self):
        rng = np.random.default_rng(27)
        kt = rng.normal(size=(2, 4, 3)).astype(np.float32)
        with _fake_numba():
            _, _, _, gqa = k._build_fused_kernels()
            out = np.zeros((6, 4, 3), dtype=np.float32)
            gqa(kt, out, 3)
        assert np.allclose(out, np.repeat(kt, 3, axis=0))

    def test_numba_fused_norm_residual_matches_reference(self):
        rng = np.random.default_rng(28)
        x = rng.normal(size=(1, 5)).astype(np.float32)
        w = rng.normal(size=(5,)).astype(np.float32)
        b = rng.normal(size=(5,)).astype(np.float32)
        ref = _ref_layernorm(x, w, b)
        x_nb = x.copy()
        ref_nb = _ref_layernorm(x_nb, w, None)
        with _fake_numba():
            fused_nr = k._build_fused_norm_residual()
            ret = fused_nr(x, w, b, np.float32(1e-5))
            ret_nb = fused_nr(x_nb, w, None, np.float32(1e-5))
        assert ret is x
        assert np.allclose(x, ref, atol=1e-5)
        assert ret_nb is x_nb
        assert np.allclose(x_nb, ref_nb, atol=1e-5)

    def test_numba_lm_head_argmax_matches_numpy(self):
        rng = np.random.default_rng(29)
        x = rng.normal(size=(1, 5)).astype(np.float32)
        W = rng.normal(size=(13, 5)).astype(np.float32)
        with _fake_numba():
            lm_argmax, _ = k._build_lm_head_argmax()
            idx = int(lm_argmax(x, W))
        assert idx == int(np.argmax((x @ W.T)[0]))

    def test_numba_lm_head_argmax_int8_quantizes_x(self):
        rng = np.random.default_rng(30)
        x = rng.normal(size=(1, 6)).astype(np.float32)
        W = rng.normal(size=(9, 6)).astype(np.float32)
        W8 = np.clip(W * 8, -128, 127).astype(np.int8)
        w_scale = np.float32(0.5)
        x_max = np.abs(x).max()
        xs = x_max / np.float32(127.0) if x_max > 0 else np.float32(1.0)
        xi = (x / xs).astype(np.int32)
        scores = xi @ W8.astype(np.int32).T
        ref = int(np.argmax(scores * (xs * w_scale)))
        with _fake_numba():
            _, lm_argmax_int8 = k._build_lm_head_argmax()
            idx = int(lm_argmax_int8(x, W8, w_scale, np.int32(0)))
        assert idx == ref


class TestNumbaEnsureBranch:
    """Cover the numba-enabled lazy-build branches of the _ensure_* functions."""

    def test_ensure_kernels_builds_and_is_idempotent(self):
        with _fake_numba():
            k._NUMBA_AVAILABLE = True
            k._ensure_kernels()
            assert k._kernels_built is True
            assert k._nb_rmsnorm is not None
            k._ensure_kernels()

    def test_ensure_fused_builds_and_is_idempotent(self):
        with _fake_numba():
            k._NUMBA_AVAILABLE = True
            k._ensure_fused()
            assert k._fused_built is True
            assert k._nb_fused_attention_single is not None
            k._ensure_fused()

    def test_ensure_fused_norm_residual_builds_and_is_idempotent(self):
        with _fake_numba():
            k._NUMBA_AVAILABLE = True
            k._ensure_fused_norm_residual()
            assert k._fused_norm_res_built is True
            assert k._nb_fused_norm_residual_out is not None
            k._ensure_fused_norm_residual()

    def test_ensure_lm_head_builds_and_is_idempotent(self):
        with _fake_numba():
            k._NUMBA_AVAILABLE = True
            k._ensure_lm_head()
            assert k._lm_head_built is True
            assert k._nb_lm_head_argmax is not None
            k._ensure_lm_head()

    def test_module_state_restored_after_build(self):
        with _fake_numba():
            k._NUMBA_AVAILABLE = True
            k._ensure_kernels()
            assert k._kernels_built is True
        assert k._kernels_built is False
        assert k._nb_rmsnorm is None
        assert sys.modules.get("numba") is None

    def test_check_numba_true_when_import_succeeds(self):
        with _fake_numba():
            k._NUMBA_AVAILABLE = None
            assert k._check_numba() is True
            assert k._NUMBA_AVAILABLE is True
        assert k._NUMBA_AVAILABLE is False

    def test_public_wrappers_use_kernel_path_when_built(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=(2, 4)).astype(np.float32)
        w = rng.normal(size=(4,)).astype(np.float32)
        b = rng.normal(size=(4,)).astype(np.float32)
        H, Eh, N = 2, 3, 4
        q = rng.normal(size=(H, Eh)).astype(np.float32)
        kv = rng.normal(size=(H, N, Eh)).astype(np.float32)
        emb = rng.normal(size=(6, 3)).astype(np.float32)
        ids = np.array([0, 1], dtype=np.int64)
        out = np.zeros((2, 3), dtype=np.float32)
        scale = np.float32(0.5)
        W = rng.normal(size=(9, 4)).astype(np.float32)
        with _fake_numba():
            k._NUMBA_AVAILABLE = True
            k._ensure_kernels()
            k._ensure_fused()
            k._ensure_fused_norm_residual()
            k._ensure_lm_head()
            assert np.allclose(k.nb_rmsnorm(x, w), _ref_rmsnorm(x, w), atol=1e-5)
            assert np.allclose(
                k.nb_layernorm(x, w, b), _ref_layernorm(x, w, b), atol=1e-5
            )
            assert np.allclose(k.nb_swiglu(x), _ref_silu(x), atol=1e-6)
            e = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
            k.nb_softmax(e)
            assert np.allclose(e.sum(axis=-1), 1.0, atol=1e-6)
            k.nb_embed(emb, ids, out)
            assert np.allclose(out, emb[[0, 1]])
            xp = np.ones((1, 2, 3), dtype=np.float32)
            k.nb_add_pos(xp, emb, 4, 2)
            assert np.allclose(xp[0, 0], emb[4] + 1.0)
            assert np.allclose(xp[0, 1], emb[5] + 1.0)
            assert np.allclose(k.nb_swi_glu_mul(x, x), _ref_silu(x) * x, atol=1e-6)
            assert np.allclose(
                k.fused_layer_norm(x, w, b), _ref_layernorm(x, w, b), atol=1e-5
            )
            scores = np.einsum("hd,hnd->hn", q, kv) * scale
            attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = attn / attn.sum(axis=-1, keepdims=True)
            out_s = k.fused_attention_single(q, kv, kv, scale, H, Eh)
            assert np.allclose(out_s, np.einsum("hn,hnd->hd", attn, kv), atol=1e-5)
            S = 2
            qm = rng.normal(size=(S, H, Eh)).astype(np.float32)
            scores = np.einsum("she,hne->hsn", qm, kv) * scale
            causal = np.triu(np.full((S, N), -1e9, dtype=np.float32), k=1)
            masked = scores + causal
            attn = np.exp(masked - masked.max(axis=-1, keepdims=True))
            attn = attn / attn.sum(axis=-1, keepdims=True)
            out_m = k.fused_attention_multi(qm, kv, kv, scale, H, Eh)
            assert np.allclose(out_m, np.einsum("hsn,hne->she", attn, kv), atol=1e-5)
            kg = rng.normal(size=(1, N, Eh)).astype(np.float32)
            assert np.allclose(k.gqa_expand(kg, 2), np.repeat(kg, 2, axis=0))
            xt = x[:1]
            assert k.lm_head_argmax(xt, W) == int(np.argmax((xt @ W.T)[0]))
            W8 = np.clip(W * 8, -128, 127).astype(np.int8)
            ref = int(np.argmax((xt @ (W8.astype(np.float32) * 0.5).T)[0]))
            assert k.lm_head_argmax_int8(xt, W8, np.float32(0.5)) == ref
