"""
Numeric tolerance contract for the parallelised slonet kernels (card 21135daa).

Card f30be275 (nogil) required bit-identical outputs. This card (parallel=True
+ prange) deliberately does NOT re-require bit-identity in general: parallel
reductions reorder floating-point adds. The shipped implementation pranges
only over axes of INDEPENDENT output elements (rows/heads/elements), keeping
every reduction sequential within its row, so the outputs observed during the
card's measurements were bit-identical to the serial forms — but the bound
this file encodes is tolerance, not identity, so a future kernel that does
reorder adds stays legal within the same contract.

Tolerance derivation (fp32 throughout, eps = 2**-23 ~= 1.19e-7):
- Element-wise kernels (swiglu, swi_glu_mul, fused_swiglu_inplace): one op
  chain per element, no accumulation — only libm ULP differences between
  numba's exp/tanh and numpy's can appear. rtol=1e-5, atol=1e-6 (~85 eps).
- Reduction kernels (layernorm, softmax, attention): sequential accumulation
  over at most E*seq = 64*128 = 8192 terms; worst-case forward error
  ~ n*eps ~= 1e-3 absolute on order-1 values, typical ~ sqrt(n)*eps < 1e-5
  relative. rtol=1e-4, atol=1e-5 (~10x margin over the typical bound).

Determinism pin: the parallel forms schedule independent-output pranges only,
so repeated calls must be exactly equal (no run-to-run scheduling noise) —
generation stability depends on this.
"""

import numpy as np

RTOL_ELEM, ATOL_ELEM = 1e-5, 1e-6
RTOL_REDUCE, ATOL_REDUCE = 1e-4, 1e-5


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class TestParallelElementwise:
    """prange over flat element indices — no accumulation anywhere."""

    def test_swiglu_matches_numpy_reference(self):
        from domain.training._internal.slonet_kernels import nb_swiglu

        rng = np.random.default_rng(42)
        h1 = rng.standard_normal((256, 896)).astype(np.float32)
        out = nb_swiglu(h1.copy())
        np.testing.assert_allclose(out, h1 * _sigmoid(h1), rtol=RTOL_ELEM, atol=ATOL_ELEM)

    def test_swiglu_deterministic_between_runs(self):
        from domain.training._internal.slonet_kernels import nb_swiglu

        rng = np.random.default_rng(7)
        h1 = rng.standard_normal((256, 896)).astype(np.float32)
        assert np.array_equal(nb_swiglu(h1.copy()), nb_swiglu(h1.copy()))

    def test_swi_glu_mul_matches_numpy_reference(self):
        from domain.training._internal.slonet_kernels import nb_swi_glu_mul

        rng = np.random.default_rng(43)
        h1 = rng.standard_normal((256, 896)).astype(np.float32)
        h3 = rng.standard_normal((256, 896)).astype(np.float32)
        out = nb_swi_glu_mul(h1, h3)
        np.testing.assert_allclose(out, (h1 * _sigmoid(h1)) * h3, rtol=RTOL_ELEM, atol=ATOL_ELEM)

    # NOTE: the internal _fused_swiglu_inplace kernel (also parallelised in
    # card 21135daa) exposes no module-level handle — it is compiled inside
    # _build_fused_kernels and reached only through the warmup/block path.
    # Its elementwise numerics are pinned by test_swi_glu_mul above (same
    # op chain) and end-to-end by the transformer benchmark.


class TestParallelReductions:
    """prange over rows/heads; every reduction stays sequential within a row."""

    def test_softmax_matches_numpy_reference(self):
        from domain.training._internal.slonet_kernels import nb_softmax

        rng = np.random.default_rng(45)
        e = rng.standard_normal((256, 896)).astype(np.float32) * 3.0
        out = nb_softmax(e.copy())
        shifted = e - e.max(axis=-1, keepdims=True)
        expected = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(out, expected, rtol=RTOL_REDUCE, atol=ATOL_REDUCE)

    def test_fused_layer_norm_matches_numpy_reference(self):
        from domain.training._internal.slonet_kernels import fused_layer_norm

        rng = np.random.default_rng(46)
        E = 896
        x = (rng.standard_normal((256, E)) * 2.0 + 0.5).astype(np.float32)
        w = rng.standard_normal(E).astype(np.float32)
        b = rng.standard_normal(E).astype(np.float32)
        eps = np.float32(1e-5)
        mu = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
        expected = ((x - mu) / np.sqrt(var + eps)) * w + b
        out = fused_layer_norm(x, w, b, eps)
        np.testing.assert_allclose(out, expected, rtol=RTOL_REDUCE, atol=ATOL_REDUCE)

    def test_fused_attention_single_matches_numpy_reference(self):
        from domain.training._internal.slonet_kernels import fused_attention_single

        rng = np.random.default_rng(47)
        H, E_head, new_len = 8, 64, 128
        q = rng.standard_normal((H, E_head)).astype(np.float32)
        k = rng.standard_normal((H, new_len, E_head)).astype(np.float32)
        v = rng.standard_normal((H, new_len, E_head)).astype(np.float32)
        scale = np.float32(1.0 / np.sqrt(E_head))
        out = fused_attention_single(q, k, v, scale, H, E_head)
        scores = np.einsum("hd,hnd->hn", q, k) * scale
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        expected = np.einsum("hn,hnd->hd", attn, v)
        np.testing.assert_allclose(out, expected, rtol=RTOL_REDUCE, atol=ATOL_REDUCE)

    def test_fused_attention_multi_matches_causal_reference(self):
        from domain.training._internal.slonet_kernels import fused_attention_multi

        rng = np.random.default_rng(48)
        H, E_head, seq_len, new_len = 8, 64, 16, 128
        q = rng.standard_normal((seq_len, H, E_head)).astype(np.float32)
        k = rng.standard_normal((H, new_len, E_head)).astype(np.float32)
        v = rng.standard_normal((H, new_len, E_head)).astype(np.float32)
        scale = np.float32(1.0 / np.sqrt(E_head))
        out = fused_attention_multi(q, k, v, scale, H, E_head)
        scores = np.einsum("she,hne->hsn", q, k) * scale
        causal = np.triu(np.full((seq_len, new_len), -1e9, dtype=np.float32), k=1)
        scores = scores + causal
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        expected = np.einsum("hsn,hne->she", attn, v)
        np.testing.assert_allclose(out, expected, rtol=RTOL_REDUCE, atol=ATOL_REDUCE)

    def test_parallel_attention_deterministic_between_runs(self):
        from domain.training._internal.slonet_kernels import fused_attention_multi

        rng = np.random.default_rng(49)
        H, E_head, seq_len, new_len = 8, 64, 16, 128
        q = rng.standard_normal((seq_len, H, E_head)).astype(np.float32)
        k = rng.standard_normal((H, new_len, E_head)).astype(np.float32)
        v = rng.standard_normal((H, new_len, E_head)).astype(np.float32)
        scale = np.float32(1.0 / np.sqrt(E_head))
        out1 = fused_attention_multi(q, k, v, scale, H, E_head)
        out2 = fused_attention_multi(q, k, v, scale, H, E_head)
        assert np.array_equal(out1, out2)
