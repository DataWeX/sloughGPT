"""Tests for numpy_forward — generic transformer forward pass."""

import numpy as np
import pytest
from domains.infrastructure.arch_config import ArchConfig, GPT2_WEIGHT_MAP, LLAMA_WEIGHT_MAP
from domains.infrastructure.numpy_forward import (
    norm_fn,
    forward,
    forward_cached,
    pre_extract_weights,
    forward_fast,
)
from domains.infrastructure.numpy_engine import KVCache


def _make_gpt2_config(n_layers=2, n_head=4, n_embed=32):
    """Create a minimal GPT-2-like ArchConfig for testing."""
    return ArchConfig(
        name="gpt2-test",
        norm="layer_norm",
        positional="absolute",
        activation="gelu",
        attention="mha",
        weight_map=GPT2_WEIGHT_MAP,
        transpose_weights=True,
        n_head=n_head,
        n_kv_head=n_head,
        n_embed=n_embed,
        n_layers=n_layers,
        head_dim=n_embed // n_head,
        rope_base=10000.0,
        tied_weights=True,
    )


def _make_gpt2_weights(arch: ArchConfig, vocab_size=100, seq_len=10):
    """Create random weights matching GPT-2 weight map."""
    W = {}
    ne = arch.n_embed

    W["wte.weight"] = np.random.randn(vocab_size, ne).astype(np.float32) * 0.02
    W["wpe.weight"] = np.random.randn(seq_len, ne).astype(np.float32) * 0.02

    for i in range(arch.n_layers):
        W[f"h.{i}.ln_1.weight"] = np.ones(ne, dtype=np.float32)
        W[f"h.{i}.ln_1.bias"] = np.zeros(ne, dtype=np.float32)
        W[f"h.{i}.attn.c_attn.weight"] = np.random.randn(ne, 3 * ne).astype(np.float32) * 0.02
        W[f"h.{i}.attn.c_attn.bias"] = np.zeros(3 * ne, dtype=np.float32)
        W[f"h.{i}.attn.c_proj.weight"] = np.random.randn(ne, ne).astype(np.float32) * 0.02
        W[f"h.{i}.attn.c_proj.bias"] = np.zeros(ne, dtype=np.float32)
        W[f"h.{i}.ln_2.weight"] = np.ones(ne, dtype=np.float32)
        W[f"h.{i}.ln_2.bias"] = np.zeros(ne, dtype=np.float32)
        W[f"h.{i}.mlp.c_fc.weight"] = np.random.randn(ne, 4 * ne).astype(np.float32) * 0.02
        W[f"h.{i}.mlp.c_fc.bias"] = np.zeros(4 * ne, dtype=np.float32)
        W[f"h.{i}.mlp.c_proj.weight"] = np.random.randn(4 * ne, ne).astype(np.float32) * 0.02
        W[f"h.{i}.mlp.c_proj.bias"] = np.zeros(ne, dtype=np.float32)

    W["ln_f.weight"] = np.ones(ne, dtype=np.float32)
    W["ln_f.bias"] = np.zeros(ne, dtype=np.float32)

    return W


def _make_llama_config(n_layers=2, n_head=4, n_embed=32, n_kv_head=2):
    """Create a minimal LLaMA-like ArchConfig."""
    return ArchConfig(
        name="llama-test",
        norm="rms_norm",
        positional="rope",
        activation="swiglu",
        attention="gqa",
        weight_map=LLAMA_WEIGHT_MAP,
        transpose_weights=False,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embed=n_embed,
        n_layers=n_layers,
        head_dim=n_embed // n_head,
        rope_base=10000.0,
        tied_weights=True,
    )


def _make_llama_weights(arch: ArchConfig, vocab_size=100, seq_len=10):
    """Create random weights matching LLaMA weight map.

    transpose_weights=False means T(w) = w.T, so weights are stored as (out, in).
    """
    W = {}
    ne = arch.n_embed
    kv_dim = arch.n_kv_head * arch.head_dim
    ffn_dim = 4 * ne

    # embed tokens stored as (vocab, ne) — not transposed in forward
    W["model.embed_tokens.weight"] = np.random.randn(vocab_size, ne).astype(np.float32) * 0.02

    for i in range(arch.n_layers):
        # Norm weights are 1-D, no transpose needed
        W[f"model.layers.{i}.input_layernorm.weight"] = np.ones(ne, dtype=np.float32)
        W[f"model.layers.{i}.post_attention_layernorm.weight"] = np.ones(ne, dtype=np.float32)

        # Attention projections stored as (out, in) — T() transposes to (in, out) for matmul
        W[f"model.layers.{i}.self_attn.q_proj.weight"] = np.random.randn(ne, ne).astype(np.float32) * 0.02
        W[f"model.layers.{i}.self_attn.k_proj.weight"] = np.random.randn(kv_dim, ne).astype(np.float32) * 0.02
        W[f"model.layers.{i}.self_attn.v_proj.weight"] = np.random.randn(kv_dim, ne).astype(np.float32) * 0.02
        W[f"model.layers.{i}.self_attn.q_proj.bias"] = np.zeros(ne, dtype=np.float32)
        W[f"model.layers.{i}.self_attn.k_proj.bias"] = np.zeros(kv_dim, dtype=np.float32)
        W[f"model.layers.{i}.self_attn.v_proj.bias"] = np.zeros(kv_dim, dtype=np.float32)
        W[f"model.layers.{i}.self_attn.o_proj.weight"] = np.random.randn(ne, ne).astype(np.float32) * 0.02

        # FFN projections stored as (out, in)
        W[f"model.layers.{i}.mlp.gate_proj.weight"] = np.random.randn(ffn_dim, ne).astype(np.float32) * 0.02
        W[f"model.layers.{i}.mlp.up_proj.weight"] = np.random.randn(ffn_dim, ne).astype(np.float32) * 0.02
        W[f"model.layers.{i}.mlp.down_proj.weight"] = np.random.randn(ne, ffn_dim).astype(np.float32) * 0.02

    W["model.norm.weight"] = np.ones(ne, dtype=np.float32)

    return W


# ── norm_fn ────────────────────────────────────────────────────────────────

class TestNormFn:
    def test_returns_rmsnorm_for_rms_norm(self):
        arch = _make_gpt2_config()
        arch.norm = "rms_norm"
        fn = norm_fn(arch)
        from domains.infrastructure.numpy_ops import rmsnorm
        assert fn is rmsnorm

    def test_returns_layer_norm_for_layer_norm(self):
        arch = _make_gpt2_config()
        arch.norm = "layer_norm"
        fn = norm_fn(arch)
        from domains.infrastructure.numpy_ops import layer_norm
        assert fn is layer_norm

    def test_norm_fn_returns_callable(self):
        arch = _make_gpt2_config()
        fn = norm_fn(arch)
        assert callable(fn)

    def test_norm_fn_with_rms_norm_actually_normalizes(self):
        arch = _make_gpt2_config()
        arch.norm = "rms_norm"
        fn = norm_fn(arch)
        x = np.array([[3.0, 4.0]], dtype=np.float32)
        w = np.ones(2, dtype=np.float32)
        result = fn(x, w)
        rms = np.sqrt(np.mean(result ** 2))
        assert np.isclose(rms, 1.0, atol=1e-5)

    def test_norm_fn_with_layer_norm_actually_normalizes(self):
        arch = _make_gpt2_config()
        arch.norm = "layer_norm"
        fn = norm_fn(arch)
        x = np.array([[10.0, 20.0]], dtype=np.float32)
        w = np.ones(2, dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        result = fn(x, w, b)
        assert np.isclose(result.mean(), 0.0, atol=1e-4)


# ── forward (GPT-2) ──────────────────────────────────────────────────────

class TestForwardGPT2:
    def test_output_shape(self):
        arch = _make_gpt2_config(n_layers=2, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch, vocab_size=100)
        logits = forward(weights, arch, [0, 1, 2])
        assert logits.shape == (100,)

    def test_output_is_finite(self):
        arch = _make_gpt2_config(n_layers=2, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch, vocab_size=100)
        logits = forward(weights, arch, [0, 1, 2])
        assert np.all(np.isfinite(logits))

    def test_different_inputs_different_outputs(self):
        arch = _make_gpt2_config(n_layers=2, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch)
        logits1 = forward(weights, arch, [0, 1])
        logits2 = forward(weights, arch, [2, 3])
        assert not np.allclose(logits1, logits2)

    def test_single_token(self):
        arch = _make_gpt2_config(n_layers=1, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch, vocab_size=100)
        logits = forward(weights, arch, [5])
        assert logits.shape == (100,)

    def test_deterministic(self):
        arch = _make_gpt2_config(n_layers=1, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch)
        logits1 = forward(weights, arch, [0, 1, 2])
        logits2 = forward(weights, arch, [0, 1, 2])
        np.testing.assert_array_equal(logits1, logits2)

    def test_long_sequence(self):
        arch = _make_gpt2_config(n_layers=1, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch, vocab_size=50, seq_len=20)
        logits = forward(weights, arch, list(range(15)))
        assert logits.shape == (50,)

    def test_varying_seq_lengths(self):
        arch = _make_gpt2_config(n_layers=1, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch, vocab_size=50)
        logits_1 = forward(weights, arch, [0])
        logits_5 = forward(weights, arch, [0, 1, 2, 3, 4])
        assert logits_1.shape == (50,)
        assert logits_5.shape == (50,)

    def test_many_layers(self):
        arch = _make_gpt2_config(n_layers=4, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch, vocab_size=50)
        logits = forward(weights, arch, [0, 1])
        assert logits.shape == (50,)
        assert np.all(np.isfinite(logits))

    def test_nonzero_bias_effect(self):
        arch = _make_gpt2_config(n_layers=1, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch, vocab_size=50)
        logits_base = forward(weights, arch, [0, 1])
        weights["h.0.attn.c_attn.bias"] = np.ones(3 * arch.n_embed, dtype=np.float32)
        logits_biased = forward(weights, arch, [0, 1])
        assert not np.allclose(logits_base, logits_biased)

    def test_zero_weights_gives_near_zero_output(self):
        arch = _make_gpt2_config(n_layers=1, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch, vocab_size=50)
        for k in weights:
            if "weight" in k and weights[k].ndim == 2:
                weights[k] = np.zeros_like(weights[k])
        logits = forward(weights, arch, [0, 1])
        assert np.all(np.isfinite(logits))

    def test_varying_embed_sizes(self):
        for ne in [16, 32, 64]:
            arch = _make_gpt2_config(n_layers=1, n_head=4, n_embed=ne)
            weights = _make_gpt2_weights(arch, vocab_size=50)
            logits = forward(weights, arch, [0, 1])
            assert logits.shape == (50,)


# ── forward (LLaMA) ──────────────────────────────────────────────────────

class TestForwardLLaMA:
    def test_llama_output_shape(self):
        arch = _make_llama_config(n_layers=2, n_head=4, n_embed=32, n_kv_head=2)
        weights = _make_llama_weights(arch, vocab_size=100)
        logits = forward(weights, arch, [0, 1, 2])
        assert logits.shape == (100,)

    def test_llama_output_is_finite(self):
        arch = _make_llama_config(n_layers=2, n_head=4, n_embed=32, n_kv_head=2)
        weights = _make_llama_weights(arch, vocab_size=100)
        logits = forward(weights, arch, [0, 1, 2])
        assert np.all(np.isfinite(logits))

    def test_llama_deterministic(self):
        arch = _make_llama_config(n_layers=1, n_head=4, n_embed=32, n_kv_head=2)
        weights = _make_llama_weights(arch, vocab_size=100)
        logits1 = forward(weights, arch, [0, 1])
        logits2 = forward(weights, arch, [0, 1])
        np.testing.assert_array_equal(logits1, logits2)

    def test_llama_single_token(self):
        arch = _make_llama_config(n_layers=1, n_head=4, n_embed=32, n_kv_head=2)
        weights = _make_llama_weights(arch, vocab_size=100)
        logits = forward(weights, arch, [5])
        assert logits.shape == (100,)

    def test_llama_gqa_heads_match(self):
        arch = _make_llama_config(n_layers=1, n_head=4, n_embed=32, n_kv_head=4)
        weights = _make_llama_weights(arch, vocab_size=50)
        logits = forward(weights, arch, [0, 1])
        assert logits.shape == (50,)
        assert np.all(np.isfinite(logits))

    def test_llama_many_layers(self):
        arch = _make_llama_config(n_layers=3, n_head=4, n_embed=32, n_kv_head=2)
        weights = _make_llama_weights(arch, vocab_size=50)
        logits = forward(weights, arch, [0, 1])
        assert logits.shape == (50,)
        assert np.all(np.isfinite(logits))


# ── pre_extract_weights ──────────────────────────────────────────────────

class TestPreExtractWeights:
    def test_extracts_all_weights(self):
        arch = _make_gpt2_config(n_layers=2, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch)
        extracted = pre_extract_weights(arch, weights)

        for i in range(arch.n_layers):
            assert f"layers.{{i}}.attn_norm.weight:{i}" in extracted
            assert f"layers.{{i}}.ffn.up.weight:{i}" in extracted

        assert "embed.token" in extracted
        assert "final_norm.weight" in extracted

    def test_values_match_original(self):
        arch = _make_gpt2_config(n_layers=1, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch)
        extracted = pre_extract_weights(arch, weights)
        np.testing.assert_array_equal(
            extracted["embed.token"],
            weights["wte.weight"]
        )

    def test_layer_weights_per_layer(self):
        arch = _make_gpt2_config(n_layers=3, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch)
        extracted = pre_extract_weights(arch, weights)
        for i in range(3):
            assert f"layers.{{i}}.attn_norm.weight:{i}" in extracted
            assert f"layers.{{i}}.ffn.down.weight:{i}" in extracted

    def test_contiguous_array(self):
        arch = _make_gpt2_config(n_layers=1, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch)
        extracted = pre_extract_weights(arch, weights)
        for v in extracted.values():
            assert v.flags["C_CONTIGUOUS"]

    def test_missing_weight_not_in_extracted(self):
        arch = _make_gpt2_config(n_layers=1, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch)
        del weights["h.0.mlp.c_proj.bias"]
        extracted = pre_extract_weights(arch, weights)
        assert "layers.{i}.ffn.down.bias:0" not in extracted

    def test_llama_extract_weights(self):
        arch = _make_llama_config(n_layers=2, n_head=4, n_embed=32, n_kv_head=2)
        weights = _make_llama_weights(arch)
        extracted = pre_extract_weights(arch, weights)
        assert "embed.token" in extracted
        assert "final_norm.weight" in extracted
        assert "layers.{i}.ffn.gate.weight:0" in extracted


# ── forward_fast ────────────────────────────────────────────────────────

class TestForwardFast:
    def test_matches_forward(self):
        arch = _make_gpt2_config(n_layers=2, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch)
        extracted = pre_extract_weights(arch, weights)

        token_ids = [0, 1, 2]
        logits_ref = forward(weights, arch, token_ids)
        logits_fast = forward_fast(extracted, arch, token_ids)
        np.testing.assert_allclose(logits_ref, logits_fast, atol=1e-5)

    def test_single_token_matches(self):
        arch = _make_gpt2_config(n_layers=1, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch)
        extracted = pre_extract_weights(arch, weights)
        logits_ref = forward(weights, arch, [3])
        logits_fast = forward_fast(extracted, arch, [3])
        np.testing.assert_allclose(logits_ref, logits_fast, atol=1e-5)

    def test_llama_fast_matches_forward(self):
        arch = _make_llama_config(n_layers=2, n_head=4, n_embed=32, n_kv_head=2)
        weights = _make_llama_weights(arch)
        extracted = pre_extract_weights(arch, weights)
        logits_ref = forward(weights, arch, [0, 1])
        logits_fast = forward_fast(extracted, arch, [0, 1])
        np.testing.assert_allclose(logits_ref, logits_fast, atol=1e-5)

    def test_fast_deterministic(self):
        arch = _make_gpt2_config(n_layers=1, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch)
        extracted = pre_extract_weights(arch, weights)
        logits1 = forward_fast(extracted, arch, [0, 1])
        logits2 = forward_fast(extracted, arch, [0, 1])
        np.testing.assert_array_equal(logits1, logits2)

    def test_fast_many_layers(self):
        arch = _make_gpt2_config(n_layers=4, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch)
        extracted = pre_extract_weights(arch, weights)
        logits_ref = forward(weights, arch, [0, 1, 2])
        logits_fast = forward_fast(extracted, arch, [0, 1, 2])
        np.testing.assert_allclose(logits_ref, logits_fast, atol=1e-5)

    def test_fast_output_shape(self):
        arch = _make_gpt2_config(n_layers=2, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch, vocab_size=80)
        extracted = pre_extract_weights(arch, weights)
        logits = forward_fast(extracted, arch, [0, 1])
        assert logits.shape == (80,)


# ── forward_cached ──────────────────────────────────────────────────────

class TestForwardCached:
    def test_single_token_matches_full(self):
        arch = _make_gpt2_config(n_layers=2, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch)

        def get_weight(name):
            return weights[name]

        token_ids = [0, 1, 2]
        logits_full = forward(weights, arch, token_ids)

        cache = KVCache(arch.n_layers)
        logits_cached = forward_cached(get_weight, arch, token_ids, kv_cache=cache)
        np.testing.assert_allclose(logits_full, logits_cached, atol=1e-5)

    def test_incremental_matches_full(self):
        arch = _make_gpt2_config(n_layers=1, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch)

        def get_weight(name):
            return weights[name]

        token_ids = [0, 1, 2, 3]
        logits_full = forward(weights, arch, token_ids)

        cache = KVCache(arch.n_layers)
        # Process all tokens at once with cache
        logits_cached = forward_cached(get_weight, arch, token_ids, kv_cache=cache)
        np.testing.assert_allclose(logits_full, logits_cached, atol=1e-5)

    def test_no_cache_matches_full(self):
        arch = _make_gpt2_config(n_layers=1, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch)

        def get_weight(name):
            return weights[name]

        token_ids = [0, 1, 2]
        logits_full = forward(weights, arch, token_ids)
        logits_cached = forward_cached(get_weight, arch, token_ids, kv_cache=None)
        np.testing.assert_allclose(logits_full, logits_cached, atol=1e-5)

    def test_deterministic(self):
        arch = _make_gpt2_config(n_layers=1, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch)

        def get_weight(name):
            return weights[name]

        cache1 = KVCache(arch.n_layers)
        logits1 = forward_cached(get_weight, arch, [0, 1, 2], kv_cache=cache1)
        cache2 = KVCache(arch.n_layers)
        logits2 = forward_cached(get_weight, arch, [0, 1, 2], kv_cache=cache2)
        np.testing.assert_array_equal(logits1, logits2)

    def test_llama_cached_matches_full(self):
        arch = _make_llama_config(n_layers=1, n_head=4, n_embed=32, n_kv_head=2)
        weights = _make_llama_weights(arch)

        def get_weight(name):
            return weights[name]

        token_ids = [0, 1]
        logits_full = forward(weights, arch, token_ids)
        cache = KVCache(arch.n_layers)
        logits_cached = forward_cached(get_weight, arch, token_ids, kv_cache=cache)
        np.testing.assert_allclose(logits_full, logits_cached, atol=1e-5)

    def test_start_pos_affects_rope(self):
        from domains.infrastructure.numpy_ops import rope
        x = np.random.randn(3, 4, 8).astype(np.float32)
        r0 = rope(x, 0, 8, 10000.0)
        r10 = rope(x, 10, 8, 10000.0)
        assert not np.allclose(r0, r10)
