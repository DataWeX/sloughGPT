"""Tests for numpy_forward — generic transformer forward pass."""

import numpy as np
import pytest
from domains.infrastructure.arch_config import ArchConfig, GPT2_WEIGHT_MAP
from domains.infrastructure.numpy_forward import (
    norm_fn,
    forward,
    forward_cached,
    pre_extract_weights,
    forward_fast,
)


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
    hd = arch.head_dim

    # Embeddings
    W["wte.weight"] = np.random.randn(vocab_size, ne).astype(np.float32) * 0.02
    W["wpe.weight"] = np.random.randn(seq_len, ne).astype(np.float32) * 0.02

    for i in range(arch.n_layers):
        # Attention norms
        W[f"h.{i}.ln_1.weight"] = np.ones(ne, dtype=np.float32)
        W[f"h.{i}.ln_1.bias"] = np.zeros(ne, dtype=np.float32)
        # Combined QKV: (ne, 3*ne) — GPT-2 stores as (in, out)
        W[f"h.{i}.attn.c_attn.weight"] = np.random.randn(ne, 3 * ne).astype(np.float32) * 0.02
        W[f"h.{i}.attn.c_attn.bias"] = np.zeros(3 * ne, dtype=np.float32)
        # Output projection
        W[f"h.{i}.attn.c_proj.weight"] = np.random.randn(ne, ne).astype(np.float32) * 0.02
        W[f"h.{i}.attn.c_proj.bias"] = np.zeros(ne, dtype=np.float32)
        # FFN norms
        W[f"h.{i}.ln_2.weight"] = np.ones(ne, dtype=np.float32)
        W[f"h.{i}.ln_2.bias"] = np.zeros(ne, dtype=np.float32)
        # FFN up/down (GELU MLP)
        W[f"h.{i}.mlp.c_fc.weight"] = np.random.randn(ne, 4 * ne).astype(np.float32) * 0.02
        W[f"h.{i}.mlp.c_fc.bias"] = np.zeros(4 * ne, dtype=np.float32)
        W[f"h.{i}.mlp.c_proj.weight"] = np.random.randn(4 * ne, ne).astype(np.float32) * 0.02
        W[f"h.{i}.mlp.c_proj.bias"] = np.zeros(ne, dtype=np.float32)

    # Final norm
    W["ln_f.weight"] = np.ones(ne, dtype=np.float32)
    W["ln_f.bias"] = np.zeros(ne, dtype=np.float32)

    return W


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


class TestForwardGPT2:
    def test_output_shape(self):
        arch = _make_gpt2_config(n_layers=2, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch, vocab_size=100)
        logits = forward(weights, arch, [0, 1, 2])
        # LM head: n_embed → vocab_size (weight tying with embed)
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


class TestPreExtractWeights:
    def test_extracts_all_weights(self):
        arch = _make_gpt2_config(n_layers=2, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch)
        extracted = pre_extract_weights(arch, weights)

        # Check layer-specific weights exist (keys use canonical:{layer} format)
        for i in range(arch.n_layers):
            assert f"layers.{{i}}.attn_norm.weight:{i}" in extracted
            assert f"layers.{{i}}.ffn.up.weight:{i}" in extracted

        # Check global weights
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


class TestForwardFast:
    def test_matches_forward(self):
        arch = _make_gpt2_config(n_layers=2, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch)
        extracted = pre_extract_weights(arch, weights)

        token_ids = [0, 1, 2]
        logits_ref = forward(weights, arch, token_ids)
        logits_fast = forward_fast(extracted, arch, token_ids)
        np.testing.assert_allclose(logits_ref, logits_fast, atol=1e-5)


class TestForwardCached:
    def test_single_token_matches_full(self):
        arch = _make_gpt2_config(n_layers=2, n_head=4, n_embed=32)
        weights = _make_gpt2_weights(arch)

        def get_weight(name):
            return weights[name]

        token_ids = [0, 1, 2]
        logits_full = forward(weights, arch, token_ids)

        # forward_cached returns per-token logits when called with all tokens
        from domains.infrastructure.numpy_engine import KVCache
        cache = KVCache(arch.n_layers)
        logits_cached = forward_cached(get_weight, arch, token_ids, kv_cache=cache)
        np.testing.assert_allclose(logits_full, logits_cached, atol=1e-5)
