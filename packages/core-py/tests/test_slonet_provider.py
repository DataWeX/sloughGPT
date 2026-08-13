"""Tests for the universal HF→SloNet weight converter."""

import numpy as np
import pytest

from domains.inference.slonet_provider import convert_hf_to_slonet


def _fake_gpt2_state_dict():
    """Create a minimal GPT-2 style state dict (fused QKV, GELU MLP)."""
    n_embed, n_layer = 64, 2
    sd = {}
    sd["wte.weight"] = np.random.randn(1000, n_embed).astype(np.float32)
    sd["wpe.weight"] = np.random.randn(512, n_embed).astype(np.float32)
    sd["ln_f.weight"] = np.ones(n_embed, dtype=np.float32)
    sd["ln_f.bias"] = np.zeros(n_embed, dtype=np.float32)
    for i in range(n_layer):
        sd[f"h.{i}.ln_1.weight"] = np.ones(n_embed, dtype=np.float32)
        sd[f"h.{i}.ln_1.bias"] = np.zeros(n_embed, dtype=np.float32)
        sd[f"h.{i}.attn.c_attn.weight"] = np.random.randn(n_embed, 3 * n_embed).astype(np.float32)
        sd[f"h.{i}.attn.c_attn.bias"] = np.random.randn(3 * n_embed).astype(np.float32)
        sd[f"h.{i}.attn.c_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
        sd[f"h.{i}.attn.c_proj.bias"] = np.zeros(n_embed, dtype=np.float32)
        sd[f"h.{i}.ln_2.weight"] = np.ones(n_embed, dtype=np.float32)
        sd[f"h.{i}.ln_2.bias"] = np.zeros(n_embed, dtype=np.float32)
        sd[f"h.{i}.mlp.c_fc.weight"] = np.random.randn(n_embed, 4 * n_embed).astype(np.float32)
        sd[f"h.{i}.mlp.c_fc.bias"] = np.zeros(4 * n_embed, dtype=np.float32)
        sd[f"h.{i}.mlp.c_proj.weight"] = np.random.randn(4 * n_embed, n_embed).astype(np.float32)
        sd[f"h.{i}.mlp.c_proj.bias"] = np.zeros(n_embed, dtype=np.float32)
    return sd


def test_universal_converter_gpt2():
    """GPT-2 style: fused QKV, GELU MLP, no positional embeddings in SloNet."""
    sd = _fake_gpt2_state_dict()
    result = convert_hf_to_slonet(sd, n_layer=2)

    # Should have: tok_emb, lm_head, 2x (attn_norm, q/k/v, o_proj, ff_norm, w1/w2/w3), final norm
    # Check key groups exist
    assert "tok_emb.weight" in result
    assert "lm_head.weight" in result
    for i in range(2):
        assert f"blocks.{i}.attn_norm.weight" in result
        assert f"blocks.{i}.attn.q_proj.weight" in result
        assert f"blocks.{i}.attn.k_proj.weight" in result
        assert f"blocks.{i}.attn.v_proj.weight" in result
        assert f"blocks.{i}.attn.o_proj.weight" in result
        assert f"blocks.{i}.ff_norm.weight" in result
        assert f"blocks.{i}.ff.w1.weight" in result
        assert f"blocks.{i}.ff.w2.weight" in result
        assert f"blocks.{i}.ff.w3.weight" in result  # synthesized identity

    # QKV split: fused is (64, 192) → transpose → (192, 64) → split into 3 × (64, 64)
    q = result["blocks.0.attn.q_proj.weight"]
    k = result["blocks.0.attn.k_proj.weight"]
    v = result["blocks.0.attn.v_proj.weight"]
    assert q.shape == (64, 64)
    assert k.shape == q.shape
    assert v.shape == q.shape

    # FF: w1 is the first linear (from c_fc), w2 is the second (from c_proj), w3 is identity
    w1 = result["blocks.0.ff.w1.weight"]
    w3 = result["blocks.0.ff.w3.weight"]
    assert w1.shape == (4 * 64, 64)  # transposed from (64, 256)
    assert w3.shape == w1.shape
    # w3 should be zeros (identity when multiplied with sigmoid-like activation)
    assert np.allclose(w3, 0.0)


def test_universal_converter_llama_style():
    """LLaMA-style: split QKV, SwiGLU MLP."""
    n_embed, n_layer = 64, 2
    sd = {}
    sd["model.embed_tokens.weight"] = np.random.randn(1000, n_embed).astype(np.float32)
    sd["model.norm.weight"] = np.ones(n_embed, dtype=np.float32)
    for i in range(n_layer):
        sd[f"model.layers.{i}.input_layernorm.weight"] = np.ones(n_embed, dtype=np.float32)
        sd[f"model.layers.{i}.self_attn.q_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
        sd[f"model.layers.{i}.self_attn.k_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
        sd[f"model.layers.{i}.self_attn.v_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
        sd[f"model.layers.{i}.self_attn.o_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = np.ones(n_embed, dtype=np.float32)
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = np.random.randn(4 * n_embed, n_embed).astype(np.float32)
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = np.random.randn(4 * n_embed, n_embed).astype(np.float32)
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = np.random.randn(n_embed, 4 * n_embed).astype(np.float32)

    result = convert_hf_to_slonet(sd, n_layer=2)

    assert "tok_emb.weight" in result
    assert "lm_head.weight" in result
    for i in range(2):
        assert f"blocks.{i}.ff.w1.weight" in result  # gate
        assert f"blocks.{i}.ff.w2.weight" in result  # down
        assert f"blocks.{i}.ff.w3.weight" in result  # up
        # No synthesized zeros — these should be real weights
        assert not np.allclose(result[f"blocks.{i}.ff.w3.weight"], 0.0)


def test_to_server_builds_guard_backed_server():
    """to_server() wraps the provider's model/tokenizer in a SloNetServer."""
    from unittest.mock import MagicMock

    from domains.inference.slonet_provider import SloNetChatProvider
    from domains.infrastructure.slonet_server import SloNetServer

    provider = SloNetChatProvider.__new__(SloNetChatProvider)
    provider._model = MagicMock()
    provider._tokenizer = MagicMock()
    provider._model_id = "test-slo"

    guard = MagicMock()
    server = provider.to_server(process_guard=guard)

    assert isinstance(server, SloNetServer)
    assert server._process_guard is guard
    assert server._model is provider._model
    assert server._tokenizer is provider._tokenizer
    assert server._model_id == "test-slo"

    no_guard = provider.to_server()
    assert no_guard._process_guard is None


class TestConvertHFToSloNet:
    def test_returns_dict(self):
        sd = _fake_gpt2_state_dict()
        result = convert_hf_to_slonet(sd, n_layer=2)
        assert isinstance(result, dict)

    def test_all_values_are_numpy(self):
        sd = _fake_gpt2_state_dict()
        result = convert_hf_to_slonet(sd, n_layer=2)
        for k, v in result.items():
            assert isinstance(v, np.ndarray), f"{k} is not ndarray"

    def test_final_norm_exists(self):
        sd = _fake_gpt2_state_dict()
        result = convert_hf_to_slonet(sd, n_layer=2)
        assert "norm.weight" in result

    def test_attn_bias_converted(self):
        sd = _fake_gpt2_state_dict()
        result = convert_hf_to_slonet(sd, n_layer=2)
        for i in range(2):
            assert f"blocks.{i}.attn.q_proj.bias" in result
            assert f"blocks.{i}.attn.k_proj.bias" in result
            assert f"blocks.{i}.attn.v_proj.bias" in result

    def test_ff_bias_converted(self):
        sd = _fake_gpt2_state_dict()
        result = convert_hf_to_slonet(sd, n_layer=2)
        for i in range(2):
            assert f"blocks.{i}.ff.w1.bias" in result
            assert f"blocks.{i}.ff.w2.bias" in result

    def test_single_layer(self):
        sd = _fake_gpt2_state_dict()
        result = convert_hf_to_slonet(sd, n_layer=1)
        assert "blocks.0.attn.q_proj.weight" in result
        assert "blocks.1.attn.q_proj.weight" not in result

    def test_large_n_embed(self):
        n_embed, n_layer = 256, 2
        sd = {}
        sd["wte.weight"] = np.random.randn(1000, n_embed).astype(np.float32)
        sd["wpe.weight"] = np.random.randn(512, n_embed).astype(np.float32)
        sd["ln_f.weight"] = np.ones(n_embed, dtype=np.float32)
        sd["ln_f.bias"] = np.zeros(n_embed, dtype=np.float32)
        for i in range(n_layer):
            sd[f"h.{i}.ln_1.weight"] = np.ones(n_embed, dtype=np.float32)
            sd[f"h.{i}.ln_1.bias"] = np.zeros(n_embed, dtype=np.float32)
            sd[f"h.{i}.attn.c_attn.weight"] = np.random.randn(n_embed, 3 * n_embed).astype(np.float32)
            sd[f"h.{i}.attn.c_attn.bias"] = np.random.randn(3 * n_embed).astype(np.float32)
            sd[f"h.{i}.attn.c_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
            sd[f"h.{i}.attn.c_proj.bias"] = np.zeros(n_embed, dtype=np.float32)
            sd[f"h.{i}.ln_2.weight"] = np.ones(n_embed, dtype=np.float32)
            sd[f"h.{i}.ln_2.bias"] = np.zeros(n_embed, dtype=np.float32)
            sd[f"h.{i}.mlp.c_fc.weight"] = np.random.randn(n_embed, 4 * n_embed).astype(np.float32)
            sd[f"h.{i}.mlp.c_fc.bias"] = np.zeros(4 * n_embed, dtype=np.float32)
            sd[f"h.{i}.mlp.c_proj.weight"] = np.random.randn(4 * n_embed, n_embed).astype(np.float32)
            sd[f"h.{i}.mlp.c_proj.bias"] = np.zeros(n_embed, dtype=np.float32)
        result = convert_hf_to_slonet(sd, n_layer=2)
        assert result["blocks.0.attn.q_proj.weight"].shape == (n_embed, n_embed)
