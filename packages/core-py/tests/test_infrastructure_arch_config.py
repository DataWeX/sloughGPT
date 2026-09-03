"""Tests for ArchConfig — architecture configuration for transformer inference."""
from __future__ import annotations

from domains.infrastructure.arch_config import (
    GPT2_WEIGHT_MAP,
    LLAMA_WEIGHT_MAP,
    ArchConfig,
    build_arch,
)


class TestArchConfig:
    def test_resolve_simple(self):
        ac = ArchConfig(name="test", norm="rms_norm", positional="rope", activation="swiglu", attention="mha")
        assert ac.resolve("embed.token") == "embed.token"

    def test_resolve_mapped(self):
        ac = ArchConfig(
            name="gpt2", norm="layer_norm", positional="absolute", activation="gelu", attention="mha",
            weight_map={"embed.token": "wte.weight"}
        )
        assert ac.resolve("embed.token") == "wte.weight"

    def test_resolve_layer_index(self):
        ac = ArchConfig(
            name="llama", norm="rms_norm", positional="rope", activation="swiglu", attention="gqa",
            weight_map={"layers.{i}.q.weight": "model.layers.{i}.self_attn.q_proj.weight"}
        )
        assert ac.resolve("layers.{i}.q.weight", layer_idx=3) == "model.layers.3.self_attn.q_proj.weight"

    def test_defaults(self):
        ac = ArchConfig(name="test", norm="rms_norm", positional="rope", activation="swiglu", attention="mha")
        assert ac.n_head == 0
        assert ac.transpose_weights is False
        assert ac.tied_weights is True


class TestBuildArch:
    def test_gpt2_detection(self):
        config = {"architectures": ["GPT2LMHeadModel"], "n_head": 12, "n_embd": 768, "n_layer": 12}
        weight_keys = {"wte.weight", "wpe.weight", "h.0.ln_1.weight"}
        arch = build_arch("gpt2", config, weight_keys)
        assert arch.norm == "layer_norm"
        assert arch.positional == "absolute"
        assert arch.transpose_weights is True
        assert arch.n_layers == 12

    def test_llama_detection(self):
        config = {"architectures": ["LlamaForCausalLM"], "num_attention_heads": 8, "hidden_size": 512, "num_hidden_layers": 6}
        weight_keys = {"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.input_layernorm.weight", "model.layers.0.mlp.gate_proj.weight"}
        arch = build_arch("llama", config, weight_keys)
        assert arch.norm == "rms_norm"
        assert arch.positional == "rope"
        assert arch.activation == "swiglu"
        assert arch.transpose_weights is False
        assert arch.n_layers == 6

    def test_unknown_fallback(self):
        config = {"architectures": ["UnknownModel"]}
        weight_keys = {"some.weight"}
        arch = build_arch("unknown", config, weight_keys)
        assert arch.norm == "layer_norm"
        assert arch.positional == "absolute"
