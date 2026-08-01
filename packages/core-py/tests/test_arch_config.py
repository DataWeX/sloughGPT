"""Tests for ArchConfig and architecture detection (arch_config.py)."""

import pytest

from domains.infrastructure.arch_config import (
    GPT2_WEIGHT_MAP,
    LLAMA_WEIGHT_MAP,
    ArchConfig,
    build_arch,
)


class TestArchConfigDefaults:
    def test_requires_name(self):
        with pytest.raises(TypeError):
            ArchConfig()

    def test_defaults(self):
        arch = ArchConfig(name="x", norm="rms_norm", positional="rope",
                          activation="swiglu", attention="gqa")
        assert arch.weight_map == {}
        assert arch.transpose_weights is False
        assert arch.n_head == 0
        assert arch.n_kv_head == 0
        assert arch.n_embed == 0
        assert arch.n_layers == 0
        assert arch.head_dim == 0
        assert arch.rope_base == 10000.0
        assert arch.tied_weights is True

    def test_constructor_args(self):
        arch = ArchConfig(
            name="x", norm="layer_norm", positional="absolute",
            activation="gelu", attention="mha",
            weight_map={"a": "b"}, transpose_weights=True,
            n_head=4, n_kv_head=4, n_embed=16, n_layers=2, head_dim=4,
        )
        assert arch.n_head == 4
        assert arch.head_dim == 4
        assert arch.transpose_weights is True


class TestResolve:
    def test_substitutes_layer(self):
        arch = ArchConfig(name="x", norm="n", positional="p",
                          activation="a", attention="m",
                          weight_map={"layers.{i}.q.weight": "model.layers.{i}.self_attn.q_proj.weight"})
        assert arch.resolve("layers.{i}.q.weight", layer_idx=3) == "model.layers.3.self_attn.q_proj.weight"

    def test_no_placeholder_key(self):
        arch = ArchConfig(name="x", norm="n", positional="p",
                          activation="a", attention="m",
                          weight_map={"embed.token": "wte.weight"})
        assert arch.resolve("embed.token") == "wte.weight"

    def test_unmapped_key_passthrough(self):
        arch = ArchConfig(name="x", norm="n", positional="p",
                          activation="a", attention="m")
        assert arch.resolve("something.else") == "something.else"

    def test_default_layer_zero(self):
        arch = ArchConfig(name="x", norm="n", positional="p",
                          activation="a", attention="m",
                          weight_map={"layers.{i}.q.weight": "q.{i}"})
        assert arch.resolve("layers.{i}.q.weight") == "q.0"


class TestWeightMaps:
    def test_gpt2_map_shape(self):
        assert GPT2_WEIGHT_MAP["embed.token"] == "wte.weight"
        assert GPT2_WEIGHT_MAP["embed.pos"] == "wpe.weight"
        assert GPT2_WEIGHT_MAP["layers.{i}.qkv.weight"] == "h.{i}.attn.c_attn.weight"
        assert GPT2_WEIGHT_MAP["final_norm.weight"] == "ln_f.weight"

    def test_llama_map_shape(self):
        assert LLAMA_WEIGHT_MAP["embed.token"] == "model.embed_tokens.weight"
        assert LLAMA_WEIGHT_MAP["layers.{i}.ffn.gate.weight"] == "model.layers.{i}.mlp.gate_proj.weight"
        assert LLAMA_WEIGHT_MAP["final_norm.weight"] == "model.norm.weight"


class TestBuildArch:
    def test_gpt2_detection(self):
        config = {"architectures": ["GPT2LMHeadModel"], "n_head": 12, "n_embd": 768, "n_layer": 12}
        keys = {"wte.weight", "wpe.weight", "h.0.ln_1.weight", "h.0.attn.c_attn.weight"}
        arch = build_arch("gpt2", config, keys)
        assert arch.norm == "layer_norm"
        assert arch.positional == "absolute"
        assert arch.activation == "gelu"
        assert arch.attention == "mha"
        assert arch.transpose_weights is True
        assert arch.n_head == 12
        assert arch.n_embed == 768
        assert arch.n_layers == 12
        assert arch.head_dim == 64

    def test_gpt2_numeric_defaults(self):
        config = {"architectures": ["GPT2LMHeadModel"]}
        arch = build_arch("gpt2", config, {"wte.weight"})
        assert arch.n_head == 12
        assert arch.n_embed == 768
        assert arch.n_layers == 12
        assert arch.head_dim == 64

    def test_llama_detection_rms_swiglu_gqa(self):
        config = {"architectures": ["LlamaForCausalLM"], "num_attention_heads": 8,
                  "num_key_value_heads": 4, "hidden_size": 512, "num_hidden_layers": 2}
        keys = {"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.input_layernorm.weight", "model.layers.0.mlp.gate_proj.weight"}
        arch = build_arch("llama", config, keys)
        assert arch.norm == "rms_norm"
        assert arch.positional == "rope"
        assert arch.activation == "swiglu"
        assert arch.attention == "gqa"
        assert arch.transpose_weights is False
        assert arch.n_head == 8
        assert arch.n_kv_head == 4
        assert arch.head_dim == 64
        assert arch.n_layers == 2

    def test_llama_without_gate_uses_gelu(self):
        config = {"architectures": ["MistralForCausalLM"], "num_attention_heads": 4,
                  "num_key_value_heads": 4, "hidden_size": 256, "num_hidden_layers": 1}
        keys = {"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.input_layernorm.weight"}
        arch = build_arch("mistral", config, keys)
        assert arch.activation == "gelu"
        assert arch.attention == "mha"  # kv heads == q heads

    def test_llama_layer_norm_when_no_rms_key(self):
        config = {"architectures": ["LlamaForCausalLM"], "num_attention_heads": 2,
                  "num_key_value_heads": 2, "hidden_size": 64, "num_hidden_layers": 1}
        keys = {"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"}
        arch = build_arch("qwen", config, keys)
        assert arch.norm == "layer_norm"

    def test_rope_base_from_config(self):
        config = {"architectures": ["LlamaForCausalLM"], "num_attention_heads": 2,
                  "num_key_value_heads": 2, "hidden_size": 64, "num_hidden_layers": 1,
                  "rope_theta": 1000000.0}
        keys = {"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"}
        arch = build_arch("qwen", config, keys)
        assert arch.rope_base == 1000000.0

    def test_unknown_arch_falls_back_to_gpt2(self):
        config = {"architectures": ["SomeNewModel"], "n_head": 4, "n_embd": 32, "n_layer": 1}
        arch = build_arch("new", config, set())
        assert arch.norm == "layer_norm"
        assert arch.positional == "absolute"
        assert arch.attention == "mha"
        assert arch.transpose_weights is False

    def test_arch_name_preserved(self):
        config = {"architectures": ["GPT2LMHeadModel"]}
        arch = build_arch("my-custom-name", config, {"wte.weight"})
        assert arch.name == "my-custom-name"
