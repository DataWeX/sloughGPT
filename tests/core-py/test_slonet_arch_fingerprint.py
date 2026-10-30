"""Architecture fingerprint: what makes SloNet the same as — and different from — other models.

Pins three contracts so "is our architecture different?" is answered by a
green/red test instead of vibes:

1. SAME recipe (by design — ``SloTransformer`` is a drop-in replacement):
   its feature flags, derived mechanically from a real instance's weights via
   ``build_arch``, match the llama/Qwen decoder-only family (pre-norm RMSNorm,
   RoPE, gated-GLU FFN, GQA).
2. DISCRIMINANT validity: a GPT-2 fingerprint differs in every flag, and our
   native tensor naming resolves differently from HF llama naming — the
   comparison can actually tell architectures apart.
3. DIFFERENCE contracts: the substrate is pure NumPy (AST import audit — no
   torch/transformers anywhere in slonet.py), the layer zoo goes beyond
   transformers (LSTM/Conv2D/BN), and every model carries soul metadata that
   no HF config has.

If a future change makes SloNet silently *more* like HF (or breaks the
drop-in compatibility it exists to provide), this file goes red.
"""

from __future__ import annotations

import ast
from pathlib import Path

import domain.training._internal.slonet as slonet_mod
from domain.infrastructure._internal.arch_config import (
    ArchConfig,
    build_arch,
)
from domain.training._internal.slonet import (
    SloAdapterLayer,
    SloBatchNorm2D,
    SloConv2D,
    SloLSTM,
    SloTransformer,
    SloTransformerBlock,
)

# Bound to the constructor below so config and instance cannot drift apart.
ROPE_BASE = 10000.0

# Qwen2.5-0.5B-class reference (llama-family layout in HF naming).
QWEN_REF_CONFIG = {
    "hidden_size": 896,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
}
LLAMA_FAMILY_KEYS = {
    "model.embed_tokens.weight",
    # Detection reads weight KEYS, not config — a real Qwen checkpoint has
    # input_layernorm, and without it build_arch classifies us as layer_norm.
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.norm.weight",
}

# GPT-2-class reference (absolute positions, layer norm, plain MHA).
GPT2_REF_CONFIG = {"n_embd": 768, "n_layer": 12, "n_head": 12}
GPT2_KEYS = {"wte.weight", "wpe.weight", "h.0.attn.c_attn.weight"}


def _tiny_native() -> SloTransformer:
    """A real SloTransformer small enough to fingerprint in a unit test."""
    return SloTransformer(
        vocab_size=64,
        n_embed=32,
        n_layer=2,
        n_head=4,
        n_kv_head=2,  # kv < heads => GQA, like Qwen
        block_size=32,
        max_seq_len=64,
        dropout=0,
        rope_base=ROPE_BASE,
        _lazy=True,
    )


def _native_fingerprint(model: SloTransformer) -> ArchConfig:
    """Derive our ArchConfig from an actual instance: real weight keys,
    real norm eps, constructor-bound rope base — nothing hardcoded twice."""
    md = model.metadata
    blocks = [layer for layer in model.layers if isinstance(layer, SloTransformerBlock)]
    assert blocks, "SloTransformer must contain at least one SloTransformerBlock"
    eps = blocks[0].attn_norm.eps
    config = {
        "hidden_size": md["n_embed"],
        "num_hidden_layers": md["n_layer"],
        "num_attention_heads": md["n_head"],
        "num_key_value_heads": md["n_kv_head"],
        "rms_norm_eps": eps,
        "rope_theta": ROPE_BASE,
    }
    return build_arch("slonet-native", config, set(model.state_dict()))


class TestSameFamilyRecipe:
    """The drop-in contract: same decoder-only recipe as the llama/Qwen family."""

    def test_native_flags_match_llama_family(self):
        ours = _native_fingerprint(_tiny_native())
        ref = build_arch("qwen2.5-0.5b-ref", QWEN_REF_CONFIG, LLAMA_FAMILY_KEYS)

        # Four feature flags — the entire per-architecture math selection.
        assert ours.norm == ref.norm == "rms_norm"
        assert ours.positional == ref.positional == "rope"
        assert ours.activation == ref.activation == "swiglu"
        assert ours.attention == ref.attention == "gqa"

    def test_native_shapes_follow_the_shared_recipe(self):
        model = _tiny_native()
        ours = _native_fingerprint(model)

        assert ours.n_head == 4 and ours.n_kv_head == 2
        assert ours.head_dim == ours.n_embed // ours.n_head
        assert ours.rope_base == ROPE_BASE
        assert ours.transpose_weights is False  # same orientation as HF llama
        # Tied weights are the family default, and SloTransformer keeps them.
        assert ours.tied_weights is model.tie_weights is True


class TestFingerprintDiscriminates:
    """A fingerprint that matches everything proves nothing — show it separates."""

    def test_gpt2_differs_in_every_flag(self):
        ours = _native_fingerprint(_tiny_native())
        gpt2 = build_arch("gpt2-ref", GPT2_REF_CONFIG, GPT2_KEYS)

        assert ours.norm != gpt2.norm  # rms_norm vs layer_norm
        assert ours.positional != gpt2.positional  # rope vs absolute
        assert ours.activation != gpt2.activation  # gated GLU vs gelu
        assert ours.attention != gpt2.attention  # gqa vs mha

    def test_native_tensor_naming_differs_from_hf(self):
        """Same math, different artifact: our tensor names are not HF names."""
        ours = _native_fingerprint(_tiny_native())
        hf = build_arch("qwen2.5-0.5b-ref", QWEN_REF_CONFIG, LLAMA_FAMILY_KEYS)
        canonical = "layers.{i}.ffn.gate.weight"

        native_name = ours.resolve(canonical, layer_idx=1)
        hf_name = hf.resolve(canonical, layer_idx=1)

        assert native_name == "blocks.1.ff.w1.weight"
        assert hf_name == "model.layers.1.mlp.gate_proj.weight"
        assert native_name != hf_name

    def test_orientation_contract_per_substrate(self):
        """Transpose need is a property of the artifact, not the recipe."""
        ours = _native_fingerprint(_tiny_native())
        gpt2 = build_arch("gpt2-ref", GPT2_REF_CONFIG, GPT2_KEYS)

        assert ours.transpose_weights is False  # native .slnc stores (out, in)
        assert gpt2.transpose_weights is True  # GPT-2 Conv1D stores (in, out)


class TestDifferenceContracts:
    """The parts of SloNet that no HF-hosted model has."""

    def test_substrate_is_pure_numpy(self):
        """AST import audit of slonet.py — 'no torch' must stay mechanically true."""
        tree = ast.parse(Path(slonet_mod.__file__).read_text(encoding="utf-8"))
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".")[0])

        assert "numpy" in modules
        for foreign in ("torch", "transformers", "tensorflow", "jax"):
            assert foreign not in modules, f"slonet.py must stay framework-free, found {foreign}"

    def test_layer_zoo_goes_beyond_transformers(self):
        """HF LLM stacks ship transformer layers only; SloNet is a general NN library."""
        for cls in (SloLSTM, SloConv2D, SloBatchNorm2D, SloAdapterLayer):
            assert cls.__module__ == slonet_mod.__name__

    def test_every_model_is_a_soul(self):
        """Soul metadata lives on the model object — no HF config.json analog."""
        model = _tiny_native()

        assert model.soul_name == "SloTransformer"
        assert model.lineage == "soultransformer"
        assert set(model.soul_traits) == {"warmth", "creativity", "curiosity", "confidence"}
        assert model.metadata["model_type"] == "sloughgpt"
        # The fingerprint itself must carry the identity fields, too.
        assert _native_fingerprint(model).name == "slonet-native"
