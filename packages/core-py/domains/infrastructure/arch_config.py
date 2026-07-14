"""
Architecture config for NumPy transformer inference.

The ONLY thing that changes per model type:
  - Weight map: canonical name → actual tensor name
  - Feature flags: norm_type, positional, attention, activation

New arch = new ArchConfig instance. Zero math changes.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger("man.infrastructure.arch_config")


@dataclass
class ArchConfig:
    """Architecture definition as data.

    weight_map: maps canonical names → actual tensor names.
        Use {i} for layer index. Example: "layers.{i}.q.weight" → "model.layers.{i}.self_attn.q_proj.weight"

    Feature flags control which math ops the generic forward pass uses.
    """

    name: str
    norm: str              # "layer_norm" | "rms_norm"
    positional: str        # "absolute" | "rope"
    activation: str        # "gelu" | "swiglu"
    attention: str         # "mha" | "gqa"
    weight_map: Dict[str, str] = field(default_factory=dict)
    transpose_weights: bool = False  # True if HF weights stored as (in, out) but SloTransformer expects (out, in) — need .T

    # Derived from config.json at load time (not hardcoded)
    n_head: int = 0
    n_kv_head: int = 0
    n_embed: int = 0
    n_layers: int = 0
    head_dim: int = 0
    rope_base: float = 10000.0
    tied_weights: bool = True  # lm_head shares embed weights

    def resolve(self, canonical: str, layer_idx: int = 0) -> str:
        """Map canonical name → actual weight tensor name."""
        key = canonical.replace("{i}", str(layer_idx))
        return self.weight_map.get(key, key)


# ── Weight maps (canonical → actual) ─────────────────────────────────────────
# Canonical names: embed.token, embed.pos, layers.{i}.attn_norm,
#   layers.{i}.qkv, layers.{i}.q, layers.{i}.k, layers.{i}.v,
#   layers.{i}.o_proj, layers.{i}.ff_norm, layers.{i}.ffn.{gate,up,down},
#   final_norm, lm_head

GPT2_WEIGHT_MAP = {
    "embed.token": "wte.weight",
    "embed.pos": "wpe.weight",
    "layers.{i}.attn_norm.weight": "h.{i}.ln_1.weight",
    "layers.{i}.attn_norm.bias": "h.{i}.ln_1.bias",
    "layers.{i}.qkv.weight": "h.{i}.attn.c_attn.weight",
    "layers.{i}.qkv.bias": "h.{i}.attn.c_attn.bias",
    "layers.{i}.o_proj.weight": "h.{i}.attn.c_proj.weight",
    "layers.{i}.o_proj.bias": "h.{i}.attn.c_proj.bias",
    "layers.{i}.ff_norm.weight": "h.{i}.ln_2.weight",
    "layers.{i}.ff_norm.bias": "h.{i}.ln_2.bias",
    "layers.{i}.ffn.up.weight": "h.{i}.mlp.c_fc.weight",
    "layers.{i}.ffn.up.bias": "h.{i}.mlp.c_fc.bias",
    "layers.{i}.ffn.down.weight": "h.{i}.mlp.c_proj.weight",
    "layers.{i}.ffn.down.bias": "h.{i}.mlp.c_proj.bias",
    "final_norm.weight": "ln_f.weight",
    "final_norm.bias": "ln_f.bias",
}

LLAMA_WEIGHT_MAP = {
    "embed.token": "model.embed_tokens.weight",
    "layers.{i}.attn_norm.weight": "model.layers.{i}.input_layernorm.weight",
    "layers.{i}.q.weight": "model.layers.{i}.self_attn.q_proj.weight",
    "layers.{i}.k.weight": "model.layers.{i}.self_attn.k_proj.weight",
    "layers.{i}.v.weight": "model.layers.{i}.self_attn.v_proj.weight",
    "layers.{i}.q.bias": "model.layers.{i}.self_attn.q_proj.bias",
    "layers.{i}.k.bias": "model.layers.{i}.self_attn.k_proj.bias",
    "layers.{i}.v.bias": "model.layers.{i}.self_attn.v_proj.bias",
    "layers.{i}.o_proj.weight": "model.layers.{i}.self_attn.o_proj.weight",
    "layers.{i}.ff_norm.weight": "model.layers.{i}.post_attention_layernorm.weight",
    "layers.{i}.ffn.gate.weight": "model.layers.{i}.mlp.gate_proj.weight",
    "layers.{i}.ffn.up.weight": "model.layers.{i}.mlp.up_proj.weight",
    "layers.{i}.ffn.down.weight": "model.layers.{i}.mlp.down_proj.weight",
    "final_norm.weight": "model.norm.weight",
}


def build_arch(name: str, config: dict, weight_keys: set) -> ArchConfig:
    """Build ArchConfig from HuggingFace config.json + actual weight keys.

    Detects the architecture from config, then selects the right weight map.
    The weight map is the ONLY architecture-specific data.
    """
    arch_name = config.get("architectures", ["unknown"])[0]

    # Detect features from config
    n_head = config.get("n_head") or config.get("num_attention_heads", 12)
    n_kv_head = config.get("num_key_value_heads", n_head)
    n_embed = config.get("n_embd") or config.get("hidden_size", 768)
    n_layers = config.get("n_layer") or config.get("num_hidden_layers", 12)
    head_dim = n_embed // n_head
    rope_base = config.get("rope_theta", 10000.0)

    # Select weight map based on which keys exist in the checkpoint
    if "wte.weight" in weight_keys:
        # GPT-2 style — safetensors stores weights as (in, out), no transpose needed
        wm = GPT2_WEIGHT_MAP
        norm, positional, activation, attention = "layer_norm", "absolute", "gelu", "mha"
        transpose = False  # GPT-2 safetensors is already (in, out), forward does h @ W directly
    elif "model.embed_tokens.weight" in weight_keys and "model.layers.0.self_attn.q_proj.weight" in weight_keys:
        # LLaMA/Qwen/Mistral style — detect sub-features
        norm = "rms_norm" if "model.layers.0.input_layernorm.weight" in weight_keys else "layer_norm"
        positional = "rope"
        has_gate = "model.layers.0.mlp.gate_proj.weight" in weight_keys
        activation = "swiglu" if has_gate else "gelu"
        attention = "gqa" if n_kv_head < n_head else "mha"
        wm = LLAMA_WEIGHT_MAP
        transpose = False  # LLaMA safetensors is already (out, in), same as SloLinear
    else:
        # Unknown — try GPT-2 as fallback
        wm = GPT2_WEIGHT_MAP
        norm, positional, activation, attention = "layer_norm", "absolute", "gelu", "mha"
        transpose = False  # GPT-2 safetensors is already (in, out), forward does h @ W directly

    arch = ArchConfig(
        name=name,
        norm=norm,
        positional=positional,
        activation=activation,
        attention=attention,
        weight_map=wm,
        transpose_weights=transpose,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embed=n_embed,
        n_layers=n_layers,
        head_dim=head_dim,
        rope_base=rope_base,
    )

    logger.info("ArchConfig: %s (norm=%s, pos=%s, act=%s, attn=%s, layers=%d, heads=%d/%d)",
                name, norm, positional, activation, attention, n_layers, n_head, n_kv_head)
    return arch
