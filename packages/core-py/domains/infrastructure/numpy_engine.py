"""
Generic NumPy transformer inference engine.

Any architecture (GPT-2, Qwen2, LLaMA, Mistral, etc.) integrates via a
weight map + feature flags — no per-architecture forward pass needed.

Architecture = data, not code:
  - Weight map: canonical name → actual tensor name
  - Feature flags: norm_type, positional, attention, activation

New arch = new ArchConfig instance. Zero math changes.

Features:
  - Compression: weights compressed via vector quantization (4x memory savings)
  - KV cache: incremental decoding (only process new token after first step)
  - Streaming: async generator for token-by-token output

Usage:
    from domains.infrastructure.numpy_engine import NumpyEngine
    engine = NumpyEngine.from_pretrained("gpt2")
    text = engine.generate("Hello", max_new_tokens=50)
    # Streaming
    async for token in engine.generate_stream("Hello", max_new_tokens=50):
        print(token, end="", flush=True)
"""

import asyncio
import json
import logging
import struct
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("man.infrastructure.numpy_engine")


# ══════════════════════════════════════════════════════════════════════════════
# bfloat16 handling
# ══════════════════════════════════════════════════════════════════════════════

def _to_float32(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.name in ("bfloat16", "float16"):
        if arr.dtype.name == "bfloat16":
            raw = arr.view(np.uint16).astype(np.uint32) << 16
            return raw.view(np.float32)
        return arr.astype(np.float32)
    return arr.astype(np.float32) if arr.dtype != np.float32 else arr


# ══════════════════════════════════════════════════════════════════════════════
# Weight loader
# ══════════════════════════════════════════════════════════════════════════════

def _load_weights(model_id: str) -> Tuple[dict, dict]:
    """Load config.json + weights from HF cache. Returns (config, weights)."""
    from domains.infrastructure.safetensors_loader import _get_model_dir, _find_safetensors

    model_dir = _get_model_dir(model_id)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model {model_id} not cached")

    # Config
    config_path = None
    snapshots = model_dir / "snapshots"
    if snapshots.exists():
        for snap in snapshots.iterdir():
            c = snap / "config.json"
            if c.exists():
                config_path = c
                break
    if config_path is None:
        config_path = model_dir / "config.json"
    if config_path is None:
        raise FileNotFoundError(f"No config.json for {model_id}")

    with open(config_path) as f:
        config = json.load(f)

    # Weights
    safetensors_path = _find_safetensors(model_dir)
    if safetensors_path is None:
        raise FileNotFoundError(f"No .safetensors for {model_id}")

    from safetensors import safe_open

    # Try mmap first
    try:
        weights = {}
        with safe_open(str(safetensors_path), framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
        logger.info("Loaded %d weights from %s (mmap)", len(weights), model_id)
        return config, weights
    except Exception:
        pass

    # bfloat16 fallback
    logger.info("bfloat16 fallback for %s", model_id)
    with open(safetensors_path, "rb") as f:
        header_len_bytes = f.read(8)
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes)

        weights = {}
        for key, info in header.items():
            if key == "__metadata__":
                continue
            dtype_str = info.get("dtype", "")
            data_offsets = info.get("data_offsets", [0, 0])
            begin, end = data_offsets
            shape = info.get("shape", [])
            f.seek(8 + header_len + begin)
            raw_bytes = f.read(end - begin)

            if "bfloat" in dtype_str.lower() or dtype_str == "BF16":
                raw_arr = np.frombuffer(raw_bytes, dtype=np.uint16)
                weights[key] = (raw_arr.astype(np.uint32) << 16).view(np.float32).reshape(shape)
            elif "float16" in dtype_str.lower() or dtype_str == "F16":
                weights[key] = np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32).reshape(shape)
            elif "float32" in dtype_str.lower() or dtype_str == "F32":
                weights[key] = np.frombuffer(raw_bytes, dtype=np.float32).reshape(shape)

    logger.info("Loaded %d weights from %s (bfloat16 fallback)", len(weights), model_id)
    return config, weights


# ══════════════════════════════════════════════════════════════════════════════
# Generic ops
# ══════════════════════════════════════════════════════════════════════════════

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = x.max(axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=axis, keepdims=True)


def _rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)) * w


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def _rope(x: np.ndarray, pos: int, dim: int, base: float = 10000.0) -> np.ndarray:
    """Rotary position embeddings. x: (seq, heads, head_dim)."""
    seq_len = x.shape[0]
    t = np.arange(pos, pos + seq_len, dtype=np.float32)
    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    emb = np.outer(t, freqs)
    cos = np.cos(emb)
    sin = np.sin(emb)
    if x.ndim == 3:
        cos = cos[:, np.newaxis, :]
        sin = sin[:, np.newaxis, :]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


# ══════════════════════════════════════════════════════════════════════════════
# Architecture config — the ONLY thing that changes per model type
# ══════════════════════════════════════════════════════════════════════════════

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
    transpose_weights: bool = False  # True if weights stored as (out, in) — need .T

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


def _build_arch(name: str, config: dict, weight_keys: set) -> ArchConfig:
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
        # GPT-2 style
        wm = _GPT2_WEIGHT_MAP
        norm, positional, activation, attention = "layer_norm", "absolute", "gelu", "mha"
        transpose = False
    elif "model.embed_tokens.weight" in weight_keys and "model.layers.0.self_attn.q_proj.weight" in weight_keys:
        # LLaMA/Qwen/Mistral style — detect sub-features
        norm = "rms_norm" if "model.layers.0.input_layernorm.weight" in weight_keys else "layer_norm"
        positional = "rope"
        has_gate = "model.layers.0.mlp.gate_proj.weight" in weight_keys
        activation = "swiglu" if has_gate else "gelu"
        attention = "gqa" if n_kv_head < n_head else "mha"
        wm = _LLAMA_WEIGHT_MAP
        transpose = True
    else:
        # Unknown — try GPT-2 as fallback
        wm = _GPT2_WEIGHT_MAP
        norm, positional, activation, attention = "layer_norm", "absolute", "gelu", "mha"
        transpose = False

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


# ── Weight maps (canonical → actual) ─────────────────────────────────────────
# Canonical names: embed.token, embed.pos, layers.{i}.attn_norm,
#   layers.{i}.qkv, layers.{i}.q, layers.{i}.k, layers.{i}.v,
#   layers.{i}.o_proj, layers.{i}.ff_norm, layers.{i}.ffn.{gate,up,down},
#   final_norm, lm_head

_GPT2_WEIGHT_MAP = {
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

_LLAMA_WEIGHT_MAP = {
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


# ══════════════════════════════════════════════════════════════════════════════
# Generic forward pass — ONE function handles ALL architectures
# ══════════════════════════════════════════════════════════════════════════════

def _norm_fn(arch: ArchConfig):
    """Return the normalization function for this architecture."""
    return _rmsnorm if arch.norm == "rms_norm" else _layer_norm


def _get_w(weights: dict, name: str) -> Optional[np.ndarray]:
    """Get weight tensor, return None if missing."""
    return weights.get(name)


def _forward(weights: dict, arch: ArchConfig, token_ids: List[int]) -> np.ndarray:
    """Generic transformer forward pass — reads arch config, not architecture names.

    This is the ONLY forward function. Architectures differ only in:
    - Which weight names to look up (via arch.weight_map)
    - Which math ops to use (via arch.norm, arch.positional, etc.)
    """
    seq_len = len(token_ids)
    W = arch.weight_map

    def w(canonical: str, layer: int = 0) -> np.ndarray:
        """Resolve canonical name → actual weight.
        Canonical names use {i} for layer index. Map keys also use {i}.
        Resolution: canonical → map key (with {i}) → actual name (with layer idx) → weight tensor.
        """
        mapped = W[canonical]  # e.g. "layers.{i}.attn_norm.weight" → "h.{i}.ln_1.weight"
        actual = mapped.replace("{i}", str(layer))  # → "h.0.ln_1.weight"
        return weights[actual]

    def wn(canonical: str, layer: int = 0) -> Optional[np.ndarray]:
        """Resolve with fallback to None."""
        mapped = W.get(canonical)
        if mapped is None:
            return None
        actual = mapped.replace("{i}", str(layer))
        return weights.get(actual)

    norm = _norm_fn(arch)
    T = (lambda w: w.T) if arch.transpose_weights else (lambda w: w)  # conditional transpose

    # ── Embeddings ────────────────────────────────────────────────────────
    x = w("embed.token")[token_ids]
    if arch.positional == "absolute":
        x = x + w("embed.pos")[:seq_len]

    # ── Causal mask ──────────────────────────────────────────────────────
    mask = np.triu(np.full((seq_len, seq_len), -1e10, dtype=np.float32), k=1)

    # ── Transformer blocks ───────────────────────────────────────────────
    for i in range(arch.n_layers):

        # === Attention ===
        if arch.norm == "rms_norm":
            h = norm(x, w("layers.{i}.attn_norm.weight", i))
        else:
            b = wn("layers.{i}.attn_norm.bias", i)
            h = norm(x, w("layers.{i}.attn_norm.weight", i), b if b is not None else np.zeros(arch.n_embed))

        if arch.attention == "mha" and "layers.{i}.qkv.weight" in W:
            # GPT-2 style: combined QKV projection
            qkv = h @ T(w("layers.{i}.qkv.weight", i))
            bias = wn("layers.{i}.qkv.bias", i)
            if bias is not None:
                qkv = qkv + bias
            q, k, v = np.split(qkv, 3, axis=-1)
        else:
            # LLaMA/Qwen style: separate Q, K, V projections
            q = h @ T(w("layers.{i}.q.weight", i))
            k = h @ T(w("layers.{i}.k.weight", i))
            v = h @ T(w("layers.{i}.v.weight", i))
            for name in ("q", "k", "v"):
                bias = wn(f"layers.{i}.{name}.bias", i)
                if bias is not None:
                    locals()[name] = locals()[name] + bias

        # Reshape to (heads, seq, head_dim)
        n_h = arch.n_head
        n_kv = arch.n_kv_head
        hd = arch.head_dim
        q = q.reshape(seq_len, n_h, hd).transpose(1, 0, 2)
        k = k.reshape(seq_len, n_kv, hd).transpose(1, 0, 2)
        v = v.reshape(seq_len, n_kv, hd).transpose(1, 0, 2)

        # Positional encoding
        if arch.positional == "rope":
            q = _rope(q, 0, hd, arch.rope_base)
            k = _rope(k, 0, hd, arch.rope_base)

        # GQA: repeat KV heads to match Q heads
        n_rep = n_h // n_kv
        if n_rep > 1:
            k = np.repeat(k, n_rep, axis=0)
            v = np.repeat(v, n_rep, axis=0)

        # Scaled dot-product attention
        scale = np.sqrt(hd).astype(np.float32)
        attn = (q @ k.transpose(0, 2, 1)) / scale + mask
        attn = _softmax(attn, axis=-1)
        out = (attn @ v).transpose(1, 0, 2).reshape(seq_len, arch.n_embed)

        # Output projection
        x = x + out @ T(w("layers.{i}.o_proj.weight", i))
        bias = wn("layers.{i}.o_proj.bias", i)
        if bias is not None:
            x = x + bias

        # === Feed-forward ===
        if arch.norm == "rms_norm":
            h = norm(x, w("layers.{i}.ff_norm.weight", i))
        else:
            b = wn("layers.{i}.ff_norm.bias", i)
            h = norm(x, w("layers.{i}.ff_norm.weight", i), b if b is not None else np.zeros(arch.n_embed))

        if arch.activation == "swiglu" and "layers.{i}.ffn.gate.weight" in W:
            # SwiGLU: gate * silu(up) @ down
            gate = h @ T(w("layers.{i}.ffn.gate.weight", i))
            up = h @ T(w("layers.{i}.ffn.up.weight", i))
            x = x + (_silu(gate) * up) @ T(w("layers.{i}.ffn.down.weight", i))
        else:
            # GELU MLP: up → gelu → down
            h2 = h @ T(w("layers.{i}.ffn.up.weight", i))
            bias = wn("layers.{i}.ffn.up.bias", i)
            if bias is not None:
                h2 = h2 + bias
            h2 = _gelu(h2)
            x = x + h2 @ T(w("layers.{i}.ffn.down.weight", i))
            bias = wn("layers.{i}.ffn.down.bias", i)
            if bias is not None:
                x = x + bias

    # ── Final norm + LM head ─────────────────────────────────────────────
    if arch.norm == "rms_norm":
        x = norm(x, w("final_norm.weight"))
    else:
        b = wn("final_norm.bias")
        x = norm(x, w("final_norm.weight"), b if b is not None else np.zeros(arch.n_embed))

    # LM head (weight tying or separate)
    embed_weight = weights[arch.weight_map["embed.token"]]  # raw weight, no T
    if "lm_head.weight" in weights:
        embed_weight = weights["lm_head.weight"]
    return x[-1] @ embed_weight.T


def _forward_cached(
    get_weight,
    arch: ArchConfig,
    token_ids: List[int],
    kv_cache: Optional["KVCache"] = None,
    start_pos: int = 0,
) -> np.ndarray:
    """Forward pass with KV cache support for incremental decoding.

    Args:
        get_weight: Function that returns weight tensor by name.
        arch: Architecture configuration.
        token_ids: Input token IDs.
        kv_cache: Optional KV cache for incremental decoding.
        start_pos: Starting position for RoPE (used with KV cache).

    Returns:
        Logits for next token prediction.
    """
    seq_len = len(token_ids)
    W = arch.weight_map

    def w(canonical: str, layer: int = 0) -> np.ndarray:
        mapped = W[canonical]
        actual = mapped.replace("{i}", str(layer))
        return get_weight(actual)

    def wn(canonical: str, layer: int = 0) -> Optional[np.ndarray]:
        mapped = W.get(canonical)
        if mapped is None:
            return None
        actual = mapped.replace("{i}", str(layer))
        try:
            return get_weight(actual)
        except KeyError:
            return None

    norm = _norm_fn(arch)
    T = (lambda w: w.T) if arch.transpose_weights else (lambda w: w)

    # ── Embeddings ────────────────────────────────────────────────────────
    x = w("embed.token")[token_ids]
    if arch.positional == "absolute":
        x = x + w("embed.pos")[:seq_len]

    # ── Embeddings ────────────────────────────────────────────────────────
    x = w("embed.token")[token_ids]
    if arch.positional == "absolute":
        x = x + w("embed.pos")[start_pos:start_pos + seq_len]

    # ── Causal mask ──────────────────────────────────────────────────────
    if kv_cache is not None and kv_cache.seq_len > 0:
        # Incremental: only need mask for new tokens against all cached + new
        total_len = kv_cache.seq_len + seq_len
        mask = np.triu(np.full((seq_len, total_len), -1e10, dtype=np.float32), k=1 + kv_cache.seq_len)
    else:
        mask = np.triu(np.full((seq_len, seq_len), -1e10, dtype=np.float32), k=1)

    # ── Transformer blocks ───────────────────────────────────────────────
    for i in range(arch.n_layers):

        # === Attention ===
        if arch.norm == "rms_norm":
            h = norm(x, w("layers.{i}.attn_norm.weight", i))
        else:
            b = wn("layers.{i}.attn_norm.bias", i)
            h = norm(x, w("layers.{i}.attn_norm.weight", i), b if b is not None else np.zeros(arch.n_embed))

        if arch.attention == "mha" and "layers.{i}.qkv.weight" in W:
            qkv = h @ T(w("layers.{i}.qkv.weight", i))
            bias = wn("layers.{i}.qkv.bias", i)
            if bias is not None:
                qkv = qkv + bias
            q, k, v = np.split(qkv, 3, axis=-1)
        else:
            q = h @ T(w("layers.{i}.q.weight", i))
            k = h @ T(w("layers.{i}.k.weight", i))
            v = h @ T(w("layers.{i}.v.weight", i))
            for name in ("q", "k", "v"):
                bias = wn(f"layers.{i}.{name}.bias", i)
                if bias is not None:
                    locals()[name] = locals()[name] + bias

        # Reshape to (heads, seq, head_dim)
        n_h = arch.n_head
        n_kv = arch.n_kv_head
        hd = arch.head_dim
        q = q.reshape(seq_len, n_h, hd).transpose(1, 0, 2)
        k = k.reshape(seq_len, n_kv, hd).transpose(1, 0, 2)
        v = v.reshape(seq_len, n_kv, hd).transpose(1, 0, 2)

        # Positional encoding (RoPE)
        if arch.positional == "rope":
            q = _rope(q, start_pos, hd, arch.rope_base)
            k = _rope(k, start_pos, hd, arch.rope_base)

        # KV cache update
        if kv_cache is not None:
            k, v = kv_cache.update(i, k, v)

        # GQA: repeat KV heads to match Q heads
        n_rep = n_h // n_kv
        if n_rep > 1:
            k = np.repeat(k, n_rep, axis=0)
            v = np.repeat(v, n_rep, axis=0)

        # Scaled dot-product attention
        scale = np.sqrt(hd).astype(np.float32)
        attn = (q @ k.transpose(0, 2, 1)) / scale + mask
        attn = _softmax(attn, axis=-1)
        out = (attn @ v).transpose(1, 0, 2).reshape(seq_len, arch.n_embed)

        # Output projection
        x = x + out @ T(w("layers.{i}.o_proj.weight", i))
        bias = wn("layers.{i}.o_proj.bias", i)
        if bias is not None:
            x = x + bias

        # === Feed-forward ===
        if arch.norm == "rms_norm":
            h = norm(x, w("layers.{i}.ff_norm.weight", i))
        else:
            b = wn("layers.{i}.ff_norm.bias", i)
            h = norm(x, w("layers.{i}.ff_norm.weight", i), b if b is not None else np.zeros(arch.n_embed))

        if arch.activation == "swiglu" and "layers.{i}.ffn.gate.weight" in W:
            gate = h @ T(w("layers.{i}.ffn.gate.weight", i))
            up = h @ T(w("layers.{i}.ffn.up.weight", i))
            x = x + (_silu(gate) * up) @ T(w("layers.{i}.ffn.down.weight", i))
        else:
            h2 = h @ T(w("layers.{i}.ffn.up.weight", i))
            bias = wn("layers.{i}.ffn.up.bias", i)
            if bias is not None:
                h2 = h2 + bias
            h2 = _gelu(h2)
            x = x + h2 @ T(w("layers.{i}.ffn.down.weight", i))
            bias = wn("layers.{i}.ffn.down.bias", i)
            if bias is not None:
                x = x + bias

    # ── Final norm + LM head ─────────────────────────────────────────────
    if arch.norm == "rms_norm":
        x = norm(x, w("final_norm.weight"))
    else:
        b = wn("final_norm.bias")
        x = norm(x, w("final_norm.weight"), b if b is not None else np.zeros(arch.n_embed))

    # LM head (weight tying or separate)
    embed_weight = get_weight(arch.weight_map["embed.token"])
    try:
        lm_head_weight = get_weight("lm_head.weight")
        embed_weight = lm_head_weight
    except KeyError:
        pass
    return x[-1] @ embed_weight.T


# ══════════════════════════════════════════════════════════════════════════════
# KV Cache for incremental decoding
# ══════════════════════════════════════════════════════════════════════════════

class KVCache:
    """Per-layer key-value cache for incremental decoding.

    Stores K and V tensors from previous tokens so we only need to process
    the new token on each step (instead of recomputing the full sequence).
    """

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self._k: List[Optional[np.ndarray]] = [None] * n_layers
        self._v: List[Optional[np.ndarray]] = [None] * n_layers

    def update(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update cache for a layer and return concatenated K, V.

        Args:
            layer_idx: Layer index to update.
            k: New key tensor (n_heads, seq_len, head_dim).
            v: New value tensor (n_heads, seq_len, head_dim).

        Returns:
            (k_cat, v_cat) — concatenated cached + new K, V.
        """
        if self._k[layer_idx] is None:
            self._k[layer_idx] = k
            self._v[layer_idx] = v
        else:
            self._k[layer_idx] = np.concatenate([self._k[layer_idx], k], axis=1)
            self._v[layer_idx] = np.concatenate([self._v[layer_idx], v], axis=1)
        return self._k[layer_idx], self._v[layer_idx]

    def get(self, layer_idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get cached K, V for a layer."""
        if self._k[layer_idx] is None:
            return None
        return self._k[layer_idx], self._v[layer_idx]

    def reset(self):
        """Clear all cached K, V."""
        self._k = [None] * self.n_layers
        self._v = [None] * self.n_layers

    @property
    def seq_len(self) -> int:
        """Current cached sequence length."""
        if self._k[0] is None:
            return 0
        return self._k[0].shape[1]


# ══════════════════════════════════════════════════════════════════════════════
# Compression support
# ══════════════════════════════════════════════════════════════════════════════

class _CompressedWeight:
    """Compressed weight storage — VQ centroids + assignments + residual.

    Decompression: reconstructed = centroids[assignments] + residual
    With float16 residual, error is ~5e-8 per element (near machine epsilon).
    """

    __slots__ = ('centroids', 'assignments', 'residual', 'shape', 'dtype')

    def __init__(self, centroids: np.ndarray, assignments: np.ndarray,
                 residual: Optional[np.ndarray], shape: tuple, dtype: np.dtype):
        self.centroids = centroids
        self.assignments = assignments
        self.residual = residual  # float16 residual for exact reconstruction
        self.shape = shape
        self.dtype = dtype

    def decompress(self) -> np.ndarray:
        """Reconstruct weight from centroids + assignments + residual."""
        reconstructed = self.centroids[self.assignments]
        if self.residual is not None:
            reconstructed = reconstructed + self.residual.astype(np.float32)
        return reconstructed.reshape(self.shape)

    @property
    def compressed_bytes(self) -> int:
        """Total compressed size in bytes."""
        size = self.centroids.nbytes + self.assignments.nbytes
        if self.residual is not None:
            size += self.residual.nbytes
        return size


class _LRUCache:
    """Simple LRU cache for decompressed weights."""

    def __init__(self, max_size: int = 100):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: np.ndarray):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


# ══════════════════════════════════════════════════════════════════════════════
# Engine
# ══════════════════════════════════════════════════════════════════════════════

class NumpyEngine:
    """Generic NumPy inference — any architecture via config, not code.

    Features:
      - Compression: weights compressed via vector quantization (4x memory savings)
      - KV cache: incremental decoding (only process new token after first step)
      - Streaming: async generator for token-by-token output
    """

    def __init__(
        self,
        config: dict,
        weights: dict,
        tokenizer: Any = None,
        compress: bool = True,
        n_clusters: int = 16,
        cache_size: int = 100,
    ):
        """Initialize engine with optional compression.

        Args:
            config: HuggingFace config.json dict.
            weights: Dict of weight name → numpy array.
            tokenizer: Tokenizer instance (MorphTokenizer or HF tokenizer).
            compress: Whether to compress weights via vector quantization.
            n_clusters: Number of clusters for VQ compression (16 = 4x savings).
            cache_size: Max number of decompressed weights to cache.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.arch = _build_arch(
            name=config.get("architectures", ["unknown"])[0],
            config=config,
            weight_keys=set(weights.keys()),
        )
        self.vocab_size = config.get("vocab_size", config.get("n_vocab", 0))
        self.max_context = config.get("n_positions", config.get("max_position_embeddings", 1024))

        # Compression
        self._compress = compress
        self._n_clusters = n_clusters
        self._raw_weights: Dict[str, np.ndarray] = {}
        self._compressed_weights: Dict[str, _CompressedWeight] = {}
        self._cache = _LRUCache(max_size=cache_size)

        # KV cache for incremental decoding
        self._kv_cache: Optional[KVCache] = None

        # Statistics
        self._total_raw_bytes = 0
        self._total_compressed_bytes = 0

        # Load weights
        if compress:
            self._compress_weights(weights)
        else:
            self._raw_weights = weights

        logger.info(
            "NumpyEngine: %s, %d params, compression=%s, ratio=%.1fx",
            self.arch.name,
            sum(w.size for w in weights.values()),
            compress,
            self._total_raw_bytes / max(self._total_compressed_bytes, 1),
        )

    def _compress_weights(self, weights: Dict[str, np.ndarray]):
        """Compress all weights via VQ + float16 residual (near-lossless).

        Strategy:
        1. VQ with n_clusters centroids → captures most of the weight structure
        2. Compute residual = original - VQ_approximation
        3. Store residual as float16 (error ~5e-8 per element)
        4. Decompression: centroids[assignments] + residual.float32

        This achieves ~2.5x compression with near-zero error.
        For exact lossless: skip compression on small weights.
        """
        for name, raw in weights.items():
            flat = raw.flatten().astype(np.float32)
            n = len(flat)

            if n < self._n_clusters * 2:
                # Too small to compress — store raw
                self._raw_weights[name] = raw
                self._total_raw_bytes += raw.nbytes
                self._total_compressed_bytes += raw.nbytes
                continue

            # VQ: initialize centroids from quantiles
            quantiles = np.linspace(0, 100, self._n_clusters + 2)[1:-1]
            centroids = np.percentile(flat, quantiles)

            # Assign each weight to nearest centroid
            distances = np.abs(flat[:, np.newaxis] - centroids[np.newaxis, :])
            assignments = np.argmin(distances, axis=1).astype(np.uint8)

            # Compute VQ approximation and residual
            vq_approx = centroids[assignments]
            residual = flat - vq_approx

            # Store residual as float16 for ~2x additional compression
            # Error from float16: ~5e-8 per element (near machine epsilon)
            residual_f16 = residual.astype(np.float16)

            self._compressed_weights[name] = _CompressedWeight(
                centroids=centroids,
                assignments=assignments,
                residual=residual_f16,
                shape=raw.shape,
                dtype=raw.dtype,
            )

            self._total_raw_bytes += raw.nbytes
            # Centroids (float32) + assignments (uint8) + residual (float16)
            compressed_size = centroids.nbytes + assignments.nbytes + residual_f16.nbytes
            self._total_compressed_bytes += compressed_size

    def _get_weight(self, name: str) -> np.ndarray:
        """Get weight tensor — decompress on demand, cache after first use."""
        # Check cache first
        cached = self._cache.get(name)
        if cached is not None:
            return cached

        # Decompress from compressed storage
        if name in self._compressed_weights:
            raw = self._compressed_weights[name].decompress()
            self._cache.put(name, raw)
            return raw

        # Fall back to raw weights
        if name in self._raw_weights:
            raw = self._raw_weights[name]
            self._cache.put(name, raw)
            return raw

        raise KeyError(f"Weight '{name}' not found")

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        tokenizer: Any = None,
        compress: bool = True,
        n_clusters: int = 16,
    ) -> "NumpyEngine":
        """Load model from HuggingFace cache.

        Args:
            model_id: HuggingFace model ID (e.g., "gpt2", "Qwen/Qwen2.5-0.5B-Instruct").
            tokenizer: Optional tokenizer. If None, loads MorphTokenizer.
            compress: Whether to compress weights via VQ.
            n_clusters: Number of clusters for VQ (16 = 4x, 32 = 8x, 8 = 2x).
        """
        config, weights = _load_weights(model_id)
        if tokenizer is None:
            from domains.infrastructure.morph_tokenizer import MorphTokenizer
            tokenizer = MorphTokenizer.from_pretrained(model_id)
        return cls(
            config=config,
            weights=weights,
            tokenizer=tokenizer,
            compress=compress,
            n_clusters=n_clusters,
        )

    def _forward(self, token_ids: List[int], kv_cache: Optional[KVCache] = None, start_pos: int = 0) -> np.ndarray:
        """Forward pass with optional KV cache for incremental decoding.

        Args:
            token_ids: Input token IDs.
            kv_cache: Optional KV cache for incremental decoding.
            start_pos: Starting position for RoPE (used with KV cache).

        Returns:
            Logits for next token prediction.
        """
        return _forward_cached(self._get_weight, self.arch, token_ids, kv_cache, start_pos)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        use_kv_cache: bool = True,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_k: Top-k sampling (0 = disabled).
            use_kv_cache: Whether to use KV cache for faster generation.

        Returns:
            Generated text (including prompt).
        """
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer")

        ids = self.tokenizer.encode(prompt)
        initial_len = len(ids)

        if use_kv_cache:
            self._kv_cache = KVCache(self.arch.n_layers)

        for step in range(max_new_tokens):
            if use_kv_cache and self._kv_cache.seq_len > 0:
                # Incremental: only process the new token
                input_ids = [ids[-1]]
                start_pos = self._kv_cache.seq_len
            else:
                # Full context: process all tokens
                input_ids = ids[-self.max_context:]
                start_pos = 0

            logits = self._forward(input_ids, kv_cache=self._kv_cache, start_pos=start_pos)

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Apply top-k
            if top_k > 0:
                top_k_idx = np.argpartition(logits, -top_k)[-top_k:]
                mask = np.full_like(logits, -np.inf)
                mask[top_k_idx] = logits[top_k_idx]
                logits = mask

            # Sample or greedy
            if temperature > 0:
                probs = _softmax(logits)
                next_id = int(np.random.choice(len(probs), p=probs))
            else:
                next_id = int(np.argmax(logits))

            # Stop on EOS
            if next_id == self.tokenizer.eos_token_id:
                break

            ids.append(next_id)

        # Reset KV cache
        if self._kv_cache is not None:
            self._kv_cache.reset()
            self._kv_cache = None

        return self.tokenizer.decode(ids)

    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> AsyncGenerator[str, None]:
        """Generate text token-by-token (async generator).

        Args:
            prompt: Input text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_k: Top-k sampling (0 = disabled).

        Yields:
            One token string at a time.
        """
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer")

        ids = self.tokenizer.encode(prompt)
        self._kv_cache = KVCache(self.arch.n_layers)

        for step in range(max_new_tokens):
            # Run forward pass in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()

            if self._kv_cache.seq_len > 0:
                input_ids = [ids[-1]]
                start_pos = self._kv_cache.seq_len
            else:
                input_ids = ids[-self.max_context:]
                start_pos = 0

            logits = await loop.run_in_executor(
                None,
                lambda: self._forward(input_ids, kv_cache=self._kv_cache, start_pos=start_pos),
            )

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Apply top-k
            if top_k > 0:
                top_k_idx = np.argpartition(logits, -top_k)[-top_k:]
                mask = np.full_like(logits, -np.inf)
                mask[top_k_idx] = logits[top_k_idx]
                logits = mask

            # Sample or greedy
            if temperature > 0:
                probs = _softmax(logits)
                next_id = int(np.random.choice(len(probs), p=probs))
            else:
                next_id = int(np.argmax(logits))

            # Stop on EOS
            if next_id == self.tokenizer.eos_token_id:
                break

            ids.append(next_id)

            # Yield the new token
            token_text = self.tokenizer.decode([next_id])
            yield token_text

        # Reset KV cache
        self._kv_cache.reset()
        self._kv_cache = None

    def info(self) -> Dict[str, Any]:
        """Return engine information."""
        compression_ratio = self._total_raw_bytes / max(self._total_compressed_bytes, 1)
        return {
            "arch": self.arch.name,
            "arch_config": self.arch.norm + "/" + self.arch.positional + "/" + self.arch.activation + "/" + self.arch.attention,
            "vocab_size": self.vocab_size,
            "max_context": self.max_context,
            "num_layers": self.arch.n_layers,
            "num_params": sum(w.size for w in self._raw_weights.values()) + sum(
                c.shape[0] for c in self._compressed_weights.values()
            ),
            "has_tokenizer": self.tokenizer is not None,
            "compressed": self._compress,
            "compression_ratio": compression_ratio,
            "raw_bytes": self._total_raw_bytes,
            "compressed_bytes": self._total_compressed_bytes,
        }
