"""
Generic transformer forward pass — ONE function handles ALL architectures.

Architectures differ only in:
  - Which weight names to look up (via arch.weight_map)
  - Which math ops to use (via arch.norm, arch.positional, etc.)
"""

from typing import List, Optional

import numpy as np

from domains.infrastructure.arch_config import ArchConfig
from domains.infrastructure.numpy_ops import rmsnorm, layer_norm, softmax, gelu, silu, rope


def norm_fn(arch: ArchConfig):
    """Return the normalization function for this architecture."""
    return rmsnorm if arch.norm == "rms_norm" else layer_norm


def forward(weights: dict, arch: ArchConfig, token_ids: List[int]) -> np.ndarray:
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

    norm = norm_fn(arch)
    T = (lambda w: w.T) if not arch.transpose_weights else (lambda w: w)  # transpose (out, in) → (in, out)

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
            h = norm(x, w("layers.{i}.attn_norm.weight", i), b if b is not None else np.zeros(arch.n_embed, dtype=x.dtype))

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

        # Reshape to (seq, heads, head_dim); RoPE needs this layout
        n_h = arch.n_head
        n_kv = arch.n_kv_head
        hd = arch.head_dim
        q = q.reshape(seq_len, n_h, hd)
        k = k.reshape(seq_len, n_kv, hd)
        v = v.reshape(seq_len, n_kv, hd)

        # Positional encoding (RoPE: rotate per-position, shared across heads)
        if arch.positional == "rope":
            q = rope(q, 0, hd, arch.rope_base)
            k = rope(k, 0, hd, arch.rope_base)

        # Attention math needs (heads, seq, head_dim)
        q = q.transpose(1, 0, 2)
        k = k.transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)

        # GQA: repeat KV heads to match Q heads
        n_rep = n_h // n_kv
        if n_rep > 1:
            k = np.repeat(k, n_rep, axis=0)
            v = np.repeat(v, n_rep, axis=0)

        # Scaled dot-product attention
        scale = np.sqrt(hd).astype(np.float32)
        attn = (q @ k.transpose(0, 2, 1)) / scale + mask
        attn = softmax(attn, axis=-1)
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
            h = norm(x, w("layers.{i}.ff_norm.weight", i), b if b is not None else np.zeros(arch.n_embed, dtype=x.dtype))

        if arch.activation == "swiglu" and "layers.{i}.ffn.gate.weight" in W:
            # SwiGLU: gate * silu(up) @ down
            gate = h @ T(w("layers.{i}.ffn.gate.weight", i))
            up = h @ T(w("layers.{i}.ffn.up.weight", i))
            x = x + (silu(gate) * up) @ T(w("layers.{i}.ffn.down.weight", i))
        else:
            # GELU MLP: up → gelu → down
            h2 = h @ T(w("layers.{i}.ffn.up.weight", i))
            bias = wn("layers.{i}.ffn.up.bias", i)
            if bias is not None:
                h2 = h2 + bias
            h2 = gelu(h2)
            x = x + h2 @ T(w("layers.{i}.ffn.down.weight", i))
            bias = wn("layers.{i}.ffn.down.bias", i)
            if bias is not None:
                x = x + bias

    # ── Final norm + LM head ─────────────────────────────────────────────
    if arch.norm == "rms_norm":
        x = norm(x, w("final_norm.weight"))
    else:
        b = wn("final_norm.bias")
        x = norm(x, w("final_norm.weight"), b if b is not None else np.zeros(arch.n_embed, dtype=x.dtype))

    # LM head (weight tying or separate)
    embed_weight = weights[arch.weight_map["embed.token"]]  # raw weight, no T
    if "lm_head.weight" in weights:
        embed_weight = weights["lm_head.weight"]
    return x[-1] @ embed_weight.T


def forward_cached(
    get_weight,
    arch: ArchConfig,
    token_ids: List[int],
    kv_cache=None,
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
    from domains.infrastructure.numpy_engine import KVCache

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

    norm = norm_fn(arch)
    T = (lambda w: w.T) if not arch.transpose_weights else (lambda w: w)

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
            h = norm(x, w("layers.{i}.attn_norm.weight", i), b if b is not None else np.zeros(arch.n_embed, dtype=x.dtype))

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

        # Reshape to (seq, heads, head_dim); RoPE needs this layout
        n_h = arch.n_head
        n_kv = arch.n_kv_head
        hd = arch.head_dim
        q = q.reshape(seq_len, n_h, hd)
        k = k.reshape(seq_len, n_kv, hd)
        v = v.reshape(seq_len, n_kv, hd)

        # Positional encoding (RoPE: rotate per-position, shared across heads)
        if arch.positional == "rope":
            q = rope(q, start_pos, hd, arch.rope_base)
            k = rope(k, start_pos, hd, arch.rope_base)

        # Attention math needs (heads, seq, head_dim)
        q = q.transpose(1, 0, 2)
        k = k.transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)

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
        attn = softmax(attn, axis=-1)
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
            h = norm(x, w("layers.{i}.ff_norm.weight", i), b if b is not None else np.zeros(arch.n_embed, dtype=x.dtype))

        if arch.activation == "swiglu" and "layers.{i}.ffn.gate.weight" in W:
            gate = h @ T(w("layers.{i}.ffn.gate.weight", i))
            up = h @ T(w("layers.{i}.ffn.up.weight", i))
            x = x + (silu(gate) * up) @ T(w("layers.{i}.ffn.down.weight", i))
        else:
            h2 = h @ T(w("layers.{i}.ffn.up.weight", i))
            bias = wn("layers.{i}.ffn.up.bias", i)
            if bias is not None:
                h2 = h2 + bias
            h2 = gelu(h2)
            x = x + h2 @ T(w("layers.{i}.ffn.down.weight", i))
            bias = wn("layers.{i}.ffn.down.bias", i)
            if bias is not None:
                x = x + bias

    # ── Final norm + LM head ─────────────────────────────────────────────
    if arch.norm == "rms_norm":
        x = norm(x, w("final_norm.weight"))
    else:
        b = wn("final_norm.bias")
        x = norm(x, w("final_norm.weight"), b if b is not None else np.zeros(arch.n_embed, dtype=x.dtype))

    # LM head (weight tying or separate)
    embed_weight = get_weight(arch.weight_map["embed.token"])
    try:
        lm_head_weight = get_weight("lm_head.weight")
        embed_weight = lm_head_weight
    except KeyError:
        pass
    return x[-1] @ embed_weight.T


def pre_extract_weights(arch: ArchConfig, weights: dict) -> dict:
    """Pre-extract all weight names for a given arch, resolving the weight map once.

    Returns a dict of canonical weight name → numpy array, ready for forward_fast().
    """
    W = arch.weight_map
    resolved = {}

    for canonical, mapped in W.items():
        if "{i}" in mapped:
            for layer in range(arch.n_layers):
                actual = mapped.replace("{i}", str(layer))
                if actual in weights:
                    resolved[f"{canonical}:{layer}"] = np.ascontiguousarray(weights[actual])
        else:
            if mapped in weights:
                resolved[canonical] = np.ascontiguousarray(weights[mapped])

    return resolved


def forward_fast(
    rw: dict,
    arch: ArchConfig,
    token_ids: List[int],
    kv_cache=None,
    start_pos: int = 0,
) -> np.ndarray:
    """Fast forward pass using pre-extracted weights. No dict lookups or string ops.

    Args:
        rw: Pre-extracted weights dict from pre_extract_weights().
        arch: Architecture config.
        token_ids: Input token IDs.
        kv_cache: Optional KVCache for incremental decoding.
        start_pos: Starting position for RoPE.

    Returns:
        Logits for next token prediction.
    """
    seq_len = len(token_ids)

    def w(canonical: str, layer: int = 0) -> np.ndarray:
        return rw[f"{canonical}:{layer}"]

    def wn(canonical: str, layer: int = 0) -> Optional[np.ndarray]:
        return rw.get(f"{canonical}:{layer}")

    norm = norm_fn(arch)
    T = (lambda w: w.T) if not arch.transpose_weights else (lambda w: w)

    # ── Embeddings ──
    x = rw["embed.token"][token_ids]
    if arch.positional == "absolute":
        x = x + rw["embed.pos"][start_pos:start_pos + seq_len]

    # ── Causal mask ──
    if kv_cache is not None and kv_cache.seq_len > 0:
        total_len = kv_cache.seq_len + seq_len
        mask = np.triu(np.full((seq_len, total_len), -1e10, dtype=np.float32), k=1 + kv_cache.seq_len)
    else:
        mask = np.triu(np.full((seq_len, seq_len), -1e10, dtype=np.float32), k=1)

    # ── Transformer blocks ──
    for i in range(arch.n_layers):
        # === Attention norm ===
        if arch.norm == "rms_norm":
            h = norm(x, w("layers.{i}.attn_norm.weight", i))
        else:
            b = wn("layers.{i}.attn_norm.bias", i)
            h = norm(x, w("layers.{i}.attn_norm.weight", i),
                     b if b is not None else np.zeros(arch.n_embed, dtype=x.dtype))

        # === QKV ===
        if arch.attention == "mha" and "layers.{i}.qkv.weight:0" in rw:
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

        # Reshape to (seq, heads, head_dim); RoPE needs this layout
        n_h = arch.n_head
        n_kv = arch.n_kv_head
        hd = arch.head_dim
        q = q.reshape(seq_len, n_h, hd)
        k = k.reshape(seq_len, n_kv, hd)
        v = v.reshape(seq_len, n_kv, hd)

        # RoPE (rotate per-position, shared across heads)
        if arch.positional == "rope":
            q = rope(q, start_pos, hd, arch.rope_base)
            k = rope(k, start_pos, hd, arch.rope_base)

        # Attention math needs (heads, seq, head_dim)
        q = q.transpose(1, 0, 2)
        k = k.transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)

        # KV cache
        if kv_cache is not None:
            k, v = kv_cache.update(i, k, v)

        # GQA
        n_rep = n_h // n_kv
        if n_rep > 1:
            k = np.repeat(k, n_rep, axis=0)
            v = np.repeat(v, n_rep, axis=0)

        # Attention
        scale = np.float32(np.sqrt(hd))
        attn = (q @ k.transpose(0, 2, 1)) / scale + mask
        attn = softmax(attn, axis=-1)
        out = (attn @ v).transpose(1, 0, 2).reshape(seq_len, arch.n_embed)

        # Output proj
        x = x + out @ T(w("layers.{i}.o_proj.weight", i))
        bias = wn("layers.{i}.o_proj.bias", i)
        if bias is not None:
            x = x + bias

        # === FFN ===
        if arch.norm == "rms_norm":
            h = norm(x, w("layers.{i}.ff_norm.weight", i))
        else:
            b = wn("layers.{i}.ff_norm.bias", i)
            h = norm(x, w("layers.{i}.ff_norm.weight", i),
                     b if b is not None else np.zeros(arch.n_embed, dtype=x.dtype))

        if arch.activation == "swiglu" and "layers.{i}.ffn.gate.weight:0" in rw:
            gate = h @ T(w("layers.{i}.ffn.gate.weight", i))
            up = h @ T(w("layers.{i}.ffn.up.weight", i))
            x = x + (silu(gate) * up) @ T(w("layers.{i}.ffn.down.weight", i))
        else:
            h2 = h @ T(w("layers.{i}.ffn.up.weight", i))
            bias = wn("layers.{i}.ffn.up.bias", i)
            if bias is not None:
                h2 = h2 + bias
            h2 = gelu(h2)
            x = x + h2 @ T(w("layers.{i}.ffn.down.weight", i))
            bias = wn("layers.{i}.ffn.down.bias", i)
            if bias is not None:
                x = x + bias

    # ── Final norm + LM head ──
    if arch.norm == "rms_norm":
        x = norm(x, rw["final_norm.weight"])
    else:
        b = rw.get("final_norm.bias")
        x = norm(x, rw["final_norm.weight"],
                 b if b is not None else np.zeros(arch.n_embed, dtype=x.dtype))

    embed_weight = rw["embed.token"]
    lm_head = rw.get("lm_head.weight")
    if lm_head is not None:
        embed_weight = lm_head
    return x[-1] @ embed_weight.T
