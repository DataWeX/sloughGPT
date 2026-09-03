"""
weight_mapper.py - Maps SLNC tensor dict to flat C weight array.

Flat layout per layer:
  attn_norm[D], q_w[D*NH*HD], q_b[NH*HD], k_w[D*NKV*HD], k_b[NKV*HD],
  v_w[D*NKV*HD], v_b[NKV*HD], o_w[NH*HD*D], o_b[D],
  ff_norm[D], gate_w[D*FF], gate_b[FF], up_w[D*FF], up_b[FF],
  down_w[FF*D], down_b[D]
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger("slo.inference.native.weight_mapper")


def map_slnc_to_native(
    tensors: Dict[str, np.ndarray],
    n_layers: int,
    hidden_dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    ff_dim: int,
    vocab_size: int,
    is_qwen_style: bool = True,
) -> Tuple[np.ndarray, dict]:
    """Map SLNC weight dict to flat float32 array for C engine.

    Args:
        tensors: SLNC weight dict (from parser.get_weights_dict())
        n_layers: Number of transformer layers
        hidden_dim: Model hidden dimension (D)
        n_heads: Number of attention heads (NH)
        n_kv_heads: Number of KV heads (NKV, for GQA)
        head_dim: Per-head dimension (HD)
        ff_dim: Feed-forward intermediate dimension (FF)
        vocab_size: Vocabulary size (V)
        is_qwen_style: If True, uses model.layers.N.* naming. If False, uses h.N.* naming.

    Returns:
        (flat_array, info_dict)
    """
    D, NH, NKV, HD, FF, V = hidden_dim, n_heads, n_kv_heads, head_dim, ff_dim, vocab_size

    def _get(name, n):
        """Return first n raveled floats of a tensor, or zeros(n) if missing.

        Missing tensors (e.g. architectures that omit a weight) must not
        break the flat layout, so they default to zeros of the block size.
        """
        t = tensors.get(name)
        if t is None:
            return np.zeros(n, dtype=np.float32)
        return t.astype(np.float32).ravel()[:n]

    def _get_or(*names):
        for name in names:
            if name in tensors:
                return tensors[name].astype(np.float32)
        return np.zeros(0, dtype=np.float32)

    def _bias_or(name, size):
        """Get bias tensor or return zeros if missing (for models without biases)."""
        t = tensors.get(name)
        if t is not None:
            return t.astype(np.float32)
        return np.zeros(size, dtype=np.float32)

    prefix_fn = lambda i: f"model.layers.{i}" if is_qwen_style else f"h.{i}"

    layer_size = (D + D*(NH*HD) + NH*HD + D*(NKV*HD) + NKV*HD
                  + D*(NKV*HD) + NKV*HD + NH*HD*D + D
                  + D + D*FF + FF + D*FF + FF + FF*D + D)

    total = V * D + layer_size * n_layers + D + V * D
    flat = np.zeros(total, dtype=np.float32)
    offset = 0

    embed = tensors.get("model.embed_tokens.weight", tensors.get("wte.weight"))
    if embed is None:
        embed = np.zeros(V*D, dtype=np.float32)
    flat[offset:offset + V*D] = embed.astype(np.float32).ravel()[:V*D]
    offset += V * D

    layer_info = {}
    for i in range(n_layers):
        p = prefix_fn(i)
        start = offset

        an_w = _get(f"{p}.input_layernorm.weight", D)
        flat[offset:offset+D] = an_w; offset += D

        qw = _get(f"{p}.self_attn.q_proj.weight", D*(NH*HD))
        flat[offset:offset+D*(NH*HD)] = qw; offset += D*(NH*HD)
        qb = _get(f"{p}.self_attn.q_proj.bias", NH*HD)
        flat[offset:offset+NH*HD] = qb; offset += NH*HD

        kw = _get(f"{p}.self_attn.k_proj.weight", D*(NKV*HD))
        flat[offset:offset+D*(NKV*HD)] = kw; offset += D*(NKV*HD)
        kb = _get(f"{p}.self_attn.k_proj.bias", NKV*HD)
        flat[offset:offset+NKV*HD] = kb; offset += NKV*HD

        vw = _get(f"{p}.self_attn.v_proj.weight", D*(NKV*HD))
        flat[offset:offset+D*(NKV*HD)] = vw; offset += D*(NKV*HD)
        vb = _get(f"{p}.self_attn.v_proj.bias", NKV*HD)
        flat[offset:offset+NKV*HD] = vb; offset += NKV*HD

        ow = _get(f"{p}.self_attn.o_proj.weight", NH*HD*D)
        flat[offset:offset+NH*HD*D] = ow; offset += NH*HD*D
        ob = _get_or(f"{p}.self_attn.o_proj.bias", f"{p}.self_attn.bias")
        if ob.size == 0:
            ob = np.zeros(D, dtype=np.float32)
        flat[offset:offset+D] = ob.ravel()[:D]; offset += D

        fnw = _get(f"{p}.post_attention_layernorm.weight", D)
        flat[offset:offset+D] = fnw; offset += D

        gw = _get(f"{p}.mlp.gate_proj.weight", D*FF)
        flat[offset:offset+D*FF] = gw; offset += D*FF
        gb = _bias_or(f"{p}.mlp.gate_proj.bias", FF)
        flat[offset:offset+FF] = gb.ravel()[:FF]; offset += FF

        uw = _get(f"{p}.mlp.up_proj.weight", D*FF)
        flat[offset:offset+D*FF] = uw; offset += D*FF
        ub = _bias_or(f"{p}.mlp.up_proj.bias", FF)
        flat[offset:offset+FF] = ub.ravel()[:FF]; offset += FF

        dw = _get(f"{p}.mlp.down_proj.weight", FF*D)
        flat[offset:offset+FF*D] = dw; offset += FF*D
        db = _bias_or(f"{p}.mlp.down_proj.bias", D)
        flat[offset:offset+D] = db.ravel()[:D]; offset += D

        layer_info[i] = {"offset": start, "size": layer_size}

    fnw_f = tensors.get("model.norm.weight")
    if fnw_f is None:
        fnw_f = tensors.get("ln_f.weight")
    if fnw_f is None:
        fnw_f = np.zeros(D, dtype=np.float32)
    flat[offset:offset+D] = fnw_f.astype(np.float32).ravel()[:D]; offset += D

    lm_head = tensors.get("model.lm_head.weight")
    if lm_head is None:
        lm_head = tensors.get("wte.weight", tensors.get("model.embed_tokens.weight"))
    if lm_head is None:
        lm_head = np.zeros(V*D, dtype=np.float32)
    flat[offset:offset+V*D] = lm_head.astype(np.float32).ravel()[:V*D]; offset += V*D

    info = {
        "total_floats": offset,
        "layer_size": layer_size,
        "layers": layer_info,
    }
    logger.info("weight_mapper: %d floats (%.1f MB), %d layers",
                offset, offset * 4 / 1e6, n_layers)
    return flat[:offset], info
