"""soul → slnc bridge — pure format conversion for the owned architecture.

Converts a native ``.soul`` checkpoint (SloughGPTTrainer output) into a
``.slnc`` file without any HuggingFace import. Tensor names stay identity
(native layout); tokenizer/vocab metadata is embedded in the SLNC config
so ``from_slnc`` can rebuild a char/tree tokenizer without HF.

Usage:
    from domain.infrastructure._internal.soul_to_slnc import soul_to_slnc
    soul_to_slnc("models/slonet-native/foo.soul", "models/slonet-native/foo.slnc")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("slo.infrastructure.soul_to_slnc")


def build_slnc_config_from_soul(soul: Any, state_dict: dict) -> dict:
    """Build an SLNC config dict from soul metadata + weights (no HF)."""
    meta = soul.metadata if hasattr(soul, "metadata") else {}
    cfg = meta.get("config") or {}
    n_embed = int(cfg.get("n_embed") or next(
        (v.shape[1] for k, v in state_dict.items() if k == "tok_emb.weight" and v.ndim == 2),
        128,
    ))
    n_layer = int(cfg.get("n_layer") or sum(
        1 for k in state_dict if k.endswith(".attn_norm.weight")
    ) or 1)
    n_head = int(cfg.get("n_head") or 4)
    block_size = int(cfg.get("block_size") or 128)
    vocab_size = int(meta.get("vocab_size") or cfg.get("vocab_size") or 256)

    has_w3 = any(k.endswith(".ff.w3.weight") for k in state_dict)
    has_gate = any(k.endswith(".ff.w1.weight") for k in state_dict)
    # Intermediate dim from w1/w3 (out features)
    n_inner = n_embed * 4
    for k, v in state_dict.items():
        if k.endswith(".ff.w1.weight") and v.ndim == 2:
            n_inner = int(v.shape[0])
            break

    config: dict[str, Any] = {
        "architectures": ["SloTransformer"],
        "model_type": str(cfg.get("model_type") or "sloughgpt"),
        "n_embd": n_embed,
        "hidden_size": n_embed,
        "n_layer": n_layer,
        "num_hidden_layers": n_layer,
        "n_head": n_head,
        "num_attention_heads": n_head,
        "n_positions": block_size,
        "max_position_embeddings": block_size,
        "n_inner": n_inner,
        "intermediate_size": n_inner,
        "vocab_size": vocab_size,
        "native_arch": True,
        "native_activation": "swiglu" if (has_w3 or has_gate) else "gelu",
        # SloughGPTModel trains with RoPE — mark it so build_model_from_config
        # reconstructs the same architecture (no absolute pos_emb).
        "position_embedding_type": "rope",
        "rope_theta": 10000.0,
        # SloughGPTModel / SloTransformer default norm is RMSNorm
        "rms_norm_eps": 1e-5,
        # build_model_from_config maps silu → SwiGLU (w1/w2/w3)
        "hidden_act": "silu" if (has_w3 or has_gate) else "gelu",
        "slo_tokenizer": _embed_tokenizer(meta),
    }
    return config


def _embed_tokenizer(meta: dict) -> dict | None:
    """Embed tokenizer/vocab info so from_slnc need not hit HuggingFace."""
    stoi = meta.get("stoi")
    itos = meta.get("itos")
    tok = meta.get("tokenizer")
    chars = meta.get("chars")
    if not stoi and not itos and not tok:
        return None
    payload: dict[str, Any] = {}
    if isinstance(tok, dict):
        # token_tree or other structured tokenizer — pass through
        payload["type"] = tok.get("type", "unknown")
        payload["tokenizer"] = tok
    else:
        payload["type"] = "char"
    if stoi:
        payload["stoi"] = stoi
    if itos:
        # JSON keys may be ints; keep as-is for roundtrip via _CharTokenizer
        payload["itos"] = itos
    if chars:
        payload["chars"] = chars
    if meta.get("vocab_size"):
        payload["vocab_size"] = meta["vocab_size"]
    return payload


def soul_to_slnc(
    soul_path: str | Path,
    slnc_path: str | Path,
    quantize: bool | str = False,
) -> str:
    """Compile a native ``.soul`` checkpoint into ``.slnc``.

    Args:
        soul_path: Path to the ``.soul`` file.
        slnc_path: Output ``.slnc`` path.
        quantize: Passed through to ``SLNCCompiler.compile_from_dict``.

    Returns:
        Path to the created ``.slnc`` file.

    Raises:
        FileNotFoundError: If ``soul_path`` does not exist.
        ValueError: If the soul file is invalid.
    """
    from domain.inference._internal.slo_format import load_soul
    from domain.infrastructure._internal.slnc.compiler import SLNCCompiler

    soul_path = Path(soul_path)
    slnc_path = Path(slnc_path)
    if not soul_path.exists():
        raise FileNotFoundError(f".soul not found: {soul_path}")

    soul, state_dict = load_soul(str(soul_path))
    if not state_dict:
        raise ValueError(f"Empty .soul (no weights): {soul_path}")

    # Ensure lm_head exists for native layout ordering (tied weights)
    if "lm_head.weight" not in state_dict and "tok_emb.weight" in state_dict:
        state_dict = dict(state_dict)
        state_dict["lm_head.weight"] = state_dict["tok_emb.weight"]

    config = build_slnc_config_from_soul(soul, state_dict)
    slnc_path.parent.mkdir(parents=True, exist_ok=True)

    out = SLNCCompiler().compile_from_dict(
        config, state_dict, str(slnc_path), quantize=quantize
    )
    logger.info(
        "soul_to_slnc: %s → %s (%d tensors, vocab=%s)",
        soul_path,
        out,
        len(state_dict),
        config.get("vocab_size"),
        extra={"tag": "INFRA"},
    )
    return out
