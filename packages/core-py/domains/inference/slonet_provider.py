"""
SloNet Chat Provider — pure NumPy inference via SloTransformer.

Loads a HuggingFace model's weights into SloTransformer and runs inference
entirely through NumPy ops. No PyTorch dependency at inference time.

Universal weight import: uses ArchConfig to auto-detect any HuggingFace
architecture and convert weights to SloTransformer format. No per-model
hardcoding — new arch = new ArchConfig instance.
"""
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncIterator
import numpy as np

logger = logging.getLogger("man.inference.slonet_provider")


# ══════════════════════════════════════════════════════════════════════════════
# Universal HF → SloTransformer converter
# ══════════════════════════════════════════════════════════════════════════════

# ArchConfig canonical → SloTransformer canonical (shared for all architectures)
_ARCH_TO_SLONET_SHARED = {
    "embed.token": "tok_emb.weight",
    "embed.pos": "pos_emb.weight",
    "layers.{i}.attn_norm.weight": "blocks.{i}.attn_norm.weight",
    "layers.{i}.attn_norm.bias": None,  # RMSNorm has no bias — drop
    "layers.{i}.q.weight": "blocks.{i}.attn.q_proj.weight",
    "layers.{i}.q.bias": "blocks.{i}.attn.q_proj.bias",
    "layers.{i}.k.weight": "blocks.{i}.attn.k_proj.weight",
    "layers.{i}.k.bias": "blocks.{i}.attn.k_proj.bias",
    "layers.{i}.v.weight": "blocks.{i}.attn.v_proj.weight",
    "layers.{i}.v.bias": "blocks.{i}.attn.v_proj.bias",
    "layers.{i}.qkv.weight": None,  # fused — handled by _split_fused_qkv
    "layers.{i}.qkv.bias": None,    # fused — handled by _split_fused_qkv
    "layers.{i}.o_proj.weight": "blocks.{i}.attn.o_proj.weight",
    "layers.{i}.o_proj.bias": "blocks.{i}.attn.o_proj.bias",
    "layers.{i}.ff_norm.weight": "blocks.{i}.ff_norm.weight",
    "layers.{i}.ff_norm.bias": None,  # RMSNorm has no bias — drop
    "layers.{i}.ffn.down.weight": "blocks.{i}.ff.w2.weight",
    "layers.{i}.ffn.down.bias": "blocks.{i}.ff.w2.bias",
    "final_norm.weight": "norm.weight",
    "final_norm.bias": None,  # RMSNorm has no bias — drop
}

# SwiGLU: has separate gate_proj + up_proj
_ARCH_TO_SLONET_SWIGLU = {
    "layers.{i}.ffn.gate.weight": "blocks.{i}.ff.w1.weight",
    "layers.{i}.ffn.gate.bias": "blocks.{i}.ff.w1.bias",
    "layers.{i}.ffn.up.weight": "blocks.{i}.ff.w3.weight",
    "layers.{i}.ffn.up.bias": "blocks.{i}.ff.w3.bias",
}

# GELU: only up + down (no gate), up maps to w1, w3 synthesized as identity
_ARCH_TO_SLONET_GELU = {
    "layers.{i}.ffn.up.weight": "blocks.{i}.ff.w1.weight",
    "layers.{i}.ffn.up.bias": "blocks.{i}.ff.w1.bias",
}


def _split_fused_qkv(
    hf_key: str, arr: np.ndarray, n_embed: int, n_layer: int,
    hf_state_dict: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Split fused QKV weight/bias into separate Q, K, V.

    GPT-2 stores QKV as a single (n_embed, 3*n_embed) tensor.
    SloTransformer expects separate Q, K, V projections.
    Also handles transpose: GPT-2 stores (in, out), SloTransformer expects (out, in).
    """
    result = {}
    # Extract layer index from key like "h.0.attn.c_attn.weight"
    parts = hf_key.split(".")
    layer_idx = None
    for j, p in enumerate(parts):
        if p == "h" and j + 1 < len(parts):
            try:
                layer_idx = int(parts[j + 1])
                break
            except ValueError:
                continue
    if layer_idx is None:
        return result

    is_bias = arr.ndim == 1
    if is_bias:
        # bias: (3*n_embed,) → split into 3 × (n_embed,)
        q_b, k_b, v_b = np.split(arr, 3, axis=0)
        result[f"blocks.{layer_idx}.attn.q_proj.bias"] = q_b
        result[f"blocks.{layer_idx}.attn.k_proj.bias"] = k_b
        result[f"blocks.{layer_idx}.attn.v_proj.bias"] = v_b
    else:
        # weight: (n_embed, 3*n_embed) → transpose → (3*n_embed, n_embed) → split
        wt = arr.T  # (3*n_embed, n_embed)
        q_w, k_w, v_w = np.split(wt, 3, axis=0)
        result[f"blocks.{layer_idx}.attn.q_proj.weight"] = q_w
        result[f"blocks.{layer_idx}.attn.k_proj.weight"] = k_w
        result[f"blocks.{layer_idx}.attn.v_proj.weight"] = v_w

    return result


def convert_hf_to_slonet(
    hf_state_dict: Dict[str, np.ndarray],
    n_layer: int,
    config: Optional[dict] = None,
) -> Dict[str, np.ndarray]:
    """Universal HF → SloTransformer weight converter.

    Auto-detects any HuggingFace architecture via ArchConfig.build_arch(),
    resolves canonical names, and maps to SloTransformer format.

    Handles:
    - Separate Q/K/V projections (LLaMA, Qwen, Mistral, Gemma, Phi, ...)
    - Fused QKV projections (GPT-2, GPT-Neo, ...)
    - SwiGLU (LLaMA-family) and GELU (GPT-2) feed-forward
    - Weight transposition where needed
    - LayerNorm → RMSNorm (bias dropped)
    - Positional embeddings skipped (RoPE)

    Args:
        hf_state_dict: HuggingFace model weights.
        n_layer: Number of transformer layers.
        config: Optional HuggingFace config.json dict. If None, auto-detected
            from weight keys via ArchConfig.

    Returns:
        Dict mapping SloTransformer canonical names → weight arrays.
    """
    from domains.infrastructure.arch_config import build_arch, ArchConfig

    # Auto-detect architecture
    if config is None:
        config = {}
    arch = build_arch(
        name=config.get("architectures", ["unknown"])[0],
        config=config,
        weight_keys=set(hf_state_dict.keys()),
    )

    logger.info("convert_hf_to_slonet: arch=%s (norm=%s, pos=%s, act=%s, attn=%s, transpose=%s)",
                arch.name, arch.norm, arch.positional, arch.activation, arch.attention, arch.transpose_weights)

    result = {}
    W = arch.weight_map

    # Embeddings and lm_head are NOT weight matrices — never transpose them
    NO_TRANSPOSE_KEYS = {"embed.token", "embed.pos", "lm_head"}

    # Track whether this arch uses fused QKV (GPT-2) or split (LLaMA-family)
    has_fused_qkv = "layers.{i}.qkv.weight" in W

    # Build the full mapping: shared + FFN-specific
    arch_to_slonet = dict(_ARCH_TO_SLONET_SHARED)
    if arch.activation == "swiglu":
        # SwiGLU: has separate gate_proj + up_proj → w1 + w3
        arch_to_slonet.update(_ARCH_TO_SLONET_SWIGLU)
    else:
        # GELU: up_proj → w1, w3 synthesized as identity
        arch_to_slonet.update(_ARCH_TO_SLONET_GELU)

    # LayerNorm has bias — include norm bias mappings (RMSNorm drops them via None)
    if arch.norm == "layer_norm":
        arch_to_slonet["layers.{i}.attn_norm.bias"] = "blocks.{i}.attn_norm.bias"
        arch_to_slonet["layers.{i}.ff_norm.bias"] = "blocks.{i}.ff_norm.bias"
        arch_to_slonet["final_norm.bias"] = "norm.bias"

    for hf_key, arr in hf_state_dict.items():
        mapped = False

        for canonical, slo_target in arch_to_slonet.items():
            if slo_target is None:
                # Skip dropped keys (bias, fused QKV, positional)
                if has_fused_qkv and "qkv" in canonical:
                    # Handle fused QKV separately
                    if hf_key == W.get(canonical, "").replace("{i}", "") or \
                       any(hf_key == W.get(canonical, "").replace("{i}", str(i)) for i in range(n_layer)):
                        result.update(_split_fused_qkv(hf_key, arr, arch.n_embed, n_layer, hf_state_dict))
                        mapped = True
                        break
                continue

            # Resolve canonical → actual HF key via weight map
            mapped_hf_key = W.get(canonical)
            if mapped_hf_key is None:
                continue

            if "{i}" in mapped_hf_key:
                # Per-layer key
                for i in range(n_layer):
                    concrete = mapped_hf_key.replace("{i}", str(i))
                    if hf_key == concrete:
                        slo_key = slo_target.replace("{i}", str(i))
                        w = arr
                        # Transpose if arch stores (in, out) but SloTransformer expects (out, in)
                        # Embeddings and lm_head are NOT linear weights — skip transposition
                        if arch.transpose_weights and w.ndim == 2 and canonical not in NO_TRANSPOSE_KEYS:
                            w = w.T
                        result[slo_key] = w
                        mapped = True
                        break
            else:
                # Global key (embed, final norm, lm_head)
                if hf_key == mapped_hf_key:
                    w = arr
                    if arch.transpose_weights and w.ndim == 2 and canonical not in NO_TRANSPOSE_KEYS:
                        w = w.T
                    result[slo_target] = w
                    mapped = True
                    break

        # Handle fused QKV for GPT-2 style
        if not mapped and has_fused_qkv:
            fused_key = W.get("layers.{i}.qkv.weight")
            if fused_key:
                for i in range(n_layer):
                    concrete = fused_key.replace("{i}", str(i))
                    if hf_key == concrete:
                        result.update(_split_fused_qkv(hf_key, arr, arch.n_embed, n_layer, hf_state_dict))
                        mapped = True
                        break

            fused_bias = W.get("layers.{i}.qkv.bias")
            if fused_bias and not mapped:
                for i in range(n_layer):
                    concrete = fused_bias.replace("{i}", str(i))
                    if hf_key == concrete:
                        result.update(_split_fused_qkv(hf_key, arr, arch.n_embed, n_layer, hf_state_dict))
                        mapped = True
                        break

    # GELU: synthesize SwiGLU gate (w3) as identity — w3=identity means
    # w2(act(w1(x)) * w3(x)) = w2(act(w1(x)) * 1) = w2(act(w1(x)))
    if arch.activation != "swiglu":
        for i in range(n_layer):
            w1_key = f"blocks.{i}.ff.w1.weight"
            if w1_key in result and f"blocks.{i}.ff.w3.weight" not in result:
                w1 = result[w1_key]
                result[f"blocks.{i}.ff.w3.weight"] = np.zeros_like(w1)
                result[f"blocks.{i}.ff.w3.bias"] = np.ones(w1.shape[0], dtype=np.float32)

    # Tie lm_head to token embedding if not separate
    if "lm_head.weight" not in result and "tok_emb.weight" in result:
        result["lm_head.weight"] = result["tok_emb.weight"]

    logger.info("convert_hf_to_slonet: mapped %d keys (arch=%s)", len(result), arch.name)
    return result


class SloNetChatProvider:
    """Pure NumPy inference via SloTransformer.

    Loads a HuggingFace model's weights into SloTransformer and runs inference
    entirely through NumPy ops. No PyTorch dependency at inference time.

    Args:
        hf_model_id: HuggingFace model ID or local path (e.g. 'gpt2', 'Qwen/Qwen2.5-0.5B-Instruct')
    """

    def __init__(self, hf_model_id: str = "gpt2"):
        import json as _json
        from pathlib import Path as _Path
        from safetensors.numpy import load_file as _load_file
        from domains.training.slonet import SloTransformer

        self._hf_model_id = hf_model_id
        self._device = "cpu"

        # Resolve model directory
        model_dir = _Path(hf_model_id)
        if not model_dir.exists():
            cache_dir = _Path.home() / ".cache/huggingface/hub"
            model_dir = cache_dir / f"models--{hf_model_id.replace('/', '--')}"
            if model_dir.exists():
                snapshots = model_dir / "snapshots"
                if snapshots.exists():
                    snaps = sorted(snapshots.iterdir())
                    if snaps:
                        model_dir = snaps[-1]

        # Load config
        config_path = model_dir / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = _json.load(f)

        n_embed = config.get("n_embd") or config.get("hidden_size", 768)
        n_head = config.get("n_head") or config.get("num_attention_heads", 12)
        n_layer = config.get("n_layer") or config.get("num_hidden_layers", 12)
        vocab_size = config.get("vocab_size", 50257)
        intermediate_size = config.get("n_inner") or config.get("intermediate_size")
        max_pos = config.get("n_positions") or config.get("max_position_embeddings", 1024)

        # Detect architecture features from weight keys
        from domains.infrastructure.arch_config import build_arch
        safetensors_path = model_dir / "model.safetensors"
        if not safetensors_path.exists():
            # Try finding any .safetensors file
            for f in model_dir.glob("*.safetensors"):
                safetensors_path = f
                break

        sd = _load_file(str(safetensors_path)) if safetensors_path.exists() else {}
        arch = build_arch(
            name=config.get("architectures", ["unknown"])[0],
            config=config,
            weight_keys=set(sd.keys()),
        )

        # GPT-2 uses absolute positional embeddings, not RoPE
        use_abs_pos = arch.positional == "absolute"

        self._model = SloTransformer(
            vocab_size=vocab_size,
            n_embed=n_embed,
            n_layer=n_layer,
            n_head=n_head,
            intermediate_size=intermediate_size,
            block_size=max_pos,
            max_seq_len=max_pos,
            use_rope=not use_abs_pos,
            dropout=0.0,
            tie_weights=True,
            use_abs_pos_emb=use_abs_pos,
            norm_type=arch.norm,
        )

        # Load weights
        if sd:
            mapped = convert_hf_to_slonet(sd, n_layer=n_layer, config=config)
            self._model.load_state_dict(mapped)

        # Load tokenizer
        self._tokenizer = self._load_tokenizer(model_dir, config)

        logger.info("SloNetChatProvider loaded: %s (embed=%d, layers=%d, heads=%d, rope=%s)",
                     hf_model_id, n_embed, n_layer, n_head, not use_abs_pos)

    def _load_tokenizer(self, model_dir, config):
        """Load tokenizer — MorphTokenizer.from_pretrained handles all parsing."""
        try:
            from domains.infrastructure.morph_tokenizer import MorphTokenizer
            # from_pretrained reads tokenizer.json and parses vocab+merges correctly
            return MorphTokenizer.from_pretrained(self._hf_model_id)
        except Exception as e:
            logger.warning("MorphTokenizer load failed: %s", e)

        raise RuntimeError(f"No tokenizer found for {self._hf_model_id}")

    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 1.0) -> str:
        """Generate text from a prompt."""
        tokens = self._tokenizer.encode(prompt)
        generated = list(tokens)
        for _ in range(max_tokens):
            import numpy as _np
            inp = _np.array([generated], dtype=_np.int64)
            result = self._model.forward(inp)
            logits = result[0].data
            next_token = int(_np.argmax(logits[0, -1]))
            generated.append(next_token)
        return self._tokenizer.decode(generated)

    async def generate_stream(self, messages, **kwargs):
        """Streaming generation — yields token strings."""
        prompt = messages[-1].get("content", "") if messages else ""
        tokens = self._tokenizer.encode(prompt)
        generated = list(tokens)
        for _ in range(kwargs.get("max_tokens", 200)):
            import numpy as _np
            inp = _np.array([generated], dtype=_np.int64)
            result = self._model.forward(inp)
            logits = result[0].data
            next_token = int(_np.argmax(logits[0, -1]))
            generated.append(next_token)
            yield self._tokenizer.decode([next_token])
