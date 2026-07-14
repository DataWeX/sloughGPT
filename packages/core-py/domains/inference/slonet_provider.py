"""
SloNet Chat Provider — pure NumPy inference via SloTransformer.

Loads a HuggingFace model's weights into SloTransformer and runs inference
entirely through NumPy ops. No PyTorch dependency at inference time.

Universal weight import: uses ArchConfig to auto-detect any HuggingFace
architecture and convert weights to SloTransformer format. No per-model
hardcoding — new arch = new ArchConfig instance.
"""
import logging
import threading
from pathlib import Path
from typing import Optional, Dict
import numpy as np

logger = logging.getLogger("man.inference.slonet_provider")

# Lazy import to avoid circular dependency
_SloLayerNorm = None

def _get_slo_layernorm():
    global _SloLayerNorm
    if _SloLayerNorm is None:
        from domains.training.slonet import SloLayerNorm
        _SloLayerNorm = SloLayerNorm
    return _SloLayerNorm


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
    from domains.infrastructure.arch_config import build_arch

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

    Satisfies the ModelProvider protocol used by ProviderRouter.

    Args:
        hf_model_id: HuggingFace model ID or local path (e.g. 'gpt2', 'Qwen/Qwen2.5-0.5B-Instruct')
    """

    def __init__(self, hf_model_id: str = "gpt2", quantize: bool = False,
                 quant_bits: int = 8, quant_mode: str = "symmetric",
                 quant_clip: float = 0.999):
        import json as _json
        from pathlib import Path as _Path
        from safetensors.numpy import load_file as _load_file
        from domains.training.slonet import SloTransformer

        self._hf_model_id = hf_model_id
        self._model_id = hf_model_id
        self._device = "cpu"
        self._quant_engine = None

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
        if intermediate_size is None:
            # GPT-2 and many models: FFN is 4× hidden size
            intermediate_size = n_embed * 4
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

        # Apply quantization if requested
        if quantize:
            from domains.infrastructure.quantization import QuantEngine, walk_slo_linears

            linear_map = walk_slo_linears(self._model)
            engine = QuantEngine(bits=quant_bits, mode=quant_mode, clip_percentile=quant_clip)
            quantized_count = 0
            for name, module in linear_map.items():
                if "norm" in name:
                    continue
                info = engine.quantize(f"{name}.weight", module.weight.data.copy())
                if info.is_quantized:
                    module.set_quantized_weight(info)
                    quantized_count += 1
            self._quant_engine = engine
            logger.info("SloNetChatProvider: quantized %d/%d layers to int%d",
                        quantized_count, len(linear_map), quant_bits)

        # Load tokenizer
        self._tokenizer = self._load_tokenizer(model_dir, config)

        logger.info("SloNetChatProvider loaded: %s (embed=%d, layers=%d, heads=%d, rope=%s, quant=%s)",
                     hf_model_id, n_embed, n_layer, n_head, not use_abs_pos,
                     f"int{quant_bits}" if quantize else "none")

    @classmethod
    def from_slnc(
        cls,
        slnc_path: str,
        model_id: str = "gpt2",
        quantize: bool = False,
        quant_bits: int = 8,
        quant_mode: str = "symmetric",
        quant_clip: float = 0.999,
    ) -> "SloNetChatProvider":
        """Create provider from .slnc file (mmap, zero-copy).

        Args:
            slnc_path: Path to .slnc file
            model_id: HuggingFace model ID for tokenizer (e.g. "gpt2")
            quantize: If True, apply per-tensor quantization to weights
            quant_bits: Bits for quantization (8 or 4)
            quant_mode: "symmetric" or "asymmetric"
            quant_clip: Outlier clipping percentile (e.g., 0.999)

        Returns:
            SloNetChatProvider using mmap-backed (optionally quantized) weights
        """
        from domains.infrastructure.slnc.parser import SLNCParser
        from domains.training.slonet import SloTransformer

        parser = SLNCParser(slnc_path)
        config = parser.config

        n_embed = config.get("n_embd", config.get("hidden_size", 768))
        n_head = config.get("n_head", config.get("num_attention_heads", 12))
        n_layer = config.get("n_layer", config.get("num_hidden_layers", 12))
        vocab_size = config.get("vocab_size", 50257)
        intermediate_size = config.get("n_inner") or config.get("intermediate_size", n_embed * 4)
        max_pos = config.get("n_positions", config.get("max_position_embeddings", 1024))

        # Auto-detect positional encoding from config
        has_rope = config.get("rope_theta") is not None or config.get("position_embedding_type") == "rope"
        use_abs_pos = not has_rope

        # Auto-detect norm type
        has_rms = "model.norm.weight" in config or "model.layers.0.input_layernorm.weight" in config
        norm_type = "layer_norm" if not has_rms else "rms_norm"
        # Also check explicit config field
        if config.get("layer_norm_type"):
            norm_type = config["layer_norm_type"]

        # Auto-detect GQA (n_kv_head < n_head)
        n_kv_head = config.get("num_key_value_heads", n_head)

        # Create SloTransformer
        model = SloTransformer(
            vocab_size=vocab_size,
            n_embed=n_embed,
            n_layer=n_layer,
            n_head=n_head,
            n_kv_head=n_kv_head,
            intermediate_size=intermediate_size,
            block_size=max_pos,
            max_seq_len=max_pos,
            use_rope=not use_abs_pos,
            rope_base=config.get("rope_theta", 10000.0),
            dropout=0.0,
            tie_weights=True,
            use_abs_pos_emb=use_abs_pos,
            norm_type=norm_type,
        )

        # Load weights directly from mmap (zero copy)
        weights_dict = parser.get_weights_dict()

        # Convert and load into model
        mapped = convert_hf_to_slonet(weights_dict, n_layer=n_layer, config=config)
        model.load_state_dict(mapped)

        # Create instance (bypass __init__)
        instance = cls.__new__(cls)
        instance._hf_model_id = model_id
        instance._model_id = model_id
        instance._device = "cpu"
        instance._model = model
        instance._parser = parser  # keep mmap alive
        instance._quant_engine = None

        # Apply quantization if requested
        if quantize:
            from domains.infrastructure.quantization import QuantEngine, walk_slo_linears
            from pathlib import Path as PathlibPath

            slnc_path_obj = PathlibPath(slnc_path)
            quant_meta_path = slnc_path_obj.with_suffix(slnc_path_obj.suffix + ".quant.json")

            # Get SloLinear layers via walk_slo_linears (not named_modules,
            # which misses layers stored in plain Python list attributes)
            linear_map = walk_slo_linears(model)
            param_names = dict(model.named_parameters())

            # Build a reverse lookup: parameter name → linear module name
            # named_parameters() uses HuggingFace naming (q_proj), while
            # walk_slo_linears uses SloNet naming (W_q). Match via the
            # actual module objects.
            param_to_module = {}
            for mod_name, module in linear_map.items():
                for pname, param in param_names.items():
                    if param is module.weight:
                        param_to_module[pname] = mod_name
                        break

            # Try loading existing quant metadata sidecar
            if quant_meta_path.exists():
                engine = QuantEngine(
                    bits=quant_bits,
                    mode=quant_mode,
                    clip_percentile=quant_clip,
                )
                engine.load_metadata(str(quant_meta_path))
                logger.info(
                    "SloNetChatProvider.from_slnc: loaded quant metadata (%d tensors) from %s",
                    len(param_names), quant_meta_path,
                )

                quantized_count = 0
                for pname, param in param_names.items():
                    if pname not in param_to_module:
                        continue
                    meta = engine._error_report.get(pname)
                    if meta is not None:
                        arr = param.data.copy()
                        info = engine.quantize_with_scale(
                            pname, arr, meta.scale, meta.zero_point,
                        )
                        if info.is_quantized:
                            linear_map[param_to_module[pname]].set_quantized_weight(info)
                            quantized_count += 1
            else:
                engine = QuantEngine(
                    bits=quant_bits,
                    mode=quant_mode,
                    clip_percentile=quant_clip,
                )

                quantized_count = 0
                for pname, param in param_names.items():
                    if pname not in param_to_module:
                        continue
                    arr = param.data.copy()
                    info = engine.quantize(pname, arr)
                    if info.is_quantized:
                        linear_map[param_to_module[pname]].set_quantized_weight(info)
                        quantized_count += 1

                engine.save_metadata(str(quant_meta_path))

            instance._quant_engine = engine
            summary = engine.summary()
            logger.info(
                "SloNetChatProvider.from_slnc: quantized %d/%d tensors (bits=%d, mode=%s, avg_cosine=%.4f)",
                quantized_count, len(linear_map), quant_bits, quant_mode,
                summary.get("avg_cosine_sim", 0.0),
            )

        # Load tokenizer
        instance._tokenizer = instance._load_tokenizer(
            Path(slnc_path).parent, config
        )

        logger.info("SloNetChatProvider.from_slnc: %s, %d layers",
                     slnc_path, n_layer)

        return instance

    def quantization_report(self) -> dict:
        """Get quantization error report (if quantized).

        Returns:
            Dict with per-tensor error metrics and aggregate summary.
            Empty dict if model was not quantized.
        """
        if self._quant_engine is None:
            return {"quantized": False}
        summary = self._quant_engine.summary()
        return {
            "quantized": True,
            "bits": summary.get("bits", 0),
            "mode": summary.get("mode", "symmetric"),
            "summary": summary,
            "per_tensor": self._quant_engine.error_report(),
        }

    @property
    def model_id(self) -> str:
        """Unique identifier for this model."""
        return self._model_id

    @property
    def capabilities(self):
        """What this model supports."""
        from domains.models.provider import ModelCapabilities
        return ModelCapabilities(
            chat=True, streaming=True, embedding=False,
            vision=False, functions=False,
        )

    def _load_tokenizer(self, model_dir, config):
        """Load tokenizer — MorphTokenizer.from_pretrained handles all parsing."""
        try:
            from domains.infrastructure.morph_tokenizer import MorphTokenizer
            # from_pretrained reads tokenizer.json and parses vocab+merges correctly
            return MorphTokenizer.from_pretrained(self._hf_model_id)
        except Exception as e:
            logger.warning("MorphTokenizer load failed: %s", e)

        raise RuntimeError(f"No tokenizer found for {self._hf_model_id}")

    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 1.0,
                 top_k: int = None, top_p: float = None, repetition_penalty: float = 1.0) -> str:
        """Generate text using pure numpy inference (KV cache + inlined ops)."""
        import numpy as _np
        tokens = self._tokenizer.encode(prompt)
        input_ids = _np.array([tokens], dtype=_np.int64)
        result = self._model.generate_numpy(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token=self._tokenizer.eos_token_id or 0,
        )
        return self._tokenizer.decode(result[0].tolist())

    async def chat(self, messages, max_tokens=512, temperature=0.8, **kwargs):
        """Blocking chat — returns complete response. Runs in thread to avoid blocking event loop."""
        import asyncio
        return await asyncio.to_thread(
            self._generate_sync, messages, max_tokens, temperature,
            kwargs.get('top_k'), kwargs.get('top_p'),
            kwargs.get('repetition_penalty', 1.0),
        )

    def _generate_sync(self, messages, max_tokens=512, temperature=0.8,
                       top_k=None, top_p=None, repetition_penalty=1.0):
        """Synchronous generate with KV cache — called from chat() via to_thread."""
        import numpy as _np
        prompt = self._build_prompt(messages)
        tokens = self._tokenizer.encode(prompt)
        input_ids = _np.array([tokens], dtype=_np.int64)
        result = self._model.generate_numpy(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token=self._tokenizer.eos_token_id or 0,
        )
        return self._tokenizer.decode(result[0].tolist())

    async def chat_stream(self, messages, max_tokens=512, temperature=0.8, **kwargs):
        """Streaming chat — yields token strings with KV cache.

        Pre-fills prompt in one forward pass, then generates one token at a time
        using KV cache (each step only processes the new token).

        Robustness features:
        - cancel_event: abort generation mid-stream (e.g. on client disconnect)
        - NaN/Inf guard: stops if forward step produces non-finite logits
        - Per-token timeout: 30s per token prevents hung threads
        - Total generation timeout: 120s total prevents unbounded generation
        - Error propagation: producer thread exceptions surface to consumer

        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Keep only top-k logits before sampling
            top_p: Nucleus threshold — keep tokens with cumulative prob <= p
            repetition_penalty: Scale factor for repeated tokens (>1 = penalize)
            cancel_event: threading.Event() — set to abort generation

        Yields:
            Each token string as it's generated.
        """
        import asyncio
        import numpy as _np
        import math
        import time

        prompt = ""
        if messages and isinstance(messages[-1], dict):
            prompt = messages[-1].get("content", "")
        elif messages and isinstance(messages[-1], str):
            prompt = messages[-1]
        token_ids = self._tokenizer.encode(prompt)
        eos_id = self._tokenizer.eos_token_id or 0
        top_k = kwargs.get('top_k')
        top_p = kwargs.get('top_p')
        repetition_penalty = kwargs.get('repetition_penalty', 1.0)
        cancel_event = kwargs.get('cancel_event')

        def _stream_generate():
            """KV-cache streaming generation — runs in a thread."""
            from domains.training.slonet import _sample_from_logits
            m = self._model
            tokens = _np.array([token_ids], dtype=_np.int64)
            prompt_len = tokens.shape[1]

            # Extract weights once (same as generate_numpy)
            blocks = []
            for l in m.layers[1:-2]:
                if hasattr(l, 'attn_norm') and hasattr(l, 'ff'):
                    b = l
                    SloLN = _get_slo_layernorm()
                    has_ln = isinstance(b.attn_norm, SloLN)
                    blocks.append({
                        'an_w': b.attn_norm.weight.data,
                        'an_b': b.attn_norm.bias.data if has_ln else None,
                        'an_eps': b.attn_norm.eps,
                        'fn_w': b.ff_norm.weight.data,
                        'fn_b': b.ff_norm.bias.data if has_ln else None,
                        'fn_eps': b.ff_norm.eps,
                        'wq': b.attn.W_q.weight.data,
                        'bq': b.attn.W_q.bias.data if b.attn.W_q.use_bias else None,
                        'wk': b.attn.W_k.weight.data,
                        'bk': b.attn.W_k.bias.data if b.attn.W_k.use_bias else None,
                        'wv': b.attn.W_v.weight.data,
                        'bv': b.attn.W_v.bias.data if b.attn.W_v.use_bias else None,
                        'wo': b.attn.W_o.weight.data,
                        'bo': b.attn.W_o.bias.data if b.attn.W_o.use_bias else None,
                        'w1': b.ff.w1.weight.data,
                        'b1': b.ff.w1.bias.data if b.ff.w1.use_bias else None,
                        'w2': b.ff.w2.weight.data,
                        'b2': b.ff.w2.bias.data if b.ff.w2.use_bias else None,
                        'w3': b.ff.w3.weight.data,
                        'b3': b.ff.w3.bias.data if b.ff.w3.use_bias else None,
                        'n_heads': b.attn.n_heads,
                        'n_kv_heads': b.attn.n_kv_head,
                        'head_dim': b.attn.head_dim,
                    })

            tok_emb_w = m.layers[0].weight.data
            pos_emb_w = m.pos_emb.weight.data if m.pos_emb is not None else None
            pos_emb_n = m.pos_emb.num_embeddings if m.pos_emb is not None else 0
            norm_layer = m.layers[-2]
            SloLN = _get_slo_layernorm()
            norm_has_bias = isinstance(norm_layer, SloLN)
            norm_w = norm_layer.weight.data
            norm_b = norm_layer.bias.data if norm_has_bias else None
            norm_eps = norm_layer.eps
            lm_w = m.layers[-1].weight.data

            E = blocks[0]['head_dim']
            H = blocks[0]['n_heads']
            K_H = blocks[0]['n_kv_heads']
            scale = 1.0 / math.sqrt(E)

            # Pre-allocate KV cache
            kv_buf_k = [None] * len(blocks)
            kv_buf_v = [None] * len(blocks)
            kv_len = [0] * len(blocks)

            def _forward_step(idx, pos, step):
                nonlocal kv_buf_k, kv_buf_v, kv_len
                B = 1
                seq_len = idx.shape[1]
                clipped = _np.clip(idx.astype(_np.int64), 0, tok_emb_w.shape[0] - 1)
                x = _np.take(tok_emb_w, clipped, axis=0)
                if pos_emb_w is not None:
                    p = _np.arange(pos, pos + seq_len, dtype=_np.int64).reshape(1, -1)
                    x = x + _np.take(pos_emb_w, _np.clip(p, 0, pos_emb_n - 1), axis=0)

                for bi, bw in enumerate(blocks):
                    mean = x.mean(axis=-1, keepdims=True)
                    var = x.var(axis=-1, keepdims=True)
                    h = (x - mean) / _np.sqrt(var + bw['an_eps']) * bw['an_w']
                    if bw['an_b'] is not None: h = h + bw['an_b']

                    q = h @ bw['wq'].T
                    if bw['bq'] is not None: q = q + bw['bq']
                    k = h @ bw['wk'].T
                    if bw['bk'] is not None: k = k + bw['bk']
                    v = h @ bw['wv'].T
                    if bw['bv'] is not None: v = v + bw['bv']

                    q = q.reshape(B, seq_len, H, E)
                    k = k.reshape(B, seq_len, K_H, E)
                    v = v.reshape(B, seq_len, K_H, E)

                    new_len = kv_len[bi] + seq_len
                    if kv_buf_k[bi] is None or new_len > kv_buf_k[bi].shape[1]:
                        cap = max(64, new_len * 2)
                        new_buf_k = _np.zeros((B, cap, K_H, E), dtype=k.dtype)
                        new_buf_v = _np.zeros((B, cap, K_H, E), dtype=v.dtype)
                        if kv_buf_k[bi] is not None:
                            old_len = kv_len[bi]
                            new_buf_k[:, :old_len] = kv_buf_k[bi][:, :old_len]
                            new_buf_v[:, :old_len] = kv_buf_v[bi][:, :old_len]
                        kv_buf_k[bi] = new_buf_k
                        kv_buf_v[bi] = new_buf_v
                    kv_buf_k[bi][:, kv_len[bi]:kv_len[bi]+seq_len] = k
                    kv_buf_v[bi][:, kv_len[bi]:kv_len[bi]+seq_len] = v
                    kv_len[bi] = new_len
                    k = kv_buf_k[bi][:, :new_len]
                    v = kv_buf_v[bi][:, :new_len]

                    if K_H < H:
                        reps = H // K_H
                        k = _np.repeat(k, reps, axis=2)
                        v = _np.repeat(v, reps, axis=2)

                    scores = _np.einsum('bnhd,bmhd->bhnm', q, k) * scale
                    if step == 0 and seq_len > 1:
                        causal = _np.triu(_np.full((seq_len, seq_len), -1e9, dtype=_np.float32), k=1)
                        scores = scores + causal
                    attn = _np.exp(scores - scores.max(axis=-1, keepdims=True))
                    attn = attn / attn.sum(axis=-1, keepdims=True)
                    ao = _np.einsum('bhnm,bmhd->bnhd', attn, v).reshape(B, seq_len, H * E)

                    ao = ao @ bw['wo'].T
                    if bw['bo'] is not None: ao = ao + bw['bo']
                    x = x + ao

                    mean = x.mean(axis=-1, keepdims=True)
                    var = x.var(axis=-1, keepdims=True)
                    h = (x - mean) / _np.sqrt(var + bw['fn_eps']) * bw['fn_w']
                    if bw['fn_b'] is not None: h = h + bw['fn_b']
                    h1 = h @ bw['w1'].T
                    if bw['b1'] is not None: h1 = h1 + bw['b1']
                    h3 = h @ bw['w3'].T
                    if bw['b3'] is not None: h3 = h3 + bw['b3']
                    h1 = 0.5 * h1 * (1.0 + _np.tanh(0.7978845608 * (h1 + 0.044715 * h1**3)))
                    h = h1 * h3
                    h = h @ bw['w2'].T
                    if bw['b2'] is not None: h = h + bw['b2']
                    x = x + h

                mean = x.mean(axis=-1, keepdims=True)
                var = x.var(axis=-1, keepdims=True)
                x = (x - mean) / _np.sqrt(var + norm_eps) * norm_w
                if norm_has_bias: x = x + norm_b
                logits = x[:, -1, :] @ lm_w.T
                return logits

            # Step 0: pre-fill full prompt
            logits = _forward_step(tokens[:, -m.block_size:], 0, step=0)

            # NaN/Inf guard on first logits
            if not _np.all(_np.isfinite(logits)):
                raise RuntimeError(f"Model produced non-finite logits (NaN/Inf) in pre-fill step. "
                                   f"logits range: [{_np.min(logits):.4f}, {_np.max(logits):.4f}]")

            generated_so_far = _np.array([], dtype=_np.int64)
            next_id = _sample_from_logits(
                logits, temperature=temperature,
                top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_ids=generated_so_far,
                eos_token=eos_id,
            )
            tokens = _np.concatenate([tokens, _np.array([[next_id]], dtype=_np.int64)], axis=1)
            generated_so_far = _np.append(generated_so_far, next_id)
            yield self._tokenizer.decode([next_id])

            # Steps 1..max_tokens: single-token forward with KV cache
            for step in range(1, max_tokens):
                # Cancel check (threading.Event is safe to poll from any thread)
                if cancel_event and cancel_event.is_set():
                    return

                if next_id == eos_id:
                    break
                pos = tokens.shape[1] - 1
                logits = _forward_step(tokens[:, -1:], pos, step=step)

                # NaN/Inf guard — stop generation if logits are corrupt
                if not _np.all(_np.isfinite(logits)):
                    raise RuntimeError(
                        f"Model produced non-finite logits at step {step} "
                        f"(prompt_len={prompt_len}, generated={len(generated_so_far)}). "
                        f"logits range: [{_np.min(logits):.4f}, {_np.max(logits):.4f}]"
                    )

                next_id = _sample_from_logits(
                    logits, temperature=temperature,
                    top_k=top_k, top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    generated_ids=generated_so_far,
                    eos_token=eos_id if step < max_tokens - 1 else None,
                )
                tokens = _np.concatenate([tokens, _np.array([[next_id]], dtype=_np.int64)], axis=1)
                generated_so_far = _np.append(generated_so_far, next_id)
                yield self._tokenizer.decode([next_id])

        # ── Robust streaming pipeline ──
        # Producer thread feeds tokens into queue; consumer yields from queue.
        # Errors propagate via a separate _error queue.
        import queue
        q = queue.Queue()
        err_q = queue.Queue()
        sentinel = object()
        _GENERATION_TIMEOUT_S = 120.0

        def _producer():
            try:
                for token in _stream_generate():
                    q.put(token)
            except Exception as e:
                err_q.put(e)
            finally:
                q.put(sentinel)

        producer_thread = threading.Thread(target=_producer, daemon=True)
        producer_thread.start()

        gen_start = time.monotonic()
        while True:
            # Check total generation timeout
            elapsed = time.monotonic() - gen_start
            if elapsed > _GENERATION_TIMEOUT_S:
                cancel_event.set() if cancel_event else None
                logger.warning("Streaming generation timed out after %.0fs", elapsed)
                yield "\n\n[Generation timed out after {:.0f}s]".format(elapsed)
                return

            try:
                # Use to_thread with timeout to prevent indefinite blocking
                token = await asyncio.wait_for(
                    asyncio.to_thread(q.get),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                # Check if producer thread is still alive
                if not producer_thread.is_alive():
                    # Thread died — drain remaining tokens
                    while not q.empty():
                        t = q.get_nowait()
                        if t is sentinel:
                            break
                        yield t
                    break
                # Thread still alive but no token for 30s — check for errors
                if not err_q.empty():
                    exc = err_q.get_nowait()
                    yield "\n\n[Generation error: {}]".format(exc)
                    return
                # Still generating — continue waiting
                continue

            if token is sentinel:
                break

            # Check for producer errors (may have been raised between queue reads)
            if not err_q.empty():
                exc = err_q.get_nowait()
                yield "\n\n[Generation error: {}]".format(exc)
                return

            yield token
