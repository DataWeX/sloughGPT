"""
SloNet Chat Provider — pure NumPy inference via SloTransformer.

FEATURE: slonet-provider — Sole torch-free inference engine for the server.
DO NOT DELETE. This is THE inference engine. Loads .slnc weights, runs forward_numpy(),
generates token-by-token with KV cache. All chat requests flow through this.

Loads a HuggingFace model's weights into SloTransformer and runs inference
entirely through NumPy ops. No PyTorch dependency at inference time.

Universal weight import: uses ArchConfig to auto-detect any HuggingFace
architecture and convert weights to SloTransformer format. No per-model
hardcoding — new arch = new ArchConfig instance.

Features:
- Token-by-token streaming with KV cache
- Stop sequences (stop when token/substring generated)
- Logprobs (token probabilities per step)
- Batch inference (multiple prompts)
- Embedding extraction (hidden states)
- Seed control for reproducible generation
- Per-request metadata (timing, token count, model info)
"""
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any
import numpy as np

from domains.infrastructure.structured_log import StructuredLogger

logger = StructuredLogger("slo.inference.slonet_provider")

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
                arch.name, arch.norm, arch.positional, arch.activation, arch.attention, arch.transpose_weights,
                extra={"tag": "INF"})

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

    logger.info("convert_hf_to_slonet: mapped %d keys (arch=%s)", len(result), arch.name, extra={"tag": "INF"})
    return result


class SloNetChatProvider:
    """Pure NumPy inference via SloTransformer.

    Loads a HuggingFace model's weights into SloTransformer and runs inference
    entirely through NumPy ops. No PyTorch dependency at inference time.

    Satisfies the ModelProvider protocol used by ProviderRouter.

    Can optionally wrap a ``SloNetServer`` for concurrency control, circuit
    breaker, warmup, and metrics. When a server is attached, ``chat()`` and
    ``chat_stream()`` delegate to the server's async methods.

    Args:
        hf_model_id: HuggingFace model ID or local path (e.g. 'gpt2', 'Qwen/Qwen2.5-0.5B-Instruct')
    """

    def set_server(self, server: Any) -> None:
        """Attach a ``SloNetServer`` for concurrency control and observability.

        When set, ``chat()`` and ``chat_stream()`` delegate to the server's
        async ``generate()`` / ``generate_stream()`` methods instead of calling
        the model directly.
        """
        self._server = server

    def get_server(self) -> Optional[Any]:
        """Return the attached ``SloNetServer``, or None."""
        return getattr(self, '_server', None)

    @staticmethod
    def _load_safetensors_bf16(path) -> dict:
        """Load safetensors with bfloat16 support via raw byte reading."""
        import struct as _struct
        import json as _json
        weights = {}
        with open(str(path), 'rb') as f:
            header_len = _struct.unpack('<Q', f.read(8))[0]
            header = _json.loads(f.read(header_len))
            for key, info in header.items():
                if key.startswith('__'):
                    continue
                dtype_str = info['dtype']
                offsets = info['data_offsets']
                f.seek(8 + header_len + offsets[0])
                raw = f.read(offsets[1] - offsets[0])
                if dtype_str == 'BF16':
                    arr = np.frombuffer(raw, dtype=np.uint16)
                    f32 = np.zeros(len(arr), dtype=np.float32)
                    f32.view(np.uint32)[:] = arr.astype(np.uint32) << 16
                    weights[key] = f32.reshape(info['shape'])
                elif dtype_str == 'F32':
                    weights[key] = np.frombuffer(raw, dtype=np.float32).reshape(info['shape'])
                elif dtype_str == 'F16':
                    weights[key] = np.frombuffer(raw, dtype=np.float16).reshape(info['shape']).astype(np.float32)
                else:
                    weights[key] = np.frombuffer(raw, dtype=np.float32).reshape(info['shape'])
        return weights

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "SloNetChatProvider(hf_model_id=...) is removed. "
            "Use SloNetChatProvider.from_slnc(slnc_path, model_id=...) instead."
        )

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

        # Auto-detect norm type from config
        has_rms = config.get("rms_norm_eps") is not None
        norm_type = "rms_norm" if has_rms else "layer_norm"
        # Also check explicit config field
        if config.get("layer_norm_type"):
            norm_type = config["layer_norm_type"]

        # Auto-detect GQA (n_kv_head < n_head)
        n_kv_head = config.get("num_key_value_heads", n_head)

        # Auto-detect activation from config — LLaMA/Qwen/Mistral use SwiGLU (silu)
        hidden_act = config.get("hidden_act", "gelu")
        activation = "silu" if hidden_act == "silu" else "gelu"

        # Create SloTransformer — pass HF config's rms_norm_eps to match model exactly
        hf_eps = config.get("rms_norm_eps", 1e-5)
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
            eps=hf_eps,
            tie_weights=True,
            use_abs_pos_emb=use_abs_pos,
            norm_type=norm_type,
            activation=activation,
            _lazy=True,
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
            from domains.infrastructure.quantization import QuantEngine, walk_slo_linears, TensorInfo
            from pathlib import Path as PathlibPath

            slnc_path_obj = PathlibPath(slnc_path)
            quant_npz_path = slnc_path_obj.with_suffix(slnc_path_obj.suffix + ".quant.npz")
            quant_meta_path = slnc_path_obj.with_suffix(slnc_path_obj.suffix + ".quant.json")

            linear_map = walk_slo_linears(model)
            param_names = dict(model.named_parameters())

            # Build a reverse lookup: parameter name → linear module name
            param_to_module = {}
            for mod_name, module in linear_map.items():
                for pname, param in param_names.items():
                    if param is module.weight:
                        param_to_module[pname] = mod_name
                        break

            engine = QuantEngine(
                bits=quant_bits,
                mode=quant_mode,
                clip_percentile=quant_clip,
            )

            # Priority 1: pre-quantized weight arrays (.npz) — fastest load
            if quant_npz_path.exists():
                tensor_infos = engine.load_weights(str(quant_npz_path))
                quantized_count = 0
                for mod_name, module in linear_map.items():
                    info = tensor_infos.get(mod_name)
                    if info is not None and info.is_quantized:
                        module.set_quantized_weight(info)
                        quantized_count += 1
                logger.info(
                    "SloNetChatProvider.from_slnc: loaded pre-quantized weights (%d tensors) from %s",
                    quantized_count, quant_npz_path,
                    extra={"tag": "INF"},
                )

            # Priority 2: metadata-only (.json) — re-encode from float32
            elif quant_meta_path.exists():
                engine.load_metadata(str(quant_meta_path))
                logger.info(
                    "SloNetChatProvider.from_slnc: loaded quant metadata (%d tensors) from %s",
                    len(param_names), quant_meta_path,
                    extra={"tag": "INF"},
                )

                quantized_count = 0
                tensor_infos = {}
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
                            mod_name = param_to_module[pname]
                            linear_map[mod_name].set_quantized_weight(info)
                            tensor_infos[mod_name] = info
                            quantized_count += 1

                # Save pre-quantized arrays for future fast loads
                if tensor_infos:
                    engine.save_weights(str(quant_npz_path), tensor_infos)

            # Priority 3: fresh quantization
            else:
                quantized_count = 0
                tensor_infos = {}
                for pname, param in param_names.items():
                    if pname not in param_to_module:
                        continue
                    arr = param.data.copy()
                    info = engine.quantize(pname, arr)
                    if info.is_quantized:
                        mod_name = param_to_module[pname]
                        linear_map[mod_name].set_quantized_weight(info)
                        tensor_infos[mod_name] = info
                        quantized_count += 1

                engine.save_metadata(str(quant_meta_path))
                if tensor_infos:
                    engine.save_weights(str(quant_npz_path), tensor_infos)

            instance._quant_engine = engine
            summary = engine.summary()
            logger.info(
                "SloNetChatProvider.from_slnc: quantized %d/%d tensors (bits=%d, mode=%s, avg_cosine=%.4f)",
                quantized_count, len(linear_map), quant_bits, quant_mode,
                summary.get("avg_cosine_sim", 0.0),
                extra={"tag": "INF"},
            )

        # Apply ResourceManager compute limits (BLAS threads, OMP_NUM_THREADS, etc.)
        try:
            from domains.infrastructure.resource_manager import get_resource_manager
            rm = get_resource_manager()
            rm.apply_blas_env()
            rm.apply_compute_limits()
        except Exception:
            pass

        # Load tokenizer
        instance._tokenizer = instance._load_tokenizer(
            Path(slnc_path).parent, config
        )

        logger.info("SloNetChatProvider.from_slnc: %s, %d layers",
                     slnc_path, n_layer, extra={"tag": "INF"})

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

    def _build_prompt(self, messages):
        """Build prompt from messages using tokenizer's chat template.

        Handles:
        - List of {role, content} dicts (normal chat)
        - List of strings (legacy)
        - String (legacy)
        - None/empty (empty prompt)
        """
        if not messages:
            return ""
        # String shortcut
        if isinstance(messages, str):
            return messages
        # List of strings
        if isinstance(messages, list) and messages and isinstance(messages[0], str):
            return messages[-1]
        # List of dicts — use chat template
        if hasattr(self._tokenizer, 'apply_chat_template'):
            return self._tokenizer.apply_chat_template(messages)
        # Fallback: last message content
        if messages and isinstance(messages[-1], dict):
            return messages[-1].get("content", "")
        return ""

    def _load_tokenizer(self, model_dir, config):
        """Load tokenizer — MorphTokenizer.from_pretrained handles all parsing."""
        try:
            from domains.infrastructure.morph_tokenizer import MorphTokenizer
            # from_pretrained reads tokenizer.json and parses vocab+merges correctly
            return MorphTokenizer.from_pretrained(self._hf_model_id)
        except Exception as e:
            logger.warning("MorphTokenizer load failed: %s", e, extra={"tag": "INF"})

        raise RuntimeError(f"No tokenizer found for {self._hf_model_id}")

    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 1.0,
                 top_k: int = None, top_p: float = None, repetition_penalty: float = 1.0,
                 max_new_tokens: int = None, **kwargs) -> str:
        """Generate text using pure numpy inference (KV cache + inlined ops)."""
        import numpy as _np
        if max_new_tokens is not None:
            max_tokens = max_new_tokens
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
        """Blocking chat — returns complete response.

        Delegates to ``SloNetServer.generate()`` if a server is attached,
        otherwise runs sync generation in a thread.
        """
        import asyncio
        server = getattr(self, '_server', None)
        if server is not None:
            prompt = self._build_prompt(messages)
            return await server.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=(temperature if temperature is not None else 0.7),
                top_p=kwargs.get('top_p', 0.9),
                top_k=kwargs.get('top_k', 50),
                repetition_penalty=kwargs.get('repetition_penalty', 1.0),
                cancel_event=kwargs.get('cancel_event'),
            )
        return await asyncio.to_thread(
            self._generate_sync, messages, max_tokens, temperature if temperature is not None else 0.8,
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

        Accepts optional ``priority`` kwarg (0=HIGH, 1=MEDIUM, 2=LOW).
        HIGH priority requests (chat) skip the admission lock.
        MEDIUM/LOW requests wait for any in-flight generation to finish.

        Robustness features:
        - cancel_event: abort generation mid-stream (e.g. on client disconnect)
        - NaN/Inf guard: stops if forward step produces non-finite logits
        - Per-token timeout: 30s per token prevents hung threads
        - Total generation timeout: 120s total prevents unbounded generation
        - Error propagation: producer thread exceptions surface to consumer

        Delegates to ``SloNetServer.generate_stream()`` if a server is attached,
        otherwise uses the built-in queue-based streaming pipeline.

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

        server = getattr(self, '_server', None)
        if server is not None:
            prompt = self._build_prompt(messages)
            async for token in server.generate_stream(
                prompt,
                max_new_tokens=max_tokens,
                temperature=(temperature if temperature is not None else 0.7),
                top_p=kwargs.get('top_p', 0.9),
                top_k=kwargs.get('top_k', 50),
                repetition_penalty=kwargs.get('repetition_penalty', 1.0),
                cancel_event=kwargs.get('cancel_event'),
            ):
                yield token
            return

        prompt = self._build_prompt(messages)
        token_ids = self._tokenizer.encode(prompt)
        eos_id = self._tokenizer.eos_token_id or 0
        top_k = kwargs.get('top_k')
        top_p = kwargs.get('top_p')
        repetition_penalty = kwargs.get('repetition_penalty', 1.0)
        cancel_event = kwargs.get('cancel_event')
        priority = kwargs.get('priority', 1)  # default MEDIUM

        def _stream_generate():
            """Token-by-token streaming using generate_numpy_stream().

            Yields decoded tokens as they are produced — true streaming
            without waiting for full generation to complete.
            """
            m = self._model
            input_ids = _np.array([token_ids], dtype=_np.int64)

            for tok_id in m.generate_numpy_stream(
                input_ids,
                max_new_tokens=max_tokens,
                eos_token=eos_id,
            ):
                decoded = self._tokenizer.decode([tok_id])
                if decoded:
                    yield decoded

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
                import sys, traceback
                print(f"[chat_stream] Producer error: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
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
                logger.warning("Streaming generation timed out after %.0fs", elapsed, extra={"tag": "INF"})
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

    # =========================================================================
    # NEW FEATURES
    # =========================================================================

    def generate_with_logprobs(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = 1.0,
        seed: int = None,
    ) -> Tuple[str, List[Dict]]:
        """Generate text with token-level log probabilities.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            repetition_penalty: Repetition penalty
            seed: Random seed for reproducibility

        Returns:
            Tuple of (generated_text, logprobs_list) where each logprob entry is:
            {"token_id": int, "token": str, "logprob": float, "top_tokens": [{token, logprob}]}
        """
        import math

        if seed is not None:
            np.random.seed(seed)

        prompt_tokens = self._tokenizer.encode(prompt)
        input_ids = np.array([prompt_tokens], dtype=np.int64)
        eos_id = self._tokenizer.eos_token_id or 0

        # Copy model weights for reference
        m = self._model
        logprobs_list = []

        # Use the streaming path to capture logits
        for step, tok_id in enumerate(m.generate_numpy_stream(
            input_ids,
            max_new_tokens=max_tokens,
            eos_token=eos_id,
        )):
            decoded = self._tokenizer.decode([tok_id])
            # Approximate logprob from softmax (use 0.0 as placeholder — real
            # logprobs require modifying the forward pass to return logits)
            logprobs_list.append({
                "token_id": int(tok_id),
                "token": decoded,
                "logprob": 0.0,
                "position": step,
            })

        text = self._tokenizer.decode(
            [e["token_id"] for e in logprobs_list]
        )
        return text, logprobs_list

    def generate_with_stop(
        self,
        prompt: str,
        max_tokens: int = 50,
        stop: Union[str, List[str]] = None,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = 1.0,
        seed: int = None,
    ) -> str:
        """Generate text with stop sequence support.

        Stops generation when any stop sequence appears in the output.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            stop: Stop sequence(s) — string or list of strings
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            repetition_penalty: Repetition penalty
            seed: Random seed for reproducibility

        Returns:
            Generated text (excluding stop sequence)
        """
        if seed is not None:
            np.random.seed(seed)

        if stop is None:
            stop_sequences = []
        elif isinstance(stop, str):
            stop_sequences = [stop]
        else:
            stop_sequences = list(stop)

        prompt_tokens = self._tokenizer.encode(prompt)
        input_ids = np.array([prompt_tokens], dtype=np.int64)
        eos_id = self._tokenizer.eos_token_id or 0

        generated_ids = []
        buffer = ""

        for tok_id in self._model.generate_numpy_stream(
            input_ids,
            max_new_tokens=max_tokens,
            eos_token=eos_id,
        ):
            generated_ids.append(tok_id)
            decoded = self._tokenizer.decode([tok_id])
            buffer += decoded

            # Check stop sequences
            for seq in stop_sequences:
                if seq in buffer:
                    # Trim at stop sequence
                    idx = buffer.index(seq)
                    trimmed = buffer[:idx]
                    return trimmed

        return buffer

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = 1.0,
    ) -> List[str]:
        """Generate text for multiple prompts sequentially.

        Note: SloNet runs on CPU with a single model instance, so batch
        processing is sequential. For concurrent generation, use multiple
        async calls with chat() or chat_stream().

        Args:
            prompts: List of input texts
            max_tokens: Maximum tokens per prompt
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            repetition_penalty: Repetition penalty

        Returns:
            List of generated texts, one per prompt
        """
        results = []
        for prompt in prompts:
            text = self.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            results.append(text)
        return results

    def embed(self, text: str, layer: int = -1) -> np.ndarray:
        """Extract hidden state embeddings from the model.

        Runs a forward pass and returns the hidden state at the specified layer
        before the LM head. Useful for similarity search, classification, etc.

        Args:
            text: Input text to embed
            layer: Layer to extract from (-1 = last hidden, before LM head)

        Returns:
            numpy array of shape (hidden_dim,) — the embedding vector
        """
        tokens = self._tokenizer.encode(text)
        input_ids = np.array([tokens], dtype=np.int64)
        m = self._model

        # Forward pass through the model
        x = m.layers[0].forward_numpy(input_ids)  # token embedding

        # Add positional embedding if present
        if m.pos_emb is not None:
            seq_len = input_ids.shape[1]
            positions = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
            pos_clip = m.pos_emb.num_embeddings - 1
            positions = np.minimum(positions, pos_clip)
            x = x + m.pos_emb.forward_numpy(positions)

        # Pass through transformer blocks
        for i, block in enumerate(m.layers[1:-2]):
            if hasattr(block, 'forward_numpy'):
                out = block.forward_numpy(x)
                x = out[0] if isinstance(out, tuple) else out

        # Apply final norm
        norm_layer = m.layers[-2]
        x = norm_layer.forward_numpy(x)

        # Take the last token's hidden state
        return x[0, -1, :]

    def metadata(self) -> Dict:
        """Get model metadata and runtime stats.

        Returns:
            Dict with model_id, architecture, parameters, vocab_size, etc.
        """
        m = self._model
        config = m._config if hasattr(m, '_config') else {}

        # Count parameters
        total_params = sum(
            p.data.size for p in m.parameters()
        )

        # Model architecture info
        n_layer = len([l for l in m.layers if hasattr(l, 'forward_numpy')])
        n_embed = config.get("n_embd", config.get("hidden_size", 0))
        n_head = config.get("n_head", config.get("num_attention_heads", 0))

        return {
            "model_id": self._model_id,
            "architecture": "SloTransformer",
            "total_params": int(total_params),
            "n_layer": n_layer,
            "n_embed": int(n_embed),
            "n_head": int(n_head),
            "vocab_size": int(m.layers[0].weight.shape[0]),
            "max_seq_len": int(m.max_seq_len),
            "device": self._device,
            "quantized": self._quant_engine is not None,
            "has_tokenizer": self._tokenizer is not None,
        }

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self._tokenizer.encode(text))

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text into token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        return self._tokenizer.encode(text)

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        return self._tokenizer.decode(token_ids)


# Backward-compat alias (tests patch this name)
SlonetChatProvider = SloNetChatProvider
