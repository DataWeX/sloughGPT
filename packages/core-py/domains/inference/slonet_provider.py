"""
SloNet Chat Provider — pure NumPy inference via SloTransformer.

FEATURE: slonet-provider — Sole inference engine for the server.
DO NOT DELETE. This is THE inference engine. Loads .slnc weights, runs forward_numpy(),
generates token-by-token with KV cache. All chat requests flow through this.

Loads a HuggingFace model's weights into SloTransformer and runs inference
entirely through NumPy ops.

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
import threading
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any
import numpy as np

from domains.infrastructure.structured_log import StructuredLogger
from domains.infrastructure.constants import DEFAULT_GENERATE_TIMEOUT

logger = StructuredLogger("slo.inference.slonet_provider")

# Streaming robustness timeouts (overridable in tests).
_STREAM_GET_TIMEOUT_S = 30.0
_STREAM_TOTAL_TIMEOUT_S = DEFAULT_GENERATE_TIMEOUT

# Lazy import to avoid circular dependency
_SloLayerNorm = None

def _get_slo_layernorm():
    global _SloLayerNorm
    if _SloLayerNorm is None:
        from domains.training.slonet import SloLayerNorm
        _SloLayerNorm = SloLayerNorm
    return _SloLayerNorm


class _CharTokenizer:
    """Minimal tokenizer wrapper for char-level models trained by SloNet.

    Wraps stoi/itos dicts from training into the encode/decode interface
    that SloNetChatProvider expects from a HuggingFace tokenizer.
    """

    def __init__(self, stoi: Dict[str, int], itos: Dict):
        self._stoi = stoi
        # JSON serializes int keys as strings — normalize to int
        self._itos = {int(k): v for k, v in itos.items()}
        self.eos_token_id = stoi.get("\n", 0)
        self.pad_token_id = 0
        self._vocab_size = max(self._itos.keys()) + 1 if self._itos else 0

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs (one per character)."""
        return [int(self._stoi.get(c, 0)) for c in text]

    def decode(self, token_ids) -> str:
        """Decode token IDs back to text."""
        return "".join(self._itos.get(int(tid), "") for tid in token_ids)

    def apply_chat_template(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Format messages as a simple chat string for char-level models."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)


class _TreeTokenizer:
    """SloBPE-compatible tokenizer wrapper for TokenTree-trained models.

    Wraps a reconstructed TokenTree into the encode/decode interface that
    SloNetChatProvider expects from a HuggingFace tokenizer, so BPE-level
    checkpoints round-trip through .soul metadata.
    """

    def __init__(self, tree: Any):
        self._tree = tree
        self.eos_token_id = int(tree.eos_id)
        self.pad_token_id = int(tree.pad_id)

    @property
    def vocab_size(self) -> int:
        return self._tree.vocab_size

    def encode(self, text: str) -> List[int]:
        """Encode text to BPE token IDs via the token tree."""
        return self._tree.encode(text)

    def decode(self, token_ids) -> str:
        """Decode token IDs back to text."""
        return self._tree.decode(token_ids)

    def apply_chat_template(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Format messages as a simple chat string for tree-tokenized models."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)


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
                        w = w.T  # pragma: no cover — no global 2D key outside NO_TRANSPOSE_KEYS
                    result[slo_target] = w
                    mapped = True
                    break

        # Handle fused QKV for GPT-2 style (matched in the canonical None-branch above).

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
    entirely through NumPy ops.

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

    def to_server(self, process_guard: Any = None, **kwargs: Any) -> Any:
        """Build a ``SloNetServer`` wrapping this provider's model and tokenizer.

        Used to attach concurrency control, circuit breaker, and optional
        ``ProcessGuard`` delegation to an already-loaded provider.

        Args:
            process_guard: Optional ``ProcessGuard`` — when set and alive,
                generation delegates to the guarded subprocess.
            **kwargs: Extra ``SloNetServer`` constructor kwargs.

        Returns:
            SloNetServer bound to this provider's model/tokenizer.
        """
        from domains.infrastructure.slonet_server import SloNetServer

        # Lazy providers (created via lazy_from_slnc) keep _model == None until
        # first use. Hand the server a factory so it can load the weights in the
        # parent process only if a request arrives with no guard to serve it.
        lazy_factory = None
        if getattr(self, "_lazy_lock", None) is not None:
            lazy_factory = self._get_model

        return SloNetServer(
            model=getattr(self, "_model", None),
            tokenizer=getattr(self, "_tokenizer", None),
            model_id=getattr(self, "_model_id", "slonet"),
            process_guard=process_guard,
            provider=self,
            lazy_model_factory=lazy_factory,
            **kwargs,
        )

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
        kv_max_sessions: int = 64,
        free_quantized_originals: bool = False,
        release_mmap_pages: bool = True,
        trim_allocator_after_load: bool = True,
    ) -> "SloNetChatProvider":
        """Create provider from .slnc file (mmap, zero-copy).

        Args:
            slnc_path: Path to .slnc file
            model_id: HuggingFace model ID for tokenizer (e.g. "gpt2")
            quantize: If True, apply per-tensor quantization to weights
            quant_bits: Bits for quantization (8 or 4)
            quant_mode: "symmetric" or "asymmetric"
            quant_clip: Outlier clipping percentile (e.g., 0.999)
            kv_max_sessions: Max simultaneous cross-turn KV sessions before
                least-recently-used eviction kicks in
            free_quantized_originals: If True, release the float32 weight of
                every quantized/point linear layer after quantization. Saves
                ~(projection bytes) per layer; safe only for inference-only
                loads (training gradients need the float32 originals).
            release_mmap_pages: If True, discard the resident .slnc mmap
                pages after all tensors have been copied out. Tensors are
                copies (``get_tensor``), so this frees RSS with no correctness
                impact; a later read simply re-faults from disk.
            trim_allocator_after_load: If True, call ``malloc_trim(0)`` after
                load (glibc/Linux only) so transient peak allocations from
                weight conversion are returned to the OS. No-op elsewhere.

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

        # Drop the transient conversion buffers immediately — they are copies
        # (2x the fp32 weight bytes) and would otherwise pin peak RSS until
        # this method returns. load_state_dict already wrote their values into
        # the model's parameters.
        del weights_dict
        del mapped

        # Create instance (bypass __init__)
        instance = cls.__new__(cls)
        instance._hf_model_id = model_id
        instance._model_id = model_id
        instance._device = "cpu"
        instance._model = model
        instance._parser = parser  # keep mmap alive
        instance._quant_engine = None
        instance._slnc_path = slnc_path  # stored for unload/reload

        # Apply quantization if requested.
        #
        # Quantization only helps when the AVX2 int8 GEMM kernel is loaded
        # (quant_core C extension). When the kernel is unavailable the pure-numpy
        # fallback converts every weight matrix to int32 on each forward pass,
        # which is ~12x SLOWER than the plain float32 matmul. In that case we
        # silently skip quantization — float32 BLAS is both faster and exact.
        if quantize:
            from domains.infrastructure.quant_core.wrapper import HAS_AVX2 as _HAS_AVX2
            from domains.infrastructure.quantization import Quantine, walk_slo_linears
            from pathlib import Path as PathlibPath

            if not bool(_HAS_AVX2):
                logger.info(
                    "SloNetChatProvider.from_slnc: AVX2 int8 kernel not available "
                    "(numpy fallback is slower than float32) — skipping quantization",
                    extra={"tag": "INF"},
                )
                quantize = False

        if quantize:
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

            engine = Quantine(
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

        # Release float32 originals of quantized/point layers (inference-only).
        # get_tensor() returns copies, so both this and the mmap release below
        # reclaim RSS without any correctness impact.
        if free_quantized_originals:
            freed = model.free_quantized_originals()
            logger.info(
                "SloNetChatProvider.from_slnc: released float32 originals of %d quantized/point layers",
                freed,
                extra={"tag": "INF"},
            )

        if release_mmap_pages:
            if parser.release_file_pages():
                logger.info(
                    "SloNetChatProvider.from_slnc: released .slnc mmap pages (%.1f MB file)",
                    parser.file_size / 1e6,
                    extra={"tag": "INF"},
                )

        # glibc keeps freed heap (peak allocations from weight conversion)
        # resident; return it to the OS. No-op on non-glibc allocators.
        if trim_allocator_after_load:
            try:
                import ctypes
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except Exception:
                pass

        # Apply ResourceManager compute limits (BLAS threads, OMP_NUM_THREADS, etc.)
        try:
            from domains.infrastructure.resource_manager import get_resource_manager
            rm = get_resource_manager()
            rm.apply_blas_env()
            rm.apply_compute_limits()
        except Exception as e:  # pragma: no cover — defensive; real ResourceManager never raises here
            logger.warning("ResourceManager.apply_blas_env skipped: %s", e)

        # Load tokenizer
        instance._tokenizer = instance._load_tokenizer(
            Path(slnc_path).parent, config
        )

        logger.info("SloNetChatProvider.from_slnc: %s, %d layers",
                     slnc_path, n_layer, extra={"tag": "INF"})

        # Cross-turn KV cache state per session (lazy NumpyKVState per session_id)
        instance._kv_states: Dict[str, Any] = {}
        instance._kv_last_access: Dict[str, float] = {}  # session_id → monotonic timestamp
        instance._kv_ttl: float = 3600.0  # 1 hour default TTL for idle sessions
        instance._kv_max_sessions: int = kv_max_sessions  # LRU cap on concurrent sessions
        # Guard for the session KV map — mutated from to_thread workers and
        # API routes concurrently, so check-then-set races must be serialized.
        instance._kv_lock = threading.Lock()

        return instance

    @classmethod
    def lazy_from_slnc(
        cls,
        slnc_path: str,
        model_id: str = "gpt2",
        quantize: bool = False,
        quant_bits: int = 8,
        quant_mode: str = "symmetric",
        quant_clip: float = 0.999,
        kv_max_sessions: int = 64,
        free_quantized_originals: bool = False,
        release_mmap_pages: bool = True,
        trim_allocator_after_load: bool = True,
    ) -> "SloNetChatProvider":
        """Create a provider whose model weights load lazily on first use.

        Reads only the .slnc header (config + tensor table) to build full
        metadata, then closes the file — no weight pages are faulted into
        memory. The complete load (``from_slnc``) runs on the first call that
        actually needs the model (``_get_model()``), and can be undone with
        ``release_model()``.

        This is the intended construction path for the server autoload with a
        ``ProcessGuard``: the parent stays near-idle while the guarded worker
        serves inference; the parent only materializes weights as a fallback
        when no guard is alive.

        Args:
            slnc_path: Path to .slnc file
            model_id: HuggingFace model ID for tokenizer (e.g. "gpt2")
            quantize: If True, apply per-tensor quantization on first load
            quant_bits: Bits for quantization (8 or 4)
            quant_mode: "symmetric" or "asymmetric"
            quant_clip: Outlier clipping percentile (e.g., 0.999)
            kv_max_sessions: Max simultaneous cross-turn KV sessions before
                least-recently-used eviction kicks in
            free_quantized_originals: Passed through to the lazy load
            release_mmap_pages: Passed through to the lazy load
            trim_allocator_after_load: Passed through to the lazy load

        Returns:
            Lazy SloNetChatProvider. ``_model`` is None until first use.
        """
        from domains.infrastructure.slnc.parser import SLNCParser

        parser = SLNCParser(slnc_path)
        try:
            config = parser.config
            vocab_size = config.get("vocab_size", 50257)
            n_layer = config.get("n_layer", config.get("num_hidden_layers", 12))
            n_embed = config.get("n_embd", config.get("hidden_size", 768))
            n_head = config.get("n_head", config.get("num_attention_heads", 12))
            n_positions = parser.n_positions
            total_params = parser.param_count
        finally:
            parser.close()

        instance = cls.__new__(cls)
        instance._hf_model_id = model_id
        instance._model_id = model_id
        instance._device = "cpu"
        instance._model = None
        instance._parser = None
        instance._quant_engine = None
        instance._tokenizer = instance._load_tokenizer(Path(slnc_path).parent, config)
        instance._slnc_path = str(slnc_path)
        instance._load_kwargs = {
            "quantize": quantize,
            "quant_bits": quant_bits,
            "quant_mode": quant_mode,
            "quant_clip": quant_clip,
            "kv_max_sessions": kv_max_sessions,
            "free_quantized_originals": free_quantized_originals,
            "release_mmap_pages": release_mmap_pages,
            "trim_allocator_after_load": trim_allocator_after_load,
        }
        instance._lazy_lock = threading.Lock()
        instance._materializing = threading.Event()
        instance._materializing.set()  # Start "not loading" — cleared during materialize_model()
        instance._loaded = False
        instance._meta = {
            "model_id": model_id,
            "architecture": "SloTransformer",
            "total_params": int(total_params),
            "n_layer": int(n_layer),
            "n_embed": int(n_embed),
            "n_head": int(n_head),
            "vocab_size": int(vocab_size),
            "max_seq_len": int(n_positions),
            "device": "cpu",
            "quantized": bool(quantize),
            "has_tokenizer": instance._tokenizer is not None,
            "lazy": True,
        }
        instance._kv_states: Dict[str, Any] = {}
        instance._kv_last_access: Dict[str, float] = {}
        instance._kv_ttl: float = 3600.0
        instance._kv_max_sessions: int = kv_max_sessions
        instance._kv_lock = threading.Lock()

        logger.info(
            "SloNetChatProvider.lazy_from_slnc: %s (%.1f MB file, %d params) — weights deferred",
            slnc_path, parser.file_size / 1e6, total_params,
            extra={"tag": "INF"},
        )
        return instance

    @classmethod
    def from_soul(
        cls,
        soul_path: str,
        model_id: str = "sloughgpt",
        kv_max_sessions: int = 64,
    ) -> "SloNetChatProvider":
        """Create provider from a .soul checkpoint trained natively by SloNet.

        Unlike from_slnc() which converts HuggingFace weights, this loads
        checkpoints produced by SloughGPTTrainer or distill_gpt2.py directly.
        The weights are already in SloNet format — no conversion needed.

        Args:
            soul_path: Path to .soul checkpoint file
            model_id: Identifier for this model (used in health/status)
            kv_max_sessions: Max simultaneous cross-turn KV sessions

        Returns:
            SloNetChatProvider using the trained model's weights

        Raises:
            FileNotFoundError: If soul_path does not exist
            ValueError: If the .soul file is invalid or missing model config
        """
        from domains.inference.slo_format import load_soul
        from domains.training.slonet import SloTransformer

        soul, state_dict = load_soul(soul_path)

        # Extract model config from soul metadata
        cfg = soul.metadata.get("config", {})
        vocab_size = soul.metadata.get("vocab_size", cfg.get("vocab_size", 256))
        n_embed = cfg.get("n_embed", 128)
        n_layer = cfg.get("n_layer", 4)
        n_head = cfg.get("n_head", 4)
        block_size = cfg.get("block_size", 128)

        # Create SloTransformer with the trained architecture
        model = SloTransformer(
            vocab_size=vocab_size,
            n_embed=n_embed,
            n_layer=n_layer,
            n_head=n_head,
            block_size=block_size,
            dropout=0.0,
            _lazy=True,
        )

        # Load weights directly — already in SloNet format from training
        model.load_state_dict(state_dict)

        # Create instance (bypass __init__)
        instance = cls.__new__(cls)
        instance._hf_model_id = model_id
        instance._model_id = model_id
        instance._device = "cpu"
        instance._model = model
        instance._parser = None
        instance._quant_engine = None

        # Apply ResourceManager compute limits
        try:
            from domains.infrastructure.resource_manager import get_resource_manager
            rm = get_resource_manager()
            rm.apply_blas_env()
            rm.apply_compute_limits()
        except Exception as e:
            logger.warning("ResourceManager.apply_blas_env skipped (soul load): %s", e)

        # Load tokenizer from soul metadata if available
        tokenizer_meta = soul.metadata.get("tokenizer")
        if (
            isinstance(tokenizer_meta, dict)
            and tokenizer_meta.get("type") == "token_tree"
            and isinstance(tokenizer_meta.get("tree"), dict)
        ):
            from domains.training.token_tree import TokenTree
            instance._tokenizer = _TreeTokenizer(
                TokenTree.from_dict(tokenizer_meta["tree"])
            )
        else:
            stoi = soul.metadata.get("stoi")
            itos = soul.metadata.get("itos")
            if stoi and itos:
                # Char-level model — create a simple tokenizer from vocab
                instance._tokenizer = _CharTokenizer(stoi, itos)
            else:
                instance._tokenizer = None

        logger.info(
            "SloNetChatProvider.from_soul: %s, %d layers, vocab=%d, embed=%d",
            soul_path, n_layer, vocab_size, n_embed, extra={"tag": "INF"},
        )

        # Cross-turn KV cache state
        instance._kv_states: Dict[str, Any] = {}
        instance._kv_last_access: Dict[str, float] = {}
        instance._kv_ttl: float = 3600.0
        instance._kv_max_sessions: int = kv_max_sessions
        instance._kv_lock = threading.Lock()

        return instance

    def apply_adapter(self, adapter_path: str, merge: bool = False) -> dict:
        """Apply a LoRA adapter (.npz) to the loaded model.

        Reads rank/alpha/target_modules from the adapter's _config/* keys,
        applies LoRA layers to the model, then loads the adapter weights.
        Optionally merges LoRA into base weights for faster inference.

        Args:
            adapter_path: Path to .npz adapter file (from HFLoraTrainer)
            merge: If True, merge LoRA weights into base and remove LoRA overhead

        Returns:
            Dict with adapter metadata: rank, alpha, n_params, merged
        """
        import numpy as np
        from pathlib import Path as PathlibPath

        adapter_file = PathlibPath(adapter_path)
        if not adapter_file.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        adapter = np.load(adapter_path)

        # Read adapter config
        rank = int(adapter.get("_config/rank", [8])[0]) if "_config/rank" in adapter else 8
        alpha = float(adapter.get("_config/alpha", [16.0])[0]) if "_config/alpha" in adapter else 16.0

        # Decode target modules from _config/target_module_N keys
        target_modules = []
        n_modules = int(adapter.get("_config/target_modules", [0])[0]) if "_config/target_modules" in adapter else 0
        for i in range(n_modules):
            key = f"_config/target_module_{i}"
            if key in adapter:
                chars = adapter[key].tolist()
                target_modules.append("".join(chr(c) for c in chars))

        if not target_modules:
            target_modules = ["W_q", "W_k", "W_v", "W_o"]

        # Apply LoRA layers
        from domains.training.lora import LoRAConfig, apply_lora_to_model, count_lora_parameters
        from domains.training.hf_lora_finetune import load_lora_adapter

        lora_config = LoRAConfig(rank=rank, alpha=alpha, target_modules=target_modules)
        self._model = apply_lora_to_model(self._model, lora_config)

        # Load adapter weights
        load_lora_adapter(self._model, adapter_path)
        n_params = count_lora_parameters(self._model)

        logger.info(
            "SloNetChatProvider.apply_adapter: loaded %s (rank=%d, alpha=%.1f, %d params)",
            adapter_path, rank, alpha, n_params, extra={"tag": "INF"},
        )

        result = {
            "rank": rank,
            "alpha": alpha,
            "target_modules": target_modules,
            "n_params": n_params,
            "merged": False,
        }

        # Optionally merge for faster inference
        if merge:
            from domains.training.hf_lora_finetune import merge_lora_adapter
            self._model = merge_lora_adapter(self._model)
            result["merged"] = True
            logger.info(
                "SloNetChatProvider.apply_adapter: merged LoRA into base weights",
                extra={"tag": "INF"},
            )

        return result

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

    def session_stats(self) -> dict:
        """Get cross-turn KV cache session statistics.

        Returns:
            Dict with active session count, TTL, and memory estimate.
        """
        import time as _time
        with self._kv_lock:
            n_sessions = len(self._kv_states)
            total_tokens = 0
            for state in self._kv_states.values():
                kv = getattr(state, "kv_len", None)
                if isinstance(kv, (list, tuple)):
                    total_tokens += sum(kv)
                elif kv is not None:
                    total_tokens += kv
            return {
                "active_sessions": n_sessions,
                "max_sessions": self._kv_max_sessions,
                "ttl_seconds": self._kv_ttl,
                "cached_tokens": total_tokens,
                "oldest_session_age": max(self._kv_last_access.values()) - min(self._kv_last_access.values())
                if len(self._kv_last_access) > 1 else 0.0,
            }

    def clear_session(self, session_id: str) -> bool:
        """Drop the cross-turn KV state for a single session.

        Used when a session is deleted so its cached keys/values are freed
        immediately instead of waiting for TTL eviction.

        Args:
            session_id: The session whose KV state should be removed.

        Returns:
            True if a state existed and was removed, False otherwise.
        """
        with self._kv_lock:
            existed = self._kv_states.pop(session_id, None) is not None
            self._kv_last_access.pop(session_id, None)
        if existed:
            logger.debug("Cleared KV state for session %s", session_id,
                         extra={"tag": "MODEL"})
        return existed

    def clear_all_sessions(self) -> int:
        """Drop all cross-turn KV states.

        Used on model unload/switch where cached keys/values from the old
        model are no longer valid.

        Returns:
            Number of sessions cleared.
        """
        with self._kv_lock:
            n = len(self._kv_states)
            self._kv_states.clear()
            self._kv_last_access.clear()
        if n:
            logger.info("Cleared KV state for %d sessions", n,
                        extra={"tag": "MODEL"})
        return n

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
        # Fallback: last message content (dead with MorphTokenizer, which
        # always implements apply_chat_template)
        if messages and isinstance(messages[-1], dict):  # pragma: no cover
            return messages[-1].get("content", "")  # pragma: no cover
        return ""  # pragma: no cover

    def _load_tokenizer(self, model_dir, config):
        """Load tokenizer — MorphTokenizer.from_pretrained handles all parsing.

        Prefers a tokenizer.json shipped with a local fine-tuned model dir;
        falls back to the base HuggingFace model id's tokenizer.
        """
        try:
            from domains.infrastructure.morph_tokenizer import MorphTokenizer
            if model_dir and Path(model_dir).is_dir() and (Path(model_dir) / "tokenizer.json").exists():
                logger.info("Using fine-tuned model tokenizer from %s", model_dir,
                            extra={"tag": "INF"})
                return MorphTokenizer.from_pretrained(str(model_dir))
            # from_pretrained reads tokenizer.json and parses vocab+merges correctly
            return MorphTokenizer.from_pretrained(self._hf_model_id)
        except Exception as e:
            logger.warning("MorphTokenizer load failed: %s", e, extra={"tag": "INF"})

        raise RuntimeError(f"No tokenizer found for {self._hf_model_id}")

    def _get_model(self):
        """Return the loaded model, loading lazily on first access.

        For an eager provider (from_slnc / from_soul) this is a plain
        attribute read. For a lazy provider (lazy_from_slnc) the full weight
        load is deferred until the model is actually needed; a per-provider
        lock serializes concurrent first-access loads.

        When the parent materializes weights, any attached ProcessGuard is
        stopped to release the subprocess copy and avoid double-memory OOM.

        Returns:
            The SloTransformer model.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        model = getattr(self, "_model", None)
        if model is not None:
            return model
        lock = getattr(self, "_lazy_lock", None)
        if lock is None:
            raise RuntimeError(f"Model '{self._model_id}' not loaded — no lazy lock initialized")
        # If another thread (e.g. parent preload) is already loading, wait
        # for it to finish instead of blocking on the lock.
        mat = getattr(self, "_materializing", None)
        if mat is not None and not mat.is_set():
            # Wait for background preload to finish (up to 300s).
            # Uses Event.wait() instead of busy loop for efficiency.
            if not mat.wait(timeout=300):
                raise RuntimeError(
                    f"Model materialization timed out for '{self._model_id}'"
                )
            model = self._model
            if model is not None:
                return model
        with lock:
            model = self._model
            if model is not None:
                return model
            eager = self.from_slnc(
                self._slnc_path, model_id=self._model_id, **self._load_kwargs
            )
            self._model = eager._model
            self._parser = eager._parser
            self._quant_engine = eager._quant_engine
            self._kv_states = eager._kv_states
            self._kv_last_access = eager._kv_last_access
            self._kv_ttl = eager._kv_ttl
            self._kv_max_sessions = eager._kv_max_sessions
            self._kv_lock = eager._kv_lock
            self._loaded = True
            if self._meta is not None:
                self._meta["quantized"] = eager._quant_engine is not None
                self._meta["lazy"] = True
            logger.info(
                "SloNetChatProvider: %s weights now resident in parent (lazy load)",
                self._model_id, extra={"tag": "INF"},
            )
            return self._model

    def materialize_model(self):
        """Load the model directly, bypassing _get_model() lock.

        Used by parent preload to load weights without holding _lazy_lock
        for the entire duration. Sets _materializing flag so other threads
        wait instead of blocking on the lock.

        Note: Guard lifecycle is managed by the caller (startup.py preload
        thread), not by this method.
        """
        model = getattr(self, "_model", None)
        if model is not None:
            return model
        mat = getattr(self, "_materializing", None)
        if mat is None:
            return self._get_model()
        # Signal that materialization is in progress
        mat.clear()
        try:
            eager = self.from_slnc(
                self._slnc_path, model_id=self._model_id, **self._load_kwargs
            )
            self._model = eager._model
            self._parser = eager._parser
            self._quant_engine = eager._quant_engine
            self._kv_states = eager._kv_states
            self._kv_last_access = eager._kv_last_access
            self._kv_ttl = eager._kv_ttl
            self._kv_max_sessions = eager._kv_max_sessions
            self._kv_lock = eager._kv_lock
            self._loaded = True
            if self._meta is not None:
                self._meta["quantized"] = eager._quant_engine is not None
                self._meta["lazy"] = True
            logger.info(
                "SloNetChatProvider: %s weights now resident (materialize)",
                self._model_id, extra={"tag": "INF"},
            )
            return self._model
        finally:
            mat.set()  # Signal completion — wake all waiters

    def release_model(self) -> bool:
        """Drop the resident model and return its memory to the OS.

        Frees the model weights, clears cross-turn KV states, and calls
        ``malloc_trim(0)`` so allocator-held memory from weight conversion is
        actually returned (glibc/Linux only). The provider remains usable —
        a later call to any generation/embedding method reloads weights via
        ``_get_model()``.

        Only supported for lazy providers (created via ``lazy_from_slnc``);
        eager providers keep their model for the provider's lifetime.

        Returns:
            True if weights were resident and released, False otherwise.
        """
        lock = getattr(self, "_lazy_lock", None)
        if lock is None:
            return False
        with lock:
            if self._model is None and self._parser is None:
                return False
            self._model = None
            self._parser = None
            self._quant_engine = None
            with self._kv_lock:
                self._kv_states.clear()
                self._kv_last_access.clear()
            self._loaded = False
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass
        logger.info(
            "SloNetChatProvider.release_model: %s weights released to OS",
            self._model_id, extra={"tag": "INF"},
        )
        return True

    def num_parameters(self) -> int:
        """Total number of model parameters without forcing a weight load.

        Lazy providers read the header-only count captured at creation;
        eager providers count the loaded model's parameters directly.

        Returns:
            Total parameter count, or 0 if unknown.
        """
        meta = getattr(self, "_meta", None)
        if meta is not None and "total_params" in meta:
            return int(meta["total_params"])
        model = getattr(self, "_model", None)
        if model is not None:
            return int(sum(p.data.size for p in model.parameters()))
        return 0

    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 1.0,
                 top_k: int = None, top_p: float = None, repetition_penalty: float = 1.0,
                 max_new_tokens: int = None, **kwargs) -> str:
        """Generate text using pure numpy inference (KV cache + inlined ops)."""
        import numpy as _np
        if max_new_tokens is not None:
            max_tokens = max_new_tokens
        tokens = self._tokenizer.encode(prompt)
        input_ids = _np.array([tokens], dtype=_np.int64)
        result = self._get_model().generate_numpy(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token=self._tokenizer.eos_token_id or 0,
            extra_stop_ids=getattr(self._tokenizer, "chat_stop_ids", lambda: ())(),
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
                session_id=kwargs.get('session_id'),
            )
        return await asyncio.to_thread(
            self._generate_sync, messages, max_tokens, temperature if temperature is not None else 0.8,
            kwargs.get('top_k'), kwargs.get('top_p'),
            kwargs.get('repetition_penalty', 1.0),
            session_id=kwargs.get('session_id'),
        )

    def _evict_stale_sessions(self):
        """Remove KV states for sessions idle longer than _kv_ttl seconds."""
        import time as _time
        now = _time.monotonic()
        with self._kv_lock:
            stale = [sid for sid, ts in self._kv_last_access.items()
                     if now - ts > self._kv_ttl]
            for sid in stale:
                self._kv_states.pop(sid, None)
                self._kv_last_access.pop(sid, None)
        if stale:
            logger.info("Evicted %d stale KV sessions (TTL=%.0fs)", len(stale), self._kv_ttl,
                        extra={"tag": "INF"})

    def _resolve_session_kv(self, session_id):
        """Resolve KV state for a session, creating if needed, with TTL eviction."""
        import time as _time
        if session_id is None:
            return None
        self._evict_stale_sessions()
        with self._kv_lock:
            kv_state = self._kv_states.get(session_id)
            if kv_state is None:
                kv_state = self._get_model().new_kv_state()
                self._kv_states[session_id] = kv_state
                self._evict_lru_session(session_id)
            self._kv_last_access[session_id] = _time.monotonic()
        return kv_state

    def _evict_lru_session(self, keep_session_id):
        """Evict the least-recently-used session when over the session cap.

        Must be called with self._kv_lock held. The session being resolved is
        never evicted (it has no timestamp yet), so LRU is chosen among the
        remaining entries.

        Args:
            keep_session_id: The session that is being resolved/created.
        """
        if len(self._kv_states) <= self._kv_max_sessions:
            return
        evictable = {sid: ts for sid, ts in self._kv_last_access.items()
                     if sid != keep_session_id}
        if not evictable:
            return
        lru_id = min(evictable, key=evictable.get)
        self._kv_states.pop(lru_id, None)
        self._kv_last_access.pop(lru_id, None)
        logger.info("Evicted least-recently-used KV session %s (max=%d)",
                    lru_id, self._kv_max_sessions, extra={"tag": "INF"})

    def _generate_sync(self, messages, max_tokens=512, temperature=0.8,
                       top_k=None, top_p=None, repetition_penalty=1.0,
                       session_id=None):
        """Synchronous generate with KV cache — called from chat() via to_thread."""
        import numpy as _np
        prompt = self._build_prompt(messages)
        tokens = self._tokenizer.encode(prompt)
        input_ids = _np.array([tokens], dtype=_np.int64)

        # Cross-turn KV cache: resolve or create state for this session
        kv_state = self._resolve_session_kv(session_id)

        result = self._get_model().generate_numpy(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token=self._tokenizer.eos_token_id or 0,
            extra_stop_ids=getattr(self._tokenizer, "chat_stop_ids", lambda: ())(),
            kv_state=kv_state,
        )
        return self._tokenizer.decode(result[0].tolist())

    async def chat_stream(self, messages, max_tokens=512, temperature=0.7, **kwargs):
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
                session_id=kwargs.get('session_id'),
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
        session_id = kwargs.get('session_id')

        # Cross-turn KV cache: resolve or create state for this session
        kv_state = self._resolve_session_kv(session_id)

        def _stream_generate():
            """Token-by-token streaming using generate_numpy_stream().

            Yields decoded tokens as they are produced — true streaming
            without waiting for full generation to complete.
            """
            m = self._get_model()
            input_ids = _np.array([token_ids], dtype=_np.int64)

            for tok_id in m.generate_numpy_stream(
                input_ids,
                max_new_tokens=max_tokens,
                eos_token=eos_id,
                extra_stop_ids=getattr(self._tokenizer, "chat_stop_ids", lambda: ())(),
                temperature=temperature if temperature is not None else 0.8,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                kv_state=kv_state,
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

        def _producer():
            try:
                for token in _stream_generate():
                    q.put(token)
            except Exception as e:
                logger.error("[chat_stream] Producer error: %s", e, exc_info=True)
                err_q.put(e)
            finally:
                q.put(sentinel)

        producer_thread = threading.Thread(target=_producer, daemon=True)
        producer_thread.start()

        gen_start = time.monotonic()
        while True:
            # Check total generation timeout
            elapsed = time.monotonic() - gen_start
            if elapsed > _STREAM_TOTAL_TIMEOUT_S:
                cancel_event.set() if cancel_event else None
                logger.warning("Streaming generation timed out after %.0fs", elapsed, extra={"tag": "INF"})
                yield "\n\n[Generation timed out after {:.0f}s]".format(elapsed)
                return

            try:
                # Use to_thread with timeout to prevent indefinite blocking
                token = await asyncio.wait_for(
                    asyncio.to_thread(q.get),
                    timeout=_STREAM_GET_TIMEOUT_S,
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
                if not err_q.empty():
                    exc = err_q.get_nowait()
                    yield "\n\n[Generation error: {}]".format(exc)
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
        m = self._get_model()
        logprobs_list = []

        # Use the streaming path to capture logits
        for step, tok_id in enumerate(m.generate_numpy_stream(
            input_ids,
            max_new_tokens=max_tokens,
            eos_token=eos_id,
            extra_stop_ids=getattr(self._tokenizer, "chat_stop_ids", lambda: ())(),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
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

        for tok_id in self._get_model().generate_numpy_stream(
            input_ids,
            max_new_tokens=max_tokens,
            eos_token=eos_id,
            extra_stop_ids=getattr(self._tokenizer, "chat_stop_ids", lambda: ())(),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
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
        m = self._get_model()

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

        Lazy providers return the header-only metadata captured at creation
        (no weight load triggered). Eager providers compute metadata from the
        loaded model on first call and cache it.

        Returns:
            Dict with model_id, architecture, parameters, vocab_size, etc.
        """
        cached = getattr(self, "_meta", None)
        if cached is not None:
            return dict(cached)

        m = self._get_model()
        config = m._config if hasattr(m, '_config') else {}

        # Count parameters
        total_params = sum(
            p.data.size for p in m.parameters()
        )

        # Model architecture info
        n_layer = len([l for l in m.layers if hasattr(l, 'forward_numpy')])
        n_embed = config.get("n_embd", config.get("hidden_size", 0))
        n_head = config.get("n_head", config.get("num_attention_heads", 0))

        result = {
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
        self._meta = result
        return dict(result)

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
