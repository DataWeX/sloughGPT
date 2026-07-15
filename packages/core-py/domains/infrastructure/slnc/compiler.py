"""
.slnc compiler — converts safetensors to memory-mapped format.

Pipeline:
  1. Parse source (safetensors or numpy dict)
  2. Compute layout (tensor offsets in computation order)
  3. Write header (magic + metadata + config)
  4. Write tensor table (offsets + checksums)
  5. Write tensor data (computation order)
  6. Verify (optional integrity check)

Usage:
    from domains.infrastructure.slnc.compiler import SLNCCompiler

    compiler = SLNCCompiler()
    compiler.compile("gpt2", output="models/gpt2.slnc")
    compiler.compile_from_dict(config, weights, output="custom.slnc")
"""

import json
import logging
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from domains.infrastructure.slnc.spec import (
    MAGIC,
    VERSION,
    FLAGS_DEFAULT,
    ALIGNMENT,
    DTYPE_FLOAT32,
    compute_header_size,
    compute_tensor_entry_size,
    compute_tensor_table_size,
    dtype_to_code,
    _align,
)

logger = logging.getLogger("slo.infrastructure.slnc.compiler")


# ══════════════════════════════════════════════════════════════════════════════
# Block tensor definitions (computation order)
# ══════════════════════════════════════════════════════════════════════════════

# GPT-2 block tensors in computation order
GPT2_BLOCK_LAYOUT = [
    "ln_1.weight",
    "ln_1.bias",
    "attn.c_attn.weight",
    "attn.c_attn.bias",
    "attn.c_proj.weight",
    "attn.c_proj.bias",
    "ln_2.weight",
    "ln_2.bias",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
    "mlp.c_proj.bias",
]

# Non-block tensors (after all blocks)
GPT2_NON_BLOCK_LAYOUT = [
    "ln_f.weight",
    "ln_f.bias",
    "wte.weight",
    "wpe.weight",
]

# LLaMA/Qwen/Mistral block tensors in computation order
LLAMA_BLOCK_LAYOUT = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]

# Non-block tensors (after all blocks)
LLAMA_NON_BLOCK_LAYOUT = [
    "model.norm.weight",
    "model.embed_tokens.weight",
    "model.lm_head.weight",
]

# Mapping: architecture detection → block/non-block layouts
_ARCH_LAYOUTS = {
    "gpt2": (GPT2_BLOCK_LAYOUT, GPT2_NON_BLOCK_LAYOUT, "h.{i}."),
    "llama": (LLAMA_BLOCK_LAYOUT, LLAMA_NON_BLOCK_LAYOUT, "model.layers.{i}."),
}


class SLNCCompiler:
    """Compiles model weights into .slnc format."""

    def __init__(self):
        self._tensor_entries: List[Tuple[str, int, bytes, np.dtype, int, int]] = []
        # (name, offset, data_bytes, dtype, ndim, crc32)

    def compile(
        self,
        model_id: str,
        output: Optional[str] = None,
    ) -> str:
        """Compile HuggingFace model to .slnc.

        Args:
            model_id: HuggingFace model ID
            output: Output path. Defaults to models/<model_id>.slnc

        Returns:
            Path to created .slnc file
        """
        from domains.infrastructure.safetensors_loader import (
            _get_model_dir,
            _find_safetensors,
            load_model_config,
        )

        config = load_model_config(model_id)
        model_dir = _get_model_dir(model_id)
        safetensors_path = _find_safetensors(model_dir)
        if safetensors_path is None:
            raise FileNotFoundError(f"No .safetensors for {model_id}")

        # Load all weights — read raw bytes to handle bfloat16
        import json as _json
        weights = {}
        with open(str(safetensors_path), "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = _json.loads(f.read(header_len))
            for key, info in header.items():
                if key.startswith("__"):
                    continue
                dtype_str = info["dtype"]
                offsets = info["data_offsets"]
                f.seek(8 + header_len + offsets[0])
                raw = f.read(offsets[1] - offsets[0])
                if dtype_str == "BF16":
                    arr = np.frombuffer(raw, dtype=np.uint16)
                    f32 = np.zeros(len(arr), dtype=np.float32)
                    f32.view(np.uint32)[:][:] = arr.astype(np.uint32) << 16
                    weights[key] = f32.reshape(info["shape"])
                elif dtype_str == "F32":
                    weights[key] = np.frombuffer(raw, dtype=np.float32).reshape(info["shape"])
                elif dtype_str == "F16":
                    weights[key] = np.frombuffer(raw, dtype=np.float16).reshape(info["shape"]).astype(np.float32)
                else:
                    weights[key] = np.frombuffer(raw, dtype=np.float32).reshape(info["shape"])

        # Determine output path
        if output is None:
            repo_root = Path(__file__).resolve().parents[4]
            models_dir = repo_root / "models"
            models_dir.mkdir(exist_ok=True)
            output = str(models_dir / f"{model_id.replace('/', '_')}.slnc")

        return self.compile_from_dict(config, weights, output)

    def compile_from_dict(
        self,
        config: dict,
        weights: Dict[str, np.ndarray],
        output: str,
    ) -> str:
        """Compile from config + weight dict.

        Args:
            config: HuggingFace config.json
            weights: Dict mapping tensor names to numpy arrays
            output: Output file path

        Returns:
            Path to created .slnc file
        """
        logger.info("Compiling %s (%d tensors)", output, len(weights),
            extra={"tag": "INFRA"})

        # Build tensor list in computation order
        tensor_list = self._order_tensors(config, weights)

        # Build tensor entries (compute sizes without storing data yet)
        entries_for_size = []
        for name, tensor in tensor_list:
            ndim = len(tensor.shape)
            # Format: (name, offset, data_bytes, ndim, dtype, crc)
            entries_for_size.append((name, 0, None, ndim, tensor.dtype, 0))

        # Compute layout
        config_json = json.dumps(config, sort_keys=True).encode()
        header_size = compute_header_size(config_json)
        tensor_table_size = compute_tensor_table_size(entries_for_size)

        # Compute offsets
        data_start = _align(header_size + tensor_table_size)
        current_offset = data_start

        self._tensor_entries = []
        for name, tensor in tensor_list:
            tensor_bytes = tensor.tobytes()
            crc = _crc32(tensor_bytes)
            dtype_code = dtype_to_code(tensor.dtype)
            ndim = len(tensor.shape)

            self._tensor_entries.append((
                name,
                current_offset,
                tensor_bytes,
                tensor.dtype,
                ndim,
                crc,
            ))
            current_offset += len(tensor_bytes)

        total_size = current_offset

        # Write file
        with open(output, "wb") as f:
            # Header
            f.write(MAGIC)
            f.write(struct.pack("<I", VERSION))
            f.write(struct.pack("<I", FLAGS_DEFAULT))

            # Model metadata (64 bytes)
            n_layer = config.get("n_layer", config.get("num_hidden_layers", 0))
            n_embd = config.get("n_embd", config.get("hidden_size", 0))
            n_head = config.get("n_head", config.get("num_attention_heads", 0))
            n_inner = config.get("n_inner", config.get("intermediate_size", n_embd * 4))
            vocab_size = config.get("vocab_size", 0)
            n_positions = config.get("n_positions", config.get("max_position_embeddings", 1024))
            block_count = n_layer
            block_size = self._compute_block_size(config)
            tensor_count = len(tensor_list)
            data_offset = data_start

            f.write(struct.pack("<I", n_layer))
            f.write(struct.pack("<I", n_embd))
            f.write(struct.pack("<I", n_head))
            f.write(struct.pack("<I", n_inner))
            f.write(struct.pack("<I", vocab_size))
            f.write(struct.pack("<I", n_positions))
            f.write(struct.pack("<I", block_count))
            f.write(struct.pack("<I", block_size))
            f.write(struct.pack("<I", tensor_count))
            f.write(struct.pack("<I", data_offset))
            f.write(b"\x00" * 24)  # reserved

            # Config JSON
            f.write(struct.pack("<I", len(config_json)))
            f.write(config_json)

            # Pad to alignment
            current = f.tell()
            if current < header_size:
                f.write(b"\x00" * (header_size - current))

            # Tensor table
            for name, offset, data_bytes, dtype, ndim, crc in self._tensor_entries:
                name_bytes = name.encode()
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("<Q", offset))
                f.write(struct.pack("<I", len(data_bytes)))
                f.write(struct.pack("<I", ndim))
                shape = tensor_list[[t[0] for t in tensor_list].index(name)][1].shape
                for dim in shape:
                    f.write(struct.pack("<I", dim))
                f.write(struct.pack("<I", dtype_code))
                f.write(struct.pack("<I", crc))

            # Pad to data start
            current = f.tell()
            if current < data_start:
                f.write(b"\x00" * (data_start - current))

            # Tensor data (computation order)
            for name, offset, data_bytes, dtype, ndim, crc in self._tensor_entries:
                f.write(data_bytes)

        logger.info(
            "Compiled %s: %d tensors, %.1f MB, %d blocks",
            output,
            len(tensor_list),
            total_size / 1e6,
            block_count,
            extra={"tag": "INFRA"},
        )
        return output

    def _order_tensors(
        self, config: dict, weights: Dict[str, np.ndarray]
    ) -> List[Tuple[str, np.ndarray]]:
        """Order tensors in computation order. Auto-detects GPT-2 vs LLaMA/Qwen."""
        n_layer = config.get("n_layer", config.get("num_hidden_layers", 12))
        result = []

        # Detect architecture from weight keys
        weight_keys = set(weights.keys())
        if "model.embed_tokens.weight" in weight_keys and "model.layers.0.self_attn.q_proj.weight" in weight_keys:
            arch = "llama"
        elif "wte.weight" in weight_keys:
            arch = "gpt2"
        else:
            arch = "gpt2"  # fallback

        block_layout, non_block_layout, prefix_template = _ARCH_LAYOUTS[arch]

        # Block tensors — include biases if they exist in weights
        bias_suffixes = [".bias"]
        for layer_idx in range(n_layer):
            prefix = prefix_template.format(i=layer_idx)
            for tensor_name in block_layout:
                key = prefix + tensor_name
                if key in weights:
                    result.append((key, weights[key]))
                # Check for corresponding bias (e.g. q_proj.weight → q_proj.bias)
                if tensor_name.endswith(".weight"):
                    bias_key = prefix + tensor_name[:-len(".weight")] + ".bias"
                    if bias_key in weights:
                        result.append((bias_key, weights[bias_key]))

        # Non-block tensors
        for tensor_name in non_block_layout:
            if tensor_name in weights:
                result.append((tensor_name, weights[tensor_name]))
            elif tensor_name == "model.lm_head.weight":
                # Weight tying: lm_head shares embed_tokens weight
                if "model.embed_tokens.weight" in weights:
                    result.append((tensor_name, weights["model.embed_tokens.weight"]))

        return result

    def _compute_block_size(self, config: dict) -> int:
        """Compute bytes per transformer block. Auto-detects GPT-2 vs LLaMA."""
        n_embd = config.get("n_embd", config.get("hidden_size", 768))
        n_inner = config.get("n_inner", config.get("intermediate_size", n_embd * 4))

        # Detect from config keys
        has_rope = config.get("rope_theta") is not None or config.get("position_embedding_type") == "rope"

        if has_rope:
            # LLaMA/Qwen — SwiGLU has 3 weight matrices; biases included if present
            shapes = {
                "input_layernorm.weight": (n_embd,),
                "self_attn.q_proj.weight": (n_embd, n_embd),
                "self_attn.k_proj.weight": (n_embd, n_embd),
                "self_attn.v_proj.weight": (n_embd, n_embd),
                "self_attn.o_proj.weight": (n_embd, n_embd),
                "post_attention_layernorm.weight": (n_embd,),
                "mlp.gate_proj.weight": (n_embd, n_inner),
                "mlp.up_proj.weight": (n_embd, n_inner),
                "mlp.down_proj.weight": (n_inner, n_embd),
            }
            # Dynamic bias shapes — will be included if present in weights
            bias_shapes = {
                "self_attn.q_proj.bias": (n_embd,),
                "self_attn.k_proj.bias": (n_embd,),
                "self_attn.v_proj.bias": (n_embd,),
                "self_attn.o_proj.bias": (n_embd,),
                "mlp.gate_proj.bias": (n_inner,),
                "mlp.up_proj.bias": (n_inner,),
                "mlp.down_proj.bias": (n_embd,),
            }
        else:
            # GPT-2
            shapes = {
                "ln_1.weight": (n_embd,),
                "ln_1.bias": (n_embd,),
                "attn.c_attn.weight": (n_embd, 3 * n_embd),
                "attn.c_attn.bias": (3 * n_embd,),
                "attn.c_proj.weight": (n_embd, n_embd),
                "attn.c_proj.bias": (n_embd,),
                "ln_2.weight": (n_embd,),
                "ln_2.bias": (n_embd,),
                "mlp.c_fc.weight": (n_embd, n_inner),
                "mlp.c_fc.bias": (n_inner,),
                "mlp.c_proj.weight": (n_inner, n_embd),
                "mlp.c_proj.bias": (n_embd,),
            }

        return sum(int(np.prod(s)) * 4 for s in shapes.values())


def _crc32(data: bytes) -> int:
    """Compute CRC32 checksum."""
    import zlib
    return zlib.crc32(data) & 0xFFFFFFFF


def _xxhash64(data: bytes) -> int:
    """Compute xxHash64 (fast hash for tensor names)."""
    # Simple implementation — use zlib if xxhash not available
    try:
        import xxhash
        return xxhash.xxh64(data).intdigest()
    except ImportError:
        # Fallback: use CRC32 as hash (not cryptographically secure, but fine for this)
        return _crc32(data) | (_crc32(data[::-1]) << 32)
