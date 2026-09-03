from __future__ import annotations

"""
.slnc compiler — converts safetensors to memory-mapped format.

Pipeline:
  1. Parse source (safetensors or numpy dict)
  2. Compute layout (tensor offsets in computation order)
  3. Write header (magic + metadata + config + optional header CRC)
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
from typing import Dict, List, Optional, Tuple

import numpy as np

from domains.shared import find_repo_root
from domains.infrastructure.slnc.spec import (
    MAGIC,
    VERSION,
    MAX_NDIM,
    MAX_TENSOR_COUNT,
    MAX_NAME_LEN,
    SLNCConfig,
    compute_header_size,
    compute_tensor_table_size,
    dtype_to_code,
    _align,
    _align_offset,
)

logger = logging.getLogger("slo.infrastructure.slnc.compiler")


# ══════════════════════════════════════════════════════════════════════════════
# Block tensor definitions (computation order)
# ══════════════════════════════════════════════════════════════════════════════

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

GPT2_NON_BLOCK_LAYOUT = [
    "ln_f.weight",
    "ln_f.bias",
    "wte.weight",
    "wpe.weight",
]

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

LLAMA_NON_BLOCK_LAYOUT = [
    "model.norm.weight",
    "model.embed_tokens.weight",
    "model.lm_head.weight",
]

_ARCH_LAYOUTS = {
    "gpt2": (GPT2_BLOCK_LAYOUT, GPT2_NON_BLOCK_LAYOUT, "h.{i}."),
    "llama": (LLAMA_BLOCK_LAYOUT, LLAMA_NON_BLOCK_LAYOUT, "model.layers.{i}."),
}


class SLNCCompiler:
    """Compiles model weights into .slnc format."""

    def __init__(self, config: Optional[SLNCConfig] = None):
        self._config = config or SLNCConfig()
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
        from domains.infrastructure.model_resolver import (
            get_model_dir as _get_model_dir,
            find_safetensors as _find_safetensors,
            load_model_config,
        )

        config = load_model_config(model_id)
        model_dir = _get_model_dir(model_id)
        safetensors_path = _find_safetensors(model_dir)
        if safetensors_path is None:
            raise FileNotFoundError(f"No .safetensors for {model_id}")

        weights = self._read_weights(safetensors_path)

        if output is None:
            repo_root = find_repo_root(Path(__file__).resolve())
            models_dir = repo_root / "models"
            models_dir.mkdir(exist_ok=True)
            output = str(models_dir / f"{model_id.replace('/', '_')}.slnc")

        try:
            from domains.infrastructure.model_protector import protect_model
            protect_model(model_id, [output])
        except Exception as e:
            logger.debug("Could not protect .slnc file: %s", e)

        return self.compile_from_dict(config, weights, output)

    def compile_from_directory(
        self,
        model_dir: str,
        output: str,
        config_path: Optional[str] = None,
    ) -> str:
        """Compile a local fine-tuned model directory to .slnc."""
        directory = Path(model_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        cfg_path = Path(config_path) if config_path else directory / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"No config.json in {model_dir}")
        with open(cfg_path) as f:
            config = json.load(f)

        from domains.infrastructure.model_resolver import find_safetensors as _find_safetensors
        safetensors_path = _find_safetensors(directory)
        if safetensors_path is None:
            raise FileNotFoundError(f"No .safetensors in {model_dir}")
        weights = self._read_weights(safetensors_path)

        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return self.compile_from_dict(config, weights, str(out_path))

    def _read_weights(self, safetensors_path: Path) -> Dict[str, np.ndarray]:
        """Read all weight arrays from a safetensors file."""
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
        return weights

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

        # Validate tensor count
        if len(weights) > MAX_TENSOR_COUNT:
            raise ValueError(f"Too many tensors: {len(weights)} (max {MAX_TENSOR_COUNT})")

        # Build tensor list in computation order
        tensor_list = self._order_tensors(config, weights)

        # Validate ndim for all tensors
        for name, tensor in tensor_list:
            if len(tensor.shape) > MAX_NDIM:
                raise ValueError(f"Tensor {name} has {len(tensor.shape)} dims (max {MAX_NDIM})")

        # Build shape lookup (O(1) instead of O(n²))
        shape_map = {name: tensor.shape for name, tensor in tensor_list}

        # Compute layout
        config_json = json.dumps(config, sort_keys=True).encode()
        header_size = compute_header_size(config_json)

        # Build tensor entries for size computation
        entries_for_size = []
        for name, tensor in tensor_list:
            ndim = len(tensor.shape)
            entries_for_size.append((name, 0, None, ndim, tensor.dtype, 0))

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

            # Align tensor data if enabled
            if self._config.align_tensors:
                current_offset = _align_offset(current_offset)

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

        # Compute flags
        flags = self._config.to_flags()

        # Write file
        with open(output, "wb") as f:
            # Header: magic + version + flags
            f.write(MAGIC)
            f.write(struct.pack("<I", VERSION))
            f.write(struct.pack("<I", flags))

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

            # Reserved region (24 bytes): header_crc + unused
            reserved_start = f.tell()
            f.write(b"\x00" * 24)  # placeholder for header CRC

            # Config JSON
            f.write(struct.pack("<I", len(config_json)))
            f.write(config_json)

            # Pad to alignment
            current = f.tell()
            if current < header_size:
                f.write(b"\x00" * (header_size - current))

            # Write header CRC into reserved region
            if self._config.write_header_crc:
                # Read back header data to compute CRC
                f_pos = f.tell()
                f.seek(0)
                header_data = f.read(header_size)
                f.seek(f_pos)

                header_crc = _crc32(header_data)
                f.seek(reserved_start)
                f.write(struct.pack("<I", header_crc))
                f.seek(f_pos)

            # Tensor table (variable-length entries)
            for name, offset, data_bytes, dtype, ndim, crc in self._tensor_entries:
                name_bytes = name.encode()
                if len(name_bytes) > MAX_NAME_LEN:
                    raise ValueError(f"Tensor name too long: {name} ({len(name_bytes)} > {MAX_NAME_LEN})")

                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("<Q", offset))
                f.write(struct.pack("<I", len(data_bytes)))
                f.write(struct.pack("<I", ndim))
                shape = shape_map[name]
                for dim in shape:
                    f.write(struct.pack("<I", dim))
                f.write(struct.pack("<I", dtype_to_code(dtype)))
                f.write(struct.pack("<I", crc))

            # Pad to data start
            current = f.tell()
            if current < data_start:
                f.write(b"\x00" * (data_start - current))

            # Tensor data (computation order, aligned)
            for name, offset, data_bytes, dtype, ndim, crc in self._tensor_entries:
                # Pad to alignment before writing
                current_pos = f.tell()
                if current_pos < offset:
                    f.write(b"\x00" * (offset - current_pos))
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

        weight_keys = set(weights.keys())
        if "model.embed_tokens.weight" in weight_keys and "model.layers.0.self_attn.q_proj.weight" in weight_keys:
            arch = "llama"
        elif "wte.weight" in weight_keys:
            arch = "gpt2"
        else:
            arch = "gpt2"

        block_layout, non_block_layout, prefix_template = _ARCH_LAYOUTS[arch]

        # Block tensors
        for layer_idx in range(n_layer):
            prefix = prefix_template.format(i=layer_idx)
            for tensor_name in block_layout:
                key = prefix + tensor_name
                if key in weights:
                    result.append((key, weights[key]))
                if tensor_name.endswith(".weight"):
                    bias_key = prefix + tensor_name[:-len(".weight")] + ".bias"
                    if bias_key in weights and bias_key not in (prefix + t for t in block_layout):
                        result.append((bias_key, weights[bias_key]))

        # Non-block tensors
        for tensor_name in non_block_layout:
            if tensor_name in weights:
                result.append((tensor_name, weights[tensor_name]))
            elif tensor_name == "model.lm_head.weight":
                if "model.embed_tokens.weight" in weights:
                    result.append((tensor_name, weights["model.embed_tokens.weight"]))

        return result

    def _compute_block_size(self, config: dict) -> int:
        """Compute bytes per transformer block."""
        n_embd = config.get("n_embd", config.get("hidden_size", 768))
        n_inner = config.get("n_inner", config.get("intermediate_size", n_embd * 4))

        has_rope = config.get("rope_theta") is not None or config.get("position_embedding_type") == "rope"

        if has_rope:
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
        else:
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
    try:
        import xxhash
        return xxhash.xxh64(data).intdigest()
    except ImportError:
        return _crc32(data) | (_crc32(data[::-1]) << 32)
