"""
.slnc — SloughGPT Neural Cache format.

Memory-mapped inference format with computation-graph-aligned layout.
Weights are stored in inference order: Block 0 → Block 1 → ... → norm → lm_head.

Benefits over safetensors:
- Zero-copy: numpy arrays are views into mmap'd memory
- Sequential access: blocks are contiguous (no seeking across file)
- Demand loading: OS pages in only accessed blocks
- No parsing: offsets computed from config, not JSON header

File layout:
  [header: SLNC magic + config + offsets]
  [block 0: ln_1, attn, ln_2, mlp (contiguous)]
  [block 1: ln_1, attn, ln_2, mlp (contiguous)]
  ...
  [block N-1]
  [non-block: wte, wpe, ln_f, lm_head]

from __future__ import annotations

Usage:
    # Convert
    from domains.infrastructure.slnc_format import convert_to_slnc
    convert_to_slnc("gpt2", "gpt2.slnc")

    # Load (mmap)
    from domains.infrastructure.slnc_loader import SLNCLoader
    loader = SLNCLoader("gpt2.slnc")
    block0_q = loader.get_tensor("blocks.0.attn.q_proj.weight")
"""

import json
import logging
import os
import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from domains.shared import find_repo_root

logger = logging.getLogger("slo.infrastructure.slnc_format")

# ══════════════════════════════════════════════════════════════════════════════
# Format constants
# ══════════════════════════════════════════════════════════════════════════════

SLNC_MAGIC = b"SLNC"
SLNC_VERSION = 1

# Header field sizes (fixed)
HEADER_FIXED_SIZE = (
    4   # magic
    + 4  # version
    + 4  # n_layer
    + 4  # n_embd
    + 4  # n_head
    + 4  # n_inner
    + 4  # vocab_size
    + 4  # n_positions
    + 4  # block_count
    + 4  # block_size_bytes
    + 4  # non_block_offset
    + 4  # non_block_size
    + 4  # tensor_count
)

# GPT-2 tensor names (in computation order within a block)
GPT2_BLOCK_TENSORS = [
    ("ln_1.weight", "weight"),
    ("ln_1.bias", "bias"),
    ("attn.c_attn.weight", "weight"),
    ("attn.c_attn.bias", "bias"),
    ("attn.c_proj.weight", "weight"),
    ("attn.c_proj.bias", "bias"),
    ("ln_2.weight", "weight"),
    ("ln_2.bias", "bias"),
    ("mlp.c_fc.weight", "weight"),
    ("mlp.c_fc.bias", "bias"),
    ("mlp.c_proj.weight", "weight"),
    ("mlp.c_proj.bias", "bias"),
]

# LLaMA/Qwen tensor names (in computation order within a block)
LLAMA_BLOCK_TENSORS = [
    ("input_layernorm.weight", "weight"),
    ("self_attn.q_proj.weight", "weight"),
    ("self_attn.k_proj.weight", "weight"),
    ("self_attn.v_proj.weight", "weight"),
    ("self_attn.o_proj.weight", "weight"),
    ("post_attention_layernorm.weight", "weight"),
    ("mlp.gate_proj.weight", "weight"),
    ("mlp.up_proj.weight", "weight"),
    ("mlp.down_proj.weight", "weight"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Layout computation
# ══════════════════════════════════════════════════════════════════════════════

def _compute_tensor_size(name: str, config: dict) -> int:
    """Compute tensor size in bytes from name and config."""
    n_embd = config["n_embd"]
    n_inner = config.get("n_inner", n_embd * 4)
    vocab = config["vocab_size"]
    n_pos = config.get("n_positions", 1024)

    shape = _get_tensor_shape(name, config)
    return int(np.prod(shape)) * 4  # float32 = 4 bytes


def _get_tensor_shape(name: str, config: dict) -> Tuple[int, ...]:
    """Get tensor shape from name and config."""
    n_embd = config["n_embd"]
    n_inner = config.get("n_inner", n_embd * 4)
    vocab = config["vocab_size"]
    n_pos = config.get("n_positions", 1024)

    if name == "ln_1.weight" or name == "ln_2.weight" or name == "ln_f.weight":
        return (n_embd,)
    elif name == "ln_1.bias" or name == "ln_2.bias" or name == "ln_f.bias":
        return (n_embd,)
    elif name == "attn.c_attn.weight":
        return (n_embd, 3 * n_embd)  # fused QKV
    elif name == "attn.c_attn.bias":
        return (3 * n_embd,)
    elif name == "attn.c_proj.weight":
        return (n_embd, n_embd)
    elif name == "attn.c_proj.bias":
        return (n_embd,)
    elif name == "mlp.c_fc.weight":
        return (n_embd, n_inner)
    elif name == "mlp.c_fc.bias":
        return (n_inner,)
    elif name == "mlp.c_proj.weight":
        return (n_inner, n_embd)
    elif name == "mlp.c_proj.bias":
        return (n_embd,)
    elif name == "wte.weight":
        return (vocab, n_embd)
    elif name == "wpe.weight":
        return (n_pos, n_embd)
    else:
        raise ValueError(f"Unknown tensor: {name}")


def compute_layout(config: dict) -> dict:
    """Compute the full file layout from config.

    Returns dict with:
        block_tensors: list of (name, offset_within_block, size_bytes)
        non_block_tensors: list of (name, global_offset, size_bytes)
        block_size: total bytes per block
        header_size: total header bytes
    """
    n_layer = config["n_layer"]

    # Compute block layout
    block_tensors = []
    offset = 0
    for name, _ in GPT2_BLOCK_TENSORS:
        size = _compute_tensor_size(name, config)
        block_tensors.append((name, offset, size))
        offset += size
    block_size = offset

    # Compute non-block layout
    non_block_tensors = []
    header_size = HEADER_FIXED_SIZE + 4  # +4 for json_len field
    json_bytes = json.dumps(config).encode()
    header_size += len(json_bytes)

    # Align to 64 bytes for cache line alignment
    header_size = (header_size + 63) & ~63

    non_block_offset = header_size + block_size * n_layer
    current_offset = non_block_offset

    for name in ["wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"]:
        size = _compute_tensor_size(name, config)
        non_block_tensors.append((name, current_offset, size))
        current_offset += size

    return {
        "block_tensors": block_tensors,
        "non_block_tensors": non_block_tensors,
        "block_size": block_size,
        "header_size": header_size,
        "total_size": current_offset,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Converter: safetensors → .slnc
# ══════════════════════════════════════════════════════════════════════════════

def convert_to_slnc(
    model_id: str,
    output_path: Optional[str] = None,
) -> str:
    """Convert HuggingFace model to .slnc format.

    Args:
        model_id: HuggingFace model ID (e.g. "gpt2")
        output_path: Output file path. Defaults to models/<model_id>.slnc

    Returns:
        Path to created .slnc file
    """
    from domains.infrastructure.safetensors_loader import (
        _get_model_dir,
        _find_safetensors,
        load_model_config,
    )
    from safetensors import safe_open

    # Load config
    config = load_model_config(model_id)

    # Compute layout
    layout = compute_layout(config)

    # Determine output path
    if output_path is None:
        repo_root = find_repo_root(Path(__file__).resolve())
        models_dir = repo_root / "models"
        models_dir.mkdir(exist_ok=True)
        output_path = str(models_dir / f"{model_id.replace('/', '_')}.slnc")

    logger.info("Converting %s → %s", model_id, output_path,
        extra={"tag": "INFRA"})
    logger.info(
        "Layout: %d layers, %d bytes/block, %d non-block tensors",
        config["n_layer"],
        layout["block_size"],
        len(layout["non_block_tensors"]),
        extra={"tag": "INFRA"},
    )

    # Load source weights
    model_dir = _get_model_dir(model_id)
    safetensors_path = _find_safetensors(model_dir)
    if safetensors_path is None:
        raise FileNotFoundError(f"No .safetensors for {model_id}")

    source_weights = {}
    with safe_open(str(safetensors_path), framework="numpy") as f:
        for key in f.keys():
            source_weights[key] = f.get_tensor(key).astype(np.float32)

    # Write .slnc file
    with open(output_path, "wb") as out:
        # Header
        out.write(SLNC_MAGIC)
        out.write(struct.pack("<I", SLNC_VERSION))
        out.write(struct.pack("<I", config["n_layer"]))
        out.write(struct.pack("<I", config["n_embd"]))
        out.write(struct.pack("<I", config.get("n_head", 12)))
        out.write(struct.pack("<I", config.get("n_inner", config["n_embd"] * 4)))
        out.write(struct.pack("<I", config["vocab_size"]))
        out.write(struct.pack("<I", config.get("n_positions", 1024)))
        out.write(struct.pack("<I", config["n_layer"]))  # block_count
        out.write(struct.pack("<I", layout["block_size"]))
        out.write(struct.pack("<I", layout["header_size"] + layout["block_size"] * config["n_layer"]))  # non_block_offset
        out.write(struct.pack("<I", sum(s for _, _, s in layout["non_block_tensors"])))  # non_block_size
        tensor_count = config["n_layer"] * len(GPT2_BLOCK_TENSORS) + len(layout["non_block_tensors"])
        out.write(struct.pack("<I", tensor_count))

        # Config JSON
        json_bytes = json.dumps(config).encode()
        out.write(struct.pack("<I", len(json_bytes)))
        out.write(json_bytes)

        # Pad to header_size
        current = out.tell()
        if current < layout["header_size"]:
            out.write(b"\x00" * (layout["header_size"] - current))

        # Write blocks in computation order
        for layer_idx in range(config["n_layer"]):
            for tensor_name, offset_in_block, size in layout["block_tensors"]:
                # Map to safetensors key
                st_key = f"h.{layer_idx}.{tensor_name}"
                if st_key not in source_weights:
                    raise KeyError(f"Missing tensor: {st_key}")

                weight = source_weights[st_key]
                expected_shape = _get_tensor_shape(tensor_name, config)
                if weight.shape != expected_shape:
                    raise ValueError(
                        f"Shape mismatch: {st_key} has {weight.shape}, expected {expected_shape}"
                    )

                out.write(weight.tobytes())

        # Write non-block tensors
        for name, _, size in layout["non_block_tensors"]:
            if name not in source_weights:
                raise KeyError(f"Missing tensor: {name}")
            out.write(source_weights[name].tobytes())

    total_size = layout["total_size"]
    logger.info(
        "Created %s (%.1f MB, %d layers, %d bytes/block)",
        output_path,
        total_size / 1e6,
        config["n_layer"],
        layout["block_size"],
        extra={"tag": "INFRA"},
    )
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# Metadata reader (for quick inspection without full mmap)
# ══════════════════════════════════════════════════════════════════════════════

def read_slnc_header(path: str) -> dict:
    """Read .slnc header without loading weights.

    Returns:
        Dict with config, layout info, and tensor map.
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != SLNC_MAGIC:
            raise ValueError(f"Invalid magic: {magic!r} (expected {SLNC_MAGIC!r})")

        version = struct.unpack("<I", f.read(4))[0]
        if version != SLNC_VERSION:
            raise ValueError(f"Unsupported version: {version}")

        n_layer = struct.unpack("<I", f.read(4))[0]
        n_embd = struct.unpack("<I", f.read(4))[0]
        n_head = struct.unpack("<I", f.read(4))[0]
        n_inner = struct.unpack("<I", f.read(4))[0]
        vocab_size = struct.unpack("<I", f.read(4))[0]
        n_positions = struct.unpack("<I", f.read(4))[0]
        block_count = struct.unpack("<I", f.read(4))[0]
        block_size = struct.unpack("<I", f.read(4))[0]
        non_block_offset = struct.unpack("<I", f.read(4))[0]
        non_block_size = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<I", f.read(4))[0]

        json_len = struct.unpack("<I", f.read(4))[0]
        pos = f.tell()
        file_size = os.fstat(f.fileno()).st_size
        if json_len < 0 or pos + json_len > file_size:
            raise ValueError(
                f"Corrupt SLNC header: json_len={json_len} exceeds "
                f"file bounds (file_size={file_size}, pos={pos})"
            )
        try:
            config = json.loads(f.read(json_len))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt SLNC config JSON: {exc}") from exc

    return {
        "version": version,
        "n_layer": n_layer,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_inner": n_inner,
        "vocab_size": vocab_size,
        "n_positions": n_positions,
        "block_count": block_count,
        "block_size": block_size,
        "non_block_offset": non_block_offset,
        "non_block_size": non_block_size,
        "tensor_count": tensor_count,
        "config": config,
    }
