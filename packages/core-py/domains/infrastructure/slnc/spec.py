"""
.slnc binary format specification.

Defines the binary layout, magic numbers, and field sizes.
This is the single source of truth for the format.

Binary layout:
  ┌─────────────────────────────────────────────────────┐
  │ Magic (4 bytes): "SLNC"                            │
  │ Version (4 bytes): uint32 LE                       │
  │ Flags (4 bytes): uint32 LE (reserved)              │
  ├─────────────────────────────────────────────────────┤
  │ Model metadata (fixed 64 bytes):                   │
  │   n_layer (4), n_embd (4), n_head (4), n_inner (4)│
  │   vocab_size (4), n_positions (4)                  │
  │   block_count (4), block_size (4)                  │
  │   tensor_count (4), data_offset (4)                │
  │   reserved (24 bytes)                              │
  ├─────────────────────────────────────────────────────┤
  │ Config JSON (variable length):                     │
  │   json_len (4 bytes), json_bytes [...]             │
  ├─────────────────────────────────────────────────────┤
  │ Tensor table (tensor_count × 32 bytes):            │
  │   Per tensor:                                      │
  │     name_hash (8 bytes): xxhash of tensor name     │
  │     offset (8 bytes): absolute file offset         │
  │     size (4 bytes): tensor size in bytes           │
  │     ndim (4 bytes): number of dimensions           │
  │     shape[ndim] (4 bytes each): dimension sizes    │
  │     dtype (4 bytes): numpy dtype code              │
  │     crc32 (4 bytes): CRC32 of tensor data         │
  ├─────────────────────────────────────────────────────┤
  │ Padding to 64-byte alignment                       │
  ├─────────────────────────────────────────────────────┤
  │ Tensor data (computation order):                   │
  │   [Block 0: ln_1, attn, ln_2, mlp]                │
  │   [Block 1: ln_1, attn, ln_2, mlp]                │
  │   ...                                              │
  │   [Block N-1]                                      │
  │   [norm: ln_f]                                     │
  │   [embeddings: wte, wpe]                           │
  │   [lm_head (if separate from wte)]                 │
  └─────────────────────────────────────────────────────┘
"""

import struct
from typing import Tuple

# ══════════════════════════════════════════════════════════════════════════════
# Magic and version
# ══════════════════════════════════════════════════════════════════════════════

MAGIC = b"SLNC"
VERSION = 1
FLAGS_DEFAULT = 0

# ══════════════════════════════════════════════════════════════════════════════
# Field sizes (bytes)
# ══════════════════════════════════════════════════════════════════════════════

# Header fixed section
MAGIC_SIZE = 4
VERSION_SIZE = 4
FLAGS_SIZE = 4
MODEL_META_SIZE = 64  # 10 uint32 fields + 24 reserved
JSON_LEN_SIZE = 4

# Tensor table entry (variable size due to name string)
# Fixed part: offset(8) + size(4) + ndim(4) + shape[ndim](4 each) + dtype(4) + crc32(4) = 28 + ndim*4
# Variable part: name_len(4) + name_bytes[name_len]

# Alignment
ALIGNMENT = 64  # cache line alignment

# ══════════════════════════════════════════════════════════════════════════════
# Dtype codes
# ══════════════════════════════════════════════════════════════════════════════

DTYPE_FLOAT32 = 0
DTYPE_FLOAT16 = 1
DTYPE_BFLOAT16 = 2
DTYPE_INT32 = 3
DTYPE_INT64 = 4
DTYPE_UINT8 = 5

DTYPE_MAP = {
    DTYPE_FLOAT32: "float32",
    DTYPE_FLOAT16: "float16",
    DTYPE_BFLOAT16: "bfloat16",
    DTYPE_INT32: "int32",
    DTYPE_INT64: "int64",
    DTYPE_UINT8: "uint8",
}

# ══════════════════════════════════════════════════════════════════════════════
# Layout helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_header_size(json_bytes: bytes) -> int:
    """Compute total header size (aligned to ALIGNMENT)."""
    size = (
        MAGIC_SIZE
        + VERSION_SIZE
        + FLAGS_SIZE
        + MODEL_META_SIZE
        + JSON_LEN_SIZE
        + len(json_bytes)
    )
    return _align(size)


def compute_tensor_entry_size(ndim: int, name_len: int) -> int:
    """Compute size of a tensor table entry for given dimensionality and name length."""
    return 32 + ndim * 4 + name_len  # fixed fields + shape + name string


def compute_tensor_table_size(entries: list) -> int:
    """Compute total size of tensor table.

    entries: list of (name, offset, data_bytes, ndim, dtype, crc)
    """
    total = 0
    for name, _, _, ndim, _, _ in entries:
        name_bytes = name.encode() if isinstance(name, str) else name
        total += compute_tensor_entry_size(ndim, len(name_bytes))
    return total


def _align(size: int) -> int:
    """Align size to ALIGNMENT boundary."""
    return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1)


def dtype_to_code(dtype) -> int:
    """Convert numpy dtype to format code."""
    import numpy as np
    if dtype == np.float32:
        return DTYPE_FLOAT32
    elif dtype == np.float16:
        return DTYPE_FLOAT16
    elif dtype == np.int32:
        return DTYPE_INT32
    elif dtype == np.int64:
        return DTYPE_INT64
    elif dtype == np.uint8:
        return DTYPE_UINT8
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def code_to_dtype(code: int):
    """Convert format code to numpy dtype."""
    import numpy as np
    name = DTYPE_MAP.get(code)
    if name is None:
        raise ValueError(f"Unknown dtype code: {code}")
    return getattr(np, name)
