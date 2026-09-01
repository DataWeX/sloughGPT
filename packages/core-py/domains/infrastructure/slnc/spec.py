"""
.slnc binary format specification.

Defines the binary layout, magic numbers, and field sizes.
This is the single source of truth for the format.

Binary layout:
  ┌─────────────────────────────────────────────────────┐
  │ Magic (4 bytes): "SLNC"                            │
  │ Version (4 bytes): uint32 LE                       │
  │ Flags (4 bytes): uint32 LE (feature bitmask)       │
  ├─────────────────────────────────────────────────────┤
  │ Model metadata (fixed 64 bytes):                   │
  │   n_layer (4), n_embd (4), n_head (4), n_inner (4)│
  │   vocab_size (4), n_positions (4)                  │
  │   block_count (4), block_size (4)                  │
  │   tensor_count (4), data_offset (4)                │
  │   reserved (24 bytes): [header_crc32:4][unused:20] │
  ├─────────────────────────────────────────────────────┤
  │ Config JSON (variable length):                     │
  │   json_len (4 bytes), json_bytes [...]             │
  ├─────────────────────────────────────────────────────┤
  │ Padding to 64-byte alignment                       │
  ├─────────────────────────────────────────────────────┤
  │ Tensor table (variable-length entries):             │
  │   Per entry:                                       │
  │     name_len (4 bytes): tensor name length         │
  │     name_bytes [...]                               │
  │     offset (8 bytes): absolute file offset         │
  │     size (4 bytes): tensor size in bytes           │
  │     ndim (4 bytes): number of dimensions           │
  │     shape[ndim] (4 bytes each): dimension sizes    │
  │     dtype (4 bytes): numpy dtype code              │
  │     crc32 (4 bytes): CRC32 of tensor data         │
  ├─────────────────────────────────────────────────────┤
  │ Padding to 64-byte alignment                       │
  ├─────────────────────────────────────────────────────┤
  │ Tensor data (computation order, each aligned):     │
  │   [Block 0: ln_1, attn, ln_2, mlp]                │
  │   [Block 1: ln_1, attn, ln_2, mlp]                │
  │   ...                                              │
  │   [Block N-1]                                      │
  │   [norm: ln_f]                                     │
  │   [embeddings: wte, wpe]                           │
  │   [lm_head (if separate from wte)]                 │
  └─────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass

# ══════════════════════════════════════════════════════════════════════════════
# Magic and version
# ══════════════════════════════════════════════════════════════════════════════

MAGIC = b"SLNC"
VERSION = 1
FLAGS_DEFAULT = 0

# ══════════════════════════════════════════════════════════════════════════════
# Feature flags (bitmask in Flags field)
# ══════════════════════════════════════════════════════════════════════════════

FLAG_HAS_HEADER_CRC   = 0x01  # reserved region contains header CRC32
FLAG_ALIGNED_TENSORS  = 0x02  # each tensor starts at 64B-aligned offset
FLAG_HAS_FILE_HASH    = 0x08  # reserved region contains file-level hash

# ══════════════════════════════════════════════════════════════════════════════
# Field sizes (bytes)
# ══════════════════════════════════════════════════════════════════════════════

# Header fixed section
MAGIC_SIZE = 4
VERSION_SIZE = 4
FLAGS_SIZE = 4
MODEL_META_SIZE = 64  # 10 uint32 fields + 24 reserved
JSON_LEN_SIZE = 4

# Reserved region layout (24 bytes within MODEL_META)
RESERVED_SIZE = 24
HEADER_CRC_OFFSET = 0    # offset within reserved region

# Tensor table entry (variable length due to name string)
# name_len(4) + name_bytes[name_len] + offset(8) + size(4) + ndim(4) + shape[ndim](4 each) + dtype(4) + crc32(4)

# Alignment
ALIGNMENT = 64  # cache line alignment

# Bounds
MAX_NDIM = 8          # no tensor > 8D
MAX_TENSOR_COUNT = 100_000  # no model has >100K tensors
MAX_NAME_LEN = 256    # max tensor name length

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
# Config dataclass (pugqeep config pattern)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SLNCConfig:
    """SLNC format configuration — single source of truth for all tunables."""
    alignment: int = ALIGNMENT
    max_ndim: int = MAX_NDIM
    max_tensor_count: int = MAX_TENSOR_COUNT
    max_name_len: int = MAX_NAME_LEN
    verify_checksums: bool = False
    align_tensors: bool = False  # default: no alignment (backward compatible)
    write_header_crc: bool = False  # default: no header CRC (backward compatible)

    @classmethod
    def from_flags(cls, flags: int) -> "SLNCConfig":
        """Create config from file flags bitmask."""
        return cls(
            align_tensors=bool(flags & FLAG_ALIGNED_TENSORS),
            write_header_crc=bool(flags & FLAG_HAS_HEADER_CRC),
        )

    def to_flags(self) -> int:
        """Convert config to flags bitmask."""
        flags = FLAGS_DEFAULT
        if self.write_header_crc:
            flags |= FLAG_HAS_HEADER_CRC
        if self.align_tensors:
            flags |= FLAG_ALIGNED_TENSORS
        return flags


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
    # name_len(4) + name_bytes[name_len] + offset(8) + size(4) + ndim(4) + shape[ndim](4 each) + dtype(4) + crc32(4)
    return 4 + name_len + 8 + 4 + 4 + ndim * 4 + 4 + 4


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


def _align_offset(offset: int) -> int:
    """Align a file offset to ALIGNMENT boundary."""
    return _align(offset)


def dtype_to_code(dtype) -> int:
    """Convert numpy dtype to format code."""
    import numpy as np
    if dtype == np.float32:
        return DTYPE_FLOAT32
    elif dtype == np.float16:
        return DTYPE_FLOAT16
    elif hasattr(np, 'bfloat16') and dtype == np.bfloat16:
        return DTYPE_BFLOAT16
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
    if code == DTYPE_BFLOAT16:
        # bfloat16 not in standard numpy; return uint16 as storage dtype
        return np.uint16
    name = DTYPE_MAP.get(code)
    if name is None or not hasattr(np, name):
        raise ValueError(f"Unknown dtype code: {code}")
    return getattr(np, name)
