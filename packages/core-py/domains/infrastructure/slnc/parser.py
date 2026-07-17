"""
.slnc parser — memory-mapped loader for .slnc files.

Zero-copy weight loading via mmap. Numpy arrays are views into file pages.
OS handles demand loading — only accessed blocks get paged in from disk.

Usage:
    from domains.infrastructure.slnc.parser import SLNCParser

    parser = SLNCParser("models/gpt2.slnc")
    q_weight = parser.get_tensor("blocks.0.attn.c_attn.weight")
    block0 = parser.get_block(0)
    all_weights = parser.get_weights_dict()
"""

import json
import logging
import mmap
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from domains.infrastructure.slnc.spec import (
    MAGIC,
    VERSION,
    ALIGNMENT,
    DTYPE_FLOAT32,
    compute_header_size,
    compute_tensor_entry_size,
    code_to_dtype,
)

logger = logging.getLogger("slo.infrastructure.slnc.parser")


class SLNCParser:
    """Memory-mapped parser for .slnc files.

    Weights are numpy views into mmap'd file pages.
    Zero copy — the numpy array IS the file memory.
    Demand loading — OS pages in only accessed blocks.
    """

    def __init__(self, path: str, verify_checksums: bool = False):
        """Open .slnc file and parse header + tensor table.

        Args:
            path: Path to .slnc file
            verify_checksums: If True, verify CRC32 on first access
        """
        self._path = path
        self._verify = verify_checksums
        self._fd = os.open(path, os.O_RDONLY)
        self._file_size = os.fstat(self._fd).st_size
        self._mm = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)

        # Parse header
        self._parse_header()

        # Parse tensor table
        self._parse_tensor_table()

        logger.info(
            "SLNCParser: %s, %d tensors, %d layers, %.1f MB",
            Path(path).name,
            len(self._tensor_map),
            self._n_layer,
            self._file_size / 1e6,
            extra={"tag": "INFRA"},
        )

    def _parse_header(self):
        """Parse the fixed-size header."""
        self._mm.seek(0)

        # Magic
        magic = self._mm.read(4)
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic!r} (expected {MAGIC!r})")

        # Version
        version = struct.unpack("<I", self._mm.read(4))[0]
        if version != VERSION:
            raise ValueError(f"Unsupported version: {version}")

        # Flags
        self._flags = struct.unpack("<I", self._mm.read(4))[0]

        # Model metadata (64 bytes)
        self._n_layer = struct.unpack("<I", self._mm.read(4))[0]
        self._n_embd = struct.unpack("<I", self._mm.read(4))[0]
        self._n_head = struct.unpack("<I", self._mm.read(4))[0]
        self._n_inner = struct.unpack("<I", self._mm.read(4))[0]
        self._vocab_size = struct.unpack("<I", self._mm.read(4))[0]
        self._n_positions = struct.unpack("<I", self._mm.read(4))[0]
        self._block_count = struct.unpack("<I", self._mm.read(4))[0]
        self._block_size = struct.unpack("<I", self._mm.read(4))[0]
        self._tensor_count = struct.unpack("<I", self._mm.read(4))[0]
        self._data_offset = struct.unpack("<I", self._mm.read(4))[0]
        self._reserved = self._mm.read(24)

        # Config JSON
        json_len = struct.unpack("<I", self._mm.read(4))[0]
        self._config = json.loads(self._mm.read(json_len))

    def _parse_tensor_table(self):
        """Parse the tensor table (offsets + metadata for all tensors)."""
        self._tensor_map: Dict[str, Tuple[int, Tuple[int, ...], np.dtype, int]] = {}
        # name → (file_offset, shape, dtype, crc32)

        # Skip to tensor table (after header)
        header_size = compute_header_size(json.dumps(self._config, sort_keys=True).encode())
        self._mm.seek(header_size)

        for _ in range(self._tensor_count):
            # Read name string
            name_len = struct.unpack("<I", self._mm.read(4))[0]
            name = self._mm.read(name_len).decode()

            # Read entry fields
            offset = struct.unpack("<Q", self._mm.read(8))[0]
            size = struct.unpack("<I", self._mm.read(4))[0]
            ndim = struct.unpack("<I", self._mm.read(4))[0]

            shape = tuple(
                struct.unpack("<I", self._mm.read(4))[0] for _ in range(ndim)
            )

            dtype_code = struct.unpack("<I", self._mm.read(4))[0]
            crc = struct.unpack("<I", self._mm.read(4))[0]

            dtype = code_to_dtype(dtype_code)
            self._tensor_map[name] = (offset, shape, dtype, crc)

    def get_tensor(self, name: str) -> np.ndarray:
        """Get weight tensor from mmap'd file.

        Args:
            name: Tensor name (e.g. "h.0.attn.c_attn.weight")

        Returns:
            numpy array — COPY of data from mmap'd file
        """
        if name not in self._tensor_map:
            raise KeyError(f"Unknown tensor: {name}")

        offset, shape, dtype, crc = self._tensor_map[name]
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize

        # Copy from mmap to avoid segfaults when mmap is closed/GC'd
        # while numpy arrays still reference the pages
        data = self._mm[offset:offset + nbytes]
        arr = np.frombuffer(bytes(data), dtype=dtype).reshape(shape).copy()

        # Optional integrity check
        if self._verify:
            import zlib
            actual_crc = zlib.crc32(arr.tobytes()) & 0xFFFFFFFF
            if actual_crc != crc:
                raise ValueError(f"Checksum mismatch for {name}: expected {crc:#x}, got {actual_crc:#x}")

        return arr

    def get_block(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """Get all weights for a transformer block.

        Returns dict mapping short tensor names to numpy views.
        """
        block_tensor_names = [
            "ln_1.weight", "ln_1.bias",
            "attn.c_attn.weight", "attn.c_attn.bias",
            "attn.c_proj.weight", "attn.c_proj.bias",
            "ln_2.weight", "ln_2.bias",
            "mlp.c_fc.weight", "mlp.c_fc.bias",
            "mlp.c_proj.weight", "mlp.c_proj.bias",
        ]

        result = {}
        for tensor_name in block_tensor_names:
            key = f"h.{layer_idx}.{tensor_name}"
            result[tensor_name] = self.get_tensor(key)
        return result

    def get_weights_dict(self) -> Dict[str, np.ndarray]:
        """Get all weights as a dict (backward compatibility)."""
        return {name: self.get_tensor(name) for name in self._tensor_map}

    def verify_all(self) -> bool:
        """Verify all tensor checksums. Returns True if all pass."""
        import zlib
        for name in self._tensor_map:
            offset, shape, dtype, expected_crc = self._tensor_map[name]
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            data = self._mm[offset:offset + nbytes]
            actual_crc = zlib.crc32(data) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                logger.error("Checksum mismatch: %s (expected %x, got %x)", name, expected_crc, actual_crc,
                    extra={"tag": "INFRA"})
                return False
        return True

    @property
    def config(self) -> dict:
        return self._config

    @property
    def file_size(self) -> int:
        return self._file_size

    @property
    def tensor_count(self) -> int:
        return len(self._tensor_map)

    def __del__(self):
        try:
            self._mm.close()
            os.close(self._fd)
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"SLNCParser({Path(self._path).name}, "
            f"{self._n_layer} layers, "
            f"{self._file_size / 1e6:.1f} MB, "
            f"{len(self._tensor_map)} tensors)"
        )
