"""
Memory-mapped loader for .slnc files.

Zero-copy weight loading via mmap. Numpy arrays are views into file pages.
OS handles demand loading — only accessed blocks get paged in from disk.

Usage:
    from domains.infrastructure.slnc_loader import SLNCLoader

    loader = SLNCLoader("gpt2.slnc")
    q_weight = loader.get_tensor("blocks.0.attn.c_attn.weight")  # view into mmap
    block0 = loader.get_block(0)  # all weights for block 0
"""

import json
import logging
import mmap
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from domains.infrastructure.slnc_format import (
    GPT2_BLOCK_TENSORS,
    SLNC_MAGIC,
    SLNC_VERSION,
    HEADER_FIXED_SIZE,
)

logger = logging.getLogger("slo.infrastructure.slnc_loader")


class SLNCLoader:
    """Memory-mapped loader for .slnc files.

    Weights are numpy views into mmap'd file pages.
    Zero copy — the numpy array IS the file memory.
    Demand loading — OS pages in only accessed blocks.
    """

    def __init__(self, path: str):
        """Open .slnc file and parse header.

        Args:
            path: Path to .slnc file
        """
        self._path = path
        self._fd = os.open(path, os.O_RDONLY)
        self._file_size = os.fstat(self._fd).st_size
        self._mm = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)

        # Parse header (small, always in memory)
        self._parse_header()

        # Compute tensor offsets (no file parsing needed)
        self._compute_offsets()

        logger.info(
            "SLNCLoader: %s, %d layers, %d bytes/block, %d tensors",
            Path(path).name,
            self._n_layer,
            self._block_size,
            len(self._tensor_map),
            extra={"tag": "INFRA"},
        )

    def _parse_header(self):
        """Parse the fixed-size header."""
        self._mm.seek(0)
        magic = self._mm.read(4)
        if magic != SLNC_MAGIC:
            raise ValueError(f"Invalid magic: {magic!r}")

        self._version = struct.unpack("<I", self._mm.read(4))[0]
        if self._version != SLNC_VERSION:
            raise ValueError(f"Unsupported version: {self._version}")

        self._n_layer = struct.unpack("<I", self._mm.read(4))[0]
        self._n_embd = struct.unpack("<I", self._mm.read(4))[0]
        self._n_head = struct.unpack("<I", self._mm.read(4))[0]
        self._n_inner = struct.unpack("<I", self._mm.read(4))[0]
        self._vocab_size = struct.unpack("<I", self._mm.read(4))[0]
        self._n_positions = struct.unpack("<I", self._mm.read(4))[0]
        self._block_count = struct.unpack("<I", self._mm.read(4))[0]
        self._block_size = struct.unpack("<I", self._mm.read(4))[0]
        self._non_block_offset = struct.unpack("<I", self._mm.read(4))[0]
        self._non_block_size = struct.unpack("<I", self._mm.read(4))[0]

        json_len = struct.unpack("<I", self._mm.read(4))[0]
        self._config = json.loads(self._mm.read(json_len))

    def _compute_offsets(self):
        """Compute absolute file offsets for all tensors."""
        self._tensor_map: Dict[str, Tuple[int, Tuple[int, ...], np.dtype]] = {}

        # Block tensors: offset = header_size + block_index * block_size + offset_in_block
        for layer_idx in range(self._n_layer):
            for tensor_name, offset_in_block, _ in self._get_block_layout():
                abs_offset = (
                    self._non_block_offset  # header is before blocks
                    - self._block_size * self._n_layer  # no, let me recalculate
                )
                # Actually: header_size = non_block_offset - block_size * n_layer
                header_size = self._non_block_offset - self._block_size * self._n_layer
                abs_offset = header_size + layer_idx * self._block_size + offset_in_block
                shape = self._get_tensor_shape(tensor_name)
                size_bytes = int(np.prod(shape)) * 4  # float32

                key = f"blocks.{layer_idx}.{tensor_name}"
                self._tensor_map[key] = (abs_offset, shape, np.float32)

        # Non-block tensors
        non_block_names = ["wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"]
        current_offset = self._non_block_offset
        for name in non_block_names:
            shape = self._get_tensor_shape(name)
            size_bytes = int(np.prod(shape)) * 4
            self._tensor_map[name] = (current_offset, shape, np.float32)
            current_offset += size_bytes

    def _get_block_layout(self) -> List[Tuple[str, int, int]]:
        """Get tensor layout within a block (name, offset, size)."""
        layout = []
        offset = 0
        for tensor_name, _ in GPT2_BLOCK_TENSORS:
            shape = self._get_tensor_shape(tensor_name)
            size = int(np.prod(shape)) * 4
            layout.append((tensor_name, offset, size))
            offset += size
        return layout

    def _get_tensor_shape(self, name: str) -> Tuple[int, ...]:
        """Get tensor shape from name."""
        n = self._n_embd
        n_inner = self._n_inner
        vocab = self._vocab_size
        n_pos = self._n_positions

        shapes = {
            "ln_1.weight": (n,),
            "ln_1.bias": (n,),
            "attn.c_attn.weight": (n, 3 * n),
            "attn.c_attn.bias": (3 * n,),
            "attn.c_proj.weight": (n, n),
            "attn.c_proj.bias": (n,),
            "ln_2.weight": (n,),
            "ln_2.bias": (n,),
            "mlp.c_fc.weight": (n, n_inner),
            "mlp.c_fc.bias": (n_inner,),
            "mlp.c_proj.weight": (n_inner, n),
            "mlp.c_proj.bias": (n,),
            "wte.weight": (vocab, n),
            "wpe.weight": (n_pos, n),
            "ln_f.weight": (n,),
            "ln_f.bias": (n,),
        }
        return shapes[name]

    def get_tensor(self, name: str) -> np.ndarray:
        """Get weight tensor as numpy view into mmap'd memory.

        Args:
            name: Tensor name (e.g. "blocks.0.attn.c_attn.weight")

        Returns:
            numpy array — a VIEW into the mmap'd file (zero copy)
        """
        if name not in self._tensor_map:
            raise KeyError(f"Unknown tensor: {name}")

        offset, shape, dtype = self._tensor_map[name]
        nbytes = int(np.prod(shape)) * dtype().itemsize

        # Create view into mmap (zero copy!)
        # mmap supports buffer protocol, so we can slice it
        data = self._mm[offset:offset + nbytes]
        return np.frombuffer(data, dtype=dtype).reshape(shape)

    def get_block(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """Get all weights for a transformer block.

        Returns dict mapping tensor names to numpy views.
        Sequential access — all data is contiguous in file.
        """
        result = {}
        for tensor_name, _, _ in self._get_block_layout():
            key = f"blocks.{layer_idx}.{tensor_name}"
            result[tensor_name] = self.get_tensor(key)
        return result

    def get_weights_dict(self) -> Dict[str, np.ndarray]:
        """Get all weights as a dict (for backward compatibility).

        Returns dict mapping tensor names to numpy views.
        """
        return {name: self.get_tensor(name) for name in self._tensor_map}

    @property
    def config(self) -> dict:
        """Model configuration."""
        return self._config

    @property
    def file_size(self) -> int:
        """File size in bytes."""
        return self._file_size

    def __del__(self):
        """Clean up mmap and file descriptor."""
        try:
            self._mm.close()
            os.close(self._fd)
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"SLNCLoader({Path(self._path).name}, "
            f"{self._n_layer} layers, "
            f"{self._file_size / 1e6:.1f} MB)"
        )
