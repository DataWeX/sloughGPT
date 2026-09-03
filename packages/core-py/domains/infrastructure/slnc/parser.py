from __future__ import annotations

"""
.slnc parser — memory-mapped loader for .slnc files.

True zero-copy weight loading via mmap. Numpy arrays are views into file
pages. OS handles demand loading — only accessed blocks get paged in from disk.

Usage:
    from domains.infrastructure.slnc.parser import SLNCParser

    parser = SLNCParser("models/gpt2.slnc")
    q_weight = parser.get_tensor("h.0.attn.c_attn.weight")  # zero-copy view
    block0 = parser.get_block(0)
    all_weights = parser.get_weights_dict()  # still zero-copy
"""

import json
import logging
import mmap
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from domains.infrastructure.slnc.spec import (
    MAGIC,
    VERSION,
    MAX_NDIM,
    MAX_TENSOR_COUNT,
    MAX_NAME_LEN,
    FLAG_HAS_HEADER_CRC,
    FLAG_ALIGNED_TENSORS,
    SLNCConfig,
    compute_header_size,
    code_to_dtype,
)

logger = logging.getLogger("slo.infrastructure.slnc.parser")


class SLNCParser:
    """Memory-mapped parser for .slnc files.

    Weights are numpy views into mmap'd file pages.
    True zero copy — the numpy array IS the file memory.
    Demand loading — OS pages in only accessed blocks.
    """

    def __init__(
        self,
        path: str,
        verify_checksums: bool = False,
        config: Optional[SLNCConfig] = None,
    ):
        """Open .slnc file and parse header + tensor table.

        Args:
            path: Path to .slnc file
            verify_checksums: If True, verify CRC32 on first access
            config: Optional config overrides
        """
        self._path = path
        self._verify = verify_checksums
        self._config = config or SLNCConfig()
        self._fd = os.open(path, os.O_RDONLY)
        self._file_size = os.fstat(self._fd).st_size

        # Read header + tensor table via os.read (fast, ~KB)
        self._header_data = self._read_header_and_table()

        # mmap for random tensor access during inference
        self._mm = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)

        # Parse header from buffered data
        self._parse_header()

        # Parse tensor table from buffered data
        self._parse_tensor_table()

        logger.info(
            "SLNCParser: %s, %d tensors, %d layers, %.1f MB",
            Path(path).name,
            len(self._tensor_map),
            self._n_layer,
            self._file_size / 1e6,
            extra={"tag": "INFRA"},
        )

    def _read_header_and_table(self) -> bytes:
        """Read header + tensor table via sequential os.read (~KB, fast).

        Only reads what's needed for init. Tensor data stays on disk
        until accessed via mmap during inference.
        """
        # Read first 64KB — covers header + tensor table for most models
        header_size = 64 * 1024
        data = os.read(self._fd, min(header_size, self._file_size))
        os.lseek(self._fd, 0, os.SEEK_SET)
        return data

    def _parse_header(self):
        """Parse the fixed-size header from buffered data (fast, no mmap)."""
        buf = self._header_data
        pos = 0

        # Magic
        magic = buf[pos:pos + 4]
        pos += 4
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic!r} (expected {MAGIC!r})")

        # Version — soft check
        version = struct.unpack("<I", buf[pos:pos + 4])[0]
        pos += 4
        if version > VERSION:
            logger.warning(
                "SLNC version %d > supported %d — some features may be unavailable",
                version, VERSION,
                extra={"tag": "INFRA"},
            )

        # Flags
        self._flags = struct.unpack("<I", buf[pos:pos + 4])[0]
        pos += 4

        # Decode flags
        self._has_header_crc = bool(self._flags & FLAG_HAS_HEADER_CRC)
        self._aligned_tensors = bool(self._flags & FLAG_ALIGNED_TENSORS)

        # Model metadata (64 bytes → 10 × uint32)
        self._n_layer = struct.unpack("<I", buf[pos:pos + 4])[0]; pos += 4
        self._n_embd = struct.unpack("<I", buf[pos:pos + 4])[0]; pos += 4
        self._n_head = struct.unpack("<I", buf[pos:pos + 4])[0]; pos += 4
        self._n_inner = struct.unpack("<I", buf[pos:pos + 4])[0]; pos += 4
        self._vocab_size = struct.unpack("<I", buf[pos:pos + 4])[0]; pos += 4
        self._n_positions = struct.unpack("<I", buf[pos:pos + 4])[0]; pos += 4
        self._block_count = struct.unpack("<I", buf[pos:pos + 4])[0]; pos += 4
        self._block_size = struct.unpack("<I", buf[pos:pos + 4])[0]; pos += 4
        self._tensor_count = struct.unpack("<I", buf[pos:pos + 4])[0]; pos += 4
        self._data_offset = struct.unpack("<I", buf[pos:pos + 4])[0]; pos += 4

        # Reserved region (24 bytes)
        reserved = buf[pos:pos + 24]
        pos += 24
        self._header_crc = struct.unpack("<I", reserved[:4])[0]

        # Validate tensor count
        if self._tensor_count > MAX_TENSOR_COUNT:
            raise ValueError(f"Too many tensors: {self._tensor_count} (max {MAX_TENSOR_COUNT})")

        # Config JSON — still in buffered data (after fixed header)
        json_len = struct.unpack("<I", buf[pos:pos + 4])[0]
        pos += 4
        self._config_dict = json.loads(buf[pos:pos + json_len])

        # Verify header CRC if present
        if self._has_header_crc and self._header_crc != 0:
            self._verify_header_crc()

    def _verify_header_crc(self):
        """Verify header integrity via CRC32."""
        import zlib
        # Read entire header up to tensor table start
        header_size = compute_header_size(
            json.dumps(self._config_dict, sort_keys=True).encode()
        )
        self._mm.seek(0)
        header_data = self._mm.read(header_size)
        actual_crc = zlib.crc32(header_data) & 0xFFFFFFFF
        if actual_crc != self._header_crc:
            raise ValueError(
                f"Header CRC mismatch: expected {self._header_crc:#x}, got {actual_crc:#x}"
            )

    def _parse_tensor_table(self):
        """Parse the tensor table from buffered data (fast, no mmap)."""
        self._tensor_map: Dict[str, Tuple[int, Tuple[int, ...], np.dtype, int]] = {}

        header_size = compute_header_size(
            json.dumps(self._config_dict, sort_keys=True).encode()
        )
        buf = self._header_data
        pos = header_size

        for _ in range(self._tensor_count):
            # Read name string
            name_len = struct.unpack("<I", buf[pos:pos + 4])[0]
            pos += 4
            if name_len > MAX_NAME_LEN:
                raise ValueError(f"Tensor name too long: {name_len} > {MAX_NAME_LEN}")
            name = buf[pos:pos + name_len].decode()
            pos += name_len

            # Read entry fields
            offset = struct.unpack("<Q", buf[pos:pos + 8])[0]
            pos += 8
            size = struct.unpack("<I", buf[pos:pos + 4])[0]
            pos += 4
            ndim = struct.unpack("<I", buf[pos:pos + 4])[0]
            pos += 4

            # Validate ndim
            if ndim > MAX_NDIM:
                raise ValueError(f"Tensor {name!r} has {ndim} dims (max {MAX_NDIM})")

            shape = tuple(
                struct.unpack("<I", buf[pos + i * 4:pos + (i + 1) * 4])[0]
                for i in range(ndim)
            )
            pos += ndim * 4

            dtype_code = struct.unpack("<I", buf[pos:pos + 4])[0]
            pos += 4
            crc = struct.unpack("<I", buf[pos:pos + 4])[0]
            pos += 4

            dtype = code_to_dtype(dtype_code)
            self._tensor_map[name] = (offset, shape, dtype, crc)

    def get_tensor(self, name: str) -> np.ndarray:
        """Get weight tensor from mmap'd file.

        Args:
            name: Tensor name (e.g. "h.0.attn.c_attn.weight")

        Returns:
            numpy array — TRUE ZERO-COPY VIEW into mmap'd file
        """
        if name not in self._tensor_map:
            raise KeyError(f"Unknown tensor: {name}")

        offset, shape, dtype, crc = self._tensor_map[name]
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize

        # True zero-copy: view into mmap without .copy()
        arr = np.frombuffer(self._mm[offset:offset + nbytes], dtype=dtype).reshape(shape)

        # Optional integrity check
        if self._verify:
            import zlib
            actual_crc = zlib.crc32(arr.tobytes()) & 0xFFFFFFFF
            if actual_crc != crc:
                raise ValueError(f"Checksum mismatch for {name}: expected {crc:#x}, got {actual_crc:#x}")

        return arr

    def get_tensor_copy(self, name: str) -> np.ndarray:
        """Get weight tensor as an independent writable copy.

        Use this when you need to modify the tensor (e.g. for inference
        computation). For read-only access, prefer get_tensor() which
        returns a zero-copy view.

        Args:
            name: Tensor name

        Returns:
            numpy array — independent copy that can be freely modified
        """
        return self.get_tensor(name).copy()

    def get_tensor_info(self, name: str) -> Tuple[int, Tuple[int, ...], np.dtype, int]:
        """Get tensor metadata without reading data.

        Returns:
            (offset, shape, dtype, crc) — file offset, array shape, element dtype, CRC32

        Raises:
            KeyError: if tensor name not found
        """
        if name not in self._tensor_map:
            raise KeyError(f"Unknown tensor: {name}")
        return self._tensor_map[name]

    def read_tensor_region(self, name: str) -> np.ndarray:
        """Read tensor directly from mmap into numpy array (writable copy).

        Like get_tensor_copy() but named for backward compatibility.
        """
        offset, shape, dtype, crc = self.get_tensor_info(name)
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        return np.frombuffer(self._mm[offset:offset + nbytes], dtype=dtype).reshape(shape).copy()

    @property
    def tensor_names(self) -> List[str]:
        """List of all tensor names in the file."""
        return list(self._tensor_map.keys())

    def get_block(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """Get all weights for a transformer block."""
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
            if key in self._tensor_map:
                result[tensor_name] = self.get_tensor(key)
        return result

    def get_weights_dict(self) -> Dict[str, np.ndarray]:
        """Get all weights as a dict (zero-copy views)."""
        return {name: self.get_tensor(name) for name in self._tensor_map}

    def get_weights_dict_parallel(self, max_workers: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Get all weights as a dict using parallel tensor loading.

        Numpy operations release the GIL, so a thread pool can load multiple
        tensors concurrently. For models with many tensors (e.g. 0.5B+),
        this can significantly reduce wall-clock load time on multi-core CPUs.

        Args:
            max_workers: Thread pool size. Defaults to min(32, os.cpu_count() + 4).

        Returns:
            Dict mapping tensor names → numpy arrays (zero-copy views).
        """
        names = list(self._tensor_map.keys())

        def _load(name):
            return name, self.get_tensor(name)

        result = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for name, arr in pool.map(lambda n: _load(n), names):
                result[name] = arr
        return result

    def release_file_pages(self) -> bool:
        """Discard resident file-backed pages back to the OS."""
        if self._mm is None:
            return False
        try:
            self._mm.madvise(mmap.MADV_DONTNEED)
            return True
        except Exception:
            logger.debug("release_file_pages: madvise not supported", extra={"tag": "INFRA"})
            return False

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
        return self._config_dict

    @property
    def file_size(self) -> int:
        return self._file_size

    @property
    def tensor_count(self) -> int:
        return len(self._tensor_map)

    @property
    def param_count(self) -> int:
        """Total number of model parameters (sum of tensor elements)."""
        return int(sum(np.prod(shape) for (_, shape, _, _) in self._tensor_map.values()))

    @property
    def n_layer(self) -> int:
        return self._n_layer

    @property
    def n_embd(self) -> int:
        return self._n_embd

    @property
    def n_head(self) -> int:
        return self._n_head

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def n_positions(self) -> int:
        return self._n_positions

    def close(self) -> None:
        """Close the mmap and file descriptor."""
        try:
            if self._mm is not None:
                self._mm.close()
        except Exception as exc:
            logger.debug("mmap close failed: %s", exc)
        try:
            os.close(self._fd)
        except OSError as exc:
            logger.debug("fd close failed: %s", exc)
        self._mm = None

    def __del__(self):
        try:
            if self._mm is not None:
                self._mm.close()
        except Exception as exc:
            logger.debug("mmap close failed in __del__: %s", exc)
        try:
            os.close(self._fd)
        except OSError as exc:
            logger.debug("fd close failed in __del__: %s", exc)

    def __repr__(self) -> str:
        return (
            f"SLNCParser({Path(self._path).name}, "
            f"{self._n_layer} layers, "
            f"{self._file_size / 1e6:.1f} MB, "
            f"{len(self._tensor_map)} tensors)"
        )


# ── Model directory utilities ────────────────────────────────────────────────

def get_model_dir(model_id: str) -> Path:
    """Resolve HuggingFace cache directory for a model.

    Searches the standard HF cache first, then the project-local
    cache (models/hf-cache/hub/) mirroring MorphTokenizer.from_pretrained.
    """
    from domains.shared import find_repo_root
    cache_id = model_id.replace("/", "--")
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    candidates = [
        Path(hf_home) / "hub" / f"models--{cache_id}",
        find_repo_root(Path(__file__).resolve()) / "models" / "hf-cache" / "hub" / f"models--{cache_id}",
        Path("models/hf-cache/hub") / f"models--{cache_id}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def find_safetensors(model_dir: Path) -> Optional[Path]:
    """Find the safetensors file in a model directory."""
    snapshots = model_dir / "snapshots"
    if snapshots.exists():
        for snapshot in snapshots.iterdir():
            st = snapshot / "model.safetensors"
            if st.exists():
                return st
    st = model_dir / "model.safetensors"
    if st.exists():
        return st
    return None


def load_model_config(model_id: str) -> dict:
    """Load model config.json from HuggingFace cache."""
    model_dir = get_model_dir(model_id)
    config_path = None
    snapshots = model_dir / "snapshots"
    if snapshots.exists():
        for snap in snapshots.iterdir():
            c = snap / "config.json"
            if c.exists():
                config_path = c
                break
    if config_path is None:
        config_path = model_dir / "config.json"
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"No config.json for {model_id}")
    with open(config_path) as f:
        return json.load(f)
