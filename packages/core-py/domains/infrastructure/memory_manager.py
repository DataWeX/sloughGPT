"""
Custom memory manager for neural network weights.

Tiered "swap" system:
  Hot tier:  uncompressed numpy arrays in RAM (fastest access)
  Warm tier: compressed arrays in RAM (fast decompression)
  Cold tier: .slnc file on disk (mmap or read)

Design principles:
  - We know the access pattern (sequential transformer blocks)
  - We can prefetch Block N+1 while computing Block N
  - Lossless compression for accuracy preservation
  - Configurable tier sizes based on available RAM

Compression:
  - ZSTD: good balance (~3 GB/s, ~3x ratio), lossless
  - Quantize: int8/int4 for extreme compression (lossy, not for GPT-2)

Quantization:
  - TensorInfo wraps every weight with optional quantization metadata
  - Per-tensor scale/zero_point (symmetric or asymmetric)
  - Calibration support: collect min/max from representative data
  - Error metrics: MSE, max_abs_error, cosine similarity per tensor

Usage:
    from domains.infrastructure.memory_manager import WeightManager
    from domains.infrastructure.quantization import TensorInfo

    manager = WeightManager.from_slnc("models/gpt2.slnc", ram_budget_mb=512)
    q_weight = manager.get("blocks.0.attn.q_proj.weight")  # transparent decompress
    manager.prefetch_block(1)  # prefetch next block

    # Quantization-aware loading:
    manager.quantize_all(bits=8, mode="asymmetric", clip_percentile=0.999)
    info = manager.get_info("blocks.0.q_proj.weight")
    weight = info.as_float()  # dequantized on the fly
"""

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from domains.infrastructure.quantization import TensorInfo, QuantEngine, QuantMeta

logger = logging.getLogger("man.infrastructure.memory_manager")


class Tier(Enum):
    HOT = "hot"      # uncompressed in RAM
    WARM = "warm"    # compressed in RAM
    COLD = "cold"    # on disk


class Compression(Enum):
    NONE = "none"
    ZSTD = "zstd"  # lossless, ~3x ratio, ~3 GB/s decompress
    QUANTIZE_INT8 = "quantize_int8"  # lossy, ~4x ratio, slower
    QUANTIZE_INT4 = "quantize_int4"  # lossy, ~8x ratio, slowest


@dataclass
class WeightEntry:
    """Metadata for a single weight tensor."""
    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    nbytes: int              # uncompressed size
    tier: Tier = Tier.COLD
    compressed_data: Optional[bytes] = None
    compressed_nbytes: int = 0
    last_access: float = 0.0
    access_count: int = 0
    pinned: bool = False     # if True, never evict (e.g., embedding table)
    quant_info: Optional[TensorInfo] = None  # quantization wrapper


@dataclass
class TierStats:
    """Statistics for a memory tier."""
    count: int = 0
    uncompressed_bytes: int = 0
    compressed_bytes: int = 0

    @property
    def ratio(self) -> float:
        if self.compressed_bytes == 0:
            return 1.0
        return self.uncompressed_bytes / self.compressed_bytes


class WeightManager:
    """Tiered memory manager for neural network weights.

    Manages hot/warm/cold tiers with automatic promotion/demotion
    based on access patterns. Supports per-tensor quantization with
    calibration metadata.
    """

    def __init__(
        self,
        ram_budget_mb: float = 512,
        hot_ratio: float = 0.2,      # 20% of RAM for hot tier
        warm_ratio: float = 0.5,     # 50% of RAM for warm tier
        compression: Compression = Compression.ZSTD,
        page_size: int = 1024 * 1024,  # 1MB pages
    ):
        """Initialize memory manager.

        Args:
            ram_budget_mb: Total RAM budget in MB
            hot_ratio: Fraction of budget for hot tier (uncompressed)
            warm_ratio: Fraction of budget for warm tier (compressed)
            compression: Compression algorithm for warm tier
            page_size: Size of memory pages (for alignment)
        """
        self._ram_budget = int(ram_budget_mb * 1024 * 1024)
        self._hot_budget = int(self._ram_budget * hot_ratio)
        self._warm_budget = int(self._ram_budget * warm_ratio)
        self._compression = compression
        self._page_size = page_size

        # Weight storage
        self._hot: Dict[str, np.ndarray] = {}        # name → uncompressed array
        self._warm: Dict[str, bytes] = {}            # name → compressed bytes
        self._metadata: Dict[str, WeightEntry] = {}  # name → metadata
        self._tensor_infos: Dict[str, TensorInfo] = {}  # name → quantized wrapper

        # LRU eviction tracking
        self._hot_lru: OrderedDict[str, None] = OrderedDict()
        self._warm_lru: OrderedDict[str, None] = OrderedDict()

        # Quantization engine (lazy init)
        self._quant_engine: Optional[QuantEngine] = None

        # Stats
        self._stats = {
            "hot_hits": 0,
            "warm_hits": 0,
            "cold_hits": 0,
            "evictions": 0,
            "decompressions": 0,
        }

        # Prefetch state
        self._prefetch_queue: List[int] = []

        logger.info(
            "WeightManager: budget=%dMB, hot=%dMB, warm=%dMB, compression=%s",
            ram_budget_mb,
            self._hot_budget // (1024 * 1024),
            self._warm_budget // (1024 * 1024),
            compression.value,
        )

    def load_from_slnc(self, path: str):
        """Load weight metadata from .slnc file (lazy — no data loaded yet)."""
        from domains.infrastructure.slnc.parser import SLNCParser

        parser = SLNCParser(path)
        self._parser = parser
        self._config = parser.config

        # Register all tensors (cold tier)
        for name in parser._tensor_map:
            offset, shape, dtype, crc = parser._tensor_map[name]
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize

            self._metadata[name] = WeightEntry(
                name=name,
                shape=shape,
                dtype=dtype,
                nbytes=nbytes,
                tier=Tier.COLD,
            )

        logger.info(
            "Loaded %d tensors from %s (%.1f MB total)",
            len(self._metadata),
            path,
            sum(e.nbytes for e in self._metadata.values()) / 1e6,
        )

    # ══════════════════════════════════════════════════════════════════════════════
    # Quantization
    # ══════════════════════════════════════════════════════════════════════════════

    def quantize_all(
        self,
        bits: int = 8,
        mode: str = "symmetric",
        clip_percentile: Optional[float] = None,
    ):
        """Quantize all registered weights in-place.

        Args:
            bits: 8 or 4
            mode: "symmetric" or "asymmetric"
            clip_percentile: outlier clipping (e.g., 0.999)
        """
        self._quant_engine = QuantEngine(
            bits=bits,
            mode=mode,
            clip_percentile=clip_percentile,
        )

        quantized_count = 0
        skipped_count = 0

        for name in list(self._metadata.keys()):
            # Load the raw array
            raw = self._get_raw(name)

            # Quantize
            info = self._quant_engine.quantize(name, raw)

            if info.is_quantized:
                self._tensor_infos[name] = info
                self._metadata[name].quant_info = info
                quantized_count += 1
            else:
                skipped_count += 1

        summary = self._quant_engine.summary()
        logger.info(
            "WeightManager.quantize_all: %d quantized, %d skipped (bits=%d, mode=%s)",
            quantized_count, skipped_count, bits, mode,
        )
        if summary.get("tensors", 0) > 0:
            logger.info(
                "  avg_mse=%.6f, avg_cosine=%.4f, worst=%s",
                summary["avg_mse"], summary["avg_cosine_sim"], summary["worst_tensor"],
            )

    def quantize_tensor(self, name: str, bits: int = 8, mode: str = "symmetric") -> TensorInfo:
        """Quantize a single tensor and return its TensorInfo.

        Useful for selective quantization (e.g., quantize all but sensitive layers).
        """
        if self._quant_engine is None:
            self._quant_engine = QuantEngine(bits=bits, mode=mode)

        raw = self._get_raw(name)
        info = self._quant_engine.quantize(name, raw)

        if info.is_quantized:
            self._tensor_infos[name] = info
            self._metadata[name].quant_info = info

        return info

    def get_quant_error_report(self) -> Dict[str, Dict[str, Any]]:
        """Get per-tensor quantization error metrics."""
        if self._quant_engine is None:
            return {}
        return self._quant_engine.error_report()

    def get_quant_summary(self) -> Dict[str, Any]:
        """Get aggregate quantization summary."""
        if self._quant_engine is None:
            return {"tensors": 0}
        return self._quant_engine.summary()

    def save_quant_metadata(self, path: str):
        """Save quantization metadata to JSON."""
        if self._quant_engine is not None:
            self._quant_engine.save_metadata(path)

    def load_quant_metadata(self, path: str):
        """Load quantization metadata from JSON."""
        if self._quant_engine is None:
            self._quant_engine = QuantEngine()
        self._quant_engine.load_metadata(path)

    def get(self, name: str) -> np.ndarray:
        """Get weight tensor as float32 (transparent tier access).

        If quantized, returns dequantized float32. Otherwise returns
        the original array (possibly cast to float32).
        """
        info = self.get_info(name)
        return info.as_float()

    def get_info(self, name: str) -> TensorInfo:
        """Get weight as TensorInfo (may be quantized).

        Returns TensorInfo with .as_float() for dequantized access.
        """
        if name not in self._metadata:
            raise KeyError(f"Unknown weight: {name}")

        entry = self._metadata[name]
        entry.last_access = time.monotonic()
        entry.access_count += 1

        # If we have a cached TensorInfo, return it
        if name in self._tensor_infos:
            return self._tensor_infos[name]

        # Get raw array from appropriate tier
        raw = self._get_raw(name)

        # Wrap in TensorInfo
        info = TensorInfo(name=name, array=raw)
        self._tensor_infos[name] = info
        return info

    def _get_raw(self, name: str) -> np.ndarray:
        """Get raw numpy array from tier storage."""
        entry = self._metadata[name]

        # Hot tier — fastest path
        if name in self._hot:
            self._stats["hot_hits"] += 1
            self._hot_lru.move_to_end(name)
            return self._hot[name]

        # Warm tier — decompress
        if name in self._warm:
            self._stats["warm_hits"] += 1
            self._stats["decompressions"] += 1
            data = self._decompress(self._warm[name], entry.shape, entry.dtype)
            self._promote_to_hot(name, data)
            return data

        # Cold tier — read from disk
        self._stats["cold_hits"] += 1
        data = self._read_from_disk(name)
        self._promote_to_hot(name, data)
        return data

    def get_block(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """Get all weights for a transformer block."""
        block_prefix = f"h.{layer_idx}."
        result = {}
        for name in self._metadata:
            if name.startswith(block_prefix):
                result[name] = self.get(name)
        return result

    def prefetch_block(self, layer_idx: int):
        """Prefetch block weights into warm tier (background)."""
        block_prefix = f"h.{layer_idx}."
        for name in self._metadata:
            if name.startswith(block_prefix) and name not in self._hot:
                self._prefetch_queue.append(layer_idx)
                break

    def pin(self, name: str):
        """Pin weight in hot tier (never evict)."""
        if name in self._metadata:
            self._metadata[name].pinned = True
            if name not in self._hot:
                data = self.get(name)
                self._promote_to_hot(name, data)

    def unpin(self, name: str):
        """Unpin weight (allow eviction)."""
        if name in self._metadata:
            self._metadata[name].pinned = False

    def _read_from_disk(self, name: str) -> np.ndarray:
        """Read weight from .slnc file (cold tier)."""
        return self._parser.get_tensor(name)

    def _promote_to_hot(self, name: str, data: np.ndarray):
        """Promote weight to hot tier (evict if needed)."""
        entry = self._metadata[name]

        # Check if we need to evict
        while self._current_hot_bytes() + entry.nbytes > self._hot_budget:
            if not self._evict_from_hot():
                break  # nothing to evict (all pinned)

        self._hot[name] = data
        self._hot_lru[name] = None
        entry.tier = Tier.HOT

    def _evict_from_hot(self) -> bool:
        """Evict least recently used weight from hot tier."""
        for name in list(self._hot_lru.keys()):
            entry = self._metadata[name]
            if entry.pinned:
                continue

            # Compress and move to warm tier
            data = self._hot[name]
            compressed = self._compress(data)

            # Check warm budget
            if self._current_warm_bytes() + len(compressed) > self._warm_budget:
                # Evict from warm too
                self._evict_from_warm()

            self._warm[name] = compressed
            self._warm_lru[name] = None
            entry.tier = Tier.WARM
            entry.compressed_data = compressed
            entry.compressed_nbytes = len(compressed)

            del self._hot[name]
            del self._hot_lru[name]

            self._stats["evictions"] += 1
            logger.debug("Evicted %s from hot → warm (%.1f KB compressed)", name, len(compressed) / 1024)
            return True

        return False

    def _evict_from_warm(self) -> bool:
        """Evict least recently used weight from warm tier (to cold)."""
        for name in list(self._warm_lru.keys()):
            entry = self._metadata[name]
            if entry.pinned:
                continue

            del self._warm[name]
            del self._warm_lru[name]
            entry.tier = Tier.COLD
            entry.compressed_data = None
            entry.compressed_nbytes = 0

            self._stats["evictions"] += 1
            logger.debug("Evicted %s from warm → cold", name)
            return True

        return False

    def _compress(self, data: np.ndarray) -> bytes:
        """Compress numpy array."""
        raw = data.tobytes()

        if self._compression == Compression.NONE:
            return raw
        elif self._compression == Compression.ZSTD:
            import zstandard as zstd
            ctx = zstd.ZstdCompressor()
            return ctx.compress(raw)
        elif self._compression == Compression.QUANTIZE_INT8:
            return self._quantize_int8(data)
        elif self._compression == Compression.QUANTIZE_INT4:
            return self._quantize_int4(data)
        else:
            return raw

    def _decompress(self, data: bytes, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Decompress bytes to numpy array."""
        if self._compression == Compression.NONE:
            return np.frombuffer(data, dtype=dtype).reshape(shape)
        elif self._compression == Compression.ZSTD:
            import zstandard as zstd
            ctx = zstd.ZstdDecompressor()
            raw = ctx.decompress(data)
            return np.frombuffer(raw, dtype=dtype).reshape(shape)
        elif self._compression == Compression.QUANTIZE_INT8:
            return self._dequantize_int8(data, shape, dtype)
        elif self._compression == Compression.QUANTIZE_INT4:
            return self._dequantize_int4(data, shape, dtype)
        else:
            return np.frombuffer(data, dtype=dtype).reshape(shape)

    def _quantize_int8(self, data: np.ndarray) -> bytes:
        """Quantize float32 to int8 (lossy, ~4x compression)."""
        flat = data.flatten().astype(np.float32)
        scale = np.max(np.abs(flat)) / 127.0
        quantized = np.clip(flat / scale, -128, 127).astype(np.int8)
        # Store scale as float32 header
        return scale.tobytes() + quantized.tobytes()

    def _dequantize_int8(self, data: bytes, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Dequantize int8 to float32."""
        scale = np.frombuffer(data[:4], dtype=np.float32)[0]
        quantized = np.frombuffer(data[4:], dtype=np.int8)
        return (quantized.astype(np.float32) * scale).reshape(shape)

    def _quantize_int4(self, data: np.ndarray) -> bytes:
        """Quantize float32 to int4 (lossy, ~8x compression)."""
        flat = data.flatten().astype(np.float32)
        scale = np.max(np.abs(flat)) / 7.0
        quantized = np.clip(flat / scale, -8, 7).astype(np.int8)
        # Pad to even length if needed
        if len(quantized) % 2 != 0:
            quantized = np.append(quantized, 0)
        # Pack two int4 per byte
        low = quantized[0::2].astype(np.uint8) & 0x0F
        high = (quantized[1::2].astype(np.uint8) & 0x0F) << 4
        packed = low | high
        return scale.tobytes() + packed.tobytes()

    def _dequantize_int4(self, data: bytes, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Dequantize int4 to float32."""
        scale = np.frombuffer(data[:4], dtype=np.float32)[0]
        packed = np.frombuffer(data[4:], dtype=np.uint8)
        # Unpack
        low = (packed & 0x0F).astype(np.int8)
        high = ((packed >> 4) & 0x0F).astype(np.int8)
        quantized = np.empty(len(packed) * 2, dtype=np.int8)
        quantized[0::2] = low
        quantized[1::2] = high
        return (quantized.astype(np.float32) * scale).reshape(shape)

    def _current_hot_bytes(self) -> int:
        """Current hot tier memory usage."""
        return sum(e.nbytes for name, e in self._metadata.items() if name in self._hot)

    def _current_warm_bytes(self) -> int:
        """Current warm tier memory usage."""
        return sum(len(self._warm.get(name, b"")) for name in self._metadata)

    def stats(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        hot = TierStats(
            count=len(self._hot),
            uncompressed_bytes=self._current_hot_bytes(),
            compressed_bytes=self._current_hot_bytes(),
        )
        warm = TierStats(
            count=len(self._warm),
            uncompressed_bytes=sum(
                e.nbytes for name, e in self._metadata.items() if name in self._warm
            ),
            compressed_bytes=self._current_warm_bytes(),
        )
        cold = TierStats(
            count=sum(1 for e in self._metadata.values() if e.tier == Tier.COLD),
            uncompressed_bytes=sum(
                e.nbytes for e in self._metadata.values() if e.tier == Tier.COLD
            ),
        )

        total = self._ram_budget
        used = self._current_hot_bytes() + self._current_warm_bytes()

        return {
            "hot": hot,
            "warm": warm,
            "cold": cold,
            "ram_budget_mb": total / (1024 * 1024),
            "ram_used_mb": used / (1024 * 1024),
            "ram_usage": used / total,
            "compression_ratio": warm.ratio,
            **self._stats,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"WeightManager("
            f"hot={s['hot'].count}, warm={s['warm'].count}, cold={s['cold'].count}, "
            f"ram={s['ram_used_mb']:.0f}/{s['ram_budget_mb']:.0f}MB)"
        )
