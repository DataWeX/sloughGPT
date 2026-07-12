"""
Per-tensor quantization engine with calibration support.

Provides accurate per-tensor quantization while maintaining the existing
data flow (numpy arrays). Every tensor can be independently quantized
with its own scale/zero_point, and quantization error is tracked per tensor.

Quantization modes:
  - symmetric: scale only, zero_point = 0 (default, simpler)
  - asymmetric: scale + zero_point (handles shifted distributions)
  - clip_percentile: clip outliers before quantizing (improves accuracy)

Design:
  TensorInfo wraps every weight tensor and exposes .as_float() to get the
  dequantized value. The quantized kernel checks info.is_quantized and
  routes accordingly. No changes needed to the model's forward pass.

Usage:
    from domains.infrastructure.quantization import QuantEngine, TensorInfo

    engine = QuantEngine(bits=8, mode="asymmetric", clip_percentile=0.999)
    info = engine.quantize(name="blocks.0.q_proj.weight", arr=weight_array)

    # In the forward pass:
    weight = info.as_float()  # dequantized on the fly
"""

import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def walk_slo_linears(model) -> dict:
    """Find all SloLinear layers in a SloTransformer model.

    SloTransformer stores sub-modules in a plain list (``model.layers``)
    rather than via ``nn.ModuleList`` or ``named_modules()``, so standard
    PyTorch module walking does not work. This function manually walks
    the known structure:

    - ``layers[-1]`` — lm_head (output projection, SloLinear)
    - ``blocks[i].attn.W_q/W_k/W_v/W_o`` — attention projections
    - ``blocks[i].ff.w1/w2/w3`` — feed-forward projections

    Returns:
        dict of ``{name: SloLinear_layer}``.
    """
    from domains.training.slonet import SloLinear

    layers = {}

    # Output projection (lm_head)
    if hasattr(model, "layers") and len(model.layers) >= 1:
        lm_head = model.layers[-1]
        if isinstance(lm_head, SloLinear):
            layers["lm_head"] = lm_head

    # Transformer blocks
    if hasattr(model, "blocks"):
        for i, block in enumerate(model.blocks):
            if hasattr(block, "attn"):
                for proj in ("W_q", "W_k", "W_v", "W_o"):
                    p = getattr(block.attn, proj, None)
                    if isinstance(p, SloLinear):
                        layers[f"blocks.{i}.attn.{proj}"] = p
            if hasattr(block, "ff"):
                for proj in ("w1", "w2", "w3"):
                    p = getattr(block.ff, proj, None)
                    if isinstance(p, SloLinear):
                        layers[f"blocks.{i}.ff.{proj}"] = p

    return layers


def walk_hf_linears(model) -> dict:
    """Find all nn.Linear layers in a HuggingFace model.

    Unlike SloNet models (which store layers in plain Python lists),
    HuggingFace models use standard ``nn.Module`` hierarchy, so
    ``named_modules()`` works. This function extracts all ``nn.Linear``
    layers, which are the targets for int8/int4 quantization.

    Returns:
        dict of ``{name: nn.Linear_layer}``.
    """
    layers = {}
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name == "Linear" and hasattr(module, "weight"):
            layers[name] = module
    return layers


# Try to load AVX2-accelerated int8 GEMM
def _numpy_fallback(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pure-numpy int8 GEMM fallback."""
    return np.matmul(a.astype(np.int32), b.astype(np.int32).T)

def _int4_numpy_fallback(A: np.ndarray, B_packed: np.ndarray, K: int) -> np.ndarray:
    """Pure-numpy int4 GEMM fallback: unpack to int8 then matmul."""
    N = B_packed.shape[0]
    B_unpacked = np.zeros((N, K), dtype=np.int8)
    for j in range(N):
        for k in range(K):
            if k % 2 == 0:
                nib = int(B_packed[j, k // 2]) & 0x0F
            else:
                nib = (int(B_packed[j, k // 2]) >> 4) & 0x0F
            B_unpacked[j, k] = np.int8((nib ^ 8) - 8)
    return np.matmul(A.astype(np.int32), B_unpacked.astype(np.int32).T)

_c_matmul = _numpy_fallback
_c_matmul_int4 = _int4_numpy_fallback
try:
    from domains.infrastructure.quant_core.wrapper import matmul_int8_c, matmul_int4_c, HAS_AVX2
    if HAS_AVX2:
        _c_matmul = matmul_int8_c
        _c_matmul_int4 = matmul_int4_c
        logger.info("Using AVX2 int8 + int4 GEMM (quant_core)")
except Exception:
    pass

logger = logging.getLogger("man.infrastructure.quantization")


class QuantMode(Enum):
    SYMMETRIC = "symmetric"      # scale only, zero_point=0
    ASYMMETRIC = "asymmetric"    # scale + zero_point


class QuantDtype(Enum):
    INT8 = "int8"
    UINT8 = "uint8"
    INT4 = "int4"


@dataclass
class QuantMeta:
    """Quantization metadata for a single tensor."""
    scale: float
    zero_point: int
    bits: int
    mode: str
    dtype_code: int            # numpy dtype code for quantized storage
    original_shape: Tuple[int, ...]
    original_dtype: str        # e.g. "float32"
    # Error metrics (set after calibration)
    mse: float = 0.0           # mean squared error vs original
    max_abs_error: float = 0.0  # worst-case absolute error
    cosine_sim: float = 1.0    # cosine similarity (1.0 = perfect)

    def to_dict(self) -> dict:
        return {
            "scale": self.scale,
            "zero_point": self.zero_point,
            "bits": self.bits,
            "mode": self.mode,
            "dtype_code": self.dtype_code,
            "original_shape": list(self.original_shape),
            "original_dtype": self.original_dtype,
            "mse": self.mse,
            "max_abs_error": self.max_abs_error,
            "cosine_sim": self.cosine_sim,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QuantMeta":
        return cls(
            scale=d["scale"],
            zero_point=d["zero_point"],
            bits=d["bits"],
            mode=d["mode"],
            dtype_code=d["dtype_code"],
            original_shape=tuple(d["original_shape"]),
            original_dtype=d["original_dtype"],
            mse=d.get("mse", 0.0),
            max_abs_error=d.get("max_abs_error", 0.0),
            cosine_sim=d.get("cosine_sim", 1.0),
        )


@dataclass
class TensorInfo:
    """Wraps a weight tensor with optional quantization metadata.

    Provides transparent access: code can always call .as_float() to get
    the dequantized tensor regardless of whether it's quantized or not.
    """
    name: str
    array: np.ndarray          # either float32 original or int8/int4 quantized
    meta: Optional[QuantMeta] = None

    @property
    def is_quantized(self) -> bool:
        return self.meta is not None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.meta.original_shape if self.meta else self.array.shape

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.meta.original_dtype) if self.meta else self.array.dtype

    @property
    def nbytes(self) -> int:
        return self.array.nbytes

    def as_float(self) -> np.ndarray:
        """Return dequantized float32 array.

        If not quantized, returns the original array (possibly cast to float32).
        If quantized, dequantizes using scale and zero_point.
        """
        if not self.is_quantized:
            if self.array.dtype != np.float32:
                return self.array.astype(np.float32)
            return self.array

        signed = (self.meta.mode == "symmetric") if self.is_quantized and self.meta.bits == 4 else True
        return _dequantize(
            self.array,
            scale=self.meta.scale,
            zero_point=self.meta.zero_point,
            bits=self.meta.bits,
            original_shape=self.meta.original_shape,
            signed=signed,
        )

    def quantized_bytes(self) -> int:
        """Size of the quantized representation in bytes."""
        if not self.is_quantized:
            return self.array.nbytes
        return self.array.nbytes

    def compression_ratio(self) -> float:
        """Original size / quantized size."""
        if not self.is_quantized:
            return 1.0
        original = int(np.prod(self.meta.original_shape)) * 4  # float32 = 4 bytes
        if self.meta.bits == 4:
            # Packed int4: each byte stores 2 values, so quantized_bytes reflects packed size
            return original / max(self.quantized_bytes(), 1)
        return original / max(self.quantized_bytes(), 1)


class QuantEngine:
    """Per-tensor quantization engine with calibration support.

    Each tensor gets independently quantized with its own scale/zero_point.
    Supports symmetric and asymmetric modes, optional outlier clipping, and
    error metrics for quality assurance.

    Args:
        bits: 8 or 4 (quantized bit width)
        mode: "symmetric" or "asymmetric"
        clip_percentile: if set (0.0-1.0), clip outliers before computing
            scale. E.g., 0.999 clips the top 0.1% of values.
    """

    # Tensors that should NEVER be quantized (embeddings, final norm)
    SKIP_PREFIXES = ("tok_emb.", "pos_emb.", "norm.")

    # Tensors that are sensitive and should use higher precision
    SENSITIVE_PREFIXES = ("attn_norm.", "ff_norm.")

    def __init__(
        self,
        bits: int = 8,
        mode: str = "symmetric",
        clip_percentile: Optional[float] = None,
        skip_quantize_if_error_above: float = 2.0,
    ):
        if bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {bits}")

        self._bits = bits
        self._mode = QuantMode(mode)
        self._clip_pct = clip_percentile
        self._error_threshold = skip_quantize_if_error_above

        # Dtype for quantized storage
        self._quant_dtype = np.int8 if bits == 8 else np.int8  # int4 packs into int8

        # Per-tensor error report
        self._error_report: Dict[str, QuantMeta] = {}

        logger.info(
            "QuantEngine: bits=%d, mode=%s, clip=%.3f, error_threshold=%.3f",
            bits, mode, clip_percentile or 0.0, skip_quantize_if_error_above,
        )

    def should_skip(self, name: str) -> bool:
        """Check if this tensor should NOT be quantized."""
        for prefix in self.SKIP_PREFIXES:
            if name.startswith(prefix):
                return True
        return False

    def is_sensitive(self, name: str) -> bool:
        """Check if this tensor needs special care."""
        for prefix in self.SENSITIVE_PREFIXES:
            if name.startswith(prefix):
                return True
        return False

    def quantize(self, name: str, arr: np.ndarray) -> TensorInfo:
        """Quantize a single weight tensor.

        Args:
            name: tensor name (e.g. "blocks.0.q_proj.weight")
            arr: float32 weight array

        Returns:
            TensorInfo with quantized array and metadata
        """
        if self.should_skip(name):
            return TensorInfo(name=name, array=arr)

        flat = arr.flatten().astype(np.float32)

        # Apply clipping if configured
        if self._clip_pct is not None:
            lo = np.percentile(flat, (1.0 - self._clip_pct) * 100)
            hi = np.percentile(flat, self._clip_pct * 100)
            flat_clipped = np.clip(flat, lo, hi)
        else:
            flat_clipped = flat
            lo, hi = float(flat.min()), float(flat.max())

        # Compute scale and zero_point
        if self._mode == QuantMode.SYMMETRIC:
            scale, zero_point = self._compute_symmetric(flat_clipped)
        else:
            scale, zero_point = self._compute_asymmetric(flat_clipped)

        # Quantize
        quantized = self._encode(flat_clipped, scale, zero_point)

        # Compute error metrics
        _signed = (self._mode == QuantMode.SYMMETRIC)
        dequantized = _dequantize(quantized, scale, zero_point, self._bits, arr.shape, signed=_signed)
        mse = float(np.mean((flat - dequantized.flatten()) ** 2))
        max_err = float(np.max(np.abs(flat - dequantized.flatten())))
        cos_sim = float(_cosine_similarity(flat, dequantized.flatten()))

        # Decision: skip quantization if error too high for this tensor
        # Use relative threshold: MSE / std^2 must be below threshold
        var = float(np.var(flat))
        relative_mse = mse / max(var, 1e-10)
        if relative_mse > self._error_threshold:
            logger.warning(
                "QuantEngine: skipping %s — rel_mse %.4f > threshold %.4f (var=%.6f)",
                name, relative_mse, self._error_threshold, var,
            )
            return TensorInfo(name=name, array=arr)

        meta = QuantMeta(
            scale=scale,
            zero_point=zero_point,
            bits=self._bits,
            mode=self._mode.value,
            dtype_code=5,  # UINT8 for int8 storage
            original_shape=arr.shape,
            original_dtype=str(arr.dtype),
            mse=mse,
            max_abs_error=max_err,
            cosine_sim=cos_sim,
        )

        # Reshape to original shape for matmul compatibility (int8)
        # For int4 (packed), keep flat since packed size differs from original shape
        quantized_reshaped = quantized if self._bits == 4 else quantized.reshape(arr.shape)

        self._error_report[name] = meta

        return TensorInfo(name=name, array=quantized_reshaped, meta=meta)

    def dequantize_to_float(self, info: TensorInfo) -> np.ndarray:
        """Dequantize a TensorInfo to float32. Convenience method."""
        return info.as_float()

    def quantize_with_scale(
        self, name: str, arr: np.ndarray, scale: float, zero_point: int = 0,
    ) -> TensorInfo:
        """Quantize using a pre-computed scale (skip calibration).

        Useful for re-applying the same quantization from saved metadata
        without re-analyzing the weight distribution.

        Args:
            name: tensor name
            arr: float32 weight array
            scale: pre-computed quantization scale
            zero_point: pre-computed zero point

        Returns:
            TensorInfo with quantized array and metadata (no error metrics)
        """
        if self.should_skip(name):
            return TensorInfo(name=name, array=arr)

        flat = arr.flatten().astype(np.float32)
        quantized = self._encode(flat, scale, zero_point)

        _signed = (self._mode == QuantMode.SYMMETRIC)
        dequantized = _dequantize(quantized, scale, zero_point, self._bits, arr.shape, signed=_signed)
        mse = float(np.mean((flat - dequantized.flatten()) ** 2))
        max_err = float(np.max(np.abs(flat - dequantized.flatten())))
        cos_sim = float(_cosine_similarity(flat, dequantized.flatten()))

        meta = QuantMeta(
            scale=scale,
            zero_point=zero_point,
            bits=self._bits,
            mode=self._mode.value,
            dtype_code=5,
            original_shape=arr.shape,
            original_dtype=str(arr.dtype),
            mse=mse,
            max_abs_error=max_err,
            cosine_sim=cos_sim,
        )
        quantized_reshaped = quantized if self._bits == 4 else quantized.reshape(arr.shape)
        self._error_report[name] = meta
        return TensorInfo(name=name, array=quantized_reshaped, meta=meta)

    def error_report(self) -> Dict[str, Dict[str, Any]]:
        """Get per-tensor quantization error metrics."""
        return {name: meta.to_dict() for name, meta in self._error_report.items()}

    def summary(self) -> Dict[str, Any]:
        """Get aggregate quantization summary."""
        if not self._error_report:
            return {"tensors": 0}

        mses = [m.mse for m in self._error_report.values()]
        cos_sims = [m.cosine_sim for m in self._error_report.values()]
        max_errs = [m.max_abs_error for m in self._error_report.values()]

        return {
            "tensors": len(self._error_report),
            "bits": self._bits,
            "mode": self._mode.value,
            "avg_mse": float(np.mean(mses)),
            "max_mse": float(np.max(mses)),
            "avg_cosine_sim": float(np.mean(cos_sims)),
            "min_cosine_sim": float(np.min(cos_sims)),
            "avg_max_abs_error": float(np.mean(max_errs)),
            "worst_tensor": max(self._error_report.keys(), key=lambda k: self._error_report[k].mse),
        }

    def _compute_symmetric(self, flat: np.ndarray) -> Tuple[float, int]:
        """Symmetric quantization: scale = max_abs / (2^(bits-1) - 1), zero_point = 0."""
        max_abs = max(np.max(np.abs(flat)), 1e-10)
        qmax = (2 ** (self._bits - 1)) - 1  # 127 for int8, 7 for int4
        scale = max_abs / qmax
        return float(scale), 0

    def _compute_asymmetric(self, flat: np.ndarray) -> Tuple[float, int]:
        """Asymmetric quantization: maps [lo, hi] → [qmin, qmax].

        Handles non-zero-centered distributions better than symmetric.
        For 8-bit: maps to signed int8 [-128, 127].
        For 4-bit: maps to unsigned int4 [0, 15].

        The encode formula: q = clip(round(x / scale) + zero_point, qmin, qmax)
        The decode formula: x = (q - zero_point) * scale

        For 8-bit, zero_point = round(-128 - lo / scale) so that
        lo maps to -128 and hi maps to 127.  For 4-bit, zero_point = round(-lo / scale)
        so that lo maps to 0 and hi maps to 15.
        """
        lo = float(np.min(flat))
        hi = float(np.max(flat))

        # Avoid division by zero
        range_val = hi - lo
        if range_val < 1e-10:
            return 1.0, 0

        if self._bits == 8:
            # Signed int8: [-128, 127]
            scale = range_val / 255
            zero_point = int(np.round(-128 - lo / scale))
        else:
            # Unsigned int4: [0, 15]
            scale = range_val / 15
            zero_point = int(np.round(-lo / scale))

        return float(scale), zero_point

    def _encode(self, flat: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
        """Encode float32 array to quantized integer array.

        For int8: returns int8 array of same length.
        For int4: returns packed int8 array with two values per byte (half the length).
        """
        if self._mode == QuantMode.SYMMETRIC:
            if self._bits == 8:
                return np.clip(np.round(flat / scale), -128, 127).astype(np.int8)
            else:
                # int4 symmetric: range [-8, 7]
                q = np.clip(np.round(flat / scale), -8, 7).astype(np.int8)
                return _pack_int4(q)
        else:
            # Asymmetric
            if self._bits == 8:
                quantized = np.clip(np.round(flat / scale) + zero_point, -128, 127).astype(np.int8)
                return quantized
            else:
                # int4 asymmetric: range [0, 15]
                q = np.clip(np.round(flat / scale) + zero_point, 0, 15).astype(np.int8)
                return _pack_int4(q)

    def save_metadata(self, path: str):
        """Save quantization metadata to JSON file."""
        report = {name: meta.to_dict() for name, meta in self._error_report.items()}
        Path(path).write_text(json.dumps(report, indent=2))
        logger.info("QuantEngine: saved metadata for %d tensors to %s", len(report), path)

    def load_metadata(self, path: str):
        """Load quantization metadata from JSON file."""
        raw = json.loads(Path(path).read_text())
        self._error_report = {name: QuantMeta.from_dict(d) for name, d in raw.items()}
        logger.info("QuantEngine: loaded metadata for %d tensors from %s", len(self._error_report), path)


# ══════════════════════════════════════════════════════════════════════════════
# int4 packing / unpacking
# ══════════════════════════════════════════════════════════════════════════════

def _pack_int4(arr: np.ndarray) -> np.ndarray:
    """Pack two int4 values into each int8 byte.

    Args:
        arr: 1D int8 array where each element is in range [-8, 7] or [0, 15]

    Returns:
        Packed int8 array with half the length.
        Each byte: low_nibble = arr[2*i] & 0x0F, high_nibble = arr[2*i+1] & 0x0F
    """
    assert arr.ndim == 1, f"_pack_int4 expects 1D, got {arr.ndim}D"
    # Pad to even length
    if len(arr) % 2 != 0:
        arr = np.append(arr, np.int8(0))
    packed = (arr[1::2].astype(np.uint8) << 4) | (arr[::2].astype(np.uint8) & 0x0F)
    return packed.astype(np.int8)


def _unpack_int4(packed: np.ndarray, original_length: int, signed: bool = True) -> np.ndarray:
    """Unpack int4 values from packed int8 bytes.

    Args:
        packed: int8 array where each byte contains two int4 values
        original_length: number of elements before packing
        signed: if True, interpret as signed int4 (range [-8, 7], symmetric)
                if False, interpret as unsigned int4 (range [0, 15], asymmetric)

    Returns:
        int8 array of original_length
    """
    assert packed.ndim == 1, f"_unpack_int4 expects 1D, got {packed.ndim}D"
    packed_u8 = packed.astype(np.uint8)
    low = (packed_u8 & 0x0F).astype(np.int8)
    high = ((packed_u8 >> 4) & 0x0F).astype(np.int8)
    result = np.empty(len(packed) * 2, dtype=np.int8)
    result[0::2] = low
    result[1::2] = high
    if signed:
        # Sign extension for symmetric int4 (values 8-15 → negative)
        result[result > 7] = result[result > 7] - 16
    return result[:original_length]


def _dequantize(
    quantized: np.ndarray,
    scale: float,
    zero_point: int,
    bits: int,
    original_shape: Tuple[int, ...],
    signed: bool = True,
) -> np.ndarray:
    """Dequantize integer array to float32.

    For symmetric mode (zero_point=0): result = quantized * scale
    For asymmetric mode: result = (quantized - zero_point) * scale

    Handles both int8 (one value per byte) and packed int4 (two values per byte).

    Args:
        quantized: quantized integer array (int8 or packed int4)
        scale: quantization scale
        zero_point: quantization zero point
        bits: 8 or 4
        original_shape: shape to reshape result to
        signed: for int4 unpacking, True for symmetric ([-8,7]), False for asymmetric ([0,15])
    """
    flat = quantized.flatten()

    if bits == 4:
        # Unpack int4 first
        n_original = int(np.prod(original_shape))
        flat = _unpack_int4(flat, n_original, signed=signed)

    if zero_point == 0:
        # Symmetric
        result = flat.astype(np.float32) * scale
    else:
        # Asymmetric: (q - zero_point) * scale
        result = (flat.astype(np.float32) - zero_point) * scale

    return result.reshape(original_shape)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flat arrays.

    Returns 1.0 for identical vectors, 0.0 for orthogonal, -1.0 for opposite.
    Returns 1.0 if both vectors are zero (vacuously similar).
    """
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)

    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)

    if norm_a < 1e-10 and norm_b < 1e-10:
        return 1.0  # both zero — vacuously similar
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0  # one zero, one not

    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def quantize_state_dict(
    state_dict: Dict[str, np.ndarray],
    bits: int = 8,
    mode: str = "symmetric",
    clip_percentile: Optional[float] = None,
) -> Dict[str, TensorInfo]:
    """Quantize an entire state dict.

    Args:
        state_dict: name → float32 array
        bits: 8 or 4
        mode: "symmetric" or "asymmetric"
        clip_percentile: outlier clipping percentile

    Returns:
        Dict mapping name → TensorInfo (quantized or original)
    """
    engine = QuantEngine(bits=bits, mode=mode, clip_percentile=clip_percentile)
    result = {}
    for name, arr in state_dict.items():
        result[name] = engine.quantize(name, arr)

    # Log summary
    summary = engine.summary()
    if summary["tensors"] > 0:
        logger.info(
            "quantize_state_dict: %d tensors, avg_mse=%.6f, avg_cosine=%.4f, worst=%s",
            summary["tensors"],
            summary["avg_mse"],
            summary["avg_cosine_sim"],
            summary["worst_tensor"],
        )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Quantized matmul kernels (int8 GEMM for real speedup)
# ══════════════════════════════════════════════════════════════════════════════

def quantize_activation(
    x: np.ndarray,
    scale: float,
    zero_point: int = 0,
) -> np.ndarray:
    """Quantize a float32 activation to int8.

    Uses the weight's scale for symmetric quantization (weight-only quantization).
    This avoids needing a separate activation scale — the weight scale acts as
    the shared quantization grid.

    Args:
        x: float32 activation array (any shape)
        scale: quantization scale (from the weight's QuantMeta)
        zero_point: quantization zero point (0 for symmetric)

    Returns:
        int8 array with same shape
    """
    if zero_point == 0:
        return np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    else:
        return np.clip(np.round(x / scale) + zero_point, -128, 127).astype(np.int8)


def _ensure_2d_packed(b_packed: np.ndarray, orig_k: int) -> np.ndarray:
    """Reshape packed int4 array from 1D ``(N * K // 2,)`` to 2D ``(N, K // 2)``."""
    if b_packed.ndim == 1:
        n = b_packed.shape[0] * 2 // orig_k
        return b_packed.reshape(n, orig_k // 2)
    return b_packed


def int4_matmul(
    a: np.ndarray,
    b_packed: np.ndarray,
    a_scale: float,
    b_scale: float,
    orig_k: int,
    a_zero_point: int = 0,
    b_zero_point: int = 0,
) -> np.ndarray:
    """INT4 × INT8 matrix multiplication with float32 output.

    ``b_packed`` is a uint8 array storing two signed int4 values per byte
    (low nibble = even index, high nibble = odd index). May be 1D
    ``(N * K // 2,)`` or 2D ``(N, K // 2)``.

    Uses AVX2 C extension if available, otherwise falls back to
    unpack→int8→numpy.

    Args:
        a: int8 matrix (M, K) — activations
        b_packed: uint8 array — packed int4 weights (1D or 2D)
        a_scale: quantization scale for a
        b_scale: quantization scale for b
        orig_k: original (unpacked) dimension K
        a_zero_point: zero point for a (0 for symmetric)
        b_zero_point: zero point for b (0 for symmetric)

    Returns:
        float32 result matrix (M, N)
    """
    b_packed = _ensure_2d_packed(b_packed, orig_k)
    if a_zero_point == 0 and b_zero_point == 0:
        accum = _c_matmul_int4(a, b_packed, orig_k)
        return accum.astype(np.float32) * (a_scale * b_scale)
    else:
        N = b_packed.shape[0]
        b_unpacked = np.zeros((N, orig_k), dtype=np.int8)
        for j in range(N):
            for k in range(orig_k):
                if k % 2 == 0:
                    nib = int(b_packed[j, k // 2]) & 0x0F
                else:
                    nib = (int(b_packed[j, k // 2]) >> 4) & 0x0F
                b_unpacked[j, k] = np.int8((nib ^ 8) - 8)
        return int8_matmul(a, b_unpacked, a_scale, b_scale, a_zero_point, b_zero_point)


def int8_matmul(
    a: np.ndarray,
    b: np.ndarray,
    a_scale: float,
    b_scale: float,
    a_zero_point: int = 0,
    b_zero_point: int = 0,
) -> np.ndarray:
    """INT8 matrix multiplication with float32 output.

    Computes: result = a_dequantized @ b_dequantized.T
    where a is (M, K) and b is (N, K) — weight is stored transposed.

    For symmetric quantization (zero_point=0):
        result_fp32 = (a_int8 @ b_int8.T) * a_scale * b_scale

    Args:
        a: int8 matrix (M, K) — activations
        b: int8 matrix (N, K) — weights (stored transposed, as produced by QuantEngine)
        a_scale: scale for matrix a
        b_scale: scale for matrix b
        a_zero_point: zero point for a (0 for symmetric)
        b_zero_point: zero point for b (0 for symmetric)

    Returns:
        float32 result matrix (M, N)
    """
    # Int32 accumulation (avoids overflow for typical sizes)
    if a_zero_point == 0 and b_zero_point == 0:
        # Pure symmetric: result = a @ b.T
        # Use AVX2 C extension if available, else numpy
        accum = _c_matmul(a, b)
        return accum.astype(np.float32) * (a_scale * b_scale)
    else:
        # Asymmetric: need to account for zero points
        a_sum = a.astype(np.int32).sum(axis=-1, keepdims=True)
        b_sum = b.astype(np.int32).sum(axis=-1, keepdims=True)

        accum = _c_matmul(a, b)
        accum = accum - a_zero_point * b_sum.T - b_zero_point * a_sum + a.shape[-1] * a_zero_point * b_zero_point
        return accum.astype(np.float32) * (a_scale * b_scale)


def quantized_linear(
    x: np.ndarray,
    weight_int8: np.ndarray,
    weight_scale: float,
    weight_zero_point: int = 0,
    bias: Optional[np.ndarray] = None,
    x_scale: Optional[float] = None,
    x_zero_point: int = 0,
) -> np.ndarray:
    """Quantized linear layer: x @ weight.T + bias.

    Performs:
        1. Quantize x to int8 using x_scale (or weight_scale for weight-only)
        2. int8 × int8 → int32 matmul
        3. Dequantize result to float32
        4. Add bias

    Args:
        x: float32 input (..., in_features)
        weight_int8: int8 weight matrix (out_features, in_features) — as stored by QuantEngine
        weight_scale: quantization scale for weight
        weight_zero_point: zero point for weight
        bias: optional float32 bias (out_features,)
        x_scale: quantization scale for x (if None, uses weight_scale)
        x_zero_point: zero point for x (0 for symmetric)

    Returns:
        float32 output (..., out_features)
    """
    orig_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])  # (M, K)

    # Compute activation scale dynamically from x (weight-only quantization)
    if x_scale is not None:
        act_scale = x_scale
    else:
        x_max = np.max(np.abs(x_flat))
        act_scale = x_max / 127.0 if x_max > 0 else 1.0

    # Quantize activation
    x_int8 = quantize_activation(x_flat, act_scale, x_zero_point)

    # INT8 matmul: x_int8 (M, K) @ weight_int8.T (K, N) → (M, N)
    # weight_int8 is (N, K), int8_matmul expects b as (N, K) and does b.T internally
    result = int8_matmul(
        x_int8, weight_int8,
        a_scale=act_scale, b_scale=weight_scale,
        a_zero_point=x_zero_point, b_zero_point=weight_zero_point,
    )

    # Add bias
    if bias is not None:
        result = result + bias

    return result.reshape(orig_shape[:-1] + (weight_int8.shape[0],))


def int4_quantized_linear(
    x: np.ndarray,
    weight_packed: np.ndarray,
    weight_scale: float,
    weight_zero_point: int,
    orig_k: int,
    bias: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Quantized linear layer with packed int4 weights: x @ weight.T + bias.

    ``weight_packed`` is (N, K//2) uint8 — two signed int4 values per byte
    (low nibble = even index, high nibble = odd index).

    Uses AVX2 C kernel if available (inline unpack during dot product),
    otherwise falls back to unpack→int8→numpy.

    Args:
        x: float32 input (..., in_features=K)
        weight_packed: uint8 packed int4 weight matrix (out_features=N, K//2)
        weight_scale: quantization scale for weight
        weight_zero_point: zero point for weight (0 for symmetric)
        orig_k: original (unpacked) dimension K
        bias: optional float32 bias (out_features,)

    Returns:
        float32 output (..., out_features)
    """
    orig_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])  # (M, K)

    # Compute activation scale dynamically
    x_max = np.max(np.abs(x_flat))
    act_scale = x_max / 127.0 if x_max > 0 else 1.0

    # Quantize activation to int8
    x_int8 = quantize_activation(x_flat, act_scale, 0)

    # Int4 GEMM
    result = int4_matmul(
        x_int8, weight_packed,
        a_scale=act_scale, b_scale=weight_scale,
        orig_k=orig_k,
        a_zero_point=0, b_zero_point=weight_zero_point,
    )

    if bias is not None:
        result = result + bias

    # Determine number of output features from packed array
    n = weight_packed.shape[0] if weight_packed.ndim == 2 else weight_packed.shape[0] * 2 // orig_k
    return result.reshape(orig_shape[:-1] + (n,))
