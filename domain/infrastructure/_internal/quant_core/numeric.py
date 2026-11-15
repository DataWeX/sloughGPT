"""Scalar quantization numerics — the canonical home for quant/dequant arithmetic.

Quantine (the serving quantization engine), pugqeep's block codecs, the .slnc
compiler and embedding quantization all express the same affine-grid formula
family. Each *convention* lives here exactly once, with the op order of its
originating site preserved bit-for-bit — the before/after dump
(``scripts``-external quant-audit) proves every re-pointed call site produces
byte-identical output.

Encode conventions (all return the clipped float array; callers cast):

- ``encode_round_div``      ``clip(round(x / scale), qmin, qmax)``
- ``encode_round_div_zp``   ``clip(round(x / scale) + zero_point, qmin, qmax)``
- ``encode_offset``         ``clip(round((x - lo) / scale), qmin, qmax)``
- ``encode_round_add_zp``   ``clip(round(x / scale + zero_point), qmin, qmax)``
  (float zero-point added *inside* the round — VQ centroid grids)
- ``encode_scale_mul``      ``clip(round(x / scale * mul) + zero_point, qmin, qmax)``
  (embedding grids: scale is per-row absmax, ``mul`` = 127)

Dequant conventions (float32 result, broadcast-safe):

- ``dequant_offset``        ``q * scale + lo``          (block codecs)
- ``dequant_zp``            ``(q - zero_point) * scale`` (Quantine / KV / VQ;
  ``zero_point=0`` is exact: subtracting 0 never changes float bits)

Scale conventions:

- ``symmetric_scale``       ``max(absmax, 1e-10) / qmax``
- ``symmetric_scale_rows``  per-output-row variant, float32 ``(N, 1)``
- ``block_scale``           ``max(range, 1e-10) / divisor`` (per-block codecs)

Nibble packing (low nibble = even index, high nibble = odd index):

- ``pack_int4`` / ``unpack_int4``  — 1-D, signed or unsigned
- ``pack_int4_rows``               — 2-D row-wise variant (per-channel int4)

Pure numpy, imports nothing from this package: importing it must never trigger
the native-kernel build (``quant_core/__init__`` keeps kernels lazy for exactly
this reason — pugqeep and the .slnc compiler consume this module from contexts
where no toolchain should be required).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "block_scale",
    "dequant_offset",
    "dequant_zp",
    "encode_offset",
    "encode_round_add_zp",
    "encode_round_div",
    "encode_round_div_zp",
    "encode_scale_mul",
    "pack_int4",
    "pack_int4_rows",
    "symmetric_scale",
    "symmetric_scale_rows",
    "unpack_int4",
]


# ── scales ──────────────────────────────────────────────────────────────────


def symmetric_scale(absmax: float, qmax: int):
    """Global symmetric grid scale: ``max(absmax, 1e-10) / qmax``.

    The 1e-10 floor keeps an all-zero tensor from dividing by zero; qmax is
    127 for int8 and 7 for int4.
    """
    return max(absmax, 1e-10) / qmax


def symmetric_scale_rows(arr: np.ndarray, qmax: int) -> np.ndarray:
    """Per-output-row symmetric scale, float32 ``(N, 1)``.

    Row max comes from the TRUE row maximum (no percentile clipping — see
    ``Quantine._quantize_per_channel``: clipping then overflowing the grid
    destroys outlier channels).
    """
    max_abs = np.maximum(np.max(np.abs(arr), axis=1, keepdims=True), 1e-10)
    return (max_abs / qmax).astype(np.float32)


def block_scale(block_range: np.ndarray, divisor: float):
    """Per-block affine scale: ``max(range, 1e-10) / divisor``.

    divisor is 15 for 4-bit blocks and 255 for 8-bit blocks; the floor keeps
    constant-valued blocks from dividing by zero.
    """
    return np.maximum(block_range, 1e-10) / divisor


# ── encode ──────────────────────────────────────────────────────────────────


def encode_round_div(x: np.ndarray, scale, qmin: int, qmax: int) -> np.ndarray:
    """``clip(round(x / scale), qmin, qmax)`` — symmetric grids (zp = 0)."""
    return np.clip(np.round(x / scale), qmin, qmax)


def encode_round_div_zp(x: np.ndarray, scale, zero_point, qmin: int, qmax: int) -> np.ndarray:
    """``clip(round(x / scale) + zero_point, qmin, qmax)`` — integer zero-point
    added *outside* the round (Quantine asymmetric grids, activations)."""
    return np.clip(np.round(x / scale) + zero_point, qmin, qmax)


def encode_offset(x: np.ndarray, lo: np.ndarray, scale, qmin: int, qmax: int) -> np.ndarray:
    """``clip(round((x - lo) / scale), qmin, qmax)`` — offset-form grids
    (pugqeep block codecs: lo is the per-block minimum)."""
    return np.clip(np.round((x - lo) / scale), qmin, qmax)


def encode_round_add_zp(x: np.ndarray, scale, zero_point, qmin: int, qmax: int) -> np.ndarray:
    """``clip(round(x / scale + zero_point), qmin, qmax)`` — float zero-point
    added *inside* the round (VQ centroid grids; not interchangeable with
    ``encode_round_div_zp`` once zero_point is fractional)."""
    return np.clip(np.round(x / scale + zero_point), qmin, qmax)


def encode_scale_mul(
    x: np.ndarray,
    scale,
    mul: float,
    qmin: int,
    qmax: int,
    zero_point: int = 0,
) -> np.ndarray:
    """``clip(round(x / scale * mul) + zero_point, qmin, qmax)`` — grids whose
    scale is a plain absmax and whose range is folded in at round time
    (``OptimizedEmbedding``: ``mul=127``, uint8 adds 128)."""
    return np.clip(np.round(x / scale * mul) + zero_point, qmin, qmax)


# ── dequant ─────────────────────────────────────────────────────────────────


def dequant_offset(q: np.ndarray, scale, lo) -> np.ndarray:
    """``q * scale + lo`` in float32 — offset-form decode (``lo`` already
    broadcast by the caller, matching the original per-site shapes)."""
    return q.astype(np.float32) * scale + lo


def dequant_zp(q: np.ndarray, scale, zero_point) -> np.ndarray:
    """``(q - zero_point) * scale`` in float32 — affine decode.

    ``zero_point=0`` is bit-exact against a bare ``q * scale`` (subtracting
    zero never changes float bits), so symmetric and asymmetric paths share
    this one function.
    """
    return (q.astype(np.float32) - zero_point) * scale


# ── int4 nibble packing ─────────────────────────────────────────────────────


def pack_int4(arr: np.ndarray) -> np.ndarray:
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


def pack_int4_rows(arr: np.ndarray) -> np.ndarray:
    """Row-wise int4 pack: each row padded to even width, then packed with the
    same low-nibble-even / high-nibble-odd layout as :func:`pack_int4`.

    This is the layout ``int4_matmul`` and the C kernel expect, so per-channel
    int4 stays fused-path compatible.

    Args:
        arr: 2D int8 array, values in [-8, 7]

    Returns:
        Packed int8 array of shape ``(N, K // 2)`` (rows padded with a 0 nibble
        when K is odd).
    """
    assert arr.ndim == 2, f"pack_int4_rows expects 2D, got {arr.ndim}D"
    if arr.shape[1] % 2 != 0:
        arr = np.pad(arr, ((0, 0), (0, 1)))
    packed = (arr[:, 1::2].astype(np.uint8) << 4) | (arr[:, ::2].astype(np.uint8) & 0x0F)
    return packed.astype(np.int8)


def unpack_int4(packed: np.ndarray, original_length: int, signed: bool = True) -> np.ndarray:
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
