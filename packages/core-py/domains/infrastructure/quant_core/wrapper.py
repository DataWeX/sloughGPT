"""
ctypes wrapper for the AVX2-accelerated int8 and int4 GEMM libraries.

Functions
---------
matmul_int8_c(A, B) → ndarray
    C[i,j] = Σₖ A[i,k] · B[j,k]  using AVX2 if available, else numpy.

matmul_int4_c(A, B_packed) → ndarray
    C[i,j] = Σₖ A[i,k] · unpack_int4(B_packed[j,k])  using AVX2 if available,
    else numpy (via unpack→int8→matmul).

_build_all() → bool
    Try to compile all C libraries. Returns True if at least one succeeded.
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys
from typing import Optional, Union

import numpy as np

logger = logging.getLogger("slo.quant_core")

# ── Paths ──────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))

# Platform-specific shared library extension
if sys.platform == "darwin":
    _EXT = ".dylib"
elif sys.platform == "win32":
    _EXT = ".dll"
else:
    _EXT = ".so"

_SRCS = {
    "matmul_int8": os.path.join(_HERE, "matmul_int8.c"),
    "matmul_int4": os.path.join(_HERE, "matmul_int4.c"),
}
_DYLIBS = {
    name: os.path.join(_HERE, f"{name}{_EXT}")
    for name in _SRCS
}


# ── Build ──────────────────────────────────────────────────────────

def _build_one(name: str) -> bool:
    """Compile a single C extension with gcc/clang.

    Requires AVX2 support (Haswell or newer Intel, or AMD Zen).
    If compilation fails, logs a warning and returns False.

    Returns:
        True if the library was compiled successfully.
    """
    src = _SRCS[name]
    dylib = _DYLIBS[name]
    try:
        if not os.path.exists(src):
            logger.warning("quant_core: source not found: %s", src,
                extra={"tag": "INFRA"})
            return False
        result = subprocess.run(
            ["gcc", "-O3", "-mavx2", "-shared", "-fPIC", "-pthread",
             "-o", dylib, src],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            logger.warning(
                "quant_core: %s compilation failed (stderr):\n%s",
                name, result.stderr,
                extra={"tag": "INFRA"},
            )
            return False
        logger.info("quant_core: compiled %s", dylib,
            extra={"tag": "INFRA"})
        return True
    except FileNotFoundError:
        logger.warning("quant_core: gcc/clang not found — using numpy fallback",
            extra={"tag": "INFRA"})
        return False
    except Exception as exc:
        logger.warning("quant_core: %s build error (%s) — using numpy fallback", name, exc,
            extra={"tag": "INFRA"})
        return False


def _build_all(force: bool = False) -> bool:
    """Compile all C extensions. Returns True if at least one succeeded.

    A library is (re)built when its shared object is missing or older than the
    C source (stale libs silently mask kernel edits). Pass ``force=True`` to
    always recompile.

    Returns:
        True if at least one library is available after building.
    """
    ok = False
    for name in _SRCS:
        src = _SRCS[name]
        dylib = _DYLIBS[name]
        if force or not os.path.exists(dylib):
            ok = _build_one(name) or ok
            continue
        # Rebuild if the source is newer than the compiled library.
        try:
            if os.path.exists(src) and os.path.getmtime(src) > os.path.getmtime(dylib):
                ok = _build_one(name) or ok
                continue
        except OSError:
            pass
        ok = True
    return ok


# ── Native library loading ─────────────────────────────────────────

_LIB = None


def _load_lib():
    """Load the shared libraries, building them first if needed."""
    global _LIB
    if _LIB is not None:
        return True

    # Build both if needed
    _build_all()

    # Load int8 library
    try:
        _LIB = ctypes.CDLL(_DYLIBS["matmul_int8"])
        _LIB.matmul_int8.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # B
            ctypes.c_void_p,  # C (output)
            ctypes.c_int,     # M
            ctypes.c_int,     # N
            ctypes.c_int,     # K
        ]
        _LIB.matmul_int8.restype = None
        _LIB.matmul_int8_f32.argtypes = [
            ctypes.c_void_p,  # A (float32)
            ctypes.c_void_p,  # B (int8)
            ctypes.c_void_p,  # B_scale (float)
            ctypes.c_void_p,  # bias (float, nullable)
            ctypes.c_void_p,  # C (float32 output)
            ctypes.c_int,     # M
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_int,     # b_scale_per_row
        ]
        _LIB.matmul_int8_f32.restype = None
        has_int8 = True
    except Exception as exc:
        logger.warning("quant_core: int8 lib load failed (%s)", exc,
            extra={"tag": "INFRA"})
        has_int8 = False

    # Load int4 library (separate dylib)
    has_int4 = False
    try:
        _lib4 = ctypes.CDLL(_DYLIBS["matmul_int4"])
        _lib4.matmul_int4.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # B_packed
            ctypes.c_void_p,  # C (output)
            ctypes.c_int,     # M
            ctypes.c_int,     # N
            ctypes.c_int,     # K
        ]
        _lib4.matmul_int4.restype = None
        # Attach to _LIB so hasattr and call syntax work
        _LIB.matmul_int4 = _lib4.matmul_int4
        has_int4 = True
    except Exception as exc:
        logger.info("quant_core: matmul_int4 not available (%s)", exc,
            extra={"tag": "INFRA"})

    if has_int4 or has_int8:
        return True
    logger.warning("quant_core: no library loaded",
        extra={"tag": "INFRA"})
    return False


# ── Public API ─────────────────────────────────────────────────────


def matmul_int8_c(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """int8 GEMM using AVX2-accelerated C library.

    Args:
        A: int8 array, shape (M, K)
        B: int8 array, shape (N, K)  — **not** transposed

    Returns:
        int32 array, shape (M, N)

    Falls back to numpy if the C library is unavailable.
    """
    if not _load_lib():
        return _fallback(A, B)

    M, K = A.shape
    N = B.shape[0]
    assert B.shape[1] == K, f"B.shape[1]={B.shape[1]} != K={K}"

    C = np.zeros((M, N), dtype=np.int32)
    _LIB.matmul_int8(
        A.ctypes.data,
        B.ctypes.data,
        C.ctypes.data,
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
    )
    return C


def matmul_int8_f32_c(
    A: np.ndarray,
    B: np.ndarray,
    B_scale: Union[float, np.ndarray],
    bias: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Fused float32 activation → quantized int8 → GEMM → float32 output.

    Performs the whole per-token symmetric W8A8 linear in one C call:
    quantize each row of ``A`` (scale = max|row|/127), int8 GEMM against
    ``B``, then dequantize with per-row ``B_scale`` and add ``bias``.

    Args:
        A: float32 activations, shape (M, K)
        B: int8 weights, shape (N, K)
        B_scale: weight scale — float (per-tensor) or ``(N,)`` float32 per-row
        bias: optional float32 bias, shape (N,)

    Returns:
        float32 result, shape (M, N), or None when the C library is
        unavailable so the caller can fall back to the unfused path.
    """
    if not _load_lib() or not hasattr(_LIB, "matmul_int8_f32"):
        return None

    M, K = A.shape
    N = B.shape[0]
    assert B.shape[1] == K, f"B.shape[1]={B.shape[1]} != K={K}"

    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.int8)
    if np.isscalar(B_scale):
        b_scale_arr = np.asarray([float(B_scale)], dtype=np.float32)
        b_scale_per_row = 0
    else:
        b_scale_arr = np.asarray(B_scale, dtype=np.float32).ravel().copy()
        assert b_scale_arr.shape[0] == N, \
            f"B_scale has {b_scale_arr.shape[0]} rows, expected {N}"
        b_scale_per_row = 1
    bias_ptr = None
    if bias is not None:
        bias = np.ascontiguousarray(bias, dtype=np.float32)
        assert bias.shape[0] == N
        bias_ptr = bias.ctypes.data

    C = np.empty((M, N), dtype=np.float32)
    _LIB.matmul_int8_f32(
        A.ctypes.data,
        B.ctypes.data,
        b_scale_arr.ctypes.data,
        bias_ptr,
        C.ctypes.data,
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
        ctypes.c_int(b_scale_per_row),
    )
    return C


def matmul_int4_c(A: np.ndarray, B_packed: np.ndarray, K: int) -> np.ndarray:
    """int4×int8 GEMM using AVX2-accelerated C library (packed int4).

    Args:
        A: int8 array, shape (M, K)
        B_packed: uint8 array, shape (N, K//2) — packed int4, two values per byte
        K: original (unpacked) dimension

    Returns:
        int32 array, shape (M, N)

    Falls back to numpy (unpack→int8→matmul) if the C library is unavailable.
    """
    if not _load_lib() or not hasattr(_LIB, "matmul_int4"):
        return _fallback_int4(A, B_packed, K)

    M = A.shape[0]
    N = B_packed.shape[0]
    assert A.shape[1] == K, f"A.shape[1]={A.shape[1]} != K={K}"
    assert B_packed.shape[1] == K // 2, f"B_packed.shape[1]={B_packed.shape[1]} != K/2={K//2}"

    C = np.zeros((M, N), dtype=np.int32)
    _LIB.matmul_int4(
        A.ctypes.data,
        B_packed.ctypes.data,
        C.ctypes.data,
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
    )
    return C


def _fallback(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pure-numpy fallback for int8 matmul."""
    return np.matmul(A.astype(np.int32), B.astype(np.int32).T)


def _fallback_int4(A: np.ndarray, B_packed: np.ndarray, K: int) -> np.ndarray:
    """Vectorized pure-numpy fallback for int4 matmul."""
    from ..quantization import _unpack_int4 as _vectorized_unpack_int4

    N = B_packed.shape[0]
    # Vectorized unpack: all rows at once using the shared utility
    n_total = N * K
    flat = _vectorized_unpack_int4(B_packed.ravel(), n_total, signed=True)
    B_unpacked = flat.reshape(N, K).astype(np.int8)
    return _fallback(A, B_unpacked)


# ── Check at import time ──────────────────────────────────────────

HAS_AVX2 = _load_lib()
if HAS_AVX2:
    if hasattr(_LIB, "matmul_int4"):
        logger.info("quant_core: AVX2 int8 + int4 GEMM loaded",
            extra={"tag": "INFRA"})
    else:
        logger.info("quant_core: AVX2 int8 GEMM loaded",
            extra={"tag": "INFRA"})
else:
    logger.info("quant_core: using numpy fallback for all quantized GEMM",
        extra={"tag": "INFRA"})
