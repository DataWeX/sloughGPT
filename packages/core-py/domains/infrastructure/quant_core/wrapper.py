"""
ctypes wrapper for the AVX2-accelerated int8 GEMM library.

Functions
---------
matmul_int8_c(A, B) → ndarray
    C[i,j] = Σₖ A[i,k] · B[j,k]  using AVX2 if available, else numpy.

_build() → bool
    Try to compile the C library. Returns True on success.
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys

import numpy as np

logger = logging.getLogger("man.quant_core")

# ── Paths ──────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_C_SRC = os.path.join(_HERE, "matmul_int8.c")
_DYLIB = os.path.join(_HERE, "matmul_int8.dylib")


# ── Build ──────────────────────────────────────────────────────────

def _build() -> bool:
    """Compile the C extension with gcc/clang.

    Requires AVX2 support (Haswell or newer Intel, or AMD Zen).
    If compilation fails, logs a warning and returns False.

    Returns:
        True if the library was compiled successfully.
    """
    try:
        result = subprocess.run(
            ["gcc", "-O3", "-mavx2", "-shared", "-fPIC",
             "-o", _DYLIB, _C_SRC],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            logger.warning(
                "quant_core: compilation failed (stderr):\n%s",
                result.stderr,
            )
            return False
        logger.info("quant_core: compiled %s", _DYLIB)
        return True
    except FileNotFoundError:
        logger.warning("quant_core: gcc/clang not found — using numpy fallback")
        return False
    except Exception as exc:
        logger.warning("quant_core: build error (%s) — using numpy fallback", exc)
        return False


# ── Native library loading ─────────────────────────────────────────

_LIB = None


def _load_lib():
    """Load the shared library, building it first if needed."""
    global _LIB
    if _LIB is not None:
        return True

    if not os.path.exists(_DYLIB):
        if not _build():
            return False

    try:
        _LIB = ctypes.CDLL(_DYLIB)
        _LIB.matmul_int8.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # B
            ctypes.c_void_p,  # C (output)
            ctypes.c_int,     # M
            ctypes.c_int,     # N
            ctypes.c_int,     # K
        ]
        _LIB.matmul_int8.restype = None
        return True
    except Exception as exc:
        logger.warning("quant_core: load failed (%s)", exc)
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


def _fallback(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pure-numpy fallback: int32 matmul + transpose."""
    return np.matmul(A.astype(np.int32), B.astype(np.int32).T)


# ── Check at import time ──────────────────────────────────────────

HAS_AVX2 = _load_lib()
if HAS_AVX2:
    logger.info("quant_core: AVX2 int8 GEMM loaded")
else:
    logger.info("quant_core: using numpy fallback for int8 GEMM")
