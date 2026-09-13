"""
ops/blas.py — Apple Accelerate BLAS bindings via ctypes.

C base layer for matrix operations. Falls back to numpy when
Accelerate is unavailable (Linux, non-macOS).
"""

from __future__ import annotations

import ctypes
import numpy as np

_accelerate = None
_unavailable = False


def _load_accelerate():
    """Load libAccelerate.dylib for cblas_sgemm."""
    global _accelerate, _unavailable
    if _unavailable:
        return None
    if _accelerate is not None:
        return _accelerate
    try:
        _accelerate = ctypes.CDLL("libAccelerate.dylib")
        _setup_sgemm(_accelerate)
        return _accelerate
    except OSError:
        _unavailable = True
        return None


def _setup_sgemm(lib):
    """Set cblas_sgemm signature."""
    # CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc
    lib.cblas_sgemm.argtypes = [
        ctypes.c_int,       # Layout (101 = RowMajor)
        ctypes.c_int,       # TransA
        ctypes.c_int,       # TransB
        ctypes.c_int,       # M
        ctypes.c_int,       # N
        ctypes.c_int,       # K
        ctypes.c_float,     # alpha
        ctypes.c_void_p,    # A
        ctypes.c_int,       # lda
        ctypes.c_void_p,    # B
        ctypes.c_int,       # ldb
        ctypes.c_float,     # beta
        ctypes.c_void_p,    # C
        ctypes.c_int,       # ldc
    ]
    lib.cblas_sgemm.restype = None


# CBLAS constants
CBLAS_ROW_MAJOR = 101
CBLAS_NO_TRANS = 111
CBLAS_TRANS = 112


def sgemm(a: np.ndarray, b: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """C = alpha * A @ B + beta * C via Apple Accelerate cblas_sgemm.

    A: (M, K) float32 row-major
    B: (K, N) float32 row-major
    Returns: (M, N) float32
    """
    lib = _load_accelerate()
    if lib is None:
        return np.matmul(a, b)

    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"matmul shape mismatch: ({M},{K}) @ ({K2},{N})"
    assert a.dtype == np.float32 and b.dtype == np.float32

    c = np.empty((M, N), dtype=np.float32)

    lib.cblas_sgemm(
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,
        CBLAS_NO_TRANS,
        M, N, K,
        alpha,
        a.ctypes.data_as(ctypes.c_void_p), K,
        b.ctypes.data_as(ctypes.c_void_p), N,
        beta,
        c.ctypes.data_as(ctypes.c_void_p), N,
    )
    return c


def is_available() -> bool:
    """Check if Accelerate BLAS is available."""
    return _load_accelerate() is not None
