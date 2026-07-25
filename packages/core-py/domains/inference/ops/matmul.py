"""
ops/matmul.py — Matrix multiply with C base (Accelerate) and numpy fallback.

Both SloTransformer and NativeEngine call ops.matmul() instead of
np.matmul() directly. C is tried first, numpy is the fallback.
"""

import numpy as np
from . import blas


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply: C = A @ B.

    C base: Apple Accelerate cblas_sgemm (float32 only).
    Fallback: numpy matmul (any dtype).

    Args:
        a: (M, K) array
        b: (K, N) array

    Returns:
        (M, N) array, same dtype as input (promoted to float32 for C path).
    """
    # C path requires float32 — promote if needed, fall back if not worth it
    if a.dtype == np.float32 and b.dtype == np.float32 and blas.is_available():
        return blas.sgemm(a, b)
    return np.matmul(a, b)
