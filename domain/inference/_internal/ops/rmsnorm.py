"""
ops/rmsnorm.py — RMS normalization. C base (future), numpy fallback.
"""

from __future__ import annotations

import numpy as np


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMS normalization: y = x / sqrt(mean(x^2) + eps) * weight.

    Args:
        x: (..., hidden) input
        weight: (hidden,) scale
        eps: epsilon for numerical stability

    Returns:
        Normalized array, same shape as x.
    """
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms * weight
