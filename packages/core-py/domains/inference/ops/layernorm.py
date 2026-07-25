"""
ops/layernorm.py — Layer normalization. C base (future), numpy fallback.
"""

import numpy as np


def layernorm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
              eps: float = 1e-5) -> np.ndarray:
    """Layer normalization: y = (x - mean) / sqrt(var + eps) * weight + bias.

    Args:
        x: (..., hidden) input
        weight: (hidden,) scale
        bias: (hidden,) shift
        eps: epsilon for numerical stability

    Returns:
        Normalized array, same shape as x.
    """
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias
