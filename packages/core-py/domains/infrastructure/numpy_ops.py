"""
Generic NumPy ops for transformer inference.

Functions:
  - softmax, rmsnorm, layer_norm, gelu, silu, rope
  - to_float32 (bfloat16/float16 handling)
"""

import numpy as np


def to_float32(arr: np.ndarray) -> np.ndarray:
    """Convert bfloat16/float16 to float32."""
    if arr.dtype.name in ("bfloat16", "float16"):
        if arr.dtype.name == "bfloat16":
            raw = arr.view(np.uint16).astype(np.uint32) << 16
            return raw.view(np.float32)
        return arr.astype(np.float32)
    return arr.astype(np.float32) if arr.dtype != np.float32 else arr


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax along axis."""
    x_max = x.max(axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=axis, keepdims=True)


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMS normalization."""
    eps = np.dtype(x.dtype).type(eps)
    return (x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)) * w


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization."""
    eps = np.dtype(x.dtype).type(eps)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps) * w
    if b is not None:
        out = out + b.astype(x.dtype)
    return out


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation."""
    T = np.dtype(x.dtype).type
    return T(0.5) * x * (T(1.0) + np.tanh(T(np.sqrt(2.0 / np.pi)) * (x + T(0.044715) * x ** 3)))


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation."""
    T = np.dtype(x.dtype).type
    return x * (T(1.0) / (T(1.0) + np.exp(-x)))


def rope(x: np.ndarray, pos: int, dim: int, base: float = 10000.0) -> np.ndarray:
    """Rotary position embeddings. x: (seq, heads, head_dim)."""
    seq_len = x.shape[0]
    t = np.arange(pos, pos + seq_len, dtype=np.float32)
    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    emb = np.outer(t, freqs)
    cos = np.cos(emb)
    sin = np.sin(emb)
    if x.ndim == 3:
        cos = cos[:, np.newaxis, :]
        sin = sin[:, np.newaxis, :]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
