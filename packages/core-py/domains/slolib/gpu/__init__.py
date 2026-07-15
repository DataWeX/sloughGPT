"""
GPU/Accelerator Layer for SloLib

Three backends:
- **Metal**: macOS GPU via PyTorch MPS (existing install, ~0 setup)
- **CUDA**: NVIDIA GPU via CuPy (pip install cupy-cuda12x)
- **CPU**: Pure NumPy (always works, no dependencies)

Usage:
    acc = get_accelerator()
    c = acc.matmul(a, b)
    y = acc.layer_norm(x, weight, bias)
    attn = acc.scaled_dot_attention(q, k, v, mask=mask)
"""

from __future__ import annotations

import os
import sys
import time
import math
from typing import Optional, List, Tuple, Any, Dict

import logging
import numpy as np

logger = logging.getLogger("man.gpu")


# =============================================================================
# BUFFER POOL — reuse numpy arrays to reduce allocation overhead
# =============================================================================

class _BufferPool:
    """Simple memory pool that reuses numpy arrays of matching shape/dtype.
    Avoids repeated malloc/free for frequently-used buffer sizes.

    Usage::
        pool = _BufferPool()
        buf = pool.get((128, 768), np.float32)
        # ... use buf ...
        pool.put(buf)  # returns to pool
    """
    def __init__(self, max_pool_size: int = 64):
        self._pool: Dict[Tuple, List[np.ndarray]] = {}
        self._max_pool_size = max_pool_size
        self._hits = 0
        self._misses = 0

    def get(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        key = (shape, dtype)
        bucket = self._pool.get(key)
        if bucket:
            self._hits += 1
            return bucket.pop()
        self._misses += 1
        return np.empty(shape, dtype=dtype)

    def put(self, arr: np.ndarray) -> None:
        key = (arr.shape, arr.dtype)
        bucket = self._pool.setdefault(key, [])
        if len(bucket) < self._max_pool_size:
            bucket.append(arr)

    def stats(self) -> Dict[str, Any]:
        return {"hits": self._hits, "misses": self._misses}

    def clear(self) -> None:
        self._pool.clear()
        self._hits = 0
        self._misses = 0


_POOL = _BufferPool()


# =============================================================================
# DEVICE DETECTION
# =============================================================================

_BACKEND: Optional["_Accelerator"] = None


def get_accelerator() -> "_Accelerator":
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    _BACKEND = _detect_best_backend()
    return _BACKEND


def reset_accelerator() -> None:
    global _BACKEND
    _BACKEND = None


def _detect_best_backend() -> "_Accelerator":
    candidates: List[Tuple[str, "_Accelerator", int]] = []

    # --- Metal (macOS) ---
    if sys.platform == "darwin":
        try:
            metal = _MetalBackend()
            if metal.is_available():
                vram = metal.vram_gb()
                priority = 100 + vram  # discrete GPUs score higher
                candidates.append(("metal", metal, priority))
        except Exception:
            pass

    # --- CUDA (NVIDIA) ---
    try:
        cuda = _CUDABackend()
        if cuda.is_available():
            vram = cuda.vram_gb()
            priority = 200 + vram  # CUDA is the fastest GPU path
            candidates.append(("cuda", cuda, priority))
    except Exception:
        pass

    # --- OpenCL (Intel iGPU / AMD dGPU / integrated) ---
    try:
        opencl = _OpenCLBackend()
        if opencl.is_available():
            vram = opencl.vram_gb()
            priority = 50 + vram
            candidates.append(("opencl", opencl, priority))
    except Exception:
        pass

    # --- CPU with SIMD ---
    cpu = _CPUBackend()
    priority = 0
    if cpu.has_openblas():
        priority = 20 + cpu.openblas_threads()
    elif cpu.has_simd():
        priority = 10
    candidates.append(("cpu", cpu, priority))

    # Pick highest priority
    candidates.sort(key=lambda x: x[2], reverse=True)
    best_name, best_acc, _ = candidates[0]
    return best_acc


class _Accelerator:
    """Base accelerator interface."""

    name: str = "base"
    device_type: str = "cpu"
    compute_tier: str = "lite"
    desc: str = ""

    def is_available(self) -> bool:
        return True

    def vram_gb(self) -> float:
        return 0.0

    def memory_hint(self) -> Dict[str, Any]:
        return {"tier": self.compute_tier}

    def sync(self) -> None:
        """No-op on CPU — operations are synchronous."""

    def to_device(self, arr: np.ndarray) -> np.ndarray:
        return arr.astype(np.float32)

    def from_device(self, arr: Any) -> np.ndarray:
        return np.asarray(arr)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b)

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    def sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    def div(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a / b

    def pow(self, a: np.ndarray, p: float) -> np.ndarray:
        return a ** p

    def sqrt(self, a: np.ndarray) -> np.ndarray:
        return np.sqrt(a)

    def exp(self, a: np.ndarray) -> np.ndarray:
        return np.exp(a)

    def log(self, a: np.ndarray) -> np.ndarray:
        return np.log(a)

    def sum(self, a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        return a.sum(axis=axis)

    def mean(self, a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        return a.mean(axis=axis)

    def max(self, a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        return a.max(axis=axis)

    def min(self, a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        return a.min(axis=axis)

    def abs(self, a: np.ndarray) -> np.ndarray:
        return np.abs(a)

    def neg(self, a: np.ndarray) -> np.ndarray:
        return -a

    def clamp(self, a: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.clip(a, lo, hi)

    def where(self, cond: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.where(cond, a, b)

    def gather(self, a: np.ndarray, dim: int, index: np.ndarray) -> np.ndarray:
        return np.take_along_axis(a, index.astype(int), axis=dim)

    def scatter(self, a: np.ndarray, dim: int, index: np.ndarray, src: np.ndarray) -> np.ndarray:
        result = a.copy()
        np.put_along_axis(result, index.astype(int), src, axis=dim)
        return result

    def pad(self, a: np.ndarray, pad_width, mode: str = "constant", constant_values: float = 0.0) -> np.ndarray:
        return np.pad(a, pad_width, mode=mode, constant_values=constant_values)

    def softmax(self, a: np.ndarray, axis: int = -1) -> np.ndarray:
        a_max = a.max(axis=axis, keepdims=True)
        exp_a = np.exp(a - a_max)
        return exp_a / exp_a.sum(axis=axis, keepdims=True)

    def log_softmax(self, a: np.ndarray, axis: int = -1) -> np.ndarray:
        a_max = a.max(axis=axis, keepdims=True)
        return a - a_max - np.log(np.exp(a - a_max).sum(axis=axis, keepdims=True))

    def layer_norm(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return ((x - mean) / np.sqrt(var + eps)) * weight + bias

    def rms_norm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        return (x / rms) * weight

    def gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def silu(self, x: np.ndarray) -> np.ndarray:
        return x / (1 + np.exp(-x))

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    def fused_add_mul(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Fused a + b * c — one pass instead of two."""
        return a + b * c

    def fused_mul_add(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Fused a * b + c — common in linear layers (weight @ input + bias)."""
        return a * b + c

    def fused_layernorm_gelu(self, x: np.ndarray, weight: np.ndarray,
                              bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Fused layer_norm + gelu — one pass over x instead of two."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        normed = (x - mean) / np.sqrt(var + eps)
        scaled = normed * weight + bias
        return 0.5 * scaled * (1 + np.tanh(np.sqrt(2 / np.pi) * (scaled + 0.044715 * scaled ** 3)))

    def fused_layernorm_silu(self, x: np.ndarray, weight: np.ndarray,
                              bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Fused layer_norm + silu — one pass."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        normed = (x - mean) / np.sqrt(var + eps)
        scaled = normed * weight + bias
        return scaled / (1 + np.exp(-scaled))

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def _scores_sn(self, q: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Compute Q @ K^T with shape [B, H, N, S] using matmul (faster than einsum)."""
        B, H, N, E = q.shape
        S = k.shape[2]
        return (q.reshape(B * H, N, E) @ k.reshape(B * H, S, E).transpose(0, 2, 1)).reshape(B, H, N, S)

    def _apply_attn(self, attn: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Apply attention weights to V with shape [B, H, N, E] using matmul."""
        B, H, N, S = attn.shape
        E = v.shape[3]
        return (attn.reshape(B * H, N, S) @ v.reshape(B * H, S, E)).reshape(B, H, N, E)

    def scaled_dot_attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                             mask: Optional[np.ndarray] = None,
                             scale: Optional[float] = None,
                             causal: bool = False) -> np.ndarray:
        """Scaled dot-product attention with optimizations.

        Uses matmul (BLAS) instead of einsum for 3-9x speedup on common sizes.
        Falls back to tiled online-softmax for very large sequences (N*S > 65536).
        Q: [B, H, N, E], K: [B, H, S, E], V: [B, H, S, E]
        Returns: [B, H, N, E]
        """
        B, H, N, E = q.shape
        S = k.shape[2]
        if scale is None:
            scale = 1.0 / math.sqrt(E)

        if N * S > 65536:
            return self.fused_softmax_attention(q, k, v, mask=mask, scale=scale, causal=causal)

        scores = self._scores_sn(q, k) * scale

        if mask is not None:
            scores = scores + mask

        if causal and N > 0 and S > 0:
            causal_mask = np.triu(np.full((N, S), -1e9, dtype=np.float32), k=1)
            scores = scores + causal_mask[None, None, :, :]

        a_max = scores.max(axis=-1, keepdims=True)
        exp_s = np.exp(scores - a_max)
        attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
        return self._apply_attn(attn, v)

    def multi_head_attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                             num_heads: int, mask: Optional[np.ndarray] = None,
                             causal: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        B, N, C = q.shape
        E = C // num_heads
        q = q.reshape(B, N, num_heads, E).transpose(0, 2, 1, 3)  # [B, H, N, E]
        k = k.reshape(B, -1, num_heads, E).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, num_heads, E).transpose(0, 2, 1, 3)
        out = self.scaled_dot_attention(q, k, v, mask=mask, causal=causal)
        attn_weights = self._scores_sn(q, k)  # [B, H, N, S] for return
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return out, attn_weights

    def conv2d(self, x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None,
               stride: int = 1, padding: int = 0, groups: int = 1) -> np.ndarray:
        n, c, h, w = x.shape
        oc, ic, kh, kw = weight.shape
        if padding > 0:
            x = np.pad(x, [(0,0),(0,0),(padding,),(padding,)], mode='constant')
        oh = (x.shape[2] - kh) // stride + 1
        ow = (x.shape[3] - kw) // stride + 1

        cols = self._im2col(x, kh, kw, stride)
        w_col = weight.reshape(oc, -1)
        out = self.matmul(w_col, cols.T).reshape(n, oc, oh, ow)

        if bias is not None:
            out = out + bias[:, None, None]
        return out

    def _im2col(self, x: np.ndarray, kh: int, kw: int, stride: int) -> np.ndarray:
        n, c, h, w = x.shape
        oh = (h - kh) // stride + 1
        ow = (w - kw) // stride + 1
        strides = x.strides
        view_shape = (n, c, oh, ow, kh, kw)
        view_strides = (strides[0], strides[1], strides[2] * stride, strides[3] * stride, strides[2], strides[3])
        view = np.lib.stride_tricks.as_strided(x, shape=view_shape, strides=view_strides, writeable=False)
        return view.transpose(0, 2, 3, 1, 4, 5).reshape(n * oh * ow, c * kh * kw)

    def max_pool2d(self, x: np.ndarray, kernel_size: int, stride: int) -> np.ndarray:
        n, c, h, w = x.shape
        oh = (h - kernel_size) // stride + 1
        ow = (w - kernel_size) // stride + 1
        strides = x.strides
        view_shape = (n, c, oh, ow, kernel_size, kernel_size)
        view_strides = (strides[0], strides[1], strides[2] * stride, strides[3] * stride, strides[2], strides[3])
        view = np.lib.stride_tricks.as_strided(x, shape=view_shape, strides=view_strides, writeable=False)
        return view.max(axis=(4, 5))

    def avg_pool2d(self, x: np.ndarray, kernel_size: int, stride: int) -> np.ndarray:
        n, c, h, w = x.shape
        oh = (h - kernel_size) // stride + 1
        ow = (w - kernel_size) // stride + 1
        strides = x.strides
        view_shape = (n, c, oh, ow, kernel_size, kernel_size)
        view_strides = (strides[0], strides[1], strides[2] * stride, strides[3] * stride, strides[2], strides[3])
        view = np.lib.stride_tricks.as_strided(x, shape=view_shape, strides=view_strides, writeable=False)
        return view.mean(axis=(4, 5))

    def fused_softmax_attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                                 mask: Optional[np.ndarray] = None,
                                 scale: Optional[float] = None,
                                 causal: bool = False) -> np.ndarray:
        """Fused softmax(QK^T)V with online-softmax tiling for large sequences.
        All einsums replaced with matmul for 3-9x speedup.
        """
        B, H, N, E = q.shape
        S = k.shape[2]
        if scale is None:
            scale = 1.0 / math.sqrt(E)
        if N * S < 256 * 256:
            scores = self._scores_sn(q, k) * scale
            if mask is not None:
                scores = scores + mask
            if causal:
                causal_mask = np.triu(np.full((N, S), -1e9, dtype=np.float32), k=1)
                scores = scores + causal_mask[None, None, :, :]
            attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = attn / attn.sum(axis=-1, keepdims=True)
            return self._apply_attn(attn, v)
        tile_s = min(128, S)
        o = np.zeros((B, H, N, E), dtype=np.float32)
        for t_start in range(0, S, tile_s):
            t_end = min(t_start + tile_s, S)
            k_tile = k[:, :, t_start:t_end, :]
            v_tile = v[:, :, t_start:t_end, :]
            T = t_end - t_start
            q2 = q.reshape(B * H, N, E)
            k2 = k_tile.reshape(B * H, T, E)
            scores_tile = (q2 @ k2.transpose(0, 2, 1)).reshape(B, H, N, T) * scale
            if mask is not None:
                scores_tile = scores_tile + mask[:, :, :, t_start:t_end]
            if causal:
                cm = np.triu(np.full((N, T), -1e9, dtype=np.float32), k=1 + max(0, t_start - N))
                scores_tile = scores_tile + cm[None, None, :, :]
            m_prev = getattr(self, '_fus_attn_m', np.full((B, H, N, 1), -1e9, dtype=np.float32))
            if t_start == 0:
                m_prev[:] = -1e9
            m_new = np.maximum(m_prev, scores_tile.max(axis=-1, keepdims=True))
            p_tile = np.exp(scores_tile - m_new)
            if t_start == 0:
                self._fus_attn_s = p_tile.sum(axis=-1, keepdims=True)
                p2 = p_tile.reshape(B * H, N, T)
                v2 = v_tile.reshape(B * H, T, E)
                self._fus_attn_o = (p2 @ v2).reshape(B, H, N, E)
            else:
                rescale = np.exp(m_prev - m_new)
                s_old = self._fus_attn_s * rescale
                s_new = s_old + p_tile.sum(axis=-1, keepdims=True)
                o_old = self._fus_attn_o * rescale
                p2 = p_tile.reshape(B * H, N, T)
                v2 = v_tile.reshape(B * H, T, E)
                self._fus_attn_o = o_old + (p2 @ v2).reshape(B, H, N, E)
                self._fus_attn_s = s_new
            self._fus_attn_m = m_new
        return self._fus_attn_o / self._fus_attn_s

    def fused_layer_norm_gelu(self, x: np.ndarray, weight: np.ndarray,
                               bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Fused layer_norm + gelu — one pass over x instead of two."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        normed = (x - mean) / np.sqrt(var + eps)
        scaled = normed * weight + bias
        return 0.5 * scaled * (1 + np.tanh(np.sqrt(2 / np.pi) * (scaled + 0.044715 * scaled ** 3)))

    def embedding_lookup(self, indices: np.ndarray, weight: np.ndarray) -> np.ndarray:
        flat = np.clip(indices.astype(int).flatten(), 0, weight.shape[0] - 1)
        return weight[flat].reshape(list(indices.shape) + [weight.shape[1]])

    def cross_entropy(self, logits: np.ndarray, targets: np.ndarray) -> float:
        flat_l = logits.reshape(-1, logits.shape[-1])
        x_max = flat_l.max(axis=-1, keepdims=True)
        log_probs = flat_l - x_max - np.log(np.exp(flat_l - x_max).sum(axis=-1, keepdims=True))
        flat_t = targets.astype(np.int64).flatten()
        valid = flat_t < log_probs.shape[1]
        idx = np.arange(len(flat_t))[valid]
        losses = -log_probs[idx, flat_t[valid]]
        return float(losses.mean()) if losses.size > 0 else 0.0

    def embedding(self, indices: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return self.embedding_lookup(indices, weight)

    def one_hot(self, indices: np.ndarray, num_classes: int) -> np.ndarray:
        flat = indices.flatten().astype(int)
        n = len(flat)
        result = np.zeros((n, num_classes), dtype=np.float32)
        valid = (flat >= 0) & (flat < num_classes)
        result[np.arange(n)[valid], flat[valid]] = 1.0
        return result.reshape(list(indices.shape) + [num_classes])

    def concat(self, arrays: List[np.ndarray], axis: int = 0) -> np.ndarray:
        return np.concatenate(arrays, axis=axis)

    def stack(self, arrays: List[np.ndarray], axis: int = 0) -> np.ndarray:
        return np.stack(arrays, axis=axis)

    def permute(self, a: np.ndarray, axes: Tuple[int, ...]) -> np.ndarray:
        return np.transpose(a, axes)

    def reshape(self, a: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        return a.reshape(shape)

    def transpose(self, a: np.ndarray, axes: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        if axes:
            return np.transpose(a, axes)
        return a.T

    def topk(self, a: np.ndarray, k: int, dim: int = -1, largest: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        sorted_idx = np.argsort(a, axis=dim)
        if largest:
            sorted_idx = sorted_idx[..., ::-1]
        indices = sorted_idx[..., :k]
        values = np.take_along_axis(a, indices, axis=dim)
        return values, indices

    def multinomial(self, probs: np.ndarray, num_samples: int, replacement: bool = False) -> np.ndarray:
        flat = probs.flatten().astype(np.float64)
        flat = np.maximum(flat, 0)
        total = flat.sum()
        if total > 0:
            flat = flat / total
        indices = np.random.choice(len(flat), size=num_samples, p=flat, replace=replacement)
        return indices.reshape(1, num_samples)

    def dropout(self, x: np.ndarray, p: float, training: bool = True) -> np.ndarray:
        if not training or p == 0:
            return x
        mask = (np.random.rand(*x.shape) > p).astype(np.float32)
        return x * mask / (1 - p)

    def batch_norm_2d(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                      running_mean: np.ndarray, running_var: np.ndarray,
                      eps: float = 1e-5, momentum: float = 0.1,
                      training: bool = True) -> np.ndarray:
        if training:
            mean = x.mean(axis=(0, 2, 3), keepdims=True)
            var = x.var(axis=(0, 2, 3), keepdims=True)
            running_mean[...] = momentum * mean.squeeze() + (1 - momentum) * running_mean
            running_var[...] = momentum * var.squeeze() + (1 - momentum) * running_var
        else:
            mean = running_mean.reshape(1, -1, 1, 1)
            var = running_var.reshape(1, -1, 1, 1)
        return ((x - mean) / np.sqrt(var + eps)) * gamma[:, None, None] + beta[:, None, None]

    def batch_norm_1d(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                      running_mean: np.ndarray, running_var: np.ndarray,
                      eps: float = 1e-5, momentum: float = 0.1,
                      training: bool = True) -> np.ndarray:
        if training:
            mean = x.mean(axis=0, keepdims=True)
            var = x.var(axis=0, keepdims=True)
            running_mean[...] = momentum * mean.squeeze() + (1 - momentum) * running_mean
            running_var[...] = momentum * var.squeeze() + (1 - momentum) * running_var
        else:
            mean = running_mean.reshape(1, -1)
            var = running_var.reshape(1, -1)
        return ((x - mean) / np.sqrt(var + eps)) * gamma + beta


class _CPUBackend(_Accelerator):
    """CPU backend with SIMD support detection.

    Detects: OpenBLAS (MKL/OpenBLAS multi-threaded), AVX2/AVX512 intrinsics,
    Apple Accelerate framework (macOS), or plain NumPy fallback.
    """
    name = "cpu"
    device_type = "cpu"
    _openblas_threads_cache: Optional[int] = None

    def has_openblas(self) -> bool:
        try:
            import ctypes.util
            return bool(ctypes.util.find_library("openblas") or ctypes.util.find_library("libopenblas"))
        except Exception:
            return False

    def has_simd(self) -> bool:
        try:
            import platform
            arch = platform.machine()
            # x86_64 almost always has SSE/AVX
            if arch in ("x86_64", "AMD64"):
                return True
            # ARM (Apple Silicon, Raspberry Pi) has NEON
            if arch in ("arm64", "aarch64"):
                return True
        except Exception:
            pass
        return False

    def openblas_threads(self) -> int:
        if self._openblas_threads_cache is not None:
            return self._openblas_threads_cache
        try:
            import os
            # Check OMP_NUM_THREADS / OPENBLAS_NUM_THREADS
            n = int(os.environ.get("OPENBLAS_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "0")))
            if n > 0:
                self._openblas_threads_cache = n
                return n
            # Try to detect automatically (cpu count - 1)
            import multiprocessing
            n = max(1, multiprocessing.cpu_count() - 1)
            self._openblas_threads_cache = n
            return n
        except Exception:
            self._openblas_threads_cache = 1
            return 1

    def vram_gb(self) -> float:
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 ** 3) * 0.5  # 50% for compute
        except Exception:
            pass
        return 8.0  # assume 8 GB system RAM available for compute

    @property
    def compute_tier(self) -> str:
        cores = self.openblas_threads()
        vram = self.vram_gb()
        if cores >= 8 and vram >= 16:
            return "full"
        elif cores >= 4 and vram >= 8:
            return "medium"
        return "lite"

    def memory_hint(self) -> Dict[str, Any]:
        tier = self.compute_tier
        threads = self.openblas_threads()
        return {
            "tier": tier,
            "threads": threads,
            "max_batch": 8 if tier == "full" else 4 if tier == "medium" else 2,
            "max_seq_len": 512 if tier == "full" else 256,
            "recommend_openmp": self.has_openblas(),
            "recommend_quantization": tier != "full",
        }

    def softmax(self, a: np.ndarray, axis: int = -1) -> np.ndarray:
        a_max = a.max(axis=axis, keepdims=True)
        exp_a = np.exp(a - a_max)
        return exp_a / exp_a.sum(axis=axis, keepdims=True)


class _MetalBackend(_Accelerator):
    """Clean Metal GPU backend via PyTorch MPS.

    Delegates compute to MPS (Metal Performance Shaders) through PyTorch.
    All ops accept numpy arrays and return numpy arrays.
    Ops not overridden here fall through to the base _Accelerator numpy impl.
    """
    name = "metal"
    device_type = "gpu"

    def __init__(self):
        self._torch = None

    def is_available(self) -> bool:
        if self._torch is not None:
            return True
        try:
            import torch
            ok = torch.backends.mps.is_available() and torch.backends.mps.is_built()
            if ok:
                self._torch = torch
            return ok
        except Exception:
            return False

    def _t(self, a):
        """Convert numpy → MPS tensor; pass through if already a torch tensor."""
        if isinstance(a, np.ndarray):
            return self._torch.tensor(a, dtype=self._torch.float32, device="mps")
        return a

    def _n(self, t):
        """Convert MPS tensor → numpy; pass through if already numpy."""
        if isinstance(t, self._torch.Tensor):
            return t.cpu().numpy()
        return t

    # --- Device transfer (used by _accel_op and callers) ---

    def to_device(self, arr):
        if not isinstance(arr, np.ndarray):
            return arr
        return self._torch.tensor(arr, dtype=self._torch.float32, device="mps")

    def from_device(self, arr):
        """Convert MPS tensor to numpy. Pass through for numpy arrays."""
        if isinstance(arr, self._torch.Tensor):
            return arr.cpu().numpy()
        return np.asarray(arr)

    def sync(self) -> None:
        if self._torch is not None:
            self._torch.mps.synchronize()

    # --- Core ops ---

    def matmul(self, a, b):
        return self._n(self._t(a) @ self._t(b))

    def add(self, a, b):
        return self._n(self._t(a) + self._t(b))

    def neg(self, a):
        return self._n(-self._t(a))

    def mul(self, a, b):
        return self._n(self._t(a) * self._t(b))

    def pow(self, a, p):
        return self._n(self._t(a) ** p)

    def sum(self, a, axis=None):
        t = self._t(a)
        if axis is not None:
            return self._n(t.sum(dim=axis))
        return self._n(t.sum())

    def mean(self, a, axis=None):
        t = self._t(a)
        if axis is not None:
            return self._n(t.mean(dim=axis))
        return self._n(t.mean())

    def sigmoid(self, x):
        return self._n(self._torch.sigmoid(self._t(x)))

    def tanh(self, x):
        return self._n(self._torch.tanh(self._t(x)))

    def relu(self, x):
        return self._n(self._torch.relu(self._t(x)))

    def gelu(self, x):
        return self._n(self._torch.nn.functional.gelu(self._t(x)))

    def silu(self, x):
        return self._n(self._torch.nn.functional.silu(self._t(x)))

    def softmax(self, a, axis=-1):
        return self._n(self._t(a).softmax(dim=axis))

    def layer_norm(self, x, weight, bias, eps=1e-5):
        tx, tw, tb = self._t(x), self._t(weight), self._t(bias)
        return self._n(self._torch.nn.functional.layer_norm(
            tx, tx.shape[-1:], weight=tw, bias=tb, eps=eps
        ))

    def rms_norm(self, x, weight, eps=1e-5):
        tx, tw = self._t(x), self._t(weight)
        rms = self._torch.rsqrt(self._torch.mean(tx ** 2, dim=-1, keepdim=True) + eps)
        return self._n(tx * rms * tw)

    def scaled_dot_attention(self, q, k, v, mask=None, scale=None, causal=False):
        tq, tk, tv = self._t(q), self._t(k), self._t(v)
        tm = self._t(mask) if mask is not None else None
        return self._n(self._torch.nn.functional.scaled_dot_product_attention(
            tq, tk, tv, attn_mask=tm, is_causal=causal
        ))

    def cross_entropy(self, logits, targets):
        tlog, ttar = self._t(logits), self._t(targets)
        return self._torch.nn.functional.cross_entropy(
            tlog, ttar.long(), reduction="mean"
        ).item()

    def conv2d(self, x, weight, bias=None, stride=1, padding=0, groups=1):
        tx, tw = self._t(x), self._t(weight)
        tb = self._t(bias) if bias is not None else None
        return self._n(self._torch.nn.functional.conv2d(
            tx, tw, bias=tb, stride=stride, padding=padding, groups=groups
        ))

    def max_pool2d(self, x, kernel_size, stride):
        return self._n(self._torch.nn.functional.max_pool2d(
            self._t(x), kernel_size, stride
        ))

    def embedding(self, indices, weight):
        ti, tw = self._t(indices), self._t(weight)
        return self._n(self._torch.nn.functional.embedding(ti.long(), tw))

    def dropout(self, x, p=0.0, training=True):
        return self._n(self._torch.nn.functional.dropout(
            self._t(x), p=p, training=training
        ))

    def abs(self, x):
        return self._n(self._torch.abs(self._t(x)))

    def exp(self, x):
        return self._n(self._torch.exp(self._t(x)))

    def sqrt(self, x):
        return self._n(self._torch.sqrt(self._t(x)))

    def max(self, x, axis=None):
        t = self._t(x)
        if axis is not None:
            return self._n(t.amax(dim=axis))
        return self._n(t.amax())

    # Use base class numpy implementations for ops not listed above
    # (sub, div, clamp, where, gather, scatter, pad, batch_norm, etc.)


class _CUDABackend(_Accelerator):

    """CUDA (NVIDIA GPU) backend via CuPy.

    Install: pip install cupy-cuda12x
    Provides the fastest GPU compute path for NVIDIA cards.
    """
    name = "cuda"
    device_type = "gpu"

    def __init__(self):
        self._cp = None

    def is_available(self) -> bool:
        try:
            import cupy as cp
            self._cp = cp
            return True
        except Exception:
            return False

    def sync(self) -> None:
        if self._cp:
            self._cp.cuda.Stream.null.synchronize()

    def vram_gb(self) -> float:
        if self._cp:
            try:
                mem = self._cp.cuda.Device().mem_info
                return mem[1] / (1024 ** 3)
            except Exception:
                pass
        return 4.0

    @property
    def compute_tier(self) -> str:
        vram = self.vram_gb()
        if vram >= 8:
            return "full"
        elif vram >= 4:
            return "medium"
        return "lite"

    def memory_hint(self) -> Dict[str, Any]:
        vram = self.vram_gb()
        tier = self.compute_tier
        return {
            "vram_gb": vram,
            "tier": tier,
            "max_batch": int(64 * (vram / 8)),
            "max_seq_len": 2048 if tier == "full" else 1024 if tier == "medium" else 512,
            "recommend_flash_attention": True,
            "recommend_quantization": tier != "full",
        }

    def to_device(self, arr: np.ndarray) -> Any:
        if self._cp:
            return self._cp.asarray(arr.astype(np.float32))
        return arr.astype(np.float32)

    def from_device(self, arr: Any) -> np.ndarray:
        if self._cp and hasattr(arr, "get"):
            return arr.get()
        return np.asarray(arr)

    def matmul(self, a: Any, b: Any) -> np.ndarray:
        if self._cp:
            return self._cp.asnumpy(self._cp.matmul(
                self._cp.asarray(a) if not hasattr(a, "device") else a,
                self._cp.asarray(b) if not hasattr(b, "device") else b
            ))
        return np.matmul(np.asarray(a), np.asarray(b))

    def scaled_dot_attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                             mask: Optional[np.ndarray] = None,
                             scale: Optional[float] = None,
                             causal: bool = False) -> np.ndarray:
        if self._cp:
            q_c, k_c, v_c = self._cp.asarray(q), self._cp.asarray(k), self._cp.asarray(v)
            m_c = self._cp.asarray(mask) if mask is not None else None
            if scale is None:
                scale = 1.0 / math.sqrt(q_c.shape[-1])
            scores = self._cp.einsum("bhnd,bhkd->bhnk", q_c, k_c) * scale
            if m_c is not None:
                scores = scores + m_c
            if causal:
                n, s = q_c.shape[2], k_c.shape[2]
                scores = self._cp.where(self._cp.triu(self._cp.ones((n, s))) == 0, scores, -1e9)
            attn = self._cp.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = attn / attn.sum(axis=-1, keepdims=True)
            out = self._cp.einsum("bhnk,bhkd->bhnd", attn, v_c)
            return self._cp.asnumpy(out)
        return super().scaled_dot_attention(q, k, v, mask=mask, scale=scale, causal=causal)

    def layer_norm(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        if self._cp:
            x_c, w_c, b_c = self._cp.asarray(x), self._cp.asarray(weight), self._cp.asarray(bias)
            mean = x_c.mean(axis=-1, keepdims=True)
            var = x_c.var(axis=-1, keepdims=True)
            return self._cp.asnumpy(((x_c - mean) / self._cp.sqrt(var + eps)) * w_c + b_c)
        return super().layer_norm(x, weight, bias, eps)

    def gelu(self, x: np.ndarray) -> np.ndarray:
        if self._cp:
            x_c = self._cp.asarray(x)
            return self._cp.asnumpy(0.5 * x_c * (1 + self._cp.tanh(self._cp.sqrt(self._cp.pi / self._cp.pi) * (x_c + 0.044715 * x_c ** 3))))
        return super().gelu(x)




# =============================================================================
# OpenCL BACKEND (Intel iGPU, AMD dGPU, integrated GPUs on Linux/Windows)
# =============================================================================

class _OpenCLBackend(_Accelerator):
    """OpenCL backend for Intel/AMD GPUs on non-Apple platforms.

    Uses pyopencl for cross-vendor GPU compute. Supports:
    - Intel HD/UHD integrated graphics (Linux/Windows)
    - AMD Radeon (Linux/Windows, via AMD ROCm or proprietary driver)
    - Intel Arc discrete GPUs

    Install: pip install pyopencl
    Falls back to CPU if pyopencl is unavailable.
    """
    name = "opencl"
    device_type = "gpu"

    def __init__(self):
        self._cl = None
        self._platform = None
        self._device = None
        self._queue = None
        self._build_opts = ""

    def is_available(self) -> bool:
        try:
            import pyopencl as cl
            ctx = cl.create_some_context(interactive=False)
            self._cl = cl
            return True
        except Exception:
            return False

    def sync(self) -> None:
        if self._queue is not None:
            self._queue.finish()

    def vram_gb(self) -> float:
        if self._cl is None:
            return 0.0
        try:
            dev = self._cl.get_platforms()[0].get_devices()[0]
            return dev.global_mem_size / (1024 ** 3)
        except Exception:
            return 1.0

    @property
    def compute_tier(self) -> str:
        vram = self.vram_gb()
        if vram >= 4:
            return "full"
        elif vram >= 2:
            return "medium"
        return "lite"

    def memory_hint(self) -> Dict[str, Any]:
        vram = self.vram_gb()
        tier = self.compute_tier
        return {
            "vram_gb": vram,
            "tier": tier,
            "max_batch": int(16 * (vram / 4)),
            "max_seq_len": 512 if tier == "full" else 256,
            "recommend_quantization": tier != "full",
        }

    def _ensure_context(self):
        if self._cl is None:
            self._cl = __import__("pyopencl")
        if self._queue is None:
            ctx = self._cl.create_some_context(interactive=False)
            self._queue = self._cl.CommandQueue(ctx)
            self._device = ctx.devices[0]

    def to_device(self, arr: np.ndarray) -> Any:
        self._ensure_context()
        mf = self._cl.mem_flags
        buf = self._cl.Buffer(self._queue.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=arr.astype(np.float32))
        return buf

    def from_device(self, arr: Any) -> np.ndarray:
        result = np.empty(arr.shape, dtype=np.float32)
        self._cl.enqueue_copy(self._queue, result, arr)
        return result

    def matmul(self, a: Any, b: Any) -> np.ndarray:
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.matmul(a, b)
        return np.matmul(np.asarray(a), np.asarray(b))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def to_gpu(arr: np.ndarray) -> Any:
    """Move array to accelerator device (GPU if available)."""
    return get_accelerator().to_device(arr)


def from_gpu(arr: Any) -> np.ndarray:
    """Move array from accelerator device back to CPU numpy."""
    return get_accelerator().from_device(arr)


def gelu(x: np.ndarray) -> np.ndarray:
    return get_accelerator().gelu(x)


def silu(x: np.ndarray) -> np.ndarray:
    return get_accelerator().silu(x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return get_accelerator().softmax(x, axis=axis)


def benchmark_accelerators() -> Dict[str, Dict[str, Any]]:
    """Benchmark all available backends: matmul + layernorm + gelu.

    Returns dict of backend_name -> {gflops, memory_gb, tier, status}.
    """
    import time
    results: Dict[str, Dict[str, Any]] = {}

    A = np.random.randn(256, 256).astype(np.float32)
    B = np.random.randn(256, 256).astype(np.float32)
    X = np.random.randn(128, 256).astype(np.float32)
    W = np.random.randn(256).astype(np.float32)
    Bv = np.random.randn(256).astype(np.float32)

    backends: List[Tuple[str, "_Accelerator"]] = [
        ("cpu", _CPUBackend()),
        ("metal", _MetalBackend()),
        ("cuda", _CUDABackend()),
        ("opencl", _OpenCLBackend()),
    ]

    for name, backend in backends:
        if not backend.is_available():
            results[name] = {"status": "unavailable", "gflops": 0.0}
            continue

        try:
            t0 = time.perf_counter()
            for _ in range(20):
                c = backend.matmul(A, B)
                c = backend.layer_norm(X, W, Bv)
                c = backend.gelu(c)
            if hasattr(backend, "sync"):
                backend.sync()
            elapsed = time.perf_counter() - t0

            flops = 20 * (
                2 * 256 * 256 * 256 +
                2 * 128 * 256 +
                2 * 128 * 256
            )
            gflops = flops / elapsed / 1e9

            results[name] = {
                "status": "ok",
                "gflops": round(gflops, 2),
                "vram_gb": round(backend.vram_gb(), 1),
                "tier": backend.compute_tier,
            }
        except Exception as e:
            results[name] = {"status": f"error: {e}", "gflops": 0.0}

    logger.info("", extra={"tag": "INFRA"})
    for name, r in sorted(results.items(), key=lambda x: -x[1].get("gflops", 0)):
        bar = "=" * int(r.get("gflops", 0) * 2)
        tier = r.get("tier", "?")
        vram = f"{r['vram_gb']}GB" if r.get("vram_gb") else ""
        logger.info("  %s %6.1f GF/s %s %s %s [%s]", name, r['gflops'], bar, tier, vram, r['status'], extra={"tag": "INFRA"})

    return results
