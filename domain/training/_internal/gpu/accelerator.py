"""GPU Acceleration Layer for SloNet

Supports Metal (macOS), CUDA (NVIDIA), and CPU fallback.
All tensors stay as numpy arrays backed by GPU buffers when available.

Usage:
    from domain.training._internal.gpu.accelerator import get_accelerator

    acc = get_accelerator()          # auto-detect best backend
    result = acc.matmul(a, b)        # GPU-accelerated matmul
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# BACKEND DETECTION
# =============================================================================


class _MetalAccelerator:
    """Metal (Apple GPU) awareness.

    Detects MPS availability but uses numpy for all compute operations.
    The torch shim handles its own autograd — for real MPS speed,
    use a torch-based training pipeline.
    """

    name = "metal"
    device_type = "gpu"

    def __init__(self):
        self._available = self._check_metal()

    def _check_metal(self) -> bool:
        """Check if MPS is available without importing torch."""
        try:
            from domain.infrastructure._internal.ml_types import _mps_available

            return _mps_available()
        except Exception:
            return False

    def is_available(self) -> bool:
        return self._available

    def to_device(self, arr: np.ndarray) -> np.ndarray:
        return arr.astype(np.float32).copy()

    def from_device(self, arr: Any) -> np.ndarray:
        return np.asarray(arr)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray(a) + np.asarray(b)

    def softmax(self, arr: np.ndarray, axis: int = -1) -> np.ndarray:
        a = np.asarray(arr)
        e = np.exp(a - np.max(a, axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)

    def gelu(self, arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr)
        return 0.5 * a * (1 + np.tanh(np.sqrt(2 / np.pi) * (a + 0.044715 * a**3)))

    def silu(self, arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr)
        return a / (1 + np.exp(-np.clip(a, -500, 500)))

    def layernorm(self, arr: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        a = np.asarray(arr)
        mean = a.mean(axis=-1, keepdims=True)
        var = a.var(axis=-1, keepdims=True)
        return ((a - mean) / np.sqrt(var + eps)) * np.asarray(weight) + np.asarray(bias)

    def rmsnorm(self, arr: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        a = np.asarray(arr)
        rms = np.sqrt(np.mean(a**2, axis=-1, keepdims=True) + eps)
        return (a / rms) * np.asarray(weight)

    def attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float = 1.0) -> np.ndarray:
        qd = np.asarray(q)
        kd = np.asarray(k)
        vd = np.asarray(v)
        scores = np.matmul(qd, kd.T) * scale
        attn = self.softmax(scores, axis=-1)
        return np.matmul(attn, vd)

    def scaled_dot_attention(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None, scale: float | None = None
    ) -> np.ndarray:
        qd = np.asarray(q)
        kd = np.asarray(k)
        vd = np.asarray(v)
        if scale is None:
            scale = 1.0 / np.sqrt(kd.shape[-1])
        scores = np.matmul(qd, np.swapaxes(kd, -2, -1)) * scale
        if mask is not None:
            scores = scores + np.asarray(mask)
        attn = self.softmax(scores, axis=-1)
        return np.matmul(attn, vd)

    def dropout(self, arr: np.ndarray, p: float = 0.0) -> np.ndarray:
        a = np.asarray(arr)
        if p <= 0:
            return a
        mask = np.random.binomial(1, 1 - p, a.shape) / (1 - p)
        return a * mask

    def embedding(self, input_indices: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return np.asarray(weight)[np.asarray(input_indices).astype(int)]

    def cross_entropy(self, logits: np.ndarray, targets: np.ndarray) -> float:
        logits_a = np.asarray(logits)
        targets_a = np.asarray(targets).astype(int)
        shifted = logits_a - logits_a.max(axis=-1, keepdims=True)
        softmax = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        batch = np.arange(len(targets_a))
        return float(-np.log(np.maximum(softmax[batch, targets_a], 1e-10)).mean())

    def conv2d(
        self, input: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None, stride: int = 1, padding: int = 0
    ) -> np.ndarray:
        return _CPUAccelerator._conv2d_impl(input, weight, bias, stride, padding)

    def max_pool2d(
        self, input: np.ndarray, kernel_size: int | tuple[int, int] = 2, stride: int | tuple[int, int] | None = None, padding: int = 0
    ) -> np.ndarray:
        inp = np.asarray(input)
        n, c, h, w = inp.shape
        kh = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        kw = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
        st = stride if stride is not None else kh
        sh = st if isinstance(st, int) else st[0]
        sw = st if isinstance(st, int) else st[1]
        out_h = (h + 2 * padding - kh) // sh + 1
        out_w = (w + 2 * padding - kw) // sw + 1
        result = np.zeros((n, c, out_h, out_w), dtype=np.float32)
        for i in range(n):
            for ci in range(c):
                for oh in range(out_h):
                    for ow in range(out_w):
                        ih = oh * sh - padding
                        iw = ow * sw - padding
                        ph = inp[i, ci, max(0, ih) : min(h, ih + kh), max(0, iw) : min(w, iw + kw)]
                        if ph.size > 0:
                            result[i, ci, oh, ow] = ph.max()
        return result


class _CUDAAccelerator:
    """CUDA (NVIDIA GPU) awareness.

    Detects CuPy availability for GPU-accelerated numpy operations.
    Falls back to CPU numpy when CuPy is not installed.
    """

    name = "cuda"
    device_type = "gpu"

    def __init__(self):
        self._cp = None
        self._available = self._check_cuda()

    def _check_cuda(self) -> bool:
        """Check if CUDA is available via CuPy; env-var detection as fallback."""
        try:
            import cupy as cp

            cp.cuda.runtime.getDeviceCount()
            self._cp = cp
            return True
        except Exception as e:
            logger.debug("cupy/CUDA check failed: %s", e)

        # Check via CUDA_VISIBLE_DEVICES (cupy missing or no device)
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
            return True
        return False

    def is_available(self) -> bool:
        return self._available and self._cp is not None

    def to_device(self, arr: np.ndarray) -> Any:
        if self._cp:
            return self._cp.asarray(arr.astype(np.float32))
        return arr.astype(np.float32)

    def from_device(self, arr: Any) -> np.ndarray:
        if self._cp and hasattr(arr, "get"):
            return arr.get()
        return np.asarray(arr)

    def matmul(self, a: Any, b: Any) -> Any:
        if self._cp:
            return self._cp.matmul(a, b)
        return np.matmul(np.asarray(a), np.asarray(b))

    def add(self, a: Any, b: Any) -> Any:
        if self._cp:
            return a + b
        return np.asarray(a) + np.asarray(b)

    def softmax(self, arr: Any, axis: int = -1) -> Any:
        a = np.asarray(arr) if not isinstance(arr, np.ndarray) and self._cp is None else arr
        exp_arr = np.exp(a - np.max(a, axis=axis, keepdims=True))
        return exp_arr / np.sum(exp_arr, axis=axis, keepdims=True)

    def gelu(self, arr: Any) -> Any:
        a = np.asarray(arr)
        return 0.5 * a * (1 + np.tanh(np.sqrt(2 / np.pi) * (a + 0.044715 * a**3)))

    def layernorm(self, arr: Any, weight: Any, bias: Any, eps: float = 1e-5) -> Any:
        a = np.asarray(arr)
        mean = a.mean(axis=-1, keepdims=True)
        var = a.var(axis=-1, keepdims=True)
        return ((a - mean) / np.sqrt(var + eps)) * np.asarray(weight) + np.asarray(bias)

    def attention(self, q: Any, k: Any, v: Any, scale: float = 1.0) -> Any:
        s = np.matmul(np.asarray(q), np.asarray(k).T) * scale
        attn = self.softmax(s, axis=-1)
        return np.matmul(attn, np.asarray(v))

    def conv2d(
        self, input: Any, weight: Any, bias: Any | None, stride: int = 1, padding: int = 0
    ) -> Any:
        inp = np.asarray(input)
        w = np.asarray(weight)
        b = np.asarray(bias) if bias is not None else None
        return _CUDAAccelerator._conv2d_impl(inp, w, b, stride, padding)

    @staticmethod
    def _conv2d_impl(
        input: np.ndarray, weight: np.ndarray, bias: np.ndarray | None, stride: int, padding: int
    ) -> np.ndarray:
        n, c, h, w = input.shape
        oc, ic, kh, kw = weight.shape
        if padding > 0:
            input = np.pad(input, [(0, 0), (0, 0), (padding, padding), (padding, padding)], mode="constant")
        out_h = (input.shape[2] - kh) // stride + 1
        out_w = (input.shape[3] - kw) // stride + 1
        result = np.zeros((n, oc, out_h, out_w), dtype=np.float32)
        for i in range(n):
            for oc_idx in range(oc):
                for oh in range(out_h):
                    for ow in range(out_w):
                        ih = oh * stride
                        iw = ow * stride
                        patch = input[i, :, ih : ih + kh, iw : iw + kw]
                        result[i, oc_idx, oh, ow] = np.sum(patch * weight[oc_idx]) + (
                            bias[oc_idx] if bias is not None else 0
                        )
        return result


class _CPUAccelerator:
    """CPU fallback — plain numpy."""

    name = "cpu"
    device_type = "cpu"

    def is_available(self) -> bool:
        return True

    def to_device(self, arr: np.ndarray) -> np.ndarray:
        return arr.astype(np.float32)

    def from_device(self, arr: Any) -> np.ndarray:
        return np.asarray(arr)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    def softmax(self, arr: np.ndarray, axis: int = -1) -> np.ndarray:
        exp_arr = np.exp(arr - np.max(arr, axis=axis, keepdims=True))
        return exp_arr / np.sum(exp_arr, axis=axis, keepdims=True)

    def gelu(self, arr: np.ndarray) -> np.ndarray:
        return 0.5 * arr * (1 + np.tanh(np.sqrt(2 / np.pi) * (arr + 0.044715 * arr**3)))

    def silu(self, arr: np.ndarray) -> np.ndarray:
        return arr / (1 + np.exp(-arr))

    def layernorm(self, arr: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = arr.mean(axis=-1, keepdims=True)
        var = arr.var(axis=-1, keepdims=True)
        return ((arr - mean) / np.sqrt(var + eps)) * weight + bias

    def rmsnorm(self, arr: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        rms = np.sqrt(np.mean(arr**2, axis=-1, keepdims=True) + eps)
        return (arr / rms) * weight

    def attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float = 1.0) -> np.ndarray:
        scores = np.matmul(q, k.T) * scale
        attn = self.softmax(scores, axis=-1)
        return np.matmul(attn, v)

    def scaled_dot_attention(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None, scale: float | None = None
    ) -> np.ndarray:
        if scale is None:
            scale = 1.0 / np.sqrt(k.shape[-1])
        scores = np.matmul(q, np.swapaxes(k, -2, -1)) * scale
        if mask is not None:
            scores = scores + mask
        attn = self.softmax(scores, axis=-1)
        return np.matmul(attn, v)

    @staticmethod
    def _conv2d_impl(
        input: np.ndarray, weight: np.ndarray, bias: np.ndarray | None, stride: int, padding: int
    ) -> np.ndarray:
        n, c, h, w = input.shape
        oc, ic, kh, kw = weight.shape
        if padding > 0:
            input = np.pad(input, [(0, 0), (0, 0), (padding, padding), (padding, padding)], mode="constant")
        out_h = (input.shape[2] - kh) // stride + 1
        out_w = (input.shape[3] - kw) // stride + 1
        result = np.zeros((n, oc, out_h, out_w), dtype=np.float32)
        for i in range(n):
            for oc_idx in range(oc):
                for oh in range(out_h):
                    for ow in range(out_w):
                        ih = oh * stride
                        iw = ow * stride
                        patch = input[i, :, ih : ih + kh, iw : iw + kw]
                        result[i, oc_idx, oh, ow] = np.sum(patch * weight[oc_idx]) + (
                            bias[oc_idx] if bias is not None else 0
                        )
        return result

    def conv2d(
        self, input: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None, stride: int = 1, padding: int = 0
    ) -> np.ndarray:
        return _CPUAccelerator._conv2d_impl(input, weight, bias, stride, padding)

    def max_pool2d(self, input: np.ndarray, kernel_size: int = 2, stride: int = 2) -> np.ndarray:
        n, c, h, w = input.shape
        out_h = (h - kernel_size) // stride + 1
        out_w = (w - kernel_size) // stride + 1
        result = np.full((n, c, out_h, out_w), -np.inf, dtype=np.float32)
        for i in range(n):
            for ch in range(c):
                for oh in range(out_h):
                    for ow in range(out_w):
                        ih = oh * stride
                        iw = ow * stride
                        result[i, ch, oh, ow] = input[i, ch, ih : ih + kernel_size, iw : iw + kernel_size].max()
        return result

    def embedding(self, indices: np.ndarray, weight: np.ndarray) -> np.ndarray:
        flat = np.clip(indices.astype(int).flatten(), 0, weight.shape[0] - 1)
        return weight[flat].reshape(list(indices.shape) + [weight.shape[-1]])

    def cross_entropy(self, logits: np.ndarray, targets: np.ndarray) -> float:
        n, vocab = logits.shape
        log_probs = logits - np.max(logits, axis=-1, keepdims=True)
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))
        targets_flat = targets.astype(int).flatten()
        loss = 0.0
        for i in range(n):
            if targets_flat[i] < vocab:
                loss -= log_probs[i, targets_flat[i]]
        return loss / n

    def dropout(self, arr: np.ndarray, p: float, training: bool = True) -> np.ndarray:
        if not training or p == 0:
            return arr
        mask = (np.random.rand(*arr.shape) > p).astype(np.float32)
        return arr * mask / (1 - p)


# =============================================================================
# GLOBAL ACCELERATOR
# =============================================================================

_accelerator: object | None = None


def get_accelerator() -> object:
    """Get the best available accelerator (CUDA > Metal > CPU)."""
    global _accelerator
    if _accelerator is not None:
        return _accelerator

    # Priority: CUDA > Metal > CPU
    if _CUDAAccelerator().is_available():
        _accelerator = _CUDAAccelerator()
    elif _MetalAccelerator().is_available():
        _accelerator = _MetalAccelerator()
    else:
        _accelerator = _CPUAccelerator()

    return _accelerator


def to_gpu(arr: np.ndarray) -> Any:
    """Move numpy array to GPU."""
    return get_accelerator().to_device(arr)


def from_gpu(arr: Any) -> np.ndarray:
    """Move GPU array back to CPU numpy."""
    return get_accelerator().from_device(arr)


def reset_accelerator() -> None:
    """Reset accelerator (for testing or switching backends)."""
    global _accelerator
    _accelerator = None


def cholesky(A: object) -> object:
    """Cholesky decomposition A = L @ L.T using numpy."""
    import numpy as np

    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                L[i, j] = np.sqrt(max(A[i, i] - s, 0))
            else:
                L[i, j] = (A[i, j] - s) / L[j, j] if L[j, j] != 0 else 0
    return L


def solve_triangular(L: object, b: object, lower: bool = True) -> object:
    """Solve L @ x = b for triangular L — forward substitution for lower,
    back substitution for upper (``lower=False``)."""
    import numpy as np

    n = L.shape[0]
    # dtype follows the inputs (float32 in ⇒ float32 out); ints promote to a
    # float so the division below never truncates.
    x = np.zeros(n, dtype=np.result_type(np.asarray(L).dtype, np.asarray(b).dtype, np.float32))
    if lower:
        for i in range(n):
            x[i] = (b[i] - sum(L[i, j] * x[j] for j in range(i))) / L[i, i] if L[i, i] != 0 else 0
    else:
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - sum(L[i, j] * x[j] for j in range(i + 1, n))) / L[i, i] if L[i, i] != 0 else 0
    return x


def solve_cholesky(A: object, b: object) -> object:
    """Solve A @ x = b via Cholesky decomposition."""
    L = cholesky(A)
    y = solve_triangular(L, b, lower=True)
    return solve_triangular(L.T, y, lower=False)


def dominant_eigen(A: object, n_eigen: int = 1, max_iter: int = 100, tol: float = 1e-10) -> tuple:
    """Dominant eigenpairs via power iteration with Hotelling deflation.

    Returns ``(vals, vecs)``: ``vals`` shape ``(n_eigen,)`` ordered by
    descending magnitude, ``vecs`` shape ``(n, n_eigen)`` with unit-norm
    columns. Eigenvalues use the Rayleigh quotient against the original ``A``
    so deflated iteration stays faithful for symmetric inputs.
    """
    import numpy as np

    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    k = max(1, min(int(n_eigen), n))
    work = A.copy()
    vals = np.zeros(k)
    vecs = np.zeros((n, k))
    for j in range(k):
        v = np.ones(n) / np.sqrt(n)
        lam = 0.0
        for _ in range(max_iter):
            w = work @ v
            norm = np.linalg.norm(w)
            if norm == 0:
                break
            v = w / norm
            lam_new = float(v @ A @ v)
            if abs(lam_new - lam) < tol:
                lam = lam_new
                break
            lam = lam_new
        vals[j] = float(v @ A @ v)
        vecs[:, j] = v / np.linalg.norm(v)
        work = work - vals[j] * np.outer(vecs[:, j], vecs[:, j])
    return vals, vecs
