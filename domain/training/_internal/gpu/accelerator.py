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
        # float32 is the accelerator's resident dtype; the extra copy guarantees
        # the host array is never aliased by (or pinned through) the device one.
        return arr.astype(np.float32).copy()

    def from_device(self, arr: Any) -> np.ndarray:
        return np.asarray(arr)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b

    # -- elementwise / activation ------------------------------------------
    # Restored verbatim from the pre-5a0b901b2 implementation: the "cleanup"
    # deleted 384 lines including every op below, and test_gpu_accelerator /
    # test_gpu_linalg had been written against them months earlier. Metal is a
    # numpy-backed backend by design (see the class docstring), so these are
    # plain numpy.

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    def softmax(self, arr: np.ndarray, axis: int = -1) -> np.ndarray:
        exp_arr = np.exp(arr - np.max(arr, axis=axis, keepdims=True))
        return exp_arr / np.sum(exp_arr, axis=axis, keepdims=True)

    def gelu(self, arr: np.ndarray) -> np.ndarray:
        return 0.5 * arr * (1 + np.tanh(np.sqrt(2 / np.pi) * (arr + 0.044715 * arr**3)))

    def silu(self, arr: np.ndarray) -> np.ndarray:
        return arr / (1 + np.exp(-np.clip(arr, -500, 500)))

    def layernorm(
        self, arr: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:
        mean = arr.mean(axis=-1, keepdims=True)
        var = arr.var(axis=-1, keepdims=True)
        return ((arr - mean) / np.sqrt(var + eps)) * weight + bias

    def rmsnorm(self, arr: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        rms = np.sqrt(np.mean(arr**2, axis=-1, keepdims=True) + eps)
        return (arr / rms) * weight

    # -- attention ----------------------------------------------------------

    def attention(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float = 1.0
    ) -> np.ndarray:
        scores = np.matmul(q, k.T) * scale
        return np.matmul(self.softmax(scores, axis=-1), v)

    def scaled_dot_attention(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: np.ndarray | None = None,
        scale: float | None = None,
    ) -> np.ndarray:
        if scale is None:
            scale = 1.0 / np.sqrt(k.shape[-1])
        scores = np.matmul(q, np.swapaxes(k, -2, -1)) * scale
        if mask is not None:
            scores = scores + mask
        return np.matmul(self.softmax(scores, axis=-1), v)

    # -- conv / pool / embedding / loss ------------------------------------

    @staticmethod
    def _conv2d_impl(
        input: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray | None,
        stride: int,
        padding: int,
    ) -> np.ndarray:
        n, _c, h, w = input.shape
        oc, _ic, kh, kw = weight.shape
        if padding > 0:
            input = np.pad(
                input, [(0, 0), (0, 0), (padding, padding), (padding, padding)], mode="constant"
            )
        out_h = (input.shape[2] - kh) // stride + 1
        out_w = (input.shape[3] - kw) // stride + 1
        result = np.zeros((n, oc, out_h, out_w), dtype=np.float32)
        for i in range(n):
            for oc_idx in range(oc):
                for oh in range(out_h):
                    for ow in range(out_w):
                        patch = input[
                            i, :, oh * stride : oh * stride + kh, ow * stride : ow * stride + kw
                        ]
                        result[i, oc_idx, oh, ow] = np.sum(patch * weight[oc_idx]) + (
                            bias[oc_idx] if bias is not None else 0
                        )
        return result

    def conv2d(
        self,
        input: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray | None = None,
        stride: int = 1,
        padding: int = 0,
    ) -> np.ndarray:
        return _MetalAccelerator._conv2d_impl(input, weight, bias, stride, padding)

    def max_pool2d(
        self,
        input: np.ndarray,
        kernel_size: Any = 2,
        stride: Any = None,
        padding: int = 0,
    ) -> np.ndarray:
        n, c, h, w = input.shape
        kh = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        kw = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
        st = stride if stride is not None else kh
        sh = st if isinstance(st, int) else st[0]
        sw = st if isinstance(st, int) else st[1]
        out_h = (h + 2 * padding - kh) // sh + 1
        out_w = (w + 2 * padding - kw) // sw + 1
        result = np.full((n, c, out_h, out_w), -np.inf, dtype=np.float32)
        for i in range(n):
            for ci in range(c):
                for oh in range(out_h):
                    for ow in range(out_w):
                        ih = oh * sh - padding
                        iw = ow * sw - padding
                        patch = input[
                            i, ci, max(0, ih) : min(h, ih + kh), max(0, iw) : min(w, iw + kw)
                        ]
                        if patch.size > 0:
                            result[i, ci, oh, ow] = patch.max()
        return result

    def embedding(self, indices: np.ndarray, weight: np.ndarray) -> np.ndarray:
        # Direct indexing: an out-of-range index is a caller bug and must raise,
        # not be silently clipped (the CPU backend clips by contract instead).
        return weight[np.asarray(indices).astype(int)]

    def cross_entropy(self, logits: np.ndarray, targets: np.ndarray) -> float:
        logits_a = np.asarray(logits) - np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(logits_a) / np.exp(logits_a).sum(axis=-1, keepdims=True)
        targets_a = np.asarray(targets).astype(int)
        batch = np.arange(len(targets_a))
        return float(-np.log(np.maximum(probs[batch, targets_a], 1e-10)).mean())

    def dropout(self, arr: np.ndarray, p: float = 0.0) -> np.ndarray:
        if p <= 0:
            return arr
        mask = np.random.binomial(1, 1 - p, arr.shape) / (1 - p)
        return arr * mask


class _CUDAAccelerator:
    """CUDA (NVIDIA GPU) awareness.

    Detects CuPy availability for GPU-accelerated numpy operations.
    Falls back to CPU numpy when CuPy is not installed.
    """

    name = "cuda"
    device_type = "gpu"

    def __init__(self):
        self._available = self._check_cuda()
        self._cp = None

    def _check_cuda(self) -> bool:
        """Check if CUDA is available via CuPy."""
        try:
            import cupy as cp

            cp.cuda.runtime.getDeviceCount()
            return True
        except Exception:
            return False

    def is_available(self) -> bool:
        return self._available

    def to_device(self, arr: np.ndarray) -> Any:
        if self._cp is None:
            import cupy as cp

            self._cp = cp
        # Resident dtype is float32, so a float64 host array must be narrowed
        # before upload (keeps device memory from doubling). Cast only when
        # needed: `astype` copies unconditionally, and that copy showed up as a
        # +77% tax on the common already-float32 path (1263 -> 2232 us/call).
        return self._cp.asarray(arr if arr.dtype == np.float32 else arr.astype(np.float32))

    def from_device(self, arr: Any) -> np.ndarray:
        if self._cp is None:
            import cupy as cp

            self._cp = cp
        return self._cp.asnumpy(arr)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self._cp is None:
            import cupy as cp

            self._cp = cp
        a_gpu = self._cp.asarray(a)
        b_gpu = self._cp.asarray(b)
        return self._cp.asnumpy(a_gpu @ b_gpu)

    # Restored from the pre-5a0b901b2 implementation. Every op below dispatches
    # to CuPy only when `_cp` is set, and otherwise falls back to numpy — that
    # fallback is the contract TestCUDAFallbackAccelerator exercises, and
    # deleting it left the class unable to answer `add`/`softmax` at all when
    # CuPy is absent.

    def add(self, a: Any, b: Any) -> Any:
        if self._cp:
            return a + b
        return np.asarray(a) + np.asarray(b)

    def softmax(self, arr: Any, axis: int = -1) -> Any:
        a = arr if isinstance(arr, np.ndarray) or self._cp is not None else np.asarray(arr)
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
        scores = np.matmul(np.asarray(q), np.asarray(k).T) * scale
        return np.matmul(self.softmax(scores, axis=-1), np.asarray(v))

    def conv2d(
        self,
        input: Any,
        weight: Any,
        bias: Any | None,
        stride: int = 1,
        padding: int = 0,
    ) -> Any:
        return _CUDAAccelerator._conv2d_impl(
            np.asarray(input),
            np.asarray(weight),
            np.asarray(bias) if bias is not None else None,
            stride,
            padding,
        )

    @staticmethod
    def _conv2d_impl(
        input: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray | None,
        stride: int,
        padding: int,
    ) -> np.ndarray:
        n, _c, h, w = input.shape
        oc, _ic, kh, kw = weight.shape
        if padding > 0:
            input = np.pad(
                input, [(0, 0), (0, 0), (padding, padding), (padding, padding)], mode="constant"
            )
        out_h = (input.shape[2] - kh) // stride + 1
        out_w = (input.shape[3] - kw) // stride + 1
        result = np.zeros((n, oc, out_h, out_w), dtype=np.float32)
        for i in range(n):
            for oc_idx in range(oc):
                for oh in range(out_h):
                    for ow in range(out_w):
                        patch = input[
                            i, :, oh * stride : oh * stride + kh, ow * stride : ow * stride + kw
                        ]
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
        # astype copies, so the returned buffer never shares memory with the
        # caller's array (test_device_transfer_roundtrip pins both properties).
        return arr.astype(np.float32)

    def from_device(self, arr: Any) -> np.ndarray:
        return np.asarray(arr)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b

    # Restored from the pre-5a0b901b2 implementation (see the Metal class note).
    # These are the reference numpy implementations the other two backends
    # mirror, and the ones test_gpu_accelerator's TestCPUAccelerator pins.

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    def softmax(self, arr: np.ndarray, axis: int = -1) -> np.ndarray:
        exp_arr = np.exp(arr - np.max(arr, axis=axis, keepdims=True))
        return exp_arr / np.sum(exp_arr, axis=axis, keepdims=True)

    def gelu(self, arr: np.ndarray) -> np.ndarray:
        return 0.5 * arr * (1 + np.tanh(np.sqrt(2 / np.pi) * (arr + 0.044715 * arr**3)))

    def silu(self, arr: np.ndarray) -> np.ndarray:
        return arr / (1 + np.exp(-np.clip(arr, -500, 500)))

    def layernorm(
        self, arr: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:
        mean = arr.mean(axis=-1, keepdims=True)
        var = arr.var(axis=-1, keepdims=True)
        return ((arr - mean) / np.sqrt(var + eps)) * weight + bias

    def rmsnorm(self, arr: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        rms = np.sqrt(np.mean(arr**2, axis=-1, keepdims=True) + eps)
        return (arr / rms) * weight

    def attention(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float = 1.0
    ) -> np.ndarray:
        scores = np.matmul(q, k.T) * scale
        return np.matmul(self.softmax(scores, axis=-1), v)

    def scaled_dot_attention(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: np.ndarray | None = None,
        scale: float | None = None,
    ) -> np.ndarray:
        if scale is None:
            scale = 1.0 / np.sqrt(k.shape[-1])
        scores = np.matmul(q, np.swapaxes(k, -2, -1)) * scale
        if mask is not None:
            scores = scores + mask
        return np.matmul(self.softmax(scores, axis=-1), v)

    @staticmethod
    def _conv2d_impl(
        input: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray | None,
        stride: int,
        padding: int,
    ) -> np.ndarray:
        n, _c, h, w = input.shape
        oc, _ic, kh, kw = weight.shape
        if padding > 0:
            input = np.pad(
                input, [(0, 0), (0, 0), (padding, padding), (padding, padding)], mode="constant"
            )
        out_h = (input.shape[2] - kh) // stride + 1
        out_w = (input.shape[3] - kw) // stride + 1
        result = np.zeros((n, oc, out_h, out_w), dtype=np.float32)
        for i in range(n):
            for oc_idx in range(oc):
                for oh in range(out_h):
                    for ow in range(out_w):
                        patch = input[
                            i, :, oh * stride : oh * stride + kh, ow * stride : ow * stride + kw
                        ]
                        result[i, oc_idx, oh, ow] = np.sum(patch * weight[oc_idx]) + (
                            bias[oc_idx] if bias is not None else 0
                        )
        return result

    def conv2d(
        self,
        input: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray | None = None,
        stride: int = 1,
        padding: int = 0,
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
                        result[i, ch, oh, ow] = input[
                            i, ch, ih : ih + kernel_size, iw : iw + kernel_size
                        ].max()
        return result

    def embedding(self, indices: np.ndarray, weight: np.ndarray) -> np.ndarray:
        # Clips, unlike the Metal backend: the CPU contract is to clamp indices
        # into range (test_embedding_2d_indices_clipped).
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

# All three backends expose the same surface (is_available/to_device/from_device/
# matmul plus the restored op set), so callers can treat the resolved backend as
# this union instead of `object` — which is what forced every call site to be
# untypeable.
Accelerator = _MetalAccelerator | _CUDAAccelerator | _CPUAccelerator

_accelerator: Accelerator | None = None


def get_accelerator() -> Accelerator:
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
    """Move a numpy array onto the active accelerator's device."""
    return get_accelerator().to_device(arr)


def from_gpu(arr: Any) -> np.ndarray:
    """Move an accelerator-side array back to host numpy."""
    return get_accelerator().from_device(arr)


def reset_accelerator() -> None:
    """Drop the cached accelerator (testing, or switching backends)."""
    global _accelerator
    _accelerator = None


def cholesky(A: np.ndarray) -> np.ndarray:
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


def solve_triangular(L: np.ndarray, b: np.ndarray, lower: bool = True) -> np.ndarray:
    """Solve L @ x = b for triangular L.

    lower=True  -> forward substitution, L lower triangular
    lower=False -> back substitution,   L upper triangular

    Both branches are required: solve_cholesky() feeds L (lower) in the first
    stage and L.T (upper) in the second, so a forward-only solver silently
    drops the sum term on the second stage and returns y[i] / L[i, i].
    """
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
            x[i] = (
                (b[i] - sum(L[i, j] * x[j] for j in range(i + 1, n))) / L[i, i]
                if L[i, i] != 0
                else 0
            )
    return x


def solve_cholesky(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A @ x = b via Cholesky decomposition."""
    L = cholesky(A)
    y = solve_triangular(L, b, lower=True)
    return solve_triangular(L.T, y, lower=False)


def dominant_eigen(
    A: np.ndarray,
    n_eigen: int = 1,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the n_eigen dominant eigenpairs via power iteration + deflation.

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
