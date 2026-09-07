"""GPU Acceleration Layer for SloNet

Supports Metal (macOS), CUDA (NVIDIA), and CPU fallback.
All tensors stay as numpy arrays backed by GPU buffers when available.

Usage:
    from domains.training.gpu.accelerator import get_accelerator

    acc = get_accelerator()          # auto-detect best backend
    result = acc.matmul(a, b)        # GPU-accelerated matmul
"""

from __future__ import annotations

import logging
from typing import Any, Optional

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
            from domains.infrastructure.ml_types import _mps_available
            return _mps_available()
        except Exception:
            return False

    def is_available(self) -> bool:
        return self._available

    def to_device(self, arr: np.ndarray) -> np.ndarray:
        return arr

    def from_device(self, arr: Any) -> np.ndarray:
        return np.asarray(arr)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b


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
        return self._cp.asarray(arr)

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


class _CPUAccelerator:
    """CPU fallback — plain numpy."""

    name = "cpu"
    device_type = "cpu"

    def is_available(self) -> bool:
        return True

    def to_device(self, arr: np.ndarray) -> np.ndarray:
        return arr

    def from_device(self, arr: Any) -> np.ndarray:
        return np.asarray(arr)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b


# =============================================================================
# GLOBAL ACCELERATOR
# =============================================================================

_accelerator: Optional[object] = None


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


def solve_triangular(L: object, b: object) -> object:
    """Solve L @ x = b for lower triangular L."""
    import numpy as np
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - sum(L[i, j] * x[j] for j in range(i))) / L[i, i] if L[i, i] != 0 else 0
    return x


def solve_cholesky(A: object, b: object) -> object:
    """Solve A @ x = b via Cholesky decomposition."""
    L = cholesky(A)
    y = solve_triangular(L, b)
    return solve_triangular(L.T, y)


def dominant_eigen(A: object, max_iter: int = 100) -> object:
    """Compute dominant eigenvalue via power iteration."""
    import numpy as np
    n = A.shape[0]
    v = np.ones(n) / np.sqrt(n)
    for _ in range(max_iter):
        w = A @ v
        norm = np.linalg.norm(w)
        if norm == 0:
            break
        v = w / norm
    return float(v @ A @ v)
