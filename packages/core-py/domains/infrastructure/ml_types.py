"""
Unified ML type system — replaces direct torch imports with numpy-first abstractions.

Every function in this module works WITHOUT torch installed. When torch IS
available, it provides backward-compatible aliases for existing code that
references torch.float32, torch.device, etc.

Design principle: numpy is the source of truth. torch types are thin wrappers
that delegate to numpy equivalents. This lets us benchmark and eventually
replace torch entirely.

Usage:
    from domains.infrastructure.ml_types import dtype, device, tensor, zeros, isnan, isinf

    x = tensor([1.0, 2.0, 3.0], dtype=float32)
    d = device("cpu")  # always "cpu" for numpy, but API-compatible
    assert isnan(tensor([float('nan')])).any()
"""

from __future__ import annotations

import platform
import sys
from typing import Any, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Dtype system
# ---------------------------------------------------------------------------

# Numpy dtypes as the canonical representation
float32 = np.float32
float16 = np.float16
float64 = np.float64
bfloat16 = np.float32  # numpy has no bfloat16 — approximate with float32
int32 = np.int32
int64 = np.int64
int16 = np.int16
int8 = np.int8
uint8 = np.uint8
bool_ = np.bool_
bool = np.bool_  # convenience alias

# Map string names to numpy dtypes (matches torch API)
_DTYPE_MAP = {
    "float32": np.float32,
    "float": np.float32,
    "fp32": np.float32,
    "float16": np.float16,
    "half": np.float16,
    "fp16": np.float16,
    "bfloat16": np.float32,  # approximate
    "bf16": np.float32,
    "float64": np.float64,
    "double": np.float64,
    "fp64": np.float64,
    "int64": np.int64,
    "long": np.int64,
    "int32": np.int32,
    "int": np.int32,
    "int16": np.int16,
    "short": np.int16,
    "int8": np.int8,
    "uint8": np.uint8,
    "bool": np.bool_,
}


def dtype(name_or_value: Any) -> np.dtype:
    """Resolve a dtype from name, numpy dtype, or torch dtype.

    Examples:
        dtype("float32") → np.float32
        dtype(float32) → np.float32
        dtype(np.float16) → np.float16
    """
    if isinstance(name_or_value, np.dtype):
        return name_or_value
    if isinstance(name_or_value, type) and issubclass(name_or_value, np.generic):
        return name_or_value
    if isinstance(name_or_value, str):
        key = name_or_value.lower().replace("torch.", "")
        if key in _DTYPE_MAP:
            return _DTYPE_MAP[key]
        raise ValueError(f"Unknown dtype: {name_or_value}")
    # Torch dtype objects — extract numpy equivalent
    if hasattr(name_or_value, "numpy"):
        # torch.float32.numpy() → np.float32
        try:
            return name_or_value.numpy()
        except Exception:
            pass
    # Try to use as-is (might be a numpy dtype already)
    return np.dtype(name_or_value)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def _mps_available() -> bool:
    """Check if Apple Metal Performance Shaders are available.

    Only returns ``True`` on Apple Silicon (arm64). On Intel Macs (x86_64),
    PyTorch 2.x can report MPS as available via ``torch.backends.mps.is_available()``
    even though it silently crashes during actual inference.
    """
    if sys.platform != "darwin":
        return False
    # Intel Macs: PyTorch may report MPS available but it crashes at runtime.
    if platform.machine() in ("x86_64", "i386"):
        return False
    try:
        import torch
        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        pass
    # Fallback: check on macOS 12+ with Apple Silicon
    if platform.system() == "Darwin":
        machine = platform.machine()
        if machine in ("arm64", "aarch64"):
            return True
    return False


def _cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except (ImportError, AttributeError):
        return False


def auto_device() -> str:
    """Resolve the best available device: mps > cuda > cpu.

    For numpy-only inference, always returns "cpu".
    For torch models, returns the best accelerator.
    """
    if _mps_available():
        return "mps"
    if _cuda_available():
        return "cuda"
    return "cpu"


class device:
    """Device abstraction — API-compatible with torch.device.

    For numpy-only inference, all devices resolve to "cpu".
    """

    def __init__(self, dev: str = "cpu"):
        if isinstance(dev, device):
            self.type = dev.type
            self.index = dev.index
            return
        dev_str = str(dev).lower()
        if ":" in dev_str:
            parts = dev_str.split(":")
            self.type = parts[0]
            self.index = int(parts[1]) if parts[1].isdigit() else None
        else:
            self.type = dev_str
            self.index = None

    @property
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, v: str):
        self._type = v

    def __str__(self) -> str:
        if self.index is not None:
            return f"{self.type}:{self.index}"
        return self.type

    def __repr__(self) -> str:
        return f"device('{self}')"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return str(self) == other.lower()
        if isinstance(other, device):
            return str(self) == str(other)
        return False

    def __hash__(self) -> int:
        return hash(str(self))


# ---------------------------------------------------------------------------
# Tensor creation (numpy-backed)
# ---------------------------------------------------------------------------

def tensor(
    data: Any,
    dtype: Any = np.float32,
    device: Optional[Union[str, device]] = None,
    requires_grad: bool = False,
) -> np.ndarray:
    """Create a numpy array — API-compatible with torch.tensor().

    The `device` and `requires_grad` params are accepted but ignored
    for numpy (always CPU, no autograd).
    """
    if isinstance(data, np.ndarray):
        arr = data
        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype)
        return arr
    arr = np.array(data, dtype=dtype)
    return arr


def zeros(shape: Tuple[int, ...], dtype: Any = np.float32, device: Optional[str] = None) -> np.ndarray:
    """Create zero-filled array — replaces torch.zeros."""
    return np.zeros(shape, dtype=dtype)


def ones(shape: Tuple[int, ...], dtype: Any = np.float32, device: Optional[str] = None) -> np.ndarray:
    """Create ones-filled array — replaces torch.ones."""
    return np.ones(shape, dtype=dtype)


def full(shape: Tuple[int, ...], fill_value: float, dtype: Any = np.float32, device: Optional[str] = None) -> np.ndarray:
    """Create constant-filled array — replaces torch.full."""
    return np.full(shape, fill_value, dtype=dtype)


def full_like(a: np.ndarray, fill_value: float, dtype: Optional[Any] = None) -> np.ndarray:
    """Create constant-filled array matching shape — replaces torch.full_like."""
    dt = dtype or a.dtype
    return np.full_like(a, fill_value, dtype=dt)


def empty(shape: Tuple[int, ...], dtype: Any = np.float32, device: Optional[str] = None) -> np.ndarray:
    """Create uninitialized array — replaces torch.empty."""
    return np.empty(shape, dtype=dtype)


def randn(*shape, dtype: Any = np.float32, device: Optional[str] = None) -> np.ndarray:
    """Create normal-distributed random array — replaces torch.randn.

    Accepts either randn(5, 5) or randn((5, 5)).
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return np.random.randn(*shape).astype(dtype)


def arange(start: int, end: int, step: int = 1, dtype: Any = np.int64) -> np.ndarray:
    """Create range array — replaces torch.arange."""
    return np.arange(start, end, step, dtype=dtype)


def from_numpy(a: np.ndarray) -> np.ndarray:
    """Identity — numpy arrays are already numpy. Replaces torch.from_numpy."""
    return a


# ---------------------------------------------------------------------------
# Inspection ops
# ---------------------------------------------------------------------------

def isnan(a: Any) -> np.ndarray:
    """Check for NaN — replaces torch.isnan."""
    arr = np.asarray(a)
    return np.isnan(arr)


def isinf(a: Any) -> np.ndarray:
    """Check for Inf — replaces torch.isinf."""
    arr = np.asarray(a)
    return np.isinf(arr)


def isfinite(a: Any) -> np.ndarray:
    """Check for finite values — replaces torch.isfinite."""
    arr = np.asarray(a)
    return np.isfinite(arr)


def numel(a: Any) -> int:
    """Count elements — replaces torch.Tensor.numel()."""
    return int(np.asarray(a).size)


def allclose(a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Compare arrays — replaces torch.allclose."""
    return bool(np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol))


def item(a: Any) -> float:
    """Extract scalar — replaces torch.Tensor.item()."""
    return float(np.asarray(a).item())


# ---------------------------------------------------------------------------
# Math ops
# ---------------------------------------------------------------------------

def cat(arrays: list, dim: int = 0) -> np.ndarray:
    """Concatenate — replaces torch.cat."""
    return np.concatenate(arrays, axis=dim)


def stack(arrays: list, dim: int = 0) -> np.ndarray:
    """Stack — replaces torch.stack."""
    return np.stack(arrays, axis=dim)


def where(condition: Any, x: Any, y: Any) -> np.ndarray:
    """Conditional select — replaces torch.where."""
    return np.where(np.asarray(condition), np.asarray(x), np.asarray(y))


def topk(a: Any, k: int, dim: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k values and indices — replaces torch.topk."""
    arr = np.asarray(a)
    axis = dim if dim >= 0 else arr.ndim + dim
    flat = np.swapaxes(arr, axis, -1).reshape(-1, arr.shape[axis])
    top_values = []
    top_indices_local = []
    for row in flat:
        idx = np.argpartition(row, -k)[-k:]
        order = np.argsort(-row[idx])
        top_values.append(row[idx[order]])
        top_indices_local.append(idx[order])
    top_values = np.array(top_values).reshape(*arr.shape[:axis], k, *arr.shape[axis + 1:])
    top_indices = np.array(top_indices_local).reshape(*arr.shape[:axis], k, *arr.shape[axis + 1:])
    return top_values, top_indices


def sort(a: Any, dim: int = -1, descending: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Sort — replaces torch.sort."""
    arr = np.asarray(a)
    if descending:
        indices = np.argsort(-arr, axis=dim)
    else:
        indices = np.argsort(arr, axis=dim)
    values = np.take_along_axis(arr, indices, axis=dim)
    return values, indices


def clamp(a: Any, min=None, max=None, min_val: Optional[float] = None, max_val: Optional[float] = None) -> np.ndarray:
    """Clamp values — replaces torch.clamp.

    Accepts both min/max (torch API) and min_val/max_val (legacy API).
    """
    lo = min if min is not None else min_val
    hi = max if max is not None else max_val
    arr = np.asarray(a)
    if lo is not None:
        arr = np.maximum(arr, lo)
    if hi is not None:
        arr = np.minimum(arr, hi)
    return arr


def multinomial(a: Any, num_samples: int) -> np.ndarray:
    """Sample from distribution — replaces torch.multinomial."""
    probs = np.asarray(a, dtype=np.float64)
    probs = probs / probs.sum()
    return np.random.choice(len(probs), size=num_samples, p=probs)


def softmax(a: Any, dim: int = -1) -> np.ndarray:
    """Softmax — replaces torch.nn.functional.softmax."""
    arr = np.asarray(a, dtype=np.float64)
    arr_max = np.max(arr, axis=dim, keepdims=True)
    exp_arr = np.exp(arr - arr_max)
    return (exp_arr / np.sum(exp_arr, axis=dim, keepdims=True)).astype(a.dtype if hasattr(a, 'dtype') else np.float32)


def matmul(a: Any, b: Any) -> np.ndarray:
    """Matrix multiply — replaces torch.matmul / @."""
    return np.matmul(np.asarray(a), np.asarray(b))


def cosine_similarity(a: Any, b: Any, dim: int = -1) -> np.ndarray:
    """Cosine similarity — replaces torch.nn.functional.cosine_similarity."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    dot = np.sum(a_arr * b_arr, axis=dim)
    norm_a = np.sqrt(np.sum(a_arr ** 2, axis=dim))
    norm_b = np.sqrt(np.sum(b_arr ** 2, axis=dim))
    return (dot / (norm_a * norm_b + 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# Platform stubs (no-ops for numpy)
# ---------------------------------------------------------------------------

class _MpsCache:
    @staticmethod
    def is_available() -> bool:
        """Check if MPS is available."""
        return _mps_available()

    @staticmethod
    def empty_cache():
        """No-op — numpy doesn't allocate GPU memory."""
        pass


class _CudaCache:
    @staticmethod
    def is_available() -> bool:
        """Check if CUDA is available."""
        return _cuda_available()

    @staticmethod
    def empty_cache():
        """No-op — numpy doesn't allocate GPU memory."""
        pass


mps = _MpsCache()
cuda = _CudaCache()


class _Backends:
    mps = _MpsCache()
    cuda = _CudaCache()


backends = _Backends()


def no_grad():
    """Context manager to disable gradient computation.

    For numpy, this is a no-op (numpy has no autograd).
    For torch compatibility, returns a context manager.
    """
    from contextlib import contextmanager

    @contextmanager
    def _no_grad():
        yield

    return _no_grad()
