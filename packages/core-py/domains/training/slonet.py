"""
SloNet — Slo-native Neural Network Library

A minimal, dependency-light neural network library built on NumPy.
Every layer carries soul metadata. Every model IS a soul.
"""

from __future__ import annotations

import os
import struct
import json
import math
import time
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Callable
from pathlib import Path
import logging

logger = logging.getLogger("man.slonet")

# Lazy GPU acceleration — import on demand, never at module load
_ACCELERATOR = None

# Lazy Numba import
_NUMBA_AVAILABLE = None
def _check_numba():
    global _NUMBA_AVAILABLE
    if _NUMBA_AVAILABLE is None:
        try:
            from numba import njit, prange
            _NUMBA_AVAILABLE = True
        except ImportError:
            _NUMBA_AVAILABLE = False
    return _NUMBA_AVAILABLE

# Numba-accelerated inference kernels (lazy import, graceful fallback)
try:
    from domains.training.slonet_kernels import (
        nb_rmsnorm as _nb_rmsnorm,
        nb_layernorm as _nb_layernorm,
        nb_swiglu as _nb_swiglu,
        nb_softmax as _nb_softmax,
        nb_swi_glu_mul as _nb_swi_glu_mul,
    )
    _KERNELS_AVAILABLE = True
except ImportError:
    _KERNELS_AVAILABLE = False

# Global no_grad mode — skips backward graph construction
_NO_GRAD = False

class no_grad:
    """Context manager / decorator to disable gradient computation.

    All tensors created inside the context have ``requires_grad=False``
    and no ``_children`` / ``_backward_fn`` are set, saving the overhead
    of building the autograd graph.

    Usage::
        with no_grad():
            y = model(x)   # no graph built
    """
    def __enter__(self):
        global _NO_GRAD
        self._prev = _NO_GRAD
        _NO_GRAD = True
        return self

    def __exit__(self, *args):
        global _NO_GRAD
        _NO_GRAD = False


def _broadcast_back(g: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum gradient over broadcast axes to match target shape."""
    if g.ndim > len(shape):
        g = g.sum(axis=tuple(range(g.ndim - len(shape))), keepdims=False)
    for i, d in enumerate(shape):
        if d == 1 and i < g.ndim and g.shape[i] > 1:
            g = np.sum(g, axis=i, keepdims=True)
    return g


def _broadcast_forward(t: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Broadcast tangent from input shape to output shape (inverse of _broadcast_back)."""
    if t.ndim < len(target_shape):
        t = np.expand_dims(t, axis=tuple(range(len(target_shape) - t.ndim)))
    for i in range(t.ndim, len(target_shape)):
        t = np.expand_dims(t, axis=i)
    for i, d in enumerate(t.shape):
        if i < len(target_shape) and target_shape[i] != d:
            t = np.repeat(t, target_shape[i] // d, axis=i) if d > 0 else t
    return t


def _get_accelerator():
    """Get GPU accelerator, lazy-loaded. Tries slolib/gpu first, then old accelerator."""
    global _ACCELERATOR
    if _ACCELERATOR is not None:
        return _ACCELERATOR if _ACCELERATOR != "none" else None
    try:
        from domains.slolib.gpu import get_accelerator as _get_slolib_acc
        acc = _get_slolib_acc()
        if acc is not None and acc.name != "cpu":
            _ACCELERATOR = acc
            return _ACCELERATOR
    except Exception:
        pass
    try:
        from domains.training.gpu.accelerator import get_accelerator as _get_old_acc
        _ACCELERATOR = _get_old_acc()
    except Exception:
        _ACCELERATOR = "none"
        return None
    if _ACCELERATOR is None:
        _ACCELERATOR = "none"
        return None
    return _ACCELERATOR


# Minimum elements to justify accelerator dispatch overhead
_ACCEL_THRESHOLD = 4096


def _accel_op(op_name: str, *args, threshold: int = _ACCEL_THRESHOLD):
    """Run an accelerator operation with numpy fallback.

    Returns the numpy result if the accelerator is available and tensor
    size exceeds the threshold, otherwise runs the numpy fallback.
    The numpy fallback is provided as the last positional argument.
    """
    numpy_fn = args[-1]
    data_args = args[:-1]
    total_elems = 0
    for a in data_args:
        if isinstance(a, np.ndarray):
            total_elems = max(total_elems, a.size)
    if total_elems < threshold:
        return numpy_fn(*data_args)
    acc = _get_accelerator()
    if acc is None:
        return numpy_fn(*data_args)
    try:
        acc_fn = getattr(acc, op_name, None)
        if acc_fn is None:
            return numpy_fn(*data_args)
        device_args = [acc.to_device(a) if isinstance(a, np.ndarray) else a for a in data_args]
        result = acc_fn(*device_args)
        return acc.from_device(result)
    except Exception:
        return numpy_fn(*data_args)


# =============================================================================
# CORE TENSOR
# =============================================================================

class _MetaTensor:
    """Placeholder for meta device tensors (no actual data)."""
    def __init__(self, shape=None):
        self.shape = shape or ()
        self.data = np.empty(0)
        self.requires_grad = False
    def __getattr__(self, name): return self
    def __call__(self, *a, **kw): return self
    def __repr__(self): return f"MetaTensor(shape={self.shape})"
    def numpy(self): return self.data


class Tensor:
    _id_counter = 0

    def __init__(self, data, requires_grad: bool = False, _children: tuple = (), _copy: bool = True):
        if isinstance(data, np.ndarray):
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            elif _copy:
                data = data.copy()
        elif isinstance(data, (list, memoryview)):
            data = np.array(data, dtype=np.float32)
        elif hasattr(data, 'detach'):  # PyTorch tensor
            data = data.detach().cpu().numpy().astype(np.float32)
        else:
            data = np.asarray(data, dtype=np.float32)
        self.data = np.asarray(data)
        self.grad: Optional[Tensor] = None
        if _NO_GRAD:
            self.requires_grad = False
            self._children = ()
        else:
            self.requires_grad = requires_grad
            self._children = _children
        self._backward_fn: Optional[Callable] = None
        self._forward_fn: Optional[Callable] = None  # (tangents...) -> output tangent
        self._consumers: list = []  # forward edges — populated by ops
        self.shape = self.data.shape
        self.id = Tensor._id_counter
        Tensor._id_counter += 1

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape})"

    def __ge__(self, other):
        other_d = other.data if isinstance(other, Tensor) else other
        return Tensor(np.greater_equal(self.data, other_d).astype(np.float32), requires_grad=False)

    def __le__(self, other):
        other_d = other.data if isinstance(other, Tensor) else other
        return Tensor(np.less_equal(self.data, other_d).astype(np.float32), requires_grad=False)

    def __gt__(self, other):
        other_d = other.data if isinstance(other, Tensor) else other
        return Tensor(np.greater(self.data, other_d).astype(np.float32), requires_grad=False)

    def __lt__(self, other):
        other_d = other.data if isinstance(other, Tensor) else other
        return Tensor(np.less(self.data, other_d).astype(np.float32), requires_grad=False)

    def __eq__(self, other):
        other_d = other.data if isinstance(other, Tensor) else other
        return Tensor(np.equal(self.data, other_d).astype(np.float32), requires_grad=False)

    def __ne__(self, other):
        other_d = other.data if isinstance(other, Tensor) else other
        return Tensor(np.not_equal(self.data, other_d).astype(np.float32), requires_grad=False)

    def __bool__(self):
        if self.data.ndim == 0 and self.data.size == 1:
            return bool(self.data.item())
        raise RuntimeError(
            "bool value of Tensor with more than one value is ambiguous"
        )

    def __len__(self):
        if self.data.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.data.shape[0]

    def t(self):
        if self.data.ndim != 2:
            raise RuntimeError("t() expects a 2D tensor")
        return Tensor(self.data.T.copy(), requires_grad=self.requires_grad)

    def all(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(np.array(np.all(self.data), dtype=np.float32), requires_grad=False)
        return Tensor(np.all(self.data, axis=dim, keepdims=keepdim).astype(np.float32), requires_grad=False)

    def any(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(np.array(np.any(self.data), dtype=np.float32), requires_grad=False)
        return Tensor(np.any(self.data, axis=dim, keepdims=keepdim).astype(np.float32), requires_grad=False)

    def tolist(self):
        return self.data.tolist()

    def item(self):
        return float(self.data.flat[0])

    def dim(self) -> int:
        return self.data.ndim

    def numel(self) -> int:
        return int(np.prod(self.shape))

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def squeeze(self, dim=None):
        if dim is None: return Tensor(self.data.squeeze(), requires_grad=False)
        return Tensor(self.data.squeeze(axis=dim), requires_grad=False)

    def unsqueeze(self, dim: int):
        return Tensor(np.expand_dims(self.data, axis=dim), requires_grad=False)

    def repeat(self, *sizes):
        return Tensor(np.tile(self.data, sizes), requires_grad=False)

    def gather(self, dim, index):
        idx = index.data.astype(int)
        result = np.take_along_axis(self.data, idx, axis=dim)
        return Tensor(result, requires_grad=False)

    def scatter_(self, dim, index, src):
        data = self.data.copy()
        if isinstance(src, Tensor): src = src.data
        idx = index.data.astype(int)
        np.put_along_axis(data, idx, src, axis=dim)
        self.data = data
        return self

    def __add__(self, other): return _add(self, _ensure(other))
    def __radd__(self, other): return _add(_ensure(other), self)
    def __sub__(self, other): return _sub(self, _ensure(other))
    def __rsub__(self, other): return _sub(_ensure(other), self)
    def __mul__(self, other): return _mul(self, _ensure(other))
    def __rmul__(self, other): return _mul(_ensure(other), self)
    def __neg__(self): return _neg(self)
    def __pow__(self, p): return _pow(self, p)
    def __truediv__(self, other): return _mul(self, _ensure(other) ** -1)
    def __getitem__(self, key):
        return _slice(self, key)
    def __setitem__(self, key, value):
        self.data[key] = value.data if isinstance(value, Tensor) else value
    def __matmul__(self, other): return _matmul(self, _ensure(other))
    def T(self): return _transpose(self)
    def reshape(self, *s): return _reshape(self, s)
    def sum(self): return _sum(self)
    def mean(self): return _mean(self)
    def max(self): return _max(self)
    def backward(self):
        if self.grad is None:
            self.grad = Tensor(np.ones_like(self.data))
        visited, topo = set(), []
        def build(v):
            if v.id in visited: return
            visited.add(v.id)
            for c in (getattr(v, '_children', None) or ()): build(c)
            topo.append(v)
        build(self)
        for node in reversed(topo):
            g = node.grad.data if node.grad is not None else np.ones_like(node.data)
            node.grad = Tensor(g)
            if node._backward_fn: node._backward_fn(g)

    def forward_grad(self, tangents: dict = None) -> dict:
        """Forward-mode automatic differentiation.

        Given seed tangents on leaf tensors, propagates them forward
        through the computation graph using each op's ``_forward_fn``.

        Args:
            tangents: dict mapping ``tensor_id → np.ndarray`` seed tangents.

        Returns:
            dict mapping ``tensor_id → np.ndarray`` for every reachable node.
        """
        tangents = dict(tangents or {})
        visited, order = set(), []
        def topo(v):
            if v.id in visited: return
            visited.add(v.id)
            for c in (getattr(v, '_children', None) or ()):
                if isinstance(c, Tensor): topo(c)
            order.append(v)
        topo(self)

        for node in order:
            if node.id in tangents:
                continue
            if not node._children or not node._forward_fn:
                continue
            child_t = []
            all_none = True
            for c in node._children:
                if isinstance(c, Tensor) and c.id in tangents:
                    child_t.append(tangents[c.id])
                    all_none = False
                else:
                    child_t.append(None)
            if not all_none:
                tangents[node.id] = node._forward_fn(*child_t)
            elif any(isinstance(c, Tensor) and c.requires_grad for c in node._children):
                # All children have zero tangent — tangent is zero too
                tangents[node.id] = np.zeros_like(node.data)

        return tangents

    def jvp(self, v: 'Tensor') -> 'Tensor':
        """Jacobian-vector product via forward-mode AD.

        Args:
            v: tangent vector (same shape as ``self``).

        Returns:
            Tensor holding ``J @ v`` at the output.
        """
        tangents = {self.id: v.data}
        result = self.forward_grad(tangents)
        out_t = result.get(self.id, np.zeros_like(self.data))
        return Tensor(out_t)

    # --- PyTorch-compat convenience methods ---

    def view(self, *shape):
        return _reshape(self, shape)

    def to(self, device=None, dtype=None, non_blocking=False):
        if device is not None and isinstance(device, (np.dtype, type(np.float32))):
            dtype = device
            device = None
        if dtype is not None:
            if hasattr(dtype, '_np'):
                dtype = dtype._np
            if dtype in (np.float16, np.float32, np.float64, 'float16', 'float32', 'float64'):
                self.data = self.data.astype(np.float32)
            elif dtype in (np.int8, np.int16, np.int32, np.int64, 'int8', 'int16', 'int32', 'int64'):
                self.data = self.data.astype(dtype)
        if device is not None and 'meta' not in str(device):
            pass  # no-op — SloNet is CPU/numpy native
        if hasattr(device, 'type') and device.type == 'meta':
            return _MetaTensor()
        return self

    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False)

    def cpu(self):
        return self

    def numpy(self):
        return self.data

    def float(self):
        if self.data.dtype != np.float32:
            self.data = self.data.astype(np.float32)
        return self

    def long(self):
        return Tensor(self.data.astype(np.int64), requires_grad=self.requires_grad)

    def int(self):
        return Tensor(self.data.astype(np.int32), requires_grad=self.requires_grad)

    def half(self):
        return Tensor(self.data.astype(np.float16), requires_grad=self.requires_grad)

    def double(self):
        return Tensor(self.data.astype(np.float64), requires_grad=self.requires_grad)

    def flatten(self, start_dim=0, end_dim=-1):
        return Tensor(self.data.reshape(-1), requires_grad=False)

    def clone(self):
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    def contiguous(self):
        return self

    def zero_(self):
        self.data.fill(0.0)
        return self

    def fill_(self, val):
        self.data.fill(val)
        return self

    def copy_(self, src):
        if isinstance(src, Tensor): src = src.data
        self.data[:] = src
        return self

    def expand(self, *sizes):
        return Tensor(np.broadcast_to(self.data, sizes), requires_grad=False)

    def transpose(self, dim0, dim1):
        return Tensor(np.swapaxes(self.data, dim0, dim1), requires_grad=self.requires_grad)

    def permute(self, *dims):
        return Tensor(np.transpose(self.data, dims), requires_grad=self.requires_grad)

    def abs(self):
        return Tensor(np.abs(self.data), requires_grad=self.requires_grad)

    def sqrt(self):
        return Tensor(np.sqrt(np.maximum(self.data, 0)), requires_grad=self.requires_grad)

    def clamp(self, min_val=None, max_val=None):
        return Tensor(np.clip(self.data, min_val, max_val), requires_grad=self.requires_grad)

    def argsort(self, dim=-1, descending=False):
        if descending:
            return Tensor(np.argsort(-self.data, axis=dim).astype(np.int64), requires_grad=False)
        return Tensor(np.argsort(self.data, axis=dim).astype(np.int64), requires_grad=False)

    def eq(self, other):
        o = other.data if isinstance(other, Tensor) else other
        return Tensor((self.data == o).astype(np.float32), requires_grad=False)

    def ne(self, other):
        o = other.data if isinstance(other, Tensor) else other
        return Tensor((self.data != o).astype(np.float32), requires_grad=False)

    def gt(self, other):
        o = other.data if isinstance(other, Tensor) else other
        return Tensor((self.data > o).astype(np.float32), requires_grad=False)

    def lt(self, other):
        o = other.data if isinstance(other, Tensor) else other
        return Tensor((self.data < o).astype(np.float32), requires_grad=False)

    def ge(self, other):
        o = other.data if isinstance(other, Tensor) else other
        return Tensor((self.data >= o).astype(np.float32), requires_grad=False)

    def le(self, other):
        o = other.data if isinstance(other, Tensor) else other
        return Tensor((self.data <= o).astype(np.float32), requires_grad=False)

    def argmax(self, dim=None):
        return Tensor(np.argmax(self.data, axis=dim).astype(np.int64), requires_grad=False)

    def argmin(self, dim=None):
        return Tensor(np.argmin(self.data, axis=dim).astype(np.int64), requires_grad=False)

    def topk(self, k, dim=-1):
        return topk(self, k)

    def softmax(self, dim=-1):
        return softmax(self, dim)

    def log_softmax(self, dim=-1):
        return log_softmax(self, dim)

    def __len__(self):
        return self.shape[0] if self.shape else 0

    def type(self, dtype):
        dtype_map = {'torch.FloatTensor': np.float32, 'torch.LongTensor': np.int64,
                     'torch.IntTensor': np.int32, 'torch.HalfTensor': np.float16,
                     'torch.DoubleTensor': np.float64}
        target = dtype_map.get(str(dtype), dtype)
        if isinstance(target, np.dtype) or (isinstance(target, type) and target in (np.float32, np.float64, np.int64, np.int32, np.float16)):
            self.data = self.data.astype(target)
        return self

    def requires_grad_(self, req=True):
        self.requires_grad = req
        return self

def _ensure(x): return x if isinstance(x, Tensor) else Tensor(x)


def _add(a, b):
    out = Tensor(_accel_op("add", a.data, b.data, lambda x,y: x + y), requires_grad=a.requires_grad or b.requires_grad, _children=(a, b), _copy=False)
    _a_shape = a.shape; _b_shape = b.shape; _out_shape = out.shape
    if out.requires_grad:
        if a.requires_grad: a._consumers.append(out)
        if b.requires_grad: b._consumers.append(out)
    def bk(g):
        if a.requires_grad:
            ga = _broadcast_back(g, _a_shape)
            a.grad = Tensor(ga if a.grad is None else a.grad.data + ga)
        if b.requires_grad:
            gb = _broadcast_back(g, _b_shape)
            b.grad = Tensor(gb if b.grad is None else b.grad.data + gb)
    out._backward_fn = bk
    def fwd(t_a, t_b):
        t_a = np.zeros_like(a.data) if t_a is None else _broadcast_forward(t_a, _out_shape)
        t_b = np.zeros_like(b.data) if t_b is None else _broadcast_forward(t_b, _out_shape)
        return t_a + t_b
    out._forward_fn = fwd
    return out


def _neg(a):
    out = Tensor(_accel_op("neg", a.data, lambda x: -x), requires_grad=a.requires_grad, _children=(a,), _copy=False)
    if out.requires_grad and a.requires_grad: a._consumers.append(out)
    def bk(g):
        if a.requires_grad: a.grad = Tensor(-g if a.grad is None else a.grad.data - g)
    out._backward_fn = bk
    def fwd(t_a):
        if t_a is None: return np.zeros_like(a.data)
        return -t_a
    out._forward_fn = fwd
    return out


def _sub(a, b): return _add(a, _neg(b))


def _mul(a, b):
    out = Tensor(_accel_op("mul", a.data, b.data, lambda x,y: x * y), requires_grad=a.requires_grad or b.requires_grad, _children=(a, b), _copy=False)
    _a_shape = a.shape; _b_shape = b.shape; _out_shape = out.shape
    if out.requires_grad:
        if a.requires_grad: a._consumers.append(out)
        if b.requires_grad: b._consumers.append(out)
    def bk(g):
        ga = g * b.data
        gb = g * a.data
        for i, d in enumerate(_a_shape):
            if d == 1 and i < g.ndim and g.shape[i] > 1: ga = np.sum(ga, axis=i, keepdims=True)
        for i, d in enumerate(_b_shape):
            if d == 1 and i < g.ndim and g.shape[i] > 1: gb = np.sum(gb, axis=i, keepdims=True)
        if a.requires_grad: a.grad = Tensor(ga if a.grad is None else a.grad.data + ga)
        if b.requires_grad: b.grad = Tensor(gb if b.grad is None else b.grad.data + gb)
    out._backward_fn = bk
    def fwd(t_a, t_b):
        t_a = np.zeros_like(a.data) if t_a is None else t_a
        t_b = np.zeros_like(b.data) if t_b is None else t_b
        return t_a * b.data + a.data * t_b
    out._forward_fn = fwd
    return out


def _pow(a, p):
    out = Tensor(_accel_op("pow", a.data, p, lambda x, pp: x ** pp), requires_grad=a.requires_grad, _children=(a,), _copy=False)
    if out.requires_grad and a.requires_grad: a._consumers.append(out)
    def bk(g):
        if a.requires_grad: a.grad = Tensor(p*(a.data**(p-1))*g)
    out._backward_fn = bk
    def fwd(t_a, _=None):
        if t_a is None: return np.zeros_like(a.data)
        return p * (a.data ** (p - 1)) * t_a
    out._forward_fn = fwd
    return out


def _matmul(a, b):
    a_data = a.data if isinstance(a, Tensor) else np.asarray(a)
    b_data = b.data if isinstance(b, Tensor) else np.asarray(b)
    a_req = isinstance(a, Tensor) and a.requires_grad
    b_req = isinstance(b, Tensor) and b.requires_grad
    children = (_ensure(a), _ensure(b)) if not (isinstance(a, Tensor) and isinstance(b, Tensor)) else (a, b)
    # Skip accelerator for tiny matmuls (Metal dispatch overhead dominates)
    _use_acc = False
    out_elems = a_data.shape[-2] * b_data.shape[-1] if a_data.ndim >= 2 else 1
    if out_elems >= 16384:
        acc = _get_accelerator()
        _use_acc = acc is not None and acc.name != "cpu"
    if _use_acc:
        try:
            result = acc.matmul(a_data, b_data)
        except Exception:
            result = np.matmul(a_data, b_data)
    else:
        result = np.matmul(a_data, b_data)
    out = Tensor(result, requires_grad=a_req or b_req, _children=children, _copy=False)
    if out.requires_grad:
        if isinstance(a, Tensor) and a.requires_grad: a._consumers.append(out)
        if isinstance(b, Tensor) and b.requires_grad: b._consumers.append(out)
    _a_shape = a_data.shape; _b_shape = b_data.shape; _out_shape = out.data.shape
    def bk(g):
        ga = None; gb = None
        if a_req or (isinstance(a, Tensor) and a._backward_fn):
            g_out = g.reshape(_out_shape) if g.shape != _out_shape else g
            if b_data.ndim == 1:
                ga = np.outer(g_out, b_data)
            else:
                ga = np.matmul(g_out, b_data.reshape(-1, b_data.shape[-1]).T)
            ga = ga.reshape(_a_shape) if ga.shape != _a_shape else ga
            if a_req:
                if a.grad is None: a.grad = Tensor(ga)
                else: a.grad.data[:] += ga
        if b_req or (isinstance(b, Tensor) and b._backward_fn):
            g_out = g.reshape(_out_shape) if g.shape != _out_shape else g
            if a_data.ndim == 1:
                # a is (n,), b is (n, m) → gb is outer(a, g) = (n, m)
                gb = np.outer(a_data, g_out)
            elif g_out.ndim == 1:
                # g_out = (n,), a_flat = (n, k) → gb = a_flat.T @ g_out = (k,)
                a_flat = a_data.reshape(-1, a_data.shape[-1])
                gb = np.matmul(a_flat.T, g_out)
            else:
                a_flat = a_data.reshape(-1, a_data.shape[-1])
                gb = np.matmul(a_flat.T, g_out.reshape(-1, g_out.shape[-1]))
            gb = gb.reshape(_b_shape) if gb.shape != _b_shape else gb
            if b_req:
                if b.grad is None: b.grad = Tensor(gb)
                else: b.grad.data[:] += gb
    out._backward_fn = bk
    def fwd(t_a, t_b):
        t_a = np.zeros_like(a_data) if t_a is None else t_a
        t_b = np.zeros_like(b_data) if t_b is None else t_b
        return np.matmul(t_a, b_data) + np.matmul(a_data, t_b)
    out._forward_fn = fwd
    return out


def _transpose(a):
    out = Tensor(a.data.T, requires_grad=a.requires_grad, _children=(a,), _copy=False)
    _ndim = a.data.ndim
    if out.requires_grad and a.requires_grad: a._consumers.append(out)
    def bk(g):
        if a.requires_grad:
            tg = g.T if _ndim == 2 else np.transpose(g, list(range(_ndim-2)) + [_ndim-1, _ndim-2])
            a.grad = Tensor(tg if a.grad is None else a.grad.data + tg)
    out._backward_fn = bk
    def fwd(t_a):
        if t_a is None: return np.zeros_like(a.data).T
        return t_a.T if _ndim == 2 else np.transpose(t_a, list(range(_ndim-2)) + [_ndim-1, _ndim-2])
    out._forward_fn = fwd
    return out


def _reshape(a, s):
    out = Tensor(a.data.reshape(s), requires_grad=a.requires_grad, _children=(a,), _copy=False)
    if out.requires_grad and a.requires_grad: a._consumers.append(out)
    def bk(g):
        if a.requires_grad:
            ga = g.reshape(a.shape)
            a.grad = Tensor(ga)
    out._backward_fn = bk
    def fwd(t_a):
        if t_a is None: return np.zeros(s, dtype=np.float32)
        return t_a.reshape(s)
    out._forward_fn = fwd
    return out


def _slice(a, key):
    key = key if isinstance(key, tuple) else (key,)
    out = Tensor(a.data[key], requires_grad=a.requires_grad, _children=(a,), _copy=False)
    if out.requires_grad and a.requires_grad: a._consumers.append(out)
    def bk(g):
        if a.requires_grad:
            full = np.zeros(a.shape, dtype=np.float32)
            np.add.at(full, key, g)
            ga = full if a.grad is None else a.grad.data + full
            a.grad = Tensor(ga)
    out._backward_fn = bk
    def fwd(t_a):
        if t_a is None: return np.zeros(out.shape, dtype=np.float32)
        return np.array(t_a[key], dtype=np.float32)
    out._forward_fn = fwd
    return out


def _sum(a):
    out = Tensor(_accel_op("sum", a.data, lambda x: x.sum()), requires_grad=a.requires_grad, _children=(a,), _copy=False)
    if out.requires_grad and a.requires_grad: a._consumers.append(out)
    def bk(g):
        if a.requires_grad: a.grad = Tensor(np.full_like(a.data, g))
    out._backward_fn = bk
    def fwd(t_a):
        if t_a is None: return np.array(0.0, dtype=np.float32)
        return np.array(t_a.sum(), dtype=np.float32)
    out._forward_fn = fwd
    return out


def _mean(a):
    out = Tensor(_accel_op("mean", a.data, lambda x: x.mean()), requires_grad=a.requires_grad, _children=(a,), _copy=False)
    if out.requires_grad and a.requires_grad: a._consumers.append(out)
    n = a.data.size
    def bk(g):
        if a.requires_grad: a.grad = Tensor(np.full_like(a.data, g/n))
    out._backward_fn = bk
    def fwd(t_a):
        if t_a is None: return np.array(0.0, dtype=np.float32)
        return np.array(t_a.mean(), dtype=np.float32)
    out._forward_fn = fwd
    return out


def _max(a):
    out = Tensor(a.data.max(), requires_grad=a.requires_grad, _children=(a,), _copy=False)
    if out.requires_grad and a.requires_grad: a._consumers.append(out)
    def bk(g):
        if a.requires_grad:
            mask = a.data == a.data.max()
            a.grad = Tensor(np.where(mask, g, 0.0), _copy=False)
    out._backward_fn = bk
    def fwd(t_a):
        if t_a is None: return np.array(0.0, dtype=np.float32)
        mask = a.data == a.data.max()
        return np.array(t_a[mask].sum() if mask.any() else 0.0, dtype=np.float32)
    out._forward_fn = fwd
    return out


def zeros(s, requires_grad=False): return Tensor(np.zeros(s, dtype=np.float32), requires_grad=requires_grad, _copy=False)
def randn(s, requires_grad=False): return Tensor(np.random.randn(*s).astype(np.float32), requires_grad=requires_grad, _copy=False)
def ones(s, requires_grad=False): return Tensor(np.ones(s, dtype=np.float32), requires_grad=requires_grad, _copy=False)
def tensor(d, requires_grad=False): return Tensor(d, requires_grad=requires_grad, _copy=False)


# =============================================================================
# ACTIVATIONS
# =============================================================================

def sigmoid(x):
    s = _accel_op("sigmoid", x.data, lambda d: 1.0/(1.0+np.exp(-np.clip(d, -500, 500))))
    out = Tensor(s, requires_grad=x.requires_grad, _children=(x,), _copy=False)
    if out.requires_grad and x.requires_grad: x._consumers.append(out)
    def bk(g):
        if x.requires_grad:
            gs = s*(1-s)*g
            x.grad = Tensor(gs, _copy=False)
    out._backward_fn = bk
    def fwd(t_x):
        if t_x is None: return np.zeros_like(s)
        return s * (1 - s) * t_x
    out._forward_fn = fwd
    return out


def tanh(x):
    t = _accel_op("tanh", x.data, lambda d: np.tanh(d))
    out = Tensor(t, requires_grad=x.requires_grad, _children=(x,), _copy=False)
    if out.requires_grad and x.requires_grad: x._consumers.append(out)
    def bk(g):
        if x.requires_grad:
            gt = (1-t*t)*g
            x.grad = Tensor(gt, _copy=False)
    out._backward_fn = bk
    def fwd(t_x):
        if t_x is None: return np.zeros_like(t)
        return (1 - t * t) * t_x
    out._forward_fn = fwd
    return out


def relu(x):
    out = Tensor(_accel_op("relu", x.data, lambda d: np.maximum(d, 0)), requires_grad=x.requires_grad, _children=(x,), _copy=False)
    if out.requires_grad and x.requires_grad: x._consumers.append(out)
    def bk(g):
        if x.requires_grad:
            gr = np.where(x.data>0, g, 0.0)
            x.grad = Tensor(gr)
    out._backward_fn = bk
    def fwd(t_x):
        if t_x is None: return np.zeros_like(out.data)
        return np.where(x.data > 0, t_x, 0.0)
    out._forward_fn = fwd
    return out


def gelu_np(d: np.ndarray) -> np.ndarray:
    """NumPy-only GELU (no Tensor wrapping)."""
    return 0.5 * d * (1 + np.tanh(np.sqrt(2 / np.pi) * (d + 0.044715 * d**3)))


def gelu(x):
    d = x.data if isinstance(x, Tensor) else x
    acc = _get_accelerator()
    if acc is not None and acc.name != "cpu":
        try:
            t = acc.gelu(d)
        except Exception:
            t = 0.5 * d * (1 + np.tanh(np.sqrt(2/np.pi) * (d + 0.044715 * d**3)))
    else:
        t = 0.5 * d * (1 + np.tanh(np.sqrt(2/np.pi) * (d + 0.044715 * d**3)))
    if isinstance(x, Tensor):
        out = Tensor(t, requires_grad=x.requires_grad, _children=(x,), _copy=False)
        if out.requires_grad and x.requires_grad: x._consumers.append(out)
        def bk(g):
            if x.requires_grad:
                d_gelu = 0.5 * np.tanh(np.sqrt(2/np.pi) * (d + 0.044715 * d**3)) + \
                         0.5 * d * (1 - np.tanh(np.sqrt(2/np.pi) * (d + 0.044715 * d**3))**2) * \
                         np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * d**2)
                x.grad = Tensor(d_gelu * g if x.grad is None else x.grad.data + d_gelu * g, _copy=False)
        out._backward_fn = bk
        def fwd(t_x):
            if t_x is None: return np.zeros_like(out.data)
            d_gelu = 0.5 * np.tanh(np.sqrt(2/np.pi) * (d + 0.044715 * d**3)) + \
                     0.5 * d * (1 - np.tanh(np.sqrt(2/np.pi) * (d + 0.044715 * d**3))**2) * \
                     np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * d**2)
            return d_gelu * t_x
        out._forward_fn = fwd
        return out
    return t


def silu_np(d: np.ndarray) -> np.ndarray:
    """NumPy-only SiLU (no Tensor wrapping)."""
    return d * (1 / (1 + np.exp(-d)))


def silu(x):
    d = x.data if isinstance(x, Tensor) else x
    s = 1 / (1 + np.exp(-d))
    t = d * s
    acc = _get_accelerator()
    if acc is not None and acc.name != "cpu":
        try:
            t = acc.silu(d)
        except Exception:
            pass
    if isinstance(x, Tensor):
        out = Tensor(t, requires_grad=x.requires_grad, _children=(x,), _copy=False)
        if out.requires_grad and x.requires_grad: x._consumers.append(out)
        def bk(g):
            if x.requires_grad:
                d_silu = s + d * s * (1 - s)
                x.grad = Tensor(d_silu * g if x.grad is None else x.grad.data + d_silu * g, _copy=False)
        out._backward_fn = bk
        def fwd(t_x):
            if t_x is None: return np.zeros_like(out.data)
            d_silu = s + d * s * (1 - s)
            return d_silu * t_x
        out._forward_fn = fwd
        return out
    return t


def softmax(x, dim=-1):
    if isinstance(x, Tensor):
        acc = _get_accelerator()
        if acc is not None and acc.name != "cpu":
            try:
                result = acc.softmax(x.data, axis=dim)
                out = Tensor(result, requires_grad=x.requires_grad, _children=(x,), _copy=False)
                out._backward_fn = lambda g: None
                return out
            except Exception:
                pass
        return _softmax(x, dim)
    d = x - x.max(axis=dim, keepdims=True)
    return np.exp(d) / np.exp(d).sum(axis=dim, keepdims=True)


def cross_entropy(logits, targets):
    orig_shape = logits.data.shape
    ndim = logits.data.ndim
    if ndim > 2:
        logits_2d = logits.data.reshape(-1, orig_shape[-1])
    else:
        logits_2d = logits.data
    mx = logits_2d.max(axis=-1, keepdims=True)
    diff = logits_2d - mx
    lp = diff - np.log(np.exp(diff).sum(axis=-1, keepdims=True))
    n = lp.shape[0]
    t = targets.data.astype(int).flatten()
    t = np.clip(t, 0, lp.shape[-1] - 1)
    loss = -lp[np.arange(n), t].mean()
    out = Tensor(loss, requires_grad=True, _children=(logits, targets))
    if logits.requires_grad: logits._consumers.append(out)
    _probs = np.exp(lp)
    def bk(g):
        probs = _probs.copy()
        probs[np.arange(n), t] -= 1
        probs /= n
        if ndim > 2:
            logits.grad = Tensor(probs.reshape(orig_shape) * g)
        else:
            logits.grad = Tensor(probs * g)
    out._backward_fn = bk
    def fwd(t_logits, _=None):
        if t_logits is None: return np.array(0.0, dtype=np.float32)
        # JVP of cross_entropy w.r.t. logits
        if ndim > 2:
            t_logits_2d = t_logits.reshape(-1, orig_shape[-1])
        else:
            t_logits_2d = t_logits
        grad = _probs.copy()
        grad[np.arange(n), t] -= 1
        grad /= n
        result = (grad * t_logits_2d).sum()
        return np.array(result, dtype=np.float32)
    out._forward_fn = fwd
    return out


def mse_loss(pred, target):
    return _mean(_mul(pred-target, pred-target))


def _to_np(x):
    if isinstance(x, Tensor): return x.data
    return np.array(x)


def topk(x: Tensor, k: int):
    data = _to_np(x)
    flat = data.reshape(-1)
    indices = flat.argsort()[-k:][::-1]
    values = flat[indices]
    return Tensor(values.reshape(1, k), requires_grad=False), Tensor(indices.reshape(1, k), requires_grad=False)


def multinomial(x: Tensor, num_samples: int):
    data = _to_np(x)
    flat = data.reshape(-1)
    flat = np.maximum(flat, 0)
    total = flat.sum()
    if total > 0: flat = flat / total
    else: flat = np.ones_like(flat) / flat.size
    indices = np.random.choice(len(flat), size=num_samples, p=flat, replace=False)
    return Tensor(indices.reshape(1, num_samples), requires_grad=False)


def stack(tensors, dim=0):
    arrays = [_to_np(t) for t in tensors]
    stacked = np.stack(arrays, axis=dim)
    return Tensor(stacked, requires_grad=False)


def concatenate(tensors, dim=-1):
    arrays = [_to_np(t) for t in tensors]
    if dim == -1: dim = len(arrays[0].shape) - 1
    concat = np.concatenate(arrays, axis=dim)
    return Tensor(concat, requires_grad=False)


def randint(low, high, shape):
    arr = np.random.randint(low, high, shape).astype(np.int64)
    return Tensor(arr, requires_grad=False)


def exp(x: Tensor):
    data = _to_np(x)
    out = np.exp(np.clip(data, -700, 700))
    return Tensor(out, requires_grad=False)


def isfinite(x: Tensor):
    return np.isfinite(_to_np(x))


def where(condition, a, b):
    c = _to_np(condition)
    aa = _to_np(a)
    bb = _to_np(b)
    return Tensor(np.where(c, aa, bb), requires_grad=False)


class _NoGrad:
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper


def no_grad():
    return _NoGrad()


def is_cuda(x: Tensor) -> bool:
    return False


def is_mps(x: Tensor) -> bool:
    return False


def cuda():
    return None


def cpu(x: Tensor) -> Tensor:
    return x


# =============================================================================
# LOGIT PROCESSORS — composable pipeline for quality generation
# =============================================================================

def _apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by temperature. Temp→0 ≈ greedy, temp=1 ≈ identity, temp>1 ≈ more random."""
    if temperature < 1e-6:
        return logits
    return logits / temperature


def _apply_top_k(logits: np.ndarray, k: int) -> np.ndarray:
    """Zero-out all logits except the top-k. Preserves relative order."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    cutoff = np.partition(logits, -k, axis=-1)[:, -k:].min(axis=-1, keepdims=True)
    return np.where(logits >= cutoff, logits, -1e9)


def _apply_top_p(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus sampling — keep smallest set of tokens whose cumulative prob >= p."""
    if p <= 0.0 or p >= 1.0:
        return logits
    sorted_idx = np.argsort(-logits, axis=-1)
    sorted_logits = np.take_along_axis(logits, sorted_idx, axis=-1)
    probs = np.exp(sorted_logits - sorted_logits.max(axis=-1, keepdims=True))
    probs = probs / (probs.sum(axis=-1, keepdims=True) + 1e-10)
    cumsum = np.cumsum(probs, axis=-1)
    remove = cumsum > p
    remove[:, 1:] = remove[:, :-1]
    remove[:, 0] = False  # always keep at least the top token
    for b in range(logits.shape[0]):
        logits[b, sorted_idx[b, remove[b]]] = -1e9
    return logits


def _apply_repetition_penalty(logits: np.ndarray, generated_ids: np.ndarray,
                              penalty: float) -> np.ndarray:
    """Penalize tokens that have already been generated. penalty>1 discourages repeats."""
    if abs(penalty - 1.0) < 1e-6 or len(generated_ids) == 0:
        return logits
    for tok in set(generated_ids):
        if tok >= logits.shape[-1]:
            continue
        if logits[0, tok] < 0:
            logits[0, tok] *= penalty
        else:
            logits[0, tok] /= penalty
    return logits


def _apply_frequency_penalty(logits: np.ndarray, generated_ids: np.ndarray,
                             penalty: float) -> np.ndarray:
    """Scale logits proportionally to token frequency. penalty>0 reduces repeats."""
    if abs(penalty) < 1e-6 or len(generated_ids) == 0:
        return logits
    from collections import Counter
    freq = Counter(generated_ids)
    for tok, count in freq.items():
        if tok >= logits.shape[-1]:
            continue
        logits[0, tok] -= penalty * count
    return logits


def _apply_presence_penalty(logits: np.ndarray, generated_ids: np.ndarray,
                            penalty: float) -> np.ndarray:
    """Penalize any token that has appeared at least once."""
    if abs(penalty) < 1e-6 or len(generated_ids) == 0:
        return logits
    for tok in set(generated_ids):
        if tok >= logits.shape[-1]:
            continue
        logits[0, tok] -= penalty
    return logits


def _sample_from_logits(logits: np.ndarray, temperature: float = 1.0,
                        top_k: Optional[int] = None, top_p: Optional[float] = None,
                        repetition_penalty: float = 1.0,
                        frequency_penalty: float = 0.0,
                        presence_penalty: float = 0.0,
                        generated_ids: Optional[np.ndarray] = None,
                        eos_token: Optional[int] = None) -> int:
    """Full logit processing pipeline: penalties → filtering → temperature → sample.

    Args:
        logits: Raw logits shape (1, vocab_size)
        temperature: Sampling temperature (>0). Low = greedy, high = random.
        top_k: Keep only top-k logits before sampling.
        top_p: Nucleus threshold — keep tokens with cumulative prob <= p.
        repetition_penalty: Scale factor for repeated tokens (>1 = penalize).
        frequency_penalty: Subtract penalty * count for each repeated token.
        presence_penalty: Subtract penalty for any token that has appeared.
        generated_ids: Array of already-generated token IDs.
        eos_token: If set, mask it during sampling (except for required termination).

    Returns:
        Sampled token ID.
    """
    if logits.ndim == 3:
        logits = logits[:, 0, :]
    logits = logits.copy()
    if temperature > 1e-6:
        logits = _apply_temperature(logits, temperature)
    logits = np.where(np.isfinite(logits), logits, -1e9)

    gen_ids = generated_ids.flatten() if generated_ids is not None else np.array([], dtype=np.int64)

    if repetition_penalty != 1.0:
        logits = _apply_repetition_penalty(logits, gen_ids, repetition_penalty)
    if frequency_penalty != 0.0:
        logits = _apply_frequency_penalty(logits, gen_ids, frequency_penalty)
    if presence_penalty != 0.0:
        logits = _apply_presence_penalty(logits, gen_ids, presence_penalty)

    # Mask EOS during generation so it only fires at appropriate times
    if eos_token is not None and eos_token < logits.shape[-1]:
        logits[0, eos_token] = -1e9

    if top_k is not None:
        logits = _apply_top_k(logits, top_k)
    if top_p is not None and top_p < 1.0:
        logits = _apply_top_p(logits, top_p)

    # Fast path: greedy (temp≈0, no penalties, no top_p) → argmax
    if temperature < 1e-6 and top_p is None:
        return int(np.argmax(logits[0]))

    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / (probs.sum(axis=-1, keepdims=True) + 1e-10)
    probs = np.where(np.isfinite(probs), probs, np.zeros_like(probs))
    probs = probs / (probs.sum(axis=-1, keepdims=True) + 1e-10)

    return int(np.random.choice(logits.shape[-1], p=probs[0]))


# =============================================================================
# SOUL LAYERS
# =============================================================================

class SloLayer:
    def __init__(self, name=""):
        self.name = name or self.__class__.__name__
        self.soul_traits: Dict[str, float] = {}

    def parameters(self) -> List[Tensor]: return []

    def train(self, mode: bool = True):
        """Set training mode. Subclasses with dropout override this."""
        pass

    def eval(self):
        """Set evaluation mode (disables dropout)."""
        self.train(False)

    def soul_signature(self) -> Dict: return {"layer": self.__class__.__name__, "name": self.name, "soul_traits": self.soul_traits}

    def __call__(self, x) -> Tensor:
        return self.forward(x)

    def named_children(self) -> List[Tuple[str, "SloLayer"]]:
        return []

    def named_modules(self, prefix="") -> List[Tuple[str, "SloLayer"]]:
        return [(prefix, self)]


class SloLinear(SloLayer):
    def __init__(self, in_f, out_f, name="", bias=True):
        super().__init__(name or f"Lin_{in_f}x{out_f}")
        s = math.sqrt(2.0/(in_f+out_f))
        self.weight = randn((out_f, in_f), requires_grad=True); self.weight.data *= s
        self.use_bias = bias
        if bias:
            self.bias = zeros((out_f,), requires_grad=True)
        self.out_features = out_f; self.in_features = in_f
        self._weight_T = None
        self.soul_traits = {"creativity": 0.5, "confidence": 0.5, "warmth": 0.5}
        self._quant_info = None  # TensorInfo for quantized weight
        self._quant_unpacked = None  # lazy int4→int8 unpack cache
        self._point_weight = None  # PointWeight: function-based weight representation

    def _get_weight_T(self) -> Tensor:
        if self._weight_T is None:
            self._weight_T = self.weight.T()
        return self._weight_T

    def set_quantized_weight(self, quant_info):
        """Set a quantized weight for this layer.

        When set, forward() uses int8 GEMM instead of float32 matmul.
        The float32 weight is kept for gradient computation (training).

        For int4 (bits=4), the packed 1D array stays packed for memory
        efficiency. On first forward() call, it is lazily unpacked into
        a cached 2D int8 matrix.
        """
        self._quant_info = quant_info
        self._quant_unpacked = None  # lazy unpack cache for int4

    def _get_quant_array(self) -> np.ndarray:
        """Return the quantized weight array as 2D int8 matrix.

        For int4, lazily unpacks the packed array and caches it.
        """
        if self._quant_info is None or not self._quant_info.is_quantized:
            return None
        if self._quant_info.meta.bits == 4:
            if self._quant_unpacked is None:
                from domains.infrastructure.quantization import _unpack_int4
                signed = self._quant_info.meta.mode == "symmetric"
                n_total = int(np.prod(self._quant_info.meta.original_shape))
                unpacked_1d = _unpack_int4(self._quant_info.array, n_total, signed=signed)
                self._quant_unpacked = unpacked_1d.reshape(self._quant_info.meta.original_shape).astype(np.int8)
            return self._quant_unpacked
        return self._quant_info.array

    def set_point_weight(self, point_weight):
        """Set a PointWeight for this layer.

        When set, forward() generates weight from the Point on-the-fly
        instead of using the raw Tensor weight.

        Args:
            point_weight: PointWeight instance (from pugqeep.point_weight)
        """
        self._point_weight = point_weight
        # Sync generated data into self.weight so training/gradients still work
        arr = point_weight.generate()
        self.weight.data = arr.astype(np.float32)

    def get_point_weight(self):
        """Return the PointWeight for this layer, or None."""
        return self._point_weight

    def compress_to_point(self, method: str = "auto", n_clusters: int = 16):
        """Compress this layer's weight tensor to a PointWeight.

        After compression, the Point replaces raw storage. The generated
        data is synced into self.weight for backward compatibility.
        """
        from domains.infrastructure.pugqeep.point_weight import PointWeight
        pw = PointWeight.from_array(
            self.weight.data,
            identity=self.name,
            method=method,
            n_clusters=n_clusters,
        )
        self.set_point_weight(pw)
        return pw

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        if self._quant_info is not None and self._quant_info.is_quantized:
            from domains.infrastructure.quantization import (
                quantized_linear, int4_quantized_linear,
            )
            bias_arr = self.bias.data if self.use_bias else None
            bits = self._quant_info.meta.bits
            if bits == 4:
                K = self._quant_info.meta.original_shape[-1]
                return int4_quantized_linear(
                    x, self._quant_info.array,
                    self._quant_info.meta.scale,
                    self._quant_info.meta.zero_point,
                    K, bias_arr,
                )
            return quantized_linear(
                x, self._get_quant_array(), self._quant_info.meta.scale,
                self._quant_info.meta.zero_point, bias_arr,
            )
        return x @ self.weight.data.T + self.bias.data

    def forward(self, x: Tensor) -> Tensor:
        if self._quant_info is not None and self._quant_info.is_quantized:
            from domains.infrastructure.quantization import (
                quantized_linear, int4_quantized_linear,
            )
            bias_arr = self.bias.data if self.use_bias else None
            bits = self._quant_info.meta.bits
            if bits == 4:
                K = self._quant_info.meta.original_shape[-1]
                result = int4_quantized_linear(
                    x.data, self._quant_info.array,
                    self._quant_info.meta.scale,
                    self._quant_info.meta.zero_point,
                    K, bias_arr,
                )
            else:
                result = quantized_linear(
                    x.data, self._get_quant_array(), self._quant_info.meta.scale,
                    self._quant_info.meta.zero_point, bias_arr,
                )
            return Tensor(result, requires_grad=x.requires_grad, _children=(x,))
        out = _matmul(x, self._get_weight_T())
        if self.use_bias:
            out = out + self.bias
        return out

    def parameters(self) -> List[Tensor]:
        params = [self.weight]
        if self.use_bias:
            params.append(self.bias)
        return params


class SloDropout(SloLayer):
    def __init__(self, p=0.1, name=""):
        super().__init__(name or f"Dropout{p}")
        self.p = p; self.soul_traits = {}
        self.training = True

    def train(self, mode: bool = True):
        self.training = mode

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0 or not self.training: return x
        mask = np.random.binomial(1, 1-self.p, x.data.shape).astype(np.float32) / (1-self.p)
        out = Tensor(x.data * mask, requires_grad=x.requires_grad, _children=(x,))
        def bk(g):
            if x.requires_grad:
                x.grad = Tensor(g * mask)
        out._backward_fn = bk; return out

    def parameters(self) -> List[Tensor]: return []


class SloEmbedding(SloLayer):
    def __init__(self, num_emb, emb_dim, name=""):
        super().__init__(name or f"Emb_{num_emb}x{emb_dim}")
        self.weight = randn((num_emb, emb_dim), requires_grad=True)
        self.weight.data *= math.sqrt(1.0/emb_dim)
        self.num_embeddings = num_emb; self.embedding_dim = emb_dim
        self.soul_traits = {"curiosity": 0.5, "warmth": 0.5}

    def forward_numpy(self, indices: np.ndarray) -> np.ndarray:
        # Handle both 2D and 3D inputs
        # 2D: (batch, seq_len) - standard format
        # 3D: (batch, seq_len, 1) - after expansion
        if indices.ndim == 3:
            # Only squeeze if the axis is actually size 1
            if indices.shape[1] == 1:
                indices = indices.squeeze(axis=1)
            elif indices.shape[2] == 1:
                indices = indices.squeeze(axis=2)
            else:
                # Reshape 3D to 2D by flattening last two dims
                indices = indices.reshape(indices.shape[0], -1)
        clipped = np.clip(indices.astype(int), 0, self.num_embeddings - 1)
        return np.take(self.weight.data, clipped, axis=0).reshape(indices.shape[0], indices.shape[1], self.embedding_dim)

    def forward(self, indices: Tensor) -> Tensor:
        data = indices.data
        if data.ndim == 3:
            data = np.squeeze(data, axis=1)
        flat = np.clip(data.astype(int).flatten(), 0, self.num_embeddings-1)
        embeds = self.weight.data[flat].reshape(data.shape[0], data.shape[1], self.embedding_dim)
        out = Tensor(embeds, requires_grad=True, _children=(self.weight,))
        def bk(g):
            g_arr = np.asarray(g)
            grad_out = g_arr.reshape(-1, self.embedding_dim)
            flat_np = flat.reshape(-1)
            w_grad = np.zeros_like(self.weight.data)
            np.add.at(w_grad, flat_np, grad_out)
            if self.weight.grad is None:
                self.weight.grad = Tensor(w_grad)
            else:
                self.weight.grad.data[:] += w_grad
        out._backward_fn = bk; return out

    def parameters(self) -> List[Tensor]: return [self.weight]


class SloLSTM(SloLayer):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.2, name=""):
        super().__init__(name or f"LSTM_{vocab_size}x{hidden_dim}")
        self.vocab_size = vocab_size; self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim; self.num_layers = num_layers; self.dropout = dropout
        self.embedding = SloEmbedding(vocab_size, embed_dim, "embed")
        self.W_ih = SloLinear(embed_dim, 4*hidden_dim, "W_ih")
        self.W_hh = SloLinear(hidden_dim, 4*hidden_dim, "W_hh")
        self.hidden_norm = SloRMSNorm(hidden_dim, 1e-5, "hidden_norm")
        if num_layers > 1:
            self.W_ih2 = SloLinear(hidden_dim, 4*hidden_dim, "W_ih2")
            self.W_hh2 = SloLinear(hidden_dim, 4*hidden_dim, "W_hh2")
            self.hidden_norm2 = SloRMSNorm(hidden_dim, 1e-5, "hidden_norm2")
        self.fc_out = SloLinear(hidden_dim, vocab_size, "fc_out")
        self.drop = SloDropout(dropout) if dropout > 0 else None
        self.soul_traits = {"warmth": 0.5, "creativity": 0.5, "curiosity": 0.5, "confidence": 0.5}
        self._init_recurrent_weights()

    def _init_recurrent_weights(self):
        """Initialize recurrent weights with smaller scale for gradient stability."""
        s = math.sqrt(1.0 / self.hidden_dim)
        self.W_hh.weight.data *= s / math.sqrt(2.0/(self.hidden_dim+4*self.hidden_dim)) if math.sqrt(2.0/(self.hidden_dim+4*self.hidden_dim)) > 0 else 1.0
        if self.num_layers > 1:
            self.W_hh2.weight.data *= s / max(math.sqrt(2.0/(self.hidden_dim+4*self.hidden_dim)), 1e-8)

    def forward(self, x: Tensor, hidden=None, adapter=None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        xd = x.data
        if xd.ndim == 3:
            xd = np.squeeze(xd, axis=1)
        xb = self.embedding.forward(type(x)(xd, requires_grad=False))
        xd = xb
        if self.drop:
            xd = self.drop.forward(type(x)(xd.data if isinstance(xd, Tensor) else xd, requires_grad=xd.requires_grad))
        xd_data = xd.data if isinstance(xd, Tensor) else xd
        h = _reshape(hidden[0] if hidden else zeros((1,self.hidden_dim)), (1,self.hidden_dim))
        c = _reshape(hidden[1] if hidden else zeros((1,self.hidden_dim)), (1,self.hidden_dim))
        hd = self.hidden_dim
        seq_len = xd_data.shape[1]
        ed = xd_data.shape[2]
        W_ih_T = self.W_ih.weight.T()
        W_hh_T = self.W_hh.weight.T()
        for t in range(seq_len):
            ce = _reshape(_slice(xd, (slice(None), t, slice(None))), (xd_data.shape[0], ed))
            gates = _add(_matmul(ce, W_ih_T), _matmul(h, W_hh_T))
            gate_i = sigmoid(_slice(gates, (slice(None), slice(hd))))
            gate_f = sigmoid(_slice(gates, (slice(None), slice(hd, 2*hd))))
            gate_g = tanh(_slice(gates, (slice(None), slice(2*hd, 3*hd))))
            gate_o = sigmoid(_slice(gates, (slice(None), slice(3*hd, None))))
            c = _add(_mul(gate_f, c), _mul(gate_i, gate_g))
            h_raw = _mul(gate_o, tanh(c))
            h = self.hidden_norm.forward(_reshape(h_raw, (1, hd)))
        if self.num_layers > 1:
            h2 = zeros((1,self.hidden_dim)); c2 = zeros((1,self.hidden_dim))
            W_ih2_T = self.W_ih2.weight.T()
            W_hh2_T = self.W_hh2.weight.T()
            for t in range(seq_len):
                gates2 = _add(_matmul(h, W_ih2_T), _matmul(h2, W_hh2_T))
                gate_i2 = sigmoid(_slice(gates2, (slice(None), slice(hd))))
                gate_f2 = sigmoid(_slice(gates2, (slice(None), slice(hd, 2*hd))))
                gate_g2 = tanh(_slice(gates2, (slice(None), slice(2*hd, 3*hd))))
                gate_o2 = sigmoid(_slice(gates2, (slice(None), slice(3*hd, None))))
                c2 = _add(_mul(gate_f2, c2), _mul(gate_i2, gate_g2))
                h2_raw = _mul(gate_o2, tanh(c2))
                h2 = self.hidden_norm2.forward(_reshape(h2_raw, (1, hd)))
            h,h2,c,c2 = h2,h,c2,c
        if adapter is not None:
            h_adapted = adapter.forward(_reshape(h, (1, self.hidden_dim)))
            h = _reshape(h_adapted, (1, self.hidden_dim))
        logits = self.fc_out.forward(_reshape(h,(self.hidden_dim,)))
        logits_2d = _reshape(logits, (1, self.vocab_size))
        return logits_2d, (Tensor(h.data.reshape(hd), requires_grad=False), Tensor(c.data.reshape(hd), requires_grad=False))

    def forward_numpy(self, x: np.ndarray, hidden=None, adapter=None, skip_embed=False) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """NumPy-only forward pass — no Tensor overhead. Pre-computes all input gates.

        Args:
            x: input — either integer token IDs (1D = (seq_len,) or 2D = (B, T)) or
               pre-embedded features (2D = (T, feat) or 3D = (B, T, feat)) when skip_embed=True
            hidden: optional (h, c) state tuple
            adapter: optional LoRA adapter
            skip_embed: if True, treat x as pre-embedded features instead of token IDs
        """
        if skip_embed:
            if x.ndim == 2:
                xd = x[np.newaxis, :, :]  # (T, feat) → (1, T, feat)
            elif x.ndim == 3:
                xd = x  # (B, T, feat)
            else:
                xd = x.reshape(1, 1, -1)  # (feat,) → (1, 1, feat)
            embeds = xd
        else:
            # Token ID mode: handle 1D (seq_len,) and 2D (B, T) input
            xd = x.reshape(1, -1) if x.ndim == 1 else x
            embeds = self.embedding.forward_numpy(xd)
        hd = self.hidden_dim
        all_igates = embeds @ self.W_ih.weight.data.T
        W_hh_T = self.W_hh.weight.data.T
        h = hidden[0].copy() if hidden else np.zeros((1, hd), dtype=np.float32)
        c = hidden[1].copy() if hidden else np.zeros((1, hd), dtype=np.float32)
        batch = xd.shape[0]
        seq_len = embeds.shape[1]
        for t in range(seq_len):
            gates = all_igates[:, t, :] + h @ W_hh_T
            g = gates[0]
            gi = 1.0 / (1.0 + np.exp(np.clip(-g[:hd], -500.0, 500.0)))
            gf = 1.0 / (1.0 + np.exp(np.clip(-g[hd:2*hd], -500.0, 500.0)))
            gg = np.tanh(g[2*hd:3*hd])
            go = 1.0 / (1.0 + np.exp(np.clip(-g[3*hd:], -500.0, 500.0)))
            c = gf * c + gi * gg
            h_raw = go * np.tanh(c)
            rms = np.sqrt(np.mean(h_raw**2, axis=-1, keepdims=True) + 1e-5)
            h = (h_raw / rms) * self.hidden_norm.weight.data
        if self.num_layers > 1:
            all_igates2 = h @ self.W_ih2.weight.data.T
            W_hh2_T = self.W_hh2.weight.data.T
            h2 = np.zeros((1, hd), dtype=np.float32)
            c2 = np.zeros((1, hd), dtype=np.float32)
            for t in range(seq_len):
                gates2 = all_igates2 + h2 @ W_hh2_T
                g2 = gates2[0]
                gi2 = 1.0 / (1.0 + np.exp(np.clip(-g2[:hd], -500.0, 500.0)))
                gf2 = 1.0 / (1.0 + np.exp(np.clip(-g2[hd:2*hd], -500.0, 500.0)))
                gg2 = np.tanh(g2[2*hd:3*hd])
                go2 = 1.0 / (1.0 + np.exp(np.clip(-g2[3*hd:], -500.0, 500.0)))
                c2 = gf2 * c2 + gi2 * gg2
                h2_raw = go2 * np.tanh(c2)
                rms2 = np.sqrt(np.mean(h2_raw**2, axis=-1, keepdims=True) + 1e-5)
                h2 = (h2_raw / rms2) * self.hidden_norm2.weight.data
            h, c = h2, c2
        if adapter is not None:
            down_w = adapter.down_proj.weight.data  # (rank, dim)
            h_down = h @ down_w.T  # (1, rank)
            h_act = np.maximum(h_down, 0)
            up_w = adapter.up_proj.weight.data  # (dim, rank)
            h_up = h_act @ up_w.T  # (1, dim)
            h = h + h_up
        logits = h.reshape(hd) @ self.fc_out.weight.data.T + self.fc_out.bias.data
        return logits.reshape(1, self.vocab_size), (h.reshape(hd), c.reshape(hd))

    def forward_numba(self, x: np.ndarray, hidden=None) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Numba-JIT accelerated forward pass — fastest inference path.
        Compiles the recurrent loop to machine code, eliminating Python loop overhead.
        Falls back to forward_numpy if Numba is unavailable.
        """
        if not _check_numba():
            return self.forward_numpy(x, hidden)
        try:
            from numba import njit, float32, int64
        except ImportError:
            return self.forward_numpy(x, hidden)

        xd = x
        if xd.ndim == 3:
            xd = np.squeeze(xd, axis=1)
        embeds = self.embedding.forward_numpy(xd)
        hd = self.hidden_dim

        W_ih_np = self.W_ih.weight.data.astype(np.float32).copy()
        W_hh_np = self.W_hh.weight.data.astype(np.float32).copy()
        fc_w = self.fc_out.weight.data.astype(np.float32).copy()
        fc_b = self.fc_out.bias.data.astype(np.float32).copy()
        W_ih_T = W_ih_np.T.copy()
        W_hh_T = W_hh_np.T.copy()

        has_layer2 = self.num_layers > 1
        if has_layer2:
            W_ih2_np = self.W_ih2.weight.data.astype(np.float32).copy()
            W_hh2_np = self.W_hh2.weight.data.astype(np.float32).copy()
            W_ih2_T = W_ih2_np.T.copy()
            W_hh2_T = W_hh2_np.T.copy()
        else:
            W_ih2_T = W_hh2_T = np.empty((1, 1), dtype=np.float32)

        h = hidden[0].copy().astype(np.float32) if hidden else np.zeros((1, hd), dtype=np.float32)
        c = hidden[1].copy().astype(np.float32) if hidden else np.zeros((1, hd), dtype=np.float32)

        @njit(cache=True)
        def _lstm_cell_1layer(embeds_flat, W_ih_T, W_hh_T, h, c, hd):
            seq_len = embeds_flat.shape[0]
            h_out = h.copy()
            c_out = c.copy()
            for t in range(seq_len):
                ce = embeds_flat[t:t+1, :]
                igates = np.dot(ce, W_ih_T)
                hgates = np.dot(h_out, W_hh_T)
                gates = igates + hgates
                gates_1d = gates[0, :]
                gi = np.float32(1.0) / (np.float32(1.0) + np.exp(np.clip(-gates_1d[:hd], np.float32(-500.0), np.float32(500.0))))
                gf = np.float32(1.0) / (np.float32(1.0) + np.exp(np.clip(-gates_1d[hd:2*hd], np.float32(-500.0), np.float32(500.0))))
                gg = np.tanh(gates_1d[2*hd:3*hd])
                go = np.float32(1.0) / (np.float32(1.0) + np.exp(np.clip(-gates_1d[3*hd:], np.float32(-500.0), np.float32(500.0))))
                c_out = gf.reshape(1, hd) * c_out + gi.reshape(1, hd) * gg.reshape(1, hd)
                h_out = go.reshape(1, hd) * np.tanh(c_out)
            return h_out, c_out

        @njit(cache=True)
        def _lstm_cell_2layer(embeds_flat, W_ih_T, W_hh_T, W_ih2_T, W_hh2_T,
                               h, c, hd):
            seq_len = embeds_flat.shape[0]
            h1 = h.copy()
            c1 = c.copy()
            for t in range(seq_len):
                ce = embeds_flat[t:t+1, :]
                igates = np.dot(ce, W_ih_T)
                hgates = np.dot(h1, W_hh_T)
                gates = igates + hgates
                g = gates[0, :]
                gi = np.float32(1.0) / (np.float32(1.0) + np.exp(np.clip(-g[:hd], np.float32(-500.0), np.float32(500.0))))
                gf = np.float32(1.0) / (np.float32(1.0) + np.exp(np.clip(-g[hd:2*hd], np.float32(-500.0), np.float32(500.0))))
                gg = np.tanh(g[2*hd:3*hd])
                go = np.float32(1.0) / (np.float32(1.0) + np.exp(np.clip(-g[3*hd:], np.float32(-500.0), np.float32(500.0))))
                c1 = gf.reshape(1, hd) * c1 + gi.reshape(1, hd) * gg.reshape(1, hd)
                h1 = go.reshape(1, hd) * np.tanh(c1)

            h2 = np.zeros((1, hd), dtype=np.float32)
            c2 = np.zeros((1, hd), dtype=np.float32)
            for t in range(seq_len):
                igates2 = np.dot(h1, W_ih2_T)
                hgates2 = np.dot(h2, W_hh2_T)
                gates2 = igates2 + hgates2
                g2 = gates2[0, :]
                gi2 = np.float32(1.0) / (np.float32(1.0) + np.exp(np.clip(-g2[:hd], np.float32(-500.0), np.float32(500.0))))
                gf2 = np.float32(1.0) / (np.float32(1.0) + np.exp(np.clip(-g2[hd:2*hd], np.float32(-500.0), np.float32(500.0))))
                gg2 = np.tanh(g2[2*hd:3*hd])
                go2 = np.float32(1.0) / (np.float32(1.0) + np.exp(np.clip(-g2[3*hd:], np.float32(-500.0), np.float32(500.0))))
                c2 = gf2.reshape(1, hd) * c2 + gi2.reshape(1, hd) * gg2.reshape(1, hd)
                h2 = go2.reshape(1, hd) * np.tanh(c2)
            return h2, c2

        embeds_flat = embeds[0]  # (seq_len, embed_dim) — contiguous
        if has_layer2:
            h, c = _lstm_cell_2layer(embeds_flat, W_ih_T, W_hh_T, W_ih2_T, W_hh2_T, h, c, hd)
        else:
            h, c = _lstm_cell_1layer(embeds_flat, W_ih_T, W_hh_T, h, c, hd)

        logits = np.dot(h.reshape(hd), fc_w.T.copy()) + fc_b
        logits_2d = logits.reshape(1, self.vocab_size).astype(np.float32)
        return logits_2d, (h.reshape(hd), c.reshape(hd))

    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data.fill(0)
                p.grad = None

    def init_hidden(self, batch=1) -> Tuple[Tensor, Tensor]:
        return zeros((batch, self.hidden_dim)), zeros((batch, self.hidden_dim))

    def parameters(self) -> List[Tensor]:
        ps = self.embedding.parameters()+self.W_ih.parameters()+self.W_hh.parameters()+self.fc_out.parameters()
        if hasattr(self,'W_ih2'): ps += self.W_ih2.parameters()+self.W_hh2.parameters()
        return [p for p in ps if p.requires_grad]


class SloAdapterLayer(SloLayer):
    """Lightweight bottleneck adapter for per-user personalization.

    Architecture: dim → down_proj (dim×rank) → gelu → up_proj (rank×dim) → + residual
    Up-projection initialized to zero so adapter is identity at start.
    Each user gets their own adapter — KB-scale personalization.
    """
    def __init__(self, dim: int = 768, rank: int = 8, name: str = ""):
        super().__init__(name or f"Adapter_dim{dim}_rank{rank}")
        self.dim = dim
        self.rank = rank
        s = math.sqrt(2.0 / dim)
        self.down_proj = SloLinear(dim, rank, "down", bias=False)
        self.down_proj.weight.data = randn((rank, dim)).data * s
        self.up_proj = SloLinear(rank, dim, "up", bias=False)
        self.up_proj.weight.data = np.zeros((dim, rank), dtype=np.float32)
        self.soul_traits = {"adaptability": 1.0}

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        h = gelu(self.down_proj.forward(x))
        out = self.up_proj.forward(h)
        return _add(residual, out)

    def parameters(self) -> List[Tensor]:
        return self.down_proj.parameters() + self.up_proj.parameters()


class SloConv2D(SloLayer):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, name=""):
        super().__init__(name or f"Conv{in_ch}x{out_ch}")
        kw = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
        kh = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        s = math.sqrt(2.0 / (kw * kh * in_ch))
        self.weight = randn((out_ch, in_ch, kh, kw), requires_grad=True)
        self.weight.data *= s
        self.bias = zeros((out_ch,), requires_grad=True)
        self.stride = stride if isinstance(stride, int) else stride[0]
        self.padding = padding
        self.in_ch = in_ch; self.out_ch = out_ch
        self.kernel_size = (kw, kh)
        self.soul_traits = {"creativity": 0.5, "warmth": 0.5}

    def forward(self, x: Tensor) -> Tensor:
        out = _conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
        return out

    def parameters(self) -> List[Tensor]: return [self.weight, self.bias]


class SloBatchNorm2D(SloLayer):
    def __init__(self, channels, momentum=0.9, eps=1e-5, name=""):
        super().__init__(name or f"BN{channels}")
        self.channels = channels
        self.momentum = momentum
        self.eps = eps
        self.gamma = ones((channels,), requires_grad=True)
        self.beta = zeros((channels,), requires_grad=True)
        self.running_mean = np.zeros(channels, dtype=np.float32)
        self.running_var = np.ones(channels, dtype=np.float32)
        self._train = True
        self.soul_traits = {"confidence": 0.5}

    def forward(self, x: Tensor) -> Tensor:
        return _batchnorm2d(x, self.gamma, self.beta, self.running_mean, self.running_var, self.eps, self._train)

    def parameters(self) -> List[Tensor]: return [self.gamma, self.beta]


class SloMaxPool2D(SloLayer):
    def __init__(self, kernel_size=2, stride=None, name=""):
        super().__init__(name or f"MaxPool{kernel_size}")
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size
        self.stride = stride if stride else self.kernel_size
        self.soul_traits = {"creativity": 0.5}

    def forward(self, x: Tensor) -> Tensor:
        return _maxpool2d(x, self.kernel_size, self.stride)

    def parameters(self) -> List[Tensor]: return []


class SloLayerNorm(SloLayer):
    def __init__(self, normalized_shape: int, eps: float = 1e-5, name=""):
        super().__init__(name or f"LayerNorm{normalized_shape}")
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.weight = ones(self.normalized_shape, requires_grad=True)
        self.bias = zeros(self.normalized_shape, requires_grad=True)
        self.soul_traits = {"warmth": 0.5, "confidence": 0.5}

    def forward(self, x: Tensor) -> Tensor:
        return _layernorm(x, self.weight, self.bias, self.eps)

    def parameters(self) -> List[Tensor]:
        return [self.weight, self.bias]


class SloRMSNorm(SloLayer):
    def __init__(self, dim: int, eps: float = 1e-5, name=""):
        super().__init__(name or f"RMSNorm{dim}")
        self.eps = eps
        self.weight = ones((dim,), requires_grad=True)
        self.soul_traits = {"confidence": 0.5, "creativity": 0.5}

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight.data

    def forward(self, x: Tensor) -> Tensor:
        return _rmsnorm(x, self.weight, self.eps)

    def parameters(self) -> List[Tensor]:
        return [self.weight]


class SloLayerNorm(SloLayer):
    """Layer normalization with weight and bias (GPT-2 style)."""

    def __init__(self, dim: int, eps: float = 1e-5, name=""):
        super().__init__(name or f"LayerNorm{dim}")
        self.eps = eps
        self.weight = ones((dim,), requires_grad=True)
        self.bias = zeros((dim,), requires_grad=True)

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        if _KERNELS_AVAILABLE:
            from domains.training.slonet_kernels import fused_layer_norm
            return fused_layer_norm(
                x.astype(np.float32),
                self.weight.data.astype(np.float32),
                self.bias.data.astype(np.float32),
                np.float32(self.eps),
            )
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + self.eps) * self.weight.data + self.bias.data

    def forward(self, x: Tensor) -> Tensor:
        return _layernorm(x, self.weight, self.bias, self.eps)

    def parameters(self) -> List[Tensor]:
        return [self.weight, self.bias]


class SloTransformerBlock(SloLayer):
    def __init__(self, d_model: int, n_heads: int, n_kv_head: Optional[int] = None,
                 dim_ff: int = None, use_rope: bool = False, max_seq_len: int = 2048,
                 rope_base: float = 10000.0, dropout: float = 0.1, eps: float = 1e-5,
                 norm_type: str = "rms_norm", name=""):
        super().__init__(name or f"Transformer{d_model}")
        dim_ff = dim_ff or d_model * 4
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        NormCls = SloLayerNorm if norm_type == "layer_norm" else SloRMSNorm
        self.attn_norm = NormCls(d_model, eps, name + "_attn_norm")
        self.attn = SloMultiHeadAttention(d_model, n_heads, n_kv_head=n_kv_head,
                                           use_rope=use_rope, max_seq_len=max_seq_len,
                                           rope_base=rope_base, name=name + "_attn")
        self.ff_norm = NormCls(d_model, eps, name + "_ff_norm")
        self.ff = SloFeedForward(d_model, dim_ff, name=name + "_ff")
        self.drop = SloDropout(dropout) if dropout > 0 else None
        self.use_checkpoint = False
        self.soul_traits = {"curiosity": 0.5, "creativity": 0.5, "warmth": 0.5}

    def train(self, mode: bool = True):
        self.attn_norm.train(mode)
        self.attn.train(mode)
        self.ff_norm.train(mode)
        self.ff.train(mode)
        if self.drop:
            self.drop.train(mode)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None,
                kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                start_pos: int = 0) -> Tensor:
        h, ca = None, None
        h = self.attn_norm.forward(x)
        h, ca = self.attn.forward(h, h, h, mask, kv_cache=kv_cache, start_pos=start_pos)
        if self.drop: h = self.drop.forward(h)
        x = x + h
        h = self.ff_norm.forward(x)
        h = self.ff.forward(h)
        if self.drop: h = self.drop.forward(h)
        return x + h, ca

    def forward_numpy(self, x: np.ndarray, mask: Optional[np.ndarray] = None,
                      kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                      start_pos: int = 0) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Numpy-only forward — no Tensor overhead, no autograd."""
        h = self.attn_norm.forward_numpy(x)
        h, ca = self.attn.forward_numpy(h, h, h, mask, kv_cache=kv_cache, start_pos=start_pos)
        x = x + h
        h = self.ff_norm.forward_numpy(x)
        h = self.ff.forward_numpy(h)
        return x + h, ca

    def parameters(self) -> List[Tensor]:
        return self.attn_norm.parameters() + self.attn.parameters() + self.ff_norm.parameters() + self.ff.parameters()


class SloRotaryEmbedding(SloLayer):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0, name=""):
        super().__init__(name or f"RoPE{dim}")
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        self.inv_freq = Tensor(inv_freq, requires_grad=False)
        self._cos_cached = None
        self._sin_cached = None
        self._cached_seq_len = 0
        self.soul_traits = {"curiosity": 0.5, "creativity": 0.5}

    def _precompute(self, seq_len: int):
        if self._cos_cached is not None and self._cached_seq_len >= seq_len:
            return
        t = np.arange(self._cached_seq_len, seq_len, dtype=np.float32)
        freqs = np.outer(t, self.inv_freq.data)
        emb = np.concatenate([freqs, freqs], axis=-1)
        new_cos = np.cos(emb)
        new_sin = np.sin(emb)
        if self._cos_cached is None:
            self._cos_cached = new_cos
            self._sin_cached = new_sin
        else:
            self._cos_cached = np.concatenate([self._cos_cached, new_cos])
            self._sin_cached = np.concatenate([self._sin_cached, new_sin])
        self._cached_seq_len = seq_len

    def forward(self, seq_len: int, start_pos: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        total = start_pos + seq_len
        if self._cos_cached is None or self._cached_seq_len < total:
            self._precompute(total)
        return self._cos_cached[start_pos:total], self._sin_cached[start_pos:total]

    def parameters(self) -> List[Tensor]:
        return []


def _rotate_half(x: np.ndarray) -> np.ndarray:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return np.concatenate([-x2, x1], axis=-1)


def _apply_rope(q: np.ndarray, k: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out, k_out


def _apply_rope_t(Q: Tensor, K: Tensor, cos: np.ndarray, sin: np.ndarray) -> Tuple[Tensor, Tensor]:
    """Apply rotary embeddings to 4D Q, K Tensors with gradient tracking.

    Uses raw numpy ops with a custom backward since the computation is
    element-wise (scale/rotate per head dimension).
    """
    q_data = Q.data * cos + _rotate_half(Q.data) * sin
    k_data = K.data * cos + _rotate_half(K.data) * sin

    Q_out = Tensor(q_data, requires_grad=True, _children=(Q,))
    K_out = Tensor(k_data, requires_grad=True, _children=(K,))

    def bk_q(g):
        if Q.requires_grad:
            g_q = g * cos + _rotate_half(g) * sin
            if Q.grad is None:
                Q.grad = Tensor(g_q)
            else:
                Q.grad.data += g_q
            if Q._backward_fn:
                Q._backward_fn(g_q)

    def bk_k(g):
        if K.requires_grad:
            g_k = g * cos + _rotate_half(g) * sin
            if K.grad is None:
                K.grad = Tensor(g_k)
            else:
                K.grad.data += g_k
            if K._backward_fn:
                K._backward_fn(g_k)

    Q_out._backward_fn = bk_q
    K_out._backward_fn = bk_k
    return Q_out, K_out


class SloMultiHeadAttention(SloLayer):
    def __init__(self, d_model: int, n_heads: int, n_kv_head: Optional[int] = None,
                 use_rope: bool = False, max_seq_len: int = 2048, rope_base: float = 10000.0, name=""):
        super().__init__(name or f"MHA{d_model}x{n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_kv_head = n_kv_head or n_heads
        self.n_rep = self.n_heads // self.n_kv_head
        self.use_rope = use_rope
        kv_dim = self.n_kv_head * self.head_dim
        self.W_q = SloLinear(d_model, n_heads * self.head_dim, name=name + "_q")
        self.W_k = SloLinear(d_model, kv_dim, name=name + "_k")
        self.W_v = SloLinear(d_model, kv_dim, name=name + "_v")
        self.W_o = SloLinear(d_model, d_model, name=name + "_o")
        if use_rope:
            self.rope = SloRotaryEmbedding(self.head_dim, max_seq_len, rope_base, name + "_rope")
        self.soul_traits = {"curiosity": 0.5}

    @staticmethod
    def _attention_4d(Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor],
                      scale: float) -> Tensor:
        """Batched attention with autograd: Q,K,V are 4D ``(B,N,H,E)`` Tensors.

        Returns 3D ``(B,N,C)`` Tensor with full gradient flow to Q, K, V.
        """
        B, N, H, E = Q.data.shape
        K_H = K.data.shape[2]
        n_rep = H // K_H
        n_kv_head = K_H

        K_exp = K
        V_exp = V
        if n_rep > 1:
            K_exp_data = K.data.repeat(n_rep, axis=2)
            V_exp_data = V.data.repeat(n_rep, axis=2)
            K_exp = Tensor(K_exp_data, requires_grad=K.requires_grad)
            V_exp = Tensor(V_exp_data, requires_grad=V.requires_grad)

        # Forward compute in numpy
        scale_f = float(scale)
        scores_np = np.einsum("bnhd,bmhd->bhnm", Q.data, K_exp.data) * scale_f
        if mask is not None:
            scores_np = scores_np + mask.data
        scores_max = scores_np.max(axis=-1, keepdims=True)
        attn_np = np.exp(scores_np - scores_max)
        attn_sum = attn_np.sum(axis=-1, keepdims=True)
        attn_np = attn_np / attn_sum
        out_np = np.einsum("bhnm,bmhd->bnhd", attn_np, V_exp.data)
        out_np = out_np.reshape(B, N, H * E)

        out_t = Tensor(out_np, requires_grad=True, _children=(Q, K_exp, V_exp))

        def bk(g):
            g_4d = g.reshape(B, N, H, E)
            g_attn = np.einsum("bnhd,bmhd->bhnm", g_4d, V_exp.data)
            g_V = np.einsum("bhnm,bnhd->bmhd", attn_np, g_4d)

            # Softmax backward: dL/dS = attn * (g_attn - sum(attn * g_attn, keepdims))
            g_softmax = attn_np * (g_attn - np.sum(attn_np * g_attn, axis=-1, keepdims=True))

            g_Q_np = np.einsum("bhnm,bmhd->bnhd", g_softmax, K_exp.data) * scale_f
            g_K_np = np.einsum("bhnm,bnhd->bmhd", g_softmax, Q.data) * scale_f

            if n_rep > 1:
                # Repeat backward: sum over repeated heads
                orig_shape = (B, g_K_np.shape[1], n_kv_head, E)
                g_K_reduced = np.zeros(orig_shape, dtype=np.float32)
                for h in range(H):
                    src_h = h // n_rep
                    g_K_reduced[:, :, src_h, :] += g_K_np[:, :, h, :]
                g_K_np = g_K_reduced
                g_V_np2 = np.zeros((B, g_V.shape[1], n_kv_head, E), dtype=np.float32)
                for h in range(H):
                    src_h = h // n_rep
                    g_V_np2[:, :, src_h, :] += g_V[:, :, h, :]
                g_V_np = g_V_np2
            else:
                g_V_np = g_V

            # Gradients stay 4D (same shape as Q, K_exp, V_exp).
            # The reshape backward (4D -> 3D) flows through the _reshape chain
            # upstream. Do NOT call _backward_fn here — the topo loop handles it.
            if Q.requires_grad:
                if Q.grad is None:
                    Q.grad = Tensor(g_Q_np)
                else:
                    Q.grad.data += g_Q_np
            if K_exp.requires_grad:
                if K_exp.grad is None:
                    K_exp.grad = Tensor(g_K_np)
                else:
                    K_exp.grad.data += g_K_np
            if V_exp.requires_grad:
                if V_exp.grad is None:
                    V_exp.grad = Tensor(g_V_np)
                else:
                    V_exp.grad.data += g_V_np

        out_t._backward_fn = bk
        return out_t

    def forward_numpy(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                      mask: Optional[np.ndarray] = None,
                      kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                      start_pos: int = 0) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        B, N, C = q.shape
        H, E, K_H = self.n_heads, self.head_dim, self.n_kv_head
        Q_r = self.W_q.forward_numpy(q).reshape(B, N, H, E)
        K_r = self.W_k.forward_numpy(k).reshape(B, N, K_H, E)
        V_r = self.W_v.forward_numpy(v).reshape(B, N, K_H, E)
        if self.use_rope:
            cos, sin = self.rope.forward(N, start_pos)
            cos_a, sin_a = cos.reshape(1, N, 1, E), sin.reshape(1, N, 1, E)
            Q_r = Q_r * cos_a + _rotate_half(Q_r) * sin_a
            K_r = K_r * cos_a + _rotate_half(K_r) * sin_a
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            K_r = np.concatenate([k_cache, K_r], axis=1)
            V_r = np.concatenate([v_cache, V_r], axis=1)
        scale_f = 1.0 / math.sqrt(E)
        if _KERNELS_AVAILABLE and B == 1 and N == 1:
            # Single-token decode: fused kernel has no causal masking needed
            from domains.training.slonet_kernels import fused_attention_single, gqa_expand
            # K_r: (B, seq, K_H, E) → (K_H, seq, E) for fused kernel
            K_np = K_r[0].transpose(1, 0, 2).astype(np.float32)
            V_np = V_r[0].transpose(1, 0, 2).astype(np.float32)
            if K_H < H:
                K_np = gqa_expand(K_np, H // K_H)
                V_np = gqa_expand(V_np, H // K_H)
            q_h = Q_r[0, 0].astype(np.float32)  # (H, E)
            out_h = fused_attention_single(q_h, K_np, V_np, np.float32(scale_f), H, E)
            out = out_h.reshape(1, 1, H * E)
        elif _KERNELS_AVAILABLE and B == 1 and mask is None:
            # Multi-token prompt: fused kernel applies built-in causal masking
            from domains.training.slonet_kernels import fused_attention_multi, gqa_expand
            K_np = K_r[0].transpose(1, 0, 2).astype(np.float32)  # (K_H, seq, E)
            V_np = V_r[0].transpose(1, 0, 2).astype(np.float32)
            if K_H < H:
                K_np = gqa_expand(K_np, H // K_H)
                V_np = gqa_expand(V_np, H // K_H)
            q_h = Q_r[0].astype(np.float32)  # (N, H, E) — already correct for fused
            out_h = fused_attention_multi(q_h, K_np, V_np, np.float32(scale_f), H, E)
            out = out_h.reshape(1, N, H * E)
        else:
            scores = np.einsum("bnhd,bmhd->bhnm", Q_r, K_r) * scale_f
            if mask is not None:
                scores = scores + mask
            attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = attn / attn.sum(axis=-1, keepdims=True)
            out = np.einsum("bhnm,bmhd->bnhd", attn, V_r).reshape(B, N, H * E)
        return self.W_o.forward_numpy(out), (K_r, V_r)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None,
                kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                start_pos: int = 0) -> Tensor:
        B, N, C = q.data.shape
        H = self.n_heads
        E = self.head_dim
        K_H = self.n_kv_head

        Q_raw = self.W_q.forward(q)
        K_raw = self.W_k.forward(k)
        V_raw = self.W_v.forward(v)

        # Reshape to 4D using _reshape (preserves gradient tracking)
        Q_4d = _reshape(Q_raw, (B, N, H, E))
        K_4d = _reshape(K_raw, (B, N, K_H, E))
        V_4d = _reshape(V_raw, (B, N, K_H, E))

        if self.use_rope:
            cos, sin = self.rope.forward(N, start_pos)
            cos_a = cos.reshape(1, N, 1, E)
            sin_a = sin.reshape(1, N, 1, E)
            Q_r, K_r = _apply_rope_t(Q_4d, K_4d, cos_a, sin_a)
        else:
            Q_r, K_r = Q_4d, K_4d

        V_r = V_4d
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            K_data = np.concatenate([k_cache, K_r.data], axis=1)
            V_data = np.concatenate([v_cache, V_r.data], axis=1)
            K_r = Tensor(K_data, requires_grad=K_r.requires_grad)
            V_r = Tensor(V_data, requires_grad=V_r.requires_grad)

        scale = 1.0 / math.sqrt(E)
        attn_out = self._attention_4d(Q_r, K_r, V_r, mask, scale)
        out_t = self.W_o.forward(attn_out)
        return out_t, (K_r.data, V_r.data)

    def parameters(self) -> List[Tensor]:
        ps = self.W_q.parameters() + self.W_k.parameters() + self.W_v.parameters() + self.W_o.parameters()
        if self.use_rope:
            ps += self.rope.parameters()
        return ps


class SloCrossAttention(SloLayer):
    """Cross-attention layer for multimodal fusion.

    Queries come from text decoder, keys/values come from image encoder.
    Used in BLIP/Flamingo-style architectures for image captioning.
    """
    def __init__(self, d_model: int, n_heads: int, name=""):
        super().__init__(name or f"CrossAttn{d_model}x{n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = SloLinear(d_model, d_model, name=name + "_q")
        self.k_proj = SloLinear(d_model, d_model, name=name + "_k")
        self.v_proj = SloLinear(d_model, d_model, name=name + "_v")
        self.o_proj = SloLinear(d_model, d_model, name=name + "_o")
        self.soul_traits = {"curiosity": 0.5, "creativity": 0.5}

    def forward(self, x: Tensor, context: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: text query (B, seq_len, d_model)
            context: image key/value (B, img_tokens, d_model)
            mask: optional attention mask
        Returns:
            attended output (B, seq_len, d_model)
        """
        B, N, C = x.data.shape
        H, E = self.n_heads, self.head_dim
        _, M, _ = context.data.shape  # M = image tokens

        Q = self.q_proj.forward(x).reshape(B, N, H, E)
        K = self.k_proj.forward(context).reshape(B, M, H, E)
        V = self.v_proj.forward(context).reshape(B, M, H, E)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(E)
        scores = np.einsum("bnhd,bmhd->bhnm", Q.data, K.data) * scale
        if mask is not None:
            scores = scores + mask.data
        scores_max = scores.max(axis=-1, keepdims=True)
        attn = np.exp(scores - scores_max)
        attn = attn / attn.sum(axis=-1, keepdims=True)
        out = np.einsum("bhnm,bmhd->bnhd", attn, V.data).reshape(B, N, H * E)

        out_t = Tensor(out, requires_grad=True, _children=(Q, K, V))
        if Q.requires_grad: Q._consumers.append(out_t)
        if K.requires_grad: K._consumers.append(out_t)
        if V.requires_grad: V._consumers.append(out_t)

        def bk(g):
            g_4d = g.reshape(B, N, H, E)
            g_attn = np.einsum("bnhd,bmhd->bhnm", g_4d, V.data)
            g_V = np.einsum("bhnm,bnhd->bmhd", attn, g_4d)
            g_softmax = attn * (g_attn - np.sum(attn * g_attn, axis=-1, keepdims=True))
            g_Q = np.einsum("bhnm,bmhd->bnhd", g_softmax, K.data) * scale
            g_K = np.einsum("bhnm,bnhd->bmhd", g_softmax, Q.data) * scale

            if Q.requires_grad:
                Q.grad = Tensor(g_Q) if Q.grad is None else Q.grad + Tensor(g_Q)
            if K.requires_grad:
                K.grad = Tensor(g_K) if K.grad is None else K.grad + Tensor(g_K)
            if V.requires_grad:
                V.grad = Tensor(g_V) if V.grad is None else V.grad + Tensor(g_V)

        out_t._backward_fn = bk
        def fwd(t_q, t_k, t_v):
            t_q_4d = np.zeros_like(Q.data) if t_q is None else t_q.reshape(B, N, H, E)
            t_k_4d = np.zeros_like(K.data) if t_k is None else t_k.reshape(B, M, H, E)
            t_v_4d = np.zeros_like(V.data) if t_v is None else t_v.reshape(B, M, H, E)
            # JVP of attention: scores = Q @ K.T, attn = softmax(scores), out = attn @ V
            # d_softmax = attn * (d_scores - sum(attn * d_scores))
            d_scores = np.einsum("bnhd,bmhd->bhnm", t_q_4d, K.data) + np.einsum("bnhd,bmhd->bhnm", Q.data, t_k_4d)
            d_scores = d_scores * scale
            d_attn = attn * (d_scores - np.sum(attn * d_scores, axis=-1, keepdims=True))
            result = np.einsum("bhnm,bmhd->bnhd", d_attn, V.data) + np.einsum("bhnm,bmhd->bnhd", attn, t_v_4d)
            return result.reshape(B, N, H * E)
        out_t._forward_fn = fwd
        return self.o_proj.forward(out_t)

    def parameters(self) -> List[Tensor]:
        return (self.q_proj.parameters() + self.k_proj.parameters() +
                self.v_proj.parameters() + self.o_proj.parameters())


class SloFeedForward(SloLayer):
    def __init__(self, d_model: int, dim_ff: int, name="", activation: str = "gelu"):
        super().__init__(name or f"FF{d_model}")
        self.w1 = SloLinear(d_model, dim_ff, name=name + "_w1")
        self.w2 = SloLinear(dim_ff, d_model, name=name + "_w2")
        self.w3 = SloLinear(d_model, dim_ff, name=name + "_w3")
        self.act_name = activation
        if activation == "silu":
            self.act = silu
            self.act_np = silu_np
        else:
            self.act = gelu
            self.act_np = gelu_np
        self.soul_traits = {"creativity": 0.5, "confidence": 0.5}

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        return self.w2.forward_numpy(self.act_np(self.w1.forward_numpy(x)) * self.w3.forward_numpy(x))

    def forward(self, x: Tensor) -> Tensor:
        return self.w2.forward(self.act(self.w1.forward(x)) * self.w3.forward(x))

    def parameters(self) -> List[Tensor]:
        return self.w1.parameters() + self.w2.parameters() + self.w3.parameters()


def _maxpool2d(x: Tensor, kernel_size: int, stride: int) -> Tensor:
    d = x.data
    n, c, h, w = d.shape
    out_h = (h - kernel_size) // stride + 1
    out_w = (w - kernel_size) // stride + 1
    result = np.full((n, c, out_h, out_w), -1e9, dtype=np.float32)
    for i in range(n):
        for ch in range(c):
            for oh in range(out_h):
                for ow in range(out_w):
                    ih = oh * stride
                    iw = ow * stride
                    result[i, ch, oh, ow] = d[i, ch, ih:ih+kernel_size, iw:iw+kernel_size].max()
    return Tensor(result, requires_grad=x.requires_grad, _children=(x,))


def _softmax(x: Tensor, dim: int = -1) -> Tensor:
    d = x.data
    meaned = d - d.max(axis=dim, keepdims=True)
    exp_d = np.exp(meaned)
    s = exp_d / exp_d.sum(axis=dim, keepdims=True)
    out = Tensor(s, requires_grad=x.requires_grad, _children=(x,))
    if out.requires_grad and x.requires_grad: x._consumers.append(out)
    def bk(g):
        if x.requires_grad:
            # Standard softmax backward: gx = s * (g - sum(s * g, dim))
            sg = np.sum(s * g, axis=dim, keepdims=True)
            gx = s * (g - sg)
            x.grad = Tensor(gx if x.grad is None else x.grad.data + gx)
    out._backward_fn = bk
    def fwd(t_x):
        if t_x is None: return np.zeros_like(s)
        # JVP: s * (t_x - sum(s * t_x))
        sg = np.sum(s * t_x, axis=dim, keepdims=True)
        return s * (t_x - sg)
    out._forward_fn = fwd
    return out


def _layernorm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    d = x.data
    acc = _get_accelerator()
    if d.ndim == 2:
        mean = d.mean(axis=1, keepdims=True); var = d.var(axis=1, keepdims=True)
        normed = (d - mean) / np.sqrt(var + eps)
    else:
        mean = d.mean(axis=-1, keepdims=True); var = d.var(axis=-1, keepdims=True)
        normed = (d - mean) / np.sqrt(var + eps)
    result = None
    if acc is not None and acc.name != "cpu":
        try: result = acc.layer_norm(d, weight.data, bias.data, eps)
        except Exception as e: logger.debug("Accelerator layernorm failed, using numpy: %s", e)
    if result is None:
        result = normed * weight.data + bias.data
    out = Tensor(result, requires_grad=x.requires_grad, _children=(x, weight, bias))
    if out.requires_grad:
        if x.requires_grad: x._consumers.append(out)
        if weight.requires_grad: weight._consumers.append(out)
        if bias.requires_grad: bias._consumers.append(out)
    def bk(g):
        sum_axes = tuple(range(g.ndim - 1))
        if x.requires_grad:
            gx = g * weight.data / np.sqrt(var + eps)
            x.grad = Tensor(gx if x.grad is None else x.grad.data + gx)
        if weight.requires_grad:
            gw = (g * normed).sum(axis=sum_axes)
            weight.grad = Tensor(gw if weight.grad is None else weight.grad.data + gw)
        if bias.requires_grad:
            gb = g.sum(axis=sum_axes)
            bias.grad = Tensor(gb if bias.grad is None else bias.grad.data + gb)
    out._backward_fn = bk
    def fwd(t_x, t_w, t_b):
        t_x = np.zeros_like(d) if t_x is None else t_x
        t_w = np.zeros_like(weight.data) if t_w is None else t_w
        t_b = np.zeros_like(bias.data) if t_b is None else t_b
        return t_x * weight.data / np.sqrt(var + eps) + normed * t_w + t_b
    out._forward_fn = fwd
    return out


def _rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    d = x.data
    acc = _get_accelerator()
    if d.ndim == 2:
        rms = np.sqrt(np.mean(d**2, axis=1, keepdims=True) + eps)
    else:
        rms = np.sqrt(np.mean(d**2, axis=-1, keepdims=True) + eps)
    x_normed = d / rms
    result = None
    if acc is not None and acc.name != "cpu" and d.size >= _ACCEL_THRESHOLD:
        try: result = acc.rms_norm(d, weight.data, eps)
        except Exception as e: logger.debug("Accelerator rmsnorm failed, using numpy: %s", e)
    if result is None:
        result = x_normed * weight.data

    out = Tensor(result, requires_grad=not _NO_GRAD and (x.requires_grad or weight.requires_grad), _children=(x, weight))
    if out.requires_grad:
        if x.requires_grad: x._consumers.append(out)
        if weight.requires_grad: weight._consumers.append(out)
    if not _NO_GRAD and (x.requires_grad or weight.requires_grad):
        _N = d.shape[-1]
        _w_data = weight.data.copy()
        def bk(g: np.ndarray):
            if weight.requires_grad:
                sum_axes = tuple(range(g.ndim - 1))
                gw = (g * x_normed).sum(axis=sum_axes)
                weight.grad = Tensor(gw if weight.grad is None else weight.grad.data + gw)
            if x.requires_grad:
                g_yw = g * _w_data
                gx = g_yw / rms - d * (g_yw * d).sum(axis=-1, keepdims=True) / (_N * rms**3)
                x.grad = Tensor(gx if x.grad is None else x.grad.data + gx)
        out._backward_fn = bk
        def fwd(t_x, t_w):
            t_x = np.zeros_like(d) if t_x is None else t_x
            t_w = np.zeros_like(weight.data) if t_w is None else t_w
            g_yw = t_x * _w_data
            gx = g_yw / rms - d * (g_yw * d).sum(axis=-1, keepdims=True) / (_N * rms**3)
            return gx + x_normed * t_w
        out._forward_fn = fwd
    return out


def _im2col(x: np.ndarray, kh: int, kw: int, stride: int) -> np.ndarray:
    """Convert image to column matrix for fast conv (im2col). Vectorized."""
    n, c, h, w = x.shape
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1
    n_patches = n * oh * ow
    feat_per_patch = c * kh * kw
    r_idx = np.arange(n_patches, dtype=np.intp)
    f_idx = np.arange(feat_per_patch, dtype=np.intp)
    n_idx = r_idx[:, None] // (oh * ow)
    spatial = r_idx[:, None] % (oh * ow)
    oh_i = spatial // ow
    ow_i = spatial % ow
    kh_i = (f_idx[None, :] % (kh * kw)) // kw
    kw_i = (f_idx[None, :] % (kh * kw)) % kw
    c_idx = f_idx[None, :] // (kh * kw)
    h_pos = oh_i * stride + kh_i
    w_pos = ow_i * stride + kw_i
    return x[n_idx, c_idx, h_pos, w_pos].reshape(n_patches, feat_per_patch)


def _conv2d(x: Tensor, weight: Tensor, bias: Tensor, stride: int = 1, padding: int = 0):
    """Fast conv2d using im2col + matmul (avoids nested Python loops). GPU-accelerated.

    Args:
        padding: int (same for H and W) or tuple (pad_h, pad_w).
    """
    if x.data.ndim != 4: raise ValueError(f"Conv2D needs 4D input, got {x.data.ndim}D")
    n, c, h, w = x.data.shape
    oc, ic, kh, kw = weight.data.shape
    if ic != c: raise ValueError(f"Channel mismatch: {ic} != {c}")

    if isinstance(padding, (tuple, list)):
        pad_h, pad_w = padding[0], padding[1] if len(padding) > 1 else padding[0]
        if pad_h > 0 or pad_w > 0:
            x_padded = np.pad(x.data, ((0,0),(0,0),(pad_h,pad_h),(pad_w,pad_w)), mode='constant')
        else:
            x_padded = x.data
    elif padding > 0:
        x_padded = np.pad(x.data, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
    else:
        x_padded = x.data

    oh = (x_padded.shape[2] - kh) // stride + 1
    ow = (x_padded.shape[3] - kw) // stride + 1

    cols = _im2col(x_padded, kh, kw, stride)
    w_col = weight.data.reshape(oc, -1)

    acc = _get_accelerator()
    if acc is not None and acc.name != "cpu":
        try:
            result = acc.matmul(w_col, cols.T).reshape(n, oc, oh, ow)
        except Exception:
            result = np.matmul(w_col, cols.T).reshape(n, oc, oh, ow)
    else:
        result = np.matmul(w_col, cols.T).reshape(n, oc, oh, ow)

    if bias is not None:
        result = result + bias.data[:, None, None]

    weight_req = weight.requires_grad
    bias_req = bias is not None and bias.requires_grad
    out = Tensor(result, requires_grad=not _NO_GRAD and (x.requires_grad or weight_req or bias_req), _children=(x, weight, bias))
    if out.requires_grad:
        if x.requires_grad: x._consumers.append(out)
        if weight.requires_grad: weight._consumers.append(out)
        if bias is not None and bias.requires_grad: bias._consumers.append(out)
    _w_col = w_col
    def bk(g):
        if x.requires_grad:
            dY_flat = g.reshape(n * oh * ow, oc)
            dX_col = dY_flat @ weight.data.reshape(oc, -1)
            n_patches = n * oh * ow
            feat_per_patch = c * kh * kw
            r_idx_b = np.arange(n_patches, dtype=np.intp)
            f_idx_b = np.arange(feat_per_patch, dtype=np.intp)
            n_idx_b = r_idx_b[:, None] // (oh * ow)
            spatial_b = r_idx_b[:, None] % (oh * ow)
            oh_i_b = spatial_b // ow
            ow_i_b = spatial_b % ow
            kh_i_b = (f_idx_b[None, :] % (kh * kw)) // kw
            kw_i_b = (f_idx_b[None, :] % (kh * kw)) % kw
            c_idx_b = f_idx_b[None, :] // (kh * kw)
            h_pos_b = oh_i_b * stride + kh_i_b
            w_pos_b = ow_i_b * stride + kw_i_b
            grad_in = np.zeros_like(x_padded)
            n_idx_b_full = np.broadcast_to(n_idx_b, (n_patches, feat_per_patch))
            c_idx_b_full = np.broadcast_to(c_idx_b, (n_patches, feat_per_patch))
            np.add.at(grad_in, (n_idx_b_full.ravel(), c_idx_b_full.ravel(), h_pos_b.ravel(), w_pos_b.ravel()), dX_col.ravel())
            if isinstance(padding, (tuple, list)):
                pad_h, pad_w = padding[0], padding[1] if len(padding) > 1 else padding[0]
                if pad_h > 0 or pad_w > 0:
                    sl_h = slice(pad_h, -pad_h or None)
                    sl_w = slice(pad_w, -pad_w or None)
                    grad_in = grad_in[:, :, sl_h, sl_w]
            elif padding > 0:
                grad_in = grad_in[:, :, padding:-padding, padding:-padding]
            x.grad = Tensor(grad_in if x.grad is None else x.grad.data + grad_in)
        if weight.requires_grad:
            dY_flat = g.reshape(n * oh * ow, oc)
            gw = (cols.T @ dY_flat).T.reshape(oc, c, kh, kw)
            weight.grad = Tensor(gw if weight.grad is None else weight.grad.data + gw)
        if bias is not None and bias.requires_grad:
            gb = g.sum(axis=(0, 2, 3))
            bias.grad = Tensor(gb if bias.grad is None else bias.grad.data + gb)
    out._backward_fn = bk
    def fwd(t_x, t_w, t_b):
        t_x_np = np.zeros_like(x.data) if t_x is None else t_x
        t_w_np = np.zeros_like(weight.data) if t_w is None else t_w
        # JVP: conv2d(x, w) = im2col(x) @ w.T → JVP = im2col(t_x) @ w.T + im2col(x) @ t_w.T
        t_cols = _im2col(np.pad(t_x_np, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant'), kh, kw, stride) if (not isinstance(t_x is None)) else 0
        w_col_t = t_w_np.reshape(oc, -1) if not (t_w is None) else 0
        result_t = (np.matmul(t_cols, _w_col.T) + np.matmul(cols, w_col_t.T)).reshape(n, oc, oh, ow)
        if bias is not None:
            t_b_np = np.zeros_like(bias.data) if t_b is None else t_b
            result_t = result_t + t_b_np[:, None, None]
        return result_t
    out._forward_fn = fwd
    return out


def _batchnorm2d(x: Tensor, gamma: Tensor, beta: Tensor, running_mean, running_var, eps, training):
    n, c, h, w = x.data.shape
    if training:
        mean = x.data.mean(axis=(0, 2, 3), keepdims=True)
        var = x.data.var(axis=(0, 2, 3), keepdims=True)
    else:
        mean = running_mean.reshape(1, c, 1, 1)
        var = running_var.reshape(1, c, 1, 1)

    norm = (x.data - mean) / np.sqrt(var + eps)
    out_data = gamma.data.reshape(1, c, 1, 1) * norm + beta.data.reshape(1, c, 1, 1)
    out = Tensor(out_data, requires_grad=x.requires_grad or gamma.requires_grad or beta.requires_grad, _children=(x, gamma, beta))
    if out.requires_grad:
        if x.requires_grad: x._consumers.append(out)
        if gamma.requires_grad: gamma._consumers.append(out)
        if beta.requires_grad: beta._consumers.append(out)

    def bk(g):
        if x.requires_grad:
            x_grad = g * gamma.data.reshape(1, c, 1, 1) / np.sqrt(var + eps) / (n * h * w)
            x.grad = Tensor(x_grad if x.grad is None else x.grad.data + x_grad)
        if gamma.requires_grad:
            g_gamma = (g * norm).sum(axis=(0, 2, 3))
            gamma.grad = Tensor(g_gamma if gamma.grad is None else gamma.grad.data + g_gamma)
        if beta.requires_grad:
            g_beta = g.sum(axis=(0, 2, 3))
            beta.grad = Tensor(g_beta if beta.grad is None else beta.grad.data + g_beta)
    out._backward_fn = bk
    def fwd(t_x, t_g, t_b):
        t_x = np.zeros_like(x.data) if t_x is None else t_x
        t_g = np.zeros_like(gamma.data) if t_g is None else t_g
        t_b = np.zeros_like(beta.data) if t_b is None else t_b
        return t_x * gamma.data.reshape(1, c, 1, 1) / np.sqrt(var + eps) + norm * t_g.reshape(1, c, 1, 1) + t_b.reshape(1, c, 1, 1)
    out._forward_fn = fwd
    return out


def _maxpool2d(x: Tensor, kernel_size, stride):
    ks = kernel_size if isinstance(kernel_size, int) else kernel_size
    s = stride if isinstance(stride, int) else stride
    n, c, h, w = x.data.shape
    out_h = (h - ks) // s + 1
    out_w = (w - ks) // s + 1
    result = np.zeros((n, c, out_h, out_w), dtype=np.float32)
    max_indices = {}
    for i in range(n):
        for ch in range(c):
            for oh in range(out_h):
                for ow in range(out_w):
                    ih = oh * s; iw = ow * s
                    patch = x.data[i, ch, ih:ih+ks, iw:iw+ks]
                    max_idx = np.unravel_index(patch.argmax(), (ks, ks))
                    result[i, ch, oh, ow] = patch.max()
                    max_indices[(i, ch, oh, ow)] = (ih + max_idx[0], iw + max_idx[1])

    out = Tensor(result, requires_grad=x.requires_grad, _children=(x,))
    if out.requires_grad and x.requires_grad: x._consumers.append(out)

    def bk(g):
        if x.requires_grad:
            grad_in = np.zeros_like(x.data)
            for i in range(n):
                for ch in range(c):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            ih, iw = max_indices[(i, ch, oh, ow)]
                            grad_in[i, ch, ih, iw] += g[i, ch, oh, ow]
            x.grad = Tensor(grad_in if x.grad is None else x.grad.data + grad_in)
    out._backward_fn = bk
    def fwd(t_x):
        if t_x is None: return np.zeros_like(result)
        out_t = np.zeros_like(result)
        for i in range(n):
            for ch in range(c):
                for oh in range(out_h):
                    for ow in range(out_w):
                        ih, iw = max_indices[(i, ch, oh, ow)]
                        out_t[i, ch, oh, ow] = t_x[i, ch, ih, iw]
        return out_t
    out._forward_fn = fwd
    return out


def flatten(x: Tensor) -> Tensor:
    """Flatten a 4D tensor to 2D for classification heads."""
    orig_shape = x.shape
    out = Tensor(x.data.reshape(x.data.shape[0], -1), requires_grad=x.requires_grad, _children=(x,))
    if out.requires_grad and x.requires_grad: x._consumers.append(out)
    def bk(g):
        if x.requires_grad: x.grad = Tensor(g.reshape(orig_shape) if x.grad is None else x.grad.data + g.reshape(orig_shape))
    out._backward_fn = bk
    def fwd(t_x):
        if t_x is None: return np.zeros((out.shape[0], np.prod(tuple(s for i, s in enumerate(orig_shape) if i > 0))))
        return t_x.reshape(out.shape)
    out._forward_fn = fwd
    return out


class _SoulTransformerBlockSoulLib(SloLayer):
    """Stateless transformer block used when model is loaded from PyTorch state dict."""
    def __init__(self, hidden: int, n_heads: int, ff_dim: int, name=""):
        super().__init__(name or f"Block{hidden}x{n_heads}")
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.ff_dim = ff_dim

    def forward(self, x: Tensor) -> Tensor:
        return x

    def parameters(self) -> List[Tensor]:
        return []

    def soul_signature(self) -> Dict:
        return {"layer": "SloTransformerBlock", "hidden": self.hidden, "n_heads": self.n_heads}


# =============================================================================
# SOUL NET MODEL
# =============================================================================

class SloNet:
    def __init__(self, layers=None, soul_name="Slo", soul_traits=None, system_prompt="", lineage="slonet", metadata=None):
        self.layers = layers or []
        self.soul_name = soul_name
        self.soul_traits = soul_traits or {"warmth":0.5,"creativity":0.5,"curiosity":0.5,"confidence":0.5,"empathy":0.5}
        self.system_prompt = system_prompt
        self.lineage = lineage
        self.metadata = metadata or {}
        self._step = 0
        self._created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        self._sd: Dict[str, np.ndarray] = {}
        self._user_adapters: Dict[str, SloAdapterLayer] = {}
        self._active_user_id: Optional[str] = None

    def set_active_user(self, user_id: Optional[str]) -> None:
        """Set the active user for per-user adapter application."""
        self._active_user_id = user_id

    def get_user_adapter(self, user_id: str, dim: int = 768, rank: int = 8) -> SloAdapterLayer:
        """Get or create a per-user adapter layer.

        Checks in-memory cache first, then tries to load from disk
        (``data/user_adapters/{user_id}_adapter.npz``), and finally
        creates a fresh identity-initialized adapter.
        """
        if user_id in self._user_adapters:
            return self._user_adapters[user_id]

        adapter = SloAdapterLayer(dim=dim, rank=rank, name=f"adapter_{user_id}")
        try:
            from pathlib import Path
            path = Path("data/user_adapters") / f"{user_id}_adapter.npz"
            if path.exists():
                data = np.load(str(path))
                if "down_weight" in data and "up_weight" in data:
                    dw = data["down_weight"]
                    uw = data["up_weight"]
                    if dw.shape == adapter.down_proj.weight.data.shape:
                        adapter.down_proj.weight.data = dw.copy()
                        adapter.up_proj.weight.data = uw.copy()
        except Exception:
            pass

        self._user_adapters[user_id] = adapter
        return adapter

    def remove_user_adapter(self, user_id: str) -> None:
        """Remove a user's adapter from memory."""
        self._user_adapters.pop(user_id, None)

    def train(self, mode: bool = True):
        for l in self.layers:
            if isinstance(l, SloLayer):
                l.train(mode)

    def eval(self):
        self.train(False)

    def parameters(self) -> List[Tensor]:
        ps = []
        for l in self.layers:
            if isinstance(l, SloLayer):
                ps.extend(l.parameters())
        return ps

    def state_dict(self) -> Dict[str, np.ndarray]:
        """Return all parameters as a dict (compatible with slo_format.save_soul)."""
        result = {}
        for i, p in enumerate(self.parameters()):
            result[f"p{i}"] = p.data.copy()
        return result

    def _rebuild_from_state_dict(self, sd: Dict[str, np.ndarray]) -> None:
        """Rebuild architecture from PyTorch state dict keys, then load weights."""
        self._sd = sd
        self.layers = []

        # Token embedding
        if "tok_emb.weight" in sd:
            vocab, hidden = sd["tok_emb.weight"].shape
            self.layers.append(SloEmbedding(vocab, hidden, "tok_emb"))
            # LM head (tied)
            self.layers.append(SloLinear(hidden, vocab, "lm_head"))

        # Count transformer blocks
        block_keys = [k for k in sd if k.startswith("blocks.")]
        num_blocks = len(set(k.split(".")[1] for k in block_keys))

        # Determine dims from first block
        norm1_w = sd.get(f"blocks.0.norm1.weight")
        hidden_dim = norm1_w.shape[0] if norm1_w is not None else 384

        q_w = sd.get(f"blocks.0.attn.q_proj.weight")
        n_heads = 4  # default
        if q_w is not None:
            head_dim = q_w.shape[0]
            n_heads = hidden_dim // head_dim

        ff_w1 = sd.get(f"blocks.0.mlp.w1.weight")
        ff_dim = ff_w1.shape[0] if ff_w1 is not None else hidden_dim * 4

        for i in range(num_blocks):
            block = _SoulTransformerBlockSoulLib(
                hidden_dim, n_heads, ff_dim,
                name=f"block{i}"
            )
            self.layers.append(block)

        # Output norm
        if "norm.weight" in sd:
            self.layers.append(SloLayerNorm(hidden_dim, name="output_norm"))

        # Load weights into _sd for forward pass to use
        self._sd = sd

    def _get_weight(self, key: str) -> Optional[np.ndarray]:
        return self._sd.get(key)

    def forward(self, x, user_id: Optional[str] = None) -> Tensor:
        if isinstance(x, np.ndarray):
            x = Tensor(x.astype(np.float32) if x.dtype != np.float32 else x.copy())
        elif isinstance(x, list):
            x = Tensor(np.array(x, dtype=np.float32))
        elif isinstance(x, memoryview):
            x = Tensor(np.array(x, dtype=np.float32))
        elif hasattr(x, 'data') and isinstance(x.data, np.ndarray):
            x = Tensor(x.data.copy())
        else:
            x = Tensor(x)

        uid = user_id or self._active_user_id

        if self._sd and "tok_emb.weight" in self._sd:
            return self._forward_state_dict(x)

        for l in self.layers:
            if callable(l):
                x = l(x)
            else:
                x = l.forward(x)

        if uid and uid in self._user_adapters:
            adapter = self._user_adapters[uid]
            adapter_out = adapter.forward(x)
            return adapter_out

        return x

    def __call__(self, x) -> Tensor:
        return self.forward(x)

    def fit(self, X, y, optimizer, epochs=10, batch_size=32, on_step=None, max_grad_norm=1.0) -> List[float]:
        losses = []
        params = self.parameters()
        n = X.shape[0]
        for ep in range(epochs):
            ep_loss, steps = 0.0, 0
            for i in range(0, n, batch_size):
                xb = Tensor(X.data[i:i+batch_size], requires_grad=True)
                yb = Tensor(y.data[i:i+batch_size])
                pred = self.forward(xb)
                loss = cross_entropy(pred, yb)
                loss.backward()
                if max_grad_norm is not None:
                    clip_grad_norm_(params, max_grad_norm)
                optimizer.step(params)
                ep_loss += loss.data[()]
                steps += 1; self._step += 1
                if on_step: on_step(self._step, loss.data[()], ep)
            losses.append(ep_loss/max(steps,1))
        return losses

    def soul_signature(self) -> Dict:
        return {"soul_name":self.soul_name,"soul_traits":self.soul_traits,"lineage":self.lineage,
                "layers":[l.soul_signature() for l in self.layers],"step":self._step,"created_at":self._created_at,"system_prompt":self.system_prompt}

    def num_parameters(self) -> int: return sum(p.data.size for p in self.parameters())

    def apply_gradient_checkpointing(self) -> None:
        for l in self.layers:
            if hasattr(l, "use_checkpoint"):
                l.use_checkpoint = True

    def named_modules(self, prefix="") -> List[Tuple[str, "SloNet"]]:
        return [(prefix, self)]

    def named_children(self) -> List[Tuple[str, "SloLayer"]]:
        return [(f"layer_{i}", l) for i, l in enumerate(self.layers) if hasattr(l, "forward")]

    def _get_weights_dict(self) -> Dict[str, Any]:
        return {f"p{i}": p.data.tolist() for i, p in enumerate(self.parameters())}

    def _forward_state_dict(self, x: Tensor) -> Tensor:
        sd = self._sd
        vocab = sd["tok_emb.weight"].shape[0]
        hidden = sd["tok_emb.weight"].shape[1]

        def _sd_get(key):
            if key in sd:
                return sd[key]
            key2 = key.replace("attn.", "").replace("mlp.", "")
            return sd.get(key2)

        indices = x.data.astype(int).clip(0, vocab - 1)
        B, N = indices.shape
        flat = indices.flatten()
        emb = sd["tok_emb.weight"][flat].reshape(B, N, hidden)
        h = np.clip(emb, -50, 50)
        num_blocks = len(set(k.split(".")[1] for k in sd if k.startswith("blocks.")))
        for i in range(num_blocks):
            n1_w = _sd_get(f"blocks.{i}.norm1.weight")
            h_norm = _layernorm_state_dict(Tensor(h), n1_w)
            h = h_norm.data

            q_w = _sd_get(f"blocks.{i}.attn.q_proj.weight")
            k_w = _sd_get(f"blocks.{i}.attn.k_proj.weight")
            v_w = _sd_get(f"blocks.{i}.attn.v_proj.weight")
            if q_w is None:
                q_w = _sd_get(f"blocks.{i}.q_proj.weight")
                k_w = _sd_get(f"blocks.{i}.k_proj.weight")
                v_w = _sd_get(f"blocks.{i}.v_proj.weight")

            H = 4
            E = hidden // H
            C = hidden
            if q_w is not None:
                q_out_dim = q_w.shape[0]
                head_dim = q_out_dim // 8
                if head_dim > 0 and q_out_dim % 8 == 0:
                    H = 8
                    E = head_dim

            q = _matmul_state_dict(Tensor(h), q_w.T).data
            k = _matmul_state_dict(Tensor(h), k_w.T).data
            v = _matmul_state_dict(Tensor(h), v_w.T).data

            q = np.clip(q, -100, 100)
            k = np.clip(k, -50, 50)
            v = np.clip(v, -50, 50)

            Q = q.reshape(B, N, H, E).transpose(0, 2, 1, 3)
            K = k.reshape(B, -1, H, E).transpose(0, 2, 1, 3)
            V = v.reshape(B, -1, H, E).transpose(0, 2, 1, 3)
            scale = 1.0 / (E ** 0.5)
            scores = np.einsum("bhnd,bhkd->bhnk", Q, K) * scale
            attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = attn / attn.sum(axis=-1, keepdims=True)
            attn_out = np.einsum("bhnk,bhkd->bhnd", attn, V).transpose(0, 2, 1, 3).reshape(B, N, C)

            o_w = _sd_get(f"blocks.{i}.attn.o_proj.weight")
            if o_w is None:
                o_w = _sd_get(f"blocks.{i}.proj.weight")
            h = np.clip(h + (attn_out.reshape(B * N, -1) @ o_w.T).reshape(B, N, -1), -50, 50)

            n2_w = _sd_get(f"blocks.{i}.norm2.weight")
            h_norm2 = _layernorm_state_dict(Tensor(h), n2_w)
            h = h_norm2.data

            w1 = _sd_get(f"blocks.{i}.mlp.w1.weight")
            w2 = _sd_get(f"blocks.{i}.mlp.w2.weight")
            w3 = _sd_get(f"blocks.{i}.mlp.w3.weight")
            if w1 is None:
                w1 = _sd_get(f"blocks.{i}.w1.weight")
                w2 = _sd_get(f"blocks.{i}.w2.weight")
                w3 = _sd_get(f"blocks.{i}.w3.weight")

            mid1 = np.clip(h @ w1.T, -30, 30)
            mid3 = np.clip(h @ w3.T, -30, 30)
            swiglu = (mid1 / (1 + np.exp(-np.clip(mid1, -30, 30)))) * mid3
            h = np.clip(h + (swiglu @ w2.T).reshape(B, N, -1), -50, 50)

        if "norm.weight" in sd:
            h_norm = _layernorm_state_dict(Tensor(h), sd["norm.weight"])
            h = h_norm.data

        logits = h @ sd["lm_head.weight"].T
        logits = np.clip(logits, -50, 50)
        logits = logits - logits.max(axis=-1, keepdims=True)
        return Tensor(logits.astype(np.float32))

    def _load_weights(self, weights: Dict[str, Any]) -> None:
        if not weights:
            return
        for i, p in enumerate(self.parameters()):
            key = f"p{i}"
            if key in weights:
                p.data[:] = np.array(weights[key], dtype=np.float32)


def _matmul_state_dict(a: Tensor, b: np.ndarray) -> Tensor:
    acc = _get_accelerator()
    if acc is not None and acc.name != "cpu":
        try:
            result = acc.matmul(a.data, b)
            return Tensor(result)
        except Exception:
            pass
    return Tensor(a.data @ b)


def _layernorm_state_dict(x: Tensor, weight: np.ndarray, eps: float = 1e-5) -> Tensor:
    acc = _get_accelerator()
    if acc is not None and acc.name != "cpu":
        try:
            result = acc.layer_norm(x.data, weight, np.ones_like(weight), eps)
            return Tensor(result)
        except Exception:
            pass
    d = x.data
    mean = d.mean(axis=-1, keepdims=True)
    var = d.var(axis=-1, keepdims=True)
    return Tensor(((d - mean) / np.sqrt(var + eps)) * weight)


# =============================================================================
# OPTIMIZERS
# =============================================================================

def _invalidate_gpu_cache():
    try:
        from domains.slolib.gpu import get_accelerator
        acc = get_accelerator()
        if hasattr(acc, 'clear_cache'):
            acc.clear_cache()
    except Exception:
        pass


def clip_grad_norm_(params, max_norm=1.0, norm_type=2.0):
    """Clip gradients to a maximum norm."""
    params_list = [p for p in params if p.grad is not None and p.requires_grad]
    if not params_list:
        return 0.0
    total_norm = 0.0
    for p in params_list:
        param_norm = np.sum(p.grad.data ** 2)
        total_norm += param_norm
    total_norm = float(np.sqrt(total_norm))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params_list:
            p.grad.data *= clip_coef
    return total_norm


class SloSGD:
    def __init__(self, lr=0.01, momentum=0.0, max_grad_norm=None):
        self.lr = lr; self.momentum = momentum; self.max_grad_norm = max_grad_norm
        self._v: Dict[int, Any] = {}

    def step(self, params):
        if self.max_grad_norm is not None:
            clip_grad_norm_(params, self.max_grad_norm)
        for p in params:
            if p.grad is None or not p.requires_grad: continue
            g = p.grad.data; pid = id(p)
            if self.momentum > 0:
                self._v[pid] = self.momentum*self._v.get(pid,0)+g if pid not in self._v else self.momentum*self._v[pid]+g
                g = self._v[pid]
            p.data -= self.lr*g
            p.grad = None


class SloAdam:
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0, max_grad_norm=None):
        self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.weight_decay = weight_decay; self.max_grad_norm = max_grad_norm
        self._m: Dict[int, Any] = {}; self._v: Dict[int, Any] = {}; self._t = 0

    @staticmethod
    def _zeros_like(t):
        """Create zeros array matching the input's framework (numpy or torch)."""
        if isinstance(t, np.ndarray):
            return np.zeros_like(t)
        return t.new_zeros(t.shape, dtype=t.dtype)

    @staticmethod
    def _sqrt(t):
        if isinstance(t, np.ndarray):
            return np.sqrt(t)
        return t.sqrt()

    def step(self, params):
        if self.max_grad_norm is not None:
            clip_grad_norm_(params, self.max_grad_norm)
        self._t += 1; b1 = self.b1; b2 = self.b2; eps = self.eps; lr = self.lr; wd = self.weight_decay
        for p in params:
            if p.grad is None or not p.requires_grad: continue
            g = p.grad.data; pid = id(p)
            if wd != 0:
                g = g + wd * p.data
            if pid not in self._m: self._m[pid] = self._zeros_like(p.data)
            if pid not in self._v: self._v[pid] = self._zeros_like(p.data)
            self._m[pid] = b1*self._m[pid]+(1-b1)*g
            self._v[pid] = b2*self._v[pid]+(1-b2)*(g**2)
            mh = self._m[pid]/(1-b1**self._t); vh = self._v[pid]/(1-b2**self._t)
            upd = lr*mh/(self._sqrt(vh)+eps)
            if upd.shape != p.data.shape: upd = upd.sum(axis=0)
            p.data -= upd; p.grad = None


# =============================================================================
# SOU EXPORT / IMPORT
# =============================================================================

SOU_MAGIC = b"SOUL"; SOU_VERSION = 1  # shared with domains.inference.slo_format


def _sanitize(obj):
    if isinstance(obj, dict): return {k:_sanitize(v) for k,v in obj.items()}
    if isinstance(obj, list): return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
    return obj


def export_to_sou(net: SloNet, path: str, include_weights=True, metadata: dict = None) -> str:
    base_metadata = {
        "version": 3, "soul_name": net.soul_name, "soul_traits": net.soul_traits,
        "lineage": net.lineage, "system_prompt": net.system_prompt,
        "soul_signature": net.soul_signature(), "metadata": net.metadata,
        "created_at": net._created_at, "step": net._step,
    }
    if metadata:
        base_metadata["metadata"] = {**(base_metadata.get("metadata") or {}), **metadata}
    metadata = base_metadata
    json_bytes = json.dumps(_sanitize(metadata), allow_nan=False).encode()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Write .meta.json first (small, fast — serves as sidecar for list endpoint)
    with open(path + ".meta.json", "w") as f:
        json.dump(_sanitize(metadata), f, indent=2)

    # Atomic write: temp file then rename
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path) or ".", suffix=".tmp",
    )
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(SOU_MAGIC)
            f.write(struct.pack("<I", 3))
            f.write(struct.pack("<I", len(json_bytes)))
            f.write(json_bytes)
            if include_weights:
                if hasattr(net, "state_dict") and isinstance(net, SloTransformer):
                    state_items = list(net.state_dict().items())
                else:
                    state_items = [(f"p{i}", p.data) for i, p in enumerate(net.parameters())]
                params = [(k, np.asarray(v, dtype=np.float32)) for k, v in state_items]
                f.write(struct.pack("<I", len(params)))
                for key, arr in params:
                    name_bytes = key.encode()
                    f.write(struct.pack("<I", len(name_bytes)))
                    f.write(name_bytes)
                    f.write(struct.pack("<I", arr.ndim))
                    for dim in arr.shape:
                        f.write(struct.pack("<I", dim))
                    f.write(arr.tobytes())
        os.rename(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return path


def import_from_sou(path: str) -> SloNet:
    with open(path, "rb") as f:
        raw = f.read()

    if raw[:4] != SOU_MAGIC:
        raise ValueError("Invalid .soul: bad magic")

    version = struct.unpack("<I", raw[4:8])[0]
    json_len = struct.unpack("<I", raw[8:12])[0]
    # Strip null padding bytes that were added for 4-byte alignment
    meta_bytes = raw[12:12+json_len].rstrip(b"\x00")
    meta = json.loads(meta_bytes.decode())

    weights = {}
    lineage = meta.get("lineage", meta.get("base_model", "slonet"))
    soul_name = meta.get("soul_name", meta.get("name", "Slo"))
    system_prompt = meta.get("system_prompt", "")

    # Weight data starts after aligned JSON section
    weight_offset = 12 + json_len

    # Parse weight data after config JSON
    if version >= 3:
        # v3+ binary float32 format with shape info
        rem = raw[weight_offset:]
        if len(rem) >= 4:
            num_params = struct.unpack("<I", rem[:4])[0]
            pos = 4
            for _ in range(num_params):
                name_len = struct.unpack("<I", rem[pos:pos+4])[0]
                pos += 4
                name = rem[pos:pos+name_len].decode("utf-8")
                pos += name_len
                ndim = struct.unpack("<I", rem[pos:pos+4])[0]
                pos += 4
                shape = tuple(struct.unpack("<I", rem[pos+4*i:pos+4*i+4])[0] for i in range(ndim))
                pos += 4 * ndim
                count = int(np.prod(shape))
                weights[name] = np.frombuffer(rem[pos:pos+count*4], dtype=np.float32).copy().reshape(shape)
                pos += count * 4
    else:
        # v1/v2 JSON weights or PyTorch ZIP
        pk_pos = raw.find(b"PK", weight_offset)
        if pk_pos > 0:
            weights = _load_pytorch_zip_weights(raw[pk_pos:])
            if weights:
                lineage = "slolib-pytorch"
        else:
            rem = raw[weight_offset:]
            if len(rem) >= 4:
                wl = struct.unpack("<I", rem[:4])[0]
                if 0 < wl <= len(rem) - 4:
                    weights = json.loads(rem[4:4+wl].decode())

    # Detect SloTransformer from lineage or named weight keys
    is_transformer = (
        lineage == "soultransformer"
        or meta.get("metadata", {}).get("model_type") == "sloughgpt"
        or (weights and any(k.startswith("blocks.") for k in weights.keys()))
    )

    if is_transformer:
        md = meta.get("metadata", {}) or meta.get("soul_signature", {})
        net = SloTransformer(
            vocab_size=md.get("vocab_size", md.get("vocab", 256)),
            n_embed=md.get("n_embed", md.get("hidden_size", 384)),
            n_layer=md.get("n_layer", md.get("num_hidden_layers", 6)),
            n_head=md.get("n_head", md.get("num_attention_heads", 8)),
            block_size=md.get("block_size", 64),
            max_seq_len=md.get("max_seq_len", 2048),
            dropout=md.get("dropout", 0.1),
            use_rope=md.get("use_rope", True),
            tie_weights=md.get("tie_weights", True),
            soul_name=soul_name,
            soul_traits=meta.get("soul_traits", {}),
        )
        net.system_prompt = system_prompt
        net.metadata = md
        net._created_at = meta.get("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ"))
        net._step = meta.get("step", 0)
        if weights:
            net.load_state_dict(weights, strict=False)
        return net

    net = SloNet(soul_name=soul_name,
                  soul_traits=meta.get("soul_traits", {}),
                  system_prompt=system_prompt,
                  lineage=lineage,
                  metadata=meta)

    if weights and any(k.startswith("tok_emb.") for k in weights.keys()):
        net._rebuild_from_state_dict(weights)
    elif weights:
        _rebuild_net_from_params(net, weights)

    return net


def _rebuild_net_from_params(net: SloNet, weights: Dict[str, Any]) -> None:
    """Rebuild a SloNet that used SloEmbedding + SloLSTM layers from flat param weights.

    Expected param order: emb.weight, lstm.emb.weight, lstm.W_ih.w, lstm.W_ih.b,
    lstm.W_hh.w, lstm.W_hh.b, lstm.fc_out.w, lstm.fc_out.b [,
    lstm.W_ih2.w, lstm.W_ih2.b, lstm.W_hh2.w, lstm.W_hh2.b]
    """
    if not weights:
        return
    keys = sorted(weights.keys(), key=lambda k: int(k[1:]))
    arrays = [np.array(weights[k], dtype=np.float32) for k in keys]

    i = 0
    if arrays[i].ndim == 2:
        vocab, embed_dim = arrays[0].shape
        emb_layer = SloEmbedding(vocab, embed_dim, "embed")
        emb_layer.weight.data[:] = arrays[0]
        net.layers.append(emb_layer)
        i += 1
    else:
        return

    lstm_arrays = arrays[i:]
    num_layers = 2 if len(lstm_arrays) >= 11 else 1
    if len(lstm_arrays) < 7:
        return
    hidden_dim = lstm_arrays[2].shape[0] // 4
    if hidden_dim <= 0:
        return

    dropout = float(net.metadata.get("lstm_dropout", 0.0))
    lstm = SloLSTM(vocab, embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
    lstm.embedding.weight.data[:] = arrays[i]; i += 1
    lstm.W_ih.weight.data[:] = arrays[i]; i += 1
    lstm.W_ih.bias.data[:] = arrays[i]; i += 1
    lstm.W_hh.weight.data[:] = arrays[i]; i += 1
    lstm.W_hh.bias.data[:] = arrays[i]; i += 1
    lstm.fc_out.weight.data[:] = arrays[i]; i += 1
    lstm.fc_out.bias.data[:] = arrays[i]; i += 1
    if num_layers > 1:
        lstm.W_ih2.weight.data[:] = arrays[i]; i += 1
        lstm.W_ih2.bias.data[:] = arrays[i]; i += 1
        lstm.W_hh2.weight.data[:] = arrays[i]; i += 1
        lstm.W_hh2.bias.data[:] = arrays[i]; i += 1
    net.layers.append(lstm)


def _load_pytorch_zip_weights(zip_data: bytes) -> Dict[str, np.ndarray]:
    """Load weights from a PyTorch ZIP checkpoint using torch-free pt_loader."""
    try:
        from domains.infrastructure.pt_loader import load_pt_bytes as _load_pt
        return _load_pt(zip_data)
    except Exception as e:
        logger.warning("_load_pytorch_zip_weights failed: %s: %s", type(e).__name__, e)
        return {}


def souls_from_directory(dir_path) -> List[SloNet]:
    souls = []
    import logging
    _log = logging.getLogger(__name__)
    for p in Path(dir_path).glob("*.soul"):
        try:
            souls.append(import_from_sou(str(p)))
        except Exception as exc:
            _log.warning("Failed to load soul %s: %s", p.name, exc)
    return souls


# =============================================================================
# TRAIN FROM GPT STREAM
# =============================================================================

def train_char_lstm_from_gpt(gpt_fn, soul_name="Slo", epochs=10, temperature=0.8, lr=0.001, embed_dim=256, hidden_dim=512, on_step=None):
    charset = list(" abcdefghijklmnopqrstuvwxyz0123456789.,!?-'")
    stoi = {c:i for i,c in enumerate(charset)}; itos = {i:c for i,c in enumerate(charset)}
    unk = 0
    net = SloNet([SloEmbedding(len(charset), embed_dim), SloLSTM(len(charset), embed_dim, hidden_dim, num_layers=2, dropout=0.2)],
                  soul_name=soul_name, soul_traits={"warmth":0.5,"creativity":0.5,"curiosity":0.5,"confidence":0.5},
                  system_prompt=f"You are {soul_name}.", lineage="gpt2-teacher-distillation")
    opt = SloAdam(lr=lr)
    topics = ["What is consciousness?","Explain machine learning","Write a haiku about time","How do neural networks learn?","What makes humans unique?"]
    for ep in range(epochs):
        for topic in topics:
            resp = gpt_fn(topic, temperature)
            if not resp: continue
            text = (topic+" "+resp)[:256]
            ids = [stoi.get(c, unk) for c in text.lower() if c in stoi]
            if len(ids) < 8: continue
            for i in range(0, len(ids)-1, 16):
                xi = ids[i:i+32]; yi = ids[i+1:i+33]
                while len(xi) < 32: xi.append(unk)
                while len(yi) < 32: yi.append(unk)
                x = tensor([[xi]], requires_grad=True); y = tensor([[yi]])
                lstm_l = net.layers[1]
                lg, _ = lstm_l.forward(x, lstm_l.init_hidden())
                loss = cross_entropy(lg, y.reshape(-1))
                loss.backward(); opt.step(net.parameters())
                if on_step: on_step(ep*len(topics)+topics.index(topic), loss.data[()], ep)
    export_to_sou(net, f"models/auto-training/{soul_name}_{int(time.time())}.soul")
    return net


# =============================================================================
# SOUL TRANSFORMER — Native SloNet Decoder-Only Causal LM
# =============================================================================

class SloTransformer(SloNet):
    """Native SloNet decoder-only Transformer: embedding → blocks → norm → lm_head.

    Architecture: RoPE, RMSNorm, SwiGLU, KV-cache, GQA support.
    Drop-in replacement for SloughGPTModel (no PyTorch).
    """

    def __init__(
        self,
        vocab_size: int = 256,
        n_embed: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        n_kv_head: Optional[int] = None,
        block_size: int = 128,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        eps: float = 1e-5,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        tie_weights: bool = True,
        intermediate_size: Optional[int] = None,
        use_abs_pos_emb: bool = False,
        norm_type: str = "rms_norm",
        soul_name: str = "SloTransformer",
        soul_traits: Optional[Dict[str, float]] = None,
    ):
        dim_ff = intermediate_size or int(n_embed * 8 // 3)
        dim_ff = ((dim_ff + 63) // 64) * 64
        layers = []
        layers.append(SloEmbedding(vocab_size, n_embed, "tok_emb"))
        if dropout > 0:
            layers.append(SloDropout(dropout, "emb_drop"))
        for i in range(n_layer):
            layers.append(SloTransformerBlock(
                n_embed, n_head, n_kv_head=n_kv_head,
                dim_ff=dim_ff, use_rope=use_rope, max_seq_len=max_seq_len,
                rope_base=rope_base, dropout=0, eps=eps, norm_type=norm_type,
                name=f"blocks.{i}",
            ))
        NormCls = SloLayerNorm if norm_type == "layer_norm" else SloRMSNorm
        layers.append(NormCls(n_embed, eps, "norm"))
        layers.append(SloLinear(n_embed, vocab_size, "lm_head"))
        super().__init__(
            layers=layers,
            soul_name=soul_name,
            soul_traits=soul_traits or {"warmth": 0.5, "creativity": 0.5, "curiosity": 0.5, "confidence": 0.5},
            system_prompt="",
            lineage="soultransformer",
            metadata={
                "vocab_size": vocab_size,
                "n_embed": n_embed,
                "n_layer": n_layer,
                "n_head": n_head,
                "n_kv_head": n_kv_head or n_head,
                "block_size": block_size,
                "max_seq_len": max_seq_len,
                "dropout": dropout,
                "model_type": "sloughgpt",
            },
        )
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        self.tie_weights = tie_weights
        self._kv_caches: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None] * n_layer

        # Absolute positional embedding (GPT-2 style, optional)
        self.pos_emb: Optional[SloEmbedding] = None
        if use_abs_pos_emb:
            self.pos_emb = SloEmbedding(max_seq_len, n_embed, "pos_emb")

        if tie_weights:
            self.layers[-1].weight.data[:] = self.layers[0].weight.data.copy()

    @property
    def tok_emb(self) -> SloEmbedding:
        return self.layers[0]

    @property
    def blocks(self) -> List[SloLayer]:
        start = 2 if isinstance(self.layers[1], SloDropout) else 1
        end = -2  # skip norm and lm_head
        blocks = []
        for l in self.layers[start:end]:
            if isinstance(l, SloTransformerBlock):
                blocks.append(l)
        return blocks

    @property
    def norm(self) -> SloRMSNorm:
        for l in reversed(self.layers):
            if isinstance(l, SloRMSNorm):
                return l
        return self.layers[-2]

    def _tie_weights(self):
        """Tie the language model head weights to the token embeddings.
        This mirrors the common practice of weight tying in many transformer
        language models (e.g., GPT‑2, Qwen). If the shapes match we copy the
        embedding matrix into the lm_head weight and zero the lm_head bias.
        """
        try:
            if self.tok_emb.weight.shape == self.lm_head.weight.shape:
                # Copy the embedding weights directly
                self.lm_head.weight.data = self.tok_emb.weight.data.copy()
                # Zero bias if present
                if self.lm_head.bias is not None:
                    self.lm_head.bias = Tensor(np.zeros_like(self.lm_head.bias.data), requires_grad=False)
        except Exception as e:
            # Fail silently – tying is optional and should not break loading
            logger.debug("weight tying error: %s", e)

    def clear_kv_cache(self):
        self._kv_caches = [None] * self.n_layer

    def __call__(self, input_ids, targets=None, **kwargs):
        return self.forward(input_ids, targets=targets, **kwargs)

    def forward(self, input_ids, targets=None, use_cache=False, start_pos=0, mask=None, **kwargs):
        if isinstance(input_ids, np.ndarray):
            x = Tensor(input_ids.astype(np.int64))
        elif isinstance(input_ids, Tensor):
            x = input_ids
        else:
            if hasattr(input_ids, 'cpu'):
                input_ids = input_ids.cpu().detach().numpy()
            x = Tensor(np.array(input_ids, dtype=np.int64))
        x = self.layers[0].forward(x)
        if self.pos_emb is not None:
            seq_len = x.data.shape[1]
            pos = Tensor(np.arange(seq_len, dtype=np.int64).reshape(1, -1))
            x = x + self.pos_emb.forward(pos)
        seq_len = x.data.shape[1]
        if mask is None:
            causal = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
            mask = Tensor(causal, requires_grad=False)
        block_idx = 0
        for l in self.layers[1:-2]:
            if isinstance(l, SloDropout):
                x = l.forward(x)
            elif isinstance(l, SloTransformerBlock):
                kv_cache = self._kv_caches[block_idx] if use_cache else None
                x, kv = l.forward(x, mask=mask, kv_cache=kv_cache, start_pos=start_pos)
                if use_cache:
                    self._kv_caches[block_idx] = kv
                block_idx += 1
        x = self.layers[-2].forward(x)
        logits = self.layers[-1].forward(x)
        if targets is not None:
            if isinstance(targets, np.ndarray):
                t = targets.astype(np.int64)
            elif isinstance(targets, Tensor):
                t = targets.data.astype(np.int64)
            else:
                if hasattr(targets, 'cpu'):
                    targets = targets.cpu().detach().numpy()
                t = np.array(targets, dtype=np.int64)
            loss_t = cross_entropy(logits.reshape(-1, self.vocab_size), Tensor(t.reshape(-1)))
            return logits, loss_t
        return logits, None

    @no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token: int = 0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ):
        if isinstance(input_ids, np.ndarray):
            tokens = input_ids.copy()
        elif isinstance(input_ids, Tensor):
            tokens = input_ids.data.copy()
        else:
            if hasattr(input_ids, 'cpu'):
                input_ids = input_ids.cpu().detach().numpy()
            tokens = np.array(input_ids, dtype=np.int64)
        if tokens.ndim == 1:
            tokens = tokens.reshape(1, -1)
        self.clear_kv_cache()
        prompt_len = tokens.shape[1]
        total_len = min(prompt_len + max_new_tokens, self.max_seq_len)
        max_gen = total_len - prompt_len
        for step in range(max_gen):
            if step == 0:
                idx = tokens[:, -self.block_size:]
                pos = 0
                seq_len = idx.shape[1]
                causal = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
                step_mask = Tensor(causal, requires_grad=False)
            else:
                idx = tokens[:, -1:]
                pos = tokens.shape[1] - 1
                step_mask = None
            x = self.layers[0].forward(Tensor(idx.astype(np.int64)))
            if self.pos_emb is not None:
                seq = x.data.shape[1]
                p = Tensor(np.arange(pos, pos + seq, dtype=np.int64).reshape(1, -1))
                x = x + self.pos_emb.forward(p)
            block_idx = 0
            for l in self.layers[1:-2]:
                if isinstance(l, SloTransformerBlock):
                    x, kv = l.forward(x, mask=step_mask, start_pos=pos,
                                      kv_cache=self._kv_caches[block_idx])
                    self._kv_caches[block_idx] = kv
                    block_idx += 1
            x = self.layers[-2].forward(x)
            logits = self.layers[-1].forward(x)
            logit_data = logits.data[:, -1:, :]  # keep 3D for _sample_from_logits
            generated = tokens[:, prompt_len:].flatten()
            next_id = _sample_from_logits(
                logit_data, temperature=temperature,
                top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                generated_ids=generated,
                eos_token=eos_token if step < max_gen - 1 else None,
            )
            tokens = np.concatenate([tokens, np.array([[next_id]], dtype=np.int64)], axis=1)
            if next_id == eos_token:
                break
        self.clear_kv_cache()
        return Tensor(tokens)

    def generate_numpy(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token: int = 0,
    ) -> np.ndarray:
        """Fully inlined numpy generation — maximum inference speed.

        Optimizations over previous version:
        - KV cache pre-allocated to max capacity (no realloc per step)
        - QKV fused into single matmul (3→1 per layer)
        - Attention uses direct matmul instead of einsum (avoids parse overhead)
        - Norm uses fused mean+var (1 pass instead of 2) with reciprocal
        - Direct fancy indexing instead of np.take
        - Causal mask pre-allocated once
        - Output buffer pre-allocated (no concat per step)
        - GQA expand avoided when K_H == H
        - Flat tuple indexing (eliminates dict hash lookups in hot loop)
        - Pre-allocated scratch buffers (avoids per-step allocation)
        - Inlined greedy sampling (avoids function call + logits copy)
        """
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        prompt_len = input_ids.shape[1]
        total_len = min(prompt_len + max_new_tokens, self.max_seq_len)
        max_gen = total_len - prompt_len

        out_buf = np.empty((1, total_len), dtype=np.int64)
        out_buf[:, :prompt_len] = input_ids

        _use_kernels = _KERNELS_AVAILABLE
        _is_greedy = temperature < 1e-6 and top_p is None and repetition_penalty == 1.0

        # Detect quantization — if any SloLinear has _quant_info, use quantized path.
        _is_quantized = False
        for l in self.layers[1:-2]:
            if isinstance(l, SloTransformerBlock):
                if getattr(l.attn.W_q, '_quant_info', None) is not None:
                    _is_quantized = True
                    break

        # Flatten block weights into parallel lists — eliminates dict hash lookups.
        # Each block is indexed by integer; inner loop uses direct list indexing.
        n_an_w = []; n_an_b = []; n_an_e = []
        n_fn_w = []; n_fn_b = []; n_fn_e = []
        m_wqkv = []; m_bqkv = []; m_wo = []; m_bo = []
        m_w13 = []; m_b13 = []; m_w2 = []; m_b2 = []
        _nkv = []
        # Quantized path: store SloLinear module references for forward_numpy()
        if _is_quantized:
            q_wq = []; q_wk = []; q_wv = []; q_wo = []
            q_w1 = []; q_w3 = []; q_w2 = []
        for l in self.layers[1:-2]:
            if isinstance(l, SloTransformerBlock):
                b = l
                has_ln = isinstance(b.attn_norm, SloLayerNorm)
                n_an_w.append(b.attn_norm.weight.data)
                n_an_b.append(b.attn_norm.bias.data if has_ln else None)
                n_an_e.append(b.attn_norm.eps)
                n_fn_w.append(b.ff_norm.weight.data)
                n_fn_b.append(b.ff_norm.bias.data if has_ln else None)
                n_fn_e.append(b.ff_norm.eps)
                if _is_quantized:
                    q_wq.append(b.attn.W_q); q_wk.append(b.attn.W_k)
                    q_wv.append(b.attn.W_v); q_wo.append(b.attn.W_o)
                    q_w1.append(b.ff.w1); q_w3.append(b.ff.w3)
                    q_w2.append(b.ff.w2)
                    _nkv.append(b.attn.n_kv_head)
                    m_wqkv.append(None); m_bqkv.append(None)
                    m_wo.append(None); m_bo.append(None)
                    m_w13.append(None); m_b13.append(None)
                    m_w2.append(None); m_b2.append(None)
                else:
                    wqkv = np.concatenate([b.attn.W_q.weight.data, b.attn.W_k.weight.data,
                                           b.attn.W_v.weight.data], axis=0)
                    bq = b.attn.W_q.bias.data if b.attn.W_q.use_bias else None
                    bk = b.attn.W_k.bias.data if b.attn.W_k.use_bias else None
                    bv = b.attn.W_v.bias.data if b.attn.W_v.use_bias else None
                    bqkv = np.concatenate([bq, bk, bv]) if bq is not None else None
                    w13 = np.concatenate([b.ff.w1.weight.data.T, b.ff.w3.weight.data.T], axis=1)
                    b1 = b.ff.w1.bias.data if b.ff.w1.use_bias else None
                    b3 = b.ff.w3.bias.data if b.ff.w3.use_bias else None
                    b13 = np.concatenate([b1, b3]) if b1 is not None else None
                    _nkv.append(b.attn.n_kv_head)
                    m_wqkv.append(wqkv); m_bqkv.append(bqkv)
                    m_wo.append(b.attn.W_o.weight.data)
                    m_bo.append(b.attn.W_o.bias.data if b.attn.W_o.use_bias else None)
                    m_w13.append(w13); m_b13.append(b13)
                    m_w2.append(b.ff.w2.weight.data)
                    m_b2.append(b.ff.w2.bias.data if b.ff.w2.use_bias else None)

        n_blocks = len(m_wqkv)
        tok_emb_w = self.layers[0].weight.data
        pos_emb_w = self.pos_emb.weight.data if self.pos_emb is not None else None
        pos_emb_n = self.pos_emb.num_embeddings if self.pos_emb is not None else 0
        lm_head_mod = self.layers[-1]  # SloLinear

        norm_layer = self.layers[-2]
        norm_has_bias = isinstance(norm_layer, SloLayerNorm)
        norm_w = norm_layer.weight.data
        norm_b = norm_layer.bias.data if norm_has_bias else None
        norm_eps = norm_layer.eps
        lm_w = self.layers[-1].weight.data

        # Extract E (head_dim) and H (n_heads) from the first transformer block
        _first_block = None
        for l in self.layers[1:-2]:
            if isinstance(l, SloTransformerBlock):
                _first_block = l
                break
        E = _first_block.attn.head_dim
        H = _first_block.attn.n_heads
        K_H = _nkv[0]
        # Compute _ff_dim from the first block's w1 weight shape (works in both paths)
        _first_w1 = None
        for l in self.layers[1:-2]:
            if isinstance(l, SloTransformerBlock):
                _first_w1 = l.ff.w1
                break
        _ff_dim = _first_w1.weight.shape[0]  # w1 weight: (ff_dim, E), shape[0] = ff_dim
        scale = np.float32(1.0 / math.sqrt(E))
        _clip_max = np.int64(tok_emb_w.shape[0] - 1)
        _pos_clip_max = np.int64(pos_emb_n - 1 if pos_emb_n > 0 else 0)
        _use_gqa = K_H < H
        _gqa_reps = H // K_H if _use_gqa else 0
        _he = H * E
        _khe = K_H * E

        # Pre-allocate KV cache
        kv_buf_k = [np.zeros((1, total_len, _nkv[i], E), dtype=np.float32) for i in range(n_blocks)]
        kv_buf_v = [np.zeros((1, total_len, _nkv[i], E), dtype=np.float32) for i in range(n_blocks)]
        kv_len = [0] * n_blocks

        # RoPE — detect and pre-compute cos/sin cache
        _use_rope = False
        _rope_inv_freq = None
        for l in self.layers[1:-2]:
            if isinstance(l, SloTransformerBlock) and l.attn.use_rope:
                _use_rope = True
                _rope_inv_freq = l.attn.rope.inv_freq.data.copy()
                break
        _rope_cos = None
        _rope_sin = None
        if _use_rope:
            t = np.arange(0, total_len, dtype=np.float32)
            freqs = np.outer(t, _rope_inv_freq)
            emb = np.concatenate([freqs, freqs], axis=-1)
            _rope_cos = np.cos(emb).astype(np.float32)
            _rope_sin = np.sin(emb).astype(np.float32)

        causal_mask = np.triu(np.full((self.block_size, self.block_size), -1e9, dtype=np.float32), k=1)

        _use_bias_bqkv = m_bqkv[0] is not None
        _use_bias_bo = m_bo[0] is not None
        _use_bias_b13 = m_b13[0] is not None
        _use_bias_b2 = m_b2[0] is not None

        for step in range(max_gen):
            if step == 0:
                idx = out_buf[:, :prompt_len]
                pos = 0
                seq_len = prompt_len
            else:
                idx = out_buf[:, step + prompt_len - 1:step + prompt_len]
                pos = step + prompt_len - 1
                seq_len = 1

            # Embedding
            clipped = np.clip(idx, 0, _clip_max)
            x = tok_emb_w[clipped].astype(np.float32)

            if pos_emb_w is not None:
                p = np.arange(pos, pos + seq_len, dtype=np.int64).reshape(1, -1)
                x = x + pos_emb_w[np.clip(p, 0, _pos_clip_max)]

            # --- Transformer blocks (flat tuple indexing) ---
            for bi in range(n_blocks):
                # Attn norm
                if _use_kernels:
                    h = _nb_layernorm(x, n_an_w[bi], n_an_b[bi], n_an_e[bi])
                else:
                    mu = x.mean(axis=-1, keepdims=True)
                    centered = x - mu
                    var = (centered * centered).mean(axis=-1, keepdims=True)
                    h = centered * (n_an_w[bi] * np.float32(1.0) / np.sqrt(var + n_an_e[bi]))
                    if n_an_b[bi] is not None:
                        h = h + n_an_b[bi]

                # QKV projection
                if _is_quantized:
                    q = q_wq[bi].forward_numpy(h)
                    k = q_wk[bi].forward_numpy(h)
                    v = q_wv[bi].forward_numpy(h)
                    q = q.reshape(1, seq_len, H, E)
                    k = k.reshape(1, seq_len, K_H, E)
                    v = v.reshape(1, seq_len, K_H, E)
                else:
                    qkv = h @ m_wqkv[bi].T
                    if _use_bias_bqkv:
                        qkv = qkv + m_bqkv[bi]
                    q = qkv[:, :, :_he].reshape(1, seq_len, H, E)
                    k = qkv[:, :, _he:_he+_khe].reshape(1, seq_len, K_H, E)
                    v = qkv[:, :, _he+_khe:].reshape(1, seq_len, K_H, E)

                # Apply RoPE
                if _use_rope and _rope_cos is not None:
                    q_cos = _rope_cos[pos:pos+seq_len]  # (seq_len, head_dim)
                    q_sin = _rope_sin[pos:pos+seq_len]
                    q = q * q_cos + np.concatenate([-q[..., E//2:], q[..., :E//2]], axis=-1) * q_sin
                    k = k * q_cos + np.concatenate([-k[..., E//2:], k[..., :E//2]], axis=-1) * q_sin

                # KV cache
                new_len = kv_len[bi] + seq_len
                kv_buf_k[bi][:, kv_len[bi]:new_len] = k
                kv_buf_v[bi][:, kv_len[bi]:new_len] = v
                kv_len[bi] = new_len
                k = kv_buf_k[bi][:, :new_len]
                v = kv_buf_v[bi][:, :new_len]

                if _use_gqa:
                    k = np.repeat(k, _gqa_reps, axis=2)
                    v = np.repeat(v, _gqa_reps, axis=2)

                # Attention
                scores = np.einsum('bnhd,bmhd->bhnm', q, k) * scale
                if step == 0 and seq_len > 1:
                    scores = scores + causal_mask[:seq_len, :new_len]
                attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
                attn = attn / attn.sum(axis=-1, keepdims=True)
                ao = np.einsum('bhnm,bmhd->bnhd', attn, v).reshape(1, seq_len, _he)

                # Output projection
                if _is_quantized:
                    ao = q_wo[bi].forward_numpy(ao)
                else:
                    ao = ao @ m_wo[bi].T
                    if _use_bias_bo:
                        ao = ao + m_bo[bi]
                x = x + ao

                # FFN norm
                if _use_kernels:
                    h = _nb_layernorm(x, n_fn_w[bi], n_fn_b[bi], n_fn_e[bi])
                else:
                    mu = x.mean(axis=-1, keepdims=True)
                    centered = x - mu
                    var = (centered * centered).mean(axis=-1, keepdims=True)
                    h = centered * (n_fn_w[bi] * np.float32(1.0) / np.sqrt(var + n_fn_e[bi]))
                    if n_fn_b[bi] is not None:
                        h = h + n_fn_b[bi]

                # FFN
                if _is_quantized:
                    h1 = q_w1[bi].forward_numpy(h)
                    h3 = q_w3[bi].forward_numpy(h)
                    if _use_kernels:
                        h = _nb_swi_glu_mul(h1, h3)
                    else:
                        s = np.float32(0.5) * h1 * (np.float32(1.0) + np.tanh(np.float32(0.7978845608) * (h1 + np.float32(0.044715) * h1**3)))
                        h = s * h3
                    h = q_w2[bi].forward_numpy(h)
                else:
                    h13 = h @ m_w13[bi]
                    if _use_bias_b13:
                        h13 = h13 + m_b13[bi]
                    h1 = h13[..., :_ff_dim]
                    h3 = h13[..., _ff_dim:]
                    if _use_kernels:
                        h = _nb_swi_glu_mul(h1, h3)
                    else:
                        s = np.float32(0.5) * h1 * (np.float32(1.0) + np.tanh(np.float32(0.7978845608) * (h1 + np.float32(0.044715) * h1**3)))
                        h = s * h3
                    h = h @ m_w2[bi].T
                    if _use_bias_b2:
                        h = h + m_b2[bi]
                x = x + h

            # Final norm
            if _use_kernels:
                x = _nb_layernorm(x, norm_w, norm_b, norm_eps)
            else:
                mu = x.mean(axis=-1, keepdims=True)
                centered = x - mu
                var = (centered * centered).mean(axis=-1, keepdims=True)
                x = centered * (norm_w * np.float32(1.0) / np.sqrt(var + norm_eps))
                if norm_has_bias:
                    x = x + norm_b

            # LM head
            if _is_quantized:
                logits = lm_head_mod.forward_numpy(x[:, -1, :])
            else:
                logits = x[:, -1, :] @ lm_w.T

            # Inlined greedy sampling (avoids function call + logits.copy + penalty checks)
            if _is_greedy:
                next_id = int(np.argmax(logits[0]))
            else:
                next_id = _sample_from_logits(
                    logits, temperature=temperature,
                    top_k=top_k, top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    generated_ids=out_buf[:, prompt_len:step + prompt_len].flatten(),
                    eos_token=eos_token if step < max_gen - 1 else None,
                )
            out_buf[0, prompt_len + step] = next_id
            if next_id == eos_token:
                return out_buf[:, :prompt_len + step + 1]

        return out_buf

    def state_dict(self) -> Dict[str, np.ndarray]:
        result = {}
        for name, param in self._named_parameters():
            result[name] = param.data.copy()
        return result

    def named_parameters(self, prefix="") -> List[Tuple[str, Tensor]]:
        return self._named_parameters(prefix=prefix)

    def _named_parameters(self, prefix="") -> List[Tuple[str, Tensor]]:
        named = []
        layer_names = ["tok_emb"]
        if len(self.layers) > 2 + self.n_layer and isinstance(self.layers[1], SloDropout):
            layer_names.append("emb_drop")
        layer_names += [f"blocks.{i}" for i in range(self.n_layer)] + ["norm", "lm_head"]
        for lname, layer in zip(layer_names, self.layers):
            if isinstance(layer, SloEmbedding):
                named.append((f"{prefix}{lname}.weight", layer.weight))
            elif isinstance(layer, SloDropout):
                continue
            elif isinstance(layer, SloTransformerBlock):
                named.append((f"{prefix}{lname}.attn_norm.weight", layer.attn_norm.weight))
                if isinstance(layer.attn_norm, SloLayerNorm):
                    named.append((f"{prefix}{lname}.attn_norm.bias", layer.attn_norm.bias))
                named.extend(_named_mha(f"{prefix}{lname}.attn", layer.attn))
                named.append((f"{prefix}{lname}.ff_norm.weight", layer.ff_norm.weight))
                if isinstance(layer.ff_norm, SloLayerNorm):
                    named.append((f"{prefix}{lname}.ff_norm.bias", layer.ff_norm.bias))
                named.extend(_named_ff(f"{prefix}{lname}.ff", layer.ff))
            elif isinstance(layer, SloRMSNorm):
                named.append((f"{prefix}{lname}.weight", layer.weight))
            elif isinstance(layer, SloLayerNorm):
                named.append((f"{prefix}{lname}.weight", layer.weight))
                named.append((f"{prefix}{lname}.bias", layer.bias))
            elif isinstance(layer, SloLinear):
                named.append((f"{prefix}{lname}.weight", layer.weight))
        if self.pos_emb is not None:
            named.append((f"{prefix}pos_emb.weight", self.pos_emb.weight))
        return named

    def load_state_dict(self, state_dict: Dict[str, np.ndarray], strict: bool = True):
        param_map = dict(self._named_parameters())
        loaded, missing = set(), []
        for key, arr in state_dict.items():
            arr = np.array(arr, dtype=np.float32)
            clean = key
            for old, new in [("tok_emb.", "tok_emb."), ("lm_head.", "lm_head."),
                             ("norm.", "norm."), ("blocks.", "blocks.")]:
                pass
            if clean in param_map:
                p = param_map[clean]
                if p.data.shape == arr.shape:
                    p.data[:] = arr
                    loaded.add(clean)
                elif p.data.ndim == 2 and arr.ndim == 2 and p.data.shape[1] == arr.shape[1]:
                    p.data[:arr.shape[0]] = arr[:p.data.shape[0]]
                    loaded.add(clean)
                elif p.data.ndim == 1 and arr.ndim == 1:
                    min_d = min(p.data.shape[0], arr.shape[0])
                    p.data[:min_d] = arr[:min_d]
                    loaded.add(clean)
            else:
                alt = clean.replace(".weight", ".weight")
                alt_layer_names = ["tok_emb"]
                if len(self.layers) > 2 + self.n_layer and isinstance(self.layers[1], SloDropout):
                    alt_layer_names.append("emb_drop")
                alt_layer_names += [f"blocks.{i}" for i in range(self.n_layer)] + ["norm", "lm_head"]
                for lname, layer in zip(alt_layer_names, self.layers):
                    if isinstance(layer, SloLinear) and not isinstance(layer, type(None)):
                        if clean.endswith(".weight") and clean.replace(".weight", "") == lname:
                            arr_2d = arr.reshape(layer.weight.data.shape)
                            layer.weight.data[:] = arr_2d
                            loaded.add(clean)
        if strict and len(loaded) < len(state_dict):
            missing = [k for k in state_dict if k not in loaded]
        return missing

    def to(self, device):
        return self

    def train(self, mode=True):
        for l in self.layers:
            if isinstance(l, SloLayer):
                l.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _named_mha(prefix: str, mha: SloMultiHeadAttention) -> List[Tuple[str, Tensor]]:
    named = [
        (f"{prefix}.q_proj.weight", mha.W_q.weight),
        (f"{prefix}.k_proj.weight", mha.W_k.weight),
        (f"{prefix}.v_proj.weight", mha.W_v.weight),
        (f"{prefix}.o_proj.weight", mha.W_o.weight),
    ]
    if mha.W_q.use_bias:
        named.append((f"{prefix}.q_proj.bias", mha.W_q.bias))
    if mha.W_k.use_bias:
        named.append((f"{prefix}.k_proj.bias", mha.W_k.bias))
    if mha.W_v.use_bias:
        named.append((f"{prefix}.v_proj.bias", mha.W_v.bias))
    if mha.W_o.use_bias:
        named.append((f"{prefix}.o_proj.bias", mha.W_o.bias))
    return named


def _named_ff(prefix: str, ff: SloFeedForward) -> List[Tuple[str, Tensor]]:
    named = [
        (f"{prefix}.w1.weight", ff.w1.weight),
        (f"{prefix}.w2.weight", ff.w2.weight),
        (f"{prefix}.w3.weight", ff.w3.weight),
    ]
    if ff.w1.use_bias:
        named.append((f"{prefix}.w1.bias", ff.w1.bias))
    if ff.w2.use_bias:
        named.append((f"{prefix}.w2.bias", ff.w2.bias))
    if ff.w3.use_bias:
        named.append((f"{prefix}.w3.bias", ff.w3.bias))
    return named


def _named_transformer_block(prefix: str, block: SloTransformerBlock) -> List[Tuple[str, Tensor]]:
    named = []
    named.append((f"{prefix}.attn_norm.weight", block.attn_norm.weight))
    named.extend(_named_mha(f"{prefix}.attn", block.attn))
    named.append((f"{prefix}.ff_norm.weight", block.ff_norm.weight))
    named.extend(_named_ff(f"{prefix}.ff", block.ff))
    return named


def train_soul_transformer(gpt_fn, soul_name="Slo", epochs=10, temperature=0.8, lr=0.001,
                           vocab_size=256, n_embed=128, n_layer=4, n_head=4, on_step=None):
    charset = list(" abcdefghijklmnopqrstuvwxyz0123456789.,!?-'")
    stoi = {c:i for i,c in enumerate(charset)}; itos = {i:c for i,c in enumerate(charset)}
    unk = 0
    net = SloTransformer(
        vocab_size=len(charset), n_embed=n_embed, n_layer=n_layer, n_head=n_head,
        block_size=64, max_seq_len=128, dropout=0.1, use_rope=True,
        soul_name=soul_name,
    )
    opt = SloAdam(lr=lr)
    topics = ["What is consciousness?", "Explain machine learning", "Write a haiku about time",
              "How do neural networks learn?", "What makes humans unique?"]
    for ep in range(epochs):
        for topic in topics:
            resp = gpt_fn(topic, temperature)
            if not resp:
                continue
            text = (topic + " " + resp)[:128]
            ids = [stoi.get(c, unk) for c in text.lower() if c in stoi]
            if len(ids) < 8:
                continue
            for i in range(0, len(ids) - 1, 16):
                xi = ids[i:i+32]; yi = ids[i+1:i+33]
                while len(xi) < 32: xi.append(unk)
                while len(yi) < 32: yi.append(unk)
                x = tensor([xi], requires_grad=True); y = tensor([yi])
                logits, loss = net.forward(x, y)
                if loss is None:
                    continue
                loss.backward()
                opt.step(net.parameters())
                if on_step:
                    on_step(ep * len(topics) + topics.index(topic), loss.data[()], ep)
    export_to_sou(net, f"models/auto-training/{soul_name}_{int(time.time())}.soul")
    return net


# =============================================================================
# ADDITIONAL LOSS FUNCTIONS
# =============================================================================


def log_softmax(x, dim=-1):
    """Numerically stable log-softmax with backward."""
    xd = x.data if isinstance(x, Tensor) else x
    mx = xd.max(axis=dim, keepdims=True)
    lp = xd - mx - np.log(np.exp(xd - mx).sum(axis=dim, keepdims=True))
    if isinstance(x, Tensor):
        out = Tensor(lp, requires_grad=x.requires_grad, _children=(x,))
        if out.requires_grad and x.requires_grad: x._consumers.append(out)
        def bk(g):
            if x.requires_grad:
                probs = np.exp(lp)
                x.grad = Tensor(g - probs * g.sum(axis=dim, keepdims=True))
        out._backward_fn = bk
        def fwd(t_x):
            if t_x is None: return np.zeros_like(lp)
            probs = np.exp(lp)
            return t_x - probs * t_x.sum(axis=dim, keepdims=True)
        out._forward_fn = fwd
        return out
    return lp


def kl_div_loss(input_log_prob, target_prob, reduction="batchmean"):
    """KL divergence D_KL(target || input) with backward.

    Computes Σ target * (log(target) - input) where input = log Q, target = P.
    This is the same formula as torch.nn.KLDivLoss.

    input_log_prob: log-probabilities (from log_softmax, i.e. log Q)
    target_prob: probabilities (from softmax, i.e. P)
    """
    ilp = input_log_prob.data if isinstance(input_log_prob, Tensor) else input_log_prob
    tp = target_prob.data if isinstance(target_prob, Tensor) else target_prob
    tp_safe = np.where(tp < 1e-15, 1e-15, tp)
    log_tp = np.log(tp_safe)
    loss_data = tp * (log_tp - ilp)
    kld = loss_data.sum(axis=-1)
    if reduction == "batchmean":
        kld = kld.mean()
    elif reduction == "sum":
        kld = kld.sum()
    else:
        kld = kld.mean()
    out = Tensor(kld, requires_grad=True, _children=(input_log_prob, target_prob) if isinstance(input_log_prob, Tensor) else ())
    def bk(g):
        if isinstance(input_log_prob, Tensor) and input_log_prob.requires_grad:
            g_inp = -tp
            g_inp = g_inp / ilp.shape[0] if reduction == "batchmean" else g_inp
            input_log_prob.grad = Tensor(g_inp * g)
    out._backward_fn = bk; return out


def normalize(x, p=2, dim=1):
    """L-p normalization along a dimension."""
    xd = x.data if isinstance(x, Tensor) else x
    norm = np.linalg.norm(xd, ord=p, axis=dim, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)
    nd = xd / norm
    if isinstance(x, Tensor):
        out = Tensor(nd, requires_grad=x.requires_grad, _children=(x,))
        def bk(g):
            if x.requires_grad:
                x.grad = Tensor(g / norm)
        out._backward_fn = bk; return out
    return nd


def pairwise_distance(x1, x2):
    """Pairwise Euclidean distance between rows of x1 and x2."""
    x1d = x1.data if isinstance(x1, Tensor) else x1
    x2d = x2.data if isinstance(x2, Tensor) else x2
    diff = x1d - x2d
    dist = np.sqrt(np.sum(diff**2, axis=-1) + 1e-8)
    if isinstance(x1, Tensor) or isinstance(x2, Tensor):
        _x1 = x1 if isinstance(x1, Tensor) else Tensor(x1d, requires_grad=False)
        _x2 = x2 if isinstance(x2, Tensor) else Tensor(x2d, requires_grad=False)
        out = Tensor(dist, requires_grad=True, _children=(_x1, _x2))
        def bk(g):
            if _x1.requires_grad:
                _x1.grad = Tensor(diff / (dist[..., np.newaxis] + 1e-8) * g[..., np.newaxis])
            if _x2.requires_grad:
                _x2.grad = Tensor(-diff / (dist[..., np.newaxis] + 1e-8) * g[..., np.newaxis])
        out._backward_fn = bk; return out
    return dist


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def argmax(x, dim=-1):
    """Returns indices of maximum values along a dimension."""
    xd = x.data if isinstance(x, Tensor) else np.array(x)
    return Tensor(np.argmax(xd, axis=dim).astype(np.int64), requires_grad=False)


def argmin(x, dim=-1):
    """Returns indices of minimum values along a dimension."""
    xd = x.data if isinstance(x, Tensor) else np.array(x)
    return Tensor(np.argmin(xd, axis=dim).astype(np.int64), requires_grad=False)


def squeeze(x, dim=None):
    """Remove single-dimensional entries from the shape."""
    xd = x.data if isinstance(x, Tensor) else np.array(x)
    if dim is not None:
        nd = np.squeeze(xd, axis=dim)
    else:
        nd = np.squeeze(xd)
    if isinstance(x, Tensor):
        return Tensor(nd, requires_grad=x.requires_grad, _children=(x,))
    return nd


def unsqueeze(x, dim):
    """Insert a new dimension at the specified position."""
    xd = x.data if isinstance(x, Tensor) else np.array(x)
    nd = np.expand_dims(xd, axis=dim)
    if isinstance(x, Tensor):
        return Tensor(nd, requires_grad=x.requires_grad, _children=(x,))
    return nd


def cat(tensors, dim=0):
    """Concatenate tensors along a dimension (alias for concatenate)."""
    return concatenate(tensors, dim=dim)


def eye(n, m=None):
    """Create an identity matrix."""
    if m is None: m = n
    return Tensor(np.eye(n, m, dtype=np.float32), requires_grad=False)


# =============================================================================
# SOUL DATASET / DATALOADER
# =============================================================================


class SloDataset:
    """Base dataset class (analogous to torch.utils.data.Dataset)."""

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class SloDataLoader:
    """Basic data loader with batching and shuffling (analogous to torch.utils.data.DataLoader).

    Args:
        dataset: SloDataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle indices each epoch
        collate_fn: Optional function to collate batch items
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, collate_fn=None, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self._idx = 0
        self._indices = None

    def __len__(self):
        n = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size != 0:
            n += 1
        return max(1, n)

    def __iter__(self):
        self._idx = 0
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)
        self._indices = indices
        return self

    def __next__(self):
        if self._idx >= len(self._indices):
            raise StopIteration
        batch_indices = self._indices[self._idx:self._idx + self.batch_size]
        self._idx += self.batch_size
        if len(batch_indices) < self.batch_size and self.drop_last:
            raise StopIteration
        batch = [self.dataset[i] for i in batch_indices]
        if self.collate_fn:
            return self.collate_fn(batch)
        return batch

    def reset(self):
        """Reset for a new epoch."""
        self._idx = 0
        self._indices = None


# =============================================================================
# LR SCHEDULERS
# =============================================================================


class SloLRScheduler:
    """Base LR scheduler (analogous to torch.optim.lr_scheduler._LRScheduler).

    Works with SloSGD / SloAdam via the .lr attribute on the optimizer.
    """

    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        if hasattr(optimizer, 'lr'):
            self.base_lrs = [optimizer.lr]
        elif optimizer.param_groups:
            self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        else:
            self.base_lrs = [0.0]
        self.last_epoch = last_epoch
        if last_epoch == -1:
            self.step()

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        new_lrs = self.get_lr()
        if new_lrs:
            self.optimizer.lr = new_lrs[0]
        self._last_lrs = new_lrs

    def get_last_lr(self):
        if hasattr(self, '_last_lrs') and self._last_lrs:
            return self._last_lrs
        return self.get_lr()

    def state_dict(self):
        return {"last_epoch": self.last_epoch, "base_lrs": self.base_lrs}

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict["last_epoch"]
        self.base_lrs = state_dict["base_lrs"]


class SloStepLR(SloLRScheduler):
    """Decays LR by gamma every step_size epochs."""

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (self.gamma ** (self.last_epoch // self.step_size)) for base_lr in self.base_lrs]


class SloCosineAnnealingLR(SloLRScheduler):
    """Cosine annealing LR scheduler."""

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.T_max:
            return [self.eta_min for _ in self.base_lrs]
        cos_val = math.cos(math.pi * self.last_epoch / self.T_max)
        return [self.eta_min + (base_lr - self.eta_min) * (1 + cos_val) / 2 for base_lr in self.base_lrs]


class SloReduceLROnPlateau:
    """Reduce LR when a metric has stopped improving.

    Analogous to torch.optim.lr_scheduler.ReduceLROnPlateau.
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.best = None
        self.mode_worse = None
        self.last_lr = optimizer.lr
        if mode == 'min':
            self.best = float('inf')
            self.mode_worse = float('inf')
        else:
            self.best = -float('inf')
            self.mode_worse = -float('inf')

    def _is_better(self, current, best):
        if self.threshold_mode == 'rel':
            diff = best * (1 - self.threshold) if self.mode == 'min' else best * (1 + self.threshold)
        else:
            diff = best - self.threshold if self.mode == 'min' else best + self.threshold
        return current < diff if self.mode == 'min' else current > diff

    def step(self, metrics):
        current = float(metrics) if hasattr(metrics, '__float__') else metrics
        if self.best is None or self.best == float('inf') or self.best == -float('inf'):
            self.best = current
            self.num_bad_epochs = 0
        elif self._is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
            self.optimizer.lr = new_lr
            self.last_lr = new_lr

    def state_dict(self):
        return {"best": self.best, "num_bad_epochs": self.num_bad_epochs,
                "cooldown_counter": self.cooldown_counter, "last_lr": self.last_lr}

    def load_state_dict(self, state_dict):
        self.best = state_dict["best"]
        self.num_bad_epochs = state_dict["num_bad_epochs"]
        self.cooldown_counter = state_dict["cooldown_counter"]
        self.last_lr = state_dict["last_lr"]


class WarmupCosineScheduler(SloLRScheduler):
    """Cosine annealing with linear warmup (same as torch-independent version in lr_schedulers.py)."""

    def __init__(self, optimizer, warmup_steps=0, total_steps=10000,
                 min_lr=1e-6, num_cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.num_cycles = num_cycles
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        progress = float(self.last_epoch - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * self.num_cycles * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]


class PolynomialDecayScheduler(SloLRScheduler):
    """Polynomial learning rate decay."""

    def __init__(self, optimizer, total_steps=10000, min_lr=0.0, power=1.0, last_epoch=-1):
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.total_steps:
            return [self.min_lr for _ in self.base_lrs]
        decay_ratio = (self.last_epoch / self.total_steps) ** self.power
        return [base_lr * (1 - decay_ratio) + self.min_lr * decay_ratio for base_lr in self.base_lrs]


class LinearWarmupScheduler(SloLRScheduler):
    """Linear warmup then hold or decay."""

    def __init__(self, optimizer, warmup_steps=500, base_lr=1e-4, hold_steps=0,
                 decay_type="none", min_lr=0.0, total_steps=None, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.hold_steps = hold_steps
        self.decay_type = decay_type
        self.min_lr = min_lr
        self.total_steps = total_steps or warmup_steps + hold_steps + 10000
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            factor = self.last_epoch / max(1, self.warmup_steps)
            return [self.base_lr * factor for _ in self.base_lrs]
        elif self.last_epoch < self.warmup_steps + self.hold_steps:
            return self.base_lrs
        else:
            if self.decay_type == "cosine":
                progress = min(1.0, (self.last_epoch - self.warmup_steps - self.hold_steps) /
                               max(1, self.total_steps - self.warmup_steps - self.hold_steps))
                cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
                return [self.min_lr + (self.base_lr - self.min_lr) * cosine_factor for _ in self.base_lrs]
            elif self.decay_type == "linear":
                progress = min(1.0, (self.last_epoch - self.warmup_steps - self.hold_steps) /
                               max(1, self.total_steps - self.warmup_steps - self.hold_steps))
                return [self.base_lr * (1 - progress) + self.min_lr * progress for _ in self.base_lrs]
            else:
                return self.base_lrs


class SloConstantLR(SloLRScheduler):
    """Constant LR (identity scheduler)."""

    def get_lr(self):
        return self.base_lrs


class SloOneCycleLR(SloLRScheduler):
    """One-cycle LR schedule: warmup to max_lr then decay."""

    def __init__(self, optimizer, max_lr, total_steps=10000, pct_start=0.1,
                 anneal_strategy="cos", div_factor=25.0, final_div_factor=1e4, last_epoch=-1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        total = self.total_steps
        if total == 0: return self.base_lrs
        phase = step / total
        if phase < self.pct_start:
            factor = phase / self.pct_start
            return [base_lr * self.div_factor * factor for base_lr in self.base_lrs]
        else:
            progress = (phase - self.pct_start) / (1 - self.pct_start)
            if self.anneal_strategy == "cos":
                cos_val = 0.5 * (1 + math.cos(math.pi * progress))
                decay_factor = (1 - 1 / self.final_div_factor) * cos_val + 1 / self.final_div_factor
            else:
                decay_factor = 1 - progress * (1 - 1 / self.final_div_factor)
            return [base_lr * self.div_factor * decay_factor for base_lr in self.base_lrs]


class SloCyclicLR(SloLRScheduler):
    """Cyclic LR with triangular/triangular2 mode."""

    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None,
                 mode="triangular2", gamma=0.5, last_epoch=-1):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        cycle_len = self.step_size_up + self.step_size_down
        cycle_pos = self.last_epoch % cycle_len
        if cycle_pos < self.step_size_up:
            factor = cycle_pos / max(1, self.step_size_up)
        else:
            factor = 1 - (cycle_pos - self.step_size_up) / max(1, self.step_size_down)
        scale = 1.0
        cycle_num = self.last_epoch // cycle_len
        if self.mode == "triangular2":
            scale = 0.5 ** cycle_num
        return [self.base_lr + (self.max_lr - self.base_lr) * factor * scale for _ in self.base_lrs]


def create_scheduler(optimizer, scheduler_type, total_steps=None, warmup_steps=0,
                     min_lr=1e-6, max_lr=1e-3, **kwargs):
    """Factory function to create LR scheduler (torch-independent)."""
    scheduler_type = scheduler_type.lower()
    if scheduler_type == "none" or scheduler_type == "constant":
        return SloConstantLR(optimizer, last_epoch=-1)
    elif scheduler_type == "cosine":
        return WarmupCosineScheduler(optimizer, warmup_steps=warmup_steps,
                                     total_steps=total_steps or 10000, min_lr=min_lr,
                                     num_cycles=kwargs.get("num_cycles", 0.5))
    elif scheduler_type == "warmup":
        return LinearWarmupScheduler(optimizer, warmup_steps=warmup_steps, base_lr=max_lr,
                                     decay_type=kwargs.get("decay_type", "cosine"),
                                     min_lr=min_lr, total_steps=total_steps)
    elif scheduler_type == "onecycle":
        return SloOneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps or 10000,
                              pct_start=kwargs.get("pct_start", 0.1),
                              anneal_strategy=kwargs.get("anneal_strategy", "cos"),
                              div_factor=kwargs.get("div_factor", 25.0),
                              final_div_factor=kwargs.get("final_div_factor", 1e4))
    elif scheduler_type == "cyclic":
        return SloCyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr,
                            step_size_up=kwargs.get("step_size_up", 2000),
                            step_size_down=kwargs.get("step_size_down", None),
                            mode=kwargs.get("mode", "triangular2"),
                            gamma=kwargs.get("gamma", 0.5))
    elif scheduler_type == "polynomial":
        return PolynomialDecayScheduler(optimizer, total_steps=total_steps or 10000,
                                        min_lr=min_lr, power=kwargs.get("power", 1.0))
    elif scheduler_type == "step":
        return SloStepLR(optimizer, step_size=kwargs.get("step_size", 30),
                          gamma=kwargs.get("gamma", 0.1))
    elif scheduler_type == "plateau":
        return SloReduceLROnPlateau(optimizer, mode=kwargs.get("mode", "min"),
                                     factor=kwargs.get("factor", 0.1),
                                     patience=kwargs.get("patience", 10))
    elif scheduler_type == "cosine_annealing":
        return SloCosineAnnealingLR(optimizer, T_max=kwargs.get("T_max", total_steps or 100),
                                     eta_min=min_lr)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def compute_sensitivity(
    output: "Tensor",
    param_groups: Dict[str, List["Tensor"]],
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Per-group parameter sensitivity via forward-mode AD.

    For each group, creates random unit-norm seed tangents on its trainable
    parameters, propagates them through the computation graph via
    ``output.forward_grad()``, and measures the norm of the resulting tangent
    at *output*. A higher score means a small perturbation to that group's
    parameters causes a large change in the output — the group is more
    "sensitive".

    Groups are processed independently (one forward propagation per group),
    so the scores reflect each group's *isolated* contribution.

    Args:
        output: Loss or output tensor (must have intact graph ancestry).
        param_groups: Dict ``{group_name: [param_tensors]}``.
        seed: Optional RNG seed for reproducible tangents.

    Returns:
        Dict ``{group_name: sensitivity_score}``. Empty dict if no params
        are trainable.

    Side effects:
        - Does **not** mutate any tensor's data or gradient.
        - The computation graph must still be reachable from *output*.
    """
    rng = np.random.RandomState(seed)
    sensitivities: Dict[str, float] = {}
    for group_name, params in param_groups.items():
        seed_tangents: Dict[int, np.ndarray] = {}
        for p in params:
            if not getattr(p, "requires_grad", False):
                continue
            v = rng.standard_normal(p.shape).astype(p.data.dtype)
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-12:
                seed_tangents[p.id] = v / v_norm
        if not seed_tangents:
            continue
        result = output.forward_grad(seed_tangents)
        loss_tangent = result.get(output.id, np.zeros(1))
        sensitivities[group_name] = float(np.linalg.norm(loss_tangent))
    return sensitivities


# =============================================================================
# __ALL__
# =============================================================================

__all__ = ["Tensor", "SloLayer", "SloLinear", "SloEmbedding", "SloLSTM", "SloLayerNorm", "SloRMSNorm",
           "SloTransformerBlock", "SloMultiHeadAttention", "SloFeedForward", "SloDropout",
           "SloRotaryEmbedding", "SloTransformer",
           "SloNet", "SloSGD", "SloAdam", "sigmoid", "tanh", "relu", "gelu", "silu", "softmax",
           "cross_entropy", "mse_loss", "log_softmax", "kl_div_loss",
           "normalize", "pairwise_distance", "argmax", "argmin",
           "squeeze", "unsqueeze", "cat", "eye",
           "SloDataset", "SloDataLoader",
           "SloLRScheduler", "SloStepLR", "SloCosineAnnealingLR", "SloReduceLROnPlateau",
           "WarmupCosineScheduler", "PolynomialDecayScheduler", "LinearWarmupScheduler",
           "SloConstantLR", "SloOneCycleLR", "SloCyclicLR", "create_scheduler",
           "zeros", "randn", "ones", "tensor", "export_to_sou", "import_from_sou", "souls_from_directory",
           "train_char_lstm_from_gpt", "train_soul_transformer", "SOU_MAGIC", "SOU_VERSION",
           "topk", "multinomial", "stack", "concatenate", "randint", "exp", "isfinite", "where",
           "no_grad", "is_cuda", "is_mps", "cuda", "cpu", "_rotate_half", "_apply_rope",
           "compute_sensitivity"]
