"""
torch shim — numpy-backed replacement for PyTorch with autograd.
All ops use numpy with SloNet-style autograd graph.
Optionally accelerated via gpu/accelerator (Metal, CUDA, or CPU fallback).
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Optional, Any, Callable

# Initialize accelerator for forward-pass compute
try:
    from domains.training.gpu.accelerator import get_accelerator as _get_accel
    _acc = _get_accel()
    _acc_ok = f"{_acc.name}/{_acc.device_type}"
except Exception as e:
    _acc = None
    _acc_ok = f"ERR:{e}"


class Tensor:
    _id_counter = 0

    def __init__(self, data, dtype=None, device=None, requires_grad=False, _children=(), _copy=True):
        if isinstance(data, list):
            arr = np.array(data, dtype=np.float32)
        elif not isinstance(data, np.ndarray):
            arr = np.array(data, dtype=np.float32)
        else:
            if data.dtype == np.float32 and _copy:
                arr = data.copy()
            elif data.dtype != np.float32 and data.dtype != np.int64 and data.dtype != np.int32:
                arr = data.astype(np.float32)
            else:
                arr = data.astype(np.float32) if data.dtype != np.float32 and data.dtype != np.int64 and data.dtype != np.int32 else data.copy()
        if dtype == np.int64 or dtype == 'long': arr = arr.astype(np.int64)
        elif dtype == np.int32: arr = arr.astype(np.int32)
        elif dtype is not None: arr = arr.astype(dtype)
        self.data: np.ndarray = np.asarray(arr)
        self.grad: Optional[Tensor] = None
        self.requires_grad = requires_grad
        self._children: tuple = _children
        self._backward_fn: Optional[Callable] = None
        self.shape = self.data.shape
        self.dtype = self.data.dtype
        self.device = device or "cpu"
        self.id = Tensor._id_counter
        Tensor._id_counter += 1

    def __repr__(self): return f"Tensor({self.shape}, dtype={self.dtype})"
    def __bool__(self): return bool(self.data.size == 1 and self.data.item())
    def __len__(self): return len(self.data)
    def __iter__(self):
        for i in range(len(self)): yield self[i]
    def __add__(self, other): return _add(self, _ensure(other))
    def __radd__(self, other): return _add(_ensure(other), self)
    def __sub__(self, other): return _sub(self, _ensure(other))
    def __rsub__(self, other): return _sub(_ensure(other), self)
    def __mul__(self, other): return _mul(self, _ensure(other))
    def __rmul__(self, other): return _mul(_ensure(other), self)
    def __neg__(self): return _neg(self)
    def __pow__(self, p): return _pow(self, p)
    def __truediv__(self, other): return _div(self, _ensure(other))
    def __rtruediv__(self, other): return _div(_ensure(other), self)
    def __lt__(self, other): return lt(self, _ensure(other))
    def __le__(self, other): return le(self, _ensure(other))
    def __gt__(self, other): return gt(self, _ensure(other))
    def __ge__(self, other): return ge(self, _ensure(other))
    def __eq__(self, other): return eq(self, _ensure(other))
    def __ne__(self, other): return ne(self, _ensure(other))
    def __matmul__(self, other): return _matmul(self, _ensure(other))
    def __getitem__(self, key):
        d = self.data
        ndim = d.ndim
        if isinstance(key, tuple):
            key = list(key)
            for i, k in enumerate(key):
                if k == Ellipsis:
                    n_fill = ndim - (len(key) - 1)
                    key = key[:i] + [slice(None)] * n_fill + key[i+1:]
                    break
            if len(key) > ndim: key = key[:ndim]
            result = _slice(self, tuple(key))
            # Remove scalar dims introduced by integer indices
            squeeze_dims = [i for i, k in enumerate(key) if isinstance(k, (int, np.integer))]
            for idx in reversed(squeeze_dims):
                if result.shape[idx] == 1:
                    result = _reshape(result, result.shape[:idx] + result.shape[idx+1:])
            return result
        return _slice(self, key)
    def __setitem__(self, key, value):
        vd = value.data if isinstance(value, Tensor) else np.asarray(value)
        if isinstance(key, Tensor):
            mask = np.asarray(key.data, dtype=np.bool_)
            self.data[mask] = vd.astype(self.data.dtype)
            return
        self.data[key] = vd.astype(self.data.dtype)
    def tolist(self): return self.data.tolist()
    def item(self): return float(self.data.flat[0])
    def dim(self) -> int: return self.data.ndim
    def numel(self) -> int: return self.data.size
    @property
    def T(self): return _transpose(self)
    def reshape(self, *s): return _reshape(self, s if len(s) > 1 else s[0])
    def view(self, *s): return _reshape(self, s if len(s) > 1 else s[0])
    def sum(self, dim=None, keepdim=False):
        if dim is None: return _reduce_sum(self)
        return _reduce_sum_dim(self, dim, keepdim)
    def mean(self, dim=None, keepdim=False):
        if dim is None: return _reduce_mean(self)
        return _reduce_mean_dim(self, dim, keepdim)
    def max(self, dim=None):
        if dim is None: return float(self.data.max())
        m = self.data.max(axis=dim); i = self.data.argmax(axis=dim)
        return Tensor(m), Tensor(i.astype(np.int64))
    def min(self, dim=None):
        if dim is None: return float(self.data.min())
        m = self.data.min(axis=dim); i = self.data.argmin(axis=dim)
        return Tensor(m), Tensor(i.astype(np.int64))
    def abs(self): return _abs(self)
    def abs_(self): self.data = np.abs(self.data); return self
    def neg(self): return _neg(self)
    def neg_(self): self.data = -self.data; return self
    def squeeze(self, dim=None):
        if dim is None: return Tensor(self.data.squeeze())
        return Tensor(self.data.squeeze(axis=dim))
    def unsqueeze(self, dim: int): return Tensor(np.expand_dims(self.data, axis=dim))
    def repeat(self, *s): return Tensor(np.tile(self.data, s))
    def flatten(self, start_dim=0, end_dim=-1):
        d = self.data
        if end_dim < 0: end_dim = d.ndim + end_dim
        sh = list(d.shape)
        ns = (*sh[:start_dim], int(np.prod(sh[start_dim:end_dim+1])), *sh[end_dim+1:])
        return _reshape(self, ns)
    def cumsum(self, dim=0): return Tensor(np.cumsum(self.data, axis=dim))
    def cumprod(self, dim=0): return Tensor(np.cumprod(self.data, axis=dim))
    def argmax(self, dim=None):
        if dim is None: return int(self.data.argmax())
        return Tensor(self.data.argmax(axis=dim).astype(np.int64))
    def argmin(self, dim=None):
        if dim is None: return int(self.data.argmin())
        return Tensor(self.data.argmin(axis=dim).astype(np.int64))
    def sort(self, dim=None, descending=False):
        d = self.data
        if dim is None:
            idx = d.flatten().argsort()
            if descending: idx = idx[::-1]
            return Tensor(d.flatten()[idx])
        idx = d.argsort(axis=dim)
        if descending: idx = idx[:, ::-1] if dim == 1 else idx[::-1]
        vals = np.take_along_axis(d, idx, axis=dim)
        return Tensor(vals), Tensor(idx.astype(np.int64))
    def argsort(self, dim=None, descending=False):
        d = self.data
        if dim is None:
            idx = d.flatten().argsort()
            if descending: idx = idx[::-1]
            return Tensor(idx.astype(np.int64))
        idx = d.argsort(axis=dim)
        if descending: idx = idx[:, ::-1] if dim == 1 else idx[::-1]
        return Tensor(idx.astype(np.int64))
    def gather(self, dim, index):
        idx = index.data.astype(int)
        return Tensor(np.take_along_axis(self.data, idx, axis=dim))
    def scatter_(self, dim, index, src):
        data = self.data.copy(); idx = index.data.astype(int)
        sd = src.data if isinstance(src, Tensor) else np.asarray(src)
        np.put_along_axis(data, idx, sd, axis=dim); self.data = data; return self
    def masked_fill_(self, mask, value):
        md = mask.data if isinstance(mask, Tensor) else np.asarray(mask)
        self.data[md] = value; return self
    def index_select(self, dim, index):
        idx = index.data.astype(int).flatten()
        return Tensor(np.take(self.data, idx, axis=dim))
    def split(self, split_size, dim=0):
        if isinstance(split_size, int):
            return [Tensor(x) for x in np.array_split(self.data, list(range(split_size, self.data.shape[dim], split_size)), axis=dim)]
        return [Tensor(x) for x in np.split(self.data, np.cumsum(split_size[:-1]), axis=dim)]
    def chunk(self, chunks, dim=0):
        return [Tensor(x) for x in np.array_split(self.data, chunks, axis=dim)]
    def narrow(self, dim, start, length):
        sl = (slice(None),) * dim + (slice(start, start + length),)
        return Tensor(self.data[sl])
    def permute(self, *dims):
        return Tensor(np.transpose(self.data, dims))
    def transpose(self, dim0, dim1):
        return _transpose(self, (dim0, dim1))
    def clamp_(self, min_val=None, max_val=None):
        self.data = np.clip(self.data, min_val, max_val); return self
    def zero_(self): self.data.fill(0); return self
    def fill_(self, v): self.data.fill(v); return self
    def long(self): return Tensor(self.data.astype(np.int64), dtype=np.int64)
    def float(self): return Tensor(self.data.astype(np.float32), dtype=np.float32)
    def half(self): return Tensor(self.data.astype(np.float16), dtype=np.float16)
    def cpu(self): return self
    def numpy(self): return self.data
    def to(self, *args, **kwargs): return self
    def det(self): return float(np.linalg.det(self.data))
    def diag(self, diagonal=0): return Tensor(np.diag(self.data, k=diagonal))
    def tril(self, k=0): return Tensor(np.tril(self.data, k=k))
    def triu(self, k=0): return Tensor(np.triu(self.data, k=k))
    def trace(self): return float(np.trace(self.data))
    def sqrt(self):
        return _sqrt(self)
    def log(self):
        return _log(self)
    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False)
    def clone(self):
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    def backward(self, gradient=None):
        if gradient is not None:
            gd = gradient.data if isinstance(gradient, Tensor) else np.asarray(gradient)
        else:
            gd = np.ones_like(self.data)
        if self.grad is None:
            self.grad = Tensor(gd)
        else:
            self.grad.data[:] += gd
        visited, topo = set(), []
        def build(v):
            if v.id in visited: return
            visited.add(v.id)
            for c in getattr(v, '_children', ()):
                if isinstance(c, Tensor): build(c)
            topo.append(v)
        build(self)
        for node in reversed(topo):
            g = node.grad.data if node.grad is not None else np.ones_like(node.data)
            node.grad = Tensor(g)
            if node._backward_fn: node._backward_fn(g)


def _ensure(x):
    if isinstance(x, Tensor): return x
    if x is None: return None
    return Tensor(x)


def _add(a, b):
    a_t = isinstance(a, Tensor); b_t = isinstance(b, Tensor)
    ad = a.data if a_t else np.asarray(a)
    bd = b.data if b_t else np.asarray(b)
    a_req = a_t and a.requires_grad; b_req = b_t and b.requires_grad
    out = Tensor(ad + bd, requires_grad=a_req or b_req,
                 _children=(a if a_t else None, b if b_t else None))
    _a_shape = a.shape if a_t else None; _b_shape = b.shape if b_t else None
    def bk(g):
        if a_req:
            ga = g
            if _a_shape and g.ndim > len(_a_shape):
                ga = ga.sum(axis=tuple(range(g.ndim - len(_a_shape))), keepdims=False)
            for i, d in enumerate(_a_shape):
                if d == 1 and i < ga.ndim and ga.shape[i] > 1: ga = np.sum(ga, axis=i, keepdims=True)
            a.grad = Tensor(ga if a.grad is None else a.grad.data + ga)
        if b_req:
            gb = g
            if _b_shape and g.ndim > len(_b_shape):
                gb = gb.sum(axis=tuple(range(g.ndim - len(_b_shape))), keepdims=False)
            for i, d in enumerate(_b_shape):
                if d == 1 and i < gb.ndim and gb.shape[i] > 1: gb = np.sum(gb, axis=i, keepdims=True)
            b.grad = Tensor(gb if b.grad is None else b.grad.data + gb)
    out._backward_fn = bk if (a_req or b_req) else None; return out


def _sub(a, b):
    return _add(a, _neg(b))


def _mul(a, b):
    a_t = isinstance(a, Tensor); b_t = isinstance(b, Tensor)
    ad = a.data if a_t else np.asarray(a)
    bd = b.data if b_t else np.asarray(b)
    a_req = a_t and a.requires_grad; b_req = b_t and b.requires_grad
    out = Tensor(ad * bd, requires_grad=a_req or b_req,
                 _children=(a if a_t else None, b if b_t else None))
    _a_shape = a.shape if a_t else None; _b_shape = b.shape if b_t else None
    def bk(g):
        ga = g * bd; gb = g * ad
        if _a_shape:
            for i, d in enumerate(_a_shape):
                if d == 1 and i < g.ndim and g.shape[i] > 1: ga = np.sum(ga, axis=i, keepdims=True)
        if _b_shape:
            for i, d in enumerate(_b_shape):
                if d == 1 and i < g.ndim and g.shape[i] > 1: gb = np.sum(gb, axis=i, keepdims=True)
        if a_req: a.grad = Tensor(ga if a.grad is None else a.grad.data + ga)
        if b_req: b.grad = Tensor(gb if b.grad is None else b.grad.data + gb)
    out._backward_fn = bk if (a_req or b_req) else None; return out


def _neg(a):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    a_req = isinstance(a, Tensor) and a.requires_grad
    out = Tensor(-ad, requires_grad=a_req, _children=(a,) if a_req else ())
    def bk(g):
        if a_req: a.grad = Tensor(-g if a.grad is None else a.grad.data - g)
    out._backward_fn = bk if a_req else None; return out


def _pow(a, p):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    a_req = isinstance(a, Tensor) and a.requires_grad
    out = Tensor(ad ** p, requires_grad=a_req, _children=(a,) if a_req else ())
    def bk(g):
        if a_req:
            a.grad = Tensor(p * (ad ** (p - 1)) * g if a.grad is None else a.grad.data + p * (ad ** (p - 1)) * g)
    out._backward_fn = bk if a_req else None; return out


def _div(a, b):
    return _mul(a, _pow(b, -1))


def _matmul(a, b):
    a_t = isinstance(a, Tensor); b_t = isinstance(b, Tensor)
    ad = a.data if a_t else np.asarray(a)
    bd = b.data if b_t else np.asarray(b)
    a_req = a_t and a.requires_grad; b_req = b_t and b.requires_grad
    fwd = _acc.matmul(ad, bd) if _acc is not None else np.matmul(ad, bd)
    out = Tensor(fwd, requires_grad=a_req or b_req,
                 _children=(a if a_t else None, b if b_t else None))
    _a_shape = ad.shape; _b_shape = bd.shape; _out_shape = out.data.shape
    _a_ndim = ad.ndim; _b_ndim = bd.ndim
    def bk(g):
        if a_req:
            g_flat = g.data if isinstance(g, Tensor) else np.asarray(g)
            if g_flat.shape != _out_shape: g_flat = g_flat.reshape(_out_shape)
            # da = grad @ b^T (over last two dims)
            ga = np.matmul(g_flat, np.swapaxes(bd, -2, -1))
            # Sum over broadcast dims if a had fewer batch dims than ga
            for _ in range(ga.ndim - _a_ndim):
                ga = ga.sum(axis=0)
            ga = ga.reshape(_a_shape) if ga.shape != _a_shape else ga
            a.grad = Tensor(ga if a.grad is None else a.grad.data + ga)
        if b_req:
            g_flat = g.data if isinstance(g, Tensor) else np.asarray(g)
            if g_flat.shape != _out_shape: g_flat = g_flat.reshape(_out_shape)
            # db = a^T @ grad (over last two dims)
            gb = np.matmul(np.swapaxes(ad, -2, -1), g_flat)
            # Sum over broadcast dims if b had fewer batch dims than gb
            for _ in range(gb.ndim - _b_ndim):
                gb = gb.sum(axis=0)
            gb = gb.reshape(_b_shape) if gb.shape != _b_shape else gb
            b.grad = Tensor(gb if b.grad is None else b.grad.data + gb)
    out._backward_fn = bk if (a_req or b_req) else None; return out


def _transpose(x, dims=None):
    ad = x.data if isinstance(x, Tensor) else np.asarray(x)
    x_req = isinstance(x, Tensor) and x.requires_grad
    if dims is not None:
        dim0, dim1 = dims
        _axes = list(range(ad.ndim))
        _axes[dim0], _axes[dim1] = _axes[dim1], _axes[dim0]
        out_data = np.transpose(ad, _axes)
        _reverse = list(range(ad.ndim))
        _reverse[dim0], _reverse[dim1] = _reverse[dim1], _reverse[dim0]
    else:
        out_data = ad.T.copy()
        _reverse = None
    out = Tensor(out_data, requires_grad=x_req, _children=(x,) if x_req else ())
    _dims = dims
    def bk(g):
        if x_req:
            if _dims is not None:
                tg = np.transpose(g.data if isinstance(g, Tensor) else np.asarray(g), _reverse)
            else:
                tg = (g.data if isinstance(g, Tensor) else np.asarray(g)).T
            x.grad = Tensor(tg if x.grad is None else x.grad.data + tg)
    out._backward_fn = bk if x_req else None; return out


def _reshape(a, s):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    a_req = isinstance(a, Tensor) and a.requires_grad
    out = Tensor(ad.reshape(s), requires_grad=a_req, _children=(a,) if a_req else ())
    _a_shape = a.shape if isinstance(a, Tensor) else None
    def bk(g):
        if a_req and _a_shape:
            ga = g.reshape(_a_shape)
            a.grad = Tensor(ga if a.grad is None else a.grad.data + ga)
    out._backward_fn = bk if a_req else None; return out


def _reduce_sum(a):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    a_req = isinstance(a, Tensor) and a.requires_grad
    out = Tensor(np.array(ad.sum(), dtype=np.float32), requires_grad=a_req, _children=(a,) if a_req else ())
    def bk(g):
        if a_req: a.grad = Tensor(np.full_like(ad, g) if a.grad is None else a.grad.data + np.full_like(ad, g))
    out._backward_fn = bk if a_req else None; return out


def _reduce_sum_dim(a, dim, keepdim):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    a_req = isinstance(a, Tensor) and a.requires_grad
    out = Tensor(ad.sum(axis=dim, keepdims=keepdim), requires_grad=a_req, _children=(a,) if a_req else ())
    def bk(g):
        if a_req:
            ga = np.expand_dims(g, axis=dim) if not keepdim else g
            ga = np.broadcast_to(ga, ad.shape)
            a.grad = Tensor(ga if a.grad is None else a.grad.data + ga)
    out._backward_fn = bk if a_req else None; return out


def _abs(a):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    a_req = isinstance(a, Tensor) and a.requires_grad
    out = Tensor(np.abs(ad), requires_grad=a_req, _children=(a,) if a_req else ())
    def bk(g):
        if a_req: a.grad = Tensor(np.sign(ad) * g if a.grad is None else a.grad.data + np.sign(ad) * g)
    out._backward_fn = bk if a_req else None; return out


def _sqrt(a):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    a_req = isinstance(a, Tensor) and a.requires_grad
    out = Tensor(np.sqrt(np.maximum(ad, 0)), requires_grad=a_req, _children=(a,) if a_req else ())
    def bk(g):
        if a_req:
            gs = g / (2 * np.maximum(ad, 0) ** 0.5 + 1e-8)
            a.grad = Tensor(gs if a.grad is None else a.grad.data + gs)
    out._backward_fn = bk if a_req else None; return out


def _log(a):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    a_req = isinstance(a, Tensor) and a.requires_grad
    out = Tensor(np.log(np.maximum(ad, 1e-8)), requires_grad=a_req, _children=(a,) if a_req else ())
    def bk(g):
        if a_req:
            gl = g / np.maximum(ad, 1e-8)
            a.grad = Tensor(gl if a.grad is None else a.grad.data + gl)
    out._backward_fn = bk if a_req else None; return out


def _reduce_mean(a):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    n = ad.size
    a_req = isinstance(a, Tensor) and a.requires_grad
    out = Tensor(np.array(ad.mean(), dtype=np.float32), requires_grad=a_req, _children=(a,) if a_req else ())
    def bk(g):
        if a_req: a.grad = Tensor(np.full_like(ad, g / n) if a.grad is None else a.grad.data + np.full_like(ad, g / n))
    out._backward_fn = bk if a_req else None; return out


def _reduce_mean_dim(a, dim, keepdim):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    a_req = isinstance(a, Tensor) and a.requires_grad
    out = Tensor(ad.mean(axis=dim, keepdims=keepdim), requires_grad=a_req, _children=(a,) if a_req else ())
    _dim = dim
    def bk(g):
        if a_req:
            if isinstance(_dim, tuple):
                ga = g
                for d in sorted(_dim, reverse=True):
                    ga = np.expand_dims(ga, axis=d) if not keepdim else ga
                ga = np.broadcast_to(ga, ad.shape)
                n = np.prod([ad.shape[d] for d in _dim])
            else:
                ga = np.expand_dims(g, axis=_dim) if not keepdim else g
                ga = np.broadcast_to(ga, ad.shape)
                n = ad.shape[_dim]
            ga = ga / n
            a.grad = Tensor(ga if a.grad is None else a.grad.data + ga)
    out._backward_fn = bk if a_req else None; return out


def _slice(a, key):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    a_req = isinstance(a, Tensor) and a.requires_grad
    sliced = ad[key]
    out = Tensor(sliced, requires_grad=a_req, _children=(a,) if a_req else ())
    _a_shape = a.shape if isinstance(a, Tensor) else None
    def bk(g):
        if a_req and _a_shape:
            full = np.zeros(a.data.shape, dtype=np.float32)
            np.add.at(full, key, g)
            a.grad = Tensor(full if a.grad is None else a.grad.data + full)
    out._backward_fn = bk if a_req else None; return out


def tensor(data, dtype=None, device=None, requires_grad=False):
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


def zeros(*shape, dtype=np.float32, **kwargs):
    s = shape[0] if len(shape) == 1 else shape
    rg = kwargs.pop("requires_grad", False)
    return Tensor(np.zeros(s, dtype=dtype), requires_grad=rg)


def ones(*shape, dtype=np.float32, **kwargs):
    s = shape[0] if len(shape) == 1 else shape
    rg = kwargs.pop("requires_grad", False)
    return Tensor(np.ones(s, dtype=dtype), requires_grad=rg)


def randn(*shape, dtype=np.float32, **kwargs):
    s = shape[0] if len(shape) == 1 else shape
    rg = kwargs.pop("requires_grad", False)
    return Tensor(np.random.randn(*s).astype(dtype), dtype=dtype, requires_grad=rg)


def randint(low, high, size=None, dtype=np.int64, **kwargs):
    if size is None: size = (1,)
    elif isinstance(size, int): size = (size,)
    return Tensor(np.random.randint(low, high, size), dtype=dtype)


def arange(start=0, end=None, step=1, dtype=None, **kwargs):
    if end is None: start, end = 0, start
    arr = np.arange(start, end, step)
    if dtype is None: return Tensor(arr)
    return Tensor(arr.astype(dtype))


def empty(*shape, dtype=np.float32, **kwargs):
    s = shape[0] if len(shape) == 1 else shape
    rg = kwargs.pop("requires_grad", False)
    return Tensor(np.empty(s, dtype=dtype), requires_grad=rg)


def zeros_like(t, dtype=None):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    dt = dtype or d.dtype
    return Tensor(np.zeros_like(d, dtype=dt))


def ones_like(t, dtype=None):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    dt = dtype or d.dtype
    return Tensor(np.ones_like(d, dtype=dt))


def full(size, fill_value, dtype=np.float32, requires_grad=False, **kwargs):
    if isinstance(size, int): size = (size,)
    return Tensor(np.full(size, fill_value, dtype=dtype), requires_grad=requires_grad)


def full_like(t, fill_value, dtype=None, **kwargs):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    dt = dtype or d.dtype
    return Tensor(np.full_like(d, fill_value, dtype=dt))


def exp(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.exp(np.clip(d, -700, 700)))


def log(t):
    if isinstance(t, Tensor): return _log(t)
    return Tensor(np.log(np.maximum(np.asarray(t), 1e-8)))


def sqrt(t):
    if isinstance(t, Tensor): return _sqrt(t)
    return Tensor(np.sqrt(np.maximum(np.asarray(t), 0)))


def abs(t):
    if isinstance(t, Tensor): return _abs(t)
    return Tensor(np.abs(np.asarray(t)))


def neg(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(-d)


def sin(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.sin(d))


def cos(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.cos(d))


def randn_like(t, **kwargs):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    rg = kwargs.pop("requires_grad", False)
    return Tensor(np.random.randn(*d.shape).astype(d.dtype), requires_grad=rg)


def sign(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.sign(d))


def sum(t, dim=None, keepdim=False):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    if dim is None: return float(d.sum())
    return Tensor(d.sum(axis=dim, keepdims=keepdim))


def mean(t, dim=None, keepdim=False):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    if dim is None: return float(d.mean())
    return Tensor(d.mean(axis=dim, keepdims=keepdim))


def prod(t, dim=None, keepdim=False):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    if dim is None: return float(d.prod())
    return Tensor(d.prod(axis=dim, keepdims=keepdim))


def softmax(t, dim=-1):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    if _acc is not None:
        s = _acc.softmax(d, axis=dim)
    else:
        e = np.exp(d - d.max(axis=dim, keepdims=True))
        s = e / e.sum(axis=dim, keepdims=True)
    t_req = isinstance(t, Tensor) and t.requires_grad
    out = Tensor(s, requires_grad=t_req, _children=(t,) if t_req else ())
    _dim = dim % d.ndim if d.ndim > 0 else 0
    def bk(g):
        if t_req:
            ds = s * (g - (s * g).sum(axis=_dim, keepdims=True))
            t.grad = Tensor(ds if t.grad is None else t.grad.data + ds)
    out._backward_fn = bk if t_req else None; return out


def sigmoid(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    s = 1.0 / (1.0 + np.exp(-np.clip(d, -500, 500)))
    t_req = isinstance(t, Tensor) and t.requires_grad
    out = Tensor(s, requires_grad=t_req, _children=(t,) if t_req else ())
    def bk(g):
        if t_req:
            gs = s * (1 - s) * g
            t.grad = Tensor(gs if t.grad is None else t.grad.data + gs)
    out._backward_fn = bk if t_req else None; return out


def tanh(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    th = np.tanh(d)
    t_req = isinstance(t, Tensor) and t.requires_grad
    out = Tensor(th, requires_grad=t_req, _children=(t,) if t_req else ())
    def bk(g):
        if t_req:
            gt = (1 - th * th) * g
            t.grad = Tensor(gt if t.grad is None else t.grad.data + gt)
    out._backward_fn = bk if t_req else None; return out


def relu(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    t_req = isinstance(t, Tensor) and t.requires_grad
    out = Tensor(np.maximum(d, 0), requires_grad=t_req, _children=(t,) if t_req else ())
    def bk(g):
        if t_req:
            gr = np.where(d > 0, g, 0.0)
            t.grad = Tensor(gr if t.grad is None else t.grad.data + gr)
    out._backward_fn = bk if t_req else None; return out


def leaky_relu(t, negative_slope=0.01):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    t_req = isinstance(t, Tensor) and t.requires_grad
    out = Tensor(np.where(d > 0, d, d * negative_slope), requires_grad=t_req, _children=(t,) if t_req else ())
    def bk(g):
        if t_req:
            gd = np.where(d > 0, g, g * negative_slope)
            t.grad = Tensor(gd if t.grad is None else t.grad.data + gd)
    out._backward_fn = bk if t_req else None; return out


def gelu(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    g = _acc.gelu(d) if _acc is not None else 0.5 * d * (1 + np.tanh(np.sqrt(2/np.pi) * (d + 0.044715 * d**3)))
    t_req = isinstance(t, Tensor) and t.requires_grad
    out = Tensor(g, requires_grad=t_req, _children=(t,) if t_req else ())
    def bk(grad):
        if t_req:
            d_gelu = 0.5 * np.tanh(np.sqrt(2/np.pi) * (d + 0.044715 * d**3)) + \
                     0.5 * d * (1 - np.tanh(np.sqrt(2/np.pi) * (d + 0.044715 * d**3))**2) * \
                     np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * d**2)
            t.grad = Tensor(d_gelu * grad if t.grad is None else t.grad.data + d_gelu * grad)
    out._backward_fn = bk if t_req else None; return out


def silu(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    s = _acc.silu(d) if _acc is not None else d / (1 + np.exp(-np.clip(d, -500, 500)))
    t_req = isinstance(t, Tensor) and t.requires_grad
    out = Tensor(s, requires_grad=t_req, _children=(t,) if t_req else ())
    def bk(g):
        if t_req:
            sig = 1 / (1 + np.exp(-np.clip(d, -500, 500)))
            ds = sig * (1 + d * (1 - sig))
            t.grad = Tensor(ds * g if t.grad is None else t.grad.data + ds * g)
    out._backward_fn = bk if t_req else None; return out


class _TopKResult:
    def __init__(self, values, indices):
        self.values = values; self.indices = indices
    def __iter__(self):
        return iter((self.values, self.indices))
    def __getitem__(self, key):
        return (self.values[key], self.indices[key])
    def __len__(self):
        return 2


def topk(t, k, dim=None, largest=True, sorted=True):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    if dim is None: dim = d.ndim - 1
    if largest: idx = d.argsort(axis=dim)[::-1]
    else: idx = d.argsort(axis=dim)
    idx = idx[..., :k]
    vals = np.take_along_axis(d, idx, axis=dim)
    return _TopKResult(Tensor(vals), Tensor(idx.astype(np.int64)))


def multinomial(t, num_samples, replacement=False):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    flat = d.reshape(-1).astype(np.float64)
    flat = np.maximum(flat, 0); total = flat.sum()
    if total > 0: flat = flat / total
    else: flat = np.ones_like(flat) / flat.size
    indices = np.random.choice(len(flat), size=num_samples, p=flat, replace=replacement)
    return Tensor(indices.reshape(1, num_samples), dtype=np.int64)


def cat(tensors, dim=0):
    arrays = [t.data if isinstance(t, Tensor) else np.asarray(t) for t in tensors]
    return Tensor(np.concatenate(arrays, axis=dim))


def stack(tensors, dim=0):
    arrays = [t.data if isinstance(t, Tensor) else np.asarray(t) for t in tensors]
    return Tensor(np.stack(arrays, axis=dim))


def where(condition, x=None, y=None):
    c = condition.data if isinstance(condition, Tensor) else np.asarray(condition)
    if x is None:
        return tuple(Tensor(a) for a in np.where(c))
    x_t = isinstance(x, Tensor); y_t = isinstance(y, Tensor)
    xd = x.data if x_t else np.asarray(x)
    yd = y.data if y_t else np.asarray(y)
    x_req = x_t and x.requires_grad; y_req = y_t and y.requires_grad
    out = Tensor(np.where(c, xd, yd), requires_grad=x_req or y_req,
                 _children=(x if x_t else None, y if y_t else None))
    def bk(g):
        g_np = g.data if isinstance(g, Tensor) else np.asarray(g)
        if x_req:
            gx = np.where(c, g_np, 0.0)
            x.grad = Tensor(gx if x.grad is None else x.grad.data + gx)
        if y_req:
            gy = np.where(c, 0.0, g_np)
            y.grad = Tensor(gy if y.grad is None else y.grad.data + gy)
    out._backward_fn = bk if (x_req or y_req) else None; return out


def isfinite(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.isfinite(d).astype(np.float32))


def isinf(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.isinf(d).astype(np.float32))


def isnan(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.isnan(d).astype(np.float32))


def allclose(a, b, rtol=1e-5, atol=1e-8):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    bd = b.data if isinstance(b, Tensor) else np.asarray(b)
    return np.allclose(ad, bd, rtol=rtol, atol=atol)


def eq(a, b):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    bd = b.data if isinstance(b, Tensor) else np.asarray(b)
    return Tensor((ad == bd).astype(np.float32))


def ne(a, b):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    bd = b.data if isinstance(b, Tensor) else np.asarray(b)
    return Tensor((ad != bd).astype(np.float32))


def gt(a, b):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    bd = b.data if isinstance(b, Tensor) else np.asarray(b)
    return Tensor((ad > bd).astype(np.float32))


def lt(a, b):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    bd = b.data if isinstance(b, Tensor) else np.asarray(b)
    return Tensor((ad < bd).astype(np.float32))


def ge(a, b):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    bd = b.data if isinstance(b, Tensor) else np.asarray(b)
    return Tensor((ad >= bd).astype(np.float32))


def le(a, b):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    bd = b.data if isinstance(b, Tensor) else np.asarray(b)
    return Tensor((ad <= bd).astype(np.float32))


def argmax(t, dim=None, keepdim=False):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    if dim is None:
        idx = d.argmax()
        return Tensor(np.array(idx).astype(np.int64))
    idx = np.argmax(d, axis=dim)
    if keepdim:
        idx = np.expand_dims(idx, axis=dim)
    return Tensor(idx.astype(np.int64))


def argsort(t, dim=-1, descending=False):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    idx = d.argsort(axis=dim)
    if descending:
        idx = idx[..., ::-1]
    return Tensor(idx.astype(np.int64))


def sort(t, dim=-1, descending=False):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    idx = d.argsort(axis=dim)
    if descending:
        idx = idx[..., ::-1]
    vals = np.take_along_axis(d, idx, axis=dim)
    return Tensor(vals), Tensor(idx.astype(np.int64))


def squeeze(t, dim=None):
    if isinstance(t, Tensor): return t.squeeze(dim)
    d = np.asarray(t)
    if dim is None: return Tensor(d.squeeze())
    return Tensor(np.squeeze(d, axis=dim))


def unsqueeze(t, dim):
    if isinstance(t, Tensor): return t.unsqueeze(dim)
    return Tensor(np.expand_dims(np.asarray(t), axis=dim))


def expand(t, *sizes):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.broadcast_to(d, sizes))


def repeat(t, *sizes):
    if isinstance(t, Tensor): return t.repeat(*sizes)
    return Tensor(np.tile(np.asarray(t), sizes))


def gather(t, dim, index):
    if isinstance(t, Tensor): return t.gather(dim, index)
    idx = index.data.astype(int) if isinstance(index, Tensor) else np.asarray(index).astype(int)
    return Tensor(np.take_along_axis(np.asarray(t), idx, axis=dim))


def scatter(t, dim, index, src):
    data = t.data.copy() if isinstance(t, Tensor) else np.asarray(t).copy()
    idx = index.data.astype(int) if isinstance(index, Tensor) else np.asarray(index).astype(int)
    sd = src.data if isinstance(src, Tensor) else np.asarray(src)
    if sd.shape != idx.shape:
        sd = np.broadcast_to(sd, idx.shape)
    # Iterate over all positions in index
    for flat_i in range(idx.size):
        multi_idx = np.unravel_index(flat_i, idx.shape)
        data[multi_idx[:dim] + (idx[multi_idx],) + multi_idx[dim+1:]] = sd.flat[flat_i]
    return Tensor(data)


def bmm(a, b):
    a_t = isinstance(a, Tensor); b_t = isinstance(b, Tensor)
    ad = a.data if a_t else np.asarray(a)
    bd = b.data if b_t else np.asarray(b)
    a_req = a_t and a.requires_grad; b_req = b_t and b.requires_grad
    out = Tensor(np.matmul(ad, bd), requires_grad=a_req or b_req,
                 _children=(a if a_t else None, b if b_t else None))
    def bk(g):
        gd = g.data if isinstance(g, Tensor) else np.asarray(g)
        if a_req:
            ga = np.matmul(gd, bd.swapaxes(-2, -1))
            a.grad = Tensor(ga if a.grad is None else a.grad.data + ga)
        if b_req:
            gb = np.matmul(ad.swapaxes(-2, -1), gd)
            b.grad = Tensor(gb if b.grad is None else b.grad.data + gb)
    out._backward_fn = bk if (a_req or b_req) else None
    return out


def from_numpy(arr):
    return Tensor(np.asarray(arr))


def eye(n, m=None, dtype=None):
    if m is None: m = n
    arr = np.eye(n, m, dtype=np.float32)
    return Tensor(arr, dtype=dtype)


class no_grad:
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def __call__(self, fn=None):
        if fn is None:
            return self
        return fn


class set_grad_enabled:
    def __init__(self, mode): self._mode = mode
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def __call__(self, fn=None):
        if fn is None:
            return self
        return fn


class autocast:
    def __init__(self, enabled=True, device_type="cpu", dtype=None): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def __call__(self, fn=None):
        if fn is None:
            return self
        return fn


class GradScaler:
    def __init__(self, init_scale=1.0, **kwargs): self._scale = init_scale
    def scale(self, loss): return loss * self._scale
    def step(self, opt): opt.step()
    def update(self): pass
    def get_scale(self): return self._scale


def save(state_dict, path_or_file, pickle_protocol=2):
    import pickle as _pickle
    if isinstance(path_or_file, str):
        with open(path_or_file, "wb") as f:
            _save(state_dict, f, pickle_protocol)
    else:
        _save(state_dict, path_or_file, pickle_protocol)


def _save(state_dict, file_obj, pickle_protocol):
    import pickle
    save_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, Tensor): save_dict[k] = v.data
        elif isinstance(v, np.ndarray): save_dict[k] = v
        else:
            try: save_dict[k] = v.data
            except: save_dict[k] = np.array(v)
    pickle.dump(save_dict, file_obj, protocol=pickle_protocol)


def load(path, map_location=None, weights_only=False, **kwargs):
    import pickle
    try:
        with open(path, "rb") as f: return pickle.load(f)
    except:
        try:
            with open(path, "rb") as f: return {"data": np.load(f)}
        except: return {}


def clamp(t, min_val=None, max_val=None):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.clip(d, min_val, max_val))


def clip(t, min_val=None, max_val=None):
    return clamp(t, min_val, max_val)


def masked_fill(t, mask, value):
    md = mask.data if isinstance(mask, Tensor) else np.asarray(mask)
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.where(md, value, d))


def index_select(t, dim, index):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    idx = index.data.astype(int).flatten()
    return Tensor(np.take(d, idx, axis=dim))


def matmul(a, b):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    bd = b.data if isinstance(b, Tensor) else np.asarray(b)
    return Tensor(np.matmul(ad, bd))


def mm(a, b):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    bd = b.data if isinstance(b, Tensor) else np.asarray(b)
    return Tensor(np.matmul(ad, bd))


def dot(a, b):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    bd = b.data if isinstance(b, Tensor) else np.asarray(b)
    return Tensor(np.dot(ad.flatten(), bd.flatten()))


def norm(t, p="fro", dim=None, keepdim=False):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    if dim is None: return float(np.linalg.norm(d.flatten(), ord=p if p != "fro" else None))
    return Tensor(np.linalg.norm(d, ord=p if p != "fro" else None, axis=dim, keepdims=keepdim))


def split(t, split_size, dim=0):
    if isinstance(split_size, int):
        arr = t.data if isinstance(t, Tensor) else np.asarray(t)
        indices = list(range(split_size, arr.shape[dim], split_size))
        return [Tensor(x) for x in np.array_split(arr, indices, axis=dim)]
    arr = t.data if isinstance(t, Tensor) else np.asarray(t)
    return [Tensor(x) for x in np.split(arr, np.cumsum(split_size[:-1]), axis=dim)]


def chunk(t, chunks, dim=0):
    arr = t.data if isinstance(t, Tensor) else np.asarray(t)
    return [Tensor(x) for x in np.array_split(arr, chunks, axis=dim)]


def narrow(t, dim, start, length):
    arr = t.data if isinstance(t, Tensor) else np.asarray(t)
    sl = (slice(None),) * dim + (slice(start, start + length),)
    return Tensor(arr[sl])


def permute(t, *dims):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.transpose(d, dims))


def transpose(t, dim0, dim1):
    return _transpose(t, (dim0, dim1))


def flatten(t, start_dim=0, end_dim=-1):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    sh = list(d.shape)
    ns = (*sh[:start_dim], np.prod(sh[start_dim:end_dim+1]), *sh[end_dim+1:])
    return Tensor(d.reshape(ns))


def repeat_interleave(t, repeats, dim=0):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.repeat(d, repeats, axis=dim))


def meshgrid(*tensors, indexing="ij"):
    arrays = [t.data.flatten() if isinstance(t, Tensor) else np.asarray(t).flatten() for t in tensors]
    grids = np.meshgrid(*arrays, indexing=indexing)
    return [Tensor(g) for g in grids]


def diag(t, diagonal=0):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.diag(d, k=diagonal))


def tril(t, k=0):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.tril(d, k=k))


def triu(t, k=0):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.triu(d, k=k))


def trace(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return float(np.trace(d))


def floor(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.floor(d))


def ceil(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.ceil(d))


def round(t, decimals=0):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.round(d, decimals=decimals))


def floor_divide(a, b):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    bd = b.data if isinstance(b, Tensor) else np.asarray(b)
    return Tensor(np.floor_divide(ad, bd))


def true_divide(a, b):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    bd = b.data if isinstance(b, Tensor) else np.asarray(b)
    return Tensor(ad / bd)


def reciprocal(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(1.0 / d)


def linspace(start, end, steps, dtype=np.float32):
    return Tensor(np.linspace(start, end, steps).astype(dtype))


def logspace(start, end, steps, base=10.0, dtype=np.float32):
    return Tensor(np.logspace(start, end, steps, base=base).astype(dtype))


def nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    d = np.where(np.isnan(d), nan, d)
    d = np.where(np.isposinf(d), posinf, d)
    d = np.where(np.isneginf(d), neginf, d)
    return Tensor(d)


def bernoulli(t, generator=None):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor((np.random.rand(*d.shape) < d).astype(np.float32))


def manual_seed(seed): np.random.seed(seed)


class Generator:
    def __init__(self, seed=0):
        self.seed = seed
        self._rng = np.random.RandomState(seed)
    def manual_seed(self, seed):
        self.seed = seed; self._rng = np.random.RandomState(seed); return self
    def randn(self, *shape): return Tensor(self._rng.randn(*shape).astype(np.float32))
    def randint(self, low, high, size=(1,)): return Tensor(self._rng.randint(low, high, size), dtype=np.int64)


def numel(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return int(np.prod(d.shape))


class device:
    def __init__(self, spec): self.spec = str(spec)
    def __repr__(self): return f"device('{self.spec}')"
    def __str__(self): return self.spec


class dtype:
    def __init__(self, np_type): self._np = np_type
    def __repr__(self): return f"dtype({self._np})"
    def __eq__(self, other):
        if other is float32: return self._np == np.float32
        if other is int64 or other is long: return self._np == np.int64
        if other is int32: return self._np == np.int32
        if other is bool: return self._np == np.bool_
        if other is bfloat16: return self._np == np.float32
        return False
    @property
    def is_floating_point(self): return np.issubdtype(self._np, np.floating)


float32 = dtype(np.float32)
float64 = dtype(np.float64)
float16 = dtype(np.float16)
int64 = dtype(np.int64)
int32 = dtype(np.int32)
int16 = dtype(np.int16)
int8 = dtype(np.int8)
uint8 = dtype(np.uint8)
bool = dtype(np.bool_)
long = dtype(np.int64)
short = dtype(np.int16)
bfloat16 = dtype(np.float32)
quint8 = dtype(np.uint8)
qint8 = dtype(np.int8)
qint32 = dtype(np.int32)
quint4x2 = dtype(np.uint8)


class cuda:
    @staticmethod
    def is_available() -> bool: return False
    @staticmethod
    def device_count() -> int: return 0
    @staticmethod
    def set_device(dev): pass
    @staticmethod
    def manual_seed(seed): np.random.seed(seed)
    @staticmethod
    def memory_allocated(dev=None) -> int: return 0
    @staticmethod
    def memory_reserved(dev=None) -> int: return 0
    @staticmethod
    def empty_cache(): pass
    @staticmethod
    def synchronize(): pass
    @staticmethod
    def get_device_properties(dev=0):
        class Props:
            name = "numpy-shim"; total_memory = 0; major = 0; minor = 0
        return Props()


class mps:
    @staticmethod
    def is_available() -> bool: return False


class backends:
    class mps:
        is_available = lambda: False
    class cuda:
        is_available = lambda: False


class distributed:
    @staticmethod
    def is_available() -> bool: return False
    init_model_parallel = lambda: None
    init_data_parallel = lambda: None
    world_size = 1; rank = 0


class _BaseOptimizer:
    def __init__(self, params, defaults):
        self.params = list(params) if params else []
        self.defaults = defaults
        self.state = {}

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None

    def load_state_dict(self, d): pass
    def state_dict(self): return {}


class optim:
    class SGD(_BaseOptimizer):
        def __init__(self, params, lr=0.01, momentum=0, weight_decay=0, **kwargs):
            defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
            super().__init__(params, defaults)
            self.lr = lr

        def step(self):
            for p in self.params:
                if p.grad is None:
                    continue
                g = p.grad.data
                wd = self.defaults.get("weight_decay", 0)
                if wd != 0:
                    g = g + wd * p.data
                p.data = p.data - self.lr * g

    class Adam(_BaseOptimizer):
        def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, **kwargs):
            defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            super().__init__(params, defaults)
            self.lr = lr
            self.betas = betas
            self.eps = eps
            self.weight_decay = weight_decay

        def step(self):
            b1, b2 = self.betas
            eps = self.eps
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad.data
                wd = self.weight_decay
                if wd != 0:
                    g = g + wd * p.data
                pid = str(id(p))
                if pid not in self.state:
                    self.state[pid] = dict(step=0, exp_avg=np.zeros_like(p.data), exp_avg_sq=np.zeros_like(p.data))
                st = self.state[pid]
                st["step"] += 1
                st["exp_avg"] = b1 * st["exp_avg"] + (1 - b1) * g
                st["exp_avg_sq"] = b2 * st["exp_avg_sq"] + (1 - b2) * (g ** 2)
                bias_corr1 = 1 - b1 ** st["step"]
                bias_corr2 = 1 - b2 ** st["step"]
                step_size = self.lr * np.sqrt(bias_corr2) / bias_corr1
                denom = np.sqrt(st["exp_avg_sq"]) + eps
                p.data = p.data - step_size * st["exp_avg"] / denom

    class AdamW(_BaseOptimizer):
        def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, **kwargs):
            defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            super().__init__(params, defaults)
            self.lr = lr
            self.betas = betas
            self.eps = eps
            self.weight_decay = weight_decay

        def step(self):
            b1, b2 = self.betas
            eps = self.eps
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad.data
                wd = self.weight_decay
                pid = str(id(p))
                if pid not in self.state:
                    self.state[pid] = dict(step=0, exp_avg=np.zeros_like(p.data), exp_avg_sq=np.zeros_like(p.data))
                st = self.state[pid]
                p.data = p.data - self.lr * wd * p.data
                st["step"] += 1
                st["exp_avg"] = b1 * st["exp_avg"] + (1 - b1) * g
                st["exp_avg_sq"] = b2 * st["exp_avg_sq"] + (1 - b2) * (g ** 2)
                bias_corr1 = 1 - b1 ** st["step"]
                bias_corr2 = 1 - b2 ** st["step"]
                step_size = self.lr * np.sqrt(bias_corr2) / bias_corr1
                denom = np.sqrt(st["exp_avg_sq"]) + eps
                p.data = p.data - step_size * st["exp_avg"] / denom


def compile(model, **kwargs): return model


def device_count() -> int: return 0


def set_printoptions(**kwargs): pass


def get_default_dtype(): return np.float32


def set_default_dtype(dtype): pass


class Size(tuple):
    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], (list, tuple, np.ndarray)):
            return super().__new__(cls, args[0])
        return super().__new__(cls, args)
    def __str__(self): return f"Size({list(self)})"
    def __repr__(self): return f"Size({list(self)})"


class functional:
    @staticmethod
    def relu(x, inplace=False): return relu(x)
    @staticmethod
    def tanh(x): return tanh(x)
    @staticmethod
    def sigmoid(x): return sigmoid(x)
    @staticmethod
    def softmax(x, dim=-1): return softmax(x, dim=dim)
    @staticmethod
    def dropout(x, p=0, training=False, inplace=False): return x
    @staticmethod
    def pad(x, pad_tuple):
        d = x.data if isinstance(x, Tensor) else np.asarray(x)
        return Tensor(np.pad(d, pad_tuple))
    @staticmethod
    def one_hot(t, num_classes):
        d = t.data if isinstance(t, Tensor) else np.asarray(t)
        return Tensor(np.eye(num_classes, dtype=np.float32)[d.astype(int).flatten()].reshape(*d.shape, num_classes))


class utils:
    class data:
        class Dataset:
            def __getitem__(self, i): raise NotImplementedError
            def __len__(self): raise NotImplementedError
        class DataLoader:
            def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, **kwargs):
                self.dataset = dataset; self.batch_size = batch_size; self.shuffle = shuffle
            def __iter__(self):
                if self.shuffle:
                    indices = list(range(len(self.dataset)))
                    np.random.shuffle(indices)
                    for i in range(0, len(indices), self.batch_size):
                        batch = [self.dataset[indices[j]] for j in indices[i:i+self.batch_size]]
                        yield _collate(batch)
                else:
                    for i in range(0, len(self.dataset), self.batch_size):
                        batch = [self.dataset[j] for j in range(i, min(i+self.batch_size, len(self.dataset)))]
                        yield _collate(batch)
            def _collate(self, batch):
                if isinstance(batch[0], tuple):
                    return tuple(stack([b[i] for b in batch]) for i in range(len(batch[0])))
                return stack(batch)
    class nn_utils:
        @staticmethod
        def clip_grad_norm_(parameters, max_norm, norm_type=2): return 0.0
        @staticmethod
        def pack_padded_sequence(sequences, lengths, batch_first=False): return sequences
        @staticmethod
        def pad_packed_sequence(sequence, batch_first=False, padding_value=0.0): return sequence, lengths
    class tensorboard:
        class SummaryWriter:
            def __init__(self, log_dir=None, **kwargs): pass
            def add_scalar(self, tag, value, step): pass
            def add_histogram(self, tag, values, step): pass
            def close(self): pass


__version__ = "2.1.2-shim"

from . import nn
from .nn import functional as F
from .nn import utils as nn_utils
