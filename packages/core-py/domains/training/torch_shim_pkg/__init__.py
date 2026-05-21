"""
torch shim — numpy-backed replacement for PyTorch.
All ops use numpy. No GPU, no downloads.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Optional, Any


class Tensor:
    _id_counter = 0

    def __init__(self, data, dtype=None, device=None, requires_grad=False):
        if isinstance(data, list):
            arr = np.array(data, dtype=np.float32)
        elif not isinstance(data, np.ndarray):
            arr = np.array(data, dtype=np.float32)
        else:
            arr = data.astype(np.float32) if data.dtype != np.float32 and data.dtype != np.int64 and data.dtype != np.int32 else data.copy()
        if dtype == np.int64 or dtype == long: arr = arr.astype(np.int64)
        elif dtype == np.int32: arr = arr.astype(np.int32)
        elif dtype is not None: arr = arr.astype(dtype)
        self.data: np.ndarray = np.asarray(arr)
        self.grad: Optional[Tensor] = None
        self.requires_grad = requires_grad
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
    def __matmul__(self, other): return _matmul(self, _ensure(other))
    def __getitem__(self, key): return Tensor(self.data[key], dtype=self.dtype)
    def __setitem__(self, key, value):
        vd = value.data if isinstance(value, Tensor) else np.asarray(value)
        self.data[key] = vd.astype(self.data.dtype)
    def tolist(self): return self.data.tolist()
    def item(self): return float(self.data.flat[0])
    def dim(self) -> int: return self.data.ndim
    def T(self): return _transpose(self)
    def reshape(self, *s): return Tensor(self.data.reshape(s))
    def view(self, *s): return Tensor(self.data.reshape(s))
    def sum(self, dim=None, keepdim=False):
        if dim is None: return float(self.data.sum())
        return Tensor(self.data.sum(axis=dim, keepdims=keepdim))
    def mean(self, dim=None, keepdim=False):
        if dim is None: return float(self.data.mean())
        return Tensor(self.data.mean(axis=dim, keepdims=keepdim))
    def max(self, dim=None):
        if dim is None: return float(self.data.max())
        m = self.data.max(axis=dim); i = self.data.argmax(axis=dim)
        return Tensor(m), Tensor(i.astype(np.int64))
    def min(self, dim=None):
        if dim is None: return float(self.data.min())
        m = self.data.min(axis=dim); i = self.data.argmin(axis=dim)
        return Tensor(m), Tensor(i.astype(np.int64))
    def abs(self): return Tensor(np.abs(self.data))
    def abs_(self): self.data = np.abs(self.data); return self
    def neg(self): return Tensor(-self.data)
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
        ns = (*sh[:start_dim], np.prod(sh[start_dim:end_dim+1]), *sh[end_dim+1:])
        return Tensor(d.reshape(ns))
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
        np.put_along_axis(data, idx, sd, axis=dim)
        self.data = data; return self
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
        axes = list(range(self.data.ndim)); axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return Tensor(np.transpose(self.data, axes))
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
    def backward(self, gradient=None): pass
    def det(self): return float(np.linalg.det(self.data))
    def diag(self, diagonal=0): return Tensor(np.diag(self.data, k=diagonal))
    def tril(self, k=0): return Tensor(np.tril(self.data, k=k))
    def triu(self, k=0): return Tensor(np.triu(self.data, k=k))
    def trace(self): return float(np.trace(self.data))


def _ensure(x):
    if isinstance(x, Tensor): return x
    if x is None: return None
    return Tensor(x)


def _add(a, b):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    bd = b.data if isinstance(b, Tensor) else np.asarray(b)
    return Tensor(ad + bd)


def _sub(a, b):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    bd = b.data if isinstance(b, Tensor) else np.asarray(b)
    return Tensor(ad - bd)


def _mul(a, b):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    bd = b.data if isinstance(b, Tensor) else np.asarray(b)
    return Tensor(ad * bd)


def _neg(a):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    return Tensor(-ad)


def _pow(a, p):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    return Tensor(ad ** p)


def _matmul(a, b):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    bd = b.data if isinstance(b, Tensor) else np.asarray(b)
    return Tensor(np.matmul(ad, bd))


def _transpose(x):
    return Tensor(x.data.T)


def tensor(data, dtype=None, device=None, requires_grad=False):
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


def zeros(*shape, dtype=np.float32, **kwargs):
    s = shape[0] if len(shape) == 1 else shape
    return Tensor(np.zeros(s, dtype=dtype))


def ones(*shape, dtype=np.float32, **kwargs):
    s = shape[0] if len(shape) == 1 else shape
    return Tensor(np.ones(s, dtype=dtype))


def randn(*shape, dtype=np.float32, **kwargs):
    s = shape[0] if len(shape) == 1 else shape
    return Tensor(np.random.randn(*s).astype(dtype), dtype=dtype)


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
    return Tensor(np.empty(s, dtype=dtype))


def zeros_like(t, dtype=None):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    dt = dtype or d.dtype
    return Tensor(np.zeros_like(d, dtype=dt))


def ones_like(t, dtype=None):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    dt = dtype or d.dtype
    return Tensor(np.ones_like(d, dtype=dt))


def full(size, fill_value, dtype=np.float32, **kwargs):
    if isinstance(size, int): size = (size,)
    return Tensor(np.full(size, fill_value, dtype=dtype))


def full_like(t, fill_value, dtype=None, **kwargs):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    dt = dtype or d.dtype
    return Tensor(np.full_like(d, fill_value, dtype=dt))


def exp(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.exp(np.clip(d, -700, 700)))


def log(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.log(np.maximum(d, 1e-8)))


def sqrt(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.sqrt(np.maximum(d, 0)))


def abs(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.abs(d))


def neg(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(-d)


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
    e = np.exp(d - d.max(axis=dim, keepdims=True))
    return Tensor(e / e.sum(axis=dim, keepdims=True))


def sigmoid(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(1.0 / (1.0 + np.exp(-np.clip(d, -500, 500))))


def tanh(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.tanh(d))


def relu(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.maximum(d, 0))


def leaky_relu(t, negative_slope=0.01):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.where(d > 0, d, d * negative_slope))


def gelu(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(0.5 * d * (1 + np.tanh(np.sqrt(2/np.pi) * (d + 0.044715 * d**3))))


class _TopKResult:
    def __init__(self, values, indices):
        self.values = values; self.indices = indices
    def __getitem__(self, key):
        r = self.values[key]
        return Tensor(r.data) if isinstance(r, Tensor) else Tensor(r)


def topk(t, k, dim=None, largest=True, sorted=True):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    if dim is None:
        flat = d.reshape(-1)
        if largest: idx = flat.argsort()[-k:][::-1]
        else: idx = flat.argsort()[:k]
        vals = flat[idx]
        return _TopKResult(Tensor(vals), Tensor(idx.astype(np.int64)))
    if largest: idx = d.argsort(axis=dim)[::-1][:,:k]
    else: idx = d.argsort(axis=dim)[:,:k]
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
    if x is None: return np.where(c)
    xd = x.data if isinstance(x, Tensor) else np.asarray(x)
    yd = y.data if isinstance(y, Tensor) else np.asarray(y)
    return Tensor(np.where(c, xd, yd))


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


class no_grad:
    def __enter__(self): return self
    def __exit__(self, *a): pass


class set_grad_enabled:
    def __init__(self, mode): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass


class autocast:
    def __init__(self, enabled=True, device_type="cpu", dtype=None): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass


class GradScaler:
    def __init__(self, init_scale=1.0, **kwargs): self._scale = init_scale
    def scale(self, loss): return loss * self._scale
    def step(self, opt): opt.step()
    def update(self): pass
    def get_scale(self): return self._scale


def save(state_dict, path_or_file, pickle_protocol=2):
    import pickle
    if isinstance(path_or_file, str):
        with open(path_or_file, "wb") as f:
            _save(state_dict, f, pickle_protocol)
    else:
        _save(state_dict, path_or_file, pickle_protocol)


def _save(state_dict, file_obj, pickle_protocol):
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
            with open(path, "rb") as f:
                return {"data": np.load(f)}
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
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    axes = list(range(d.ndim)); axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return Tensor(np.transpose(d, axes))


def flatten(t, start_dim=0, end_dim=-1):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    sh = list(d.shape)
    ns = (*sh[:start_dim], np.prod(sh[start_dim:end_dim+1]), *sh[end_dim+1:])
    return Tensor(d.reshape(ns))


def repeat_interleave(t, repeats, dim=0):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.repeat(d, repeats, axis=dim))


def meshgrid(*tensors, indexing="ij"):
    arrays = [t.data.flatten() for t in tensors]
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


class _NoParamOptimizer:
    def __init__(self, params, **kwargs):
        self.params = params if params else []
    def step(self): pass
    def zero_grad(self): pass
    def state(self): return {}
    def load_state_dict(self, d): pass
    def state_dict(self): return {}


class optim:
    class Adam(_NoParamOptimizer):
        def __init__(self, params, lr=0.001, **kwargs):
            super().__init__(params); self.lr = lr
    class SGD(_NoParamOptimizer):
        def __init__(self, params, lr=0.01, momentum=0, **kwargs):
            super().__init__(params); self.lr = lr
    class AdamW(_NoParamOptimizer):
        def __init__(self, params, lr=0.001, weight_decay=0.01, **kwargs):
            super().__init__(params); self.lr = lr


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
