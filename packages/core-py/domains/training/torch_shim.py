"""
torch compatibility shim — replaces torch with NumPy equivalents for inference.
No GPU dependency, no downloads. All ops are numpy-backed.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Optional, Tuple, Any


class Tensor:
    """Numpy-backed tensor matching torch API surface used in inference code."""

    _id_counter = 0

    def __init__(self, data, dtype=None, device=None, requires_grad=False):
        if isinstance(data, list):
            data = np.array(data, dtype=dtype or np.float32)
        elif not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype or np.float32)
        else:
            data = data.astype(np.float32) if np.float32 not in (data.dtype, dtype) else data.copy()
        self.data = data.astype(np.float32) if dtype == np.float32 else data.astype(np.float32)
        if dtype in (np.int64, np.int32, "long"):
            self.data = data.astype(np.int64)
        self.grad: Optional[Tensor] = None
        self.requires_grad = requires_grad
        self.shape = self.data.shape
        self.dtype = data.dtype
        self.device = device or "cpu"
        self.id = Tensor._id_counter
        Tensor._id_counter += 1

    def __repr__(self): return f"Tensor({self.shape}, dtype={self.dtype})"
    def __add__(self, other): return _add(self, _ensure(other))
    def __radd__(self, other): return _add(_ensure(other), self)
    def __sub__(self, other): return _sub(self, _ensure(other))
    def __rsub__(self, other): return _sub(_ensure(other), self)
    def __mul__(self, other): return _mul(self, _ensure(other))
    def __rmul__(self, other): return _mul(_ensure(other), self)
    def __neg__(self): return _neg(self)
    def __pow__(self, p): return _pow(self, p)
    def __matmul__(self, other): return _matmul(self, _ensure(other))
    def __getitem__(self, key): return Tensor(self.data[key], dtype=self.dtype, device=self.device)
    def __setitem__(self, key, value):
        val_data = value.data if isinstance(value, Tensor) else np.array(value)
        self.data[key] = val_data.astype(self.data.dtype)
    def tolist(self): return self.data.tolist()
    def item(self): return float(self.data.flat[0])
    def dim(self) -> int: return self.data.ndim
    def T(self): return _transpose(self)
    def reshape(self, *s): return Tensor(self.data.reshape(s))
    def sum(self, dim=None, keepdim=False):
        if dim is None: return Tensor(np.array(self.data.sum()))
        s = self.data.sum(axis=dim, keepdims=keepdim); return Tensor(s)
    def max(self, dim=None):
        if dim is None: return Tensor(np.array(self.data.max()))
        m = self.data.max(axis=dim); i = self.data.argmax(axis=dim); return Tensor(m), Tensor(i.astype(np.int64))
    def squeeze(self, dim=None):
        if dim is None: return Tensor(self.data.squeeze())
        return Tensor(self.data.squeeze(axis=dim))
    def unsqueeze(self, dim: int): return Tensor(np.expand_dims(self.data, axis=dim))
    def repeat(self, *sizes): return Tensor(np.tile(self.data, sizes))
    def scatter_(self, dim, index, src):
        data = self.data.copy()
        idx = index.data.astype(int)
        if isinstance(src, Tensor): src = src.data
        np.put_along_axis(data, idx, src, axis=dim)
        self.data = data; return self
    def gather(self, dim, index):
        idx = index.data.astype(int); return Tensor(np.take_along_axis(self.data, idx, axis=dim))
    def clamp_(min, max): self.data = np.clip(self.data, min, max); return self
    def zero_(self): self.data.fill(0); return self
    def fill_(v): self.data.fill(v); return self
    def long(self): return Tensor(self.data.astype(np.int64), dtype=np.int64)


def _ensure(x):
    if isinstance(x, Tensor): return x
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
    return Tensor(np.zeros(shape if len(shape) == 1 else shape[0], dtype=dtype), dtype=dtype)


def ones(*shape, dtype=np.float32, **kwargs):
    return Tensor(np.ones(shape if len(shape) == 1 else shape[0], dtype=dtype), dtype=dtype)


def randn(*shape, dtype=np.float32, **kwargs):
    return Tensor(np.random.randn(*shape).astype(dtype), dtype=dtype)


def randint(low, high, size):
    if isinstance(size, int): size = (size,)
    return Tensor(np.random.randint(low, high, size), dtype=np.int64)


def arange(start, end=None, step=1, dtype=np.float32):
    if end is None: start, end = 0, start
    return Tensor(np.arange(start, end, step).astype(dtype), dtype=dtype)


def empty(*shape, dtype=np.float32, **kwargs):
    return Tensor(np.empty(shape if len(shape) == 1 else shape[0], dtype=dtype), dtype=dtype)


def zeros_like(t, dtype=None):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    dt = dtype or d.dtype
    return Tensor(np.zeros_like(d, dtype=dt))


def ones_like(t, dtype=None):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    dt = dtype or d.dtype
    return Tensor(np.ones_like(d, dtype=dt))


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


def sum(t, dim=None, keepdim=False):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    if dim is None: return float(d.sum())
    return Tensor(d.sum(axis=dim, keepdims=keepdim))


def mean(t, dim=None, keepdim=False):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    if dim is None: return float(d.mean())
    return Tensor(d.mean(axis=dim, keepdims=keepdim))


def max(t, dim=None):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    if dim is None: return float(d.max())
    m = d.max(axis=dim); i = d.argmax(axis=dim)
    return Tensor(m), Tensor(i.astype(np.int64))


def min(t, dim=None):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    if dim is None: return float(d.min())
    m = d.min(axis=dim); i = d.argmin(axis=dim)
    return Tensor(m), Tensor(i.astype(np.int64))


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


def topk(t, k, dim=None, largest=True, sorted=True):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    if dim is None:
        flat = d.reshape(-1)
        indices = flat.argsort()[-k:][::-1]
        return _TopKResult(Tensor(flat[indices]), Tensor(indices.astype(np.int64)))
    indices = d.argsort(axis=dim)[::-1 if largest else 1][:,:k]
    values = np.take_along_axis(d, indices, axis=dim)
    return _TopKResult(Tensor(values), Tensor(indices.astype(np.int64)))


class _TopKResult:
    def __init__(self, values, indices):
        self.values = values
        self.indices = indices
    def __getitem__(self, key): return self.values[key]


def multinomial(t, num_samples, replacement=False):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    flat = d.reshape(-1).astype(np.float64)
    flat = np.maximum(flat, 0)
    total = flat.sum()
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


def concatenate(tensors, dim=0):
    arrays = [t.data if isinstance(t, Tensor) else np.asarray(t) for t in tensors]
    return Tensor(np.concatenate(arrays, axis=dim))


def where(condition, a=None, b=None):
    c = condition.data if isinstance(condition, Tensor) else np.asarray(condition)
    if a is None:
        return np.where(c)
    aa = a.data if isinstance(a, Tensor) else np.asarray(a)
    bb = b.data if isinstance(b, Tensor) else np.asarray(b)
    return Tensor(np.where(c, aa, bb))


def isfinite(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return np.isfinite(d)


defisfinite = isfinite


def isinf(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return np.isinf(d)


def isnan(t):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return np.isnan(d)


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


class set_enable_grad:
    def __init__(self, enabled): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass


def save(state_dict, path):
    import pickle
    with open(path, "wb") as f:
        save_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, Tensor):
                save_dict[k] = v.data
            else:
                save_dict[k] = v
        pickle.dump(save_dict, f)


def load(path, map_location=None, weights_only=False):
    import pickle, io
    with open(path, "rb") as f:
        return pickle.load(f)


def clamp(t, min_val=None, max_val=None):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.clip(d, min_val, max_val))


def argmax(t, dim=None):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    if dim is None: return int(d.argmax())
    return Tensor(d.argmax(axis=dim).astype(np.int64))


def argmin(t, dim=None):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    if dim is None: return int(d.argmin())
    return Tensor(d.argmin(axis=dim).astype(np.int64))


def sort(t, dim=None, descending=False):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    if dim is None:
        idx = d.flatten().argsort()
        if descending: idx = idx[::-1]
        return Tensor(d.flatten()[idx])
    idx = d.argsort(axis=dim)
    if descending: idx = idx[:, ::-1] if dim == 1 else idx[::-1]
    vals = np.take_along_axis(d, idx, axis=dim)
    return Tensor(vals), Tensor(idx.astype(np.int64))


def argsort(t, dim=None, descending=False):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    if dim is None:
        idx = d.flatten().argsort()
        if descending: idx = idx[::-1]
        return Tensor(idx.astype(np.int64))
    idx = d.argsort(axis=dim)
    if descending: idx = idx[:, ::-1] if dim == 1 else idx[::-1]
    return Tensor(idx.astype(np.int64))


def split(t, split_size, dim=0):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    if isinstance(split_size, int):
        return [Tensor(x) for x in np.split(d, list(range(split_size, d.shape[dim], split_size)), axis=dim)]
    return [Tensor(x) for x in np.split(d, np.cumsum(split_size[:-1]), axis=dim)]


def chunk(t, chunks, dim=0):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return [Tensor(x) for x in np.array_split(d, chunks, axis=dim)]


def narrow(t, dim, start, length):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(d[(slice(None),) * dim + (slice(start, start + length),)])


def permute(t, *dims):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.transpose(d, dims))


def transpose(t, dim0, dim1):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    axes = list(range(d.ndim)); axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return Tensor(np.transpose(d, axes))


def view(t, *shape):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(d.reshape(shape))


def flatten(t, start_dim=0, end_dim=-1):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    if end_dim < 0: end_dim = d.ndim + end_dim
    shape = list(d.shape)
    new_shape = (*shape[:start_dim], np.prod(shape[start_dim:end_dim+1]), *shape[end_dim+1:])
    return Tensor(d.reshape(new_shape))


def clamp_(t, min_val=None, max_val=None):
    t.data = np.clip(t.data, min_val, max_val); return t


def masked_fill_(t, mask, value):
    mask_d = mask.data if isinstance(mask, Tensor) else np.asarray(mask)
    t.data[mask_d] = value; return t


def index_select(t, dim, index):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    idx = index.data.astype(int).flatten()
    return Tensor(np.take(d, idx, axis=dim))


def matmul(a, b):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    bd = b.data if isinstance(b, Tensor) else np.asarray(b)
    return Tensor(np.matmul(ad, bd))


def dot(a, b):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    bd = b.data if isinstance(b, Tensor) else np.asarray(b)
    return Tensor(np.dot(ad.flatten(), bd.flatten()))


def ger(a, b):
    ad = a.data.flatten() if isinstance(a, Tensor) else np.asarray(a).flatten()
    bd = b.data.flatten() if isinstance(b, Tensor) else np.asarray(b).flatten()
    return Tensor(np.outer(ad, bd))


def norm(t, p="fro", dim=None):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    if dim is None: return float(np.linalg.norm(d.flatten(), ord=p if p != "fro" else None))
    return Tensor(np.linalg.norm(d, ord=p if p != "fro" else None, axis=dim, keepdims=True))


def bmm(a, b):
    ad = a.data if isinstance(a, Tensor) else np.asarray(a)
    bd = b.data if isinstance(b, Tensor) else np.asarray(b)
    return Tensor(np.batched_matmul(ad, bd))


def cumsum(t, dim=0):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.cumsum(d, axis=dim))


def cumprod(t, dim=0):
    d = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.cumprod(d, axis=dim))


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


def abs_(t):
    t.data = np.abs(t.data); return t


def neg_(t):
    t.data = np.neg(t.data); return t


def pow_(t, p):
    t.data = t.data ** p; return t


class device:
    def __init__(self, spec): self.spec = spec


class dtype:
    def __init__(self, spec): self._spec = spec
    def __eq__(self, other):
        if other is torch.float32: return self._spec == np.float32
        if other is torch.int64: return self._spec == np.int64
        return False


float32 = dtype(np.float32)
int64 = dtype(np.int64)
int32 = dtype(np.int32)
float64 = dtype(np.float64)
long = int64
bfloat16 = dtype(np.float32)


class cuda:
    @staticmethod
    def is_available() -> bool: return False


class mps:
    @staticmethod
    def is_available() -> bool: return False


class backends:
    class mps:
        is_available = lambda: False


class distributed:
    @staticmethod
    def is_available() -> bool: return False


def no_grad():
    return no_grad()


class _NoParamOptimizer:
    def __init__(self, params, **kwargs): self.params = params
    def step(self): pass
    def zero_grad(self): pass


class optim:
    class Adam(_NoParamOptimizer): pass
    class SGD(_NoParamOptimizer): pass
    class AdamW(_NoParamOptimizer): pass


def meshgrid(*args, **kwargs): return meshgrid(*args, **kwargs)


__version__ = "2.1.0-shim"


class nn:
    class Module:
        def __init__(self): pass
        def parameters(self): return []
        def named_parameters(self): return []
        def state_dict(self): return {}
        def load_state_dict(self, d): pass
        def train(self): return self
        def eval(self): return self
        def to(self, *args, **kwargs): return self
        def cpu(self): return self
        def cuda(self): return self
        def half(self): return self
        def float(self): return self
        def zero_grad(self): pass

    class Linear(Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.weight = randn(out_features, in_features)
            self.bias = zeros(out_features) if bias else None

    class Embedding(Module):
        def __init__(self, num_embeddings, embedding_dim):
            super().__init__()
            self.weight = randn(num_embeddings, embedding_dim)

    class LayerNorm(Module):
        def __init__(self, features, eps=1e-5):
            super().__init__()
            self.weight = ones(features)
            self.bias = zeros(features)

    class Dropout(Module):
        def __init__(self, p=0.5): super().__init__(); self.p = p
        def forward(self, x): return x


def compile(model, **kwargs): return model


def device_count() -> int: return 0


def manual_seed(seed): np.random.seed(seed)


class Generator:
    def __init__(self, seed=0): np.random.seed(seed)


__all__ = [
    "Tensor", "tensor", "zeros", "ones", "randn", "randint", "arange", "empty",
    "zeros_like", "ones_like", "exp", "log", "sqrt", "abs",
    "sum", "mean", "max", "min", "softmax", "sigmoid", "tanh", "relu",
    "topk", "multinomial", "cat", "stack", "concatenate",
    "where", "isfinite", "isinf", "isnan", "allclose",
    "eq", "ne", "gt", "lt", "ge", "le",
    "no_grad", "save", "load", "clamp", "clamp_",
    "argmax", "argmin", "sort", "argsort",
    "split", "chunk", "narrow", "permute", "transpose", "view", "flatten",
    "masked_fill_", "index_select", "matmul", "dot", "ger", "norm", "bmm",
    "cumsum", "cumprod", "repeat_interleave", "meshgrid", "diag", "tril", "triu", "trace",
    "abs_", "neg_", "pow_",
    "device", "dtype", "float32", "int64", "int32", "float64", "long", "bfloat16",
    "cuda", "mps", "backends", "distributed", "optim", "device_count", "manual_seed", "Generator",
    "compile", "nn", "__version__",
]
