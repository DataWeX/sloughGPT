"""
SloNet Torch Compatibility Module

Provides a drop-in ``torch`` replacement backed by SloNet/numpy.
Change::

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

to::

    from domains.training.slonet_compat import torch
    import torch.nn as nn
    import torch.nn.functional as F

Everything works identically without PyTorch installed.
"""

import math

import numpy as np

from domains.training.slonet import (
    Tensor as _Tensor,
    no_grad as _no_grad,
    SloLayer as _SoulLayer,
    SloLinear as _SoulLinear,
    SloEmbedding as _SoulEmbedding,
    SloDropout as _SoulDropout,
    SloAdam as _SoulAdam,
    SloSGD as _SoulSGD,
    cross_entropy as _cross_entropy,
    mse_loss as _mse_loss,
    kl_div_loss as _kl_div_loss,
    log_softmax as _log_softmax,
    softmax as _softmax,
    normalize as _normalize,
    pairwise_distance as _pairwise_distance,
    relu as _relu,
    sigmoid as _sigmoid,
    tanh as _tanh,
    gelu as _gelu,
    silu as _silu,
    eye as _eye,
    stack as _stack,
    cat as _cat,
    where as _where,
    topk as _topk,
    argmax as _argmax,
    argmin as _argmin,
    squeeze as _squeeze,
    unsqueeze as _unsqueeze,
    SloDataset as _SoulDataset,
    SloDataLoader as _SoulDataLoader,
    WarmupCosineScheduler as _WarmupCosine,
    PolynomialDecayScheduler as _Polynomial,
    LinearWarmupScheduler as _LinearWarmup,
    SloStepLR as _SoulStepLR,
    SloCosineAnnealingLR as _SoulCosineAnnealing,
    SloReduceLROnPlateau as _SoulReduceLROnPlateau,
    SloConstantLR as _SoulConstantLR,
    SloOneCycleLR as _SoulOneCycleLR,
    SloCyclicLR as _SoulCyclicLR,
    SloLRScheduler as _SoulLRScheduler,
    SloNet as _SoulNet,
    isfinite as _isfinite,
    multinomial as _multinomial,
    randint as _randint,
)


# =============================================================================
# Tensor
# =============================================================================


class Tensor(_Tensor):
    """SloNet Tensor with extra torch‑compatible type‑hint surface."""

    def __init__(self, data, requires_grad: bool = False, _children: tuple = (), _copy: bool = True, dtype=None):
        if dtype is not None:
            real_dtype = _resolve_dtype(dtype)
            if isinstance(data, np.ndarray):
                data = data.astype(real_dtype)
            elif isinstance(data, (list, memoryview)):
                data = np.array(data, dtype=real_dtype)
            elif hasattr(data, 'detach'):
                data = data.detach().cpu().numpy().astype(real_dtype)
            else:
                data = np.asarray(data, dtype=real_dtype)
            self.data = np.asarray(data)
            self.grad = None
            self.requires_grad = requires_grad
            self._children = _children
            self._backward_fn = None
            self.shape = self.data.shape
            self.id = Tensor._id_counter
            Tensor._id_counter += 1
        elif isinstance(data, np.ndarray) and data.dtype != np.float32:
            self.data = data.copy() if _copy else data
            self.grad = None
            self.requires_grad = requires_grad
            self._children = _children
            self._backward_fn = None
            self.shape = self.data.shape
            self.id = Tensor._id_counter
            Tensor._id_counter += 1
        else:
            super().__init__(data, requires_grad=requires_grad, _children=_children, _copy=_copy)

    @property
    def device(self):
        return _TorchNamespace().device("cpu")

    @property
    def dtype(self):
        return np.dtype(self.data.dtype).type

    def split(self, split_size, dim=0):
        n = self.data.shape[dim]
        sections = list(range(split_size, n, split_size))
        return tuple(Tensor(np.split(self.data, sections, axis=dim), requires_grad=self.requires_grad))

    def all(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(np.array(np.all(self.data), dtype=np.float32), requires_grad=False)
        return Tensor(np.all(self.data, axis=dim, keepdims=keepdim).astype(np.float32), requires_grad=False)

    def any(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(np.array(np.any(self.data), dtype=np.float32), requires_grad=False)
        return Tensor(np.any(self.data, axis=dim, keepdims=keepdim).astype(np.float32), requires_grad=False)



    def norm(self, p=2, dim=None, keepdim=False):
        if dim is not None:
            return Tensor(np.linalg.norm(self.data, ord=p, axis=dim, keepdims=keepdim), requires_grad=False)
        return Tensor(np.array(np.linalg.norm(self.data, ord=p)), requires_grad=False)

    def cos(self):
        return Tensor(np.cos(self.data), requires_grad=self.requires_grad)

    def sin(self):
        return Tensor(np.sin(self.data), requires_grad=self.requires_grad)

    def pow(self, exp):
        return Tensor(np.power(self.data, exp), requires_grad=self.requires_grad)

    def scatter_(self, dim, index, src):
        d = self.data.copy()
        if isinstance(src, Tensor):
            src = src.data
        idx = index.data if isinstance(index, Tensor) else np.asarray(index)
        if dim == 0:
            d[idx, np.arange(d.shape[1])] = src
        elif dim == 1:
            d[np.arange(d.shape[0])[:, None], idx] = src
        self.data[:] = d
        return self

    def scatter(self, dim, index, src):
        return self.clone().scatter_(dim, index, src)

    def amax(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(np.array(self.data.max(), dtype=np.float32), requires_grad=False)
        return Tensor(self.data.max(axis=dim, keepdims=keepdim), requires_grad=False)

    def var(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(np.array(self.data.var(), dtype=np.float32), requires_grad=False)
        return Tensor(self.data.var(axis=dim, keepdims=keepdim), requires_grad=False)

    def element_size(self):
        return self.data.dtype.itemsize

    def masked_fill(self, mask, value):
        result = self.data.copy()
        mask_data = _data(mask)
        result[mask_data] = value
        return Tensor(result, requires_grad=False)


# =============================================================================
# no_grad
# =============================================================================


class no_grad:
    """Drop‑in for ``torch.no_grad`` — context manager / decorator."""
    def __init__(self):
        self._ctx = _no_grad()
    def __enter__(self):
        return self._ctx.__enter__()
    def __exit__(self, *a):
        return self._ctx.__exit__(*a)
    def __call__(self, func):
        return self._ctx(func)


# =============================================================================
# nn — Neural Network module equivalents
# =============================================================================


class _NNModule:
    """Namespace that lazily provides ``nn.Module``, ``nn.Linear``, etc."""

    class Module(_SoulLayer):
        """Drop‑in for ``torch.nn.Module``, backed by SloLayer."""
        def parameters(self):
            return list(self._collect_params())

        def _collect_params(self):
            params = []
            for attr_name in dir(self):
                attr = getattr(self, attr_name, None)
                if isinstance(attr, Tensor):
                    if attr.requires_grad:
                        params.append(attr)
                elif isinstance(attr, _SoulLayer):
                    params.extend(attr.parameters())
            return params

        def named_parameters(self, prefix=""):
            result = []
            for i, p in enumerate(self.parameters()):
                result.append((f"{prefix}p{i}", p))
            return result

        def named_modules(self, prefix=""):
            result = [(prefix, self)]
            for attr_name in dir(self):
                attr = getattr(self, attr_name, None)
                if isinstance(attr, _SoulLayer) and attr is not self:
                    result.extend(attr.named_modules(prefix=f"{prefix}.{attr_name}" if prefix else attr_name))
            return result

        def children(self):
            for attr_name in dir(self):
                attr = getattr(self, attr_name, None)
                if isinstance(attr, _SoulLayer) and attr is not self:
                    yield attr

        def apply(self, fn):
            fn(self)
            for child in self.children():
                if hasattr(child, 'apply'):
                    child.apply(fn)
            return self

        def zero_grad(self):
            for p in self.parameters():
                p.grad = None

        def to(self, *args, **kwargs):
            if args:
                arg = args[0]
                if isinstance(arg, (np.dtype, type(np.float32))):
                    t = Tensor(self.data.astype(arg), requires_grad=self.requires_grad)
                    t.grad = self.grad
                    return t
            return self

        def train(self, mode=True):
            for child in self.children():
                if hasattr(child, 'train'):
                    child.train(mode)
            return self

        def eval(self):
            return self.train(False)

        def state_dict(self):
            return {}

        def load_state_dict(self, d, strict=True):
            return []

        def register_buffer(self, name, tensor, persistent=True):
            setattr(self, name, tensor)

        def register_parameter(self, name, param):
            setattr(self, name, param)

        def add_module(self, name, module):
            setattr(self, name, module)

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    class Linear(_SoulLinear):
        """Drop‑in for ``torch.nn.Linear``."""
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            super().__init__(in_features, out_features)
            if not bias:
                self.bias = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=False)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = Tensor(
                (np.random.randn(out_features, in_features) * math.sqrt(2.0 / (in_features + out_features))).astype(np.float32),
                requires_grad=True,
            )
        def apply(self, fn):
            fn(self)
            return self

    class Embedding(_SoulEmbedding):
        """Drop‑in for ``torch.nn.Embedding``."""
        def __init__(self, num_embeddings, embedding_dim, padding_idx=None, device=None, dtype=None):
            super().__init__(num_embeddings, embedding_dim)
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.padding_idx = padding_idx

    class Dropout(_SoulDropout):
        """Drop‑in for ``torch.nn.Dropout``."""
        def __init__(self, p=0.5, inplace=False):
            super().__init__(p=p)

    class LayerNorm(_SoulLayer):
        """Drop‑in for ``torch.nn.LayerNorm`` — numpy-based."""
        def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, device=None, dtype=None):
            super().__init__("LayerNorm")
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = normalized_shape
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            shape = tuple(normalized_shape)
            if elementwise_affine:
                self.weight = Tensor(np.ones(shape, dtype=np.float32), requires_grad=True)
                self.bias = Tensor(np.zeros(shape, dtype=np.float32), requires_grad=True)
        def forward(self, x):
            d = _data(x)
            axis = tuple(range(-len(self.normalized_shape), 0)) if self.normalized_shape else None
            mean = d.mean(axis=axis, keepdims=True)
            var = d.var(axis=axis, keepdims=True)
            out = (d - mean) / np.sqrt(var + self.eps)
            if self.elementwise_affine:
                out = out * _data(self.weight) + _data(self.bias)
            if isinstance(x, Tensor):
                return Tensor(out, requires_grad=x.requires_grad)
            return out

    class Conv1d(_SoulLayer):
        """Drop‑in for ``torch.nn.Conv1d`` — stub."""
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, device=None, dtype=None):
            super().__init__("Conv1d")
            self.in_channels = in_channels; self.out_channels = out_channels
            self.kernel_size = kernel_size; self.stride = stride
            self.padding = padding; self.dilation = dilation; self.groups = groups
            self.weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size).astype(np.float32) * 0.1, requires_grad=True)
            self.bias = Tensor(np.zeros(out_channels, dtype=np.float32), requires_grad=True) if bias else None
        def forward(self, x): return x

    class Conv2d(_SoulLayer):
        """Drop‑in for ``torch.nn.Conv2d`` — numpy-based."""
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
            super().__init__("Conv2d")
            self.in_channels = in_channels; self.out_channels = out_channels
            self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
            self.stride = stride if isinstance(stride, tuple) else (stride, stride)
            self.padding = padding if isinstance(padding, tuple) else (padding, padding)
            self.dilation = dilation; self.groups = groups; self.padding_mode = padding_mode
            k = np.prod(self.kernel_size) * in_channels
            self.weight = Tensor(np.random.randn(out_channels, in_channels, *self.kernel_size).astype(np.float32) * np.sqrt(2.0 / k), requires_grad=True)
            self.bias = Tensor(np.zeros(out_channels, dtype=np.float32), requires_grad=True) if bias else None
        def forward(self, x):
            from domains.training.slonet import _conv2d as _do_conv2d
            return _do_conv2d(x, self.weight, stride=self.stride, padding=self.padding)

    class BatchNorm1d(_SoulLayer):
        """Drop‑in for ``torch.nn.BatchNorm1d`` — stub."""
        def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
            super().__init__("BatchNorm1d")
            self.num_features = num_features; self.eps = eps; self.momentum = momentum
            self.weight = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=affine)
            self.bias = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=affine)
        def forward(self, x): return (x - x.mean()) / np.sqrt(x.var() + self.eps) * self.weight + self.bias if hasattr(x, 'mean') else x

    class BatchNorm2d(_SoulLayer):
        """Drop‑in for ``torch.nn.BatchNorm2d`` — numpy-based."""
        def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
            super().__init__("BatchNorm2d")
            self.num_features = num_features; self.eps = eps; self.momentum = momentum
            self.weight = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=affine)
            self.bias = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=affine)
        def forward(self, x):
            d = _data(x)
            if d.ndim < 4:
                d = d.reshape(d.shape[0], d.shape[1], 1, 1) if d.ndim == 2 else d.reshape(d.shape[0], d.shape[1], 1)
            c = d.shape[1]
            mean = d.mean(axis=(0, 2, 3), keepdims=True)
            var = d.var(axis=(0, 2, 3), keepdims=True)
            out = (d - mean) / np.sqrt(var + self.eps)
            w = _data(self.weight).reshape(1, c, 1, 1)
            b = _data(self.bias).reshape(1, c, 1, 1)
            out = out * w + b
            if isinstance(x, Tensor):
                return Tensor(out, requires_grad=x.requires_grad)
            return out

    class MaxPool2d(_SoulLayer):
        """Drop‑in for ``torch.nn.MaxPool2d`` — numpy-based."""
        def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
            super().__init__("MaxPool2d")
            self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
            self.stride = stride if stride else self.kernel_size
            if isinstance(self.stride, int): self.stride = (self.stride, self.stride)
            self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        def forward(self, x):
            d = _data(x)
            n, c, h, w = d.shape
            kh, kw = self.kernel_size
            sh, sw = self.stride
            ph, pw = self.padding
            if ph or pw:
                d = np.pad(d, ((0,0),(0,0),(ph,ph),(pw,pw)), mode='constant')
            oh = (h + 2*ph - kh) // sh + 1
            ow = (w + 2*pw - kw) // sw + 1
            out = np.zeros((n, c, oh, ow), dtype=d.dtype)
            for i in range(oh):
                for j in range(ow):
                    out[:, :, i, j] = d[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw].max(axis=(2,3))
            if isinstance(x, Tensor):
                return Tensor(out, requires_grad=x.requires_grad)
            return out

    class AvgPool2d(_SoulLayer):
        """Drop‑in for ``torch.nn.AvgPool2d`` — numpy-based."""
        def __init__(self, kernel_size, stride=None, padding=0):
            super().__init__("AvgPool2d")
            self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
            self.stride = stride if stride else self.kernel_size
            if isinstance(self.stride, int): self.stride = (self.stride, self.stride)
            self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        def forward(self, x):
            d = _data(x)
            n, c, h, w = d.shape
            kh, kw = self.kernel_size
            sh, sw = self.stride
            ph, pw = self.padding
            if ph or pw:
                d = np.pad(d, ((0,0),(0,0),(ph,ph),(pw,pw)), mode='constant')
            oh = (h + 2*ph - kh) // sh + 1
            ow = (w + 2*pw - kw) // sw + 1
            out = np.zeros((n, c, oh, ow), dtype=d.dtype)
            for i in range(oh):
                for j in range(ow):
                    out[:, :, i, j] = d[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw].mean(axis=(2,3))
            if isinstance(x, Tensor):
                return Tensor(out, requires_grad=x.requires_grad)
            return out

    class GRU(_SoulLayer):
        """Drop‑in for ``torch.nn.GRU`` — stub."""
        def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False, device=None, dtype=None):
            super().__init__("GRU")
            self.hidden_size = hidden_size; self.num_layers = num_layers; self.batch_first = batch_first
        def forward(self, x, h=None):
            if self.batch_first:
                b, s, d = _data(x).shape
            else:
                s, b, d = _data(x).shape
            out = Tensor(np.zeros((s, b, self.hidden_size), dtype=np.float32), requires_grad=False)
            return out, Tensor(np.zeros((self.num_layers, b, self.hidden_size), dtype=np.float32), requires_grad=False)

    class LSTM(_SoulLayer):
        """Drop‑in for ``torch.nn.LSTM`` — stub."""
        def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False, proj_size=0, device=None, dtype=None):
            super().__init__("LSTM")
            self.hidden_size = hidden_size; self.num_layers = num_layers; self.batch_first = batch_first
        def forward(self, x, hx=None):
            if self.batch_first:
                b, s, d = _data(x).shape
            else:
                s, b, d = _data(x).shape
            out = Tensor(np.zeros((s, b, self.hidden_size), dtype=np.float32), requires_grad=False)
            h = Tensor(np.zeros((self.num_layers, b, self.hidden_size), dtype=np.float32), requires_grad=False)
            c = Tensor(np.zeros((self.num_layers, b, self.hidden_size), dtype=np.float32), requires_grad=False)
            return out, (h, c)

    class MultiheadAttention(_SoulLayer):
        """Drop‑in for ``torch.nn.MultiheadAttention`` — stub."""
        def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
            super().__init__("MultiheadAttention")
            self.embed_dim = embed_dim; self.num_heads = num_heads; self.batch_first = batch_first
        def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
            d = _data(query)
            if self.batch_first:
                out = Tensor(d, requires_grad=False)
            else:
                out = Tensor(d, requires_grad=False)
            return out, None

    class PReLU(_SoulLayer):
        """Drop‑in for ``torch.nn.PReLU`` — stub."""
        def __init__(self, num_parameters=1, init=0.25, device=None, dtype=None):
            super().__init__("PReLU")
            self.weight = Tensor(np.array([init], dtype=np.float32), requires_grad=True)
        def forward(self, x):
            d = _data(x)
            w = _data(self.weight)
            out = np.where(d > 0, d, d * w)
            if isinstance(x, Tensor):
                return Tensor(out, requires_grad=x.requires_grad)
            return out

    class Identity:
        """Identity layer — returns input unchanged."""
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x
        def forward(self, x): return x
        def parameters(self): return []
        def train(self, mode=True): pass
        def eval(self): pass
        def to(self, device): return self

    class Parameter(Tensor):
        """Drop‑in for ``torch.nn.Parameter``."""
        def __new__(cls, data, requires_grad=True):
            if isinstance(data, Tensor):
                data.requires_grad = requires_grad
                result = object.__new__(cls)
                result.data = data.data.copy()
                result.grad = data.grad
                result.requires_grad = requires_grad
                result._children = data._children
                result._backward_fn = data._backward_fn
                result.shape = data.shape
                return result
            if isinstance(data, np.ndarray):
                return object.__new__(cls)
            return object.__new__(cls)

        def __init__(self, data, requires_grad=True):
            if not hasattr(self, 'data'):
                Tensor.__init__(self, data, requires_grad=requires_grad)

    class CrossEntropyLoss:
        """Cross‑entropy loss."""
        def __call__(self, logits, targets):
            return _cross_entropy(_to_tensor(logits), _to_tensor(targets))
        def forward(self, logits, targets):
            return self(logits, targets)
        def parameters(self): return []

    class MSELoss:
        """MSE loss."""
        def __call__(self, a, b):
            return _mse_loss(_to_tensor(a), _to_tensor(b))
        def forward(self, a, b):
            return self(a, b)
        def parameters(self): return []

    class KLDivLoss:
        """KL divergence loss."""
        def __init__(self, reduction="batchmean"):
            self.reduction = reduction
        def __call__(self, input, target):
            return _kl_div_loss(_to_tensor(input), _to_tensor(target), reduction=self.reduction)
        def forward(self, input, target):
            return self(input, target)
        def parameters(self): return []

    class ModuleList(list):
        """``torch.nn.ModuleList`` — wraps list for model parameter collection."""
        def parameters(self):
            params = []
            for item in self:
                if hasattr(item, 'parameters'):
                    params.extend(item.parameters())
            return params
        def named_modules(self, prefix=""):
            result = []
            for i, item in enumerate(self):
                result.extend(item.named_modules(prefix=f"{prefix}.{i}"))
            return result

    class Sequential(_SoulLayer):
        """``torch.nn.Sequential`` — chain of layers."""
        def __init__(self, *layers):
            super().__init__("Sequential")
            self._layers = list(layers)
        def forward(self, x):
            for l in self._layers:
                x = l(x) if callable(l) else x
            return x
        def parameters(self):
            params = []
            for l in self._layers:
                if hasattr(l, 'parameters'):
                    params.extend(l.parameters())
            return params
        def __getitem__(self, i):
            return self._layers[i]
        def __len__(self):
            return len(self._layers)
        def __iter__(self):
            return iter(self._layers)
        def append(self, module):
            self._layers.append(module)

    # Activation aliases
    ReLU = staticmethod(lambda: _ReLU())
    GELU = staticmethod(lambda: _GELU())
    SiLU = staticmethod(lambda: _SiLU())
    Sigmoid = staticmethod(lambda: _Sigmoid())
    Tanh = staticmethod(lambda: _Tanh())
    Softmax = staticmethod(lambda dim=None: _Softmax(dim))

    # functional assigned below after _Functional is defined
    functional = None
    # init assigned below after _Init is defined
    init = None

    class utils:
        """``torch.nn.utils`` — gradient clipping stubs."""
        @staticmethod
        def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
            """Clip gradients (no‑op stub for compat)."""
            return 0.0

    class parallel:
        """``torch.nn.parallel`` — distributed data parallel stub."""
        class DistributedDataParallel:
            def __init__(self, module, **kwargs):
                self.module = module
            def __getattr__(self, name):
                return getattr(self.module, name)
            def parameters(self):
                return self.module.parameters()
            def forward(self, *a, **kw):
                return self.module(*a, **kw)
            def train(self, mode=True):
                return self.module.train(mode)
            def eval(self):
                return self.module.eval()
            def state_dict(self, **kw):
                return self.module.state_dict(**kw)
            def load_state_dict(self, sd, **kw):
                return self.module.load_state_dict(sd, **kw)
            def named_parameters(self, **kw):
                return self.module.named_parameters(**kw)
            def zero_grad(self):
                self.module.zero_grad()


class _ReLU:
    def forward(self, x): return _relu(_to_tensor(x))
    def __call__(self, x): return self.forward(x)
    def parameters(self): return []

class _GELU:
    def forward(self, x): return _gelu(_to_tensor(x))
    def __call__(self, x): return self.forward(x)
    def parameters(self): return []

class _SiLU:
    def forward(self, x): return _silu(_to_tensor(x))
    def __call__(self, x): return self.forward(x)
    def parameters(self): return []

class _Sigmoid:
    def forward(self, x): return _sigmoid(_to_tensor(x))
    def __call__(self, x): return self.forward(x)
    def parameters(self): return []

class _Tanh:
    def forward(self, x): return _tanh(_to_tensor(x))
    def __call__(self, x): return self.forward(x)
    def parameters(self): return []

class _Softmax:
    def __init__(self, dim=None): self.dim = dim
    def forward(self, x): return _softmax(_to_tensor(x), dim=self.dim or -1)
    def __call__(self, x): return self.forward(x)
    def parameters(self): return []


# =============================================================================
# nn.init — Weight initialization
# =============================================================================


class _Init:
    """``torch.nn.init`` — weight initialization helpers."""

    @staticmethod
    def kaiming_uniform_(tensor, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu'):
        if isinstance(tensor, Tensor):
            fan = tensor.shape[1] if len(tensor.shape) >= 2 else tensor.shape[0]
            bound = math.sqrt(3.0 / fan)
            tensor.data[:] = np.random.uniform(-bound, bound, tensor.shape).astype(np.float32)
        return tensor

    @staticmethod
    def xavier_uniform_(tensor, gain=1.0):
        if isinstance(tensor, Tensor) and len(tensor.shape) >= 2:
            a = gain * math.sqrt(6.0 / (tensor.shape[0] + tensor.shape[1]))
            tensor.data[:] = np.random.uniform(-a, a, tensor.shape).astype(np.float32)
        return tensor

    @staticmethod
    def zeros_(tensor):
        if isinstance(tensor, Tensor):
            tensor.data.fill(0.0)
        return tensor

    @staticmethod
    def normal_(tensor, mean=0.0, std=1.0):
        if isinstance(tensor, Tensor):
            tensor.data[:] = (np.random.randn(*tensor.data.shape) * std + mean).astype(np.float32)
        return tensor

    @staticmethod
    def ones_(tensor):
        if isinstance(tensor, Tensor):
            tensor.data.fill(1.0)
        return tensor

    @staticmethod
    def constant_(tensor, val):
        if isinstance(tensor, Tensor):
            tensor.data.fill(val)
        return tensor

    @staticmethod
    def xavier_normal_(tensor, gain=1.0):
        if isinstance(tensor, Tensor) and len(tensor.data.shape) >= 2:
            std = gain * math.sqrt(2.0 / (tensor.data.shape[0] + tensor.data.shape[1]))
            tensor.data[:] = (np.random.randn(*tensor.data.shape) * std).astype(np.float32)
        return tensor

    @staticmethod
    def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        if isinstance(tensor, Tensor) and len(tensor.data.shape) >= 2:
            fan = tensor.data.shape[1] if mode == 'fan_in' else tensor.data.shape[0]
            gain = math.sqrt(2.0 / (1 + a**2)) if nonlinearity == 'leaky_relu' else 1.0
            std = gain / math.sqrt(fan)
            tensor.data[:] = (np.random.randn(*tensor.data.shape) * std).astype(np.float32)
        return tensor



def _to_tensor(x):
    if isinstance(x, Tensor):
        return x
    if isinstance(x, np.ndarray):
        return Tensor(x, requires_grad=False)
    if hasattr(x, 'data'):
        return Tensor(x.data, requires_grad=False)
    return Tensor(np.asarray(x, dtype=np.float32), requires_grad=False)


def _data(x):
    if isinstance(x, Tensor):
        return x.data
    if hasattr(x, 'data'):
        if hasattr(x, 'detach'):  # PyTorch tensor
            return x.detach().cpu().numpy()
        return x.data
    return np.asarray(x, dtype=np.float32)


def _linear_fn(x, weight, bias=None):
    xd = _data(x)
    wd = _data(weight)
    out = xd @ wd.T
    if bias is not None:
        out = out + _data(bias)
    if isinstance(x, Tensor) or isinstance(weight, Tensor):
        return Tensor(out, requires_grad=(isinstance(x, Tensor) and x.requires_grad))
    return out


def _pad_to_pairs(pad):
    """Convert flat pad spec to pairs for np.pad."""
    pairs = []
    for i in range(len(pad) // 2):
        pairs.append((pad[2 * i], pad[2 * i + 1]))
    return tuple(pairs)


def _one_hot(t, num_classes=-1):
    d = _data(t).astype(int).flatten()
    if num_classes <= 0:
        num_classes = d.max() + 1
    result = np.eye(num_classes, dtype=np.float32)[d]
    return Tensor(result, requires_grad=False)


def softmax(tensor, dim=-1):
    return _softmax(_to_tensor(tensor), dim=dim)


def log_softmax(tensor, dim=-1):
    return _log_softmax(_to_tensor(tensor), dim=dim)


def cross_entropy(logits, targets, **kw):
    return _cross_entropy(_to_tensor(logits), _to_tensor(targets))


def mse_loss(a, b, **kw):
    return _mse_loss(_to_tensor(a), _to_tensor(b))


def is_tensor(obj):
    return isinstance(obj, Tensor)


def numel(tensor):
    if isinstance(tensor, Tensor):
        return tensor.data.size
    return np.asarray(tensor).size


def _walk_replace(obj, fn):
    """Walk a nested dict/list structure, applying ``fn`` to each leaf value."""
    if isinstance(obj, dict):
        return {k: _walk_replace(v, fn) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_walk_replace(v, fn) for v in obj]
    return fn(obj)


def _is_sou_file(f):
    """Check if a file path has a .soul extension."""
    if isinstance(f, str):
        return f.endswith(".soul")
    if hasattr(f, "name"):
        return f.name.endswith(".soul")
    return False


def _is_npz_file(f):
    """Check if a file path has a .npz extension."""
    if isinstance(f, str):
        return f.endswith(".npz")
    if hasattr(f, "name"):
        return f.name.endswith(".npz")
    return False


def save(obj, f, **kw):
    """Save an object to disk.

    - ``SloNet`` models → ``.soul`` format.
    - Dicts containing tensors/numpy arrays → pickle-based ``.pt`` / ``.npz``.
    - Everything else → JSON.
    """
    if isinstance(obj, _SoulNet):
        from domains.training.slonet import export_to_sou
        return export_to_sou(obj, f, include_weights=True)

    has_tensors = False
    try:
        def _check_tensor(v):
            nonlocal has_tensors
            if isinstance(v, (Tensor, _Tensor, np.ndarray)):
                has_tensors = True
        _walk_replace(obj, _check_tensor)
    except Exception:
        pass

    if has_tensors:
        import pickle
        # Convert SloNet Tensors → numpy arrays for serialization
        obj_clean = _walk_replace(obj, lambda v: v.data if isinstance(v, (Tensor, _Tensor)) else v)
        with open(f, 'wb') as fh:
            pickle.dump(obj_clean, fh, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        import json
        with open(f, 'w') as fh:
            json.dump(obj, fh)


def load(f, **kw):
    """Load an object previously saved by :func:`save`.

    Auto-detects format by file extension (``.soul`` → SloNet model,
    ``.pt`` / ``.npz`` → pickle with tensors, else → JSON).
    ``map_location`` and other keyword arguments are accepted (for API
    compatibility with ``torch.load``) but ignored — no GPU mapping is
    needed in the SloNet runtime.
    """
    import numpy as np

    if _is_sou_file(f):
        from domains.training.slonet import import_from_sou
        return import_from_sou(f)

    if _is_npz_file(f):
        return dict(np.load(f))

    # Try pickle first (.pt or generic binary)
    try:
        import pickle
        with open(f, 'rb') as fh:
            data = pickle.load(fh)
        # Convert numpy arrays back to SloNet Tensors
        return _walk_replace(data, lambda v: Tensor(v, requires_grad=False) if isinstance(v, np.ndarray) else v)
    except (pickle.UnpicklingError, EOFError, ModuleNotFoundError, AttributeError):
        pass

    # Fallback to JSON
    import json
    with open(f, 'r') as fh:
        return json.load(fh)


def multinomial(tensor, num_samples, replacement=False):
    return _multinomial(_to_tensor(tensor), num_samples)


def topk(tensor, k, dim=None):
    return _topk(_to_tensor(tensor), k)


def argmax(tensor, dim=None):
    return _argmax(_to_tensor(tensor), dim=dim)


def argmin(tensor, dim=None):
    return _argmin(_to_tensor(tensor), dim=dim)


def squeeze(tensor, dim=None):
    return _squeeze(_to_tensor(tensor), dim=dim)


def unsqueeze(tensor, dim):
    return _unsqueeze(_to_tensor(tensor), dim)


def transpose(tensor, dim0, dim1):
    return Tensor(np.swapaxes(_data(tensor), dim0, dim1), requires_grad=False)


def flatten(tensor, start_dim=0, end_dim=-1):
    return Tensor(_data(tensor).reshape(-1), requires_grad=False)


def cat(tensors, dim=0):
    return _cat([_to_tensor(t) for t in tensors], dim=dim)


def stack(tensors, dim=0):
    return _stack([_to_tensor(t) for t in tensors], dim=dim)


def sum(tensor, dim=None, keepdim=False):
    if dim is None:
        return Tensor(np.array(_data(tensor).sum(), dtype=np.float32), requires_grad=False)
    return Tensor(_data(tensor).sum(axis=dim, keepdims=keepdim), requires_grad=False)


def mean(tensor, dim=None, keepdim=False):
    if dim is None:
        return Tensor(np.array(_data(tensor).mean(), dtype=np.float32), requires_grad=False)
    return Tensor(_data(tensor).mean(axis=dim, keepdims=keepdim), requires_grad=False)


def max(tensor, dim=None):
    if dim is None:
        return Tensor(np.array(_data(tensor).max(), dtype=np.float32), requires_grad=False)
    values = Tensor(_data(tensor).max(axis=dim), requires_grad=False)
    indices = Tensor(_data(tensor).argmax(axis=dim).astype(np.int64), requires_grad=False)
    return values, indices


def min(tensor, dim=None):
    if dim is None:
        return Tensor(np.array(_data(tensor).min(), dtype=np.float32), requires_grad=False)
    values = Tensor(_data(tensor).min(axis=dim), requires_grad=False)
    indices = Tensor(_data(tensor).argmin(axis=dim).astype(np.int64), requires_grad=False)
    return values, indices


def cumsum(tensor, dim=0):
    return Tensor(np.cumsum(_data(tensor), axis=dim), requires_grad=False)


def sort(tensor, dim=-1, descending=False):
    if descending:
        idx = np.argsort(-_data(tensor), axis=dim)
    else:
        idx = np.argsort(_data(tensor), axis=dim)
    values = np.take_along_axis(_data(tensor), idx, axis=dim)
    return Tensor(values, requires_grad=False), Tensor(idx.astype(np.int64), requires_grad=False)


def bmm(a, b):
    """Batch matrix multiplication."""
    return Tensor(np.matmul(_data(a), _data(b)), requires_grad=False)


def matmul(a, b):
    return Tensor(np.matmul(_data(a), _data(b)), requires_grad=False)


def einsum(equation, *operands):
    return Tensor(np.einsum(equation, *[_data(o) for o in operands]), requires_grad=False)


def meshgrid(*tensors, indexing='xy'):
    arrays = [_data(t) for t in tensors]
    result = np.meshgrid(*arrays, indexing=indexing)
    return [Tensor(r, requires_grad=False) for r in result]


def logical_not(tensor):
    return Tensor((~_data(tensor).astype(bool)).astype(np.float32), requires_grad=False)


def logical_and(a, b):
    return Tensor((_data(a).astype(bool) & _data(b).astype(bool)).astype(np.float32), requires_grad=False)


def logical_or(a, b):
    return Tensor((_data(a).astype(bool) | _data(b).astype(bool)).astype(np.float32), requires_grad=False)


# — Tensor creation
def tensor(data, dtype=None, device=None, requires_grad=False):
    resolved = _resolve_dtype(dtype)
    arr = np.asarray(data, dtype=resolved)
    return Tensor(arr, requires_grad=requires_grad, dtype=resolved)


def _resolve_shape(size):
    """Unwrap (tuple,) passed as a single arg to *size."""
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        return tuple(size[0])
    return size


def _resolve_dtype(dtype):
    """Extract numpy dtype from compat dtype, real torch dtype, or pass through."""
    if dtype is None:
        return np.float32
    if hasattr(dtype, '_np'):
        return dtype._np
    tname = str(dtype)
    if tname.startswith('torch.'):
        name = tname.split('.')[-1]
        np_type = getattr(np, name, None)
        if np_type is not None:
            return np_type
    return dtype


def zeros(*size, dtype=None, device=None, requires_grad=False):
    shape = _resolve_shape(size)
    resolved = _resolve_dtype(dtype)
    return Tensor(np.zeros(shape, dtype=resolved), requires_grad=requires_grad, dtype=resolved)


def ones(*size, dtype=None, device=None, requires_grad=False):
    shape = _resolve_shape(size)
    resolved = _resolve_dtype(dtype)
    return Tensor(np.ones(shape, dtype=resolved), requires_grad=requires_grad, dtype=resolved)


def randn(*size, dtype=None, device=None, requires_grad=False):
    shape = _resolve_shape(size)
    resolved = _resolve_dtype(dtype)
    arr = np.random.randn(*shape).astype(resolved)
    return Tensor(arr, requires_grad=requires_grad, dtype=resolved)


def rand(*size, dtype=None, device=None, requires_grad=False):
    shape = _resolve_shape(size)
    resolved = _resolve_dtype(dtype)
    arr = np.random.rand(*shape).astype(resolved)
    return Tensor(arr, requires_grad=requires_grad, dtype=resolved)


def eye(n, m=None, dtype=None, device=None, requires_grad=False):
    return _eye(n, m)


def empty(*size, dtype=None, device=None, requires_grad=False):
    shape = _resolve_shape(size)
    resolved = _resolve_dtype(dtype)
    return Tensor(np.empty(shape, dtype=resolved), requires_grad=requires_grad, dtype=resolved)


def full(size, fill_value, dtype=None, device=None, requires_grad=False):
    resolved = _resolve_dtype(dtype)
    return Tensor(np.full(size, fill_value, dtype=resolved), requires_grad=requires_grad, dtype=resolved)


def arange(start, end=None, step=1, dtype=None, device=None, requires_grad=False):
    resolved = _resolve_dtype(dtype or np.int64)
    arr = np.arange(start, end, step).astype(resolved)
    return Tensor(arr, requires_grad=requires_grad, dtype=resolved)


def linspace(start, end, steps, dtype=None, device=None, requires_grad=False):
    resolved = _resolve_dtype(dtype or np.float32)
    arr = np.linspace(start, end, steps).astype(resolved)
    return Tensor(arr, requires_grad=requires_grad, dtype=resolved)


def from_numpy(ndarray, requires_grad=False):
    return Tensor(ndarray, requires_grad=requires_grad)


def zeros_like(tensor_inp, **kw):
    return Tensor(np.zeros_like(_data(tensor_inp), dtype=_resolve_dtype(kw.get('dtype'))), **kw)


def ones_like(tensor_inp, **kw):
    return Tensor(np.ones_like(_data(tensor_inp), dtype=_resolve_dtype(kw.get('dtype'))), **kw)


def randn_like(tensor_inp, **kw):
    return Tensor(np.random.randn(*_data(tensor_inp).shape).astype(_resolve_dtype(kw.get('dtype', np.float32))), **kw)


# — Tensor ops
def stack(tensors, dim=0):
    return _stack([_to_tensor(t) for t in tensors], dim=dim)


def cat(tensors, dim=0):
    return _cat([_to_tensor(t) for t in tensors], dim=dim)


def where(condition, a, b):
    return _where(_to_tensor(condition), _to_tensor(a), _to_tensor(b))


def gather(tensor_inp, dim, index):
    if isinstance(tensor_inp, Tensor):
        return tensor_inp.gather(dim, _to_tensor(index))
    return _Tensor(_data(tensor_inp), requires_grad=False).gather(dim, _to_tensor(index))


def nonzero(tensor_inp):
    d = _data(tensor_inp)
    idx = np.nonzero(d)
    n = len(idx[0])
    result = np.zeros((n, len(idx)), dtype=np.int64)
    for i in range(len(idx)):
        result[:, i] = idx[i]
    return Tensor(result, requires_grad=False)


def abs(tensor_inp):
    return Tensor(np.abs(_data(tensor_inp)), requires_grad=False)


def sqrt(tensor_inp):
    return Tensor(np.sqrt(np.maximum(_data(tensor_inp), 0)), requires_grad=False)


def clamp(tensor_inp, min_val=None, max_val=None):
    return Tensor(np.clip(_data(tensor_inp), min_val, max_val), requires_grad=False)


def sigmoid(tensor_inp):
    return _sigmoid(_to_tensor(tensor_inp))


def tanh(tensor_inp):
    return _tanh(_to_tensor(tensor_inp))


def relu(tensor_inp):
    return _relu(_to_tensor(tensor_inp))


def exp(tensor_inp):
    return Tensor(np.exp(_data(tensor_inp)), requires_grad=False)


# =============================================================================
# nn.functional — the ``F.*`` namespace
# =============================================================================


class _Functional:
    """``torch.nn.functional``."""
    relu = staticmethod(lambda x: _relu(_to_tensor(x)))
    sigmoid = staticmethod(lambda x: _sigmoid(_to_tensor(x)))
    tanh = staticmethod(lambda x: _tanh(_to_tensor(x)))
    gelu = staticmethod(lambda x: _gelu(_to_tensor(x)))
    silu = staticmethod(lambda x: _silu(_to_tensor(x)))
    softmax = staticmethod(lambda x, dim=-1: _softmax(_to_tensor(x), dim=dim))
    log_softmax = staticmethod(lambda x, dim=-1: _log_softmax(_to_tensor(x), dim=dim))
    cross_entropy = staticmethod(lambda logits, targets, **kw: _cross_entropy(_to_tensor(logits), _to_tensor(targets)))
    mse_loss = staticmethod(lambda a, b, **kw: _mse_loss(_to_tensor(a), _to_tensor(b)))
    kl_div = staticmethod(lambda a, b, reduction='batchmean': _kl_div_loss(_to_tensor(a), _to_tensor(b), reduction=reduction))
    normalize = staticmethod(lambda x, p=2, dim=1: _normalize(_to_tensor(x), p=p, dim=dim))
    pairwise_distance = staticmethod(lambda a, b: _pairwise_distance(_to_tensor(a), _to_tensor(b)))
    linear = staticmethod(lambda x, weight, bias=None: _linear_fn(x, weight, bias))
    one_hot = staticmethod(lambda tensor, num_classes=-1: _one_hot(_to_tensor(tensor), num_classes))

    @staticmethod
    def relu_(x):
        return _relu(_to_tensor(x))

    @staticmethod
    def dropout(x, p=0.5, training=True, inplace=False):
        if not training or p == 0:
            return x
        mask = np.random.binomial(1, 1 - p, _data(x).shape).astype(np.float32) / (1 - p)
        result = _data(x) * mask
        if isinstance(x, Tensor):
            return Tensor(result, requires_grad=x.requires_grad)
        return result

    @staticmethod
    def binary_cross_entropy_with_logits(logits, targets, **kw):
        d = _data(logits)
        t = _data(targets)
        loss = np.mean(np.maximum(d, 0) - d * t + np.log(1 + np.exp(-np.abs(d))))
        return Tensor(np.array(loss, dtype=np.float32), requires_grad=True)

    @staticmethod
    def pad(x, pad, mode='constant', value=0):
        d = _data(x)
        padded = np.pad(d, _pad_to_pairs(pad), mode=mode, constant_values=value)
        if isinstance(x, Tensor):
            return Tensor(padded, requires_grad=x.requires_grad)
        return padded

    @staticmethod
    def nll_loss(log_probs, targets, reduction='mean'):
        """Negative log-likelihood loss. log_probs: (N, C), targets: (N,)"""
        d = _data(log_probs)
        t = _data(targets).astype(int).flatten()
        n = d.shape[0]
        loss_arr = -d[np.arange(n, dtype=int), t % d.shape[1]]
        if reduction == 'mean':
            loss = Tensor(np.array(loss_arr.mean(), dtype=np.float32), requires_grad=True)
        elif reduction == 'sum':
            loss = Tensor(np.array(loss_arr.sum(), dtype=np.float32), requires_grad=True)
        else:
            loss = Tensor(loss_arr.astype(np.float32), requires_grad=True)
        return loss

    @staticmethod
    def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, scale=None):
        """Simple scaled dot-product attention fallback."""
        qd = _data(query)
        kd = _data(key)
        vd = _data(value)
        d_k = qd.shape[-1]
        scale = scale or (1.0 / math.sqrt(d_k))
        scores = np.matmul(qd, kd.transpose(0, 1, 3, 2)) * scale
        if attn_mask is not None:
            mask_data = _data(attn_mask).astype(bool)
            scores = np.where(mask_data, -1e9, scores)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True)
        out = np.matmul(weights, vd)
        if isinstance(query, Tensor):
            return Tensor(out, requires_grad=False)
        return out


# Wire up _NNModule attributes defined after class
_NNModule.functional = _Functional()
_NNModule.init = _Init()

# =============================================================================
# optim — Optimizers
# =============================================================================


class _Optim:
    class AdamW(_SoulAdam):
        """Drop‑in for ``torch.optim.AdamW``."""
        def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
            super().__init__(lr=lr, b1=betas[0], b2=betas[1], eps=eps)
            self._params = params
            self.weight_decay = weight_decay
            self.defaults = {'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay}

        def step(self, closure=None):
            plist = list(self._params) if hasattr(self._params, '__iter__') else [self._params]
            params = []
            for p in plist:
                if hasattr(p, 'parameters'):
                    params.extend(p.parameters())
                else:
                    params.append(p)
            super().step(params)

        def zero_grad(self):
            plist = list(self._params) if hasattr(self._params, '__iter__') else [self._params]
            for p in plist:
                if hasattr(p, 'parameters'):
                    for sp in p.parameters():
                        sp.grad = None
                else:
                    p.grad = None

    class SGD(_SoulSGD):
        """Drop‑in for ``torch.optim.SGD``."""
        def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
            super().__init__(lr=lr, momentum=momentum)
            self._params = params
            self.defaults = {'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}

        def step(self, closure=None):
            plist = list(self._params) if hasattr(self._params, '__iter__') else [self._params]
            params = []
            for p in plist:
                if hasattr(p, 'parameters'):
                    params.extend(p.parameters())
                else:
                    params.append(p)
            super().step(params)

        def zero_grad(self):
            plist = list(self._params) if hasattr(self._params, '__iter__') else [self._params]
            for p in plist:
                if hasattr(p, 'parameters'):
                    for sp in p.parameters():
                        sp.grad = None
                else:
                    p.grad = None

    class Optimizer:
        """Base class for optimizers."""
        def __init__(self):
            self.defaults = {}
        def zero_grad(self): pass
        def step(self, closure=None): pass
        def state_dict(self): return {}
        def load_state_dict(self, d): pass


# =============================================================================
# optim.lr_scheduler
# =============================================================================


class _LRScheduler:
    """``torch.optim.lr_scheduler`` namespace — re‑exports soul schedulers."""

    class _LRScheduler(_SoulLRScheduler):
        pass

    class StepLR(_SoulStepLR):
        pass

    class CosineAnnealingLR(_SoulCosineAnnealing):
        pass

    class ReduceLROnPlateau(_SoulReduceLROnPlateau):
        pass

    class ConstantLR(_SoulConstantLR):
        pass

    class OneCycleLR(_SoulOneCycleLR):
        pass

    class CyclicLR(_SoulCyclicLR):
        pass

    class WarmupCosineScheduler(_WarmupCosine):
        pass

    class PolynomialDecayScheduler(_Polynomial):
        pass

    class LinearWarmupScheduler(_LinearWarmup):
        pass

    @staticmethod
    def CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=2, eta_min=0, last_epoch=-1):
        return _SoulCosineAnnealing(optimizer, T_max=T_0, eta_min=eta_min, last_epoch=last_epoch)


# =============================================================================
# utils.data — Dataset / DataLoader
# =============================================================================


class _Utils:
    class data:
        class Dataset(_SoulDataset):
            pass
        class DataLoader(_SoulDataLoader):
            pass

    class checkpoint:
        @staticmethod
        def checkpoint(function, *args, **kwargs):
            return function(*args)


class _Onnx:
    @staticmethod
    def export(model, args, f, **kwargs):
        pass


class _Jit:
    def script(self, fn):
        return fn

    def trace(self, fn, *args, **kwargs):
        return fn


class _Quantization:
    @staticmethod
    def quantize_dynamic(model, *args, **kwargs):
        return model

    @staticmethod
    def get_default_qconfig(backend='fbgemm'):
        return type('QConfig', (), {'__repr__': lambda s: 'default_qconfig'})()

    @staticmethod
    def prepare(model, *args, **kwargs):
        return model

    @staticmethod
    def convert(model, *args, **kwargs):
        return model


class _Version:
    __version__ = "0.0.0+slonet"


# =============================================================================
# distributed / multiprocessing stubs
# =============================================================================


class _Distributed:
    """``torch.distributed`` — no‑op stubs."""
    @staticmethod
    def is_available():
        return False
    @staticmethod
    def init_process_group(*a, **kw):
        pass
    @staticmethod
    def destroy_process_group(*a, **kw):
        pass
    @staticmethod
    def get_rank():
        return 0
    @staticmethod
    def get_world_size():
        return 1
    @staticmethod
    def all_reduce(*a, **kw):
        pass
    @staticmethod
    def all_gather(*a, **kw):
        return []
    @staticmethod
    def broadcast(*a, **kw):
        pass
    @staticmethod
    def barrier(*a, **kw):
        pass
    ReduceOp = type('ReduceOp', (), {'SUM': 'SUM'})()


class _MultiProcessing:
    """``torch.multiprocessing`` — no‑op stubs."""
    class Process:
        def __init__(self, **kw): pass
        def start(self): pass
        def join(self): pass
    @staticmethod
    def spawn(fn, args=(), nprocs=1, **kw):
        return []


# =============================================================================
# cuda / backends / device helpers
# =============================================================================


class _Cuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        return 0

    @staticmethod
    def set_device(device):
        pass

    @staticmethod
    def current_device():
        return -1

    @staticmethod
    def synchronize(device=None):
        pass

    @staticmethod
    def empty_cache():
        pass

    @staticmethod
    def memory_allocated(device=None):
        return 0

    @staticmethod
    def memory_reserved(device=None):
        return 0

    @staticmethod
    def max_memory_allocated(device=None):
        return 0

    @staticmethod
    def reset_peak_memory_stats(device=None):
        pass

    @staticmethod
    def get_device_name(device=None):
        return "cpu"

    @staticmethod
    def get_device_properties(device):
        return type('DeviceProps', (), {'total_memory': 0})()

    @staticmethod
    def get_device_capability(device=None):
        return (0, 0)

    @staticmethod
    def get_rng_state_all():
        return [np.random.get_state()]

    CUDAGraph = type('CUDAGraph', (), {'__init__': lambda self: None, 'replay': lambda self: None})

    set_sync_debug_mode = staticmethod(lambda mode: None)
    sync_debug_mode = type('sync_debug_mode', (), {'OFF': 0})()

    class amp:
        class GradScaler:
            def __init__(self, **kw): pass
            def scale(self, loss): return loss
            def step(self, optimizer, **kw):
                if hasattr(optimizer, 'step'):
                    optimizer.step()
            def update(self): pass

        def autocast(device_type=None, dtype=None, enabled=True):
            return _no_grad()


class _Backends:
    class mps:
        @staticmethod
        def is_available():
            return False
        @staticmethod
        def synchronize():
            pass

    class cudnn:
        enabled = False
        deterministic = False
        benchmark = False
        allow_tf32 = True

    class mkldnn:
        enabled = False

    class cuda:
        class matmul:
            allow_tf32 = True

    class opt_einsum:
        enabled = False


# =============================================================================
# Build the torch namespace
# =============================================================================


class _TorchNamespace:
    """Namespace that acts like the ``torch`` module."""
    Tensor = Tensor
    no_grad = no_grad
    nn = _NNModule()
    nn_functional = _Functional()
    F = _Functional()
    optim = _Optim()
    optim_lr_scheduler = _LRScheduler()
    utils = _Utils()
    utils = _Utils()
    cuda = _Cuda()
    backends = _Backends()
    distributed = _Distributed()
    multiprocessing = _MultiProcessing()
    onnx = _Onnx()
    jit = _Jit()
    quantization = _Quantization()
    version = _Version()
    nn_init = _Init()
    init = _Init()

    # Float types
    float32 = np.float32
    float64 = np.float64
    float16 = np.float16
    float = np.float32
    double = np.float64
    half = np.float16
    # Int types
    int8 = np.int8
    int16 = np.int16
    int32 = np.int32
    int64 = np.int64
    int = np.int64
    long = np.int64
    short = np.int16
    # UInt types
    uint8 = np.uint8
    # Bool
    bool = np.bool_
    bfloat16 = np.float32  # approximated
    # Quantization dtypes (stubs for compat)
    qint8 = type('qint8', (), {'__repr__': lambda s: 'qint8'})()
    qint32 = type('qint32', (), {'__repr__': lambda s: 'qint32'})()
    quint8 = type('quint8', (), {'__repr__': lambda s: 'quint8'})()
    quint4x2 = type('quint4x2', (), {'__repr__': lambda s: 'quint4x2'})()

    # Device
    device = type('device', (), {'__init__': lambda self, t: setattr(self, 'type', t.split(':')[0] if ':' in t else t)})
    # dtype type alias
    dtype = type('dtype', (), {'__repr__': lambda self: 'torch.dtype'})()

    # Top-level functions (bound to namespace)
    tensor = staticmethod(tensor)
    zeros = staticmethod(zeros)
    ones = staticmethod(ones)
    randn = staticmethod(randn)
    rand = staticmethod(rand)
    eye = staticmethod(eye)
    empty = staticmethod(empty)
    full = staticmethod(full)
    arange = staticmethod(arange)
    linspace = staticmethod(linspace)
    from_numpy = staticmethod(from_numpy)
    stack = staticmethod(stack)
    cat = staticmethod(cat)
    where = staticmethod(where)
    gather = staticmethod(gather)
    nonzero = staticmethod(nonzero)
    abs = staticmethod(abs)
    sqrt = staticmethod(sqrt)
    clamp = staticmethod(clamp)
    sigmoid = staticmethod(sigmoid)
    tanh = staticmethod(tanh)
    relu = staticmethod(relu)
    exp = staticmethod(exp)
    softmax = staticmethod(softmax)
    log_softmax = staticmethod(log_softmax)
    cross_entropy = staticmethod(cross_entropy)
    mse_loss = staticmethod(mse_loss)
    is_tensor = staticmethod(is_tensor)
    numel = staticmethod(numel)
    save = staticmethod(save)
    load = staticmethod(load)
    multinomial = staticmethod(multinomial)
    topk = staticmethod(topk)
    argmax = staticmethod(argmax)
    argmin = staticmethod(argmin)
    squeeze = staticmethod(squeeze)
    unsqueeze = staticmethod(unsqueeze)
    transpose = staticmethod(transpose)
    flatten = staticmethod(flatten)
    sum = staticmethod(sum)
    mean = staticmethod(mean)
    max = staticmethod(max)
    min = staticmethod(min)
    cumsum = staticmethod(cumsum)
    sort = staticmethod(sort)
    bmm = staticmethod(bmm)
    matmul = staticmethod(matmul)
    einsum = staticmethod(einsum)
    meshgrid = staticmethod(meshgrid)

    # Missing tensor creation/manipulation
    @staticmethod
    def tril(tensor, diagonal=0):
        d = _data(tensor)
        return Tensor(np.tril(d, k=diagonal), requires_grad=False)

    @staticmethod
    def triu(tensor, diagonal=0):
        d = _data(tensor)
        return Tensor(np.triu(d, k=diagonal), requires_grad=False)

    @staticmethod
    def rsqrt(tensor):
        return Tensor(1.0 / np.sqrt(np.maximum(_data(tensor), 1e-10)), requires_grad=False)

    @staticmethod
    def log(tensor):
        return Tensor(np.log(np.maximum(_data(tensor), 1e-12)).astype(np.float32), requires_grad=False)

    @staticmethod
    def quantile(tensor, q, dim=None, keepdim=False):
        d = _data(tensor)
        if dim is not None:
            return Tensor(np.quantile(d, q, axis=dim, keepdims=keepdim), requires_grad=False)
        return Tensor(np.array(np.quantile(d, q), dtype=np.float32), requires_grad=False)

    @staticmethod
    def unique(tensor, sorted=True, return_inverse=False, return_counts=False, dim=None):
        d = _data(tensor)
        if dim is not None:
            uniq, idx, inv, cnt = np.unique(d, return_index=True, return_inverse=True, return_counts=True, axis=dim)
        else:
            uniq, inv, cnt = np.unique(d, return_inverse=True, return_counts=True)
            idx = np.arange(len(uniq))
        result = [Tensor(uniq, requires_grad=False)]
        if return_inverse:
            result.append(Tensor(inv.astype(np.int64), requires_grad=False))
        if return_counts:
            result.append(Tensor(cnt.astype(np.int64), requires_grad=False))
        return tuple(result) if len(result) > 1 else result[0]

    @staticmethod
    def bincount(tensor, weights=None, minlength=0):
        d = _data(tensor).astype(int).flatten()
        if weights is not None:
            w = _data(weights).flatten()
        result = np.bincount(d, weights=w if weights is not None else None, minlength=minlength).astype(np.float32)
        return Tensor(result, requires_grad=False)

    @staticmethod
    def round(tensor, decimals=0):
        return Tensor(np.round(_data(tensor), decimals=decimals), requires_grad=False)

    @staticmethod
    def channels_last():
        return type('channels_last', (), {'__repr__': lambda s: 'channels_last'})()

    @staticmethod
    def set_num_threads(n):
        pass

    # torch.compile stub
    @staticmethod
    def compile(model, *args, **kwargs):
        return model

    # Convenience
    @staticmethod
    def set_grad_enabled(mode=True):
        return _no_grad() if not mode else _no_grad().__enter__()

    @staticmethod
    def inference_mode(mode=True):
        return _no_grad()

    @staticmethod
    def enable_grad():
        class _EG:
            def __enter__(self): pass
            def __exit__(self, *a): pass
        return _EG()

    @staticmethod
    def no_grad():
        return _no_grad()

    @staticmethod
    def manual_seed(seed):
        np.random.seed(seed)

    @staticmethod
    def seed():
        np.random.seed()
        return 0

    @staticmethod
    def get_rng_state():
        return np.random.get_state()

    @staticmethod
    def set_rng_state(state):
        np.random.set_state(state)

    @staticmethod
    def zeros_like(tensor, **kw):
        return Tensor(np.zeros_like(_data(tensor)), **kw)

    @staticmethod
    def ones_like(tensor, **kw):
        return Tensor(np.ones_like(_data(tensor)), **kw)

    @staticmethod
    def randn_like(tensor, **kw):
        return Tensor(np.random.randn(*_data(tensor).shape).astype(np.float32), **kw)

    @staticmethod
    def full_like(tensor, fill_value, **kw):
        return Tensor(np.full_like(_data(tensor), fill_value), **kw)

    @staticmethod
    def empty_like(tensor, **kw):
        return Tensor(np.empty_like(_data(tensor)), **kw)

    @staticmethod
    def norm(tensor, p=2, dim=None, keepdim=False):
        if dim is not None:
            return Tensor(np.linalg.norm(_data(tensor), ord=p, axis=dim, keepdims=keepdim), requires_grad=False)
        return Tensor(np.array(np.linalg.norm(_data(tensor), ord=p)), requires_grad=False)

    @staticmethod
    def dot(a, b):
        return Tensor(np.array(np.dot(_data(a), _data(b)), dtype=np.float32), requires_grad=False)

    @staticmethod
    def vdot(a, b):
        return Tensor(np.array(np.vdot(_data(a), _data(b)), dtype=np.float32), requires_grad=False)

    @staticmethod
    def equal(a, b):
        return np.array_equal(_data(a), _data(b))

    @staticmethod
    def allclose(a, b, rtol=1e-5, atol=1e-8):
        return np.allclose(_data(a), _data(b), rtol=rtol, atol=atol)

    @staticmethod
    def result_type(a, b):
        return np.result_type(_data(a).dtype, _data(b).dtype)

    @staticmethod
    def randint(low, high, size, **kw):
        return _randint(low, high, size)

    @staticmethod
    def randperm(n, **kw):
        return Tensor(np.random.permutation(n).astype(np.int64), requires_grad=False)

    @staticmethod
    def isinf(tensor):
        return Tensor(np.isinf(_data(tensor)).astype(np.float32), requires_grad=False)

    @staticmethod
    def isnan(tensor):
        return Tensor(np.isnan(_data(tensor)).astype(np.float32), requires_grad=False)

    @staticmethod
    def isfinite(tensor):
        return _isfinite(_to_tensor(tensor))

    @staticmethod
    def logical_not(tensor):
        return Tensor((~_data(tensor).astype(bool)).astype(np.float32), requires_grad=False)

    @staticmethod
    def logical_and(a, b):
        return Tensor((_data(a).astype(bool) & _data(b).astype(bool)).astype(np.float32), requires_grad=False)

    @staticmethod
    def logical_or(a, b):
        return Tensor((_data(a).astype(bool) | _data(b).astype(bool)).astype(np.float32), requires_grad=False)

    @staticmethod
    def meshgrid(*tensors, indexing='xy'):
        arrays = [_data(t) for t in tensors]
        result = np.meshgrid(*arrays, indexing=indexing)
        return [Tensor(r, requires_grad=False) for r in result]

    @staticmethod
    def gather(tensor_inp, dim, index):
        return gather(tensor_inp, dim, index)

    @staticmethod
    def nonzero(tensor_inp):
        return nonzero(tensor_inp)


# =============================================================================
# Singleton torch instance
# =============================================================================

torch = _TorchNamespace()

# Module-level exports for convenient importing:
#   from domains.training.slonet_compat import torch, nn, F, optim, utils
nn = torch.nn
F = torch.F
optim = torch.optim
utils = torch.utils
