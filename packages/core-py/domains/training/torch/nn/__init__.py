"""
torch.nn — numpy-backed neural network module.
"""

import numpy as np
from . import functional
from . import utils as nn_utils


def _to_tensor(x):
    """Convert nested structures to Tensors."""
    import sys
    T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
    if T is None:
        return x
    if isinstance(x, T.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        return type(x)(_to_tensor(v) for v in x)
    if isinstance(x, dict):
        return {k: _to_tensor(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return T.Tensor(x)
    return x


class Module:
    def __init__(self):
        self._parameters = {}
        self._buffers = {}
        self.training = True

    def __call__(self, *args, **kwargs):
        import sys
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        # Convert numpy inputs to Tensor for autograd
        args = tuple(T.Tensor(a) if isinstance(a, np.ndarray) else a for a in args)
        kwargs = {k: T.Tensor(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
        result = self.forward(*args, **kwargs)
        return _to_tensor(result)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _collect_params(self, obj, result, recurse, T):
        if isinstance(obj, Module):
            if recurse:
                result.extend(obj.parameters(recurse))
        elif T is not None and isinstance(obj, T.Tensor) and obj.requires_grad:
            result.append(obj)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._collect_params(item, result, recurse, T)

    def parameters(self, recurse=True):
        import sys
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        result = []
        for v in self.__dict__.values():
            self._collect_params(v, result, recurse, T)
        for p in self._parameters.values():
            if p is not None: result.append(p)
        return result

    def named_parameters(self, prefix="", recurse=True):
        import sys
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        for k, v in self._parameters.items():
            if v is not None: yield prefix + ("." if prefix else "") + k, v
        if recurse:
            for name, m in self.named_children():
                if m is not None:
                    yield from m.named_parameters(prefix + ("." if prefix else "") + name, recurse)

    def children(self):
        for v in self.__dict__.values():
            if isinstance(v, Module) and not isinstance(v, (ModuleList, ModuleDict)): yield v

    def named_children(self):
        for k, v in self.__dict__.items():
            if isinstance(v, Module) and not isinstance(v, (ModuleList, ModuleDict)): yield k, v

    def modules(self):
        yield self
        for m in self.children(): yield from m.modules()

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        result = {}
        for k, v in self._parameters.items():
            if v is not None: result[prefix + k] = v
        for k, v in self._buffers.items(): result[prefix + k] = v
        for k, m in self.named_children():
            sd = m.state_dict(prefix=prefix + k + ".", keep_vars=keep_vars); result.update(sd)
        return result

    def load_state_dict(self, state_dict, strict=True):
        import sys
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        self._parameters.update({k: T.Tensor(v.data) if T is not None and not isinstance(v, T.Tensor) else v
                                for k, v in state_dict.items() if "." not in k})

    def train(self): self.training = True; return self
    def eval(self): self.training = False; return self
    def to(self, *args, **kwargs): return self
    def cpu(self): return self
    def cuda(self): return self
    def half(self): return self
    def float(self): return self
    def bfloat16(self): return self
    def zero_grad(self): pass
    def register_buffer(self, name, tensor): self._buffers[name] = tensor
    def register_parameter(self, name, param): self._parameters[name] = param
    def register_forward_hook(self, hook): pass
    def apply(self, fn):
        fn(self)
        for m in self.children(): m.apply(fn)
        return self


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__(); self.layers = list(layers)
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x


class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__(); self.layers = modules or []
    def __getitem__(self, i): return self.layers[i]
    def __iter__(self): return iter(self.layers)


class ModuleDict(Module):
    def __init__(self, mapping=None):
        super().__init__(); self.map = mapping or {}


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        import math
        import sys
        super().__init__()
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        scale = math.sqrt(1.0 / in_features)
        self.weight = T.Tensor(np.random.randn(out_features, in_features).astype(np.float32) * scale, requires_grad=True) if T else np.random.randn(out_features, in_features).astype(np.float32)
        self.bias = T.Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True) if T and bias else None
        self.in_features = in_features; self.out_features = out_features

    def forward(self, x):
        import sys
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        if not isinstance(x, T.Tensor) and hasattr(x, 'data'):
            x = x.data if not isinstance(x, T.Tensor) else x
        if not isinstance(x, T.Tensor):
            x = T.Tensor(x)
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        import math
        import sys
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        kh, kw = kernel_size
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        self.weight = T.Tensor(np.random.randn(out_channels, in_channels//groups, kh, kw).astype(np.float32) * math.sqrt(1.0/(in_channels*np.prod(kernel_size)))) if T else np.random.randn(out_channels, in_channels//groups, kh, kw).astype(np.float32)
        self.bias = T.Tensor(np.zeros(out_channels, dtype=np.float32)) if T and bias else None
        self.stride = stride; self.padding = padding; self.dilation = dilation; self.groups = groups

    def forward(self, x):
        from scipy.ndimage import convolve
        import sys
        d = x.data if hasattr(x, 'data') else x
        if d.ndim == 2: d = d[:, np.newaxis, np.newaxis, :]
        N, C, H, W = d.shape
        kh, kw = self.weight.data.shape[2:] if hasattr(self.weight, 'data') else self.weight.shape[2:]
        if self.padding > 0:
            d = np.pad(d, [(0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)], mode='constant')
        H2 = (H - kh)//self.stride + 1; W2 = (W - kw)//self.stride + 1
        wd = self.weight.data if hasattr(self.weight, 'data') else self.weight
        bd = self.bias.data if self.bias is not None and hasattr(self.bias, 'data') else None
        result = np.zeros((N, wd.shape[0], H2, W2), dtype=np.float32)
        for n in range(N):
            for oc in range(wd.shape[0]):
                for ic in range(min(C, wd.shape[1])):
                    conv = convolve(d[n, ic], wd[oc, ic], mode='constant')
                    if self.dilation > 1:
                        conv = conv[::self.dilation, ::self.dilation]
                    result[n, oc] += conv[:H2, :W2]
        if bd is not None: result += bd.reshape(1, -1, 1, 1)
        return result


class ConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        import sys
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        self.weight = T.Tensor(np.random.randn(in_channels, out_channels, *kernel_size).astype(np.float32) * 0.001) if T else np.random.randn(in_channels, out_channels, *kernel_size).astype(np.float32)
        self.bias = T.Tensor(np.zeros(out_channels, dtype=np.float32)) if T and bias else None
        self.stride = stride; self.padding = padding; self.output_padding = output_padding

    def forward(self, x):
        import sys
        d = x.data if hasattr(x, 'data') else x
        N, C2, H, W = d.shape
        wd = self.weight.data if hasattr(self.weight, 'data') else self.weight
        bd = self.bias.data if self.bias is not None and hasattr(self.bias, 'data') else None
        KH, KW = wd.shape[2:]
        OH = (H-1)*self.stride - 2*self.padding + KH + self.output_padding
        OW = (W-1)*self.stride - 2*self.padding + KW + self.output_padding
        result = np.zeros((N, wd.shape[1], OH, OW), dtype=np.float32)
        for n in range(N):
            for ic in range(C2):
                for oc in range(wd.shape[1]):
                    ws = wd[ic, oc, :, :]
                    padded = np.zeros((H*self.stride+KH-1, W*self.stride+KW-1), dtype=np.float32)
                    for i in range(H):
                        for j in range(W):
                            padded[i*self.stride:i*self.stride+KH, j*self.stride:j*self.stride+KW] += d[n, ic, i, j] * ws
                    result[n, oc] += padded[self.padding:self.padding+OH, self.padding:self.padding+OW]
        if bd is not None: result += bd.reshape(1, -1, 1, 1)
        return result


class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        import math
        import sys
        super().__init__()
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        self.weight = T.Tensor(np.random.randn(out_channels, in_channels//groups, kernel_size).astype(np.float32) * math.sqrt(1.0/(in_channels*kernel_size))) if T else np.random.randn(out_channels, in_channels//groups, kernel_size).astype(np.float32)
        self.bias = T.Tensor(np.zeros(out_channels, dtype=np.float32)) if T and bias else None
        self.stride = stride; self.padding = padding; self.dilation = dilation; self.groups = groups

    def forward(self, x):
        from scipy.ndimage import convolve1d
        import sys
        d = x.data if hasattr(x, 'data') else x
        if d.ndim == 2: d = np.expand_dims(d, 1)
        N, C, L = d.shape; W = self.weight.data.shape[2:] if hasattr(self.weight, 'data') else self.weight.shape[2:]
        wd = self.weight.data if hasattr(self.weight, 'data') else self.weight
        out_len = (L + 2*self.padding - self.dilation*(W[0]-1) - 1)//self.stride + 1
        result = np.zeros((N, wd.shape[0], out_len), dtype=np.float32)
        for n in range(N):
            for oc in range(wd.shape[0]):
                for ic in range(min(C, wd.shape[1])):
                    result[n, oc] += convolve1d(d[n, ic], wd[oc, ic], mode='constant')[::self.stride][:out_len]
        if self.bias is not None:
            bd = self.bias.data if hasattr(self.bias, 'data') else self.bias
            result += bd.reshape(1, -1, 1)
        return result


def _embedding_lookup(weight, indices):
    """Embedding lookup with autograd support.
    weight: Tensor (vocab, dim)
    indices: Tensor (...,) of integer indices
    Returns: Tensor (..., dim)
    """
    import sys
    T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
    if T is None:
        T = type(weight)
    w_req = isinstance(weight, T.Tensor) and weight.requires_grad
    idx_np = indices.data.astype(np.int64) if isinstance(indices, T.Tensor) else np.asarray(indices, dtype=np.int64)
    idx_np = np.clip(idx_np, 0, weight.shape[0] - 1)
    result_data = weight.data[idx_np]
    out = T.Tensor(result_data, requires_grad=w_req, _children=(weight,) if w_req else ())
    _vocab = weight.shape[0]; _dim = weight.shape[1]; _idx = idx_np
    def bk(g):
        if w_req:
            if weight.grad is None:
                weight.grad = T.Tensor(np.zeros((_vocab, _dim), dtype=np.float32))
            g_np = g.data if isinstance(g, T.Tensor) else np.asarray(g)
            # Accumulate gradients at each index position
            flat_idx = _idx.ravel()
            for i in range(_vocab):
                mask = (flat_idx == i)
                if mask.any():
                    grad_sum = g_np.reshape(-1, _dim)[mask].sum(axis=0)
                    weight.grad.data[i] += grad_sum
    out._backward_fn = bk if w_req else None
    return out


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None):
        import sys
        super().__init__()
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        self.weight = T.Tensor(np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.1, requires_grad=True) if T else np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
        self.num_embeddings = num_embeddings; self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

    def forward(self, x):
        import sys
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        if not isinstance(x, T.Tensor) and hasattr(x, 'data'):
            x = x.data if not isinstance(x, T.Tensor) else x
        if not isinstance(x, T.Tensor):
            x = T.Tensor(x)
        return _embedding_lookup(self.weight, x.long())


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        import sys
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps; self.elementwise_affine = elementwise_affine
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        if elementwise_affine:
            self.weight = T.Tensor(np.ones(shape, dtype=np.float32), requires_grad=True) if T else None
            self.bias = T.Tensor(np.zeros(shape, dtype=np.float32), requires_grad=True) if T else None
        else:
            self.weight = None; self.bias = None

    def forward(self, x):
        import sys
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        if not isinstance(x, T.Tensor) and hasattr(x, 'data'):
            x = x.data if not isinstance(x, T.Tensor) else x
        if not isinstance(x, T.Tensor):
            x = T.Tensor(x)
        axis = tuple(range(-len(self.normalized_shape if isinstance(self.normalized_shape, tuple) else (self.normalized_shape,)), 0))
        mean = x.mean(dim=axis, keepdim=True)
        centered = x - mean
        var = (centered ** 2).mean(dim=axis, keepdim=True)
        out = centered / (var + self.eps).sqrt()
        if self.elementwise_affine and self.weight is not None:
            w = self.weight; b = self.bias
            # Reshape to broadcast over non-last dims
            sh = [1] * len(out.shape)
            if isinstance(self.normalized_shape, int):
                sh[-1] = int(self.normalized_shape)
            else:
                for i, s in enumerate(self.normalized_shape):
                    sh[i - len(self.normalized_shape)] = int(s)
            w_reshaped = w.reshape(*sh)
            b_reshaped = b.reshape(*sh)
            out = out * w_reshaped + b_reshaped
        return out


class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        import sys
        super().__init__()
        self.num_features = num_features; self.eps = eps
        self.momentum = momentum; self.affine = affine; self.track_running_stats = track_running_stats
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        if affine:
            self.weight = T.Tensor(np.ones(num_features, dtype=np.float32)) if T else None
            self.bias = T.Tensor(np.zeros(num_features, dtype=np.float32)) if T else None
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def forward(self, x):
        d = x.data if hasattr(x, 'data') else x
        if d.ndim == 3: d = d.reshape(-1, self.num_features)
        mean = d.mean(axis=0); var = d.var(axis=0)
        out = (d - mean) / np.sqrt(var + self.eps)
        if self.affine:
            wd = self.weight.data if hasattr(self.weight, 'data') else self.weight
            bd = self.bias.data if hasattr(self.bias, 'data') else self.bias
            out = out * wd + bd
        return out.astype(np.float32)


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        import sys
        super().__init__()
        self.num_features = num_features; self.eps = eps
        self.momentum = momentum; self.affine = affine
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        if affine:
            self.weight = T.Tensor(np.ones(num_features, dtype=np.float32)) if T else None
            self.bias = T.Tensor(np.zeros(num_features, dtype=np.float32)) if T else None

    def forward(self, x):
        d = x.data if hasattr(x, 'data') else x
        N, C, H, W = d.shape
        d_flat = d.transpose(1, 0).reshape(C, -1)
        mean = d_flat.mean(axis=1, keepdims=True); var = d_flat.var(axis=1, keepdims=True)
        out = (d - mean.reshape(1, C, 1, 1)) / np.sqrt(var.reshape(1, C, 1, 1) + self.eps)
        if self.affine:
            wd = self.weight.data if hasattr(self.weight, 'data') else self.weight
            bd = self.bias.data if hasattr(self.bias, 'data') else self.bias
            out = out * wd.reshape(1, C, 1, 1) + bd.reshape(1, C, 1, 1)
        return out.astype(np.float32)


class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__(); self.p = p; self.inplace = inplace
    def forward(self, x):
        import sys
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        if not isinstance(x, T.Tensor) and hasattr(x, 'data'):
            x = x.data if not isinstance(x, T.Tensor) else x
        if not isinstance(x, T.Tensor):
            x = T.Tensor(x)
        if self.training and self.p > 0:
            mask = T.Tensor(np.random.rand(*x.shape) > self.p, dtype=np.float32)
            return x * mask / (1.0 - self.p)
        return x


class Dropout2d(Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__(); self.p = p; self.inplace = inplace
    def forward(self, x):
        import sys
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        if not isinstance(x, T.Tensor) and hasattr(x, 'data'):
            x = x.data if not isinstance(x, T.Tensor) else x
        if not isinstance(x, T.Tensor):
            x = T.Tensor(x)
        if self.training and self.p > 0:
            mask = T.Tensor(np.random.rand(1, x.shape[1], x.shape[2], 1) > self.p, dtype=np.float32)
            return x * mask / (1.0 - self.p)
        return x


class GELU(Module):
    def forward(self, x):
        import sys
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        if not isinstance(x, T.Tensor): x = T.Tensor(x)
        return T.gelu(x)


class ReLU(Module):
    def forward(self, x):
        import sys
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        if not isinstance(x, T.Tensor): x = T.Tensor(x)
        return T.relu(x)


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__(); self.negative_slope = negative_slope
    def forward(self, x):
        import sys
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        if not isinstance(x, T.Tensor): x = T.Tensor(x)
        return T.leaky_relu(x, self.negative_slope)


class Sigmoid(Module):
    def forward(self, x):
        import sys
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        if not isinstance(x, T.Tensor): x = T.Tensor(x)
        return T.sigmoid(x)


class Tanh(Module):
    def forward(self, x):
        import sys
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        if not isinstance(x, T.Tensor): x = T.Tensor(x)
        return T.tanh(x)


class Softmax(Module):
    def __init__(self, dim=-1): super().__init__(); self.dim = dim
    def forward(self, x):
        import sys
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        if not isinstance(x, T.Tensor): x = T.Tensor(x)
        return T.softmax(x, dim=self.dim)


class LogSoftmax(Module):
    def __init__(self, dim=-1): super().__init__(); self.dim = dim
    def forward(self, x):
        import sys
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        if not isinstance(x, T.Tensor): x = T.Tensor(x)
        return T.log(T.softmax(x, dim=self.dim))


class Flatten(Module):
    def __init__(self, start_dim=0, end_dim=-1):
        super().__init__(); self.start_dim = start_dim; self.end_dim = end_dim
    def forward(self, x):
        d = x.data if hasattr(x, 'data') else x
        sh = list(d.shape)
        ns = (*sh[:self.start_dim], np.prod(sh[self.start_dim:self.end_dim+1]), *sh[self.end_dim+1:])
        return d.reshape(ns)


class Unflatten(Module):
    def __init__(self, dim, unflattened_size):
        super().__init__(); self.dim = dim; self.size = unflattened_size
    def forward(self, x):
        d = x.data if hasattr(x, 'data') else x
        sh = list(d.shape[:self.dim]) + list(self.size) + list(d.shape[self.dim+1:])
        return d.reshape(sh)


class Identity(Module):
    def forward(self, x): return x.data if hasattr(x, 'data') else x


class PixelShuffle(Module):
    def __init__(self, upscale_factor): super().__init__(); self.upscale_factor = upscale_factor
    def forward(self, x):
        d = x.data if hasattr(x, 'data') else x
        N, C, H, W = d.shape; uf = self.upscale_factor
        C2 = C // (uf*uf)
        return d.reshape(N, C2, uf, uf, H, W).transpose(0,1,4,2,5,3).reshape(N, C2, H*uf, W*uf)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size=1): super().__init__(); self.output_size = output_size
    def forward(self, x):
        from scipy.ndimage import zoom
        d = x.data if hasattr(x, 'data') else x
        if isinstance(self.output_size, int): OH = OW = self.output_size
        else: OH, OW = self.output_size
        N, C, H, W = d.shape
        if H == OH and W == OW: return d
        z = [1, 1, max(OH/H, 0.01), max(OW/W, 0.01)]
        return zoom(d, z, order=1).astype(np.float32)


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride or self.kernel_size; self.padding = padding; self.dilation = dilation
    def forward(self, x):
        from scipy.ndimage import maximum_filter
        d = x.data if hasattr(x, 'data') else x
        if d.ndim == 2: d = d[:, np.newaxis, np.newaxis, :]
        if self.padding > 0:
            d = np.pad(d, [(0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)])
        kh, kw = self.kernel_size
        sh = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        result = maximum_filter(d, size=(1,1,kh,kw))
        return result[:,:,::sh[0],::sh[1]].astype(np.float32)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride or self.kernel_size; self.padding = padding
    def forward(self, x):
        from scipy.ndimage import uniform_filter
        d = x.data if hasattr(x, 'data') else x
        if d.ndim == 2: d = d[:, np.newaxis, np.newaxis, :]
        if self.padding > 0:
            d = np.pad(d, [(0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)])
        sh = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        filtered = uniform_filter(d, size=(1,1,self.kernel_size[0],self.kernel_size[1]), mode='constant')
        return filtered[:,:,::sh[0],::sh[1]].astype(np.float32)


class ReflectionPad2d(Module):
    def __init__(self, padding): super().__init__(); self.padding = padding
    def forward(self, x):
        d = x.data if hasattr(x, 'data') else x
        return np.pad(d, [(0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)], mode='reflect')


class ZeroPad2d(Module):
    def __init__(self, padding): super().__init__(); self.padding = padding
    def forward(self, x):
        d = x.data if hasattr(x, 'data') else x
        return np.pad(d, [(0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)], mode='constant')


class ConstantPad2d(Module):
    def __init__(self, padding, value=0): super().__init__(); self.padding = padding; self.value = value
    def forward(self, x):
        d = x.data if hasattr(x, 'data') else x
        return np.pad(d, [(0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)], constant_values=self.value)


class Upsample(Module):
    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
        super().__init__()
        self.size = size; self.scale_factor = scale_factor; self.mode = mode
        self.align_corners = align_corners
    def forward(self, x):
        from scipy.ndimage import zoom
        d = x.data if hasattr(x, 'data') else x
        if self.scale_factor is not None:
            sf = self.scale_factor if isinstance(self.scale_factor, tuple) else (self.scale_factor, self.scale_factor)
            z = [1, 1, sf[0], sf[1]]
        else:
            from skimage.transform import resize
            target = self.size if isinstance(self.size, tuple) else (self.size, self.size)
            N, C, H, W = d.shape
            result = np.zeros((N, C, target[0], target[1]), dtype=np.float32)
            for n in range(N):
                for c in range(C):
                    result[n, c] = resize(d[n, c], target, order=1).astype(np.float32)
            return result
        return zoom(d, z, order=1).astype(np.float32)


class Resize(Module):
    def __init__(self, size=None, scale_factor=None, mode="bilinear", align_corners=None):
        super().__init__()
        self.size = size; self.scale_factor = scale_factor; self.mode = mode
        self.align_corners = align_corners
    def forward(self, x):
        from skimage.transform import resize
        d = x.data if hasattr(x, 'data') else x
        N, C, H, W = d.shape
        target = self.size if isinstance(self.size, tuple) else (self.size, self.size)
        result = np.zeros((N, C, target[0], target[1]), dtype=np.float32)
        for n in range(N):
            for c in range(C):
                result[n, c] = resize(d[n, c], target, order=1).astype(np.float32)
        return result


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity="tanh", bias=True, dropout=0, batch_first=True):
        import math, sys
        super().__init__()
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        self.weight_ih = T.Tensor(np.random.randn(num_layers, input_size, hidden_size).astype(np.float32) * 0.1) if T else np.random.randn(num_layers, input_size, hidden_size).astype(np.float32)
        self.weight_hh = T.Tensor(np.random.randn(num_layers, hidden_size, hidden_size).astype(np.float32) * 0.1) if T else np.random.randn(num_layers, hidden_size, hidden_size).astype(np.float32)
        self.bias = T.Tensor(np.zeros(num_layers, hidden_size, dtype=np.float32)) if T and bias else None
        self.hidden_size = hidden_size; self.num_layers = num_layers

    def forward(self, x, hx=None):
        d = x.data if hasattr(x, 'data') else x
        if d.ndim == 3: d = d[:, -1, :]
        h = hx.data if hx is not None else np.zeros((self.num_layers, d.shape[0], self.hidden_size), dtype=np.float32)
        for t in range(d.shape[0]):
            h_new = np.tanh(np.matmul(d[t:t+1], self.weight_ih.data.T) + np.matmul(h[-1], self.weight_hh.data.T))
            h = np.concatenate([h, h_new[np.newaxis]], axis=0)
        return h[-1], h


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0, batch_first=True):
        import math
        import sys
        super().__init__()
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        self.weight_ih_l0 = T.Tensor(np.random.randn(4*hidden_size, input_size).astype(np.float32) * 0.1) if T else np.random.randn(4*hidden_size, input_size).astype(np.float32)
        self.weight_hh_l0 = T.Tensor(np.random.randn(4*hidden_size, hidden_size).astype(np.float32) * 0.1) if T else np.random.randn(4*hidden_size, hidden_size).astype(np.float32)
        self.bias_ih_l0 = T.Tensor(np.zeros(4*hidden_size, dtype=np.float32)) if T and bias else None
        self.bias_hh_l0 = T.Tensor(np.zeros(4*hidden_size, dtype=np.float32)) if T and bias else None
        self.hidden_size = hidden_size; self.num_layers = num_layers; self.batch_first = batch_first

    def forward(self, x, hx=None):
        d = x.data if hasattr(x, 'data') else x
        if self.batch_first and d.ndim == 3: d = d.transpose(1, 0)
        T_dim, B, _ = d.shape
        h = hx[0].data if hx is not None else np.zeros((self.num_layers, B, self.hidden_size), dtype=np.float32)
        c = hx[1].data if hx is not None else np.zeros((self.num_layers, B, self.hidden_size), dtype=np.float32)
        hs = []
        for t in range(T_dim):
            inp = d[t]
            gates_ih = np.matmul(inp, self.weight_ih_l0.data.T) + (self.bias_ih_l0.data if self.bias_ih_l0 else 0)
            gates_hh = np.matmul(h[-1], self.weight_hh_l0.data.T) + (self.bias_hh_l0.data if self.bias_hh_l0 else 0)
            gates = gates_ih + gates_hh
            i, f, g, o = np.split(gates, 4, axis=-1)
            i, f, g, o = (1/(1+np.exp(-i)), 1/(1+np.exp(-f)), np.tanh(g), 1/(1+np.exp(-o)))
            c_new = f * c[-1] + i * g; h_new = o * np.tanh(c_new)
            h = np.concatenate([h, h_new[np.newaxis]], axis=0); hs.append(h_new)
        if self.batch_first: hs = np.stack(hs, axis=0).transpose(1, 0, 2)
        else: hs = np.stack(hs, axis=0)
        return hs, (h[-1], c_new)


class GRU(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True):
        import sys
        super().__init__()
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        self.weight_ih = T.Tensor(np.random.randn(3*hidden_size, input_size).astype(np.float32) * 0.1) if T else np.random.randn(3*hidden_size, input_size).astype(np.float32)
        self.weight_hh = T.Tensor(np.random.randn(3*hidden_size, hidden_size).astype(np.float32) * 0.1) if T else np.random.randn(3*hidden_size, hidden_size).astype(np.float32)
        self.hidden_size = hidden_size; self.num_layers = num_layers; self.batch_first = batch_first

    def forward(self, x, hx=None):
        d = x.data if hasattr(x, 'data') else x
        if self.batch_first and d.ndim == 3: d = d.transpose(1, 0)
        T_dim, B, _ = d.shape
        h = hx.data if hx is not None else np.zeros((self.num_layers, B, self.hidden_size), dtype=np.float32)
        for t in range(T_dim):
            inp = d[t]
            xh = np.matmul(inp, self.weight_ih.data.T) + np.matmul(h[-1], self.weight_hh.data.T)
            r = 1/(1+np.exp(-xh[:, :self.hidden_size]))
            z = 1/(1+np.exp(-xh[:, self.hidden_size:2*self.hidden_size]))
            n = np.tanh(xh[:, 2*self.hidden_size:] + r * h[-1, :, 2*self.hidden_size:])
            h_new = (1-z) * n + z * h[-1]
            h = np.concatenate([h, h_new[np.newaxis]], axis=0)
        return h[-1], h


class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward); self.dropout = Dropout(dropout); self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model); self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout); self.dropout2 = Dropout(dropout)

    def forward(self, src, src_mask=None):
        attn_out = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(attn_out); src = self.norm1(src)
        ff = self.linear2(self.dropout(relu(self.linear1(src))))
        src = src + self.dropout2(ff); src = self.norm2(src)
        return src


class TransformerDecoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward); self.dropout = Dropout(dropout); self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model); self.norm2 = LayerNorm(d_model); self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout); self.dropout2 = Dropout(dropout); self.dropout3 = Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2); tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2); tgt = self.norm2(tgt)
        ff = self.linear2(self.dropout(relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(ff); tgt = self.norm3(tgt)
        return tgt


class Transformer(Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, batch_first=False):
        super().__init__()
        self.encoder_layers = ModuleList([TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first) for _ in range(num_encoder_layers)])
        self.decoder_layers = ModuleList([TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first) for _ in range(num_decoder_layers)])
        self.batch_first = batch_first

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        for layer in self.encoder_layers: src = layer(src, src_mask)
        for layer in self.decoder_layers: tgt = layer(tgt, src, tgt_mask, memory_mask)
        return src, tgt


class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, kdim=None, vdim=None, batch_first=False):
        import math
        super().__init__()
        self.embed_dim = embed_dim; self.num_heads = num_heads; self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias); self.k_proj = Linear(kdim or embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(vdim or embed_dim, embed_dim, bias=bias); self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        import sys
        _T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        q = self.q_proj(query); k = self.k_proj(key); v = self.v_proj(value)
        # Ensure internal format is (T, N, E) where T=seq_len, N=batch
        if self.batch_first:
            # Input is (N, T, E) → transpose to (T, N, E)
            q = q.transpose(0, 1); k = k.transpose(0, 1); v = v.transpose(0, 1)
        # q/k/v: (T, N, embed_dim)
        T_len = q.shape[0]; N_batch = q.shape[1]; S = k.shape[0]
        H = self.num_heads; D = self.head_dim
        # Reshape to (T_len, N_batch*H, D) then transpose to (N_batch*H, T_len, D)
        q = q.reshape(T_len, N_batch * H, D).transpose(0, 1)
        k = k.reshape(S, N_batch * H, D).transpose(0, 1)
        v = v.reshape(S, N_batch * H, D).transpose(0, 1)
        # Attention scores: (N_batch*H, T_len, S)
        scale = 1.0 / (D ** 0.5)
        attn = (q @ k.transpose(1, 2)) * scale
        if attn_mask is not None:
            if not isinstance(attn_mask, _T.Tensor):
                attn_mask = _T.Tensor(attn_mask)
            # attn_mask is (T_len, S) or broadcastable
            attn = _T.where(attn_mask, attn, -1e9)
        probs = _T.softmax(attn, dim=-1)
        probs = self.dropout(probs)
        out = (probs @ v).transpose(0, 1).reshape(T_len, N_batch, -1)
        if self.batch_first:
            out = out.transpose(0, 1)
        return self.out_proj(out), _T.zeros(N_batch, T_len, T_len)


class RNNCellBase(Module):
    def __init__(self, input_size, hidden_size, bias=True, num_chunks=4):
        import sys
        super().__init__()
        T = sys.modules.get("torch", sys.modules.get("domains.training.torch"))
        self.weight_ih = T.Tensor(np.random.randn(num_chunks*hidden_size, input_size).astype(np.float32) * 0.1) if T else np.random.randn(num_chunks*hidden_size, input_size).astype(np.float32)
        self.weight_hh = T.Tensor(np.random.randn(num_chunks*hidden_size, hidden_size).astype(np.float32) * 0.1) if T else np.random.randn(num_chunks*hidden_size, hidden_size).astype(np.float32)
        self.bias = T.Tensor(np.zeros(num_chunks*hidden_size, dtype=np.float32)) if T and bias else None
        self.hidden_size = hidden_size


class LSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True): super().__init__(input_size, hidden_size, bias, 4)
    def forward(self, x, hx=None):
        h, c = (hx[0].data, hx[1].data) if hx is not None else (np.zeros((x.shape[0], self.hidden_size), dtype=np.float32), np.zeros((x.shape[0], self.hidden_size), dtype=np.float32))
        xd = x.data if hasattr(x, 'data') else x
        gates = np.matmul(xd, self.weight_ih.data.T) + np.matmul(h, self.weight_hh.data.T)
        if self.bias is not None: gates += self.bias.data
        i, f, g, o = np.split(gates, 4, axis=-1)
        i, f, g, o = (1/(1+np.exp(-i)), 1/(1+np.exp(-f)), np.tanh(g), 1/(1+np.exp(-o)))
        c_new = f * c + i * g; h_new = o * np.tanh(c_new)
        return h_new, c_new


class GRUCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True): super().__init__(input_size, hidden_size, bias, 3)
    def forward(self, x, hx=None):
        h = hx.data if hx is not None else np.zeros((x.shape[0], self.hidden_size), dtype=np.float32)
        xd = x.data if hasattr(x, 'data') else x
        xh = np.matmul(xd, self.weight_ih.data.T) + np.matmul(h, self.weight_hh.data.T)
        r = 1/(1+np.exp(-(xh[:, :self.hidden_size])))
        z = 1/(1+np.exp(-(xh[:, self.hidden_size:2*self.hidden_size])))
        n = np.tanh(xh[:, 2*self.hidden_size:] + r * h[:, 2*self.hidden_size:])
        h_new = (1-z) * n + z * h
        return h_new


def relu(x):
    d = x.data if hasattr(x, 'data') else x
    return np.maximum(d, 0)
