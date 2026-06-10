"""
torch.nn.functional — numpy-backed functional API.
"""

import numpy as np


def _tensor(x):
    import sys
    T = sys.modules.get("torch")
    if T is not None and hasattr(T, "Tensor"):
        if isinstance(x, T.Tensor):
            return x
        return T.Tensor(x)
    return x


def softmax(x, dim=None):
    d = x.data if hasattr(x, "data") else np.asarray(x)
    if dim is None:
        dim = max(0, d.ndim - 1)
    exp_d = np.exp(d - np.max(d, axis=dim, keepdims=True))
    return _tensor(exp_d / np.sum(exp_d, axis=dim, keepdims=True))


def log_softmax(x, dim=None):
    d = x.data if hasattr(x, "data") else np.asarray(x)
    if dim is None:
        dim = max(0, d.ndim - 1)
    x_max = np.max(d, axis=dim, keepdims=True)
    return _tensor(d - x_max - np.log(np.sum(np.exp(d - x_max), axis=dim, keepdims=True)))


def cross_entropy(inputs, target, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0):
    nll = nll_loss(inputs, target, weight=weight, ignore_index=ignore_index, reduction=reduction)
    if label_smoothing > 0:
        num_classes = inputs.shape[-1] if inputs.ndim > 0 else inputs.shape[0]
        smooth = _tensor(np.full_like(inputs, label_smoothing / max(num_classes, 1)))
        return (1 - label_smoothing) * nll + label_smoothing * smooth.mean()
    return nll


def nll_loss(inputs, target, weight=None, ignore_index=-100, reduction="mean"):
    d = inputs.data if hasattr(inputs, "data") else np.asarray(inputs)
    t = target.data if hasattr(target, "data") else np.asarray(target)
    if t.ndim > 0:
        t = t.astype(int).flatten()
    valid_mask = (t != ignore_index)
    if not np.any(valid_mask):
        return _tensor(np.array(0.0))
    batch_size = t.shape[0]
    flat_d = d.reshape(batch_size, -1)
    flat_t = t.flatten()
    valid_indices = np.where(valid_mask)[0]
    losses = []
    for i in valid_indices:
        idx = flat_t[i]
        if idx < flat_d.shape[1]:
            losses.append(-flat_d[i, int(idx)])
    if not losses:
        return _tensor(np.array(0.0))
    loss = np.array(losses)
    if weight is not None:
        w = weight.data if hasattr(weight, "data") else np.asarray(weight)
        loss = loss * w[flat_t[valid_indices]]
    if reduction == "mean":
        return _tensor(loss.mean())
    elif reduction == "sum":
        return _tensor(loss.sum())
    return _tensor(loss)


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    d = x.data if hasattr(x, "data") else np.asarray(x)
    if isinstance(normalized_shape, int):
        axis = tuple(range(d.ndim - 1))
    else:
        axis = tuple(range(d.ndim - len(normalized_shape)))
    mean = d.mean(axis=axis, keepdims=True)
    std = d.std(axis=axis, keepdims=True)
    y = (d - mean) / (std + eps)
    if weight is not None:
        w = weight.data if hasattr(weight, "data") else np.asarray(weight)
        y = y * w
    if bias is not None:
        b = bias.data if hasattr(bias, "data") else np.asarray(bias)
        y = y + b
    return _tensor(y)


def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    d = input.data if hasattr(input, "data") else np.asarray(input)
    flat = d.flatten().astype(int)
    if padding_idx is not None:
        mask = flat != padding_idx
        flat = flat[mask]
    rows = weight.data if hasattr(weight, "data") else np.asarray(weight)
    out = rows[flat]
    return _tensor(out)


def embedding_bag(input, weight, offsets=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, mode="mean", sparse=False, per_sample_weights=None, include_last_offset=False, padding_idx=None):
    d = input.data if hasattr(input, "data") else np.asarray(input)
    flat = d.flatten().astype(int)
    rows = weight.data if hasattr(weight, "data") else np.asarray(weight)
    if offsets is not None:
        off = offsets.data if hasattr(offsets, "data") else np.asarray(offsets)
        segments = []
        for i in range(len(off) - 1):
            seg = flat[off[i]:off[i+1]]
            if len(seg) > 0:
                segments.append(seg)
        if len(segments) == 0:
            return _tensor(np.array(0.0))
        embeddings = [rows[s] for s in segments]
        if mode == "mean":
            return _tensor(np.stack([e.mean(axis=0) for e in embeddings]))
        elif mode == "sum":
            return _tensor(np.stack([e.sum(axis=0) for e in embeddings]))
    return _tensor(rows[flat])


def relu(x, inplace=False):
    d = x.data if hasattr(x, "data") else np.asarray(x)
    return _tensor(np.maximum(d, 0))


def leaky_relu(x, negative_slope=1e-2, inplace=False):
    d = x.data if hasattr(x, "data") else np.asarray(x)
    return _tensor(np.where(d > 0, d, d * negative_slope))


def tanh(x):
    d = x.data if hasattr(x, "data") else np.asarray(x)
    return _tensor(np.tanh(d))


def sigmoid(x):
    d = x.data if hasattr(x, "data") else np.asarray(x)
    return _tensor(1 / (1 + np.exp(-d)))


def dropout(x, p=0.5, training=True, inplace=False):
    if not training or p == 0:
        return x
    d = x.data if hasattr(x, "data") else np.asarray(x)
    mask = np.random.binomial(1, 1 - p, d.shape)
    return _tensor(d * mask / (1 - p))


def linear(input, weight, bias=None):
    d = input.data if hasattr(input, "data") else np.asarray(input)
    w = weight.data if hasattr(weight, "data") else np.asarray(weight)
    out = np.matmul(d, w.T)
    if bias is not None:
        b = bias.data if hasattr(bias, "data") else np.asarray(bias)
        out = out + b
    return _tensor(out)


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    d = input.data if hasattr(input, "data") else np.asarray(input)
    w = weight.data if hasattr(weight, "data") else np.asarray(weight)
    if isinstance(padding, int):
        pad = (padding, padding)
    else:
        pad = tuple(padding)
    if pad != (0, 0):
        d = np.pad(d, [(0, 0)] * (d.ndim - 2) + [pad, pad], mode='constant')
    if isinstance(stride, int):
        stride = (stride,)
    s = stride[0]
    out_len = (d.shape[-1] - w.shape[-1]) // s + 1
    out = np.zeros((d.shape[0], w.shape[0] * groups, out_len))
    out_c = w.shape[0] * groups
    for c in range(out_c):
        k = c % w.shape[0]
        for i in range(out_len):
            out[:, c, i] = np.correlate(d[:, c // groups, i*s:i*s+w.shape[-1]], w[k, :, 0], mode='valid')
    if bias is not None:
        b = bias.data if hasattr(bias, "data") else np.asarray(bias)
        out = out + b[:, None]
    return _tensor(out)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    d = input.data if hasattr(input, "data") else np.asarray(input)
    w = weight.data if hasattr(weight, "data") else np.asarray(weight)
    if isinstance(padding, int):
        pad_h = pad_w = padding
    elif len(padding) == 2:
        pad_h, pad_w = padding if isinstance(padding[0], int) else (padding[0][0], padding[0][1])
    else:
        pad_h = pad_w = 0
    if pad_h > 0 or pad_w > 0:
        pad_width = [(0, 0)] * (d.ndim - 2) + [(pad_h, pad_h), (pad_w, pad_w)]
        d = np.pad(d, pad_width, mode='constant')
    if isinstance(stride, int):
        s_h = s_w = stride
    else:
        s_h, s_w = stride[0], stride[1]
    out_h = (d.shape[-2] - w.shape[-2]) // s_h + 1
    out_w = (d.shape[-1] - w.shape[-1]) // s_w + 1
    out = np.zeros((d.shape[0], w.shape[0] * groups, out_h, out_w))
    for b in range(d.shape[0]):
        for c_out in range(w.shape[0] * groups):
            c_in = c_out % (d.shape[1] // groups)
            k_idx = c_out % w.shape[0]
            for i in range(0, d.shape[-2] - w.shape[-2] + 1, s_h):
                for j in range(0, d.shape[-1] - w.shape[-1] + 1, s_w):
                    patch = d[b, c_in, i:i+w.shape[-2], j:j+w.shape[-1]]
                    out[b, c_out, i // s_h, j // s_w] = (patch * w[k_idx]).sum()
    if bias is not None:
        b = bias.data if hasattr(bias, "data") else np.asarray(bias)
        out = out + b[:, None, None]
    return _tensor(out)


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    d = query.data if hasattr(query, "data") else np.asarray(query)
    dk = key.data if hasattr(key, "data") else np.asarray(key)
    dv = value.data if hasattr(value, "data") else np.asarray(value)
    if scale is None:
        scale = dk.shape[-1] ** -0.5
    scores = np.matmul(d * scale, dk.transpose(-2, -1))
    if attn_mask is not None:
        mask = attn_mask.data if hasattr(attn_mask, "data") else np.asarray(attn_mask)
        scores = scores + mask
    attn = softmax(_tensor(scores), dim=-1)
    return _tensor(np.matmul(attn.data if hasattr(attn, "data") else np.asarray(attn), dv))


def gelu(x, approximate="none"):
    d = x.data if hasattr(x, "data") else np.asarray(x)
    if approximate == "tanh":
        return _tensor(0.5 * d * (1 + np.tanh(np.sqrt(2/np.pi) * (d + 0.044715 * d**3))))
    return _tensor(0.5 * d * (1 + np.math.erf(d / np.sqrt(2))))


def silu(x):
    return _tensor(x.data * sigmoid(x).data if hasattr(x, "data") else x * (1 / (1 + np.exp(-x))))


def binary_cross_entropy_with_logits(input, target, weight=None, pos_weight=None, reduction="mean"):
    d = input.data if hasattr(input, "data") else np.asarray(input)
    t = target.data if hasattr(target, "data") else np.asarray(target)
    max_val = np.maximum(-d, 0)
    loss = (1 - t) * d + max_val + np.log(np.exp(-max_val) + np.exp(-d - max_val))
    if pos_weight is not None:
        p = pos_weight.data if hasattr(pos_weight, "data") else np.asarray(pos_weight)
        loss = loss * (t * (p - 1) + 1)
    if weight is not None:
        w = weight.data if hasattr(weight, "data") else np.asarray(weight)
        loss = loss * w
    if reduction == "mean":
        return _tensor(loss.mean())
    elif reduction == "sum":
        return _tensor(loss.sum())
    return _tensor(loss)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    d1 = x1.data if hasattr(x1, "data") else np.asarray(x1)
    d2 = x2.data if hasattr(x2, "data") else np.asarray(x2)
    dot = np.sum(d1 * d2, axis=dim)
    norm1 = np.linalg.norm(d1, axis=dim)
    norm2 = np.linalg.norm(d2, axis=dim)
    return _tensor(dot / (norm1 * norm2 + eps))


def pad(input, pad, mode='constant', value=0.0):
    d = input.data if hasattr(input, "data") else np.asarray(input)
    if isinstance(pad, int):
        pad = (pad,)
    pad_width = [(0, 0)] * d.ndim
    for i, p in enumerate(pad):
        axis = d.ndim - 1 - (i // 2)
        if i % 2 == 0:
            pad_width[axis] = (p, pad_width[axis][1])
        else:
            pad_width[axis] = (pad_width[axis][0], p)
    return _tensor(np.pad(d, pad_width, mode='constant', constant_values=value))


def one_hot(tensor, num_classes=-1):
    d = tensor.data if hasattr(tensor, "data") else np.asarray(tensor)
    if num_classes < 0:
        num_classes = int(d.max()) + 1
    one = np.zeros(list(d.shape) + [num_classes])
    for idx in np.ndindex(d.shape):
        one[idx + (int(d[idx]),)] = 1
    return _tensor(one)