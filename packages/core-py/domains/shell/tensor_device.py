"""
TensorDevice — pure compute engine (no I/O).

Just math: matmul, relu, softmax, add, mul, conv, pooling, etc.
No model awareness, no file I/O, no assembly interface.
"""

from __future__ import annotations

import numpy as np
from typing import Any


class TensorDevice:
    """Pure compute engine — wraps numpy for tensor operations.

    No I/O. No model awareness. Just math.
    """

    def __init__(self):
        self._ops = {
            # Linear algebra
            "matmul": self._matmul,
            "dot": self._dot,
            "inv": self._inv,
            "svd": self._svd,
            "eig": self._eig,

            # Activation functions
            "relu": self._relu,
            "leaky_relu": self._leaky_relu,
            "sigmoid": self._sigmoid,
            "tanh": self._tanh,
            "softmax": self._softmax,
            "log_softmax": self._log_softmax,
            "gelu": self._gelu,
            "silu": self._silu,
            "elu": self._elu,
            "selu": self._selu,

            # Arithmetic
            "add": self._add,
            "sub": self._sub,
            "mul": self._mul,
            "div": self._div,
            "neg": self._neg,
            "abs": self._abs,
            "pow": self._pow,
            "sqrt": self._sqrt,
            "exp": self._exp,
            "log": self._log,

            # Reduction
            "sum": self._sum,
            "mean": self._mean,
            "std": self._std,
            "var": self._var,
            "max": self._max,
            "min": self._min,
            "argmax": self._argmax,
            "argmin": self._argmin,

            # Shape
            "reshape": self._reshape,
            "transpose": self._transpose,
            "flatten": self._flatten,
            "squeeze": self._squeeze,
            "unsqueeze": self._unsqueeze,
            "cat": self._cat,
            "stack": self._stack,

            # Convolution
            "conv1d": self._conv1d,
            "conv2d": self._conv2d,

            # Pooling
            "max_pool1d": self._max_pool1d,
            "max_pool2d": self._max_pool2d,
            "avg_pool1d": self._avg_pool1d,
            "avg_pool2d": self._avg_pool2d,
            "adaptive_avg_pool1d": self._adaptive_avg_pool1d,
            "adaptive_avg_pool2d": self._adaptive_avg_pool2d,

            # Normalization
            "batch_norm": self._batch_norm,
            "layer_norm": self._layer_norm,
            "rms_norm": self._rms_norm,
            "instance_norm": self._instance_norm,

            # Attention
            "attention": self._attention,
            "scaled_dot_product_attention": self._scaled_dot_product_attention,

            # Loss functions
            "cross_entropy": self._cross_entropy,
            "mse": self._mse,
            "mae": self._mae,
            "binary_cross_entropy": self._binary_cross_entropy,

            # Optimizers (stateless — return updates)
            "sgd_step": self._sgd_step,
            "adam_step": self._adam_step,

            # Utility
            "clip_grad_norm": self._clip_grad_norm,
            "dropout": self._dropout,
            "embedding": self._embedding,
            "linear": self._linear,
        }

    def __call__(self, op: str, *args, **kwargs) -> Any:
        """Execute operation."""
        fn = self._ops.get(op)
        if fn is None:
            raise ValueError(f"unknown op: {op}")
        return fn(*args, **kwargs)

    def list_ops(self) -> list[str]:
        """List all available operations."""
        return sorted(self._ops.keys())

    # ── Helpers ───────────────────────────────────────────────────────────

    def _to_arr(self, v: Any) -> np.ndarray:
        if isinstance(v, np.ndarray):
            return v
        if isinstance(v, (int, float)):
            return np.float64(v)
        if isinstance(v, list):
            return np.array(v, dtype=np.float64)
        if isinstance(v, tuple):
            return np.array(v, dtype=np.float64)
        return np.array(v, dtype=np.float64)

    # ── Linear algebra ────────────────────────────────────────────────────

    def _matmul(self, a, b):
        return self._to_arr(a) @ self._to_arr(b)

    def _dot(self, a, b):
        return np.dot(self._to_arr(a), self._to_arr(b))

    def _inv(self, a):
        return np.linalg.inv(self._to_arr(a))

    def _svd(self, a):
        return np.linalg.svd(self._to_arr(a))

    def _eig(self, a):
        return np.linalg.eig(self._to_arr(a))

    # ── Activation functions ──────────────────────────────────────────────

    def _relu(self, a):
        return np.maximum(0, self._to_arr(a))

    def _leaky_relu(self, a, alpha=0.01):
        x = self._to_arr(a)
        return np.where(x > 0, x, alpha * x)

    def _sigmoid(self, a):
        x = self._to_arr(a)
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _tanh(self, a):
        return np.tanh(self._to_arr(a))

    def _softmax(self, a, axis=-1):
        x = self._to_arr(a)
        shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _log_softmax(self, a, axis=-1):
        return np.log(self._softmax(a, axis=axis) + 1e-8)

    def _gelu(self, a):
        x = self._to_arr(a)
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def _silu(self, a):
        x = self._to_arr(a)
        return x * self._sigmoid(x)

    def _elu(self, a, alpha=1.0):
        x = self._to_arr(a)
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def _selu(self, a):
        x = self._to_arr(a)
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

    # ── Arithmetic ────────────────────────────────────────────────────────

    def _add(self, a, b):
        return self._to_arr(a) + self._to_arr(b)

    def _sub(self, a, b):
        return self._to_arr(a) - self._to_arr(b)

    def _mul(self, a, b):
        return self._to_arr(a) * self._to_arr(b)

    def _div(self, a, b):
        return self._to_arr(a) / self._to_arr(b)

    def _neg(self, a):
        return -self._to_arr(a)

    def _abs(self, a):
        return np.abs(self._to_arr(a))

    def _pow(self, a, exp):
        return np.power(self._to_arr(a), exp)

    def _sqrt(self, a):
        return np.sqrt(self._to_arr(a))

    def _exp(self, a):
        return np.exp(self._to_arr(a))

    def _log(self, a):
        return np.log(self._to_arr(a) + 1e-8)

    # ── Reduction ─────────────────────────────────────────────────────────

    def _sum(self, a, axis=None):
        return np.sum(self._to_arr(a), axis=axis)

    def _mean(self, a, axis=None):
        return np.mean(self._to_arr(a), axis=axis)

    def _std(self, a, axis=None):
        return np.std(self._to_arr(a), axis=axis)

    def _var(self, a, axis=None):
        return np.var(self._to_arr(a), axis=axis)

    def _max(self, a, axis=None):
        return np.max(self._to_arr(a), axis=axis)

    def _min(self, a, axis=None):
        return np.min(self._to_arr(a), axis=axis)

    def _argmax(self, a, axis=-1):
        return np.argmax(self._to_arr(a), axis=axis)

    def _argmin(self, a, axis=-1):
        return np.argmin(self._to_arr(a), axis=axis)

    # ── Shape ─────────────────────────────────────────────────────────────

    def _reshape(self, a, shape):
        return self._to_arr(a).reshape(shape)

    def _transpose(self, a, axes=None):
        return self._to_arr(a).transpose(axes)

    def _flatten(self, a):
        return self._to_arr(a).flatten()

    def _squeeze(self, a, axis=None):
        return np.squeeze(self._to_arr(a), axis=axis)

    def _unsqueeze(self, a, axis=0):
        return np.expand_dims(self._to_arr(a), axis=axis)

    def _cat(self, arrays, axis=0):
        return np.concatenate([self._to_arr(a) for a in arrays], axis=axis)

    def _stack(self, arrays, axis=0):
        return np.stack([self._to_arr(a) for a in arrays], axis=axis)

    # ── Convolution ───────────────────────────────────────────────────────

    def _conv1d(self, input, weight, stride=1, padding=0, dilation=1, groups=1):
        """1D convolution."""
        x = self._to_arr(input)
        w = self._to_arr(weight)
        # Simple implementation using im2col
        batch, in_channels, seq_len = x.shape
        out_channels, _, kernel_size = w.shape
        out_seq = (seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        # Pad input
        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode='constant')

        # im2col
        cols = np.zeros((batch, in_channels, kernel_size, out_seq))
        for i in range(kernel_size):
            start = i * dilation
            cols[:, :, i, :] = x[:, :, start:start + out_seq * stride:stride]

        # Reshape for matmul
        cols = cols.transpose(0, 3, 1, 2).reshape(batch * out_seq, in_channels * kernel_size)
        w_flat = w.reshape(out_channels, -1)

        # Convolution as matmul
        out = cols @ w_flat.T
        out = out.reshape(batch, out_seq, out_channels).transpose(0, 2, 1)
        return out

    def _conv2d(self, input, weight, stride=1, padding=0, dilation=1, groups=1):
        """2D convolution."""
        x = self._to_arr(input)
        w = self._to_arr(weight)
        batch, in_channels, h, w_in = x.shape
        out_channels, _, kh, kw = w.shape
        h_out = (h + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
        w_out = (w_in + 2 * padding - dilation * (kw - 1) - 1) // stride + 1

        # Pad input
        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

        # im2col
        cols = np.zeros((batch, in_channels, kh, kw, h_out, w_out))
        for i in range(kh):
            for j in range(kw):
                h_start = i * dilation
                w_start = j * dilation
                cols[:, :, i, j, :, :] = x[:, :, h_start:h_start + h_out * stride:stride,
                                               w_start:w_start + w_out * stride:stride]

        # Reshape for matmul
        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(
            batch * h_out * w_out, in_channels * kh * kw)
        w_flat = w.reshape(out_channels, -1)

        # Convolution as matmul
        out = cols @ w_flat.T
        out = out.reshape(batch, h_out, w_out, out_channels).transpose(0, 3, 1, 2)
        return out

    # ── Pooling ───────────────────────────────────────────────────────────

    def _max_pool1d(self, input, kernel_size, stride=None, padding=0):
        x = self._to_arr(input)
        if stride is None:
            stride = kernel_size
        batch, channels, seq_len = x.shape
        out_len = (seq_len + 2 * padding - kernel_size) // stride + 1

        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode='constant',
                       constant_values=-np.inf)

        out = np.zeros((batch, channels, out_len))
        for i in range(out_len):
            start = i * stride
            out[:, :, i] = np.max(x[:, :, start:start + kernel_size], axis=2)
        return out

    def _max_pool2d(self, input, kernel_size, stride=None, padding=0):
        x = self._to_arr(input)
        if stride is None:
            stride = kernel_size
        batch, channels, h, w = x.shape
        kh = kw = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        sh = sw = stride if isinstance(stride, int) else stride[0]
        h_out = (h + 2 * padding - kh) // sh + 1
        w_out = (w + 2 * padding - kw) // sw + 1

        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                       mode='constant', constant_values=-np.inf)

        out = np.zeros((batch, channels, h_out, w_out))
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * sh
                w_start = j * sw
                out[:, :, i, j] = np.max(x[:, :, h_start:h_start + kh, w_start:w_start + kw],
                                         axis=(2, 3))
        return out

    def _avg_pool1d(self, input, kernel_size, stride=None, padding=0):
        x = self._to_arr(input)
        if stride is None:
            stride = kernel_size
        batch, channels, seq_len = x.shape
        out_len = (seq_len + 2 * padding - kernel_size) // stride + 1

        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode='constant')

        out = np.zeros((batch, channels, out_len))
        for i in range(out_len):
            start = i * stride
            out[:, :, i] = np.mean(x[:, :, start:start + kernel_size], axis=2)
        return out

    def _avg_pool2d(self, input, kernel_size, stride=None, padding=0):
        x = self._to_arr(input)
        if stride is None:
            stride = kernel_size
        batch, channels, h, w = x.shape
        kh = kw = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        sh = sw = stride if isinstance(stride, int) else stride[0]
        h_out = (h + 2 * padding - kh) // sh + 1
        w_out = (w + 2 * padding - kw) // sw + 1

        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                       mode='constant')

        out = np.zeros((batch, channels, h_out, w_out))
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * sh
                w_start = j * sw
                out[:, :, i, j] = np.mean(x[:, :, h_start:h_start + kh, w_start:w_start + kw],
                                          axis=(2, 3))
        return out

    def _adaptive_avg_pool1d(self, output_size):
        def pool(input):
            x = self._to_arr(input)
            batch, channels, seq_len = x.shape
            out = np.zeros((batch, channels, output_size))
            for i in range(output_size):
                start = int(i * seq_len / output_size)
                end = int((i + 1) * seq_len / output_size)
                out[:, :, i] = np.mean(x[:, :, start:end], axis=2)
            return out
        return pool

    def _adaptive_avg_pool2d(self, output_size):
        def pool(input):
            x = self._to_arr(input)
            batch, channels, h, w = x.shape
            h_out, w_out = output_size
            out = np.zeros((batch, channels, h_out, w_out))
            for i in range(h_out):
                for j in range(w_out):
                    h_start = int(i * h / h_out)
                    h_end = int((i + 1) * h / h_out)
                    w_start = int(j * w / w_out)
                    w_end = int((j + 1) * w / w_out)
                    out[:, :, i, j] = np.mean(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))
            return out
        return pool

    # ── Normalization ─────────────────────────────────────────────────────

    def _batch_norm(self, input, weight, bias, mean, var, eps=1e-5, momentum=0.1, training=True):
        x = self._to_arr(input)
        w = self._to_arr(weight)
        b = self._to_arr(bias)
        m = self._to_arr(mean)
        v = self._to_arr(var)

        if training:
            x_hat = (x - m) / np.sqrt(v + eps)
        else:
            x_hat = (x - m) / np.sqrt(v + eps)
        return w * x_hat + b

    def _layer_norm(self, input, weight, bias, eps=1e-5):
        x = self._to_arr(input)
        w = self._to_arr(weight)
        b = self._to_arr(bias)

        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + eps)
        return w * x_hat + b

    def _rms_norm(self, input, weight, eps=1e-6):
        x = self._to_arr(input)
        w = self._to_arr(weight)
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        return x / rms * w

    def _instance_norm(self, input, weight, bias, eps=1e-5):
        x = self._to_arr(input)
        w = self._to_arr(weight)
        b = self._to_arr(bias)

        mean = np.mean(x, axis=(2, 3), keepdims=True)
        var = np.var(x, axis=(2, 3), keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + eps)
        return w * x_hat + b

    # ── Attention ─────────────────────────────────────────────────────────

    def _attention(self, query, key, value, mask=None):
        """Scaled dot-product attention."""
        q = self._to_arr(query)
        k = self._to_arr(key)
        v = self._to_arr(value)

        d_k = q.shape[-1]
        scores = np.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        weights = self._softmax(scores, axis=-1)
        return np.matmul(weights, v)

    def _scaled_dot_product_attention(self, query, key, value, mask=None, scale=None):
        return self._attention(query, key, value, mask)

    # ── Loss functions ────────────────────────────────────────────────────

    def _cross_entropy(self, input, target, ignore_index=-100):
        """Cross entropy loss."""
        logits = self._to_arr(input)
        targets = self._to_arr(target).astype(int)

        # Numerically stable log softmax
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True) + 1e-8)
        log_probs = shifted - log_sum_exp

        # Gather log probs at target indices
        nll = -np.take_along_axis(log_probs, targets[..., np.newaxis], axis=-1).squeeze(-1)

        # Mask ignored indices
        if ignore_index >= 0:
            mask = targets != ignore_index
            nll = nll * mask
            return np.sum(nll) / np.sum(mask)
        return np.mean(nll)

    def _mse(self, input, target):
        return np.mean((self._to_arr(input) - self._to_arr(target)) ** 2)

    def _mae(self, input, target):
        return np.mean(np.abs(self._to_arr(input) - self._to_arr(target)))

    def _binary_cross_entropy(self, input, target, eps=1e-8):
        x = self._to_arr(input)
        t = self._to_arr(target)
        return -np.mean(t * np.log(x + eps) + (1 - t) * np.log(1 - x + eps))

    # ── Optimizers ────────────────────────────────────────────────────────

    def _sgd_step(self, params, grads, lr=0.01, momentum=0.0, state=None):
        """SGD step — returns updated params."""
        updates = {}
        for name in params:
            grad = grads.get(name)
            if grad is None:
                continue
            g = self._to_arr(grad)
            if momentum > 0 and state is not None:
                v = momentum * state.get(name, np.zeros_like(g)) + g
                state[name] = v
                updates[name] = params[name] - lr * v
            else:
                updates[name] = params[name] - lr * g
        return updates

    def _adam_step(self, params, grads, lr=0.001, betas=(0.9, 0.999), eps=1e-8, state=None):
        """Adam step — returns updated params."""
        updates = {}
        b1, b2 = betas
        for name in params:
            grad = grads.get(name)
            if grad is None:
                continue
            g = self._to_arr(grad)

            if state is not None:
                m = b1 * state.get(f"{name}_m", np.zeros_like(g)) + (1 - b1) * g
                v = b2 * state.get(f"{name}_v", np.zeros_like(g)) + (1 - b2) * g ** 2
                state[f"{name}_m"] = m
                state[f"{name}_v"] = v
                m_hat = m / (1 - b1)
                v_hat = v / (1 - b2)
                updates[name] = params[name] - lr * m_hat / (np.sqrt(v_hat) + eps)
            else:
                updates[name] = params[name] - lr * g
        return updates

    # ── Utility ───────────────────────────────────────────────────────────

    def _clip_grad_norm(self, grads, max_norm=1.0):
        """Clip gradients by global norm."""
        total_norm = 0.0
        for g in grads.values():
            arr = self._to_arr(g)
            total_norm += np.sum(arr ** 2)
        total_norm = np.sqrt(total_norm)

        clip_coef = max_norm / (total_norm + 1e-8)
        if clip_coef < 1:
            return {k: self._to_arr(v) * clip_coef for k, v in grads.items()}
        return dict(grads)

    def _dropout(self, input, p=0.5, training=True):
        """Dropout."""
        if not training:
            return self._to_arr(input)
        x = self._to_arr(input)
        mask = np.random.binomial(1, 1 - p, size=x.shape) / (1 - p)
        return x * mask

    def _embedding(self, input, weight):
        """Embedding lookup."""
        indices = self._to_arr(input).astype(int)
        w = self._to_arr(weight)
        return w[indices]

    def _linear(self, input, weight, bias=None):
        """Linear layer (matmul + bias)."""
        out = self._matmul(input, weight)
        if bias is not None:
            out = out + self._to_arr(bias)
        return out
