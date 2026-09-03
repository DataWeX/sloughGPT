"""
TensorDevice — standalone compute hardware.

Pure compute engine with clean ioctl interface.
No inheritance. No decorators. Just a device.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from .ioctl import IoctlCommand
from .kernel_syscall import SyscallResult


class TensorDevice:
    """Standalone compute hardware — wraps numpy.

    Has clean ioctl interface for assembly.
    Has function calls for direct use.
    """

    def __init__(self, name: str = "tensor"):
        self._name = name
        self._ops = {
            # Linear algebra
            IoctlCommand.MATMUL: self._matmul,
            IoctlCommand.DOT: self._dot,
            IoctlCommand.INV: self._inv,
            IoctlCommand.SVD: self._svd,
            IoctlCommand.EIG: self._eig,

            # Activation functions
            IoctlCommand.RELU: self._relu,
            IoctlCommand.LEAKY_RELU: self._leaky_relu,
            IoctlCommand.SIGMOID: self._sigmoid,
            IoctlCommand.TANH: self._tanh,
            IoctlCommand.SOFTMAX: self._softmax,
            IoctlCommand.LOG_SOFTMAX: self._log_softmax,
            IoctlCommand.GELU: self._gelu,
            IoctlCommand.SILU: self._silu,
            IoctlCommand.ELU: self._elu,
            IoctlCommand.SELU: self._selu,

            # Arithmetic
            IoctlCommand.ADD: self._add,
            IoctlCommand.SUB: self._sub,
            IoctlCommand.MUL: self._mul,
            IoctlCommand.DIV: self._div,
            IoctlCommand.NEG: self._neg,
            IoctlCommand.ABS: self._abs,
            IoctlCommand.POW: self._pow,
            IoctlCommand.SQRT: self._sqrt,
            IoctlCommand.EXP: self._exp,
            IoctlCommand.LOG: self._log,

            # Reduction
            IoctlCommand.SUM: self._sum,
            IoctlCommand.MEAN: self._mean,
            IoctlCommand.STD: self._std,
            IoctlCommand.VAR: self._var,
            IoctlCommand.MAX: self._max,
            IoctlCommand.MIN: self._min,
            IoctlCommand.ARGMAX: self._argmax,
            IoctlCommand.ARGMIN: self._argmin,

            # Shape
            IoctlCommand.RESHAPE: self._reshape,
            IoctlCommand.TRANSPOSE: self._transpose,
            IoctlCommand.FLATTEN: self._flatten,
            IoctlCommand.SQUEEZE: self._squeeze,
            IoctlCommand.UNSQUEEZE: self._unsqueeze,
            IoctlCommand.CAT: self._cat,
            IoctlCommand.STACK: self._stack,

            # Convolution
            IoctlCommand.CONV1D: self._conv1d,
            IoctlCommand.CONV2D: self._conv2d,

            # Pooling
            IoctlCommand.MAX_POOL1D: self._max_pool1d,
            IoctlCommand.MAX_POOL2D: self._max_pool2d,
            IoctlCommand.AVG_POOL1D: self._avg_pool1d,
            IoctlCommand.AVG_POOL2D: self._avg_pool2d,

            # Normalization
            IoctlCommand.BATCH_NORM: self._batch_norm,
            IoctlCommand.LAYER_NORM: self._layer_norm,
            IoctlCommand.RMS_NORM: self._rms_norm,

            # Attention
            IoctlCommand.ATTENTION: self._attention,

            # Loss functions
            IoctlCommand.CROSS_ENTROPY: self._cross_entropy,
            IoctlCommand.MSE: self._mse,
            IoctlCommand.MAE: self._mae,

            # Optimizers
            IoctlCommand.SGD_STEP: self._sgd_step,
            IoctlCommand.ADAM_STEP: self._adam_step,

            # Utility
            IoctlCommand.CLIP_GRAD_NORM: self._clip_grad_norm,
            IoctlCommand.DROPOUT: self._dropout,
            IoctlCommand.EMBEDDING: self._embedding,
            IoctlCommand.LINEAR: self._linear,
        }

    @property
    def name(self) -> str:
        return self._name

    def info(self) -> dict:
        return {
            "name": self._name,
            "type": "tensor",
            "commands": len(self._ops),
        }

    # ── ioctl interface ───────────────────────────────────────────────────

    def ioctl(self, command: str | IoctlCommand, *args: Any) -> SyscallResult:
        """Clean ioctl interface — type-safe, documented."""
        try:
            if isinstance(command, str):
                try:
                    cmd = IoctlCommand(command)
                except ValueError:
                    return SyscallResult.fail(f"unknown command: {command}")
            else:
                cmd = command

            fn = self._ops.get(cmd)
            if fn is None:
                return SyscallResult.fail(f"command not implemented: {cmd.value}")

            result = fn(*args)
            return SyscallResult.ok(result)
        except Exception as e:
            return SyscallResult.fail(f"ioctl error: {e}")

    def list_commands(self) -> list[str]:
        """List all available commands."""
        return sorted([cmd.value for cmd in self._ops.keys()])

    # ── Function calls (direct use) ───────────────────────────────────────

    def matmul(self, a, b):
        """Matrix multiplication."""
        return self._to_arr(a) @ self._to_arr(b)

    def relu(self, a):
        """ReLU activation."""
        return np.maximum(0, self._to_arr(a))

    def softmax(self, a, axis=-1):
        """Softmax activation."""
        x = self._to_arr(a)
        shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def add(self, a, b):
        """Element-wise addition."""
        return self._to_arr(a) + self._to_arr(b)

    def mul(self, a, b):
        """Element-wise multiplication."""
        return self._to_arr(a) * self._to_arr(b)

    def linear(self, input, weight, bias=None):
        """Linear layer (matmul + bias)."""
        out = self.matmul(input, weight)
        if bias is not None:
            out = out + self._to_arr(bias)
        return out

    def embedding(self, input, weight):
        """Embedding lookup."""
        indices = self._to_arr(input).astype(int)
        w = self._to_arr(weight)
        return w[indices]

    def attention(self, query, key, value, mask=None):
        """Scaled dot-product attention."""
        q = self._to_arr(query)
        k = self._to_arr(key)
        v = self._to_arr(value)
        d_k = q.shape[-1]
        scores = np.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        weights = self.softmax(scores, axis=-1)
        return np.matmul(weights, v)

    def cross_entropy(self, input, target, ignore_index=-100):
        """Cross entropy loss."""
        logits = self._to_arr(input)
        targets = self._to_arr(target).astype(int)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True) + 1e-8)
        log_probs = shifted - log_sum_exp
        nll = -np.take_along_axis(log_probs, targets[..., np.newaxis], axis=-1).squeeze(-1)
        if ignore_index >= 0:
            mask = targets != ignore_index
            nll = nll * mask
            return np.sum(nll) / np.sum(mask)
        return np.mean(nll)

    def dropout(self, input, p=0.5, training=True):
        """Dropout."""
        if not training:
            return self._to_arr(input)
        x = self._to_arr(input)
        mask = np.random.binomial(1, 1 - p, size=x.shape) / (1 - p)
        return x * mask

    # ── Private methods (ioctl handlers) ──────────────────────────────────

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

    def _matmul(self, *args):
        return self.matmul(args[0], args[1])

    def _dot(self, *args):
        return np.dot(self._to_arr(args[0]), self._to_arr(args[1]))

    def _inv(self, *args):
        return np.linalg.inv(self._to_arr(args[0]))

    def _svd(self, *args):
        return np.linalg.svd(self._to_arr(args[0]))

    def _eig(self, *args):
        return np.linalg.eig(self._to_arr(args[0]))

    def _relu(self, *args):
        return self.relu(args[0])

    def _leaky_relu(self, *args):
        x = self._to_arr(args[0])
        alpha = args[1] if len(args) > 1 else 0.01
        return np.where(x > 0, x, alpha * x)

    def _sigmoid(self, *args):
        x = self._to_arr(args[0])
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _tanh(self, *args):
        return np.tanh(self._to_arr(args[0]))

    def _softmax(self, *args):
        return self.softmax(args[0])

    def _log_softmax(self, *args):
        return np.log(self.softmax(args[0]) + 1e-8)

    def _gelu(self, *args):
        x = self._to_arr(args[0])
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def _silu(self, *args):
        x = self._to_arr(args[0])
        return x * self._sigmoid(x)

    def _elu(self, *args):
        x = self._to_arr(args[0])
        alpha = args[1] if len(args) > 1 else 1.0
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def _selu(self, *args):
        x = self._to_arr(args[0])
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def _add(self, *args):
        return self.add(args[0], args[1])

    def _sub(self, *args):
        return self._to_arr(args[0]) - self._to_arr(args[1])

    def _mul(self, *args):
        return self.mul(args[0], args[1])

    def _div(self, *args):
        return self._to_arr(args[0]) / self._to_arr(args[1])

    def _neg(self, *args):
        return -self._to_arr(args[0])

    def _abs(self, *args):
        return np.abs(self._to_arr(args[0]))

    def _pow(self, *args):
        return np.power(self._to_arr(args[0]), args[1])

    def _sqrt(self, *args):
        return np.sqrt(self._to_arr(args[0]))

    def _exp(self, *args):
        return np.exp(self._to_arr(args[0]))

    def _log(self, *args):
        return np.log(self._to_arr(args[0]) + 1e-8)

    def _sum(self, *args):
        axis = args[1] if len(args) > 1 else None
        return np.sum(self._to_arr(args[0]), axis=axis)

    def _mean(self, *args):
        axis = args[1] if len(args) > 1 else None
        return np.mean(self._to_arr(args[0]), axis=axis)

    def _std(self, *args):
        axis = args[1] if len(args) > 1 else None
        return np.std(self._to_arr(args[0]), axis=axis)

    def _var(self, *args):
        axis = args[1] if len(args) > 1 else None
        return np.var(self._to_arr(args[0]), axis=axis)

    def _max(self, *args):
        axis = args[1] if len(args) > 1 else None
        return np.max(self._to_arr(args[0]), axis=axis)

    def _min(self, *args):
        axis = args[1] if len(args) > 1 else None
        return np.min(self._to_arr(args[0]), axis=axis)

    def _argmax(self, *args):
        axis = args[1] if len(args) > 1 else -1
        return np.argmax(self._to_arr(args[0]), axis=axis)

    def _argmin(self, *args):
        axis = args[1] if len(args) > 1 else -1
        return np.argmin(self._to_arr(args[0]), axis=axis)

    def _reshape(self, *args):
        return self._to_arr(args[0]).reshape(args[1])

    def _transpose(self, *args):
        axes = args[1] if len(args) > 1 else None
        return self._to_arr(args[0]).transpose(axes)

    def _flatten(self, *args):
        return self._to_arr(args[0]).flatten()

    def _squeeze(self, *args):
        axis = args[1] if len(args) > 1 else None
        return np.squeeze(self._to_arr(args[0]), axis=axis)

    def _unsqueeze(self, *args):
        axis = args[1] if len(args) > 1 else 0
        return np.expand_dims(self._to_arr(args[0]), axis=axis)

    def _cat(self, *args):
        arrays = args[0]
        axis = args[1] if len(args) > 1 else 0
        return np.concatenate([self._to_arr(a) for a in arrays], axis=axis)

    def _stack(self, *args):
        arrays = args[0]
        axis = args[1] if len(args) > 1 else 0
        return np.stack([self._to_arr(a) for a in arrays], axis=axis)

    def _conv1d(self, *args):
        input, weight = args[0], args[1]
        stride = args[2] if len(args) > 2 else 1
        padding = args[3] if len(args) > 3 else 0
        x = self._to_arr(input)
        w = self._to_arr(weight)
        batch, in_channels, seq_len = x.shape
        out_channels, _, kernel_size = w.shape
        out_seq = (seq_len + 2 * padding - kernel_size) // stride + 1
        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode='constant')
        cols = np.zeros((batch, in_channels, kernel_size, out_seq))
        for i in range(kernel_size):
            cols[:, :, i, :] = x[:, :, i:i + out_seq * stride:stride]
        cols = cols.transpose(0, 3, 1, 2).reshape(batch * out_seq, in_channels * kernel_size)
        w_flat = w.reshape(out_channels, -1)
        out = cols @ w_flat.T
        return out.reshape(batch, out_seq, out_channels).transpose(0, 2, 1)

    def _conv2d(self, *args):
        input, weight = args[0], args[1]
        stride = args[2] if len(args) > 2 else 1
        padding = args[3] if len(args) > 3 else 0
        x = self._to_arr(input)
        w = self._to_arr(weight)
        batch, in_channels, h, w_in = x.shape
        out_channels, _, kh, kw = w.shape
        h_out = (h + 2 * padding - kh) // stride + 1
        w_out = (w_in + 2 * padding - kw) // stride + 1
        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        cols = np.zeros((batch, in_channels, kh, kw, h_out, w_out))
        for i in range(kh):
            for j in range(kw):
                cols[:, :, i, j, :, :] = x[:, :, i:i + h_out * stride:stride, j:j + w_out * stride:stride]
        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(batch * h_out * w_out, in_channels * kh * kw)
        w_flat = w.reshape(out_channels, -1)
        out = cols @ w_flat.T
        return out.reshape(batch, h_out, w_out, out_channels).transpose(0, 3, 1, 2)

    def _max_pool1d(self, *args):
        input, kernel_size = args[0], args[1]
        stride = args[2] if len(args) > 2 else kernel_size
        padding = args[3] if len(args) > 3 else 0
        x = self._to_arr(input)
        batch, channels, seq_len = x.shape
        out_len = (seq_len + 2 * padding - kernel_size) // stride + 1
        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode='constant', constant_values=-np.inf)
        out = np.zeros((batch, channels, out_len))
        for i in range(out_len):
            start = i * stride
            out[:, :, i] = np.max(x[:, :, start:start + kernel_size], axis=2)
        return out

    def _max_pool2d(self, *args):
        input, kernel_size = args[0], args[1]
        stride = args[2] if len(args) > 2 else kernel_size
        padding = args[3] if len(args) > 3 else 0
        x = self._to_arr(input)
        batch, channels, h, w = x.shape
        h_out = (h + 2 * padding - kernel_size) // stride + 1
        w_out = (w + 2 * padding - kernel_size) // stride + 1
        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=-np.inf)
        out = np.zeros((batch, channels, h_out, w_out))
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * stride
                w_start = j * stride
                out[:, :, i, j] = np.max(x[:, :, h_start:h_start + kernel_size, w_start:w_start + kernel_size], axis=(2, 3))
        return out

    def _avg_pool1d(self, *args):
        input, kernel_size = args[0], args[1]
        stride = args[2] if len(args) > 2 else kernel_size
        padding = args[3] if len(args) > 3 else 0
        x = self._to_arr(input)
        batch, channels, seq_len = x.shape
        out_len = (seq_len + 2 * padding - kernel_size) // stride + 1
        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode='constant')
        out = np.zeros((batch, channels, out_len))
        for i in range(out_len):
            start = i * stride
            out[:, :, i] = np.mean(x[:, :, start:start + kernel_size], axis=2)
        return out

    def _avg_pool2d(self, *args):
        input, kernel_size = args[0], args[1]
        stride = args[2] if len(args) > 2 else kernel_size
        padding = args[3] if len(args) > 3 else 0
        x = self._to_arr(input)
        batch, channels, h, w = x.shape
        h_out = (h + 2 * padding - kernel_size) // stride + 1
        w_out = (w + 2 * padding - kernel_size) // stride + 1
        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        out = np.zeros((batch, channels, h_out, w_out))
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * stride
                w_start = j * stride
                out[:, :, i, j] = np.mean(x[:, :, h_start:h_start + kernel_size, w_start:w_start + kernel_size], axis=(2, 3))
        return out

    def _batch_norm(self, *args):
        input, weight, bias, mean, var = args[0], args[1], args[2], args[3], args[4]
        eps = args[5] if len(args) > 5 else 1e-5
        x = self._to_arr(input)
        w = self._to_arr(weight)
        b = self._to_arr(bias)
        m = self._to_arr(mean)
        v = self._to_arr(var)
        x_hat = (x - m) / np.sqrt(v + eps)
        return w * x_hat + b

    def _layer_norm(self, *args):
        input, weight, bias = args[0], args[1], args[2]
        eps = args[3] if len(args) > 3 else 1e-5
        x = self._to_arr(input)
        w = self._to_arr(weight)
        b = self._to_arr(bias)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + eps)
        return w * x_hat + b

    def _rms_norm(self, *args):
        input, weight = args[0], args[1]
        eps = args[2] if len(args) > 2 else 1e-6
        x = self._to_arr(input)
        w = self._to_arr(weight)
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        return x / rms * w

    def _attention(self, *args):
        query, key, value = args[0], args[1], args[2]
        mask = args[3] if len(args) > 3 else None
        return self.attention(query, key, value, mask)

    def _cross_entropy(self, *args):
        input, target = args[0], args[1]
        ignore_index = args[2] if len(args) > 2 else -100
        return self.cross_entropy(input, target, ignore_index)

    def _mse(self, *args):
        return np.mean((self._to_arr(args[0]) - self._to_arr(args[1])) ** 2)

    def _mae(self, *args):
        return np.mean(np.abs(self._to_arr(args[0]) - self._to_arr(args[1])))

    def _sgd_step(self, *args):
        params, grads, lr = args[0], args[1], args[2]
        momentum = args[3] if len(args) > 3 else 0.0
        state = args[4] if len(args) > 4 else None
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

    def _adam_step(self, *args):
        params, grads, lr = args[0], args[1], args[2]
        betas = args[3] if len(args) > 3 else (0.9, 0.999)
        eps = args[4] if len(args) > 4 else 1e-8
        state = args[5] if len(args) > 5 else None
        b1, b2 = betas
        updates = {}
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

    def _clip_grad_norm(self, *args):
        grads, max_norm = args[0], args[1]
        total_norm = 0.0
        for g in grads.values():
            arr = self._to_arr(g)
            total_norm += np.sum(arr ** 2)
        total_norm = np.sqrt(total_norm)
        clip_coef = max_norm / (total_norm + 1e-8)
        if clip_coef < 1:
            return {k: self._to_arr(v) * clip_coef for k, v in grads.items()}
        return dict(grads)

    def _dropout(self, *args):
        input = args[0]
        p = args[1] if len(args) > 1 else 0.5
        training = args[2] if len(args) > 2 else True
        return self.dropout(input, p, training)

    def _embedding(self, *args):
        return self.embedding(args[0], args[1])

    def _linear(self, *args):
        input, weight = args[0], args[1]
        bias = args[2] if len(args) > 2 else None
        return self.linear(input, weight, bias)
