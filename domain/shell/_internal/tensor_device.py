"""
TensorDevice — standalone compute hardware.

Pure compute engine with clean ioctl interface.
No inheritance. No decorators. Just a device.
"""

from __future__ import annotations

import ctypes
import logging
import os
from typing import Any

import numpy as np

from .ioctl import IoctlCommand
from .kernel_syscall import SyscallResult

logger = logging.getLogger("slo.tensor.native")

# ── native engine — ONE C source, two engines (card d74e2cef phase 2) ──────
#
# tensor_kernels.c is compiled by domain/slolib/_internal/native.py
# (gcc+OpenMP ctypes / NVRTC RawKernel) and routed through the ioctl
# handlers below behind SLO_TENSOR_NATIVE (default on; =0 forces the
# numpy oracle, read dynamically — same contract as SLO_CYCLES_NATIVE).
# Everything the kernel does not take (float32, 0-d, 3-D softmax,
# strided shapes the guards reject) keeps the numpy path byte-identical.
# matmul is deliberately NOT routed: BLAS owns it (the benchmark keeps
# it as the control).

_NATIVE_ENTRIES: dict[str, tuple[str, str]] = {
    "SOFTMAX": ("tens_softmax_cpu", "tens_softmax_gpu"),
    "LAYER_NORM": ("tens_layernorm_cpu", "tens_layernorm_gpu"),
    "RMS_NORM": ("tens_rmsnorm_cpu", "tens_rmsnorm_gpu"),
    "GELU": ("tens_gelu_cpu", "tens_gelu_gpu"),
    "RELU": ("tens_relu_cpu", "tens_relu_gpu"),
}
# Benchmark-gated routing set (scripts/benchmark_tensor_kernels.py,
# threads=1, recorded on card d74e2cef): softmax 2.7x, layernorm 3.6x,
# rmsnorm 1.5x, gelu 1.5x — all ship. RELU stays in _NATIVE_ENTRIES
# (the C kernel ships and is parity-tested on both engines) but is NOT
# routed: np.maximum(0, x) is already a single-pass SIMD ufunc, 0.97x
# measured = no headroom, so the branch does not ship. matmul never
# enters the entries at all — BLAS owns it (the benchmark's control).
_ROUTED_OPS = frozenset({"GELU", "LAYER_NORM", "RMS_NORM", "SOFTMAX"})
_NATIVE_SRC: str | None = None
_NATIVE_ENGINES: dict[str, Any] = {}


def _native_disabled() -> bool:
    """SLO_TENSOR_NATIVE=0 forces the numpy oracle path.

    Read dynamically (not at import) so tests/CLIs can toggle at
    runtime without a module reload.
    """
    return os.environ.get("SLO_TENSOR_NATIVE", "1") == "0"


def _build_engines(cmd: str):
    """Compiled engines for this op's entries, or None. Capability
    probe (no env check) — caches successes AND failures; one source
    means a successful CPU compile carries every entry."""
    global _NATIVE_SRC
    if cmd in _NATIVE_ENGINES:
        return _NATIVE_ENGINES[cmd]
    eng = None
    try:
        from pathlib import Path

        from domain.slolib._internal.native import NativeEngines

        if _NATIVE_SRC is None:
            _NATIVE_SRC = (Path(__file__).parent / "tensor_kernels.c").read_text()
        cpu_entry, gpu_entry = _NATIVE_ENTRIES[cmd]
        cand = NativeEngines(_NATIVE_SRC, cpu_entry, gpu_entry)
        eng = cand if cand.any else None
    except Exception as exc:  # noqa: BLE001 — any loader failure = numpy oracle
        logger.info("native tensor unavailable (%s); numpy oracle path", exc)
        eng = None
    _NATIVE_ENGINES[cmd] = eng
    return eng


def _engines(cmd: str):
    """Runtime gate for the handlers: disabled → None WITHOUT caching,
    so re-enabling works immediately."""
    if _native_disabled():
        return None
    return _build_engines(cmd)


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.c_void_p)


def _run(fn, argtypes: list, *args) -> None:
    fn.argtypes = argtypes
    fn.restype = None
    fn(*args)


def _row_op(cmd: str, x: Any):
    """(contig_x, out, rows, cols, cpu_fn) for a routable float64
    array of ndim 1-2, else None (caller keeps the numpy path)."""
    if not isinstance(x, np.ndarray) or x.dtype != np.float64:
        return None
    if x.ndim < 1 or x.ndim > 2 or x.size == 0:
        return None
    eng = _engines(cmd)
    if eng is None or eng.cpu is None:
        return None
    fn = getattr(eng.cpu, _NATIVE_ENTRIES[cmd][0], None)  # symbol, not the CDLL
    if fn is None:
        return None
    xc = np.ascontiguousarray(x)
    out = np.empty(x.shape, dtype=np.float64)
    rows = int(x.shape[0]) if x.ndim == 2 else 1
    cols = int(x.shape[-1])
    return xc, out, rows, cols, fn


def _flat_op(cmd: str, x: Any):
    """(contig_flat_x, out_flat, n, cpu_fn) for a routable float64
    array, else None."""
    if not isinstance(x, np.ndarray) or x.dtype != np.float64 or x.size == 0:
        return None
    eng = _engines(cmd)
    if eng is None or eng.cpu is None:
        return None
    fn = getattr(eng.cpu, _NATIVE_ENTRIES[cmd][0], None)  # symbol, not the CDLL
    if fn is None:
        return None
    xc = np.ascontiguousarray(x).reshape(-1)
    out = np.empty(x.size, dtype=np.float64)
    return xc, out, int(x.size), fn


class TensorDevice:
    """Standalone compute hardware — wraps numpy.

    Has clean ioctl interface for assembly.
    Has function calls for direct use.
    """

    def __init__(self, name: str = "tensor"):
        self._name = name
        self._ops = {
            # Contract
            IoctlCommand.CAPS: self._caps,
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

    def call(self, method: str, *args: Any) -> Any:
        """VM Device interface — delegates to ioctl."""
        result = self.ioctl(method, *args)
        if result.success:
            return result.value
        raise Exception(result.error)

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

    def caps(self) -> dict:
        """Capability matrix — the device's machine-readable contract.

        ONE device, three faces (the CPU's tensor backend via
        ``vm._default_tensor``, the kernel NPUDevice's ``_compute``, the
        runtime DeviceManager's registered ``"tensor"``) — each reaches
        this through the same ``ioctl()``/``call()`` path, so the matrix
        is declared once here and never per-face.

        backend/parity report reality, never intent: the probe compiles
        tensor_kernels.c (cached on disk by the loader) and the fields
        fill only if it actually compiled. The kill-switch is a runtime
        concern and does not change what the machine CAN do, so this
        capability view ignores it.
        """
        native = _build_engines("SOFTMAX") is not None  # one source — any entry proves it
        native_ops = sorted(_ROUTED_OPS) if native else []
        return {
            "name": self._name,
            "type": "tensor",
            "ops": self.list_commands(),
            # _to_arr coerces scalar/list inputs to float64; ndarray inputs
            # pass through opaquely, so float64 is the one guaranteed
            # computation dtype.
            "dtypes": ["float64"],
            "backend": {
                "engine": "native-cpu" if native_ops else "numpy",
                "native_available": bool(native_ops),
                "native_ops": native_ops,
            },
            "parity": list(native_ops),
        }

    def _caps(self, *args) -> dict:
        """Ioctl handler — extra args ignored (DEV_CALL may pass junk)."""
        return self.caps()

    # ── Function calls (direct use) ───────────────────────────────────────

    def matmul(self, a, b):
        """Matrix multiplication."""
        return self._to_arr(a) @ self._to_arr(b)

    def relu(self, a):
        """ReLU activation — deliberately NOT routed (benchmark: 0.97x,
        np.maximum is already single-pass; see _ROUTED_OPS)."""
        return np.maximum(0, self._to_arr(a))

    def softmax(self, a, axis=-1):
        """Softmax activation.

        Native rows (axis=-1, <=2-D) and the global form (axis=None,
        the VM SOFTMAX opcode — one flattened row) route through
        tensor_kernels.c; anything else (3-D attention scores, non-
        float64, odd axes) keeps the numpy path.
        """
        x = self._to_arr(a)
        if axis == -1:
            got = _row_op("SOFTMAX", x)
            if got is not None:
                xc, out, rows, cols, fn = got
                _run(
                    fn,
                    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_long],
                    _ptr(xc),
                    _ptr(out),
                    rows,
                    cols,
                )
                return out
        elif axis is None and isinstance(x, np.ndarray):
            got = _row_op("SOFTMAX", np.ravel(x))
            if got is not None:
                xc, out, rows, cols, fn = got
                _run(
                    fn,
                    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_long],
                    _ptr(xc),
                    _ptr(out),
                    rows,
                    cols,
                )
                return out.reshape(x.shape)
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
        # Transpose last two axes; for 3D (batch, seq, features) use (0, 2, 1)
        if k.ndim >= 3:
            axes = list(range(k.ndim))
            axes[-2], axes[-1] = axes[-1], axes[-2]
            k_t = k.transpose(axes)
        else:
            k_t = k.T
        scores = np.matmul(q, k_t) / np.sqrt(d_k)
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
        got = _flat_op("GELU", x)
        if got is not None:
            xc, out, n, fn = got
            _run(fn, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long], _ptr(xc), _ptr(out), n)
            return out.reshape(x.shape)
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
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode="constant")
        cols = np.zeros((batch, in_channels, kernel_size, out_seq))
        for i in range(kernel_size):
            cols[:, :, i, :] = x[:, :, i : i + out_seq * stride : stride]
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
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
        cols = np.zeros((batch, in_channels, kh, kw, h_out, w_out))
        for i in range(kh):
            for j in range(kw):
                cols[:, :, i, j, :, :] = x[
                    :, :, i : i + h_out * stride : stride, j : j + w_out * stride : stride
                ]
        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(
            batch * h_out * w_out, in_channels * kh * kw
        )
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
            x = np.pad(
                x, ((0, 0), (0, 0), (padding, padding)), mode="constant", constant_values=-np.inf
            )
        out = np.zeros((batch, channels, out_len))
        for i in range(out_len):
            start = i * stride
            out[:, :, i] = np.max(x[:, :, start : start + kernel_size], axis=2)
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
            x = np.pad(
                x,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                mode="constant",
                constant_values=-np.inf,
            )
        out = np.zeros((batch, channels, h_out, w_out))
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * stride
                w_start = j * stride
                out[:, :, i, j] = np.max(
                    x[:, :, h_start : h_start + kernel_size, w_start : w_start + kernel_size],
                    axis=(2, 3),
                )
        return out

    def _avg_pool1d(self, *args):
        input, kernel_size = args[0], args[1]
        stride = args[2] if len(args) > 2 else kernel_size
        padding = args[3] if len(args) > 3 else 0
        x = self._to_arr(input)
        batch, channels, seq_len = x.shape
        out_len = (seq_len + 2 * padding - kernel_size) // stride + 1
        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode="constant")
        out = np.zeros((batch, channels, out_len))
        for i in range(out_len):
            start = i * stride
            out[:, :, i] = np.mean(x[:, :, start : start + kernel_size], axis=2)
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
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
        out = np.zeros((batch, channels, h_out, w_out))
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * stride
                w_start = j * stride
                out[:, :, i, j] = np.mean(
                    x[:, :, h_start : h_start + kernel_size, w_start : w_start + kernel_size],
                    axis=(2, 3),
                )
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
        routed = self._affine_row("LAYER_NORM", x, w, b)
        if routed is not None:
            xc, wc, bc, out, rows, cols, fn = routed
            _run(
                fn,
                [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_long,
                    ctypes.c_long,
                    ctypes.c_double,
                ],
                _ptr(xc),
                _ptr(wc),
                _ptr(bc),
                _ptr(out),
                rows,
                cols,
                float(eps),
            )
            return out
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + eps)
        return w * x_hat + b

    def _rms_norm(self, *args):
        input, weight = args[0], args[1]
        eps = args[2] if len(args) > 2 else 1e-6
        x = self._to_arr(input)
        w = self._to_arr(weight)
        routed = self._affine_row("RMS_NORM", x, w, None)
        if routed is not None:
            xc, wc, _bc, out, rows, cols, fn = routed
            _run(
                fn,
                [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_long,
                    ctypes.c_long,
                    ctypes.c_double,
                ],
                _ptr(xc),
                _ptr(wc),
                _ptr(out),
                rows,
                cols,
                float(eps),
            )
            return out
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
        return x / rms * w

    @staticmethod
    def _affine_row(cmd: str, x: Any, w: Any, b: Any):
        """Row-op prep + weight/bias shape gate; None = numpy path.

        w must be float64 of shape (cols,) — anything else (None,
        lists, broadcastable (1, cols) rows) stays on numpy, which
        accepts what the C kernel does not.
        """
        base = _row_op(cmd, x)
        if base is None:
            return None
        xc, out, rows, cols, fn = base
        if not isinstance(w, np.ndarray) or w.dtype != np.float64 or w.shape != (cols,):
            return None
        if b is not None and (
            not isinstance(b, np.ndarray) or b.dtype != np.float64 or b.shape != (cols,)
        ):
            return None
        return xc, np.ascontiguousarray(w), b, out, rows, cols, fn

    # ── VM-instruction tensor ops (the board CPU delegates these here) ──────
    # Explicit signatures because the CPU calls them directly with fixed arity.
    # Operands arrive already parsed by the CPU, so they are consumed as-is —
    # this mirrors the expressions that used to live inline in the CPU shims.

    def _norm(self, x):
        """L2 norm (VM NORM)."""
        return np.linalg.norm(x)

    def _shape(self, x):
        """Tensor shape as a list (VM SHAPE)."""
        return list(x.shape)

    def _size(self, x):
        """Element count (VM SIZE)."""
        return int(x.size)

    def _layernorm_global(self, x):
        """Plain weightless layernorm, global reduction (VM LAYERNORM).

        Distinct from ``_layer_norm`` (affine: weight/bias, axis=-1) — this is
        the statistical, axis-free normalization the VM opcode specifies.
        """
        if isinstance(x, np.ndarray):
            got = _row_op("LAYER_NORM", np.ravel(x))
            if got is not None:
                xc, out, rows, cols, fn = got
                _run(
                    fn,
                    [
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_long,
                        ctypes.c_long,
                        ctypes.c_double,
                    ],
                    _ptr(xc),
                    None,  # NULL w/b = the weightless form in the C kernel
                    None,
                    _ptr(out),
                    rows,
                    cols,
                    1e-5,
                )
                return out.reshape(x.shape)
        return (x - np.mean(x)) / np.sqrt(np.var(x) + 1e-5)

    def _rmsnorm_global(self, x):
        """Plain weightless rmsnorm, global reduction (VM RMSNORM).

        Distinct from ``_rms_norm`` (affine: weight, axis=-1).
        """
        if isinstance(x, np.ndarray):
            got = _row_op("RMS_NORM", np.ravel(x))
            if got is not None:
                xc, out, rows, cols, fn = got
                _run(
                    fn,
                    [
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_long,
                        ctypes.c_long,
                        ctypes.c_double,
                    ],
                    _ptr(xc),
                    None,  # NULL w = the weightless form in the C kernel
                    _ptr(out),
                    rows,
                    cols,
                    1e-5,
                )
                return out.reshape(x.shape)
        return x / np.sqrt(np.mean(x**2) + 1e-5)

    def _randn(self, rows, cols):
        """Standard-normal tensor (VM RANDN)."""
        return np.random.randn(rows, cols)

    def _randunif(self, rows, cols, lo, hi):
        """Uniform tensor (VM RANDUNIF)."""
        return np.random.uniform(lo, hi, (rows, cols))

    def _safe_div(self, a, b):
        """Division with inf/nan -> 0 (VM DIV: the guard is part of the contract)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            result = self._div(a, b)
            result = np.where(np.isinf(result), 0.0, result)
            result = np.where(np.isnan(result), 0.0, result)
        return result

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
                v = b2 * state.get(f"{name}_v", np.zeros_like(g)) + (1 - b2) * g**2
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
            total_norm += np.sum(arr**2)
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
