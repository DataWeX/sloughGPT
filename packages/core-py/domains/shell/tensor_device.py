"""
TensorDevice — flip-flops for compute.

Inherits DeviceDriver.
Only ioctl. No open/close.
Commands read/write registers (flip-flops).
"""

from __future__ import annotations

import numpy as np
from typing import Any

from .baseboard import DeviceDriver, DeviceType
from .ioctl import IoctlCommand
from .kernel_syscall import SyscallResult


class TensorDevice(DeviceDriver):
    """Compute flip-flops — holds tensor state.

    No open/close. Only ioctl.
    Commands read/write registers.
    """

    def __init__(self, name: str = "tensor"):
        super().__init__(name, DeviceType.INFERENCE)
        self._ops = {
            IoctlCommand.MATMUL: self._matmul,
            IoctlCommand.RELU: self._relu,
            IoctlCommand.SOFTMAX: self._softmax,
            IoctlCommand.ADD: self._add,
            IoctlCommand.MUL: self._mul,
            IoctlCommand.LINEAR: self._linear,
            IoctlCommand.EMBEDDING: self._embedding,
            IoctlCommand.ATTENTION: self._attention,
            IoctlCommand.CROSS_ENTROPY: self._cross_entropy,
            IoctlCommand.DROPOUT: self._dropout,
        }

    def ioctl(self, command: str, *args: Any) -> SyscallResult:
        """Read/write flip-flops."""
        try:
            cmd = IoctlCommand(command)
            fn = self._ops.get(cmd)
            if fn is None:
                return SyscallResult.fail(f"not implemented: {command}")
            result = fn(*args)
            return SyscallResult.ok(result)
        except Exception as e:
            return SyscallResult.fail(f"ioctl error: {e}")

    # ── Flip-flop operations ──────────────────────────────────────────────

    def _matmul(self, *args):
        a = np.array(args[0], dtype=np.float64)
        b = np.array(args[1], dtype=np.float64)
        return (a @ b).tolist()

    def _relu(self, *args):
        a = np.array(args[0], dtype=np.float64)
        return np.maximum(0, a).tolist()

    def _softmax(self, *args):
        a = np.array(args[0], dtype=np.float64)
        shifted = a - np.max(a)
        exp_a = np.exp(shifted)
        return (exp_a / np.sum(exp_a)).tolist()

    def _add(self, *args):
        a = np.array(args[0], dtype=np.float64)
        b = np.array(args[1], dtype=np.float64)
        return (a + b).tolist()

    def _mul(self, *args):
        a = np.array(args[0], dtype=np.float64)
        b = np.array(args[1], dtype=np.float64)
        return (a * b).tolist()

    def _linear(self, *args):
        x = np.array(args[0], dtype=np.float64)
        w = np.array(args[1], dtype=np.float64)
        bias = np.array(args[2], dtype=np.float64) if len(args) > 2 else None
        out = x @ w.T
        if bias is not None:
            out = out + bias
        return out.tolist()

    def _embedding(self, *args):
        indices = np.array(args[0], dtype=np.int64)
        weights = np.array(args[1], dtype=np.float64)
        return weights[indices].tolist()

    def _attention(self, *args):
        q = np.array(args[0], dtype=np.float64)
        k = np.array(args[1], dtype=np.float64)
        v = np.array(args[2], dtype=np.float64)
        d_k = q.shape[-1]
        scores = np.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        return np.matmul(weights, v).tolist()

    def _cross_entropy(self, *args):
        logits = np.array(args[0], dtype=np.float64)
        targets = np.array(args[1], dtype=np.int64)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True) + 1e-8)
        log_probs = shifted - log_sum_exp
        nll = -np.take_along_axis(log_probs, targets[..., np.newaxis], axis=-1).squeeze(-1)
        return float(np.mean(nll))

    def _dropout(self, *args):
        x = np.array(args[0], dtype=np.float64)
        p = float(args[1]) if len(args) > 1 else 0.5
        training = bool(args[2]) if len(args) > 2 else True
        if not training:
            return x.tolist()
        mask = np.random.binomial(1, 1 - p, size=x.shape) / (1 - p)
        return (x * mask).tolist()
