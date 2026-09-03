"""
NPUDevice — flip-flops for neural processing.

Inherits DeviceDriver.
Only ioctl. No open/close.
Commands read/write registers (flip-flops).
"""

from __future__ import annotations

import time
import numpy as np
from typing import Any

from .baseboard import DeviceDriver, DeviceType
from .tensor_device import TensorDevice
from .ioctl import IoctlCommand
from .kernel_syscall import SyscallResult


class NPUDevice(DeviceDriver):
    """Neural processing flip-flops — holds model state.

    No open/close. Only ioctl.
    Commands read/write registers.
    Uses TensorDevice for computation.
    """

    def __init__(self, name: str = "npu"):
        super().__init__(name, DeviceType.INFERENCE)
        self._compute = TensorDevice()
        self._models: dict[str, Any] = {}
        self._default_model: str = ""

        self._ops = {
            IoctlCommand.INFO: self._info,
            IoctlCommand.LOAD: self._load,
            IoctlCommand.UNLOAD: self._unload,
            IoctlCommand.CALL: self._call,
            IoctlCommand.COMPUTE: self._compute_op,
        }

    def ioctl(self, command: str, *args: Any) -> SyscallResult:
        """Read/write flip-flops."""
        try:
            cmd = IoctlCommand(command)
            fn = self._ops.get(cmd)
            if fn is None:
                return SyscallResult.fail(f"not implemented: {command}")
            return fn(*args)
        except Exception as e:
            return SyscallResult.fail(f"ioctl error: {e}")

    # ── Flip-flop operations ──────────────────────────────────────────────

    def _info(self, *args) -> SyscallResult:
        return SyscallResult.ok({
            "device": self._name,
            "models": list(self._models.keys()),
            "default": self._default_model,
        })

    def _load(self, *args) -> SyscallResult:
        if len(args) < 1:
            return SyscallResult.fail("LOAD requires path")
        path = args[0]
        name = args[1] if len(args) > 1 else path.rsplit("/", 1)[-1]

        try:
            from domains.inference.slonet_provider import SlonetChatProvider
            provider = SlonetChatProvider.from_slnc(path, model_id=name)
            self._models[name] = provider
            if not self._default_model:
                self._default_model = name
            return SyscallResult.ok({"model": name})
        except Exception as e:
            return SyscallResult.fail(f"load failed: {e}")

    def _unload(self, *args) -> SyscallResult:
        if len(args) < 1:
            return SyscallResult.fail("UNLOAD requires name")
        name = args[0]
        if name in self._models:
            del self._models[name]
            if self._default_model == name:
                self._default_model = next(iter(self._models), "")
            return SyscallResult.ok({"unloaded": name})
        return SyscallResult.fail(f"model not found: {name}")

    def _call(self, *args) -> SyscallResult:
        if len(args) < 2:
            return SyscallResult.fail("CALL requires name and input")
        name = args[0]
        inp = args[1]
        provider = self._models.get(name or self._default_model)
        if provider is None:
            return SyscallResult.fail(f"model not loaded: {name}")

        try:
            if isinstance(inp, str):
                tokens = provider.tokenize(inp)
                ids = np.array([tokens], dtype=np.int64)
                gen = provider.generate_numpy(ids, max_new_tokens=100)
                result = provider.detokenize(gen[0].tolist())
                return SyscallResult.ok({"text": result})
            else:
                ids = np.array([inp], dtype=np.int64) if isinstance(inp, list) else inp
                logits = provider.forward_numpy(ids)
                return SyscallResult.ok({"logits": logits.tolist()})
        except Exception as e:
            return SyscallResult.fail(f"call failed: {e}")

    def _compute_op(self, *args) -> SyscallResult:
        if len(args) < 1:
            return SyscallResult.fail("COMPUTE requires op")
        op = args[0]
        op_args = args[1:]
        return self._compute.ioctl(op, *op_args)
