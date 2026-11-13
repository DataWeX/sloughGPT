"""Kernel Core — the implementations (the plural) of the kernel ops.

Layer 3 of the Kernel Engine stack (engine = policy, ops = contract,
core = capability). A core is a STATELESS function over the arrays it is
handed: ownership follows the constructor (only the model engine owns
weights), custody moves freely, and a core never holds, caches, or
mutates state it did not create. The ``device`` field declares WHERE a
core can execute — capability, never custody.

The ``numpy`` core is the oracle: every other core is parity-gated
against it in tests/core-py/test_kernel_engine.py (elementwise
rtol=1e-5/atol=1e-6, reductions rtol=1e-4/atol=1e-5, matmul looser for
k-term accumulation — derivations in that file).

Registered cores:
  numpy  — always available; BLAS matmul + ufunc chains (the referee)
  numba  — opt-in (SLO_USE_NUMBA=1, per commit b70a0d085: serial kernels
           forfeit multithreaded BLAS, so the flag stays off by default)
  c32    — shell's tensor_kernels.c ported f32 (double->float), compiled
           once via slolib native.py; 4 ops, no matmul (BLAS owns it)
  gpu    — the slolib accelerator backend (CuPy); compute on device,
           returns host arrays (residency pass-through lands with the
           residency build)

Selection among these is the ENGINE's job, not this module's: cores
declare, the engine picks, ops dispatches.
"""

from __future__ import annotations

import ctypes
import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

logger = logging.getLogger("slo.kernel.core")

#: The op vocabulary — shared with kernel_ops (the contract) and
#: kernel_engine (the selection key). Adding an op is an ops+core edit,
#: never a call-site edit.
OPS: tuple[str, ...] = ("matmul", "softmax", "layernorm", "rmsnorm", "gelu")

EPS = 1e-5


# ── numpy oracle semantics (the referee) ─────────────────────────────────────


def _np_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.matmul(a, b)  # type: ignore[no-any-return]


def _np_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)  # type: ignore[no-any-return]


def _np_layernorm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = EPS):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mu) / np.sqrt(var + eps)) * w + b


def _np_rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = EPS):
    return x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps) * w


def _np_gelu(x: np.ndarray) -> np.ndarray:
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))  # type: ignore[no-any-return]


#: Per-op oracle implementations — kernel_ops.oracle is a view of this.
ORACLE: dict[str, Callable[..., np.ndarray]] = {
    "matmul": _np_matmul,
    "softmax": _np_softmax,
    "layernorm": _np_layernorm,
    "rmsnorm": _np_rmsnorm,
    "gelu": _np_gelu,
}


def _split(op: str, args: tuple, eps: float):
    """Canonical arg layout per op: layernorm/rmsnorm carry eps last."""
    if op == "layernorm":
        x, w, b = args[0], args[1], args[2]
        return (x, w, b, eps)
    if op == "rmsnorm":
        return (args[0], args[1], eps)
    return args


class _Core:
    """Base: name, capability declaration, op support, stateless run."""

    name: str = "base"
    device: str = "host"  # where it CAN execute: host | device | both

    def available(self) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def supports(self, op: str) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def run(self, op: str, *args, eps: float = EPS) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover


class NumpyCore(_Core):
    """The oracle: always available, BLAS for matmul, ufuncs for the rest."""

    name = "numpy"
    device = "host"

    def available(self) -> bool:
        return True

    def supports(self, op: str) -> bool:
        return op in ORACLE

    def run(self, op: str, *args, eps: float = EPS) -> np.ndarray:
        return ORACLE[op](*_split(op, args, eps))


class NumbaCore(_Core):
    """slonet_kernels' njit forms — OPT-IN only (repo decision b70a0d085).

    Serial kernels measured 1.8x slower end-to-end on this box (they
    forfeit multithreaded BLAS), so availability is gated on
    SLO_USE_NUMBA: the engine must not silently re-select what the repo
    deliberately made a flag. nb_softmax mutates in place — this core
    copies first (cores never mutate their inputs).
    """

    name = "numba"
    device = "host"

    _SUPPORT = ("softmax", "layernorm", "rmsnorm")
    _ok: bool | None = None

    def available(self) -> bool:
        if os.environ.get("SLO_USE_NUMBA", "0").strip().lower() not in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return False
        if self._ok is None:
            try:
                from domain.training._internal import slonet_kernels  # noqa: F401

                self._ok = True
            except Exception as exc:  # noqa: BLE001 - capability probe
                logger.info("numba core unavailable (%s)", exc)
                self._ok = False
        return self._ok

    def supports(self, op: str) -> bool:
        return op in self._SUPPORT

    def run(self, op: str, *args, eps: float = EPS) -> np.ndarray:
        from domain.training._internal import slonet_kernels as sk

        if op == "softmax":
            # nb_softmax writes through its argument — copy keeps the
            # no-mutation ownership contract (custody passes through).
            return np.asarray(sk.nb_softmax(args[0].copy()))
        if op == "layernorm":
            return np.asarray(sk.nb_layernorm(args[0], args[1], args[2], np.float32(eps)))
        return np.asarray(sk.nb_rmsnorm(args[0], args[1], np.float32(eps)))


class C32Core(_Core):
    """shell's fused row kernels, ported f32, compiled once via native.py.

    tensor_kernels.c is f64-only (shell's dtype gate at
    tensor_device.py:107); training computes f32, so this core applies
    the mechanical double->float port (the same transform the probe
    validated) and compiles it through the shared native.py loader —
    ONE source, cached by hash, no recompile per call.

    No matmul by design: BLAS owns it (the benchmark's control).
    """

    name = "c32"
    device = "host"

    _SUPPORT = ("softmax", "layernorm", "rmsnorm", "gelu")
    _ok: bool | None = None
    _fns: dict = None  # type: ignore[assignment]

    def _src(self) -> str:
        shell_c = Path(__file__).resolve().parents[2] / "shell" / "_internal" / "tensor_kernels.c"
        return shell_c.read_text().replace("double", "float")

    def available(self) -> bool:
        if self._ok is None:
            self._ok = False
            try:
                from domain.slolib._internal.native import compile_cpu

                src = self._src()
                entries = {
                    "softmax": "tens_softmax_cpu",
                    "layernorm": "tens_layernorm_cpu",
                    "rmsnorm": "tens_rmsnorm_cpu",
                    "gelu": "tens_gelu_cpu",
                }
                cdll = compile_cpu(src, next(iter(entries.values())))
                if cdll is None:
                    raise RuntimeError("cpu compile failed")
                self._fns = {op: getattr(cdll, sym) for op, sym in entries.items()}
                for op, fn in self._fns.items():
                    if op == "softmax":
                        fn.argtypes = [
                            ctypes.c_void_p,
                            ctypes.c_void_p,
                            ctypes.c_long,
                            ctypes.c_long,
                        ]
                    elif op == "gelu":
                        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
                    elif op == "layernorm":
                        fn.argtypes = [
                            ctypes.c_void_p,
                            ctypes.c_void_p,
                            ctypes.c_void_p,
                            ctypes.c_void_p,
                            ctypes.c_long,
                            ctypes.c_long,
                            ctypes.c_float,
                        ]
                    else:
                        fn.argtypes = [
                            ctypes.c_void_p,
                            ctypes.c_void_p,
                            ctypes.c_void_p,
                            ctypes.c_long,
                            ctypes.c_long,
                            ctypes.c_float,
                        ]
                    fn.restype = None
                self._ok = True
            except Exception as exc:  # noqa: BLE001 - capability probe
                logger.info("c32 core unavailable (%s)", exc)
                self._ok = False
        return self._ok

    def supports(self, op: str) -> bool:
        return op in self._SUPPORT

    def run(self, op: str, *args, eps: float = EPS) -> np.ndarray:
        fn = self._fns[op]
        p = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
        if op == "softmax":
            x: np.ndarray = args[0]
            out = np.empty_like(x)
            fn(p(x), p(out), x.shape[0], x.shape[-1])
            return out
        if op == "gelu":
            x = args[0]
            out = np.empty_like(x)
            fn(p(x), p(out), x.size)
            return out
        x, w = args[0], args[1]
        out = np.empty_like(x)
        rows, cols = x.shape[0], x.shape[-1]
        if op == "layernorm":
            fn(p(x), p(w), p(args[2]), p(out), rows, cols, np.float32(eps))
        else:
            fn(p(x), p(w), p(out), rows, cols, np.float32(eps))
        return out


class GpuCore(_Core):
    """The slolib accelerator backend (CuPy) — compute on device.

    Capability declaration: executes ON device. Custody today still
    round-trips (the measured naive arm); residency pass-through is the
    residency build's job, not this core's — a core never arranges
    custody, it only states where it can run.
    """

    name = "gpu"
    device = "device"

    _SUPPORT = OPS
    _acc = None
    _ok: bool | None = None

    def available(self) -> bool:
        if self._ok is None:
            try:
                from domain.slolib._internal.gpu import get_accelerator

                acc = get_accelerator()
                self._ok = acc is not None and acc.device_type != "cpu"
                if self._ok:
                    self._acc = acc
            except Exception as exc:  # noqa: BLE001 - capability probe
                logger.info("gpu core unavailable (%s)", exc)
                self._ok = False
        return self._ok

    def supports(self, op: str) -> bool:
        return op in self._SUPPORT

    def run(self, op: str, *args, eps: float = EPS) -> np.ndarray:
        acc = self._acc
        if acc is None:  # available() gates every run — belt and braces
            raise RuntimeError("gpu core has no accelerator")
        if op == "matmul":
            return acc.matmul(args[0], args[1])
        if op == "softmax":
            return acc.softmax(args[0], axis=-1)
        if op == "layernorm":
            return acc.layer_norm(args[0], args[1], args[2], eps)
        if op == "rmsnorm":
            return acc.rms_norm(args[0], args[1], eps)
        return acc.gelu(args[0])


# ── registry ────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, _Core] = {}


def _build() -> dict[str, _Core]:
    if not _REGISTRY:
        for cls in (NumpyCore, NumbaCore, C32Core, GpuCore):
            core = cls()
            _REGISTRY[core.name] = core
    return _REGISTRY


def names() -> list[str]:
    """Registered core names (build order: numpy first — the oracle)."""
    return list(_build())


def get(name: str) -> _Core | None:
    """The core registered under ``name``, or None (never raises)."""
    return _build().get(name)
