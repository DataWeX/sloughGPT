"""Kernel Ops — the f32 contract every caller sees (layer 1 of 3).

engine = policy · ops = contract · core = capability. This module is the
STABLE layer: callers (training, inference) hold nothing but these
functions, so the engine (autotune policy) and the cores (implementations)
are free to change — or disappear — without a single call site noticing.
No device vocabulary appears above this line: selection over cores is
the engine's private business.

Contract:
  * f32 in, f32 out (the dtype gate — training computes float32; the
    float64 shell path is not this contract's problem).
  * residency passes through unchanged (device in -> device out); today's
    cores all return host arrays, the residency build adds device cores —
    the signature fact is declared here, implemented there.
  * outputs match the numpy ORACLE within the tolerance contract
    (tests/core-py/test_kernel_engine.py — elementwise rtol=1e-5,
    reductions rtol=1e-4, matmul looser for k-term accumulation).
  * deterministic: one core chosen per (op, shape) per process — the
    engine memoizes, so run-to-run equality holds (generation stability).

``oracle`` is exposed for tests and for callers that want the referee
directly (benchmarks, parity harnesses) — it is kernel_core.ORACLE.
"""

from __future__ import annotations

import logging
import types

import numpy as np

from domain.slolib._internal import kernel_core, kernel_engine

logger = logging.getLogger("slo.kernel.ops")

#: Reference implementations (the referee) — re-exported, not redefined.
# Attribute access (kernel_ops.oracle.layernorm) over the same dict.
oracle = types.SimpleNamespace(**kernel_core.ORACLE)

#: The op vocabulary — one declaration, shared with core and engine.
OPS = kernel_core.OPS

EPS = kernel_core.EPS


def _dispatch(op: str, *args, eps: float = EPS) -> np.ndarray:
    """Pick a core for this (op, shape), run it, uphold the f32 contract.

    Fallback is layered and never raises for non-default cores: a core
    that vanishes between pick and run, or fails mid-call, degrades to
    the numpy oracle AND logs a warning (silent degradation must leave a
    trace — job_c87a2dd6). A failure in the oracle itself propagates:
    correctness bugs must not be papered over.
    """
    shape = np.shape(args[0]) if args else ()
    name = kernel_engine.pick(op, shape)
    core = kernel_core.get(name)
    try:
        if core is None or not (core.available() and core.supports(op)):
            raise LookupError(f"core {name!r} cannot run {op}")
        out = core.run(op, *args, eps=eps)
    except Exception as exc:
        # Silent degradation must leave a trace (job_c87a2dd6 signature):
        # a core that fails at RUN time is a real event, not a shrug.
        if name != kernel_engine.DEFAULT_CORE:
            logger.warning("core %s failed %s, falling back to numpy: %s", name, op, exc)
            fallback = kernel_core.get(kernel_engine.DEFAULT_CORE)
            if fallback is None:  # numpy core is registered by construction
                raise
            out = fallback.run(op, *args, eps=eps)
        else:
            raise
    return np.asarray(out, dtype=np.float32)


def matmul(a, b) -> np.ndarray:
    """f32 matmul — BLAS owns the CPU path (numpy core), GPU core optional."""
    return _dispatch("matmul", a, b)


def softmax(x) -> np.ndarray:
    """Softmax over the last axis."""
    return _dispatch("softmax", x)


def layernorm(x, w, b, eps: float = EPS) -> np.ndarray:
    """LayerNorm over the last axis with weight and bias."""
    return _dispatch("layernorm", x, w, b, eps=eps)


def rmsnorm(x, w, eps: float = EPS) -> np.ndarray:
    """RMSNorm over the last axis with weight (no bias)."""
    return _dispatch("rmsnorm", x, w, eps=eps)


def gelu(x) -> np.ndarray:
    """tanh-approximate GELU (the formulation training uses)."""
    return _dispatch("gelu", x)
