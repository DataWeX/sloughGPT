#!/usr/bin/env python3
"""Tensor kernel benchmark — measure the NPU tensor ops, pick routing winners.

Card d74e2cef phase 2 (benchmark-first). Measures per-(op, shape) cost of
TensorDevice ioctl ops at representative model-stack shapes, then — once
native routing lands — A/B's it against the native engine:

    --engine numpy   force the numpy oracle (SLO_TENSOR_NATIVE=0)
    --engine native  force native routing   (SLO_TENSOR_NATIVE=1)
    --engine both    side-by-side with a ratio (default; falls back to
                     numpy-only automatically while routing is unlanded,
                     decided from TensorDevice.caps() — never lies)

Candidates are chosen for STRUCTURAL headroom (numpy does multi-pass with
temporaries: softmax/layernorm/rmsnorm/gelu) plus matmul as the BLAS
control (expected to lose — the harness must be honest) and attention as
a measured-but-unrouted entry for the future fused-attention card.

Timing: per-call timer overhead would drown microsecond ops, so each
sample batches K calls (batch >= ~2 ms) and p50/p95 come from 40 samples.

Run:
    PYTHONNOUSERSITE=1 python scripts/benchmark_tensor_kernels.py
    PYTHONNOUSERSITE=1 python scripts/benchmark_tensor_kernels.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

# Pin thread budgets BEFORE numpy loads (openblas reads its env at
# library init). A shared box with parallel lanes inflates every number
# if both sides spawn 8 threads into contention — measured here: BLAS
# collapsed from 0.28 ms to 14 ms per 128x128 matmul under load. Default
# 1 = honest single-core kernel-vs-kernel; export SLO_BENCH_THREADS=8 on
# an idle box for production-shaped numbers (both sides same budget).
_BUDGET = os.environ.setdefault("SLO_BENCH_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", _BUDGET)
os.environ.setdefault("OMP_NUM_THREADS", _BUDGET)
os.environ.setdefault("MKL_NUM_THREADS", _BUDGET)

import numpy as np  # noqa: E402 — env pinned above, must precede this load

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from domain.shell._internal.tensor_device import TensorDevice  # noqa: E402

SAMPLES = 40
TARGET_BATCH_S = 0.002  # every sample times >= this many seconds of work
WARMUP = 3


# (op label, ioctl command, shape label, arg factory)
def _cases() -> list[tuple[str, str, str, Callable[[], tuple]]]:
    rng = np.random.default_rng(7)

    def x(r: int, c: int) -> np.ndarray:
        return rng.standard_normal((r, c))

    w512 = rng.standard_normal(512)
    b512 = rng.standard_normal(512)
    att = rng.standard_normal((2, 512, 64))

    def mm(n: int):
        a = rng.standard_normal((n, n))
        b = rng.standard_normal((n, n))
        return a, b

    cases: list[tuple[str, str, str, Callable[[], tuple]]] = [
        # structural-headroom candidates: multi-pass + temporaries in numpy
        ("softmax", "SOFTMAX", "512x512", lambda: (x(512, 512),)),
        ("softmax", "SOFTMAX", "8192x512", lambda: (x(8192, 512),)),
        ("layernorm", "LAYER_NORM", "4096x512", lambda: (x(4096, 512), w512, b512)),
        ("rmsnorm", "RMS_NORM", "4096x512", lambda: (x(4096, 512), w512)),
        ("gelu", "GELU", "4M flat", lambda: (rng.standard_normal(4_194_304),)),
        ("relu", "RELU", "4M flat", lambda: (rng.standard_normal(4_194_304),)),
        # BLAS control: expected numpy win on CPU — keeps the harness honest
        ("matmul", "MATMUL", "128x128", lambda: mm(128)),
        ("matmul", "MATMUL", "512x512", lambda: mm(512)),
        ("matmul", "MATMUL", "1024x1024", lambda: mm(1024)),
        # measured, NOT routed in v1 (fused attention = future work)
        ("attention", "ATTENTION", "b2 s512 d64", lambda: (att, att, att)),
    ]
    return cases


def _percentile(samples: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(samples), q))


def _time_op(td: TensorDevice, cmd: str, args: tuple) -> tuple[float, float]:
    """Returns (p50_s, p95_s) per call — batched samples, 40 rounds."""
    td.ioctl(cmd, *args)  # warm
    for _ in range(WARMUP):
        td.ioctl(cmd, *args)
    t1 = time.perf_counter()
    td.ioctl(cmd, *args)
    one = max(time.perf_counter() - t1, 1e-9)
    k = max(1, int(TARGET_BATCH_S / one))
    samples = []
    for _ in range(SAMPLES):
        t0 = time.perf_counter()
        for _ in range(k):
            td.ioctl(cmd, *args)
        samples.append((time.perf_counter() - t0) / k)
    return _percentile(samples, 50), _percentile(samples, 95)


def _set_engine(engine: str) -> None:
    # Read dynamically by the device per call (SLO_CYCLES_NATIVE pattern),
    # so toggling here works without a reload.
    os.environ["SLO_TENSOR_NATIVE"] = "0" if engine == "numpy" else "1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=["numpy", "native", "both"], default="both")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    td = TensorDevice("tensor")
    caps = td.caps()
    routed = caps["backend"]["engine"] != "numpy"
    if args.engine == "native" and not routed:
        print("error: native routing not landed (caps engine=numpy) — use --engine numpy")
        return 2

    engines = [args.engine]
    if args.engine == "both":
        engines = ["native", "numpy"] if routed else ["numpy"]
        if not routed:
            print("note: native routing not landed (caps engine=numpy) — numpy baseline only\n")

    cases = _cases()
    routed_ops = set(caps["backend"]["native_ops"])
    rows = []
    for label, cmd, shape, make in cases:
        case_args = make()
        entry: dict = {"op": label, "command": cmd, "shape": shape}
        # unrouted ops (RELU 0.97x, matmul control, attention future)
        # time ONCE on numpy — a second identical run would fake a ratio
        case_engines = [e for e in engines if e == "numpy" or cmd in routed_ops]
        for eng in case_engines:
            _set_engine(eng)
            p50, p95 = _time_op(td, cmd, case_args)
            entry[f"{eng}_p50_us"] = round(p50 * 1e6, 3)
            entry[f"{eng}_p95_us"] = round(p95 * 1e6, 3)
        if len(case_engines) == 2:
            entry["speedup"] = round(entry["numpy_p50_us"] / entry["native_p50_us"], 3)
        rows.append(entry)

    if args.json:
        print(
            json.dumps(
                {
                    "caps_engine": caps["backend"]["engine"],
                    "threads": int(_BUDGET),
                    "results": rows,
                },
                indent=2,
            )
        )
        return 0

    print(f"threads: {_BUDGET} (OPENBLAS/OMP/MKL pinned)\n")
    header = f"{'op':<10} {'shape':<14} {'numpy p50 us':>12} {'numpy p95 us':>12}"
    if len(engines) == 2:
        header += f" {'native p50 us':>14} {'speedup':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = (
            f"{r['op']:<10} {r['shape']:<14} {r['numpy_p50_us']:>12.3f} {r['numpy_p95_us']:>12.3f}"
        )
        if "native_p50_us" in r:
            line += f" {r['native_p50_us']:>14.3f} {r['speedup']:>8.3f}"
        elif len(engines) == 2:
            line += f" {'- (unrouted)':>14} {'-':>8}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
