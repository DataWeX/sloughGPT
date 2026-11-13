#!/usr/bin/env python3
"""Kernel Engine benchmark — dispatch overhead + per-core ratios at training shapes.

What it measures (the two claims the engine must keep):

1. DISPATCH OVERHEAD: kernel_ops.<op>() vs the raw numpy oracle called
   directly. The engine's pick() is memoized, so the seam must cost
   ~nothing per call — this is the small-chattiness disease (numba/CuPy/
   OpenMP each measured fast as callees and lost as conversations); the
   engine itself must not become a fifth chatty layer. Pass criterion:
   overhead is a small single-digit % at training shapes (reported, not
   asserted — ratio runs under contention are for trends).

2. PER-CORE RATIOS: every available core vs numpy (the oracle) for each
   (op, shape), interleaved and rotated (drift cancellation), p50. The
   numbers that feed the engine's tuning cache decisions.

Usage:
    scripts/benchmark_kernel_engine.py                # both sections
    scripts/benchmark_kernel_engine.py --json out.json
    SLO_BENCH_THREADS=8 scripts/benchmark_kernel_engine.py   # prod budget

Threads pinned BEFORE numpy loads (same convention as
benchmark_tensor_kernels.py: a shared box with parallel lanes inflates
every number if both sides spawn 8 threads into contention). Ratio
measurements are robust under load; absolute µs are not — check
/proc/loadavg before quoting absolutes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_BUDGET = os.environ.setdefault("SLO_BENCH_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", _BUDGET)
os.environ.setdefault("OMP_NUM_THREADS", _BUDGET)
os.environ.setdefault("MKL_NUM_THREADS", _BUDGET)

import numpy as np  # noqa: E402 — env pinned above, must precede this load

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from domain.slolib._internal import kernel_core, kernel_engine, kernel_ops  # noqa: E402

# gate (1024x64 / 4096x32) and medium (4096x128 / 16384x128) training shapes
CASES = [
    ("layernorm", (1024, 64)),
    ("layernorm", (4096, 128)),
    ("rmsnorm", (1024, 64)),
    ("rmsnorm", (4096, 128)),
    ("softmax", (4096, 32)),
    ("softmax", (16384, 128)),
    ("gelu", (1024 * 256,)),
    ("gelu", (4096 * 512,)),
    ("matmul", (1024, 64)),
    ("matmul", (4096, 128)),
]

SAMPLES = 20
TARGET_S = 0.002  # every sample times >= this many seconds of work
ROUNDS = 3  # rotated interleaving: round r starts at arm r


def _inputs(op: str, shape: tuple, seed: int = 7):
    rng = np.random.default_rng(seed)
    if op == "matmul":
        a = rng.standard_normal(shape).astype(np.float32)
        b = rng.standard_normal((shape[-1], 32)).astype(np.float32)
        return (a, b)
    x = rng.standard_normal(shape).astype(np.float32)
    if op in ("gelu", "softmax"):
        return (x,)
    w = rng.standard_normal(shape[-1]).astype(np.float32)
    if op == "layernorm":
        b = rng.standard_normal(shape[-1]).astype(np.float32)
        return (x, w, b)
    return (x, w)


def _timeit(fn) -> float:
    """p50 µs per call — warm-up, then batched samples (amortizes the clock)."""
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    fn()
    one = max(time.perf_counter() - t0, 1e-9)
    k = max(1, int(TARGET_S / one))
    samples = []
    for _ in range(SAMPLES):
        t0 = time.perf_counter()
        for _ in range(k):
            fn()
        samples.append((time.perf_counter() - t0) / k)
    return float(np.median(samples)) * 1e6  # µs


def section_dispatch() -> list[dict]:
    """ops contract vs raw oracle — what the seam costs per call."""
    print("\n── dispatch overhead: kernel_ops vs raw numpy oracle ──")
    print(f"{'op':<10} {'shape':<16} {'ops p50 us':>11} {'raw p50 us':>11} {'overhead':>9}")
    rows = []
    for op, shape in CASES:
        args = _inputs(op, shape)
        raw = getattr(kernel_ops.oracle, op)
        ops_fn = getattr(kernel_ops, op)
        arms = [lambda: raw(*args), lambda: ops_fn(*args)]
        samples: list[list[float]] = [[], []]
        for r in range(ROUNDS):
            for i, order in enumerate([range(2), range(1, -1)][r % 2]):
                arm = arms[order]
                samples[order].append(_timeit(arm))
        ops_p50 = float(np.median(samples[1]))
        raw_p50 = float(np.median(samples[0]))
        ratio = ops_p50 / raw_p50 if raw_p50 > 0 else float("inf")
        rows.append(
            {"op": op, "shape": list(shape), "ops_us": ops_p50, "raw_us": raw_p50, "ratio": ratio}
        )
        print(f"{op:<10} {str(shape):<16} {ops_p50:>11.1f} {raw_p50:>11.1f} {ratio:>8.2f}x")
    return rows


def section_cores() -> list[dict]:
    """Every available core vs numpy, rotated interleaved, p50."""
    print("\n── per-core ratios vs numpy oracle (rotated, p50) ──")
    print(f"{'op':<10} {'shape':<16} {'core':<8} {'p50 us':>10} {'vs numpy':>9}")
    rows = []
    for op, shape in CASES:
        args = _inputs(op, shape)
        oracle_fn = getattr(kernel_ops.oracle, op)
        arms = {"numpy": lambda: oracle_fn(*args)}
        for name in kernel_core.names():
            if name == "numpy":
                continue
            core = kernel_core.get(name)
            if core is None or not core.available() or not core.supports(op):
                continue
            arms[name] = lambda c=core: c.run(op, *args)
        names = list(arms)
        samples: dict[str, list[float]] = {n: [] for n in names}
        for r in range(ROUNDS):
            order = names[r:] + names[:r]
            for n in order:
                samples[n].append(_timeit(arms[n]))
        base = float(np.median(samples["numpy"]))
        for n in names:
            p50 = float(np.median(samples[n]))
            ratio = base / p50 if p50 > 0 else float("inf")
            rows.append({"op": op, "shape": list(shape), "core": n, "us": p50, "speedup": ratio})
            print(f"{op:<10} {str(shape):<16} {n:<8} {p50:>10.1f} {ratio:>8.2f}x")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", help="write machine-readable results here")
    args = ap.parse_args()

    load = open("/proc/loadavg").read().split()[0]
    print(f"threads={_BUDGET} (OPENBLAS/OMP/MKL pinned)  load={load}")
    avail = [
        f"{n}={'y' if (c := kernel_core.get(n)) and c.available() else 'n'}"
        for n in kernel_core.names()
    ]
    print(f"cores: {' '.join(avail)}")

    dispatch = section_dispatch()
    cores = section_cores()

    if args.json:
        out = {
            "threads": int(_BUDGET),
            "loadavg": load,
            "fingerprint": kernel_engine.fingerprint(),
            "dispatch": dispatch,
            "cores": cores,
        }
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
