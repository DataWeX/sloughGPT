#!/usr/bin/env python3
"""Benchmark int8 GEMM thread scaling across batch M, including the default policy.

Why this exists
---------------
`_gemm_threads()` in ``quant_core/matmul_int8.c`` used to return 1 unconditionally,
with a measured rationale taken on **decode** (M=1): warm weights already sit near
DRAM bandwidth, extra hyperthreads add memory-controller contention and push the
i5-9300H into thermal throttle (measured 124-130ms bimodal at 4 threads vs a
stable 54-67ms serial).

**Prefill is a different regime** and that verdict was never re-measured there:

* B is reused across all M rows, so it stays cache-resident and DRAM traffic is
  a fraction of decode's — which is what caused the contention,
* the work is M times larger, so thread spawn cost is negligible,
* M is hundreds-to-thousands.

The same file also gated threading on ``N*K >= 6MB`` — M never entered — so a
601-token prefill of a 0.8MB weight block took the serial branch while the
single-token decode of that same block (identical N*K) was eligible.

This script answers three things:

1. does threading help prefill, and at what thread count,
2. is threaded output **bit-identical** to serial (required before shipping),
3. what does the shipped default policy do (``default`` = MAN_GEMM_THREADS
   unset -> decode serial, prefill online-CPUs-capped-8).

Reports one row per shape x M with wall time per spec and speedup vs serial.

Usage:
    .venv/bin/python3 scripts/benchmark_gemm_threading.py
    .venv/bin/python3 scripts/benchmark_gemm_threading.py --quick
    .venv/bin/python3 scripts/benchmark_gemm_threading.py --threads 1,4,8,default
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from domains.infrastructure.quant_core.wrapper import matmul_int8_c  # noqa: E402

# Qwen2.5-0.5B (the model the chat server serves): hidden 896,
# intermediate 4864, vocab 151936, 2 kv-heads x 64 head_dim.
# (name, N, K) — N*K in bytes is what the old gate compared against 6MB.
SHAPES = [
    ("qkv_proj", 1152, 896),  #  1.03 MB
    ("attn_out", 896, 896),  #  0.80 MB
    ("gate_up", 4864, 896),  #  4.36 MB
    ("down_proj", 896, 4864),  #  4.36 MB
    ("lm_head", 151936, 896),  # 136.13 MB
]

# M=1 is decode (the regime the old serial verdict was measured on);
# the rest are prefill sizes — 601 is an observed prompt length.
BATCHES = [1, 64, 256, 601, 1024]


def _apply(spec: str) -> None:
    if spec == "default":
        os.environ.pop("MAN_GEMM_THREADS", None)
    else:
        os.environ["MAN_GEMM_THREADS"] = spec


def _time_one(A: np.ndarray, B: np.ndarray, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        matmul_int8_c(A, B)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _calibrate(A: np.ndarray, B: np.ndarray, budget: float) -> int:
    t = _time_one(A, B, 1)
    if t <= 0:
        return 3
    return max(3, min(40, int(budget / t)))


def run(specs: list[str], quick: bool) -> int:
    batches = BATCHES[:3] if quick else BATCHES
    shapes = SHAPES[:4] if quick else SHAPES
    budget = 0.12 if quick else 0.30

    print(f"host: {_host_summary()}")
    print(f"specs: {specs}   shapes: {len(shapes)}   batches: {batches}")
    print(f"budget/config: {budget * 1000:.0f}ms (median of N timed calls)\n")

    widths = max(len(s) for s in specs)
    header = (
        f"{'shape':<10} {'N*K':>9} {'M':>5} {'MB':>7} "
        + " ".join(f"{('t=' + s):>{widths + 4}}" for s in specs)
        + f" {'best':>7} {'parity':>7}"
    )
    print(header)
    print("-" * len(header))

    parity_ok = True
    for name, N, K in shapes:
        B = np.ascontiguousarray(
            np.random.default_rng(0).integers(-127, 128, size=(N, K), dtype=np.int8)
        )
        mk = (N * K) / 1e6
        for M in batches:
            A = np.ascontiguousarray(
                np.random.default_rng(1).integers(-128, 127, size=(M, K), dtype=np.int8)
            )
            repeats = _calibrate(A, B, budget)

            times: dict[str, float] = {}
            ref = None
            for spec in specs:
                _apply(spec)
                C = matmul_int8_c(A, B)
                if ref is None:
                    ref = C
                elif not np.array_equal(ref, C):
                    parity_ok = False
                times[spec] = _time_one(A, B, repeats)

            t1 = times.get("1") or times[specs[0]]
            best = min(times, key=lambda s: times[s])
            cells = " ".join(f"{times[s] * 1e3:>{widths + 4}.2f}ms" for s in specs)
            print(
                f"{name:<10} {N * K:>9} {M:>5} {mk:>7.2f} {cells} "
                f"{t1 / times[best]:>6.2f}x {'ok' if parity_ok else 'FAIL':>7}"
            )

    os.environ.pop("MAN_GEMM_THREADS", None)
    print(f"\nparity vs {'serial':}: {'bit-identical at every spec' if parity_ok else 'MISMATCH'}")
    return 0 if parity_ok else 1


def _host_summary() -> str:
    try:
        flags = Path("/proc/cpuinfo").read_text().split("flags", 1)[1].split("\n", 1)[0]
    except Exception:
        return "unknown host"
    return f"{os.cpu_count()} logical cores, AVX2={'avx2' in flags} AVX512={'avx512' in flags}"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--threads",
        default="1,4,8,default",
        help="comma-separated MAN_GEMM_THREADS values; 'default' = unset (shipped policy)",
    )
    p.add_argument("--quick", action="store_true", help="fewer shapes/batches (~3x faster)")
    a = p.parse_args()
    specs = [x.strip() for x in a.threads.split(",") if x.strip()]
    if "1" not in specs:
        specs.insert(0, "1")
    return run(specs, a.quick)


if __name__ == "__main__":
    raise SystemExit(main())
