"""Benchmark the int8 GEMM kernels against each other and against fp32.

Compares three execution paths for the same MxNxK shapes:
  * "2"   -- 256-bit AVX2 int8 kernel (MAN_QUANT_KERNEL=2)
  * "512" -- 512-bit AVX-512 VNNI int8 kernel (MAN_QUANT_KERNEL=512)
  * "1024" -- float32 numpy BLAS (the K=1024 crossover baseline that adaptive
              quantization keeps small layers on)

Each mode runs in a fresh process with MAN_QUANT_KERNEL set before the C
library loads, so the measured path is the one requested. Timing uses warmup +
median-of-N to filter noise.

Run:
    python scripts/benchmark_quant_kernels.py 2
    python scripts/benchmark_quant_kernels.py 512
    python scripts/benchmark_quant_kernels.py 1024

Each prints one JSON line per shape:
{"mode","M","N","K","median_ms","gflop"}
"""
import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

# ``domains`` lives at packages/core-py; add that to the import path.
_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))


def _timeit(fn, runs=7, warmup=2):
    """Warm up, then return the median wall time of ``runs`` executions."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = perf_counter()
        fn()
        samples.append(perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def _bench_mode(mode):
    # Kernel choice is decided in C at first use and cached in the loaded .so,
    # so set the knob before importing the wrapper.
    if mode in ("2", "512"):
        os.environ["MAN_QUANT_KERNEL"] = mode

    matmul_int8_c = None
    if mode in ("2", "512"):
        from domains.infrastructure.quant_core import wrapper
        from domains.infrastructure.quant_core.wrapper import HAS_AVX2, HAS_AVX512

        matmul_int8_c = wrapper.matmul_int8_c
        if mode == "512" and not HAS_AVX512:
            sys.exit(f"mode 512 requested but AVX-512 kernel not active (HAS_AVX512={HAS_AVX512})")
        if mode == "2" and not HAS_AVX2:
            sys.exit("mode 2 requested but AVX2 kernel not available")
    print(f"mode={mode} HAS_AVX2={locals().get('HAS_AVX2', None)} "
          f"HAS_AVX512={locals().get('HAS_AVX512', None)}", file=sys.stderr)

    grid = [
        (m, n, k)
        for m in (1, 8, 64)
        for k in (256, 512, 768, 1024, 1536, 2048)
        for n in (256, 1024, 4096)
    ]
    rng = np.random.default_rng(7)
    results = []
    for m, n, k in grid:
        flops = 2.0 * m * n * k
        if mode in ("2", "512"):
            a = rng.integers(-128, 128, size=(m, k), dtype=np.int8)
            b = rng.integers(-128, 128, size=(n, k), dtype=np.int8)

            def run(a=a, b=b):
                return matmul_int8_c(a, b)
        else:
            a = rng.standard_normal((m, k)).astype(np.float32)
            b = rng.standard_normal((n, k)).astype(np.float32)

            def run(a=a, b=b):
                return a @ b.T
        med = _timeit(run)
        results.append({
            "mode": mode, "M": m, "N": n, "K": k,
            "median_ms": round(med * 1e3, 4),
            "gflop": round((flops / 1e9) / med if med > 0 else 0.0, 3),
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["2", "512", "1024"])
    args = ap.parse_args()
    for row in _bench_mode(args.mode):
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
