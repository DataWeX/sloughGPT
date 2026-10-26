"""Crossover probe: is parallel=True still a win at SMALL shapes?

The main probe (probe_parallel_kernels.py) showed 4-5x on realistic shapes,
but numba's parallel layer carries a per-call fork/join cost. Decode at the
START of a generation has a tiny KV cache (new_len 1..64) and prefill of a
short prompt is small too — exactly where the thread-team overhead could eat
the gain. Measure the crossover before deciding whether the shipped kernels
need a size gate.

Usage:
    python -u scripts/probe_parallel_small.py
"""

from __future__ import annotations

import os
import time

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("SLO_OPENBLAS_NUM_THREADS", "1"))
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("SLO_OMP_NUM_THREADS", "1"))

from probe_parallel_kernels import (  # noqa: E402
    parallel_attn_multi,
    parallel_attn_single,
    serial_attn_multi,
    serial_attn_single,
)


def _time(fn, args, repeat):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main():
    import sys

    rep = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rng = np.random.default_rng(1)
    H, E = 14, 64
    scale = np.float32(1.0 / np.sqrt(E))

    print(
        f"repeat={rep} (min-of-N; machine load ~{os.getloadavg()[1]:.1f} on {os.cpu_count()} cores)"
    )
    print(f"{'case':28} {'serial us':>11} {'par us':>10} {'speedup':>8}")
    print("-" * 60)

    # --- decode: tiny KV cache (start of generation) --------------------------
    for new_len in (1, 4, 8, 16, 32, 64, 128):
        q = rng.standard_normal((H, E)).astype(np.float32)
        k = rng.standard_normal((H, new_len, E)).astype(np.float32)
        v = rng.standard_normal((H, new_len, E)).astype(np.float32)
        o_s = np.zeros((H, E), dtype=np.float32)
        o_p = np.zeros((H, E), dtype=np.float32)
        s = _time(serial_attn_single, (q, k, v, o_s, scale, H, E, 1, new_len), rep)
        p = _time(parallel_attn_single, (q, k, v, o_p, scale, H, E, 1, new_len), rep)
        print(f"{'decode kv=' + str(new_len):28} {s * 1e3:11.1f} {p * 1e3:10.1f} {s / p:7.2f}x")

    # --- prefill: short prompt ------------------------------------------------
    print()
    for seq_len in (1, 2, 4, 8, 16, 32):
        q = rng.standard_normal((seq_len, H, E)).astype(np.float32)
        k = rng.standard_normal((H, seq_len, E)).astype(np.float32)
        v = rng.standard_normal((H, seq_len, E)).astype(np.float32)
        o_s = np.zeros((seq_len, H, E), dtype=np.float32)
        o_p = np.zeros((seq_len, H, E), dtype=np.float32)
        s = _time(serial_attn_multi, (q, k, v, o_s, scale, H, E, seq_len, seq_len), rep)
        p = _time(parallel_attn_multi, (q, k, v, o_p, scale, H, E, seq_len, seq_len), rep)
        print(f"{'prefill seq=' + str(seq_len):28} {s * 1e3:11.1f} {p * 1e3:10.1f} {s / p:7.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
