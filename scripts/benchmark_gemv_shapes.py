"""Per-shape GEMV microbench — where the decode matmul budget goes.

Companion to profile_decode.py (card 20261024_111415): the profile says
matmul_int8_f32_c is 84.6% of decode time but lumps all five projection
shapes into one entry. This times each shape in isolation — exactly the
fused quantize→GEMM→dequant path the decode loop calls — across
MAN_GEMM_THREADS configs, and reports ms, GB/s of weight traffic, and the
share of the ~389MB/token weight stream.

Designed for an always-loaded box: it measures at whatever load is present
and records loadavg alongside, instead of waiting for a quiet window.

Shapes = Qwen2.5-0.5B (hidden 896, ffn 4864, vocab 151936, 24 layers,
kv dim 128).

Usage:
    python scripts/benchmark_gemv_shapes.py
    python scripts/benchmark_gemv_shapes.py --threads 1,2,4 --reps 7 --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

# (name, N, K, calls_per_token) — B is int8 (N, K) row-major.
SHAPES = [
    ("qkv", 1152, 896, 24),
    ("wo", 896, 896, 24),
    ("w13", 4864, 896, 24),
    ("w2", 896, 4864, 24),
    ("lm_head", 151936, 896, 1),
]


def _bench_shape(name: str, n: int, k: int, reps: int) -> dict:
    import numpy as np

    from domain.infrastructure._internal.quant_core import wrapper as qw

    a = np.random.randn(1, k).astype(np.float32)
    b = np.random.randint(-127, 128, size=(n, k), dtype=np.int8)
    b_scale = np.ones(n, dtype=np.float32)

    qw.matmul_int8_f32_c(a[:1, :k], b, b_scale)  # warmup + dispatch
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = qw.matmul_int8_f32_c(a, b, b_scale)
        times.append(time.perf_counter() - t0)
    assert out is not None and out.shape == (1, n)

    ms = statistics.median(times) * 1000
    bytes_moved = n * k  # int8 weights streamed per call
    return {
        "name": name,
        "n": n,
        "k": k,
        "ms": round(ms, 3),
        "gb_s": round(bytes_moved / 1e9 / (ms / 1000), 2),
        "mb_per_call": round(bytes_moved / 1e6, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", default="1,2,4", help="MAN_GEMM_THREADS values to sweep")
    ap.add_argument("--reps", type=int, default=7, help="timed reps per shape (median)")
    ap.add_argument("--json", default=None, metavar="PATH")
    args = ap.parse_args()

    rows = []
    for th in [int(x) for x in args.threads.split(",")]:
        os.environ["MAN_GEMM_THREADS"] = str(th)
        print(
            f"\n── MAN_GEMM_THREADS={th}  loadavg="
            f"{' '.join(f'{v:.1f}' for v in os.getloadavg())} ──"
        )
        print(f"  {'shape':<10} {'ms':>9} {'GB/s':>7} {'MB/call':>9} {'×tok':>5}")
        for name, n, k, calls in SHAPES:
            r = _bench_shape(name, n, k, args.reps)
            r["threads"] = th
            r["calls_per_token"] = calls
            rows.append(r)
            print(
                f"  {name:<10} {r['ms']:9.2f} {r['gb_s']:7.2f} {r['mb_per_call']:9.1f} {calls:5d}"
            )

        # Serialized per-token total for this thread config.
        total_ms = sum(r["ms"] * r["calls_per_token"] for r in rows if r["threads"] == th)
        total_mb = sum(r["mb_per_call"] * r["calls_per_token"] for r in rows if r["threads"] == th)
        print(
            f"  → matmul/token {total_ms:.1f}ms  ({total_mb:.0f}MB streamed)  "
            f"matmul-bound ceiling {1000 / total_ms:.1f} tok/s"
        )

    os.environ.pop("MAN_GEMM_THREADS", None)
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {"loadavg": list(os.getloadavg()), "reps": args.reps, "rows": rows},
                indent=2,
            )
        )
        print(f"\njson → {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
