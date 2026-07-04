"""
Benchmark: ml_types (numpy) vs torch for dtype, device, and tensor operations.

Measures the overhead of torch imports and verifies that numpy-first
operations are at least as fast as torch equivalents.

Usage:
    python scripts/benchmark_ml_types.py
"""

import time
from typing import Dict, Tuple


def bench(label: str, fn, n: int = 1000) -> float:
    """Run fn n times and return average seconds per call."""
    start = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = time.perf_counter() - start
    return elapsed / n


def run_benchmarks() -> Dict[str, Tuple[float, float, str]]:
    """Run all benchmarks, return {name: (ml_time, torch_time, winner)}."""
    import numpy as np
    import sys
    sys.path.insert(0, "packages/core-py")
    from domains.infrastructure import ml_types as ml
    import torch

    results = {}

    # ── 1. Tensor creation ────────────────────────────────────────────
    def ml_zeros():
        ml.zeros((64, 128), dtype=ml.float32)

    def torch_zeros():
        torch.zeros((64, 128), dtype=torch.float32)

    ml_t = bench("zeros(64,128)", ml_zeros)
    torch_t = bench("zeros(64,128)", torch_zeros)
    winner = "ml_types" if ml_t <= torch_t else "torch"
    results["zeros"] = (ml_t, torch_t, winner)

    # ── 2. Matmul ─────────────────────────────────────────────────────
    a_ml = ml.tensor(np.random.randn(64, 128).astype(np.float32))
    b_ml = ml.tensor(np.random.randn(128, 64).astype(np.float32))
    a_t = torch.randn(64, 128, dtype=torch.float32)
    b_t = torch.randn(128, 64, dtype=torch.float32)

    def ml_matmul():
        ml.matmul(a_ml, b_ml)

    def torch_matmul():
        torch.matmul(a_t, b_t)

    ml_t = bench("matmul(64x128, 128x64)", ml_matmul)
    torch_t = bench("matmul(64x128, 128x64)", torch_matmul)
    winner = "ml_types" if ml_t <= torch_t else "torch"
    results["matmul"] = (ml_t, torch_t, winner)

    # ── 3. Softmax ────────────────────────────────────────────────────
    x_ml = ml.tensor(np.random.randn(1, 1000).astype(np.float32))
    x_t = torch.randn(1, 1000, dtype=torch.float32)

    def ml_softmax():
        ml.softmax(x_ml, dim=-1)

    def torch_softmax():
        torch.nn.functional.softmax(x_t, dim=-1)

    ml_t = bench("softmax(1000)", ml_softmax)
    torch_t = bench("softmax(1000)", torch_softmax)
    winner = "ml_types" if ml_t <= torch_t else "torch"
    results["softmax"] = (ml_t, torch_t, winner)

    # ── 4. isnan/isinf ────────────────────────────────────────────────
    arr_ml = ml.tensor(np.random.randn(10000).astype(np.float32))
    arr_t = torch.randn(10000, dtype=torch.float32)

    def ml_isnan():
        ml.isnan(arr_ml)

    def torch_isnan():
        torch.isnan(arr_t)

    ml_t = bench("isnan(10k)", ml_isnan)
    torch_t = bench("isnan(10k)", torch_isnan)
    winner = "ml_types" if ml_t <= torch_t else "torch"
    results["isnan"] = (ml_t, torch_t, winner)

    # ── 5. Device detection ───────────────────────────────────────────
    def ml_auto():
        ml.auto_device()

    def torch_device():
        if torch.cuda.is_available():
            return "cuda"
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    ml_t = bench("auto_device()", ml_auto, n=10000)
    torch_t = bench("device detection", torch_device, n=10000)
    winner = "ml_types" if ml_t <= torch_t else "torch"
    results["device_detect"] = (ml_t, torch_t, winner)

    # ── 6. cat (concat) ───────────────────────────────────────────────
    arrs_ml = [ml.tensor(np.random.randn(32).astype(np.float32)) for _ in range(10)]
    arrs_t = [torch.randn(32) for _ in range(10)]

    def ml_cat():
        ml.cat(arrs_ml)

    def torch_cat():
        torch.cat(arrs_t)

    ml_t = bench("cat(10x32)", ml_cat)
    torch_t = bench("cat(10x32)", torch_cat)
    winner = "ml_types" if ml_t <= torch_t else "torch"
    results["cat"] = (ml_t, torch_t, winner)

    # ── 7. where ──────────────────────────────────────────────────────
    cond_ml = ml.tensor(np.random.randn(1000).astype(np.float32) > 0)
    x_w_ml = ml.tensor(np.random.randn(1000).astype(np.float32))
    y_w_ml = ml.tensor(np.random.randn(1000).astype(np.float32))
    cond_t = torch.randn(1000) > 0
    x_w_t = torch.randn(1000)
    y_w_t = torch.randn(1000)

    def ml_where():
        ml.where(cond_ml, x_w_ml, y_w_ml)

    def torch_where():
        torch.where(cond_t, x_w_t, y_w_t)

    ml_t = bench("where(1000)", ml_where)
    torch_t = bench("where(1000)", torch_where)
    winner = "ml_types" if ml_t <= torch_t else "torch"
    results["where"] = (ml_t, torch_t, winner)

    return results


def main():
    print("=" * 70)
    print("ml_types (numpy) vs torch benchmark")
    print("=" * 70)

    results = run_benchmarks()

    print(f"\n{'Operation':<28} {'ml_types (μs)':<15} {'torch (μs)':<15} {'winner':<10} {'speedup'}")
    print("-" * 70)

    for name, (ml_t, torch_t, winner) in results.items():
        ml_us = ml_t * 1e6
        torch_us = torch_t * 1e6
        if ml_t > 0 and torch_t > 0:
            speedup = torch_t / ml_t if ml_t < torch_t else -(ml_t / torch_t)
        else:
            speedup = 0
        print(f"{name:<28} {ml_us:<15.1f} {torch_us:<15.1f} {winner:<10} {speedup:+.1f}x")

    print("\n" + "=" * 70)
    ml_wins = sum(1 for _, (_, _, w) in results.items() if w == "ml_types")
    torch_wins = sum(1 for _, (_, _, w) in results.items() if w == "torch")
    print(f"Result: ml_types wins {ml_wins}/{len(results)}, torch wins {torch_wins}/{len(results)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
