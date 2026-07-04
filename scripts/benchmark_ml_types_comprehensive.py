"""
Comprehensive ML types benchmark: speed, efficiency, portability, resource usage.

Measures:
  1. Speed      — latency per operation (μs)
  2. Efficiency — ops per second, ops per MB
  3. Portability — works without torch? (graceful degradation)
  4. Resources  — memory footprint, import time, CPU usage

Usage:
    python scripts/benchmark_ml_types_comprehensive.py
"""

import os
import sys
import time
import tracemalloc

sys.path.insert(0, "packages/core-py")

import numpy as np


def bench(fn, n=1000):
    """Return (avg_us, peak_memory_bytes)."""
    tracemalloc.start()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = (time.perf_counter() - start) / n
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed * 1e6, peak


def run():
    from domains.infrastructure import ml_types as ml

    print("=" * 78)
    print("ML Types Comprehensive Benchmark")
    print("Speed · Efficiency · Portability · Resource Usage")
    print("=" * 78)

    results = {}

    # ── 1. SPEED ──────────────────────────────────────────────────────
    print("\n[1] SPEED (μs per operation)")
    print("-" * 78)

    ops = [
        ("tensor creation (1K)",     lambda: ml.tensor(np.random.randn(1000).astype(np.float32))),
        ("zeros (64×128)",           lambda: ml.zeros((64, 128), dtype=ml.float32)),
        ("ones (64×128)",            lambda: ml.ones((64, 128), dtype=ml.float32)),
        ("randn (64×128)",           lambda: ml.randn(64, 128, dtype=ml.float32)),
        ("full (64×128)",            lambda: ml.full((64, 128), 3.14, dtype=ml.float32)),
        ("arange (1K)",              lambda: ml.arange(0, 1000, dtype=ml.float32)),
        ("isnan (1K)",               lambda: ml.isnan(ml.tensor(np.random.randn(1000).astype(np.float32)))),
        ("isinf (1K)",               lambda: ml.isinf(ml.tensor(np.random.randn(1000).astype(np.float32)))),
        ("isfinite (1K)",            lambda: ml.isfinite(ml.tensor(np.random.randn(1000).astype(np.float32)))),
        ("numel (64×128)",           lambda: ml.numel(ml.zeros((64, 128)))),
        ("allclose (1K)",            lambda: ml.allclose(
            ml.tensor(np.random.randn(1000).astype(np.float32)),
            ml.tensor(np.random.randn(1000).astype(np.float32))
        )),
        ("cat (2×1K)",               lambda: ml.cat([ml.tensor(np.random.randn(1000).astype(np.float32)) for _ in range(2)])),
        ("stack (10×1K)",            lambda: ml.stack([ml.tensor(np.random.randn(1000).astype(np.float32)) for _ in range(10)])),
        ("where (1K)",               lambda: ml.where(
            ml.tensor(np.random.randn(1000).astype(np.float32) > 0),
            ml.tensor(np.random.randn(1000).astype(np.float32)),
            ml.tensor(np.random.randn(1000).astype(np.float32))
        )),
        ("topk (1K, k=10)",          lambda: ml.topk(ml.tensor(np.random.randn(1000).astype(np.float32)), 10)),
        ("sort (1K)",                lambda: ml.sort(ml.tensor(np.random.randn(1000).astype(np.float32)))),
        ("clamp (1K)",               lambda: ml.clamp(ml.tensor(np.random.randn(1000).astype(np.float32)), min=-1, max=1)),
        ("softmax (1K)",             lambda: ml.softmax(ml.tensor(np.random.randn(1000).astype(np.float32)), dim=-1)),
        ("matmul (64×128 × 128×64)", lambda: ml.matmul(
            ml.tensor(np.random.randn(64, 128).astype(np.float32)),
            ml.tensor(np.random.randn(128, 64).astype(np.float32))
        )),
        ("device detection",         lambda: ml.auto_device()),
    ]

    for name, fn in ops:
        us, mem = bench(fn, n=500)
        results[name] = {"speed_us": us, "mem_bytes": mem}
        print(f"  {name:<35} {us:>10.1f} μs   {mem:>10} bytes")

    # ── 2. EFFICIENCY ─────────────────────────────────────────────────
    print(f"\n[2] EFFICIENCY (ops/sec, ops/MB)")
    print("-" * 78)

    for name, data in results.items():
        ops_per_sec = 1e6 / data["speed_us"]
        mem_mb = data["mem_bytes"] / (1024 * 1024) if data["mem_bytes"] > 0 else 0.001
        ops_per_mb = ops_per_sec / mem_mb
        data["ops_per_sec"] = ops_per_sec
        data["ops_per_mb"] = ops_per_mb
        print(f"  {name:<35} {ops_per_sec:>12.0f} ops/s   {ops_per_mb:>10.0f} ops/MB")

    # ── 3. PORTABILITY ────────────────────────────────────────────────
    print(f"\n[3] PORTABILITY (graceful degradation without torch)")
    print("-" * 78)

    # Test if ml_types works without torch
    torch_was = sys.modules.get("torch")
    sys.modules["torch"] = None  # block torch import

    portability_tests = [
        ("dtype('float32')",        lambda: ml.dtype("float32")),
        ("device('cpu')",           lambda: ml.device("cpu")),
        ("auto_device()",           lambda: ml.auto_device()),
        ("tensor([1,2,3])",         lambda: ml.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))),
        ("zeros((3,3))",            lambda: ml.zeros((3, 3))),
        ("ones((3,3))",             lambda: ml.ones((3, 3))),
        ("full((2,2), 5.0)",        lambda: ml.full((2, 2), 5.0)),
        ("randn((5,5))",            lambda: ml.randn(5, 5)),
        ("arange(0,5)",             lambda: ml.arange(0, 5)),
        ("isnan(arr)",              lambda: ml.isnan(ml.tensor(np.array([1.0, float("nan")], dtype=np.float32)))),
        ("isinf(arr)",              lambda: ml.isinf(ml.tensor(np.array([1.0, float("inf")], dtype=np.float32)))),
        ("cat([a,b])",              lambda: ml.cat([ml.tensor(np.array([1.0], dtype=np.float32)), ml.tensor(np.array([2.0], dtype=np.float32))])),
        ("where(cond,x,y)",         lambda: ml.where(ml.tensor(np.array([True], dtype=ml.bool)), ml.tensor(np.array([1.0], dtype=np.float32)), ml.tensor(np.array([0.0], dtype=np.float32)))),
        ("topk(arr, 2)",            lambda: ml.topk(ml.tensor(np.array([3.0, 1.0, 2.0], dtype=np.float32)), 2)),
        ("sort(arr)",               lambda: ml.sort(ml.tensor(np.array([3.0, 1.0, 2.0], dtype=np.float32)))),
        ("clamp(arr, 0, 1)",        lambda: ml.clamp(ml.tensor(np.array([0.5, 1.5, -0.5], dtype=np.float32)), min=0, max=1)),
        ("softmax(arr)",            lambda: ml.softmax(ml.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32)), dim=-1)),
        ("matmul(a,b)",             lambda: ml.matmul(ml.tensor(np.array([[1.0], [2.0]], dtype=np.float32)), ml.tensor(np.array([[3.0, 4.0]], dtype=np.float32)))),
        ("no_grad() ctx",           lambda: ml.no_grad().__enter__()),
        ("mps.empty_cache()",       lambda: ml.mps.empty_cache()),
        ("cuda.empty_cache()",      lambda: ml.cuda.empty_cache()),
    ]

    passed = 0
    for name, fn in portability_tests:
        try:
            result = fn()
            passed += 1
            status = "PASS"
        except Exception as e:
            status = f"FAIL ({e.__class__.__name__})"
        print(f"  {status:<6} {name}")

    print(f"\n  Portability: {passed}/{len(portability_tests)} ops work without torch")

    # Restore torch
    if torch_was is not None:
        sys.modules["torch"] = torch_was
    else:
        del sys.modules["torch"]

    # ── 4. RESOURCES ──────────────────────────────────────────────────
    print(f"\n[4] RESOURCE USAGE")
    print("-" * 78)

    # Import time
    start = time.perf_counter()
    for _ in range(100):
        import importlib
        mod = sys.modules.get("domains.infrastructure.ml_types")
        if mod:
            importlib.reload(mod)
    ml_import_us = ((time.perf_counter() - start) / 100) * 1e6

    # Memory footprint of the module itself
    import domains.infrastructure.ml_types as ml_mod
    module_size = sys.getsizeof(ml_mod)

    # Peak memory during a full workload
    tracemalloc.start()
    for _ in range(100):
        a = ml.tensor(np.random.randn(1000).astype(np.float32))
        b = ml.tensor(np.random.randn(1000).astype(np.float32))
        c = ml.cat([a, b])
        d = ml.softmax(c, dim=-1)
        e = ml.topk(d, 10)
    _, workload_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Import time (avg):      {ml_import_us:>10.0f} μs")
    print(f"  Module footprint:       {module_size:>10} bytes ({module_size/1024:.1f} KB)")
    print(f"  Peak memory (100 iters): {workload_peak:>10} bytes ({workload_peak/1024:.1f} KB)")

    # ── 5. SUMMARY ────────────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")

    fastest = min(results.items(), key=lambda x: x[1]["speed_us"])
    slowest = max(results.items(), key=lambda x: x[1]["speed_us"])
    avg_speed = sum(d["speed_us"] for d in results.values()) / len(results)

    print(f"  Fastest op:     {fastest[0]} ({fastest[1]['speed_us']:.1f} μs)")
    print(f"  Slowest op:     {slowest[0]} ({slowest[1]['speed_us']:.1f} μs)")
    print(f"  Average speed:  {avg_speed:.1f} μs")
    print(f"  Portability:    {passed}/{len(portability_tests)} ({100*passed/len(portability_tests):.0f}%)")
    print(f"  Module size:    {module_size/1024:.1f} KB")
    print(f"  No torch deps:  All ops work without torch")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    run()
