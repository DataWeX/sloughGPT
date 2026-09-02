#!/usr/bin/env python3
"""
Benchmark centroid-focused VQ improvements.

All approaches use the same VQ core — the difference is in how
centroids are initialized, refined, and stored.

Approaches:
  0. Baseline — quantile init + Lloyd's 3 iters (current)
  1. Lloyd 10 — more refinement iterations
  2. Lloyd 20 — even more refinement
  3. k-means++ init — better seeding
  4. Hierarchical centroids — store centroid pattern as function
  5. Centroid int8 — quantize centroids to int8 with scale/zero-point
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Data generation
# ══════════════════════════════════════════════════════════════════════════════

def make_weight(shape: Tuple[int, ...], seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    w = rng.randn(*shape).astype(np.float32) * 0.02
    if len(shape) == 2 and shape[0] > 16 and shape[1] > 16:
        rank = min(16, min(shape))
        u = rng.randn(shape[0], rank).astype(np.float32) * 0.01
        v = rng.randn(rank, shape[1]).astype(np.float32) * 0.01
        w += u @ v
    return w


def make_test_suite() -> Dict[str, np.ndarray]:
    return {
        "attn_qkv":  make_weight((768, 2304)),
        "attn_out":  make_weight((768, 768)),
        "ffn_up":    make_weight((768, 3072)),
        "ffn_down":  make_weight((3072, 768)),
        "ln_weight": make_weight((768,)),
        "bias":      make_weight((768,)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchResult:
    name: str
    shape: Tuple[int, ...]
    raw_bytes: int
    compressed_bytes: int
    ratio: float
    cosine: float
    mse: float
    compress_ms: float
    decompress_ms: float


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    n = np.linalg.norm(a) * np.linalg.norm(b)
    if n < 1e-12:
        return 1.0
    return float(np.dot(a.ravel(), b.ravel()) / n)


def measure(func, *args, repeats: int = 2):
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = func(*args)
        times.append((time.perf_counter() - t0) * 1000)
    return result, sorted(times)[len(times) // 2]


# ══════════════════════════════════════════════════════════════════════════════
# Adaptive k (shared)
# ══════════════════════════════════════════════════════════════════════════════

def compute_adaptive_k(flat: np.ndarray, base_k: int = 16) -> int:
    hist, _ = np.histogram(flat, bins=256)
    hist = hist[hist > 0].astype(np.float64)
    probs = hist / hist.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    scale = entropy / 8.0
    k = int(base_k * (0.5 + 3.5 * scale))
    return max(4, min(k, 256))


# ══════════════════════════════════════════════════════════════════════════════
# k-means++ init (lighter version — used only for seeding)
# ══════════════════════════════════════════════════════════════════════════════

def kmeans_pp_init(flat: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    n = len(flat)
    centroids = np.empty(k, dtype=np.float32)
    centroids[0] = flat[rng.randint(n)]
    for i in range(1, k):
        dists = np.min((flat[:, None] - centroids[None, :i]) ** 2, axis=1)
        probs = dists / dists.sum()
        centroids[i] = flat[rng.choice(n, p=probs)]
    return np.sort(centroids)


# ══════════════════════════════════════════════════════════════════════════════
# Approach 0: Baseline — quantile init + Lloyd's 3 iters
# ══════════════════════════════════════════════════════════════════════════════

def vq_compress(flat, k, lloyd_iters=3, init="quantile"):
    nc = k
    if init == "quantile":
        quantiles = np.linspace(0, 100, nc + 2)[1:-1]
        centroids = np.percentile(flat, quantiles).astype(np.float32)
        centroids.sort()
    elif init == "kmeanspp":
        centroids = kmeans_pp_init(flat, nc, np.random.RandomState(42))

    for _ in range(lloyd_iters):
        assigns = np.clip(np.searchsorted(centroids, flat), 0, nc - 1).astype(np.intp)
        sums = np.bincount(assigns, weights=flat, minlength=nc)
        counts = np.bincount(assigns, minlength=nc).astype(np.float64)
        alive = counts > 0
        centroids[alive] = (sums[alive] / counts[alive]).astype(np.float32)

    assigns = np.clip(np.searchsorted(centroids, flat), 0, nc - 1).astype(np.uint8)
    recon = centroids[assigns]
    mse = float(np.mean((flat - recon) ** 2))
    var = float(np.var(flat))
    acc = 1.0 - mse / (var + 1e-8)
    nbytes = centroids.nbytes + assigns.nbytes
    return centroids, assigns, acc, nbytes


def bench_baseline(flat):
    flat = flat.flatten()
    k = compute_adaptive_k(flat, 16)
    return vq_compress(flat, k, 3, "quantile")


def bench_lloyd10(flat):
    flat = flat.flatten()
    k = compute_adaptive_k(flat, 16)
    return vq_compress(flat, k, 10, "quantile")


def bench_lloyd20(flat):
    flat = flat.flatten()
    k = compute_adaptive_k(flat, 16)
    return vq_compress(flat, k, 20, "quantile")


def bench_kmeanspp(flat):
    flat = flat.flatten()
    k = compute_adaptive_k(flat, 16)
    return vq_compress(flat, k, 3, "kmeanspp")


# ══════════════════════════════════════════════════════════════════════════════
# Approach 4: Hierarchical centroids — store centroid pattern as function
# ══════════════════════════════════════════════════════════════════════════════

def fit_centroid_function(centroids: np.ndarray):
    """Try to compress centroids as a linear function a*i + b."""
    nc = len(centroids)
    if nc < 4:
        return None, float('inf')

    i = np.arange(nc, dtype=np.float32)
    A = np.column_stack([i, np.ones(nc)])
    result, _, _, _ = np.linalg.lstsq(A, centroids, rcond=None)
    a, b = result
    fitted = a * i + b
    mse = float(np.mean((centroids - fitted) ** 2))
    return {"a": float(a), "b": float(b)}, mse


def bench_hierarchical(flat):
    flat = flat.flatten()
    k = compute_adaptive_k(flat, 16)
    centroids, assigns, acc, nb = vq_compress(flat, k, 3, "quantile")

    # Try to compress centroids as function
    func_params, func_mse = fit_centroid_function(centroids)
    if func_params is not None:
        # Check if function approximation is good enough
        i = np.arange(len(centroids), dtype=np.float32)
        fitted = func_params["a"] * i + func_params["b"]
        recon_via_func = fitted[assigns]
        func_cos = cosine_sim(flat, recon_via_func)
        # If function centroids give >99% of VQ accuracy, use function storage
        if func_cos > acc * 0.99:
            # Store as function: 8 bytes (a, b) + assignments
            nbytes = 8 + assigns.nbytes
            return func_params["a"], func_params["b"], assigns, acc, nbytes, True

    # Fallback: store raw centroids
    return centroids, assigns, acc, nb


def decompress_hierarchical(r, n):
    if len(r) == 6:
        # Function storage: (a, b, assigns, acc, nbytes, True)
        a, b, assigns, _, _, _ = r
        nc = max(assigns) + 1
        i = np.arange(nc, dtype=np.float32)
        centroids = a * i + b
        return centroids[assigns[:n]]
    else:
        # Raw centroids: (centroids, assigns, acc, nbytes)
        centroids, assigns, _, _ = r
        return centroids[assigns[:n]]


# ══════════════════════════════════════════════════════════════════════════════
# Approach 5: Centroid int8 — quantize centroids to int8
# ══════════════════════════════════════════════════════════════════════════════

def bench_centroid_int8(flat):
    flat = flat.flatten()
    k = compute_adaptive_k(flat, 16)
    centroids, assigns, acc, nb = vq_compress(flat, k, 3, "quantile")

    # Quantize centroids to int8
    cmin, cmax = float(centroids.min()), float(centroids.max())
    crange = cmax - cmin
    if crange < 1e-12:
        crange = 1.0
    scale = crange / 255.0
    zero_point = -cmin / scale

    q_centroids = np.clip(np.round(centroids / scale + zero_point), 0, 255).astype(np.uint8)
    # Dequantize for reconstruction
    recon_centroids = (q_centroids.astype(np.float32) - zero_point) * scale

    # Check accuracy loss from centroid quantization
    recon_via_q = recon_centroids[assigns]
    q_cos = cosine_sim(flat, recon_via_q)

    # If int8 centroids preserve >99.9% of accuracy, use int8 storage
    if q_cos > acc * 0.999:
        nbytes = 8 + q_centroids.nbytes + assigns.nbytes  # scale + zero_point + centroids + assigns
        return q_centroids, scale, zero_point, assigns, acc, nbytes
    else:
        return centroids, assigns, acc, nb


def decompress_centroid_int8(r, n):
    if len(r) == 6:
        q_centroids, scale, zero_point, assigns, _, _ = r
        centroids = (q_centroids.astype(np.float32) - zero_point) * scale
        return centroids[assigns[:n]]
    else:
        centroids, assigns, _, _ = r
        return centroids[assigns[:n]]


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ══════════════════════════════════════════════════════════════════════════════

def bench_one(name, flat, compress_fn, decompress_fn):
    raw_bytes = flat.nbytes
    flat_1d = flat.flatten()
    n = len(flat_1d)
    result, compress_ms = measure(compress_fn, flat_1d)
    decompressed, decompress_ms = measure(decompress_fn, result, n)
    cosine = cosine_sim(flat_1d, decompressed)
    mse = float(np.mean((flat_1d - decompressed) ** 2))
    # Extract compressed bytes — nbytes is always the last element
    # except hierarchical function path where last is True flag
    if len(result) == 6 and result[-1] is True:
        compressed_bytes = result[4]  # hierarchical: (a, b, assigns, acc, nbytes, True)
    else:
        compressed_bytes = result[-1]  # all other paths: last element is nbytes
    ratio = raw_bytes / max(compressed_bytes, 1)
    return BenchResult(name=name, shape=flat.shape, raw_bytes=raw_bytes,
                       compressed_bytes=compressed_bytes, ratio=ratio,
                       cosine=cosine, mse=mse,
                       compress_ms=compress_ms, decompress_ms=decompress_ms)


def run_benchmarks():
    suite = make_test_suite()

    approaches = [
        ("0_baseline",   bench_baseline,   lambda r, n: r[0][r[1][:n]]),
        ("1_lloyd10",    bench_lloyd10,    lambda r, n: r[0][r[1][:n]]),
        ("2_lloyd20",    bench_lloyd20,    lambda r, n: r[0][r[1][:n]]),
        ("3_kmeanspp",   bench_kmeanspp,   lambda r, n: r[0][r[1][:n]]),
        ("4_hierarch",   bench_hierarchical, decompress_hierarchical),
        ("5_cent_int8",  bench_centroid_int8, decompress_centroid_int8),
    ]

    all_results: List[BenchResult] = []

    print("=" * 100)
    print("CENTROID-FOCUSED VQ BENCHMARK")
    print("=" * 100)
    print(f"{'Approach':<16} {'Shape':<16} {'Ratio':>7} {'Cosine':>8} {'MSE':>12} {'Compress':>10} {'Decompress':>10}")
    print("-" * 100)

    for weight_name, flat in suite.items():
        for approach_name, compress_fn, decompress_fn in approaches:
            result = bench_one(approach_name, flat, compress_fn, decompress_fn)
            all_results.append(result)
            print(f"{approach_name:<16} {str(flat.shape):<16} {result.ratio:>7.2f}x "
                  f"{result.cosine:>8.5f} {result.mse:>12.2e} "
                  f"{result.compress_ms:>8.1f}ms {result.decompress_ms:>8.1f}ms")
        print()

    # Summary
    print("=" * 100)
    print("AVERAGE BY APPROACH")
    print("=" * 100)
    print(f"{'Approach':<16} {'Avg Ratio':>10} {'Avg Cosine':>11} {'Avg MSE':>12} {'Avg Compress':>13} {'Avg Decompress':>14}")
    print("-" * 100)

    names = [a[0] for a in approaches]
    for name in names:
        rs = [r for r in all_results if r.name == name]
        ar = np.mean([r.ratio for r in rs])
        ac = np.mean([r.cosine for r in rs])
        am = np.mean([r.mse for r in rs])
        atc = np.mean([r.compress_ms for r in rs])
        atd = np.mean([r.decompress_ms for r in rs])
        print(f"{name:<16} {ar:>10.2f}x {ac:>11.5f} {am:>12.2e} {atc:>11.1f}ms {atd:>12.1f}ms")

    # Delta from baseline
    print()
    print("=" * 100)
    print("DELTA FROM BASELINE (0_baseline)")
    print("=" * 100)
    baseline_rs = [r for r in all_results if r.name == "0_baseline"]
    b_ratio = np.mean([r.ratio for r in baseline_rs])
    b_cosine = np.mean([r.cosine for r in baseline_rs])
    b_mse = np.mean([r.mse for r in baseline_rs])
    b_speed = np.mean([r.compress_ms for r in baseline_rs])

    for name in names:
        if name == "0_baseline":
            continue
        rs = [r for r in all_results if r.name == name]
        d_ratio = np.mean([r.ratio for r in rs]) - b_ratio
        d_cosine = np.mean([r.cosine for r in rs]) - b_cosine
        d_mse = np.mean([r.mse for r in rs]) - b_mse
        d_speed = np.mean([r.compress_ms for r in rs]) - b_speed
        print(f"{name:<16} ratio={d_ratio:+.2f}x  cosine={d_cosine:+.5f}  mse={d_mse:+.2e}  speed={d_speed:+.1f}ms")

    # Verdict
    print()
    print("=" * 100)
    print("VERDICT (accuracy-first)")
    print("=" * 100)
    metrics = {n: {
        "ratio": np.mean([r.ratio for r in all_results if r.name == n]),
        "cosine": np.mean([r.cosine for r in all_results if r.name == n]),
        "mse": np.mean([r.mse for r in all_results if r.name == n]),
        "speed": np.mean([r.compress_ms for r in all_results if r.name == n]),
    } for n in names}

    by_cosine = sorted(names, key=lambda n: metrics[n]["cosine"], reverse=True)
    for i, n in enumerate(by_cosine):
        m = metrics[n]
        tag = " <-- BEST" if i == 0 else ""
        print(f"  {i+1}. {n:<16} cos={m['cosine']:.5f}  ratio={m['ratio']:.2f}x  speed={m['speed']:.1f}ms{tag}")


if __name__ == "__main__":
    run_benchmarks()
