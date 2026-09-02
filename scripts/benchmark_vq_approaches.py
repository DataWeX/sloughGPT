#!/usr/bin/env python3
"""
Benchmark 6 VQ improvement approaches against the current baseline.

Approaches:
  0. Baseline VQ — quantile init + Lloyd's (current implementation)
  1. Adaptive k — cluster count varies per layer by weight entropy
  2. k-means++ init — better centroid seeding
  3. Product Quantization (OPQ) — sub-vector quantization with learned rotation
  4. Residual VQ (RVQ) — cascade of VQ stages on residuals
  5. Finite Scalar Quantization (FSQ) — discrete bottleneck rounding

Metrics:
  - Compression ratio (raw_bytes / compressed_bytes)
  - Cosine similarity (1.0 = perfect)
  - MSE
  - Compress time (ms)
  - Decompress time (ms)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Data generation — realistic NN weight distributions
# ══════════════════════════════════════════════════════════════════════════════

def make_weight(shape: Tuple[int, ...], seed: int = 0) -> np.ndarray:
    """Generate realistic NN weights: near-zero mean, small variance, low-rank structure."""
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
        "attn_qkv":    make_weight((768, 2304)),
        "attn_out":    make_weight((768, 768)),
        "ffn_up":      make_weight((768, 3072)),
        "ffn_down":    make_weight((3072, 768)),
        "ln_weight":   make_weight((768,)),
        "bias":        make_weight((768,)),
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
# VQ core — used by multiple approaches
# ══════════════════════════════════════════════════════════════════════════════

def vq_core(flat: np.ndarray, n_clusters: int, lloyd_iters: int = 3,
            init: str = "quantile", rng: np.random.RandomState = None):
    """Core VQ: returns (centroids, assignments, accuracy, nbytes)."""
    n = len(flat)
    n_clusters = min(n_clusters, n)
    nc = n_clusters

    if init == "quantile":
        quantiles = np.linspace(0, 100, nc + 2)[1:-1]
        centroids = np.percentile(flat, quantiles).astype(np.float32)
        centroids.sort()
    elif init == "kmeanspp":
        centroids = _kmeans_pp_init(flat, nc, rng or np.random.RandomState(42))
    else:
        raise ValueError(f"Unknown init: {init}")

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
    accuracy = 1.0 - mse / (var + 1e-8)
    nbytes = centroids.nbytes + assigns.nbytes
    return centroids, assigns, accuracy, nbytes


def _kmeans_pp_init(flat: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    n = len(flat)
    centroids = np.empty(k, dtype=np.float32)
    centroids[0] = flat[rng.randint(n)]
    for i in range(1, k):
        dists = np.min((flat[:, None] - centroids[None, :i]) ** 2, axis=1)
        probs = dists / dists.sum()
        centroids[i] = flat[rng.choice(n, p=probs)]
    return np.sort(centroids)


def compute_entropy_k(flat: np.ndarray, base_k: int = 16) -> int:
    hist, _ = np.histogram(flat, bins=256)
    hist = hist[hist > 0].astype(np.float64)
    probs = hist / hist.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    scale = entropy / 8.0
    k = int(base_k * (0.5 + 3.5 * scale))
    return max(4, min(k, 256))


# ══════════════════════════════════════════════════════════════════════════════
# Approach 0: Baseline VQ
# ══════════════════════════════════════════════════════════════════════════════

def bench_baseline(flat):
    flat = flat.flatten()
    c, a, acc, nb = vq_core(flat, 16, 3, "quantile")
    return c, a, acc, nb

def decompress_baseline(r, n):
    return r[0][r[1][:n]]


# ══════════════════════════════════════════════════════════════════════════════
# Approach 1: Adaptive k
# ══════════════════════════════════════════════════════════════════════════════

def bench_adaptive_k(flat):
    flat = flat.flatten()
    k = compute_entropy_k(flat, 16)
    c, a, acc, nb = vq_core(flat, k, 3, "quantile")
    return c, a, acc, nb

def decompress_adaptive_k(r, n):
    return r[0][r[1][:n]]


# ══════════════════════════════════════════════════════════════════════════════
# Approach 2: k-means++ init
# ══════════════════════════════════════════════════════════════════════════════

def bench_kmeanspp(flat):
    flat = flat.flatten()
    c, a, acc, nb = vq_core(flat, 16, 3, "kmeanspp")
    return c, a, acc, nb

def decompress_kmeanspp(r, n):
    return r[0][r[1][:n]]


# ══════════════════════════════════════════════════════════════════════════════
# Approach 3: Product Quantization (OPQ)
# ══════════════════════════════════════════════════════════════════════════════

def bench_pq(flat, n_sub=8, nc_per_sub=16):
    flat = flat.flatten()
    n = len(flat)
    pad = (n_sub - n % n_sub) % n_sub
    padded = np.concatenate([flat, np.zeros(pad, dtype=np.float32)]) if pad else flat
    dim = len(padded) // n_sub
    subs = padded.reshape(n_sub, dim)

    all_c, all_a = [], []
    total_bytes = 0
    for i in range(n_sub):
        c, a, _, nb = vq_core(subs[i], nc_per_sub, 3, "quantile")
        all_c.append(c)
        all_a.append(a)
        total_bytes += nb

    # Reconstruct
    recon_subs = np.array([all_c[i][all_a[i]] for i in range(n_sub)])
    recon = recon_subs.ravel()[:n]

    mse = float(np.mean((flat - recon) ** 2))
    var = float(np.var(flat))
    acc = 1.0 - mse / (var + 1e-8)
    return all_c, all_a, acc, total_bytes

def decompress_pq(r, n):
    all_c, all_a = r[0], r[1]
    n_sub = len(all_c)
    recon_subs = np.array([all_c[i][all_a[i]] for i in range(n_sub)])
    return recon_subs.ravel()[:n]


# ══════════════════════════════════════════════════════════════════════════════
# Approach 4: Residual VQ (RVQ)
# ══════════════════════════════════════════════════════════════════════════════

def bench_rvq(flat, n_stages=3, nc=16):
    flat = flat.flatten()
    residual = flat.copy()
    all_c, all_a = [], []
    total_bytes = 0
    for _ in range(n_stages):
        c, a, _, nb = vq_core(residual, nc, 3, "quantile")
        all_c.append(c)
        all_a.append(a)
        total_bytes += nb
        recon_stage = c[a[:len(residual)]]
        residual = residual - recon_stage

    # Reconstruct
    recon = np.zeros_like(flat)
    for s in range(n_stages):
        recon += all_c[s][all_a[s][:len(flat)]]

    mse = float(np.mean((flat - recon) ** 2))
    var = float(np.var(flat))
    acc = 1.0 - mse / (var + 1e-8)
    return all_c, all_a, acc, total_bytes

def decompress_rvq(r, n):
    all_c, all_a = r[0], r[1]
    recon = np.zeros(n, dtype=np.float32)
    for s in range(len(all_c)):
        recon += all_c[s][all_a[s][:n]]
    return recon


# ══════════════════════════════════════════════════════════════════════════════
# Approach 5: Finite Scalar Quantization (FSQ)
# ══════════════════════════════════════════════════════════════════════════════

def bench_fsq(flat, n_levels=8):
    flat = flat.flatten()
    vmin, vmax = float(flat.min()), float(flat.max())
    vrange = vmax - vmin
    if vrange < 1e-12:
        vrange = 1.0
    scale = (n_levels - 1) / vrange
    zero_point = -vmin * scale

    q = np.clip(np.round(flat * scale + zero_point), 0, n_levels - 1).astype(np.uint8)
    recon = (q.astype(np.float32) - zero_point) / scale

    mse = float(np.mean((flat - recon) ** 2))
    var = float(np.var(flat))
    acc = 1.0 - mse / (var + 1e-8)
    nbytes = 8 + q.nbytes  # scale + zero_point + quantized
    return scale, zero_point, q, acc, nbytes

def decompress_fsq(r, n):
    scale, zero_point, q = r[0], r[1], r[2]
    return (q[:n].astype(np.float32) - zero_point) / scale


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
    compressed_bytes = result[-1]
    ratio = raw_bytes / max(compressed_bytes, 1)
    return BenchResult(name=name, shape=flat.shape, raw_bytes=raw_bytes,
                       compressed_bytes=compressed_bytes, ratio=ratio,
                       cosine=cosine, mse=mse,
                       compress_ms=compress_ms, decompress_ms=decompress_ms)


def run_benchmarks():
    suite = make_test_suite()

    approaches = [
        ("0_baseline",    bench_baseline,    decompress_baseline),
        ("1_adaptive_k",  bench_adaptive_k,  decompress_adaptive_k),
        ("2_kmeanspp",    bench_kmeanspp,    decompress_kmeanspp),
        ("3_pq_8x16",     bench_pq,          decompress_pq),
        ("4_rvq_3stage",  bench_rvq,         decompress_rvq),
        ("5_fsq_8level",  bench_fsq,         decompress_fsq),
    ]

    all_results: List[BenchResult] = []

    print("=" * 95)
    print("VQ APPROACH BENCHMARK — 6 METHODS")
    print("=" * 95)
    print(f"{'Approach':<18} {'Shape':<16} {'Ratio':>7} {'Cosine':>8} {'MSE':>12} {'Compress':>10} {'Decompress':>10}")
    print("-" * 95)

    for weight_name, flat in suite.items():
        for approach_name, compress_fn, decompress_fn in approaches:
            result = bench_one(approach_name, flat, compress_fn, decompress_fn)
            all_results.append(result)
            print(f"{approach_name:<18} {str(flat.shape):<16} {result.ratio:>7.2f}x "
                  f"{result.cosine:>8.5f} {result.mse:>12.2e} "
                  f"{result.compress_ms:>8.1f}ms {result.decompress_ms:>8.1f}ms")
        print()

    # Summary
    print("=" * 95)
    print("AVERAGE BY APPROACH")
    print("=" * 95)
    print(f"{'Approach':<18} {'Avg Ratio':>10} {'Avg Cosine':>11} {'Avg MSE':>12} {'Avg Compress':>13} {'Avg Decompress':>14}")
    print("-" * 95)

    names = [a[0] for a in approaches]
    for name in names:
        rs = [r for r in all_results if r.name == name]
        ar = np.mean([r.ratio for r in rs])
        ac = np.mean([r.cosine for r in rs])
        am = np.mean([r.mse for r in rs])
        atc = np.mean([r.compress_ms for r in rs])
        atd = np.mean([r.decompress_ms for r in rs])
        print(f"{name:<18} {ar:>10.2f}x {ac:>11.5f} {am:>12.2e} {atc:>11.1f}ms {atd:>12.1f}ms")

    # Verdict
    print()
    print("=" * 95)
    print("VERDICT")
    print("=" * 95)

    metrics = {n: {
        "ratio": np.mean([r.ratio for r in all_results if r.name == n]),
        "cosine": np.mean([r.cosine for r in all_results if r.name == n]),
        "mse": np.mean([r.mse for r in all_results if r.name == n]),
        "speed": np.mean([r.compress_ms for r in all_results if r.name == n]),
    } for n in names}

    best_cos = max(names, key=lambda n: metrics[n]["cosine"])
    best_ratio = max(names, key=lambda n: metrics[n]["ratio"])
    best_speed = min(names, key=lambda n: metrics[n]["speed"])

    print(f"Best accuracy:  {best_cos} ({metrics[best_cos]['cosine']:.5f})")
    print(f"Best ratio:     {best_ratio} ({metrics[best_ratio]['ratio']:.2f}x)")
    print(f"Fastest:        {best_speed} ({metrics[best_speed]['speed']:.1f}ms)")

    # Combined score
    max_r = max(m["ratio"] for m in metrics.values())
    max_c = max(m["cosine"] for m in metrics.values())
    min_s = min(m["speed"] for m in metrics.values())

    print()
    print("Combined score (0.4*ratio + 0.4*cosine + 0.2*speed):")
    scores = {}
    for n in names:
        rn = metrics[n]["ratio"] / max_r
        cn = metrics[n]["cosine"] / max_c
        sn = min_s / max(metrics[n]["speed"], 1e-6)
        scores[n] = 0.4 * rn + 0.4 * cn + 0.2 * sn

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for i, (n, s) in enumerate(ranked):
        m = metrics[n]
        tag = " <-- WINNER" if i == 0 else ""
        print(f"  {i+1}. {n:<18} score={s:.4f}  (ratio={m['ratio']:.2f}x, cos={m['cosine']:.5f}, "
              f"speed={m['speed']:.1f}ms){tag}")


if __name__ == "__main__":
    run_benchmarks()
