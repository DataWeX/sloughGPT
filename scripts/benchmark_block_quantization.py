#!/usr/bin/env python3
"""
Benchmark: Lloyd+Huffman (cluster) vs Block Q4 vs Block Q8.

Compares compression methods across realistic NN weight shapes.
Metrics: ratio, cosine similarity, MSE, compress time, decompress time.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Ensure packages/core-py is on sys.path
_core = str(Path(__file__).resolve().parents[1] / "packages" / "core-py")
if _core not in sys.path:
    sys.path.insert(0, _core)

# Also add root for domain.* imports needed by pugqeep __init__
_root = str(Path(__file__).resolve().parents[1])
if _root not in sys.path:
    sys.path.insert(0, _root)

from domain.infrastructure._internal.pugqeep.compressor import PointCompressor
from domain.infrastructure._internal.pugqeep.point import Point


def make_weight(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    w = rng.randn(*shape).astype(np.float32) * 0.02
    if len(shape) == 2 and shape[0] > 16 and shape[1] > 16:
        rank = min(16, min(shape))
        u = rng.randn(shape[0], rank).astype(np.float32) * 0.01
        v = rng.randn(rank, shape[1]).astype(np.float32) * 0.01
        w += u @ v
    return w


def bench_one(compressor: PointCompressor, weights: np.ndarray, name: str, method: str) -> dict:
    flat = weights.flatten()
    raw_bytes = flat.nbytes

    t0 = time.perf_counter()
    if method == "cluster":
        point = compressor.compress_cluster(flat, "bench", n_clusters=min(64, len(flat) // 16))
    elif method == "block_q4":
        point = compressor.compress_block_q4(flat, "bench")
    elif method == "block_q8":
        point = compressor.compress_block_q8(flat, "bench")
    else:
        raise ValueError(method)
    compress_ms = (time.perf_counter() - t0) * 1000

    comp_bytes = point.nbytes()
    ratio = raw_bytes / max(comp_bytes, 1)

    # Decompress
    t0 = time.perf_counter()
    if method == "block_q4":
        decompressed = compressor.decompress_block_q4(point)
    elif method == "block_q8":
        decompressed = compressor.decompress_block_q8(point)
    else:
        decompressed = point.generate(len(flat))
    decompress_ms = (time.perf_counter() - t0) * 1000

    n = min(len(flat), len(decompressed))
    mse = float(np.mean((flat[:n] - decompressed[:n]) ** 2))
    cos_sim = float(
        np.dot(flat[:n], decompressed[:n])
        / (np.linalg.norm(flat[:n]) * np.linalg.norm(decompressed[:n]) + 1e-10)
    )

    # Serialize roundtrip (only for block types — cluster uses Huffman bitstream)
    ser_ms = 0.0
    if method in ("block_q4", "block_q8"):
        t0 = time.perf_counter()
        data = point.to_bytes()
        Point.from_bytes(data, identity="bench")
        ser_ms = (time.perf_counter() - t0) * 1000

    return {
        "method": method,
        "shape": weights.shape,
        "raw_bytes": raw_bytes,
        "comp_bytes": comp_bytes,
        "ratio": ratio,
        "cosine": cos_sim,
        "mse": mse,
        "compress_ms": compress_ms,
        "decompress_ms": decompress_ms,
        "serialize_ms": ser_ms,
        "accuracy": point.accuracy,
    }


def run_benchmark():
    compressor = PointCompressor()

    shapes = {
        "attn_qkv (128x512)": (128, 512),
        "attn_out (128x128)": (128, 128),
        "ffn_up (128x256)": (128, 256),
        "ffn_down (256x128)": (256, 128),
        "tiny (32x32)": (32, 32),
    }

    methods = ["cluster", "block_q4", "block_q8"]

    all_results = []
    for name, shape in shapes.items():
        w = make_weight(shape, seed=hash(name) % 1000)
        for method in methods:
            r = bench_one(compressor, w, name, method)
            r["layer"] = name
            all_results.append(r)

    # Print results
    print(
        f"{'Layer':<25} {'Method':<12} {'Ratio':>7} {'Cosine':>8} {'MSE':>10} {'Comp ms':>8} {'Decomp ms':>9} {'Ser ms':>7} {'Acc':>6}"
    )
    print("-" * 110)

    for r in all_results:
        print(
            f"{r['layer']:<25} {r['method']:<12} {r['ratio']:>6.1f}x {r['cosine']:>8.5f} {r['mse']:>10.2e} {r['compress_ms']:>7.1f} {r['decompress_ms']:>8.1f} {r['serialize_ms']:>6.1f} {r['accuracy']:>5.3f}"
        )

    # Summary
    print("\n" + "=" * 110)
    print("SUMMARY (averages across all layers)")
    print("=" * 110)
    for method in methods:
        subset = [r for r in all_results if r["method"] == method]
        avg_ratio = np.mean([r["ratio"] for r in subset])
        avg_cosine = np.mean([r["cosine"] for r in subset])
        avg_mse = np.mean([r["mse"] for r in subset])
        avg_comp = np.mean([r["compress_ms"] for r in subset])
        avg_decomp = np.mean([r["decompress_ms"] for r in subset])
        avg_ser = np.mean([r["serialize_ms"] for r in subset])
        print(
            f"{method:<12} ratio={avg_ratio:.1f}x  cosine={avg_cosine:.5f}  mse={avg_mse:.2e}  compress={avg_comp:.1f}ms  decompress={avg_decomp:.1f}ms  serialize={avg_ser:.1f}ms"
        )


if __name__ == "__main__":
    run_benchmark()
