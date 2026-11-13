#!/usr/bin/env python3
"""
Benchmark: Lloyd+Huffman (cluster) vs Block Q4 vs Block Q8.

Card vq-huffman-04. Measures, per realistic NN weight shape (low-rank
+ small-variance, the make_weight distribution shared with
benchmark_vq_approaches.py):

  * compression ratio — raw float32 bytes / Point.nbytes() stored bytes
    (the huffman-aware size: centroids + codes + bitstream for cluster,
    packed 4-bit + mins/scales for block_q4)
  * round-trip fidelity — cosine similarity + MSE of decompress(compress(w))
    against the source, plus Point's self-reported accuracy
  * encode / decode / serialize wall time — median of --rounds samples:
    the block codecs are sub-ms (single-sample timings are noise) and
    Lloyd refinement is seconds (median keeps both citable)
  * serialize round-trip integrity — to_bytes -> from_bytes -> decode must
    reproduce the first-pass decompression BIT-EXACTLY; a mismatch aborts
    the run (throughput/quality numbers are meaningless without it).
    Cluster is included: series cards 01-03 landed ClusterCodec
    serialization, the old block-only guard was stale.

Seeds derive from zlib.crc32 of the layer name — the old `hash(name)`
carried PYTHONHASHSEED, so runs weren't comparable process to process.

History is appended to scripts/benchmark_block_quantization.json
(downcraft precedent: benchmark_results.py has no compression category).

Usage:
    ./scripts/python scripts/benchmark_block_quantization.py [--rounds 3] [--no-history]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from datetime import UTC, datetime
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

from domain.infrastructure._internal.pugqeep.compressor import PointCompressor  # noqa: E402
from domain.infrastructure._internal.pugqeep.point import Point  # noqa: E402

HISTORY_PATH = Path(__file__).with_suffix(".json")
HISTORY_KEEP = 50


def make_weight(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    """Realistic NN weights: near-zero mean, small variance, low-rank structure."""
    rng = np.random.RandomState(seed)
    w = rng.randn(*shape).astype(np.float32) * 0.02
    if len(shape) == 2 and shape[0] > 16 and shape[1] > 16:
        rank = min(16, min(shape))
        u = rng.randn(shape[0], rank).astype(np.float32) * 0.01
        v = rng.randn(rank, shape[1]).astype(np.float32) * 0.01
        w += u @ v
    return w


def _stable_seed(name: str) -> int:
    """Process-independent seed (hash() carries PYTHONHASHSEED)."""
    return zlib.crc32(name.encode()) % 1000


def _compress(compressor: PointCompressor, flat: np.ndarray, method: str, seed: int) -> Point:
    if method == "cluster":
        # seeded: k-means++ sampling deterministic across runs/processes
        return compressor.compress_cluster(
            flat, "bench", n_clusters=min(64, len(flat) // 16), seed=seed
        )
    if method == "block_q4":
        return compressor.compress_block_q4(flat, "bench")
    if method == "block_q8":
        return compressor.compress_block_q8(flat, "bench")
    raise ValueError(method)


def _decode(compressor: PointCompressor, point: Point, n: int, method: str) -> np.ndarray:
    if method == "block_q4":
        return compressor.decompress_block_q4(point)
    if method == "block_q8":
        return compressor.decompress_block_q8(point)
    return point.generate(n)


def bench_one(
    compressor: PointCompressor,
    weights: np.ndarray,
    name: str,
    method: str,
    rounds: int,
) -> dict:
    flat = weights.flatten()
    raw_bytes = flat.nbytes
    n = len(flat)
    seed = _stable_seed(name)

    # encode: median over rounds (deterministic seed → same point each pass)
    comp_times: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        point = _compress(compressor, flat, method, seed)
        comp_times.append((time.perf_counter() - t0) * 1000)
    compress_ms = float(np.median(comp_times))

    # decode: median over rounds
    decomp_times: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        decompressed = _decode(compressor, point, n, method)
        decomp_times.append((time.perf_counter() - t0) * 1000)
    decompress_ms = float(np.median(decomp_times))

    k = min(n, len(decompressed))
    mse = float(np.mean((flat[:k] - decompressed[:k]) ** 2))
    cos_sim = float(
        np.dot(flat[:k], decompressed[:k])
        / (np.linalg.norm(flat[:k]) * np.linalg.norm(decompressed[:k]) + 1e-10)
    )

    # serialize roundtrip — ALL methods (cluster streams landed in card 03);
    # decode of the roundtripped Point must be bit-exact vs the first pass
    ser_times: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        data = point.to_bytes()
        back = Point.from_bytes(data, identity="bench")
        ser_times.append((time.perf_counter() - t0) * 1000)
    serialize_ms = float(np.median(ser_times))
    rt = _decode(compressor, back, n, method)
    if not np.array_equal(decompressed[:k], rt[:k]):
        raise SystemExit(
            f"roundtrip mismatch for {name}/{method} — aborting (numbers would be meaningless)"
        )

    return {
        "method": method,
        "layer": name,
        "shape": list(weights.shape),
        "raw_bytes": raw_bytes,
        "comp_bytes": point.nbytes(),
        "ratio": raw_bytes / max(point.nbytes(), 1),
        "cosine": cos_sim,
        "mse": mse,
        "compress_ms": compress_ms,
        "decompress_ms": decompress_ms,
        "serialize_ms": serialize_ms,
        "accuracy": float(point.accuracy),
    }


def run_benchmark(rounds: int, history: bool) -> dict:
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
        w = make_weight(shape, seed=_stable_seed(name))
        for method in methods:
            all_results.append(bench_one(compressor, w, name, method, rounds))

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
    print(f"SUMMARY (average across layers, median of {rounds} timing rounds each)")
    print("=" * 110)
    summary: dict[str, dict] = {}
    for method in methods:
        subset = [r for r in all_results if r["method"] == method]
        avg = {
            "ratio": float(np.mean([r["ratio"] for r in subset])),
            "cosine": float(np.mean([r["cosine"] for r in subset])),
            "mse": float(np.mean([r["mse"] for r in subset])),
            "compress_ms": float(np.mean([r["compress_ms"] for r in subset])),
            "decompress_ms": float(np.mean([r["decompress_ms"] for r in subset])),
            "serialize_ms": float(np.mean([r["serialize_ms"] for r in subset])),
        }
        summary[method] = avg
        print(
            f"{method:<12} ratio={avg['ratio']:.1f}x  cosine={avg['cosine']:.5f}  mse={avg['mse']:.2e}  "
            f"compress={avg['compress_ms']:.1f}ms  decompress={avg['decompress_ms']:.1f}ms  serialize={avg['serialize_ms']:.1f}ms"
        )

    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "rounds": rounds,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "rows": all_results,
        "summary": summary,
    }
    if history:
        _append_history(record)
        print(f"\nhistory appended: {HISTORY_PATH}")
    return record


def _append_history(record: dict) -> None:
    """Append-run history (downcraft precedent; last HISTORY_KEEP kept)."""
    entries: list = []
    if HISTORY_PATH.exists():
        try:
            entries = json.loads(HISTORY_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            entries = []
    entries.append(record)
    HISTORY_PATH.write_text(json.dumps(entries[-HISTORY_KEEP:], indent=1) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rounds", type=int, default=3, help="timing samples per codec (median reported)"
    )
    parser.add_argument(
        "--no-history", action="store_true", help="skip appending to the results JSON"
    )
    args = parser.parse_args()
    run_benchmark(args.rounds, history=not args.no_history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
