#!/usr/bin/env python3
"""
Benchmark: VQ approaches vs the SHIPPED cluster codec (vq-huffman-06 truth-up).

The old script framed its own local `vq_core` (quantile init, 3 Lloyd iters)
as "current baseline" and listed adaptive-k + k-means++ as PROPOSED approaches
— both shipped into `PointCompressor` by series card vq-huffman-02 (k-means++
init, adaptive_k by weight entropy, 5 Lloyd iters) and card -03 (Huffman-coded
assignments). Rows are now anchored on the real codec:

  * shipped_vq            real PointCompressor defaults — k-means++ init,
                          adaptive_k, Huffman assignments, quantized centroids.
                          This is the reference everything else is judged against.
  * shipped_vq_k16        same codec with adaptive_k=False, n_clusters=16 —
                          isolates the adaptive-k effect on the REAL stack.
  * legacy_quantile_k16   the pre-card-02 local baseline (quantile init) — kept
                          as historical context, NOT a candidate.
  * pq_8x16 / rvq_3stage / fsq_8level   exploration set — the genuinely
                          unexplored candidates, still worth measuring.

Measures per BERT-scale layer (attn/ffn/ln/bias, make_weight low-rank
distribution): compression ratio (raw float32 bytes / stored bytes — shipped
rows include the Huffman bitstream; exploration rows are centroids + dense
uint8 assignments, i.e. as-implemented, NOT Huffman-coded: their ratio would
improve if shipped), cosine similarity, MSE of decompress(compress(w)), and
encode/decode wall time — median of --rounds samples. Integrity gate: a second
decode must be bit-exact (SystemExit otherwise) — numbers are meaningless
without it.

Reading the results: `shipped_vq_k16` reporting near-zero MSE is the codec's
residual fallback firing (accuracy < residual_threshold=0.99 → the leftover is
stored too, compressor.py:357). That is a REAL shipped behavior, not a
benchmark artifact: on clean low-rank weight-like data, fixed k=16 + residual
dominates adaptive-k lossy VQ on BOTH ratio and fidelity — worth citing when
tuning n_clusters / residual_threshold for the model stack.

History appended to scripts/benchmark_vq_approaches.json (downcraft precedent).

Usage:
    ./scripts/python scripts/benchmark_vq_approaches.py [--rounds 3] [--no-history]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_core = str(Path(__file__).resolve().parents[1] / "packages" / "core-py")
if _core not in sys.path:
    sys.path.insert(0, _core)
_root = str(Path(__file__).resolve().parents[1])
if _root not in sys.path:
    sys.path.insert(0, _root)

from domain.infrastructure._internal.pugqeep.compressor import (  # noqa: E402
    PointCompressor,
)

HISTORY_PATH = Path(__file__).with_suffix(".json")
HISTORY_KEEP = 50


# ══════════════════════════════════════════════════════════════════════════════
# Data generation — realistic NN weight distributions
# ══════════════════════════════════════════════════════════════════════════════


def make_weight(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    """Generate realistic NN weights: near-zero mean, small variance, low-rank structure."""
    rng = np.random.RandomState(seed)
    w = rng.randn(*shape).astype(np.float32) * 0.02
    if len(shape) == 2 and shape[0] > 16 and shape[1] > 16:
        rank = min(16, min(shape))
        u = rng.randn(shape[0], rank).astype(np.float32) * 0.01
        v = rng.randn(rank, shape[1]).astype(np.float32) * 0.01
        w += u @ v
    return w


def make_test_suite() -> dict[str, np.ndarray]:
    return {
        "attn_qkv": make_weight((768, 2304)),
        "attn_out": make_weight((768, 768)),
        "ffn_up": make_weight((768, 3072)),
        "ffn_down": make_weight((3072, 768)),
        "ln_weight": make_weight((768,)),
        "bias": make_weight((768,)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Shipped codec rows — the real PointCompressor (cards vq-huffman-02/03)
# ══════════════════════════════════════════════════════════════════════════════


def bench_shipped(flat):
    c = PointCompressor()
    p = c.compress_cluster(flat, "bench", seed=0)
    return p, float(p.accuracy), p.nbytes()


def bench_shipped_k16(flat):
    c = PointCompressor(adaptive_k=False, n_clusters=16)
    p = c.compress_cluster(flat, "bench", seed=0)
    return p, float(p.accuracy), p.nbytes()


def decompress_shipped(r, n):
    return r[0].generate(n)


# ══════════════════════════════════════════════════════════════════════════════
# Legacy local VQ core — pre-card-02 baseline, kept as historical context
# ══════════════════════════════════════════════════════════════════════════════


def vq_core(
    flat: np.ndarray,
    n_clusters: int,
    lloyd_iters: int = 3,
    init: str = "quantile",
    rng: np.random.RandomState = None,
):
    """Legacy local VQ: returns (centroids, assignments, accuracy, nbytes)."""
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


def bench_legacy_quantile(flat):
    flat = flat.flatten()
    return vq_core(flat, 16, 3, "quantile")


def decompress_legacy(r, n):
    return r[0][r[1][:n]]


# ══════════════════════════════════════════════════════════════════════════════
# Exploration rows — PQ / RVQ / FSQ (un-Huffman-coded, as-implemented)
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
        recon_stage = c[a[: len(residual)]]
        residual = residual - recon_stage

    recon = np.zeros_like(flat)
    for s in range(n_stages):
        recon += all_c[s][all_a[s][: len(flat)]]

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


def bench_one(name: str, flat: np.ndarray, compress_fn, decompress_fn, rounds: int) -> dict:
    raw_bytes = flat.nbytes
    flat_1d = flat.flatten()
    n = len(flat_1d)

    comp_times: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        result = compress_fn(flat_1d)
        comp_times.append((time.perf_counter() - t0) * 1000)
    compress_ms = float(np.median(comp_times))

    decomp_times: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        decompressed = decompress_fn(result, n)
        decomp_times.append((time.perf_counter() - t0) * 1000)
    decompress_ms = float(np.median(decomp_times))

    # Integrity: decode must be deterministic — a second decode is bit-exact
    again = decompress_fn(result, n)
    if not np.array_equal(decompressed, again):
        raise SystemExit(
            f"decode non-deterministic for {name} — aborting (numbers would be meaningless)"
        )

    cosine = float(
        np.dot(flat_1d, decompressed)
        / (np.linalg.norm(flat_1d) * np.linalg.norm(decompressed) + 1e-10)
    )
    mse = float(np.mean((flat_1d - decompressed) ** 2))
    compressed_bytes = result[-1]
    return {
        "approach": name,
        "shape": list(flat.shape),
        "raw_bytes": int(raw_bytes),
        "compressed_bytes": int(compressed_bytes),
        "ratio": float(raw_bytes / max(compressed_bytes, 1)),
        "cosine": cosine,
        "mse": mse,
        "compress_ms": compress_ms,
        "decompress_ms": decompress_ms,
        "accuracy": float(result[-2]),
    }


def run_benchmarks(rounds: int, history: bool) -> dict:
    suite = make_test_suite()

    approaches = [
        ("shipped_vq", bench_shipped, decompress_shipped),
        ("shipped_vq_k16", bench_shipped_k16, decompress_shipped),
        ("legacy_quantile_k16", bench_legacy_quantile, decompress_legacy),
        ("pq_8x16", bench_pq, decompress_pq),
        ("rvq_3stage", bench_rvq, decompress_rvq),
        ("fsq_8level", bench_fsq, decompress_fsq),
    ]

    all_results: list[dict] = []

    print("=" * 95)
    print("VQ APPROACH BENCHMARK — SHIPPED CODEC + EXPLORATION (truth-up vq-huffman-06)")
    print("=" * 95)
    print(
        f"{'Approach':<22} {'Shape':<16} {'Ratio':>7} {'Cosine':>8} {'MSE':>12} {'Compress':>10} {'Decompress':>10}"
    )
    print("-" * 95)

    for _weight_name, flat in suite.items():
        for approach_name, compress_fn, decompress_fn in approaches:
            result = bench_one(approach_name, flat, compress_fn, decompress_fn, rounds)
            all_results.append(result)
            print(
                f"{approach_name:<22} {str(tuple(flat.shape)):<16} {result['ratio']:>7.2f}x "
                f"{result['cosine']:>8.5f} {result['mse']:>12.2e} "
                f"{result['compress_ms']:>8.1f}ms {result['decompress_ms']:>8.1f}ms"
            )
        print()

    # Summary
    print("=" * 95)
    print(f"AVERAGE BY APPROACH (median of {rounds} timing rounds each)")
    print("=" * 95)
    print(
        f"{'Approach':<22} {'Avg Ratio':>10} {'Avg Cosine':>11} {'Avg MSE':>12} {'Avg Compress':>13} {'Avg Decompress':>14}"
    )
    print("-" * 95)

    names = [a[0] for a in approaches]
    summary: dict[str, dict] = {}
    for name in names:
        rs = [r for r in all_results if r["approach"] == name]
        avg = {
            "ratio": float(np.mean([r["ratio"] for r in rs])),
            "cosine": float(np.mean([r["cosine"] for r in rs])),
            "mse": float(np.mean([r["mse"] for r in rs])),
            "compress_ms": float(np.mean([r["compress_ms"] for r in rs])),
            "decompress_ms": float(np.mean([r["decompress_ms"] for r in rs])),
        }
        summary[name] = avg
        print(
            f"{name:<22} {avg['ratio']:>10.2f}x {avg['cosine']:>11.5f} {avg['mse']:>12.2e} "
            f"{avg['compress_ms']:>11.1f}ms {avg['decompress_ms']:>12.1f}ms"
        )

    # Verdict
    print()
    print("=" * 95)
    print("VERDICT vs shipped_vq (the reference)")
    print("=" * 95)

    ref = summary["shipped_vq"]
    for name in names:
        if name == "shipped_vq":
            continue
        m = summary[name]
        d_ratio = m["ratio"] / max(ref["ratio"], 1e-9)
        d_mse = m["mse"] / max(ref["mse"], 1e-12)
        print(
            f"  {name:<22} ratio x{d_ratio:>5.2f} of shipped, MSE x{d_mse:>8.2f} of shipped, "
            f"encode {m['compress_ms']:.1f}ms vs {ref['compress_ms']:.1f}ms"
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
    run_benchmarks(args.rounds, history=not args.no_history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
