#!/usr/bin/env python3
"""
Benchmark: compression-knob frontier — n_clusters x residual_threshold x adaptive_k
on the REAL PointCompressor (vq-huffman-07, card 6eaa413c).

vq-huffman-06 found one point: fixed k=16 + the codec's residual fallback
(accuracy < residual_threshold=0.99 -> leftover stored too, compressor.py:357)
dominated adaptive-k lossy VQ on weight-like data — 8.22x ratio AND lossless vs
adaptive's 4.91x lossy. This script maps the FRONTIER around that point so the
model stack can pick defaults with evidence instead of vibes.

Grid (15 configs, real codec, seed=0, shipped quantize_centroids):

  fixed_k{8,16,32,64}_rt{0,0.99,1}   adaptive_k off, k fixed.
      rt semantics ("store residual when accuracy < rt"):
        rt=0    never  — pure lossy VQ at that k
        rt=0.99 default shipped threshold
        rt=1    always — VQ + residual (near-lossless class)
  adaptive_k16_rt{0,0.99,1}          shipped default is rt0.99 (reference row).

Measures per config x per BERT-scale layer (suite imported from
benchmark_vq_approaches): compression ratio (raw float32 / stored bytes incl.
Huffman bitstream AND any stored residual), cosine, MSE of decode(encode(w)),
residual actually fired (reported accuracy < rt), encode/decode wall time.
Integrity gate: a second decode must be bit-exact (SystemExit otherwise).

MEASUREMENT TRUTH (discovered by vq-huffman-07, FIXED by card d9028c8f,
2026-10-21): ratio is computed from point.to_bytes() — the on-disk/wire
layout, which packs the residual. The pre-fix ClusterCodec.nbytes omitted
the residual in the Huffman and quantized branches (which compress_cluster
always takes), so every residual-carrying point's reported nbytes was short
by residual.nbytes (= the FULL raw size, float32), and the Huffman bitstream
itself never survived serialization. nbytes is now the honest payload
contract — residual in every branch, and the Huffman bitstream persists on
the wire under the CLH1 marker. Reported-vs-true ratios are kept side by
side in the JSON (nbytes_gap) — post-fix the gap is wire headers only.

Output: compact per-config frontier table (6-layer averages) + best-config-per-
fidelity-class verdict vs the shipped default. Per-layer rows live in JSON
history only (summarize, don't dump). Recommendation only — ADOPTING new
defaults in CompressorConfig is out of scope, that is a user decision.

Usage:
    ./scripts/python scripts/benchmark_compression_knobs.py [--no-history]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
_root = _here.parent
for _p in (str(_root / "packages" / "core-py"), str(_root), str(_here)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark_vq_approaches import make_test_suite  # noqa: E402

from domain.infrastructure._internal.pugqeep.compressor import (  # noqa: E402
    PointCompressor,
)

HISTORY_PATH = Path(__file__).with_suffix(".json")
HISTORY_KEEP = 50

SHIPPED_DEFAULT = "adaptive_k16_rt0.99"


# ══════════════════════════════════════════════════════════════════════════════
# Grid
# ══════════════════════════════════════════════════════════════════════════════


def build_grid() -> list[tuple[str, dict]]:
    configs: list[tuple[str, dict]] = []
    for k in (8, 16, 32, 64):
        for rt in (0.0, 0.99, 1.0):
            configs.append(
                (
                    f"fixed_k{k}_rt{rt:g}",
                    {
                        "adaptive_k": False,
                        "n_clusters": k,
                        "residual_threshold": rt,
                    },
                )
            )
    for rt in (0.0, 0.99, 1.0):
        configs.append(
            (
                f"adaptive_k16_rt{rt:g}",
                {
                    "adaptive_k": True,
                    "n_clusters": 16,
                    "residual_threshold": rt,
                },
            )
        )
    return configs


# ══════════════════════════════════════════════════════════════════════════════
# Measurement
# ══════════════════════════════════════════════════════════════════════════════


def measure(config_name: str, kw: dict, layer_name: str, flat: np.ndarray) -> dict:
    flat_1d = flat.flatten()
    n = len(flat_1d)
    raw_bytes = int(flat.nbytes)

    t0 = time.perf_counter()
    point = PointCompressor(**kw).compress_cluster(flat_1d, "knobs", seed=0)
    encode_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    decoded = point.generate(n)
    decode_ms = (time.perf_counter() - t0) * 1000

    # integrity: decode must be deterministic — a second decode is bit-exact
    again = point.generate(n)
    if not np.array_equal(decoded, again):
        raise SystemExit(f"decode non-deterministic for {config_name}/{layer_name} — aborting")

    # Ground truth = to_bytes() (serialize_params packs the residual).
    # Pre-d9028c8f, point.nbytes() UNDERCOUNTED residual-carrying points
    # (residual omitted in the Huffman/quantized branches) — the reported
    # ratio was inflated. nbytes is now the honest payload contract; the
    # remaining gap is wire headers only. Keep both numbers side by side:
    # the gap doubles as a tripwire against accounting regressions.
    stored_bytes = int(len(point.to_bytes()))
    reported_nbytes = int(point.nbytes())
    acc = float(point.accuracy)
    rt = float(kw["residual_threshold"])
    return {
        "config": config_name,
        "layer": layer_name,
        "shape": list(flat.shape),
        "raw_bytes": raw_bytes,
        "stored_bytes": stored_bytes,
        "reported_nbytes": reported_nbytes,
        "nbytes_gap": stored_bytes - reported_nbytes,
        "ratio": float(raw_bytes / max(stored_bytes, 1)),
        "cosine": float(
            np.dot(flat_1d, decoded) / (np.linalg.norm(flat_1d) * np.linalg.norm(decoded) + 1e-10)
        ),
        "mse": float(np.mean((flat_1d - decoded) ** 2)),
        "accuracy": acc,
        "residual_fired": bool(acc < rt),
        "encode_ms": encode_ms,
        "decode_ms": decode_ms,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════


def run(history: bool) -> dict:
    suite = make_test_suite()
    grid = build_grid()

    rows: list[dict] = []
    print("=" * 100)
    print("COMPRESSION KNOB FRONTIER — real PointCompressor (vq-huffman-07)")
    print("=" * 100)

    for config_name, kw in grid:
        for layer_name, flat in suite.items():
            rows.append(measure(config_name, kw, layer_name, flat))
        fired = sum(1 for r in rows if r["config"] == config_name and r["residual_fired"])
        print(f"  measured {config_name:<22} ({fired}/6 layers residual fired)")

    # per-config frontier table (6-layer averages)
    print()
    print("=" * 100)
    print("FRONTIER (avg over 6 BERT-scale layers)")
    print("=" * 100)
    print(
        f"{'config':<22} {'ratio':>7} {'cosine':>8} {'min-cos':>8} {'MSE':>11} "
        f"{'resid':>5} {'encode':>9}"
    )
    print("-" * 100)

    summary: dict[str, dict] = {}
    for config_name, _ in grid:
        rs = [r for r in rows if r["config"] == config_name]
        s = {
            "ratio": float(np.mean([r["ratio"] for r in rs])),
            "reported_ratio": float(
                np.mean([r["raw_bytes"] / max(r["reported_nbytes"], 1) for r in rs])
            ),
            "nbytes_gap": int(sum(r["nbytes_gap"] for r in rs)),
            "cosine": float(np.mean([r["cosine"] for r in rs])),
            "min_cosine": float(min(r["cosine"] for r in rs)),
            "mse": float(np.mean([r["mse"] for r in rs])),
            "residual_fired": int(sum(r["residual_fired"] for r in rs)),
            "encode_ms": float(np.mean([r["encode_ms"] for r in rs])),
            "decode_ms": float(np.mean([r["decode_ms"] for r in rs])),
        }
        summary[config_name] = s
        star = "  <- shipped default" if config_name == SHIPPED_DEFAULT else ""
        print(
            f"{config_name:<22} {s['ratio']:>6.2f}x {s['cosine']:>8.5f} "
            f"{s['min_cosine']:>8.5f} {s['mse']:>11.2e} "
            f"{s['residual_fired']:>3}/6 {s['encode_ms']:>7.0f}ms{star}"
        )

    # verdict: best ratio within each fidelity class vs shipped default
    # tripwire: post-d9028c8f, nbytes is honest payload — the gap vs the
    # wire should be headers only (~50 B/point, ~300 B per 6-layer config).
    # Anything approaching residual scale (>512/config) is a regression.
    gaps = [n for n, s in summary.items() if s["nbytes_gap"] > 512]
    if gaps:
        print(
            f"\n  NOTE: point.nbytes() understates the wire by >512 B for "
            f"{len(gaps)}/{len(grid)} configs — EXPECTED ~300 B (headers only) "
            "post-d9028c8f; investigate as an accounting regression."
        )

    def best_in(pred) -> str:
        eligible = [n for n, s in summary.items() if pred(s)]
        return max(eligible, key=lambda n: summary[n]["ratio"]) if eligible else "n/a"

    classes = {
        "near-lossless (avg MSE < 1e-12)": lambda s: s["mse"] < 1e-12,
        "high-fidelity (avg cosine >= 0.999, min >= 0.995)": (
            lambda s: s["cosine"] >= 0.999 and s["min_cosine"] >= 0.995
        ),
        "any (max ratio)": lambda s: True,
    }
    ref = summary[SHIPPED_DEFAULT]
    verdict: dict[str, str] = {}
    print()
    print("=" * 100)
    print(
        f"VERDICT vs shipped default ({SHIPPED_DEFAULT}: ratio "
        f"{ref['ratio']:.2f}x, cosine {ref['cosine']:.5f}, MSE {ref['mse']:.2e})"
    )
    print("=" * 100)
    for cls, pred in classes.items():
        winner = best_in(pred)
        if winner == "n/a":
            print(f"  {cls:<52} : no config qualifies")
            verdict[cls] = "n/a"
            continue
        w = summary[winner]
        d_ratio = w["ratio"] / max(ref["ratio"], 1e-9)
        d_mse = w["mse"] / max(ref["mse"], 1e-15)
        note = (
            ""
            if winner == SHIPPED_DEFAULT
            else f"  ({d_ratio:.2f}x ratio of ref, MSE x{d_mse:.2g})"
        )
        print(f"  {cls:<52} : {winner}{note}")
        verdict[cls] = winner

    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "grid": [n for n, _ in grid],
        "shipped_default": SHIPPED_DEFAULT,
        "rows": rows,
        "summary": summary,
        "verdict": verdict,
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
        "--no-history", action="store_true", help="skip appending to the results JSON"
    )
    args = parser.parse_args()
    run(history=not args.no_history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
