#!/usr/bin/env python3
"""Benchmark: edge compression gateway — bytes-on-wire vs identity.

Measures gzip/zstd ratio and encode throughput on synthetic multi-MB
payloads. Mirrors apps/gateway/src/compression.rs codecs.

Two payload classes:
    download  semi-structured records — file-download traffic (default)
    sse       model token stream — `data:` frames of generation deltas,
              the `text/event-stream` traffic the edge compresses from
              byte 0 (streaming, no Content-Length)

Usage:
    .venv/bin/python scripts/benchmark_gateway_compression.py [--mib 64]
    .venv/bin/python scripts/benchmark_gateway_compression.py --class sse --mib 8
    .venv/bin/python scripts/benchmark_gateway_compression.py --file checkpoints/step_25.pt
    .venv/bin/python scripts/benchmark_gateway_compression.py --file apps/web/public/v86/v86-fallback.wasm
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import time
from pathlib import Path

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover
    zstd = None


def make_payload(mib: int) -> bytes:
    """Semi-structured records — realistic for logs/JSONL/CSV downloads."""
    line = "id=%08d name=item-%d status=ok tags=a,b,c payload=%s\n"
    pad = "x" * 200
    target = mib * 1024 * 1024
    out = bytearray()
    i = 0
    while len(out) < target:
        out.extend((line % (i, i, pad)).encode())
        i += 1
    return bytes(out[:target])


def make_sse_payload(mib: int) -> bytes:
    """Synthetic model token stream — SSE `data:` frames of generation deltas.

    The `text/event-stream` traffic class: streaming, no Content-Length,
    highly redundant JSON lines (repetitive keys, incrementing indices).
    """
    frame = (
        'data: {"type":"delta","idx":%08d,"content":"the quick brown fox jumps '
        'over the lazy dog token %d padding padding","finish":false}\n\n'
    )
    target = mib * 1024 * 1024
    out = bytearray()
    i = 0
    while len(out) < target:
        out.extend((frame % (i, i)).encode())
        i += 1
    return bytes(out[:target])


def bench_gzip(data: bytes, level: int = 6) -> dict:
    t0 = time.perf_counter()
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=level, mtime=0) as f:
        f.write(data)
    elapsed = time.perf_counter() - t0
    encoded = buf.getvalue()
    t0 = time.perf_counter()
    gzip.decompress(encoded)
    dec = time.perf_counter() - t0
    return {
        "codec": f"gzip-{level}",
        "raw_bytes": len(data),
        "wire_bytes": len(encoded),
        "ratio": round(len(encoded) / len(data), 4),
        "savings_pct": round((1 - len(encoded) / len(data)) * 100, 2),
        "enc_mibs": round(len(data) / (1024 * 1024) / elapsed, 1) if elapsed else None,
        "dec_mibs": round(len(data) / (1024 * 1024) / dec, 1) if dec else None,
    }


def bench_zstd(data: bytes, level: int = 3) -> dict | None:
    if zstd is None:
        return None
    cctx = zstd.ZstdCompressor(level=level)
    t0 = time.perf_counter()
    encoded = cctx.compress(data)
    elapsed = time.perf_counter() - t0
    dctx = zstd.ZstdDecompressor()
    t0 = time.perf_counter()
    dctx.decompress(encoded)
    dec = time.perf_counter() - t0
    return {
        "codec": f"zstd-{level}",
        "raw_bytes": len(data),
        "wire_bytes": len(encoded),
        "ratio": round(len(encoded) / len(data), 4),
        "savings_pct": round((1 - len(encoded) / len(data)) * 100, 2),
        "enc_mibs": round(len(data) / (1024 * 1024) / elapsed, 1) if elapsed else None,
        "dec_mibs": round(len(data) / (1024 * 1024) / dec, 1) if dec else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mib", type=int, default=64, help="payload size in MiB")
    ap.add_argument(
        "--class",
        dest="payload_class",
        choices=["download", "sse"],
        default="download",
        help="payload class: download (records) or sse (model token stream)",
    )
    ap.add_argument(
        "--file",
        type=Path,
        default=None,
        help="real artifact (.pt/.so/.wasm/…) — measure actual entropy, not synthetic",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("scripts/benchmark_gateway_compression.json"),
        help="result history path",
    )
    ap.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="write nested metrics JSON for benchmark_results.py record --kind compression",
    )
    args = ap.parse_args()

    if args.file is not None:
        data = args.file.read_bytes()
        label = str(args.file)
        source = "file"
    elif args.payload_class == "sse":
        data = make_sse_payload(args.mib)
        label = f"{args.mib} MiB synthetic SSE token stream"
        source = "synthetic"
    else:
        data = make_payload(args.mib)
        label = f"{args.mib} MiB synthetic download records"
        source = "synthetic"

    results = [bench_gzip(data), bench_zstd(data)]
    results = [r for r in results if r]

    # identity baseline
    results.insert(
        0,
        {
            "codec": "identity",
            "raw_bytes": len(data),
            "wire_bytes": len(data),
            "ratio": 1.0,
            "savings_pct": 0.0,
            "enc_mibs": None,
            "dec_mibs": None,
        },
    )

    print(f"payload: {label} ({source}, {len(data)} bytes)")
    print(
        f"{'codec':<12} {'wire':>12} {'ratio':>8} {'savings':>9} {'enc MiB/s':>10} {'dec MiB/s':>10}"
    )
    for r in results:
        print(
            f"{r['codec']:<12} {r['wire_bytes']:>12} {r['ratio']:>8.4f} "
            f"{r['savings_pct']:>8.1f}% {str(r['enc_mibs']):>10} {str(r['dec_mibs']):>10}"
        )

    history = []
    if args.out.exists():
        try:
            history = json.loads(args.out.read_text())
        except json.JSONDecodeError:
            history = []
    entry = {
        "mib": args.mib if args.file is None else None,
        "class": args.payload_class if args.file is None else "file",
        "results": results,
    }
    if args.file is not None:
        entry["file"] = str(args.file)
        entry["raw_bytes"] = len(data)
    history.append(entry)
    args.out.write_text(json.dumps(history, indent=2) + "\n")
    print(f"\nrecorded → {args.out}")

    if args.metrics_out is not None:
        metrics = {
            "payload": {
                "class": entry["class"],
                "mib": entry["mib"],
                "raw_bytes": len(data),
            }
        }
        for r in results:
            # normalise codec keys ("gzip-6"/"zstd-3") so thresholds stay stable
            metrics[r["codec"].split("-")[0]] = {
                "ratio": r["ratio"],
                "savings_pct": r["savings_pct"],
                "enc_mibs": r["enc_mibs"],
                "dec_mibs": r["dec_mibs"],
            }
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"metrics → {args.metrics_out}")


if __name__ == "__main__":
    main()
