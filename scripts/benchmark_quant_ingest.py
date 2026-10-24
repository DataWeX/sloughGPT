#!/usr/bin/env python3
"""Quantize-on-ingest benchmark — the lossy int8 lever a byte codec cannot pull (card 019464d8).

A fp32/bf16 safetensors is byte-level incompressible (gzip/zstd ~1.0x, proven
in the codec benchmark), so the honest saving is *quantization at store time*:
int8 = ~4x vs fp32 at cosine ~0.9996. This measures that trade on a synthetic
fp32 "model" and proves the round-trip contract the loader depends on.

Measures, per file:
  * put_quantized ops/s (read + quantize + write + metadata upsert)
  * load_quantized ops/s (read + dequantize back to float32)
  * on-disk ratio (stored_size / fp32 source size)
  * worst-tensor cosine (fidelity floor — the honest cost of lossy)

Integrity is a hard gate: every dequantized tensor's cosine vs the original
must clear --min-cosine, else the run FAILs. This is opt-in lossy — the gate
is what keeps it honest.

Usage:
    ./scripts/python scripts/benchmark_quant_ingest.py [--files 8] [--mib 4]
        [--min-cosine 0.99] [--json [PATH]]
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from domain.files import FileStore, load_quantized  # noqa: E402

MIB = 1024 * 1024


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Minimal fp32 safetensors writer (8-byte LE header len + JSON + data)."""
    header: dict = {}
    blobs: list[bytes] = []
    off = 0
    for name in sorted(tensors):
        arr = tensors[name].astype("<f4")
        data = arr.tobytes()
        header[name] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    h = json.dumps(header).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def make_fp32_model(path: Path, mib: int) -> dict[str, np.ndarray]:
    """A weight-shaped fp32 model totalling ~mib MiB (2D mats + 1D biases)."""
    rng = np.random.default_rng(7)
    target = mib * MIB
    tensors: dict[str, np.ndarray] = {}
    i = 0
    total = 0
    while total < target:
        if i % 3 == 2:  # a bias so per-tensor (non-per-channel) path is covered
            arr = rng.normal(0, 0.05, (512,)).astype(np.float32)
        else:
            arr = rng.normal(0, 0.05, (512, 512)).astype(np.float32)
        tensors[f"blocks.{i}.weight"] = arr
        total += arr.nbytes
        i += 1
    write_safetensors(path, tensors)
    return tensors


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description="Quantize-on-ingest benchmark")
    ap.add_argument("--files", type=int, default=8, help="files to quantize")
    ap.add_argument("--mib", type=int, default=4, help="source size per file, MiB")
    ap.add_argument("--min-cosine", type=float, default=0.99, help="fidelity floor (hard gate)")
    ap.add_argument(
        "--json",
        nargs="?",
        const=Path(__file__).resolve().parent / "benchmark_quant_ingest.json",
        type=Path,
        default=None,
    )
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="quant-ingest-bench-"))
    try:
        store = FileStore(root=tmp / "uploads")
        srcs = [tmp / f"src{i}.safetensors" for i in range(args.files)]
        tensors = [make_fp32_model(p, args.mib) for p in srcs]

        # ── quantize-on-ingest: read + quantize + write + metadata ──
        t0 = time.perf_counter()
        metas = [store.put_quantized(p, f"model{i}.safetensors") for i, p in enumerate(srcs)]
        quant_s = time.perf_counter() - t0

        # ── load: read + dequantize back to float32 ──
        t0 = time.perf_counter()
        restored = [load_quantized(store.root / m["filename"]) for m in metas]
        load_s = time.perf_counter() - t0

        # ── fidelity: every tensor must clear the cosine floor ──
        worst = 1.0
        bad: list[str] = []
        for i, (orig, got) in enumerate(zip(tensors, restored)):
            for name, oarr in orig.items():
                c = cosine(oarr, got[name])
                worst = min(worst, c)
                if c < args.min_cosine:
                    bad.append(f"file{i}:{name}={c:.4f}")

        fp32_bytes = sum(a.nbytes for t in tensors for a in t.values())
        stored = sum(m["stored_size"] for m in metas)
        n = args.files
        ratios = [m["quantization"]["ratio"] for m in metas]
        codecs = {m["codec"] for m in metas}
        qbits = {m["quantization"]["bits"] for m in metas}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"payload: {args.files} files x ~{args.mib} MiB fp32 safetensors")
    print(f"  quantize-on-ingest : {n / quant_s:6.1f} files/s   ({fp32_bytes / quant_s / MIB:5.1f} MiB/s)")
    print(f"  load (dequant)     : {n / load_s:6.1f} files/s   ({fp32_bytes / load_s / MIB:5.1f} MiB/s)")
    print(f"  on-disk ratio      : {stored / fp32_bytes:6.4f}x fp32  (mean per-file {sum(ratios) / len(ratios):.2f}x)")
    print(f"  codecs/bits        : {','.join(sorted(codecs))} / {sorted(qbits)}-bit")
    print(f"  worst cosine       : {worst:6.4f}   (floor {args.min_cosine})")

    ok = not bad and worst >= args.min_cosine and stored / fp32_bytes < 0.4
    print(
        "\ngate: every tensor clears cosine floor AND on-disk < 0.4x fp32: "
        + ("PASS" if ok else "FAIL")
    )
    if bad:
        print("  below floor: " + ", ".join(bad[:5]) + (" ..." if len(bad) > 5 else ""))

    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "files": n,
                    "mib": args.mib,
                    "quant_ops_s": round(n / quant_s, 1),
                    "load_ops_s": round(n / load_s, 1),
                    "ratio": round(stored / fp32_bytes, 4),
                    "worst_cosine": round(worst, 5),
                    "pass": ok,
                },
                indent=2,
            )
        )
        print(f"json: {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
