#!/usr/bin/env python3
"""FileStore benchmark — at-rest ratio + put/get throughput per payload class.

Proves the store half of the codec policy (card ff2ecade, on top of the
codec proof fba1ce5e): text/code files must show the zstd compress-at-rest
win on disk, model-weight artifacts must show raw passthrough (zero CPU
wasted on incompressible bytes).

Measures, per class:
  * put ops/s (chunked write + codec + metadata upsert)
  * get ops/s (read + decompress + sha256 verify)
  * aggregate on-disk ratio (sum stored_size / sum size_bytes)
  * download-stream throughput (iter_chunks — what /download serves)

Integrity is a hard gate: every get() is sha256-verified by the store
itself, and the benchmark re-verifies one sample per class.

Usage:
    ./scripts/python scripts/benchmark_file_store.py [--files 40] [--json [PATH]]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from domain.files import FileStore  # noqa: E402


def make_code(size: int, seed: int) -> bytes:
    """Source-like text with per-line variety (lands near real-source ratio)."""
    out = bytearray()
    j = 0
    while len(out) < size:
        tok = f"{(seed * 1000003 + j) % 0xFFFFFFFF:08x}"
        out.extend(
            f"def handler_{seed}_{j}(event, ctx=None):\n    # tok={tok}\n    return {{'ok': True, 'n': {j}}}\n".encode()
        )
        j += 1
    return bytes(out[:size])


def make_model(size: int) -> bytes:
    """Incompressible weight-like binary (os.urandom)."""
    import os

    return os.urandom(size)


def bench_class(store: FileStore, name: str, blobs: list[tuple[str, bytes]]) -> dict:
    t0 = time.perf_counter()
    metas = [store.put(data, fname) for fname, data in blobs]
    put_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for meta, (_, data) in zip(metas, blobs):
        got, _ = store.get(meta["file_id"])
        if got != data:
            raise SystemExit(f"INTEGRITY FAIL: {meta['file_id']}")
    get_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    stream_bytes = 0
    for meta in metas:
        stream_bytes += sum(len(c) for c in store.iter_chunks(meta["file_id"]))
    stream_s = time.perf_counter() - t0

    raw = sum(m["size_bytes"] for m in metas)
    stored = sum(m["stored_size"] for m in metas)
    codecs = {m["codec"] for m in metas}
    n = len(metas)

    t0 = time.perf_counter()
    for meta in metas:
        store.delete(meta["file_id"])
    del_s = time.perf_counter() - t0

    return {
        "class": name,
        "files": n,
        "raw_bytes": raw,
        "stored_bytes": stored,
        "ratio": round(stored / raw, 4) if raw else 1.0,
        "codecs": sorted(codecs),
        "put_ops_s": round(n / put_s, 1) if put_s else None,
        "get_ops_s": round(n / get_s, 1) if get_s else None,
        "stream_mibs": round(stream_bytes / (1024 * 1024) / stream_s, 1) if stream_s else None,
        "del_ops_s": round(n / del_s, 1) if del_s else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="FileStore benchmark")
    ap.add_argument("--files", type=int, default=40, help="files per class")
    ap.add_argument("--kib", type=int, default=64, help="file size in KiB")
    ap.add_argument(
        "--json",
        nargs="?",
        const=Path(__file__).resolve().parent / "benchmark_file_store.json",
        type=Path,
        default=None,
    )
    args = ap.parse_args()

    size = args.kib * 1024
    tmp = Path(tempfile.mkdtemp(prefix="filestore-bench-"))
    try:
        store = FileStore(root=tmp / "uploads")
        rows = [
            bench_class(store, "code(text/zstd)", [(f"h{i}.py", make_code(size, i)) for i in range(args.files)]),
            bench_class(store, "model(raw)", [(f"w{i}.gguf", make_model(size)) for i in range(args.files)]),
        ]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"payload: {args.files} files x {args.kib} KiB per class")
    print(
        f"  {'class':<16} {'ratio':>7} {'codecs':<14} "
        f"{'put/s':>8} {'get/s':>8} {'stream MiB/s':>12} {'del/s':>8}"
    )
    for r in rows:
        print(
            f"  {r['class']:<16} {r['ratio']:>7.4f} {','.join(r['codecs']):<14} "
            f"{str(r['put_ops_s']):>8} {str(r['get_ops_s']):>8} "
            f"{str(r['stream_mibs']):>12} {str(r['del_ops_s']):>8}"
        )

    code_row, model_row = rows
    ok = code_row["ratio"] < 0.5 and model_row["ratio"] >= 0.99
    print(
        "\ngate: code compresses (<0.5x) AND model stays raw (>=0.99x): "
        + ("PASS" if ok else "FAIL")
    )

    if args.json is not None:
        args.json.write_text(
            json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": rows}, indent=2)
        )
        print(f"json: {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
