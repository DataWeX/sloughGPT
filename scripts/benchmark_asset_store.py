"""Asset store throughput benchmark — slice 5 of the asset-store campaign.

Measures the unified asset store's hot paths on a SCRATCH directory (never
the real data dir) so regressions in the storage layer are caught by numbers,
not vibes:

- ``put_cold``      distinct payloads -> CAS + metadata write throughput
                    (files/s, MB/s): the upload / generate path.
- ``put_dedupe``    identical payload repeated -> the content-addressed
                    fast path (blob already exists, metadata-only write).
- ``get_cold``      reads with ``cache_bytes=0`` -> raw blob reads (disk path).
- ``get_warm``      second pass with the default 64MB LRU -> cache-hit path;
                    the gap vs ``get_cold`` is what the slice-4 cache buys.
- ``migrate``       one-shot legacy adoption (``migrate_uploads``): what the
                    first request after an upgrade pays to bring existing
                    ``data/uploads`` files into the store.

Standalone by design: ``benchmark_results.py record --kind`` predates storage
kinds (stability|latency|execution|training|startup); wiring a sixth kind
touches its history/compare schema and earns its own card. ``--json`` emits a
machine-readable record for that future wiring (and for CI diffing).

Usage::

    ./scripts/python scripts/benchmark_asset_store.py
    ./scripts/python scripts/benchmark_asset_store.py --count 2000 --size 262144
    ./scripts/python scripts/benchmark_asset_store.py --json /tmp/asset_store_bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from domain.infrastructure.asset_store import AssetStore  # noqa: E402


def _rate(count: int, seconds: float) -> float:
    return count / seconds if seconds > 0 else 0.0


def _mbps(total_bytes: int, seconds: float) -> float:
    return total_bytes / seconds / 1e6 if seconds > 0 else 0.0


def run_benchmarks(count: int, size: int) -> dict:
    scratch = Path(tempfile.mkdtemp(prefix="slo-asset-bench-"))
    workloads: dict[str, dict] = {}
    try:
        payloads = [os.urandom(size) for _ in range(count)]
        total_bytes = count * size

        # put_cold — distinct bytes: full CAS + metadata path.
        store = AssetStore(scratch / "cold")
        t0 = time.perf_counter()
        for i in range(count):
            store.put(payloads[i], original_name=f"cold_{i}.bin", origin="upload")
        dt = time.perf_counter() - t0
        workloads["put_cold"] = {
            "seconds": round(dt, 4),
            "ops_per_sec": round(_rate(count, dt), 1),
            "mb_per_sec": round(_mbps(total_bytes, dt), 1),
        }

        # put_dedupe — one blob written once, metadata-only writes after.
        dedupe = AssetStore(scratch / "dedupe")
        blob = payloads[0]
        t0 = time.perf_counter()
        for i in range(count):
            dedupe.put(blob, original_name=f"same_{i}.bin", origin="upload")
        dt = time.perf_counter() - t0
        workloads["put_dedupe"] = {
            "seconds": round(dt, 4),
            "ops_per_sec": round(_rate(count, dt), 1),
            "mb_per_sec": round(_mbps(total_bytes, dt), 1),
        }

        # get_cold — cache disabled: every read touches the blob on disk.
        ids = [m.id for m in store.list()]
        nocache = AssetStore(scratch / "cold", cache_bytes=0)
        t0 = time.perf_counter()
        for asset_id in ids:
            assert nocache.get_bytes(asset_id) is not None
        dt = time.perf_counter() - t0
        workloads["get_cold"] = {
            "seconds": round(dt, 4),
            "ops_per_sec": round(_rate(count, dt), 1),
            "mb_per_sec": round(_mbps(total_bytes, dt), 1),
        }

        # get_warm — default LRU: first pass primes, second pass measures hits.
        warm = AssetStore(scratch / "cold", cache_bytes=64 * 1024 * 1024)
        for asset_id in ids:
            warm.get_bytes(asset_id)
        t0 = time.perf_counter()
        for asset_id in ids:
            assert warm.get_bytes(asset_id) is not None
        dt = time.perf_counter() - t0
        workloads["get_warm"] = {
            "seconds": round(dt, 4),
            "ops_per_sec": round(_rate(count, dt), 1),
            "mb_per_sec": round(_mbps(total_bytes, dt), 1),
        }

        # migrate — legacy adoption of a data/uploads-shaped directory.
        legacy = scratch / "uploads"
        legacy.mkdir()
        entries = []
        for i in range(count):
            path = legacy / f"{i:06d}.bin"
            path.write_bytes(payloads[i])
            entries.append((path, {"filename": path.name, "original_name": path.name, "tags": []}))
        migrating = AssetStore(scratch / "migrate")
        t0 = time.perf_counter()
        stats = migrating.migrate_uploads(entries)
        dt = time.perf_counter() - t0
        assert stats["migrated"] == count, stats
        workloads["migrate"] = {
            "seconds": round(dt, 4),
            "ops_per_sec": round(_rate(count, dt), 1),
            "mb_per_sec": round(_mbps(total_bytes, dt), 1),
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    return {
        "kind": "asset_store",
        "ts": time.time(),
        "count": count,
        "size_bytes": size,
        "workloads": workloads,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=500, help="assets per workload")
    parser.add_argument("--size", type=int, default=65536, help="bytes per asset")
    parser.add_argument("--json", default=None, help="write the record to this JSON file")
    args = parser.parse_args()

    record = run_benchmarks(args.count, args.size)
    w = record["workloads"]
    print(f"asset-store benchmark — {args.count} assets x {args.size} B (scratch dir)")
    print(f"{'workload':<12} {'seconds':>9} {'ops/s':>10} {'MB/s':>9}")
    for name in ("put_cold", "put_dedupe", "get_cold", "get_warm", "migrate"):
        row = w[name]
        print(
            f"{name:<12} {row['seconds']:>9.4f} {row['ops_per_sec']:>10.1f} {row['mb_per_sec']:>9.1f}"
        )
    gap = (
        w["get_cold"]["ops_per_sec"] and w["get_warm"]["ops_per_sec"] / w["get_cold"]["ops_per_sec"]
    )
    print(f"cache speedup (get_warm / get_cold): {gap:.1f}x" if gap else "")

    if args.json:
        Path(args.json).write_text(json.dumps(record, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
