"""
Quantized weight-load benchmark (cards 20261103_082215 / 20261103_085029).

Measures ``Quantine.load_weights`` on the real pre-quantized artifact and
compares a serial inflate against the parallel one, in the same process so the
page cache and the machine are identical for both. Wall time is the gate; peak
RSS is reported but not enforced (allocator retention makes it noisy).

Why this exists: the archive is 338 DEFLATE entries (505 MB raw / 421 MB
stored, ratio 1.15× — int8 weights barely compress). One zlib stream inflates
at ~82 MB/s, so the serial loop cost ~8 s of every cold start, and the old
unconditional ``.copy(order="C")`` was a redundant 505 MB memcpy on top.

Usage:
    ./scripts/python scripts/benchmark_model_load.py
    ./scripts/python scripts/benchmark_model_load.py --json out.json
    ./scripts/python scripts/benchmark_model_load.py --serial-first   # safer peak RSS
    ./scripts/python scripts/benchmark_model_load.py --no-serial      # parallel only
Exit code is non-zero when parallel is not at least ``--min-speedup`` faster
than serial, or exceeds ``--max-seconds`` (defaults: 2.0×, 3.0 s).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in (
    str(_REPO),
    str(_REPO / "packages" / "downcraft"),
    str(_REPO / "packages" / "core-py"),
    str(_REPO / "packages" / "mogdb" / "src"),
    str(_REPO / "apps" / "api" / "server"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_DEFAULT_ARCHIVE = (
    _REPO
    / "models"
    / "hf-cache"
    / "hub"
    / "models--Qwen--Qwen2.5-0.5B-Instruct"
    / "model.slnc.quant.npz"
)


def _rss_mb() -> float:
    """Current RSS in MB (Linux); 0.0 when unavailable."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _peak_rss_mb() -> float:
    """VmHWM (high-water mark) in MB; 0.0 when unavailable."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _run(archive: Path, workers: int) -> dict:
    """Load the archive once with a fixed worker count."""
    from domain.infrastructure._internal.quantization import Quantine

    os.environ["SLO_QUANT_LOAD_WORKERS"] = str(workers)
    engine = Quantine(bits=8, mode="asymmetric")
    started = time.perf_counter()
    infos = engine.load_weights(str(archive))
    elapsed = time.perf_counter() - started

    tensors = len(infos)
    total_bytes = sum(int(info.array.nbytes) for info in infos.values())
    non_contiguous = sum(1 for info in infos.values() if not info.array.flags.c_contiguous)
    result = {
        "workers": workers,
        "seconds": round(elapsed, 3),
        "tensors": tensors,
        "raw_mb": round(total_bytes / 1e6, 1),
        "throughput_mb_s": round((total_bytes / 1e6) / elapsed, 1) if elapsed > 0 else 0.0,
        "non_contiguous": non_contiguous,
        "rss_mb": round(_rss_mb(), 1),
        "peak_rss_mb": round(_peak_rss_mb(), 1),
    }
    del infos, engine
    gc.collect()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--archive", default=str(_DEFAULT_ARCHIVE))
    ap.add_argument(
        "--workers", type=int, default=0, help="override worker count (0 = Quantine default)"
    )
    ap.add_argument("--no-serial", action="store_true", help="skip the serial baseline")
    ap.add_argument("--serial-first", action="store_true", help="run serial before parallel")
    ap.add_argument("--min-speedup", type=float, default=2.0)
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="absolute wall-time gate; 0 disables it (wall time tracks core count and "
        "thermal headroom — the speedup ratio is the machine-independent invariant)",
    )
    ap.add_argument("--json", default="", help="write results to this path")
    args = ap.parse_args()

    archive = Path(args.archive)
    if not archive.exists():
        print(f"archive not found: {archive}", file=sys.stderr)
        return 2

    from domain.infrastructure._internal.quantization import Quantine

    parallel_workers = args.workers or Quantine._load_worker_count(10_000)
    print(f"archive: {archive}")
    print(f"size:    {archive.stat().st_size / 1e6:.1f} MB on disk")
    print(f"workers: {parallel_workers} (cpu_count={os.cpu_count()})")

    runs: list[dict] = []
    if not args.no_serial and args.serial_first:
        runs.append(_run(archive, 1))
        runs.append(_run(archive, parallel_workers))
    else:
        runs.append(_run(archive, parallel_workers))
        if not args.no_serial:
            runs.append(_run(archive, 1))

    by_workers = {run["workers"]: run for run in runs}
    parallel = by_workers[parallel_workers]
    serial = by_workers.get(1)

    print()
    print(f"{'mode':<10} {'seconds':>8} {'MB/s':>8} {'tensors':>8} {'peak RSS':>10}")
    for run in runs:
        label = "serial" if run["workers"] == 1 else f"par x{run['workers']}"
        print(
            f"{label:<10} {run['seconds']:>8.3f} {run['throughput_mb_s']:>8.1f} "
            f"{run['tensors']:>8} {run['peak_rss_mb']:>9.0f}M"
        )

    failures: list[str] = []
    if parallel["tensors"] == 0:
        failures.append("parallel load returned no tensors")
    if parallel["non_contiguous"]:
        failures.append(f"{parallel['non_contiguous']} tensors not C-contiguous")
    if parallel["seconds"] > args.max_seconds > 0:
        failures.append(
            f"parallel load {parallel['seconds']:.2f}s exceeds {args.max_seconds:.2f}s gate"
        )

    speedup = 0.0
    if serial:
        if serial["tensors"] != parallel["tensors"]:
            failures.append(
                f"tensor count mismatch: serial={serial['tensors']} parallel={parallel['tensors']}"
            )
        if parallel["seconds"] > 0:
            speedup = serial["seconds"] / parallel["seconds"]
        if speedup < args.min_speedup:
            failures.append(f"speedup {speedup:.2f}× below {args.min_speedup:.2f}× gate")

        print()
        print(
            f"speedup: {speedup:.2f}× (serial {serial['seconds']:.2f}s -> "
            f"parallel {parallel['seconds']:.2f}s, gate >= {args.min_speedup:.2f}×)"
        )

    print()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
    else:
        print("PASS: parallel inflate within gate")

    payload = {
        "archive": str(archive),
        "cpu_count": os.cpu_count(),
        "speedup": round(speedup, 2),
        "runs": runs,
        "failures": failures,
        "passed": not failures,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
