"""
Soul Filename-Grammar + Checkpoint I/O Benchmark

Measures the cost the .soul filename grammar adds to the checkpoint path
(domain/inference/_internal/slo_format.py: soul_path / soul_read_candidates /
load_soul's candidate probe).

What this is guarding:
  - save_soul() now canonicalizes the destination before writing, and
    load_soul() probes an ordered candidate list instead of opening one path.
    Both are new work on a hot path (every trainer checkpoint, every model
    load), so the per-call cost has to stay negligible next to the I/O.
  - The probe has a worst case: the canonical name is absent and the legacy
    double-appended spelling is what exists on disk. That path does two
    os.path.isfile() calls and one open instead of one open.

Benchmarks:
  soul_path_write      pure canonicalization (string work only, no I/O)
  read_candidates      ordered probe-list construction (string work only)
  save_canonical       save_soul() to an already-canonical name (adds soul_path)
  save_legacy_name     save_soul() from a legacy .soul.soul name (canonicalizes)
  load_canonical_hit   load_soul() where the canonical file exists (1 probe)
  load_legacy_hit      load_soul() where only .soul.soul exists (2 probes + miss)

Interpretation: the four string benchmarks should read as microseconds; the
I/O benchmarks should be dominated by the read itself. A regression shows up
as load_legacy_hit pulling away from load_canonical_hit by more than one
stat() call's worth (~0.01ms), which means the probe loop is doing real work
per candidate rather than a single failed stat.

Usage:
    python scripts/benchmark_soul_paths.py
    python scripts/benchmark_soul_paths.py --json
    python scripts/benchmark_soul_paths.py --repeats 2000
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class BenchResult:
    name: str
    repeats: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    ops_per_s: float


def _summarize(name: str, lat: list[float], repeats: int) -> BenchResult:
    lat = sorted(lat)
    return BenchResult(
        name=name,
        repeats=repeats,
        mean_ms=round(1000 * statistics.fmean(lat), 6),
        p50_ms=round(1000 * lat[len(lat) // 2], 6),
        p95_ms=round(1000 * lat[int(len(lat) * 0.95)], 6),
        ops_per_s=round(1.0 / statistics.fmean(lat)) if statistics.fmean(lat) else 0.0,
    )


def _time(fn, repeats: int) -> list[float]:
    """Wall-clock each call in seconds. First call is warmed up and dropped."""
    fn()
    lat = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        lat.append(time.perf_counter() - t0)
    return lat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=500, help="iterations per benchmark")
    ap.add_argument("--io-repeats", type=int, default=150, help="iterations for I/O benchmarks")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    from domain.inference import load_soul, save_soul, soul_path, soul_read_candidates

    results: list[BenchResult] = []

    # ── Pure string work: the grammar itself, no filesystem ──────────────
    results.append(
        _summarize(
            "soul_path_write",
            _time(lambda: soul_path("models/v2.3/gpt2.5-model.soul.soul"), args.repeats),
            args.repeats,
        )
    )
    results.append(
        _summarize(
            "read_candidates",
            _time(lambda: soul_read_candidates("m/x.soul.soul"), args.repeats),
            args.repeats,
        )
    )

    tmp = Path(tempfile.mkdtemp(prefix="soul_bench_"))
    try:
        weights_only = {"weights_only": True}

        # ── Write path: canonicalization cost against real I/O ───────────
        def save_canonical():
            save_soul(None, str(tmp / "canon.soul"), **weights_only)

        def save_legacy():
            save_soul(None, str(tmp / "legacy.soul.soul"), **weights_only)

        results.append(
            _summarize("save_canonical", _time(save_canonical, args.io_repeats), args.io_repeats)
        )
        results.append(
            _summarize("save_legacy_name", _time(save_legacy, args.io_repeats), args.io_repeats)
        )

        # ── Read path: 1 probe vs the 2-probe legacy worst case ─────────
        save_soul(None, str(tmp / "hit.soul"), **weights_only)
        save_soul(None, str(tmp / "miss.soul"), **weights_only)
        # Legacy-only layout: file is at the double-appended name, canonical
        # is absent, so load_soul must fail the first isfile() before it
        # reaches the real file on the second candidate.
        save_soul(None, str(tmp / "legacyhit.soul"), **weights_only)
        (tmp / "legacyhit.soul").rename(tmp / "legacyhit.soul.soul")

        results.append(
            _summarize(
                "load_canonical_hit",
                _time(lambda: load_soul(str(tmp / "hit.soul")), args.io_repeats),
                args.io_repeats,
            )
        )
        results.append(
            _summarize(
                "load_legacy_hit",
                _time(lambda: load_soul(str(tmp / "legacyhit.soul")), args.io_repeats),
                args.io_repeats,
            )
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    by_name = {r.name: r for r in results}
    hit, legacy = by_name["load_canonical_hit"], by_name["load_legacy_hit"]

    if args.json:
        print(
            json.dumps(
                {
                    "results": [asdict(r) for r in results],
                    "probe_overhead_ms": round(legacy.p50_ms - hit.p50_ms, 6),
                },
                indent=2,
            )
        )
        return 0

    print("\nSoul filename grammar + checkpoint I/O")
    print("=" * 66)
    print(f"{'benchmark':<22}{'p50 ms':>12}{'p95 ms':>12}{'ops/s':>14}")
    print("-" * 66)
    for r in results:
        print(f"{r.name:<22}{r.p50_ms:>12.4f}{r.p95_ms:>12.4f}{r.ops_per_s:>14,.0f}")
    print("-" * 66)
    print(f"probe overhead (2nd candidate miss): {legacy.p50_ms - hit.p50_ms:+.4f} ms median")
    print(
        "guard: overhead should stay within noise of one failed stat() "
        "(~0.01ms). A larger gap means the probe loop is doing per-candidate "
        "work beyond the existence check."
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
