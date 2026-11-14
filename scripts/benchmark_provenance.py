#!/usr/bin/env python3
"""Benchmark: provenance stamping overhead on the LoRA adapter write path.

Card 20261101_113742. Provenance must never sit on a hot path — this
measures the wall-clock cost of stamping a sidecar on adapter create and
on ``aggregate_best_adapters`` (the merge write path), and fails if the
added cost exceeds a small absolute budget.

What it measures:
  - baseline ``aggregate_best_adapters`` with provenance disabled
  - the same call with provenance enabled (sidecar write per merge)
  - ``provenance_snapshot()`` alone (git + platform capture), the expensive
    part — it is memoized per process, so the first call is the slow one.

Pass/fail: stamping stays under _BUDGET_MS per aggregate call (median).
Loadavg is printed because this box is shared and timing swings with it.

Usage::

    ./scripts/python scripts/benchmark_provenance.py [--json]
    ./scripts/python scripts/benchmark_provenance.py --record   # via benchmark_results
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from domain.feedback._internal.per_user_lora import PerUserLoRAStore  # noqa: E402
from domain.infrastructure._internal.artifact_registry import provenance_snapshot  # noqa: E402

# Absolute budget per aggregate call. Stamping is a single small JSON write
# plus a memoized git lookup — well under a millisecond steady-state; 5ms
# is generous headroom that still catches an accidental git subprocess per
# record or a full-file hash sneaking onto the path.
_BUDGET_MS = 5.0


def _loadavg() -> float:
    try:
        import os

        return os.getloadavg()[0]
    except Exception:  # noqa: BLE001
        return 0.0


def _time_aggregate(store: PerUserLoRAStore, runs: int) -> float:
    """Median wall-clock (ms) for one aggregate_best_adapters call."""
    samples: list[float] = []
    for i in range(runs):
        t0 = time.perf_counter()
        store.aggregate_best_adapters(
            top_k=3, min_feedback_count=1, output_name=f"m{i}", run_eval=False
        )
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def _make_store(td: str) -> PerUserLoRAStore:
    store = PerUserLoRAStore(
        store_path=td,
        adapter_rank=8,
        model_dim=64,
        run_eval=False,
        min_feedback_for_aggregation=1,
    )
    for u in ("u1", "u2", "u3"):
        store.create_adapter(u)
        for _ in range(3):
            store.update_adapter(u, feedback_signal=1.0)
    return store


def run(runs: int = 25, as_json: bool = False) -> dict:
    # First-call snapshot is the worst case (git subprocess not yet memoized).
    t0 = time.perf_counter()
    snap = provenance_snapshot()
    cold_snapshot_ms = (time.perf_counter() - t0) * 1000.0

    # Warm snapshot — what steady-state stamping actually pays.
    t0 = time.perf_counter()
    for _ in range(100):
        provenance_snapshot()
    warm_snapshot_us = (time.perf_counter() - t0) * 1000000.0 / 100

    with tempfile.TemporaryDirectory() as td_off, tempfile.TemporaryDirectory() as td_on:
        store_off = _make_store(td_off)
        store_on = _make_store(td_on)

        # Baseline: strip provenance stamping out of the write path.
        import domain.feedback._internal.per_user_lora as lora_mod

        original = lora_mod.PerUserLoRAStore._stamp_provenance
        lora_mod.PerUserLoRAStore._stamp_provenance = lambda self, *a, **k: None
        try:
            baseline_ms = _time_aggregate(store_off, runs)
        finally:
            lora_mod.PerUserLoRAStore._stamp_provenance = original

        stamped_ms = _time_aggregate(store_on, runs)

        # Confirm the sidecar actually landed (provenance is doing work).
        sidecars = [p for p in Path(td_on).glob("*.prov.json")]
        delta_ms = stamped_ms - baseline_ms
        within_budget = delta_ms <= _BUDGET_MS

        report = {
            "baseline_ms": round(baseline_ms, 4),
            "stamped_ms": round(stamped_ms, 4),
            "delta_ms": round(delta_ms, 4),
            "budget_ms": _BUDGET_MS,
            "within_budget": within_budget,
            "cold_snapshot_ms": round(cold_snapshot_ms, 3),
            "warm_snapshot_us": round(warm_snapshot_us, 3),
            "sidecars_written": len(sidecars),
            "commit": snap.get("commit"),
            "runs": runs,
            "loadavg": _loadavg(),
        }
    if not as_json:
        print("provenance stamping benchmark")
        print(f"  baseline (no provenance) : {baseline_ms:8.3f} ms")
        print(f"  stamped   (with provenance): {stamped_ms:8.3f} ms")
        print(f"  delta                    : {delta_ms:8.3f} ms  (budget {_BUDGET_MS} ms)")
        print(f"  cold snapshot (1st call) : {cold_snapshot_ms:8.3f} ms")
        print(f"  warm snapshot (steady)   : {warm_snapshot_us:8.3f} us")
        print(f"  sidecars written         : {len(sidecars)}")
        print(f"  loadavg                  : {report['loadavg']}")
        print(f"  commit                   : {snap.get('commit')}")
        print(f"  RESULT                   : {'PASS' if within_budget else 'FAIL'}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=25)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--record", action="store_true", help="record via benchmark_results.py")
    args = ap.parse_args()

    report = run(runs=args.runs, as_json=args.json or args.record)

    if args.record:
        import subprocess

        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "benchmark_results.py"),
                "record",
                "--kind",
                "effects",
                "--json",
            ],
            input=json.dumps(report),
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        sys.stderr.write(proc.stderr)
        print(proc.stdout)

    if args.json:
        print(json.dumps(report, indent=2))
    return 0 if report["within_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
