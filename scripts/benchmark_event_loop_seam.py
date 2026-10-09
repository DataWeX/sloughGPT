#!/usr/bin/env python3
"""Benchmark the async seam: event-loop stall caused by store writes.

A MogDB journal append takes a cross-process ``flock`` and fsyncs, and
``compact()`` rewrites the whole snapshot under that lock. Called inline from
an ``async def`` handler that blocks the *whole* event loop, not just the
request that triggered it. This benchmark measures that stall directly:

  - a ticker coroutine tries to wake every ``--tick`` seconds and records the
    gap between wakes; the longest gap is the worst single stall the loop
    suffered;
  - the same N store writes are run twice: ``inline`` (called directly from
    the coroutine, the pattern the seam guard now forbids) and ``offloaded``
    (``asyncio.to_thread``, the pattern ``test_event_loop_seam_guard``
    enforces).

Metrics (all lower is better):
  - ``inline_stall_ms`` / ``offloaded_stall_ms``: worst ticker gap
  - ``inline_mean_stall_ms`` / ``offloaded_mean_stall_ms``: mean gap
  - ``inline_stalled_fraction`` / ``offloaded_stalled_fraction``: share of
    the run's wall time during which the loop serviced nothing
  - ``stall_reduction``: inline worst / offloaded worst

``stall_*`` and ``stalled_fraction`` are the load-bearing numbers; total
work time is dominated by fsync variance (measured swings of ~6x for the
same volume on the same disk, because ext4 ``fsync`` blocks on the journal
commit), so do not read a speedup into ``*_seconds``.

Usage:
  PYTHONPATH=packages/mogdb/src python scripts/benchmark_event_loop_seam.py \
      --records 50 --json /tmp/opencode/event_loop_seam.json

Companion to ``tests/api-server/test_event_loop_seam_guard.py`` (stops the
regression) and ``scripts/benchmark_mogdb_journal.py`` (costs the primitive).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "mogdb" / "src"))

from mogdb.collection import Collection  # noqa: E402


def _write_inline(col: Collection, count: int) -> None:
    """The pre-guard pattern: blocking store work inside the coroutine."""
    for i in range(count):
        col.insert_one({"n": i, "text": "seam benchmark record", "tag": "bench"})


async def _measure(fn, count: int, tick: float) -> dict:
    """Run *fn* while a ticker probes how long the loop stops servicing it."""
    gaps: list[float] = []
    stop = False
    last = time.perf_counter()

    async def ticker() -> None:
        nonlocal last
        while not stop:
            await asyncio.sleep(tick)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    task = asyncio.create_task(ticker())
    await asyncio.sleep(tick * 5)  # let the ticker settle
    try:
        # No await between here and the call below, so no gap can be recorded
        # in between: every gap from this index on belongs to the measured run.
        last = time.perf_counter()
        start = time.perf_counter()
        at = len(gaps)
        result = await fn(count)
        elapsed = time.perf_counter() - start
    finally:
        stop = True
        await task
    # The ticker can still append the gap spanning the block when it resumes.
    measured = gaps[at:] or [0.0]
    # Contiguous gaps always sum to ~elapsed, so that would report 100% for
    # any run. The loop was only unresponsive for the part of each gap beyond
    # the interval the ticker was promised.
    stalled = sum(max(0.0, g - tick) for g in measured)
    return {
        "seconds": elapsed,
        "worst_stall_ms": max(measured) * 1000,
        "mean_stall_ms": statistics.fmean(measured) * 1000,
        # Share of the run's wall time during which the loop serviced nothing.
        "stalled_fraction": stalled / elapsed if elapsed else 0.0,
        "ticks": len(measured),
        "result": result,
    }


async def _run(root: Path, records: int, tick: float) -> dict:
    col = Collection("bench", root / "store")
    col.insert_one({"_id": "warm", "n": -1})  # force the journal into existence

    async def inline(n: int) -> None:
        # The pre-guard pattern: blocking store work called straight from the
        # coroutine, so the loop cannot service anything else meanwhile.
        _write_inline(col, n)

    async def offloaded(n: int) -> None:
        await asyncio.to_thread(_write_inline, col, n)

    stall = await _measure(inline, records, tick)

    # Fresh store so both modes see the same journal volume on write.
    col2 = Collection("bench", root / "store2")
    col2.insert_one({"_id": "warm", "n": -1})
    off = await _measure(offloaded, records, tick)

    return {
        "records": records,
        "tick_ms": tick * 1000,
        "inline": {k: v for k, v in stall.items() if k != "result"},
        "offloaded": {k: v for k, v in off.items() if k != "result"},
        "inline_docs": len(col.find()),
        "offloaded_docs": len(col2.find()),
        "stall_reduction": stall["worst_stall_ms"] / max(off["worst_stall_ms"], 1e-6),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=int, default=50, help="store writes per mode")
    ap.add_argument("--tick", type=float, default=0.002, help="ticker interval seconds")
    ap.add_argument("--json", type=Path, help="write flat metrics to this path")
    args = ap.parse_args()

    root = Path(tempfile.mkdtemp(prefix="seam_bench_"))
    try:
        print(f"{args.records} store writes, ticker every {args.tick * 1000:.1f} ms")
        data = asyncio.run(_run(root, args.records, args.tick))
    finally:
        shutil.rmtree(root, ignore_errors=True)

    i, o = data["inline"], data["offloaded"]
    print(f"  inline    worst stall {i['worst_stall_ms']:8.1f} ms   mean {i['mean_stall_ms']:7.1f} ms")
    print(
        f"  offloaded worst stall {o['worst_stall_ms']:8.1f} ms   "
        f"mean {o['mean_stall_ms']:7.1f} ms"
    )
    print(f"  worst-stall reduction: {data['stall_reduction']:.1f}x")
    print(
        f"  loop unresponsive {i['stalled_fraction'] * 100:5.1f}% of the inline run "
        f"vs {o['stalled_fraction'] * 100:5.1f}% offloaded "
        f"(work time {i['seconds']:.3f}s vs {o['seconds']:.3f}s)"
    )

    if args.json:
        flat = {
            "records": data["records"],
            "tick_ms": data["tick_ms"],
            "inline_stall_ms": i["worst_stall_ms"],
            "inline_mean_stall_ms": i["mean_stall_ms"],
            "inline_seconds": i["seconds"],
            "inline_stalled_fraction": i["stalled_fraction"],
            "offloaded_stall_ms": o["worst_stall_ms"],
            "offloaded_mean_stall_ms": o["mean_stall_ms"],
            "offloaded_seconds": o["seconds"],
            "offloaded_stalled_fraction": o["stalled_fraction"],
            "stall_reduction": data["stall_reduction"],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(flat, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
