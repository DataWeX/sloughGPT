#!/usr/bin/env python3
"""Benchmark MogDB journal durability: append throughput + replay + compact.

Measures (per mode):
  - insert: ``insert_one`` records/sec and µs/op (the journal write path)
  - reopen: seconds to replay the journal into a fresh Collection
  - compact: seconds to compact + fsync a snapshot

Modes:
  --fsync on    durable appends (fsync per record) — the crash-resistant mode
  --fsync off   buffered appends (throughput ceiling)
  --fsync auto  whatever the current code supports (baseline runs on code
                that has no fsync knob: falls back to off and says so)

Usage:
  PYTHONPATH=packages/mogdb/src python scripts/benchmark_mogdb_journal.py \
      --records 20000 --json /tmp/opencode/mogdb_journal_baseline.json

Card faa1cfa7 (crash-resistant stores). Compare runs with
``scripts/benchmark_results.py compare`` or by diffing the JSON.
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "mogdb" / "src"))

from mogdb.collection import Collection  # noqa: E402


def _fsync_supported() -> bool:
    return "fsync" in inspect.signature(Collection.__init__).parameters


def make_collection(path: Path, fsync: str) -> Collection:
    """Build a collection honouring --fsync (with baseline fallback)."""
    if fsync == "on":
        if _fsync_supported():
            return Collection("bench", path, fsync=True)
        print("  [warn] fsync knob not present in this build; running buffered (baseline)")
        return Collection("bench", path)
    if _fsync_supported():
        return Collection("bench", path, fsync=False)
    return Collection("bench", path)


def bench_insert(path: Path, records: int, fsync: str) -> dict:
    payload = {"n": 0, "text": "benchmark record for journal durability", "tag": "bench"}
    times: list[float] = []
    for run in range(3):
        target = path / f"run{run}"
        target.mkdir(parents=True, exist_ok=True)
        c = make_collection(target, fsync)
        t0 = time.perf_counter()
        for i in range(records):
            c.insert_one({**payload, "n": i})
        times.append(time.perf_counter() - t0)
        shutil.rmtree(target)
    best = min(times)
    return {
        "records": records,
        "seconds_best": round(best, 4),
        "seconds_all": [round(t, 4) for t in times],
        "rec_per_s": int(records / best),
        "us_per_op": round(best / records * 1e6, 2),
    }


def bench_reopen(path: Path, records: int, fsync: str) -> dict:
    target = path / "reopen"
    target.mkdir(parents=True, exist_ok=True)
    c = make_collection(target, fsync)
    for i in range(records):
        c.insert_one({"n": i, "text": "replay payload"})
    t0 = time.perf_counter()
    c2 = Collection("bench", target)
    elapsed = time.perf_counter() - t0
    loaded = len(c2.find())
    shutil.rmtree(target)
    return {"records": records, "reopen_seconds": round(elapsed, 4), "loaded": loaded}


def bench_compact(path: Path, records: int, fsync: str) -> dict:
    target = path / "compact"
    target.mkdir(parents=True, exist_ok=True)
    c = make_collection(target, fsync)
    for i in range(records):
        c.insert_one({"n": i, "text": "compact payload"})
    t0 = time.perf_counter()
    n = c.compact()
    elapsed = time.perf_counter() - t0
    shutil.rmtree(target)
    return {"records": records, "docs": n, "compact_seconds": round(elapsed, 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=int, default=20000)
    ap.add_argument("--fsync", choices=("on", "off", "auto"), default="auto")
    ap.add_argument("--json", type=Path, default=None, help="write results JSON here")
    args = ap.parse_args()

    fsync_mode = "off" if args.fsync == "auto" else args.fsync
    results = {
        "benchmark": "mogdb_journal",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": _git_head(),
        "python": platform.python_version(),
        "fsync_requested": args.fsync,
        "fsync_effective": fsync_mode if _fsync_supported() else "unsupported(baseline)",
        "insert": None,
        "reopen": None,
        "compact": None,
    }

    with tempfile.TemporaryDirectory(prefix="mogdb_bench_") as tmp:
        root = Path(tmp)
        print(f"insert x3 runs of {args.records} records (fsync={results['fsync_effective']}) ...")
        results["insert"] = bench_insert(root, args.records, fsync_mode)
        i = results["insert"]
        print(f"  best: {i['rec_per_s']} rec/s  ({i['us_per_op']} µs/op)  all={i['seconds_all']}")
        print("reopen (fresh replay of journal) ...")
        results["reopen"] = bench_reopen(root, args.records, fsync_mode)
        r = results["reopen"]
        print(f"  {r['reopen_seconds']}s for {r['loaded']} docs")
        print("compact ...")
        results["compact"] = bench_compact(root, args.records, fsync_mode)
        c = results["compact"]
        print(f"  {c['compact_seconds']}s for {c['docs']} docs")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


def _git_head() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
