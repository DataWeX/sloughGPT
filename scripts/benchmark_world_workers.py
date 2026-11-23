"""
World workers benchmark: serial vs pooled seeded world runs.

Measures wall-clock speedup of ``run_world_runs`` across worker processes
and verifies seeded parity — pooled results must equal serial results
exactly (each run seeds its own process, so seed S matches an inline run).

Usage:
    python scripts/benchmark_world_workers.py                 # default sweep
    python scripts/benchmark_world_workers.py --runs 8 --ticks 40
    python scripts/benchmark_world_workers.py --json-out run.json   # for
        `benchmark_results.py record --kind world-workers --json-file run.json`
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Repo root on sys.path so `domain` resolves regardless of invocation cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Before numpy: OpenBLAS reads these once at library load; the repo default
# is a single thread per op (SLO_OPENBLAS_NUM_THREADS=1).
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("SLO_OPENBLAS_NUM_THREADS", "1"))
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("SLO_OMP_NUM_THREADS", "1"))

from domain.shell._internal.simulation import WorldParams  # noqa: E402
from domain.shell._internal.world_workers import run_world_runs  # noqa: E402


def _specs(seeds: list[int], grid: tuple[int, int, int], ticks: int) -> list[dict]:
    """One independent ticks-run spec per seed (each sets its own world seed)."""
    return [
        {
            "kind": "ticks",
            "seed": seed,
            "params": WorldParams(
                grid_size=grid, world_seed=seed, generate_world=True, start_agents=4
            ),
            "ticks": ticks,
        }
        for seed in seeds
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="World workers benchmark (serial vs pooled).")
    parser.add_argument("--runs", type=int, default=8, help="seeded runs in the sweep")
    parser.add_argument("--ticks", type=int, default=60, help="ticks per run")
    parser.add_argument("--grid", type=str, default="24,12,24", help="world size as W,H,D")
    parser.add_argument("--seed", type=int, default=42, help="first seed (runs use seed..seed+N)")
    parser.add_argument("--json-out", type=str, default=None, help="dump metrics to this file")
    args = parser.parse_args()

    grid = tuple(int(p) for p in args.grid.split(","))
    seeds = [args.seed + i for i in range(args.runs)]
    specs = _specs(seeds, grid, args.ticks)
    cpu = os.cpu_count() or 1

    print(f"runs={args.runs} ticks={args.ticks} grid={args.grid} cpus={cpu}")
    print(f"{'mode':<8} {'workers':>7} {'wall_s':>8} {'speedup':>8} {'parity':>7}")

    rows: list[dict] = []
    serial_results = None
    serial_wall = None
    worker_counts = [w for w in (2, 4) if w <= cpu]
    for workers in [1, *worker_counts]:
        started = time.perf_counter()
        results = run_world_runs(specs, max_workers=workers)
        wall = time.perf_counter() - started
        if workers == 1:
            serial_results, serial_wall = results, wall
            parity, speedup = "ok", 1.0
        else:
            parity = "ok" if results == serial_results else "FAIL"
            speedup = serial_wall / wall if wall else float("inf")
        mode = "serial" if workers == 1 else "pooled"
        print(f"{mode:<8} {workers:>7} {wall:>8.3f} {speedup:>7.2f}x {parity:>7}")
        rows.append(
            {
                "mode": mode,
                "workers": workers,
                "wall_s": round(wall, 4),
                "speedup": round(speedup, 3),
                "parity": parity,
            }
        )

    failed = any(r["parity"] == "FAIL" for r in rows)
    if failed:
        print("PARITY FAIL: pooled results differ from serial — seeded contract broken")
        return 1

    if args.json_out:
        payload = {
            "kind": "world-workers",
            "runs": args.runs,
            "ticks": args.ticks,
            "grid": list(grid),
            "seed": args.seed,
            "cpu_count": cpu,
            "results": rows,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
