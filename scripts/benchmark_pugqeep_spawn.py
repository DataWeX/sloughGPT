"""Benchmark: pugqeep SubprocessProcess spawn cost across start methods.

The L2 forkserver switch trades fork's shared-memory child for a fresh
interpreter (no inherited threads -> no BLAS fork-deadlock class). This
measures what that trade costs through the real SubprocessProcess.start()
path (preflight + preload + module-level worker included):

  - fork          baseline: child inherits parent memory
  - forkserver    new default: fresh child; engine + target module are
                  preloaded into the forkserver, so steady state should
                  match fork (without preload every child re-imports the
                  pugqeep chain, ~1s each -- the tax preload removes)
  - spawn         fresh child, pickle path shared with forkserver, but
                  no preload mechanism (every child re-imports)

First spawn and steady-state samples are reported separately: the first
forkserver spawn carries the one-time server launch + preload imports.

Run from the repo root: ./scripts/python scripts/benchmark_pugqeep_spawn.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from domain.infrastructure._internal.pugqeep.config import SubprocessConfig
from domain.infrastructure._internal.pugqeep.engine import (
    Process,
    ProcessStatus,
    SubprocessProcess,
)

METHODS = ("fork", "forkserver", "spawn")


def _probe() -> int:
    return 42


def run_scenario(method: str, iterations: int, wait_s: float) -> dict:
    """Spawn the same module-level target N times with the given method."""
    config = SubprocessConfig(enabled=True, start_method=method)
    samples: list[float] = []
    for i in range(iterations):
        proc = Process(fn=_probe, name=f"probe-{method}-{i}")
        sub = SubprocessProcess(proc, config)
        t0 = time.perf_counter()
        sub.start()
        sub.monitor()
        deadline = time.time() + wait_s
        while not proc.is_done and time.time() < deadline:
            time.sleep(0.005)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if proc.status != ProcessStatus.COMPLETED or proc.result != 42:
            raise SystemExit(
                f"[{method}] spawn {i} failed: status={proc.status} "
                f"result={proc.result!r} error={proc.error!r} "
                f"(expected COMPLETED/42)"
            )
        samples.append(elapsed_ms)

    steady = samples[1:] or samples
    return {
        "method": method,
        "first_spawn_ms": round(samples[0], 2),
        "steady_mean_ms": round(sum(steady) / len(steady), 2),
        "steady_min_ms": round(min(steady), 2),
        "steady_max_ms": round(max(steady), 2),
        "iterations": iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--wait", type=float, default=30.0, help="per-spawn timeout s")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    print(f"pugqeep spawn benchmark: {args.iterations} spawns x {args.methods}")
    print(f"{'method':<12} {'first':>10} {'steady mean':>12} {'min':>9} {'max':>9}")
    results = []
    for method in args.methods:
        r = run_scenario(method, args.iterations, args.wait)
        results.append(r)
        print(
            f"{r['method']:<12} {r['first_spawn_ms']:>9.1f}m "
            f"{r['steady_mean_ms']:>11.1f}m {r['steady_min_ms']:>8.1f}m "
            f"{r['steady_max_ms']:>8.1f}m"
        )

    if len(results) >= 2:
        by_method = {r["method"]: r for r in results}
        if "fork" in by_method and "forkserver" in by_method:
            fork = by_method["fork"]["steady_mean_ms"]
            fsv = by_method["forkserver"]["steady_mean_ms"]
            delta = ((fsv - fork) / fork) * 100 if fork else float("inf")
            print(f"forkserver steady vs fork: {delta:+.1f}% ({fork:.1f}m -> {fsv:.1f}m)")

    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"[JSON] {args.json}")


if __name__ == "__main__":
    main()
