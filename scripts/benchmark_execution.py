"""Benchmark fire-and-forget dispatch (execution consolidation, stage 1).

Measures:
  - per-dispatch latency of ``FireAndForgetPool.submit`` (queue put, µs)
  - peak live threads while bursting N blocking fire-and-forget tasks
    through the bounded pool (vs the old raw ``threading.Thread`` spawn)

Run from the repo root: ``scripts/python scripts/benchmark_execution.py``
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import timeit
from collections.abc import Callable
from pathlib import Path

from domain.infrastructure._internal.fire_and_forget import FireAndForgetPool, get_pool

N_BURST = 128
N_DISPATCH = 20_000


def drain_pool(pool: FireAndForgetPool) -> None:
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline:
        finished = threading.Event()

        def probe() -> None:
            finished.set()

        if not pool.submit(probe):
            continue
        finished.wait(timeout=0.1)
        if pool.stats["queued"] == 0 and finished.is_set():
            break


def peak_threads(submit: Callable[[Callable[[], None]], bool | None], n: int) -> int:
    gate = threading.Event()
    done = {"n": 0, "queued": 0}
    lock = threading.Lock()
    peak = {"n": 0}

    def blocker() -> None:
        gate.wait()
        with lock:
            done["n"] += 1
            peak["n"] = max(peak["n"], threading.active_count())

    for _ in range(n):
        if submit(blocker) is not False:
            with lock:
                done["queued"] += 1
    gate.set()
    deadline = time.perf_counter() + 10.0
    while time.perf_counter() < deadline:
        with lock:
            peak["n"] = max(peak["n"], threading.active_count())
            complete = done["n"] >= done["queued"]
        if complete:
            break
        time.sleep(0.005)
    return peak["n"]


def raw_spawn(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="fire-and-forget dispatch benchmark")
    parser.add_argument(
        "--json", type=Path, default=None, help="write metrics JSON for benchmark_results.py record"
    )
    args = parser.parse_args()

    print("fire-and-forget dispatch benchmark (execution-consolidation stage 1)")
    print("=" * 64)

    submit = get_pool().submit
    drain_pool(get_pool())
    t0 = time.perf_counter()
    peak_pool = peak_threads(lambda fn: submit(fn), N_BURST)
    t_pool = time.perf_counter() - t0
    active = threading.active_count()
    print(
        f"pool   peak threads  : {peak_pool:3d}  (burst {N_BURST}, live {active}, {t_pool * 1e3:.1f} ms)"
    )

    per_op = timeit.timeit(lambda: submit(lambda: None), number=N_DISPATCH) / N_DISPATCH * 1e6
    print(f"pool   per-dispatch  : {per_op:8.2f} µs  ({N_DISPATCH} submits, NO wait)")

    t0 = time.perf_counter()
    peak_raw = peak_threads(raw_spawn, N_BURST)
    t_raw = time.perf_counter() - t0
    active = threading.active_count()
    print(
        f"raw    peak threads  : {peak_raw:3d}  (burst {N_BURST}, live {active}, {t_raw * 1e3:.1f} ms)"
    )

    stats = get_pool().stats
    print("=" * 64)
    print(f"pool stats: {stats}")
    print(f"summary: dispatch {per_op:.2f}µs, peak threads {peak_pool} vs {peak_raw} (raw)")

    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    "dispatch_us": round(per_op, 2),
                    "peak_threads": peak_pool,
                    "peak_threads_raw": peak_raw,
                },
                indent=2,
            )
        )
        print(f"[JSON] {args.json}")


if __name__ == "__main__":
    main()
