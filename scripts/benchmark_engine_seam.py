#!/usr/bin/env python
"""Benchmark: run_on_engine seam vs asyncio.to_thread (stage-4 charter).

For the same synthetic blocking compute it measures, per method:

1. **Max event-loop stall** during the await — the seam exists to be as
   loop-safe as ``to_thread``; a blocking wait would stall the ticker for
   the whole job (hundreds of ms), a healthy seam stays near the ticker's
   0.5ms quantum.
2. **End-to-end await latency** — engine dispatch (0.1s run-loop poll +
   pool pickup) vs the executor's immediate pickup. This quantifies the
   duration rule: short jobs pay the poll, long jobs don't care.
3. **Dispatch delay** (fn start − await start) — the seam-specific
   overhead, isolated from the compute itself.

Run:
    PYTHONNOUSERSITE=1 ./scripts/python scripts/benchmark_engine_seam.py
"""

from __future__ import annotations

import asyncio
import statistics
import time

from domain.infrastructure._internal.engine_seam import run_on_engine

DURATIONS = [0.005, 0.05, 0.3]  # seconds of synthetic blocking compute
ITERS = {0.005: 20, 0.05: 10, 0.3: 5}


def _blocking_compute(seconds: float) -> float:
    """Sync work that must not run on the loop thread."""
    t0 = time.perf_counter()
    time.sleep(seconds)
    return time.perf_counter() - t0


async def measure(method: str, duration: float) -> dict:
    """One iteration: return e2e/stall/dispatch-delay seconds."""
    max_gap = 0.0
    stop = False

    async def ticker() -> None:
        nonlocal max_gap
        last = time.perf_counter()
        while not stop:
            await asyncio.sleep(0.0005)
            now = time.perf_counter()
            max_gap = max(max_gap, now - last)
            last = now

    tick = asyncio.create_task(ticker())
    await asyncio.sleep(0)  # let the ticker stamp `last` before the job

    stamp: dict[str, float] = {}

    def fn() -> float:
        stamp["start"] = time.perf_counter()
        return _blocking_compute(duration)

    await_start = time.perf_counter()
    if method == "to_thread":
        await asyncio.to_thread(fn)
    else:
        await run_on_engine(fn, name=f"bench-{method}")
    e2e = time.perf_counter() - await_start

    stop = True
    await tick
    delay = stamp.get("start", await_start) - await_start
    return {"e2e": e2e, "stall": max_gap, "delay": delay}


async def main() -> None:
    print("run_on_engine seam vs asyncio.to_thread (synthetic blocking compute)")
    print(
        f"{'compute':>8} {'method':>14} {'n':>3} {'e2e med ms':>11} "
        f"{'delay med ms':>13} {'stall max ms':>13}"
    )
    print("-" * 68)

    for duration in DURATIONS:
        n = ITERS[duration]
        for method in ("to_thread", "engine"):
            runs = [await measure(method, duration) for _ in range(n)]
            e2e = statistics.median(r["e2e"] for r in runs) * 1000
            delay = statistics.median(r["delay"] for r in runs) * 1000
            stall = max(r["stall"] for r in runs) * 1000
            print(
                f"{duration * 1000:>7.0f}ms {method:>14} {n:>3} "
                f"{e2e:>11.2f} {delay:>13.2f} {stall:>13.2f}"
            )
        print()

    print(
        "Reading: stall near 0.5ms = loop never blocked (both methods); "
        "delay = dispatch overhead only.\n"
        "The engine's 0.1s run-loop poll shows up as delay on short jobs — "
        "the duration rule keeps micro-I/O on to_thread for exactly this."
    )


if __name__ == "__main__":
    asyncio.run(main())
