#!/usr/bin/env python
"""Benchmark: server-side SelectiveGZipMiddleware (kanban 223001e4).

Measures two things for synthetic JSON payloads at several sizes:

1. End-to-end middleware behavior via the real ASGI stack: total time,
   content-encoding outcome, and compression ratio.
2. **Max event-loop stall** during compression — the fix under test.
   Baseline (pre-fix) ran ``gzip.compress`` synchronously on the loop, so the
   loop stalls for the whole compress. The fixed middleware runs it at the
   seam (``asyncio.to_thread``), so a ticker task keeps ticking while a
   worker thread compresses. A stall of ~0.5ms = the ticker's sleep quantum;
   a stall of hundreds of ms = the loop was blocked.

Run:
    PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/packages/downcraft:$PWD/packages/core-py:$PWD/apps/api/server" \
        python scripts/benchmark_server_compression.py
"""

from __future__ import annotations

import asyncio
import gzip
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "apps/api/server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from infrastructure.compression import SelectiveGZipMiddleware  # noqa: E402

SIZES = [100_000, 1_000_000, 4_000_000]  # bytes of JSON payload


def make_payload(size: int) -> str:
    """Realistic mixed-entropy JSON: repeated keys + random values.

    A payload of identical characters would gzip at ~1000:1 in ~1ms and hide
    the very stall this benchmark exists to measure.
    """
    import random

    rng = random.Random(42)
    parts: list[str] = []
    written = 0
    while written < size:
        item = (
            f'{{"id":{rng.randrange(10**9)},"tok":"{rng.getrandbits(128):032x}",'
            f'"msg":"latency saw {rng.randrange(10**6)}us"}}'
        )
        parts.append(item)
        written += len(item)
    return "".join(parts)


def make_client(payload: str) -> TestClient:
    app = FastAPI()

    @app.get("/payload")
    def payload_route():
        return {"data": payload}

    app.add_middleware(SelectiveGZipMiddleware)
    return TestClient(app)


async def measure_loop_stall(job) -> tuple[float, float]:
    """Run *job* on the loop; return (job_seconds, max_loop_stall_seconds)."""
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
    # Yield once so the ticker starts and stamps `last` BEFORE the job runs —
    # otherwise the task is still unscheduled during the block and can never
    # observe it (the original measurement silently reported floor noise).
    await asyncio.sleep(0)
    start = time.perf_counter()
    await job()
    elapsed = time.perf_counter() - start
    stop = True
    await tick
    return elapsed, max_gap


async def main() -> None:
    print(
        f"{'payload':>10} {'e2e ms':>9} {'ratio':>7} {'enc':>6} "
        f"{'stall_old ms':>13} {'stall_new ms':>13}"
    )
    print("-" * 64)

    for size in SIZES:
        payload = make_payload(size)
        body = json.dumps({"data": payload}).encode()

        # e2e through the real middleware
        client = make_client(payload)
        start = time.perf_counter()
        resp = client.get("/payload", headers={"accept-encoding": "gzip"})
        e2e = (time.perf_counter() - start) * 1000
        encoded = resp.headers.get("content-encoding", "identity")
        wire = int(resp.headers.get("content-length", len(resp.content)))
        ratio = len(body) / wire

        # loop stall: old (sync compress on the loop) vs new (to_thread)
        _, stall_old = await measure_loop_stall(lambda: asyncio.sleep(0, sync_compress(body)))
        _, stall_new = await measure_loop_stall(lambda: asyncio.to_thread(gzip.compress, body, 6))

        print(
            f"{size:>10,} {e2e:>9.1f} {ratio:>6.2f}x {encoded:>6} "
            f"{stall_old * 1000:>13.1f} {stall_new * 1000:>13.1f}"
        )


def sync_compress(body: bytes) -> None:
    """Blocking compress, executed inline — the pre-fix loop behavior."""
    gzip.compress(body, 6)


if __name__ == "__main__":
    asyncio.run(main())
