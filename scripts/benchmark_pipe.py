"""Bounded execution stack: admission overhead and bound enforcement.

Measures:

- **admission overhead** -- the cost of an ``admit``/``release`` cycle on
  the process queue, and of ``Engine.spawn`` with that door in its path
  (capacity far above the spawn count, so the door is never contested).
- **bound enforcement** -- peak ``usage()`` under concurrent spawning must
  never exceed ``diameter()``, nothing may be rejected while the engine is
  open, and a producer arriving at a full stack must park and proceed once
  a process finishes rather than be turned away.

Run from the repo root: ``.venv/bin/python scripts/benchmark_pipe.py``

Options::

    --json-out PATH   write raw metrics JSON
    --update          store the current metrics as the baseline
    --ci              exit non-zero if an invariant fails or spawn cost
                      regresses >25% against the baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "packages" / "core-py")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from domains.infrastructure.pugqeep import Pipe, PipeClosed  # noqa: E402
from domains.infrastructure.pugqeep.config import EngineConfig  # noqa: E402
from domains.infrastructure.pugqeep.engine import Engine  # noqa: E402

BASELINE_FILE = Path("data/benchmark_pipe_baseline.json")

N_ADMIT = 200_000
N_SPAWN = 20_000
CAPACITY_SPAWN = 100_000
BOUND_CAPACITY = 64
BOUND_WORKERS = 8
REGRESSION_TOLERANCE = 0.25


def _noop() -> None:
    return None


class _Stub:
    """Minimal process stand-in: an id and a done flag."""

    __slots__ = ("id", "is_done")

    def __init__(self, pid: str) -> None:
        self.id = pid
        self.is_done = False


# ── admission overhead ───────────────────────────────────────────────────────


def bench_admit_release(iters: int = N_ADMIT) -> dict:
    """Raw door cost: admit one process, release it, repeat."""
    pipe = Pipe("bench", capacity=1024)
    stub = _Stub("s")

    start = time.perf_counter()
    for _ in range(iters):
        pipe.admit(stub)
        pipe.release(stub)
    elapsed = time.perf_counter() - start

    return {
        "cycles": iters,
        "ns_per_cycle": round(elapsed / iters * 1e9, 1),
        "leftover_usage": pipe.usage(),
    }


def bench_spawn(count: int = N_SPAWN) -> dict:
    """``Engine.spawn`` with the door in its path, never contested."""
    engine = Engine(config=EngineConfig(name="bench", queue_size=CAPACITY_SPAWN))
    try:
        start = time.perf_counter()
        for i in range(count):
            engine.spawn(_noop, name=f"p{i}")
        elapsed = time.perf_counter() - start

        return {
            "spawns": count,
            "ns_per_spawn": round(elapsed / count * 1e9, 1),
            "spawns_per_s": round(count / elapsed, 1),
            "usage": engine._pipe.usage(),
        }
    finally:
        engine.stop()


# ── bound enforcement ────────────────────────────────────────────────────────


def bench_bound(capacity: int = BOUND_CAPACITY) -> dict:
    """Fill the stack to its ceiling from many threads, then prove it binds.

    A bench that spawns-and-completes in a loop never approaches the bound
    and therefore tests nothing: an earlier version reported peak 0 against
    a diameter of 64. Here the workers fill to exactly ``diameter`` without
    finishing anything, so the ceiling must be reached precisely, and the
    first admission past it must park rather than slip in.
    """
    if capacity % BOUND_WORKERS:
        raise ValueError("capacity must divide evenly across workers")

    engine = Engine(config=EngineConfig(name="bound", queue_size=capacity))
    observed: list[int] = []
    admitted: list = []
    rejections = 0
    lock = threading.Lock()
    sampling = threading.Event()

    def sample() -> None:
        while not sampling.is_set():
            with lock:
                observed.append(engine._pipe.usage())
            time.sleep(0.001)

    def worker() -> None:
        nonlocal rejections
        for _ in range(capacity // BOUND_WORKERS):
            try:
                proc = engine.spawn(_noop)
            except PipeClosed:
                rejections += 1
                continue
            with lock:
                admitted.append(proc)

    past_ceiling: dict = {}

    def one_more() -> None:
        """One more admission at exactly full -- it must park, then proceed."""
        start = time.perf_counter()
        try:
            engine.spawn(_noop)
            past_ceiling["proceeded"] = True
        except PipeClosed:
            past_ceiling["proceeded"] = False
        past_ceiling["parked_ns"] = round((time.perf_counter() - start) * 1e9, 0)

    try:
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()

        workers = [threading.Thread(target=worker) for _ in range(BOUND_WORKERS)]
        start = time.perf_counter()
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        fill_ms = round((time.perf_counter() - start) * 1e3, 2)

        usage_at_ceiling = engine._pipe.usage()
        headroom_at_ceiling = engine._pipe.headroom()

        # Exactly full: a further admission must park, not slip in.
        beyond = threading.Thread(target=one_more)
        beyond.start()
        time.sleep(0.15)
        still_parked = "proceeded" not in past_ceiling

        # Return one slot; the parked producer must take it.
        admitted[0].complete(None)
        beyond.join(timeout=5)

        for proc in admitted[1:]:
            proc.complete(None)
        sampling.set()
        sampler.join(timeout=5)
    finally:
        engine.stop()

    diameter = engine._pipe.diameter()
    peak = max(observed) if observed else 0
    return {
        "diameter": diameter,
        "usage_at_ceiling": usage_at_ceiling,
        "headroom_at_ceiling": headroom_at_ceiling,
        "peak_usage": peak,
        "high_water_mark": engine._pipe.peak_usage,
        "reached_ceiling": usage_at_ceiling == diameter and headroom_at_ceiling == 0,
        "parked_past_ceiling": still_parked,
        "admitted_after_release": past_ceiling.get("proceeded"),
        "parked_ms": round((past_ceiling.get("parked_ns") or 0) / 1e6, 2),
        "bound_held": peak <= diameter and engine._pipe.peak_usage <= diameter,
        "rejections": rejections,
        "samples": len(observed),
        "fill_ms": fill_ms,
        "spawns": len(admitted),
    }


def bench_backpressure() -> dict:
    """A producer at a full stack parks, then proceeds when room appears."""
    engine = Engine(config=EngineConfig(name="bp", queue_size=1))
    outcome: dict = {"admitted_after_release": None, "parked_ns": None}

    def producer() -> None:
        start = time.perf_counter()
        try:
            engine.spawn(_noop)
            outcome["admitted_after_release"] = True
        except PipeClosed:
            outcome["admitted_after_release"] = False
        outcome["parked_ns"] = round((time.perf_counter() - start) * 1e9, 0)

    held = engine.spawn(_noop)
    thread = threading.Thread(target=producer)
    thread.start()
    try:
        time.sleep(0.15)
        still_parked = outcome["admitted_after_release"] is None
        held.complete(None)  # free the only slot; the producer must wake
        thread.join(timeout=5)
    finally:
        engine.stop()

    return {
        "still_parked_at_full": still_parked,
        "admitted_after_release": outcome["admitted_after_release"],
        "parked_ms": round((outcome["parked_ns"] or 0) / 1e6, 2),
    }


# ── reporting ────────────────────────────────────────────────────────────────


def run() -> dict:
    return {
        "admit_release": bench_admit_release(),
        "spawn": bench_spawn(),
        "bound": bench_bound(),
        "backpressure": bench_backpressure(),
    }


def _print(metrics: dict) -> None:
    ar = metrics["admit_release"]
    sp = metrics["spawn"]
    bd = metrics["bound"]
    bp = metrics["backpressure"]

    print("PGQ pipe benchmark")
    print(f"  admit+release      {ar['ns_per_cycle']:>9.1f} ns/cycle  ({ar['cycles']:,} cycles)")
    print(f"  Engine.spawn       {sp['ns_per_spawn']:>9.1f} ns/spawn   ({sp['spawns']:,} spawns)")
    print(f"  spawn throughput   {sp['spawns_per_s']:>9,.0f} /s")
    print()
    verdict = "HELD" if bd["bound_held"] else "BREACHED"
    reached = "reached" if bd["reached_ceiling"] else "NEVER REACHED"
    parked = "parked" if bd["parked_past_ceiling"] else "SLIPPED THROUGH"
    print(f"  diameter           {bd['diameter']:>9}            (queue_size)")
    print(
        f"  usage at ceiling   {bd['usage_at_ceiling']:>9}  [{reached}]  headroom {bd['headroom_at_ceiling']}"
    )
    print(f"  peak usage         {bd['peak_usage']:>9}  [{verdict}]")
    print(f"  high-water mark    {bd['high_water_mark']:>9}            (observed, never declared)")
    print(
        f"  one past ceiling   {'':>9}  [{parked}] for {bd['parked_ms']:.1f} ms, then "
        f"{'proceeded' if bd['admitted_after_release'] else 'REFUSED'}"
    )
    print(f"  rejections         {bd['rejections']:>9}            (0 while open)")
    print(
        f"  fill to ceiling    {bd['fill_ms']:>9.1f} ms          ({bd['spawns']} spawns, {bd['samples']} samples)"
    )
    print()
    parked = "parked" if bp["still_parked_at_full"] else "DID NOT PARK"
    admitted = "proceeded" if bp["admitted_after_release"] else "REFUSED"
    print(f"  at capacity        {parked} for {bp['parked_ms']:.1f} ms, then {admitted}")


def _invariants(metrics: dict) -> list[str]:
    failures = []
    if not metrics["bound"]["bound_held"]:
        failures.append(
            f"bound breached: peak {metrics['bound']['peak_usage']} > "
            f"diameter {metrics['bound']['diameter']}"
        )
    if not metrics["bound"]["reached_ceiling"]:
        failures.append(
            f"fill stopped short: usage {metrics['bound']['usage_at_ceiling']} of "
            f"{metrics['bound']['diameter']} -- the bench would prove nothing"
        )
    if not metrics["bound"]["parked_past_ceiling"]:
        failures.append("admission at exactly full slipped past the bound")
    if not metrics["bound"]["admitted_after_release"]:
        failures.append("producer parked at the ceiling never proceeded once a slot freed")
    if metrics["bound"]["rejections"]:
        failures.append(f"{metrics['bound']['rejections']} admissions rejected while open")
    if not metrics["backpressure"]["still_parked_at_full"]:
        failures.append("full stack did not park the producer")
    if not metrics["backpressure"]["admitted_after_release"]:
        failures.append("parked producer did not proceed once capacity freed")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-out", default=None, help="write raw metrics JSON here")
    ap.add_argument("--update", action="store_true", help="store metrics as the baseline")
    ap.add_argument("--ci", action="store_true", help="exit non-zero on invariant/regression")
    args = ap.parse_args()

    metrics = run()
    _print(metrics)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(metrics, indent=2))
        print(f"\nwrote {args.json_out}")

    if args.update:
        BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_FILE.write_text(json.dumps(metrics, indent=2))
        print(f"baseline written to {BASELINE_FILE}")

    failures = _invariants(metrics)

    if args.ci and BASELINE_FILE.exists() and not args.update:
        baseline = json.loads(BASELINE_FILE.read_text())
        base_ns = baseline.get("spawn", {}).get("ns_per_spawn")
        now_ns = metrics["spawn"]["ns_per_spawn"]
        if base_ns:
            limit = base_ns * (1 + REGRESSION_TOLERANCE)
            if now_ns > limit:
                failures.append(
                    f"spawn cost regressed: {now_ns:.1f} ns > {limit:.1f} ns "
                    f"(baseline {base_ns:.1f}, +{REGRESSION_TOLERANCE:.0%} budget)"
                )

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
