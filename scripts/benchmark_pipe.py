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
import contextlib
import gc
import json
import os
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
# Timed passes are reported as the BEST of REPEATS, never a single pass: a
# single timed pass on a loaded box measured spawn at 46.7 us where min-of-7
# gives 13.4 us, and that error went straight into the --ci regression gate.
REPEATS = 5
WARMUP = 512


def _noop() -> None:
    return None


class _Stub:
    """Minimal process stand-in: an id and a done flag."""

    __slots__ = ("id", "is_done")

    def __init__(self, pid: str) -> None:
        self.id = pid
        self.is_done = False


@contextlib.contextmanager
def _collector_out():
    """Take the garbage collector out of the timed window.

    One ``Engine.spawn`` allocates ~11 objects (two ``threading.Event``, a
    kwargs dict, seven lists), so gen-0 collections fire every ~60 spawns and
    drop 60-100 us spikes into the measurement: banded spawn cost spread 3.3x
    with the collector running and 1.3x with it held. The gate wants the cost
    of the code, not the collector's schedule -- a 12% regression is smaller
    than a single collection, so leaving GC in would bury every signal this
    benchmark exists to catch.

    Necessary but not sufficient on its own: it removes in-process noise only.
    The throughput benches also read CPU time for the other half -- see
    ``_cpu_seconds``.
    """
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


def _load1() -> float:
    """1-minute load average -- contention on this box that we do not control.

    Recorded so ``--ci`` can refuse to call a number a regression when the
    machine is simply busier than it was at baseline: the same spawn measured
    17.2 us at moderate load and 23.0 us at load 18, which is 33% and would
    trip a 25% budget with no change to the code at all.
    """
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:  # platforms without getloadavg
        return 0.0


def _cpu_seconds() -> float:
    """CPU seconds this process has consumed -- not wall-clock seconds.

    Throughput benches read this instead of ``perf_counter`` because the box
    is shared. A wall clock bills the process for time it spent descheduled:
    at load average 18, banded spawn cost spread 4.09x on wall time with the
    collector already held, and 1.28x on CPU time. CPU time is the only one of
    the two that survives another session competing for the same cores.

    Deliberately NOT used by bench_bound or bench_backpressure: those measure
    waiting for a slot, which is real elapsed time and must stay wall-clock.
    """
    return time.process_time()


# ── admission overhead ───────────────────────────────────────────────────────


def _drain(engine: Engine) -> None:
    """Return every admitted slot so the next timed pass starts at zero usage."""
    for proc in engine._processes.values():
        engine._pipe.release(proc)
    engine._processes.clear()
    engine._pending.clear()


def bench_admit_release(iters: int = N_ADMIT, repeats: int = REPEATS) -> dict:
    """Raw door cost: admit one process, release it, repeat.

    Best of ``repeats`` passes -- the number feeds ``--ci``, so a single pass
    would gate against scheduler noise rather than against the code.
    """
    pipe = Pipe("bench", capacity=1024)
    stub = _Stub("s")
    best = float("inf")

    for _ in range(repeats):
        pipe.admit(stub)
        pipe.release(stub)  # warm
        with _collector_out():
            start = _cpu_seconds()
            for _ in range(iters):
                pipe.admit(stub)
                pipe.release(stub)
            elapsed = _cpu_seconds() - start
        best = min(best, elapsed / iters * 1e9)

    return {
        "cycles": iters,
        "repeats": repeats,
        "ns_per_cycle": round(best, 1),
        "leftover_usage": pipe.usage(),
    }


def bench_spawn(count: int = N_SPAWN, repeats: int = REPEATS) -> dict:
    """``Engine.spawn`` with the door in its path, never contested.

    Best of ``repeats`` passes, collector held off (see ``_collector_out``).
    Containers and the pipe are drained between passes so pass 5 measures the
    same state as pass 1 -- otherwise the pipe would fill across passes and
    start blocking, which measures backpressure instead of spawn.
    """
    engine = Engine(config=EngineConfig(name="bench", queue_size=CAPACITY_SPAWN))
    try:
        best = float("inf")
        for _ in range(repeats):
            for i in range(WARMUP):
                engine.spawn(_noop, name=f"w{i}")
            _drain(engine)

            with _collector_out():
                start = _cpu_seconds()
                for i in range(count):
                    engine.spawn(_noop, name=f"p{i}")
                elapsed = _cpu_seconds() - start
            best = min(best, elapsed / count * 1e9)
            _drain(engine)

        return {
            "spawns": count,
            "repeats": repeats,
            "ns_per_spawn": round(best, 1),
            "spawns_per_s": round(1e9 / best, 1),
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
        "load1": _load1(),
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

    print(
        f"PGQ pipe benchmark   (best of {ar['repeats']} passes, cpu time, gc off, "
        f"load {metrics.get('load1', 0.0)})"
    )
    print(f"  admit+release      {ar['ns_per_cycle']:>9.1f} ns/cycle  ({ar['cycles']:,} cycles)")
    print(
        f"  Engine.spawn       {sp['ns_per_spawn']:>9.1f} ns/spawn   ({sp['spawns']:,} spawns/pass)"
    )
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
                detail = (
                    f"spawn cost {now_ns:.1f} ns > {limit:.1f} ns "
                    f"(baseline {base_ns:.1f}, +{REGRESSION_TOLERANCE:.0%} budget)"
                )
                # CPU time still drifts with contention: enough page faults and
                # cache pressure and the same code bills more. Failing here
                # would teach the next reader to ignore this gate, so only
                # compare when the box is about as busy as it was at baseline.
                base_load = baseline.get("load1", 0.0)
                now_load = metrics.get("load1", 0.0)
                if base_load and now_load > max(base_load * 1.5, base_load + 2.0):
                    print(
                        f"\n[advisory] not failing on {detail} -- machine busy: "
                        f"load {now_load} vs baseline {base_load}"
                    )
                else:
                    failures.append(f"spawn cost regressed: {detail}")

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
