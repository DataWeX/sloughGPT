"""Benchmark the SloEngine consciousness post-generation hook (Build Order #6).

Three signals per mode (off = ConsciousnessConfig level 0, the production
default; on = level 1):

1. sloengine_init_us_avg — SloEngine() construction cost (init now includes
   the one-time get_consciousness() handshake).
2. generate:us_per_gen — full generate() cost. NOTE: dominated by pre-existing
   HD-memory/quantum search (~1.6s/gen under load), so only large regressions
   are visible here; the hook's own cost is measured directly by (3).
3. hook:us_per_call — direct _post_consciousness() cost: off = the per-
   generate gate overhead in production (expect sub-microseconds); on =
   full process()+save() feature cost.

No-regression gate: "off" hook cost must stay ~sub-µs; init cost must not
regress materially vs main (run with --mode off on main for comparison).

Usage:
    python scripts/benchmark_consciousness_hook.py
    python scripts/benchmark_consciousness_hook.py --mode off --iters 30
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain.cognition._internal.consciousness.config import ConsciousnessConfig
from domain.cognition._internal.consciousness.engine import (
    get_consciousness,
    reset_consciousness,
)
from domain.core._internal.soul import SloEngine


def _bench_generate(iters: int) -> dict[str, float]:
    eng = SloEngine()
    prompt = "benchmark the consciousness hook"
    eng.generate(prompt, include_reasoning=False)  # warmup
    start = time.perf_counter()
    for _ in range(iters):
        eng.generate(prompt, include_reasoning=False)
    elapsed = time.perf_counter() - start
    return {
        "iters": iters,
        "elapsed_s": elapsed,
        "gens_per_s": iters / elapsed if elapsed else float("inf"),
        "us_per_gen": (elapsed / iters) * 1e6 if iters else 0.0,
    }


def _bench_init(count: int) -> float:
    start = time.perf_counter()
    for _ in range(count):
        SloEngine()
    return (time.perf_counter() - start) * 1e6 / count if count else 0.0


def _bench_hook(mode: str) -> str:
    eng = SloEngine()
    if not hasattr(eng, "_post_consciousness"):
        return "hook n/a (pre-wiring build)"
    calls = 20000 if mode == "off" else 30
    start = time.perf_counter()
    for _ in range(calls):
        eng._post_consciousness("prompt text", "generated response text")
    elapsed = time.perf_counter() - start
    return f"{(elapsed / calls) * 1e6:.3f} us/call (n={calls})"


def run(mode: str, iters: int, init_count: int) -> None:
    print(f"mode={mode} iters={iters}", flush=True)
    print(f"sloengine_init_us_avg={_bench_init(init_count):.1f} (n={init_count})", flush=True)

    store = tempfile.mkdtemp(prefix="consc-bench-")
    try:
        reset_consciousness()
        level = 1 if mode == "on" else 0
        get_consciousness(ConsciousnessConfig(level=level, store_path=store))

        result = _bench_generate(iters)
        print(
            f"generate: gens_per_s={result['gens_per_s']:.2f} "
            f"us_per_gen={result['us_per_gen']:.1f} elapsed_s={result['elapsed_s']:.3f}",
            flush=True,
        )
        print(f"hook: {_bench_hook(mode)}", flush=True)
    finally:
        reset_consciousness()
        shutil.rmtree(store, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("off", "on", "both"), default="both")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--init-count", type=int, default=10)
    args = parser.parse_args()

    modes = ("off", "on") if args.mode == "both" else (args.mode,)
    for mode in modes:
        run(mode, args.iters, args.init_count)


if __name__ == "__main__":
    main()
