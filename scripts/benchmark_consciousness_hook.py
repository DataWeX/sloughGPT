"""Benchmark the SloEngine consciousness post-generation hook (Build Order #6).

Measures generate() throughput with the hook off (default, ConsciousnessConfig
level 0 — the production default) vs on (level 1), plus init cost of a
SloEngine construction. Useful as a no-regression gate: "off" on this branch
must match "off" on main (the hook is then a single None-check per generate).

Usage:
    python scripts/benchmark_consciousness_hook.py --iters 2000
    python scripts/benchmark_consciousness_hook.py --mode off --iters 2000
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


def run(mode: str, iters: int, init_count: int) -> None:
    print(f"mode={mode} iters={iters}")
    print(f"sloengine_init_us_avg={_bench_init(init_count):.1f} (n={init_count})")

    store = tempfile.mkdtemp(prefix="consc-bench-")
    try:
        reset_consciousness()
        if mode == "on":
            get_consciousness(ConsciousnessConfig(level=1, store_path=store))
        elif mode == "off":
            get_consciousness(ConsciousnessConfig(level=0, store_path=store))

        result = _bench_generate(iters)
        print(
            f"generate: gens_per_s={result['gens_per_s']:.1f} "
            f"us_per_gen={result['us_per_gen']:.1f} elapsed_s={result['elapsed_s']:.3f}"
        )
    finally:
        reset_consciousness()
        shutil.rmtree(store, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("off", "on", "both"), default="both")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--init-count", type=int, default=200)
    args = parser.parse_args()

    modes = ("off", "on") if args.mode == "both" else (args.mode,)
    for mode in modes:
        run(mode, args.iters, args.init_count)


if __name__ == "__main__":
    main()
