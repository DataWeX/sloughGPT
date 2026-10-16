#!/usr/bin/env python3
"""World–Model Bridge benchmark — ModelBrain vs Perceptron brains.

Measures (card 7cefbbcf acceptance):
  1. Ticks/sec: a model-brained world vs the perceptron baseline at the
     same population — the <20% regression gate (Step 2 of the bridge plan;
     decide-every-N-ticks is the fallback lever if it trips).
  2. Emergence: the standard evolved-vs-frozen proof re-run with
     brain_kind="model" — a model brain must still emerge (evolved beats
     frozen-random) or the trainable hidden layer bought nothing.
  3. Learning: in-life backprop moves the hidden projection (H changes
     after learning_enabled ticks) — the capability step over the frozen
     Perceptron feature projection.

Usage:
    PYTHONPATH=packages/core-py python3 scripts/benchmark_world_brain.py
    PYTHONPATH=packages/core-py python3 scripts/benchmark_world_brain.py --ticks 30 --pop 12
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "core-py"))

import numpy as np  # noqa: E402


def bench_throughput(ticks: int, pop: int, seed: int) -> dict:
    """Ticks/sec for perceptron vs model brains on identical worlds."""
    from domain.shell._internal.evolution import Genome
    from domain.shell._internal.simulation import (
        SimBaby,
        SimScene,
        Simulation,
        WorldParams,
    )

    out: dict = {}
    for kind in ("perceptron", "model"):
        params = WorldParams(
            grid_size=(16, 8, 16),
            generate_world=True,
            world_seed=seed,
            start_agents=pop,
            brain_hidden_units=16,
            brain_kind=kind,
        )
        scene = SimScene(params=params)
        rng = np.random.default_rng(seed)
        for _ in range(pop):
            b = SimBaby(params=params, initial_energy=params.start_energy)
            Genome.random(params, rng).apply_to(b)
            scene.add_baby(b)
        sim = Simulation(scene, max_ticks=ticks)
        t0 = time.perf_counter()
        sim.run()
        dt = time.perf_counter() - t0
        out[kind] = {
            "ticks": ticks,
            "seconds": dt,
            "ticks_per_sec": ticks / dt if dt > 0 else float("inf"),
        }
    base = out["perceptron"]["ticks_per_sec"]
    model = out["model"]["ticks_per_sec"]
    out["regression_pct"] = (1.0 - model / base) * 100.0 if base > 0 else 0.0
    return out


def bench_emergence(seed: int, hidden: int) -> dict:
    """The standard emergence proof with model brains."""
    from domain.shell._internal.evolution import benchmark_emergence
    from domain.shell._internal.simulation import WorldParams

    params = WorldParams(grid_size=(16, 8, 16), brain_kind="model")
    return benchmark_emergence(params, hidden_units=hidden, seed=seed)


def bench_learning(ticks: int, seed: int) -> dict:
    """In-life backprop must move the hidden projection (and help)."""
    from domain.shell._internal.simulation import (
        SimBaby,
        SimScene,
        Simulation,
        WorldParams,
    )

    results: dict = {}
    for kind in ("perceptron", "model"):
        params = WorldParams(
            grid_size=(16, 8, 16),
            generate_world=True,
            world_seed=seed,
            start_agents=8,
            brain_hidden_units=16,
            brain_kind=kind,
            learning_enabled=True,
        )
        scene = SimScene(params=params)
        babies = []
        for _ in range(8):
            b = SimBaby(params=params, initial_energy=params.start_energy)
            scene.add_baby(b)
            babies.append(b)
        h0 = [b.perceptron_cells.H.copy() for b in babies if b.perceptron_cells.H is not None]
        e0 = float(sum(b.entity.energy for b in babies))
        Simulation(scene, max_ticks=ticks).run()
        h1 = [b.perceptron_cells.H.copy() for b in babies if b.perceptron_cells.H is not None]
        e1 = float(sum(b.entity.energy for b in babies))
        drift = float(np.mean([np.abs(a - b).mean() for a, b in zip(h0, h1)])) if h0 else 0.0
        results[kind] = {"energy_delta": e1 - e0, "H_mean_abs_drift": drift}
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ticks", type=int, default=40, help="ticks per throughput/learning run")
    ap.add_argument("--pop", type=int, default=12, help="babies in the throughput run")
    ap.add_argument("--hidden", type=int, default=16, help="brain hidden units")
    ap.add_argument("--seed", type=int, default=7, help="world + RNG seed")
    args = ap.parse_args()

    print(f"== throughput ({args.ticks} ticks, pop={args.pop}, hidden={args.hidden}) ==")
    tp = bench_throughput(args.ticks, args.pop, args.seed)
    for kind in ("perceptron", "model"):
        r = tp[kind]
        print(f"  {kind:11s} {r['ticks_per_sec']:8.2f} ticks/s  ({r['seconds']:.3f}s)")
    print(f"  model-brain regression: {tp['regression_pct']:+.1f}% (gate: < 20%)")
    gate_tp = tp["regression_pct"] < 20.0

    print(f"== emergence (evolved vs frozen, hidden={args.hidden}) ==")
    em = bench_emergence(args.seed, args.hidden)
    print(f"  evolved last avg: {em['evolved_last_avg']:.4f}")
    print(f"  frozen  last avg: {em['frozen_last_avg']:.4f}")
    print(f"  emerged={'yes' if em['emerged'] else 'no'}")
    gate_em = bool(em["emerged"])

    print(f"== in-life learning ({args.ticks} ticks) ==")
    ln = bench_learning(args.ticks, args.seed)
    for kind in ("perceptron", "model"):
        r = ln[kind]
        print(f"  {kind:11s} energy {r['energy_delta']:+9.2f}  H drift {r['H_mean_abs_drift']:.6f}")
    # The capability step: the model brain's hidden projection must MOVE
    # under learning (the perceptron's H is frozen by design).
    gate_ln = ln["model"]["H_mean_abs_drift"] > 0.0

    ok = gate_tp and gate_em and gate_ln
    print(
        f"\nRESULT: throughput={'PASS' if gate_tp else 'FAIL'} "
        f"emergence={'PASS' if gate_em else 'FAIL'} "
        f"learning={'PASS' if gate_ln else 'FAIL'} -> {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
