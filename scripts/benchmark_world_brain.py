#!/usr/bin/env python3
"""World–Model Brain benchmark — every registered brain kind, measured fairly.

Card 7ff49d20 replaced the original single-seed, baseline-first benchmark
(measured the baseline world cold and the model world hot, hence the bogus
"-33.9% model is faster") with a comparative harness:

  * multi-seed averaging (the ``--seeds`` seed list), each arm built from the
    same seed so a kind never gets a luckier world than the baseline;
  * a warmup window that is run but NOT timed, so import caches, branch
    prediction and the world's first-tick setup are excluded from every arm
    equally — both numbers are reported, warmup included and excluded;
  * repeats of the timed window, averaged;
  * four comparative gates, one per claim a brain kind makes.

Gates:
  throughput  each kind vs the perceptron baseline, < 20% regression.
  emergence   evolved beats frozen-random. Genome-compatible kinds only:
              ``slonet`` is fail-loud in ``Genome`` by design (the flat
              W/b[/H/bh] format cannot represent a native stack), so it has no
              generational evolution to prove and is reported as n/a.
  learning    in-life updates must actually move the trainable tensors.
  energy      trained vs UNTRAINED on mean baby energy, identical seeds — the
              only gate that proves learning HELPS rather than merely moving a
              tensor (both prior implementations left this ungated).

Usage:
    PYTHONNOUSERSITE=1 PYTHONPATH=packages/core-py \
        /home/mana/miniconda3/envs/sloughgpt/bin/python scripts/benchmark_world_brain.py
    ... --ticks 30 --pop 12 --seeds 5 --warmup 10 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "core-py"))

import numpy as np  # noqa: E402

#: Kinds compared against the perceptron baseline for throughput/learning.
COMPARATIVE_KINDS = ("model", "slonet", "random")
#: Kinds that learn in-world and therefore take the energy gate.
LEARNABLE_KINDS = ("perceptron", "model", "slonet")
#: Throughput regression a kind may show against the baseline before failing.
THROUGHPUT_BUDGET_PCT = 20.0


# ── measurement helpers ─────────────────────────────────────────────────────


def _brain_kw(args: argparse.Namespace) -> dict:
    """WorldParams knobs every factory receives (flat kinds drop them)."""
    return {
        "brain_hidden_units": args.hidden,
        "brain_n_embed": args.n_embed,
        "brain_n_layer": args.n_layer,
        "brain_decide_every": args.decide_every,
        "brain_learn_every": args.learn_every,
        "brain_optimizer": args.optimizer,
    }


def _scene_for(kind: str, seed: int, pop: int, brain_kw: dict) -> tuple:
    """One world built from ``seed``; seeding the global stream before the
    build makes every arm's spawn layout and flat-kind weights reproducible,
    so a kind never gets a luckier world than the baseline."""
    from domain.shell._internal.simulation import SimBaby, SimScene, WorldParams

    np.random.seed(seed)
    params = WorldParams(
        grid_size=(16, 8, 16),
        generate_world=True,
        world_seed=seed,
        start_agents=pop,
        brain_kind=kind,
        **brain_kw,
    )
    scene = SimScene(params=params)
    for _ in range(pop):
        scene.add_baby(SimBaby(params=params, initial_energy=params.start_energy))
    return params, scene


def measure_throughput(
    kind: str, seed: int, pop: int, ticks: int, warmup: int, repeats: int, brain_kw: dict
) -> dict:
    """Ticks/sec for one kind, with and without the warmup window excluded.

    A single scene is built, warmed up for ``warmup`` ticks (untimed), then
    timed for ``repeats`` windows of ``ticks`` ticks. Reporting both numbers
    from the same run is what makes the warmup exclusion auditable: if they
    agree, the warmup never mattered.
    """
    from domain.shell._internal.simulation import Simulation

    _params, scene = _scene_for(kind, seed, pop, brain_kw)
    if warmup > 0:
        t0 = time.perf_counter()
        Simulation(scene, max_ticks=warmup).run()
        warmup_dt = time.perf_counter() - t0
    else:
        warmup_dt = 0.0

    excluded: list[float] = []
    t0 = time.perf_counter()
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        Simulation(scene, max_ticks=ticks).run()
        window = time.perf_counter() - start
        excluded.append(ticks / window if window > 0 else float("inf"))
    wall = time.perf_counter() - t0

    total_ticks = warmup + ticks * max(1, repeats)
    return {
        "tps_included_warmup": total_ticks / (warmup_dt + wall),
        "tps_excluded_warmup": float(np.mean(excluded)),
        "repeats": max(1, repeats),
    }


def bench_throughput(args: argparse.Namespace, brain_kw: dict) -> dict:
    """Baseline + every comparative kind, multi-seed averaged, both tps views."""
    rows: dict[str, dict] = {}
    for kind in ("perceptron", *COMPARATIVE_KINDS):
        incl, excl = [], []
        for seed in args.seeds:
            r = measure_throughput(
                kind, seed, args.pop, args.ticks, args.warmup, args.repeats, brain_kw
            )
            incl.append(r["tps_included_warmup"])
            excl.append(r["tps_excluded_warmup"])
        rows[kind] = {
            "tps_included_warmup": float(np.mean(incl)),
            "tps_excluded_warmup": float(np.mean(excl)),
        }
    base_in = rows["perceptron"]["tps_included_warmup"]
    base_ex = rows["perceptron"]["tps_excluded_warmup"]
    for kind in COMPARATIVE_KINDS:
        rows[kind]["regression_pct_included"] = (
            (1.0 - rows[kind]["tps_included_warmup"] / base_in) * 100.0 if base_in > 0 else 0.0
        )
        rows[kind]["regression_pct_excluded"] = (
            (1.0 - rows[kind]["tps_excluded_warmup"] / base_ex) * 100.0 if base_ex > 0 else 0.0
        )
    return rows


def bench_emergence(args: argparse.Namespace, brain_kw: dict) -> dict:
    """The evolved-vs-frozen proof, re-run for the model brain.

    ``perceptron`` and ``model`` are provably identical under this benchmark
    (``learning_enabled=False`` and the genome overwrites both brains' tensors
    identically), so measuring both would only print the same numbers twice —
    the baseline's emergence is covered by the locked evolution proofs.

    ``slonet``/``random`` are genome-incompatible (``Genome`` raises for the
    native stack, and there is nothing to select for in a coin flip), so they
    get no generational evolution and are reported as n/a rather than faked.
    """
    from domain.shell._internal.evolution import benchmark_emergence
    from domain.shell._internal.simulation import WorldParams

    params = WorldParams(grid_size=(16, 8, 16), brain_kind="model", **brain_kw)
    r = benchmark_emergence(params, hidden_units=args.hidden, seed=args.seeds[0])
    return {
        "perceptron": {"emerged": None, "note": "identical to 'model' under frozen learning"},
        "model": {
            "evolved_last_avg": r["evolved_last_avg"],
            "frozen_last_avg": r["frozen_last_avg"],
            "emerged": r["emerged"],
        },
        "slonet": {"emerged": None, "note": "Genome evolution is not supported"},
        "random": {"emerged": None, "note": "Genome evolution is not supported"},
    }


def _snapshot(kind: str, seed: int, pop: int, ticks: int, brain_kw: dict) -> tuple[list, list]:
    """The initial and post-run behavior-brain states of every baby."""
    from domain.shell._internal.simulation import Simulation

    _params, scene = _scene_for(kind, seed, pop, brain_kw)
    babies = list(scene.babies)
    before = [json.dumps(b.perceptron_cells.to_dict(), sort_keys=True) for b in babies]
    Simulation(scene, max_ticks=ticks).run()
    after = [json.dumps(b.perceptron_cells.to_dict(), sort_keys=True) for b in babies]
    return before, after


def bench_learning(args: argparse.Namespace, brain_kw: dict) -> dict:
    """In-life learning must move the trainable tensors (plumbing check).

    The window has to outlast ``brain_learn_every``: a baby updates one brain
    roughly once per tick, so a brain only takes a real optimizer step once
    every ``brain_learn_every`` ticks.
    """
    ticks = max(12, 2 * args.learn_every + 2)
    out: dict[str, float] = {}
    for kind in LEARNABLE_KINDS:
        moved = []
        for seed in args.seeds[:2]:
            before, after = _snapshot(kind, seed, args.pop, ticks, brain_kw)
            moved.append(float(np.mean([a != b for a, b in zip(before, after)])))
        out[kind] = float(np.mean(moved))
    return out


def _mean_energy(
    kind: str, learning: bool, seed: int, pop: int, ticks: int, brain_kw: dict
) -> float:
    """Mean baby energy over a full run — the world's honest scoreboard."""
    from domain.shell._internal.simulation import (
        SimBaby,
        SimScene,
        Simulation,
        WorldParams,
    )

    # Identical global seed for both arms => identical world AND identical
    # initial weights, so learning_enabled is the ONLY difference.
    np.random.seed(seed)
    params = WorldParams(
        grid_size=(16, 8, 16),
        generate_world=True,
        world_seed=seed,
        start_agents=pop,
        brain_kind=kind,
        learning_enabled=learning,
        **brain_kw,
    )
    scene = SimScene(params=params)
    for _ in range(pop):
        scene.add_baby(SimBaby(params=params, initial_energy=params.start_energy))
    sim = Simulation(scene, max_ticks=ticks)
    sim.run()
    return float(np.mean([r["energy"] for r in sim.tick_log]))


def bench_energy(args: argparse.Namespace, brain_kw: dict) -> dict:
    """Trained vs untrained mean energy, per learnable kind, multi-seed."""
    out: dict[str, dict] = {}
    for kind in LEARNABLE_KINDS:
        deltas = []
        for seed in args.seeds:
            trained = _mean_energy(kind, True, seed, args.pop, args.energy_ticks, brain_kw)
            untrained = _mean_energy(kind, False, seed, args.pop, args.energy_ticks, brain_kw)
            deltas.append(trained - untrained)
        out[kind] = {
            "mean_energy_delta": float(np.mean(deltas)),
            "per_seed_delta": [round(d, 4) for d in deltas],
            "seeds_won": int(sum(d > 0 for d in deltas)),
        }
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ticks", type=int, default=40, help="ticks per timed throughput window")
    ap.add_argument("--pop", type=int, default=12, help="babies per world")
    ap.add_argument("--warmup", type=int, default=10, help="untimed warmup ticks per arm")
    ap.add_argument("--repeats", type=int, default=2, help="timed windows per arm")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="world seeds")
    ap.add_argument("--energy-ticks", type=int, default=40, help="ticks per energy arm")
    ap.add_argument("--hidden", type=int, default=16, help="brain hidden units")
    ap.add_argument("--n-embed", type=int, default=0, help="slonet stack width (0 = auto)")
    ap.add_argument("--n-layer", type=int, default=1, help="slonet hidden blocks")
    ap.add_argument("--decide-every", type=int, default=1, help="slonet action-reuse window")
    ap.add_argument("--learn-every", type=int, default=8, help="slonet rewards per backward()")
    ap.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    args = ap.parse_args()
    brain_kw = _brain_kw(args)

    print(
        f"== throughput (ticks={args.ticks}x{args.repeats}, warmup={args.warmup}, "
        f"pop={args.pop}, hidden={args.hidden}, seeds={args.seeds}) =="
    )
    tp = bench_throughput(args, brain_kw)
    print(f"  {'kind':11s} {'tps (warmup in)':>16s} {'tps (excluded)':>15s} "
          f"{'reg (in)':>10s} {'reg (ex)':>9s}")
    for kind in ("perceptron", *COMPARATIVE_KINDS):
        r = tp[kind]
        reg_in = r.get("regression_pct_included")
        reg_ex = r.get("regression_pct_excluded")
        print(
            f"  {kind:11s} {r['tps_included_warmup']:16.2f} {r['tps_excluded_warmup']:15.2f} "
            f"{('-' if reg_in is None else f'{reg_in:+.1f}%'):>10s} "
            f"{('-' if reg_ex is None else f'{reg_ex:+.1f}%'):>9s}"
        )
    # The gate reads the warmup-EXCLUDED number; the included column is the
    # audit trail that shows how much (if anything) the warmup was hiding.
    gate_tp = all(tp[k]["regression_pct_excluded"] < THROUGHPUT_BUDGET_PCT for k in COMPARATIVE_KINDS)
    print(f"  gate: every kind within {THROUGHPUT_BUDGET_PCT:.0f}% of baseline (warmup excluded)")

    print(f"\n== emergence (evolved vs frozen, hidden={args.hidden}, seed={args.seeds[0]}) ==")
    em = bench_emergence(args, brain_kw)
    for kind, r in em.items():
        if r.get("emerged") is None:
            print(f"  {kind:11s} n/a — {r['note']}")
        else:
            print(
                f"  {kind:11s} evolved {r['evolved_last_avg']:8.4f}  "
                f"frozen {r['frozen_last_avg']:8.4f}  emerged={'yes' if r['emerged'] else 'no'}"
            )
    # Gated on 'model' alone: it is the only genome-compatible non-baseline
    # kind, and 'slonet'/'random' are n/a by design (see bench_emergence).
    gate_em = bool(em["model"]["emerged"])

    print("\n== in-life learning (share of babies whose brains changed) ==")
    ln = bench_learning(args, brain_kw)
    for kind, share in ln.items():
        print(f"  {kind:11s} moved {share * 100:5.1f}%")
    gate_ln = all(v > 0.0 for v in ln.values())

    print(f"\n== energy: trained - untrained mean energy ({args.energy_ticks} ticks) ==")
    en = bench_energy(args, brain_kw)
    for kind, r in en.items():
        print(
            f"  {kind:11s} delta {r['mean_energy_delta']:+9.4f} energy  "
            f"(per-seed {r['per_seed_delta']}, won {r['seeds_won']}/{len(args.seeds)})"
        )
    # Gated on the new native brain — the card's item 3. The flat kinds are
    # printed for comparison but not gated: their learning behaviour is pinned
    # by the locked proofs and is not this card's to change.
    gate_en = en["slonet"]["mean_energy_delta"] > 0.0

    ok = gate_tp and gate_em and gate_ln and gate_en
    print(
        f"\nRESULT: throughput={'PASS' if gate_tp else 'FAIL'} "
        f"emergence={'PASS' if gate_em else 'FAIL'} "
        f"learning={'PASS' if gate_ln else 'FAIL'} "
        f"energy={'PASS' if gate_en else 'FAIL'} -> {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
