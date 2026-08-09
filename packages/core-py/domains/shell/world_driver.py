"""
World driver — headless observability harness for the world realm.

Runs a world (empty grid or deterministic generated terrain) with baby agents
for a requested number of ticks and reports energy economy, material
populations, and conservation invariants. Provides a CLI wrapper so the world
can be observed and tuned without the interactive shell or a UI.

Public API:
- WorldDriver: build a world, run ticks, snapshot state, audit energy.
- main(): command-line interface (``python3 -m domains.shell.world_driver``).

The driver is pure observability: it never mutates simulation behavior and
adds no new physics. All numbers are computed from live grid arrays, never
cached placeholders.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

from .evolution import EvolutionEngine, benchmark_emergence, benchmark_social
from .simulation import (
    NUM_MATERIALS,
    SimScene,
    Simulation,
    WorldParams,
)

_MATERIAL_NAMES: dict[int, str] = {
    getattr(sys.modules["domains.shell.simulation"], _const): _const
    .removeprefix("MATERIAL_")
    .lower()
    for _const in dir(sys.modules["domains.shell.simulation"])
    if _const.startswith("MATERIAL_")
    and isinstance(getattr(sys.modules["domains.shell.simulation"], _const), int)
}


def _material_name(material_id: int) -> str:
    """Name a material id from the MATERIAL_* constants, never hardcoded."""
    return _MATERIAL_NAMES.get(int(material_id), f"material_{int(material_id)}")


class WorldDriver:
    """Headless harness: build a world, run ticks, observe energy and materials.

    Args:
        params: WorldParams to build the scene with. The driver mutates the
            live scene, so pass a fresh instance per run.
        seed: optional int. When set, ``np.random.seed(seed)`` is applied
            before scene construction so generated terrain and spawn positions
            are reproducible.
    """

    def __init__(self, params: WorldParams | None = None, seed: int | None = None):
        self.params = params or WorldParams()
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.scene = SimScene(self.params)
        self.scene.spawn_babies()
        self.sim = Simulation(self.scene, max_ticks=1 << 62)

    def material_populations(self) -> dict[int, int]:
        """
        Count cells holding each material id.

        Returns:
            Dict mapping material id to cell count. ids with no cells are 0.
        """
        counts = np.bincount(
            self.scene.world.material.astype(np.int64), minlength=NUM_MATERIALS
        )
        return {int(i): int(c) for i, c in enumerate(counts)}

    def energy_ledger(self) -> dict[str, Any]:
        """
        Energy bookkeeping: grid total, entity totals, and per-material sums.

        Returns:
            Dict with ``grid`` (sum of all cell energy), ``entities`` (sum of
            every baby's energy), ``total`` (grid + entities), and
            ``per_material`` (dict material id -> summed cell energy).
        """
        g = self.scene.world
        grid_total = float(np.sum(g.energy))
        entity_total = float(sum(b.energy for b in self.scene.babies))
        per_material: dict[int, float] = {}
        for m in range(NUM_MATERIALS):
            mask = g.material == m
            per_material[int(m)] = float(np.sum(g.energy[mask]))
        return {
            "grid": grid_total,
            "entities": entity_total,
            "total": grid_total + entity_total,
            "per_material": per_material,
        }

    def snapshot(self) -> dict[str, Any]:
        """
        Capture a point-in-time report of the world.

        Returns:
            Dict with tick, alive_babies, grid_energy, entity_energy,
            total_energy, total_signal, mean_baby_energy, and materials
            (material id -> cell count).
        """
        g = self.scene.world
        alive = self.scene.alive_babies
        grid_energy = float(np.sum(g.energy))
        entity_energy = float(sum(b.energy for b in self.scene.babies))
        mean_energy = float(np.mean([b.energy for b in alive])) if alive else 0.0
        return {
            "tick": self.scene.tick,
            "alive_babies": len(alive),
            "grid_energy": grid_energy,
            "entity_energy": entity_energy,
            "total_energy": grid_energy + entity_energy,
            "total_signal": float(np.sum(g.signal)),
            "mean_baby_energy": mean_energy,
            "materials": self.material_populations(),
        }

    def run_ticks(self, ticks: int) -> list[dict[str, Any]]:
        """
        Run ``ticks`` simulation steps, returning one snapshot per tick.

        Args:
            ticks: number of steps to advance. At least 1.

        Returns:
            List of snapshots, one per completed tick, oldest first.

        Side effects:
            - advances the underlying SimScene state.
        """
        out: list[dict[str, Any]] = []
        for _ in range(max(1, int(ticks))):
            self.sim.step()
            out.append(self.snapshot())
        return out

    def conservation_report(self, snapshots: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Check total energy never increases across consecutive snapshots.

        The world has energy sinks (rot, ember conversion to heat, living
        growth loss, passive drain, world energy_loss) but no creation, so
        grid + entity energy must be non-increasing. An increase flags a
        physics regression.

        Args:
            snapshots: list of dicts from ``run_ticks`` (or ``snapshot``).

        Returns:
            Dict with ``monotonic`` (bool), ``violations`` (list of
            ``(tick, prev_total, new_total)`` tuples), and ``start_total`` /
            ``end_total`` floats.
        """
        totals = [s["total_energy"] for s in snapshots]
        violations: list[tuple[int, float, float]] = []
        for i in range(1, len(totals)):
            if totals[i] > totals[i - 1] + 1e-6:
                violations.append((i + 1, totals[i - 1], totals[i]))
        return {
            "monotonic": len(violations) == 0,
            "violations": violations,
            "start_total": float(totals[0]) if totals else 0.0,
            "end_total": float(totals[-1]) if totals else 0.0,
        }

    def run_evolution(
        self,
        generations: int = 5,
        population: int = 8,
        ticks_per_generation: int = 20,
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run a genetic-algorithm sweep over baby genomes.

        Args:
            generations: number of generations to evolve.
            population: babies per generation.
            ticks_per_generation: ticks each generation runs.
            seed: optional seed for reproducibility (defaults to the driver's
                seed).
            kwargs: extra EvolutionEngine options.

        Returns:
            EvolutionEngine.run() summary dict.

        Side effects:
            - calls np.random.seed(seed) inside EvolutionEngine.run.
        """
        engine = EvolutionEngine(
            params=self.params,
            population_size=population,
            generations=generations,
            ticks_per_generation=ticks_per_generation,
            seed=seed if seed is not None else self.seed,
            **kwargs,
        )
        return engine.run()


def _parse_grid(value: str) -> tuple[int, int, int]:
    """Parse ``W,H,D`` into a positive 3-tuple. Raises ValueError if malformed."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"grid must be W,H,D (got {value!r})")
    dims = tuple(int(p) for p in parts)
    if any(d <= 0 for d in dims):
        raise ValueError(f"grid dimensions must be positive (got {value!r})")
    return dims  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point. Runs a world headlessly and prints a tick table + summary.

    Args:
        argv: argument list (defaults to sys.argv[1:]).

    Returns:
        Process exit code (0 on success, 1 on runtime error, 2 on bad args).

    Side effects:
        - prints report to stdout.
    """
    parser = argparse.ArgumentParser(
        prog="world_driver",
        description="Headless world-realm observability harness.",
    )
    parser.add_argument("--grid", type=_parse_grid, default=None,
                        help="world size as W,H,D (default 64,32,64; "
                             "16,8,16 for --social)")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for terrain and spawns")
    parser.add_argument("--world", action="store_true", default=True,
                        help="generate deterministic terrain (default)")
    parser.add_argument("--no-world", dest="world", action="store_false",
                        help="use an empty air grid")
    parser.add_argument("--babies", type=int, default=4,
                        help="baby agents to spawn")
    parser.add_argument("--ticks", type=int, default=50,
                        help="ticks to run")
    parser.add_argument("--every", type=int, default=5,
                        help="print a row every N ticks")
    parser.add_argument("--evolution", action="store_true",
                        help="run an evolution sweep instead of ticks")
    parser.add_argument("--generations", type=int, default=None,
                        help="generations for --evolution/--emergence/--social "
                             "(default 5, or 12 for --social)")
    parser.add_argument("--population", type=int, default=8,
                        help="population per generation for --evolution")
    parser.add_argument("--ticks-per-gen", type=int, default=20,
                        help="ticks per generation for --evolution")
    parser.add_argument("--emergence", action="store_true",
                        help="run the emergence benchmark (evolved vs frozen)")
    parser.add_argument("--hidden-units", type=int, default=0,
                        help="brain hidden units for --emergence")
    parser.add_argument("--social", action="store_true",
                        help="run the social benchmark (trait-group vs individual)")
    parser.add_argument("--social-pools", type=int, default=3,
                        help="organic food pools for --social")
    parser.add_argument("--social-ticks-per-gen", type=int, default=24,
                        help="ticks per generation for --social")
    parser.add_argument("--group-count", type=int, default=2,
                        help="tribes/territories for --social")
    parser.add_argument("--group-weight", type=float, default=0.5,
                        help="tribe-mean share of group-arm selection for --social")
    parser.add_argument("--messages", action="store_true",
                        help="enable directed inter-agent messaging (Stage 6)")
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return 2

    params = WorldParams(
        grid_size=args.grid if args.grid is not None else (
            (16, 8, 16) if args.social else (64, 32, 64)
        ),
        generate_world=args.world,
        start_agents=args.babies,
        world_seed=args.seed,
        message_enabled=args.messages,
    )

    driver = WorldDriver(params, seed=args.seed)

    generations = args.generations if args.generations is not None else (
        12 if args.social else 5
    )

    if args.social:
        result = benchmark_social(
            params,
            population_size=args.population,
            generations=generations,
            ticks_per_generation=args.social_ticks_per_gen,
            organic_pools=args.social_pools,
            group_count=args.group_count,
            group_weight=args.group_weight,
            hidden_units=args.hidden_units,
            seed=args.seed,
        )
        ind = result["individual"]
        grp = result["group"]
        print("generation ind_coop grp_coop ind_contest grp_contest "
              "ind_best grp_best")
        for ih, gh in zip(ind["history"], grp["history"]):
            print(f"{ih['generation']:<10d} {ih['cooperate_rate']:<8.4f} "
                  f"{gh['cooperate_rate']:<8.4f} {ih['contest_rate']:<12.4f} "
                  f"{gh['contest_rate']:<12.4f} {ih['best_fitness']:<8.1f} "
                  f"{gh['best_fitness']:.1f}")
        print(f"ind_coop={result['individual_cooperate_rate']:.4f} "
              f"grp_coop={result['group_cooperate_rate']:.4f} "
              f"ind_contest={result['individual_contest_rate']:.4f} "
              f"grp_contest={result['group_contest_rate']:.4f}")
        print(f"group_count={result['group_count']} "
              f"group_weight={result['group_weight']:.2f}")
        print(f"cooperation_emerged={'yes' if result['cooperation_emerged'] else 'no'}")
        return 0

    if args.emergence:
        result = benchmark_emergence(
            params,
            population_size=args.population,
            generations=generations,
            ticks_per_generation=args.ticks_per_gen,
            hidden_units=args.hidden_units,
            seed=args.seed,
        )
        evo = result["evolved"]["history"]
        fro = result["frozen"]["history"]
        print("generation evolved_avg frozen_avg evolved_best frozen_best")
        for eh, fh in zip(evo, fro):
            print(f"{eh['generation']:<10d} {eh['avg_fitness']:<12.4f} "
                  f"{fh['avg_fitness']:<12.4f} {eh['best_fitness']:<13.4f} "
                  f"{fh['best_fitness']:.4f}")
        print(f"evolved_last_avg={result['evolved_last_avg']:.4f}")
        print(f"frozen_last_avg={result['frozen_last_avg']:.4f}")
        print(f"emergence={'yes' if result['emerged'] else 'no'}")
        return 0

    if args.evolution:
        result = driver.run_evolution(
            generations=generations,
            population=args.population,
            ticks_per_generation=args.ticks_per_gen,
        )
        history = result.get("history", [])
        print("generation best_fitness avg_fitness alive")
        for h in history:
            print(f"{h['generation']:<10d} {h['best_fitness']:<12.4f} "
                  f"{h['avg_fitness']:<12.4f} {h['alive']}")
        print(f"overall_best_fitness={result.get('best_fitness', -1.0):.4f}")
        return 0

    header = "tick alive grid_energy entity_energy total_energy total_signal"
    print(header)
    for snap in driver.run_ticks(args.ticks):
        if snap["tick"] % args.every != 0:
            continue
        print(f"{snap['tick']:<5d} {snap['alive_babies']:<5d} "
              f"{snap['grid_energy']:<12.1f} {snap['entity_energy']:<14.1f} "
              f"{snap['total_energy']:<13.1f} {snap['total_signal']:<12.3f}")

    ledger = driver.energy_ledger()
    populations = driver.material_populations()
    print("\nmaterial_populations")
    for m in range(NUM_MATERIALS):
        print(f"  {_material_name(m):<8s} {populations[m]:>8d}")
    print("\nenergy_ledger")
    print(f"  grid={ledger['grid']:.1f} entities={ledger['entities']:.1f} "
          f"total={ledger['total']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
