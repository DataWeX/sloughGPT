"""
Evolution engine — genetic algorithm over baby genomes.

Each baby's genome is the weights of its three perceptrons
(cells, body, entity). A population of babies is simulated each generation;
fitness is energy at the end of the generation (0 if dead). The fittest
genomes are kept as elite, the rest are bred via tournament selection,
uniform crossover, and gaussian mutation.

Because the entity perceptron gates social behavior (cooperate/contest),
evolution can genuinely select for energy-preserving strategies: a
contest-gated genome concentrates energy into the fitter baby, while a
cooperating population keeps everyone alive. No behavior is hardcoded —
the genomes compete, and the world's energy accounting decides winners.

Stage 6 (multi-agent) splits the population into tribes with per-group
territories (spawn regions plus local food pools). Selection uses honest
trait-group (multilevel) selection when ``group_weight > 0``: tribes compete
by the geometric mean of member energy, and mating within a chosen tribe is
uniform — so an act that keeps a starving member alive (cooperation) raises
the tribe's score and spreads, instead of being out-bred by the free-rider
it fed. The social benchmark (``benchmark_social``) proves the effect by
evolving the same grouped world under pure individual selection vs
group-inclusive selection and comparing cooperation rates.

Memory is inherited across generations (the memotype): each genome also
carries the consolidated highest-reward episodes of its parent lineage,
and every offspring is born with those episodes seeded into its episodic
memory. Experience survives death through the lineage — only the weights
mutate, never the inherited memories.

Key classes:
  - Genome: serializable perceptron weights (W, b per perceptron) + inherited episodes + tribe id
  - EvolutionEngine: generation loop (simulate → score → select → breed)

Standalone: depends only on the simulation module. No API, no UI.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .simulation import (
    MATERIAL_ORGANIC,
    SimBaby,
    SimScene,
    Simulation,
    WorldParams,
)


class Genome:
    """
    Serializable perceptron weights plus an inherited memory (memotype).

    ``tensors`` are the genetic material — the three perceptrons' weights.
    ``memories`` are consolidated episodes inherited from a parent lineage:
    the highest-reward lived experiences of the parent, detached from its
    living ring buffer so they survive death and seed the offspring's own
    episodic memory at birth. Memory does not mutate — it is cultural
    (inherited as-is), not genetic.
    ``group_id`` is the tribe a lineage belongs to: offspring inherit their
    winning parent's tribe, so tribes persist and drift across generations.
    """

    def __init__(self, tensors: dict[str, np.ndarray],
                 memories: list[dict] | None = None,
                 group_id: int = 0):
        self.tensors: dict[str, np.ndarray] = {
            k: np.array(v, dtype=np.float32) for k, v in tensors.items()
        }
        self.memories: list[dict] = [
            dict(m, features=list(m["features"]), action=list(m["action"]))
            for m in (memories or [])
        ]
        self.group_id: int = int(group_id)

    @property
    def memory_count(self) -> int:
        """Number of inherited episodes carried by this genome."""
        return len(self.memories)

    @classmethod
    def from_baby(cls, baby: SimBaby, group_id: int = 0) -> "Genome":
        """
        Extract a genome from a baby: perceptron weights plus a consolidated
        memotype.

        The memotype is the baby's highest-reward episodes (capped at the
        world's memory_inherit), copied out of the living ring buffer so it
        survives death and can seed the offspring's memory at birth.

        Args:
            baby: the source agent.
            group_id: the tribe the baby's lineage belongs to.

        Returns:
            Genome with copies of cells/body/entity W and b tensors and
            serializable inherited episodes.
        """
        tensors: dict[str, np.ndarray] = {}
        names = ("cells", "body", "entity", "move")
        if hasattr(baby, "perceptron_message") and baby.perceptron_message is not None:
            names = names + ("message",)
        for name in names:
            p = getattr(baby, f"perceptron_{name}")
            tensors[f"{name}.W"] = p.W.copy()
            tensors[f"{name}.b"] = p.b.copy()
            if p.H is not None:
                tensors[f"{name}.H"] = p.H.copy()
                tensors[f"{name}.bh"] = p.bh.copy()
        cap = max(1, int(getattr(baby.params, "memory_inherit", 8)))
        memories: list[dict] = []
        for e in baby.memory.recall(cap, by_reward=True):
            memories.append({
                "features": [float(x) for x in e.features],
                "action": [float(a) for a in e.action],
                "reward": float(e.reward),
                "tick": int(e.tick),
            })
        return cls(tensors, memories=memories, group_id=group_id)

    @classmethod
    def random(cls, params: WorldParams, rng: np.random.Generator,
               group_id: int = 0) -> "Genome":
        """
        Create a random genome matching the perceptron shapes for params.

        Args:
            params: world params (defines perceptron input dims).
            rng: seeded random generator.
            group_id: the tribe this genome starts in.

        Returns:
            Genome with random W tensors, zero biases, and no memories.
        """
        shapes = {
            "cells": (params.cells_input_dim, 3),
            "body": (params.body_input_dim, 2),
            "entity": (params.entity_input_dim, 2),
            "move": (params.cells_input_dim, 3),
        }
        if params.message_enabled:
            # Directed-message brain: one gate over the target's entity
            # features. Appended last so existing genomes keep their exact
            # RNG draw order when messages are off (locked proofs).
            shapes["message"] = (params.entity_input_dim, 1)
        hidden = max(0, int(params.brain_hidden_units))
        tensors: dict[str, np.ndarray] = {}
        for name, (n_in, n_out) in shapes.items():
            readout_in = n_in + hidden if (hidden > 0 and name in ("cells", "move")) else n_in
            tensors[f"{name}.W"] = (rng.standard_normal((readout_in, n_out)) * 0.1).astype(np.float32)
            tensors[f"{name}.b"] = np.zeros(n_out, dtype=np.float32)
            if hidden > 0 and name in ("cells", "move"):
                tensors[f"{name}.H"] = (rng.standard_normal((n_in, hidden)) * 0.5).astype(np.float32)
                tensors[f"{name}.bh"] = (rng.standard_normal(hidden) * 0.1).astype(np.float32)
        return cls(tensors, group_id=group_id)

    def apply_to(self, baby: SimBaby) -> None:
        """
        Overwrite a baby's perceptron weights and seed its episodic memory
        with the genome's inherited episodes (memotype).

        Args:
            baby: the target agent (weights and memory modified in place).

        Side effects:
            - sets baby.perceptron_*.W and .b
            - records the genome's memories into baby.memory
        """
        for name in ("cells", "body", "entity", "move"):
            p = getattr(baby, f"perceptron_{name}")
            p.W[:] = self.tensors[f"{name}.W"]
            p.b[:] = self.tensors[f"{name}.b"]
            if f"{name}.H" in self.tensors:
                p.H = np.array(self.tensors[f"{name}.H"], dtype=np.float32).copy()
                p.bh = np.array(self.tensors[f"{name}.bh"], dtype=np.float32).copy()
        if (hasattr(baby, "perceptron_message")
                and baby.perceptron_message is not None
                and "message.W" in self.tensors):
            p = baby.perceptron_message
            p.W[:] = self.tensors["message.W"]
            p.b[:] = self.tensors["message.b"]
            if "message.H" in self.tensors:
                p.H = np.array(self.tensors["message.H"], dtype=np.float32).copy()
                p.bh = np.array(self.tensors["message.bh"], dtype=np.float32).copy()
        for m in self.memories:
            baby.memory.record(
                features=np.asarray(m["features"], dtype=np.float32),
                action=tuple(m["action"]),
                reward=float(m["reward"]),
                tick=int(m["tick"]),
            )

    def crossover(self, other: "Genome", rng: np.random.Generator) -> "Genome":
        """
        Uniform crossover — each element is inherited from either parent.

        Memories are inherited wholesale from the first parent (self): the
        memotype travels with the winning lineage rather than recombining.
        The tribe (group_id) is likewise inherited from the first parent, so
        a child breeds into its winning parent's territory.

        Args:
            other: the second parent genome.
            rng: seeded random generator.

        Returns:
            New Genome mixing both parents' weights element-wise and carrying
            this genome's inherited memories and group_id.
        """
        mixed: dict[str, np.ndarray] = {}
        for k, v in self.tensors.items():
            mask = rng.random(v.shape) < 0.5
            mixed[k] = np.where(mask, v, other.tensors[k]).astype(np.float32)
        return Genome(mixed, memories=self.memories, group_id=self.group_id)

    def mutate(self, rng: np.random.Generator, rate: float = 0.05,
               scale: float = 0.1) -> "Genome":
        """
        Gaussian mutation — add noise to a random subset of elements.

        Args:
            rng: seeded random generator.
            rate: probability each element is mutated.
            scale: stddev of the gaussian noise.

        Returns:
            self (mutated in place).
        """
        for k in list(self.tensors):
            v = self.tensors[k]
            mask = rng.random(v.shape) < rate
            noise = rng.standard_normal(v.shape) * scale
            self.tensors[k] = (v + mask * noise).astype(np.float32)
        return self


class EvolutionEngine:
    """
    Generation loop: simulate a population, score by energy, select and breed.

        Args:
            params: world params. Defaults to a small grid so evolution is fast.
            population_size: babies per generation (>= 2).
            generations: number of evolution cycles.
            ticks_per_generation: simulation ticks per generation.
            elite_count: top-N genomes carried forward unchanged (>= 1).
            mutation_rate: probability each weight element mutates.
            mutation_scale: stddev of mutation noise.
            organic_pools: organic food clusters placed per generation.
            learning_enabled: when False (default) the babies run with in-life
                delta-rule weight updates disabled, so selection pressure is
                the only teacher. Memory recording is unaffected.
            spawn_positions: optional fixed list of start positions, one per
                genome, reused every generation. When given, fitness is
                measured from identical starts each generation instead of
                random spawns. With ``group_count > 1`` and no explicit
                positions, one spawn per genome is auto-generated and
                clustered into per-group territories.
            group_count: number of tribes. When > 1 the population is split
                into tribes, each gets its own x-slab territory (spawn region
                plus food pools), and group membership is inherited by
                offspring so tribes persist across generations.
            group_weight: fraction of selection fitness taken from the tribe
                mean energy (rest from the individual's own energy). 0.0 is
                pure individual selection; a positive value makes a tribe's
                collective survival count toward a member's breeding chances,
                which can select for cooperative acts.
            seed: random seed for reproducible runs.
    """

    def __init__(self, params: WorldParams | None = None,
                 population_size: int = 8,
                 generations: int = 10,
                 ticks_per_generation: int = 20,
                 elite_count: int = 2,
                 mutation_rate: float = 0.05,
                 mutation_scale: float = 0.1,
                 organic_pools: int = 3,
                 learning_enabled: bool = False,
                 spawn_positions: list[np.ndarray] | None = None,
                 group_count: int = 1,
                 group_weight: float = 0.0,
                 seed: int | None = None):
        self.params = params or WorldParams(grid_size=(16, 8, 16))
        self.population_size = max(2, int(population_size))
        self.generations = max(1, int(generations))
        self.ticks_per_generation = max(1, int(ticks_per_generation))
        self.elite_count = max(1, min(int(elite_count), self.population_size))
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.organic_pools = organic_pools
        self.learning_enabled = bool(learning_enabled)
        self.spawn_positions = spawn_positions
        self.group_count = max(1, int(group_count))
        self.group_weight = float(group_weight)
        self.seed = seed
        self.history: list[dict] = []
        if self.spawn_positions is None and self.group_count > 1:
            self.spawn_positions = self._grouped_spawn_positions()

    def _group_edges(self, nx: int) -> list[int]:
        """
        X-slab boundaries partitioning the grid into group territories.

        Args:
            nx: grid width in x.

        Returns:
            List of ``group_count + 1`` boundaries; group ``g`` owns the x
            range ``[edges[g], edges[g+1])``.
        """
        return [int(round(i * nx / self.group_count))
                for i in range(self.group_count + 1)]

    def _grouped_spawn_positions(self) -> list[np.ndarray]:
        """
        One fixed spawn per genome, clustered into per-group territories.

        Genome ``i`` belongs to group ``i % group_count`` and spawns inside
        that group's x-slab on the world's surface, so each tribe starts in
        its own home region. Deterministic for a given seed.

        Returns:
            List of ``population_size`` spawn positions.
        """
        rng = np.random.default_rng(self.seed)
        ref = SimScene(params=self.params)
        nx, _, nz = self.params.grid_size
        edges = self._group_edges(nx)
        positions: list[np.ndarray] = []
        for i in range(self.population_size):
            g = i % self.group_count
            lo, hi = edges[g], max(edges[g] + 1, edges[g + 1])
            x = int(rng.integers(lo, hi))
            z = int(rng.integers(0, nz))
            y = int(ref._surface_y(x, z))
            positions.append(np.array([x, y, z], dtype=np.float64))
        return positions

    def _place_food(self, scene: SimScene, rng: np.random.Generator) -> None:
        """Scatter organic food pools at random cells, per-group when grouped.

        Each pool is a small surface cluster of organic cells (3x3 in x/z).
        A baby must absorb more than it burns per tick to build a surplus;
        with single-cell pools the net balance is negative and the cooperate
        gate (energy > start_energy) can never open, making cooperation
        structurally impossible. Clusters make surplus genuinely reachable.

        With ``group_count > 1`` the pools are distributed into each group's
        x-slab (round-robin), so every territory carries its own food supply
        and staying home is locally rewarding.
        """
        nx, ny, nz = self.params.grid_size
        edges = self._group_edges(nx) if self.group_count > 1 else None
        cluster_r = 1  # 3x3 surface cluster per pool
        for pool_idx in range(self.organic_pools):
            if edges is not None:
                g = pool_idx % self.group_count
                lo, hi = edges[g], max(edges[g] + 1, edges[g + 1])
                cx = int(rng.integers(lo, hi))
                cz = int(rng.integers(0, nz))
            else:
                cx = int(rng.integers(0, nx))
                cz = int(rng.integers(0, nz))
            for dx in range(-cluster_r, cluster_r + 1):
                for dz in range(-cluster_r, cluster_r + 1):
                    x = min(max(cx + dx, 0), nx - 1)
                    z = min(max(cz + dz, 0), nz - 1)
                    y = int(scene._surface_y(x, z))
                    scene.place_material(
                        x, y, z, MATERIAL_ORGANIC,
                        energy=float(rng.uniform(200, 800)),
                    )

    @staticmethod
    def _fitness(baby: SimBaby) -> float:
        """Fitness = energy at end of generation (0 if dead)."""
        return baby.energy

    def _group_means(self, babies: list[SimBaby],
                     genomes: list[Genome]) -> dict[int, float]:
        """Mean end-of-generation energy per tribe (for observability)."""
        sums: dict[int, float] = {}
        counts: dict[int, int] = {}
        for b, g in zip(babies, genomes):
            key = g.group_id
            sums[key] = sums.get(key, 0.0) + b.energy
            counts[key] = counts.get(key, 0) + 1
        return {k: round(sums[k] / counts[k], 4) for k in sums}

    def _home_displacement(self, babies: list[SimBaby],
                           genomes: list[Genome]) -> float:
        """
        Territoriality measure: mean distance of alive babies from their
        tribe's spawn centroid. Lower means the tribe stays in its home
        region (its food supply) rather than wandering.
        """
        if self.spawn_positions is None or not babies:
            return 0.0
        groups = [g.group_id for g in genomes]
        centroids: dict[int, list[np.ndarray]] = {}
        for i, pos in enumerate(self.spawn_positions):
            if i < len(groups):
                centroids.setdefault(groups[i], []).append(np.asarray(pos))
        means = {g: np.mean(arr, axis=0)
                 for g, arr in centroids.items() if arr}
        dists = [
            float(np.linalg.norm(b.position - means[g.group_id]))
            for b, g in zip(babies, genomes)
            if b.alive and g.group_id in means
        ]
        return float(np.mean(dists)) if dists else 0.0

    def _run_generation(self, genomes: list[Genome],
                        rng: np.random.Generator) -> tuple[list[SimBaby], list[float], dict]:
        """
        Build a scene, place food, simulate one generation.

        In a generated world the terrain comes from ``world_seed`` and the
        food pools are re-seeded from the same constant every generation, so
        each generation faces an identical environment; combined with fixed
        ``spawn_positions`` this makes fitness measure genetic quality rather
        than spawn/layout luck. Returns the babies, their raw energy fitness,
        and a social summary (cooperations, contests, territoriality).
        """
        gen_params = replace(self.params, learning_enabled=self.learning_enabled)
        if self.params.generate_world:
            food_rng = np.random.default_rng(int(self.params.world_seed))
        else:
            food_rng = rng
        scene = SimScene(params=gen_params)
        self._place_food(scene, food_rng)
        babies: list[SimBaby] = []
        for i, g in enumerate(genomes):
            position = (self.spawn_positions[i] if self.spawn_positions
                        and i < len(self.spawn_positions) else None)
            b = SimBaby(position=position, initial_energy=self.params.start_energy,
                        params=self.params, group_id=g.group_id)
            g.apply_to(b)
            scene.add_baby(b)
            babies.append(b)
        sim = Simulation(scene, max_ticks=self.ticks_per_generation)
        sim.run()
        social = sim.summary()
        total_baby_ticks = max(int(social.get("total_baby_ticks", 1)), 1)
        social["cooperate_rate"] = social["cooperations"] / total_baby_ticks
        social["contest_rate"] = social["contests"] / total_baby_ticks
        social["mean_home_displacement"] = self._home_displacement(babies, genomes)
        return babies, [self._fitness(b) for b in babies], social

    def _select(self, babies: list[SimBaby], fitnesses: list[float],
                rng: np.random.Generator,
                groups: list[int] | None = None) -> list[Genome]:
        """
        Elitism + tournament selection, then crossover + mutation to refill.

        Single-population mode (``groups`` all 0, or ``group_weight <= 0``):
        both parents of every child are population-wide tournament winners,
        so the breeding pool stays above median.

        Group mode (``group_count > 1`` and ``group_weight > 0``): two-level
        (trait-group) selection. Tribes compete BETWEEN themselves by the
        geometric mean of member energy — a tribe is only as strong as its
        weakest member, so an act that keeps a starving member alive (or
        evens out energy) lifts the whole tribe's score. Parents are then
        drawn UNIFORMLY from a chosen tribe (no within-tribe energy
        tournament, so a cooperative donor is not out-bred by the free-rider
        it fed), and the child inherits its tribe, so cooperative gene pools
        spread across generations.

        Args:
            babies: this generation's agents.
            fitnesses: parallel per-baby raw energy scores.
            rng: seeded random generator.
            groups: parallel tribe ids; None means a single tribe (group 0).

        Returns:
            Next generation's genomes.
        """
        if groups is None:
            groups = [0] * len(babies)
        if self.group_count > 1 and self.group_weight > 0.0:
            return self._select_groups(babies, fitnesses, rng, groups)

        ranked = sorted(zip(babies, fitnesses, groups),
                        key=lambda x: x[1], reverse=True)
        next_gen = [Genome.from_baby(b, group_id=grp)
                    for b, _, grp in ranked[:self.elite_count]]
        pool_size = len(ranked)
        while len(next_gen) < self.population_size:
            i = int(rng.integers(0, pool_size))
            j = int(rng.integers(0, pool_size))
            parent_a = ranked[i] if ranked[i][1] >= ranked[j][1] else ranked[j]
            k = int(rng.integers(0, pool_size))
            l = int(rng.integers(0, pool_size))
            parent_b = ranked[k] if ranked[k][1] >= ranked[l][1] else ranked[l]
            child = Genome.from_baby(parent_a[0], group_id=parent_a[2]).crossover(
                Genome.from_baby(parent_b[0], group_id=parent_b[2]), rng,
            )
            child.mutate(rng, self.mutation_rate, self.mutation_scale)
            next_gen.append(child)
        return next_gen

    def _select_groups(self, babies: list[SimBaby], fitnesses: list[float],
                       rng: np.random.Generator,
                       groups: list[int]) -> list[Genome]:
        """
        Two-level (multilevel / trait-group) selection.

        Tribe fitness is the geometric mean of member energies. Unlike the
        arithmetic mean — which a pure transfer (sharing) leaves unchanged —
        the geometric mean rewards tribes whose energy is spread evenly:
        a tribe is only as strong as its weakest member, so an act that keeps
        a starving member alive lifts the whole tribe's score, and a tribe
        whose members hoard rather than share scores lower.

        Between tribes: a tournament over that geometric-mean score decides
        which tribes breed. Within a chosen tribe both parents are drawn
        UNIFORMLY at random — not by an energy tournament — so the donor of a
        cooperative act is not out-bred by the free-rider it fed. Offspring
        inherit their tribe, so a tribe's cooperative genes travel with its
        lineage.

        Args:
            babies: this generation's agents.
            fitnesses: parallel per-baby raw energy scores.
            rng: seeded random generator.
            groups: parallel tribe ids.

        Returns:
            Next generation's genomes.
        """
        tribes: dict[int, list[tuple[SimBaby, float]]] = {}
        for b, f, g in zip(babies, fitnesses, groups):
            tribes.setdefault(g, []).append((b, f))
        tribe_ids = list(tribes)

        def geometric_mean(members: list[tuple[SimBaby, float]]) -> float:
            energies = [f for _, f in members]
            if not energies or any(e <= 0.0 for e in energies):
                return 0.0
            return float(np.exp(np.mean(np.log(energies))))

        tribe_score = {g: geometric_mean(members)
                       for g, members in tribes.items()}

        def pick_tribe() -> int:
            i, j = int(rng.integers(0, len(tribe_ids))), int(rng.integers(0, len(tribe_ids)))
            a, b = tribe_ids[i], tribe_ids[j]
            return a if tribe_score[a] >= tribe_score[b] else b

        ranked_tribes = sorted(tribe_ids, key=lambda g: tribe_score[g], reverse=True)
        next_gen = [
            Genome.from_baby(max(tribes[g], key=lambda x: x[1])[0], group_id=g)
            for g in ranked_tribes[:min(self.elite_count, len(ranked_tribes))]
        ]
        while len(next_gen) < self.population_size:
            ga = pick_tribe()
            gb = pick_tribe()
            members_a = tribes[ga]
            members_b = tribes[gb]
            parent_a = members_a[int(rng.integers(0, len(members_a)))][0]
            parent_b = members_b[int(rng.integers(0, len(members_b)))][0]
            child = Genome.from_baby(parent_a, group_id=ga).crossover(
                Genome.from_baby(parent_b, group_id=gb), rng,
            )
            child.mutate(rng, self.mutation_rate, self.mutation_scale)
            next_gen.append(child)
        return next_gen

    def run(self) -> dict:
        """
        Run the full evolution loop.

        Offspring inherit the winning parents' consolidated memories, so
        experience accumulates across generations even though the living
        ring buffers die with their babies. With ``group_count > 1`` each
        genome is born into a tribe, selection fitness blends individual and
        tribe energy, and the per-generation history carries social stats.

        Returns:
            Dict with generations, population_size, best_fitness,
            best_genome (Genome | None), inherited_episodes (episodes the
            best genome carries), and history (per-generation stats).
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        genomes = [Genome.random(self.params, rng, group_id=i % self.group_count)
                   for i in range(self.population_size)]
        best_genome: Genome | None = None
        best_fitness = -1.0

        for gen in range(1, self.generations + 1):
            babies, raw, social = self._run_generation(genomes, rng)
            gen_best = float(max(raw))
            gen_avg = float(np.mean(raw))
            alive = sum(1 for b in babies if b.alive)

            if gen_best > best_fitness:
                best_fitness = gen_best
                best_idx = int(np.argmax(raw))
                best_genome = Genome.from_baby(
                    babies[best_idx], group_id=genomes[best_idx].group_id,
                )

            entry = {
                "generation": gen,
                "best_fitness": gen_best,
                "avg_fitness": gen_avg,
                "alive": alive,
                "cooperations": social["cooperations"],
                "contests": social["contests"],
                "cooperate_rate": social["cooperate_rate"],
                "contest_rate": social["contest_rate"],
                "social_energy_moved": social["social_energy_moved"],
                "mean_home_displacement": social["mean_home_displacement"],
            }
            if self.group_count > 1:
                entry["group_means"] = self._group_means(babies, genomes)
            self.history.append(entry)

            if gen < self.generations:
                genomes = self._select(
                    babies, raw, rng,
                    groups=[g.group_id for g in genomes],
                )

        return {
            "generations": self.generations,
            "population_size": self.population_size,
            "best_fitness": best_fitness,
            "best_genome": best_genome,
            "inherited_episodes": best_genome.memory_count if best_genome else 0,
            "history": list(self.history),
        }

    def run_frozen(self) -> dict:
        """
        Baseline arm of the emergence proof: fresh random genomes every
        generation, no selection, no breeding.

        The environment (terrain, food pools, spawn positions) is identical
        to an evolved run with the same seed, so any difference in fitness
        is attributable to selection alone. Social stats are still recorded.

        Returns:
            Dict with the same keys as ``run()`` (best_genome is always None).
        """
        rng = np.random.default_rng(self.seed)
        if self.seed is not None:
            np.random.seed(self.seed)
        best_fitness = -1.0
        history: list[dict] = []
        for gen in range(1, self.generations + 1):
            genomes = [Genome.random(self.params, rng, group_id=i % self.group_count)
                       for i in range(self.population_size)]
            babies, raw, social = self._run_generation(genomes, rng)
            gen_best = float(max(raw))
            gen_avg = float(np.mean(raw))
            alive = sum(1 for b in babies if b.alive)
            if gen_best > best_fitness:
                best_fitness = gen_best
            history.append({
                "generation": gen,
                "best_fitness": gen_best,
                "avg_fitness": gen_avg,
                "alive": alive,
                "cooperations": social["cooperations"],
                "contests": social["contests"],
                "cooperate_rate": social["cooperate_rate"],
                "contest_rate": social["contest_rate"],
                "social_energy_moved": social["social_energy_moved"],
                "mean_home_displacement": social["mean_home_displacement"],
            })
        return {
            "generations": self.generations,
            "population_size": self.population_size,
            "best_fitness": best_fitness,
            "best_genome": None,
            "inherited_episodes": 0,
            "history": history,
        }


def benchmark_emergence(params: WorldParams | None = None, *,
                        population_size: int = 8,
                        generations: int = 14,
                        ticks_per_generation: int = 30,
                        organic_pools: int = 3,
                        hidden_units: int = 0,
                        shared_spawn: bool = True,
                        seed: int = 1) -> dict:
    """
    Deterministic emergence proof: evolved vs frozen-random babies.

    Both arms re-evaluate their population every generation on the SAME
    generated world (``generate_world=True``, ``world_seed=seed``), the SAME
    food pools, and the SAME fixed spawn positions, so fitness measures
    genetic quality rather than spawn/layout luck. In-life delta learning is
    disabled in both arms — selection is the only teacher.

    Args:
        params: base world rules; the benchmark forces ``generate_world=True``
            and ``world_seed=seed`` and disables learning.
        population_size: genomes per generation.
        generations: evolution cycles.
        ticks_per_generation: simulation ticks per generation.
        organic_pools: food pools scattered in the fixed world.
        spawn_positions: fixed starts for each genome, reused every
            generation; ``shared_spawn=True`` gives every genome the same
            start so all genes target the same policy.
        hidden_units: hidden projection width for the babies' brains.
        seed: world + RNG seed (identical for both arms).

    Returns:
        Dict with ``evolved`` and ``frozen`` runs (as returned by ``run()``
        and ``run_frozen()``) plus ``spawn_positions`` for reproducibility.
    """
    base = params or WorldParams(grid_size=(16, 8, 16))
    base = replace(base, generate_world=True, world_seed=seed,
                   learning_enabled=False, brain_hidden_units=int(hidden_units),
                   social_enabled=False)

    ref = SimScene(params=base)
    nx, ny, nz = base.grid_size
    rng = np.random.default_rng(seed)
    xs = rng.permutation(nx)
    zs = rng.permutation(nz)
    positions = [
        np.array([int(xs[i % nx]), int(ref._surface_y(int(xs[i % nx]), int(zs[i % nz]))),
                  int(zs[i % nz])], dtype=np.float64)
        for i in range(population_size)
    ]
    if shared_spawn:
        positions = [positions[0]] * population_size

    shared = dict(population_size=population_size, generations=generations,
                  ticks_per_generation=ticks_per_generation,
                  organic_pools=organic_pools, spawn_positions=positions,
                  seed=seed)
    evolved = EvolutionEngine(params=base, **shared).run()
    frozen = EvolutionEngine(params=base, **shared).run_frozen()

    evolved_last = [h["avg_fitness"] for h in evolved["history"]]
    frozen_last = [h["avg_fitness"] for h in frozen["history"]]
    return {
        "evolved": evolved,
        "frozen": frozen,
        "spawn_positions": positions,
        "evolved_last_avg": evolved_last[-1],
        "frozen_last_avg": frozen_last[-1],
        "emerged": bool(evolved_last[-1] > frozen_last[-1]),
    }


def benchmark_social(params: WorldParams | None = None, *,
                     population_size: int = 8,
                     generations: int = 12,
                     ticks_per_generation: int = 24,
                     organic_pools: int = 3,
                     group_count: int = 2,
                     group_weight: float = 0.5,
                     hidden_units: int = 0,
                     seed: int = 1) -> dict:
    """
    Deterministic social emergence proof (Stage 6): two selection objectives.

    Both arms evolve on the SAME grouped world: the same generated terrain,
    the same per-group food pools, and the same per-tribe spawn territories
    (auto-grouped by ``group_count``). The only difference is the selection
    objective:
      - ``individual``: pure individual fitness (``group_weight = 0``) — a
        baby that shares surplus energy lowers its own breeding chances.
      - ``group``: trait-group selection (``group_weight > 0``) — tribes
        compete by the geometric mean of member energy, so an act that keeps
        a tribe member alive preserves the tribe's score and cooperative
        genes spread.

    In-life delta learning is disabled in both arms — selection is the only
    teacher. Cooperation is never hardcoded: the entity perceptron must learn
    to open its cooperate gate (surplus shared) and close its contest gate.

    The world is deliberately scarce (``organic_pools=3`` food clusters,
    ``ticks_per_generation=24``): some babies finish the generation with a
    surplus while others near starvation, so an act that rescues a starving
    tribe-mate materially raises the tribe's geometric mean — and the donor
    pays the same cost in the individual arm, where selection punishes it.

    Args:
        params: base world rules; the benchmark forces ``generate_world=True``
            and ``world_seed=seed`` and disables learning.
        population_size: genomes per generation.
        generations: evolution cycles.
        ticks_per_generation: simulation ticks per generation.
        organic_pools: food pools distributed across the group territories.
        group_count: number of tribes (territories).
        group_weight: tribe-mean share of group-arm selection fitness.
        hidden_units: hidden projection width for the babies' brains.
        seed: world + RNG seed (identical for both arms).

    Returns:
        Dict with ``individual`` and ``group`` runs, plus their last-generation
        ``cooperate_rate``/``contest_rate`` and the ``cooperation_emerged``
        verdict (group-arm cooperation rate beats the individual arm's).
    """
    base = params or WorldParams(grid_size=(16, 8, 16))
    base = replace(base, generate_world=True, world_seed=seed,
                   learning_enabled=False, brain_hidden_units=int(hidden_units))

    shared = dict(population_size=population_size, generations=generations,
                  ticks_per_generation=ticks_per_generation,
                  organic_pools=organic_pools, group_count=group_count,
                  seed=seed)
    individual = EvolutionEngine(params=base, group_weight=0.0, **shared).run()
    group = EvolutionEngine(params=base, group_weight=group_weight, **shared).run()

    ind_rate = individual["history"][-1]["cooperate_rate"]
    grp_rate = group["history"][-1]["cooperate_rate"]
    ind_contest = individual["history"][-1]["contest_rate"]
    grp_contest = group["history"][-1]["contest_rate"]
    return {
        "individual": individual,
        "group": group,
        "group_count": group_count,
        "group_weight": group_weight,
        "individual_cooperate_rate": float(ind_rate),
        "group_cooperate_rate": float(grp_rate),
        "individual_contest_rate": float(ind_contest),
        "group_contest_rate": float(grp_contest),
        "cooperation_emerged": bool(grp_rate > ind_rate),
    }
