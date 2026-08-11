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
from .memory import WorldMemory

# Dedicated RNG stream for the cultural (teach) brain. The teach brain is
# genetic material too, but drawing it from the SHARED stream would change the
# four behavior brains' draws (and the perception-noise stream) whenever
# teaching is enabled — breaking the locked proofs and un-controlling the
# culture benchmark. A dedicated stream keeps every other draw bit-identical
# whether or not teaching is on.
_TEACH_RNG_SEED = 0xC001


def _teach_rng(group_id: int = 0) -> np.random.Generator:
    """Seeded Generator for the cultural brain, independent of the shared one.

    Args:
        group_id: tribe id, so the initial teach gate varies across tribes.

    Returns:
        np.random.Generator whose draws never touch the shared RNG streams.
    """
    return np.random.default_rng(_TEACH_RNG_SEED + int(group_id))


# Seed for the predator brain's dedicated RNG stream. The predation weights
# are drawn from their own generator (``_PREDATION_RNG_SEED + group_id``) so
# enabling predation never perturbs the shared stream — and therefore the
# four behavior brains' draws (and the perception-noise stream) stay
# bit-identical whether or not predation is enabled (locked proofs and a
# controlled predator-prey benchmark).
_PREDATION_RNG_SEED = 0x0BAD


def _predation_rng(group_id: int = 0) -> np.random.Generator:
    """Seeded Generator for the predator brain, independent of the shared one.

    Args:
        group_id: tribe id, so the initial predation gate varies across
            tribes.

    Returns:
        np.random.Generator whose draws never touch the shared RNG streams.
    """
    return np.random.default_rng(_PREDATION_RNG_SEED + int(group_id))


# Seed for the territory brain's dedicated RNG stream. The territory weights
# are drawn from their own generator (``_TERRITORY_RNG_SEED + group_id``) so
# enabling territoriality never perturbs the shared stream — and therefore
# the four behavior brains' draws (and the perception-noise stream) stay
# bit-identical whether or not territoriality is enabled (locked proofs and
# a controlled territoriality benchmark).
_TERRITORY_RNG_SEED = 0x1E07

_REPRODUCE_RNG_SEED = 0x1E08


def _reproduce_rng(group_id: int = 0) -> np.random.Generator:
    """Seeded Generator for the reproduction brain, independent of the shared one.

    Args:
        group_id: tribe id, so the initial reproduce gate varies across
            tribes.

    Returns:
        np.random.Generator whose draws never touch the shared RNG streams.
    """
    return np.random.default_rng(_REPRODUCE_RNG_SEED + int(group_id))


# Stage 11: the role (Builder/Warrior) brain draws from its own dedicated
# stream so the shared stream — and therefore the four behavior brains'
# draws — stay bit-identical whether or not specialization is enabled
# (locked proofs and a controlled division-of-labor benchmark).
_ROLE_RNG_SEED = 0x1E09


def _role_rng(group_id: int = 0) -> np.random.Generator:
    """Seeded Generator for the role brain, independent of the shared one.

    Args:
        group_id: tribe id, so the initial role posture varies across
            tribes.

    Returns:
        np.random.Generator whose draws never touch the shared RNG streams.
    """
    return np.random.default_rng(_ROLE_RNG_SEED + int(group_id))


def _territory_rng(group_id: int = 0) -> np.random.Generator:
    """Seeded Generator for the territory brain, independent of the shared one.

    Args:
        group_id: tribe id, so the initial territory gate varies across
            tribes.

    Returns:
        np.random.Generator whose draws never touch the shared RNG streams.
    """
    return np.random.default_rng(_TERRITORY_RNG_SEED + int(group_id))


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
        if hasattr(baby, "perceptron_teach") and baby.perceptron_teach is not None:
            names = names + ("teach",)
        if (hasattr(baby, "perceptron_predation")
                and baby.perceptron_predation is not None):
            names = names + ("predation",)
        if (hasattr(baby, "perceptron_territory")
                and baby.perceptron_territory is not None):
            names = names + ("territory",)
        if (hasattr(baby, "perceptron_reproduce")
                and baby.perceptron_reproduce is not None):
            names = names + ("reproduce",)
        if (hasattr(baby, "perceptron_role")
                and baby.perceptron_role is not None):
            names = names + ("role",)
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
        if params.teaching_enabled:
            # Cultural brain: one gate over the target's entity features,
            # drawn from the DEDICATED teach stream so the shared stream —
            # and therefore the four behavior brains' draws — is unchanged
            # whether or not teaching is enabled (locked proofs and a
            # perfectly controlled culture benchmark).
            teach = _teach_rng(group_id)
            tensors["teach.W"] = (teach.standard_normal((params.entity_input_dim, 1)) * 0.1).astype(np.float32)
            tensors["teach.b"] = np.zeros(1, dtype=np.float32)
        if params.predation_enabled:
            # Predator brain: one gate over the target's entity features,
            # drawn from the DEDICATED predation stream so the shared stream
            # — and therefore the four behavior brains' draws — is unchanged
            # whether or not predation is enabled (locked proofs and a
            # perfectly controlled predator-prey benchmark).
            pred = _predation_rng(group_id)
            tensors["predation.W"] = (pred.standard_normal((params.entity_input_dim, 1)) * 0.1).astype(np.float32)
            tensors["predation.b"] = np.zeros(1, dtype=np.float32)
        if params.territoriality_enabled:
            # Territory brain: one gate over a trespasser's entity features,
            # drawn from the DEDICATED territory stream so the shared stream
            # — and therefore the four behavior brains' draws — is unchanged
            # whether or not territoriality is enabled (locked proofs and a
            # perfectly controlled territoriality benchmark).
            terr = _territory_rng(group_id)
            tensors["territory.W"] = (terr.standard_normal((params.entity_input_dim, 1)) * 0.1).astype(np.float32)
            tensors["territory.b"] = np.zeros(1, dtype=np.float32)
        if params.lifecycle_enabled:
            # Reproduction brain: one gate over the parent's own body state,
            # drawn from the DEDICATED reproduce stream so the shared stream
            # — and therefore the four behavior brains' draws — is unchanged
            # whether or not lifecycle is enabled (locked proofs and a
            # perfectly controlled lifecycle benchmark).
            repro = _reproduce_rng(group_id)
            tensors["reproduce.W"] = (repro.standard_normal((params.body_input_dim, 1)) * 0.1).astype(np.float32)
            tensors["reproduce.b"] = np.zeros(1, dtype=np.float32)
        if params.specialization_enabled:
            # Role brain: one gate over the baby's own body state — its
            # posture — drawn from the DEDICATED role stream so the shared
            # stream — and therefore the four behavior brains' draws — is
            # unchanged whether or not specialization is enabled (locked
            # proofs and a perfectly controlled division-of-labor
            # benchmark).
            role = _role_rng(group_id)
            tensors["role.W"] = (role.standard_normal((params.body_input_dim, 1)) * 0.1).astype(np.float32)
            tensors["role.b"] = np.zeros(1, dtype=np.float32)
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
        if (hasattr(baby, "perceptron_teach")
                and baby.perceptron_teach is not None
                and "teach.W" in self.tensors):
            p = baby.perceptron_teach
            p.W[:] = self.tensors["teach.W"]
            p.b[:] = self.tensors["teach.b"]
        if (hasattr(baby, "perceptron_predation")
                and baby.perceptron_predation is not None
                and "predation.W" in self.tensors):
            p = baby.perceptron_predation
            p.W[:] = self.tensors["predation.W"]
            p.b[:] = self.tensors["predation.b"]
        if (hasattr(baby, "perceptron_territory")
                and baby.perceptron_territory is not None
                and "territory.W" in self.tensors):
            p = baby.perceptron_territory
            p.W[:] = self.tensors["territory.W"]
            p.b[:] = self.tensors["territory.b"]
        if (hasattr(baby, "perceptron_reproduce")
                and baby.perceptron_reproduce is not None
                and "reproduce.W" in self.tensors):
            p = baby.perceptron_reproduce
            p.W[:] = self.tensors["reproduce.W"]
            p.b[:] = self.tensors["reproduce.b"]
        if (hasattr(baby, "perceptron_role")
                and baby.perceptron_role is not None
                and "role.W" in self.tensors):
            p = baby.perceptron_role
            p.W[:] = self.tensors["role.W"]
            p.b[:] = self.tensors["role.b"]
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
            if k.startswith("teach."):
                mask_rng = _teach_rng(self.group_id)
            elif k.startswith("predation."):
                mask_rng = _predation_rng(self.group_id)
            elif k.startswith("territory."):
                mask_rng = _territory_rng(self.group_id)
            elif k.startswith("reproduce."):
                mask_rng = _reproduce_rng(self.group_id)
            elif k.startswith("role."):
                mask_rng = _role_rng(self.group_id)
            else:
                mask_rng = rng
            mask = mask_rng.random(v.shape) < 0.5
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
            if k.startswith("teach."):
                draw_rng = _teach_rng(self.group_id)
            elif k.startswith("predation."):
                draw_rng = _predation_rng(self.group_id)
            elif k.startswith("territory."):
                draw_rng = _territory_rng(self.group_id)
            elif k.startswith("reproduce."):
                draw_rng = _reproduce_rng(self.group_id)
            elif k.startswith("role."):
                draw_rng = _role_rng(self.group_id)
            else:
                draw_rng = rng
            mask = draw_rng.random(v.shape) < rate
            noise = draw_rng.standard_normal(v.shape) * scale
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

        The engine carries a ``WorldMemory`` reservoir whenever
        ``params.memory_enabled``: every generation's scene shares the same
        reservoir, dead babies deposit in-sim, survivors deposit at the
        generation boundary, and the next generation's newborns are seeded
        from it — so lived experience accumulates across generations, beyond
        the capped parent->child memotype.
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
        # World-level long-term memory: one reservoir per engine run, shared
        # by every generation's scene so deposits survive across generations.
        self.world_memory = (WorldMemory()
                             if self.params.memory_enabled else None)
        self._run_seeds_given = 0
        self._run_seeds_total = 0
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
        scene = SimScene(params=gen_params, world_memory=self.world_memory)
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
        # Generation boundary: every survivor also deposits its best episodes
        # into the world reservoir (dead babies deposited in-sim at death), so
        # the collective record grows monotonically across generations.
        for b in babies:
            if b.alive:
                scene.deposit_memory(b)
        self._run_seeds_given = scene.memory_seeds_given
        self._run_seeds_total += self._run_seeds_given
        social = sim.summary()
        total_baby_ticks = max(int(social.get("total_baby_ticks", 1)), 1)
        social["cooperate_rate"] = social["cooperations"] / total_baby_ticks
        social["contest_rate"] = social["contests"] / total_baby_ticks
        social["teach_rate"] = social["lessons"] / total_baby_ticks
        social["predation_rate"] = social["predations"] / total_baby_ticks
        social["defend_rate"] = social["defenses"] / total_baby_ticks
        social["role_deposit_rate"] = social["role_deposits"] / total_baby_ticks
        social["role_raid_rate"] = social["role_raids"] / total_baby_ticks
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

        self._run_seeds_total = 0

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

            seeds = self._run_seeds_given
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
                "lessons": social["lessons"],
                "teach_rate": social["teach_rate"],
                "predations": social["predations"],
                "predation_rate": social["predation_rate"],
                "predation_energy_moved": social["predation_energy_moved"],
                "defenses": social["defenses"],
                "defend_rate": social["defend_rate"],
                "defend_energy_moved": social["defend_energy_moved"],
                "raids": social["raids"],
                "raid_energy_moved": social["raid_energy_moved"],
                "nests_built": social["nests_built"],
                "births": social["births"],
                "birth_energy_moved": social["birth_energy_moved"],
                "role_deposits": social["role_deposits"],
                "role_deposit_rate": social["role_deposit_rate"],
                "role_deposit_energy": social["role_deposit_energy"],
                "role_raids": social["role_raids"],
                "role_raid_rate": social["role_raid_rate"],
                "role_raid_energy": social["role_raid_energy"],
                "deaths": social["deaths"],
                "alive_count": social["alive_count"],
                "memory_size": len(self.world_memory)
                if self.world_memory is not None else 0,
                "memory_seeds": seeds,
                "solar_energy_deposited": float(
                    social.get("solar_energy_deposited", 0.0)),
                "sunshine": float(social.get("sunshine", 0.0)),
                "light_final": float(social.get("light_final", 0.0)),
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
            "memory_size": len(self.world_memory)
            if self.world_memory is not None else 0,
            "memory_seeds_total": self._run_seeds_total,
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
                "lessons": social["lessons"],
                "teach_rate": social["teach_rate"],
                "predations": social["predations"],
                "predation_rate": social["predation_rate"],
                "predation_energy_moved": social["predation_energy_moved"],
                "defenses": social["defenses"],
                "defend_rate": social["defend_rate"],
                "defend_energy_moved": social["defend_energy_moved"],
                "raids": social["raids"],
                "raid_energy_moved": social["raid_energy_moved"],
                "nests_built": social["nests_built"],
                "role_deposits": social["role_deposits"],
                "role_deposit_rate": social["role_deposit_rate"],
                "role_deposit_energy": social["role_deposit_energy"],
                "role_raids": social["role_raids"],
                "role_raid_rate": social["role_raid_rate"],
                "role_raid_energy": social["role_raid_energy"],
                "memory_size": 0,
                "memory_seeds": 0,
            })
        return {
            "generations": self.generations,
            "population_size": self.population_size,
            "best_fitness": best_fitness,
            "best_genome": None,
            "inherited_episodes": 0,
            "history": history,
            "memory_size": 0,
            "memory_seeds_total": 0,
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


def benchmark_specialization(params: WorldParams | None = None, *,
                             population_size: int = 8,
                             generations: int = 12,
                             ticks_per_generation: int = 24,
                             organic_pools: int = 3,
                             group_count: int = 2,
                             group_weight: float = 0.5,
                             hidden_units: int = 0,
                             seed: int = 1) -> dict:
    """
    Deterministic division-of-labor proof (Stage 11): two role channels.

    Both arms evolve on the SAME grouped, structured world: the same
    generated terrain, the same per-group food pools, the same per-tribe
    spawn territories, nests that cell-writes bank into
    (``structure_enabled``), and territoriality — so banks exist and can be
    fed or drained in both arms. In-life delta learning is disabled in both
    arms — selection is the only teacher. The only difference is the role
    channel:
      - ``control``: specialization disabled. Nests are fed only by the
        noisy cell-write deposits; a baby draws only when starving, and
        raids only when hungry on foreign ground.
      - ``specialization``: specialization enabled. Each baby carries a
        heritable role brain (drawn from the DEDICATED role stream, so the
        four behavior brains are bit-identical between arms) whose gate is a
        posture: below ``role_gate_threshold`` the baby is a BUILDER that
        deliberately banks surplus into its tribe's nearest nest; at or
        above it the baby is a WARRIOR that raids a foreign tribe's bank
        even when not hungry. Selection is trait-group
        (``group_weight > 0``): tribes compete by the geometric mean of
        member energy, so complementary postures — one lifting the tribe's
        famine floor, one lifting its mean — are rewarded as a package.

    Args:
        params: base world rules; the benchmark forces ``generate_world=True``,
            ``world_seed=seed``, ``structure_enabled=True``,
            ``territoriality_enabled=True`` and disables learning.
        population_size: genomes per generation.
        generations: evolution cycles.
        ticks_per_generation: simulation ticks per generation.
        organic_pools: food pools distributed across the group territories.
        group_count: number of tribes (territories).
        group_weight: tribe-mean share of selection fitness.
        hidden_units: hidden projection width for the babies' brains.
        seed: world + RNG seed (identical for both arms).

    Returns:
        Dict with ``control`` and ``specialization`` runs, each arm's
        last-generation ``role_deposit_rate``/``role_raid_rate`` and final
        ``avg_fitness``, and the ``specialization_emerged`` verdict: the
        specialization arm actually fielded BOTH postures (positive deposit
        AND raid rates) while matching the control arm's final mean fitness.
    """
    base = params or WorldParams(grid_size=(16, 8, 16))
    base = replace(base, generate_world=True, world_seed=seed,
                   learning_enabled=False, brain_hidden_units=int(hidden_units),
                   structure_enabled=True, territoriality_enabled=True,
                   write_energy_scale=10.0)

    shared = dict(population_size=population_size, generations=generations,
                  ticks_per_generation=ticks_per_generation,
                  organic_pools=organic_pools, group_count=group_count,
                  group_weight=group_weight, seed=seed)
    control = EvolutionEngine(params=base, **shared).run()
    spec = EvolutionEngine(
        params=replace(base, specialization_enabled=True), **shared,
    ).run()

    def last(arm: dict, key: str) -> float:
        return float(arm["history"][-1].get(key, 0.0))

    c_dep = last(control, "role_deposit_rate")
    c_raid = last(control, "role_raid_rate")
    s_dep = last(spec, "role_deposit_rate")
    s_raid = last(spec, "role_raid_rate")
    c_avg = last(control, "avg_fitness")
    s_avg = last(spec, "avg_fitness")
    return {
        "control": control,
        "specialization": spec,
        "group_count": group_count,
        "group_weight": group_weight,
        "control_role_deposit_rate": c_dep,
        "control_role_raid_rate": c_raid,
        "specialization_role_deposit_rate": s_dep,
        "specialization_role_raid_rate": s_raid,
        "control_final_avg_fitness": c_avg,
        "specialization_final_avg_fitness": s_avg,
        "specialization_emerged": bool(
            s_dep > 0.0 and s_raid > 0.0 and s_avg >= c_avg
        ),
    }


def benchmark_culture(params: WorldParams | None = None, *,
                      population_size: int = 8,
                      generations: int = 12,
                      ticks_per_generation: int = 24,
                      organic_pools: int = 3,
                      group_count: int = 2,
                      group_weight: float = 0.5,
                      hidden_units: int = 0,
                      seed: int = 1) -> dict:
    """
    Deterministic cultural transmission proof (Stage 7): two transmission
    channels.

    Both arms evolve on the SAME grouped world — the same generated terrain,
    the same per-group food pools, the same per-tribe spawn territories, and
    the same initial core genomes (the teach brain's weights are drawn last,
    so the four behavior brains are bit-identical between arms). In-life
    delta-rule learning is enabled in BOTH arms: the only difference is the
    transmission channel:
      - ``control``: teaching disabled. Learned behavior moves vertically
        only — consolidated into an offspring's memotype at birth.
      - ``culture``: teaching enabled. A baby with surplus can teach the
        neediest tribe-mate in range: the student's behavior weights blend
        toward the teacher's learned weights and up to ``teach_memotype_cap``
        of the teacher's best episodes are copied into the student's memory
        (default 1; bulk copies raised the student's reward baseline and
        dampened its learning, so episode transfer is kept minimal). Learned
        behavior now also moves laterally between living agents.

    Teaching is never hardcoded: the teach perceptron must learn to open its
    gate (and pay the energy cost) only when the lesson is worth it.

    Args:
        params: base world rules; the benchmark forces ``generate_world=True``,
            ``world_seed=seed`` and ``learning_enabled=True``.
        population_size: genomes per generation.
        generations: evolution cycles.
        ticks_per_generation: simulation ticks per generation.
        organic_pools: food pools distributed across the group territories.
        group_count: number of tribes (territories).
        group_weight: tribe-mean share of group-arm selection fitness.
        hidden_units: hidden projection width for the babies' brains.
        seed: world + RNG seed (identical for both arms).

    Returns:
        Dict with ``control`` and ``culture`` runs, their last-generation
        ``avg_fitness``/``teach_rate`` and the ``culture_emerged`` verdict
        (culture-arm final average fitness beats the control arm's).
    """
    base = params or WorldParams(grid_size=(16, 8, 16))
    base = replace(base, generate_world=True, world_seed=seed,
                   learning_enabled=True, brain_hidden_units=int(hidden_units),
                   teaching_enabled=False)

    shared = dict(population_size=population_size, generations=generations,
                  ticks_per_generation=ticks_per_generation,
                  organic_pools=organic_pools, group_count=group_count,
                  group_weight=group_weight, seed=seed)
    control = EvolutionEngine(params=base, **shared).run()
    culture = EvolutionEngine(params=replace(base, teaching_enabled=True),
                              **shared).run()

    ctrl_avg = control["history"][-1]["avg_fitness"]
    cult_avg = culture["history"][-1]["avg_fitness"]
    return {
        "control": control,
        "culture": culture,
        "group_count": group_count,
        "group_weight": group_weight,
        "control_last_avg": float(ctrl_avg),
        "culture_last_avg": float(cult_avg),
        "control_teach_rate": float(control["history"][-1]["teach_rate"]),
        "culture_teach_rate": float(culture["history"][-1]["teach_rate"]),
        "culture_emerged": bool(cult_avg > ctrl_avg),
    }


def benchmark_memory(params: WorldParams | None = None, *,
                     population_size: int = 8,
                     generations: int = 12,
                     ticks_per_generation: int = 24,
                     organic_pools: int = 3,
                     group_count: int = 2,
                     group_weight: float = 0.5,
                     hidden_units: int = 0,
                     seed: int = 1) -> dict:
    """
    Deterministic long-term memory proof (Stage 7): two memory channels.

    Both arms evolve on the SAME grouped world — the same generated terrain,
    the same per-group food pools, the same per-tribe spawn territories, and
    the same initial core genomes. In-life delta-rule learning is enabled in
    BOTH arms and teaching is off in both: the only difference is how lived
    experience persists:
      - ``control``: memory disabled. Experience survives only through the
        memotype — each offspring inherits its winning parents' consolidated
        episodes at birth, capped and parent->child only.
      - ``memory``: the world keeps an append-only reservoir. Every dead baby
        deposits its best episodes in-sim and every survivor deposits at the
        generation boundary; the reservoir never evicts, so the collective
        record spans the whole run; and each newborn is seeded with the
        reservoir's best episodes on top of its memotype, so experience also
        crosses lineages.

    Memory is never hardcoded: the reservoir holds honest episode rewards and
    only the episodes babies actually learned from.

    Args:
        params: base world rules; the benchmark forces ``generate_world=True``,
            ``world_seed=seed`` and ``learning_enabled=True``.
        population_size: genomes per generation.
        generations: evolution cycles.
        ticks_per_generation: simulation ticks per generation.
        organic_pools: food pools distributed across the group territories.
        group_count: number of tribes (territories).
        group_weight: tribe-mean share of group-arm selection fitness.
        hidden_units: hidden projection width for the babies' brains.
        seed: world + RNG seed (identical for both arms).

    Returns:
        Dict with ``control`` and ``memory`` runs, the last-generation
        ``avg_fitness`` of each, the final reservoir ``memory_size`` and total
        ``memory_seeds`` of the memory arm, and the ``memory_emerged`` verdict
        (memory-arm final average fitness beats the control arm's).
    """
    base = params or WorldParams(grid_size=(16, 8, 16))
    base = replace(base, generate_world=True, world_seed=seed,
                   learning_enabled=True, brain_hidden_units=int(hidden_units),
                   teaching_enabled=False, memory_enabled=False)

    shared = dict(population_size=population_size, generations=generations,
                  ticks_per_generation=ticks_per_generation,
                  organic_pools=organic_pools, group_count=group_count,
                  group_weight=group_weight, seed=seed)
    control = EvolutionEngine(params=base, **shared).run()
    memory = EvolutionEngine(params=replace(base, memory_enabled=True),
                             **shared).run()

    ctrl_avg = control["history"][-1]["avg_fitness"]
    mem_avg = memory["history"][-1]["avg_fitness"]
    return {
        "control": control,
        "memory": memory,
        "group_count": group_count,
        "group_weight": group_weight,
        "control_last_avg": float(ctrl_avg),
        "memory_last_avg": float(mem_avg),
        "memory_size": int(memory["memory_size"]),
        "memory_seeds": int(memory["memory_seeds_total"]),
        "memory_emerged": bool(mem_avg > ctrl_avg),
    }


def benchmark_predation(params: WorldParams | None = None, *,
                        population_size: int = 8,
                        generations: int = 12,
                        ticks_per_generation: int = 24,
                        organic_pools: int = 3,
                        group_count: int = 2,
                        group_weight: float = 0.5,
                        hidden_units: int = 0,
                        seed: int = 1) -> dict:
    """
    Deterministic predator-prey proof (Stage 8): two interaction channels.

    Both arms evolve on the SAME grouped world — the same generated terrain,
    the same per-group food pools, the same per-tribe spawn territories, and
    the same initial core genomes (the predation brain's weights are drawn
    last from a dedicated stream, so the four behavior brains are
    bit-identical between arms). In-life delta-rule learning is enabled in
    BOTH arms and teaching/memory are off in both: the only difference is the
    interaction channel:
      - ``control``: predation disabled. Agents gather, forage, and may
        contest neighbors (a small energy theft), but no strike is lethal.
      - ``predation``: a baby can hunt the weakest nearby baby within range
        when its predation gate clears ``predation_gate_threshold``: the
        strike is lethal, transfers the prey's full energy to the predator,
        and costs ``predation_cost``.

    Predation is never hardcoded: the predation perceptron must learn to
    open its gate only when the hunt pays (prey energy exceeds the strike
    cost), and group selection shapes the balance between predators and prey
    — the kin feature lets a tribe learn not to eat its own members while
    hunting rival tribes.

    Args:
        params: base world rules; the benchmark forces ``generate_world=True``,
            ``world_seed=seed`` and ``learning_enabled=True``.
        population_size: genomes per generation.
        generations: evolution cycles.
        ticks_per_generation: simulation ticks per generation.
        organic_pools: food pools distributed across the group territories.
        group_count: number of tribes (territories).
        group_weight: tribe-mean share of group-arm selection fitness.
        hidden_units: hidden projection width for the babies' brains.
        seed: world + RNG seed (identical for both arms).

    Returns:
        Dict with ``control`` and ``predation`` runs, their last-generation
        ``avg_fitness``, the predation arm's ``predation_rate``,
        ``predations`` count and total ``predation_energy_moved``, and the
        ``predation_emerged`` verdict (predation-arm final average fitness
        beats the control arm's).
    """
    base = params or WorldParams(grid_size=(16, 8, 16))
    base = replace(base, generate_world=True, world_seed=seed,
                   learning_enabled=True, brain_hidden_units=int(hidden_units),
                   teaching_enabled=False, memory_enabled=False,
                   predation_enabled=False)

    shared = dict(population_size=population_size, generations=generations,
                  ticks_per_generation=ticks_per_generation,
                  organic_pools=organic_pools, group_count=group_count,
                  group_weight=group_weight, seed=seed)
    control = EvolutionEngine(params=base, **shared).run()
    predation = EvolutionEngine(params=replace(base, predation_enabled=True),
                                **shared).run()

    ctrl_avg = control["history"][-1]["avg_fitness"]
    pred_avg = predation["history"][-1]["avg_fitness"]
    return {
        "control": control,
        "predation": predation,
        "group_count": group_count,
        "group_weight": group_weight,
        "control_last_avg": float(ctrl_avg),
        "predation_last_avg": float(pred_avg),
        "predation_rate": float(predation["history"][-1]["predation_rate"]),
        "predations": int(predation["history"][-1]["predations"]),
        "predation_energy_moved": float(
            predation["history"][-1]["predation_energy_moved"]),
        "predation_emerged": bool(pred_avg > ctrl_avg),
    }


def benchmark_territoriality(params: WorldParams | None = None, *,
                             population_size: int = 8,
                             generations: int = 12,
                             ticks_per_generation: int = 24,
                             organic_pools: int = 3,
                             group_count: int = 2,
                             group_weight: float = 0.5,
                             hidden_units: int = 0,
                             seed: int = 1) -> dict:
    """
    Deterministic territoriality proof (Stage 9): the defense channel.

    Both arms evolve on the SAME grouped world — the same generated terrain,
    the same per-group food pools, the same per-tribe spawn territories, and
    the same initial core genomes. Durable nests are enabled in BOTH arms
    (territories are claimed by building nests, so the same structures exist
    either way); ``write_energy_scale`` is raised so energy-depositing writes
    actually seed nests and territories form. The territory brain's weights
    are drawn last from a dedicated stream, so the four behavior brains are
    bit-identical between arms. In-life delta-rule learning is enabled in both
    arms and teaching/memory are off in both: the only difference is the
    interaction channel:
      - ``control``: territoriality disabled. Tribes gather, forage, and
        build nests, but the nest banks are never raided and standing on
        their own ground never evicts anyone.
      - ``territoriality``: the full two-sided channel. A hungry baby
        standing on foreign ground (within ``territory_radius`` of a foreign
        tribe's nearest nest) RAIDS that bank — ``nest_draw_rate`` per tick,
        a one-way drain. Standing within ``territory_radius`` of its tribe's
        nearest nest, a baby whose territory gate clears
        ``defend_gate_threshold`` evicts the nearest foreign baby within
        ``defend_range`` — shoving it ``defend_push`` cells away (a pure
        relocation, never a kill) and transferring ``defend_take_fraction``
        of its energy to the defender (a toll that scales with what the
        trespasser carries), who pays ``defend_cost``. Eviction keeps a rival
        raider off the tribe's shared bank, so defense protects a real
        resource (all transfers — the world conserves energy).

    Territoriality is never hardcoded: the territory perceptron must learn to
    open its gate only when the eviction pays (the trespasser carries more
    energy than the eviction costs), and the honest same-tick net reward
    shapes the gate from the true outcome. The kin feature lets a tribe
    defend its own region while leaving foreign ground alone.

    Args:
        params: base world rules; the benchmark forces ``generate_world=True``,
            ``world_seed=seed``, ``structure_enabled=True`` and
            ``learning_enabled=True``.
        population_size: genomes per generation.
        generations: evolution cycles.
        ticks_per_generation: simulation ticks per generation.
        organic_pools: food pools distributed across the group territories.
        group_count: number of tribes (territories).
        group_weight: tribe-mean share of group-arm selection fitness.
        hidden_units: hidden projection width for the babies' brains.
        seed: world + RNG seed (identical for both arms).

    Returns:
        Dict with ``control`` and ``territoriality`` runs, their last-
        generation ``avg_fitness``, the territoriality arm's ``defend_rate``,
        ``defenses`` count, total ``defend_energy_moved``, ``raids`` and total
        ``raid_energy_moved``, and the ``territoriality_emerged`` verdict
        (territoriality-arm final average fitness beats the control arm's).
    """
    base = params or WorldParams(grid_size=(16, 8, 16))
    base = replace(base, generate_world=True, world_seed=seed,
                   learning_enabled=True, brain_hidden_units=int(hidden_units),
                   teaching_enabled=False, memory_enabled=False,
                   structure_enabled=True, write_energy_scale=10.0,
                   territoriality_enabled=False)

    shared = dict(population_size=population_size, generations=generations,
                  ticks_per_generation=ticks_per_generation,
                  organic_pools=organic_pools, group_count=group_count,
                  group_weight=group_weight, seed=seed)
    control = EvolutionEngine(params=base, **shared).run()
    territory = EvolutionEngine(
        params=replace(base, territoriality_enabled=True), **shared).run()

    ctrl_avg = control["history"][-1]["avg_fitness"]
    terr_avg = territory["history"][-1]["avg_fitness"]
    return {
        "control": control,
        "territoriality": territory,
        "group_count": group_count,
        "group_weight": group_weight,
        "control_last_avg": float(ctrl_avg),
        "territoriality_last_avg": float(terr_avg),
        "defend_rate": float(territory["history"][-1]["defend_rate"]),
        "defenses": int(territory["history"][-1]["defenses"]),
        "defend_energy_moved": float(
            territory["history"][-1]["defend_energy_moved"]),
        "raids": int(territory["history"][-1]["raids"]),
        "raid_energy_moved": float(
            territory["history"][-1]["raid_energy_moved"]),
        "territoriality_emerged": bool(terr_avg > ctrl_avg),
    }


def benchmark_lifecycle(params: WorldParams | None = None, *,
                        population_size: int = 8,
                        generations: int = 12,
                        ticks_per_generation: int = 24,
                        organic_pools: int = 3,
                        group_count: int = 2,
                        group_weight: float = 0.5,
                        hidden_units: int = 0,
                        seed: int = 1) -> dict:
    """
    Deterministic life-cycle proof (Stage 10): births and deaths in-tick.

    Both arms evolve on the SAME grouped world — the same generated terrain,
    the same per-group food pools, the same per-tribe spawn territories, and
    the same initial core genomes. Durable nests are enabled in BOTH arms
    (the tribe's nest bank is the pool a birth can draw from, so the same
    structures exist either way) and in-life learning is enabled in both. The
    reproduce brain's weights are drawn last from a dedicated stream, so the
    four behavior brains are bit-identical between arms; the only difference
    is the interaction channel:
      - ``control``: lifecycle disabled. The engine re-seeds the full
        population every generation (itself replaced by selection), so
        population size is constant by construction and nothing breeds inside
        a tick.
      - ``lifecycle``: the in-world life cycle. A baby whose reproduce gate
        clears while it stands above ``reproduce_energy_threshold`` spawns an
        offspring near itself; the child's ``birth_cost`` is transferred from
        the tribe's nearest nest bank and the parent (never created), so the
        world conserves energy while its population self-sustains within the
        ``max_entities`` cap. Starvation still removes babies every tick, so
        deaths happen in the same loop.

    Lifecycle is never hardcoded: the reproduce perceptron must learn to open
    its gate only when breeding is affordable (the honest same-tick net reward
    lands the birth outlay on the parent), and the world's conserved energy is
    the ultimate carrying capacity — a birth is a transfer, not creation.

    Args:
        params: base world rules; the benchmark forces ``generate_world=True``,
            ``world_seed=seed``, ``structure_enabled=True``,
            ``learning_enabled=True`` and a generous ``max_entities``.
        population_size: genomes per generation.
        generations: evolution cycles.
        ticks_per_generation: simulation ticks per generation.
        organic_pools: food pools distributed across the group territories.
        group_count: number of tribes (territories).
        group_weight: tribe-mean share of group-arm selection fitness.
        hidden_units: hidden projection width for the babies' brains.
        seed: world + RNG seed (identical for both arms).

    Returns:
        Dict with ``control`` and ``lifecycle`` runs, their last-generation
        ``avg_fitness``, the lifecycle arm's ``births``, total
        ``birth_energy_moved``, ``deaths``, final ``alive_count`` and
        ``population_size``, and the ``lifecycle_emerged`` verdict (the
        lifecycle arm bred inside its own ticks — births outnumber zero —
        i.e. the channel demonstrably fired).
    """
    base = params or WorldParams(grid_size=(16, 8, 16))
    base = replace(base, generate_world=True, world_seed=seed,
                   learning_enabled=True, brain_hidden_units=int(hidden_units),
                   teaching_enabled=False, memory_enabled=False,
                   structure_enabled=True, write_energy_scale=10.0,
                   lifecycle_enabled=False,
                   max_entities=max(int(population_size) * 4, 16))

    shared = dict(population_size=population_size, generations=generations,
                  ticks_per_generation=ticks_per_generation,
                  organic_pools=organic_pools, group_count=group_count,
                  group_weight=group_weight, seed=seed)
    control = EvolutionEngine(params=base, **shared).run()
    lifecycle = EvolutionEngine(
        params=replace(base, lifecycle_enabled=True), **shared).run()

    ctrl_avg = control["history"][-1]["avg_fitness"]
    life_avg = lifecycle["history"][-1]["avg_fitness"]
    last = lifecycle["history"][-1]
    births = int(last["births"])
    return {
        "control": control,
        "lifecycle": lifecycle,
        "group_count": group_count,
        "group_weight": group_weight,
        "control_last_avg": float(ctrl_avg),
        "lifecycle_last_avg": float(life_avg),
        "births": births,
        "birth_energy_moved": float(last["birth_energy_moved"]),
        "deaths": int(last["deaths"]),
        "alive_count": int(last["alive_count"]),
        "population_size": population_size,
        "lifecycle_emerged": bool(births > 0),
    }


def _conservation_sweep(params: WorldParams, genomes: list[Genome],
                        ticks: int) -> dict:
    """
    Live physics tripwire: grid + entity + nest energy must never increase.

    Builds one fully-loaded generation (every opt-in channel on) and steps
    the real tick loop one tick at a time, recomputing the world total from
    the LIVE scene after each tick — the sum of all grid cell energy, every
    baby's energy, and every nest bank. Each channel is individually
    transfer-safe; an increase here flags a channel that created energy when
    combined with the others. The food pools use the exact same deterministic
    placement the engine uses, so the sweep is reproducible on ``world_seed``.

    When ``params.solar_enabled`` the sky is a legit BOUNDARY source: each
    tick's deposit (tracked by ``scene.solar_energy_deposited``) is subtracted
    before the comparison, so the invariant becomes "internal channels never
    create energy" — the exact Stage 13 claim. With the sun off the deposit is
    always zero and the sweep is the strict closed-world tripwire.

    Args:
        params: world rules (the caller passes the ALL-ON ruleset).
        genomes: the generation's genomes, applied to the spawned babies.
        ticks: simulation ticks to sweep.

    Returns:
        Dict with ``monotonic`` (bool), ``violations`` (list of
        ``(tick, prev_total, new_total)`` tuples), ``start_total`` /
        ``end_total`` floats, and ``boundary_deposit_total`` (energy the sky
        added over the sweep).
    """
    if params.world_seed is not None:
        np.random.seed(int(params.world_seed))
    scene = SimScene(params=params)
    food_rng = np.random.default_rng(int(params.world_seed))
    engine = EvolutionEngine(params=params, population_size=len(genomes),
                             generations=1, ticks_per_generation=ticks,
                             seed=int(params.world_seed))
    engine._place_food(scene, food_rng)
    for g in genomes:
        b = SimBaby(position=None, initial_energy=params.start_energy,
                    params=params, group_id=g.group_id)
        g.apply_to(b)
        scene.add_baby(b)
    sim = Simulation(scene, max_ticks=ticks)

    def total() -> float:
        return (float(np.sum(scene.world.energy))
                + float(sum(x.energy for x in scene.babies))
                + float(sum(n.stored_energy for n in scene.nests)))

    start_total = total()
    prev = start_total
    prev_solar = scene.solar_energy_deposited
    violations: list[tuple[int, float, float]] = []
    for t in range(1, ticks + 1):
        sim.step()
        cur = total()
        boundary = scene.solar_energy_deposited - prev_solar
        prev_solar = scene.solar_energy_deposited
        if cur > prev + boundary + 1e-6:
            violations.append((t, prev, cur))
        prev = cur
    return {
        "monotonic": len(violations) == 0,
        "violations": violations,
        "start_total": float(start_total),
        "end_total": float(prev),
        "boundary_deposit_total": float(scene.solar_energy_deposited),
    }


def benchmark_civilization(params: WorldParams | None = None, *,
                           population_size: int = 8,
                           generations: int = 12,
                           ticks_per_generation: int = 24,
                           organic_pools: int = 3,
                           group_count: int = 2,
                           group_weight: float = 0.5,
                           hidden_units: int = 0,
                           seed: int = 1) -> dict:
    """
    Integrated world proof (Stage 12): every channel in one living world.

    Each opt-in channel — structures, teaching, memory, messages, predation,
    territoriality, lifecycle, specialization — is proven in its own
    benchmark. This benchmark turns them ALL on at once in a single evolution
    run and checks four invariants that only hold when the channels are
    genuinely composable:

      1. CONSERVATION under full load: over a live, fully-loaded generation
         the world total (grid + entity + nest energy) never increases. Each
         channel is individually transfer-safe; together they must still
         never create energy. ``conservation_monotonic``.
      2. RNG ISOLATION under total load: the four behavior brains (cells,
         body, entity, move) drawn with the same seed are bit-identical
         whether every dedicated-stream channel is off or ALL on. The locked
         selection proofs keep their exact genome layout and energy flow no
         matter how many channels coexist. ``brains_identical``.
      3. CHANNEL LIVENESS: every opt-in channel demonstrably fires somewhere
         in the run — lessons (teaching), predations, defenses AND raids
         (territoriality), nests_built (structures), births (lifecycle),
         role deposits/raids (specialization), and a growing world reservoir
         (memory). ``channels_live``.
      4. SUSTAINABILITY: births > 0 with survivors at the final generation —
         the world grows its own population while predation and contest are
         taking lives. ``civilization_emerged``.

    Two arms evolve on the SAME generated world with the same initial core
    genome draws:
      - ``control``: the bare world — every opt-in channel off, learning off.
        This is the classic emergence baseline; it produces zero channel
        activity by construction (the negative control).
      - ``civilization``: ALL opt-in channels on, in-life learning on, and a
        generous ``max_entities`` so births have room to land.

    Nothing is hardcoded: the channel brains are still evolved material, and
    the conservation sweep is the honest physics tripwire over the live tick
    loop.

    Args:
        params: base world rules; the benchmark forces ``generate_world=True``
            and ``world_seed=seed`` on both arms and every opt-in channel plus
            ``learning_enabled=True`` on the civilization arm.
        population_size: genomes per generation.
        generations: evolution cycles.
        ticks_per_generation: simulation ticks per generation.
        organic_pools: food pools distributed across the group territories.
        group_count: number of tribes (territories).
        group_weight: tribe-mean share of group-arm selection fitness.
        hidden_units: hidden projection width for the babies' brains.
        seed: world + RNG seed (identical for both arms).

    Returns:
        Dict with ``control`` and ``civilization`` runs, their last-generation
        ``avg_fitness``, the conservation verdict (``conservation_monotonic``,
        ``conservation_violations``, ``conservation_start_total``,
        ``conservation_end_total``), the RNG-isolation verdict
        (``brains_identical``), per-channel liveness (``channels_live`` and
        ``channels_live_all``), ``births``, final ``alive_count``, and the
        ``civilization_emerged`` verdict.
    """
    base = params or WorldParams(grid_size=(16, 8, 16))
    bare = replace(base, generate_world=True, world_seed=seed,
                   learning_enabled=False, brain_hidden_units=int(hidden_units),
                   message_enabled=False, structure_enabled=False,
                   teaching_enabled=False, memory_enabled=False,
                   predation_enabled=False, territoriality_enabled=False,
                   lifecycle_enabled=False, specialization_enabled=False)
    civil = replace(base, generate_world=True, world_seed=seed,
                    learning_enabled=True, brain_hidden_units=int(hidden_units),
                    message_enabled=True, structure_enabled=True,
                    teaching_enabled=True, memory_enabled=True,
                    predation_enabled=True, territoriality_enabled=True,
                    lifecycle_enabled=True, specialization_enabled=True,
                    write_energy_scale=10.0,
                    max_entities=max(int(population_size) * 4, 16))

    shared = dict(population_size=population_size, generations=generations,
                  ticks_per_generation=ticks_per_generation,
                  organic_pools=organic_pools, group_count=group_count,
                  group_weight=group_weight, seed=seed)
    control = EvolutionEngine(params=bare, **shared).run()
    civilization = EvolutionEngine(params=civil, **shared).run()

    # RNG isolation: same-seed genome draws, bare vs ALL-ON. The dedicated
    # channel streams must not perturb the four behavior brains even when
    # every channel is enabled at once.
    g_off = Genome.random(bare, np.random.default_rng(seed), group_id=0)
    g_on = Genome.random(civil, np.random.default_rng(seed), group_id=0)
    brains_identical = all(
        np.allclose(g_off.tensors[f"{name}.{suf}"],
                    g_on.tensors[f"{name}.{suf}"])
        for name in ("cells", "body", "entity", "move")
        for suf in ("W", "b")
    )

    # Conservation sweep: one fully-loaded generation, live tick loop.
    genomes = [Genome.random(civil, np.random.default_rng(seed),
                             group_id=i % group_count)
               for i in range(population_size)]
    sweep = _conservation_sweep(civil, genomes, ticks_per_generation)

    # Channel liveness across the whole civilization run.
    fired = {
        k: any(h.get(k, 0) > 0 for h in civilization["history"])
        for k in ("lessons", "predations", "defenses", "raids",
                  "nests_built", "births", "role_deposits", "role_raids")
    }
    fired["memory"] = any(h.get("memory_size", 0) > 0
                          for h in civilization["history"])

    last = civilization["history"][-1]
    births = int(last["births"])
    all_live = all(fired.values())
    emerged = bool(sweep["monotonic"] and brains_identical and all_live
                   and births > 0)
    return {
        "control": control,
        "civilization": civilization,
        "group_count": group_count,
        "group_weight": group_weight,
        "control_last_avg": float(control["history"][-1]["avg_fitness"]),
        "civilization_last_avg": float(last["avg_fitness"]),
        "conservation_monotonic": bool(sweep["monotonic"]),
        "conservation_violations": list(sweep["violations"]),
        "conservation_start_total": float(sweep["start_total"]),
        "conservation_end_total": float(sweep["end_total"]),
        "brains_identical": bool(brains_identical),
        "channels_live": dict(fired),
        "channels_live_all": bool(all_live),
        "births": births,
        "alive_count": int(last["alive_count"]),
        "population_size": population_size,
        "civilization_emerged": bool(emerged),
    }


def benchmark_solar(params: WorldParams | None = None, *,
                    population_size: int = 8,
                    generations: int = 12,
                    ticks_per_generation: int = 48,
                    organic_pools: int = 3,
                    solar_deposit_rate: float = 0.1,
                    hidden_units: int = 0,
                    seed: int = 7) -> dict:
    """
    Diurnal energy cycle proof (Stage 13): the world's first external source.

    The sun is the boundary source that breaks the world's closed system —
    and Stage 13's whole claim is that the conservation invariant survives
    the break. Two arms evolve on the SAME generated world with the same
    initial core genome draws, in-life learning on in both, every other
    opt-in channel off:

      - ``control``: solar off. The world is a closed system and every tick
        drains energy — the classic heat-death that the earlier stages
        accepted as the cost of honest physics.
      - ``solar``: the sun rises and sets once per generation
        (``solar_day_ticks == ticks_per_generation``). At day the sky
        deposits energy onto the topmost exposed surface cell of every
        column, scaled by a half-wave diurnal curve; at night it deposits
        nothing. The surface is fed, diffusion carries the gift down, and
        the population does not starve.

    Four invariants must hold:

      1. CONSERVATION under a boundary source: over a live solar generation
         the world total never increases beyond the tick's sky deposit.
         ``solar_conservation_exact`` — the sweep subtracts
         ``scene.solar_energy_deposited`` per tick, so any internal channel
         that created energy still trips the wire.
      2. CLOSED-WORLD TRIPWIRE: the same sweep on the solar-OFF ruleset is
         strictly monotonic — the sun is the only way energy can appear.
         ``closed_monotonic``.
      3. RNG ISOLATION: same-seed genome draws produce bit-identical four
         behavior brains whether the sky is on or off (dims match, so the
         dedicated solar stream never perturbs the core draw).
         ``brains_identical``.
      4. DAYLIGHT LIVENESS: the solar arm's history shows sunshine and a
         positive boundary deposit, and the sun keeps the world alive:
         ``solar_last_avg > control_last_avg`` (``solar_emerged``).

    Args:
        params: base world rules; the benchmark forces ``generate_world=True``,
            ``world_seed=seed``, ``learning_enabled=True`` and every opt-in
            channel off (structures, teaching, memory, messages, predation,
            territoriality, lifecycle, specialization).
        population_size: genomes per generation.
        generations: evolution cycles.
        ticks_per_generation: simulation ticks per generation AND the length
            of one full day/night cycle in the solar arm.
        organic_pools: food pools distributed across the world.
        solar_deposit_rate: energy per lit surface cell per tick at noon.
        hidden_units: hidden projection width for the babies' brains.
        seed: world + RNG seed (identical for both arms).

    Returns:
        Dict with ``control`` and ``solar`` runs, their last-generation
        ``avg_fitness``, the conservation verdicts (``solar_conservation_exact``,
        ``closed_monotonic``, violations and totals), the RNG-isolation
        verdict ``brains_identical``, daylight stats (``deposited``,
        ``sunshine``, ``light_final``), and the ``solar_emerged`` verdict.
    """
    base = params or WorldParams(grid_size=(16, 8, 16))
    closed = replace(base, generate_world=True, world_seed=seed,
                     learning_enabled=True, brain_hidden_units=int(hidden_units),
                     message_enabled=False, structure_enabled=False,
                     teaching_enabled=False, memory_enabled=False,
                     predation_enabled=False, territoriality_enabled=False,
                     lifecycle_enabled=False, specialization_enabled=False,
                     solar_enabled=False)
    solar = replace(base, generate_world=True, world_seed=seed,
                    learning_enabled=True, brain_hidden_units=int(hidden_units),
                    message_enabled=False, structure_enabled=False,
                    teaching_enabled=False, memory_enabled=False,
                    predation_enabled=False, territoriality_enabled=False,
                    lifecycle_enabled=False, specialization_enabled=False,
                    solar_enabled=True,
                    solar_day_ticks=int(ticks_per_generation),
                    solar_phase=0, solar_min_intensity=0.0,
                    solar_max_intensity=1.0, solar_deposit_rate=float(solar_deposit_rate))

    shared = dict(population_size=population_size, generations=generations,
                  ticks_per_generation=ticks_per_generation,
                  organic_pools=organic_pools, seed=seed)
    control = EvolutionEngine(params=closed, **shared).run()
    day = EvolutionEngine(params=solar, **shared).run()

    ctrl_avg = control["history"][-1]["avg_fitness"]
    solar_avg = day["history"][-1]["avg_fitness"]

    # RNG isolation: same-seed genome draws, sky on vs off (same dims).
    g_off = Genome.random(closed, np.random.default_rng(seed), group_id=0)
    g_on = Genome.random(solar, np.random.default_rng(seed), group_id=0)
    brains_identical = all(
        np.allclose(g_off.tensors[f"{name}.{suf}"],
                    g_on.tensors[f"{name}.{suf}"])
        for name in ("cells", "body", "entity", "move")
        for suf in ("W", "b")
    )

    # Conservation under the boundary source: solar-aware sweep.
    genomes = [Genome.random(solar, np.random.default_rng(seed), group_id=0)
               for _ in range(population_size)]
    solar_sweep = _conservation_sweep(solar, genomes, ticks_per_generation)
    # Closed-world tripwire: strict monotonic with the sky off.
    closed_genomes = [Genome.random(closed, np.random.default_rng(seed), group_id=0)
                      for _ in range(population_size)]
    closed_sweep = _conservation_sweep(closed, closed_genomes,
                                       ticks_per_generation)

    last = day["history"][-1]
    deposited = float(last["solar_energy_deposited"])
    sunshine = float(last["sunshine"])
    light_final = float(last["light_final"])

    emerged = bool(solar_sweep["monotonic"] and closed_sweep["monotonic"]
                   and brains_identical and deposited > 0.0
                   and sunshine > 0.0 and solar_avg > ctrl_avg)
    return {
        "control": control,
        "solar": day,
        "control_last_avg": float(ctrl_avg),
        "solar_last_avg": float(solar_avg),
        "solar_conservation_exact": bool(solar_sweep["monotonic"]),
        "solar_violations": list(solar_sweep["violations"]),
        "solar_start_total": float(solar_sweep["start_total"]),
        "solar_end_total": float(solar_sweep["end_total"]),
        "solar_boundary_deposit": float(solar_sweep["boundary_deposit_total"]),
        "closed_monotonic": bool(closed_sweep["monotonic"]),
        "closed_violations": list(closed_sweep["violations"]),
        "closed_start_total": float(closed_sweep["start_total"]),
        "closed_end_total": float(closed_sweep["end_total"]),
        "brains_identical": bool(brains_identical),
        "deposited": deposited,
        "sunshine": sunshine,
        "light_final": light_final,
        "population_size": population_size,
        "solar_emerged": bool(emerged),
    }


def benchmark_seasons(params: WorldParams | None = None, *,
                      population_size: int = 6,
                      generations: int = 3,
                      ticks_per_generation: int = 24,
                      organic_pools: int = 2,
                      solar_deposit_rate: float = 0.1,
                      seasonality: float = 1.0,
                      seasons_per_year: int = 4,
                      hidden_units: int = 0,
                      seed: int = 7) -> dict:
    """
    Seasonal year envelope proof (Stage 14): the diurnal cycle rides a year.

    The Stage 13 sun follows a fixed half-wave diurnal curve every day. Stage
    14 wraps that curve inside a slower cosine year — ``solar_season_ticks``
    ticks long — so a summer noon outshines a winter noon while the sky still
    deposits ONLY onto exposed surface cells. Two arms evolve on the SAME
    generated world with the same initial core genome draws, in-life learning
    on in both, every other opt-in channel off:

      - ``control``: the sun runs the plain Stage 13 diurnal cycle
        (``solar_season_ticks == 0``) — the exact pre-season world.
      - ``seasonal``: the same sun rides the year envelope
        (``solar_season_ticks == seasons_per_year * ticks_per_generation``).

    The invariants must hold:

      1. CONSERVATION under the seasonal boundary: over a live seasonal
         generation the world total never increases beyond the tick's sky
         deposit. ``seasonal_conservation_exact`` — the sweep subtracts
         ``scene.solar_energy_deposited`` per tick.
      2. CLOSED-WORLD TRIPWIRE: the same sweep on a solar-OFF ruleset is
         strictly monotonic. ``closed_monotonic``.
      3. RNG ISOLATION: same-seed genome draws are bit-identical whether the
         year is on or off — the envelope consumes no RNG.
         ``brains_identical``.
      4. SEASON LIVENESS: a summer noon is brighter than a winter noon
         (``summer_noon > winter_noon``), the seasonal arm sees sunshine and a
         positive boundary deposit, and conservation survived the boundary.

    Args:
        params: base world rules; the benchmark forces ``generate_world=True``,
            ``world_seed=seed``, ``learning_enabled=True`` and every opt-in
            channel off.
        population_size: genomes per generation.
        generations: evolution cycles.
        ticks_per_generation: simulation ticks per generation AND one full
            day/night cycle length in both arms.
        organic_pools: food pools distributed across the world.
        solar_deposit_rate: energy per lit surface cell per tick at noon.
        seasonality: 0 = flat diurnal mean, 1 = full seasonal swing.
        seasons_per_year: how many diurnal cycles fit in one year.
        hidden_units: hidden projection width for the babies' brains.
        seed: world + RNG seed (identical for both arms).

    Returns:
        Dict with ``control`` and ``seasonal`` runs, their last-generation
        ``avg_fitness``, the conservation verdicts
        (``seasonal_conservation_exact``, ``closed_monotonic``, violations and
        totals), the RNG-isolation verdict ``brains_identical``, the noon
        comparison (``summer_noon``, ``winter_noon``), daylight stats
        (``deposited``, ``sunshine``) and the ``seasons_emerged`` verdict.
    """
    base = params or WorldParams(grid_size=(16, 8, 16))
    closed = replace(base, generate_world=True, world_seed=seed,
                     learning_enabled=True, brain_hidden_units=int(hidden_units),
                     message_enabled=False, structure_enabled=False,
                     teaching_enabled=False, memory_enabled=False,
                     predation_enabled=False, territoriality_enabled=False,
                     lifecycle_enabled=False, specialization_enabled=False,
                     solar_enabled=False)
    day_ticks = max(int(ticks_per_generation), 1)
    control = replace(closed, solar_enabled=True,
                      solar_day_ticks=day_ticks,
                      solar_phase=0, solar_min_intensity=0.0,
                      solar_max_intensity=1.0,
                      solar_deposit_rate=float(solar_deposit_rate),
                      solar_season_ticks=0, solar_seasonality=1.0)
    season_ticks = max(int(seasons_per_year), 1) * day_ticks
    seasonal = replace(control, solar_season_ticks=season_ticks,
                       solar_seasonality=float(seasonality))

    shared = dict(population_size=population_size, generations=generations,
                  ticks_per_generation=ticks_per_generation,
                  organic_pools=organic_pools, seed=seed)
    ctrl_run = EvolutionEngine(params=control, **shared).run()
    seasonal_run = EvolutionEngine(params=seasonal, **shared).run()

    ctrl_avg = ctrl_run["history"][-1]["avg_fitness"]
    seasonal_avg = seasonal_run["history"][-1]["avg_fitness"]

    # RNG isolation: same-seed genome draws, year on vs off (same dims).
    g_off = Genome.random(control, np.random.default_rng(seed), group_id=0)
    g_on = Genome.random(seasonal, np.random.default_rng(seed), group_id=0)
    brains_identical = all(
        np.allclose(g_off.tensors[f"{name}.{suf}"],
                    g_on.tensors[f"{name}.{suf}"])
        for name in ("cells", "body", "entity", "move")
        for suf in ("W", "b")
    )

    # Conservation under the seasonal boundary: solar-aware sweep.
    genomes = [Genome.random(seasonal, np.random.default_rng(seed), group_id=0)
               for _ in range(population_size)]
    seasonal_sweep = _conservation_sweep(seasonal, genomes, day_ticks)
    # Closed-world tripwire: strict monotonic with the sky off.
    closed_genomes = [Genome.random(closed, np.random.default_rng(seed), group_id=0)
                      for _ in range(population_size)]
    closed_sweep = _conservation_sweep(closed, closed_genomes, day_ticks)

    def _noon(params: WorldParams, tick: int) -> float:
        scene = SimScene(params=params)
        scene._tick = tick
        scene.apply_solar()
        return float(scene.world.light)

    noon = max(day_ticks // 4, 1)
    summer_noon = _noon(seasonal, noon)
    winter_noon = _noon(seasonal, season_ticks // 2 + noon)

    last = seasonal_run["history"][-1]
    deposited = float(last["solar_energy_deposited"])
    sunshine = float(last["sunshine"])

    emerged = bool(seasonal_sweep["monotonic"] and closed_sweep["monotonic"]
                   and brains_identical and deposited > 0.0
                   and sunshine > 0.0 and summer_noon > winter_noon)
    return {
        "control": ctrl_run,
        "seasonal": seasonal_run,
        "control_last_avg": float(ctrl_avg),
        "seasonal_last_avg": float(seasonal_avg),
        "seasonal_conservation_exact": bool(seasonal_sweep["monotonic"]),
        "seasonal_violations": list(seasonal_sweep["violations"]),
        "seasonal_start_total": float(seasonal_sweep["start_total"]),
        "seasonal_end_total": float(seasonal_sweep["end_total"]),
        "seasonal_boundary_deposit": float(seasonal_sweep["boundary_deposit_total"]),
        "closed_monotonic": bool(closed_sweep["monotonic"]),
        "closed_violations": list(closed_sweep["violations"]),
        "closed_start_total": float(closed_sweep["start_total"]),
        "closed_end_total": float(closed_sweep["end_total"]),
        "brains_identical": bool(brains_identical),
        "summer_noon": summer_noon,
        "winter_noon": winter_noon,
        "deposited": deposited,
        "sunshine": sunshine,
        "population_size": population_size,
        "generations": generations,
        "seasons_per_year": int(seasons_per_year),
        "seasonality": float(seasonality),
        "seasons_emerged": emerged,
    }
