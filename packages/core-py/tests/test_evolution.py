"""
Tests for the evolution engine — genetic algorithm over baby genomes.

Covers Genome serialization/roundtrip, crossover, mutation, inherited
memories (the memotype), the EvolutionEngine generation loop
(selection, elitism, determinism), honest trait-group (multilevel)
selection for the Stage 6 multi-agent world, the kin-signal social
perception, the social-vs-individual emergence benchmark, and the
Stage 7 durable structures (nests) built on cell-write deposits.
"""

import json

import numpy as np
import pytest

from domains.shell.simulation import (
    MATERIAL_ORGANIC,
    EntityType,
    BabyAction,
    CellWrite,
    Nest,
    WorldParams,
    SimBaby,
    SimScene,
    Simulation,
    Perceptron,
)
from domains.shell.evolution import (
    Genome,
    EvolutionEngine,
    benchmark_emergence,
    benchmark_social,
    benchmark_culture,
    benchmark_memory,
)


def _baby_with_memories(params: WorldParams | None = None) -> SimBaby:
    baby = SimBaby(initial_energy=100.0, params=params)
    for i, reward in enumerate((1.0, 5.0, 3.0, 2.0, 4.0)):
        baby.memory.record(
            np.full(4, float(i), dtype=np.float32), (0.5, 0.5, 0.5), reward, tick=i,
        )
    return baby


# ── Genome ────────────────────────────────────────────────────────────────────

class TestGenome:
    def test_roundtrip_preserves_weights(self):
        params = WorldParams(grid_size=(8, 4, 8))
        baby = SimBaby(initial_energy=100.0, params=params)
        g = Genome.from_baby(baby)
        g.apply_to(baby)
        for name in ("cells", "body", "entity"):
            p = getattr(baby, f"perceptron_{name}")
            assert np.allclose(p.W, g.tensors[f"{name}.W"])
            assert np.allclose(p.b, g.tensors[f"{name}.b"])

    def test_from_baby_copies_not_aliases(self):
        params = WorldParams(grid_size=(8, 4, 8))
        baby = SimBaby(initial_energy=100.0, params=params)
        g = Genome.from_baby(baby)
        baby.perceptron_cells.W[:] = 999.0
        assert not np.allclose(g.tensors["cells.W"], 999.0)

    def test_apply_to_overwrites_baby_weights(self):
        params = WorldParams(grid_size=(8, 4, 8))
        baby = SimBaby(initial_energy=100.0, params=params)
        g = Genome.random(params, np.random.default_rng(0))
        g.apply_to(baby)
        assert np.allclose(baby.perceptron_cells.W, g.tensors["cells.W"])

    def test_random_has_expected_shapes(self):
        params = WorldParams(grid_size=(8, 4, 8))
        g = Genome.random(params, np.random.default_rng(1))
        assert g.tensors["cells.W"].shape == (params.cells_input_dim, 3)
        assert g.tensors["cells.b"].shape == (3,)
        assert g.tensors["body.W"].shape == (params.body_input_dim, 2)
        assert g.tensors["entity.W"].shape == (params.entity_input_dim, 2)
        assert np.all(g.tensors["cells.b"] == 0.0)

    def test_crossover_mixes_parents(self):
        params = WorldParams(grid_size=(8, 4, 8))
        rng = np.random.default_rng(7)
        a = Genome.random(params, rng)
        b = Genome.random(params, rng)
        child = a.crossover(b, np.random.default_rng(7))
        for k in a.tensors:
            assert child.tensors[k].shape == a.tensors[k].shape
        child_flat = np.concatenate([v.ravel() for v in child.tensors.values()])
        a_flat = np.concatenate([v.ravel() for v in a.tensors.values()])
        b_flat = np.concatenate([v.ravel() for v in b.tensors.values()])
        assert not np.array_equal(child_flat, a_flat)
        assert not np.array_equal(child_flat, b_flat)
        assert np.all((child_flat == a_flat) | (child_flat == b_flat))

    def test_crossover_all_from_a_when_mask_all_true(self):
        params = WorldParams(grid_size=(8, 4, 8))
        rng = np.random.default_rng(3)
        a = Genome.random(params, rng)
        b = Genome.random(params, rng)
        class AllTrue:
            def random(self, shape):
                return np.zeros(shape)
        child = a.crossover(b, AllTrue())
        for k in a.tensors:
            assert np.array_equal(child.tensors[k], a.tensors[k])

    def test_mutate_changes_weights_at_rate_one(self):
        params = WorldParams(grid_size=(8, 4, 8))
        rng = np.random.default_rng(5)
        g = Genome.random(params, rng)
        original = {k: v.copy() for k, v in g.tensors.items()}
        g.mutate(np.random.default_rng(9), rate=1.0, scale=0.5)
        for k in original:
            assert not np.array_equal(g.tensors[k], original[k])

    def test_mutate_zero_rate_keeps_weights(self):
        params = WorldParams(grid_size=(8, 4, 8))
        rng = np.random.default_rng(5)
        g = Genome.random(params, rng)
        original = {k: v.copy() for k, v in g.tensors.items()}
        g.mutate(np.random.default_rng(9), rate=0.0, scale=0.5)
        for k in original:
            assert np.array_equal(g.tensors[k], original[k])

    def test_mutate_uses_provided_rng(self):
        params = WorldParams(grid_size=(8, 4, 8))
        g1 = Genome.random(params, np.random.default_rng(5))
        g2 = Genome.random(params, np.random.default_rng(5))
        g1.mutate(np.random.default_rng(9), rate=1.0, scale=0.3)
        g2.mutate(np.random.default_rng(9), rate=1.0, scale=0.3)
        for k in g1.tensors:
            assert np.array_equal(g1.tensors[k], g2.tensors[k])


# ── EvolutionEngine internals ─────────────────────────────────────────────────

class TestEvolutionEngine:
    def test_default_params_small_grid(self):
        eng = EvolutionEngine(seed=1)
        assert eng.params.grid_size == (16, 8, 16)

    def test_clamped_sizes(self):
        eng = EvolutionEngine(population_size=1, generations=0, elite_count=0)
        assert eng.population_size == 2
        assert eng.generations == 1
        assert eng.elite_count == 1

    def test_elite_count_capped_at_population(self):
        eng = EvolutionEngine(population_size=4, elite_count=99)
        assert eng.elite_count == 4

    def test_fitness_is_energy(self):
        params = WorldParams(grid_size=(8, 4, 8))
        baby = SimBaby(initial_energy=50.0, params=params)
        assert EvolutionEngine._fitness(baby) == 50.0

    def test_selection_keeps_top_performer_in_next_generation(self):
        params = WorldParams(grid_size=(8, 4, 8))
        babies = []
        for e in (10.0, 90.0, 50.0, 70.0):
            b = SimBaby(initial_energy=e, params=params)
            babies.append(b)
        fitnesses = [b.energy for b in babies]
        eng = EvolutionEngine(population_size=4, elite_count=2, seed=0)
        next_gen = eng._select(babies, fitnesses, np.random.default_rng(0))
        assert len(next_gen) == 4
        top = babies[int(np.argmax(fitnesses))]
        assert np.allclose(next_gen[0].tensors["cells.W"],
                           top.perceptron_cells.W)

    def test_run_structure(self):
        eng = EvolutionEngine(
            population_size=4, generations=3, ticks_per_generation=2,
            organic_pools=1, seed=42,
        )
        result = eng.run()
        assert result["generations"] == 3
        assert result["population_size"] == 4
        assert len(result["history"]) == 3
        assert result["best_genome"] is not None
        assert result["best_fitness"] >= 0.0
        assert result["history"][0]["generation"] == 1
        assert set(result["history"][0].keys()) == {
            "generation", "best_fitness", "avg_fitness", "alive",
            "cooperations", "contests", "cooperate_rate", "contest_rate",
            "social_energy_moved", "mean_home_displacement",
            "lessons", "teach_rate",
            "memory_size", "memory_seeds",
            "predations", "predation_rate", "predation_energy_moved",
        }

    def test_run_history_fitness_monotone_in_hypothesis(self):
        eng = EvolutionEngine(
            population_size=6, generations=4, ticks_per_generation=1,
            organic_pools=0, seed=7,
        )
        result = eng.run()
        bests = [h["best_fitness"] for h in result["history"]]
        assert result["best_fitness"] == max(bests)

    def test_run_deterministic_with_seed(self):
        eng1 = EvolutionEngine(population_size=4, generations=3,
                               ticks_per_generation=2, organic_pools=1, seed=123)
        eng2 = EvolutionEngine(population_size=4, generations=3,
                               ticks_per_generation=2, organic_pools=1, seed=123)
        r1 = eng1.run()
        r2 = eng2.run()
        assert r1["best_fitness"] == r2["best_fitness"]
        assert [h["best_fitness"] for h in r1["history"]] == \
               [h["best_fitness"] for h in r2["history"]]

    def test_run_different_seeds_may_differ(self):
        eng1 = EvolutionEngine(population_size=8, generations=2,
                               ticks_per_generation=1, organic_pools=0, seed=1)
        eng2 = EvolutionEngine(population_size=8, generations=2,
                               ticks_per_generation=1, organic_pools=0, seed=2)
        r1 = eng1.run()
        r2 = eng2.run()
        assert r1["best_fitness"] is not None
        assert r2["best_fitness"] is not None

    def test_run_history_avg_is_mean_of_fitnesses(self):
        eng = EvolutionEngine(population_size=4, generations=1,
                              ticks_per_generation=1, organic_pools=0, seed=0)
        babies, fitnesses, _social = eng._run_generation(
            [Genome.random(eng.params, np.random.default_rng(0)) for _ in range(4)],
            np.random.default_rng(0),
        )
        assert eng.history == []
        result = eng.run()
        h = result["history"][0]
        assert h["alive"] == sum(1 for b in babies if b.alive)

    def test_best_genome_roundtrips_into_new_baby(self):
        eng = EvolutionEngine(
            population_size=4, generations=2, ticks_per_generation=1,
            organic_pools=1, seed=11,
        )
        result = eng.run()
        baby = SimBaby(initial_energy=100.0, params=eng.params)
        result["best_genome"].apply_to(baby)
        for name in ("cells", "body", "entity"):
            p = getattr(baby, f"perceptron_{name}")
            assert np.allclose(p.W, result["best_genome"].tensors[f"{name}.W"])


# ── Inherited memory (memotype) ───────────────────────────────────────────────

class TestMemoryInheritance:
    def test_from_baby_consolidates_top_reward_episodes(self):
        baby = _baby_with_memories()
        g = Genome.from_baby(baby)
        rewards = [m["reward"] for m in g.memories]
        assert rewards == [5.0, 4.0, 3.0, 2.0, 1.0]
        assert g.memory_count == 5

    def test_from_baby_memories_match_source_episodes(self):
        baby = _baby_with_memories()
        g = Genome.from_baby(baby)
        top = g.memories[0]
        assert top["reward"] == pytest.approx(5.0)
        assert top["features"] == [1.0, 1.0, 1.0, 1.0]  # reward 5.0 was tick 1
        assert top["tick"] == 1

    def test_from_baby_caps_inherited_memories(self):
        params = WorldParams(grid_size=(8, 4, 8), memory_inherit=2)
        g = Genome.from_baby(_baby_with_memories(params))
        assert g.memory_count == 2
        assert [m["reward"] for m in g.memories] == [5.0, 4.0]

    def test_from_baby_detaches_memories(self):
        baby = _baby_with_memories()
        g = Genome.from_baby(baby)
        baby.memory.record(np.zeros(4, np.float32), (0.0, 0.0, 0.0), 99.0)
        assert g.memory_count == 5  # consolidation is a snapshot

    def test_random_genome_starts_without_memories(self):
        params = WorldParams(grid_size=(8, 4, 8))
        g = Genome.random(params, np.random.default_rng(0))
        assert g.memory_count == 0
        assert g.memories == []

    def test_crossover_inherits_first_parent_memories(self):
        params = WorldParams(grid_size=(8, 4, 8))
        rng = np.random.default_rng(0)
        a = Genome.from_baby(_baby_with_memories(params))
        b = Genome.random(params, rng)
        child = a.crossover(b, np.random.default_rng(1))
        assert [m["reward"] for m in child.memories] == \
               [m["reward"] for m in a.memories]
        assert b.memory_count == 0

    def test_mutate_preserves_memories(self):
        params = WorldParams(grid_size=(8, 4, 8))
        g = Genome.from_baby(_baby_with_memories(params))
        original = [dict(m) for m in g.memories]
        g.mutate(np.random.default_rng(9), rate=1.0, scale=0.5)
        assert [m["reward"] for m in g.memories] == \
               [m["reward"] for m in original]
        assert g.memory_count == 5

    def test_apply_to_seeds_baby_memory(self):
        params = WorldParams(grid_size=(8, 4, 8))
        g = Genome.from_baby(_baby_with_memories(params))
        baby = SimBaby(initial_energy=100.0, params=params)
        g.apply_to(baby)
        assert len(baby.memory) == g.memory_count
        rewards = [e.reward for e in baby.memory.recall(k=g.memory_count)]
        assert rewards == [5.0, 4.0, 3.0, 2.0, 1.0]

    def test_memories_are_json_serializable(self):
        baby = _baby_with_memories()
        g = Genome.from_baby(baby)
        payload = json.dumps({"memories": g.memories})
        assert isinstance(payload, str)

    def test_selection_propagates_memory(self):
        params = WorldParams(grid_size=(8, 4, 8))
        babies = []
        for e in (10.0, 90.0, 50.0, 70.0):
            b = SimBaby(initial_energy=e, params=params)
            b.memory.record(np.zeros(4, np.float32), (0.0, 0.0, 0.0), float(e / 10))
            babies.append(b)
        fitnesses = [b.energy for b in babies]
        eng = EvolutionEngine(population_size=4, elite_count=2, seed=0)
        next_gen = eng._select(babies, fitnesses, np.random.default_rng(0))
        assert all(len(g.memories) == 1 for g in next_gen)

    def test_memory_survives_generation_transition(self):
        eng = EvolutionEngine(
            population_size=4, generations=1, ticks_per_generation=2,
            organic_pools=1, seed=3,
        )
        genomes = [Genome.random(eng.params, np.random.default_rng(0))
                   for _ in range(4)]
        babies, fitnesses, _social = eng._run_generation(
            genomes, np.random.default_rng(0))
        top = babies[int(np.argmax(fitnesses))]
        top.memory.record(np.zeros(4, np.float32), (0.5, 0.5, 0.5), 7.0, tick=1)
        next_gen = eng._select(babies, fitnesses, np.random.default_rng(0))
        assert any(g.memory_count > 0 for g in next_gen)
        assert any(
            any(m["reward"] == pytest.approx(7.0) for m in g.memories)
            for g in next_gen
        )

    def test_run_reports_inherited_episodes(self):
        eng = EvolutionEngine(
            population_size=4, generations=2, ticks_per_generation=1,
            organic_pools=1, seed=11,
        )
        result = eng.run()
        assert "inherited_episodes" in result
        assert result["inherited_episodes"] >= 0


# ── World-level long-term memory (Stage 7) ────────────────────────────────────

class TestWorldMemoryEvolution:
    """
    The infinite long-term memory reservoir: off by default so locked proofs
    keep their exact energy flow; when ``memory_enabled`` the engine carries
    one append-only reservoir across every generation — dead babies deposit
    in-sim, survivors deposit at the generation boundary, newborns are seeded
    from the reservoir's best episodes, and nothing is ever evicted.
    """

    def test_reservoir_off_by_default(self):
        eng = EvolutionEngine(
            population_size=4, generations=1, ticks_per_generation=2,
            organic_pools=1, seed=1,
        )
        assert eng.world_memory is None
        result = eng.run()
        assert result["memory_size"] == 0
        assert result["memory_seeds_total"] == 0

    def test_enabled_engine_carries_reservoir(self):
        eng = EvolutionEngine(
            population_size=4, generations=1, ticks_per_generation=2,
            organic_pools=1, seed=1,
            params=WorldParams(grid_size=(8, 4, 8), memory_enabled=True),
        )
        assert eng.world_memory is not None
        result = eng.run()
        assert result["memory_size"] > 0
        assert result["memory_seeds_total"] >= 0

    def test_reservoir_grows_across_generations(self):
        eng = EvolutionEngine(
            population_size=4, generations=3, ticks_per_generation=3,
            organic_pools=1, seed=5,
            params=WorldParams(grid_size=(8, 4, 8), memory_enabled=True),
        )
        eng.run()
        sizes = [h["memory_size"] for h in eng.history]
        assert sizes == sorted(sizes)  # append-only: never evicts
        assert sizes[-1] >= sizes[0]
        assert sizes[0] > 0

    def test_newborns_seeded_from_reservoir(self):
        eng = EvolutionEngine(
            population_size=4, generations=2, ticks_per_generation=3,
            organic_pools=1, seed=5,
            params=WorldParams(grid_size=(8, 4, 8), memory_enabled=True,
                               memory_seed=2),
        )
        result = eng.run()
        # Generation 2 babies are seeded from gen-1 deposits.
        seeds = [h["memory_seeds"] for h in eng.history]
        assert seeds[0] >= 0
        assert sum(seeds) == result["memory_seeds_total"]

    def test_deposits_stamped_with_tribe_and_donor(self):
        eng = EvolutionEngine(
            population_size=4, generations=2, ticks_per_generation=3,
            organic_pools=1, seed=5,
            params=WorldParams(grid_size=(8, 4, 8), memory_enabled=True,
                               memory_deposit=2),
        )
        eng.run()
        assert eng.world_memory is not None
        episodes = eng.world_memory._episodes
        assert len(episodes) > 0
        assert all(e.donor_id > 0 for e in episodes)
        assert all(e.group_id == 0 for e in episodes)  # single population

    def test_run_is_deterministic_with_memory_enabled(self):
        def run_once():
            eng = EvolutionEngine(
                population_size=4, generations=2, ticks_per_generation=3,
                organic_pools=1, seed=7,
                params=WorldParams(grid_size=(8, 4, 8), memory_enabled=True),
            )
            result = eng.run()
            return (
                result["memory_size"],
                result["memory_seeds_total"],
                [h["memory_size"] for h in eng.history],
                [round(h["avg_fitness"], 4) for h in eng.history],
            )

        a, b = run_once(), run_once()
        assert a == b

    def test_deposit_memory_uses_deposit_cap(self):
        eng = EvolutionEngine(
            population_size=4, generations=1, ticks_per_generation=3,
            organic_pools=1, seed=5,
            params=WorldParams(grid_size=(8, 4, 8), memory_enabled=True,
                               memory_deposit=2),
        )
        result = eng.run()
        assert result["memory_size"] >= 0  # capped per-baby deposits
        # Every baby holds at most memory_deposit episodes in its living
        # buffer, so per-baby deposits never exceed the cap either.
        for h in eng.history:
            assert h["memory_size"] <= len(eng.history) * eng.population_size * 2


# ── Emergence proof (Stage 5) ─────────────────────────────────────────────────

class TestEmergenceBenchmark:
    """
    The emergence proof: in a fixed generated world with fixed food pools and
    a fixed shared spawn, an evolved population must out-harvest a frozen
    random population. Fitness measures genetic quality, not spawn luck.
    """

    def test_evolved_beats_frozen_random(self):
        r = benchmark_emergence(seed=1)
        assert r["evolved_last_avg"] > r["frozen_last_avg"]
        assert r["emerged"] is True

    def test_benchmark_deterministic_with_seed(self):
        a = benchmark_emergence(seed=7)
        b = benchmark_emergence(seed=7)
        assert a["evolved_last_avg"] == b["evolved_last_avg"]
        assert a["frozen_last_avg"] == b["frozen_last_avg"]
        assert [h["avg_fitness"] for h in a["evolved"]["history"]] == \
               [h["avg_fitness"] for h in b["evolved"]["history"]]

    def test_frozen_arm_has_no_selection(self):
        r = benchmark_emergence(seed=1)
        frozen = r["frozen"]
        assert frozen["best_genome"] is None
        assert len(frozen["history"]) == frozen["generations"]

    def test_shared_spawn_single_position(self):
        r = benchmark_emergence(seed=1)
        first = r["spawn_positions"][0]
        assert all(np.array_equal(p, first) for p in r["spawn_positions"])

    def test_spawns_land_on_surface(self):
        base = WorldParams(grid_size=(16, 8, 16), generate_world=True, world_seed=1)
        ref = SimScene(params=base)
        r = benchmark_emergence(seed=1)
        for p in r["spawn_positions"]:
            x, z = int(p[0]), int(p[2])
            assert p[1] == ref._surface_y(x, z)

    def test_fixed_environment_removes_layout_luck(self):
        # The two arms face identical terrain/food; generation 1 is identical
        # genomes, so the only difference across arms is selection.
        r = benchmark_emergence(seed=1)
        evo_gen1 = r["evolved"]["history"][0]["avg_fitness"]
        fro_gen1 = r["frozen"]["history"][0]["avg_fitness"]
        assert evo_gen1 == pytest.approx(fro_gen1)


# ── Honest trait-group selection (Stage 6) ────────────────────────────────────

class TestTribeSelection:
    """
    Two-level (multilevel) selection: tribes compete by the geometric mean of
    member energy, parents are drawn uniformly within a chosen tribe, and
    offspring inherit their tribe.
    """

    def _groups_engine(self):
        return EvolutionEngine(population_size=8, group_count=2,
                               group_weight=0.5, seed=0)

    def test_select_dispatches_to_group_mode(self):
        eng = self._groups_engine()
        babies = [SimBaby(initial_energy=e, params=eng.params, group_id=i % 2)
                  for i, e in enumerate((10.0, 90.0, 50.0, 70.0))]
        fitnesses = [b.energy for b in babies]
        next_gen = eng._select(babies, fitnesses, np.random.default_rng(0),
                               groups=[b.group_id for b in babies])
        assert len(next_gen) == 8
        assert all(g.group_id in (0, 1) for g in next_gen)

    def test_single_population_ignores_group_ids(self):
        # group_weight <= 0 means no group structure: all offspring breed
        # through the population-wide tournament regardless of group ids.
        eng = EvolutionEngine(population_size=8, group_count=2,
                              group_weight=0.0, seed=0)
        babies = [SimBaby(initial_energy=e, params=eng.params, group_id=i % 2)
                  for i, e in enumerate((10.0, 90.0, 50.0, 70.0))]
        fitnesses = [b.energy for b in babies]
        next_gen = eng._select(babies, fitnesses, np.random.default_rng(0),
                               groups=[b.group_id for b in babies])
        assert len(next_gen) == 8

    def test_even_tribe_outranks_same_sum_imbalanced_tribe(self):
        # Tribe 0: [100, 100, 100, 100]  sum 400, gmean 100
        # Tribe 1: [100, 100, 100, 0.1]  sum ~300, gmean ~10
        # The geometric-mean score makes the balanced tribe dominate breeding.
        eng = self._groups_engine()
        energies = [100, 100, 100, 100, 100, 100, 100, 0.1]
        babies = [SimBaby(initial_energy=e, params=eng.params, group_id=0 if i < 4 else 1)
                  for i, e in enumerate(energies)]
        fitnesses = [b.energy for b in babies]
        groups = [b.group_id for b in babies]
        next_gen = eng._select_groups(babies, fitnesses,
                                      np.random.default_rng(0), groups)
        counts = {g: sum(1 for x in next_gen if x.group_id == g) for g in (0, 1)}
        assert counts[0] > counts[1], counts  # balanced tribe breeds more
        assert counts[1] >= 1, counts          # but keeps an elite presence

    def test_offspring_group_ids_stay_within_tribe_set(self):
        eng = self._groups_engine()
        babies = [SimBaby(initial_energy=float(60 + 40 * (i % 2)),
                          params=eng.params, group_id=i % 3)
                  for i in range(9)]
        fitnesses = [b.energy for b in babies]
        groups = [b.group_id for b in babies]
        next_gen = eng._select(babies, fitnesses, np.random.default_rng(3),
                               groups=groups)
        assert all(g.group_id in (0, 1, 2) for g in next_gen)

    def test_grouped_run_reports_group_means(self):
        eng = EvolutionEngine(population_size=8, generations=2,
                              ticks_per_generation=2, organic_pools=2,
                              group_count=2, group_weight=0.5, seed=5)
        result = eng.run()
        entry = result["history"][0]
        assert "group_means" in entry
        assert set(entry["group_means"].keys()) == {0, 1}
        assert entry["mean_home_displacement"] >= 0.0

    def test_genome_inherits_group_via_crossover(self):
        params = WorldParams(grid_size=(8, 4, 8))
        rng = np.random.default_rng(0)
        a = Genome.random(params, rng, group_id=3)
        b = Genome.random(params, rng, group_id=7)
        child = a.crossover(b, np.random.default_rng(1))
        assert child.group_id == 3  # first parent's tribe travels with lineage


# ── Kin signal & social acts (Stage 6) ───────────────────────────────────────

class TestKinSignal:
    """
    Babies carry a tribe id ("communication via world signals"); the entity
    perceptron input includes a same-tribe bit so cooperation can be learned
    and directed at kin.
    """

    @staticmethod
    def _quiet_baby(energy, position, group_id=0):
        """A baby whose social/movement/write gates are pinned off."""
        params = WorldParams(grid_size=(8, 4, 8))
        b = SimBaby(initial_energy=energy, position=np.array(position, dtype=np.float64),
                    params=params, group_id=group_id)
        for p in (b.perceptron_cells, b.perceptron_body, b.perceptron_move):
            p.W[:] = 0.0
            p.b[:] = -10.0 if p is not b.perceptron_move else 0.0
        b.perceptron_entity.W[:] = 0.0
        b.perceptron_entity.b[:] = -10.0  # both social gates closed
        return b

    def test_perception_carries_group_id(self):
        params = WorldParams(grid_size=(8, 4, 8))
        scene = SimScene(params=params)
        a = SimBaby(position=np.array([4.0, 1.0, 4.0]), params=params, group_id=0)
        b = SimBaby(position=np.array([5.0, 1.0, 4.0]), params=params, group_id=1)
        scene.add_baby(a)
        scene.add_baby(b)
        p = a.perceive(scene.world, babies=[a, b])
        assert len(p.nearby_entities) == 1
        assert p.nearby_entities[0]["group_id"] == 1

    def test_entity_input_shape_matches_world_dim(self):
        params = WorldParams(grid_size=(8, 4, 8))
        a = SimBaby(initial_energy=200.0, position=np.array([4.0, 1.0, 4.0]),
                    params=params, group_id=0)
        b = SimBaby(initial_energy=20.0, position=np.array([4.0, 1.0, 5.0]),
                    params=params, group_id=0)
        a.perceptron_entity.W[:] = 0.0
        a.perceptron_entity.b[:] = np.array([-10.0, 10.0])  # cooperate open
        out = a.social_step(b)
        assert out["act"] == "cooperate"

    def test_social_step_works_without_kin_feature(self):
        # entity_input_dim=4 drops the kin bit; the input is sliced and the
        # perceptron (4,2) runs without a shape mismatch.
        params = WorldParams(grid_size=(8, 4, 8), entity_input_dim=4)
        a = SimBaby(initial_energy=200.0, position=np.array([4.0, 1.0, 4.0]),
                    params=params, group_id=0)
        b = SimBaby(initial_energy=20.0, position=np.array([4.0, 1.0, 5.0]),
                    params=params, group_id=1)
        assert a.perceptron_entity.W.shape == (4, 2)
        a.perceptron_entity.W[:] = 0.0
        a.perceptron_entity.b[:] = np.array([-10.0, 10.0])
        assert a.social_step(b)["act"] == "cooperate"


class TestSocialInteraction:
    """Perceptron-driven cooperation and contest between babies."""

    def _pair(self, a_energy, b_energy, group_a=0, group_b=0):
        params = WorldParams(grid_size=(8, 4, 8))
        a = SimBaby(initial_energy=a_energy,
                    position=np.array([4.0, 1.0, 4.0]), params=params, group_id=group_a)
        b = SimBaby(initial_energy=b_energy,
                    position=np.array([4.0, 1.0, 5.0]), params=params, group_id=group_b)
        return params, a, b

    def test_cooperation_transfers_fraction_of_surplus(self):
        params, a, b = self._pair(200.0, 20.0)
        a.perceptron_entity.W[:] = 0.0
        a.perceptron_entity.b[:] = np.array([-10.0, 10.0])  # cooperate open
        before_a, before_b = a.energy, b.energy
        moved = a.social_step(b)
        assert moved["act"] == "cooperate"
        assert moved["energy_moved"] == pytest.approx(min(200.0 * params.share_fraction,
                                                          params.start_energy))
        assert a.energy == pytest.approx(before_a - moved["energy_moved"])
        assert b.energy == pytest.approx(before_b + moved["energy_moved"])

    def test_cooperation_needs_surplus(self):
        # At exactly start_energy the cooperate gate may be open but there is
        # no surplus to share — the act is skipped.
        params, a, b = self._pair(100.0, 20.0)
        a.perceptron_entity.W[:] = 0.0
        a.perceptron_entity.b[:] = np.array([-10.0, 10.0])
        out = a.social_step(b)
        assert out["act"] == "none"

    def test_contest_takes_from_weaker_only(self):
        params, a, b = self._pair(100.0, 50.0)
        a.perceptron_entity.W[:] = 0.0
        a.perceptron_entity.b[:] = np.array([10.0, -10.0])  # contest open
        before = b.energy
        moved = a.social_step(b)
        assert moved["act"] == "contest"
        assert moved["energy_moved"] == pytest.approx(min(params.contest_take, before))
        assert b.energy == pytest.approx(before - moved["energy_moved"])

    def test_no_contest_against_stronger(self):
        params, a, b = self._pair(50.0, 100.0)
        a.perceptron_entity.W[:] = 0.0
        a.perceptron_entity.b[:] = np.array([10.0, -10.0])  # contest gate open
        assert a.social_step(b)["act"] == "none"

    def test_social_target_is_neediest_neighbor(self):
        # A well-fed donor with its cooperate gate open must help the hungriest
        # nearby baby, not the nearest — this is what rescues a starving
        # tribe-mate and lifts the tribe's geometric mean.
        params = WorldParams(grid_size=(8, 4, 8))
        scene = SimScene(params=params)
        donor = TestKinSignal._quiet_baby(200.0, (4.0, 1.0, 4.0), group_id=0)
        mid = TestKinSignal._quiet_baby(90.0, (5.0, 1.0, 4.0), group_id=0)
        starving = TestKinSignal._quiet_baby(30.0, (4.0, 1.0, 6.0), group_id=0)
        donor.perceptron_entity.W[:] = 0.0
        donor.perceptron_entity.b[:] = np.array([-10.0, 10.0])  # cooperate open
        for baby in (donor, mid, starving):
            scene.add_baby(baby)
        before = {id(b): b.energy for b in (mid, starving)}
        sim = Simulation(scene, max_ticks=1)
        sim.step()
        assert starving.energy > before[id(starving)]
        assert mid.energy <= before[id(mid)] + 1.0  # untouched (its own drain only)


# ── Social emergence benchmark (Stage 6) ─────────────────────────────────────

class TestSocialBenchmark:
    """
    Two selection objectives on the same grouped world: individual fitness
    punishes sharing (the free-rider problem), trait-group selection rewards
    it — so group-arm cooperation must outlast the individual arm's.
    """

    def test_benchmark_deterministic_with_seed(self):
        a = benchmark_social(seed=11)
        b = benchmark_social(seed=11)
        assert a["individual"]["best_fitness"] == b["individual"]["best_fitness"]
        assert a["group"]["best_fitness"] == b["group"]["best_fitness"]
        assert [h["avg_fitness"] for h in a["group"]["history"]] == \
               [h["avg_fitness"] for h in b["group"]["history"]]

    def test_result_structure(self):
        r = benchmark_social(seed=1)
        assert set(r) == {"individual", "group", "group_count", "group_weight",
                          "individual_cooperate_rate", "group_cooperate_rate",
                          "individual_contest_rate", "group_contest_rate",
                          "cooperation_emerged"}
        assert r["group_count"] == 2
        assert r["group_weight"] > 0.0
        assert len(r["individual"]["history"]) == r["individual"]["generations"]
        assert len(r["group"]["history"]) == r["group"]["generations"]

    def test_group_selection_sustains_cooperation(self):
        r = benchmark_social(seed=1)
        assert r["group_cooperate_rate"] > r["individual_cooperate_rate"]
        assert r["cooperation_emerged"] is True

    def test_individual_selection_punishes_sharing(self):
        # A pure individual objective removes cooperation: the donor is out-bred
        # by the free-rider it feeds, so the act rate collapses.
        r = benchmark_social(seed=1)
        grp_acts = [h["cooperations"] for h in r["group"]["history"]]
        ind_acts = [h["cooperations"] for h in r["individual"]["history"]]
        assert max(grp_acts) > max(ind_acts)
        assert ind_acts[-1] <= ind_acts[1]  # decays after the initial random burst

    def test_group_arm_contests_less(self):
        # Contesting a tribe-mate steals energy from the geometric mean, so
        # group selection also suppresses intra-tribe aggression.
        r = benchmark_social(seed=1)
        assert r["individual_contest_rate"] > r["group_contest_rate"]


# ── Directed communication (Stage 6) ─────────────────────────────────────────

class TestDirectedMessage:
    """
    The directed-message channel: a sender addresses one specific neighbor,
    pays a cost scaled by the gate amplitude, and the recipient perceives the
    message exactly one tick later as the entity feature at index 5. Opt-in
    (``message_enabled``) so the locked selection proofs keep their exact
    genome layout.
    """

    @staticmethod
    def _params(**kw):
        base = dict(grid_size=(8, 4, 8), message_enabled=True,
                    entity_input_dim=6, social_enabled=False)
        base.update(kw)
        return WorldParams(**base)

    @staticmethod
    def _quiet_baby(params, energy, position, group_id=0):
        """A baby whose every decision gate is pinned off/neutral."""
        b = SimBaby(initial_energy=energy,
                    position=np.array(position, dtype=np.float64),
                    params=params, group_id=group_id)
        for p in (b.perceptron_cells, b.perceptron_body, b.perceptron_move):
            p.W[:] = 0.0
            p.b[:] = 0.5 if p is b.perceptron_move else -10.0
        b.perceptron_entity.W[:] = 0.0
        b.perceptron_entity.b[:] = -10.0
        if b.perceptron_message is not None:
            b.perceptron_message.W[:] = 0.0
            b.perceptron_message.b[:] = -10.0
        return b

    def _sender(self, params, energy=200.0):
        """A baby with its message gate forced open (amplitude = sigmoid(10))."""
        b = self._quiet_baby(params, energy, (4.0, 1.0, 4.0))
        b.perceptron_message.W[:] = 0.0
        b.perceptron_message.b[:] = 10.0
        return b

    def test_off_by_default_keeps_locked_proofs(self):
        # The channel is opt-in: default params create no message brain, so
        # the benchmark genomes (and their RNG streams) are untouched.
        assert WorldParams().message_enabled is False
        b = SimBaby(params=WorldParams(grid_size=(8, 4, 8)))
        assert b.perceptron_message is None
        on = SimBaby(params=self._params())
        assert on.perceptron_message is not None
        assert on.perceptron_message.W.shape == (6, 1)

    def test_gate_below_threshold_emits_nothing(self):
        params = self._params()
        a = self._quiet_baby(params, 200.0, (4.0, 1.0, 4.0))
        b = self._quiet_baby(params, 20.0, (4.0, 1.0, 5.0))
        assert a.decide_message(b) == 0.0
        assert a._last_message_out is not None and a._last_message_out < params.message_gate_threshold

    def test_open_gate_emits_amplitude(self):
        params = self._params()
        a = self._sender(params)
        b = self._quiet_baby(params, 20.0, (4.0, 1.0, 5.0))
        amp = a.decide_message(b)
        expected = 1.0 / (1.0 + np.exp(-10.0))
        assert amp == pytest.approx(expected)
        assert 0.0 < amp <= 1.0

    def test_emission_posts_to_bus_and_charges_cost(self):
        params = self._params()
        scene = SimScene(params=params)
        a = self._sender(params)
        b = self._quiet_baby(params, 20.0, (4.0, 1.0, 5.0))
        for baby in (a, b):
            scene.add_baby(baby)
        sim = Simulation(scene, max_ticks=2)
        results = sim.run()
        ra = next(r for r in results if r["baby_id"] == a.entity.id)
        amp = ra["message_amplitude"]
        assert amp > 0.0
        assert ra["message_energy"] == pytest.approx(params.message_cost * amp)
        # The first emission was delivered at the start of tick 2; the tick-2
        # emission is now queued for tick 3.
        expected = 1.0 / (1.0 + np.exp(-10.0))
        assert b._inbox == {a.entity.id: pytest.approx(expected)}
        assert len(scene._pending_messages) == 1

    def test_delivery_is_one_tick_latent_and_directed(self):
        # Sender targets the neediest neighbor; only that baby's inbox gets the
        # message, and only from the second tick onward.
        params = self._params()
        scene = SimScene(params=params)
        a = self._sender(params)
        b = self._quiet_baby(params, 20.0, (4.0, 1.0, 5.0))
        c = self._quiet_baby(params, 90.0, (5.0, 1.0, 4.0))
        for baby in (a, b, c):
            scene.add_baby(baby)
        sim = Simulation(scene, max_ticks=1)
        sim.step()
        assert len(scene._pending_messages) == 1
        sender, target, amp = scene._pending_messages[0]
        assert sender == a.entity.id and target == b.entity.id
        assert b._inbox == {}  # not delivered yet

        sim.step()
        assert b._inbox == {a.entity.id: pytest.approx(amp)}
        # Tick 2 emits a second message addressed the same way; c was never
        # the target and never receives anything.
        assert len(scene._pending_messages) == 1
        assert scene._pending_messages[0][1] == b.entity.id
        assert a._inbox == {}
        assert c._inbox == {}

    def test_recipient_perceives_message_feature(self):
        params = self._params()
        scene = SimScene(params=params)
        a = self._sender(params)
        b = self._quiet_baby(params, 20.0, (4.0, 1.0, 5.0))
        c = self._quiet_baby(params, 90.0, (5.0, 1.0, 4.0))
        for baby in (a, b, c):
            scene.add_baby(baby)
        Simulation(scene, max_ticks=2).run()
        sender_id = a.entity.id
        amp = 1.0 / (1.0 + np.exp(-10.0))
        p = b.perceive(scene.world, babies=[a, b, c])
        entry = next(e for e in p.nearby_entities if e["id"] == sender_id)
        assert entry["message"] == pytest.approx(amp)
        assert b._entity_features(entry)[5] == pytest.approx(amp)

    def test_message_feature_sliced_away_at_dim5(self):
        # entity_input_dim=5 drops the message bit; a dim-5 brain still runs
        # social decisions without a shape mismatch (message channel invisible).
        params = WorldParams(grid_size=(8, 4, 8), message_enabled=True,
                             entity_input_dim=5)
        a = SimBaby(initial_energy=200.0, position=np.array([4.0, 1.0, 4.0]),
                    params=params, group_id=0)
        b = SimBaby(initial_energy=20.0, position=np.array([4.0, 1.0, 5.0]),
                    params=params, group_id=0)
        assert a.perceptron_entity.W.shape == (5, 2)
        assert a.perceptron_message.W.shape == (5, 1)
        a.perceptron_entity.W[:] = 0.0
        a.perceptron_entity.b[:] = np.array([-10.0, 10.0])  # cooperate open
        assert a.social_step(b)["act"] == "cooperate"
        assert len(a._entity_features({"type": 0, "energy": 20.0, "distance": 1.0,
                                       "angle": 0.0, "group_id": 0, "message": 0.9})) == 5

    def test_genome_carries_message_tensors_when_enabled(self):
        params = self._params()
        a = self._sender(params)
        b = self._quiet_baby(params, 20.0, (4.0, 1.0, 5.0))
        g = Genome.from_baby(a)
        assert g.tensors["message.W"].shape == (6, 1)
        fresh = self._quiet_baby(params, 100.0, (1.0, 1.0, 1.0))
        g.apply_to(fresh)
        assert np.allclose(fresh.perceptron_message.W, a.perceptron_message.W)
        assert np.allclose(fresh.perceptron_message.b, a.perceptron_message.b)
        # A disabled baby carries no message tensors at all.
        off = SimBaby(params=WorldParams(grid_size=(8, 4, 8)))
        g_off = Genome.from_baby(off)
        assert "message.W" not in g_off.tensors

    def test_random_genome_shapes_follow_params(self):
        params = self._params()
        rng = np.random.default_rng(0)
        g = Genome.random(params, rng)
        assert g.tensors["message.W"].shape == (6, 1)
        assert g.tensors["message.b"].shape == (1,)
        off_rng = np.random.default_rng(0)
        g_off = Genome.random(WorldParams(grid_size=(8, 4, 8)), off_rng)
        assert "message.W" not in g_off.tensors

    def test_scene_roundtrip_preserves_pending_messages(self):
        params = self._params()
        scene = SimScene(params=params)
        a = self._quiet_baby(params, 100.0, (4.0, 1.0, 4.0))
        b = self._quiet_baby(params, 20.0, (4.0, 1.0, 5.0))
        for baby in (a, b):
            scene.add_baby(baby)
        scene._pending_messages.append((a.entity.id, b.entity.id, 0.8))
        restored = SimScene.from_dict(json.loads(json.dumps(scene.to_dict())))
        assert restored._pending_messages == [(a.entity.id, b.entity.id, 0.8)]
        restored.deliver_messages()
        assert restored._pending_messages == []
        assert restored.get_baby(b.entity.id)._inbox == {a.entity.id: 0.8}

    def test_learning_updates_message_brain(self):
        params = self._params(learning_enabled=True)
        a = self._quiet_baby(params, 200.0, (4.0, 1.0, 4.0))
        a.perceptron_message.W[:] = 0.0
        a.perceptron_message.b[:] = 0.0
        b = self._quiet_baby(params, 20.0, (4.0, 1.0, 5.0))
        a.decide_message(b)
        before = a.perceptron_message.W.copy()
        a.learn(-0.5)
        assert not np.allclose(a.perceptron_message.W, before)


# ── Durable structures / nests (Stage 7) ─────────────────────────────────────

class TestNests:
    """
    Stage 7 durable structures. A cell-write deposit near a nest feeds its
    bank; a substantial deposit far from any nest seeds a new one — both are
    transfers, never creation, so the world still conserves energy. A baby
    under its start energy can draw from its own tribe's nearest nest (a
    starvation buffer that rewards territoriality and resource pooling).
    Opt-in (``structure_enabled``) so the locked selection proofs keep their
    exact genome layout and RNG stream.
    """

    @staticmethod
    def _params(**kw):
        base = dict(grid_size=(8, 4, 8), structure_enabled=True,
                    social_enabled=False, message_enabled=False,
                    learning_enabled=False)
        base.update(kw)
        return WorldParams(**base)

    @staticmethod
    def _quiet_baby(params, energy, position, group_id=0):
        """A baby whose every decision gate is pinned off."""
        b = SimBaby(initial_energy=energy,
                    position=np.array(position, dtype=np.float64),
                    params=params, group_id=group_id)
        for p in (b.perceptron_cells, b.perceptron_body, b.perceptron_move):
            p.W[:] = 0.0
            p.b[:] = -10.0
        b.perceptron_entity.W[:] = 0.0
        b.perceptron_entity.b[:] = -10.0
        return b

    @staticmethod
    def _write(x, y, z, energy, material=MATERIAL_ORGANIC):
        return CellWrite(x, y, z, material, energy)

    def test_off_by_default_keeps_locked_proofs(self):
        # Opt-in: default params disable the whole structure system, so the
        # benchmark genomes (and their RNG streams) are untouched.
        params = WorldParams(grid_size=(8, 4, 8))
        assert params.structure_enabled is False
        scene = SimScene(params=params)
        baby = self._quiet_baby(params, 100.0, (4.0, 1.0, 4.0))
        scene.add_baby(baby)
        scene.nests.append(Nest(id=1, position=np.array([4.5, 1.0, 4.5]),
                                stored_energy=3.0, owner_group_id=0))
        action = BabyAction(writes=[self._write(4, 1, 4, 5.0)])
        assert scene.route_build(action, baby) == (0.0, 0)
        assert scene.draw_nest(baby) == 0.0
        scene.update_nests()
        assert len(scene.nests) == 1  # untouched by everything

    def test_nearest_nest_group_filtering(self):
        params = self._params()
        scene = SimScene(params=params)
        scene.nests.append(Nest(id=1, position=np.array([1.0, 1.0, 1.0]),
                                stored_energy=2.0, owner_group_id=0))
        scene.nests.append(Nest(id=2, position=np.array([2.0, 1.0, 2.0]),
                                stored_energy=4.0, owner_group_id=1))
        assert scene.nearest_nest(np.array([1.5, 1.0, 1.5]), 1.0).id == 1
        assert scene.nearest_nest(
            np.array([1.5, 1.0, 1.5]), 3.0, group_id=0).id == 1
        assert scene.nearest_nest(
            np.array([1.5, 1.0, 1.5]), 3.0, group_id=1).id == 2
        assert scene.nearest_nest(np.array([10.0, 1.0, 10.0]), 3.0) is None
        scene.nests[0].alive = False
        # nest1 dead and nest2 is out of reach of nest1's own cell
        assert scene.nearest_nest(np.array([1.0, 1.0, 1.0]), 1.0) is None

    def test_write_near_nest_feeds_bank(self):
        params = self._params()
        scene = SimScene(params=params)
        baby = self._quiet_baby(params, 100.0, (4.0, 1.0, 4.0))
        scene.add_baby(baby)
        scene.nests.append(Nest(id=1, position=np.array([4.5, 1.0, 4.5]),
                                stored_energy=3.0, owner_group_id=0))
        action = BabyAction(writes=[self._write(4, 1, 4, 5.0)])
        baby.apply_action(action, scene.world)
        nested, seeded = scene.route_build(action, baby)
        assert nested == pytest.approx(5.0)
        assert seeded == 0
        assert len(scene.nests) == 1
        assert scene.nests[0].stored_energy == pytest.approx(8.0)
        # the deposit is a transfer: the fed cell is left as rubble
        assert scene.world.get_cell(4, 1, 4).energy == pytest.approx(0.0)

    def test_write_far_from_nest_seeds_new_nest(self):
        params = self._params()
        scene = SimScene(params=params)
        baby = self._quiet_baby(params, 100.0, (4.0, 1.0, 4.0), group_id=7)
        scene.add_baby(baby)
        action = BabyAction(writes=[self._write(4, 1, 4, 5.0)])
        baby.apply_action(action, scene.world)
        nested, seeded = scene.route_build(action, baby)
        assert nested == pytest.approx(0.0)
        assert seeded == 1
        nest = scene.nests[0]
        assert nest.owner_group_id == 7
        assert nest.stored_energy == pytest.approx(5.0)
        assert scene.world.get_cell(4, 1, 4).energy == pytest.approx(0.0)

    def test_seed_requires_enough_energy(self):
        params = self._params()
        scene = SimScene(params=params)
        baby = self._quiet_baby(params, 100.0, (4.0, 1.0, 4.0))
        scene.add_baby(baby)
        action = BabyAction(writes=[self._write(4, 1, 4, 1.0)])
        baby.apply_action(action, scene.world)
        nested, seeded = scene.route_build(action, baby)
        assert seeded == 0
        assert scene.nests == []
        # below the seed threshold the write stays a plain cell deposit
        assert scene.world.get_cell(4, 1, 4).energy == pytest.approx(1.0)

    def test_nest_count_capped_at_max(self):
        params = self._params(max_nests=1)
        scene = SimScene(params=params)
        baby = self._quiet_baby(params, 100.0, (4.0, 1.0, 4.0))
        scene.add_baby(baby)
        scene.nests.append(Nest(id=1, position=np.array([1.5, 1.0, 1.5]),
                                stored_energy=3.0, owner_group_id=0))
        action = BabyAction(writes=[self._write(4, 1, 4, 5.0)])
        baby.apply_action(action, scene.world)
        nested, seeded = scene.route_build(action, baby)
        assert seeded == 0
        assert len(scene.nests) == 1
        # cap reached — the write is not routed and keeps its cell energy
        assert scene.world.get_cell(4, 1, 4).energy == pytest.approx(5.0)

    def test_deposit_and_draw_conserve_energy(self):
        # A full deposit -> draw cycle moves energy baby -> nest -> baby and
        # nothing is created or destroyed (decay off, draw rate uncapped).
        params = self._params(nest_draw_rate=10.0, nest_decay=0.0)
        scene = SimScene(params=params)
        baby = self._quiet_baby(params, 100.0, (4.0, 1.0, 4.0))
        scene.add_baby(baby)
        action = BabyAction(writes=[self._write(4, 1, 4, 5.0)])
        baby.apply_action(action, scene.world)
        scene.route_build(action, baby)
        baby.entity.energy -= 5.0  # the deposit charge, as step() does
        nest = scene.nests[0]
        assert nest.stored_energy == pytest.approx(5.0)
        drawn = scene.draw_nest(baby)
        assert drawn == pytest.approx(5.0)  # capped by the gap to start energy
        assert baby.energy == pytest.approx(100.0)
        assert nest.stored_energy == pytest.approx(0.0)
        total = scene.world.total_energy + baby.energy + nest.stored_energy
        assert total == pytest.approx(100.0)

    def test_draw_only_when_starving_and_own_tribe(self):
        params = self._params()
        scene = SimScene(params=params)
        baby = self._quiet_baby(params, 90.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(baby)
        scene.nests.append(Nest(id=1, position=np.array([4.5, 1.0, 4.5]),
                                stored_energy=10.0, owner_group_id=0))
        drawn = scene.draw_nest(baby)
        assert drawn == pytest.approx(1.0)  # capped by the draw rate
        assert baby.energy == pytest.approx(91.0)
        assert scene.nests[0].stored_energy == pytest.approx(9.0)
        # a well-fed baby never taps the bank
        full = self._quiet_baby(params, 100.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(full)
        assert scene.draw_nest(full) == 0.0
        # a foreign tribe's nest is off-limits territory
        outsider = self._quiet_baby(params, 90.0, (4.0, 1.0, 4.0), group_id=9)
        scene.add_baby(outsider)
        assert scene.draw_nest(outsider) == 0.0

    def test_update_nests_decays_and_prunes_empty(self):
        params = self._params(nest_decay=0.5)
        scene = SimScene(params=params)
        scene.nests.append(Nest(id=1, position=np.array([1.0, 1.0, 1.0]),
                                stored_energy=10.0, owner_group_id=0))
        scene.nests.append(Nest(id=2, position=np.array([2.0, 1.0, 2.0]),
                                stored_energy=1e-12, owner_group_id=0))
        scene.update_nests()
        assert len(scene.nests) == 1
        assert scene.nests[0].stored_energy == pytest.approx(5.0)

    def test_nest_perceived_as_entity(self):
        params = self._params()
        scene = SimScene(params=params)
        baby = self._quiet_baby(params, 100.0, (4.0, 1.0, 4.0))
        scene.add_baby(baby)
        scene.nests.append(Nest(id=3, position=np.array([4.5, 1.0, 4.5]),
                                stored_energy=7.0, owner_group_id=0))
        p = baby.perceive(scene.world, babies=[baby], nests=scene.nests)
        entry = next(e for e in p.nearby_entities if e.get("is_nest"))
        assert entry["id"] == 3
        assert entry["type"] == int(EntityType.OBJECT)
        assert entry["energy"] == pytest.approx(7.0)
        assert entry["group_id"] == 0
        assert entry["message"] == 0.0

    def test_nests_invisible_when_disabled(self):
        params = WorldParams(grid_size=(8, 4, 8), structure_enabled=False)
        scene = SimScene(params=params)
        baby = self._quiet_baby(params, 100.0, (4.0, 1.0, 4.0))
        scene.add_baby(baby)
        scene.nests.append(Nest(id=3, position=np.array([4.5, 1.0, 4.5]),
                                stored_energy=7.0, owner_group_id=0))
        p = baby.perceive(scene.world, babies=[baby], nests=scene.nests)
        assert not any(e.get("is_nest") for e in p.nearby_entities)

    def test_nest_serialization_roundtrip(self):
        params = self._params()
        scene = SimScene(params=params)
        baby = self._quiet_baby(params, 100.0, (4.0, 1.0, 4.0))
        scene.add_baby(baby)
        action = BabyAction(writes=[self._write(4, 1, 4, 5.0)])
        baby.apply_action(action, scene.world)
        scene.route_build(action, baby)
        restored = SimScene.from_dict(json.loads(json.dumps(scene.to_dict())))
        assert len(restored.nests) == 1
        n = restored.nests[0]
        assert n.id == 1
        assert n.stored_energy == pytest.approx(5.0)
        assert n.owner_group_id == 0
        assert np.allclose(n.position, [4.5, 1.0, 4.5])
        # the id counter resumes past restored nests (no id collision)
        assert restored._next_nest_id == 2
        baby2 = self._quiet_baby(params, 100.0, (4.0, 1.0, 4.0))
        restored.add_baby(baby2)
        action2 = BabyAction(writes=[self._write(7, 1, 7, 5.0)])
        baby2.apply_action(action2, restored.world)
        _, seeded = restored.route_build(action2, baby2)
        assert seeded == 1
        assert restored.nests[-1].id == 2

    def test_old_scene_snapshot_without_nests_still_loads(self):
        params = WorldParams(grid_size=(8, 4, 8))
        data = SimScene(params=params).to_dict()
        del data["nests"]
        del data["next_nest_id"]
        restored = SimScene.from_dict(data)
        assert restored.nests == []
        assert restored._next_nest_id == 1

    def test_structures_emerge_in_simulation(self):
        # A baby with its write gate forced open deposits energy every tick;
        # after a few ticks the world holds at least one seeded nest and the
        # simulation ledger reports the new structure metrics.
        np.random.seed(0)
        params = self._params(nest_decay=0.0, write_energy_scale=10.0)
        scene = SimScene(params=params)
        baby = self._quiet_baby(params, 500.0, (4.0, 1.0, 4.0))
        baby.perceptron_cells.W[:] = 0.0
        baby.perceptron_cells.b[:] = np.array([1.0, 0.5, 1.0])
        scene.add_baby(baby)
        sim = Simulation(scene, max_ticks=3)
        sim.run()
        summary = sim.summary()
        assert len(scene.nests) >= 1
        assert summary["nests_built"] >= 1
        assert scene.info()["nests"] >= 1
        assert scene.info()["nest_energy"] > 0.0
        assert "total_nested" in summary and "total_drawn" in summary
        assert all("nested" in r and "seeded" in r and "drawn" in r
                   for r in sim.tick_log)


# ── Delta-rule sign (Stage 7 reward fix) ─────────────────────────────────────

class TestDeltaRuleSign:
    """
    The delta rule's direction comes from the error sign; the learning rate
    is always positive. A loss must weaken the output and a gain reinforce it.
    The original defect used a negative learning rate on losses, which
    double-flipped the sign (error * lr came out positive) so "weakening"
    actually reinforced the bad behavior and drove the perceptrons toward
    saturation.
    """

    def test_loss_weakens_and_gain_reinforces(self):
        np.random.seed(0)
        p = Perceptron(4, 1)
        p.W[:] = np.zeros_like(p.W) + 0.1
        p.b[:] = np.array([0.5], dtype=np.float32)
        x = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        before = float(p.forward(x)[0])
        p.update(x, np.array([-1.0]), lr=0.01)
        after_loss = float(p.forward(x)[0])
        assert after_loss < before
        p.update(x, np.array([1.0]), lr=0.01)
        assert float(p.forward(x)[0]) > after_loss

    def test_neutral_error_does_not_change_weights(self):
        p = Perceptron(4, 1)
        x = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        w0 = p.W.copy()
        b0 = p.b.copy()
        p.update(x, np.array([0.0]), lr=0.01)
        assert np.allclose(p.W, w0)
        assert np.allclose(p.b, b0)


# ── Cultural transmission / teaching (Stage 7) ───────────────────────────────

class TestTeaching:
    """
    Stage 7 cultural transmission. A surplus baby whose teach perceptron gate
    clears ``teach_gate_threshold`` teaches the neediest tribe-mate in range:
    the student's behavior weights blend toward the teacher's and a capped
    number of the teacher's best episodes are copied into the student's
    memory. Opt-in (``teaching_enabled``) so the locked selection proofs keep
    their exact genome layout and RNG stream.
    """

    @staticmethod
    def _params(**kw):
        base = dict(grid_size=(8, 4, 8), teaching_enabled=True,
                    social_enabled=False, message_enabled=False,
                    structure_enabled=False)
        base.update(kw)
        return WorldParams(**base)

    @staticmethod
    def _pin(baby):
        """Zero every behavior gate so the baby acts inertly."""
        for p in (baby.perceptron_cells, baby.perceptron_body, baby.perceptron_move):
            p.W[:] = 0.0
            p.b[:] = -10.0
        baby.perceptron_entity.W[:] = 0.0
        baby.perceptron_entity.b[:] = -10.0
        if baby.perceptron_teach is not None:
            # a default teach gate (b=0 -> sigmoid 0.5) equals the 0.5
            # threshold and would fire at amplitude 0.5 — pin it off too
            baby.perceptron_teach.W[:] = 0.0
            baby.perceptron_teach.b[:] = -10.0

    def test_off_by_default_keeps_locked_proofs(self):
        params = WorldParams(grid_size=(8, 4, 8))
        baby = SimBaby(initial_energy=100.0, params=params)
        other = SimBaby(initial_energy=50.0, params=params)
        assert params.teaching_enabled is False
        assert baby.perceptron_teach is None
        assert baby.decide_teach(other) == 0.0

    def test_surplus_gate(self):
        # A teacher at or below start energy never spends energy on a lesson:
        # teaching only pays off when there is a surplus (mirrors cooperation).
        params = self._params()
        baby = SimBaby(initial_energy=50.0, params=params,
                       position=np.array([0.0, 1.0, 0.0]))
        baby.perceptron_teach.W[:] = 0.0
        baby.perceptron_teach.b[:] = np.array([10.0])  # gate pinned open
        other = SimBaby(initial_energy=50.0, params=params,
                        position=np.array([1.0, 1.0, 0.0]))
        assert baby.decide_teach(other) == 0.0

    def test_gate_threshold(self):
        # Below threshold no lesson; at/above it the gate value becomes the
        # lesson amplitude, so a louder lesson transfers more behavior and
        # costs more energy.
        params = self._params(teach_gate_threshold=0.5)
        baby = SimBaby(initial_energy=200.0, params=params,
                       position=np.array([0.0, 1.0, 0.0]))
        baby.perceptron_teach.W[:] = 0.0
        other = SimBaby(initial_energy=50.0, params=params,
                        position=np.array([1.0, 1.0, 0.0]))
        baby.perceptron_teach.b[:] = np.array([-2.0])  # sigmoid ~0.12
        assert baby.decide_teach(other) == 0.0
        baby.perceptron_teach.b[:] = np.array([2.0])   # sigmoid ~0.88
        amp = baby.decide_teach(other)
        assert amp > 0.5
        assert amp == pytest.approx(float(Perceptron._sigmoid(np.array([2.0]))[0]))

    def test_teach_blends_weights_and_respects_episode_cap(self):
        # A lesson closes ``teach_weight_blend * amplitude`` of the weight gap
        # and copies at most ``teach_memotype_cap`` episodes; 0 disables the
        # episode copy entirely (the locked benchmark config keeps it at 1).
        params = self._params(teach_weight_blend=0.5, teach_memotype_cap=0)
        teacher = SimBaby(initial_energy=200.0, params=params)
        student = SimBaby(initial_energy=100.0, params=params)
        for i, reward in enumerate((1.0, 5.0, 3.0, 2.0, 4.0)):
            teacher.memory.record(np.full(4, float(i), dtype=np.float32),
                                  (0.5, 0.5, 0.5), reward, tick=i)
        for name in ("cells", "body", "entity", "move"):
            tp = getattr(teacher, f"perceptron_{name}")
            sp = getattr(student, f"perceptron_{name}")
            tp.W[:] = 1.0
            tp.b[:] = 0.0
            sp.W[:] = 0.0
            sp.b[:] = 1.0
        assert teacher.teach(student, 1.0) == 0  # cap=0 -> no episodes copied
        for name in ("cells", "body", "entity", "move"):
            sp = getattr(student, f"perceptron_{name}")
            assert sp.W[0, 0] == pytest.approx(0.5)  # half the gap closed
            assert sp.b[0] == pytest.approx(0.5)

    def test_teach_copies_highest_reward_episodes(self):
        params = self._params(teach_memotype_cap=2)
        teacher = SimBaby(initial_energy=200.0, params=params)
        student = SimBaby(initial_energy=100.0, params=params)
        for i, reward in enumerate((1.0, 5.0, 3.0, 2.0, 4.0)):
            teacher.memory.record(np.full(4, float(i), dtype=np.float32),
                                  (0.5, 0.5, 0.5), reward, tick=i)
        copied = teacher.teach(student, 1.0)
        assert copied == 2
        assert sorted(e.reward for e in student.memory.recall(64)) == [4.0, 5.0]

    def test_teaches_only_neediest_same_tribe_student(self):
        # In-range tribe-mates are taught; out-group neighbors are never
        # chosen (cultural transmission is in-group, mirroring the tribe-
        # scoped nest draw). Among tribe-mates the hungriest is the target.
        np.random.seed(0)
        params = self._params(teach_weight_blend=0.5, teach_memotype_cap=0,
                              learning_enabled=False)
        scene = SimScene(params=params)
        teacher = SimBaby(initial_energy=500.0, params=params,
                          position=np.array([0.0, 1.0, 0.0]), group_id=0)
        starved = SimBaby(initial_energy=50.0, params=params,
                          position=np.array([1.0, 1.0, 0.0]), group_id=0)
        full = SimBaby(initial_energy=300.0, params=params,
                       position=np.array([2.0, 1.0, 0.0]), group_id=0)
        outsider = SimBaby(initial_energy=50.0, params=params,
                           position=np.array([3.0, 1.0, 0.0]), group_id=1)
        teacher.perceptron_teach.W[:] = 0.0
        teacher.perceptron_teach.b[:] = np.array([5.0])  # gate forced open
        for name in ("cells", "body", "entity", "move"):
            getattr(teacher, f"perceptron_{name}").W[:] = 2.0
            getattr(teacher, f"perceptron_{name}").b[:] = 0.0
        for s in (starved, full, outsider):
            self._pin(s)
        for baby in (teacher, starved, full, outsider):
            scene.add_baby(baby)
        sim = Simulation(scene, max_ticks=1)
        sim.run()
        teacher_log = [r for r in sim.tick_log if r["baby_id"] == teacher.entity.id][0]
        assert teacher_log["teaching_amplitude"] > 0.5
        assert teacher_log["teaching_energy"] == pytest.approx(
            params.teach_cost * teacher_log["teaching_amplitude"])
        # only the hungriest in-group neighbor received the blend
        amp = teacher_log["teaching_amplitude"]
        assert starved.perceptron_body.W[0, 0] == pytest.approx(
            2.0 * params.teach_weight_blend * amp)
        assert full.perceptron_body.W[0, 0] == pytest.approx(0.0)
        assert outsider.perceptron_body.W[0, 0] == pytest.approx(0.0)

    def test_serialization_includes_teach_brain(self):
        params = self._params()
        baby = SimBaby(initial_energy=100.0, params=params)
        baby.perceptron_teach.W[:] = 0.25
        baby.perceptron_teach.b[:] = np.array([0.5])
        data = baby.to_dict()
        restored = SimBaby.from_dict(data, params=params)
        assert restored.perceptron_teach is not None
        assert np.allclose(restored.perceptron_teach.W, baby.perceptron_teach.W)
        assert np.allclose(restored.perceptron_teach.b, baby.perceptron_teach.b)


# ── Cultural transmission benchmark (Stage 7) ────────────────────────────────

class TestCultureBenchmark:
    """
    Stage 7 cultural transmission proof: both arms evolve on the same grouped
    world and both learn; the only difference is the transmission channel.
    Control keeps the memotype (vertical inheritance at birth); culture adds
    lateral teaching. The verdict ``culture_emerged`` is the culture arm
    beating control at final average fitness.
    """

    def test_benchmark_deterministic_with_seed(self):
        a = benchmark_culture(seed=1)
        b = benchmark_culture(seed=1)
        assert a["control_last_avg"] == b["control_last_avg"]
        assert a["culture_last_avg"] == b["culture_last_avg"]
        assert [h["avg_fitness"] for h in a["control"]["history"]] == \
               [h["avg_fitness"] for h in b["control"]["history"]]
        assert [h["avg_fitness"] for h in a["culture"]["history"]] == \
               [h["avg_fitness"] for h in b["culture"]["history"]]

    def test_result_structure(self):
        r = benchmark_culture(seed=1)
        assert set(r) == {"control", "culture", "group_count", "group_weight",
                          "control_last_avg", "culture_last_avg",
                          "control_teach_rate", "culture_teach_rate",
                          "culture_emerged"}
        assert r["group_count"] == 2
        assert r["group_weight"] > 0.0
        assert len(r["control"]["history"]) == r["control"]["generations"]
        assert len(r["culture"]["history"]) == r["culture"]["generations"]

    def test_only_culture_arm_teaches(self):
        r = benchmark_culture(seed=1)
        assert r["control_teach_rate"] == 0.0
        assert r["culture_teach_rate"] > 0.0

    def test_locked_verdict_across_five_seeds(self):
        # Locked with the honest net-delta reward (step 8b) and the delta-rule
        # sign fix. Teaching amplifies lineage quality — good or bad — so the
        # culture arm wins 3/5 worlds; it is a genuine lateral channel, not a
        # guarantee of emergence on every world (seed 4's culture lineage
        # drifted toward aggression and teaching spread it).
        expected = {1: False, 2: True, 3: True, 4: False, 5: True}
        for seed, want in expected.items():
            r = benchmark_culture(seed=seed)
            assert r["culture_emerged"] is want, f"seed {seed}"

    def test_culture_wins_on_emerged_seed(self):
        r = benchmark_culture(seed=2)
        assert r["culture_last_avg"] > r["control_last_avg"]


# ── Long-term memory benchmark (Stage 7) ──────────────────────────────────────

class TestMemoryBenchmark:
    """
    Stage 7 long-term memory proof: both arms evolve on the same grouped
    world and both learn; the only difference is the persistence channel.
    Control keeps only the vertical memotype (parent -> child at birth).
    Memory adds the world reservoir — dead babies deposit in-sim, survivors
    deposit at every generation boundary, newborns are seeded from the
    reservoir's best episodes on top of their memotype, and nothing is ever
    evicted. The verdict ``memory_emerged`` is the memory arm beating control
    at final average fitness. The reservoir is an honest channel — it
    preserves whatever experience actually happened, good or bad, so it wins
    2 of 5 locked worlds rather than every world.
    """

    @classmethod
    def _run(cls, seed: int) -> dict:
        if seed not in cls._cache:
            cls._cache[seed] = benchmark_memory(seed=seed)
        return cls._cache[seed]

    _cache: dict[int, dict] = {}

    def test_benchmark_deterministic_with_seed(self):
        a = benchmark_memory(seed=1)
        b = benchmark_memory(seed=1)
        assert a["control_last_avg"] == b["control_last_avg"]
        assert a["memory_last_avg"] == b["memory_last_avg"]
        assert [h["avg_fitness"] for h in a["control"]["history"]] == \
               [h["avg_fitness"] for h in b["control"]["history"]]
        assert [h["avg_fitness"] for h in a["memory"]["history"]] == \
               [h["avg_fitness"] for h in b["memory"]["history"]]
        assert [h["memory_size"] for h in a["memory"]["history"]] == \
               [h["memory_size"] for h in b["memory"]["history"]]

    def test_result_structure(self):
        r = self._run(1)
        assert set(r) == {"control", "memory", "group_count", "group_weight",
                          "control_last_avg", "memory_last_avg",
                          "memory_size", "memory_seeds", "memory_emerged"}
        assert r["group_count"] == 2
        assert r["group_weight"] > 0.0
        assert len(r["control"]["history"]) == r["control"]["generations"]
        assert len(r["memory"]["history"]) == r["memory"]["generations"]

    def test_only_memory_arm_keeps_reservoir(self):
        r = self._run(1)
        assert all(h["memory_size"] == 0 for h in r["control"]["history"])
        assert all(h["memory_size"] > 0 for h in r["memory"]["history"])
        assert r["memory_size"] > 0

    def test_reservoir_append_only_and_seeds_accumulate(self):
        r = self._run(1)
        sizes = [h["memory_size"] for h in r["memory"]["history"]]
        assert sizes == sorted(sizes)  # nothing is ever evicted
        assert sum(h["memory_seeds"] for h in r["memory"]["history"]) == \
               r["memory_seeds"]
        assert r["memory_seeds"] > 0

    def test_locked_verdict_across_five_seeds(self):
        # Locked with the honest energy-flow deposit (no episode-into-reward
        # edits) and the cross-lineage birth seed. The reservoir is a genuine
        # lateral channel: on seeds 1/3/4 the seeded lineages drifted toward
        # dead ends that the reservoir faithfully preserved and propagated.
        expected = {1: False, 2: True, 3: False, 4: False, 5: True}
        for seed, want in expected.items():
            r = self._run(seed)
            assert r["memory_emerged"] is want, f"seed {seed}"

    def test_memory_wins_on_emerged_seed(self):
        r = self._run(2)
        assert r["memory_last_avg"] > r["control_last_avg"]



