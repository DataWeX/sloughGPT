"""Tests for packages/core-py/domains/shell/evolution.py — pure logic only."""

from __future__ import annotations

import numpy as np
import pytest

from domains.shell.evolution import (
    Genome,
    EvolutionEngine,
    benchmark_emergence,
)
from domains.shell.simulation import (
    Perceptron,
    SimBaby,
    WorldParams,
)


# ── Genome ──────────────────────────────────────────────────────────────────

class TestGenome:
    def test_init(self):
        tensors = {
            "cells.W": np.zeros((5, 3), dtype=np.float32),
            "cells.b": np.zeros(3, dtype=np.float32),
            "body.W": np.zeros((3, 2), dtype=np.float32),
            "body.b": np.zeros(2, dtype=np.float32),
            "entity.W": np.zeros((5, 2), dtype=np.float32),
            "entity.b": np.zeros(2, dtype=np.float32),
            "move.W": np.zeros((5, 3), dtype=np.float32),
            "move.b": np.zeros(3, dtype=np.float32),
        }
        g = Genome(tensors, group_id=1)
        assert g.group_id == 1
        assert g.memory_count == 0

    def test_init_with_memories(self):
        tensors = {"cells.W": np.zeros((5, 3), dtype=np.float32),
                   "cells.b": np.zeros(3, dtype=np.float32),
                   "body.W": np.zeros((3, 2), dtype=np.float32),
                   "body.b": np.zeros(2, dtype=np.float32),
                   "entity.W": np.zeros((5, 2), dtype=np.float32),
                   "entity.b": np.zeros(2, dtype=np.float32),
                   "move.W": np.zeros((5, 3), dtype=np.float32),
                   "move.b": np.zeros(3, dtype=np.float32)}
        memories = [
            {"features": [1.0, 2.0], "action": [0.5, 0.3], "reward": 10.0, "tick": 1},
            {"features": [3.0, 4.0], "action": [0.2, 0.7], "reward": -5.0, "tick": 2},
        ]
        g = Genome(tensors, memories=memories)
        assert g.memory_count == 2

    def test_from_baby(self):
        params = WorldParams(cells_input_dim=5, body_input_dim=3,
                             entity_input_dim=5)
        baby = SimBaby(position=np.array([4.0, 4.0, 4.0]), params=params)
        g = Genome.from_baby(baby, group_id=3)
        assert g.group_id == 3
        assert "cells.W" in g.tensors
        assert "cells.b" in g.tensors
        assert "body.W" in g.tensors
        assert "entity.W" in g.tensors
        assert "move.W" in g.tensors
        assert g.tensors["cells.W"].shape == (5, 3)
        assert g.tensors["body.W"].shape == (3, 2)
        assert g.tensors["entity.W"].shape == (5, 2)

    def test_from_baby_with_optional_brains(self):
        p = WorldParams(cells_input_dim=5, body_input_dim=3,
                        entity_input_dim=5, message_enabled=True,
                        teaching_enabled=True, predation_enabled=True,
                        territoriality_enabled=True, lifecycle_enabled=True,
                        specialization_enabled=True)
        baby = SimBaby(params=p)
        g = Genome.from_baby(baby)
        assert "message.W" in g.tensors
        assert "teach.W" in g.tensors
        assert "predation.W" in g.tensors
        assert "territory.W" in g.tensors
        assert "reproduce.W" in g.tensors
        assert "role.W" in g.tensors

    def test_random(self):
        params = WorldParams(cells_input_dim=5, body_input_dim=3,
                             entity_input_dim=5)
        rng = np.random.default_rng(42)
        g = Genome.random(params, rng, group_id=0)
        assert g.tensors["cells.W"].shape == (5, 3)
        assert g.tensors["body.W"].shape == (3, 2)
        assert g.tensors["entity.W"].shape == (5, 2)
        assert g.tensors["move.W"].shape == (5, 3)

    def test_random_with_hidden(self):
        params = WorldParams(cells_input_dim=5, body_input_dim=3,
                             entity_input_dim=5, brain_hidden_units=4)
        rng = np.random.default_rng(42)
        g = Genome.random(params, rng)
        assert "cells.H" in g.tensors
        assert "cells.bh" in g.tensors
        assert g.tensors["cells.H"].shape == (5, 4)

    def test_apply_to(self):
        params = WorldParams(cells_input_dim=5, body_input_dim=3,
                             entity_input_dim=5)
        baby = SimBaby(params=params)
        rng = np.random.default_rng(99)
        g = Genome.random(params, rng)
        old_W = baby.perceptron_cells.W.copy()
        g.apply_to(baby)
        np.testing.assert_array_almost_equal(baby.perceptron_cells.W, g.tensors["cells.W"])
        assert not np.allclose(baby.perceptron_cells.W, old_W)

    def test_apply_to_with_hidden(self):
        params = WorldParams(cells_input_dim=5, body_input_dim=3,
                             entity_input_dim=5, brain_hidden_units=4)
        baby = SimBaby(params=params)
        rng = np.random.default_rng(99)
        g = Genome.random(params, rng)
        g.apply_to(baby)
        assert baby.perceptron_cells.H is not None
        np.testing.assert_array_almost_equal(baby.perceptron_cells.H, g.tensors["cells.H"])

    def test_crossover(self):
        params = WorldParams(cells_input_dim=5, body_input_dim=3,
                             entity_input_dim=5)
        rng = np.random.default_rng(42)
        g1 = Genome.random(params, rng, group_id=0)
        g2 = Genome.random(params, rng, group_id=1)
        child = g1.crossover(g2, rng)
        for key in g1.tensors:
            assert key in child.tensors
            assert child.tensors[key].shape == g1.tensors[key].shape
        assert child.group_id == 0

    def test_crossover_mixed_elements(self):
        params = WorldParams(cells_input_dim=5, body_input_dim=3,
                             entity_input_dim=5)
        rng = np.random.default_rng(42)
        g1 = Genome.random(params, rng)
        g2 = Genome.random(params, rng)
        child = g1.crossover(g2, rng)
        for key in g1.tensors:
            mask = np.isclose(child.tensors[key], g1.tensors[key]) | np.isclose(
                child.tensors[key], g2.tensors[key])
            assert mask.all()

    def test_mutate(self):
        params = WorldParams(cells_input_dim=5, body_input_dim=3,
                             entity_input_dim=5)
        rng = np.random.default_rng(42)
        g = Genome.random(params, rng)
        original = {k: v.copy() for k, v in g.tensors.items()}
        g.mutate(rng, rate=1.0, scale=0.5)
        changed = False
        for key in g.tensors:
            if not np.allclose(g.tensors[key], original[key]):
                changed = True
                break
        assert changed

    def test_mutate_zero_rate(self):
        params = WorldParams(cells_input_dim=5, body_input_dim=3,
                             entity_input_dim=5)
        rng = np.random.default_rng(42)
        g = Genome.random(params, rng)
        original = {k: v.copy() for k, v in g.tensors.items()}
        g.mutate(rng, rate=0.0, scale=0.1)
        for key in g.tensors:
            np.testing.assert_array_equal(g.tensors[key], original[key])


# ── EvolutionEngine ─────────────────────────────────────────────────────────

class TestEvolutionEngine:
    def test_init_defaults(self):
        engine = EvolutionEngine()
        assert engine.population_size >= 2
        assert engine.generations >= 1
        assert engine.elite_count >= 1

    def test_run_single_generation(self):
        params = WorldParams(grid_size=(8, 8, 8))
        engine = EvolutionEngine(
            params=params, population_size=4, generations=1,
            ticks_per_generation=3, organic_pools=1, seed=42,
        )
        result = engine.run()
        assert result["generations"] == 1
        assert result["population_size"] == 4
        assert result["best_fitness"] > 0.0 or result["best_fitness"] == 0.0
        assert len(result["history"]) == 1
        assert result["best_genome"] is not None

    def test_run_multiple_generations(self):
        params = WorldParams(grid_size=(8, 8, 8))
        engine = EvolutionEngine(
            params=params, population_size=4, generations=3,
            ticks_per_generation=3, organic_pools=1, seed=42,
        )
        result = engine.run()
        assert len(result["history"]) == 3
        for entry in result["history"]:
            assert "best_fitness" in entry
            assert "avg_fitness" in entry
            assert "alive" in entry
            assert "cooperations" in entry
            assert "contests" in entry

    def test_run_frozen(self):
        params = WorldParams(grid_size=(8, 8, 8))
        engine = EvolutionEngine(
            params=params, population_size=4, generations=2,
            ticks_per_generation=3, organic_pools=1, seed=42,
        )
        result = engine.run_frozen()
        assert result["generations"] == 2
        assert result["best_genome"] is None
        assert result["inherited_episodes"] == 0

    def test_elite_count_clamped(self):
        engine = EvolutionEngine(population_size=4, elite_count=10)
        assert engine.elite_count == 4

    def test_history_accumulates(self):
        params = WorldParams(grid_size=(8, 8, 8))
        engine = EvolutionEngine(
            params=params, population_size=4, generations=2,
            ticks_per_generation=3, organic_pools=1, seed=42,
        )
        engine.run()
        assert len(engine.history) == 2

    def test_group_edges(self):
        engine = EvolutionEngine(group_count=4)
        edges = engine._group_edges(64)
        assert len(edges) == 5
        assert edges[0] == 0
        assert edges[-1] == 64

    def test_grouped_spawn_positions(self):
        params = WorldParams(grid_size=(16, 8, 16))
        engine = EvolutionEngine(
            params=params, population_size=6, group_count=2, seed=42,
        )
        positions = engine._grouped_spawn_positions()
        assert len(positions) == 6

    def test_fitness(self):
        baby = SimBaby(initial_energy=150.0)
        assert EvolutionEngine._fitness(baby) == 150.0

    def test_fitness_dead(self):
        baby = SimBaby(initial_energy=0.0)
        baby.entity.alive = False
        assert EvolutionEngine._fitness(baby) == 0.0

    def test_group_means(self):
        params = WorldParams(grid_size=(8, 8, 8))
        engine = EvolutionEngine(params=params)
        babies = [SimBaby(initial_energy=100.0), SimBaby(initial_energy=200.0)]
        genomes = [Genome.random(params, np.random.default_rng(0), group_id=0),
                   Genome.random(params, np.random.default_rng(1), group_id=0)]
        means = engine._group_means(babies, genomes)
        assert 0 in means
        assert means[0] == pytest.approx(150.0)


class TestBenchmarkEmergence:
    def test_runs(self):
        result = benchmark_emergence(
            population_size=4, generations=2,
            ticks_per_generation=3, organic_pools=1, seed=42,
        )
        assert "evolved" in result
        assert "frozen" in result
        assert "emerged" in result
        assert isinstance(result["emerged"], bool)
