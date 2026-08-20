import numpy as np
import pytest

from domains.collections.baby_evolution import (
    EvolutionConfig, Genome, FitnessTracker, SelectionOperator,
    EvolutionEngine,
)


class TestGenome:
    def test_create_random(self):
        g = Genome()
        assert len(g.genes) == 16
        assert g.fitness == 0.0

    def test_create_with_genes(self):
        genes = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        g = Genome(genes)
        assert len(g) == 3
        assert g.genes[0] == 1.0

    def test_copy(self):
        g = Genome()
        g.fitness = 0.5
        g2 = g.copy()
        assert g2.fitness == 0.5
        g2.genes[0] = 999.0
        assert g.genes[0] != 999.0

    def test_mutate(self):
        g = Genome(np.zeros(10, dtype=np.float32))
        g.mutate(rate=1.0, strength=1.0)
        assert not np.all(g.genes == 0)

    def test_crossover(self):
        g1 = Genome(np.ones(10, dtype=np.float32))
        g2 = Genome(np.zeros(10, dtype=np.float32))
        child = g1.crossover(g2)
        assert len(child) == 10
        assert child is not g1
        assert child is not g2

    def test_distance(self):
        g1 = Genome(np.zeros(10, dtype=np.float32))
        g2 = Genome(np.ones(10, dtype=np.float32))
        assert g1.distance(g2) > 0

    def test_to_dict(self):
        g = Genome(np.array([1.0, 2.0], dtype=np.float32))
        d = g.to_dict()
        assert "genes" in d
        assert "fitness" in d

    def test_from_dict(self):
        d = {"genes": [1.0, 2.0], "fitness": 0.5, "generation": 3}
        g = Genome.from_dict(d)
        assert g.genes[0] == 1.0
        assert g.fitness == 0.5
        assert g.generation == 3


class TestFitnessTracker:
    def test_record(self):
        tracker = FitnessTracker()
        tracker.record(1, 0.5)
        tracker.record(1, 0.6)
        assert len(tracker._scores[1]) == 2

    def test_get_fitness(self):
        tracker = FitnessTracker()
        tracker.record(1, 0.5)
        tracker.record(1, 0.7)
        assert abs(tracker.get_fitness(1) - 0.6) < 1e-6

    def test_get_fitness_empty(self):
        tracker = FitnessTracker()
        assert tracker.get_fitness(999) == 0.0

    def test_get_all_fitness(self):
        tracker = FitnessTracker()
        tracker.record(1, 0.5)
        tracker.record(2, 0.7)
        all_f = tracker.get_all_fitness()
        assert 1 in all_f
        assert 2 in all_f

    def test_get_trend(self):
        tracker = FitnessTracker()
        for i in range(20):
            tracker.record(1, 0.1 + i * 0.01)
        trend = tracker.get_trend(1)
        assert trend > 0

    def test_summary(self):
        tracker = FitnessTracker()
        tracker.record(1, 0.5)
        s = tracker.summary()
        assert s["tracked_babies"] == 1


class TestSelectionOperator:
    def test_tournament_select(self):
        op = SelectionOperator(tournament_size=2)
        pop = [Genome(np.array([i], dtype=np.float32)) for i in range(10)]
        for g in pop:
            g.fitness = float(g.genes[0])
        selected = op.tournament_select(pop)
        assert selected in pop

    def test_rank_select(self):
        op = SelectionOperator()
        pop = [Genome(np.array([i], dtype=np.float32)) for i in range(10)]
        for g in pop:
            g.fitness = float(g.genes[0])
        selected = op.rank_select(pop)
        assert selected in pop

    def test_roulette_select(self):
        op = SelectionOperator()
        pop = [Genome(np.array([i], dtype=np.float32)) for i in range(10)]
        for g in pop:
            g.fitness = float(g.genes[0]) + 1.0
        selected = op.roulette_select(pop)
        assert selected in pop


class TestEvolutionEngine:
    def test_initialize_population(self):
        engine = EvolutionEngine(EvolutionConfig(population_size=10))
        engine.initialize_population(gene_size=8)
        assert engine.population_size == 10

    def test_add_genome(self):
        engine = EvolutionEngine()
        g = Genome()
        engine.add_genome(g)
        assert engine.population_size == 1

    def test_step(self):
        engine = EvolutionEngine(EvolutionConfig(population_size=10))
        engine.initialize_population()
        pop = engine.step()
        assert len(pop) == 10
        assert engine.generation == 1

    def test_step_multiple(self):
        engine = EvolutionEngine(EvolutionConfig(population_size=10))
        engine.initialize_population()
        for _ in range(5):
            engine.step()
        assert engine.generation == 5

    def test_get_best(self):
        engine = EvolutionEngine(EvolutionConfig(population_size=10))
        engine.initialize_population()
        engine._population[0].fitness = 1.0
        engine._population[1].fitness = 0.5
        best = engine.get_best(1)
        assert best[0].fitness == 1.0

    def test_export_import(self):
        engine = EvolutionEngine(EvolutionConfig(population_size=5))
        engine.initialize_population()
        data = engine.export_population()
        engine2 = EvolutionEngine()
        engine2.import_population(data)
        assert engine2.population_size == 5

    def test_diversity(self):
        engine = EvolutionEngine(EvolutionConfig(population_size=10))
        engine.initialize_population()
        diversity = engine._diversity()
        assert diversity >= 0.0

    def test_summary(self):
        engine = EvolutionEngine(EvolutionConfig(population_size=5))
        engine.initialize_population()
        s = engine.summary()
        assert "generation" in s
        assert "population_size" in s
        assert "fitness_tracker" in s

    def test_elitism(self):
        config = EvolutionConfig(population_size=10, elitism_count=3)
        engine = EvolutionEngine(config)
        engine.initialize_population()
        engine._population[0].fitness = 1.0
        engine._population[1].fitness = 0.9
        engine._population[2].fitness = 0.8
        engine.step()
        fitnesses = [g.fitness for g in engine._population]
        assert max(fitnesses) > 0.5
