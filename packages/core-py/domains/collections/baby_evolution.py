import numpy as np
from dataclasses import dataclass, field
from typing import Callable
import random


@dataclass
class EvolutionConfig:
    population_size: int = 20
    mutation_rate: float = 0.1
    mutation_strength: float = 0.2
    crossover_rate: float = 0.7
    tournament_size: int = 3
    elitism_count: int = 2
    fitness_window: int = 50
    reproduction_threshold: float = 80.0
    max_offspring: int = 2


class Genome:
    def __init__(self, genes: np.ndarray | None = None, gene_names: list[str] | None = None):
        if genes is not None:
            self.genes = genes.astype(np.float32)
        else:
            self.genes = np.random.randn(16).astype(np.float32) * 0.1
        self.gene_names = gene_names or [f"gene_{i}" for i in range(len(self.genes))]
        self.fitness: float = 0.0
        self.generation: int = 0
        self.parent_ids: list[int] = []

    def copy(self):
        g = Genome(self.genes.copy(), self.gene_names.copy())
        g.fitness = self.fitness
        g.generation = self.generation
        g.parent_ids = list(self.parent_ids)
        return g

    def mutate(self, rate: float = 0.1, strength: float = 0.2):
        mask = np.random.random(len(self.genes)) < rate
        noise = np.random.randn(mask.sum()).astype(np.float32) * strength
        self.genes[mask] += noise

    def crossover(self, other: 'Genome') -> 'Genome':
        child_genes = self.genes.copy()
        mask = np.random.random(len(self.genes)) < 0.5
        child_genes[mask] = other.genes[mask]
        child = Genome(child_genes, self.gene_names.copy())
        child.parent_ids = [id(self), id(other)]
        return child

    def distance(self, other: 'Genome') -> float:
        return float(np.linalg.norm(self.genes - other.genes))

    def to_dict(self) -> dict:
        return {
            "genes": self.genes.tolist(),
            "fitness": self.fitness,
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Genome':
        g = cls(np.array(d["genes"], dtype=np.float32))
        g.fitness = d.get("fitness", 0.0)
        g.generation = d.get("generation", 0)
        return g

    def __len__(self):
        return len(self.genes)


class FitnessTracker:
    def __init__(self, window: int = 50):
        self.window = window
        self._scores: dict[int, list[float]] = {}

    def record(self, baby_id: int, score: float):
        if baby_id not in self._scores:
            self._scores[baby_id] = []
        self._scores[baby_id].append(score)
        if len(self._scores[baby_id]) > self.window:
            self._scores[baby_id] = self._scores[baby_id][-self.window:]

    def get_fitness(self, baby_id: int) -> float:
        scores = self._scores.get(baby_id, [])
        if not scores:
            return 0.0
        return float(np.mean(scores[-self.window:]))

    def get_all_fitness(self) -> dict[int, float]:
        return {bid: self.get_fitness(bid) for bid in self._scores}

    def get_trend(self, baby_id: int) -> float:
        scores = self._scores.get(baby_id, [])
        if len(scores) < 2:
            return 0.0
        recent = np.mean(scores[-10:])
        older = np.mean(scores[:-10]) if len(scores) > 10 else scores[0]
        return float(recent - older)

    def summary(self) -> dict:
        all_fitness = self.get_all_fitness()
        return {
            "tracked_babies": len(all_fitness),
            "avg_fitness": float(np.mean(list(all_fitness.values()))) if all_fitness else 0.0,
            "max_fitness": float(max(all_fitness.values())) if all_fitness else 0.0,
        }


class SelectionOperator:
    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    def tournament_select(self, population: list[Genome]) -> Genome:
        contestants = random.sample(population, min(self.tournament_size, len(population)))
        return max(contestants, key=lambda g: g.fitness)

    def rank_select(self, population: list[Genome]) -> Genome:
        sorted_pop = sorted(population, key=lambda g: g.fitness)
        ranks = np.arange(1, len(sorted_pop) + 1, dtype=np.float32)
        probs = ranks / ranks.sum()
        idx = np.random.choice(len(sorted_pop), p=probs)
        return sorted_pop[idx]

    def roulette_select(self, population: list[Genome]) -> Genome:
        fitnesses = np.array([max(g.fitness, 0.01) for g in population])
        probs = fitnesses / fitnesses.sum()
        idx = np.random.choice(len(population), p=probs)
        return population[idx]


class EvolutionEngine:
    def __init__(self, config: EvolutionConfig | None = None):
        self.config = config or EvolutionConfig()
        self._population: list[Genome] = []
        self._generation = 0
        self._history: list[dict] = []
        self._fitness_tracker = FitnessTracker(self.config.fitness_window)
        self._selection = SelectionOperator(self.config.tournament_size)
        self._next_id = 0

    def initialize_population(self, gene_size: int = 16, gene_names: list[str] | None = None):
        self._population = []
        for _ in range(self.config.population_size):
            g = Genome(gene_names=gene_names)
            if len(g) != gene_size:
                g.genes = np.random.randn(gene_size).astype(np.float32) * 0.1
            self._population.append(g)

    def add_genome(self, genome: Genome):
        self._population.append(genome)

    def record_fitness(self, baby_id: int, fitness: float):
        self._fitness_tracker.record(baby_id, fitness)

    def update_population_fitness(self, babies: list):
        for baby in babies:
            if baby.alive:
                fitness = self._calculate_fitness(baby)
                self._fitness_tracker.record(baby.entity.id, fitness)

        for genome in self._population:
            genome.fitness = self._fitness_tracker.get_fitness(genome.id if hasattr(genome, 'id') else 0)

    def _calculate_fitness(self, baby) -> float:
        energy_score = baby.energy / 100.0
        age_score = min(baby._total_ticks / 100.0, 1.0) if hasattr(baby, '_total_ticks') else 0.0
        memory_score = 0.0
        if hasattr(baby, 'memory'):
            memory_score = len(baby.memory._buffer) / baby.memory.capacity if hasattr(baby.memory, '_buffer') else 0.0
        return energy_score * 0.5 + age_score * 0.3 + memory_score * 0.2

    def step(self) -> list[Genome]:
        if len(self._population) < 2:
            return list(self._population)

        self._generation += 1
        new_population = []

        sorted_pop = sorted(self._population, key=lambda g: g.fitness, reverse=True)
        for i in range(min(self.config.elitism_count, len(sorted_pop))):
            elite = sorted_pop[i].copy()
            elite.generation = self._generation
            new_population.append(elite)

        while len(new_population) < self.config.population_size:
            if random.random() < self.config.crossover_rate:
                parent1 = self._selection.tournament_select(self._population)
                parent2 = self._selection.tournament_select(self._population)
                child = parent1.crossover(parent2)
            else:
                parent = self._selection.tournament_select(self._population)
                child = parent.copy()

            child.mutate(self.config.mutation_rate, self.config.mutation_strength)
            child.generation = self._generation
            child.fitness = 0.0
            new_population.append(child)

        self._population = new_population[:self.config.population_size]

        stats = self._generation_stats()
        self._history.append(stats)
        return list(self._population)

    def _generation_stats(self) -> dict:
        fitnesses = [g.fitness for g in self._population]
        return {
            "generation": self._generation,
            "population_size": len(self._population),
            "avg_fitness": float(np.mean(fitnesses)) if fitnesses else 0.0,
            "max_fitness": float(max(fitnesses)) if fitnesses else 0.0,
            "min_fitness": float(min(fitnesses)) if fitnesses else 0.0,
            "diversity": self._diversity(),
        }

    def _diversity(self) -> float:
        if len(self._population) < 2:
            return 0.0
        total_dist = 0.0
        count = 0
        for i in range(len(self._population)):
            for j in range(i + 1, min(i + 5, len(self._population))):
                total_dist += self._population[i].distance(self._population[j])
                count += 1
        return float(total_dist / max(count, 1))

    def get_best(self, n: int = 1) -> list[Genome]:
        return sorted(self._population, key=lambda g: g.fitness, reverse=True)[:n]

    def get_population(self) -> list[Genome]:
        return list(self._population)

    def export_population(self) -> list[dict]:
        return [g.to_dict() for g in self._population]

    def import_population(self, data: list[dict]):
        self._population = [Genome.from_dict(d) for d in data]

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def population_size(self) -> int:
        return len(self._population)

    def summary(self) -> dict:
        stats = self._generation_stats() if self._population else {}
        return {
            "generation": self._generation,
            "population_size": len(self._population),
            "history_length": len(self._history),
            "fitness_tracker": self._fitness_tracker.summary(),
            "current_stats": stats,
        }
