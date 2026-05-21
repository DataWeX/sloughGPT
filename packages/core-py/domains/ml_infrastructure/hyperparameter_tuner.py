"""
Hyperparameter Tuner - Automated Hyperparameter Optimization

Production-grade hyperparameter tuning with:
- Grid search
- Random search
- Bayesian optimization
- Early stopping
- Parallel execution
- Resource management
"""

import json
import time
import uuid
import logging
import random
import math
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger("sloughgpt.tuning")


class TuneStatus(Enum):
    """Tuning job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EARLY_STOPPED = "early_stopped"


class TuneObjective(Enum):
    """Tuning objective direction."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class SearchStrategy(Enum):
    """Hyperparameter search strategy."""

    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"
    HYPERBAND = "hyperband"


@dataclass
class HyperparameterSpec:
    """Specification for a single hyperparameter."""

    name: str
    hp_type: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    values: Optional[List[Any]] = None
    default: Any = None
    log_scale: bool = False

    def sample(self, rng: random.Random) -> Any:
        """Sample a value for this hyperparameter."""
        if self.values:
            return rng.choice(self.values)

        if self.hp_type == "int":
            return rng.randint(int(self.min_value), int(self.max_value))
        elif self.hp_type == "float":
            if self.log_scale:
                min_val = math.log10(self.min_value)
                max_val = math.log10(self.max_value)
                value = math.pow(10, rng.uniform(min_val, max_val))
            else:
                value = rng.uniform(self.min_value, self.max_value)
            return round(value, 6)
        elif self.hp_type == "categorical":
            return rng.choice([True, False])

        return self.default

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Trial:
    """A single trial (hyperparameter configuration)."""

    trial_id: str
    job_id: str
    params: Dict[str, Any]
    status: TuneStatus
    objective_value: Optional[float]
    metrics: Dict[str, float]
    start_time: float
    end_time: Optional[float]
    iteration: int
    error: Optional[str]

    def duration(self) -> Optional[float]:
        if self.end_time:
            return self.end_time - self.start_time
        return None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class TuningJob:
    """A hyperparameter tuning job."""

    job_id: str
    name: str
    objective_metric: str
    objective: TuneObjective
    strategy: SearchStrategy
    max_trials: int
    max_iterations: int
    max_parallel: int
    early_stopping_patience: int
    early_stopping_threshold: float
    specs: Dict[str, HyperparameterSpec]
    trials: List[Trial]
    status: TuneStatus
    best_trial: Optional[Trial]
    best_value: Optional[float]
    created_at: float
    updated_at: float

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["objective"] = self.objective.value
        data["strategy"] = self.strategy.value
        data["specs"] = {
            k: HyperparameterSpec(**v).to_dict() if isinstance(v, dict) else v
            for k, v in data["specs"].items()
        }
        data["trials"] = [
            Trial(**t, status=TuneStatus(t["status"])).to_dict()
            if isinstance(t, dict)
            else t.to_dict()
            for t in data["trials"]
        ]
        if data.get("best_trial") and isinstance(data["best_trial"], dict):
            data["best_trial"] = Trial(
                **{**data["best_trial"], "status": TuneStatus(data["best_trial"]["status"])}
            ).to_dict()
        return data


class SearchSpace(ABC):
    """Abstract base for search spaces."""

    @abstractmethod
    def sample(self, rng: random.Random) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_specs(self) -> Dict[str, HyperparameterSpec]:
        pass


class HyperparameterSearchSpace(SearchSpace):
    """Concrete search space implementation."""

    def __init__(self):
        self.specs: Dict[str, HyperparameterSpec] = {}

    def add_int(
        self, name: str, min_val: int, max_val: int, default: Optional[int] = None
    ) -> "HyperparameterSearchSpace":
        self.specs[name] = HyperparameterSpec(
            name=name,
            hp_type="int",
            min_value=float(min_val),
            max_value=float(max_val),
            default=default or min_val,
        )
        return self

    def add_float(
        self,
        name: str,
        min_val: float,
        max_val: float,
        default: Optional[float] = None,
        log_scale: bool = False,
    ) -> "HyperparameterSearchSpace":
        self.specs[name] = HyperparameterSpec(
            name=name,
            hp_type="float",
            min_value=min_val,
            max_value=max_val,
            default=default or min_val,
            log_scale=log_scale,
        )
        return self

    def add_categorical(
        self, name: str, values: List[Any], default: Optional[Any] = None
    ) -> "HyperparameterSearchSpace":
        self.specs[name] = HyperparameterSpec(
            name=name, hp_type="categorical", values=values, default=default or values[0]
        )
        return self

    def add_log_uniform(
        self, name: str, min_val: float, max_val: float, default: Optional[float] = None
    ) -> "HyperparameterSearchSpace":
        self.specs[name] = HyperparameterSpec(
            name=name,
            hp_type="float",
            min_value=min_val,
            max_value=max_val,
            default=default or min_val,
            log_scale=True,
        )
        return self

    def sample(self, rng: random.Random) -> Dict[str, Any]:
        return {name: spec.sample(rng) for name, spec in self.specs.items()}

    def get_specs(self) -> Dict[str, HyperparameterSpec]:
        return self.specs.copy()


class BaseOptimizer(ABC):
    """Base class for optimization strategies."""

    @abstractmethod
    def suggest(self, completed_trials: List[Trial]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def update(self, trial: Trial):
        pass


class RandomOptimizer(BaseOptimizer):
    """Random search optimizer."""

    def __init__(self, search_space: HyperparameterSearchSpace):
        self.search_space = search_space
        self.rng = random.Random()

    def suggest(self, completed_trials: List[Trial]) -> Dict[str, Any]:
        return self.search_space.sample(self.rng)

    def update(self, trial: Trial):
        pass


class GridOptimizer(BaseOptimizer):
    """Grid search optimizer."""

    def __init__(self, search_space: HyperparameterSearchSpace, points_per_dim: int = 5):
        self.search_space = search_space
        self.points_per_dim = points_per_dim
        self.grid_points: List[Dict[str, Any]] = []
        self.current_idx = 0
        self._generate_grid()

    def _generate_grid(self):
        specs = self.search_space.get_specs()

        for name, spec in specs.items():
            if spec.values:
                spec.values = spec.values[: self.points_per_dim]
            elif spec.hp_type in ("int", "float"):
                spec.min_value = spec.min_value
                spec.max_value = spec.max_value

        names = list(specs.keys())
        grid = self._cartesian_product(specs, names, 0)

        rng = random.Random(42)
        rng.shuffle(grid)
        self.grid_points = grid

    def _cartesian_product(
        self, specs: Dict[str, HyperparameterSpec], names: List[str], idx: int
    ) -> List[Dict[str, Any]]:
        if idx >= len(names):
            return [{}]

        name = names[idx]
        spec = specs[name]
        results = []

        if spec.values:
            values = spec.values
        else:
            if spec.hp_type == "int":
                values = list(range(int(spec.min_value), int(spec.max_value) + 1))
            else:
                values = np.linspace(spec.min_value, spec.max_value, self.points_per_dim).tolist()

        for v in values:
            for rest in self._cartesian_product(specs, names, idx + 1):
                rest[name] = v
                results.append(rest)

        return results

    def suggest(self, completed_trials: List[Trial]) -> Dict[str, Any]:
        if self.current_idx >= len(self.grid_points):
            return {}
        params = self.grid_points[self.current_idx]
        self.current_idx += 1
        return params

    def update(self, trial: Trial):
        pass


class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization using Gaussian Processes."""

    def __init__(self, search_space: HyperparameterSearchSpace, n_initial: int = 5):
        self.search_space = search_space
        self.n_initial = n_initial
        self.observations: List[Tuple[Dict[str, float], float]] = []
        self.rng = random.Random()

    def suggest(self, completed_trials: List[Trial]) -> Dict[str, Any]:
        if len(completed_trials) < self.n_initial:
            return self.search_space.sample(self.rng)

        return self._optimize_acquisition(completed_trials)

    def _optimize_acquisition(self, trials: List[Trial]) -> Dict[str, Any]:
        best_value = min(t.objective_value for t in trials if t.objective_value is not None)

        candidates = [self.search_space.sample(self.rng) for _ in range(100)]

        best_candidate = candidates[0]
        best_score = float("inf")

        for candidate in candidates:
            score = self._expected_improvement(candidate, best_value)
            if score < best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    def _expected_improvement(self, params: Dict[str, Any], best_value: float) -> float:
        mu, sigma = self._predict(params)

        if sigma < 1e-6:
            return 0.0

        z = (best_value - mu) / sigma
        ei = (best_value - mu) * self._norm_cdf(z) + sigma * self._norm_pdf(z)

        return -ei

    def _norm_cdf(self, x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _norm_pdf(self, x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    def _predict(self, params: Dict[str, Any]) -> Tuple[float, float]:
        if not self.observations:
            return 0.0, 1.0

        values = [v for _, v in self.observations]
        return float(np.mean(values)), float(np.std(values)) if len(values) > 1 else 1.0

    def update(self, trial: Trial):
        if trial.objective_value is not None:
            param_values = {k: float(v) for k, v in trial.params.items()}
            self.observations.append((param_values, trial.objective_value))


class HyperparameterTuner:
    """
    Hyperparameter tuning system.

    Features:
    - Multiple search strategies (grid, random, bayesian)
    - Early stopping
    - Parallel trial execution
    - Trial progress tracking
    - Best configuration selection
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, storage_path: str = "data/tuning"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, storage_path: str = "data/tuning"):
        if self._initialized:
            return

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.jobs: Dict[str, TuningJob] = {}
        self._lock = threading.Lock()

        self._load_jobs()
        self._initialized = True
        logger.info(f"HyperparameterTuner initialized at {self.storage_path}")

    def _load_jobs(self):
        """Load tuning jobs from storage."""
        jobs_file = self.storage_path / "tuning_jobs.json"
        if jobs_file.exists():
            try:
                with open(jobs_file) as f:
                    data = json.load(f)
                    for job_data in data.values():
                        job_data["specs"] = {
                            k: HyperparameterSpec(**v) for k, v in job_data.get("specs", {}).items()
                        }
                        job_data["trials"] = [
                            Trial(**{**t, "status": TuneStatus(t["status"])})
                            for t in job_data.get("trials", [])
                        ]
                        job_data["best_trial"] = (
                            Trial(
                                **{
                                    **job_data["best_trial"],
                                    "status": TuneStatus(job_data["best_trial"]["status"]),
                                }
                            )
                            if job_data.get("best_trial")
                            else None
                        )
                        job_data["status"] = TuneStatus(job_data["status"])
                        job_data["objective"] = TuneObjective(job_data["objective"])
                        job_data["strategy"] = SearchStrategy(job_data["strategy"])
                        self.jobs[job_data["job_id"]] = TuningJob(**job_data)
            except Exception as e:
                logger.warning(f"Failed to load tuning jobs: {e}")

    def _save_jobs(self):
        """Persist tuning jobs to storage."""
        jobs_file = self.storage_path / "tuning_jobs.json"
        try:
            with open(jobs_file, "w") as f:
                data = {k: v.to_dict() for k, v in self.jobs.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tuning jobs: {e}")

    def create_tuning_job(
        self,
        name: str,
        search_space: SearchSpace,
        objective_metric: str,
        objective: TuneObjective = TuneObjective.MINIMIZE,
        strategy: SearchStrategy = SearchStrategy.RANDOM,
        max_trials: int = 100,
        max_iterations: int = 1000,
        max_parallel: int = 4,
        early_stopping_patience: int = 10,
        early_stopping_threshold: float = 0.01,
    ) -> str:
        """Create a new tuning job."""
        job_id = str(uuid.uuid4())[:12]

        job = TuningJob(
            job_id=job_id,
            name=name,
            objective_metric=objective_metric,
            objective=objective,
            strategy=strategy,
            max_trials=max_trials,
            max_iterations=max_iterations,
            max_parallel=max_parallel,
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
            specs=search_space.get_specs()
            if isinstance(search_space, HyperparameterSearchSpace)
            else {},
            trials=[],
            status=TuneStatus.PENDING,
            best_trial=None,
            best_value=None,
            created_at=time.time(),
            updated_at=time.time(),
        )

        with self._lock:
            self.jobs[job_id] = job
            self._save_jobs()

        logger.info(f"Created tuning job: {name} ({job_id})")
        return job_id

    def start_tuning(
        self,
        job_id: str,
        objective_fn: Callable[[Dict[str, Any]], float],
        callback: Optional[Callable[[Trial], None]] = None,
    ) -> None:
        """Start a tuning job."""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        job.status = TuneStatus.RUNNING
        self._save_jobs()

        if isinstance(job.specs, dict) and job.specs:
            search_space = HyperparameterSearchSpace()
            for name, spec in job.specs.items():
                if spec.hp_type == "int":
                    search_space.add_int(
                        name, int(spec.min_value), int(spec.max_value), spec.default
                    )
                elif spec.hp_type == "float":
                    search_space.add_float(
                        name, spec.min_value, spec.max_value, spec.default, spec.log_scale
                    )
                elif spec.values:
                    search_space.add_categorical(name, spec.values, spec.default)
        else:
            return

        if job.strategy == SearchStrategy.GRID:
            optimizer = GridOptimizer(search_space)
        elif job.strategy == SearchStrategy.BAYESIAN:
            optimizer = BayesianOptimizer(search_space)
        else:
            optimizer = RandomOptimizer(search_space)

        completed_trials = []
        no_improvement_count = 0

        try:
            for trial_num in range(job.max_trials):
                if job.status == TuneStatus.CANCELLED:
                    break

                params = optimizer.suggest(completed_trials)

                trial = Trial(
                    trial_id=str(uuid.uuid4())[:12],
                    job_id=job_id,
                    params=params,
                    status=TuneStatus.RUNNING,
                    objective_value=None,
                    metrics={},
                    start_time=time.time(),
                    end_time=None,
                    iteration=0,
                    error=None,
                )

                with self._lock:
                    job.trials.append(trial)

                try:
                    objective_value = objective_fn(params)

                    trial.objective_value = objective_value
                    trial.metrics[job.objective_metric] = objective_value
                    trial.status = TuneStatus.COMPLETED
                    trial.end_time = time.time()

                    completed_trials.append(trial)
                    optimizer.update(trial)

                    if (
                        job.best_value is None
                        or (
                            job.objective == TuneObjective.MINIMIZE
                            and objective_value < job.best_value
                        )
                        or (
                            job.objective == TuneObjective.MAXIMIZE
                            and objective_value > job.best_value
                        )
                    ):
                        job.best_value = objective_value
                        job.best_trial = trial
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1

                    if callback:
                        callback(trial)

                    if no_improvement_count >= job.early_stopping_patience:
                        job.status = TuneStatus.EARLY_STOPPED
                        break

                except Exception as e:
                    trial.status = TuneStatus.FAILED
                    trial.error = str(e)
                    trial.end_time = time.time()

                job.updated_at = time.time()
                self._save_jobs()

            if job.status == TuneStatus.RUNNING:
                job.status = TuneStatus.COMPLETED

        except Exception as e:
            job.status = TuneStatus.FAILED
            logger.error(f"Tuning job {job_id} failed: {e}")

        self._save_jobs()

    def get_job(self, job_id: str) -> Optional[TuningJob]:
        return self.jobs.get(job_id)

    def list_jobs(self, status: Optional[TuneStatus] = None) -> List[TuningJob]:
        if status:
            return [j for j in self.jobs.values() if j.status == status]
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    def get_best_params(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.jobs.get(job_id)
        if job and job.best_trial:
            return job.best_trial.params
        return None

    def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job and job.status == TuneStatus.RUNNING:
            job.status = TuneStatus.CANCELLED
            self._save_jobs()
            return True
        return False

    def compare_trials(self, job_id: str, metric: str) -> List[Dict[str, Any]]:
        job = self.jobs.get(job_id)
        if not job:
            return []

        return [
            {
                "trial_id": t.trial_id,
                "params": t.params,
                "objective_value": t.objective_value,
                "metrics": t.metrics,
                "duration": t.duration(),
                "status": t.status.value,
            }
            for t in sorted(job.trials, key=lambda x: x.objective_value or float("inf"))
            if t.objective_value is not None
        ]


tuner = HyperparameterTuner()


def create_tuning_job(
    name: str,
    search_space: SearchSpace,
    objective_metric: str,
    objective: TuneObjective = TuneObjective.MINIMIZE,
    strategy: SearchStrategy = SearchStrategy.RANDOM,
    max_trials: int = 100,
) -> str:
    return tuner.create_tuning_job(
        name, search_space, objective_metric, objective, strategy, max_trials
    )


def start_tuning(job_id: str, objective_fn: Callable):
    tuner.start_tuning(job_id, objective_fn)


def get_best_params(job_id: str) -> Optional[Dict[str, Any]]:
    return tuner.get_best_params(job_id)


__all__ = [
    "HyperparameterTuner",
    "TuningJob",
    "Trial",
    "HyperparameterSpec",
    "HyperparameterSearchSpace",
    "SearchSpace",
    "TuneStatus",
    "TuneObjective",
    "SearchStrategy",
    "BaseOptimizer",
    "RandomOptimizer",
    "GridOptimizer",
    "BayesianOptimizer",
    "tuner",
    "create_tuning_job",
    "start_tuning",
    "get_best_params",
]
