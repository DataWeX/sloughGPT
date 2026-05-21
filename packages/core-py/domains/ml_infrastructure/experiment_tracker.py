"""
Experiment Tracker - ML Experiment Management

Production-grade experiment tracking system inspired by MLflow/TensorBoard.
Tracks parameters, metrics, artifacts, and enables comparison across runs.
"""

import json
import time
import uuid
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import threading

logger = logging.getLogger("sloughgpt.experiments")


class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class MetricPoint:
    """Single metric data point."""

    timestamp: float
    step: int
    value: float


@dataclass
class Experiment:
    """Experiment metadata and results."""

    experiment_id: str
    name: str
    description: str
    status: ExperimentStatus
    parameters: Dict[str, Any]
    metrics: Dict[str, List[MetricPoint]]
    artifacts: Dict[str, str]
    start_time: float
    end_time: Optional[float]
    tags: Dict[str, str]
    run_id: str

    def duration(self) -> Optional[float]:
        if self.end_time:
            return self.end_time - self.start_time
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Shape used for JSON persistence (``experiments.json``)."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "parameters": self.parameters,
            "metrics": {
                k: [{"timestamp": m.timestamp, "step": m.step, "value": m.value} for m in v]
                for k, v in self.metrics.items()
            },
            "artifacts": self.artifacts,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "tags": self.tags,
            "run_id": self.run_id,
        }

    def to_api_dict(self) -> Dict[str, Any]:
        """REST-friendly experiment object (flat latest metrics, ``id``, ``created_at``)."""
        latest: Dict[str, float] = {}
        for key, points in self.metrics.items():
            if points:
                latest[key] = points[-1].value
        return {
            "experiment_id": self.experiment_id,
            "id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "parameters": self.parameters,
            "params": self.parameters,
            "metrics": latest,
            "artifacts": self.artifacts,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "tags": self.tags,
            "run_id": self.run_id,
            "created_at": datetime.fromtimestamp(self.start_time, tz=timezone.utc).isoformat(),
        }


@dataclass
class RunRecord:
    """Single run row exposed by ``GET /runs/{run_id}`` (backed by experiment storage)."""

    experiment_id: str
    run_id: str
    status: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    start_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.run_id,
            "experiment_id": self.experiment_id,
            "status": self.status,
            "metrics": self.metrics,
            "params": self.parameters,
            "created_at": datetime.fromtimestamp(self.start_time, tz=timezone.utc).isoformat(),
        }


class ExperimentTracker:
    """
    Experiment tracking system with local storage.

    Features:
    - Parameter tracking
    - Metric logging (with step and timestamp)
    - Artifact storage
    - Experiment comparison
    - Status management
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, storage_path: str = "data/experiments"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, storage_path: str = "data/experiments"):
        if self._initialized:
            return

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.experiments: Dict[str, Experiment] = {}
        self.current_run: Optional[Experiment] = None
        self._lock = threading.Lock()

        self._load_experiments()
        self._initialized = True
        logger.info(f"ExperimentTracker initialized at {self.storage_path}")

    def _load_experiments(self):
        """Load existing experiments from storage."""
        experiments_file = self.storage_path / "experiments.json"
        if experiments_file.exists():
            try:
                with open(experiments_file) as f:
                    data = json.load(f)
                    for exp_data in data.values():
                        exp_data["status"] = ExperimentStatus(exp_data["status"])
                        metrics_dict = exp_data.get("metrics", {})
                        exp_data["metrics"] = {
                            k: [MetricPoint(**m) for m in v] if isinstance(v, list) else []
                            for k, v in metrics_dict.items()
                        }
                        self.experiments[exp_data["experiment_id"]] = Experiment(**exp_data)
            except Exception as e:
                logger.warning(f"Failed to load experiments: {e}")

    def _save_experiments(self):
        """Persist experiments to storage."""
        experiments_file = self.storage_path / "experiments.json"
        try:
            with open(experiments_file, "w") as f:
                data = {k: v.to_dict() for k, v in self.experiments.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save experiments: {e}")

    def create_experiment(
        self,
        name: str,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create a new experiment."""
        experiment_id = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:12]

        run_id = str(uuid.uuid4())[:8]

        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            status=ExperimentStatus.PENDING,
            parameters=parameters or {},
            metrics={},
            artifacts={},
            start_time=time.time(),
            end_time=None,
            tags=tags or {},
            run_id=run_id,
        )

        with self._lock:
            self.experiments[experiment_id] = experiment
            self._save_experiments()

        logger.info(f"Created experiment {experiment_id}: {name}")
        return experiment_id

    @contextmanager
    def start_run(
        self,
        experiment_name: str,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Context manager for running an experiment."""
        experiment_id = self.create_experiment(
            name=experiment_name, description=description, parameters=parameters, tags=tags
        )

        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_time = time.time()

        self.current_run = experiment
        self._save_experiments()

        try:
            yield experiment
            experiment.status = ExperimentStatus.COMPLETED
            experiment.end_time = time.time()
        except Exception as e:
            experiment.status = ExperimentStatus.FAILED
            experiment.end_time = time.time()
            logger.error(f"Experiment {experiment_id} failed: {e}")
            raise
        finally:
            self._save_experiments()
            self.current_run = None

    def log_parameter(self, key: str, value: Any):
        """Log a parameter for current run."""
        if not self.current_run:
            logger.warning("No active run, parameter not logged")
            return

        with self._lock:
            self.current_run.parameters[key] = value
            self._save_experiments()

    def log_parameters(self, parameters: Dict[str, Any]):
        """Log multiple parameters."""
        for key, value in parameters.items():
            self.log_parameter(key, value)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a metric for current run."""
        if not self.current_run:
            logger.warning("No active run, metric not logged")
            return

        with self._lock:
            if key not in self.current_run.metrics:
                self.current_run.metrics[key] = []

            metric_point = MetricPoint(
                timestamp=time.time(), step=step or len(self.current_run.metrics[key]), value=value
            )
            self.current_run.metrics[key].append(metric_point)
            self._save_experiments()

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_artifact(self, name: str, path: str):
        """Log an artifact (file path)."""
        if not self.current_run:
            logger.warning("No active run, artifact not logged")
            return

        with self._lock:
            self.current_run.artifacts[name] = path
            self._save_experiments()

    def set_status(self, status: ExperimentStatus):
        """Set experiment status."""
        if not self.current_run:
            return

        with self._lock:
            self.current_run.status = status
            if status in (
                ExperimentStatus.COMPLETED,
                ExperimentStatus.FAILED,
                ExperimentStatus.CANCELLED,
            ):
                self.current_run.end_time = time.time()
            self._save_experiments()

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        return self.experiments.get(experiment_id)

    def _flatten_last_metrics(self, exp: Experiment) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, points in exp.metrics.items():
            if points:
                out[key] = points[-1].value
        return out

    def get_experiment_runs(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Runs for an experiment (current storage uses one ``run_id`` per experiment)."""
        exp = self.get_experiment(experiment_id)
        if exp is None:
            return []
        rec = RunRecord(
            experiment_id=exp.experiment_id,
            run_id=exp.run_id,
            status=exp.status.value,
            parameters=exp.parameters,
            metrics=self._flatten_last_metrics(exp),
            start_time=exp.start_time,
        )
        return [rec.to_dict()]

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        """Lookup a run by ``run_id`` (matches experiment ``run_id``)."""
        for exp in self.experiments.values():
            if exp.run_id == run_id:
                return RunRecord(
                    experiment_id=exp.experiment_id,
                    run_id=exp.run_id,
                    status=exp.status.value,
                    parameters=exp.parameters,
                    metrics=self._flatten_last_metrics(exp),
                    start_time=exp.start_time,
                )
        return None

    def get_experiment_by_name(self, name: str) -> Optional[Experiment]:
        """Get experiment by name (returns latest if multiple)."""
        matches = [e for e in self.experiments.values() if e.name == name]
        if not matches:
            return None
        return max(matches, key=lambda e: e.start_time)

    def list_experiments(
        self, status: Optional[ExperimentStatus] = None, tags: Optional[Dict[str, str]] = None
    ) -> List[Experiment]:
        """List experiments with optional filtering."""
        results = list(self.experiments.values())

        if status:
            results = [e for e in results if e.status == status]

        if tags:
            results = [e for e in results if all(e.tags.get(k) == v for k, v in tags.items())]

        return sorted(results, key=lambda e: e.start_time, reverse=True)

    def compare_experiments(
        self, experiment_ids: List[str], metric_keys: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Compare metrics across multiple experiments."""
        comparison = {}

        for exp_id in experiment_ids:
            exp = self.experiments.get(exp_id)
            if not exp:
                continue

            comparison[exp_id] = {
                "name": exp.name,
                "status": exp.status.value,
                "duration": exp.duration(),
                "metrics": {},
            }

            for metric_key in metric_keys:
                if metric_key in exp.metrics:
                    points = exp.metrics[metric_key]
                    values = [p.value for p in points]

                    comparison[exp_id]["metrics"][metric_key] = {
                        "final": values[-1] if values else None,
                        "min": min(values) if values else None,
                        "max": max(values) if values else None,
                        "mean": sum(values) / len(values) if values else None,
                        "history": values,
                    }

        return comparison

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        if experiment_id in self.experiments:
            del self.experiments[experiment_id]
            self._save_experiments()
            return True
        return False


tracker = ExperimentTracker()


def create_experiment(
    name: str,
    description: str = "",
    parameters: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    return tracker.create_experiment(name, description, parameters, tags)


@contextmanager
def start_run(
    experiment_name: str,
    description: str = "",
    parameters: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
):
    with tracker.start_run(experiment_name, description, parameters, tags) as exp:
        yield exp


def log_parameter(key: str, value: Any):
    tracker.log_parameter(key, value)


def log_parameters(parameters: Dict[str, Any]):
    tracker.log_parameters(parameters)


def log_metric(key: str, value: float, step: Optional[int] = None):
    tracker.log_metric(key, value, step)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    tracker.log_metrics(metrics, step)


def log_artifact(name: str, path: str):
    tracker.log_artifact(name, path)


__all__ = [
    "ExperimentTracker",
    "Experiment",
    "RunRecord",
    "MetricPoint",
    "ExperimentStatus",
    "tracker",
    "create_experiment",
    "start_run",
    "log_parameter",
    "log_parameters",
    "log_metric",
    "log_metrics",
    "log_artifact",
]
