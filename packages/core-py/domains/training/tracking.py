"""
MLflow/W&B Integration for SloughGPT

Experiment tracking and logging.
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("sloughgpt.tracking")


class TrackerBackend(Enum):
    """Tracking backend options."""

    MLFLOW = "mlflow"
    WANDB = "wandb"
    COMET = "comet"
    NONE = "none"


@dataclass
class TrackingConfig:
    """Configuration for experiment tracking."""

    backend: TrackerBackend = TrackerBackend.NONE
    experiment_name: str = "sloughgpt_experiment"
    run_name: Optional[str] = None
    tracking_uri: Optional[str] = None
    api_key: Optional[str] = None
    project: str = "sloughgpt"
    entity: Optional[str] = None
    #: Passed to ``wandb.init`` (e.g. ``train``, ``server``).
    job_type: Optional[str] = None
    #: Optional W&B run tags (e.g. ``["sloughgpt", "api"]``).
    tags: Optional[list] = None

    def __post_init__(self):
        if self.backend == TrackerBackend.WANDB:
            self.api_key = self.api_key or os.getenv("WANDB_API_KEY")
        elif self.backend == TrackerBackend.MLFLOW:
            self.tracking_uri = self.tracking_uri or os.getenv("MLFLOW_TRACKING_URI")


class ExperimentTracker:
    """
    Unified experiment tracker supporting multiple backends.

    Supports:
    - MLflow
    - Weights & Biases (WandB)
    - Comet.ml
    """

    def __init__(self, config: TrackingConfig):
        self.config = config
        self._client = None
        self._run = None
        self._initialize()

    def _initialize(self):
        """Initialize the tracking backend."""
        if self.config.backend == TrackerBackend.MLFLOW:
            self._init_mlflow()
        elif self.config.backend == TrackerBackend.WANDB:
            self._init_wandb()
        elif self.config.backend == TrackerBackend.COMET:
            self._init_comet()
        else:
            logger.info("No tracking backend selected")

    def _init_mlflow(self):
        """Initialize MLflow."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)
            self._client = mlflow
            logger.info(f"MLflow initialized: {self.config.tracking_uri}")
        except ImportError:
            logger.warning("MLflow not installed: pip install mlflow")

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        try:
            import wandb

            from domains.training.wandb_helpers import default_wandb_project

            kwargs: Dict[str, Any] = {
                "project": self.config.project or default_wandb_project(),
                "name": self.config.run_name,
                "entity": self.config.entity,
                "api_key": self.config.api_key,
            }
            mode = os.environ.get("WANDB_MODE", "online")
            if mode:
                kwargs["mode"] = mode
            if self.config.job_type:
                kwargs["job_type"] = self.config.job_type
            if self.config.tags:
                kwargs["tags"] = list(self.config.tags)
            wandb_dir = os.environ.get("WANDB_DIR")
            if wandb_dir:
                kwargs["dir"] = wandb_dir
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            wandb.init(**kwargs)
            self._client = wandb
            logger.info("WandB initialized: project=%s", kwargs.get("project"))
        except ImportError:
            logger.warning("WandB not installed: pip install wandb")

    def _init_comet(self):
        """Initialize Comet.ml."""
        try:
            from comet_ml import Experiment

            experiment = Experiment(
                project_name=self.config.project,
                api_key=self.config.api_key,
            )
            self._client = experiment
            logger.info(f"Comet initialized: {self.config.project}")
        except ImportError:
            logger.warning("Comet not installed: pip install comet-ml")

    def start_run(self, run_name: Optional[str] = None):
        """Start a new run."""
        if self.config.backend == TrackerBackend.MLFLOW:
            self._run = self._client.start_run(run_name=run_name)
        elif self.config.backend == TrackerBackend.WANDB:
            self._run = self._client
        elif self.config.backend == TrackerBackend.COMET:
            self._run = self._client

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric."""
        if self._run is None:
            return

        if self.config.backend == TrackerBackend.MLFLOW:
            self._client.log_metric(name, value, step=step)
        elif self.config.backend == TrackerBackend.WANDB:
            self._client.log({name: value}, step=step)
        elif self.config.backend == TrackerBackend.COMET:
            self._client.log_metric(name, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics."""
        if self._run is None:
            return
        if self.config.backend == TrackerBackend.WANDB:
            self._client.log(metrics, step=step)
            return
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_param(self, name: str, value: Any):
        """Log a parameter."""
        if self._run is None:
            return

        if self.config.backend == TrackerBackend.MLFLOW:
            self._client.log_param(name, value)
        elif self.config.backend == TrackerBackend.WANDB:
            self._client.config.update({name: value})
        elif self.config.backend == TrackerBackend.COMET:
            self._client.log_parameter(name, value)

    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        if self._run is None:
            return

        if self.config.backend == TrackerBackend.MLFLOW:
            self._client.log_params(params)
        elif self.config.backend == TrackerBackend.WANDB:
            self._client.config.update(params)
        elif self.config.backend == TrackerBackend.COMET:
            for k, v in params.items():
                self._client.log_parameter(k, v)

    def log_artifact(self, local_path: str, name: Optional[str] = None):
        """Log an artifact."""
        if self._run is None:
            return

        if self.config.backend == TrackerBackend.MLFLOW:
            self._client.log_artifact(local_path, name)
        elif self.config.backend == TrackerBackend.WANDB:
            self._client.log_artifact(local_path)
        elif self.config.backend == TrackerBackend.COMET:
            self._client.log_artifact(local_path)

    def log_model(self, model, name: str = "model"):
        """Log a model."""
        if self._run is None:
            return

        if self.config.backend == TrackerBackend.MLFLOW:
            self._client.pyfunc.log_model(name, python_model=model)
        elif self.config.backend == TrackerBackend.WANDB:
            self._client.log_model(model, name)

    def end_run(self):
        """End the current run."""
        if self._run is None:
            return

        if self.config.backend == TrackerBackend.MLFLOW:
            self._client.end_run()
        elif self.config.backend == TrackerBackend.WANDB:
            self._client.finish()
        elif self.config.backend == TrackerBackend.COMET:
            self._client.end()

        self._run = None

    def __enter__(self):
        self.start_run(self.config.run_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()


# =============================================================================
# Convenience Functions
# =============================================================================


def create_tracker(backend: str = "mlflow", **kwargs) -> ExperimentTracker:
    """
    Create an experiment tracker.

    Args:
        backend: "mlflow", "wandb", "comet", or "none"
        **kwargs: Additional config options

    Returns:
        ExperimentTracker instance
    """
    backend_map = {
        "mlflow": TrackerBackend.MLFLOW,
        "wandb": TrackerBackend.WANDB,
        "comet": TrackerBackend.COMET,
        "none": TrackerBackend.NONE,
    }

    config = TrackingConfig(backend=backend_map.get(backend.lower(), TrackerBackend.NONE), **kwargs)

    return ExperimentTracker(config)


def log_training_metrics(
    tracker: ExperimentTracker,
    epoch: int,
    metrics: Dict[str, float],
    lr: float,
):
    """Log standard training metrics."""
    tracker.log_metrics(
        {"epoch": epoch, "learning_rate": lr, **{f"train/{k}": v for k, v in metrics.items()}},
        step=epoch,
    )


def log_eval_metrics(
    tracker: ExperimentTracker,
    epoch: int,
    metrics: Dict[str, float],
):
    """Log evaluation metrics."""
    tracker.log_metrics(
        {"epoch": epoch, **{f"eval/{k}": v for k, v in metrics.items()}}, step=epoch
    )


__all__ = [
    "TrackerBackend",
    "TrackingConfig",
    "ExperimentTracker",
    "create_tracker",
    "log_training_metrics",
    "log_eval_metrics",
]
