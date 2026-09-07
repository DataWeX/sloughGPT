"""Tests for training.tracking — experiment tracking."""

from __future__ import annotations

import pytest
from domains.training.tracking import (
    TrackerBackend,
    TrackingConfig,
    ExperimentTracker,
)


# ── TrackerBackend ──────────────────────────────────────────────────────────


class TestTrackerBackend:

    def test_values(self):
        assert TrackerBackend.MLFLOW.value == "mlflow"
        assert TrackerBackend.WANDB.value == "wandb"
        assert TrackerBackend.COMET.value == "comet"
        assert TrackerBackend.NONE.value == "none"


# ── TrackingConfig ──────────────────────────────────────────────────────────


class TestTrackingConfig:

    def test_default(self):
        config = TrackingConfig()
        assert config.backend == TrackerBackend.NONE
        assert config.experiment_name == "sloughgpt_experiment"

    def test_custom(self):
        config = TrackingConfig(backend=TrackerBackend.MLFLOW, experiment_name="test")
        assert config.backend == TrackerBackend.MLFLOW
        assert config.experiment_name == "test"


# ── ExperimentTracker ──────────────────────────────────────────────────────


class TestExperimentTracker:

    def test_init_none(self):
        config = TrackingConfig(backend=TrackerBackend.NONE)
        tracker = ExperimentTracker(config)
        assert tracker._client is None

    def test_log_metric(self):
        config = TrackingConfig(backend=TrackerBackend.NONE)
        tracker = ExperimentTracker(config)
        tracker.log_metric("loss", 0.5)

    def test_log_metrics(self):
        config = TrackingConfig(backend=TrackerBackend.NONE)
        tracker = ExperimentTracker(config)
        tracker.log_metrics({"loss": 0.5, "acc": 0.9})

    def test_log_param(self):
        config = TrackingConfig(backend=TrackerBackend.NONE)
        tracker = ExperimentTracker(config)
        tracker.log_param("lr", 0.001)

    def test_log_params(self):
        config = TrackingConfig(backend=TrackerBackend.NONE)
        tracker = ExperimentTracker(config)
        tracker.log_params({"lr": 0.001, "batch_size": 32})

    def test_log_artifact(self):
        config = TrackingConfig(backend=TrackerBackend.NONE)
        tracker = ExperimentTracker(config)
        tracker.log_artifact("model.pt")

    def test_start_run(self):
        config = TrackingConfig(backend=TrackerBackend.NONE)
        tracker = ExperimentTracker(config)
        tracker.start_run()

    def test_end_run(self):
        config = TrackingConfig(backend=TrackerBackend.NONE)
        tracker = ExperimentTracker(config)
        tracker.end_run()
