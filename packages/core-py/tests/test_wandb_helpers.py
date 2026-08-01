"""Tests for domains.training.wandb_helpers: env flags, config flattening, API job tracker."""

import sys
import types
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from domains.infrastructure.config import get_config
from domains.training.wandb_helpers import (
    create_training_tracker_for_api_job,
    default_wandb_project,
    flatten_for_wandb_config,
    wandb_server_enabled_from_env,
    wandb_training_enabled_from_env,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("MAN_WANDB_TRAINING", raising=False)


@pytest.fixture
def tracking_config(monkeypatch):
    cfg = get_config()
    old = {
        "wandb_training_enabled": cfg.tracking.wandb_training_enabled,
        "wandb_project": cfg.tracking.wandb_project,
        "wandb_entity": cfg.tracking.wandb_entity,
        "wandb_api_key": cfg.tracking.wandb_api_key,
        "wandb_server_enabled": cfg.tracking.wandb_server_enabled,
    }
    yield cfg
    for key, value in old.items():
        setattr(cfg.tracking, key, value)


class TestWandbTrainingEnabled:
    def test_env_true(self, tracking_config, monkeypatch):
        tracking_config.tracking.wandb_training_enabled = False
        monkeypatch.setenv("MAN_WANDB_TRAINING", "1")
        assert wandb_training_enabled_from_env() is True

    def test_env_true_aliases(self, tracking_config, monkeypatch):
        for value in ("true", "yes"):
            monkeypatch.setenv("MAN_WANDB_TRAINING", value)
            assert wandb_training_enabled_from_env() is True

    def test_env_false_falls_back_to_config(self, tracking_config):
        tracking_config.tracking.wandb_training_enabled = True
        assert wandb_training_enabled_from_env() is True

    def test_env_empty_falls_back_to_config_false(self, tracking_config):
        tracking_config.tracking.wandb_training_enabled = False
        assert wandb_training_enabled_from_env() is False


class TestWandbServerEnabled:
    def test_config_true(self, tracking_config):
        tracking_config.tracking.wandb_server_enabled = True
        assert wandb_server_enabled_from_env() is True

    def test_config_false(self, tracking_config):
        tracking_config.tracking.wandb_server_enabled = False
        assert wandb_server_enabled_from_env() is False

    def test_env_does_not_override(self, tracking_config, monkeypatch):
        tracking_config.tracking.wandb_server_enabled = False
        monkeypatch.setenv("MAN_WANDB_SERVER", "1")
        assert wandb_server_enabled_from_env() is False


class TestDefaultWandbProject:
    def test_returns_config(self, tracking_config):
        tracking_config.tracking.wandb_project = "my-proj"
        assert default_wandb_project() == "my-proj"


class TestFlattenForWandbConfig:
    def test_none(self):
        assert flatten_for_wandb_config(None) == {}

    def test_flat_dict(self):
        assert flatten_for_wandb_config({"a": 1, "b": "x"}) == {"a": 1, "b": "x"}

    def test_nested_dict(self):
        result = flatten_for_wandb_config({"a": {"b": {"c": 3}}})
        assert result == {"a.b.c": 3}

    def test_dataclass(self):
        @dataclass
        class Inner:
            x: int = 1
            y: str = "z"

        @dataclass
        class Outer:
            name: str = "n"
            inner: Inner = field(default_factory=Inner)

        result = flatten_for_wandb_config(Outer())
        assert result == {"name": "n", "inner.x": 1, "inner.y": "z"}

    def test_list_truncated_to_string(self):
        result = flatten_for_wandb_config({"items": [1, 2, 3]})
        assert result == {"items": "[1, 2, 3]"}

    def test_none_value_preserved(self):
        assert flatten_for_wandb_config({"a": None}) == {"a": None}

    def test_primitives_truncation(self):
        result = flatten_for_wandb_config({"huge": ["x" * 5000]})
        assert len(result["huge"]) == 2000

    def test_nested_under_prefix(self):
        result = flatten_for_wandb_config({"job": {"id": "j1"}}, prefix="meta")
        assert result == {"meta.job.id": "j1"}

    def test_scalar_without_prefix(self):
        assert flatten_for_wandb_config(42) == {"value": 42}

    def test_scalar_with_prefix(self):
        assert flatten_for_wandb_config(42, prefix="lr") == {"lr": 42}

    def test_nested_dataclass_in_dict(self):
        @dataclass
        class Cfg:
            rate: float = 0.5

        result = flatten_for_wandb_config({"trainer": {"cfg": Cfg()}})
        assert result == {"trainer.cfg.rate": 0.5}

    def test_object_strified(self):
        result = flatten_for_wandb_config({"obj": object()})
        assert isinstance(result["obj"], str)


class FakeTrackingModule:
    """In-memory stand-in for domains.training.tracking to avoid wandb/network."""

    def __init__(self):
        self.runs = []
        self.TrackerBackend = types.SimpleNamespace(WANDB="wandb")
        self.TrackingConfig = MagicMock()
        self.ExperimentTracker = MagicMock()

    def reset(self):
        self.runs = []
        self.TrackingConfig = MagicMock()
        self.ExperimentTracker = MagicMock()


@pytest.fixture
def fake_tracking(monkeypatch):
    fake = FakeTrackingModule()
    monkeypatch.setitem(sys.modules, "domains.training.tracking", fake)
    return fake


class TestCreateTrainingTrackerForApiJob:
    def test_disabled_returns_none(self, tracking_config, monkeypatch):
        tracking_config.tracking.wandb_training_enabled = False
        assert create_training_tracker_for_api_job(
            job_id="j1", job_name="n", data_path="p", hyperparams={}
        ) is None

    def test_missing_tracking_module_returns_none(self, tracking_config, monkeypatch):
        tracking_config.tracking.wandb_training_enabled = True
        monkeypatch.setitem(sys.modules, "domains.training.tracking", None)
        assert create_training_tracker_for_api_job(
            job_id="j1", job_name="n", data_path="p", hyperparams={}
        ) is None

    def test_starts_run_and_logs_params(self, tracking_config, fake_tracking):
        tracking_config.tracking.wandb_training_enabled = True
        tracking_config.tracking.wandb_project = "proj"
        tracker = create_training_tracker_for_api_job(
            job_id="j1", job_name="run", data_path="/data/x", hyperparams={"lr": 1e-3}
        )
        assert tracker is not None
        fake_tracking.ExperimentTracker.assert_called_once()
        tracker.start_run.assert_called_once()
        tracker.log_params.assert_called_once()
        flat = tracker.log_params.call_args[0][0]
        assert flat["job.id"] == "j1"
        assert flat["job.name"] == "run"
        assert flat["data.path"] == "/data/x"
        assert flat["trainer.lr"] == 1e-3

    def test_run_name_is_job_id_and_name(self, tracking_config, fake_tracking):
        tracking_config.tracking.wandb_training_enabled = True
        create_training_tracker_for_api_job(
            job_id="abc123", job_name="my run", data_path="p", hyperparams={}
        )
        tracker = fake_tracking.ExperimentTracker.return_value
        tracker.start_run.assert_called_once_with(run_name="abc123_my run")

    def test_init_failure_returns_none(self, tracking_config, fake_tracking):
        tracking_config.tracking.wandb_training_enabled = True
        fake_tracking.ExperimentTracker.side_effect = RuntimeError("no wandb")
        assert create_training_tracker_for_api_job(
            job_id="j1", job_name="n", data_path="p", hyperparams={}
        ) is None

    def test_run_name_truncated(self, tracking_config, fake_tracking):
        tracking_config.tracking.wandb_training_enabled = True
        create_training_tracker_for_api_job(
            job_id="j" * 100, job_name="n" * 100, data_path="p", hyperparams={}
        )
        run_name = fake_tracking.ExperimentTracker.return_value.start_run.call_args[1]["run_name"]
        assert len(run_name) == 128
