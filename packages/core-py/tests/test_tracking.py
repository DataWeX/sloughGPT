"""Tests for domains/training/tracking.py."""

import sys

import pytest

from domains.training.tracking import (
    ExperimentTracker,
    TrackingConfig,
    TrackerBackend,
    create_tracker,
    log_eval_metrics,
    log_training_metrics,
)


class FakeTracking:
    """Minimal fake config.tracking surface used by TrackingConfig init."""

    wandb_api_key = ""
    mlflow_tracking_uri = ""
    wandb_mode = ""
    wandb_dir = ""
    wandb_project = "sloughgpt"


@pytest.fixture(autouse=True)
def fake_config(monkeypatch):
    cfg = type(
        "FakeConfig",
        (),
        {"tracking": FakeTracking()},
    )
    monkeypatch.setattr(
        "domains.infrastructure.config.get_config", lambda: cfg
    )
    monkeypatch.setattr(
        "domains.training.tracking.get_config", lambda: cfg
    )


class TestTrackerBackend:
    def test_values(self):
        assert TrackerBackend.MLFLOW.value == "mlflow"
        assert TrackerBackend.WANDB.value == "wandb"
        assert TrackerBackend.COMET.value == "comet"
        assert TrackerBackend.NONE.value == "none"


class TestTrackingConfig:
    def test_defaults(self):
        cfg = TrackingConfig()
        assert cfg.backend == TrackerBackend.NONE
        assert cfg.experiment_name == "sloughgpt_experiment"
        assert cfg.project == "sloughgpt"

    def test_wandb_api_key_fallback(self):
        cfg = TrackingConfig(backend=TrackerBackend.WANDB)
        assert cfg.api_key == "" or cfg.api_key is None

    def test_mlflow_uri_fallback(self):
        cfg = TrackingConfig(backend=TrackerBackend.MLFLOW)
        assert cfg.tracking_uri is None or cfg.tracking_uri == ""

    def test_custom_fields(self):
        cfg = TrackingConfig(
            backend=TrackerBackend.WANDB,
            run_name="run1",
            entity="my-entity",
            tags=["a", "b"],
        )
        assert cfg.run_name == "run1"
        assert cfg.entity == "my-entity"
        assert cfg.tags == ["a", "b"]


class TestNoneBackend:
    def test_no_crash(self):
        tracker = ExperimentTracker(TrackingConfig())
        assert tracker._client is None

    def test_logging_noops_without_run(self):
        tracker = ExperimentTracker(TrackingConfig())
        tracker.log_metric("x", 1.0)
        tracker.log_metrics({"a": 1.0})
        tracker.log_param("p", 1)
        tracker.log_artifact("/tmp/f")
        tracker.log_model(object())
        tracker.end_run()

    def test_context_manager(self):
        tracker = ExperimentTracker(TrackingConfig())
        with tracker:
            pass


class TestMLflowBackend:
    def _fake_mlflow(self, monkeypatch):
        calls = []

        class FakeMLflow:
            @staticmethod
            def set_tracking_uri(uri):
                calls.append(("uri", uri))

            @staticmethod
            def set_experiment(name):
                calls.append(("exp", name))

            @staticmethod
            def start_run(run_name=None):
                calls.append(("start", run_name))
                return object()

            @staticmethod
            def log_metric(name, value, step=None):
                calls.append(("metric", name, value, step))

            @staticmethod
            def log_param(name, value):
                calls.append(("param", name, value))

            @staticmethod
            def end_run():
                calls.append(("end",))

        monkeypatch.setitem(sys.modules, "mlflow", FakeMLflow)
        return calls

    def test_init(self, monkeypatch):
        calls = self._fake_mlflow(monkeypatch)
        tracker = ExperimentTracker(
            TrackingConfig(backend=TrackerBackend.MLFLOW, tracking_uri="sqlite:///x")
        )
        assert tracker._client is not None
        assert ("uri", "sqlite:///x") in calls
        assert ("exp", "sloughgpt_experiment") in calls

    def test_metrics_and_params(self, monkeypatch):
        calls = self._fake_mlflow(monkeypatch)
        tracker = ExperimentTracker(
            TrackingConfig(backend=TrackerBackend.MLFLOW)
        )
        tracker.start_run(run_name="r")
        assert ("start", "r") in calls
        tracker.log_metric("loss", 0.5, step=2)
        tracker.log_param("lr", 1e-3)
        assert ("metric", "loss", 0.5, 2) in calls
        assert ("param", "lr", 1e-3) in calls

    def test_end_run(self, monkeypatch):
        calls = self._fake_mlflow(monkeypatch)
        tracker = ExperimentTracker(
            TrackingConfig(backend=TrackerBackend.MLFLOW)
        )
        tracker.start_run()
        tracker.end_run()
        assert ("end",) in calls
        assert tracker._run is None


class TestWandbBackend:
    def test_init(self, monkeypatch):
        init_calls = {}

        class FakeWandb:
            @staticmethod
            def init(**kwargs):
                init_calls.update(kwargs)

            @staticmethod
            def log(data, step=None):
                pass

            @staticmethod
            def finish():
                pass

        class FakeWandbHelpers:
            @staticmethod
            def default_wandb_project():
                return "default-project"

        monkeypatch.setitem(sys.modules, "wandb", FakeWandb)
        monkeypatch.setitem(
            sys.modules,
            "domains.training.wandb_helpers",
            FakeWandbHelpers,
        )
        tracker = ExperimentTracker(
            TrackingConfig(
                backend=TrackerBackend.WANDB,
                run_name="myrun",
                entity="me",
                tags=["tag1"],
            )
        )
        assert tracker._client is FakeWandb
        assert init_calls["name"] == "myrun"
        assert init_calls["entity"] == "me"
        assert init_calls["tags"] == ["tag1"]

    def test_metrics_logs(self, monkeypatch):
        logged = []

        class FakeWandb:
            @staticmethod
            def init(**kwargs):
                pass

            @staticmethod
            def log(data, step=None):
                logged.append((data, step))

            @staticmethod
            def finish():
                pass

        monkeypatch.setitem(sys.modules, "wandb", FakeWandb)
        monkeypatch.setitem(
            sys.modules,
            "domains.training.wandb_helpers",
            type("H", (), {"default_wandb_project": staticmethod(lambda: "p")}),
        )
        tracker = ExperimentTracker(TrackingConfig(backend=TrackerBackend.WANDB))
        tracker.start_run()
        tracker.log_metric("loss", 0.5, step=1)
        tracker.log_metrics({"acc": 0.9}, step=2)
        assert logged == [({"loss": 0.5}, 1), ({"acc": 0.9}, 2)]

    def test_end_run_calls_finish(self, monkeypatch):
        finished = []

        class FakeWandb:
            @staticmethod
            def init(**kwargs):
                pass

            @staticmethod
            def finish():
                finished.append(True)

        monkeypatch.setitem(sys.modules, "wandb", FakeWandb)
        monkeypatch.setitem(
            sys.modules,
            "domains.training.wandb_helpers",
            type("H", (), {"default_wandb_project": staticmethod(lambda: "p")}),
        )
        tracker = ExperimentTracker(TrackingConfig(backend=TrackerBackend.WANDB))
        tracker.start_run()
        tracker.end_run()
        assert finished == [True]


class TestCometBackend:
    def test_init_and_metric(self, monkeypatch):
        logged = []

        class FakeExperiment:
            def __init__(self, project_name=None, api_key=None):
                self.project_name = project_name
                self.api_key = api_key

            def log_metric(self, name, value, step=None):
                logged.append((name, value, step))

            def end(self):
                pass

        monkeypatch.setitem(
            sys.modules, "comet_ml", type("C", (), {"Experiment": FakeExperiment})
        )
        tracker = ExperimentTracker(
            TrackingConfig(
                backend=TrackerBackend.COMET,
                project="proj",
                api_key="key",
            )
        )
        assert tracker._client is not None
        tracker.start_run()
        tracker.log_metric("x", 1.0)
        assert logged == [("x", 1.0, None)]


class TestMissingBackendDeps:
    def test_mlflow_missing(self, monkeypatch):
        def no_mlflow(name, *a, **k):
            if name == "mlflow":
                raise ImportError("no mlflow")
            return __import__(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", no_mlflow)
        tracker = ExperimentTracker(
            TrackingConfig(backend=TrackerBackend.MLFLOW)
        )
        assert tracker._client is None

    def test_wandb_missing(self, monkeypatch):
        def no_wandb(name, *a, **k):
            if name == "wandb":
                raise ImportError("no wandb")
            return __import__(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", no_wandb)
        tracker = ExperimentTracker(
            TrackingConfig(backend=TrackerBackend.WANDB)
        )
        assert tracker._client is None


class TestCreateTracker:
    def test_known_backend(self):
        tracker = create_tracker("none")
        assert tracker.config.backend == TrackerBackend.NONE

    def test_unknown_backend_defaults_none(self):
        tracker = create_tracker("bogus")
        assert tracker.config.backend == TrackerBackend.NONE

    def test_case_insensitive(self):
        tracker = create_tracker("MLFLOW")
        assert tracker.config.backend == TrackerBackend.MLFLOW


class TestLogHelpers:
    def test_log_training_metrics(self, monkeypatch):
        recorded = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                recorded.append((metrics, step))

        log_training_metrics(FakeTracker(), 3, {"loss": 0.5}, lr=0.001)
        metrics, step = recorded[0]
        assert metrics["epoch"] == 3
        assert metrics["learning_rate"] == 0.001
        assert metrics["train/loss"] == 0.5
        assert step == 3

    def test_log_eval_metrics(self, monkeypatch):
        recorded = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                recorded.append((metrics, step))

        log_eval_metrics(FakeTracker(), 2, {"acc": 0.8})
        metrics, step = recorded[0]
        assert metrics["epoch"] == 2
        assert metrics["eval/acc"] == 0.8
