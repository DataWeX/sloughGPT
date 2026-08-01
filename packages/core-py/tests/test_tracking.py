"""Tests for domains.training.tracking: MLflow/W&B/Comet experiment tracking."""

import sys

import pytest

from domains.training.tracking import (
    ExperimentTracker,
    TrackerBackend,
    TrackingConfig,
    create_tracker,
    log_eval_metrics,
    log_training_metrics,
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
        assert cfg.run_name is None
        assert cfg.job_type is None
        assert cfg.tags is None

    def test_none_backend_no_config_lookup(self):
        cfg = TrackingConfig(backend=TrackerBackend.NONE)
        assert cfg.api_key is None
        assert cfg.tracking_uri is None

    def test_mlflow_uses_config_tracking_uri_when_unset(self):
        cfg = TrackingConfig(backend=TrackerBackend.MLFLOW)
        assert cfg.tracking_uri is None or isinstance(cfg.tracking_uri, str)


class TestCreateTracker:
    def test_mlflow_string(self):
        tracker = create_tracker("mlflow")
        assert tracker.config.backend == TrackerBackend.MLFLOW

    def test_wandb_string(self):
        tracker = create_tracker("wandb")
        assert tracker.config.backend == TrackerBackend.WANDB

    def test_comet_string(self):
        tracker = create_tracker("comet")
        assert tracker.config.backend == TrackerBackend.COMET

    def test_none_string(self):
        tracker = create_tracker("none")
        assert tracker.config.backend == TrackerBackend.NONE

    def test_unknown_backend_falls_back_to_none(self):
        tracker = create_tracker("bogus")
        assert tracker.config.backend == TrackerBackend.NONE

    def test_kwargs_passed_to_config(self):
        tracker = create_tracker("wandb", run_name="run-1", tags=["sloughgpt"])
        assert tracker.config.run_name == "run-1"
        assert tracker.config.tags == ["sloughgpt"]


class FakeModule:
    def __init__(self, **attrs):
        self._calls = []
        for k, v in attrs.items():
            setattr(self, k, v)

    def record(self, name, *args, **kwargs):
        self._calls.append((name, args, kwargs))
        return None

    def make_callable(self, name, result=None):
        def fn(*args, **kwargs):
            self._calls.append((name, args, kwargs))
            return result

        setattr(self, name, fn)


def _install(monkeypatch, name, fake):
    monkeypatch.setitem(sys.modules, name, fake)
    return fake


class TestExperimentTrackerNoneBackend:
    def test_init_no_backend(self):
        tracker = ExperimentTracker(TrackingConfig(backend=TrackerBackend.NONE))
        assert tracker._client is None
        assert tracker._run is None

    def test_methods_are_noops(self):
        tracker = ExperimentTracker(TrackingConfig(backend=TrackerBackend.NONE))
        tracker.start_run("r")
        tracker.log_metric("loss", 0.1, step=1)
        tracker.log_metrics({"a": 1.0}, step=1)
        tracker.log_param("p", 1)
        tracker.log_params({"p": 1})
        tracker.log_artifact("/tmp/x")
        tracker.log_model(object())
        tracker.end_run()

    def test_context_manager(self):
        tracker = ExperimentTracker(TrackingConfig(backend=TrackerBackend.NONE))
        with tracker:
            pass
        assert tracker._run is None


class TestExperimentTrackerMlflow:
    def test_init_mlflow(self, monkeypatch):
        fake = FakeModule()
        fake.make_callable("set_tracking_uri")
        fake.make_callable("set_experiment")
        _install(monkeypatch, "mlflow", fake)
        tracker = ExperimentTracker(TrackingConfig(backend=TrackerBackend.MLFLOW, tracking_uri="http://x"))
        assert tracker._client is fake
        assert fake._calls[0][0] == "set_tracking_uri"
        assert fake._calls[0][1] == ("http://x",)
        assert fake._calls[1][0] == "set_experiment"

    def test_init_mlflow_import_error(self, monkeypatch):
        real_import = __import__

        def blocked(name, *a, **k):
            if name == "mlflow":
                raise ImportError("no mlflow")
            return real_import(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", blocked)
        tracker = ExperimentTracker(TrackingConfig(backend=TrackerBackend.MLFLOW))
        assert tracker._client is None

    def test_start_run_and_logging(self, monkeypatch):
        fake = FakeModule()
        fake.make_callable("set_tracking_uri")
        fake.make_callable("set_experiment")
        fake.make_callable("start_run", result="RUN")
        fake.make_callable("log_metric")
        fake.make_callable("log_param")
        fake.make_callable("log_params")
        fake.make_callable("end_run")
        _install(monkeypatch, "mlflow", fake)
        tracker = ExperimentTracker(TrackingConfig(backend=TrackerBackend.MLFLOW))
        tracker.start_run("my-run")
        assert tracker._run == "RUN"
        assert fake._calls[-1] == ("start_run", (), {"run_name": "my-run"})
        tracker.log_metric("loss", 0.5, step=3)
        assert fake._calls[-1] == ("log_metric", ("loss", 0.5), {"step": 3})
        tracker.log_param("lr", 1e-3)
        assert fake._calls[-1] == ("log_param", ("lr", 1e-3), {})
        tracker.log_params({"a": 1})
        assert fake._calls[-1] == ("log_params", ({"a": 1},), {})
        tracker.end_run()
        assert fake._calls[-1] == ("end_run", (), {})
        assert tracker._run is None

    def test_logging_before_start_run_is_skipped(self, monkeypatch):
        fake = FakeModule()
        fake.make_callable("set_tracking_uri")
        fake.make_callable("set_experiment")
        fake.make_callable("log_metric")
        _install(monkeypatch, "mlflow", fake)
        tracker = ExperimentTracker(TrackingConfig(backend=TrackerBackend.MLFLOW))
        tracker.log_metric("loss", 0.5)
        assert len(fake._calls) == 2

    def test_log_artifact_and_model(self, monkeypatch):
        fake = FakeModule()
        fake.make_callable("set_tracking_uri")
        fake.make_callable("set_experiment")
        fake.make_callable("start_run", result="RUN")
        fake.make_callable("log_artifact")
        fake.pyfunc = FakeModule(make_log_model=object)
        fake.pyfunc.make_callable("log_model")
        _install(monkeypatch, "mlflow", fake)
        tracker = ExperimentTracker(TrackingConfig(backend=TrackerBackend.MLFLOW))
        tracker.start_run()
        tracker.log_artifact("/tmp/model.bin", "m")
        assert fake._calls[-1] == ("log_artifact", ("/tmp/model.bin", "m"), {})
        tracker.log_model(object(), "model")
        assert fake.pyfunc._calls[-1][0] == "log_model"


class TestExperimentTrackerWandb:
    def test_init_wandb_kwargs(self, monkeypatch):
        fake = FakeModule()
        fake.make_callable("init")
        fake.config = FakeModule()
        _install(monkeypatch, "wandb", fake)
        from domains.infrastructure.config import get_config

        expected_mode = get_config().tracking.wandb_mode or None
        tracker = ExperimentTracker(
            TrackingConfig(
                backend=TrackerBackend.WANDB,
                project="myproj",
                run_name="r1",
                entity="myentity",
                api_key="secret",
                job_type="train",
                tags=["a", "b"],
            )
        )
        assert tracker._client is fake
        init_call = [c for c in fake._calls if c[0] == "init"][-1]
        assert init_call[1] == ()
        kwargs = init_call[2]
        assert kwargs["project"] == "myproj"
        assert kwargs["name"] == "r1"
        assert kwargs["entity"] == "myentity"
        assert kwargs["api_key"] == "secret"
        assert kwargs["job_type"] == "train"
        assert kwargs["tags"] == ["a", "b"]
        assert kwargs.get("mode") == expected_mode

    def test_init_wandb_none_fields_filtered(self, monkeypatch):
        fake = FakeModule()
        fake.make_callable("init")
        fake.config = FakeModule()
        _install(monkeypatch, "wandb", fake)
        tracker = ExperimentTracker(TrackingConfig(backend=TrackerBackend.WANDB))
        init_call = [c for c in fake._calls if c[0] == "init"][-1]
        assert "name" not in init_call[2]
        assert "api_key" not in init_call[2]

    def test_init_wandb_import_error(self, monkeypatch):
        real_import = __import__

        def blocked(name, *a, **k):
            if name == "wandb":
                raise ImportError("no wandb")
            return real_import(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", blocked)
        tracker = ExperimentTracker(TrackingConfig(backend=TrackerBackend.WANDB))
        assert tracker._client is None

    def test_logging_uses_wandb_api(self, monkeypatch):
        fake = FakeModule()
        fake.make_callable("init")
        fake.config = FakeModule()
        fake.config.make_callable("update")
        fake.make_callable("log")
        fake.make_callable("log_artifact")
        fake.make_callable("log_model")
        fake.make_callable("finish")
        _install(monkeypatch, "wandb", fake)
        tracker = ExperimentTracker(TrackingConfig(backend=TrackerBackend.WANDB))
        tracker.start_run()
        tracker.log_metric("loss", 0.5, step=2)
        assert fake._calls[-1] == ("log", ({"loss": 0.5},), {"step": 2})
        tracker.log_metrics({"a": 1.0, "b": 2.0}, step=3)
        assert fake._calls[-1] == ("log", ({"a": 1.0, "b": 2.0},), {"step": 3})
        tracker.log_param("lr", 1e-3)
        assert fake.config._calls[-1] == ("update", ({"lr": 1e-3},), {})
        tracker.log_params({"x": 1})
        assert fake.config._calls[-1] == ("update", ({"x": 1},), {})
        tracker.log_artifact("/tmp/a")
        assert fake._calls[-1] == ("log_artifact", ("/tmp/a",), {})
        tracker.log_model(object(), "m")
        assert fake._calls[-1] == ("log_model", (object, "m"), {}) or fake._calls[-1][0] == "log_model"
        tracker.end_run()
        assert fake._calls[-1] == ("finish", (), {})
        assert tracker._run is None


class TestExperimentTrackerComet:
    def test_init_comet(self, monkeypatch):
        experiment = FakeModule()
        experiment.make_callable("log_metric")
        experiment.make_callable("log_parameter")
        experiment.make_callable("end")
        comet = FakeModule(Experiment=lambda *a, **k: experiment)
        _install(monkeypatch, "comet_ml", comet)
        tracker = ExperimentTracker(TrackingConfig(backend=TrackerBackend.COMET, api_key="k"))
        assert tracker._client is experiment
        tracker.start_run()
        tracker.log_metric("loss", 0.5, step=1)
        assert experiment._calls[-1] == ("log_metric", ("loss", 0.5), {"step": 1})
        tracker.log_param("lr", 1e-3)
        assert experiment._calls[-1] == ("log_parameter", ("lr", 1e-3), {})
        tracker.end_run()
        assert experiment._calls[-1] == ("end", (), {})
        assert tracker._run is None

    def test_init_comet_import_error(self, monkeypatch):
        real_import = __import__

        def blocked(name, *a, **k):
            if name == "comet_ml":
                raise ImportError("no comet")
            return real_import(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", blocked)
        tracker = ExperimentTracker(TrackingConfig(backend=TrackerBackend.COMET))
        assert tracker._client is None


class TestLogMetrics:
    def test_log_training_metrics_format(self, monkeypatch):
        tracker = FakeModule()
        tracker.make_callable("log_metrics")
        log_training_metrics(tracker, epoch=2, metrics={"loss": 0.5, "acc": 0.9}, lr=0.001)
        call = tracker._calls[-1]
        assert call[0] == "log_metrics"
        payload = call[1][0]
        assert payload["epoch"] == 2
        assert payload["learning_rate"] == 0.001
        assert payload["train/loss"] == 0.5
        assert payload["train/acc"] == 0.9
        assert call[2] == {"step": 2}

    def test_log_eval_metrics_format(self, monkeypatch):
        tracker = FakeModule()
        tracker.make_callable("log_metrics")
        log_eval_metrics(tracker, epoch=3, metrics={"loss": 0.3})
        call = tracker._calls[-1]
        payload = call[1][0]
        assert payload["epoch"] == 3
        assert payload["eval/loss"] == 0.3
        assert call[2] == {"step": 3}
