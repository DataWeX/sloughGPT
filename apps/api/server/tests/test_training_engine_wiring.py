"""Unit tests for TrainingEngine executor/export/turbo/outcome methods.

These methods were added to route system/models/dashboard training access
through the engine instead of domain.training._internal (Feature-Level
API Rule). Router-level behavior is covered by test_system_router,
test_models_router, and test_dashboard_router; this file pins the
engine envelope semantics (success flags, error codes, data shapes).
"""

from contextlib import nullcontext

import pytest

from domain.training.engine import TrainingEngine


@pytest.fixture()
def engine() -> TrainingEngine:
    return TrainingEngine()


class _FakeExecutor:
    max_workers = 3
    job_count = 1

    def __init__(self, status=None, summary=None, purged=0, cancelled=True):
        self._status = status
        self._summary = summary
        self._purged = purged
        self._cancelled = cancelled

    def active_count(self) -> int:
        return 1

    def list_jobs(self) -> list:
        return [{"id": "j1"}]

    def status(self, job_id: str):
        return self._status if job_id == "j1" else None

    def result_summary(self, job_id: str):
        return self._summary if job_id == "j1" else None

    def purge_completed(self, max_age_s: float = 3600.0) -> int:
        return self._purged

    def cancel(self, job_id: str) -> bool:
        return self._cancelled


class TestExecutorStatus:
    def test_uninitialized(self, engine, monkeypatch):
        from domain.training._internal import executor as executor_mod

        monkeypatch.setattr(executor_mod, "_instance", None)
        result = engine.executor_status()
        assert result.success
        assert result.data == {
            "initialized": False,
            "active_jobs": 0,
            "max_workers": 0,
            "total_tracked": 0,
            "jobs": [],
        }

    def test_initialized(self, engine, monkeypatch):
        from domain.training._internal import executor as executor_mod

        monkeypatch.setattr(executor_mod, "_instance", _FakeExecutor())
        result = engine.executor_status()
        assert result.success
        assert result.data == {
            "initialized": True,
            "active_jobs": 1,
            "max_workers": 3,
            "total_tracked": 1,
            "jobs": [{"id": "j1"}],
        }


class TestExecutorJob:
    def test_not_initialized(self, engine, monkeypatch):
        from domain.training._internal import executor as executor_mod

        monkeypatch.setattr(executor_mod, "_instance", None)
        result = engine.executor_job("j1")
        assert not result.success
        assert result.error == "executor not initialized"
        assert result.metadata["code"] == "E_INFRA_STARTUP"

    def test_not_found(self, engine, monkeypatch):
        from domain.training._internal import executor as executor_mod

        monkeypatch.setattr(executor_mod, "_instance", _FakeExecutor())
        result = engine.executor_job("missing")
        assert not result.success
        assert result.metadata["code"] == "E_NOT_FOUND"

    def test_found(self, engine, monkeypatch):
        from domain.training._internal import executor as executor_mod

        monkeypatch.setattr(
            executor_mod, "_instance", _FakeExecutor(status={"id": "j1", "state": "done"})
        )
        result = engine.executor_job("j1")
        assert result.success
        assert result.data == {"id": "j1", "state": "done"}


class TestExecutorJobResult:
    def test_not_completed(self, engine, monkeypatch):
        from domain.training._internal import executor as executor_mod

        monkeypatch.setattr(
            executor_mod, "_instance", _FakeExecutor(status={"id": "j1", "state": "running"})
        )
        result = engine.executor_job_result("j1")
        assert not result.success
        assert result.metadata["code"] == "E_DOMAIN"

    def test_completed(self, engine, monkeypatch):
        from domain.training._internal import executor as executor_mod

        monkeypatch.setattr(
            executor_mod,
            "_instance",
            _FakeExecutor(status={"id": "j1"}, summary={"shape": [2, 2]}),
        )
        result = engine.executor_job_result("j1")
        assert result.success
        assert result.data == {"shape": [2, 2]}


class TestExecutorControl:
    def test_purge_uninitialized(self, engine, monkeypatch):
        from domain.training._internal import executor as executor_mod

        monkeypatch.setattr(executor_mod, "_instance", None)
        result = engine.purge_executor_jobs()
        assert result.success
        assert result.data == {"purged": 0}

    def test_purge_delegates_max_age(self, engine, monkeypatch):
        from domain.training._internal import executor as executor_mod

        monkeypatch.setattr(executor_mod, "_instance", _FakeExecutor(purged=7))
        result = engine.purge_executor_jobs(max_age_s=60.0)
        assert result.success
        assert result.data == {"purged": 7}

    def test_cancel_uninitialized(self, engine, monkeypatch):
        from domain.training._internal import executor as executor_mod

        monkeypatch.setattr(executor_mod, "_instance", None)
        result = engine.cancel_executor_job("j1")
        assert result.success
        assert result.data == {"cancelled": False, "reason": "executor not initialized"}

    def test_cancel_delegates(self, engine, monkeypatch):
        from domain.training._internal import executor as executor_mod

        monkeypatch.setattr(executor_mod, "_instance", _FakeExecutor())
        result = engine.cancel_executor_job("j1")
        assert result.success
        assert result.data == {"cancelled": True}


class TestExport:
    def test_list_export_formats(self, engine, monkeypatch):
        import domain.training._internal.export as export_mod

        monkeypatch.setattr(export_mod, "list_export_formats", lambda: ["gguf", "safetensors"])
        result = engine.list_export_formats()
        assert result.success
        assert result.data == ["gguf", "safetensors"]

    def test_export_model_delegates(self, engine, monkeypatch):
        import domain.training._internal.export as export_mod

        captured = {}

        def fake_do_export(config, model, tokenizer):
            captured["format"] = config.format
            captured["output"] = config.output_path
            captured["metadata"] = config.metadata
            return ["model.gguf"]

        monkeypatch.setattr(export_mod, "export_model", fake_do_export)
        result = engine.export_model(
            model=object(),
            tokenizer=object(),
            output_path="/tmp/out",
            format="gguf",
            metadata={"model_type": "tiny"},
        )
        assert result.success
        assert result.data == ["model.gguf"]
        assert captured == {"format": "gguf", "output": "/tmp/out", "metadata": {"model_type": "tiny"}}


class TestTurboAndOutcomes:
    def test_turbo_state_snapshot(self, engine, monkeypatch):
        import domain.training._internal.service as service_mod

        state = {"status": "running", "epoch": 2}
        monkeypatch.setattr(service_mod, "get_turbo_lock", nullcontext)
        monkeypatch.setattr(service_mod, "get_turbo_state", lambda: state)
        result = engine.turbo_state()
        assert result.success
        assert result.data == state
        result.data["status"] = "mutated"
        assert state["status"] == "running"

    def test_outcome_stats(self, engine, monkeypatch):
        import domain.training._internal.outcome_tracker as tracker_mod

        class _Tracker:
            def get_stats(self):
                return {"total_runs": 5}

        monkeypatch.setattr(tracker_mod, "TrainingOutcomeTracker", _Tracker)
        result = engine.outcome_stats()
        assert result.success
        assert result.data == {"total_runs": 5}
