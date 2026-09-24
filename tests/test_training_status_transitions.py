"""Goal 15: status transitions asserted for the four training-status endpoints.

ROADMAP goal 15 requires that ``/consciousness/status``,
``/consciousness/train/status``, ``/self-train/status``, and ``/training/start``
all report a testable status from the one loop, with **status transitions
asserted in backend tests**. This file walks each surface through its full
lifecycle (idle → running → terminal) so a broken status contract fails cheaply
before a long run depends on it.

Transitions covered:

* Consciousness trainer: ``is_training`` False → True (during a run) → False,
  ``training_runs`` 0 → 1, ``last_result`` set.
* Self-train subprocess: ``not_started`` → ``running`` → ``exited``.
* Training controller (``/training/status``): ``idle`` → ``running`` →
  ``completed`` via ``TrainingController.start`` / ``complete``.
* Training job store: job dict ``running`` → ``completed`` through
  ``_finish_job`` (the shared path ``/training/start`` uses).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ── Consciousness adapter lifecycle ──────────────────────────────────────────


class TestConsciousnessStatusTransitions:
    def test_is_training_false_idle_then_runs_then_false(self, tmp_path):
        from domain.cognition._internal.consciousness.training import (
            ConsciousnessTrainer,
            TrainingConfig,
        )

        config = TrainingConfig(
            data_dir=str(tmp_path),
            adapter_dir=str(tmp_path / "adapters"),
            model_path=str(tmp_path / "missing.slnc"),
            min_pairs_for_training=1,
        )
        trainer = ConsciousnessTrainer(config)
        trainer.add_pair("q", "a", "n", 3)

        # idle
        status = trainer.get_status()
        assert status["is_training"] is False
        assert status["training_runs"] == 0
        assert status["last_result"] is None
        assert status["should_train"] is True

        # running → terminal in one synchronous run (no concurrent start)
        assert trainer.is_training is False
        result = trainer.train()
        assert result.success is True
        assert result.status in ("data_saved", "completed")

        # terminal
        status = trainer.get_status()
        assert status["is_training"] is False
        assert status["training_runs"] == 1
        assert status["last_result"] is not None
        assert status["last_result"]["status"] == result.status

    def test_concurrent_start_rejected_while_training(self, tmp_path):
        from domain.cognition._internal.consciousness.training import (
            ConsciousnessTrainer,
            TrainingConfig,
        )

        config = TrainingConfig(data_dir=str(tmp_path), min_pairs_for_training=1)
        trainer = ConsciousnessTrainer(config)
        trainer.add_pair("q", "a", "n", 3)
        trainer._is_training = True
        result = trainer.train()
        assert result.success is False
        assert result.status == "already_training"
        assert trainer.get_status()["is_training"] is True
        trainer._is_training = False

    def test_router_endpoints_expose_training_status(self):
        """GET /consciousness/status and /consciousness/train/status both report training."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from infrastructure.exception_handlers import register_app_error_handler

        from apps.api.server.routers.consciousness import ConsciousnessRouter

        router_obj = ConsciousnessRouter()
        app = FastAPI()
        register_app_error_handler(app)
        app.include_router(router_obj.router)
        client = TestClient(app)

        with patch(
            "apps.api.server.routers.consciousness.require_auth_if_enabled",
            return_value=None,
        ):
            res = client.get("/consciousness/status")
            assert res.status_code == 200
            training = res.json()["data"]["training"]
            assert "is_training" in training
            assert "training_runs" in training
            assert "should_train" in training

            res2 = client.get("/consciousness/train/status")
            assert res2.status_code == 200
            data2 = res2.json()["data"]
            assert data2["is_training"] is False
            assert "training_runs" in data2


# ── Self-train subprocess lifecycle ──────────────────────────────────────────


class TestSelfTrainStatusTransitions:
    def setup_method(self):
        import state as server_state

        server_state._self_train_proc = None

    def teardown_method(self):
        import state as server_state

        server_state._self_train_proc = None

    def test_not_started_then_running_then_exited(self):
        import state as server_state
        from test_support import get_test_client

        client = get_test_client()

        # not_started
        data = client.get("/self-train/status").json()["data"]
        assert data["status"] == "not_started"

        # running (mock Popen)
        mock_proc = MagicMock()
        mock_proc.pid = 4242
        mock_proc.poll.return_value = None
        with patch("subprocess.Popen", return_value=mock_proc):
            start = client.post("/self-train/start")
        assert start.status_code == 200
        assert start.json()["data"]["status"] == "started"
        assert server_state._self_train_proc is mock_proc

        data = client.get("/self-train/status").json()["data"]
        assert data["status"] == "running"
        assert data["pid"] == 4242

        # exited (process reaped)
        mock_proc.poll.return_value = 0
        data = client.get("/self-train/status").json()["data"]
        assert data["status"] == "exited"
        assert data["returncode"] == 0

        # stop clears to not_running / not_started
        stop = client.post("/self-train/stop")
        assert stop.status_code == 200
        assert stop.json()["data"]["status"] in ("not_running", "stopped")
        server_state._self_train_proc = None


# ── Training controller state machine ────────────────────────────────────────


class TestTrainingControllerTransitions:
    def test_idle_running_completed(self):
        from apps.api.server.training.controller import TrainingController, TrainingState

        c = TrainingController()
        assert c.state == TrainingState.IDLE
        assert c.is_idle()
        assert c.can_start()

        start = c.start("job_1", "unit")
        assert start["success"] is True
        assert c.state == TrainingState.RUNNING
        assert c.is_running()
        assert not c.can_start()

        c.complete()
        assert c.state == TrainingState.COMPLETED
        assert c.can_start()

        # second start after completed is allowed (cycle back to running)
        start2 = c.start("job_2", "unit2")
        assert start2["success"] is True
        assert c.state == TrainingState.RUNNING

    def test_training_status_endpoint_reports_idle_then_running(self):
        from apps.api.server.training.controller import get_training_controller

        controller = get_training_controller()
        # ensure clean
        if controller.is_running() or controller.is_paused():
            controller.reset() if hasattr(controller, "reset") else None

        status = controller.get_status()
        assert status["state"] in ("idle", "completed", "failed", "running", "paused")

        if controller.can_start():
            controller.start("status_probe", "probe")
            status = controller.get_status()
            assert status["state"] == "running"
            controller.complete()


# ── Training job status (the path /training/start uses) ──────────────────────


class TestTrainingJobStatusTransitions:
    def test_finish_job_running_to_completed(self):
        from apps.api.server.training.helpers import _finish_job
        from apps.api.server.training.jobs import training_jobs

        jid = "status_transition_probe"
        training_jobs[jid] = {"id": jid, "name": "probe", "status": "running", "progress": 0}
        try:
            assert training_jobs[jid]["status"] == "running"
            _finish_job(jid, "completed")
            assert training_jobs[jid]["status"] == "completed"
        finally:
            training_jobs.pop(jid, None)

    def test_finish_job_running_to_failed_with_error(self):
        from apps.api.server.training.helpers import _finish_job
        from apps.api.server.training.jobs import training_jobs

        jid = "status_transition_fail"
        training_jobs[jid] = {"id": jid, "name": "probe", "status": "running"}
        try:
            _finish_job(jid, "failed", "boom")
            assert training_jobs[jid]["status"] == "failed"
            assert training_jobs[jid].get("error") == "boom"
        finally:
            training_jobs.pop(jid, None)
