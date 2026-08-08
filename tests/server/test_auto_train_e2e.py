"""
End-to-end smoke test for the auto-train pipeline.

Tests: start config → status → checkpoints → schema validation → cancel.
"""

import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.auto_train import router

app = FastAPI()
app.include_router(router)
client = TestClient(app, raise_server_exceptions=False)


class TestAutoTrainStart:
    """Test POST /auto-train/start — stores config for stream."""

    def test_start_requires_data(self):
        resp = client.post("/auto-train/start", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert "Provide" in data["message"]

    def test_start_with_source_text(self):
        resp = client.post("/auto-train/start", json={
            "source_text": "hello world " * 100,
            "epochs": 1,
            "batch_size": 4,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["epochs"] == 1

    def test_start_config_accessible_via_status(self):
        client.post("/auto-train/start", json={
            "source_text": "test training data",
            "epochs": 3,
            "learning_rate": 0.001,
        })
        resp = client.get("/auto-train/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("running") is True
        assert data["config"]["epochs"] == 3


class TestAutoTrainCheckpoints:
    """Test GET /auto-train/checkpoints."""

    def test_returns_wrapped_list(self):
        resp = client.get("/auto-train/checkpoints")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert isinstance(data["data"], list)

    def test_each_checkpoint_has_name(self):
        resp = client.get("/auto-train/checkpoints")
        ckpts = resp.json()["data"]
        for ckpt in ckpts:
            assert "name" in ckpt
            assert "download_url" in ckpt


class TestAutoTrainSchema:
    """Test StartRequest schema validation."""

    def test_invalid_epochs(self):
        resp = client.post("/auto-train/start", json={
            "source_text": "test",
            "epochs": -1,
        })
        assert resp.status_code == 422

    def test_invalid_learning_rate(self):
        resp = client.post("/auto-train/start", json={
            "source_text": "test",
            "learning_rate": 0,
        })
        assert resp.status_code == 422

    def test_valid_config(self):
        resp = client.post("/auto-train/start", json={
            "source_text": "hello world",
            "epochs": 5,
            "learning_rate": 0.001,
            "batch_size": 32,
            "soul_name": "test-soul",
            "algo": "bpe",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"


class TestAutoTrainNativeConfig:
    """POST /auto-train/start must forward the native SloNet architecture
    params the web frontend sends for method=native."""

    def _start_native(self, **overrides):
        body = {
            "source_text": "native training sample text " * 60,
            "epochs": 1,
            "learning_rate": 0.001,
            "batch_size": 8,
            "n_embed": 128,
            "n_layer": 4,
            "n_head": 4,
            "block_size": 128,
            "checkpoint_dir": "models/slonet-native",
        }
        body.update(overrides)
        return client.post("/auto-train/start", json=body)

    def test_native_params_forwarded_to_config(self):
        resp = self._start_native()
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["config"]["n_embed"] == 128
        assert data["config"]["n_layer"] == 4
        assert data["config"]["n_head"] == 4
        assert data["config"]["block_size"] == 128
        assert data["config"]["checkpoint_dir"] == "models/slonet-native"

    def test_native_params_visible_via_status(self):
        self._start_native(n_embed=256, n_layer=6, n_head=8, block_size=64)
        resp = client.get("/auto-train/status")
        assert resp.status_code == 200
        cfg = resp.json()["config"]
        assert cfg["n_embed"] == 256
        assert cfg["n_layer"] == 6
        assert cfg["n_head"] == 8
        assert cfg["block_size"] == 64

    def test_native_defaults_applied(self):
        resp = client.post("/auto-train/start", json={
            "source_text": "default native config " * 60,
            "epochs": 1,
        })
        assert resp.status_code == 200
        cfg = resp.json()["config"]
        assert cfg["n_embed"] == 128
        assert cfg["n_layer"] == 4
        assert cfg["n_head"] == 4
        assert cfg["block_size"] == 128

    def test_native_schema_validation(self):
        for bad in (
            {"n_embed": 2},
            {"n_head": 100},
            {"block_size": 4},
            {"n_layer": 0},
            {"dropout": 1.5},
        ):
            resp = self._start_native(**bad)
            assert resp.status_code == 422, bad


class TestAutoTrainCancel:
    """Test POST /auto-train/cancel."""

    def test_cancel_returns_ok(self):
        resp = client.post("/auto-train/cancel")
        assert resp.status_code in (200, 404, 409)


class TestAutoTrainStatus:
    """Test GET /auto-train/status."""

    def test_status_returns_config(self):
        resp = client.get("/auto-train/status")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert "running" in data
        assert "config" in data


@pytest.fixture(autouse=True)
def _reset_auto_train_state():
    """Reset the module router singleton + global events before each test."""
    import routers.auto_train as mod

    yield

    mod._auto_train_instance.state.running = False
    mod._auto_train_instance.state.config = {}
    mod._auto_train_cancel_event = None
    mod._auto_train_pause_event = None
    mod._complete_enqueued[0] = False


class TestAutoTrainStopPauseResume:
    """Deterministic state-machine branches: stop / pause / resume."""

    def test_stop_when_no_run(self):
        resp = client.post("/auto-train/stop")
        assert resp.status_code == 200
        assert resp.json() == {"status": "stopped"}

    def test_stop_with_active_cancel_event(self):
        import routers.auto_train as mod
        mod._auto_train_cancel_event = threading.Event()
        resp = client.post("/auto-train/stop")
        assert resp.json()["status"] == "cancelling"
        assert mod._auto_train_cancel_event.is_set()

    def test_pause_when_no_run(self):
        resp = client.post("/auto-train/pause")
        assert resp.json() == {"success": False, "message": "No active training to pause"}

    def test_pause_and_resume_cycle(self):
        import routers.auto_train as mod
        mod._auto_train_pause_event = threading.Event()
        pause = client.post("/auto-train/pause").json()
        assert pause["success"] is True
        again = client.post("/auto-train/pause").json()
        assert again["success"] is False
        resume = client.post("/auto-train/resume").json()
        assert resume["success"] is True
        assert mod._auto_train_pause_event.is_set() is False

    def test_resume_when_not_paused(self):
        import routers.auto_train as mod
        mod._auto_train_pause_event = threading.Event()
        resp = client.post("/auto-train/resume").json()
        assert resp["success"] is False


class TestAutoTrainStreamState:
    """Stream endpoint guards on missing state."""

    def test_stream_requires_start(self):
        resp = client.get("/auto-train/stream")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text
        assert "No training state" in body

    def test_from_sessions_stream_requires_start(self):
        resp = client.get("/auto-train/from-sessions/stream")
        assert resp.status_code == 200
        assert "No training state" in resp.text


class TestAutoTrainStartTurbo:
    """POST /auto-train/start-turbo — validation + missing-data branches."""

    def test_turbo_requires_data_path(self):
        resp = client.post("/auto-train/start-turbo", json={"epochs": 1})
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"
        assert "No data_path or dataset_id" in resp.json()["message"]

    def test_turbo_epochs_validation(self):
        resp = client.post("/auto-train/start-turbo", json={"epochs": 0, "data_path": "x.txt"})
        assert resp.status_code == 422

    def test_turbo_block_size_validation(self):
        resp = client.post("/auto-train/start-turbo", json={"epochs": 1, "data_path": "x.txt", "block_size": 2})
        assert resp.status_code == 422


class TestAutoTrainFromSessions:
    """POST /auto-train/from-sessions/start — state transition + guards."""

    def test_start_sets_config(self):
        resp = client.post("/auto-train/from-sessions/start", json={"epochs": 3})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["method"] == "from-sessions"
        assert data["epochs"] == 3

    def test_start_when_running_rejected(self):
        import routers.auto_train as mod
        mod._auto_train_instance.state.running = True
        resp = client.post("/auto-train/from-sessions/start", json={"epochs": 3})
        assert resp.status_code == 200
        assert "already" in resp.json()["message"].lower()

    def test_start_epochs_validation(self):
        resp = client.post("/auto-train/from-sessions/start", json={"epochs": 0})
        assert resp.status_code == 422

    def test_cancel_sends_signal(self):
        resp = client.get("/auto-train/from-sessions/cancel")
        assert resp.status_code == 200


class TestAutoTrainCheckpointValidation:
    """Checkpoint metadata / validation endpoints with no real model."""

    def test_download_invalid_name_400(self):
        resp = client.get("/auto-train/checkpoints/a..b/download")
        assert resp.status_code == 400

    def test_download_missing_404(self):
        resp = client.get("/auto-train/checkpoints/definitely_missing_ckpt/download")
        assert resp.status_code == 404

    def test_info_invalid_name_400(self):
        resp = client.get("/auto-train/checkpoints/bad%20name/info")
        assert resp.status_code == 400

    def test_info_missing_404(self):
        resp = client.get("/auto-train/checkpoints/definitely_missing_ckpt/info")
        assert resp.status_code == 404

    def test_delete_missing_returns_not_found(self):
        resp = client.delete("/auto-train/checkpoints/definitely_missing_ckpt")
        assert resp.status_code == 200
        assert resp.json()["message"] == "not_found"

    def test_load_missing_checkpoint(self):
        resp = client.post("/auto-train/checkpoints/definitely_missing_ckpt/load")
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"

    def test_export_mobile_missing_404(self):
        resp = client.get("/auto-train/checkpoints/definitely_missing_ckpt/export-mobile")
        assert resp.status_code == 404


class TestAutoTrainMetricsExport:
    """GET /auto-train/metrics/export — downloadable JSON."""

    def test_exports_json_with_structure(self):
        resp = client.get("/auto-train/metrics/export")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        data = resp.json()
        assert "exported_at" in data
        assert "total_checkpoints" in data
        assert "checkpoints" in data


class TestAutoTrainLog:
    """GET /auto-train/log — server buffer tail."""

    def test_returns_lines(self):
        resp = client.get("/auto-train/log")
        assert resp.status_code == 200
        body = resp.json()
        assert "lines" in body
        assert isinstance(body["lines"], list)
        assert "total" in body
