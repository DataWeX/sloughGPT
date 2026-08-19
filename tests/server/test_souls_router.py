"""
Tests for the souls router — list, current, switch, weights, snapshots, stats.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from infrastructure.exception_handlers import register_all_handlers
from apps.api.server.routers.souls import SoulsRouter


@pytest.fixture
def souls_router():
    return SoulsRouter()


@pytest.fixture
def app(souls_router):
    _app = FastAPI()
    register_all_handlers(_app)
    _app.include_router(souls_router.router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def _make_slo_info(name, description="A soul", traits=None):
    s = MagicMock()
    s.name = name
    s.path = f"models/{name}.soul"
    s.description = description
    s.personality = {"warmth": 0.7, "creativity": 0.5}
    s.traits = traits or []
    return s


class TestListSouls:
    """GET /souls"""

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_list_souls(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.list_souls.return_value = [
            _make_slo_info("sage", "Wise advisor", ["analytical"]),
            _make_slo_info("friend", "Warm companion", ["empathetic"]),
        ]
        mgr.get_current_soul.return_value = _make_slo_info("sage")
        mock_get_mgr.return_value = mgr

        resp = client.get("/souls")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        souls = body["data"]
        assert len(souls) == 2
        names = [s["name"] for s in souls]
        assert "sage" in names
        assert "friend" in names
        assert body["meta"]["current_soul"] == "sage"

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_list_souls_empty(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.list_souls.return_value = []
        mgr.get_current_soul.return_value = None
        mock_get_mgr.return_value = mgr

        resp = client.get("/souls")
        assert resp.status_code == 200
        assert resp.json()["data"] == []


class TestGetCurrentSoul:
    """GET /souls/current"""

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_get_current_soul(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.get_current_soul.return_value = _make_slo_info("sage", "Wise advisor", ["analytical"])
        mock_get_mgr.return_value = mgr

        resp = client.get("/souls/current")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"]["name"] == "sage"
        assert "analytical" in body["data"]["traits"]

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_get_current_soul_none(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.get_current_soul.return_value = None
        mock_get_mgr.return_value = mgr

        resp = client.get("/souls/current")
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] is None


class TestSwitchSoul:
    """POST /souls/switch"""

    @patch("domains.inference.slo_manager.get_slo_manager")
    @patch("domains.infrastructure.context_core.get_context_core")
    @patch("domains.core.soul.SloEngine")
    @patch("domains.models.provider.update_personality_traits")
    def test_switch_soul_success(self, mock_update_traits, mock_engine_cls,
                                  mock_get_ctx, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.switch_soul.return_value = {"success": True}
        mgr.get_soul.return_value = _make_slo_info("sage", "Wise advisor")
        mock_get_mgr.return_value = mgr

        mock_ctx_core = MagicMock()
        mock_get_ctx.return_value = mock_ctx_core

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.load_soul.return_value = MagicMock()

        resp = client.post("/souls/switch", json={"name": "sage"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        mgr.switch_soul.assert_called_once_with("sage")

    def test_switch_returns_status(self, client):
        resp = client.post("/souls/switch", json={"name": "test"})
        assert resp.status_code in (200, 500)
        data = resp.json()
        assert "status" in data


class TestGetTraitWeights:
    """GET /souls/weights"""

    def test_returns_weights(self, client):
        resp = client.get("/souls/weights")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "personality" in data
        assert "cognition" in data
        assert "emotion" in data

    def test_weights_have_trait_values(self, client):
        resp = client.get("/souls/weights")
        data = resp.json()["data"]
        assert isinstance(data["personality"], dict)
        assert len(data["personality"]) > 0


class TestGetTraitModes:
    """GET /souls/weights/modes"""

    def test_returns_modes(self, client):
        resp = client.get("/souls/weights/modes")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "personality" in data
        assert "memory" in data
        assert "style" in data
        assert "task" in data

    def test_modes_have_labels(self, client):
        resp = client.get("/souls/weights/modes")
        data = resp.json()["data"]
        assert "label" in data["personality"]
        assert "confidence" in data["personality"]


class TestListWeightSnapshots:
    """GET /souls/weights/snapshots"""

    def test_returns_snapshots(self, client):
        resp = client.get("/souls/weights/snapshots")
        assert resp.status_code == 200
        assert isinstance(resp.json()["data"], list)

    def test_empty_snapshots(self, client):
        resp = client.get("/souls/weights/snapshots")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert isinstance(data, list)


class TestGetSoulStats:
    """GET /souls/stats"""

    def test_returns_stats(self, client):
        resp = client.get("/souls/stats")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "total_souls" in data
        assert "available_souls" in data

    def test_stats_has_current_soul(self, client):
        resp = client.get("/souls/stats")
        data = resp.json()["data"]
        assert "current_soul" in data


class TestGetSoul:
    """GET /souls/{soul_name}"""

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_get_existing_soul(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.list_souls.return_value = [_make_slo_info("sage", "Wise advisor")]
        mock_get_mgr.return_value = mgr
        resp = client.get("/souls/sage")
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "sage"

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_get_nonexistent_soul(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.list_souls.return_value = []
        mock_get_mgr.return_value = mgr
        resp = client.get("/souls/nonexistent")
        assert resp.status_code == 404


class TestSoulChat:
    """POST /souls/chat"""

    def test_chat_requires_fields(self, client):
        resp = client.post("/souls/chat", json={"message": "Hi"})
        assert resp.status_code == 422

    def test_chat_with_valid_fields(self, client):
        resp = client.post("/souls/chat", json={
            "checkpoint_name": "test",
            "prompt": "Hello",
            "max_new_tokens": 10,
        })
        assert resp.status_code == 404

    def test_chat_invalid_checkpoint_name(self, client):
        resp = client.post("/souls/chat", json={
            "checkpoint_name": "../evil",
            "prompt": "Hello",
        })
        assert resp.status_code == 422
        assert resp.json()["error"] == "Invalid checkpoint name"

    def test_chat_missing_checkpoint_name(self, client):
        resp = client.post("/souls/chat", json={
            "prompt": "Hello",
            "max_new_tokens": 5,
        })
        assert resp.status_code == 422


class TestSaveTraitWeights:
    """POST /souls/weights"""

    @patch("domains.context.managers.get_trait_config")
    def test_save_flattens_groups(self, mock_get_config, client):
        config = MagicMock()
        mock_get_config.return_value = config
        resp = client.post("/souls/weights", json={
            "personality": {"warmth": 0.8},
            "cognition": {"curiosity": 0.3},
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        config.set_many.assert_called_once_with({"warmth": 0.8, "curiosity": 0.3})

    @patch("domains.context.managers.get_trait_config")
    def test_save_empty_body(self, mock_get_config, client):
        config = MagicMock()
        mock_get_config.return_value = config
        resp = client.post("/souls/weights", json={})
        assert resp.status_code == 200
        config.set_many.assert_called_once_with({})

    @patch("domains.context.managers.get_trait_config")
    def test_save_propagates_error(self, mock_get_config, client):
        mock_get_config.side_effect = RuntimeError("boom")
        resp = client.post("/souls/weights", json={"personality": {"warmth": 0.5}})
        assert resp.status_code == 500
        assert resp.json()["error"] == "An unexpected error occurred."


class TestWeightSnapshotLifecycle:
    """CRUD for /souls/weights/snapshot/{name}"""

    @patch("domains.context.managers.get_trait_config")
    def test_save_snapshot(self, mock_get_config, client):
        config = MagicMock()
        config.save_snapshot.return_value = "/tmp/snap_1.json"
        mock_get_config.return_value = config
        resp = client.post("/souls/weights/snapshot/preset-a")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"]["path"] == "/tmp/snap_1.json"
        config.save_snapshot.assert_called_once_with("preset-a")

    @patch("domains.context.managers.get_trait_config")
    def test_load_snapshot(self, mock_get_config, client):
        config = MagicMock()
        config.load_snapshot.return_value = 7
        mock_get_config.return_value = config
        resp = client.post("/souls/weights/snapshot/preset-a/load")
        assert resp.status_code == 200
        assert resp.json()["data"]["traits_loaded"] == 7
        config.load_snapshot.assert_called_once_with("preset-a")

    @patch("domains.context.managers.get_trait_config")
    def test_delete_snapshot(self, mock_get_config, client):
        config = MagicMock()
        config.delete_snapshot.return_value = True
        mock_get_config.return_value = config
        resp = client.delete("/souls/weights/snapshot/preset-a")
        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] is True

    @patch("domains.context.managers.get_trait_config")
    def test_delete_missing_snapshot(self, mock_get_config, client):
        config = MagicMock()
        config.delete_snapshot.return_value = False
        mock_get_config.return_value = config
        resp = client.delete("/souls/weights/snapshot/nope")
        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] is False

    @patch("domains.context.managers.get_trait_config")
    def test_snapshot_error_propagates(self, mock_get_config, client):
        mock_get_config.side_effect = RuntimeError("disk full")
        resp = client.post("/souls/weights/snapshot/x")
        assert resp.status_code == 500
        assert resp.json()["error"] == "An unexpected error occurred."


class TestListWeightSnapshotsPatched:
    """GET /souls/weights/snapshots with manager patched."""

    @patch("domains.context.managers.get_trait_config")
    def test_returns_names(self, mock_get_config, client):
        config = MagicMock()
        config.list_snapshots.return_value = ["a", "b"]
        mock_get_config.return_value = config
        resp = client.get("/souls/weights/snapshots")
        assert resp.status_code == 200
        assert resp.json()["data"] == ["a", "b"]


class TestGetTraitWeightsPatched:
    """GET /souls/weights with manager patched."""

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_returns_full_weights(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.get_trait_weights.return_value = {
            "personality": {"warmth": 0.6},
            "cognition": {},
            "emotion": {},
        }
        mock_get_mgr.return_value = mgr
        resp = client.get("/souls/weights")
        assert resp.status_code == 200
        assert resp.json()["data"]["personality"] == {"warmth": 0.6}

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_error_propagates(self, mock_get_mgr, client):
        mock_get_mgr.side_effect = RuntimeError("boom")
        resp = client.get("/souls/weights")
        assert resp.status_code == 500
        assert resp.json()["error"] == "An unexpected error occurred."


class TestSwitchSoulCheckpoint:
    """POST /souls/switch with checkpoint_name."""

    @patch.object(SoulsRouter, "_load_checkpoint_into_model")
    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_checkpoint_result_in_response(self, mock_get_mgr, mock_load, client):
        mgr = MagicMock()
        mgr.switch_soul.return_value = {"success": True}
        mgr.get_soul.return_value = None
        mock_get_mgr.return_value = mgr
        mock_load.return_value = {"status": "invalid_name"}

        resp = client.post("/souls/switch", json={
            "name": "sage", "checkpoint_name": "../evil",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"]["checkpoint_loaded"]["status"] == "invalid_name"
        mock_load.assert_called_once_with("../evil")

    @patch.object(SoulsRouter, "_load_checkpoint_into_model")
    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_switch_without_checkpoint(self, mock_get_mgr, mock_load, client):
        mgr = MagicMock()
        mgr.switch_soul.return_value = {"success": True}
        mgr.get_soul.return_value = None
        mock_get_mgr.return_value = mgr
        resp = client.post("/souls/switch", json={"name": "sage"})
        assert resp.status_code == 200
        mock_load.assert_not_called()


class TestGetSoulStatsPatched:
    """GET /souls/stats with manager patched."""

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_stats_forwarded(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.get_stats.return_value = {"total_souls": 3, "last_switch": "2026-01-01"}
        mock_get_mgr.return_value = mgr
        resp = client.get("/souls/stats")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total_souls"] == 3
        assert data["last_switch"] == "2026-01-01"

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_error_propagates(self, mock_get_mgr, client):
        mock_get_mgr.side_effect = RuntimeError("boom")
        resp = client.get("/souls/stats")
        assert resp.status_code == 500
        assert resp.json()["error"] == "An unexpected error occurred."


class TestGetSoulErrorPath:
    """GET /souls/{soul_name} error propagation."""

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_manager_error_raises_http(self, mock_get_mgr, client):
        from domains.infrastructure.errors import classify_exception
        err = classify_exception(RuntimeError("boom"))
        mock_get_mgr.side_effect = RuntimeError("boom")
        resp = client.get("/souls/broken")
        assert resp.status_code == err.http_status
