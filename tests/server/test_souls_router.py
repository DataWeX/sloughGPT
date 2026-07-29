"""
Tests for the souls router — list, current, switch.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.souls import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
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

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_switch_soul_failure(self, mock_get_mgr, client):
        mgr = MagicMock()
        mgr.switch_soul.side_effect = RuntimeError("soul not found")
        mock_get_mgr.return_value = mgr

        resp = client.post("/souls/switch", json={"name": "nonexistent"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"
