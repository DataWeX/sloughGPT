"""Tests for the souls API router (routers/souls.py).

Covers: list, get_current, switch, weight snapshots.
SloManager is mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.souls import SoulsRouter, SloRouterState  # noqa: E402


def _mock_manager(**overrides) -> MagicMock:
    mgr = MagicMock()
    soul1 = MagicMock()
    soul1.name = "assistant"
    soul1.path = "/souls/assistant.soul"
    soul1.description = "Helpful assistant"
    soul1.personality = {}
    soul1.traits = ["helpful", "friendly"]
    soul2 = MagicMock()
    soul2.name = "creative"
    soul2.path = "/souls/creative.soul"
    soul2.description = "Creative soul"
    soul2.personality = {}
    soul2.traits = ["creative"]
    mgr.list_souls.return_value = [soul1, soul2]
    current = MagicMock()
    current.name = "assistant"
    current.description = "Helpful assistant"
    current.traits = ["helpful"]
    mgr.get_current_soul.return_value = current
    mgr.switch_soul.return_value = {"success": True, "name": "creative", "description": "Creative soul"}
    mgr.get_soul.return_value = soul2
    mgr.get_soul_prompt.return_value = "You are a helpful assistant."
    mgr.get_trait_weights.return_value = {"warmth": 0.8, "creativity": 0.5}
    return mgr


def _app(sr: SoulsRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(sr.router)
    return app


class TestListSouls:
    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_list(self, mock_get):
        mock_get.return_value = _mock_manager()
        sr = SoulsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/souls")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) == 2
        assert data[0]["name"] == "assistant"


class TestGetCurrentSoul:
    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_get_current(self, mock_get):
        mock_get.return_value = _mock_manager()
        sr = SoulsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/souls/current")
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "assistant"


class TestSwitchSoul:
    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_switch(self, mock_get):
        mock_get.return_value = _mock_manager()
        sr = SoulsRouter()
        sr.state = SloRouterState()
        client = TestClient(_app(sr))
        resp = client.post("/souls/switch", json={"name": "creative"})
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "creative"

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_switch_with_checkpoint(self, mock_get):
        mgr = _mock_manager()
        mgr.load_checkpoint.return_value = {"name": "assistant", "loss": 1.5}
        mock_get.return_value = mgr
        sr = SoulsRouter()
        sr.state = SloRouterState()
        client = TestClient(_app(sr))
        resp = client.post("/souls/switch", json={"name": "assistant", "checkpoint_name": "ckpt-1"})
        assert resp.status_code == 200


class TestWeightSnapshots:
    def test_list_snapshots(self):
        sr = SoulsRouter()
        sr.state = SloRouterState()
        client = TestClient(_app(sr))
        resp = client.get("/souls/weights/snapshots")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert isinstance(data["data"], list)

    @patch("domains.context.managers.get_trait_config")
    def test_save_snapshot(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.save_snapshot.return_value = "/tmp/snap.json"
        mock_get_config.return_value = mock_config
        sr = SoulsRouter()
        sr.state = SloRouterState()
        client = TestClient(_app(sr))
        resp = client.post("/souls/weights/snapshot/test-snap")
        assert resp.status_code == 200
        assert resp.json()["data"]["path"] == "/tmp/snap.json"

    @patch("domains.context.managers.get_trait_config")
    def test_delete_nonexistent_snapshot(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.delete_snapshot.return_value = False
        mock_get_config.return_value = mock_config
        sr = SoulsRouter()
        sr.state = SloRouterState()
        client = TestClient(_app(sr))
        resp = client.delete("/souls/weights/snapshot/nonexistent")
        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] is False
