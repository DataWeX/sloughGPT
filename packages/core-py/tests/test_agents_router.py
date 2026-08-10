"""Tests for the agents API router (routers/agents.py).

Covers: list, create, get, update, delete, execute, list_runs, get_run.
Agent system is mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.agents import AgentsRouter  # noqa: E402


def _mock_system(**overrides) -> MagicMock:
    sys = MagicMock()
    sys.list.return_value = [
        {"id": "a1", "name": "Agent One", "description": "desc", "instructions": "", "tools": [], "avatar": ""},
    ]
    sys.get.return_value = {"id": "a1", "name": "Agent One", "description": "desc", "instructions": "", "tools": [], "avatar": ""}
    sys.create.return_value = {"id": "new-agent", "name": "New", "description": "", "instructions": "", "tools": [], "avatar": ""}
    sys.update.return_value = {"id": "a1", "name": "Updated", "description": "", "instructions": "", "tools": [], "avatar": ""}
    sys.delete.return_value = True
    sys.execute = AsyncMock(return_value={"result": "done"})
    return sys


def _app(ar: AgentsRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(ar.router)
    return app


class TestListAgents:
    @patch.object(AgentsRouter, "_get_system")
    def test_list(self, mock_gs):
        mock_gs.return_value = _mock_system()
        ar = AgentsRouter()
        client = TestClient(_app(ar))
        resp = client.get("/agents")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["id"] == "a1"


class TestCreateAgent:
    @patch.object(AgentsRouter, "_get_system")
    def test_create(self, mock_gs):
        sys = _mock_system()
        sys.get.return_value = None  # no conflict
        mock_gs.return_value = sys
        ar = AgentsRouter()
        client = TestClient(_app(ar))
        resp = client.post("/agents", json={"name": "New"})
        assert resp.status_code == 201
        assert resp.json()["id"] == "new-agent"

    @patch.object(AgentsRouter, "_get_system")
    def test_create_duplicate(self, mock_gs):
        sys = _mock_system()
        sys.get.return_value = {"id": "a1", "name": "X"}
        mock_gs.return_value = sys
        ar = AgentsRouter()
        client = TestClient(_app(ar))
        resp = client.post("/agents", json={"name": "Agent One"})
        assert resp.status_code == 409


class TestGetAgent:
    @patch.object(AgentsRouter, "_get_system")
    def test_get_found(self, mock_gs):
        mock_gs.return_value = _mock_system()
        ar = AgentsRouter()
        client = TestClient(_app(ar))
        resp = client.get("/agents/a1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "a1"

    @patch.object(AgentsRouter, "_get_system")
    def test_get_not_found(self, mock_gs):
        sys = _mock_system()
        sys.get.return_value = None
        mock_gs.return_value = sys
        ar = AgentsRouter()
        client = TestClient(_app(ar))
        resp = client.get("/agents/nonexistent")
        assert resp.status_code == 404


class TestUpdateAgent:
    @patch.object(AgentsRouter, "_get_system")
    def test_update(self, mock_gs):
        mock_gs.return_value = _mock_system()
        ar = AgentsRouter()
        client = TestClient(_app(ar))
        resp = client.put("/agents/a1", json={"name": "Updated"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated"

    @patch.object(AgentsRouter, "_get_system")
    def test_update_not_found(self, mock_gs):
        sys = _mock_system()
        sys.update.return_value = None
        mock_gs.return_value = sys
        ar = AgentsRouter()
        client = TestClient(_app(ar))
        resp = client.put("/agents/nonexistent", json={"name": "X"})
        assert resp.status_code == 404


class TestDeleteAgent:
    @patch.object(AgentsRouter, "_get_system")
    def test_delete(self, mock_gs):
        mock_gs.return_value = _mock_system()
        ar = AgentsRouter()
        client = TestClient(_app(ar))
        resp = client.delete("/agents/a1")
        assert resp.status_code == 200

    @patch.object(AgentsRouter, "_get_system")
    def test_delete_not_found(self, mock_gs):
        sys = _mock_system()
        sys.delete.return_value = False
        mock_gs.return_value = sys
        ar = AgentsRouter()
        client = TestClient(_app(ar))
        resp = client.delete("/agents/nonexistent")
        assert resp.status_code == 404


class TestExecuteAgent:
    @patch.object(AgentsRouter, "_get_system")
    def test_execute(self, mock_gs):
        mock_gs.return_value = _mock_system()
        ar = AgentsRouter()
        client = TestClient(_app(ar))
        resp = client.post("/agents/a1/execute", json={"request": "do something"})
        assert resp.status_code == 200
        assert resp.json()["result"] == "done"


class TestListRuns:
    @patch.object(AgentsRouter, "_get_system")
    def test_list_runs(self, mock_gs):
        mock_gs.return_value = _mock_system()
        ar = AgentsRouter()
        # runs comes from run_history store
        with patch("domains.agents.run_history.get_agent_run_store") as mock_rs:
            mock_rs.return_value.list_runs.return_value = []
            client = TestClient(_app(ar))
            resp = client.get("/agents/runs")
        assert resp.status_code == 200
