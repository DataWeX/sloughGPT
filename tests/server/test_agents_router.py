"""
Tests for the agents router — CRUD and execution.
"""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.agents import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestListAgents:
    @patch("domains.agents.system.get_agent_system")
    def test_returns_empty_list(self, mock_get_sys, client):
        sys = mock_get_sys.return_value
        sys.list.return_value = []
        resp = client.get("/agents")
        assert resp.status_code == 200
        assert resp.json() == []


class TestCreateAgent:
    @patch("domains.agents.system.get_agent_system")
    def test_creates_agent(self, mock_get_sys, client):
        sys = mock_get_sys.return_value
        sys.get.return_value = None
        sys.create.return_value = {
            "id": "helper", "name": "Helper", "description": "",
            "instructions": "", "tools": [], "avatar": "",
        }
        resp = client.post("/agents", json={"name": "Helper"})
        assert resp.status_code == 201
        assert resp.json()["id"] == "helper"

    @patch("domains.agents.system.get_agent_system")
    def test_rejects_duplicate(self, mock_get_sys, client):
        sys = mock_get_sys.return_value
        sys.get.return_value = {"id": "helper"}
        resp = client.post("/agents", json={"name": "Helper"})
        assert resp.status_code == 409


class TestGetAgent:
    @patch("domains.agents.system.get_agent_system")
    def test_returns_agent(self, mock_get_sys, client):
        sys = mock_get_sys.return_value
        sys.get.return_value = {"id": "helper", "name": "Helper", "description": "", "instructions": "", "tools": [], "avatar": ""}
        resp = client.get("/agents/helper")
        assert resp.status_code == 200

    @patch("domains.agents.system.get_agent_system")
    def test_returns_404_for_missing(self, mock_get_sys, client):
        sys = mock_get_sys.return_value
        sys.get.return_value = None
        resp = client.get("/agents/nonexistent")
        assert resp.status_code == 404


class TestUpdateAgent:
    @patch("domains.agents.system.get_agent_system")
    def test_updates_agent(self, mock_get_sys, client):
        sys = mock_get_sys.return_value
        sys.update.return_value = {"id": "helper", "name": "Updated", "description": "", "instructions": "", "tools": [], "avatar": ""}
        resp = client.put("/agents/helper", json={"name": "Updated"})
        assert resp.status_code == 200

    @patch("domains.agents.system.get_agent_system")
    def test_returns_404_for_missing(self, mock_get_sys, client):
        sys = mock_get_sys.return_value
        sys.update.return_value = None
        resp = client.put("/agents/nonexistent", json={"name": "X"})
        assert resp.status_code == 404


class TestDeleteAgent:
    @patch("domains.agents.system.get_agent_system")
    def test_deletes_agent(self, mock_get_sys, client):
        sys = mock_get_sys.return_value
        sys.delete.return_value = True
        resp = client.delete("/agents/helper")
        assert resp.status_code == 200

    @patch("domains.agents.system.get_agent_system")
    def test_returns_404_for_missing(self, mock_get_sys, client):
        sys = mock_get_sys.return_value
        sys.delete.return_value = False
        resp = client.delete("/agents/nonexistent")
        assert resp.status_code == 404


class TestExecuteAgent:
    @patch("domains.agents.system.get_agent_system")
    def test_executes_agent(self, mock_get_sys, client):
        sys = mock_get_sys.return_value
        sys.execute = AsyncMock(return_value={"response": "hello"})
        resp = client.post("/agents/helper/execute", json={"request": "say hi"})
        assert resp.status_code == 200

    @patch("domains.agents.system.get_agent_system")
    def test_returns_404_on_error(self, mock_get_sys, client):
        sys = mock_get_sys.return_value
        sys.execute = AsyncMock(return_value={"error": "Agent not found"})
        resp = client.post("/agents/nonexistent/execute", json={"request": "hi"})
        assert resp.status_code == 404
