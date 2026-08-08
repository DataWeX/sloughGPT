"""
Tests for the agents router — CRUD and execution.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from apps.api.server.routers.agents import router


def _make_app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


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

    @patch("domains.agents.system.get_agent_system")
    def test_name_with_spaces_and_underscores_slugged(self, mock_get_sys, client):
        sys = mock_get_sys.return_value
        sys.get.return_value = None
        sys.create.return_value = {
            "id": "risk", "name": "Risk", "description": "",
            "instructions": "", "tools": [], "avatar": "",
        }
        client.post("/agents", json={"name": "Risk Analyst_Writer"})
        args, kwargs = sys.create.call_args
        assert kwargs["agent_id"] == "risk-analyst-writer"

    @patch("domains.agents.system.get_agent_system")
    def test_explicit_id_used(self, mock_get_sys, client):
        sys = mock_get_sys.return_value
        sys.get.return_value = None
        sys.create.return_value = {
            "id": "my-agent", "name": "Name", "description": "",
            "instructions": "", "tools": [], "avatar": "",
        }
        client.post("/agents", json={"name": "Name", "id": "my-agent"})
        args, kwargs = sys.create.call_args
        assert kwargs["agent_id"] == "my-agent"

    @patch("domains.agents.system.get_agent_system")
    def test_tools_and_avatar_and_instructions_passthrough(self, mock_get_sys, client):
        sys = mock_get_sys.return_value
        sys.get.return_value = None
        sys.create.return_value = {
            "id": "helper", "name": "Helper", "description": "d",
            "instructions": "i", "tools": ["search"], "avatar": "a",
        }
        client.post("/agents", json={
            "name": "Helper", "description": "d", "instructions": "i",
            "tools": ["search"], "avatar": "a",
        })
        args, kwargs = sys.create.call_args
        assert kwargs["tools"] == ["search"]
        assert kwargs["avatar"] == "a"
        assert kwargs["instructions"] == "i"

    def test_empty_name_is_422(self, client):
        resp = client.post("/agents", json={"name": ""})
        assert resp.status_code == 422

    def test_missing_name_is_422(self, client):
        resp = client.post("/agents", json={})
        assert resp.status_code == 422


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

    @patch("domains.agents.system.get_agent_system")
    def test_execute_passes_session_and_user(self, mock_get_sys, client):
        sys = mock_get_sys.return_value
        sys.execute = AsyncMock(return_value={"response": "ok"})
        client.post("/agents/helper/execute", json={
            "request": "go", "session_id": "s1", "user_id": "u7",
        })
        args, kwargs = sys.execute.call_args
        assert kwargs["session_id"] == "s1"
        assert kwargs["user_id"] == "u7"
        assert kwargs["agent_id"] == "helper"

    def test_execute_empty_request_is_422(self, client):
        resp = client.post("/agents/helper/execute", json={"request": ""})
        assert resp.status_code == 422


class TestListRuns:
    @patch("domains.agents.run_history.get_agent_run_store")
    def test_lists_runs(self, mock_get_store, client):
        store = mock_get_store.return_value
        store.list_runs.return_value = [{"id": "run_1", "goal": "Research", "status": "completed"}]
        resp = client.get("/agents/runs")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        assert body["runs"][0]["id"] == "run_1"

    @patch("domains.agents.run_history.get_agent_run_store")
    def test_lists_empty_runs(self, mock_get_store, client):
        store = mock_get_store.return_value
        store.list_runs.return_value = []
        resp = client.get("/agents/runs")
        assert resp.status_code == 200
        assert resp.json() == {"runs": [], "count": 0}

    @patch("domains.agents.run_history.get_agent_run_store")
    def test_limit_clamped_rooted_at_one(self, mock_get_store, client):
        store = mock_get_store.return_value
        store.list_runs.return_value = []
        resp = client.get("/agents/runs?limit=0")
        assert resp.status_code == 200
        args, kwargs = store.list_runs.call_args
        assert kwargs.get("limit") == 1

    @patch("domains.agents.run_history.get_agent_run_store")
    def test_limit_capped_at_200(self, mock_get_store, client):
        store = mock_get_store.return_value
        store.list_runs.return_value = []
        resp = client.get("/agents/runs?limit=9999")
        assert resp.status_code == 200
        args, kwargs = store.list_runs.call_args
        assert kwargs.get("limit") == 200

    @patch("domains.agents.run_history.get_agent_run_store")
    def test_custom_valid_limit_passthrough(self, mock_get_store, client):
        store = mock_get_store.return_value
        store.list_runs.return_value = []
        resp = client.get("/agents/runs?limit=42")
        assert resp.status_code == 200
        args, kwargs = store.list_runs.call_args
        assert kwargs.get("limit") == 42


class TestGetRun:
    @patch("domains.agents.run_history.get_agent_run_store")
    def test_returns_run(self, mock_get_store, client):
        store = mock_get_store.return_value
        store.get.return_value = {"id": "run_1", "goal": "Research", "status": "completed"}
        resp = client.get("/agents/runs/run_1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "run_1"

    @patch("domains.agents.run_history.get_agent_run_store")
    def test_returns_404_for_missing(self, mock_get_store, client):
        store = mock_get_store.return_value
        store.get.return_value = None
        resp = client.get("/agents/runs/nonexistent")
        assert resp.status_code == 404


class TestOrchestrate:
    """POST /agents/orchestrate — SSE planning pipeline."""

    @patch("domains.agents.run_history.get_agent_run_store")
    @patch("domains.agents.multi.MultiAgentOrchestrator")
    def test_empty_goal_is_422(self, mock_orch, mock_store, client):
        resp = client.post("/agents/orchestrate", json={"goal": ""})
        assert resp.status_code == 422

    @patch("domains.agents.run_history.get_agent_run_store")
    @patch("domains.agents.multi.MultiAgentOrchestrator")
    def test_plan_failure_streams_error(self, mock_orch, mock_store, client):
        store = mock_store.return_value
        store.start.return_value = "run-x"
        orch = mock_orch.return_value
        orch._async_plan = AsyncMock(return_value=[])
        resp = client.post("/agents/orchestrate", json={"goal": "Do something"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert "Could not plan this goal" in resp.text
        store.fail.assert_called_once_with("run-x", "Could not plan this goal")

    @patch("domains.agents.run_history.get_agent_run_store")
    @patch("domains.agents.multi.MultiAgentOrchestrator")
    def test_full_pipeline_completes(self, mock_orch, mock_store, client):
        from types import SimpleNamespace

        store = mock_store.return_value
        store.start.return_value = "run-y"
        orch = mock_orch.return_value

        task = SimpleNamespace(
            id="t1", description="do it", assigned_agent="writer",
            status="pending", result=None, error=None,
            to_dict=lambda: {"id": "t1", "status": "pending"},
        )
        orch._async_plan = AsyncMock(return_value=[task])
        orch._compute_levels = MagicMock(return_value=[["t1"]])
        orch._build_dep_context = MagicMock(return_value={})
        orch._async_run_agent = AsyncMock(return_value="output text")
        orch._async_compose = AsyncMock(return_value="Final summary")

        resp = client.post("/agents/orchestrate", json={"goal": "Do x", "context": "ctx"})
        assert resp.status_code == 200
        body = resp.text
        assert "Final summary" in body
        assert "output text" in body
        assert task.status == "completed"
        store.complete.assert_called_once()

    @patch("domains.agents.run_history.get_agent_run_store")
    @patch("domains.agents.multi.MultiAgentOrchestrator")
    def test_task_failure_streams_error_event(self, mock_orch, mock_store, client):
        from types import SimpleNamespace

        store = mock_store.return_value
        store.start.return_value = "run-z"
        orch = mock_orch.return_value

        task = SimpleNamespace(
            id="t2", description="do it", assigned_agent="a",
            status=[], result=None, error=None,
            to_dict=lambda: {"id": "t2", "status": "pending"},
        )
        orch._async_plan = AsyncMock(return_value=[task])
        orch._compute_levels = MagicMock(return_value=[["t2"]])
        orch._build_dep_context = MagicMock(return_value={})
        orch._async_run_agent = AsyncMock(side_effect=RuntimeError("inference down"))
        orch._async_compose = AsyncMock(return_value="composed")

        resp = client.post("/agents/orchestrate", json={"goal": "Do y"})
        assert resp.status_code == 200
        body = resp.text
        assert "inference down" in body
        assert task.error == "inference down"
        assert task.status == "failed"
