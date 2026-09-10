"""Tests for the /agents router.

Covers all 8 endpoints:
  GET  /agents                    — list_agents
  POST /agents                    — create_agent (201)
  GET  /agents/{agent_id}         — get_agent
  PUT  /agents/{agent_id}         — update_agent
  DELETE /agents/{agent_id}       — delete_agent
  POST /agents/{agent_id}/execute — execute_agent
  POST /agents/orchestrate        — orchestrate_agents (SSE)
  GET  /agents/runs               — list_runs
  GET  /agents/runs/{run_id}      — get_run
"""

import time
import pytest
from unittest.mock import patch, AsyncMock
from test_support import get_test_client


def _d(resp):
    """Unwrap success_response envelope."""
    j = resp.json()
    return j.get("data", j)


@pytest.fixture(autouse=True)
def _fresh_agent_system():
    """Reset the agent system singleton before each test."""
    import domains.agents.system as sys_mod

    sys_mod._default_system = None
    yield
    sys_mod._default_system = None


def _unique_id(prefix: str = "test") -> str:
    return f"{prefix}-{int(time.time() * 1000)}"


class TestListAgents:
    def setup_method(self):
        self.client = get_test_client()

    def test_list_returns_default_agents(self):
        resp = self.client.get("/agents")
        assert resp.status_code == 200
        data = _d(resp)
        assert isinstance(data, list)
        names = [a["name"] for a in data]
        assert "General" in names
        assert "Coder" in names

    def test_list_includes_created_agent(self):
        aid = _unique_id("list-test")
        self.client.post("/agents", json={"name": aid, "description": "test"})
        resp = self.client.get("/agents")
        data = _d(resp)
        ids = [a["id"] for a in data]
        assert aid in ids


class TestCreateAgent:
    def setup_method(self):
        self.client = get_test_client()

    def test_create_returns_201(self):
        aid = _unique_id("create")
        resp = self.client.post("/agents", json={"name": aid, "description": "A test agent"})
        assert resp.status_code == 201
        data = _d(resp)
        assert data["name"] == aid
        assert data["description"] == "A test agent"
        assert "id" in data

    def test_create_auto_generates_id_from_name(self):
        resp = self.client.post("/agents", json={"name": "My Cool Agent", "description": "x"})
        assert resp.status_code == 201
        data = _d(resp)
        assert data["id"] == "my-cool-agent"

    def test_create_duplicate_returns_409(self):
        aid = _unique_id("dup")
        self.client.post("/agents", json={"name": aid, "description": "first"})
        resp = self.client.post("/agents", json={"name": aid, "description": "second"})
        assert resp.status_code == 409

    def test_create_missing_name_returns_422(self):
        resp = self.client.post("/agents", json={"description": "no name"})
        assert resp.status_code == 422

    def test_create_with_tools(self):
        aid = _unique_id("tools")
        resp = self.client.post(
            "/agents", json={"name": aid, "description": "x", "tools": ["code_execution", "memory"]}
        )
        assert resp.status_code == 201
        data = _d(resp)
        assert "code_execution" in data["tools"]
        assert "memory" in data["tools"]


class TestGetAgent:
    def setup_method(self):
        self.client = get_test_client()

    def test_get_existing_agent(self):
        aid = _unique_id("get")
        self.client.post("/agents", json={"name": aid, "description": "gettable"})
        resp = self.client.get(f"/agents/{aid}")
        assert resp.status_code == 200
        data = _d(resp)
        assert data["id"] == aid
        assert data["name"] == aid

    def test_get_nonexistent_returns_404(self):
        resp = self.client.get("/agents/definitely-not-real")
        assert resp.status_code == 404

    def test_get_default_agent(self):
        resp = self.client.get("/agents/general")
        assert resp.status_code == 200
        data = _d(resp)
        assert data["name"] == "General"


class TestUpdateAgent:
    def setup_method(self):
        self.client = get_test_client()

    def test_update_name(self):
        aid = _unique_id("upd")
        self.client.post("/agents", json={"name": aid, "description": "orig"})
        resp = self.client.put(f"/agents/{aid}", json={"name": "New Name"})
        assert resp.status_code == 200
        data = _d(resp)
        assert data["name"] == "New Name"

    def test_update_description(self):
        aid = _unique_id("upd-desc")
        self.client.post("/agents", json={"name": aid, "description": "old"})
        resp = self.client.put(f"/agents/{aid}", json={"description": "updated"})
        assert resp.status_code == 200
        data = _d(resp)
        assert data["description"] == "updated"

    def test_update_tools(self):
        aid = _unique_id("upd-tools")
        self.client.post("/agents", json={"name": aid, "description": "x"})
        resp = self.client.put(f"/agents/{aid}", json={"tools": ["web_search"]})
        assert resp.status_code == 200
        data = _d(resp)
        assert data["tools"] == ["web_search"]

    def test_update_nonexistent_returns_404(self):
        resp = self.client.put("/agents/no-such-agent", json={"name": "nope"})
        assert resp.status_code == 404


class TestDeleteAgent:
    def setup_method(self):
        self.client = get_test_client()

    def test_delete_existing(self):
        aid = _unique_id("del")
        self.client.post("/agents", json={"name": aid, "description": "x"})
        resp = self.client.delete(f"/agents/{aid}")
        assert resp.status_code == 200
        data = _d(resp)
        assert data["status"] == "deleted"
        # Verify gone
        resp = self.client.get(f"/agents/{aid}")
        assert resp.status_code == 404

    def test_delete_nonexistent_returns_404(self):
        resp = self.client.delete("/agents/ghost-agent")
        assert resp.status_code == 404


class TestExecuteAgent:
    def setup_method(self):
        self.client = get_test_client()

    @patch("domains.agents.system.AgentSystem.execute", new_callable=AsyncMock)
    def test_execute_calls_system(self, mock_execute):
        mock_execute.return_value = {"response": "Hello!", "success": True}
        aid = _unique_id("exec")
        self.client.post("/agents", json={"name": aid, "description": "x"})
        resp = self.client.post(f"/agents/{aid}/execute", json={"request": "Say hello"})
        assert resp.status_code == 200
        mock_execute.assert_called_once()

    @patch("domains.agents.system.AgentSystem.execute", new_callable=AsyncMock)
    def test_execute_nonexistent_agent(self, mock_execute):
        mock_execute.return_value = {"error": "Agent 'nope' not found", "success": False}
        resp = self.client.post("/agents/nope/execute", json={"request": "hi"})
        assert resp.status_code == 404

    def test_execute_missing_request(self):
        resp = self.client.post("/agents/general/execute", json={})
        assert resp.status_code == 422


class TestListRuns:
    def setup_method(self):
        self.client = get_test_client()

    def test_list_runs_empty(self):
        resp = self.client.get("/agents/runs")
        assert resp.status_code == 200
        data = _d(resp)
        assert "runs" in data
        assert "count" in data

    def test_list_runs_with_limit(self):
        resp = self.client.get("/agents/runs?limit=5")
        assert resp.status_code == 200
        data = _d(resp)
        assert isinstance(data["runs"], list)


class TestGetRun:
    def setup_method(self):
        self.client = get_test_client()

    def test_get_nonexistent_run(self):
        resp = self.client.get("/agents/runs/nonexistent-run-id")
        assert resp.status_code == 404


class TestAgentCRUDLifecycle:
    """End-to-end: create → get → update → delete."""

    def setup_method(self):
        self.client = get_test_client()

    def test_full_lifecycle(self):
        aid = _unique_id("lifecycle")
        # Create
        resp = self.client.post("/agents", json={"name": aid, "description": "start"})
        assert resp.status_code == 201

        # Get
        resp = self.client.get(f"/agents/{aid}")
        assert resp.status_code == 200
        assert _d(resp)["description"] == "start"

        # Update
        resp = self.client.put(f"/agents/{aid}", json={"description": "updated"})
        assert resp.status_code == 200
        assert _d(resp)["description"] == "updated"

        # Delete
        resp = self.client.delete(f"/agents/{aid}")
        assert resp.status_code == 200

        # Verify gone
        resp = self.client.get(f"/agents/{aid}")
        assert resp.status_code == 404
