"""Tests for the agents API router (routers/agents.py).

Covers: list, create, get, update, delete, execute, list_runs, get_run.
Agent system is mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

_server_dir = str(Path(__file__).resolve().parents[2] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.agents import AgentsRouter  # noqa: E402


def _mock_system(**overrides) -> MagicMock:
    sys = MagicMock()
    sys.list.return_value = [
        {
            "id": "a1",
            "name": "Agent One",
            "description": "desc",
            "instructions": "",
            "tools": [],
            "avatar": "",
        },
    ]
    sys.get.return_value = {
        "id": "a1",
        "name": "Agent One",
        "description": "desc",
        "instructions": "",
        "tools": [],
        "avatar": "",
    }
    sys.create.return_value = {
        "id": "new-agent",
        "name": "New",
        "description": "",
        "instructions": "",
        "tools": [],
        "avatar": "",
    }
    sys.update.return_value = {
        "id": "a1",
        "name": "Updated",
        "description": "",
        "instructions": "",
        "tools": [],
        "avatar": "",
    }
    sys.delete.return_value = True
    sys.execute = AsyncMock(return_value={"result": "done"})
    return sys


def _app(ar: AgentsRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(ar.router)
    # The routine stream lives in its own descriptor-projected module (the
    # declaration ratchet forbids new hand-built routes in agents.py).
    from routers.agent_routines import router as routines_router

    app.include_router(routines_router)
    from infrastructure.exception_handlers import register_all_handlers

    register_all_handlers(app)
    return app


class TestListAgents:
    @patch.object(AgentsRouter, "_get_system")
    def test_list(self, mock_gs):
        mock_gs.return_value = _mock_system()
        ar = AgentsRouter()
        client = TestClient(_app(ar))
        resp = client.get("/agents")
        assert resp.status_code == 200
        body = resp.json()
        data = body.get("data", body)
        if isinstance(data, dict):
            data = data.get("agents", data.get("items", []))
        assert len(data) == 1
        assert data[0]["id"] == "a1"


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
        body = resp.json()
        data = body.get("data", body)
        assert data["id"] == "new-agent"

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
        body = resp.json()
        data = body.get("data", body)
        assert data["id"] == "a1"

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
        body = resp.json()
        data = body.get("data", body)
        assert data["name"] == "Updated"

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
        body = resp.json()
        data = body.get("data", body)
        assert data["result"] == "done"


class TestListRuns:
    @patch.object(AgentsRouter, "_get_system")
    def test_list_runs(self, mock_gs):
        mock_gs.return_value = _mock_system()
        ar = AgentsRouter()
        # runs comes from run_history store
        with patch("domain.agents._internal.run_history.get_agent_run_store") as mock_rs:
            mock_rs.return_value.list_runs.return_value = []
            client = TestClient(_app(ar))
            resp = client.get("/agents/runs")
        assert resp.status_code == 200


class TestRoutinesStream:
    """SSE surface for the background routine runtime (agent phase 2)."""

    def _runner_with_events(self):
        """A runner that never touches disk — record_event needs no store."""
        from domain.agents._internal.routines import AgentRoutineRunner

        class _NullStore:
            def list(self):
                return []

            def get(self, routine_id):
                return None

            def save(self, routine):
                return True

            def delete(self, routine_id):
                return False

        return AgentRoutineRunner(system=None, store=_NullStore(), queue=None)

    @patch("fastapi.Request.is_disconnected", new=AsyncMock(side_effect=[False, True]))
    @patch("asyncio.sleep", new=AsyncMock(return_value=None))
    def test_stream_content_type_and_snapshot(self):
        ar = AgentsRouter()
        client = TestClient(_app(ar))
        with client.stream("GET", "/agents/routines/stream") as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            body = resp.read().decode()
        assert '"stream": "agent-routines"' in body
        assert '"phase": "SNAPSHOT"' in body
        assert '"routines"' in body

    @patch("fastapi.Request.is_disconnected", new=AsyncMock(side_effect=[False, True]))
    @patch("asyncio.sleep", new=AsyncMock(return_value=None))
    def test_stream_terminates_on_disconnect(self):
        ar = AgentsRouter()
        client = TestClient(_app(ar))
        with client.stream("GET", "/agents/routines/stream") as resp:
            body = resp.read().decode()
        # A complete frame is emitted after the loop exits.
        assert '"phase": "COMPLETE"' in body

    @patch("fastapi.Request.is_disconnected", new=AsyncMock(side_effect=[False, True]))
    @patch("asyncio.sleep", new=AsyncMock(return_value=None))
    def test_stream_forwards_routine_run_events(self):
        from domain.agents._internal import routines as _rt

        runner = self._runner_with_events()
        runner.record_event(
            {"event": "final", "routine_id": "r1", "agent_id": "helper", "text": "done"}
        )
        _rt._default_runner = runner

        ar = AgentsRouter()
        client = TestClient(_app(ar))
        with client.stream("GET", "/agents/routines/stream") as resp:
            body = resp.read().decode()

        assert '"routine_id": "r1"' in body
        assert '"phase": "FINAL"' in body

        _rt._default_runner = None
