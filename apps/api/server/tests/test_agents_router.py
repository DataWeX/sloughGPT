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

Plus the two SSE vocabularies (agent visualization contract): the
factory-floor reducer on the web side is derived from the phase tables
pinned in ``TestExecuteStreamVocabulary`` / ``TestOrchestrateVocabulary``
— a rename on either side must fail a test, not the floor view.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from test_support import get_test_client


def _d(resp):
    """Unwrap success_response envelope."""
    j = resp.json()
    return j.get("data", j)


@pytest.fixture(autouse=True)
def _fresh_agent_system():
    """Reset the agent system singleton before each test."""
    import domain.agents._internal.system as sys_mod

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
        aid = _unique_id("auto")
        resp = self.client.post("/agents", json={"name": f"{aid} Agent", "description": "x"})
        assert resp.status_code == 201
        data = _d(resp)
        assert data["id"] == f"{aid}-agent"

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

    @pytest.fixture
    def _mock_system(self):
        from unittest.mock import MagicMock

        system = MagicMock()
        system.execute = AsyncMock(return_value={"response": "Hello!", "success": True})
        with patch("domain.agents.get_agent_system", return_value=system):
            yield system

    def test_execute_calls_system(self, _mock_system):
        aid = _unique_id("exec")
        self.client.post("/agents", json={"name": aid, "description": "x"})
        resp = self.client.post(f"/agents/{aid}/execute", json={"request": "Say hello"})
        assert resp.status_code == 200
        _mock_system.execute.assert_called_once()

    def test_execute_nonexistent_agent(self, _mock_system):
        _mock_system.execute.return_value = {"error": "Agent 'nope' not found", "success": False}
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


def _sse_frames(text: str) -> list[dict]:
    """Parse 'data: {json}' SSE lines from a buffered response body."""
    import json

    frames = []
    for line in text.splitlines():
        if line.startswith("data: "):
            frames.append(json.loads(line[len("data: ") :]))
    return frames


class TestRunLoop:
    """POST /agents/loop — canonical AgentLoop over the live provider (SSE)."""

    def setup_method(self):
        self.client = get_test_client()

    def test_no_provider_yields_503_frame(self):
        from domain.models._internal.provider import registry as reg

        saved = dict(reg._providers)
        reg.clear_providers()
        try:
            resp = self.client.post("/agents/loop", json={"task": "hello"})
            assert resp.status_code == 200  # SSE always 200; error in-band
            frames = _sse_frames(resp.text)
            assert frames, "no SSE frames"
            assert frames[0]["status"] == "error"
            assert frames[0]["data"]["code"] == "E_MODEL_NOT_READY"
            assert frames[0]["data"]["http_status"] == 503
        finally:
            reg._providers.update(saved)

    def test_unknown_tool_yields_400_frame(self):
        from domain.models._internal.provider import registry as reg

        reg.register_provider("slonet-native", object())  # truthy sentinel
        try:
            resp = self.client.post(
                "/agents/loop", json={"task": "hello", "tools": ["no_such_tool"]}
            )
            frames = _sse_frames(resp.text)
            assert frames[0]["status"] == "error"
            assert frames[0]["data"]["code"] == "E_INVALID_REQUEST"
            assert "no_such_tool" in frames[0]["data"]["error"]
        finally:
            del reg._providers["slonet-native"]

    def test_happy_path_streams_tool_round_trip(self):
        from domain.models._internal.provider import registry as reg

        class FakeProvider:
            def __init__(self):
                self.calls = 0

            async def chat(self, messages, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    block = '{"name": "calculator", "args": {"expression": "2+2"}}'
                    return f"```tool_call\n{block}\n```"
                return "The answer is 4."

        reg.register_provider("slonet-native", FakeProvider())
        try:
            resp = self.client.post(
                "/agents/loop",
                json={"task": "what is 2+2?", "tools": ["calculator"], "max_iters": 3},
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            frames = _sse_frames(resp.text)
            phases = [f["phase"] for f in frames]
            assert phases[0] == "PLAN"
            assert "TOOL_CALL" in phases
            assert "TOOL_RESULT" in phases
            assert phases[-1] == "FINAL"
            final = frames[-1]
            assert final["status"] == "complete"
            assert final["data"]["text"] == "The answer is 4."
            call = next(f for f in frames if f["phase"] == "TOOL_CALL")
            assert call["data"]["name"] == "calculator"
            assert call["data"]["args"] == {"expression": "2+2"}
        finally:
            del reg._providers["slonet-native"]


class TestExecuteStreamVocabulary:
    """POST /agents/{agent_id}/execute/stream — event vocabulary pin.

    The web factory-floor reducer maps these frames to station states, so
    the ``event → (phase, status)`` table is a contract: pin it here.
    """

    def setup_method(self):
        self.client = get_test_client()

    @pytest.fixture
    def _stream_system(self):
        system = MagicMock()
        events = [
            {"event": "run_start", "agent_id": "a1"},
            {"event": "tool_start", "tool": "web_search"},
            {"event": "tool_result", "tool": "web_search", "ok": True},
            {"event": "final", "text": "done"},
            {"event": "run_end", "status": "completed"},
        ]

        async def _gen(**_kwargs):
            for ev in events:
                yield ev

        system.execute_stream = MagicMock(return_value=_gen())
        with patch("domain.agents.get_agent_system", return_value=system):
            yield system

    def test_event_vocabulary_is_pinned(self, _stream_system):
        with patch("fastapi.Request.is_disconnected", new=AsyncMock(return_value=False)):
            resp = self.client.post("/agents/a1/execute/stream", json={"request": "hi"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        frames = _sse_frames(resp.text)
        assert [(f["stream"], f["phase"], f["status"]) for f in frames] == [
            ("agent", "START", "working"),
            ("agent", "TOOL_CALL", "working"),
            ("agent", "TOOL_RESULT", "working"),
            ("agent", "FINAL", "working"),
            ("agent", "COMPLETE", "complete"),  # run_end
            ("agent", "COMPLETE", "complete"),  # sse_complete tail
        ]

    def test_error_event_maps_to_error_phase(self):
        system = MagicMock()

        async def _gen(**_kwargs):
            yield {"event": "error", "error": "boom"}

        system.execute_stream = MagicMock(return_value=_gen())
        with patch("domain.agents.get_agent_system", return_value=system):
            with patch("fastapi.Request.is_disconnected", new=AsyncMock(return_value=False)):
                resp = self.client.post("/agents/a1/execute/stream", json={"request": "hi"})
        frames = _sse_frames(resp.text)
        assert [(f["phase"], f["status"]) for f in frames] == [("ERROR", "error")]
        assert frames[0]["data"]["error"] == "boom"


class TestOrchestrateVocabulary:
    """POST /agents/orchestrate — phase vocabulary pin (floor reducer source)."""

    def setup_method(self):
        self.client = get_test_client()

    @pytest.fixture
    def _fake_orchestrator(self):
        class _Task:
            def __init__(self, tid, desc):
                self.id = tid
                self.description = desc
                self.assigned_agent = "worker"
                self.status = "pending"
                self.result = ""
                self.error = ""

            def to_dict(self):
                return {
                    "id": self.id,
                    "description": self.description,
                    "agent": self.assigned_agent,
                    "status": self.status,
                    "result_preview": self.result[:200],
                    "depends_on": [],
                }

        class _Orch:
            def __init__(self, **_kw):
                self.agents = {}

            async def _async_plan(self, _goal, _ctx):
                return [_Task("t1", "do the thing")]

            def _compute_levels(self, tasks):
                return [[t.id for t in tasks]]

            def _build_dep_context(self, _task, _task_map, _results):
                return ""

            async def _async_run_agent(self, _task, _goal, _dep):
                return "ok"

            async def _async_compose(self, _goal, _tasks):
                return "final answer"

        class _Store:
            def start(self, _goal, _ctx):
                return "run-1"

            def set_tasks(self, _rid, _tasks):
                return None

            def complete(self, _rid, response, tasks):
                return None

            def fail(self, _rid, _err):
                return None

        with patch("domain.agents.MultiAgentOrchestrator", _Orch):
            with patch("domain.agents.get_agent_run_store", lambda: _Store()):
                yield

    def test_phase_sequence_is_pinned(self, _fake_orchestrator):
        with patch("fastapi.Request.is_disconnected", new=AsyncMock(return_value=False)):
            resp = self.client.post("/agents/orchestrate", json={"goal": "test goal"})
        assert resp.status_code == 200
        frames = _sse_frames(resp.text)
        assert [(f["stream"], f["phase"], f["status"]) for f in frames] == [
            ("agent-orchestrate", "PLAN", "working"),
            ("agent-orchestrate", "PLAN", "success"),
            ("agent-orchestrate", "EXECUTE", "working"),  # starting execution
            ("agent-orchestrate", "EXECUTE", "working"),  # level dispatch
            ("agent-orchestrate", "EXECUTE", "success"),  # task done
            ("agent-orchestrate", "COMPOSE", "working"),
            ("agent-orchestrate", "COMPLETE", "complete"),
        ]

    def test_plan_failure_is_pinned(self):
        class _Orch:
            def __init__(self, **_kw):
                self.agents = {}

            async def _async_plan(self, _goal, _ctx):
                return []

        class _Store:
            def start(self, _goal, _ctx):
                return "run-1"

            def fail(self, _rid, _err):
                return None

        with patch("domain.agents.MultiAgentOrchestrator", _Orch):
            with patch("domain.agents.get_agent_run_store", lambda: _Store()):
                with patch("fastapi.Request.is_disconnected", new=AsyncMock(return_value=False)):
                    resp = self.client.post("/agents/orchestrate", json={"goal": "nope"})
        frames = _sse_frames(resp.text)
        assert [(f["phase"], f["status"]) for f in frames] == [
            ("PLAN", "working"),
            ("PLAN", "error"),
        ]
        assert frames[-1]["data"]["error"] == "Could not plan this goal"
