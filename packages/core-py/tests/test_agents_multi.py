"""Tests for agents.multi — MultiAgentOrchestrator."""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from domains.agents.multi import (
    SpecializedAgent,
    DEFAULT_AGENTS,
    TaskStatus,
    AgentTask,
    MultiAgentOrchestrator,
    get_orchestrator,
    reset_orchestrator,
)


# ── SpecializedAgent ───────────────────────────────────────────────────────


class TestSpecializedAgent:

    def test_init(self):
        a = SpecializedAgent(name="R", role="research", system_prompt="Do research")
        assert a.name == "R"
        assert a.role == "research"
        assert a.tools == []

    def test_init_with_tools(self):
        a = SpecializedAgent(name="C", role="code", system_prompt="Code", tools=["exec"])
        assert a.tools == ["exec"]

    def test_to_dict(self):
        a = SpecializedAgent(name="W", role="write", system_prompt="A" * 100)
        d = a.to_dict()
        assert d["name"] == "W"
        assert d["role"] == "write"
        assert "..." in d["description"]


# ── TaskStatus ──────────────────────────────────────────────────────────────


class TestTaskStatus:

    def test_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"


# ── AgentTask ───────────────────────────────────────────────────────────────


class TestAgentTask:

    def test_init(self):
        t = AgentTask(id="1", description="do stuff", assigned_agent="researcher")
        assert t.status == TaskStatus.PENDING
        assert t.depends_on == []
        assert t.result == ""

    def test_to_dict(self):
        t = AgentTask(id="1", description="task1", assigned_agent="writer")
        d = t.to_dict()
        assert d["id"] == "1"
        assert d["agent"] == "writer"
        assert d["status"] == "pending"
        assert d["result_preview"] == ""

    def test_to_dict_result_preview(self):
        t = AgentTask(id="1", description="t", assigned_agent="r", result="A" * 200)
        d = t.to_dict()
        assert len(d["result_preview"]) == 100


# ── DEFAULT_AGENTS ──────────────────────────────────────────────────────────


class TestDefaultAgents:

    def test_has_all(self):
        assert "researcher" in DEFAULT_AGENTS
        assert "writer" in DEFAULT_AGENTS
        assert "coder" in DEFAULT_AGENTS
        assert "critic" in DEFAULT_AGENTS

    def test_types(self):
        for agent in DEFAULT_AGENTS.values():
            assert isinstance(agent, SpecializedAgent)
            assert len(agent.system_prompt) > 20


# ── MultiAgentOrchestrator ─────────────────────────────────────────────────


class TestOrchestrator:

    def test_init(self):
        orch = MultiAgentOrchestrator()
        assert len(orch.agents) >= 4

    def test_init_custom_agents(self):
        custom = {"my_agent": SpecializedAgent(name="M", role="mine", system_prompt="prompt")}
        orch = MultiAgentOrchestrator(agents=custom)
        assert "my_agent" in orch.agents

    def test_list_agents(self):
        orch = MultiAgentOrchestrator()
        agents = orch.list_agents()
        assert len(agents) >= 4
        assert all("name" in a for a in agents)

    def test_get_agent(self):
        orch = MultiAgentOrchestrator()
        a = orch.get_agent("researcher")
        assert a is not None
        assert a.name == "Researcher"

    def test_get_agent_missing(self):
        orch = MultiAgentOrchestrator()
        assert orch.get_agent("nonexistent") is None

    def test_simple_plan(self):
        orch = MultiAgentOrchestrator()
        tasks = orch._simple_plan("test goal")
        assert len(tasks) == 2
        assert tasks[0].assigned_agent == "researcher"
        assert tasks[1].assigned_agent == "writer"
        assert tasks[1].depends_on == ["1"]


# ── _compute_levels ─────────────────────────────────────────────────────────


class TestComputeLevels:

    def test_no_deps(self):
        orch = MultiAgentOrchestrator()
        tasks = [
            AgentTask(id="1", description="a", assigned_agent="researcher"),
            AgentTask(id="2", description="b", assigned_agent="writer"),
        ]
        levels = orch._compute_levels(tasks)
        assert len(levels) == 1
        assert set(levels[0]) == {"1", "2"}

    def test_chain(self):
        orch = MultiAgentOrchestrator()
        tasks = [
            AgentTask(id="1", description="a", assigned_agent="researcher"),
            AgentTask(id="2", description="b", assigned_agent="writer", depends_on=["1"]),
        ]
        levels = orch._compute_levels(tasks)
        assert len(levels) == 2
        assert levels[0] == ["1"]
        assert levels[1] == ["2"]

    def test_diamond(self):
        orch = MultiAgentOrchestrator()
        tasks = [
            AgentTask(id="1", description="a", assigned_agent="researcher"),
            AgentTask(id="2", description="b", assigned_agent="writer", depends_on=["1"]),
            AgentTask(id="3", description="c", assigned_agent="critic", depends_on=["1"]),
            AgentTask(id="4", description="d", assigned_agent="coder", depends_on=["2", "3"]),
        ]
        levels = orch._compute_levels(tasks)
        assert len(levels) == 3
        assert set(levels[0]) == {"1"}
        assert set(levels[1]) == {"2", "3"}
        assert levels[2] == ["4"]


# ── _build_dep_context ─────────────────────────────────────────────────────


class TestBuildDepContext:

    def test_no_deps(self):
        orch = MultiAgentOrchestrator()
        task = AgentTask(id="1", description="a", assigned_agent="researcher")
        ctx = orch._build_dep_context(task, {}, {})
        assert ctx == ""

    def test_with_deps(self):
        orch = MultiAgentOrchestrator()
        dep = AgentTask(id="1", description="research", assigned_agent="researcher")
        task = AgentTask(id="2", description="write", assigned_agent="writer", depends_on=["1"])
        results = {"1": "found info"}
        ctx = orch._build_dep_context(task, {"1": dep}, results)
        assert "found info" in ctx
        assert "researcher" in ctx


# ── execute (mocked) ────────────────────────────────────────────────────────


class TestExecute:

    def test_execute_plan_failure(self):
        orch = MultiAgentOrchestrator()
        # Empty string triggers _simple_plan fallback, not "Could not plan"
        # To get empty plan, return a non-JSON, non-regex-matchable response
        with patch.object(orch, "_generate", return_value="no plan here"):
            result = orch.execute("do something")
            # _simple_plan always returns tasks, so we get a result
            assert "response" in result

    def test_execute_simple(self):
        orch = MultiAgentOrchestrator()
        responses = [
            json.dumps([{"id": "1", "description": "r1", "agent": "researcher", "depends_on": []}]),
            "research output",
            "final summary",
        ]
        call_count = 0

        def mock_generate(prompt, max_tokens=200):
            nonlocal call_count
            r = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return r

        with patch.object(orch, "_generate", side_effect=mock_generate):
            result = orch.execute("test goal")
            assert "response" in result
            assert len(result["tasks"]) == 1
            assert result["tasks"][0]["status"] == "completed"

    def test_execute_all_failed(self):
        orch = MultiAgentOrchestrator()
        plan = json.dumps([{"id": "1", "description": "t", "agent": "researcher", "depends_on": []}])

        call_count = 0

        def mock_generate(prompt, max_tokens=200):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return plan
            raise RuntimeError("LLM failed")

        with patch.object(orch, "_generate", side_effect=mock_generate):
            result = orch.execute("goal")
            assert result["response"] == "All agents failed."

    def test_execute_unknown_agent(self):
        orch = MultiAgentOrchestrator()
        plan = json.dumps([{"id": "1", "description": "t", "agent": "unknown", "depends_on": []}])
        call_count = 0

        def mock_generate(prompt, max_tokens=200):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return plan
            return "output"

        with patch.object(orch, "_generate", side_effect=mock_generate):
            result = orch.execute("goal")
            assert result["tasks"][0]["status"] == "completed"


# ── _plan edge cases ────────────────────────────────────────────────────────


class TestPlanEdgeCases:

    def test_plan_json_decode_error(self):
        orch = MultiAgentOrchestrator()
        with patch.object(orch, "_generate", return_value="not json"):
            tasks = orch._plan("goal", "")
            assert len(tasks) == 2
            assert tasks[0].assigned_agent == "researcher"

    def test_plan_regex_fallback(self):
        orch = MultiAgentOrchestrator()
        # Use a simple plan without nested brackets so regex works
        resp = 'Plan: [{"id":"1","description":"t","agent":"writer"}]'
        with patch.object(orch, "_generate", return_value=resp):
            tasks = orch._plan("goal", "")
            assert len(tasks) == 1
            assert tasks[0].assigned_agent == "writer"

    def test_plan_unknown_agent_defaults_to_researcher(self):
        orch = MultiAgentOrchestrator()
        resp = json.dumps([{"id": "1", "description": "t", "agent": "nonexistent"}])
        with patch.object(orch, "_generate", return_value=resp):
            tasks = orch._plan("goal", "")
            assert tasks[0].assigned_agent == "researcher"

    def test_plan_string_depends_on(self):
        orch = MultiAgentOrchestrator()
        resp = json.dumps([{"id": "1", "description": "t", "agent": "writer", "depends_on": "0"}])
        with patch.object(orch, "_generate", return_value=resp):
            tasks = orch._plan("goal", "")
            assert tasks[0].depends_on == ["0"]

    def test_plan_empty_list_fallback(self):
        orch = MultiAgentOrchestrator()
        with patch.object(orch, "_generate", return_value="[]"):
            tasks = orch._plan("goal", "")
            assert len(tasks) == 2


# ── _compose ────────────────────────────────────────────────────────────────


class TestCompose:

    def test_compose_no_completed(self):
        orch = MultiAgentOrchestrator()
        tasks = [AgentTask(id="1", description="t", assigned_agent="researcher", status=TaskStatus.FAILED)]
        result = orch._compose("goal", tasks)
        assert result == "All agents failed."

    def test_compose_with_completed(self):
        orch = MultiAgentOrchestrator()
        tasks = [
            AgentTask(id="1", description="t", assigned_agent="researcher",
                      result="research done", status=TaskStatus.COMPLETED),
        ]
        with patch.object(orch, "_generate", return_value="synthesized response"):
            result = orch._compose("goal", tasks)
            assert result == "synthesized response"


# ── _run_agent ──────────────────────────────────────────────────────────────


class TestRunAgent:

    def test_run_agent_no_agent(self):
        orch = MultiAgentOrchestrator()
        task = AgentTask(id="1", description="t", assigned_agent="nonexistent")
        result = orch._run_agent(task, "goal", "")
        assert "No agent" in result

    def test_run_agent_success(self):
        orch = MultiAgentOrchestrator()
        task = AgentTask(id="1", description="t", assigned_agent="researcher")
        with patch.object(orch, "_generate", return_value="result text"):
            result = orch._run_agent(task, "goal", "")
            assert result == "result text"


# ── Singleton ───────────────────────────────────────────────────────────────


class TestSingleton:

    def test_get_orchestrator(self):
        reset_orchestrator()
        orch = get_orchestrator()
        assert isinstance(orch, MultiAgentOrchestrator)
        assert get_orchestrator() is orch

    def test_reset(self):
        reset_orchestrator()
        orch1 = get_orchestrator()
        reset_orchestrator()
        orch2 = get_orchestrator()
        assert orch1 is not orch2


# ── async_execute (mocked) ──────────────────────────────────────────────────


class TestAsyncExecute:

    @pytest.mark.asyncio
    async def test_async_execute_no_plan(self):
        orch = MultiAgentOrchestrator()
        with patch.object(orch, "_async_generate", new_callable=AsyncMock, return_value="no plan"):
            result = await orch.async_execute("goal")
            # _simple_plan always returns tasks
            assert "response" in result

    @pytest.mark.asyncio
    async def test_async_execute_simple(self):
        orch = MultiAgentOrchestrator()
        plan = json.dumps([{"id": "1", "description": "t", "agent": "researcher", "depends_on": []}])
        responses = [plan, "agent output", "final"]
        call_count = 0

        async def mock_async_generate(prompt, max_tokens=200):
            nonlocal call_count
            r = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return r

        with patch.object(orch, "_async_generate", side_effect=mock_async_generate):
            result = await orch.async_execute("goal")
            assert "response" in result
            assert len(result["tasks"]) == 1
