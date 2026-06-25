"""Tests for the multi-agent orchestration system."""

from typing import Any, Dict, List
import pytest

from domains.agents.multi import (
    MultiAgentOrchestrator,
    SpecializedAgent,
    AgentTask,
    TaskStatus,
    DEFAULT_AGENTS,
    get_orchestrator,
    reset_orchestrator,
)


# ── SpecializedAgent tests ────────────────────────────────────────────


class TestSpecializedAgent:
    def test_create_minimal(self):
        a = SpecializedAgent(name="test", role="testing", system_prompt="You test.")
        assert a.name == "test"
        assert a.role == "testing"
        assert a.system_prompt == "You test."
        assert a.tools == []

    def test_create_with_tools(self):
        a = SpecializedAgent(name="t", role="r", system_prompt="p", tools=["code", "search"])
        assert a.tools == ["code", "search"]

    def test_to_dict(self):
        a = SpecializedAgent(name="TestBot", role="testing", system_prompt="You are a test bot.")
        d = a.to_dict()
        assert d["name"] == "TestBot"
        assert d["role"] == "testing"
        assert "description" in d

    def test_default_agents_present(self):
        assert "researcher" in DEFAULT_AGENTS
        assert "writer" in DEFAULT_AGENTS
        assert "coder" in DEFAULT_AGENTS
        assert "critic" in DEFAULT_AGENTS

    def test_default_agent_has_name_and_role(self):
        for name, agent in DEFAULT_AGENTS.items():
            assert agent.name
            assert agent.role
            assert agent.system_prompt


# ── AgentTask tests ───────────────────────────────────────────────────


class TestAgentTask:
    def test_create(self):
        t = AgentTask(id="1", description="do thing", assigned_agent="researcher")
        assert t.id == "1"
        assert t.description == "do thing"
        assert t.assigned_agent == "researcher"
        assert t.status == TaskStatus.PENDING
        assert t.result == ""
        assert t.error == ""

    def test_to_dict_pending(self):
        t = AgentTask(id="2", description="task", assigned_agent="writer")
        d = t.to_dict()
        assert d["id"] == "2"
        assert d["agent"] == "writer"
        assert d["status"] == TaskStatus.PENDING

    def test_to_dict_with_result(self):
        t = AgentTask(id="3", description="write code", assigned_agent="coder")
        t.status = TaskStatus.COMPLETED
        t.result = "def hello(): pass"
        d = t.to_dict()
        assert d["status"] == TaskStatus.COMPLETED
        assert "def hello" in d["result_preview"]


# ── MultiAgentOrchestrator tests ──────────────────────────────────────


class TestMultiAgentOrchestrator:
    def test_init_default_agents(self):
        orch = MultiAgentOrchestrator()
        assert len(orch.agents) == 4
        assert "researcher" in orch.agents

    def test_init_custom_agents(self):
        custom = {
            "bot": SpecializedAgent(name="Bot", role="bot", system_prompt="beep"),
        }
        orch = MultiAgentOrchestrator(agents=custom)
        assert list(orch.agents.keys()) == ["bot"]

    def test_list_agents(self):
        orch = MultiAgentOrchestrator()
        agents = orch.list_agents()
        assert len(agents) == 4
        names = [a["name"] for a in agents]
        assert "Researcher" in names
        assert "Writer" in names

    def test_get_agent_found(self):
        orch = MultiAgentOrchestrator()
        a = orch.get_agent("researcher")
        assert a is not None
        assert a.name == "Researcher"

    def test_get_agent_not_found(self):
        orch = MultiAgentOrchestrator()
        a = orch.get_agent("nonexistent")
        assert a is None

    def test_simple_plan_fallback(self):
        orch = MultiAgentOrchestrator()
        tasks = orch._simple_plan("test goal")
        assert len(tasks) == 2
        assert tasks[0].assigned_agent == "researcher"
        assert tasks[1].assigned_agent == "writer"

    def test_plan_fallback_when_llm_returns_junk(self):
        """When LLM returns non-JSON, should fall back to simple plan."""
        orch = MultiAgentOrchestrator()
        # Mock _generate to return junk
        orch._generate = lambda prompt, max_tokens=200: "this is not json at all"
        tasks = orch._plan("some goal", "")
        assert len(tasks) == 2
        assert tasks[0].assigned_agent == "researcher"

    def test_plan_from_llm_json(self):
        orch = MultiAgentOrchestrator()
        orch._generate = lambda prompt, max_tokens=200: (
            '[{"id": "1", "description": "research topic", "agent": "researcher"},'
            '{"id": "2", "description": "write output", "agent": "writer"}]'
        )
        tasks = orch._plan("test goal", "")
        assert len(tasks) == 2
        assert tasks[0].assigned_agent == "researcher"
        assert tasks[1].assigned_agent == "writer"

    def test_plan_rejects_unknown_agent(self):
        orch = MultiAgentOrchestrator()
        orch._generate = lambda prompt, max_tokens=200: (
            '[{"id": "1", "description": "hack", "agent": "hacker"}]'
        )
        tasks = orch._plan("test", "")
        # Unknown agent should default to researcher
        assert tasks[0].assigned_agent == "researcher"

    def test_run_agent_returns_generated_text(self):
        orch = MultiAgentOrchestrator()
        orch._generate = lambda prompt, max_tokens=300: "Research findings here"
        task = AgentTask(id="1", description="research", assigned_agent="researcher")
        result = orch._run_agent(task, "test goal", "")
        assert result == "Research findings here"

    def test_run_agent_unknown_agent(self):
        orch = MultiAgentOrchestrator()
        task = AgentTask(id="1", description="x", assigned_agent="ghost")
        result = orch._run_agent(task, "goal", "")
        assert "No agent" in result

    def test_run_agent_includes_system_prompt(self):
        recorded = []
        orch = MultiAgentOrchestrator()
        orch._generate = lambda prompt, max_tokens=300: recorded.append(prompt) or "ok"
        task = AgentTask(id="1", description="test task", assigned_agent="coder")
        orch._run_agent(task, "goal", "prev work")
        assert recorded
        assert "test task" in recorded[0]
        assert "prev work" in recorded[0]
        assert "coding" in recorded[0].lower()

    def test_compose_no_completed_tasks(self):
        orch = MultiAgentOrchestrator()
        tasks = [AgentTask(id="1", description="fail", assigned_agent="researcher")]
        result = orch._compose("goal", tasks)
        assert "All agents failed" in result

    def test_execute_returns_tasks(self):
        orch = MultiAgentOrchestrator()
        orch._generate = lambda prompt, max_tokens=200: (
            '[{"id": "1", "description": "research", "agent": "researcher"}]'
        )
        # For run_agent
        original_run = orch._run_agent
        orch._run_agent = lambda task, goal, ctx: "results here"
        # For compose
        orch._compose = lambda goal, tasks: "final response"

        result = orch.execute("test")
        assert "response" in result
        assert "tasks" in result
        assert result["response"] == "final response"

    def test_execute_failed_task_reported(self):
        orch = MultiAgentOrchestrator()
        orch._generate = lambda prompt, max_tokens=200: (
            '[{"id": "1", "description": "research", "agent": "researcher"}]'
        )
        def failing_run(task, goal, ctx):
            raise RuntimeError("API down")
        orch._run_agent = failing_run
        orch._compose = lambda goal, tasks: "fallback"

        result = orch.execute("test")
        tasks = result["tasks"]
        assert tasks[0]["status"] == "failed"


# ── Parallel execution tests ──────────────────────────────────────────


class TestParallelExecution:
    def test_compute_levels_all_independent(self):
        orch = MultiAgentOrchestrator()
        tasks = [
            AgentTask(id="1", description="a", assigned_agent="researcher", depends_on=[]),
            AgentTask(id="2", description="b", assigned_agent="writer", depends_on=[]),
        ]
        levels = orch._compute_levels(tasks)
        assert len(levels) == 1
        assert set(levels[0]) == {"1", "2"}

    def test_compute_levels_linear_chain(self):
        orch = MultiAgentOrchestrator()
        tasks = [
            AgentTask(id="1", description="a", assigned_agent="researcher", depends_on=[]),
            AgentTask(id="2", description="b", assigned_agent="writer", depends_on=["1"]),
            AgentTask(id="3", description="c", assigned_agent="coder", depends_on=["2"]),
        ]
        levels = orch._compute_levels(tasks)
        assert len(levels) == 3
        assert levels[0] == ["1"]
        assert levels[1] == ["2"]
        assert levels[2] == ["3"]

    def test_compute_levels_diamond(self):
        orch = MultiAgentOrchestrator()
        tasks = [
            AgentTask(id="1", description="root", assigned_agent="researcher", depends_on=[]),
            AgentTask(id="2", description="branch_a", assigned_agent="writer", depends_on=["1"]),
            AgentTask(id="3", description="branch_b", assigned_agent="coder", depends_on=["1"]),
            AgentTask(id="4", description="merge", assigned_agent="critic", depends_on=["2", "3"]),
        ]
        levels = orch._compute_levels(tasks)
        assert len(levels) == 3
        assert levels[0] == ["1"]
        assert set(levels[1]) == {"2", "3"}
        assert levels[2] == ["4"]

    def test_compute_levels_circular_deps_broken(self):
        orch = MultiAgentOrchestrator()
        tasks = [
            AgentTask(id="1", description="a", assigned_agent="researcher", depends_on=["2"]),
            AgentTask(id="2", description="b", assigned_agent="writer", depends_on=["1"]),
        ]
        levels = orch._compute_levels(tasks)
        # Can't resolve — runs all remaining in a single level
        assert len(levels) == 1
        assert set(levels[0]) == {"1", "2"}

    def test_build_dep_context_with_results(self):
        orch = MultiAgentOrchestrator()
        task = AgentTask(id="2", description="write", assigned_agent="writer", depends_on=["1"])
        task_map = {
            "1": AgentTask(id="1", description="research", assigned_agent="researcher"),
            "2": task,
        }
        results = {"1": "found important data"}
        ctx = orch._build_dep_context(task, task_map, results)
        assert "researcher" in ctx
        assert "found important data" in ctx

    def test_build_dep_context_no_completed(self):
        orch = MultiAgentOrchestrator()
        task = AgentTask(id="1", description="research", assigned_agent="researcher", depends_on=[])
        ctx = orch._build_dep_context(task, {}, {})
        assert ctx == ""

    def test_build_dep_context_missing_dep(self):
        orch = MultiAgentOrchestrator()
        task = AgentTask(id="2", description="write", assigned_agent="writer", depends_on=["ghost"])
        task_map = {"2": task}
        ctx = orch._build_dep_context(task, task_map, {})
        assert ctx == ""


# ── Singleton tests ───────────────────────────────────────────────────


class TestOrchestratorSingleton:
    def teardown_method(self):
        reset_orchestrator()

    def test_singleton(self):
        o1 = get_orchestrator()
        o2 = get_orchestrator()
        assert o1 is o2

    def test_reset(self):
        o1 = get_orchestrator()
        reset_orchestrator()
        o2 = get_orchestrator()
        assert o1 is not o2
