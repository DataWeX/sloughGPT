"""Tests for domains.agents.multi — SpecializedAgent, AgentTask, TaskStatus,
MultiAgentOrchestrator._compute_levels, _simple_plan, _build_dep_context.

Covers: dataclass creation, to_dict, topological sort into parallel levels,
dependency context building, singleton access.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.agents.multi import (
    SpecializedAgent,
    AgentTask,
    TaskStatus,
    MultiAgentOrchestrator,
    DEFAULT_AGENTS,
    get_orchestrator,
    reset_orchestrator,
)


# ── SpecializedAgent ─────────────────────────────────────────────────

class TestSpecializedAgent:
    def test_creation(self):
        a = SpecializedAgent(name="Test", role="test things", system_prompt="Be a tester.")
        assert a.name == "Test"
        assert a.role == "test things"
        assert a.tools == []

    def test_to_dict(self):
        a = SpecializedAgent(name="R", role="research", system_prompt="You research stuff.")
        d = a.to_dict()
        assert d["name"] == "R"
        assert d["role"] == "research"
        assert "description" in d

    def test_default_agents_exist(self):
        assert "researcher" in DEFAULT_AGENTS
        assert "writer" in DEFAULT_AGENTS
        assert "coder" in DEFAULT_AGENTS
        assert "critic" in DEFAULT_AGENTS

    def test_to_dict_truncates_prompt(self):
        long_prompt = "x" * 200
        a = SpecializedAgent(name="A", role="r", system_prompt=long_prompt)
        d = a.to_dict()
        assert len(d["description"]) <= 83  # 80 chars + "..."

    def test_tools_default_empty(self):
        a = SpecializedAgent(name="A", role="r", system_prompt="p")
        assert a.tools == []

    def test_tools_custom(self):
        a = SpecializedAgent(name="A", role="r", system_prompt="p", tools=["web_search", "memory"])
        assert len(a.tools) == 2
        assert "web_search" in a.tools

    def test_default_agents_count(self):
        assert len(DEFAULT_AGENTS) == 4

    def test_default_agent_names(self):
        names = [a.name for a in DEFAULT_AGENTS.values()]
        assert "Researcher" in names
        assert "Writer" in names
        assert "Coder" in names
        assert "Critic" in names

    def test_default_agent_roles(self):
        roles = {k: a.role for k, a in DEFAULT_AGENTS.items()}
        assert "research" in roles["researcher"]
        assert "write" in roles["writer"]
        assert "code" in roles["coder"]
        assert "review" in roles["critic"]

    def test_default_agent_has_tools(self):
        for agent in DEFAULT_AGENTS.values():
            assert isinstance(agent.tools, list)

    def test_default_agent_system_prompts_nonempty(self):
        for agent in DEFAULT_AGENTS.values():
            assert len(agent.system_prompt) > 0

    def test_to_dict_description_format(self):
        a = SpecializedAgent(name="A", role="r", system_prompt="Short prompt.")
        d = a.to_dict()
        assert d["description"].endswith("...")

    def test_creation_with_empty_role(self):
        a = SpecializedAgent(name="A", role="", system_prompt="p")
        assert a.role == ""

    def test_creation_with_empty_prompt(self):
        a = SpecializedAgent(name="A", role="r", system_prompt="")
        assert a.system_prompt == ""


# ── AgentTask ────────────────────────────────────────────────────────

class TestAgentTask:
    def test_creation(self):
        t = AgentTask(id="1", description="do stuff", assigned_agent="writer")
        assert t.id == "1"
        assert t.status == TaskStatus.PENDING
        assert t.depends_on == []

    def test_to_dict(self):
        t = AgentTask(id="2", description="research", assigned_agent="researcher",
                      result="Found data", depends_on=["1"])
        d = t.to_dict()
        assert d["id"] == "2"
        assert d["agent"] == "researcher"
        assert d["status"] == TaskStatus.PENDING
        assert "depends_on" in d

    def test_to_dict_result_preview(self):
        t = AgentTask(id="1", description="d", assigned_agent="a", result="x" * 200)
        d = t.to_dict()
        assert len(d["result_preview"]) <= 100

    def test_to_dict_empty_result(self):
        t = AgentTask(id="1", description="d", assigned_agent="a")
        d = t.to_dict()
        assert d["result_preview"] == ""

    def test_status_default_pending(self):
        t = AgentTask(id="1", description="d", assigned_agent="a")
        assert t.status == TaskStatus.PENDING

    def test_depends_on_default_empty(self):
        t = AgentTask(id="1", description="d", assigned_agent="a")
        assert t.depends_on == []

    def test_depends_on_custom(self):
        t = AgentTask(id="1", description="d", assigned_agent="a", depends_on=["x", "y"])
        assert t.depends_on == ["x", "y"]

    def test_context_default_empty(self):
        t = AgentTask(id="1", description="d", assigned_agent="a")
        assert t.context == ""

    def test_error_default_empty(self):
        t = AgentTask(id="1", description="d", assigned_agent="a")
        assert t.error == ""

    def test_result_default_empty(self):
        t = AgentTask(id="1", description="d", assigned_agent="a")
        assert t.result == ""

    def test_to_dict_has_all_keys(self):
        t = AgentTask(id="1", description="d", assigned_agent="a")
        d = t.to_dict()
        assert "id" in d
        assert "description" in d
        assert "agent" in d
        assert "status" in d
        assert "result_preview" in d
        assert "depends_on" in d

    def test_empty_id(self):
        t = AgentTask(id="", description="d", assigned_agent="a")
        assert t.id == ""

    def test_long_description(self):
        t = AgentTask(id="1", description="d" * 1000, assigned_agent="a")
        assert len(t.description) == 1000


# ── TaskStatus ───────────────────────────────────────────────────────

class TestTaskStatus:
    def test_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"

    def test_all_values_are_strings(self):
        assert isinstance(TaskStatus.PENDING, str)
        assert isinstance(TaskStatus.IN_PROGRESS, str)
        assert isinstance(TaskStatus.COMPLETED, str)
        assert isinstance(TaskStatus.FAILED, str)

    def test_unique_values(self):
        values = [TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED, TaskStatus.FAILED]
        assert len(values) == len(set(values))

    def test_count(self):
        values = [TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED, TaskStatus.FAILED]
        assert len(values) == 4

    def test_can_be_used_in_set(self):
        s = {TaskStatus.PENDING, TaskStatus.IN_PROGRESS}
        assert len(s) == 2

    def test_can_be_compared_directly(self):
        assert TaskStatus.PENDING == "pending"

    def test_can_be_used_as_dict_key(self):
        d = {TaskStatus.PENDING: 1}
        assert d["pending"] == 1


# ── MultiAgentOrchestrator (pure logic) ─────────────────────────────

class TestOrchestratorLevels:
    def test_independent_tasks(self):
        orch = MultiAgentOrchestrator(agents={})
        tasks = [
            AgentTask(id="1", description="a", assigned_agent="x", depends_on=[]),
            AgentTask(id="2", description="b", assigned_agent="y", depends_on=[]),
        ]
        levels = orch._compute_levels(tasks)
        assert len(levels) == 1
        assert set(levels[0]) == {"1", "2"}

    def test_linear_chain(self):
        orch = MultiAgentOrchestrator(agents={})
        tasks = [
            AgentTask(id="1", description="a", assigned_agent="x", depends_on=[]),
            AgentTask(id="2", description="b", assigned_agent="y", depends_on=["1"]),
            AgentTask(id="3", description="c", assigned_agent="z", depends_on=["2"]),
        ]
        levels = orch._compute_levels(tasks)
        assert len(levels) == 3
        assert levels[0] == ["1"]
        assert levels[1] == ["2"]
        assert levels[2] == ["3"]

    def test_mixed_dependencies(self):
        orch = MultiAgentOrchestrator(agents={})
        tasks = [
            AgentTask(id="1", description="a", assigned_agent="x", depends_on=[]),
            AgentTask(id="2", description="b", assigned_agent="y", depends_on=[]),
            AgentTask(id="3", description="c", assigned_agent="z", depends_on=["1", "2"]),
        ]
        levels = orch._compute_levels(tasks)
        assert len(levels) == 2
        assert set(levels[0]) == {"1", "2"}
        assert levels[1] == ["3"]

    def test_single_task(self):
        orch = MultiAgentOrchestrator(agents={})
        tasks = [AgentTask(id="1", description="a", assigned_agent="x", depends_on=[])]
        levels = orch._compute_levels(tasks)
        assert len(levels) == 1
        assert levels[0] == ["1"]

    def test_empty_tasks(self):
        orch = MultiAgentOrchestrator(agents={})
        levels = orch._compute_levels([])
        assert levels == []

    def test_diamond_dependency(self):
        orch = MultiAgentOrchestrator(agents={})
        tasks = [
            AgentTask(id="1", description="a", assigned_agent="x", depends_on=[]),
            AgentTask(id="2", description="b", assigned_agent="x", depends_on=["1"]),
            AgentTask(id="3", description="c", assigned_agent="x", depends_on=["1"]),
            AgentTask(id="4", description="d", assigned_agent="x", depends_on=["2", "3"]),
        ]
        levels = orch._compute_levels(tasks)
        assert len(levels) == 3
        assert set(levels[0]) == {"1"}
        assert set(levels[1]) == {"2", "3"}
        assert levels[2] == ["4"]

    def test_circular_dependency_fallback(self):
        orch = MultiAgentOrchestrator(agents={})
        tasks = [
            AgentTask(id="1", description="a", assigned_agent="x", depends_on=["2"]),
            AgentTask(id="2", description="b", assigned_agent="y", depends_on=["1"]),
        ]
        levels = orch._compute_levels(tasks)
        assert len(levels) == 1
        assert set(levels[0]) == {"1", "2"}

    def test_three_way_parallel(self):
        orch = MultiAgentOrchestrator(agents={})
        tasks = [
            AgentTask(id="1", description="a", assigned_agent="x", depends_on=[]),
            AgentTask(id="2", description="b", assigned_agent="x", depends_on=[]),
            AgentTask(id="3", description="c", assigned_agent="x", depends_on=[]),
        ]
        levels = orch._compute_levels(tasks)
        assert len(levels) == 1
        assert set(levels[0]) == {"1", "2", "3"}


class TestOrchestratorBuildDepContext:
    def test_with_deps(self):
        orch = MultiAgentOrchestrator(agents={})
        task = AgentTask(id="2", description="b", assigned_agent="writer", depends_on=["1"])
        task_map = {
            "1": AgentTask(id="1", description="a", assigned_agent="researcher"),
            "2": task,
        }
        results_ctx = {"1": "Research findings here"}
        ctx = orch._build_dep_context(task, task_map, results_ctx)
        assert "Research findings here" in ctx
        assert "researcher" in ctx

    def test_no_deps(self):
        orch = MultiAgentOrchestrator(agents={})
        task = AgentTask(id="1", description="a", assigned_agent="x")
        ctx = orch._build_dep_context(task, {}, {})
        assert ctx == ""

    def test_multiple_deps(self):
        orch = MultiAgentOrchestrator(agents={})
        task = AgentTask(id="3", description="c", assigned_agent="z", depends_on=["1", "2"])
        task_map = {
            "1": AgentTask(id="1", description="a", assigned_agent="researcher"),
            "2": AgentTask(id="2", description="b", assigned_agent="coder"),
            "3": task,
        }
        results_ctx = {"1": "Research done", "2": "Code done"}
        ctx = orch._build_dep_context(task, task_map, results_ctx)
        assert "Research done" in ctx
        assert "Code done" in ctx
        assert "researcher" in ctx
        assert "coder" in ctx

    def test_dep_not_in_results(self):
        orch = MultiAgentOrchestrator(agents={})
        task = AgentTask(id="2", description="b", assigned_agent="x", depends_on=["1"])
        task_map = {
            "1": AgentTask(id="1", description="a", assigned_agent="y"),
            "2": task,
        }
        ctx = orch._build_dep_context(task, task_map, {})
        assert ctx == ""

    def test_dep_not_in_task_map(self):
        orch = MultiAgentOrchestrator(agents={})
        task = AgentTask(id="2", description="b", assigned_agent="x", depends_on=["999"])
        ctx = orch._build_dep_context(task, {"2": task}, {"999": "data"})
        assert ctx == ""

    def test_dep_result_with_error(self):
        orch = MultiAgentOrchestrator(agents={})
        task = AgentTask(id="2", description="b", assigned_agent="x", depends_on=["1"])
        task_map = {
            "1": AgentTask(id="1", description="a", assigned_agent="y"),
            "2": task,
        }
        ctx = orch._build_dep_context(task, task_map, {"1": "[error: timeout]"})
        assert "[error: timeout]" in ctx


class TestOrchestratorSimplePlan:
    def test_simple_plan(self):
        orch = MultiAgentOrchestrator(agents={})
        tasks = orch._simple_plan("test goal")
        assert len(tasks) == 2
        assert tasks[0].assigned_agent == "researcher"
        assert tasks[1].assigned_agent == "writer"
        assert tasks[1].depends_on == ["1"]

    def test_simple_plan_ids(self):
        orch = MultiAgentOrchestrator(agents={})
        tasks = orch._simple_plan("goal")
        assert tasks[0].id == "1"
        assert tasks[1].id == "2"

    def test_simple_plan_descriptions_contain_goal(self):
        orch = MultiAgentOrchestrator(agents={})
        tasks = orch._simple_plan("my goal")
        assert "my goal" in tasks[0].description

    def test_simple_plan_writing_depends_on_research(self):
        orch = MultiAgentOrchestrator(agents={})
        tasks = orch._simple_plan("goal")
        assert tasks[1].depends_on == [tasks[0].id]


class TestOrchestratorListAgents:
    def test_list_agents(self):
        orch = MultiAgentOrchestrator()
        agents = orch.list_agents()
        assert len(agents) >= 4

    def test_get_agent(self):
        orch = MultiAgentOrchestrator()
        a = orch.get_agent("researcher")
        assert a is not None
        assert a.name == "Researcher"

    def test_get_missing_agent(self):
        orch = MultiAgentOrchestrator()
        assert orch.get_agent("nonexistent") is None

    def test_list_agents_returns_dicts(self):
        orch = MultiAgentOrchestrator()
        agents = orch.list_agents()
        for agent in agents:
            assert "name" in agent
            assert "role" in agent
            assert "description" in agent

    def test_get_agent_returns_specialized_agent(self):
        orch = MultiAgentOrchestrator()
        a = orch.get_agent("writer")
        assert isinstance(a, SpecializedAgent)

    def test_list_agents_all_present(self):
        orch = MultiAgentOrchestrator()
        agents = orch.list_agents()
        names = [a["name"] for a in agents]
        assert "Researcher" in names
        assert "Writer" in names
        assert "Coder" in names
        assert "Critic" in names

    def test_get_agent_returns_correct_role(self):
        orch = MultiAgentOrchestrator()
        a = orch.get_agent("coder")
        assert "code" in a.role

    def test_custom_agents_dict(self):
        custom = {
            "analyst": SpecializedAgent(name="Analyst", role="analyze", system_prompt="Analyze stuff."),
        }
        orch = MultiAgentOrchestrator(agents=custom)
        assert orch.get_agent("analyst") is not None
        assert orch.get_agent("researcher") is None

    def test_empty_agents_dict(self):
        orch = MultiAgentOrchestrator(agents={})
        # _load_custom_agents may add agents from ~/.config/sloughgpt/custom_agents.json
        agents = orch.list_agents()
        # With empty dict, only custom agents (if file exists) are loaded
        assert isinstance(agents, list)

    def test_orchestrator_stores_agents(self):
        orch = MultiAgentOrchestrator()
        assert "researcher" in orch.agents
        assert "writer" in orch.agents


class TestOrchestratorCompose:
    def test_compose_all_failed(self):
        orch = MultiAgentOrchestrator(agents={})
        tasks = [
            AgentTask(id="1", description="a", assigned_agent="x", status=TaskStatus.FAILED),
        ]
        result = orch._compose("goal", tasks)
        assert "All agents failed" in result

    def test_compose_with_completed_tasks(self):
        orch = MultiAgentOrchestrator()
        tasks = [
            AgentTask(id="1", description="research", assigned_agent="researcher",
                      status=TaskStatus.COMPLETED, result="Findings here"),
            AgentTask(id="2", description="write", assigned_agent="writer",
                      status=TaskStatus.COMPLETED, result="Written content"),
        ]
        result = orch._compose("test goal", tasks)
        assert isinstance(result, str)
        assert len(result) > 0


class TestSingleton:
    def test_get_and_reset(self):
        reset_orchestrator()
        orch = get_orchestrator()
        assert isinstance(orch, MultiAgentOrchestrator)
        reset_orchestrator()

    def test_singleton_returns_same_instance(self):
        reset_orchestrator()
        a = get_orchestrator()
        b = get_orchestrator()
        assert a is b
        reset_orchestrator()

    def test_reset_creates_new(self):
        reset_orchestrator()
        a = get_orchestrator()
        reset_orchestrator()
        b = get_orchestrator()
        assert a is not b
        reset_orchestrator()

    def test_get_orchestrator_has_default_agents(self):
        reset_orchestrator()
        orch = get_orchestrator()
        assert len(orch.agents) >= 4
        reset_orchestrator()

    def test_reset_then_get(self):
        reset_orchestrator()
        orch = get_orchestrator()
        assert orch is not None
        reset_orchestrator()
        orch2 = get_orchestrator()
        assert orch is not orch2
