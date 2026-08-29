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


# ── TaskStatus ───────────────────────────────────────────────────────

class TestTaskStatus:
    def test_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"


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


class TestOrchestratorSimplePlan:
    def test_simple_plan(self):
        orch = MultiAgentOrchestrator(agents={})
        tasks = orch._simple_plan("test goal")
        assert len(tasks) == 2
        assert tasks[0].assigned_agent == "researcher"
        assert tasks[1].assigned_agent == "writer"
        assert tasks[1].depends_on == ["1"]


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


class TestSingleton:
    def test_get_and_reset(self):
        reset_orchestrator()
        orch = get_orchestrator()
        assert isinstance(orch, MultiAgentOrchestrator)
        reset_orchestrator()
