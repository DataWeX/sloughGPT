"""Tests for domains.agents.multi — SpecializedAgent, TaskStatus, AgentTask, MultiAgentOrchestrator."""

import json
from domains.agents.multi import (
    SpecializedAgent,
    DEFAULT_AGENTS,
    TaskStatus,
    AgentTask,
    MultiAgentOrchestrator,
    get_orchestrator,
    reset_orchestrator,
)


class TestSpecializedAgent:
    def test_fields(self):
        sa = SpecializedAgent(name="test", role="do things", system_prompt="You are test")
        assert sa.name == "test"
        assert sa.role == "do things"
        assert sa.tools == []

    def test_to_dict(self):
        sa = SpecializedAgent(name="Researcher", role="research", system_prompt="You are a research agent")
        d = sa.to_dict()
        assert d["name"] == "Researcher"
        assert d["role"] == "research"

    def test_to_dict_truncates_prompt(self):
        long_prompt = "x" * 200
        sa = SpecializedAgent(name="A", role="r", system_prompt=long_prompt)
        d = sa.to_dict()
        assert len(d["description"]) == 83  # 80 chars + "..."

    def test_custom_tools(self):
        sa = SpecializedAgent(name="A", role="r", system_prompt="p", tools=["web_search", "code_execution"])
        assert sa.tools == ["web_search", "code_execution"]

    def test_empty_tools_default(self):
        sa = SpecializedAgent(name="A", role="r", system_prompt="p")
        assert sa.tools == []

    def test_name_preserved(self):
        sa = SpecializedAgent(name="MyAgent", role="r", system_prompt="p")
        assert sa.name == "MyAgent"

    def test_role_preserved(self):
        sa = SpecializedAgent(name="A", role="my role", system_prompt="p")
        assert sa.role == "my role"

    def test_system_prompt_preserved(self):
        sa = SpecializedAgent(name="A", role="r", system_prompt="Do things well")
        assert sa.system_prompt == "Do things well"


class TestDefaultAgents:
    def test_default_agents_exist(self):
        assert "researcher" in DEFAULT_AGENTS
        assert "writer" in DEFAULT_AGENTS
        assert "coder" in DEFAULT_AGENTS
        assert "critic" in DEFAULT_AGENTS

    def test_default_agents_are_specialized(self):
        for name, agent in DEFAULT_AGENTS.items():
            assert isinstance(agent, SpecializedAgent)
            assert agent.name
            assert agent.role

    def test_default_agents_count(self):
        assert len(DEFAULT_AGENTS) == 4

    def test_researcher_has_tools(self):
        r = DEFAULT_AGENTS["researcher"]
        assert "web_search" in r.tools
        assert "memory" in r.tools

    def test_writer_has_memory(self):
        w = DEFAULT_AGENTS["writer"]
        assert "memory" in w.tools

    def test_coder_has_code_execution(self):
        c = DEFAULT_AGENTS["coder"]
        assert "code_execution" in c.tools
        assert "file_search" in c.tools

    def test_critic_has_memory(self):
        cr = DEFAULT_AGENTS["critic"]
        assert "memory" in cr.tools

    def test_each_agent_has_unique_name(self):
        names = [a.name for a in DEFAULT_AGENTS.values()]
        assert len(set(names)) == len(names)

    def test_each_agent_has_unique_role(self):
        roles = [a.role for a in DEFAULT_AGENTS.values()]
        assert len(set(roles)) == len(roles)

    def test_all_agents_to_dict(self):
        for agent in DEFAULT_AGENTS.values():
            d = agent.to_dict()
            assert "name" in d
            assert "role" in d
            assert "description" in d


class TestTaskStatus:
    def test_pending(self):
        assert TaskStatus.PENDING == "pending"

    def test_in_progress(self):
        assert TaskStatus.IN_PROGRESS == "in_progress"

    def test_completed(self):
        assert TaskStatus.COMPLETED == "completed"

    def test_failed(self):
        assert TaskStatus.FAILED == "failed"

    def test_all_statuses_are_strings(self):
        statuses = [TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED, TaskStatus.FAILED]
        for s in statuses:
            assert isinstance(s, str)


class TestAgentTask:
    def test_fields(self):
        t = AgentTask(id="1", description="do stuff", assigned_agent="researcher")
        assert t.id == "1"
        assert t.description == "do stuff"
        assert t.assigned_agent == "researcher"

    def test_defaults(self):
        t = AgentTask(id="1", description="d", assigned_agent="r")
        assert t.context == ""
        assert t.result == ""
        assert t.status == TaskStatus.PENDING
        assert t.error == ""
        assert t.depends_on == []

    def test_to_dict(self):
        t = AgentTask(id="1", description="Research X", assigned_agent="researcher")
        d = t.to_dict()
        assert d["id"] == "1"
        assert d["agent"] == "researcher"
        assert d["status"] == "pending"
        assert d["depends_on"] == []

    def test_to_dict_result_preview(self):
        t = AgentTask(id="1", description="d", assigned_agent="r", result="x" * 200)
        d = t.to_dict()
        assert len(d["result_preview"]) == 100

    def test_to_dict_empty_result(self):
        t = AgentTask(id="1", description="d", assigned_agent="r")
        d = t.to_dict()
        assert d["result_preview"] == ""

    def test_depends_on(self):
        t = AgentTask(id="2", description="d", assigned_agent="w", depends_on=["1"])
        assert t.depends_on == ["1"]

    def test_multiple_depends(self):
        t = AgentTask(id="3", description="d", assigned_agent="c", depends_on=["1", "2"])
        assert len(t.depends_on) == 2

    def test_status_change(self):
        t = AgentTask(id="1", description="d", assigned_agent="r")
        assert t.status == TaskStatus.PENDING
        t.status = TaskStatus.IN_PROGRESS
        assert t.status == TaskStatus.IN_PROGRESS

    def test_result_assignment(self):
        t = AgentTask(id="1", description="d", assigned_agent="r")
        t.result = "completed work"
        assert t.result == "completed work"

    def test_error_assignment(self):
        t = AgentTask(id="1", description="d", assigned_agent="r")
        t.error = "something broke"
        assert t.error == "something broke"


class TestMultiAgentOrchestrator:
    def _make_orchestrator(self):
        return MultiAgentOrchestrator()

    def test_list_agents(self):
        orch = self._make_orchestrator()
        agents = orch.list_agents()
        assert len(agents) >= 4
        assert all("name" in a for a in agents)

    def test_get_agent(self):
        orch = self._make_orchestrator()
        r = orch.get_agent("researcher")
        assert r is not None
        assert r.name == "Researcher"

    def test_get_agent_missing(self):
        orch = self._make_orchestrator()
        assert orch.get_agent("nonexistent") is None

    def test_simple_plan(self):
        orch = self._make_orchestrator()
        tasks = orch._simple_plan("test goal")
        assert len(tasks) == 2
        assert tasks[0].assigned_agent == "researcher"
        assert tasks[1].assigned_agent == "writer"

    def test_simple_plan_depends(self):
        orch = self._make_orchestrator()
        tasks = orch._simple_plan("test goal")
        assert tasks[0].depends_on == []
        assert tasks[1].depends_on == ["1"]

    def test_compute_levels_no_deps(self):
        orch = self._make_orchestrator()
        tasks = [
            AgentTask(id="1", description="a", assigned_agent="researcher"),
            AgentTask(id="2", description="b", assigned_agent="writer"),
        ]
        levels = orch._compute_levels(tasks)
        assert len(levels) == 1
        assert set(levels[0]) == {"1", "2"}

    def test_compute_levels_linear(self):
        orch = self._make_orchestrator()
        tasks = [
            AgentTask(id="1", description="a", assigned_agent="researcher"),
            AgentTask(id="2", description="b", assigned_agent="writer", depends_on=["1"]),
        ]
        levels = orch._compute_levels(tasks)
        assert len(levels) == 2
        assert levels[0] == ["1"]
        assert levels[1] == ["2"]

    def test_compute_levels_diamond(self):
        orch = self._make_orchestrator()
        tasks = [
            AgentTask(id="1", description="a", assigned_agent="researcher"),
            AgentTask(id="2", description="b", assigned_agent="writer", depends_on=["1"]),
            AgentTask(id="3", description="c", assigned_agent="coder", depends_on=["1"]),
            AgentTask(id="4", description="d", assigned_agent="critic", depends_on=["2", "3"]),
        ]
        levels = orch._compute_levels(tasks)
        assert len(levels) == 3
        assert "1" in levels[0]
        assert set(levels[1]) == {"2", "3"}
        assert "4" in levels[2]

    def test_build_dep_context_empty(self):
        orch = self._make_orchestrator()
        t = AgentTask(id="1", description="a", assigned_agent="researcher")
        ctx = orch._build_dep_context(t, {}, {})
        assert ctx == ""

    def test_build_dep_context_with_deps(self):
        orch = self._make_orchestrator()
        task_map = {
            "1": AgentTask(id="1", description="Research", assigned_agent="researcher"),
            "2": AgentTask(id="2", description="Write", assigned_agent="writer", depends_on=["1"]),
        }
        ctx = orch._build_dep_context(task_map["2"], task_map, {"1": "research result"})
        assert "research result" in ctx
        assert "researcher" in ctx

    def test_compose_all_failed(self):
        orch = self._make_orchestrator()
        tasks = [AgentTask(id="1", description="a", assigned_agent="researcher", status=TaskStatus.FAILED)]
        result = orch._compose("goal", tasks)
        assert result == "All agents failed."

    def test_default_agents_in_orchestrator(self):
        orch = self._make_orchestrator()
        assert "researcher" in orch.agents
        assert "writer" in orch.agents

    def test_custom_agents(self):
        custom = {"myagent": SpecializedAgent(name="MyAgent", role="custom", system_prompt="Custom")}
        orch = MultiAgentOrchestrator(agents=custom)
        assert "myagent" in orch.agents
        assert orch.get_agent("myagent").name == "MyAgent"


class TestSingleton:
    def test_get_orchestrator_returns_same(self):
        reset_orchestrator()
        o1 = get_orchestrator()
        o2 = get_orchestrator()
        assert o1 is o2

    def test_reset_orchestrator(self):
        o1 = get_orchestrator()
        reset_orchestrator()
        o2 = get_orchestrator()
        assert o1 is not o2

    def test_reset_creates_new(self):
        reset_orchestrator()
        o = get_orchestrator()
        assert isinstance(o, MultiAgentOrchestrator)
