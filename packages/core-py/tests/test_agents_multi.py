"""Tests for domains.agents.multi — SpecializedAgent, DEFAULT_AGENTS."""

from domains.agents.multi import SpecializedAgent, DEFAULT_AGENTS


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
