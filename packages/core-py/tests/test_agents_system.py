"""Tests for domains.agents.system — AgentSystem CRUD."""

import asyncio
import pytest
import domains.agents.system as _mod
from domains.agents.system import AgentSystem, get_agent_system


@pytest.fixture(autouse=True)
def _reset_and_event_loop():
    """Reset singleton and ensure event loop exists for each test."""
    _mod._default_system = None
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    yield
    _mod._default_system = None


class TestAgentSystem:
    def test_list_has_defaults(self):
        sys = AgentSystem()
        agents = sys.list()
        assert len(agents) >= 1

    def test_get_existing(self):
        sys = AgentSystem()
        agents = sys.list()
        first_id = agents[0]["id"]
        agent = sys.get(first_id)
        assert agent is not None
        assert agent["id"] == first_id

    def test_get_nonexistent(self):
        sys = AgentSystem()
        assert sys.get("nonexistent_xyz") is None

    def test_create_and_get(self):
        sys = AgentSystem()
        result = sys.create("test_agent_1", "Test Agent", "A test agent")
        assert result["name"] == "Test Agent"
        fetched = sys.get("test_agent_1")
        assert fetched is not None
        assert fetched["description"] == "A test agent"

    def test_create_default_tools(self):
        sys = AgentSystem()
        result = sys.create("test_agent_2", "Test Agent 2", "desc")
        assert result["tools"] == ["memory"]

    def test_update(self):
        sys = AgentSystem()
        sys.create("test_agent_3", "Old Name", "old desc")
        updated = sys.update("test_agent_3", name="New Name")
        assert updated is not None
        assert updated["name"] == "New Name"

    def test_update_nonexistent(self):
        sys = AgentSystem()
        assert sys.update("nonexistent_xyz", name="x") is None

    def test_delete(self):
        sys = AgentSystem()
        sys.create("test_agent_del", "Delete Me", "desc")
        assert sys.delete("test_agent_del") is True
        assert sys.get("test_agent_del") is None

    def test_get_instructions(self):
        sys = AgentSystem()
        sys.create("test_agent_inst", "Inst Agent", "desc", instructions="Be helpful.")
        instr = sys.get_instructions("test_agent_inst")
        assert "helpful" in instr

    def test_get_instructions_nonexistent(self):
        sys = AgentSystem()
        assert sys.get_instructions("nonexistent_xyz") == ""
