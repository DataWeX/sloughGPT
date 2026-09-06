"""Tests for agents.system — AgentSystem CRUD and default agents."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from domains.agents.system import (
    AgentSystem,
    DEFAULT_AGENTS,
    get_agent_system,
    _default_inference_fn,
)


@pytest.fixture
def sys(tmp_path, monkeypatch):
    """Create an AgentSystem with a temp directory."""
    monkeypatch.setattr("domains.agents.system.AGENTS_DIR", str(tmp_path))
    monkeypatch.setattr("domains.agents.system._agent_repo", None)
    from domains.agents.system import _agent_repo as _
    # Reset singleton
    import domains.agents.system as mod
    mod._default_system = None

    # Patch the repo
    from domains.infrastructure.repository import FileRepository, JsonSerializer
    repo = FileRepository(
        directory=str(tmp_path),
        serializer=JsonSerializer(dict),
        key_suffix=".json",
    )
    repo.enable_cache(ttl_seconds=5.0)
    monkeypatch.setattr("domains.agents.system._agent_repo", repo)
    monkeypatch.setattr("domains.agents.system.get_agent", lambda: MagicMock())
    monkeypatch.setattr("domains.agents.system._default_inference_fn", lambda *a, **kw: {})

    return AgentSystem()


# ── Default agents ────────────────────────────────────────────────────────


class TestDefaultAgents:

    def test_default_agents_structure(self):
        assert "general" in DEFAULT_AGENTS
        assert "coder" in DEFAULT_AGENTS
        assert "researcher" in DEFAULT_AGENTS
        assert "writer" in DEFAULT_AGENTS
        assert "analyst" in DEFAULT_AGENTS

    def test_default_agent_has_required_fields(self):
        for aid, data in DEFAULT_AGENTS.items():
            assert "name" in data, f"{aid} missing name"
            assert "description" in data, f"{aid} missing description"
            assert "tools" in data, f"{aid} missing tools"
            assert "avatar" in data, f"{aid} missing avatar"

    def test_default_agent_tools_are_lists(self):
        for aid, data in DEFAULT_AGENTS.items():
            assert isinstance(data["tools"], list), f"{aid} tools not list"


# ── AgentSystem CRUD ──────────────────────────────────────────────────────


class TestAgentSystemCRUD:

    def test_create_agent(self, sys):
        result = sys.create("test1", "Test Agent", "A test agent", "Be helpful")
        assert result["id"] == "test1"
        assert result["name"] == "Test Agent"
        assert result["instructions"] == "Be helpful"

    def test_create_agent_defaults(self, sys):
        result = sys.create("test2", "Test", "Desc")
        assert result["instructions"] == "You are a Test assistant."
        assert result["tools"] == ["memory"]

    def test_create_agent_default_avatar(self, sys):
        result = sys.create("test3", "Alice", "Desc")
        assert result["avatar"] == "A"

    def test_get_agent(self, sys):
        sys.create("test1", "Test", "Desc")
        result = sys.get("test1")
        assert result is not None
        assert result["id"] == "test1"
        assert result["name"] == "Test"

    def test_get_nonexistent(self, sys):
        assert sys.get("nonexistent") is None

    def test_list_agents(self, sys):
        sys.create("a1", "Agent 1", "Desc 1")
        sys.create("a2", "Agent 2", "Desc 2")
        agents = sys.list()
        ids = [a["id"] for a in agents]
        assert "a1" in ids
        assert "a2" in ids

    def test_update_agent(self, sys):
        sys.create("test1", "Test", "Desc")
        result = sys.update("test1", name="Updated", description="New desc")
        assert result["name"] == "Updated"
        assert result["description"] == "New desc"

    def test_update_nonexistent(self, sys):
        assert sys.update("nonexistent", name="X") is None

    def test_update_ignores_invalid_keys(self, sys):
        sys.create("test1", "Test", "Desc")
        result = sys.update("test1", name="Valid", invalid_key="Ignored")
        assert result["name"] == "Valid"

    def test_delete_agent(self, sys):
        sys.create("test1", "Test", "Desc")
        assert sys.delete("test1") is True
        assert sys.get("test1") is None

    def test_get_instructions(self, sys):
        sys.create("test1", "Test", "Desc", instructions="Be helpful")
        assert sys.get_instructions("test1") == "Be helpful"

    def test_get_instructions_nonexistent(self, sys):
        assert sys.get_instructions("nonexistent") == ""


# ── Default loading ───────────────────────────────────────────────────────


class TestDefaultsLoading:

    def test_defaults_loaded_on_init(self, sys):
        agents = sys.list()
        ids = [a["id"] for a in agents]
        for default_id in DEFAULT_AGENTS:
            assert default_id in ids

    def test_defaults_not_overwritten(self, sys):
        sys.create("general", "Custom General", "Overridden")
        result = sys.get("general")
        assert result["name"] == "Custom General"


# ── Inference function ────────────────────────────────────────────────────


class TestInferenceFunction:

    def test_default_inference_fn_returns_dict(self):
        result = _default_inference_fn("test prompt")
        assert isinstance(result, dict)


# ── Singleton ─────────────────────────────────────────────────────────────


class TestSingleton:

    def test_get_returns_same(self):
        with patch("domains.agents.system.get_agent", return_value=MagicMock()):
            import domains.agents.system as mod
            mod._default_system = None
            s1 = get_agent_system()
            s2 = get_agent_system()
            assert s1 is s2
            mod._default_system = None
