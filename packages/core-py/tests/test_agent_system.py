"""Tests for the agent system CRUD layer (file-backed agent definitions)."""

import sys
import types

import pytest

import domains.agents as agents_pkg
import domains.agents.system as system_mod
from domains.agents.system import AgentSystem, get_agent_system, DEFAULT_AGENTS, _default_inference_fn
from domains.infrastructure.repository import FileRepository, JsonSerializer


@pytest.fixture
async def agent_system(tmp_path, monkeypatch):
    repo = FileRepository[dict](
        directory=str(tmp_path / "agents"),
        serializer=JsonSerializer(dict),
        key_suffix=".json",
    )
    repo.enable_cache(ttl_seconds=5.0)
    monkeypatch.setattr(system_mod, "_agent_repo", repo)
    monkeypatch.setattr(system_mod, "_default_system", None)
    monkeypatch.setattr(agents_pkg, "_agent", None)
    yield AgentSystem()
    monkeypatch.setattr(system_mod, "_default_system", None)


# ── AgentSystem init / defaults ───────────────────────────────────────────


class TestInit:
    def test_loads_all_defaults(self, agent_system):
        ids = {a["id"] for a in agent_system.list()}
        assert ids == set(DEFAULT_AGENTS.keys())

    def test_default_entries_have_fields(self, agent_system):
        by_id = {a["id"]: a for a in agent_system.list()}
        coder = by_id["coder"]
        assert coder["name"] == "Coder"
        assert "code_execution" in coder["tools"]

    def test_does_not_overwrite_existing_default(self, agent_system):
        agent_system.create("coder", name="Custom Coder", description="mine")
        agent_system._load_defaults()
        data = agent_system.get("coder")
        assert data["name"] == "Custom Coder"


# ── CRUD ──────────────────────────────────────────────────────────────────


class TestGet:
    def test_get_existing(self, agent_system):
        data = agent_system.get("general")
        assert data is not None
        assert data["id"] == "general"
        assert data["name"] == "General"

    def test_get_missing(self, agent_system):
        assert agent_system.get("nope") is None


class TestList:
    def test_list_returns_all_with_ids(self, agent_system):
        agents = agent_system.list()
        assert len(agents) == len(DEFAULT_AGENTS)
        assert all("id" in a for a in agents)

    def test_list_skips_unreadable_files(self, agent_system, tmp_path, monkeypatch):
        bad = tmp_path / "agents" / "bad.json"
        bad.write_text("{ not valid json")
        agents = agent_system.list()
        assert all(a["id"] != "bad" for a in agents)


class TestGetInstructions:
    def test_returns_instructions(self, agent_system):
        instructions = agent_system.get_instructions("general")
        assert "helpful AI assistant" in instructions

    def test_missing_returns_empty(self, agent_system):
        assert agent_system.get_instructions("nope") == ""


class TestCreate:
    def test_create_full(self, agent_system):
        result = agent_system.create("scientist", "Scientist", "Does science",
                                     instructions="Be rigorous", tools=["file_search"], avatar="S")
        assert result["id"] == "scientist"
        assert result["name"] == "Scientist"
        assert result["instructions"] == "Be rigorous"
        assert result["tools"] == ["file_search"]
        assert result["avatar"] == "S"
        assert agent_system.get("scientist")["id"] == "scientist"

    def test_create_default_instructions(self, agent_system):
        result = agent_system.create("bot", "Bot", "desc")
        assert result["instructions"] == "You are a Bot assistant."

    def test_create_default_tools(self, agent_system):
        result = agent_system.create("bot", "Bot", "desc")
        assert result["tools"] == ["memory"]

    def test_create_default_avatar_from_name(self, agent_system):
        assert agent_system.create("bot", "Bob", "desc")["avatar"] == "B"

    def test_create_default_avatar_empty_name(self, agent_system):
        assert agent_system.create("bot", "", "desc")["avatar"] == "A"

    def test_create_uses_given_avatar(self, agent_system):
        assert agent_system.create("bot", "Bob", "desc", avatar="X")["avatar"] == "X"

    def test_create_persists_to_disk(self, agent_system, tmp_path):
        agent_system.create("sci", "Sci", "d")
        assert (tmp_path / "agents" / "sci.json").exists()


class TestUpdate:
    def test_updates_fields(self, agent_system):
        result = agent_system.update("general", name="General 2", instructions="New")
        assert result["name"] == "General 2"
        assert result["instructions"] == "New"
        assert agent_system.get("general")["name"] == "General 2"

    def test_ignores_none_values(self, agent_system):
        before = agent_system.get("general")
        result = agent_system.update("general", name=None, instructions="Keep")
        assert result["instructions"] == "Keep"
        assert result["name"] == before["name"]

    def test_ignores_unknown_keys(self, agent_system):
        result = agent_system.update("general", not_a_field="x")
        assert "not_a_field" not in result

    def test_update_missing_returns_none(self, agent_system):
        assert agent_system.update("nope", name="X") is None


class TestDelete:
    def test_delete_existing(self, agent_system):
        assert agent_system.delete("general") is True
        assert agent_system.get("general") is None

    def test_delete_missing(self, agent_system):
        assert agent_system.delete("nope") is False


class TestPrivateIO:
    def test_save_load_roundtrip(self, agent_system):
        agent_system._save("custom", {"name": "N", "description": "D"})
        data = agent_system._load("custom")
        assert data["name"] == "N"

    def test_load_missing(self, agent_system):
        assert agent_system._load("missing") is None


# ── Execute ───────────────────────────────────────────────────────────────


class TestExecute:
    async def test_missing_agent(self, agent_system):
        result = await agent_system.execute("nope", "hello")
        assert result["success"] is False
        assert "not found" in result["error"]

    async def test_empty_plan(self, agent_system, monkeypatch):
        monkeypatch.setattr(agent_system._agent, "_inference_fn", lambda prompt: "[]")
        result = await agent_system.execute("general", "hello", session_id="s1")
        assert result["session_id"] == "s1"
        assert result["tools_used"] == []

    async def test_llm_plan_runs_code_tool(self, agent_system, monkeypatch):
        monkeypatch.setattr(
            agent_system._agent, "_inference_fn",
            lambda prompt: '[{"tool": "code_execution", "args": {"code": "print(41+1)"}}]',
        )
        result = await agent_system.execute("coder", "run code", user_id="u1")
        assert result["tools_used"]
        tool_result = result["tools_used"][0]["result"]
        assert tool_result["success"] is True

    async def test_invalid_plan_falls_back_to_keywords(self, agent_system, monkeypatch):
        monkeypatch.setattr(agent_system._agent, "_inference_fn", lambda prompt: "not json at all")
        result = await agent_system.execute("general", "hello there", user_id="u1")
        assert "response" in result

    async def test_unknown_tool_filtered_from_capabilities(self, agent_system, monkeypatch):
        monkeypatch.setattr(agent_system._agent, "_inference_fn", lambda prompt: "[]")
        agent_system.update("general", tools=["code_execution", "not_a_capability"])
        result = await agent_system.execute("general", "hi")
        assert result["tools_used"] == []


# ── Default inference fn ──────────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeRequests:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def post(self, url, json=None, timeout=None):
        if self._error:
            raise self._error
        return self._response


class TestDefaultInferenceFn:
    def test_success_path(self, monkeypatch):
        fake = _FakeRequests(_FakeResponse(200, {"text": "hi"}))
        monkeypatch.setitem(sys.modules, "requests", fake)
        assert _default_inference_fn("prompt") == {"text": "hi"}

    def test_non_200(self, monkeypatch):
        fake = _FakeRequests(_FakeResponse(500, {}))
        monkeypatch.setitem(sys.modules, "requests", fake)
        result = _default_inference_fn("prompt")
        assert result["error"] == "HTTP 500"

    def test_exception(self, monkeypatch):
        fake = _FakeRequests(error=RuntimeError("boom"))
        monkeypatch.setitem(sys.modules, "requests", fake)
        result = _default_inference_fn("prompt")
        assert result["error"] == "boom"


# ── Singleton ─────────────────────────────────────────────────────────────


class TestSingleton:
    async def test_get_agent_system_singleton(self, monkeypatch):
        monkeypatch.setattr(system_mod, "_default_system", None)
        a = get_agent_system()
        b = get_agent_system()
        assert a is b
        monkeypatch.setattr(system_mod, "_default_system", None)

    def test_system_uses_shared_agent(self, agent_system):
        a = get_agent_system()
        assert a._agent is not None
