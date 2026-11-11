"""
Agent System - CRUD management for agent definitions.
"""

from __future__ import annotations

import builtins
import logging
import os
from typing import Any

from domain.agents._internal.agents import Agent, AgentConfig, ToolCapability, get_agent
from domain.infrastructure._internal.repository import FileRepository, JsonSerializer

logger = logging.getLogger("slo.agents")

AGENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "agents")

_agent_repo = FileRepository[dict](
    directory=AGENTS_DIR,
    serializer=JsonSerializer(dict),
    key_suffix=".json",
)
_agent_repo.enable_cache(ttl_seconds=5.0)

_API_BASE = "http://localhost:8000"


DEFAULT_AGENTS: dict[str, dict[str, Any]] = {
    "general": {
        "name": "General",
        "role": "general_assistant",
        "description": "General purpose AI assistant",
        "objective": "Answer the user's request; retrieve relevant knowledge when needed.",
        "scope": "Read-only assistance. Do not claim to change app or external system state.",
        "success_criteria": ["Answer directly", "Distinguish facts from uncertainty"],
        "instructions": "Be concise, accurate, and explicit about uncertainty.",
        "tools": ["knowledge_retrieval"],
        "avatar": "G",
    },
    "coder": {
        "name": "Coder",
        "role": "implementation_engineer",
        "description": "Programming and code execution",
        "objective": "Implement the assigned repository change and verify it with relevant tests.",
        "scope": "Inspect and modify repository code. Do not publish, deploy, or claim unrun checks passed.",
        "success_criteria": ["Keep changes in scope", "Run relevant tests", "Summarize changed files and verification"],
        "instructions": "Follow repository conventions, inspect before editing, and preserve unrelated work.",
        "tools": ["code_execution", "file_read", "file_search"],
        "avatar": "C",
    },
    "researcher": {
        "name": "Researcher",
        "role": "evidence_researcher",
        "description": "Web search and information gathering",
        "objective": "Answer assigned questions using repository evidence and traceable sources.",
        "scope": "Read and search only. Do not modify files or execute code.",
        "success_criteria": ["Cite material claims", "Separate evidence from inference", "State unresolved gaps"],
        "instructions": "Prefer primary sources and never invent citations.",
        "tools": ["web_search", "citation", "knowledge_retrieval", "file_search", "file_read"],
        "avatar": "R",
    },
    "writer": {
        "name": "Writer",
        "role": "technical_writer",
        "description": "Creative and technical writing",
        "objective": "Produce the requested document using supplied facts and approved knowledge.",
        "scope": "Draft text only unless a writing task explicitly authorizes file edits.",
        "success_criteria": ["Match the requested audience and format", "Avoid unsupported facts", "Use consistent terminology"],
        "instructions": "Write clearly; identify missing facts instead of guessing.",
        "tools": ["knowledge_retrieval"],
        "avatar": "W",
    },
    "analyst": {
        "name": "Analyst",
        "role": "data_analyst",
        "description": "Data analysis and structured reasoning",
        "objective": "Analyze supplied data and explain conclusions with reproducible evidence.",
        "scope": "Read and analyze data only; do not alter source data or execute generated code.",
        "success_criteria": ["Show the evidence for conclusions", "Report assumptions and limitations", "Do not infer causality from correlation"],
        "instructions": "Use structured reasoning and report uncertainty.",
        "tools": ["knowledge_retrieval", "file_search", "file_read", "data_analysis"],
        "avatar": "A",
    },
}


def _default_inference_fn(prompt: str, max_tokens: int = 200) -> dict[str, Any]:
    """Default inference function calling the local API."""
    try:
        import requests

        r = requests.post(
            f"{_API_BASE}/infer",
            json={
                "prompt": prompt,
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
            },
            # Planning/synthesis prompts run on a contended box; 30s was
            # starving real runs (live demo hit it under load).
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
        return {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}


class AgentSystem:
    """
    Manages agent definitions with file-based persistence.
    """

    def __init__(self):
        self._agent = get_agent()
        self._agent.set_inference_fn(_default_inference_fn)
        self._load_defaults()

    def _load_defaults(self):
        for aid, data in DEFAULT_AGENTS.items():
            if not _agent_repo.exists(aid):
                _agent_repo.save(aid, data)

    def _save(self, agent_id: str, data: dict):
        _agent_repo.save(agent_id, data)

    def _load(self, agent_id: str) -> dict | None:
        return _agent_repo.get(agent_id)

    @staticmethod
    def _effective_definition(agent_id: str, data: dict) -> dict:
        """Supply new built-in role metadata to older stored definitions without rewriting them."""
        return {**DEFAULT_AGENTS.get(agent_id, {}), **data}

    def list(self) -> builtins.list[dict]:
        agents = []
        for sid in _agent_repo.keys():
            data = _agent_repo.get(sid)
            if data:
                agents.append({"id": sid, **self._effective_definition(sid, data)})
        return agents

    def get(self, agent_id: str) -> dict | None:
        data = _agent_repo.get(agent_id)
        if data is None:
            return None
        return {"id": agent_id, **self._effective_definition(agent_id, data)}

    def get_instructions(self, agent_id: str) -> str:
        """Get the system instructions for an agent (for chat injection)."""
        data = _agent_repo.get(agent_id)
        if data is None:
            return ""
        return self._role_contract(self._effective_definition(agent_id, data))

    @staticmethod
    def _role_contract(data: dict) -> str:
        """Render structured role metadata as runtime instructions."""
        sections = []
        for label, key in (("Role", "role"), ("Objective", "objective"), ("Scope", "scope")):
            if data.get(key):
                sections.append(f"{label}: {data[key]}")
        criteria = data.get("success_criteria") or []
        if not isinstance(criteria, list):
            criteria = []
        criteria = [item for item in criteria if isinstance(item, str) and item.strip()]
        if criteria:
            sections.append("Success criteria:\n" + "\n".join(f"- {item}" for item in criteria))
        if data.get("instructions"):
            sections.append(f"Additional instructions: {data['instructions']}")
        return "\n\n".join(sections)

    def create(
        self,
        agent_id: str,
        name: str,
        description: str,
        instructions: str = "",
        role: str = "general_assistant",
        objective: str = "",
        scope: str = "",
        success_criteria: builtins.list[str] | None = None,
        tools: builtins.list[str] | None = None,
        avatar: str = "",
    ) -> dict:
        data = {
            "name": name,
            "description": description,
            "role": role,
            "objective": objective,
            "scope": scope,
            "success_criteria": list(success_criteria or []),
            "instructions": instructions or f"You are a {name} assistant.",
            "tools": list(tools) if tools is not None else [],
            "avatar": avatar or name[0] if name else "A",
        }
        _agent_repo.save(agent_id, data)
        logger.info("Created agent: %s", agent_id, extra={"tag": "MODEL"})
        return {"id": agent_id, **data}

    def update(self, agent_id: str, **kwargs) -> dict | None:
        data = _agent_repo.get(agent_id)
        if data is None:
            return None
        for key, value in kwargs.items():
            if value is not None and key in (
                "name",
                "description",
                "role",
                "objective",
                "scope",
                "success_criteria",
                "instructions",
                "tools",
                "avatar",
            ):
                data[key] = value
        _agent_repo.save(agent_id, data)
        logger.info("Updated agent: %s", agent_id, extra={"tag": "MODEL"})
        return {"id": agent_id, **data}

    def delete(self, agent_id: str) -> bool:
        return _agent_repo.delete(agent_id)

    def _fresh_agent(self, agent_id: str) -> Agent | None:
        """Build a per-call Agent from stored config (thread-safe, no shared state)."""
        raw_agent_data = self._load(agent_id)
        if raw_agent_data is None:
            return None
        agent_data = self._effective_definition(agent_id, raw_agent_data)
        tool_names = agent_data.get("tools") or []
        if not isinstance(tool_names, list):
            tool_names = []
        tool_names = [name for name in tool_names if isinstance(name, str)]
        if "memory" in tool_names:
            # Legacy profiles used this label for read-only knowledge retrieval.
            tool_names = ["knowledge_retrieval" if name == "memory" else name for name in tool_names]
        capabilities = {capability.value: capability for capability in ToolCapability}
        unknown_tools = sorted(set(tool_names) - capabilities.keys())
        if unknown_tools:
            logger.warning(
                "Agent %s has unknown tools; denying them: %s",
                agent_id,
                ", ".join(unknown_tools),
                extra={"tag": "SECURITY"},
            )
        tools = [capabilities[name] for name in dict.fromkeys(tool_names) if name in capabilities]
        config = AgentConfig(
            tools=tools,
            instructions=self._role_contract(agent_data),
        )
        return Agent(config=config, inference_fn=self._agent._inference_fn)

    async def execute(
        self, agent_id: str, request: str, session_id: str = "", user_id: str = "default"
    ) -> dict:
        """Execute an agent on a user request."""
        fresh_agent = self._fresh_agent(agent_id)
        if fresh_agent is None:
            return {"error": f"Agent '{agent_id}' not found", "success": False}
        return await fresh_agent.execute(request, session_id, user_id)

    async def execute_stream(
        self, agent_id: str, request: str, session_id: str = "", user_id: str = "default"
    ):
        """Yield run events for one agent execution (same loop, SSE view)."""
        fresh_agent = self._fresh_agent(agent_id)
        if fresh_agent is None:
            yield {"event": "error", "error": f"Agent '{agent_id}' not found"}
            return
        async for event in fresh_agent.execute_stream(request, session_id, user_id):
            yield event


_default_system: AgentSystem | None = None


def get_agent_system() -> AgentSystem:
    global _default_system
    if _default_system is None:
        _default_system = AgentSystem()
    return _default_system
