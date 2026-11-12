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
        "description": "General purpose AI assistant",
        "instructions": "You are a helpful AI assistant. Answer questions clearly and concisely.",
        "tools": ["memory"],
        "avatar": "G",
    },
    "coder": {
        "name": "Coder",
        "description": "Programming and code execution",
        "instructions": "You are an expert programmer. Write clean, well-documented code.",
        "tools": ["code_execution", "file_search", "terminal"],
        "avatar": "C",
    },
    "researcher": {
        "name": "Researcher",
        "description": "Web search and information gathering",
        "instructions": "You are a research assistant. Find accurate and well-sourced information.",
        "tools": ["web_search", "citation", "memory"],
        "avatar": "R",
    },
    "writer": {
        "name": "Writer",
        "description": "Creative and technical writing",
        "instructions": "You are a skilled writer. Produce clear, engaging, and well-structured content.",
        "tools": ["memory"],
        "avatar": "W",
    },
    "analyst": {
        "name": "Analyst",
        "description": "Data analysis and structured reasoning",
        "instructions": "You are a data analyst. Break down complex problems and present findings clearly.",
        "tools": ["memory", "file_search"],
        "avatar": "A",
    },
}


def _default_inference_fn(prompt: str, max_tokens: int = 200) -> dict[str, Any]:
    """Default inference function calling the local API."""
    try:
        import requests

        r = requests.post(
            f"{_API_BASE}/inference/generate",
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


def _fast_tier_fn(base_fn: Any) -> Any:
    """Wrap an inference fn with a cache-first fast path.

    A repeated prompt is served from the shared :class:`ResponseCache`
    without calling the model at all; a miss calls through and stores the
    answer. Keys are built the same way the rest of the cache builds them,
    so a model generation bump invalidates these entries too.
    """
    from domain.inference._internal.response_cache import ResponseCache, get_response_cache

    cache = get_response_cache()

    def fast_inference(prompt: str, max_tokens: int = 200) -> dict[str, Any]:
        key = ResponseCache.build_key(
            [{"role": "user", "content": str(prompt)}],
            model="agent-fast-tier",
            temperature=0.0,
            max_tokens=int(max_tokens),
            system_prompt="",
        )
        hit = cache.get(key)
        if hit is not None:
            return {"text": hit, "cached": True}
        response = base_fn(prompt, max_tokens)
        if isinstance(response, dict):
            text = str(response.get("text", "") or "")
            if text:
                cache.put(key, text, model="agent-fast-tier", prompt_for_semantic=str(prompt))
        return response

    return fast_inference


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

    def list(self) -> builtins.list[dict]:
        agents = []
        for sid in _agent_repo.keys():
            data = _agent_repo.get(sid)
            if data:
                agents.append({"id": sid, **data})
        return agents

    def get(self, agent_id: str) -> dict | None:
        data = _agent_repo.get(agent_id)
        if data is None:
            return None
        return {"id": agent_id, **data}

    def get_instructions(self, agent_id: str) -> str:
        """Get the system instructions for an agent (for chat injection)."""
        data = _agent_repo.get(agent_id)
        if data is None:
            return ""
        return data.get("instructions", "")

    def create(
        self,
        agent_id: str,
        name: str,
        description: str,
        instructions: str = "",
        tools: builtins.list[str] | None = None,
        avatar: str = "",
        inference_tier: str = "main",
    ) -> dict:
        data = {
            "name": name,
            "description": description,
            "instructions": instructions or f"You are a {name} assistant.",
            "tools": tools or ["memory"],
            "avatar": avatar or name[0] if name else "A",
            "inference_tier": inference_tier if inference_tier in ("fast", "main") else "main",
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
                "instructions",
                "tools",
                "avatar",
                "inference_tier",
            ):
                if key == "inference_tier" and value not in ("fast", "main"):
                    continue
                data[key] = value
        _agent_repo.save(agent_id, data)
        logger.info("Updated agent: %s", agent_id, extra={"tag": "MODEL"})
        return {"id": agent_id, **data}

    def delete(self, agent_id: str) -> bool:
        return _agent_repo.delete(agent_id)

    def _fresh_agent(self, agent_id: str) -> Agent | None:
        """Build a per-call Agent from stored config (thread-safe, no shared state)."""
        agent_data = self.get(agent_id)
        if agent_data is None:
            return None
        tool_names = agent_data.get("tools", [])
        tools = [ToolCapability(t) for t in tool_names if t in [c.value for c in ToolCapability]]
        tier = agent_data.get("inference_tier", "main")
        if tier not in ("fast", "main"):
            tier = "main"
        config = AgentConfig(
            tools=tools,
            instructions=agent_data.get("instructions", ""),
            inference_tier=tier,
        )
        inference_fn = self._agent._inference_fn
        # Tier is resolved HERE, once per construction: the run loop never
        # branches on it. Fast agents get a cache-first wrapper so repeated
        # prompts are served without touching the slow model.
        if tier == "fast" and inference_fn is not None:
            inference_fn = _fast_tier_fn(inference_fn)
        return Agent(config=config, inference_fn=inference_fn)

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
