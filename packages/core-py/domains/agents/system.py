"""
Agent System - CRUD management for agent definitions.
"""

import json
import logging
import os
from typing import Callable, Dict, List, Optional, Any

from . import Agent, AgentConfig, ToolCapability, get_agent

logger = logging.getLogger("man.agents")

AGENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "agents")

_API_BASE = "http://localhost:8000"


def _ensure_dir():
    os.makedirs(AGENTS_DIR, exist_ok=True)


def _path(name: str) -> str:
    return os.path.join(AGENTS_DIR, f"{name}.json")


DEFAULT_AGENTS: Dict[str, Dict[str, Any]] = {
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


def _default_inference_fn(prompt: str, max_tokens: int = 200) -> Dict[str, Any]:
    """Default inference function calling the local API."""
    try:
        import requests
        r = requests.post(f"{_API_BASE}/inference/generate", json={
            "prompt": prompt,
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
        }, timeout=30)
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
        _ensure_dir()
        self._agent = get_agent()
        self._agent.set_inference_fn(_default_inference_fn)
        self._load_defaults()

    def _load_defaults(self):
        for aid, data in DEFAULT_AGENTS.items():
            if not os.path.exists(_path(aid)):
                self._save(aid, data)

    def _save(self, agent_id: str, data: dict):
        with open(_path(agent_id), "w") as f:
            json.dump(data, f, indent=2)

    def _load(self, agent_id: str) -> Optional[dict]:
        path = _path(agent_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def list(self) -> List[dict]:
        _ensure_dir()
        agents = []
        for fname in sorted(os.listdir(AGENTS_DIR)):
            if fname.endswith(".json"):
                agent_id = fname[:-5]
                data = self._load(agent_id)
                if data:
                    agents.append({"id": agent_id, **data})
        return agents

    def get(self, agent_id: str) -> Optional[dict]:
        data = self._load(agent_id)
        if data is None:
            return None
        return {"id": agent_id, **data}

    def get_instructions(self, agent_id: str) -> str:
        """Get the system instructions for an agent (for chat injection)."""
        data = self._load(agent_id)
        if data is None:
            return ""
        return data.get("instructions", "")

    def create(self, agent_id: str, name: str, description: str,
               instructions: str = "", tools: Optional[List[str]] = None,
               avatar: str = "") -> dict:
        data = {
            "name": name,
            "description": description,
            "instructions": instructions or f"You are a {name} assistant.",
            "tools": tools or ["memory"],
            "avatar": avatar or name[0] if name else "A",
        }
        self._save(agent_id, data)
        logger.info(f"Created agent: {agent_id}")
        return {"id": agent_id, **data}

    def update(self, agent_id: str, **kwargs) -> Optional[dict]:
        data = self._load(agent_id)
        if data is None:
            return None
        for key, value in kwargs.items():
            if value is not None and key in ("name", "description", "instructions", "tools", "avatar"):
                data[key] = value
        self._save(agent_id, data)
        logger.info(f"Updated agent: {agent_id}")
        return {"id": agent_id, **data}

    def delete(self, agent_id: str) -> bool:
        path = _path(agent_id)
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Deleted agent: {agent_id}")
            return True
        return False

    async def execute(self, agent_id: str, request: str, session_id: str = "",
                      user_id: str = "default") -> dict:
        """Execute an agent on a user request."""
        agent_data = self.get(agent_id)
        if agent_data is None:
            return {"error": f"Agent '{agent_id}' not found", "success": False}

        # Build fresh config per call (thread-safe)
        tool_names = agent_data.get("tools", [])
        tools = [ToolCapability(t) for t in tool_names if t in [c.value for c in ToolCapability]]
        config = AgentConfig(
            tools=tools,
            instructions=agent_data.get("instructions", ""),
        )
        # Create a fresh Agent with per-call config to avoid shared-state races
        fresh_agent = Agent(config=config, inference_fn=self._agent._inference_fn)
        return await fresh_agent.execute(request, session_id, user_id)


_default_system: Optional[AgentSystem] = None


def get_agent_system() -> AgentSystem:
    global _default_system
    if _default_system is None:
        _default_system = AgentSystem()
    return _default_system
