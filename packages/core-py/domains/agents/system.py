"""
Agent System - CRUD management for agent definitions.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any

from . import Agent, AgentConfig, ToolCapability, get_agent

logger = logging.getLogger("agents")

AGENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "agents")


def _ensure_dir():
    os.makedirs(AGENTS_DIR, exist_ok=True)


def _path(name: str) -> str:
    return os.path.join(AGENTS_DIR, f"{name}.json")


DEFAULT_AGENTS: Dict[str, Dict[str, Any]] = {
    "assistant": {
        "name": "Assistant",
        "description": "General purpose AI assistant",
        "instructions": "You are a helpful AI assistant. Answer questions clearly and concisely.",
        "tools": ["memory", "file_search"],
        "avatar": "A",
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
}


class AgentSystem:
    """
    Manages agent definitions with file-based persistence.
    """

    def __init__(self):
        _ensure_dir()
        self._agent = get_agent()
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

        tools = agent_data.get("tools", [])
        config = AgentConfig(tools=[ToolCapability(t) for t in tools if t in [c.value for c in ToolCapability]])
        self._agent.config = config
        return await self._agent.execute(request, session_id, user_id)


_default_system: Optional[AgentSystem] = None


def get_agent_system() -> AgentSystem:
    global _default_system
    if _default_system is None:
        _default_system = AgentSystem()
    return _default_system
