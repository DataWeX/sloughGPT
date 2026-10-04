"""agents — Agentic infrastructure for safe internal tool execution.

Public API:
    SecurityConfig, SecurityBoundary, ToolCapability, ToolDefinition
    ToolExecutionContext, ToolRunner, AgentConfig, Agent
    ToolSpec, ToolParam
    get_agent, get_runner, get_agent_system, get_tool_registry
"""

from domain.agents._internal.agents import (
    Agent,
    AgentConfig,
    SecurityBoundary,
    SecurityConfig,
    ToolCapability,
    ToolDefinition,
    ToolExecutionContext,
    ToolRunner,
    get_agent,
    get_runner,
)
from domain.agents._internal.system import get_agent_system
from domain.agents._internal.tools import ToolParam, ToolSpec, get_tool_registry

__all__ = [
    "SecurityConfig",
    "SecurityBoundary",
    "ToolCapability",
    "ToolDefinition",
    "ToolExecutionContext",
    "ToolRunner",
    "AgentConfig",
    "Agent",
    "ToolSpec",
    "ToolParam",
    "get_agent",
    "get_runner",
    "get_agent_system",
    "get_tool_registry",
]
