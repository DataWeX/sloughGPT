"""agents — Agentic infrastructure for safe internal tool execution.

Public API:
    SecurityConfig, SecurityBoundary, ToolCapability, ToolDefinition
    ToolExecutionContext, ToolRunner, AgentConfig, Agent
    get_agent, get_runner, get_agent_system, get_tool_registry
"""


def __getattr__(name):
    _lazy = {
        "MultiAgentOrchestrator": "domain.agents._internal.multi",
        "get_agent_run_store": "domain.agents._internal.run_history",
    }
    if name in _lazy:
        import importlib

        return getattr(importlib.import_module(_lazy[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
from domain.agents._internal.tools import get_tool_registry

_LAZY_IMPORTS = {
    "MultiAgentOrchestrator": ("._internal.multi", "MultiAgentOrchestrator"),
    "get_agent_run_store": ("._internal.run_history", "get_agent_run_store"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SecurityConfig",
    "SecurityBoundary",
    "ToolCapability",
    "ToolDefinition",
    "ToolExecutionContext",
    "ToolRunner",
    "AgentConfig",
    "Agent",
    "get_agent",
    "get_runner",
    "get_agent_system",
    "get_tool_registry",
    "MultiAgentOrchestrator",
    "get_agent_run_store",
]
