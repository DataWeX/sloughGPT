"""Backward-compatibility shim — imports from the new ``domain.agents`` package."""

from domain.agents._internal.agents import (
    SecurityConfig,
    SecurityBoundary,
    ToolCapability,
    ToolDefinition,
    ToolExecutionContext,
    ToolRunner,
    AgentConfig,
    Agent,
)

_agent = None
_runner = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent

def get_runner():
    global _runner
    if _runner is None:
        _runner = ToolRunner()
    return _runner

_LAZY = {
    "MultiAgentOrchestrator",
    "SpecializedAgent",
    "AgentTask",
    "get_orchestrator",
}

def __getattr__(name):
    if name in _LAZY:
        from . import multi
        val = getattr(multi, name)
        globals()[name] = val
        return val
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
]
