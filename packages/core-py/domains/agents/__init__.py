"""Backward-compatibility shim — imports from the new ``domain.agents`` package."""

from domain.agents._internal.agents import (
    Agent,
    AgentConfig,
    SecurityBoundary,
    SecurityConfig,
    ToolCapability,
    ToolDefinition,
    ToolExecutionContext,
    ToolRunner,
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


_LAZY_MULTI = {
    "MultiAgentOrchestrator",
    "SpecializedAgent",
    "AgentTask",
    "get_orchestrator",
}

_LAZY_MODULES = {
    "run_history": "domain.agents._internal.run_history",
}


def __getattr__(name):
    if name in _LAZY_MULTI:
        from domain.agents._internal import multi

        val = getattr(multi, name)
        globals()[name] = val
        return val
    if name in _LAZY_MODULES:
        import importlib

        mod = importlib.import_module(_LAZY_MODULES[name])
        globals()[name] = mod
        return mod
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
